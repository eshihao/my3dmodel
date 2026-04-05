# VQA 诊断（针对：为何 stage3_2 在 70000 steps 最好也只有 ~49%）

> 结论先行：不是单一原因，而是 **架构实现与训练目标存在错位**，叠加 **多任务稀释 + 严格判分口径 + 部分2D特征缺失**。70000 steps 更像是中期平衡点，而不是最终收敛点。

---

## 1) 你当前“实际生效”的模型架构（按代码路径还原）

1. 语言主干：`LamedPhi3ForCausalLM`（Phi-3 Causal LM）。
2. 视觉塔：`vit_stage2_dual_encoders` -> `ViT3DTower_dual_encoders`（stage1 + stage2 双路）。
3. projector：`VisualPacker_3d_phi_v3`，输入期望是 3D patch token 序列（后续 reshape 到 `(8,16,16,768)` 结构）。
4. 训练可学习参数（stage3）：`model.requires_grad_(False)` 后仅打开 `mm_projector` + LoRA。

### 关键实现错位（非常重要）

- `ViT3DTower_dual_encoders.forward` 返回的是双分支 `(image_features_stage1, image_features_stage2)`。
- 但 `lamed_phi3.py` 中 `encode_images()` 遇到 tuple 时只取 `vision_out[0]`，即只用第一分支特征；第二分支基本被丢弃。
- 同时代码把 token 数强制插值到 `EXPECTED_N=2048` 再送入 `mm_projector`。

这意味着：
- 你名义上是“dual encoders + 2D增强”，但实际训练/推理可能主要靠第一路；
- 双路设计收益无法充分兑现；
- 也能解释为什么中期能到49%，但上不去。

---

## 2) 三份日志和训练脚本给出的事实

### stage2
- `train_stage2.log` 的 `eval_accuracy=0.957` 是 stage2 对齐任务指标，不是 close-ended VQA 选项准确率。

### stage3_1
- `train_3-1.sh` 设置了 `evaluation_strategy="no"`，没有在线验证 VQA 指标。
- `train_stage3_1.log` 末段 loss 仍约 2.3~2.4，说明更像过渡对齐阶段。

### stage3_2
- `UniDatasets` 是 `Caption + Close VQA + Open VQA` 直接 concat，任务目标天然冲突（尤其对 close-ended 分类）。
- `train_stage3_2.log` 明确出现多次“2D特征缺失 -> 零向量兜底”警告，样本质量在训练中有波动。
- 末段学习率已经很低（接近衰减尾部），继续训练容易把模型推向“通用生成分布”，对 close-ended 可能反而变差。

---

## 3) 为什么 70000 steps 最优（但只有49%）

这个现象与当前代码逻辑高度一致：

1. **前期（<70k）**：projector + LoRA 仍在快速适配，欠拟合明显。
2. **中期（~70k）**：视觉-语言对齐、生成格式、VQA决策暂时达到平衡，出现局部峰值。
3. **后期（>70k）**：
   - 多任务训练继续拉扯 close-ended 目标；
   - 部分2D缺失样本持续注入噪声；
   - 评估脚本 `max_new_tokens=5` + `[A-D]` 正则抓取过严，格式偏差被计错。

因此：70000 是“局部最优检查点”，49% 是当前 pipeline 的“结构性上限”表现，而不是偶然值。

---

## 4) 可落地的解决方案（按优先级）

### P0（必须先做）
1. **修正双路视觉融合**：在 `encode_images()` 中不要只取 tuple 第一路，至少恢复双路合并逻辑（如 stage1/stage2 分路 projector 后 concat 或门控融合）。
2. **保证 `mm_projector2` 策略一致**：若启用 dual 分支，就显式训练/加载 `mm_projector2`；否则统一改为单路配置，避免“名义双路、实际单路”。
3. **修复2D特征缺失**：补齐 `_2D.npy`，把零向量兜底比例降到接近0。

### P1（显著提分）
4. **close-ended 专项训练收尾**：混训后追加 close-ended-only SFT（例如 5k~20k steps）。
5. **训练采样重加权**：提高 close-ended VQA 采样权重（建议 2~4x），降低 caption/open 比例。
6. **checkpoint 密集评测**：40k~100k 每 2k 离线评估一次，精确找峰值而不是只看大步长。

### P2（评测口径修正）
7. **答案归一化后再判分**：支持 `A.`、`Option A`、`答案是A` 等格式，减少假阴性。
8. **close-ended 推理放宽 token 上限**：`max_new_tokens` 可从 5 提到 8~12，降低截断造成的格式错误。

---

## 5) 一句话总结

你现在遇到的“stage3_2 70000 steps 最高且仅49%”主要由：
**双路视觉在实现上未充分生效 + 多任务冲突 + 2D特征缺失 + 严格判分口径** 共同导致；先修 P0，再做 P1/P2，通常会比盲目延长训练步数更有效。


---

## 6) 是否需要从 stage2 重新训练？

**短答：通常不需要立刻重训 stage2。**

优先顺序建议：

1. 先修 `stage3` 的实现/训练问题（双路融合、`mm_projector2` 一致性、2D特征缺失、close-ended收尾SFT、评估口径）。
2. 只有在完成上面修复后，close-ended 仍长期卡在低位（例如持续 <55%）时，再考虑重训 stage2。

为什么先不重训 stage2：
- 现有证据显示主要瓶颈在 stage3 与 eval 口径，而不是 stage2 视觉底座本身。
- stage2 日志里的高对齐指标说明底座并非完全失效；当前更像“下游对齐与使用方式”问题。

什么时候该重训 stage2：
- 你确认双路视觉在 stage3 已正确用上，且 2D 特征缺失基本清零；
- close-ended 专项微调也做过，但仍无明显提升；
- 此时再重训 stage2（或替换更强视觉预训练）才更有性价比。


---

## 7) `mm_projector` 和 `mm_projector2` 是什么？

### `mm_projector`
- 作用：把视觉塔输出的 patch 特征映射到 LLM hidden space（即把视觉 token 变成语言模型可直接消费的 token）。
- 在你的配置中，类型是 `VisualPacker_3d_phi_v3`。

### `mm_projector2`
- 作用：给“第二路视觉分支”准备的并行 projector。
- 只有当 `vision_tower` 是 `vit_stage2_dual_encoders` 且 `use_parallel_projector=True` 时才会创建。

### 为什么会有两个 projector
- 双路视觉（stage1/stage2）出来的是两路特征分布，分别用独立 projector 更容易各自对齐到 LLM 空间。
- 如果只用一个 projector 处理两路，往往会出现分布挤压，导致某一路信息被弱化。

### 你这个仓库里的对应关系
- `mm_projector`：默认第一路视觉特征。
- `mm_projector2`：第二路视觉特征；如果不存在则回退用 `mm_projector`。

简单说：
- **projector = 视觉到语言的翻译器**；
- **两个 projector = 两个翻译器，分别翻译两路视觉信号，再做融合**。

### 和 Spatial Packer 的关系
- 在你当前配置里，`mm_projector_type` 设为 `VisualPacker_3d_phi_v3`。
- 也就是说：**`mm_projector` 本体就是 Spatial Packer（该类实例）**。
- 若启用并行双路，则 `mm_projector2` 是第二个 Spatial Packer 实例（结构相同，参数独立）。
- 流程上可理解为：
  1) 视觉塔输出 patch token；
  2) Spatial Packer(`mm_projector`/`mm_projector2`)做下采样+映射到 LLM 维度；
  3) 再把得到的视觉 token 注入到 LLM 输入。



---

## 8) 改了 `lamed_phi3.py` / `train.py` 就一定涨点吗？

**不能保证“一定”涨点**。这两处修改是“必要但不充分条件”。

- 为什么不是充分条件：
  1) 训练数据本身噪声（如 2D 特征缺失、标签质量、样本分布）仍会限制上限；
  2) 多任务混训权重不调整时，close-ended 目标仍会被稀释；
  3) eval 口径若仍过严，格式问题会继续吃掉准确率。

- 但为什么仍值得改：
  - 这些修改先把“架构实现与设计目标不一致”的硬伤修掉，避免你在错误实现上继续调参。

- 建议验证方式（务必做 A/B）：
  1) 固定同一数据、同一 seed、同一训练步数；
  2) 对比改前/改后在 40k~100k 的 close-ended 曲线；
  3) 同时统计格式错判率（输出非规范 A/B/C/D 的比例）。

如果 A/B 后仍无提升，再回到数据与采样权重层面继续优化。


---

## 9) 你这次报错（NCCL Duplicate GPU）怎么处理

错误 `Duplicate GPU detected` 不是模型结构报错，而是 **DDP 进程数与可见 GPU 映射冲突**。

已做的脚本修复：
- 移除了 `train_3-1.sh` / `train_3-2.sh` 里硬编码的 `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`；
- 改为自动读取当前可见 GPU，并让 `--num_processes` 跟可见 GPU 数一致；
- 设置 `CUDA_DEVICE_ORDER=PCI_BUS_ID` 降低设备映射歧义。

你运行时建议：
1. 先手动设置你要用的卡，例如 `export CUDA_VISIBLE_DEVICES=0,1,2,3`；
2. 可选指定 `export NUM_PROCESSES=4`；
3. 再执行训练脚本。


---

## 10) `train_3-1` 和 `train_3-2` 分别用哪个数据集？

按你当前仓库脚本 + `train.py` 实际逻辑：

### `train_3-1.sh`
- 脚本里传入：
  - `cap_data_path`（M3D-Cap）
  - `vqa_data_train_path`（M3D-VQA train）
- `train.py` 中训练集固定是 `UniDatasets(data_args, tokenizer, mode='train')`，而 `UniDatasets` =
  1) `CapDataset`
  2) `VQADataset(close_ended=True)`
  3) `VQADataset(close_ended=False)`
- 所以 **train_3-1 实际也是三者混训**（不是只训 projector 对齐数据）。

### `train_3-2.sh`
- 同样传入 `cap_data_path` + `vqa_data_train_path`。
- 训练集同样走 `UniDatasets`，因此 **也是 Cap + Close VQA + Open VQA 混训**。

### 关键点
- 你当前这两个脚本在“训练数据构成”上本质一致，主要差异是训练超参（如 LoRA、epoch、batch size、lr、eval/save 策略），而不是数据集类型本身。


---

## 11) 建议：`train_3-1` / `train_3-2` 分别该用什么数据集

### `train_3-1`（对齐期）
目标：先把视觉 token 映射到语言空间，避免一上来被多任务拉扯。

建议数据：
- **主数据：Caption（M3D-Cap）**
- **可选少量：Close-ended VQA（10%~30%）** 用于提前对齐“选项格式”
- **不建议主训：Open-ended VQA**（容易把 projector 对齐阶段变成生成偏置学习）

一句话：`train_3-1` 以 Caption 为主最稳。

### `train_3-2`（任务微调期）
目标：冲 close-ended VQA 指标。

建议数据：
- **主数据：Close-ended VQA（60%~80%）**
- **辅助：Caption（10%~30%）** 防止语言退化
- **少量：Open-ended VQA（0%~20%）** 仅作泛化正则

一句话：`train_3-2` 应该以 Close-ended VQA 为主，不要三者等权。

### 推荐起始配比（可直接试）
- `train_3-1`: `Cap : Close : Open = 8 : 2 : 0`
- `train_3-2`: `Cap : Close : Open = 2 : 7 : 1`

如果你的目标是 `eval_close_vqa.csv`，`train_3-2` 的 close-ended 占比要继续上调。


---

## 12) 你贴的 stage3_1 日志是否正常？（loss≈2.5，epoch≈0.03~0.04）

**是正常的（至少从你给的片段看）。**

原因：
1. 这属于训练最早期（3%~4%），语言建模交叉熵在 2.x 很常见；
2. `learning_rate` 仍接近初始值（`~1e-4`），说明处于早期调参阶段，loss 小幅上下波动正常；
3. `grad_norm` 在 0.5~0.8 附近，没有爆炸迹象。

什么时候不正常：
- loss 长时间（例如 20%~30% 训练进度后）仍完全不下降；
- `grad_norm` 长期飙升或频繁 NaN；
- 生成样本持续空输出/乱输出。

建议：
- 不要只看前 300~500 step，至少观察到 2k~5k step 再判断趋势；
- 同时看一个固定小验证集的 close-ended 准确率曲线。


---

## 13) 只用 CapDataset 时，stage3_1 在 13%~14% 仍是 loss≈2.4~2.5 正常吗？

**大概率正常。**

你这段日志特征：
- `loss` 在 2.40~2.50 区间小幅震荡；
- `grad_norm` 大约 0.34~0.64；
- `learning_rate` 从 `9.73e-5` 缓慢降到 `9.70e-5`（余弦调度早期）。

这通常表示：
1. 训练稳定（无 NaN、无梯度爆炸）；
2. 仍在“早中期平台整理”而非明显发散；
3. 仅训练 projector（或少量参数）时，loss 下降会比全参数微调更慢。

- 你这次补充到 25%~26% 的日志仍在 `2.35~2.50` 区间且 `grad_norm` 平稳，依旧更像稳定平台而非异常发散。
- 到 0.94~0.96 epoch 时若仍在 `2.17~2.46` 且学习率已降到 `~5e-7`，通常说明进入收敛尾段：训练稳定但进一步下降空间有限。
建议判断阈值：
- 若到 30%~40% 进度仍长期卡在同一水平，且验证集无提升，再考虑调整：
  - 降低学习率（如 1e-4 -> 5e-5）
  - 提高有效 batch（梯度累积）
  - 增加 close-ended 小比例样本进行格式对齐


---

## 14) stage3_2 到底用“混合数据”还是“只用 VQA”？

如果你的核心指标是 `eval_close_vqa.csv`：

- **首选方案：以 VQA 为主，而不是三者等权混合。**
- 最实用做法是两段式：
  1) 先用“轻混合”稳住语言（例如 `Cap:Close:Open = 2:7:1`）
  2) 再用 **Close-ended VQA-only** 做收尾（最后 5k~20k step）

什么时候只用 VQA：
- 你主要追 close-ended 指标排名；
- 且已经有可用的 caption 能力，不担心描述能力轻微回落。

什么时候继续混合：
- 你同时要求 caption/open-vqa 的多任务能力；
- 且 close-ended 不是唯一 KPI。

一句话：
- **冲 close-ended 分数：stage3_2 后半段建议 VQA-only。**
- **要多任务平衡：stage3_2 保留混合，但 close-ended 占比至少 60%~80%。**

现在代码已支持 `--dataset_mix_mode`：`mix / cap_only / vqa_only / close_open_vqa`，可直接在脚本里切换。


---

## 15) 只用 close-ended VQA 时 loss 很快到 0.00x 是否正常？

你贴的现象（`loss` 很快从 0.18 -> 0.05 -> 0.006）在 close-ended-only 里**可能发生**，但不代表真实能力大幅提升。

常见原因：
1. close-ended 答案空间很小（A/B/C/D + 高频短文本），token 预测任务本身更容易；
2. 监督目标是“完整答案串”，模型容易先学会格式模板，导致 loss 迅速变小；
3. 训练 loss 与最终 `eval_close_vqa` 并不一一对应。

如何避免“低 loss 假繁荣”：
- 训练时优先监督 `Answer Choice`（A/B/C/D）而不是长答案描述；
- 评估时固定看 close-ended 准确率曲线，而不是只看 CE loss；
- 保留少量混合样本（或后半段再混合）防止过拟合模板。

判断标准：
- 若 loss 很低但 `eval_close_vqa` 不升，说明学到的是“格式捷径”而非判别能力。


---

## 16) 你说“看起来过拟合”怎么处理（已给代码开关）

你这个 close-ended-only 低 loss 现象，确实可能是“模板过拟合”。

已新增开关：
- `--vqa_answer_mode full|choice_only`（默认 `full`）
- 当设为 `choice_only` 时，close-ended 只监督 `A/B/C/D`，不再监督整段长答案文本。

推荐：
- stage3_2 若以 close-ended 指标为核心，优先 `--vqa_answer_mode choice_only`；
- 同时继续看 `eval_close_vqa`，不要只看训练 loss。
