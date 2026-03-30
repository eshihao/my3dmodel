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
