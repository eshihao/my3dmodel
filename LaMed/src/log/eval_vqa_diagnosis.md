# VQA Eval 准确率诊断（基于当前代码与日志）

## 1) 你当前的真实模型架构（按代码还原）

1. **LLM 主干**：`LamedPhi3ForCausalLM`（Phi-3 Causal LM）。
2. **视觉塔**：`vit_stage2_dual_encoders`（`ViT3DTower_dual_encoders`），内部同时有：
   - `vision_tower_stage1 = ViT_stage1`
   - `vision_tower_stage2 = ViT_stage2`
3. **2D 特征融合路径**：`vision_tower_stage2(images, slice_features=images_2d)`，说明 2D slice 特征直接进入 stage2 编码分支。
4. **多模态映射**：视觉特征经 `mm_projector`（类型 `VisualPacker_3d_phi_v3`）映射到 LLM hidden 后，插入到文本 embedding。
5. **stage3 可训练参数范围**：在 `train.py` 的有效代码中，`model.requires_grad_(False)` 后只显式打开 `mm_projector` 与 LoRA 参数，视觉塔默认冻结。

---

## 2) 日志告诉我们的关键事实

### A. stage2 很高的 accuracy 不是 VQA accuracy

- `train_stage2.log` 里 `eval_accuracy=0.957...`，但这是 stage2 的图文对齐训练指标，不是 close-ended VQA 选项准确率。

### B. stage3_1 更像“对齐过渡阶段”，不是 VQA 收敛阶段

- `train_3-1.sh` 中：`evaluation_strategy="no"`，无在线验证。
- `train_stage3_1.log` 末段 loss 仍约 2.3~2.4，说明远未达到高质量 VQA 收敛态。

### C. stage3_2 存在混合任务冲突与输入质量波动

- 训练集是 `CapDataset + Close VQA + Open VQA` 直接 concat 混训，close-ended 目标被稀释。
- `train_stage3_2.log` 出现多次“找不到 2D 特征文件，使用零矩阵兜底”警告，代表部分样本视觉条件被弱化。

### D. 你的评估脚本对 close-ended 判分偏“苛刻”

- 推理时 `max_new_tokens=5`。
- 判分逻辑是从生成文本里用正则直接抓 `[A-D]`，格式稍偏就会被记错（如先输出解释、先出现无关字母）。

---

## 3) 为什么“stage3_2 约 70000 steps 最好，也才 49%”

这在你当前 pipeline 下是**典型中期峰值**：

1. **前期欠拟合**：LoRA + projector 刚开始适配，VQA 还没学稳。
2. **中期（~70k）平衡点**：视觉对齐、语言格式、VQA 任务三者短暂达到较佳平衡。
3. **后期退化**：混合任务训练继续推进后，模型更偏向“通用生成/caption/open VQA”分布，close-ended 分类能力被冲淡；再叠加严格判分规则，离线评估准确率继续下降。

因此：
- 70000 step 出现峰值并不奇怪；
- 49% 停在较低水平，也符合“任务目标与训练目标不完全一致 + 判分口径偏严格 + 部分样本2D特征缺失”的组合效应。

---

## 4) 直接可执行的改进建议（按优先级）

1. **把 close-ended VQA 权重提上去**：不要与 caption/open-vqa 等权（建议 2~4 倍采样权重）。
2. **增加 close-ended 专项收尾**：混训后再做一段 close-ended-only SFT（例如 5k~20k steps）。
3. **修复 2D 特征缺失链路**：优先补齐 `_2D.npy`，避免零向量兜底频繁发生。
4. **放宽 eval 判分鲁棒性**：先做答案归一化（`A.`/`Option A`/`答案是A` 等），再提取选项字母。
5. **把 checkpoint 网格加密到峰值区间**：围绕 40k~100k 每 2k 保存+评估，精确定位最优点。

---

## 5) 一句话结论

你的 49% 上限与“只有 stage3_2 的 70000 steps 最好”并非单点 bug，而是：
**stage2 指标与目标不一致 + stage3 混合任务冲突 + stage3 中后期分布漂移 + 部分2D特征缺失 + close-ended判分口径偏严格** 共同导致。
