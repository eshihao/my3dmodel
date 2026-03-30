# VQA Eval 准确率诊断（基于当前代码与日志）

## 1) 现有模型实际架构（与你当前训练/评测流程一致）

- 主干是 **Phi-3 Causal LM**，并通过 `LamedPhi3ForCausalLM` 接入视觉分支。  
- 视觉分支是 `vit_stage2_dual_encoders`，并通过 `mm_projector`（可选 `mm_projector2`）将视觉 token 映射到 LLM 隐空间后插入文本 embedding。  
- 在你当前 `train.py` 的有效代码里，训练时 **冻结了视觉塔**，只训练 `mm_projector` + LLM 的 LoRA 参数。

## 2) 为什么只有 stage3_2 的约 70000 step 最好（且只有 ~49%）

> 结论：这是“任务冲突 + 训练/评估错配 + 评估口径严格 + 中后期退化”的叠加结果，70k 更像一个中期平衡点。

### A. Stage2 指标高，不等于 VQA 强

- `train_stage2.log` 的高 `eval_accuracy=0.957` 来自 CLIP 风格图文对齐任务，不是多选 VQA 选项决策准确率。  
- 因此 stage2 给的是视觉语义底座，不直接保证 close-ended VQA 高分。

### B. Stage3_1 本质上几乎没把 VQA 学好

- 你当前 `train_3-1.sh` 虽写“特征对齐”，但实际调用 `--stage_mode finetune`，且 `evaluation_strategy="no"`，没有在线验证 VQA 泛化。  
- `train_stage3_1.log` 训练 loss 仍在 ~2.3~2.4，说明主要还在做粗对齐/混合任务适配，不是 VQA 收敛态。

### C. Stage3_2 训练目标是混合任务，VQA 被“稀释”

- `UniDatasets` 把 `Caption + Close VQA + Open VQA` 直接 `ConcatDataset` 混训，默认近似 1:1:1。  
- 对 close-ended VQA 来说，模型同时还要学 caption 与 open generation，参数更新方向不完全一致，VQA 专项能力会被稀释。

### D. 评估脚本对答案提取很“苛刻”

- close-ended 判分仅通过正则提取首个 `[A-D]`；如果模型输出解释句、格式偏移、或先输出其他字母，很容易被判错。  
- `max_new_tokens=5` 对某些输出模板也偏紧，增加格式性错误（并非纯知识错误）。

### E. 你给出的“70k 最好”符合典型中期最优现象

- 在联合微调里，早期（如 30k）常欠拟合，后期（如 100k+）容易向训练混合分布过拟合或偏向非 close-ended 目标；中间 step（如 70k）最平衡。  
- 你现有 stage3_2 日志文件只保留了末段（接近 202k），看不到 70k 附近训练态细节，这也解释了“日志里难直接证明 70k 最优”。

## 3) 可操作改进（按优先级）

1. **改训练采样权重**：把 close-ended VQA 采样权重提高（例如 2~4 倍），不要与 caption/open-vqa 完全等权。  
2. **分阶段微调**：先混训，再用 close-ended-only 做短程 SFT（如 5k~20k steps）作为最后对齐。  
3. **评估口径更稳健**：在判分前做更严格答案归一化（优先匹配 `^([A-D])\b`，并处理 `"A."`、`"Option A"` 等）。  
4. **保存点更密**：围绕 40k~100k 每 2k 保存并批量 eval，真实定位峰值区间。  
5. **区分指标**：训练中单独记录 close-ended val acc，避免只看总 loss。

## 4) 对你问题的一句话回答

- 之所以只有 stage3_2 的 70000 steps 达到最高且只有 49%，不是单一 bug，而是：**stage2 与目标指标不一致 + stage3 混合任务冲突 + checkpoint 中后期退化 + 评估提取规则严格** 共同导致；70k 只是当前设置下“欠拟合与过拟合之间”的局部最优点。
