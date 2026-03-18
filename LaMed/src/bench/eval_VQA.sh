#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONPATH="${PYTHONPATH}:/data/esh/HSENet/Preprint"

# 启动 8 卡并行推理
# 注意：我在这里把 batch_size 提到了 2，这意味着 8 张卡同时处理 16 个样本！
accelerate launch --num_processes 8 --main_process_port 29503 eval_VQA.py \
    --close_ended False \
    --batch_size 1
