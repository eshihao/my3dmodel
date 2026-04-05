#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:/data/esh/HSENet/Preprint"
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 建议：由外部传入 CUDA_VISIBLE_DEVICES。
# 若未设置，则自动使用当前可见 GPU。
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
  CUDA_VISIBLE_DEVICES=$(python - <<'PY'
import torch
print(",".join(str(i) for i in range(torch.cuda.device_count())))
PY
)
  export CUDA_VISIBLE_DEVICES
fi

# 根据可见 GPU 数自动设置进程数，避免固定 8 卡导致 DDP/NCCL 设备冲突
NUM_PROCESSES=${NUM_PROCESSES:-$(python - <<'PY'
import os
v=os.environ.get("CUDA_VISIBLE_DEVICES","").strip()
if not v:
    print(1)
else:
    print(len([x for x in v.split(",") if x.strip()!=""]))
PY
)}

# 阶段 3.1：特征对齐 (Projector Alignment)
accelerate launch --num_processes ${NUM_PROCESSES} --main_process_port 29502 LaMed/src/train/train.py \
    --stage_mode "finetune" \
    --model_name_or_path "/data/esh/HSENet/phi-4-mini-instruct/Phi-4-mini-instruct" \
    --pretrained_vision_tower_path "/data/esh/HSENet/Preprint/LaMed/output/stage2/model.safetensors" \
    --output_dir "/data/esh/HSENet/Preprint/LaMed/output/stage3_1_alignment" \
    --data_root "/data/esh/HSENet/m3d_data" \
    --cap_data_path "/data/esh/HSENet/m3d_data/M3D-Cap/M3D_Cap/M3D_Cap.json" \
    --vqa_data_train_path "/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_train_new.csv" \
    --dataset_mix_mode "cap_only" \
    --seg_data_path "/data/esh/HSENet/m3d_data/M3D-Seg/M3D_Seg" \
    --refseg_data_train_path "/data/esh/HSENet/m3d_data/M3D_RefSeg_npy/M3D_RefSeg.csv" \
    --vision_tower "vit_stage2_dual_encoders" \
    --mm_projector_type "VisualPacker_3d_phi_v3" \
    --segmentation_module "segvol" \
    --lora_enable False \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing True \
    --dataloader_pin_memory True \
    --dataloader_num_workers 24 \
    --report_to "tensorboard"
