#!/bin/bash

# Ensure you run `accelerate config` first to configure the multi-GPU setup.

# Set environment variables (adjust accordingly based on your machine configuration)
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONPATH="${PYTHONPATH}:/data/esh/HSENet/Preprint"  # Replace with the path to your LaMed project

# Launch training using accelerate
accelerate launch LaMed/src/train/train_1.py \
    --version v0 \
    --model_name_or_path "/data/esh/HSENet/phi-4-mini-instruct/Phi-4-mini-instruct" \  # Path to your model (e.g., Phi-3-mini or other)
    --pretrained_stage2_path "/data/esh/HSENet/Preprint/LaMed/output/stage2/model.safetensors" \  # Pretrained Stage 2 model
    --vision_tower "vit_stage2_dual_encoders" \  # Vision tower model
    --freeze_vision_tower True \  # Whether to freeze the vision tower
    --mm_projector_type "VisualPacker_3d_phi_v3" \  # Type of the projector for multimodal data
    --segmentation_module "segvol" \  # Segmentation module for multi-task learning
    --tune_mm_mlp_adapter True \  # Whether to fine-tune the multimodal adapter
    --lora_enable True \  # Set to True to enable LoRA fine-tuning (optional)
    --output_dir "./LaMed/output/stage3_finetune" \  # Output directory to save model
    --num_train_epochs 5 \  # Number of training epochs
    --per_device_train_batch_size 2 \  # Batch size per GPU for training
    --per_device_eval_batch_size 2 \  # Batch size per GPU for evaluation
    --gradient_accumulation_steps 2 \  # Number of gradient accumulation steps
    --evaluation_strategy "steps" \  # Evaluate during training
    --eval_steps 500 \  # Evaluate every 500 steps
    --save_strategy "steps" \  # Save model checkpoint every steps
    --save_steps 1000 \  # Save model every 1000 steps
    --save_total_limit 3 \  # Keep at most 3 checkpoints
    --learning_rate 5e-5 \  # Learning rate for optimization
    --weight_decay 0.01 \  # Weight decay for optimization
    --warmup_ratio 0.03 \  # Learning rate warm-up ratio
    --lr_scheduler_type "cosine" \  # Learning rate scheduler
    --logging_steps 100 \  # Log every 100 steps
    --gradient_checkpointing True \  # Enable gradient checkpointing to save memory
    --dataloader_pin_memory True \  # Pin memory for faster data loading
    --dataloader_num_workers 8 \  # Number of data loading workers
    --report_to "tensorboard"  # Enable TensorBoard reporting for monitoring
