import os
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from dataclasses import dataclass, field
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM
from LaMed.src.train.lamed_trainer import LaMedTrainer
from LaMed.src.model.language_model import LamedPhi3ForCausalLM
from LaMed.src.dataset.multi_dataset import UniDatasets, CapDataset, TextDatasets, VQADataset
from LaMed.src.model.CLIP import M3DCLIP, M3DCLIPConfig
from safetensors.torch import load_file

# Set up CUDA devices and initialize distributed processing
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import torch.distributed as dist
dist.init_process_group(backend='nccl')

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="microsoft/Phi-3-mini-4k-instruct", metadata={"help": "Path to the LLM or MLLM."})
    pretrained_stage2_path: str = field(default="/path/to/your/model.safetensors", metadata={"help": "Path to the pretrained Stage 2 model."})
    
    vision_tower: str = field(default="vit_stage2_dual_encoders")
    freeze_vision_tower: bool = field(default=True)
    
    mm_projector_type: str = field(default='VisualPacker_3d_phi_v3')
    segmentation_module: str = field(default="segvol")
    
    tune_mm_mlp_adapter: bool = field(default=True)
    lora_enable: bool = field(default=False)

@dataclass
class DataArguments:
    data_root: str = field(default="/path/to/data")
    cap_data_path: str = field(default="/path/to/caption/data")
    vqa_data_train_path: str = field(default="/path/to/vqa_train")
    seg_data_path: str = field(default="/path/to/segmentation/data")

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = field(default="./output")
    num_train_epochs: int = field(default=5)
    per_device_train_batch_size: int = field(default=1)
    learning_rate: float = field(default=5e-5)
    save_steps: int = field(default=1000)

def prepare_model(model_args: ModelArguments, training_args: TrainingArguments, tokenizer: transformers.PreTrainedTokenizer):
    # Load the pretrained Stage 2 model
    model = LamedPhi3ForCausalLM.from_pretrained(
        model_args.pretrained_stage2_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.float16 if training_args.bf16 else torch.float32,
    )

    # Initialize vision modules
    model.get_model().initialize_vision_modules(model_args=model_args)

    # Initialize segmentation modules
    if model_args.segmentation_module:
        model.get_model().initialize_seg_modules(model_args=model_args)

    # Freeze vision tower if specified
    if model_args.freeze_vision_tower:
        for param in model.get_model().vision_tower.parameters():
            param.requires_grad = False

    # Optionally enable LoRA for fine-tuning
    if model_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=['lm_head', 'embed_tokens', 'vision_tower'],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    
    # Initialize tokenizer
    tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]})
    return model

def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Prepare model
    model = prepare_model(model_args, training_args, tokenizer)

    # Data preparation
    train_dataset = UniDatasets(data_args, tokenizer, mode="train")
    eval_dataset = CapDataset(data_args, tokenizer, mode="validation")

    # Data collator
    data_collator = DataCollator(seg_enable=True)

    # Training
    trainer = LaMedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Start training
    trainer.train()

    # Save final model
    trainer.save_model(model_args.output_dir)

if __name__ == "__main__":
    main()
