
# import os
# import logging
# import sys
# from typing import Optional, List
# import torch
# import transformers
# from transformers import AutoTokenizer, AutoModel
# from dataclasses import dataclass, field
# import torch.nn as nn
# from safetensors.torch import load_file

# # Add project root
# sys.path.append("/data/esh/HSENet/Preprint")

# # Import Models
# from LaMed.src.model.language_model import LamedPhi3ForCausalLM 
# from LaMed.src.train.lamed_trainer import LaMedTrainer
# from LaMed.src.dataset.dataset import UniDatasets, TextDatasets, CapDataset

# from peft import LoraConfig, get_peft_model

# local_rank = None

# def rank0_print(*args):
#     if local_rank == 0:
#         print(*args)

# @dataclass
# class ModelArguments:
#     version: Optional[str] = field(default="v0")
#     model_name_or_path: Optional[str] = field(default="/data/esh/HSENet/phi-4-mini-instruct/Phi-4-mini-instruct", metadata={"help": "LLM path"})
#     stage_mode: str = field(default="alignment", metadata={"help": "Options: 'alignment' (3.1) or 'finetune' (3.2)"})
#     vision_tower: str = field(default="vit_stage2_dual_encoders") 
#     pretrained_vision_tower_path: str = field(default="/data/esh/HSENet/Preprint/LaMed/output/stage2/model.safetensors")
#     freeze_vision_tower: bool = field(default=True)
#     image_channel: int = field(default=1)
#     image_size: tuple = field(default=(32, 256, 256))
#     patch_size: tuple = field(default=(4, 16, 16))
#     vision_select_layer: int = field(default=-1)
#     vision_select_feature: str = field(default="patch")
#     use_parallel_projector: bool = field(default=True) 
#     remain_2d3d_ViT_type: str = field(default='dual_vits')
#     mm_projector_type: str = field(default='VisualPacker_3d_phi_v3') 
#     proj_layer_type: str = field(default="mlp")
#     proj_layer_num: int = field(default=2)
#     proj_pooling_type: str = field(default="spatial")
#     proj_pooling_size: int = field(default=2)
#     pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
#     tune_mm_mlp_adapter: bool = field(default=True) 
#     segmentation_module: str = field(default="segvol")
#     pretrain_seg_module: str = field(default=None)

# @dataclass
# class DataArguments:
#     data_root: str = field(default="/data/esh/HSENet/m3d_data")
#     cap_data_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-Cap/M3D_Cap/M3D_Cap.json")
#     vqa_data_train_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_train_new.csv")
#     vqa_data_val_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_val_new.csv")
#     seg_data_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_Seg/M3D_Seg")
#     refseg_data_train_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_RefSeg_npy/M3D_RefSeg.csv")
#     refseg_data_test_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_RefSeg_npy/M3D_RefSeg_test.csv")
#     proj_out_num: int = 144 
#     seg_enable: bool = False
#     max_length: int = 2048

# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     lora_enable: bool = False
#     lora_r: int = 16
#     lora_alpha: int = 32
#     lora_dropout: float = 0.05
#     lora_bias: str = "none"
#     cache_dir: Optional[str] = field(default=None)
#     model_max_length: int = field(default=2048)
#     bf16: bool = True
#     output_dir: str = "./output/stage3"
#     num_train_epochs: float = 3
#     per_device_train_batch_size: int = 1
#     gradient_accumulation_steps: int = 4
#     save_strategy: str = "steps"
#     save_steps: int = 500
#     save_total_limit: int = 2
#     learning_rate: float = 2e-5
#     weight_decay: float = 0.0
#     warmup_ratio: float = 0.03
#     lr_scheduler_type: str = "cosine"
#     logging_steps: float = 1
#     gradient_checkpointing: bool = True
#     dataloader_num_workers: int = 8
#     report_to: str = "tensorboard"
#     remove_unused_columns: bool = False

# def find_all_linear_names(model):
#     cls = torch.nn.Linear
#     lora_module_names = set()
#     ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_module']
#     for name, module in model.named_modules():
#         if any(keyword in name for keyword in ignore_keywords): continue
#         if isinstance(module, cls): lora_module_names.add(name)
#     return list(lora_module_names)

# def compute_metrics(eval_preds): return {"accuracy": 0.0}
# def preprocess_logits_for_metrics(logits, labels): return torch.argmax(logits, dim=-1)

# @dataclass
# class DataCollator:
#     def __init__(self, seg_enable, pad_token_id, model_max_length=2048):
#         self.seg_enable = seg_enable
#         self.pad_token_id = pad_token_id
#         self.model_max_length = model_max_length
        
#     def __call__(self, batch: list) -> dict:
#         # 1. 过滤掉 Dataset 加载失败返回的 None
#         valid_batch = [b for b in batch if b is not None]
        
#         # 2. [FIX] 如果 Batch 全空，生成能产生梯度的 Dummy 数据
#         if len(valid_batch) == 0:
#             L = self.model_max_length
#             print(f"[Warning] BATCH IS EMPTY! Generating functional {L}-len dummy data.")
#             dummy = {
#                 'image': torch.zeros((1, 32, 256, 256), dtype=torch.float),
#                 # 必须让 label 的某一位不为 -100，且 attention_mask 为 1，才能产生梯度
#                 'input_id': torch.full((L,), self.pad_token_id, dtype=torch.long),
#                 'label': torch.full((L,), -100, dtype=torch.long),
#                 'attention_mask': torch.ones((L,), dtype=torch.long), # 全 1
#                 'image_2d': torch.zeros((4, 768), dtype=torch.float)
#             }
#             dummy['label'][0] = 0 # 强制产生极小 Loss 以维持梯度同步
#             if self.seg_enable: dummy['seg'] = torch.zeros((1, 32, 256, 256), dtype=torch.int)
#             valid_batch = [dummy]

#         # 3. 标准堆叠逻辑
#         data_dict = {
#             'images': torch.stack([b['image'] for b in valid_batch]),
#             'input_ids': torch.stack([b['input_id'] for b in valid_batch]),
#             'labels': torch.stack([b['label'] for b in valid_batch]),
#             'attention_mask': torch.stack([b['attention_mask'] for b in valid_batch]),
#             'images_2d': torch.stack([b['image_2d'] for b in valid_batch]) 
#         }

#         if self.seg_enable:
#             data_dict['segs'] = torch.stack([
#                 b['seg'] if 'seg' in b else torch.zeros((1, 32, 256, 256), dtype=torch.int) 
#                 for b in valid_batch
#             ])

#         return data_dict
# # ... 前面的代码保持不变 ...

# # def main():
# #     global local_rank
# #     parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
# #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
# #     local_rank = training_args.local_rank
# #     rank0_print(f"Current Stage: {model_args.stage_mode.upper()}")

# #     # 1. Tokenizer & Model Loading
# #     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right", use_fast=False)
# #     tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]})
# #     tokenizer.add_tokens("[SEG]")
# #     if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.unk_token
    
# #     model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
# #     model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
# #     model_args.vocab_size = len(tokenizer)
# #     data_args.max_length = training_args.model_max_length

# #     model = LamedPhi3ForCausalLM.from_pretrained(
# #         model_args.model_name_or_path,
# #         cache_dir=training_args.cache_dir,
# #         torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
# #     )
# #     model.config.seg_token_id = model_args.seg_token_id
# #     model.config.use_cache = False
    
# #     model.get_model().initialize_vision_modules(model_args=model_args)
# #     if model_args.stage_mode == "finetune":
# #         model.get_model().initialize_seg_modules(model_args=model_args)

# #     model_args.num_new_tokens = 4
# #     model.initialize_vision_tokenizer(model_args, tokenizer)

# #     # 2. Weights Loading
# #     if model_args.pretrained_vision_tower_path:
# #         stage2_dict = load_file(model_args.pretrained_vision_tower_path) if model_args.pretrained_vision_tower_path.endswith(".safetensors") else torch.load(model_args.pretrained_vision_tower_path, map_location="cpu")
# #         vision_dict = {k.replace("vision_encoder.", ""): v for k, v in stage2_dict.items() if "vision_encoder." in k}
# #         model.get_model().vision_tower.load_state_dict(vision_dict, strict=False)

# #     if model_args.stage_mode == "finetune" and model_args.pretrain_mm_mlp_adapter:
# #         proj_ckpt = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
# #         model.load_state_dict(proj_ckpt, strict=False)

# #     # 3. LoRA & Checkpointing Configuration
# #     model.requires_grad_(False) 
# #     if model_args.stage_mode == "alignment":
# #         for p in model.get_model().mm_projector.parameters(): p.requires_grad = True
# #     elif model_args.stage_mode == "finetune":
# #         for p in model.get_model().mm_projector.parameters(): p.requires_grad = True
# #         if hasattr(model.get_model(), "seg_module"):
# #             for p in model.get_model().seg_module.parameters(): p.requires_grad = True
# #         if training_args.lora_enable:
# #             lora_config = LoraConfig(r=training_args.lora_r, lora_alpha=training_args.lora_alpha, target_modules=find_all_linear_names(model), lora_dropout=training_args.lora_dropout, bias=training_args.lora_bias, task_type="CAUSAL_LM")
# #             model = get_peft_model(model, lora_config)
# #             for n, p in model.named_parameters():
# #                 if any(k in n for k in ['mm_projector', 'seg_module']): p.requires_grad = True

# #     # [CRITICAL FIX 1] 显式配置 Gradient Checkpointing
# #     if training_args.gradient_checkpointing:
# #         # 必须在 Trainer 初始化之前完成，并设置 use_reentrant=False
# #         model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
# #         model.enable_input_require_grads()

# #     # 4. Trainer Initialization
# #     data_args.proj_out_num = model.get_model().mm_projector.proj_out_num
# #     data_args.seg_enable = (model_args.stage_mode == "finetune")
# #     train_dataset = TextDatasets(data_args, tokenizer, mode='train') if model_args.stage_mode == "alignment" else UniDatasets(data_args, tokenizer, mode='train')
# #     eval_dataset = CapDataset(data_args, tokenizer, mode='validation')
# #     data_collator = DataCollator(data_args.seg_enable, tokenizer.pad_token_id, training_args.model_max_length)

# #     # [CRITICAL FIX 2] 强制 Trainer 配置 DDP 行为
# #     # 注意：ddp_find_unused_parameters 必须为 False 才能配合 static graph
# #     training_args.ddp_find_unused_parameters = False

# #     trainer = LaMedTrainer(
# #         model=model,
# #         args=training_args,
# #         data_collator=data_collator,
# #         train_dataset=train_dataset,
# #         eval_dataset=eval_dataset,
# #         compute_metrics=compute_metrics,
# #         preprocess_logits_for_metrics=preprocess_logits_for_metrics
# #     )

# #     # Enable gradient checkpointing before wrapping with DDP
# #     if training_args.gradient_checkpointing:
# #         model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
# #         model.enable_input_require_grads()

# #     # Wrap model with DistributedDataParallel
# #     if training_args.world_size > 1:
# #         from torch.nn.parallel import DistributedDataParallel as DDP
# #         model = DDP(model, find_unused_parameters=True)
# #     # [CRITICAL FIX 3] 解决 "marked ready twice" 报错的最核心代码
# #     if training_args.world_size > 1:
# #         # 获取被 DDP 包装后的底层模型并设置静态图
# #         # Trainer 会在创建时进行 accelerator 准备
# #         from torch.nn.parallel import DistributedDataParallel as DDP
# #         if hasattr(trainer.model, "_set_static_graph"):
# #             trainer.model._set_static_graph()
# #         rank0_print("DDP Static Graph optimization enabled to prevent double-marking error.")

# #     rank0_print("Starting Training...")
# #     trainer.train()
    
# #     # 5. Saving Logic
# #     rank0_print(f"Saving to {training_args.output_dir}")
# #     if model_args.stage_mode == "alignment":
# #         weight_to_save = {k: v for k, v in model.state_dict().items() if "mm_projector" in k}
# #         torch.save(weight_to_save, os.path.join(training_args.output_dir, "mm_projector.bin"))
# #     else:
# #         model.save_pretrained(training_args.output_dir)
# #         non_lora = {k: v for k, v in model.state_dict().items() if any(x in k for x in ["mm_projector", "seg_module"])}
# #         torch.save(non_lora, os.path.join(training_args.output_dir, "non_lora_weights.bin"))

# # if __name__ == "__main__":
# #     main()
# def main():
#     global local_rank
#     parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#     local_rank = training_args.local_rank
    
#     # 强制设置：无分割任务
#     data_args.seg_enable = False 

#     # 1. Tokenizer & Model Loading
#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right", use_fast=False)
#     tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]})
#     if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.unk_token
    
#     model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
#     model_args.vocab_size = len(tokenizer)

#     model = LamedPhi3ForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
#     )
#     model_args.num_new_tokens = 4
    
#     model.get_model().initialize_vision_modules(model_args=model_args)
#     model.initialize_vision_tokenizer(model_args, tokenizer)

#     # 加载预训练好的 3D-SGAT (Stage 2)
#     if model_args.pretrained_vision_tower_path:
#         rank0_print(f"Loading Pretrained 3D-SGAT: {model_args.pretrained_vision_tower_path}")
#         stage2_dict = load_file(model_args.pretrained_vision_tower_path) if model_args.pretrained_vision_tower_path.endswith(".safetensors") else torch.load(model_args.pretrained_vision_tower_path, map_location="cpu")
#         vision_dict = {k.replace("vision_encoder.", ""): v for k, v in stage2_dict.items() if "vision_encoder." in k}
#         model.get_model().vision_tower.load_state_dict(vision_dict, strict=False)

#     # --- [核心改动：解冻 Projector 和 LLM] ---
#     model.requires_grad_(False) 
    
#     # 1. 解冻 Spatial Packer (Projector)
#     rank0_print("Unfreezing Spatial Packer...")
#     for p in model.get_model().mm_projector.parameters():
#         p.requires_grad = True
    
#     # 2. 解冻 LLM (通过 LoRA，这在 VQA/Caption 任务中最高效)
#     if training_args.lora_enable:
#         rank0_print("Adding LoRA to LLM for joint training...")
#         lora_config = LoraConfig(
#             r=training_args.lora_r,
#             lora_alpha=training_args.lora_alpha,
#             target_modules=find_all_linear_names(model),
#             lora_dropout=training_args.lora_dropout,
#             bias=training_args.lora_bias,
#             task_type="CAUSAL_LM"
#         )
#         model = get_peft_model(model, lora_config)
#         # 确保 Projector 在 PEFT 包装后依然保持可训练
#         for n, p in model.named_parameters():
#             if 'mm_projector' in n:
#                 p.requires_grad = True

#     # [FIX] DDP 兼容性配置
#     if training_args.gradient_checkpointing:
#         model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
#         model.enable_input_require_grads()

#     # 4. Trainer Initialization
#     train_dataset = TextDatasets(data_args, tokenizer, mode='train')
#     eval_dataset = CapDataset(data_args, tokenizer, mode='validation')
#     data_collator = DataCollator(False, tokenizer.pad_token_id, training_args.model_max_length)

#     training_args.ddp_find_unused_parameters = False # 配合 static_graph 提高性能

#     trainer = LaMedTrainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         compute_metrics=compute_metrics,
#         preprocess_logits_for_metrics=preprocess_logits_for_metrics
#     )

#     # [FIX] 解决 Marked Ready Twice 报错
#     if training_args.world_size > 1:
#         if hasattr(trainer.model, "_set_static_graph"):
#             trainer.model._set_static_graph()
#         rank0_print("DDP Static Graph enabled for Joint Training.")

#     rank0_print("Starting Joint Training (Projector + LLM)...")
#     trainer.train()
    
#     # 5. Saving Logic
#     model.save_pretrained(training_args.output_dir)
#     # 额外手动保存 projector 权重，方便后续单独调用
#     projector_weights = {k: v for k, v in model.state_dict().items() if "mm_projector" in k}
#     torch.save(projector_weights, os.path.join(training_args.output_dir, "mm_projector.bin"))

# if __name__ == "__main__":
#     main()

# import os
# import logging
# import sys
# from typing import Optional, List
# import torch
# import transformers
# from transformers import AutoTokenizer, AutoModel
# from dataclasses import dataclass, field
# import torch.nn as nn
# from safetensors.torch import load_file

# # Add project root
# sys.path.append("/data/esh/HSENet/Preprint")

# # Import Models
# from LaMed.src.model.language_model import LamedPhi3ForCausalLM 
# from LaMed.src.train.lamed_trainer import LaMedTrainer
# from LaMed.src.dataset.dataset import UniDatasets, CapDataset

# from peft import LoraConfig, get_peft_model

# local_rank = None

# def rank0_print(*args):
#     if local_rank == 0:
#         print(*args)

# @dataclass
# class ModelArguments:
#     version: Optional[str] = field(default="v0")
#     model_name_or_path: Optional[str] = field(default="/data/esh/HSENet/phi-4-mini-instruct/Phi-4-mini-instruct", metadata={"help": "LLM path"})
#     stage_mode: str = field(default="finetune", metadata={"help": "Options: 'alignment' or 'finetune'"})
#     vision_tower: str = field(default="vit_stage2_dual_encoders") 
#     pretrained_vision_tower_path: str = field(default="/data/esh/HSENet/Preprint/LaMed/output/stage2/model.safetensors")
#     freeze_vision_tower: bool = field(default=True)
#     image_channel: int = field(default=1)
#     image_size: tuple = field(default=(32, 256, 256))
#     patch_size: tuple = field(default=(4, 16, 16))
#     vision_select_layer: int = field(default=-1)
#     vision_select_feature: str = field(default="patch")
#     use_parallel_projector: bool = field(default=True) 
#     remain_2d3d_ViT_type: str = field(default='dual_vits')
#     mm_projector_type: str = field(default='VisualPacker_3d_phi_v3') 
#     proj_layer_type: str = field(default="mlp")
#     proj_layer_num: int = field(default=2)
#     proj_pooling_type: str = field(default="spatial")
#     proj_pooling_size: int = field(default=2)
#     pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
#     tune_mm_mlp_adapter: bool = field(default=True) 
#     segmentation_module: str = field(default="segvol")
#     pretrain_seg_module: str = field(default=None)

# @dataclass
# class DataArguments:
#     data_root: str = field(default="/data/esh/HSENet/m3d_data")
#     cap_data_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-Cap/M3D_Cap/M3D_Cap.json")
#     vqa_data_train_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_train_new.csv")
#     vqa_data_val_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_val_new.csv")
#     seg_data_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_Seg/M3D_Seg")
#     refseg_data_train_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_RefSeg_npy/M3D_RefSeg.csv")
#     refseg_data_test_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_RefSeg_npy/M3D_RefSeg_test.csv")
#     proj_out_num: int = 144 
#     seg_enable: bool = False
#     max_length: int = 2048

# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     lora_enable: bool = True 
#     lora_r: int = 16
#     lora_alpha: int = 32
#     lora_dropout: float = 0.05
#     lora_bias: str = "none"
#     cache_dir: Optional[str] = field(default=None)
#     model_max_length: int = field(default=2048)
#     bf16: bool = True
#     output_dir: str = "./output/stage3"
#     num_train_epochs: float = 3
#     per_device_train_batch_size: int = 1
#     gradient_accumulation_steps: int = 4
#     save_strategy: str = "steps"
#     save_steps: int = 500
#     save_total_limit: int = 2
#     learning_rate: float = 2e-5
#     weight_decay: float = 0.0
#     warmup_ratio: float = 0.03
#     lr_scheduler_type: str = "cosine"
#     logging_steps: float = 1
#     gradient_checkpointing: bool = True
#     dataloader_num_workers: int = 8
#     report_to: str = "tensorboard"
#     remove_unused_columns: bool = False

# def find_all_linear_names(model):
#     cls = torch.nn.Linear
#     lora_module_names = set()
#     ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_module']
#     for name, module in model.named_modules():
#         if any(keyword in name for keyword in ignore_keywords): continue
#         if isinstance(module, cls): lora_module_names.add(name)
#     return list(lora_module_names)

# def compute_metrics(eval_preds): return {"accuracy": 0.0}
# def preprocess_logits_for_metrics(logits, labels): return torch.argmax(logits, dim=-1)

# @dataclass
# class DataCollator:
#     def __init__(self, seg_enable, pad_token_id, model_max_length=2048):
#         self.seg_enable = False 
#         self.pad_token_id = pad_token_id
#         self.model_max_length = model_max_length
        
#     def __call__(self, batch: list) -> dict:
#         valid_batch = [b for b in batch if b is not None and 'image' in b and 'input_id' in b]
        
#         if len(valid_batch) == 0:
#             L = self.model_max_length
#             rank0_print(f"[Warning] BATCH IS EMPTY! Generating functional {L}-len dummy data.")
#             dummy = {
#                 'image': torch.zeros((1, 32, 256, 256), dtype=torch.float),
#                 'input_id': torch.full((L,), self.pad_token_id, dtype=torch.long),
#                 'label': torch.full((L,), -100, dtype=torch.long),
#                 'attention_mask': torch.ones((L,), dtype=torch.long), 
#                 'image_2d': torch.zeros((4, 768), dtype=torch.float)
#             }
#             dummy['label'][0] = 0 
#             valid_batch = [dummy]

#         data_dict = {
#             'images': torch.stack([b['image'] for b in valid_batch]),
#             'input_ids': torch.stack([b['input_id'] for b in valid_batch]),
#             'labels': torch.stack([b['label'] for b in valid_batch]),
#             'attention_mask': torch.stack([b['attention_mask'] for b in valid_batch]),
#             'images_2d': torch.stack([b['image_2d'] for b in valid_batch]) 
#         }

#         if self.seg_enable:
#             data_dict['segs'] = torch.stack([torch.zeros((1, 32, 256, 256), dtype=torch.int) for _ in valid_batch])
            
#         return data_dict

# def main():
#     global local_rank
#     parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#     local_rank = training_args.local_rank
#     rank0_print(f"Current Stage Mode: {model_args.stage_mode.upper()}")
    
#     data_args.seg_enable = False 

#     # 1. Loading Tokenizer & Model
#     tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=training_args.cache_dir, model_max_length=training_args.model_max_length, padding_side="right", use_fast=False)
#     tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]})
#     if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.unk_token
    
#     model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
#     model_args.vocab_size = len(tokenizer)

#     model = LamedPhi3ForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
#     )
#     model.config.use_cache = False
#     model_args.num_new_tokens = 4
#     model.get_model().initialize_vision_modules(model_args=model_args)
#     model.initialize_vision_tokenizer(model_args, tokenizer)

#     # Load 3D-SGAT weights
#     if model_args.pretrained_vision_tower_path and os.path.exists(model_args.pretrained_vision_tower_path):
#         rank0_print(f"Loading Vision Tower: {model_args.pretrained_vision_tower_path}")
#         stage2_dict = load_file(model_args.pretrained_vision_tower_path) if model_args.pretrained_vision_tower_path.endswith(".safetensors") else torch.load(model_args.pretrained_vision_tower_path, map_location="cpu")
#         vision_dict = {k.replace("vision_encoder.", ""): v for k, v in stage2_dict.items() if "vision_encoder." in k}
#         model.get_model().vision_tower.load_state_dict(vision_dict, strict=False)

#     # 2. Joint Training Strategy (LLM LoRA + Projector)
#     model.requires_grad_(False)
#     rank0_print("Joint Training: Unfreezing Projector...")
#     for p in model.get_model().mm_projector.parameters():
#         p.requires_grad = True

#     if training_args.lora_enable:
#         rank0_print("Joint Training: Enabling LoRA for LLM...")
#         lora_config = LoraConfig(
#             r=training_args.lora_r, lora_alpha=training_args.lora_alpha,
#             target_modules=find_all_linear_names(model),
#             lora_dropout=training_args.lora_dropout, bias=training_args.lora_bias, task_type="CAUSAL_LM"
#         )
#         model = get_peft_model(model, lora_config)
#         for n, p in model.named_parameters():
#             if 'mm_projector' in n: p.requires_grad = True

#     # [终极防覆盖修复]
#     if training_args.gradient_checkpointing:
#         model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
#         model.enable_input_require_grads()
#         # 强制阻止 Trainer 内部使用默认配置覆盖掉我们上面的 use_reentrant=False 设置
#         if hasattr(training_args, "gradient_checkpointing_kwargs"):
#             training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
#         else:
#             training_args.gradient_checkpointing = False

#     training_args.ddp_find_unused_parameters = True
    
#     train_dataset = UniDatasets(data_args, tokenizer, mode='train')
#     eval_dataset = CapDataset(data_args, tokenizer, mode='validation')
#     data_collator = DataCollator(data_args.seg_enable, tokenizer.pad_token_id, training_args.model_max_length)

#     trainer = LaMedTrainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         compute_metrics=compute_metrics,
#         preprocess_logits_for_metrics=preprocess_logits_for_metrics
#     )

#     rank0_print("Starting Joint Training Execution...")
#     trainer.train()
    
#     # 4. Save Results
#     rank0_print(f"Saving joint model to {training_args.output_dir}")
#     model.save_pretrained(training_args.output_dir)
#     proj_weights = {k: v for k, v in model.state_dict().items() if "mm_projector" in k}
#     torch.save(proj_weights, os.path.join(training_args.output_dir, "mm_projector.bin"))

# if __name__ == "__main__":
#     main()



##********************************************************************************************************
# import os
# import logging
# import sys
# import numpy as np
# from typing import Optional, List
# import torch
# import transformers
# from transformers import AutoTokenizer, AutoModel
# from dataclasses import dataclass, field
# import torch.nn as nn
# from safetensors.torch import load_file
# from peft import LoraConfig, get_peft_model
# from transformers import TrainerCallback

# # Add project root
# sys.path.append("/data/esh/HSENet/Preprint")

# # Import Models
# from LaMed.src.model.language_model import LamedPhi3ForCausalLM 
# from LaMed.src.train.lamed_trainer import LaMedTrainer
# from LaMed.src.dataset.dataset import UniDatasets, CapDataset

# local_rank = None

# def rank0_print(*args):
#     if local_rank == 0:
#         print(*args)

# @dataclass
# class ModelArguments:
#     version: Optional[str] = field(default="v0")
#     model_name_or_path: Optional[str] = field(default="/data/esh/HSENet/phi-4-mini-instruct/Phi-4-mini-instruct")
#     stage_mode: str = field(default="finetune")
#     vision_tower: str = field(default="vit_stage2_dual_encoders") 
#     pretrained_vision_tower_path: str = field(default="/data/esh/HSENet/Preprint/LaMed/output/stage2/model.safetensors")
#     freeze_vision_tower: bool = field(default=True)
#     image_channel: int = field(default=1)
#     image_size: tuple = field(default=(32, 256, 256))
#     patch_size: tuple = field(default=(4, 16, 16))
#     vision_select_layer: int = field(default=-1)
#     vision_select_feature: str = field(default="patch")
#     use_parallel_projector: bool = field(default=True) 
#     remain_2d3d_ViT_type: str = field(default='dual_vits')
#     mm_projector_type: str = field(default='VisualPacker_3d_phi_v3') 
#     proj_layer_type: str = field(default="mlp")
#     proj_layer_num: int = field(default=2)
#     proj_pooling_type: str = field(default="spatial")
#     proj_pooling_size: int = field(default=2)
#     pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
#     tune_mm_mlp_adapter: bool = field(default=True) 
#     segmentation_module: str = field(default="segvol")
#     pretrain_seg_module: str = field(default=None)

# @dataclass
# class DataArguments:
#     data_root: str = field(default="/data/esh/HSENet/m3d_data")
#     cap_data_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-Cap/M3D_Cap/M3D_Cap.json")
#     vqa_data_train_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_train_new.csv")
#     vqa_data_val_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_val_new.csv")
#     seg_data_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_Seg/M3D_Seg")
#     refseg_data_train_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_RefSeg_npy/M3D_RefSeg.csv")
#     refseg_data_test_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_RefSeg_npy/M3D_RefSeg_test.csv")
#     proj_out_num: int = 144 
#     seg_enable: bool = False
#     max_length: int = 2048

# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     lora_enable: bool = True 
#     lora_r: int = 16
#     lora_alpha: int = 32
#     lora_dropout: float = 0.05
#     lora_bias: str = "none"
#     cache_dir: Optional[str] = field(default=None)
#     model_max_length: int = field(default=2048)
#     bf16: bool = True
#     output_dir: str = "./output/stage3"
#     num_train_epochs: float = 3
#     per_device_train_batch_size: int = 1
#     gradient_accumulation_steps: int = 4
#     save_strategy: str = "steps"
#     save_steps: int = 500
#     save_total_limit: int = 2
#     learning_rate: float = 2e-5
#     weight_decay: float = 0.0
#     warmup_ratio: float = 0.03
#     lr_scheduler_type: str = "cosine"
#     logging_steps: float = 1
#     gradient_checkpointing: bool = True
#     dataloader_num_workers: int = 8
#     report_to: str = "tensorboard"
#     remove_unused_columns: bool = False
#     save_step_names_list: List[int] = field(default_factory=lambda: [500, 1000, 2000, 4000])

# # ================= 补充函数 =================

# def compute_metrics(eval_preds):
#     labels_ids = eval_preds.label_ids
#     pred_ids = eval_preds.predictions
#     labels = labels_ids[:, 1:]
#     preds = pred_ids[:, :-1]
#     labels_flatten = labels.reshape(-1)
#     preds_flatten = preds.reshape(-1)
#     valid_indices = np.where(labels_flatten != -100)
#     filtered_preds = preds_flatten[valid_indices]
#     filtered_labels = labels_flatten[valid_indices]
#     acc_score = sum(filtered_preds==filtered_labels) / len(filtered_labels)
#     return {"accuracy": acc_score}

# def preprocess_logits_for_metrics(logits, labels):
#     if isinstance(logits, tuple): logits = logits[0]
#     pred_ids = torch.argmax(logits, dim=-1)
#     return pred_ids

# def maybe_zero_3(param, ignore_status=False, name=None):
#     from deepspeed import zero
#     from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
#     if hasattr(param, "ds_id"):
#         if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
#             if not ignore_status:
#                 logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
#         with zero.GatheredParameters([param]):
#             param = param.data.detach().cpu().clone()
#     else:
#         param = param.detach().cpu().clone()
#     return param

# def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
#     to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
#     to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
#     return to_return

# def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
#     if getattr(trainer.args, "tune_mm_mlp_adapter", False):
#         keys_to_match = ['mm_projector', 'embed_tokens']
#         weight_to_save = get_mm_projector_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
#         trainer.model.config.save_pretrained(output_dir)
#         current_folder = output_dir.split('/')[-1]
#         parent_folder = os.path.dirname(output_dir)
#         if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
#             if current_folder.startswith('checkpoint-'):
#                 mm_projector_folder = os.path.join(parent_folder, "mm_projector")
#                 os.makedirs(mm_projector_folder, exist_ok=True)
#                 torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
#             else:
#                 torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
#         return

#     if trainer.deepspeed:
#         torch.cuda.synchronize()
#         trainer.save_model(output_dir)
#         return

#     state_dict = trainer.model.state_dict()
#     if trainer.args.should_save:
#         cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
#         del state_dict
#         trainer._save(output_dir, state_dict=cpu_state_dict) 

# def find_all_linear_names(model):
#     cls = torch.nn.Linear
#     lora_module_names = set()
#     ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
#     for name, module in model.named_modules():
#         if any(mm_keyword in name for mm_keyword in ignore_keywords):
#             continue
#         if isinstance(module, cls):
#             lora_module_names.add(name)
#     return list(lora_module_names)

# @dataclass
# class DataCollator:
#     def __init__(self, seg_enable, pad_token_id=0, model_max_length=2048):
#         self.seg_enable = False # 阶段三强制关闭分割
#         self.pad_token_id = pad_token_id
#         self.model_max_length = model_max_length
        
#     def __call__(self, batch: list) -> dict:
#         # 严格过滤以防 KeyError
#         valid_batch = [b for b in batch if b is not None and 'image' in b and 'input_id' in b]
        
#         # Dummy 生成逻辑，确保能产生微小梯度
#         if len(valid_batch) == 0:
#             L = self.model_max_length
#             rank0_print(f"[Warning] BATCH IS EMPTY! Generating functional dummy data.")
#             dummy = {
#                 'image': torch.zeros((1, 32, 256, 256), dtype=torch.float),
#                 'input_id': torch.full((L,), self.pad_token_id, dtype=torch.long),
#                 'label': torch.full((L,), -100, dtype=torch.long),
#                 'attention_mask': torch.ones((L,), dtype=torch.long),
#                 'image_2d': torch.zeros((4, 768), dtype=torch.float)
#             }
#             dummy['label'][0] = 0
#             valid_batch = [dummy]

#         images = torch.stack([b['image'] for b in valid_batch])
#         input_ids = torch.stack([b['input_id'] for b in valid_batch])
#         labels = torch.stack([b['label'] for b in valid_batch])
#         attention_mask = torch.stack([b['attention_mask'] for b in valid_batch])
#         images_2d = torch.stack([b['image_2d'] for b in valid_batch])

#         return_dict = dict(
#             images=images,
#             input_ids=input_ids,
#             labels=labels,
#             attention_mask=attention_mask,
#             images_2d=images_2d,
#         )
#         return return_dict

# # =================================================

# def main():
#     global local_rank
#     parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
#     local_rank = training_args.local_rank

#     rank0_print("vision_tower: ", model_args.vision_tower)
#     rank0_print("mm_projector_type: ", model_args.mm_projector_type)
#     rank0_print("model_max_length: ", training_args.model_max_length)
#     data_args.seg_enable = False 

#     rank0_print("="*20 + " Tokenizer preparation " + "="*20)
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_args.model_name_or_path, cache_dir=training_args.cache_dir,
#         model_max_length=training_args.model_max_length, padding_side="right", use_fast=False,
#     )
#     tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]})
#     tokenizer.add_tokens("[SEG]")
#     if tokenizer.unk_token is not None and tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.unk_token

#     model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
#     model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
#     model_args.vocab_size = len(tokenizer)

#     rank0_print("="*20 + " Model preparation " + "="*20)
#     model = LamedPhi3ForCausalLM.from_pretrained(
#         model_args.model_name_or_path,
#         cache_dir=training_args.cache_dir,
#         torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
#     )
#     model.config.seg_token_id = model_args.seg_token_id
#     model.config.use_cache = False

#     model.get_model().initialize_vision_modules(model_args=model_args)
#     model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
#     model_args.num_new_tokens = 4
#     model.initialize_vision_tokenizer(model_args, tokenizer)

#     # 载入 Stage 2 预训练的 3D-SGAT
#     if model_args.pretrained_vision_tower_path and os.path.exists(model_args.pretrained_vision_tower_path):
#         rank0_print(f"Loading Stage 2 weights: {model_args.pretrained_vision_tower_path}")
#         stage2_dict = load_file(model_args.pretrained_vision_tower_path) if model_args.pretrained_vision_tower_path.endswith(".safetensors") else torch.load(model_args.pretrained_vision_tower_path, map_location="cpu")
#         vision_dict = {k.replace("vision_encoder.", ""): v for k, v in stage2_dict.items() if "vision_encoder." in k}
#         model.get_model().vision_tower.load_state_dict(vision_dict, strict=False)

#     # 联合训练：解冻 Projector
#     model.requires_grad_(False)
#     for p in model.get_model().mm_projector.parameters(): p.requires_grad = True

#     # 联合训练：启用 LLM LoRA
#     if training_args.lora_enable:
#         lora_config = LoraConfig(
#             r=training_args.lora_r, lora_alpha=training_args.lora_alpha,
#             target_modules=find_all_linear_names(model),
#             lora_dropout=training_args.lora_dropout, bias=training_args.lora_bias, task_type="CAUSAL_LM"
#         )
#         model = get_peft_model(model, lora_config)
#         for n, p in model.named_parameters():
#             if 'mm_projector' in n: p.requires_grad = True

#     model.print_trainable_parameters()

#     # [核心防护] 强行覆写重入机制，防止 Trainer 内部重置导致 DDP 崩溃
#     if training_args.gradient_checkpointing:
#         model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
#         model.enable_input_require_grads()
#         if hasattr(training_args, "gradient_checkpointing_kwargs"):
#             training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
#         else:
#             training_args.gradient_checkpointing = False

#     # [核心防护] 允许部分参数无梯度以兼容数据跳过
#     training_args.ddp_find_unused_parameters = True

#     rank0_print("="*20 + " Dataset preparation " + "="*20)
#     data_args.max_length = training_args.model_max_length
#     data_args.proj_out_num = model.get_model().mm_projector.proj_out_num

#     train_dataset = UniDatasets(data_args, tokenizer, mode='train')
#     eval_dataset = CapDataset(data_args, tokenizer, mode='validation')
#     data_collator = DataCollator(data_args.seg_enable, tokenizer.pad_token_id, training_args.model_max_length)

#     # 自定义 Callback
#     class CustomSaveCallback(TrainerCallback):
#         def on_step_end(self, args, state, control, **kwargs):
#             if state.global_step in args.save_step_names_list:
#                 save_idx = state.global_step
#                 save_model_path = os.path.join(args.output_dir, f"saved_{save_idx}")
#                 model_c = kwargs['model'] 
#                 os.makedirs(save_model_path, exist_ok=True)
#                 projector_params = {name: param for name, param in model_c.named_parameters() if 'mm_projector' in name}
#                 lora_params = {name: param for name, param in model_c.named_parameters() if 'lora' in name}
#                 torch.save({**projector_params, **lora_params}, os.path.join(save_model_path, 'pytorch_model.bin'))
#                 torch.save(args, os.path.join(save_model_path, "training_args.bin"))
#                 print(f"step {save_idx} saved to {save_model_path}")   

#     rank0_print("="*20 + " Training " + "="*20)
#     trainer = LaMedTrainer(
#         model=model,
#         args=training_args,
#         data_collator=data_collator,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         compute_metrics=compute_metrics,
#         preprocess_logits_for_metrics=preprocess_logits_for_metrics,
#         callbacks=[CustomSaveCallback()]
#     )

#     trainer.train()
#     trainer.save_state()
#     model.config.use_cache = True

#     rank0_print("="*20 + " Save model " + "="*20)
#     if training_args.lora_enable:
#         projector_params = {name: param for name, param in model.named_parameters() if 'mm_projector' in name}
#         lora_params = {name: param for name, param in model.named_parameters() if 'lora' in name}
#         torch.save({**projector_params, **lora_params}, os.path.join(training_args.output_dir, 'mm_projector_and_lora.bin'))
#         print(f"Model weights saved to {training_args.output_dir}")
#     else:
#         safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

# if __name__ == "__main__":
#     main()





import os
import logging
import sys
import numpy as np
from typing import Optional, List
import torch
import transformers
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass, field
import torch.nn as nn
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback

sys.path.append("/data/esh/HSENet/Preprint")
from LaMed.src.model.language_model import LamedPhi3ForCausalLM 
from LaMed.src.train.lamed_trainer import LaMedTrainer
from LaMed.src.dataset.dataset import UniDatasets, CapDataset

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="/data/esh/HSENet/phi-4-mini-instruct/Phi-4-mini-instruct")
    stage_mode: str = field(default="finetune")
    vision_tower: str = field(default="vit_stage2_dual_encoders") 
    pretrained_vision_tower_path: str = field(default="/data/esh/HSENet/Preprint/LaMed/output/stage2/model.safetensors")
    freeze_vision_tower: bool = field(default=True)
    image_channel: int = field(default=1)
    image_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))
    vision_select_layer: int = field(default=-1)
    vision_select_feature: str = field(default="patch")
    use_parallel_projector: bool = field(default=True) 
    remain_2d3d_ViT_type: str = field(default='dual_vits')
    mm_projector_type: str = field(default='VisualPacker_3d_phi_v3') 
    proj_layer_type: str = field(default="mlp")
    proj_layer_num: int = field(default=2)
    proj_pooling_type: str = field(default="spatial")
    proj_pooling_size: int = field(default=2)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=True) 
    segmentation_module: str = field(default="segvol")
    pretrain_seg_module: str = field(default=None)

@dataclass
class DataArguments:
    data_root: str = field(default="/data/esh/HSENet/m3d_data")
    cap_data_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-Cap/M3D_Cap/M3D_Cap.json")
    vqa_data_train_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_train_new.csv")
    vqa_data_val_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_val_new.csv")
    seg_data_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_Seg/M3D_Seg")
    refseg_data_train_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_RefSeg_npy/M3D_RefSeg.csv")
    refseg_data_test_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_RefSeg_npy/M3D_RefSeg_test.csv")
    proj_out_num: int = 144 
    seg_enable: bool = False
    max_length: int = 1024

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    lora_enable: bool = True 
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_bias: str = "none"
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(default=1024)
    bf16: bool = True
    output_dir: str = "./output/stage3"
    num_train_epochs: float = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 1
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 8
    report_to: str = "tensorboard"
    remove_unused_columns: bool = False
    save_step_names_list: List[int] = field(default_factory=lambda: [500, 1000, 2000, 4000])

# ================= 补充函数 =================
def compute_metrics(eval_preds): return {"accuracy": 0.0}
def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple): logits = logits[0]
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            pass
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

def get_mm_projector_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        keys_to_match = ['mm_projector', 'embed_tokens']
        weight_to_save = get_mm_projector_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)
        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict) 

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords): continue
        if isinstance(module, cls): lora_module_names.add(name)
    return list(lora_module_names)

@dataclass
class DataCollator:
    def __init__(self, seg_enable, pad_token_id=0, model_max_length=1024):
        self.seg_enable = False 
        self.pad_token_id = pad_token_id
        self.model_max_length = model_max_length
        
    def __call__(self, batch: list) -> dict:
        valid_batch = [b for b in batch if b is not None and 'image' in b and 'input_id' in b]
        
        if len(valid_batch) == 0:
            L = self.model_max_length
            rank0_print(f"[Warning] BATCH IS EMPTY! Generating functional dummy data.")
            dummy = {
                'image': torch.zeros((1, 32, 256, 256), dtype=torch.float),
                'input_id': torch.full((L,), self.pad_token_id, dtype=torch.long),
                'label': torch.full((L,), -100, dtype=torch.long),
                'attention_mask': torch.ones((L,), dtype=torch.long),
                'image_2d': torch.zeros((32, 512), dtype=torch.float)
            }
            dummy['label'][0] = 0
            valid_batch = [dummy]

        images = torch.stack([b['image'] for b in valid_batch])
        input_ids = torch.stack([b['input_id'] for b in valid_batch])
        labels = torch.stack([b['label'] for b in valid_batch])
        attention_mask = torch.stack([b['attention_mask'] for b in valid_batch])
        images_2d = torch.stack([b['image_2d'] for b in valid_batch])

        return dict(images=images, input_ids=input_ids, labels=labels, attention_mask=attention_mask, images_2d=images_2d)

# =================================================

def main():
    global local_rank
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    rank0_print("vision_tower: ", model_args.vision_tower)
    rank0_print("mm_projector_type: ", model_args.mm_projector_type)
    rank0_print("model_max_length: ", training_args.model_max_length)
    data_args.seg_enable = False 

    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length, padding_side="right", use_fast=False,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]})
    tokenizer.add_tokens("[SEG]")
    
    # [核心修复] 坚决防止 pad 污染
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    model_args.vocab_size = len(tokenizer)

    rank0_print("="*20 + " Model preparation " + "="*20)
    model = LamedPhi3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16 if training_args.bf16 else torch.float32,
        attn_implementation="flash_attention_2"
    )
    # 如果添加了新token，这里需要扩充embedding以防报错
    model.resize_token_embeddings(len(tokenizer))
    
    model.config.seg_token_id = model_args.seg_token_id
    model.config.use_cache = False

    model.get_model().initialize_vision_modules(model_args=model_args)
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model_args.num_new_tokens = 4
    model.initialize_vision_tokenizer(model_args, tokenizer)

    # ================= [终极绝杀：视觉参数精准提取机] =================
# ================= [终极修复：视觉双塔权重精准映射] =================
    if model_args.pretrained_vision_tower_path and os.path.exists(model_args.pretrained_vision_tower_path):
        rank0_print(f"🚀 正在执行手术级权重映射: {model_args.pretrained_vision_tower_path}")
        
        # 支持 safetensors 和 torch 格式
        if model_args.pretrained_vision_tower_path.endswith(".safetensors"):
            stage2_dict = load_file(model_args.pretrained_vision_tower_path)
        else:
            stage2_dict = torch.load(model_args.pretrained_vision_tower_path, map_location="cpu")
        
        vision_dict = {}
        for k, v in stage2_dict.items():
            # 1. 映射 Stage 1 视觉塔 (从 CLIP 的 vision_encoder 提取)
            if k.startswith("stage1_pretrained_CLIP.vision_encoder."):
                new_k = k.replace("stage1_pretrained_CLIP.vision_encoder.", "vision_tower_stage1.")
                vision_dict[new_k] = v
            
            # 2. 映射 Stage 2 视觉塔 (处理 vision_encoder 前缀)
            elif k.startswith("vision_encoder."):
                new_k = k.replace("vision_encoder.", "vision_tower_stage2.")
                vision_dict[new_k] = v
                
            # 3. 映射 Stage 2 视觉塔 (处理处于顶层的 blocks, sgat 等参数)
            elif any(k.startswith(p) for p in ["blocks.", "patch_embedding.", "cls_token", "norm.", "sgat.", "sga.", "sga_adapter", "kd_proj"]):
                new_k = f"vision_tower_stage2.{k}"
                vision_dict[new_k] = v
        
        # 使用 strict=False 加载，因为我们要自动过滤掉权重里自带的 language_encoder (BERT)
        res = model.get_model().vision_tower.load_state_dict(vision_dict, strict=False)
        
        # 关键验证：确保视觉块没有丢失
        missing_vision = [m for m in res.missing_keys if "vision_tower_stage2.blocks.0" in m]
        if missing_vision:
            rank0_print(f"❌ [致命警告] 视觉核心组件加载失败！请检查权重前缀。示例缺失: {missing_vision[:2]}")
        else:
            rank0_print(f"✅ 视觉双塔对齐成功！已成功加载 {len(vision_dict)} 个核心参数。")

    # ================= [终极修复：双路 Projector 并行加载] =================
    if model_args.pretrain_mm_mlp_adapter and os.path.exists(model_args.pretrain_mm_mlp_adapter):
        rank0_print(f"🔗 正在加载双路并行 Projector: {model_args.pretrain_mm_mlp_adapter}")
        projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
        
        cleaned_projector_dict = {}
        for k, v in projector_weights.items():
            # 寻找 mm_projector 或 mm_projector2 的起始位置，保留其完整后缀
            if "mm_projector2" in k:
                new_k = k[k.find("mm_projector2"):]
                cleaned_projector_dict[new_k] = v
            elif "mm_projector" in k:
                new_k = k[k.find("mm_projector"):]
                cleaned_projector_dict[new_k] = v
                
        if len(cleaned_projector_dict) > 0:
            # 这里的加载建议也用 strict=False 兼容 Embedding 等参数
            res = model.get_model().load_state_dict(cleaned_projector_dict, strict=False)
            rank0_print(f"✅ Projector 加载完毕！匹配项: {len(cleaned_projector_dict)}。结果: {res}")
        else:
            rank0_print("❌ [错误] 权重文件中未探测到任何 Projector 参数！")

    model.requires_grad_(False)
    for p in model.get_model().mm_projector.parameters(): p.requires_grad = True

    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r, lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout, bias=training_args.lora_bias, task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
        for n, p in model.named_parameters():
            if 'mm_projector' in n: p.requires_grad = True

    # model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        if hasattr(training_args, "gradient_checkpointing_kwargs"):
            training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        else:
            training_args.gradient_checkpointing = False

    training_args.ddp_find_unused_parameters = True

    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num

    train_dataset = UniDatasets(data_args, tokenizer, mode='train')
    eval_dataset = CapDataset(data_args, tokenizer, mode='validation')
    data_collator = DataCollator(data_args.seg_enable, tokenizer.pad_token_id, training_args.model_max_length)

    class CustomSaveCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step in args.save_step_names_list:
                save_idx = state.global_step
                save_model_path = os.path.join(args.output_dir, f"saved_{save_idx}")
                model_c = kwargs['model'] 
                os.makedirs(save_model_path, exist_ok=True)
                projector_params = {name: param for name, param in model_c.named_parameters() if 'mm_projector' in name}
                lora_params = {name: param for name, param in model_c.named_parameters() if 'lora' in name}
                torch.save({**projector_params, **lora_params}, os.path.join(save_model_path, 'pytorch_model.bin'))
                torch.save(args, os.path.join(save_model_path, "training_args.bin"))
                print(f"step {save_idx} saved to {save_model_path}")   

    rank0_print("="*20 + " Training " + "="*20)
    trainer = LaMedTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[CustomSaveCallback()]
    )

    trainer.train()
    trainer.save_state()
    model.config.use_cache = True

    rank0_print("="*20 + " Save model " + "="*20)
    if training_args.lora_enable:
        projector_params = {name: param for name, param in model.named_parameters() if 'mm_projector' in name}
        lora_params = {name: param for name, param in model.named_parameters() if 'lora' in name}
        torch.save({**projector_params, **lora_params}, os.path.join(training_args.output_dir, 'mm_projector_and_lora.bin'))
        print(f"Model weights saved to {training_args.output_dir}")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

if __name__ == "__main__":
    main()
