# from typing import Optional
# import transformers
# from transformers import Trainer
# from typing import List
# from dataclasses import dataclass, field
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig

# import sys
# sys.path.append("/data/esh/HSENet/Preprint")

# from LaMed.src.model.CLIP_stage1 import M3DCLIP_stage1, M3DCLIPConfig_stage1
# from LaMed.src.dataset.multi_dataset import ITRDataset, CT_RateDataset_stage2, TextDatasets, CapDataset, Stage2CapDataset

# from LaMed.src.model.CLIP_stage2 import M3DCLIP_stage2, M3DCLIPConfig_stage2


# from transformers import BertTokenizer
# import torch
# from safetensors.torch import load_file

# normal_repr = torch.Tensor.__repr__
# torch.Tensor.__repr__ = lambda self: f"{self.shape} {normal_repr(self)}"

# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,5"

# is_use_single_GPU = False

# if is_use_single_GPU:
#     os.environ['RANK'] = '0' 
#     os.environ['WORLD_SIZE'] = '1' 
#     os.environ['MASTER_ADDR'] = 'localhost' 
#     os.environ['MASTER_PORT'] = '12382' 
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#     import torch.distributed as dist
#     dist.init_process_group(backend='nccl')

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True
# torch._dynamo.config.verbose = True
# torch._dynamo.config.disable = True


# @dataclass
# class ModelArguments:
#     version: Optional[str] = field(default="v0")
#     language_model_name_or_path: str = field(default="/data/esh/HSENet/model/bert-base-uncased/")
#     use_mask: bool = field(default=False)  # True False
#     mask_rate: float = field(default=0.08)
#     use_2D_Encoder: bool = field(default=False)  # True False

#     gather_loss: bool = field(default=True, metadata={"help": "Gather all distributed batch data of multiple GPUs and calculate contrastive loss together."})
#     local_loss: bool = field(default=False)
    
#     # load our pretrained stage1 CLIP
#     pretrained_model: str = field(default="/data/esh/HSENet/model/HSENet_CLIP")
#     # initialize stage2_pretrained_CLIP using M3D-CLIP weight
#     stage2_pretrained_CLIP_path: str = field(default="/data/esh/HSENet/model/M3D-CLIP")
#     in_channels: int = field(default=1)
#     img_size: tuple = field(default=(32, 256, 256))
#     patch_size: tuple = field(default=(4, 16, 16))

#     hidden_size: int = field(default=768)
#     mlp_dim: int = field(default=3072)
#     num_layers: int = field(default=12)
#     num_heads: int = field(default=12)
#     pos_embed: str = field(default="perceptron")
#     dropout_rate: float = field(default=0.0)
#     spatial_dims: int = field(default=3)
#     max_text_len: int = field(default=128)
#     vocab_size: int = field(default=30522)


# # @dataclass
# # class DataArguments:
# #     data_root: str = field(default=".", metadata={"help": "Root directory for all data."})
# #     cap_data_path: str = field(default="/data/esh/HSENet/Data/CT_Rate/ct_rate_dataset_ordered_with_biomedclip.json", metadata={"help": "Path to caption data."})
# #     max_length: int = field(default=512)


# # esh do
# @dataclass
# class DataArguments:
#     data_root: str = field(default="/data/esh/HSENet/m3d_data", metadata={"help": "Root directory for all data."})

#     # caption data
#     cap_data_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-Cap/M3D_Cap/M3D_Cap.json", metadata={"help": "Path to caption data."})

#     # VQA data
#     vqa_data_train_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_train.csv", metadata={"help": "Path to training VQA data."})
#     vqa_data_val_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_val.csv", metadata={"help": "Path to validation VQA data."})
#     vqa_data_test_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_test.csv", metadata={"help": "Path to testing VQA data."})

#     vqa_yn_data_train_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_yn_train.csv", metadata={"help": "Path to training VQA Yes or No data."})

#     # positioning & segmentation data
#     seg_data_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_Seg_npy/", metadata={"help": "Path to segmentation data."})
#     refseg_data_train_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_RefSeg_npy/M3D_RefSeg.csv", metadata={"help": "Path to refering segmentation data."})
#     refseg_data_test_path: str = field(default="/data/esh/HSENet/m3d_data/M3D_RefSeg_npy/M3D_RefSeg_test.csv", metadata={"help": "Path to refering segmentation data."})

# @dataclass
# class TrainingArguments(transformers.TrainingArguments):
#     cache_dir: Optional[str] = field(default=None)
#     optim: str = field(default="adamw_torch")
#     remove_unused_columns: bool = field(default=False)

#     ddp_backend: str = "nccl"
#     ddp_find_unused_parameters: bool = True

#     bf16: bool = True
#     output_dir: str = "/data/esh/HSENet/Preprint/LaMed/output/CLIP-202511"
#     save_step_names_list: List[int] = field(default_factory=lambda: [200000])
#     num_train_epochs: int = 10
#     per_device_train_batch_size: int = 3
#     per_device_eval_batch_size: int = 4
#     gradient_accumulation_steps: int = 1
#     evaluation_strategy: str = "steps"
#     eval_accumulation_steps: int = 1
#     eval_steps: float = 0.04
#     save_strategy: str = "steps"
#     save_steps: int = 1000
#     save_total_limit: int = 1
#     learning_rate: float = 1e-4
#     weight_decay: float = 0.1
#     warmup_ratio: float = 0.03
#     lr_scheduler_type: str = "cosine"
#     logging_steps: float = 0.001
#     gradient_checkpointing: bool = False
#     dataloader_pin_memory: bool = True
#     dataloader_num_workers: int = 8
#     report_to: str = "tensorboard"


# def compute_metrics(eval_pred):
#     preds = eval_pred.predictions
#     labels = eval_pred.label_ids
#     correct = (preds == labels).sum()
#     total = labels.size
#     acc = correct / total
#     return {"accuracy": acc}

# def preprocess_logits_for_metrics(logits, labels):
#     try:
#         preds = torch.argmax(logits, dim=-1)
#     except:
#         print("Error in preprocess_logits_for_metrics")
#         print(logits)
#         preds = None
#     return preds

# @dataclass
# class DataCollator:
#     def __init__(self, gather_all):
#         self.gather_all = gather_all

#     def __call__(self, batch: list) -> dict:
#         images, texts, input_ids, attention_mask, images_2d = tuple(
#             [b[key] for b in batch] for key in ('image', 'text', 'input_id', 'attention_mask', 'image_2d'))

#         images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
#         input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
#         attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)
#         images_2d = torch.cat([_.unsqueeze(0) for _ in images_2d], dim=0)

#         batch_size = images.shape[0]
#         if self.gather_all:
#             world_size = torch.distributed.get_world_size()
#             batch_size *= world_size

#         labels = torch.arange(batch_size, device=images.device, dtype=torch.long)

#         return_dict = dict(
#             images=images, 
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             labels=labels,
#             images_2d=images_2d, 
#         )

#         return return_dict


# def main():
    

#     parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
#     model_args, data_args, training_args = parser.parse_args_into_dataclasses()

#     tokenizer = BertTokenizer.from_pretrained(model_args.language_model_name_or_path)

#     config = M3DCLIPConfig_stage2.from_dict(vars(model_args))
#     model = M3DCLIP_stage2(config)


#     if model_args.pretrained_model:
#         ckpt_stage1 = AutoModel.from_pretrained(model_args.pretrained_model, trust_remote_code=True)
#         ckpt_stage1_params = ckpt_stage1.state_dict()
#         print("load stage1 pretrained model to self.stage1_pretrained_CLIP.") 
#         missing_keys, unexpected_keys = model.stage1_pretrained_CLIP.load_state_dict(ckpt_stage1_params, strict=True)
#         matched_keys = [key for key in ckpt_stage1_params.keys() if key in model.stage1_pretrained_CLIP.state_dict()] 
#         print(f"matched_keys number: {len(matched_keys)}")
#         print(f"missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")
#         ckpt_stage2 = AutoModel.from_pretrained(model_args.stage2_pretrained_CLIP_path, trust_remote_code=True)
#         ckpt_stage2_params = ckpt_stage2.state_dict()
#         print("load stage1 pretrained model to initialize stage2 model.") 
#         matched_keys = [key for key in ckpt_stage2_params.keys() if key in model.state_dict()]
#         missing_keys, unexpected_keys = model.load_state_dict(ckpt_stage2_params, strict=False)
#         print(f"matched_keys number: {len(matched_keys)}")
#         print("load pretrained model finished.")

#     train_dataset = Stage2CapDataset(data_args, tokenizer, mode="train")
#     eval_dataset  = Stage2CapDataset(data_args, tokenizer, mode="validation")


#     if model_args.gather_loss and not model_args.local_loss:
#         gather_all = True
#     else:
#         gather_all = False
#     data_collator = DataCollator(gather_all)


#     from transformers import TrainerCallback
#     TRAINING_ARGS_NAME = "training_args.bin"
#     class CustomSaveCallback(TrainerCallback):
#         def on_step_end(self, args, state, control, **kwargs):
#             if state.global_step in args.save_step_names_list:
#                 save_idx = state.global_step
#                 output_dir = args.output_dir
#                 save_model_path = output_dir + f"/saved_{save_idx}" 
#                 model = kwargs['model'] 

#                 os.makedirs(save_model_path, exist_ok=True)

#                 model.config.save_pretrained(save_model_path) 
#                 model.save_pretrained(save_model_path)
#                 tokenizer.save_pretrained(save_model_path)

#                 state_dict = model.state_dict()
#                 torch.save(state_dict, os.path.join(save_model_path, 'model_params.bin'))
#                 torch.save(args, os.path.join(save_model_path, "training_args.bin"))

#                 print(f"step {save_idx} saved to {save_model_path}")  

#     class GradientMonitorCallback(TrainerCallback):
#         def on_backward_end(self, args, state, control, **kwargs):
#             model = kwargs['model']
#             for name, param in model.named_parameters():
#                 if param.grad is not None:
#                     grad_mean = param.grad.abs().mean().item()
#                     grad_max = param.grad.abs().max().item()
#                     print(f"{name}: mean={grad_mean:.6f}, max={grad_max:.6f}")
#                 else:
#                     print(f"{name}: no grad (requires_grad={param.requires_grad})")



#     class MyTrainer(Trainer):
#         def training_step(self, model, inputs, num_items_in_batch):
#             global_step = self.state.global_step
#             epoch = self.state.epoch
            
#             if global_step is None:
#                 print("global_step is None in MyTrainer")
#                 global_step = 0

#             inputs["global_step"] = global_step
#             inputs["epoch"] = epoch

#             loss = super().training_step(model, inputs, num_items_in_batch)

#             return loss
        

#     trainer = MyTrainer(
#                         model=model,
#                         args=training_args,
#                         data_collator=data_collator,
#                         train_dataset=train_dataset,
#                         eval_dataset=eval_dataset,
#                         compute_metrics=compute_metrics,
#                         preprocess_logits_for_metrics=preprocess_logits_for_metrics,
#                         callbacks=[GradientMonitorCallback(), CustomSaveCallback()]
#     )
   
#     trainer.train()
#     trainer.save_state()
#     model.config.save_pretrained(training_args.output_dir) 
#     model.save_pretrained(training_args.output_dir)
#     tokenizer.save_pretrained(training_args.output_dir)

#     state_dict = model.state_dict()
#     torch.save(state_dict, os.path.join(training_args.output_dir, 'model_params.bin'))
#     print("model saved.")

# if __name__ == "__main__":
#     main()


from typing import Optional
import transformers
from transformers import Trainer
from typing import List
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig
import sys
import os
import torch
from transformers import BertTokenizer

# 确保路径包含 LaMed
sys.path.append("/data/esh/HSENet/Preprint")

# 引入我们修改好的 Dataset
from LaMed.src.dataset.multi_dataset import Stage2CapDataset
from LaMed.src.model.CLIP_stage1 import M3DCLIP_stage1, M3DCLIPConfig_stage1
from LaMed.src.model.CLIP_stage2 import M3DCLIP_stage2, M3DCLIPConfig_stage2

# 优化 Tensor 打印格式
normal_repr = torch.Tensor.__repr__
torch.Tensor.__repr__ = lambda self: f"{self.shape} {normal_repr(self)}"

# 环境变量设置 (根据实际情况调整)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

# 分布式设置
is_use_single_GPU = False
if is_use_single_GPU:
    os.environ['RANK'] = '0' 
    os.environ['WORLD_SIZE'] = '1' 
    os.environ['MASTER_ADDR'] = 'localhost' 
    os.environ['MASTER_PORT'] = '12381' 
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch.distributed as dist
    dist.init_process_group(backend='nccl')

import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True
torch._dynamo.config.disable = True


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    language_model_name_or_path: str = field(default="/data/esh/HSENet/model/bert-base-uncased/")
    use_mask: bool = field(default=False) 
    mask_rate: float = field(default=0.08)
    
    # 【关键】设置为 True 以启用 Adapter 逻辑 (虽然不再加载 BiomedCLIP 大模型，但 CLIP_stage2 需要此开关来初始化 Teacher Adapter)
    use_2D_Encoder: bool = field(default=True)  

    gather_loss: bool = field(default=True, metadata={"help": "Gather all distributed batch data."})
    local_loss: bool = field(default=False)
    
    pretrained_model: str = field(default="/data/esh/HSENet/model/HSENet_CLIP")
    stage2_pretrained_CLIP_path: str = field(default="/data/esh/HSENet/model/M3D-CLIP")
    in_channels: int = field(default=1)
    img_size: tuple = field(default=(32, 256, 256))
    patch_size: tuple = field(default=(4, 16, 16))
    hidden_size: int = field(default=768)
    mlp_dim: int = field(default=3072)
    num_layers: int = field(default=12)
    num_heads: int = field(default=12)
    pos_embed: str = field(default="perceptron")
    dropout_rate: float = field(default=0.0)
    spatial_dims: int = field(default=3)
    max_text_len: int = field(default=128)
    vocab_size: int = field(default=30522)


@dataclass
class DataArguments:
    data_root: str = field(default="/data/esh/HSENet/m3d_data")
    cap_data_path: str = field(default="/data/esh/HSENet/m3d_data/M3D-Cap/M3D_Cap/M3D_Cap.json")
    # 其他路径保持不变...
    vqa_data_train_path: str = field(default="") # 略去不常用路径以精简代码

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = True 
    bf16: bool = True
    output_dir: str = "/data/esh/HSENet/Preprint/LaMed/output/CLIP-Stage2-SGAT-Distill"
    
    # 训练参数
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 6 # 适当调大一点点，视显存而定
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 2 # 累积梯度以模拟大 Batch
    
    evaluation_strategy: str = "steps"
    eval_steps: float = 0.04
    save_strategy: str = "steps"
    save_steps: int = 20000
    save_total_limit: int = 2
    
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    logging_steps: float = 0.001
    
    # 【重要】开启梯度检查点以节省显存 (3D模型很吃显存)
    gradient_checkpointing: bool = True 
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 8
    report_to: str = "tensorboard"


def compute_metrics(eval_pred):
    # 简化的 Accuracy 计算
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    # 确保维度匹配
    if isinstance(preds, tuple): preds = preds[0]
    correct = (preds == labels).sum()
    total = labels.size
    acc = correct / total
    return {"accuracy": acc}

def preprocess_logits_for_metrics(logits, labels):
    try:
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = torch.argmax(logits, dim=-1)
    except:
        preds = None
    return preds

@dataclass
class DataCollator:
    """
    修正后的 DataCollator: 
    能够正确处理双流 2D 特征 (image_2d_kd 和 image_2d_sga)
    """
    def __init__(self, gather_all):
        self.gather_all = gather_all

    def __call__(self, batch: list) -> dict:
        # 1. 提取基础数据
        # 注意: Dataset 返回的是 'image_2d_kd' 和 'image_2d_sga'
        images = torch.stack([b['image'] for b in batch], dim=0)
        input_ids = torch.stack([b['input_id'] for b in batch], dim=0)
        attention_mask = torch.stack([b['attention_mask'] for b in batch], dim=0)
        
        # 2. 提取双流 2D 特征并堆叠
        # [B, slices_kd, 768]
        image_2d_kd = torch.stack([b['image_2d_kd'] for b in batch], dim=0)
        # [B, slices_sga, 768]
        image_2d_sga = torch.stack([b['image_2d_sga'] for b in batch], dim=0)

        # 3. 构造 Labels (对比学习用)
        batch_size = images.shape[0]
        if self.gather_all:
            world_size = torch.distributed.get_world_size()
            batch_size *= world_size
        labels = torch.arange(batch_size, device=images.device, dtype=torch.long)

        # 返回字典，键名必须与 model.forward 的参数名一致
        return {
            "images": images, 
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_2d_kd": image_2d_kd,   # 对应 CLIP_stage2.forward 的参数
            "image_2d_sga": image_2d_sga, # 对应 CLIP_stage2.forward 的参数
        }

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = BertTokenizer.from_pretrained(model_args.language_model_name_or_path)

    config = M3DCLIPConfig_stage2.from_dict(vars(model_args))
    model = M3DCLIP_stage2(config)

    # --- 权重加载 (严格参考你的代码) ---
    if model_args.pretrained_model:
        print(f"Loading Stage 1 pretrained model from: {model_args.pretrained_model}")
        
        # 1. 加载 Stage 1 权重 (RR Loss 用)
        # 关键：这里使用 trust_remote_code=True 让 AutoModel 尝试运行 checkpoint 里的代码
        ckpt_stage1 = AutoModel.from_pretrained(model_args.pretrained_model, trust_remote_code=True)
        ckpt_stage1_params = ckpt_stage1.state_dict()
        
        print("Load stage1 pretrained model to self.stage1_pretrained_CLIP.") 
        # 将加载的权重塞入 model.stage1_pretrained_CLIP
        missing_keys, unexpected_keys = model.stage1_pretrained_CLIP.load_state_dict(ckpt_stage1_params, strict=False) # 建议 False 以防微小差异，或者按你原来的 True
        
        matched_keys = [key for key in ckpt_stage1_params.keys() if key in model.stage1_pretrained_CLIP.state_dict()] 
        print(f"Stage 1 matched_keys number: {len(matched_keys)}")
        # print(f"missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")

        # 2. 加载 Stage 2 初始权重 (Student 用)
        print(f"Loading Stage 2 init weights from: {model_args.stage2_pretrained_CLIP_path}")
        ckpt_stage2 = AutoModel.from_pretrained(model_args.stage2_pretrained_CLIP_path, trust_remote_code=True)
        ckpt_stage2_params = ckpt_stage2.state_dict()
        
        print("Initialize stage2 model with pretrained weights.") 
        matched_keys = [key for key in ckpt_stage2_params.keys() if key in model.state_dict()]
        
        # strict=False 必须的，因为我们加了 SGAT, KD_Proj, Teacher_Adapter 等新层
        missing_keys, unexpected_keys = model.load_state_dict(ckpt_stage2_params, strict=False)
        print(f"Stage 2 matched_keys number: {len(matched_keys)}")
        print("Load pretrained model finished.")
    # --- 初始化 Dataset ---
    # 可以在这里调整 num_slices_kd 和 num_slices_sga
    train_dataset = Stage2CapDataset(data_args, tokenizer, mode="train", num_slices_kd=8, num_slices_sga=4)
    eval_dataset  = Stage2CapDataset(data_args, tokenizer, mode="validation", num_slices_kd=8, num_slices_sga=4)

    # --- Data Collator ---
    gather_all = model_args.gather_loss and not model_args.local_loss
    data_collator = DataCollator(gather_all)

    # --- Callbacks ---
    from transformers import TrainerCallback
    class GradientMonitorCallback(TrainerCallback):
        def on_backward_end(self, args, state, control, **kwargs):
            if state.global_step % 100 == 0:
                model = kwargs['model']
                for name, param in model.named_parameters():
                    if param.grad is not None and ("sgat" in name or "teacher_adapter" in name):
                        print(f"[Grad] {name}: {param.grad.abs().mean().item():.6f}")

    # --- Trainer ---
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=[GradientMonitorCallback()]
    )
   
    print("Starting training...")
    trainer.train()
    
    # Save
    trainer.save_state()
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    print("Training finished.")

if __name__ == "__main__":
    main()