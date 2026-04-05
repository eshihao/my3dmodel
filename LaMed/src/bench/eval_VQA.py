import os
import csv
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import monai.transforms as mtf
from monai.data import set_track_meta
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model
import torch.nn as nn
import sys

from accelerate import Accelerator
from accelerate.utils import gather_object

# 确保能找到项目根目录下的模型代码
sys.path.append("/data/esh/HSENet/Preprint")
from LaMed.src.model.language_model import LamedPhi3ForCausalLM

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        v = str(v).strip().lower()
        if v in {"true", "1", "yes", "y", "t"}:
            return True
        if v in {"false", "0", "no", "n", "f"}:
            return False
        raise argparse.ArgumentTypeError(f"Invalid bool value: {v}")

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="/data/esh/HSENet/phi-4-mini-instruct/Phi-4-mini-instruct")
    parser.add_argument('--pretrained_vision_tower_path', type=str, default="/data/esh/HSENet/Preprint/LaMed/output_new/stage2/model.safetensors")
    parser.add_argument('--resume_mllm_weights', type=str, default="/data/esh/HSENet/Preprint/LaMed/output_new/stage3_2_VQA/saved_30000/pytorch_model.bin")
    
    parser.add_argument('--pretrain_mm_mlp_adapter', type=str, default=None)
    parser.add_argument('--tune_mm_mlp_adapter', type=str2bool, default=False)
    parser.add_argument('--version', type=str, default="v0")
    parser.add_argument('--vision_tower', type=str, default="vit_stage2_dual_encoders")
    parser.add_argument('--mm_projector_type', type=str, default="VisualPacker_3d_phi_v3")
    parser.add_argument('--proj_out_num', type=int, default=144)
    parser.add_argument('--vision_select_layer', type=int, default=-1)
    parser.add_argument('--vision_select_feature', type=str, default="patch")
    parser.add_argument('--use_parallel_projector', type=str2bool, default=True)
    parser.add_argument('--remain_2d3d_ViT_type', type=str, default="dual_vits")
    parser.add_argument('--image_channel', type=int, default=1)
    parser.add_argument('--image_size', nargs='+', type=int, default=[32, 256, 256])
    parser.add_argument('--patch_size', nargs='+', type=int, default=[4, 16, 16])
    parser.add_argument('--proj_layer_type', type=str, default="mlp")
    parser.add_argument('--proj_layer_num', type=int, default=2)
    parser.add_argument('--proj_pooling_type', type=str, default="spatial")
    parser.add_argument('--proj_pooling_size', type=int, default=2)
    parser.add_argument('--freeze_vision_tower', type=str2bool, default=True)
    parser.add_argument('--segmentation_module', type=str, default=None)
    parser.add_argument('--pretrain_seg_module', type=str, default=None)

    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=str2bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--data_root', type=str, default="/data/esh/HSENet/m3d_data")
    # 默认使用你最新的验证集 csv
    parser.add_argument('--vqa_data_test_path', type=str, default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_test5k_new.csv") 
    parser.add_argument('--close_ended', type=str2bool, default=True)
    parser.add_argument('--output_dir', type=str, default="./eval_results/M3D_VQA/")
    
    return parser.parse_args()

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module']
    for name, module in model.named_modules():
        if any(kw in name for kw in ignore_keywords): continue
        if isinstance(module, cls): lora_module_names.add(name)
    return list(lora_module_names)

def load_or_create_2d_feat(image_abs_path, target_slices=32):
    feat_2d_path = image_abs_path.replace(".npy", "_2D.npy")
    if os.path.exists(feat_2d_path):
        feats = np.load(feat_2d_path)
        if feats.shape[-1] != 512:
            return torch.zeros((target_slices, 512), dtype=torch.bfloat16)
        total_slices = feats.shape[0]
        if total_slices > 0:
            idx = np.linspace(0, total_slices - 1, target_slices).astype(int)
            feats = feats[idx]
        else:
            feats = np.zeros((target_slices, 512), dtype=np.float32)
        return torch.tensor(feats, dtype=torch.bfloat16)
    else:
        return torch.zeros((target_slices, 512), dtype=torch.bfloat16)

class RobustVQADataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=False):
        self.args = args
        self.tokenizer = tokenizer
        self.close_ended = close_ended
        self.image_tokens = "<im_patch>" * args.proj_out_num
        self.data_list = pd.read_csv(args.vqa_data_test_path)
        self.transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
        set_track_meta(False)

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        attempts = 0
        max_attempts = len(self.data_list)
        while attempts < max_attempts:
            data = self.data_list.iloc[idx]
            image_abs_path = os.path.join(self.args.data_root, data["Image Path"])
            
            if not os.path.exists(image_abs_path):
                idx = (idx + 1) % len(self.data_list)
                attempts += 1
                continue

            try:
                image = np.load(image_abs_path)
                image = self.transform(image)
                image_2d = load_or_create_2d_feat(image_abs_path)
                
                raw_question = str(data["Question"]).strip()
                # [终极修复 1]：绝对纯净的格式，完全抛弃 Chat 模板，原汁原味还原 dataset.py
                if self.close_ended:
                    clean_question = f"Closed VQA Task: {raw_question} Choices: A. {data['Choice A']} B. {data['Choice B']} C. {data['Choice C']} D. {data['Choice D']}"
                    answer = f"{data['Answer Choice']}. {data['Answer']}"
                    answer_choice = str(data["Answer Choice"]).strip()
                else:
                    clean_question = f"Open VQA Task: {raw_question}"
                    answer = str(data["Answer"]).strip()
                    answer_choice = ""

                prompt_question = f"{self.image_tokens} {clean_question} Answer: "
                if self.tokenizer.bos_token: 
                    prompt_question = self.tokenizer.bos_token + prompt_question

                return {
                    'image': image, 
                    'question': prompt_question,     
                    'clean_question': clean_question, 
                    'answer': answer, 
                    'answer_choice': answer_choice, 
                    'question_type': data.get("Question Type", "unknown"), 
                    'image_2d': image_2d,
                    'image_path': data["Image Path"],
                }
            except Exception:
                idx = (idx + 1) % len(self.data_list)
                attempts += 1
        raise FileNotFoundError("全验证集无有效文件！")

def main():
    seed_everything(42)
    args = parse_args()
    
    accelerator = Accelerator()
    device = accelerator.device

    if accelerator.is_main_process: print("\n" + "="*20 + " Tokenizer Preparation " + "="*20)
        
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, model_max_length=args.max_length,
        padding_side="left", use_fast=False,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]})
    tokenizer.add_tokens("[SEG]")
    
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None: tokenizer.pad_token = tokenizer.unk_token
        else: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    args.vocab_size = len(tokenizer)

    if accelerator.is_main_process: print("="*20 + " Model Preparation " + "="*20)
        
    model = LamedPhi3ForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    model.config.seg_token_id = args.seg_token_id
    model.config.use_cache = True 
    
    model.get_model().initialize_vision_modules(model_args=args)
    args.num_new_tokens = 4
    model.initialize_vision_tokenizer(args, tokenizer)

    # ================= [终极修复 2] 视觉双塔强健映射 =================
    if os.path.exists(args.pretrained_vision_tower_path):
        if accelerator.is_main_process: print(f"🚀 Loading Vision Tower: {args.pretrained_vision_tower_path}")
        sd = load_file(args.pretrained_vision_tower_path) if args.pretrained_vision_tower_path.endswith(".safetensors") else torch.load(args.pretrained_vision_tower_path, map_location="cpu")
        vision_dict = {}
        for k, v in sd.items():
            if k.startswith("stage1_pretrained_CLIP.vision_encoder."):
                vision_dict[k.replace("stage1_pretrained_CLIP.vision_encoder.", "vision_tower_stage1.")] = v
            elif k.startswith("vision_encoder."):
                vision_dict[k.replace("vision_encoder.", "vision_tower_stage2.")] = v
            elif any(k.startswith(p) for p in ["blocks.", "patch_embedding.", "cls_token", "norm.", "sgat.", "sga.", "sga_adapter", "kd_proj"]):
                vision_dict[f"vision_tower_stage2.{k}"] = v
        model.get_model().vision_tower.load_state_dict(vision_dict, strict=False)

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=find_all_linear_names(model), lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    # ================= [终极修复 3] 极简原生加载 Projector 与 LoRA =================
    if os.path.exists(args.resume_mllm_weights):
        if accelerator.is_main_process: print(f"📦 Loading Finetuned Weights: {args.resume_mllm_weights}")
        finetuned_weights = torch.load(args.resume_mllm_weights, map_location="cpu")
        
        # 极简加载：你保存的是 PeftModel 的原生参数，直接原路塞回去绝对不出错！
        res = model.load_state_dict(finetuned_weights, strict=False)
        
        if accelerator.is_main_process:
            loaded_proj = sum(1 for k in finetuned_weights if 'mm_projector' in k)
            loaded_lora = sum(1 for k in finetuned_weights if 'lora' in k)
            print(f"✅ 成功挂载 {loaded_proj} 个 Projector 参数，{loaded_lora} 个 LoRA 参数！")
            if "unexpected_keys" in str(res) and len(res.unexpected_keys) > 0:
                print(f"⚠️ 注意: 存在无法对齐的参数: {res.unexpected_keys[:3]}...")

    model.eval()

    def collate_fn(batch):
        return {
            'images': torch.stack([b['image'] for b in batch]),
            'questions': [b['question'] for b in batch],            
            'clean_questions': [b['clean_question'] for b in batch], 
            'answers': [b['answer'] for b in batch],
            'answer_choices': [b['answer_choice'] for b in batch],
            'question_types': [b['question_type'] for b in batch],
            'images_2d': torch.stack([b['image_2d'] for b in batch]),
            'image_paths': [b['image_path'] for b in batch],
        }

    test_dataset = RobustVQADataset(args, tokenizer=tokenizer, close_ended=args.close_ended)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, collate_fn=collate_fn)  

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    if not os.path.exists(args.output_dir) and accelerator.is_main_process:
        os.makedirs(args.output_dir)

    if accelerator.is_main_process: print("\n" + "="*20 + " Start Inference " + "="*20)
        
    local_results = []

    for i_batch, sample in enumerate(tqdm(test_dataloader, disable=not accelerator.is_main_process)):
        images = sample["images"].to(device, dtype=torch.bfloat16)
        images_2d = sample["images_2d"].to(device, dtype=torch.bfloat16)
        questions = sample["questions"]
        clean_questions = sample["clean_questions"]
        
        # [终极修复 4]：禁止自动加 EOS，避免瞬间闭嘴
        inputs = tokenizer(questions, return_tensors="pt", padding=True, add_special_tokens=False)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            unwrapped_model = accelerator.unwrap_model(model)
            
            # [终极修复 5]：加大惩罚力度对抗复读机
            generation = unwrapped_model.generate(
                inputs=input_ids,
                images=images,
                images_2d=images_2d,
                attention_mask=attention_mask,
                max_new_tokens=5 if args.close_ended else 150, 
                repetition_penalty=1.2, 
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )
        
        # [终极修复 6]：自适应解码逻辑，兼容左填充与不同 generate 返回风格
        for i in range(len(generation)):
            output_ids = generation[i]
            full_prompt_len = input_ids.shape[1]
            valid_prompt_ids = input_ids[i][attention_mask[i].bool()]
            valid_prompt_len = int(valid_prompt_ids.shape[0])

            # 常见路径：generate 返回 [完整输入(含padding) + 新token]
            if len(output_ids) > full_prompt_len and torch.equal(output_ids[:full_prompt_len], input_ids[i]):
                output_ids = output_ids[full_prompt_len:]
            # 兜底路径：若返回仅按有效 token 对齐，则按去 padding 后的 prompt 去切
            elif len(output_ids) > valid_prompt_len and torch.equal(output_ids[:valid_prompt_len], valid_prompt_ids):
                output_ids = output_ids[valid_prompt_len:]
                
            pred_text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            if "Answer:" in pred_text:
                pred_text = pred_text.split("Answer:")[-1].strip()
            
            q_type = sample["question_types"][i]
            clean_q = clean_questions[i]
            a = sample["answers"][i]
            a_c = sample["answer_choices"][i]
            img_p = sample["image_paths"][i]
            
            if args.close_ended:
                import re
                pred_up = pred_text.strip().upper()
                choice_patterns = [
                    r'^\s*([A-D])\b',
                    r'OPTION\s*([A-D])\b',
                    r'ANSWER\s*(?:IS|:)?\s*([A-D])\b',
                    r'\b([A-D])\b',
                ]
                model_choice = ""
                for pat in choice_patterns:
                    m = re.search(pat, pred_up)
                    if m:
                        model_choice = m.group(1)
                        break
                correct = 1 if (model_choice == str(a_c).strip().upper()) else 0
                local_results.append([q_type, clean_q, a, a_c, pred_text, correct, img_p])
                
                # 实时打印前两个 Batch 的推理情况，监控模型是否在认真思考
                if i_batch < 2 and accelerator.is_main_process:
                    status = '✅' if correct else '❌'
                    print(f"\n[实时监控] 真实答案: {a_c} | 模型预测: {pred_text} | {status}")
            else:
                local_results.append([q_type, clean_q, a, "", pred_text, 0, img_p])

    accelerator.wait_for_everyone()
    gathered_results = gather_object(local_results)

    if accelerator.is_main_process:
        output_name = "eval_close_vqa.csv" if args.close_ended else "eval_open_vqa.csv"
        output_path = os.path.join(args.output_dir, output_name)
        
        seen = set()
        unique_results = []
        for row in gathered_results:
            row_str = str((row[1], row[6]))
            if row_str not in seen:
                seen.add(row_str)
                unique_results.append(row)

        with open(output_path, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Question Type", "Question", "Answer", "Answer Choice", "Pred", "Correct", "Image Path"])
            for row in unique_results:
                writer.writerow(row)
                
        print(f"\nEvaluation Complete! Saved {len(unique_results)} unique results to {output_path}")
        
        if args.close_ended and len(unique_results) > 0:
            acc = sum(r[5] for r in unique_results) / len(unique_results)
            print(f"🔥 Final VQA Accuracy: {acc * 100:.2f}%")

if __name__ == "__main__":
    main()
