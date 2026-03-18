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
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
import torch.nn as nn
import sys

# 引入 accelerate 处理多卡并行
from accelerate import Accelerator
from accelerate.utils import gather_object

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default="/data/esh/HSENet/phi-4-mini-instruct/Phi-4-mini-instruct")
    parser.add_argument('--pretrained_vision_tower_path', type=str, default="/data/esh/HSENet/Preprint/LaMed/output_new/stage2/model.safetensors")
    parser.add_argument('--resume_mllm_weights', type=str, default="/data/esh/HSENet/Preprint/LaMed/output_new/stage3_2_finetune_test/mm_projector_and_lora.bin")
    
    parser.add_argument('--pretrain_mm_mlp_adapter', type=str, default=None)
    parser.add_argument('--tune_mm_mlp_adapter', type=bool, default=False)
    parser.add_argument('--version', type=str, default="v0")
    parser.add_argument('--vision_tower', type=str, default="vit_stage2_dual_encoders")
    parser.add_argument('--mm_projector_type', type=str, default="VisualPacker_3d_phi_v3")
    parser.add_argument('--proj_out_num', type=int, default=144)
    parser.add_argument('--vision_select_layer', type=int, default=-1)
    parser.add_argument('--vision_select_feature', type=str, default="patch")
    parser.add_argument('--use_parallel_projector', type=bool, default=True)
    parser.add_argument('--remain_2d3d_ViT_type', type=str, default="dual_vits")
    parser.add_argument('--image_channel', type=int, default=1)
    parser.add_argument('--image_size', nargs='+', type=int, default=[32, 256, 256])
    parser.add_argument('--patch_size', nargs='+', type=int, default=[4, 16, 16])
    parser.add_argument('--proj_layer_type', type=str, default="mlp")
    parser.add_argument('--proj_layer_num', type=int, default=2)
    parser.add_argument('--proj_pooling_type', type=str, default="spatial")
    parser.add_argument('--proj_pooling_size', type=int, default=2)
    parser.add_argument('--freeze_vision_tower', type=bool, default=True)
    parser.add_argument('--segmentation_module', type=str, default=None)
    parser.add_argument('--pretrain_seg_module', type=str, default=None)

    parser.add_argument('--max_length', type=int, default=2048)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--top_p', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--data_root', type=str, default="/data/esh/HSENet/m3d_data")
    parser.add_argument('--vqa_data_test_path', type=str, default="/data/esh/HSENet/m3d_data/M3D-VQA/M3D_VQA_test5k_new_1.csv") 
    parser.add_argument('--close_ended', type=bool, default=True) # 注意这里可以按需修改
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

# [核心修复 1]：严格对齐训练时的 32 层重采样和纯净 512 维特征
def load_or_create_2d_feat(image_abs_path, target_slices=32):
    feat_2d_path = image_abs_path.replace(".npy", "_2D.npy")
    if os.path.exists(feat_2d_path):
        feats = np.load(feat_2d_path)
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
                
                raw_question = data["Question"]
                if self.close_ended:
                    clean_question = "Closed VQA Task: " + raw_question + '\n' + "Choices: A. {} B. {} C. {} D. {}".format(data["Choice A"], data["Choice B"], data["Choice C"], data["Choice D"])
                    answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
                    answer_choice = data["Answer Choice"]
                else:
                    clean_question = "Open VQA Task: " + raw_question
                    answer = str(data["Answer"])
                    answer_choice = ""

                # [核心修复 2]：严格应用 Phi-4 Instruct 的对话模板！
                # 只有加上 <|assistant|>，模型才会知道该轮到它作答了，否则它会直接输出 <|end|>
                prompt_question = f"<|user|>\n{self.image_tokens}\n{clean_question}<|end|>\n<|assistant|>\n"

                return {
                    'image': image, 
                    'question': prompt_question,     
                    'clean_question': clean_question, 
                    'answer': answer, 
                    'answer_choice': answer_choice, 
                    'question_type': data["Question Type"], 
                    'image_2d': image_2d 
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

    if accelerator.is_main_process:
        print("="*20 + " Tokenizer Preparation " + "="*20)
        
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, model_max_length=args.max_length,
        padding_side="left", use_fast=False,
    )
    tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]})
    tokenizer.add_tokens("[SEG]")
    
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")
    args.vocab_size = len(tokenizer)

    if accelerator.is_main_process:
        print("="*20 + " Model Preparation " + "="*20)
        
    model = LamedPhi3ForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    model.config.seg_token_id = args.seg_token_id
    model.config.use_cache = True 
    
    model.get_model().initialize_vision_modules(model_args=args)
    args.num_new_tokens = 4
    model.initialize_vision_tokenizer(args, tokenizer)

    if os.path.exists(args.pretrained_vision_tower_path):
        sd = load_file(args.pretrained_vision_tower_path) if args.pretrained_vision_tower_path.endswith(".safetensors") else torch.load(args.pretrained_vision_tower_path, map_location="cpu")
        model.get_model().vision_tower.load_state_dict({k.replace("vision_encoder.", ""): v for k, v in sd.items() if "vision_encoder." in k}, strict=False)

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=find_all_linear_names(model), lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    if os.path.exists(args.resume_mllm_weights):
        if accelerator.is_main_process:
            print(f"Loading finetuned weights from: {args.resume_mllm_weights}")
        finetuned_weights = torch.load(args.resume_mllm_weights, map_location="cpu")
        
        projector_weights = {k.replace("mm_projector.", ""): v for k, v in finetuned_weights.items() if 'mm_projector' in k}
        if len(projector_weights) > 0:
            # 兼容保存时的 key name 可能带有多余前缀
            model.get_model().mm_projector.load_state_dict(projector_weights, strict=False)
            
        from peft import set_peft_model_state_dict
        lora_weights = {}
        for k, v in finetuned_weights.items():
            if 'lora' in k:
                new_key = k.replace("base_model.model.", "")
                lora_weights[new_key] = v
        if len(lora_weights) > 0:
            set_peft_model_state_dict(model, lora_weights)

    model.eval()

    def collate_fn(batch):
        return {
            'images': torch.stack([b['image'] for b in batch]),
            'questions': [b['question'] for b in batch],            
            'clean_questions': [b['clean_question'] for b in batch], 
            'answers': [b['answer'] for b in batch],
            'answer_choices': [b['answer_choice'] for b in batch],
            'question_types': [b['question_type'] for b in batch],
            'images_2d': torch.stack([b['image_2d'] for b in batch])
        }

    test_dataset = RobustVQADataset(args, tokenizer=tokenizer, close_ended=args.close_ended)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, collate_fn=collate_fn)  

    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    if not os.path.exists(args.output_dir) and accelerator.is_main_process:
        os.makedirs(args.output_dir)

    if accelerator.is_main_process:
        print("="*20 + " Start Inference " + "="*20)
        
    local_results = []

    for sample in tqdm(test_dataloader, disable=not accelerator.is_main_process):
        images = sample["images"].to(device, dtype=torch.bfloat16)
        images_2d = sample["images_2d"].to(device, dtype=torch.bfloat16)
        questions = sample["questions"]
        clean_questions = sample["clean_questions"]
        
        inputs = tokenizer(questions, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            unwrapped_model = accelerator.unwrap_model(model)
            
            # [核心修复 3]：增加 max_new_tokens 并修正传参规范
            generation_kwargs = dict(
                input_ids=input_ids,             # 必须显式传 input_ids
                attention_mask=attention_mask,
                max_new_tokens=20 if args.close_ended else 150, # 闭卷VQA给 20 个token裕量防阶段
                repetition_penalty=1.0, 
                do_sample=False, 
                pad_token_id=tokenizer.pad_token_id, 
                eos_token_id=tokenizer.eos_token_id
            )
            
            generation = unwrapped_model.generate(
                images=images,
                images_2d=images_2d,
                **generation_kwargs
            )
        
        input_len = input_ids.shape[1]
        
        # 逐个解析 Batch 内的生成结果
        for i in range(len(generation)):
            # 只截取模型新生成的部分
            gen_toks = generation[i][input_len:]
            pred_text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
            
            q_type = sample["question_types"][i]
            clean_q = clean_questions[i]
            a = sample["answers"][i]
            a_c = sample["answer_choices"][i]
            
            if args.close_ended:
                import re
                # 兼容模型输出 "A", "The answer is A", "A." 等情况
                match = re.search(r'([A-D])', pred_text)
                if match:
                    model_choice = match.group(1)
                    correct = 1 if model_choice == a_c else 0
                else:
                    correct = 0
                local_results.append([q_type, clean_q, a, a_c, pred_text, correct])
            else:
                local_results.append([q_type, clean_q, a, "", pred_text, 0])

    accelerator.wait_for_everyone()
    gathered_results = gather_object(local_results)

    if accelerator.is_main_process:
        output_name = "eval_close_vqa.csv" if args.close_ended else "eval_open_vqa.csv"
        output_path = os.path.join(args.output_dir, output_name)
        
        seen = set()
        unique_results = []
        for row in gathered_results:
            row_str = str(row[1]) 
            if row_str not in seen:
                seen.add(row_str)
                unique_results.append(row)

        with open(output_path, mode='w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["Question Type", "Question", "Answer", "Answer Choice", "Pred", "Correct"])
            for row in unique_results:
                writer.writerow(row)
                
        print(f"\nEvaluation Complete! Saved {len(unique_results)} unique results to {output_path}")

if __name__ == "__main__":
    main()
