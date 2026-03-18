import os
import sys
import torch
import numpy as np
import monai.transforms as mtf
from transformers import AutoTokenizer
from safetensors.torch import load_file
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict

# 确保能找到你的模型代码
sys.path.append("/data/esh/HSENet/Preprint")
from LaMed.src.model.language_model import LamedPhi3ForCausalLM

# ================= 配置区 =================
BASE_MODEL_PATH = "/data/esh/HSENet/phi-4-mini-instruct/Phi-4-mini-instruct"
STAGE2_VISION_PATH = "/data/esh/HSENet/Preprint/LaMed/output_new/stage2/model.safetensors"
STAGE3_WEIGHTS_PATH = "/data/esh/HSENet/Preprint/LaMed/output_new/stage3_2_finetune_test/mm_projector_and_lora.bin"
PROJ_OUT_NUM = 144

class MiniArgs:
    version = "v0"
    vision_tower = "vit_stage2_dual_encoders"
    mm_projector_type = "VisualPacker_3d_phi_v3"
    proj_out_num = PROJ_OUT_NUM
    vision_select_layer = -1
    vision_select_feature = "patch"
    use_parallel_projector = True
    remain_2d3d_ViT_type = "dual_vits"
    image_channel = 1
    image_size = [32, 256, 256]
    patch_size = [4, 16, 16]
    proj_layer_type = "mlp"
    proj_layer_num = 2
    proj_pooling_type = "spatial"
    proj_pooling_size = 2
    freeze_vision_tower = True
    segmentation_module = None
    pretrain_seg_module = None
    tune_mm_mlp_adapter = False
    num_new_tokens = 4
    pretrain_mm_mlp_adapter = None

args = MiniArgs()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

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

def main():
    print("\n" + "="*50)
    print("🚀 正在加载 3D 医疗大模型 (请耐心等待...)")
    print("="*50)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, padding_side="left", use_fast=False)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]})
    tokenizer.add_tokens("[SEG]")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")

    model = LamedPhi3ForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))
    model.config.seg_token_id = args.seg_token_id
    model.config.use_cache = True 

    model.get_model().initialize_vision_modules(model_args=args)
    model.initialize_vision_tokenizer(args, tokenizer)
    
    # ==============================================================
    # [绝杀修复 1] 强健挂载 Stage 2 视觉塔
    # ==============================================================
    if os.path.exists(STAGE2_VISION_PATH):
        sd = load_file(STAGE2_VISION_PATH) if STAGE2_VISION_PATH.endswith(".safetensors") else torch.load(STAGE2_VISION_PATH, map_location="cpu")
        vision_dict = {}
        for k, v in sd.items():
            if "vision_encoder." in k:
                new_k = k.split("vision_encoder.")[-1]
                vision_dict[new_k] = v
        res = model.get_model().vision_tower.load_state_dict(vision_dict, strict=False)
        print(f"👁️ 加载 Vision Tower 成功! (遗漏的键如果是无关组件则正常): \n{res}")

    lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=find_all_linear_names(model), lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)

    # ==============================================================
    # [绝杀修复 2] 强健挂载 Stage 3.2 权重 (Projector & 词表 & LoRA)
    # ==============================================================
    if os.path.exists(STAGE3_WEIGHTS_PATH):
        finetuned_weights = torch.load(STAGE3_WEIGHTS_PATH, map_location="cpu")
        
        # 精准截取 Projector
        projector_weights = {}
        for k, v in finetuned_weights.items():
            if 'mm_projector' in k:
                new_k = k.split('mm_projector.')[-1] # 只取 mm_projector. 后面的纯净名字
                projector_weights[new_k] = v
        if len(projector_weights) > 0: 
            res = model.get_model().mm_projector.load_state_dict(projector_weights, strict=False)
            print(f"\n✅ 成功挂载 {len(projector_weights)} 个 Projector 权重！视觉通道已彻底打通！\n状态: {res}")

        # 词表 Embedding
        embed_weights = {k: v for k, v in finetuned_weights.items() if 'embed_tokens' in k or 'lm_head' in k}
        if len(embed_weights) > 0: 
            model.load_state_dict(embed_weights, strict=False)

        # LoRA
        lora_weights = {k.replace("base_model.model.", ""): v for k, v in finetuned_weights.items() if 'lora' in k}
        if len(lora_weights) > 0: 
            set_peft_model_state_dict(model, lora_weights)
    
    model = model.to(device)
    model.eval()
    transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
    
    print("\n🎉 模型加载完毕！进入交互模式。输入 'quit' 或 'exit' 退出程序。")

    while True:
        image_path = input("\n👉 请输入 3D 图像的绝对路径 (.npy): ").strip()
        if image_path.lower() in ['quit', 'exit']: break
        if not os.path.exists(image_path):
            print("❌ 文件不存在，请检查路径。")
            continue

        question = input("💬 请输入你的问题: ").strip()
        if question.lower() in ['quit', 'exit']: break

        try:
            image_np = np.load(image_path)
            image_tensor = transform(image_np).unsqueeze(0).to(device, dtype=torch.bfloat16) 
            image_2d_tensor = load_or_create_2d_feat(image_path).unsqueeze(0).to(device)

            q_raw = question.replace("Question:", "").strip()
            if q_raw.endswith("Answer:"): q_raw = q_raw[:-7].strip()

            if "Choice A:" in q_raw:
                q_raw = q_raw.replace("Choice A:", "A.").replace("Choice B:", "B.").replace("Choice C:", "C.").replace("Choice D:", "D.")
            
            if "Choices:" in q_raw or "A." in q_raw:
                if "Choices:" not in q_raw: q_raw = q_raw.replace("A.", "Choices: A.")
                clean_q = f"Closed VQA Task: {q_raw}"
            else:
                clean_q = f"Open VQA Task: {q_raw}"

            image_tokens = "<im_patch>" * PROJ_OUT_NUM
            prompt_text = f"{image_tokens} {clean_q} Answer: "
            if tokenizer.bos_token: prompt_text = tokenizer.bos_token + prompt_text
            
            print(f"\n[后台处理后的真实 Prompt]:\n...{clean_q} Answer: ")

            inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            print("🧠 思考中...")
            
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                generation = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    images_2d=image_2d_tensor,
                    attention_mask=attention_mask,
                    max_new_tokens=15,             
                    repetition_penalty=1.2,        
                    do_sample=False,         
                    pad_token_id=tokenizer.pad_token_id, 
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )

            output_ids = generation[0]
            if len(output_ids) > input_ids.shape[1] and torch.equal(output_ids[:input_ids.shape[1]], input_ids[0]):
                output_ids = output_ids[input_ids.shape[1]:]
                
            raw_answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

            if "Answer:" in raw_answer:
                final_answer = raw_answer.split("Answer:")[-1].strip()
            else:
                final_answer = raw_answer.strip()

            print(f"\n🤖 模型回答: \033[92m{final_answer}\033[0m\n")

        except Exception as e:
            print(f"❌ 推理出错: {e}")

if __name__ == "__main__":
    main()
