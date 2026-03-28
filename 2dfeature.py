# import os
# import numpy as np
# import torch
# from PIL import Image
# import tqdm
# import glob
# import argparse
# import multiprocessing
# from multiprocessing import Pool
# import json
# from transformers import AutoTokenizer
# import open_clip
# from open_clip.factory import _MODEL_CONFIGS, create_model_and_transforms

# '''
# 修复版 HSENet Stage2 离线特征提取脚本：
# 1. 兼容本地 BiomedCLIP 配置文件
# 2. 修复 open_clip 加载 RuntimeError
# 3. 512 -> 768 维度投影（注意：此处使用固定随机种子或加载训练好的权重）
# '''

# def extract_features(slice_dir, output_npy, model, preprocess, projector, device):
#     """提取一个目录下的所有切片特征"""
#     # 获取目录下所有图片 (保持 sorted 确保均匀采样时的顺序稳定)
#     slice_paths = sorted(glob.glob(os.path.join(slice_dir, "*.png")) + 
#                          glob.glob(os.path.join(slice_dir, "*.jpg")) +
#                          glob.glob(os.path.join(slice_dir, "*.jpeg")))
    
#     if not slice_paths:
#         return None
    
#     imgs = []
#     for path in slice_paths:
#         try:
#             img = Image.open(path).convert("RGB")
#             imgs.append(preprocess(img))
#         except Exception as e:
#             print(f"Warning: skip corrupt image {path}")
#             continue
    
#     if not imgs: return None

#     # 堆叠并分批处理 (防止显存溢出)
#     batch_tensor = torch.stack(imgs).to(device)
    
#     with torch.no_grad():
#         # 1. 提取原始特征 (BiomedCLIP 默认是 512 维)
#         # 注意：使用 encode_image 得到的是 projection 后的向量
#         features = model.encode_image(batch_tensor)
        
#         # 2. 投影到 768 维 (适配 HSENet)
#         features = projector(features)
    
#     features_array = features.cpu().numpy().astype(np.float32)
#     np.save(output_npy, features_array)
#     return output_npy

# def process_gpu_batch(batch_args):
#     """每个 GPU 进程执行的逻辑"""
#     gpu_id, dir_list, model_path = batch_args
#     device = torch.device(f"cuda:{gpu_id}")
#     torch.cuda.set_device(gpu_id)

#     # --- 核心修复：仿照 Dataset 加载方式 ---
#     config_path = os.path.join(model_path, "open_clip_config.json")
#     ckpt_path = os.path.join(model_path, "open_clip_pytorch_model.bin")

#     with open(config_path, "r") as f:
#         config_data = json.load(f)
#         model_cfg = config_data["model_cfg"]
#         preprocess_cfg = config_data["preprocess_cfg"]

#     # 注册模型配置到 open_clip
#     model_name = "biomedclip_local"
#     if model_name not in _MODEL_CONFIGS:
#         _MODEL_CONFIGS[model_name] = model_cfg

#     print(f"[GPU {gpu_id}] Initializing BiomedCLIP...")
#     model, _, preprocess = create_model_and_transforms(
#         model_name=model_name,
#         pretrained=ckpt_path,
#         **{f"image_{k}": v for k, v in preprocess_cfg.items()}
#     )
#     model = model.to(device).eval()

#     # 维度对齐投影层 (512 -> 768)
#     # 警告：此处的 Linear 层权重必须与训练时加载的权重一致！
#     # 如果你还没有保存好的 projector 权重，建议这里先设置固定种子
#     torch.manual_seed(42)
#     projector = torch.nn.Linear(512, 768).to(device).eval()

#     for slice_dir in tqdm.tqdm(dir_list, desc=f"GPU {gpu_id}", position=gpu_id):
#         # 仿照 Dataset 的保存命名：在目录同级生成 .npy
#         output_npy = slice_dir + "_2D.npy"
#         if os.path.exists(output_npy): continue
        
#         try:
#             extract_features(slice_dir, output_npy, model, preprocess, projector, device)
#         except Exception as e:
#             print(f"\n[GPU {gpu_id}] Error in {slice_dir}: {e}")

# def get_all_target_dirs(base_dir):
#     """递归查找所有包含图片的末级目录"""
#     target_dirs = []
#     print("Scanning directories...")
#     for root, dirs, files in os.walk(base_dir):
#         if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
#             target_dirs.append(root)
#     return target_dirs
    

# def main():
#     parser = argparse.ArgumentParser()
#     # base_dir 应该是包含所有病人文件夹的根目录
#     parser.add_argument('--base_dir', type=str, default="/data/esh/HSENet/m3d_data/M3D-Cap/M3D_Cap/ct_quizze")
#     parser.add_argument('--model_path', type=str, default="/data/esh/HSENet/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
#     parser.add_argument('--num_gpus', type=int, default=4)
#     args = parser.parse_args()

#     all_dirs = get_all_target_dirs(args.base_dir)
#     print(f"Total directories to process: {len(all_dirs)}")

#     num_gpus = min(args.num_gpus, torch.cuda.device_count())
#     if num_gpus == 0:
#         print("No GPU found!"); return

#     # 均匀分配目录到各个 GPU
#     chunks = [all_dirs[i::num_gpus] for i in range(num_gpus)]
    
#     mp_args = [(i, chunks[i], args.model_path) for i in range(num_gpus)]
    
#     # 使用 'spawn' 启动方法防止 CUDA 运行环境冲突
#     ctx = multiprocessing.get_context('spawn')
#     with ctx.Pool(num_gpus) as p:
#         p.map(process_gpu_batch, mp_args)

# if __name__ == "__main__":
#     main()

import os
import numpy as np
import torch
from PIL import Image
import tqdm
import glob
import argparse
import multiprocessing
import json
import re
from torch.utils.data import Dataset, DataLoader
import open_clip
from open_clip.factory import _MODEL_CONFIGS, create_model_and_transforms

# [修复 1] 自然排序，防止 1, 10, 2 物理错乱
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# [加速核心 1]：全局扁平化 Dataset
class FastSliceDataset(Dataset):
    def __init__(self, dir_list, preprocess):
        self.samples = []
        self.dir_boundaries = {} 
        self.preprocess = preprocess
        
        current_idx = 0
        for d in dir_list:
            output_npy = d + "_2D.npy"
            
            
            slice_paths = glob.glob(os.path.join(d, "*.png")) + \
                          glob.glob(os.path.join(d, "*.jpg")) + \
                          glob.glob(os.path.join(d, "*.jpeg"))
            slice_paths = sorted(slice_paths, key=natural_sort_key)
            
            if not slice_paths: 
                continue
            
            start_idx = current_idx
            for p in slice_paths:
                self.samples.append(p)
                current_idx += 1
            # 记录这个文件夹在全局特征矩阵中的起止索引
            self.dir_boundaries[d] = (start_idx, current_idx)

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            tensor = self.preprocess(img)
        except Exception:
            # 图片损坏则使用黑图兜底，保证形状对齐不断层
            tensor = torch.zeros((3, 224, 224), dtype=torch.float32)
        return tensor, idx

def process_gpu_batch(batch_args):
    gpu_id, dir_list, model_path, batch_size, num_workers = batch_args
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(gpu_id)

    # 1. 加载 BiomedCLIP
    config_path = os.path.join(model_path, "open_clip_config.json")
    ckpt_path = os.path.join(model_path, "open_clip_pytorch_model.bin")
    with open(config_path, "r") as f:
        config_data = json.load(f)
        model_cfg = config_data["model_cfg"]
        preprocess_cfg = config_data["preprocess_cfg"]

    model_name = "biomedclip_local"
    if model_name not in _MODEL_CONFIGS:
        _MODEL_CONFIGS[model_name] = model_cfg

    print(f"[GPU {gpu_id}] Loading BiomedCLIP...")
    model, _, preprocess = create_model_and_transforms(
        model_name=model_name, pretrained=ckpt_path,
        **{f"image_{k}": v for k, v in preprocess_cfg.items()}
    )
    model = model.to(device).eval()

    # 2. 构建 Dataset 和 Dataloader
    print(f"[GPU {gpu_id}] Scanning directories and building dataset...")
    dataset = FastSliceDataset(dir_list, preprocess)
    if len(dataset) == 0:
        print(f"[GPU {gpu_id}] No new images to process.")
        return

    # [加速核心 2]：开启 num_workers 后台异步读图，pin_memory 加速显存拷贝
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True,
        drop_last=False
    )

    print(f"[GPU {gpu_id}] Start inference on {len(dataset)} slices (Batch Size: {batch_size})...")
    
    # 预先在 CPU 内存分配一整块大张量 (避免频繁 List append 导致内存碎片化)
    all_features = torch.zeros((len(dataset), 512), dtype=torch.float32)

    # 3. 极速推理循环
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for batch_imgs, batch_idxs in tqdm.tqdm(dataloader, desc=f"GPU {gpu_id}", position=gpu_id):
            # non_blocking=True 允许后台异步数据传输
            batch_imgs = batch_imgs.to(device, non_blocking=True)
            
            # [修复 2]：不再随机映射，保留最纯净的 512 维医学特征！
            feats = model.encode_image(batch_imgs)
            feats = torch.nn.functional.normalize(feats, dim=-1)
            
            # 存回 CPU
            all_features[batch_idxs] = feats.cpu().float()

    # 4. 根据起止索引，批量切分并保存
    print(f"[GPU {gpu_id}] Saving features to disk...")
    for d, (start, end) in dataset.dir_boundaries.items():
        output_npy = d + "_2D.npy"
        np.save(output_npy, all_features[start:end].numpy())
        
    print(f"[GPU {gpu_id}] Finished!")

def get_all_target_dirs(base_dir):
    target_dirs = []
    print("Scanning directories...")
    for root, dirs, files in os.walk(base_dir):
        if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files):
            target_dirs.append(root)
    return target_dirs

def main():
    parser = argparse.ArgumentParser()
    # base_dir 应该是包含所有病人文件夹的根目录
    parser.add_argument('--base_dir', type=str, default="/data/esh/HSENet/m3d_data/M3D-Cap/M3D_Cap/ct_case")
    parser.add_argument('--model_path', type=str, default="/data/esh/HSENet/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=2) 
    args = parser.parse_args()

    all_dirs = get_all_target_dirs(args.base_dir)
    print(f"Total directories to process: {len(all_dirs)}")

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus == 0:
        print("No GPU found!"); return

    # 均匀分配目录到各个 GPU
    chunks = [all_dirs[i::num_gpus] for i in range(num_gpus)]
    mp_args = [(i, chunks[i], args.model_path, args.batch_size, args.num_workers) for i in range(num_gpus)]
    
    # [核心修复]：放弃使用 Pool，改用原生的 Process，解除 Daemon 限制！
    ctx = multiprocessing.get_context('spawn')
    processes = []
    
    # 启动所有 GPU 进程
    for i in range(num_gpus):
        p = ctx.Process(target=process_gpu_batch, args=(mp_args[i],))
        p.start()
        processes.append(p)
        
    # 等待所有 GPU 进程结束
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
