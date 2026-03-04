# import random
# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset, ConcatDataset
# import json
# import pandas as pd
# import monai.transforms as mtf
# from monai.data import load_decathlon_datalist, set_track_meta
# from .dataset_info import dataset_info
# from .prompt_templates import Caption_templates, PosREC_templates, PosREG_templates, Seg_templates, Radgeome_vqa_templates
# from .term_dictionary import term_dict
# import glob
# from PIL import Image
# import torchvision.transforms as transforms
# from functools import partial
# import torch.nn.functional as F
# import nibabel as nib
# import tqdm

# # --- Helper: Efficiently Load or Create 2D Features ---
# def load_or_create_2d_feat(image_path, num_slices=4, hidden_dim=768):
#     try:
#         # Assuming rule: /path/to/image.npy -> /path/to/image_2D.npy
#         feat_path = image_path.replace(".npy", "_2D.npy")
#         if os.path.exists(feat_path):
#             # Load with mmap to speed up if file is huge, though for npy it loads all usually
#             feats = np.load(feat_path, mmap_mode='r') 
            
#             total_slices = feats.shape[0]
#             if total_slices >= num_slices:
#                 indices = np.sort(np.random.choice(total_slices, num_slices, replace=False))
#                 feats = feats[indices]
#             else:
#                 pad = np.zeros((num_slices - total_slices, feats.shape[1]), dtype=feats.dtype)
#                 feats = np.concatenate([feats, pad], axis=0)
            
#             # Convert to tensor copy
#             return torch.from_numpy(np.array(feats)).float()
#     except Exception:
#         pass
    
#     # Fallback: Return zero tensor (on CPU initially)
#     return torch.zeros((num_slices, hidden_dim), dtype=torch.float)


# class ITRDataset(Dataset):
#     def __init__(self, args, tokenizer, mode="train"):
#         self.args = args
#         self.data_root = args.data_root
#         self.tokenizer = tokenizer
#         self.mode = mode

#         with open(args.cap_data_path, 'r') as file:
#             self.json_file = json.load(file)
#         self.data_list = self.json_file[mode]

#         train_transform = mtf.Compose([
#             mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlip(prob=0.10, spatial_axis=0),
#             mtf.RandFlip(prob=0.10, spatial_axis=1),
#             mtf.RandFlip(prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensity(factors=0.1, prob=0.5),
#             mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
#             mtf.ToTensor(dtype=torch.float),
#         ])
#         val_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
#         set_track_meta(False)

#         if mode == 'train':
#             self.transform = train_transform
#         elif mode == 'validation':
#             self.transform = val_transform
#             self.data_list = self.data_list[:512]
#         elif 'test' in mode:
#             self.transform = val_transform

#     def __len__(self):
#         return len(self.data_list)

#     def truncate_text(self, input_text, max_tokens):
#         def count_tokens(text):
#             tokens = self.tokenizer.encode(text, add_special_tokens=True)
#             return len(tokens)
#         if count_tokens(input_text) <= max_tokens:
#             return input_text
#         sentences = input_text.split('.')
#         selected_sentences = []
#         current_tokens = 0
#         if sentences: selected_sentences.append(sentences.pop(0))
#         while current_tokens <= max_tokens and sentences:
#             random_sentence = random.choice(sentences)
#             new_tokens_len = count_tokens(random_sentence)
#             if current_tokens + new_tokens_len <= max_tokens and random_sentence not in selected_sentences:
#                 selected_sentences.append(random_sentence)
#                 current_tokens += new_tokens_len
#             else:
#                 sentences.remove(random_sentence)
#         return '.'.join(selected_sentences)

#     def __getitem__(self, idx):
#         max_attempts = 100
#         for _ in range(max_attempts):
#             try:
#                 data = self.data_list[idx]
#                 image_path = data["image"]
#                 image_abs_path = os.path.join(self.data_root, image_path)
#                 image = np.load(image_abs_path)
#                 image = self.transform(image)

#                 text_path = data["text"]
#                 text_abs_path = os.path.join(self.data_root, text_path)
#                 with open(text_abs_path, 'r') as text_file:
#                     raw_text = text_file.read()
#                 text = self.truncate_text(raw_text, self.args.max_length)

#                 text_tensor = self.tokenizer(text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
#                 input_id = text_tensor["input_ids"][0]
#                 attention_mask = text_tensor["attention_mask"][0]

#                 return {
#                     'image': image,
#                     'text': text,
#                     'input_id': input_id,
#                     'attention_mask': attention_mask,
#                     'question_type': "Image_text_retrieval",
#                 }
#             except Exception as e:
#                 idx = random.randint(0, len(self.data_list) - 1)

# class CT_RateDataset(Dataset):
#     def __init__(self, args, tokenizer, mode="train"):
#         self.args = args
#         self.data_root = args.data_root
#         self.tokenizer = tokenizer
#         self.mode = mode
#         with open(args.cap_data_path, 'r') as file:
#             self.json_file = json.load(file)
#         self.data_list = self.json_file[mode]
#         train_transform = mtf.Compose([
#             mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlip(prob=0.10, spatial_axis=0),
#             mtf.RandFlip(prob=0.10, spatial_axis=1),
#             mtf.RandFlip(prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensity(factors=0.1, prob=0.5),
#             mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
#             mtf.ToTensor(dtype=torch.float),
#         ])
#         val_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
#         set_track_meta(False)
#         self.transform = train_transform if mode == 'train' else val_transform
#         if mode == 'validation': self.data_list = self.data_list[:512]

#     def __len__(self): return len(self.data_list)
#     def truncate_text(self, input_text, max_tokens): return input_text 

#     def __getitem__(self, idx):
#         max_attempts = 100
#         for _ in range(max_attempts):
#             try:
#                 data = self.data_list[idx]
#                 image_path = data["image"]
#                 image = np.load(image_path)
#                 image = self.transform(image)
#                 raw_text = data["text"].replace('"', '').replace('\'', '').replace('(', '').replace(')', '')
#                 text = self.truncate_text(raw_text, self.args.max_length)
#                 text_tensor = self.tokenizer(text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
#                 return {
#                     'image': image,
#                     'text': text,
#                     'input_id': text_tensor["input_ids"][0],
#                     'attention_mask': text_tensor["attention_mask"][0],
#                     'question_type': "Image_text_retrieval",
#                 }
#             except Exception:
#                 idx = random.randint(0, len(self.data_list) - 1)

# class CapDataset_CT_Rate(Dataset):
#     def __init__(self, args, tokenizer, mode="train"):
#         self.args = args
#         self.data_root = args.data_root
#         self.tokenizer = tokenizer
#         self.mode = mode
#         self.image_tokens = "<im_patch>" * args.proj_out_num
#         with open(args.cap_data_path, 'r') as file:
#             self.json_file = json.load(file)
#         self.data_list = self.json_file[mode]
#         self.caption_prompts = Caption_templates
#         train_transform = mtf.Compose([
#             mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlip(prob=0.10, spatial_axis=0),
#             mtf.RandFlip(prob=0.10, spatial_axis=1),
#             mtf.RandFlip(prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensity(factors=0.1, prob=0.5),
#             mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
#             mtf.ToTensor(dtype=torch.float),
#         ])
#         val_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
#         set_track_meta(False)
#         self.transform = train_transform if mode == 'train' else val_transform

#     def __len__(self): return len(self.data_list)

#     def __getitem__(self, idx):
#         max_attempts = 100
#         for _ in range(max_attempts):
#             try:
#                 data = self.data_list[idx]
#                 image_path = data["image"]
#                 image = np.load(image_path)
#                 image = self.transform(image)
                
#                 # [NEW] Load 2D Features
#                 image_2d = np.zeros((4, 768), dtype=np.float32)
#                 if "biomedclip_features" in data:
#                      image_2d = np.load(data["biomedclip_features"])
#                 image_2d = torch.from_numpy(image_2d).float()

#                 raw_text = data["text"].replace('"', '').replace('\'', '').replace('(', '').replace(')', '')
#                 prompt_question = random.choice(self.caption_prompts)
#                 question = self.image_tokens + prompt_question
#                 if self.tokenizer.bos_token: question = self.tokenizer.bos_token + question
                
#                 text_tensor = self.tokenizer(question + ' ' + raw_text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#                 input_id = text_tensor["input_ids"][0]
#                 attention_mask = text_tensor["attention_mask"][0]
#                 q_tensor = self.tokenizer(question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#                 label = input_id.clone()
#                 label[:torch.sum(q_tensor["attention_mask"][0])] = -100
#                 if self.tokenizer.pad_token_id: label[label == self.tokenizer.pad_token_id] = -100

#                 return {
#                     'image': image,
#                     'input_id': input_id,
#                     'label': label,
#                     'attention_mask': attention_mask,
#                     'question': question,
#                     'answer': raw_text,
#                     'question_type': "Caption",
#                     'image_2d': image_2d # [NEW]
#                 }
#             except Exception:
#                 idx = random.randint(0, len(self.data_list) - 1)

# class VQADataset_CT_Rate(Dataset):
#     def __init__(self, args, tokenizer, mode="train"):
#         self.args = args
#         self.data_root = args.data_root
#         self.tokenizer = tokenizer
#         self.mode = mode
#         self.image_tokens = "<im_patch>" * args.proj_out_num
#         with open(args.cap_data_path, 'r') as file:
#             self.json_file = json.load(file)
#         self.data_list = self.json_file[mode]
#         self.caption_prompts = Caption_templates
#         self.radgeome_vqa_prompts = Radgeome_vqa_templates
#         train_transform = mtf.Compose([
#             mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlip(prob=0.10, spatial_axis=0),
#             mtf.RandFlip(prob=0.10, spatial_axis=1),
#             mtf.RandFlip(prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensity(factors=0.1, prob=0.5),
#             mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
#             mtf.ToTensor(dtype=torch.float),
#         ])
#         val_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
#         set_track_meta(False)
#         self.transform = train_transform if mode == 'train' else val_transform
#         if mode == 'validation': self.data_list = self.data_list[:1024]

#     def __len__(self): return len(self.data_list)

#     def __getitem__(self, idx):
#         max_attempts = 100
#         for _ in range(max_attempts):
#             try:
#                 data = self.data_list[idx]
#                 image_path = data["image"]
#                 image = np.load(image_path)
#                 image = self.transform(image)
                
#                 # [NEW]
#                 image_2d = np.zeros((4, 768), dtype=np.float32)
#                 if "biomedclip_features" in data:
#                      image_2d = np.load(data["biomedclip_features"])
#                 image_2d = torch.from_numpy(image_2d).float()

#                 anatomy_name = data["anatomy"]
#                 prompt_question_temp = random.choice(self.radgeome_vqa_prompts["location"])
#                 abnormality_name = data["abnormality"]
#                 prompt_question = prompt_question_temp.split("{abnormality}")[0] + abnormality_name + prompt_question_temp.split("{abnormality}")[-1]
                
#                 question = self.image_tokens + prompt_question
#                 if self.tokenizer.bos_token: question = self.tokenizer.bos_token + question
                
#                 text_tensor = self.tokenizer(question + ' ' + anatomy_name, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#                 input_id = text_tensor["input_ids"][0]
#                 attention_mask = text_tensor["attention_mask"][0]
#                 q_tensor = self.tokenizer(question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#                 label = input_id.clone()
#                 label[:torch.sum(q_tensor["attention_mask"][0])] = -100
#                 if self.tokenizer.pad_token_id: label[label == self.tokenizer.pad_token_id] = -100

#                 ret = {
#                     'image': image,
#                     'input_id': input_id,
#                     'label': label,
#                     'attention_mask': attention_mask,
#                     'question': question,
#                     'answer': anatomy_name,
#                     'question_type': "Caption",
#                     'image_2d': image_2d # [NEW]
#                 }
#                 if self.args.seg_enable: ret.update({'seg': torch.zeros_like(image)})
#                 return ret
#             except Exception:
#                 idx = random.randint(0, len(self.data_list) - 1)
# class CapDataset(Dataset):
#     def __init__(self, args, tokenizer, mode="train"):
#         self.args = args
#         self.data_root = args.data_root
#         self.tokenizer = tokenizer
#         self.mode = mode
#         self.image_tokens = "<im_patch>" * args.proj_out_num
#         with open(args.cap_data_path, 'r') as file:
#             self.json_file = json.load(file)
#         self.data_list = self.json_file[mode]
#         self.caption_prompts = Caption_templates
        
#         train_transform = mtf.Compose([
#             mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlip(prob=0.10, spatial_axis=0),
#             mtf.RandFlip(prob=0.10, spatial_axis=1),
#             mtf.RandFlip(prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensity(factors=0.1, prob=0.5),
#             mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
#             mtf.ToTensor(dtype=torch.float),
#         ])
#         val_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
#         set_track_meta(False)
#         self.transform = train_transform if mode == 'train' else val_transform

#     def __len__(self): return len(self.data_list)

#     def __getitem__(self, idx):
#         data = self.data_list[idx]
#         image_path = os.path.join(self.data_root, data["image"])
        
#         # Check if image exists
#         if not os.path.exists(image_path):
#             print(f"\n[CRITICAL ERROR] Image not found at: {image_path}")
#             print(f"Skipping this sample.")
#             return self.__getitem__((idx + 1) % len(self.data_list))  # Skip the missing file and continue

#         try:
#             image = np.load(image_path)
#             image = self.transform(image)
#             image_2d = load_or_create_2d_feat(image_path)

#             text_path = os.path.join(self.data_root, data["text"])
#             with open(text_path, 'r') as text_file: raw_text = text_file.read()
            
#             prompt_question = random.choice(self.caption_prompts)
#             question = self.image_tokens + prompt_question
#             if self.tokenizer.bos_token: question = self.tokenizer.bos_token + question

#             text_tensor = self.tokenizer(question + ' ' + raw_text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#             input_id = text_tensor["input_ids"][0]
#             attention_mask = text_tensor["attention_mask"][0]
#             q_tensor = self.tokenizer(question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#             label = input_id.clone()
#             label[:torch.sum(q_tensor["attention_mask"][0])] = -100
#             if self.tokenizer.pad_token_id: label[label == self.tokenizer.pad_token_id] = -100

#             return {
#                 'image': image,
#                 'input_id': input_id,
#                 'label': label,
#                 'attention_mask': attention_mask,
#                 'question': question,
#                 'answer': raw_text,
#                 'question_type': "Caption",
#                 'image_2d': image_2d
#             }
#         except Exception as e:
#             print(f"[Error loading idx {idx}]: {e}")
#             return self.__getitem__((idx + 1) % len(self.data_list))  # Skip this sample and try another one
        

# class VQADataset(Dataset):
#     def __init__(self, args, tokenizer, close_ended=True, mode="train"):
#         self.args = args
#         self.data_root = args.data_root
#         self.is_vqa_mark = getattr(args, 'is_vqa_mark', False)
#         self.tokenizer = tokenizer
#         self.mode = mode
#         self.close_ended = close_ended
#         self.image_tokens = "<im_patch>" * args.proj_out_num
        
#         if mode == "train": path = args.vqa_data_train_path
#         elif mode == "validation": path = args.vqa_data_val_path
#         else: path = args.vqa_data_test_path
#         self.data_list = pd.read_csv(path)
#         if mode == "validation": self.data_list = self.data_list[:2048]

#         train_transform = mtf.Compose([
#             mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlip(prob=0.10, spatial_axis=0),
#             mtf.RandFlip(prob=0.10, spatial_axis=1),
#             mtf.RandFlip(prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensity(factors=0.1, prob=0.5),
#             mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
#             mtf.ToTensor(dtype=torch.float),
#         ])
#         val_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
#         set_track_meta(False)
#         self.transform = train_transform if mode == 'train' else val_transform

#     def __len__(self): return len(self.data_list)

#     def __getitem__(self, idx):
#         data = self.data_list.iloc[idx]
#         image_abs_path = os.path.join(self.args.data_root, data["Image Path"])
        
#         if not os.path.exists(image_abs_path):
#             print(f"\n[CRITICAL ERROR] VQA Image not found: {image_abs_path}")
#             print(f"Skipping this sample.")
#             return self.__getitem__((idx + 1) % len(self.data_list))  # Skip the missing file and continue

#         try:
#             image = np.load(image_abs_path)
#             image = self.transform(image)
#             image_2d = load_or_create_2d_feat(image_abs_path)

#             question = data["Question"]
#             if self.close_ended:
#                 if self.is_vqa_mark: question = "Closed VQA Task: " + question
#                 choices = "Choices: A. {} B. {} C. {} D. {}".format(data["Choice A"], data["Choice B"], data["Choice C"], data["Choice D"])
#                 question = question + ' ' + choices
#                 answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
#             else:
#                 if self.is_vqa_mark: question = "Open VQA Task: " + question
#                 answer = str(data["Answer"])

#             question = self.image_tokens + ' ' + question
#             if self.tokenizer.bos_token: question = self.tokenizer.bos_token + question

#             text_tensor = self.tokenizer(question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#             input_id = text_tensor["input_ids"][0]
#             attention_mask = text_tensor["attention_mask"][0]
#             q_tensor = self.tokenizer(question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#             label = input_id.clone()
#             label[:torch.sum(q_tensor["attention_mask"][0])] = -100
#             if self.tokenizer.pad_token_id: label[label == self.tokenizer.pad_token_id] = -100

#             return {
#                 'image': image,
#                 'input_id': input_id,
#                 'label': label,
#                 'attention_mask': attention_mask,
#                 'question': question,
#                 'answer': answer,
#                 'answer_choice': data["Answer Choice"] if self.close_ended else "",
#                 'question_type': data["Question Type"],
#                 'image_2d': image_2d 
#             }
#         except Exception as e:
#             return self.__getitem__((idx + 1) % len(self.data_list))  # Skip this sample and try another one

# class VQAYNDataset(Dataset):
#     def __init__(self, args, tokenizer, mode="train"):
#         self.args = args
#         self.data_root = args.data_root
#         self.tokenizer = tokenizer
#         self.mode = mode
#         self.image_tokens = "<im_patch>" * args.proj_out_num
        
#         if mode == "train": path = args.vqa_yn_data_train_path
#         elif mode == "validation": path = args.vqa_yn_data_val_path
#         else: path = args.vqa_yn_data_test_path
#         self.data_list = pd.read_csv(path)
#         if mode == "validation": self.data_list = self.data_list[:2048]

#         train_transform = mtf.Compose([
#             mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlip(prob=0.10, spatial_axis=0),
#             mtf.RandFlip(prob=0.10, spatial_axis=1),
#             mtf.RandFlip(prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensity(factors=0.1, prob=0.5),
#             mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
#             mtf.ToTensor(dtype=torch.float),
#         ])
#         val_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
#         set_track_meta(False)
#         self.transform = train_transform if mode == 'train' else val_transform

#     def __len__(self): return len(self.data_list)

#     def __getitem__(self, idx):
#         max_attempts = 100
#         for _ in range(max_attempts):
#             try:
#                 data = self.data_list.iloc[idx]
#                 image_abs_path = os.path.join(self.args.data_root, data["Image Path"])
#                 image = np.load(image_abs_path)
#                 image = self.transform(image)

#                 # [NEW] Add 2D Features
#                 image_2d = load_or_create_2d_feat(image_abs_path)

#                 question = self.image_tokens + ' ' + data["Question"]
#                 if self.tokenizer.bos_token: question = self.tokenizer.bos_token + question
#                 answer = str(data["Answer"])

#                 text_tensor = self.tokenizer(question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#                 input_id = text_tensor["input_ids"][0]
#                 attention_mask = text_tensor["attention_mask"][0]
#                 q_tensor = self.tokenizer(question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#                 label = input_id.clone()
#                 label[:torch.sum(q_tensor["attention_mask"][0])] = -100
#                 if self.tokenizer.pad_token_id: label[label == self.tokenizer.pad_token_id] = -100

#                 return {
#                     'image': image,
#                     'input_id': input_id,
#                     'label': label,
#                     'attention_mask': attention_mask,
#                     'question': question,
#                     'answer': answer,
#                     'answer_choice': data["Answer Choice"],
#                     'question_type': data["Question Type"],
#                     'image_2d': image_2d # [NEW]
#                 }
#             except Exception:
#                 idx = random.randint(0, len(self.data_list) - 1)

# class PosRECDataset(Dataset):
#     def __init__(self, args, tokenizer, tag="0000", description=True, mode='train'):
#         self.args = args
#         self.tokenizer = tokenizer
#         self.tag = tag
#         self.mode = mode
#         self.description = description
#         self.dataset_info = dataset_info
#         self.image_tokens = "<im_patch>" * args.proj_out_num
#         self.box_tokens = ["<bx_start>", "<bx_end>"]
        
#         root_path = args.seg_data_path
#         json_path = os.path.join(root_path, tag, f'{tag}.json')
#         key = "train" if mode == "train" else "test"
#         self.data_list = load_decathlon_datalist(base_dir=root_path, data_list_file_path=json_path, is_segmentation=True, data_list_key=key)

#         train_transform = mtf.Compose([
#             mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
#             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
#             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
#             mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
#             mtf.ToTensord(keys=["image"], dtype=torch.float),
#             mtf.ToTensord(keys=["seg"], dtype=torch.int),
#         ])
#         val_transform = mtf.Compose([
#             mtf.ToTensord(keys=["image"], dtype=torch.float),
#             mtf.ToTensord(keys=["seg"], dtype=torch.int),
#         ])
#         set_track_meta(False)
#         self.transform = train_transform if mode == 'train' else val_transform
#         self.cls_questions = PosREC_templates["cls_questions"]
#         self.des_qustions = PosREC_templates["des_questions"]
#         self.cls_answers = PosREC_templates["cls_answers"]
#         self.des_answers = PosREC_templates["des_answers"]
#         self.cls_no_answers = PosREC_templates["cls_no_answers"]
#         self.des_no_answers = PosREC_templates["des_no_answers"]

#     def __len__(self): return len(self.data_list)

#     def __getitem__(self, idx):
#         max_attempts = 100
#         for _ in range(max_attempts):
#             data = self.data_list[idx]
#             image_array = np.load(data['image'])
#             seg_array = np.load(data['label'])
#             # cls_id = int(os.path.basename(data['label']).split('_')[1].split('.')[0])
#             filename = os.path.basename(data['label'])
#             content_in_brackets = filename.split('(')[1].split(')')[0]
#             cls_id = int(content_in_brackets.split(',')[0].strip())
#             try:
#                 item = self.transform({'image': image_array, 'seg': seg_array})
#                 image = item['image']
#                 seg = item['seg']

#                 # [NEW] 2D Features (Fallback to Zeros for Seg)
#                 image_2d = load_or_create_2d_feat(data['image'])

#                 cls_list = self.dataset_info[self.tag]
#                 vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                
#                 if vld_cls:
#                     box = mask2box(seg[0])
#                     if not self.description:
#                         q = self.image_tokens + ' ' + random.choice(self.cls_questions).format(cls_list[cls_id])
#                         ans = random.choice(self.cls_answers).format(self.box_tokens[0] + str(box) + self.box_tokens[1])
#                     else:
#                         q = self.image_tokens + ' ' + random.choice(self.des_qustions).format(random.choice(term_dict[cls_list[cls_id]]))
#                         ans = random.choice(self.des_answers).format(cls_list[cls_id], self.box_tokens[0] + str(box) + self.box_tokens[1])
#                 else:
#                     if not self.description:
#                         q = self.image_tokens + ' ' + random.choice(self.cls_questions).format(cls_list[cls_id])
#                         ans = random.choice(self.cls_no_answers).format(cls_list[cls_id])
#                     else:
#                         q = self.image_tokens + ' ' + random.choice(self.des_qustions).format(random.choice(term_dict[cls_list[cls_id]]))
#                         ans = random.choice(self.des_no_answers).format(cls_list[cls_id])

#                 text_tensor = self.tokenizer(q + ' ' + ans, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
#                 input_id = text_tensor["input_ids"][0]
#                 attention_mask = text_tensor["attention_mask"][0]
#                 q_tensor = self.tokenizer(q, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
#                 label = input_id.clone()
#                 label[:torch.sum(q_tensor["attention_mask"][0])] = -100
#                 if self.tokenizer.pad_token_id: label[label == self.tokenizer.pad_token_id] = -100

#                 return {
#                     'image': image,
#                     'input_id': input_id,
#                     'label': label,
#                     'attention_mask': attention_mask,
#                     'question': q,
#                     'answer': ans,
#                     'question_type': "REC",
#                     'image_2d': image_2d # [NEW]
#                 }
#             except Exception:
#                 idx = random.randint(0, len(self.data_list) - 1)

# class PosREGDataset(Dataset):
#     def __init__(self, args, tokenizer, tag="0000", description=True, mode='train'):
#         self.args = args
#         self.tokenizer = tokenizer
#         self.tag = tag
#         self.mode = mode
#         self.description = description
#         self.dataset_info = dataset_info
#         self.image_tokens = "<im_patch>" * args.proj_out_num
#         self.box_tokens = ["<bx_start>", "<bx_end>"]
        
#         root_path = args.seg_data_path
#         json_path = os.path.join(root_path, tag, f'{tag}.json')
#         key = "train" if mode == "train" else "test"
#         self.data_list = load_decathlon_datalist(base_dir=root_path, data_list_file_path=json_path, is_segmentation=True, data_list_key=key)

#         train_transform = mtf.Compose([
#             mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
#             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
#             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
#             mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
#             mtf.ToTensord(keys=["image"], dtype=torch.float),
#             mtf.ToTensord(keys=["seg"], dtype=torch.int),
#         ])
#         val_transform = mtf.Compose([
#             mtf.ToTensord(keys=["image"], dtype=torch.float),
#             mtf.ToTensord(keys=["seg"], dtype=torch.int),
#         ])
#         set_track_meta(False)
#         self.transform = train_transform if mode == 'train' else val_transform
#         self.cls_questions = PosREG_templates["cls_questions"]
#         self.des_questions = PosREG_templates["des_questions"]
#         self.cls_answers = PosREG_templates["cls_answers"]
#         self.des_answers = PosREG_templates["des_answers"]
#         self.cls_no_questions = PosREC_templates["cls_questions"]
#         self.des_no_questions = PosREC_templates["des_questions"]
#         self.cls_no_answers = PosREG_templates["cls_no_answers"]
#         self.des_no_answers = PosREG_templates["des_no_answers"]

#     def __len__(self): return len(self.data_list)

#     def __getitem__(self, idx):
#         max_attempts = 100
#         for _ in range(max_attempts):
#             data = self.data_list[idx]
#             image_array = np.load(data['image'])
#             seg_array = np.load(data['label'])
#             # cls_id = int(os.path.basename(data['label']).split('_')[1].split('.')[0])
#             filename = os.path.basename(data['label'])
#             content_in_brackets = filename.split('(')[1].split(')')[0]
#             cls_id = int(content_in_brackets.split(',')[0].strip())
#             try:
#                 item = self.transform({'image': image_array, 'seg': seg_array})
#                 image = item['image']
#                 seg = item['seg']

#                 # [NEW] 2D Features
#                 image_2d = torch.zeros((4, 768), dtype=torch.float)

#                 cls_list = self.dataset_info[self.tag]
#                 vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                
#                 if vld_cls:
#                     box = mask2box(seg[0])
#                     box_text = self.box_tokens[0] + str(box) + self.box_tokens[1]
#                     if not self.description:
#                         q = self.image_tokens + ' ' + random.choice(self.cls_questions).format(box_text)
#                         ans = random.choice(self.cls_answers).format(cls_list[cls_id])
#                     else:
#                         q = self.image_tokens + ' ' + random.choice(self.des_questions).format(box_text)
#                         ans = random.choice(self.des_answers).format(cls_list[cls_id], random.choice(term_dict[cls_list[cls_id]]))
#                 else:
#                     if not self.description:
#                         q = self.image_tokens + ' ' + random.choice(self.cls_no_questions).format(cls_list[cls_id])
#                         ans = random.choice(self.cls_no_answers).format(cls_list[cls_id])
#                     else:
#                         q = self.image_tokens + ' ' + random.choice(self.des_no_questions).format(random.choice(term_dict[cls_list[cls_id]]))
#                         ans = random.choice(self.des_no_answers).format(cls_list[cls_id])

#                 text_tensor = self.tokenizer(q + ' ' + ans, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
#                 input_id = text_tensor["input_ids"][0]
#                 attention_mask = text_tensor["attention_mask"][0]
#                 q_tensor = self.tokenizer(q, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
#                 label = input_id.clone()
#                 label[:torch.sum(q_tensor["attention_mask"][0])] = -100
#                 if self.tokenizer.pad_token_id: label[label == self.tokenizer.pad_token_id] = -100

#                 return {
#                     'image': image,
#                     'input_id': input_id,
#                     'label': label,
#                     'attention_mask': attention_mask,
#                     'question': q,
#                     'answer': ans,
#                     'question_type': "REG",
#                     'image_2d': image_2d # [NEW]
#                 }
#             except Exception:
#                 idx = random.randint(0, len(self.data_list) - 1)

# class SegDataset(Dataset):
#     def __init__(self, args, tokenizer, tag="0000", description=False, mode='train'):
#         self.args = args
#         self.tokenizer = tokenizer
#         self.tag = tag
#         self.description = description
#         self.mode = mode
#         self.dataset_info = dataset_info
#         self.image_tokens = "<im_patch>" * args.proj_out_num
        
#         root_path = args.seg_data_path
#         json_path = os.path.join(root_path, tag, f'{tag}.json')
#         key = "train" if mode == "train" else "test"
#         self.data_list = load_decathlon_datalist(base_dir=root_path, data_list_file_path=json_path, is_segmentation=True, data_list_key=key)

#         train_transform = mtf.Compose([
#             mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
#             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
#             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
#             mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
#             mtf.ToTensord(keys=["image"], dtype=torch.float),
#             mtf.ToTensord(keys=["seg"], dtype=torch.int),
#         ])
#         val_transform = mtf.Compose([
#             mtf.ToTensord(keys=["image"], dtype=torch.float),
#             mtf.ToTensord(keys=["seg"], dtype=torch.int),
#         ])
#         set_track_meta(False)
#         self.transform = train_transform if mode == 'train' else val_transform
#         self.cls_questions = Seg_templates["cls_questions"]
#         self.des_questions = Seg_templates["des_questions"]
#         self.cls_answers = Seg_templates["cls_answers"]
#         self.des_answers = Seg_templates["des_answers"]
#         self.cls_no_answers = Seg_templates["cls_no_answers"]
#         self.des_no_answers = Seg_templates["des_no_answers"]

#     def __len__(self): return len(self.data_list)

#     def __getitem__(self, idx):
#         max_attempts = 100
#         for _ in range(max_attempts):
#             data = self.data_list[idx]
#             image_array = np.load(data['image'])
#             seg_array = np.load(data['label'])
#             # cls_id = int(os.path.basename(data['label']).split('_')[1].split('.')[0])
#             filename = os.path.basename(data['label'])
#             content_in_brackets = filename.split('(')[1].split(')')[0]
#             cls_id = int(content_in_brackets.split(',')[0].strip())
#             try:
#                 item = self.transform({'image': image_array, 'seg': seg_array})
#                 image = item['image']
#                 seg = item['seg']

#                 # [NEW] 2D Features
#                 image_2d = torch.zeros((4, 768), dtype=torch.float)

#                 cls_list = self.dataset_info[self.tag]
#                 vld_cls = torch.nonzero(torch.sum(seg, dim=(1, 2, 3))).flatten().tolist()
                
#                 if vld_cls:
#                     if not self.description:
#                         q = self.image_tokens + ' ' + random.choice(self.cls_questions).format(cls_list[cls_id])
#                         ans = random.choice(self.cls_answers)
#                     else:
#                         q = self.image_tokens + ' ' + random.choice(self.des_questions).format(random.choice(term_dict[cls_list[cls_id]]))
#                         ans = random.choice(self.des_answers).format(cls_list[cls_id])
#                 else:
#                     if not self.description:
#                         q = self.image_tokens + ' ' + random.choice(self.cls_questions).format(cls_list[cls_id])
#                         ans = random.choice(self.cls_no_answers).format(cls_list[cls_id])
#                     else:
#                         q = self.image_tokens + ' ' + random.choice(self.des_questions).format(random.choice(term_dict[cls_list[cls_id]]))
#                         ans = random.choice(self.des_no_answers).format(cls_list[cls_id])

#                 text_tensor = self.tokenizer(q + ' ' + ans, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
#                 input_id = text_tensor["input_ids"][0]
#                 attention_mask = text_tensor["attention_mask"][0]
#                 q_tensor = self.tokenizer(q, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
#                 label = input_id.clone()
#                 label[:torch.sum(q_tensor["attention_mask"][0])] = -100
#                 if self.tokenizer.pad_token_id: label[label == self.tokenizer.pad_token_id] = -100

#                 ret = {
#                     'image': image,
#                     'input_id': input_id,
#                     'label': label,
#                     'seg': seg,
#                     'attention_mask': attention_mask,
#                     'question': q,
#                     'answer': ans,
#                     'question_type': "seg",
#                     'image_2d': image_2d # [NEW]
#                 }
#                 return ret
#             except Exception:
#                 idx = random.randint(0, len(self.data_list) - 1)

# class RefSegDataset(Dataset):
#     def __init__(self, args, tokenizer, mode="train"):
#         self.args = args
#         self.tokenizer = tokenizer
#         self.mode = mode
#         self.image_tokens = "<im_patch>" * args.proj_out_num
#         path = args.refseg_data_train_path if mode == 'train' else args.refseg_data_test_path
#         self.data_list = pd.read_csv(path, engine='python')

#         train_transform = mtf.Compose([
#             mtf.RandRotate90d(keys=["image", "seg"], prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=0),
#             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=1),
#             mtf.RandFlipd(keys=["image", "seg"], prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
#             mtf.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
#             mtf.ToTensord(keys=["image"], dtype=torch.float),
#             mtf.ToTensord(keys=["seg"], dtype=torch.int),
#         ])
#         val_transform = mtf.Compose([
#             mtf.ToTensord(keys=["image"], dtype=torch.float),
#             mtf.ToTensord(keys=["seg"], dtype=torch.int),
#         ])
#         set_track_meta(False)
#         self.transform = train_transform if mode == 'train' else val_transform

#     def __len__(self): return len(self.data_list)

#     def __getitem__(self, idx):
#         max_attempts = 100
#         for _ in range(max_attempts):
#             try:
#                 data = self.data_list.iloc[idx]
#                 image_array = np.load(os.path.join(self.args.data_root, data["Image"]))
#                 seg_path = os.path.join(self.args.data_root, data["Mask"])
#                 seg_array = np.load(seg_path)
#                 seg_array = (seg_array == data["Mask_ID"]).astype(np.int8)
                
#                 item = self.transform({"image": image_array, "seg": seg_array})
#                 image = item['image']
#                 seg = item['seg']

#                 # [NEW] 2D Features
#                 image_2d = torch.zeros((4, 768), dtype=torch.float)

#                 q = self.image_tokens + ' ' + data["Question"]
#                 ans = data["Answer"]
                
#                 text_tensor = self.tokenizer(q + ' ' + ans, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
#                 input_id = text_tensor["input_ids"][0]
#                 attention_mask = text_tensor["attention_mask"][0]
#                 q_tensor = self.tokenizer(q, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt")
#                 label = input_id.clone()
#                 label[:torch.sum(q_tensor["attention_mask"][0])] = -100
#                 if self.tokenizer.pad_token_id: label[label == self.tokenizer.pad_token_id] = -100

#                 ret = {
#                     'image': image,
#                     'input_id': input_id,
#                     'label': label,
#                     'seg': seg,
#                     'attention_mask': attention_mask,
#                     'question': q,
#                     'answer': ans,
#                     'question_type': "refseg",
#                     'image_2d': image_2d # [NEW]
#                 }
#                 return ret
#             except Exception:
#                 idx = random.randint(0, len(self.data_list) - 1)

# class MultiSegDataset(Dataset):
#     def __init__(self, args, tokenizer, mode='train'):
#         super(MultiSegDataset, self).__init__()
#         self.tokenizer = tokenizer
#         self.dataset_info = dataset_info
#         self.ds_list = []
#         for dataset_code in self.dataset_info.keys():
#             self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
#             self.ds_list.append(SegDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
#         self.dataset = ConcatDataset(self.ds_list)
#     def __len__(self): return len(self.dataset)
#     def __getitem__(self, idx): return self.dataset[idx]

# class MultiPosDataset(Dataset):
#     def __init__(self, args, tokenizer, mode='train'):
#         super(MultiPosDataset, self).__init__()
#         self.tokenizer = tokenizer
#         self.dataset_info = dataset_info
#         self.ds_list = []
#         for dataset_code in self.dataset_info.keys():
#             self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
#             self.ds_list.append(PosRECDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
#             self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=False, mode=mode))
#             self.ds_list.append(PosREGDataset(args, tokenizer, tag=dataset_code, description=True, mode=mode))
#         self.dataset = ConcatDataset(self.ds_list)
#     def __len__(self): return len(self.dataset)
#     def __getitem__(self, idx): return self.dataset[idx]

# class PosSegDatasets(Dataset):
#     def __init__(self, args, tokenizer, mode='train'):
#         super(PosSegDatasets, self).__init__()
#         self.ds_list = [MultiPosDataset(args, tokenizer, mode), MultiSegDataset(args, tokenizer, mode)]
#         self.dataset = ConcatDataset(self.ds_list)
#     def __len__(self): return len(self.dataset)
#     def __getitem__(self, idx): return self.dataset[idx]

# # --- 保持原有结构，仅在内部使用了更新后的 Dataset 类 ---
# class TextDatasets(Dataset):
#     def __init__(self, args, tokenizer, mode='train'):
#         super(TextDatasets, self).__init__()
#         self.ds_list = [
#             CapDataset(args, tokenizer, mode),
#             # VQADataset(args, tokenizer, close_ended=True, mode=mode),
#             # VQADataset(args, tokenizer, close_ended=False, mode=mode),
#         ]
#         self.dataset = ConcatDataset(self.ds_list)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

# class UniDatasets(Dataset):
#     def __init__(self, args, tokenizer, mode='train'):
#         super(UniDatasets, self).__init__()
#         self.ds_list = [
#             CapDataset(args, tokenizer, mode),
#             VQADataset(args, tokenizer, close_ended=True, mode=mode),
#             VQADataset(args, tokenizer, close_ended=False, mode=mode),
#             # MultiPosDataset(args, tokenizer, mode),
#             # MultiSegDataset(args, tokenizer, mode),
#         ]
#         self.dataset = ConcatDataset(self.ds_list)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]

##**************************************************************************************************************
# import os
# import random
# import numpy as np
# import torch
# import pandas as pd
# import json
# import monai.transforms as mtf
# from torch.utils.data import Dataset, ConcatDataset
# from monai.data import set_track_meta

# # 假设你的特征加载函数在外部定义，这里给一个兜底实现防止报错
# def load_or_create_2d_feat(image_path):
#     # 这里应为你原本的逻辑，若无则返回 dummy 占位
#     return torch.zeros((4, 768), dtype=torch.float)

# Caption_templates = ["Describe this medical image.", "What does this image show?", "Provide a caption for this scan."]

# class CapDataset(Dataset):
#     def __init__(self, args, tokenizer, mode="train"):
#         self.args = args
#         self.data_root = args.data_root
#         self.tokenizer = tokenizer
#         self.mode = mode
#         self.image_tokens = "<im_patch>" * args.proj_out_num
#         with open(args.cap_data_path, 'r') as file:
#             self.json_file = json.load(file)
#         self.data_list = self.json_file[mode]
#         self.caption_prompts = Caption_templates
        
#         train_transform = mtf.Compose([
#             mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlip(prob=0.10, spatial_axis=0),
#             mtf.RandFlip(prob=0.10, spatial_axis=1),
#             mtf.RandFlip(prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensity(factors=0.1, prob=0.5),
#             mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
#             mtf.ToTensor(dtype=torch.float),
#         ])
#         val_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
#         set_track_meta(False)
#         self.transform = train_transform if mode == 'train' else val_transform

#     def __len__(self): return len(self.data_list)

#     def __getitem__(self, idx):
#         data = self.data_list[idx]
#         image_path = os.path.join(self.data_root, data["image"])
        
#         # [核心修复] 检查文件是否存在，不存在则安全跳过，不报错
#         if not os.path.exists(image_path):
#             return self.__getitem__((idx + 1) % len(self.data_list))

#         try:
#             image = np.load(image_path)
#             image = self.transform(image)
#             image_2d = load_or_create_2d_feat(image_path)

#             text_path = os.path.join(self.data_root, data["text"])
#             with open(text_path, 'r') as text_file: raw_text = text_file.read()
            
#             prompt_question = random.choice(self.caption_prompts)
#             question = self.image_tokens + prompt_question
#             if self.tokenizer.bos_token: question = self.tokenizer.bos_token + question

#             text_tensor = self.tokenizer(question + ' ' + raw_text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#             input_id = text_tensor["input_ids"][0]
#             attention_mask = text_tensor["attention_mask"][0]
#             q_tensor = self.tokenizer(question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#             label = input_id.clone()
#             label[:torch.sum(q_tensor["attention_mask"][0])] = -100
#             if self.tokenizer.pad_token_id: label[label == self.tokenizer.pad_token_id] = -100

#             # [核心修复] 严格统一下游获取的 Key
#             return {
#                 'image': image,
#                 'input_id': input_id,
#                 'label': label,
#                 'attention_mask': attention_mask,
#                 'question': question,
#                 'answer': raw_text,
#                 'question_type': "Caption",
#                 'image_2d': image_2d
#             }
#         except Exception as e:
#             return self.__getitem__((idx + 1) % len(self.data_list))
        
# class VQADataset(Dataset):
#     def __init__(self, args, tokenizer, close_ended=True, mode="train"):
#         self.args = args
#         self.data_root = args.data_root
#         self.is_vqa_mark = getattr(args, 'is_vqa_mark', False)
#         self.tokenizer = tokenizer
#         self.mode = mode
#         self.close_ended = close_ended
#         self.image_tokens = "<im_patch>" * args.proj_out_num
        
#         if mode == "train": path = args.vqa_data_train_path
#         elif mode == "validation": path = args.vqa_data_val_path
#         else: path = args.vqa_data_test_path
#         self.data_list = pd.read_csv(path)
#         if mode == "validation": self.data_list = self.data_list[:2048]

#         train_transform = mtf.Compose([
#             mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
#             mtf.RandFlip(prob=0.10, spatial_axis=0),
#             mtf.RandFlip(prob=0.10, spatial_axis=1),
#             mtf.RandFlip(prob=0.10, spatial_axis=2),
#             mtf.RandScaleIntensity(factors=0.1, prob=0.5),
#             mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
#             mtf.ToTensor(dtype=torch.float),
#         ])
#         val_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
#         set_track_meta(False)
#         self.transform = train_transform if mode == 'train' else val_transform

#     def __len__(self): return len(self.data_list)

#     def __getitem__(self, idx):
#         data = self.data_list.iloc[idx]
#         image_abs_path = os.path.join(self.args.data_root, data["Image Path"])
        
#         # [核心修复] 文件丢失直接跳过
#         if not os.path.exists(image_abs_path):
#             return self.__getitem__((idx + 1) % len(self.data_list))

#         try:
#             image = np.load(image_abs_path)
#             image = self.transform(image)
#             image_2d = load_or_create_2d_feat(image_abs_path)

#             question = data["Question"]
#             if self.close_ended:
#                 if self.is_vqa_mark: question = "Closed VQA Task: " + question
#                 choices = "Choices: A. {} B. {} C. {} D. {}".format(data["Choice A"], data["Choice B"], data["Choice C"], data["Choice D"])
#                 question = question + ' ' + choices
#                 answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
#             else:
#                 if self.is_vqa_mark: question = "Open VQA Task: " + question
#                 answer = str(data["Answer"])

#             question = self.image_tokens + ' ' + question
#             if self.tokenizer.bos_token: question = self.tokenizer.bos_token + question

#             text_tensor = self.tokenizer(question + ' ' + answer, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#             input_id = text_tensor["input_ids"][0]
#             attention_mask = text_tensor["attention_mask"][0]
#             q_tensor = self.tokenizer(question, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
#             label = input_id.clone()
#             label[:torch.sum(q_tensor["attention_mask"][0])] = -100
#             if self.tokenizer.pad_token_id: label[label == self.tokenizer.pad_token_id] = -100

#             # [核心修复] 严格统一下游获取的 Key
#             return {
#                 'image': image,
#                 'input_id': input_id,
#                 'label': label,
#                 'attention_mask': attention_mask,
#                 'question': question,
#                 'answer': answer,
#                 'answer_choice': data["Answer Choice"] if self.close_ended else "",
#                 'question_type': data["Question Type"],
#                 'image_2d': image_2d 
#             }
#         except Exception as e:
#             return self.__getitem__((idx + 1) % len(self.data_list))

# class UniDatasets(Dataset):
#     def __init__(self, args, tokenizer, mode='train'):
#         super(UniDatasets, self).__init__()
#         self.ds_list = [
#             CapDataset(args, tokenizer, mode),
#             VQADataset(args, tokenizer, close_ended=True, mode=mode),
#             VQADataset(args, tokenizer, close_ended=False, mode=mode),
#         ]
#         self.dataset = ConcatDataset(self.ds_list)


#     def __len__(self): 
#         return len(self.dataset)
        
#     def __getitem__(self, idx): 
#         return self.dataset[idx]




import os
import random
import numpy as np
import torch
import pandas as pd
import json
import monai.transforms as mtf
from torch.utils.data import Dataset, ConcatDataset
from monai.data import set_track_meta

# 兜底 2D 特征
def load_or_create_2d_feat(image_path):
    return torch.zeros((4, 768), dtype=torch.bfloat16)

Caption_templates = ["Describe this medical image.", "What does this image show?", "Provide a caption for this scan."]

class CapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.image_tokens = "<im_patch>" * args.proj_out_num
        with open(args.cap_data_path, 'r') as file:
            self.json_file = json.load(file)
        self.data_list = self.json_file[mode]
        self.caption_prompts = Caption_templates
        
        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        val_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
        set_track_meta(False)
        self.transform = train_transform if mode == 'train' else val_transform

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        image_path = os.path.join(self.data_root, data["image"])
        
        if not os.path.exists(image_path):
            return self.__getitem__((idx + 1) % len(self.data_list))

        try:
            image = np.load(image_path)
            image = self.transform(image)
            image_2d = load_or_create_2d_feat(image_path)

            text_path = os.path.join(self.data_root, data["text"])
            with open(text_path, 'r') as text_file: raw_text = text_file.read()
            
            # [致命修复 1] 组装 Prompt
            prompt_question = random.choice(self.caption_prompts)
            prompt_text = self.image_tokens + " " + prompt_question
            if self.tokenizer.bos_token: 
                prompt_text = self.tokenizer.bos_token + prompt_text

            # [致命修复 2] 给答案强行加上 EOS Token，教模型闭嘴
            eos = self.tokenizer.eos_token if self.tokenizer.eos_token else "<|endoftext|>"
            full_text = prompt_text + " Answer: " + raw_text + eos

            text_tensor = self.tokenizer(full_text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]
            
            # [致命修复 3] 精准计算 prompt 长度，把 prompt 部分的 label 设为 -100
            prompt_tensor = self.tokenizer(prompt_text + " Answer: ", max_length=self.args.max_length, truncation=True, return_tensors="pt", add_special_tokens=False)
            prompt_len = prompt_tensor["input_ids"].shape[1]

            label = input_id.clone()
            label[:prompt_len] = -100 # 不计算问题部分的 Loss
            if self.tokenizer.pad_token_id is not None: 
                label[label == self.tokenizer.pad_token_id] = -100 # 不计算填充部分的 Loss

            return {
                'image': image,
                'input_id': input_id,
                'label': label,
                'attention_mask': attention_mask,
                'question': prompt_text,
                'answer': raw_text,
                'question_type': "Caption",
                'image_2d': image_2d
            }
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.data_list))
        
class VQADataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.is_vqa_mark = getattr(args, 'is_vqa_mark', False)
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended
        self.image_tokens = "<im_patch>" * args.proj_out_num
        
        if mode == "train": path = args.vqa_data_train_path
        elif mode == "validation": path = args.vqa_data_val_path
        else: path = args.vqa_data_test_path
        self.data_list = pd.read_csv(path)
        if mode == "validation": self.data_list = self.data_list[:2048]

        train_transform = mtf.Compose([
            mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
            mtf.RandFlip(prob=0.10, spatial_axis=0),
            mtf.RandFlip(prob=0.10, spatial_axis=1),
            mtf.RandFlip(prob=0.10, spatial_axis=2),
            mtf.RandScaleIntensity(factors=0.1, prob=0.5),
            mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
            mtf.ToTensor(dtype=torch.float),
        ])
        val_transform = mtf.Compose([mtf.ToTensor(dtype=torch.float)])
        set_track_meta(False)
        self.transform = train_transform if mode == 'train' else val_transform

    def __len__(self): return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list.iloc[idx]
        image_abs_path = os.path.join(self.args.data_root, data["Image Path"])
        
        if not os.path.exists(image_abs_path):
            return self.__getitem__((idx + 1) % len(self.data_list))

        try:
            image = np.load(image_abs_path)
            image = self.transform(image)
            image_2d = load_or_create_2d_feat(image_abs_path)

            question = data["Question"]
            
            # [致命修复 1] 保证训练与推理格式百分之百一致
            if self.close_ended:
                question = "Closed VQA Task: " + question
                choices = "Choices: A. {} B. {} C. {} D. {}".format(data["Choice A"], data["Choice B"], data["Choice C"], data["Choice D"])
                question = question + ' ' + choices
                answer = "{}. {}".format(data["Answer Choice"], data["Answer"])
            else:
                question = "Open VQA Task: " + question
                answer = str(data["Answer"])

            prompt_text = self.image_tokens + ' ' + question
            if self.tokenizer.bos_token: 
                prompt_text = self.tokenizer.bos_token + prompt_text

            # [致命修复 2] 尾部添加结束符
            eos = self.tokenizer.eos_token if self.tokenizer.eos_token else "<|endoftext|>"
            full_text = prompt_text + " Answer: " + answer + eos

            text_tensor = self.tokenizer(full_text, max_length=self.args.max_length, truncation=True, padding="max_length", return_tensors="pt", add_special_tokens=False)
            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]
            
            # [致命修复 3] 计算问题长度，设置 Mask
            prompt_tensor = self.tokenizer(prompt_text + " Answer: ", max_length=self.args.max_length, truncation=True, return_tensors="pt", add_special_tokens=False)
            prompt_len = prompt_tensor["input_ids"].shape[1]

            label = input_id.clone()
            label[:prompt_len] = -100 # 不背诵题目
            if self.tokenizer.pad_token_id is not None: 
                label[label == self.tokenizer.pad_token_id] = -100 # 不背诵 PAD

            return {
                'image': image,
                'input_id': input_id,
                'label': label,
                'attention_mask': attention_mask,
                'question': prompt_text,
                'answer': answer,
                'answer_choice': data["Answer Choice"] if self.close_ended else "",
                'question_type': data["Question Type"],
                'image_2d': image_2d 
            }
        except Exception as e:
            return self.__getitem__((idx + 1) % len(self.data_list))

class UniDatasets(Dataset):
    def __init__(self, args, tokenizer, mode='train'):
        super(UniDatasets, self).__init__()
        self.ds_list = [
            CapDataset(args, tokenizer, mode),
            VQADataset(args, tokenizer, close_ended=True, mode=mode),
            VQADataset(args, tokenizer, close_ended=False, mode=mode),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self): 
        return len(self.dataset)
        
    def __getitem__(self, idx): 
        return self.dataset[idx]