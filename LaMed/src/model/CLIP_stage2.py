# import numpy as np
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import PreTrainedModel, PretrainedConfig, BertModel, AutoConfig, AutoModel
# from LaMed.src.model.multimodal_encoder.vit import ViT
# from LaMed.src.model.multimodal_encoder.vit import ViT_stage2
# from LaMed.src.utils.dist_utils import gather_features
# import os
# import json

# loss_logs = []

# class M3DCLIPConfig_stage2(PretrainedConfig):
#     model_type = "m3d_clip_stage2"

#     def __init__(
#         self,
#         language_model_name_or_path: str = "",
#         stage1_pretrained_CLIP_path = "",
#         local_loss: bool = False,
#         gather_loss: bool = True,
#         in_channels: int = 1,
#         img_size: tuple = (32, 256, 256),
#         patch_size: tuple = (4, 16, 16),
#         hidden_size: int = 768,
#         mlp_dim: int = 3072,
#         num_layers: int = 12,
#         num_heads: int = 12,
#         pos_embed: str = "perceptron",
#         dropout_rate: float = 0,
#         spatial_dims: int = 3,
#         max_text_len: int = 128,
#         vocab_size: int = 30522,
#         **kwargs,
#     ):
#         self.language_model_name_or_path = language_model_name_or_path
#         self.in_channels = in_channels
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.hidden_size = hidden_size
#         self.mlp_dim = mlp_dim
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.pos_embed = pos_embed
#         self.dropout_rate = dropout_rate
#         self.spatial_dims = spatial_dims
#         self.local_loss = local_loss
#         self.gather_loss = gather_loss
#         self.max_text_len = max_text_len
#         self.vocab_size = vocab_size
#         super().__init__(**kwargs)

# import open_clip
# current_step = 0

# class M3DCLIP_stage2(PreTrainedModel):
#     config_class = M3DCLIPConfig_stage2

#     def __init__(self, config):
#         super().__init__(config)

#         # Stage2 2D-Enhanced 3D Vision Encoder (Trainable)
#         self.vision_encoder = ViT_stage2(
#             in_channels=config.in_channels,
#             img_size=config.img_size,
#             patch_size=config.patch_size,
#             hidden_size=config.hidden_size,
#             mlp_dim=config.mlp_dim,
#             num_layers=config.num_layers,
#             num_heads=config.num_heads,
#             pos_embed=config.pos_embed,
#             dropout_rate=config.dropout_rate,
#             spatial_dims=config.spatial_dims,
#             classification=True,
#         )

#         # Stage2 Text Encoder (Trainable)
#         self.language_encoder = BertModel.from_pretrained(config.language_model_name_or_path)

#         self.mm_vision_proj = nn.Linear(config.hidden_size, config.hidden_size) 
#         self.mm_language_proj = nn.Linear(config.hidden_size, config.hidden_size)


#         # Stage1 Pretrained CLIP (Frozen)
#         self.stage1_pretrained_CLIP = AutoModel.from_pretrained(config.pretrained_model, trust_remote_code=True)
#         self.stage1_pretrained_CLIP.requires_grad_(False)


#         self.visual_encoder_2D = None
#         if config.use_mask and config.use_2D_Encoder: 
#             model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
#             self.visual_encoder_2D = model.visual.trunk
#             self.visual_encoder_2D.requires_grad_(False)


#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

#         self.local_loss = config.local_loss
#         self.gather_loss = config.gather_loss

#         self.mask_config = config

#     # def encode_image(self, image, images_2d, mask_ratio=None, text_features=None, image_path=None):
#     #     image_feats, _ = self.vision_encoder(image, images_2d, mask_ratio, self.visual_encoder_2D, text_features, image_path) 

#     #     image_feats = self.mm_vision_proj(image_feats) 
#     #     image_feats = F.normalize(image_feats, dim=-1)  
#     #     return image_feats

#     def encode_text(self, input_id, attention_mask):
#         text_feats_language = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"] 
#         text_feats = self.mm_language_proj(text_feats_language) 
#         text_feats = F.normalize(text_feats, dim=-1) 

#         return text_feats, text_feats_language
    
#     # esh do
#     def loss_A1(self, cls_feats, text_feats):
#         # cls_feats:  [B, C]
#         # text_feats: [B, C]

#         # normalize
#         cls_norm = F.normalize(cls_feats, dim=-1)      # [B, C]
#         txt_norm = F.normalize(text_feats, dim=-1)     # [B, C]

#         # batch-wise similarity: [B, B]
#         sim = cls_norm @ txt_norm.t()

#         # InfoNCE
#         labels = torch.arange(sim.size(0), device=sim.device)
#         loss = F.cross_entropy(sim, labels)

#         return loss

    
#     def loss_A2(self, patch_feats, text_segments, patch_scores, patch2segment):

#         if text_segments.dim() == 2:
#             text_segments = text_segments.unsqueeze(1)  # [B,1,C]

#         B, N, C = patch_feats.shape
#         _, K, _ = text_segments.shape

#         # [B, N]
#         w = F.softmax(patch_scores, dim=-1)

#         # normalize
#         patch_norm = F.normalize(patch_feats, dim=-1)      # [B,N,C]
#         txt_norm = F.normalize(text_segments, dim=-1)      # [B,K,C]

#         # sim: [B, N, K]
#         sim = torch.einsum("bnc,bkc->bnk", patch_norm, txt_norm)

#         # 根据 patch2segment 选出 pos： [B, N]
#         pos = sim.gather(2, patch2segment.unsqueeze(-1)).squeeze(-1)

#         # denom: softmax 分母
#         denom = torch.exp(sim).sum(dim=(1, 2))  # [B]

#         # loss: (w * exp(pos)) sum over all patches
#         loss = -(w * torch.exp(pos)).sum() / denom.sum()

#         return loss

    
#     def loss_A3(self, post, neighbor_idx):
#         """
#         post: SGAT 输出的 patch 特征 (B, N, C)
#         neighbor_idx: (N, K) long tensor (已 register_buffer)
#         """
#         B, N, C = post.shape
#         K = neighbor_idx.shape[1]

#         # 取邻居特征
#         nbr = post[:, neighbor_idx, :]         # (B, N, K, C)
#         nbr_mean = nbr.mean(dim=2)             # (B, N, C)

#         # Laplacian smooth loss
#         loss = F.mse_loss(post, nbr_mean)
#         return loss

#     def loss_B1_text(self, patch_feats, text_cls):
#         """
#         patch_feats: [B, N, C]
#         text_cls: [B, C]
#         """
#         text_cls = text_cls.unsqueeze(1).expand_as(patch_feats)
#         return F.mse_loss(patch_feats, text_cls)

    
#     def encode_image(self, image, images_2d, mask_ratio=None, text_features=None, image_path=None):
#         """
#         兼容两种 vision_encoder 返回形式：
#           - 旧形式: returns (image_feats, hidden_states)
#           - 新形式: returns dict 包含 keys: 'cls_feats' (B,D), 'patch_feats' (B,N,D), 'patch_scores', 'hidden_states', ...
#         最终构造 image_feats 的格式保持原来（sequence, 带 cls token 在 index 0 的形状）：(B, N+1, D)
#         """
#         # 调用 vision encoder（可能返回 tuple 或 dict）
#         res = self.vision_encoder(image, images_2d, mask_ratio, self.visual_encoder_2D, text_features, image_path)

#         if isinstance(res, dict):
#             # 新版：从 dict 中重建 sequence 形式的 image_feats (cls + patches)
#             # 期望 res 至少包含 'cls_feats' (B,D) 与 'patch_feats' (B,N,D)
#             if 'cls_feats' in res and 'patch_feats' in res:
#                 cls_feats = res['cls_feats']            # (B, D)
#                 patch_feats = res['patch_feats']        # (B, N, D)
#                 # 重建 sequence：cls + patches -> (B, 1+N, D)
#                 image_feats = torch.cat([cls_feats.unsqueeze(1), patch_feats], dim=1)
#                 hidden_states = res.get('hidden_states', None)
#             else:
#                 # 如果 dict 不符合预期，尽量从 hidden_states 中恢复（兼容性回退）
#                 hidden_states = res.get('hidden_states', None)
#                 if hidden_states is not None and len(hidden_states) > 0:
#                     # 取最后一层 hidden_states[-1] 作为 (B, L, D)
#                     image_feats = hidden_states[-1]
#                 else:
#                     # 万一都没有，构建一个零张量（保持不崩溃）
#                     B = image.size(0)
#                     C = self.mm_vision_proj.out_features if hasattr(self, "mm_vision_proj") else image.device.type
#                     image_feats = torch.zeros(B, 1, self.config.hidden_size, device=image.device)
#                     hidden_states = None
#         else:
#             # 旧版行为：tuple (image_feats, hidden_states)
#             try:
#                 image_feats, hidden_states = res
#             except Exception:
#                 # 若返回非预期 tuple 也尝试按原样使用
#                 image_feats = res
#                 hidden_states = None

#         # 继续原来的后处理：投影 + L2 归一化
#         image_feats = self.mm_vision_proj(image_feats) 
#         image_feats = F.normalize(image_feats, dim=-1)

#         # 返回 image_feats（保持原 encode_image 的单值返回）
#         return image_feats


#     def forward(self, images, input_ids, attention_mask, labels, images_2d, global_step=None, epoch=None, **kwargs):
#         mask_ratio = None
#         image_path = None
    
#         # Semantic Matching Loss
#         with torch.inference_mode():
#             text_features_stage1, text_features_language_stage1 = self.stage1_pretrained_CLIP.encode_text(input_ids, attention_mask)
#             text_features_stage1 = text_features_stage1[:, 0]
#             image_features_stage1 = self.stage1_pretrained_CLIP.encode_image(images, None, text_features_language_stage1)[:, 0]
#         loss_CL_stage1, logits_per_image_stage1, logits_per_text_stage1 = self.image_text_contrastive_learning(image_features_stage1, text_features_stage1, labels)

#         text_features, text_feats_language = self.encode_text(input_ids, attention_mask) 
#         image_features = self.encode_image(images, images_2d, mask_ratio, text_features_language_stage1, image_path) 

#         text_features = text_features[:, 0]
#         image_features = image_features[:, 0]

#         # CL Loss for image-text matching
#         loss_CL_stage2, logits_per_image_stage2, logits_per_text_stage2 = self.image_text_contrastive_learning(image_features, text_features, labels)

#         loss_relation = self.image_text_relation_regulation(logits_per_image_stage1.detach(), logits_per_text_stage1.detach(), logits_per_image_stage2, logits_per_text_stage2)

#         relation_loss_weight = None
#         max_weighted_step = 5000
#         try:
#             if global_step < max_weighted_step:
#                 relation_loss_weight = 0.1 * (1 - global_step / max_weighted_step)
#             else:
#                 relation_loss_weight = 0.0
#         except:
#             relation_loss_weight = 0.0

#         # loss = loss_CL_stage2 + relation_loss_weight*loss_relation

#         # esh do
#         vision_out = self.vision_encoder(images, images_2d, mask_ratio,
#                                         self.visual_encoder_2D, text_features_language_stage1)

#         cls_feats = vision_out["cls_feats"]
#         patch_feats = vision_out["patch_feats"]
#         patch_scores = vision_out["patch_scores"]
#         att_map = vision_out["att_map"]
#         patch_kd = vision_out["patch_kd_feats"]
#         pre_sgat = vision_out["x_pre_sgat"]
#         post_sgat = vision_out["x_post_sgat"]

#         # -------- A1/A2/A3/B1 losses --------
#         # loss_a1 = self.loss_A1(cls_feats, text_features_language_stage1)
#         # # loss_a2 = self.loss_A2(patch_feats, text_features_language_stage1, patch_scores, att_map)
#         loss_a3 = self.loss_A3(post_sgat, self.vision_encoder.sgat.neighbor_idx)
#         # loss_b1 = self.loss_B1(patch_feats, patch_kd)

#         loss = (
#             loss_CL_stage2
#             # + 1.0 * loss_a1
#             # + 1.0 * loss_a2
#             + 0.2 * loss_a3
#             # + 0.5 * loss_b1
#             + relation_loss_weight * loss_relation
#         )

        

#         print_interval = 100  # 1
#         try:
#             if global_step % print_interval == 0 and global_step != 0:
#                 print()
#                 if global_step < max_weighted_step: 
#                     print(f"Step {global_step}, loss: {loss}, loss_CL_stage1: {loss_CL_stage1}, loss_CL_stage2: {loss_CL_stage2}, loss_relation: {loss_relation}, relation_loss_weight: {relation_loss_weight}")
#                 else: 
#                     print(f"Step {global_step}, loss: {loss}, loss_CL_stage1: {loss_CL_stage1}, loss_CL_stage2: {loss_CL_stage2}, loss_relation: {loss_relation}")
#         except:
#             pass
#         ret = {
#             "loss": loss,
#             "logits": (logits_per_image_stage2 + logits_per_text_stage2) / 2.0,
#         }

#         return ret

#     def image_text_contrastive_learning(self, image_features, text_features, labels):
#         if self.gather_loss:
#             all_image_features, all_text_features = gather_features(image_features, text_features)
#             if self.local_loss:
#                 logits_per_image = self.logit_scale * image_features @ all_text_features.T
#                 logits_per_text = self.logit_scale * text_features @ all_image_features.T
#             else:
#                 logits_per_image = self.logit_scale * all_image_features @ all_text_features.T
#                 logits_per_text = logits_per_image.T
#         else:
#             logits_per_image = self.logit_scale * image_features @ text_features.T
#             logits_per_text = self.logit_scale * text_features @ image_features.T  
#         loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2

#         return loss, logits_per_image, logits_per_text

#     def image_text_relation_regulation(self, logits_per_image, logits_per_text, logits_per_image_masked, logits_per_text_masked):
#         loss_per_image = F.mse_loss(logits_per_image, logits_per_image_masked)
#         loss_per_text = F.mse_loss(logits_per_text, logits_per_text_masked)
#         return (loss_per_image + loss_per_text) / 2




# AutoConfig.register("m3d_clip_stage2", M3DCLIPConfig_stage2)
# AutoModel.register(M3DCLIPConfig_stage2, M3DCLIP_stage2)



# import numpy as np
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from transformers import PreTrainedModel, PretrainedConfig, BertModel, AutoConfig, AutoModel
# from LaMed.src.model.multimodal_encoder.vit import ViT_stage2
# from LaMed.src.utils.dist_utils import gather_features
# import os
# import json

# loss_logs = []

# class M3DCLIPConfig_stage2(PretrainedConfig):
#     model_type = "m3d_clip_stage2"

#     def __init__(
#         self,
#         language_model_name_or_path: str = "",
#         stage1_pretrained_CLIP_path = "",
#         local_loss: bool = False,
#         gather_loss: bool = True,
#         in_channels: int = 1,
#         img_size: tuple = (32, 256, 256),
#         patch_size: tuple = (4, 16, 16),
#         hidden_size: int = 768,
#         mlp_dim: int = 3072,
#         num_layers: int = 12,
#         num_heads: int = 12,
#         pos_embed: str = "perceptron",
#         dropout_rate: float = 0,
#         spatial_dims: int = 3,
#         max_text_len: int = 128,
#         vocab_size: int = 30522,
#         use_2D_Encoder: bool = True, # 用于控制是否使用2D特征分支
#         **kwargs,
#     ):
#         self.language_model_name_or_path = language_model_name_or_path
#         self.in_channels = in_channels
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.hidden_size = hidden_size
#         self.mlp_dim = mlp_dim
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.pos_embed = pos_embed
#         self.dropout_rate = dropout_rate
#         self.spatial_dims = spatial_dims
#         self.local_loss = local_loss
#         self.gather_loss = gather_loss
#         self.max_text_len = max_text_len
#         self.vocab_size = vocab_size
#         self.use_2D_Encoder = use_2D_Encoder
#         super().__init__(**kwargs)

# class M3DCLIP_stage2(PreTrainedModel):
#     config_class = M3DCLIPConfig_stage2

#     def __init__(self, config):
#         super().__init__(config)

#         # 1. Student Vision Encoder (HSENet Stage 2)
#         self.vision_encoder = ViT_stage2(
#             in_channels=config.in_channels,
#             img_size=config.img_size,
#             patch_size=config.patch_size,
#             hidden_size=config.hidden_size,
#             mlp_dim=config.mlp_dim,
#             num_layers=config.num_layers,
#             num_heads=config.num_heads,
#             dropout_rate=config.dropout_rate,
#             spatial_dims=config.spatial_dims,
#         )

#         # 2. Text Encoder
#         self.language_encoder = BertModel.from_pretrained(config.language_model_name_or_path)

#         # 3. Projection Heads
#         self.mm_vision_proj = nn.Linear(config.hidden_size, config.hidden_size) 
#         self.mm_language_proj = nn.Linear(config.hidden_size, config.hidden_size)


#         self.stage1_pretrained_CLIP = AutoModel.from_pretrained(config.pretrained_model, trust_remote_code=True)
#         self.stage1_pretrained_CLIP.requires_grad_(False)

#         self.teacher_adapter = nn.Identity()
#         if config.use_2D_Encoder:
#             # 输入特征 768 -> 映射到 Hidden Size 768
#             # 即使维度相同，加一个 Linear 层作为 buffer 也是好的，有助于适应特征分布
#             self.teacher_adapter = nn.Linear(768, config.hidden_size)

#         self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
#         self.local_loss = config.local_loss
#         self.gather_loss = config.gather_loss

#     def encode_text(self, input_id, attention_mask):
#         text_feats_language = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"] 
#         text_feats = self.mm_language_proj(text_feats_language) 
#         text_feats = F.normalize(text_feats, dim=-1) 
#         return text_feats, text_feats_language
    
#     # --- Loss A3: Topology Consistency (Laplacian Smoothing) ---
#     def loss_A3(self, post, neighbor_idx):
#         """
#         计算 3D 特征的空间平滑度。
#         """
#         B, N, C = post.shape
#         # neighbor_idx: [N, K] -> expanded [B, N, K]
#         expanded_idx = neighbor_idx.unsqueeze(0).expand(B, -1, -1)
#         batch_indices = torch.arange(B, device=post.device).view(B, 1, 1).expand(B, N, neighbor_idx.shape[1])
        
#         # Gather
#         neighbor_feats = post[batch_indices, expanded_idx] # [B, N, K, C]
#         neighbor_mean = neighbor_feats.mean(dim=2) # [B, N, C]
        
#         return F.mse_loss(post, neighbor_mean)

#     # --- Loss B1: Knowledge Distillation ---
#     def loss_B1(self, student_3d_feats, teacher_2d_feats, grid_shape):
#         """
#         student_3d_feats: [B, N, 768]
#         teacher_2d_feats: [B, S, 768]
#         """
#         B, N, C = student_3d_feats.shape
#         S = teacher_2d_feats.shape[1]
#         _, _, D = grid_shape
        
#         # 将 2D Slice 特征扩展到 3D 空间
#         patch_idx = torch.arange(N, device=student_3d_feats.device)
#         d_indices = patch_idx % D
#         slice_indices = (d_indices.float() / (D - 1) * (S - 1)).round().long().clamp(0, S-1)
        
#         teacher_expanded = []
#         for b in range(B):
#             teacher_expanded.append(teacher_2d_feats[b, slice_indices])
#         teacher_target = torch.stack(teacher_expanded, dim=0)
        
#         return F.mse_loss(student_3d_feats, teacher_target)

#     def image_text_contrastive_learning(self, image_features, text_features, labels):
#         if self.gather_loss:
#             all_image_features, all_text_features = gather_features(image_features, text_features)
#             if self.local_loss:
#                 logits_per_image = self.logit_scale * image_features @ all_text_features.T
#                 logits_per_text = self.logit_scale * text_features @ all_image_features.T
#             else:
#                 logits_per_image = self.logit_scale * all_image_features @ all_text_features.T
#                 logits_per_text = logits_per_image.T
#         else:
#             logits_per_image = self.logit_scale * image_features @ text_features.T
#             logits_per_text = self.logit_scale * text_features @ image_features.T  
#         loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
#         return loss, logits_per_image, logits_per_text

#     def image_text_relation_regulation(self, logits_per_image, logits_per_text, logits_per_image_masked, logits_per_text_masked):
#         loss_per_image = F.mse_loss(logits_per_image, logits_per_image_masked)
#         loss_per_text = F.mse_loss(logits_per_text, logits_per_text_masked)
#         return (loss_per_image + loss_per_text) / 2

#     def forward(self, images, input_ids, attention_mask, labels, image_2d_kd=None, image_2d_sga=None, global_step=None, **kwargs):
#         """
#         Args:
#             images: [B, C, D, H, W]
#             image_2d_kd: [B, S_kd, 768] (Dataset直接提供的特征)
#             image_2d_sga: [B, S_sga, 768] (Dataset直接提供的特征)
#         """
        
#         # --- 1. 计算 Stage 1 Loss (Relation Regulation) ---
#         # 严格参照你提供的原始代码逻辑
#         loss_relation = torch.tensor(0.0, device=images.device)
#         logits_per_image_stage1 = None
#         logits_per_text_stage1 = None
        

#         with torch.inference_mode():
#             # 计算 Stage 1 文本特征
#             text_features_stage1, text_features_language_stage1 = self.stage1_pretrained_CLIP.encode_text(input_ids, attention_mask)
#             text_features_stage1 = text_features_stage1[:, 0]
            
#             # 计算 Stage 1 图像特征
#             # 注意：这里我们使用 text_features_language_stage1 作为条件，参考了原代码
#             image_features_stage1 = self.stage1_pretrained_CLIP.encode_image(images, None, text_features_language_stage1)[:, 0]
        
#         # 计算 Stage 1 的 CL Loss 和 Logits (作为 Target)
#         loss_CL_stage1, logits_per_image_stage1, logits_per_text_stage1 = self.image_text_contrastive_learning(
#             image_features_stage1, text_features_stage1, labels
#         )

#         # --- 2. 适配 2D 特征 (Teacher Adapter) ---
#         feat_kd = None
#         feat_sga = None
#         if image_2d_kd is not None:
#             feat_kd = self.teacher_adapter(image_2d_kd) 
#         if image_2d_sga is not None:
#             feat_sga = self.teacher_adapter(image_2d_sga)

#         # --- 3. 运行 HSENet Stage 2 (Student) ---
#         # 使用 SGA 用的特征进行融合
#         vision_out = self.vision_encoder(images, slice_features=feat_sga)

#         cls_feats = vision_out["cls_feats"]
#         f_3d_raw = vision_out["f_3d_raw"]
        
#         # --- 4. 文本编码 ---
#         # 使用当前 Stage 2 的 text encoder
#         text_features, _ = self.encode_text(input_ids, attention_mask)
#         image_features_cl = F.normalize(self.mm_vision_proj(cls_feats), dim=-1)
#         text_features_cl = text_features[:, 0]

#         # --- 5. 计算损失 ---

#         # (A) CL Loss (全局对齐)
#         loss_CL_stage2, logits_per_image_stage2, logits_per_text_stage2 = self.image_text_contrastive_learning(
#             image_features_cl, text_features_cl, labels
#         )

#         # (B) KD Loss (知识蒸馏)
#         loss_KD = torch.tensor(0.0, device=images.device)
#         if feat_kd is not None:
#             student_kd_proj = self.vision_encoder.kd_proj(f_3d_raw)
#             loss_KD = self.loss_B1(student_kd_proj, feat_kd, self.vision_encoder.sgat.grid_shape)

#         # (C) Topology Loss (结构平滑)
#         neighbor_idx = self.vision_encoder.sgat.neighbor_idx
#         loss_Topo = self.loss_A3(f_3d_raw, neighbor_idx)

#         # (D) RR Loss (关系正则)
#         if logits_per_image_stage1 is not None:
#             loss_relation = self.image_text_relation_regulation(
#                 logits_per_image_stage1.detach(), logits_per_text_stage1.detach(),
#                 logits_per_image_stage2, logits_per_text_stage2
#             )

#         # 权重控制
#         relation_weight = 0.0
#         max_weighted_step = 5000
#         if global_step is not None and global_step < max_weighted_step:
#             relation_weight = 0.1 * (1 - global_step / max_weighted_step)

#         # 总损失
#         total_loss = loss_CL_stage2 + 0.5 * loss_KD + 0.2 * loss_Topo + relation_weight * loss_relation

#         # Logging
#         if global_step is not None and global_step % 100 == 0:
#              print(f"Step {global_step} | Total: {total_loss:.4f} | CL: {loss_CL_stage2:.4f} | KD: {loss_KD:.4f} | RR: {loss_relation:.4f}")

#         return {
#             "loss": total_loss,
#             "logits": logits_per_image_stage2
#         }

# AutoConfig.register("m3d_clip_stage2", M3DCLIPConfig_stage2)
# AutoModel.register(M3DCLIPConfig_stage2, M3DCLIP_stage2)

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, BertModel, AutoConfig, AutoModel
# 确保引用路径正确
from LaMed.src.model.multimodal_encoder.vit import ViT_stage2
from LaMed.src.utils.dist_utils import gather_features
import os
import json

loss_logs = []

class M3DCLIPConfig_stage2(PretrainedConfig):
    model_type = "m3d_clip_stage2"

    def __init__(
        self,
        language_model_name_or_path: str = "",
        stage1_pretrained_CLIP_path = "",
        local_loss: bool = False,
        gather_loss: bool = True,
        in_channels: int = 1,
        img_size: tuple = (32, 256, 256),
        patch_size: tuple = (4, 16, 16),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        dropout_rate: float = 0,
        spatial_dims: int = 3,
        max_text_len: int = 128,
        vocab_size: int = 30522,
        use_2D_Encoder: bool = True,
        **kwargs,
    ):
        self.language_model_name_or_path = language_model_name_or_path
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pos_embed = pos_embed
        self.dropout_rate = dropout_rate
        self.spatial_dims = spatial_dims
        self.local_loss = local_loss
        self.gather_loss = gather_loss
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        self.use_2D_Encoder = use_2D_Encoder
        super().__init__(**kwargs)

class M3DCLIP_stage2(PreTrainedModel):
    config_class = M3DCLIPConfig_stage2

    def __init__(self, config):
        super().__init__(config)

        # 1. Student Vision Encoder (HSENet Stage 2)
        self.vision_encoder = ViT_stage2(
            in_channels=config.in_channels,
            img_size=config.img_size,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            mlp_dim=config.mlp_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout_rate=config.dropout_rate,
            spatial_dims=config.spatial_dims,
        )

        # 2. Text Encoder
        self.language_encoder = BertModel.from_pretrained(config.language_model_name_or_path)

        # 3. Projection Heads
        self.mm_vision_proj = nn.Linear(config.hidden_size, config.hidden_size) 
        self.mm_language_proj = nn.Linear(config.hidden_size, config.hidden_size)

        # 4. Stage1 Pretrained CLIP (用于 Relation Regulation Loss)
        if hasattr(config, 'pretrained_model') and config.pretrained_model:
             try:
                 self.stage1_pretrained_CLIP = AutoModel.from_pretrained(config.pretrained_model, trust_remote_code=True)
                 self.stage1_pretrained_CLIP.requires_grad_(False)
             except Exception:
                 print("Warning: Could not load Stage 1 model. RR Loss will be disabled.")
                 self.stage1_pretrained_CLIP = None

        # 5. Teacher Adapter (用于适配预提取的 768 维 2D 特征)
        # 我们不再加载庞大的 BiomedCLIP 模型，而是直接使用一个线性层来微调特征
        self.teacher_adapter = nn.Identity() 
        if config.use_2D_Encoder:
            # 如果预提取特征与 Student 维度完全一致且不需要微调，可以用 Identity
            # 但通常加一个 Linear 层能更好地将 2D 语义映射到 3D 上下文
            self.teacher_adapter = nn.Linear(768, config.hidden_size)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.local_loss = config.local_loss
        self.gather_loss = config.gather_loss

    def encode_text(self, input_id, attention_mask):
        text_feats_language = self.language_encoder(input_id, attention_mask=attention_mask)["last_hidden_state"] 
        text_feats = self.mm_language_proj(text_feats_language) 
        text_feats = F.normalize(text_feats, dim=-1) 
        return text_feats, text_feats_language
    
    # --- Loss A3: Topology Consistency (Laplacian Smoothing) ---
    def loss_A3(self, post, neighbor_idx):
        """
        计算 3D 特征的空间平滑度。
        post: [B, N, C]
        neighbor_idx: [N, K]
        """
        # post: [B, N, C] -> [B, N, K, C]
        # 利用 neighbor_idx 收集邻居特征
        # 注意: neighbor_idx 需要扩展维度以匹配 Batch
        B, N, C = post.shape
        K = neighbor_idx.shape[1]
        
        # 扩展 idx: [1, N, K] -> [B, N, K]
        expanded_idx = neighbor_idx.unsqueeze(0).expand(B, -1, -1)
        
        # 为了使用 gather, 我们需要将 post 视为扁平化的 [B, N, C]
        # 使用高级索引 (Advanced Indexing)
        # batch_indices: [B, N, K] -> values 0..B-1
        batch_indices = torch.arange(B, device=post.device).view(B, 1, 1).expand(B, N, K)
        
        # Gather neighbors
        neighbor_feats = post[batch_indices, expanded_idx] # [B, N, K, C]
        
        # 计算邻居均值
        neighbor_mean = neighbor_feats.mean(dim=2) # [B, N, C]
        
        # 计算节点与其邻居均值的 MSE
        loss = F.mse_loss(post, neighbor_mean)
        return loss

    # --- Loss B1: Knowledge Distillation ---
    def loss_B1(self, student_3d_feats, teacher_2d_feats, grid_shape):
        """
        student_3d_feats: [B, N, 768] (来自 KD_Proj)
        teacher_2d_feats: [B, S, 768] (来自 Dataset KD 分支)
        grid_shape: (H, W, D)
        """
        B, N, C = student_3d_feats.shape
        S = teacher_2d_feats.shape[1]
        H, W, D = grid_shape
        
        # 这里的逻辑是：将 2D Slice 特征扩展到 3D 空间，作为 Target
        # 1. 计算每个 Patch 对应的 Slice Index
        patch_idx = torch.arange(N, device=student_3d_feats.device)
        # 假设 N 排列为 H->W->D, 则深度索引 d = patch_idx % D

        h_indices = patch_idx // (W * D) 
        slice_indices = (h_indices.float() / (H - 1) * (S - 1)).round().long().clamp(0, S-1)
                
        # 3. 扩展 Teacher 特征 [B, S, C] -> [B, N, C]
        teacher_expanded = []
        for b in range(B):
            teacher_expanded.append(teacher_2d_feats[b, slice_indices])
        teacher_target = torch.stack(teacher_expanded, dim=0)
        
        # 4. 计算 MSE
        student_norm = F.normalize(student_3d_feats, dim=-1)
        teacher_norm = F.normalize(teacher_target, dim=-1)
        return F.mse_loss(student_norm, teacher_norm)

    def image_text_contrastive_learning(self, image_features, text_features, labels):
        if self.gather_loss:
            all_image_features, all_text_features = gather_features(image_features, text_features)
            if self.local_loss:
                logits_per_image = self.logit_scale * image_features @ all_text_features.T
                logits_per_text = self.logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = self.logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = self.logit_scale * image_features @ text_features.T
            logits_per_text = self.logit_scale * text_features @ image_features.T  
        loss = (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2
        return loss, logits_per_image, logits_per_text

    def image_text_relation_regulation(self, logits_per_image, logits_per_text, logits_per_image_masked, logits_per_text_masked):
        loss_per_image = F.mse_loss(logits_per_image, logits_per_image_masked)
        loss_per_text = F.mse_loss(logits_per_text, logits_per_text_masked)
        return (loss_per_image + loss_per_text) / 2

    def forward(self, images, input_ids, attention_mask, labels, image_2d_kd=None, image_2d_sga=None, global_step=None, **kwargs):
        """
        Args:
            images: [B, C, D, H, W]
            image_2d_kd: [B, S_kd, 768] (用于蒸馏)
            image_2d_sga: [B, S_sga, 768] (用于融合)
        """
        
        # --- 1. 适配 2D 特征 (Teacher Adapter) ---
        # 即使输入已经是 768 维，加上一个 Linear 层能更好地适应
        feat_kd = None
        feat_sga = None
        if image_2d_kd is not None:
            feat_kd = self.teacher_adapter(image_2d_kd) # [B, S_kd, 768]
        if image_2d_sga is not None:
            feat_sga = self.teacher_adapter(image_2d_sga) # [B, S_sga, 768]

        # --- 2. 运行 HSENet (Student) ---
        # 这里我们将 SGA 用的特征传入 Vision Encoder，用于 Cross-Attention
        # ViT 返回字典: {'cls_feats', 'f_3d_raw', 'patch_feats', ...}
        vision_out = self.vision_encoder(images, slice_features=feat_sga)

        cls_feats = vision_out["cls_feats"]
        f_3d_raw = vision_out["f_3d_raw"]   # [B, N, 768] (SGAT 纯 3D 输出)
        
        # --- 3. 文本编码 ---
        text_features, _ = self.encode_text(input_ids, attention_mask)
        image_features_cl = F.normalize(self.mm_vision_proj(cls_feats), dim=-1)
        text_features_cl = text_features[:, 0]

        # --- 4. 计算损失 (Multi-Objective) ---

        # (A) CL Loss (全局对齐)
        loss_CL, logits_img, _ = self.image_text_contrastive_learning(image_features_cl, text_features_cl, labels)

        # (B) KD Loss (知识蒸馏) - 关键修改
        # 我们使用 dataset 提供的 image_2d_kd (feat_kd) 和 纯 3D 特征 (f_3d_raw)
        # 需要先将 f_3d_raw 通过 ViT 内部的 KD Projector 投影
        loss_KD = torch.tensor(0.0, device=images.device)
        if feat_kd is not None:
            # 调用 ViT 内部的投影层 (Student Projection)
            student_kd_proj = self.vision_encoder.kd_proj(f_3d_raw)
            # 计算 Loss: Student Proj vs Teacher KD Slices
            loss_KD = self.loss_B1(student_kd_proj, feat_kd, self.vision_encoder.sgat.grid_shape)

        # (C) Topology Loss (结构平滑)
        neighbor_idx = self.vision_encoder.sgat.neighbor_idx
        loss_Topo = self.loss_A3(f_3d_raw, neighbor_idx)

        # (D) RR Loss (关系正则)
        loss_RR = torch.tensor(0.0, device=images.device)
        if hasattr(self, "stage1_pretrained_CLIP") and self.stage1_pretrained_CLIP is not None:
            with torch.no_grad():
                t_s1, _ = self.stage1_pretrained_CLIP.encode_text(input_ids, attention_mask)
                # 假设 Stage1 接口兼容，若不同需调整
                i_s1 = self.stage1_pretrained_CLIP.encode_image(images, None, None)[:, 0]
                logits_s1 = self.logit_scale * i_s1 @ t_s1[:, 0].T
            
            logits_s2 = self.logit_scale * image_features_cl @ text_features_cl.T
            loss_RR = F.mse_loss(logits_s2, logits_s1.detach())

        # --- 5. 总损失 ---
        # 权重可调: CL=1.0, KD=0.5, Topo=0.2, RR=0.1(Decay)
        relation_weight = 0.0
        if global_step is not None and global_step < 5000:
            relation_weight = 0.1 * (1 - global_step / 5000)

        total_loss = loss_CL + 0.5 * loss_KD + 0.2 * loss_Topo + relation_weight * loss_RR

        # Logging (Optional)
        if global_step is not None and global_step % 50 == 0:
             print(f"Step {global_step} | Total: {total_loss:.4f} | CL: {loss_CL:.4f} | KD: {loss_KD:.4f} | Topo: {loss_Topo:.4f}")

        return {
            "loss": total_loss,
            "logits": logits_img
        }

AutoConfig.register("m3d_clip_stage2", M3DCLIPConfig_stage2)
AutoModel.register(M3DCLIPConfig_stage2, M3DCLIP_stage2)
