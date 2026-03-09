
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from __future__ import annotations

# from collections.abc import Sequence

# import torch
# import torch.nn as nn
# import math
# import torch.nn.functional as F
# from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
# from monai.networks.blocks.transformerblock import TransformerBlock
# import open_clip

# def attention(query, key, value, mask=None, dropout=None):
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#     p_attn = F.softmax(scores, dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn


# class regular_attention(nn.Module):

#     def __init__(self, in_channels=16, out_channels=8, emb_dim=768, output_dim=768, dropout=0.1, aropout=0.0):
#         super(regular_attention, self).__init__()
#         self.emb_dim = emb_dim
#         self.Wq = nn.Linear(emb_dim, emb_dim)
#         self.Wk = nn.Linear(emb_dim, emb_dim)
#         self.Wv = nn.Linear(emb_dim, emb_dim)
#         self.attn = None
#         self.output_linear = nn.Linear(emb_dim,emb_dim)
#         self.dropout = nn.Dropout(p=dropout)
#         self.dropout_2 = nn.Dropout(p=dropout)
#         self.norm = nn.LayerNorm(emb_dim)
    
#     def forward(self, Query, Key, Value, context=None, mask=None):
#         '''
#         :param x: [1-4, 32, 768]
#         :param context: [batch_szie, seq_len, emb_dim]
#         :param pad_mask: [batch_size, seq_len, seq_len]
#         :return:
#         ''' 
#         query_list=self.Wq(Query)
#         key_list=self.Wk(Key)
#         value_list=self.Wv(Value)
#         x, self.attn = attention(query_list, key_list, value_list, mask=mask, dropout=self.dropout)
#         x = self.output_linear(x)
#         x = self.norm(query_list + self.dropout_2(x)) 

#         return x, self.attn

# # esh do
# class ThreeDSGAT(nn.Module):

#     def __init__(self,
#                  hidden_size: int,
#                  img_size=(32, 256, 256),
#                  patch_size=(4, 16, 16),
#                  num_heads: int = 8,
#                  num_layers: int = 1,
#                  neighbor_mode: str = "26",    # "6" or "18" or "26" 
#                  distance_gamma: float = 1.0,
#                  dropout: float = 0.1):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.head_dim = hidden_size // num_heads
#         assert self.head_dim * num_heads == hidden_size, "hidden_size not divisible by num_heads"
#         self.num_layers = num_layers
#         self.distance_gamma = float(distance_gamma)
#         self.dropout = nn.Dropout(dropout)

#         # projection layers (shared across layers for simplicity; could make per-layer)
#         self.Wq = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.Wk = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.Wv = nn.Linear(hidden_size, hidden_size, bias=False)
#         self.Wo = nn.Linear(hidden_size, hidden_size, bias=True)

#         self.norm = nn.LayerNorm(hidden_size)

#         # Grid and patch info
#         if isinstance(img_size, int):
#             img_size = (img_size,)
#         if isinstance(patch_size, int):
#             patch_size = (patch_size,)

#         # require 3D shapes
#         assert len(img_size) == 3 and len(patch_size) == 3, "img_size and patch_size must be 3-tuples (D,H,W) or (slice, H, W)"
#         self.img_size = tuple(img_size)
#         self.patch_size = tuple(patch_size)

#         # patch grid (h, w, d) where first dimension corresponds to slice dimension
#         self.grid_shape = (
#             img_size[0] // patch_size[0],
#             img_size[1] // patch_size[1],
#             img_size[2] // patch_size[2],
#         )
#         H, W, D = self.grid_shape
#         N = H * W * D
#         self.num_patches = N

#         # build coordinates [N,3] in (h,w,d) indexing
#         h_coords = torch.arange(H)
#         w_coords = torch.arange(W)
#         d_coords = torch.arange(D)
#         gh, gw, gd = torch.meshgrid(h_coords, w_coords, d_coords, indexing='ij')
#         coords = torch.stack([gh.reshape(-1), gw.reshape(-1), gd.reshape(-1)], dim=-1)  # (N,3)
#         # register coords as buffer (so it moves with model)
#         self.register_buffer("coords", coords.long())  # (N,3)

#         # build relative neighbor offsets according to neighbor_mode
#         rels = []
#         for dz in (-1, 0, 1):
#             for dy in (-1, 0, 1):
#                 for dx in (-1, 0, 1):
#                     if dz == 0 and dy == 0 and dx == 0:
#                         continue
#                     rels.append((dz, dy, dx))
#         # rels length 26
#         rels = torch.tensor(rels, dtype=torch.long)  # (26,3)
#         if neighbor_mode == "6":
#             # 6-neighborhood: +/- axis only
#             rels6 = torch.tensor([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=torch.long)
#             rels = rels6
#         elif neighbor_mode == "18":
#             # remove some diagonals: keep those where sum(abs)<=2
#             rels = rels[(rels.abs().sum(dim=-1) <= 2)]
#         # else keep full 26

#         # compute neighbor idx for each node (N, K). For out-of-bound neighbors we map to self index (so gather still valid).
#         rel = rels  # (K,3)
#         K = rel.shape[0]
#         # coords: (N,3), rel: (K,3) -> coords[:,None,:] + rel[None,:,:] => (N,K,3)
#         nbr_coords = coords.unsqueeze(1) + rel.unsqueeze(0)  # (N, K, 3)
#         # check bounds
#         in_bounds = (
#             (nbr_coords[...,0] >= 0) & (nbr_coords[...,0] < H) &
#             (nbr_coords[...,1] >= 0) & (nbr_coords[...,1] < W) &
#             (nbr_coords[...,2] >= 0) & (nbr_coords[...,2] < D)
#         )  # (N,K)
#         # flatten coordinates to index id = h * (W*D) + w * D + d
#         idx_flat = (nbr_coords[...,0] * (W * D) + nbr_coords[...,1] * D + nbr_coords[...,2])  # (N,K)
#         # for out-of-bounds, set neighbor idx to self index
#         self_idx = torch.arange(N).unsqueeze(1).expand(-1, K)
#         idx_flat_valid = torch.where(in_bounds, idx_flat, self_idx)  # (N,K)
#         self.register_buffer("neighbor_idx", idx_flat_valid.long())   # (N,K)

#         # precompute neighbor distances (Euclidean in grid coords)
#         coord_self = coords.unsqueeze(1).float()  # (N,1,3)
#         nbr_coords_float = nbr_coords.float()     # (N,K,3)
#         distances = torch.norm((coord_self - nbr_coords_float), dim=-1)  # (N,K)
#         self.register_buffer("neighbor_dist", distances.float())  # used for spatial weighting

#         # store K
#         self.K = K

#     def forward(self, x):
#         """
#         x: (B, N, C)
#         returns: (B, N, C) same dtype as input
#         """
#         B, N, C = x.shape
#         assert N == self.num_patches, f"Input N ({N}) != expected num_patches ({self.num_patches})"

#         device = x.device
#         dtype = x.dtype

#         # linear projections
#         q = self.Wq(x)  # (B,N,C)
#         # reshape to multi-head: (B, N, H, D)
#         q = q.view(B, N, self.num_heads, self.head_dim)

#         # gather neighbor features: neighbor_idx: (N,K) -> expand to (B,N,K)
#         neighbor_idx = self.neighbor_idx.to(device=device)  # (N,K)
#         batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, N, self.K)  # (B,N,K)
#         neighbor_idx_exp = neighbor_idx.unsqueeze(0).expand(B, -1, -1)  # (B,N,K)
#         # advanced indexing to get neighbor features: result (B, N, K, C)
#         nbr_feats = x[batch_idx, neighbor_idx_exp]  # (B,N,K,C)

#         # project neighbors: K and V
#         k = self.Wk(nbr_feats)  # (B,N,K,C)
#         v = self.Wv(nbr_feats)  # (B,N,K,C)

#         # reshape k/v to heads: (B,N,K,H,D)
#         k = k.view(B, N, self.K, self.num_heads, self.head_dim)
#         v = v.view(B, N, self.K, self.num_heads, self.head_dim)
#         # bring head dim forward for einsum convenience
#         # k: (B,N,K,H,D), q: (B,N,H,D)
#         # compute attention scores via einsum: (B,N,H,K)
#         # ensure dtype consistent for distance weighting
#         q_ = q  # (B,N,H,D)
#         # compute raw scores
#         attn_scores = torch.einsum("bnhd,bnkhd->bnhk", q_, k)  # (B,N,H,K)
#         # scale
#         attn_scores = attn_scores / math.sqrt(self.head_dim)

#         # apply distance weighting
#         spatial_weight = torch.exp(- (self.distance_gamma * self.neighbor_dist.to(device=device).unsqueeze(0).unsqueeze(2)).to(dtype))  
#         # neighbor_dist: (N,K) -> unsqueeze -> (1,N,1,K) -> expand to broadcast with (B,N,H,K)
#         # convert to same dtype
#         spatial_weight = spatial_weight  # shape (1,N,1,K)
#         attn_scores = attn_scores * spatial_weight  # broadcast

#         # softmax over neighbor dim K
#         attn_weights = F.softmax(attn_scores, dim=-1)  # (B,N,H,K)
#         attn_weights = self.dropout(attn_weights)

#         # aggregate: einsum over K with v -> (B,N,H,D)
#         agg = torch.einsum("bnhk,bnkhd->bnhd", attn_weights, v)  # (B,N,H,D)
#         # reshape back to (B,N,C)
#         agg = agg.contiguous().view(B, N, C)

#         # output projection + residual + norm
#         out = self.Wo(agg)
#         out = self.norm(x + self.dropout(out))

#         return out



# class KDProjection(nn.Module):
#     """
#     B1: 2D→3D KD 映射
#     将每个 3D patch 对应映射到其所属的 slice。
#     """
#     def __init__(self, hidden_size, n_slices):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.n_slices = n_slices

#     def forward(self, patch_feats, slice_feats, grid_shape):
#         B, N, C = patch_feats.shape
#         H, W, D = grid_shape
#         device = patch_feats.device

#         # 每个 patch 的 z-index
#         z_patch = torch.arange(N, device=device) % D
#         z_slice = (z_patch.float() / (D - 1) * (self.n_slices - 1)).round().long()

#         out = []
#         for b in range(B):
#             sf = slice_feats[b]       # [S, C]
#             sf_expand = sf[z_slice]   # [N, C]
#             out.append(sf_expand)
#         return torch.stack(out, dim=0)


# class ViT_stage2(nn.Module):
#     """
#     Vision Transformer (ViT), based on: "Dosovitskiy et al.,
#     An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

#     ViT supports Torchscript but only works for Pytorch after 1.8.
#     """

#     def __init__(
#         self,
#         in_channels: int,
#         img_size: Sequence[int] | int,
#         patch_size: Sequence[int] | int,
#         hidden_size: int = 768,
#         mlp_dim: int = 3072,
#         num_layers: int = 12,
#         num_heads: int = 12,
#         pos_embed: str = "conv",
#         classification: bool = False,
#         num_classes: int = 2,
#         dropout_rate: float = 0.0,
#         spatial_dims: int = 3,
#         post_activation="Tanh",
#         qkv_bias: bool = False,
#         save_attn: bool = False,
#     ) -> None:
#         """
#         Args:
#             in_channels (int): dimension of input channels.
#             img_size (Union[Sequence[int], int]): dimension of input image.
#             patch_size (Union[Sequence[int], int]): dimension of patch size.
#             hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
#             mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
#             num_layers (int, optional): number of transformer blocks. Defaults to 12.
#             num_heads (int, optional): number of attention heads. Defaults to 12.
#             pos_embed (str, optional): position embedding layer type. Defaults to "conv".
#             classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
#             num_classes (int, optional): number of classes if classification is used. Defaults to 2.
#             dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
#             spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
#             post_activation (str, optional): add a final acivation function to the classification head
#                 when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
#                 Set to other values to remove this function.
#             qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
#             save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

#         Examples::

#             # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
#             >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

#             # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
#             >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

#             # for 3-channel with image size of (224,224), 12 layers and classification backbone
#             >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

#         """

#         super().__init__()

#         if not (0 <= dropout_rate <= 1):
#             raise ValueError("dropout_rate should be between 0 and 1.")

#         if hidden_size % num_heads != 0:
#             raise ValueError("hidden_size should be divisible by num_heads.")
#         self.hidden_size = hidden_size
#         self.classification = classification
#         self.patch_embedding = PatchEmbeddingBlock(
#             in_channels=in_channels,
#             img_size=img_size,
#             patch_size=patch_size,
#             hidden_size=hidden_size,
#             num_heads=num_heads,
#             pos_embed=pos_embed,
#             dropout_rate=dropout_rate,
#             spatial_dims=spatial_dims,
#         )  # Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=4, p2=16, p3=16) + Linear(in_features=1024, out_features=768, bias=True)
#         self.blocks = nn.ModuleList(
#             [
#                 TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
#                 for i in range(num_layers)
#             ]
#         )


#         self.patch_score_proj = nn.Linear(hidden_size, 1)
#         self.patch_score_norm = nn.Sigmoid()
#         self.slice_guided_attention = regular_attention()
#         self.norm = nn.LayerNorm(hidden_size)
#         self.sgat = ThreeDSGAT(
#                 hidden_size=hidden_size,
#                 img_size=img_size,       # ensure ViT_stage2 stored self.img_size
#                 patch_size=patch_size,   # ensure ViT_stage2 stored self.patch_size
#                 num_heads=num_heads,
#                 num_layers=1,                 # keep it light by default
#                 neighbor_mode="6",            # "6" or "18" or "26"; choose smaller for speed
#                 distance_gamma=1.0,
#                 dropout=dropout_rate
#             )
#         self.kd_proj = KDProjection(hidden_size=hidden_size, n_slices=32)
#         if self.classification:
#             self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

#     # esh do
#     def forward(self, x, image_2d, k=None, visual_encoder_2D=None, text_features=None, image_path=None):
#         """
#         修正要点：
#         - 更可靠地从 self.img_size / self.patch_size 推断 (H,W,D)
#         - 向 ThreeDSGAT 传入正确 grid_shape
#         - 保证 slice_features 来自 visual_encoder_2D 或由 image_2d 传入
#         - 返回字典（兼容后续损失计算）
#         """
#         batch = x.size(0)

#         # ---- 1) patch embedding ----
#         if x.device.type == "cuda":
#             self.patch_embedding = self.patch_embedding.cuda()
#         x = self.patch_embedding(x.clone())  # (B, N, C)
#         B, N, C = x.shape

#         # ---- 2) infer grid shape robustly ----
#         # N = number of patches
#         B, N, C = x.shape

#         # Helper: try compute grid dims using self.patch_embedding / self.img_size if available,
#         # but try all axis permutations to match N.
#         def infer_grid_shape_from_attrs(patch_embedding):
#             # try to read img_size and patch_size attributes if present
#             img_sz = None
#             p_sz = None
#             try:
#                 img_sz = tuple(patch_embedding.img_size)
#                 p_sz = tuple(patch_embedding.patch_size)
#             except Exception:
#                 # fallback if attributes don't exist
#                 try:
#                     # some implementations store as .image_size / .patch_shape
#                     img_sz = tuple(patch_embedding.image_size)
#                     p_sz = tuple(patch_embedding.patch_shape)
#                 except Exception:
#                     img_sz = None
#                     p_sz = None
#             if img_sz is None or p_sz is None:
#                 return None

#             # ensure length equals spatial dims (3)
#             if len(img_sz) != 3 or len(p_sz) != 3:
#                 return None

#             # compute candidate grid dims = img_sz // p_sz axis-wise
#             cand = [img_sz[i] // p_sz[i] for i in range(3)]
#             return tuple(cand), tuple(img_sz), tuple(p_sz)

#         grid_found = False
#         grid_shape = None

#         inferred = infer_grid_shape_from_attrs(self.patch_embedding)
#         if inferred is not None:
#             cand_grid, img_sz, p_sz = inferred
#             # try all permutations of cand_grid to match N
#             import itertools
#             perms = list(itertools.permutations(cand_grid))
#             for (h, w, d) in perms:
#                 if h * w * d == N:
#                     grid_shape = (int(h), int(w), int(d))
#                     grid_found = True
#                     break

#         if not grid_found:
#             # fallback: try using direct division based on self.img_size/self.patch_size if those exist on self
#             try:
#                 if hasattr(self, "img_size") and hasattr(self, "patch_size"):
#                     img_sz = tuple(self.img_size)
#                     p_sz = tuple(self.patch_size)
#                     if len(img_sz) == 3 and len(p_sz) == 3:
#                         cand = [img_sz[i] // p_sz[i] for i in range(3)]
#                         import itertools
#                         for perm in itertools.permutations(cand):
#                             if perm[0] * perm[1] * perm[2] == N:
#                                 grid_shape = (int(perm[0]), int(perm[1]), int(perm[2]))
#                                 grid_found = True
#                                 break
#             except Exception:
#                 pass

#         if not grid_found:
#             # last resort: try to find integer factors close to cubic root using simple heuristics
#             # compute cube root and try find sensible H,W,D
#             cube = int(round(N ** (1.0 / 3.0)))
#             # try a few candidate triples based on typical medical dims (depth small)
#             candidates = [
#                 (cube, cube, max(1, N // (cube * cube))),
#                 (max(1, N // (cube * cube)), cube, cube),
#                 (8, 16, N // (8 * 16)) if N % (8 * 16) == 0 else (cube, cube, max(1, N // (cube * cube))),
#                 (N // (16 * 16), 16, 16) if N % (16 * 16) == 0 else None
#             ]
#             # filter None and valid
#             candidates = [c for c in candidates if c is not None and c[0] * c[1] * c[2] == N]
#             if len(candidates) > 0:
#                 grid_shape = candidates[0]
#                 grid_found = True

#         if not grid_found:
#             # As a safe fallback choose permutation of inferred cand (if available) whose product closest to N
#             # This avoids raising, but logs a warning.
#             try:
#                 import warnings, itertools
#                 warnings.warn(f"Could not robustly determine grid_shape to match N={N}. Trying best-effort heuristics.")
#                 if inferred is not None:
#                     cand_grid = list(inferred[0])
#                     best = None
#                     best_diff = None
#                     for perm in itertools.permutations(cand_grid):
#                         prod = perm[0] * perm[1] * perm[2]
#                         diff = abs(prod - N)
#                         if best is None or diff < best_diff:
#                             best = perm
#                             best_diff = diff
#                     grid_shape = (int(best[0]), int(best[1]), int(best[2]))
#                 else:
#                     # fallback to cube
#                     cube = int(round(N ** (1.0 / 3.0)))
#                     # choose H=cube, W=cube, D=ceil(N/(cube*cube))
#                     import math
#                     H = cube
#                     W = cube
#                     D = int(math.ceil(N / (H * W)))
#                     grid_shape = (H, W, D)
#                 # warn about potential mismatch
#                 warnings.warn(f"Using best-effort grid_shape={grid_shape} for N={N}. If ordering mismatch persists, please verify patch_embedding.img_size and patch_embedding.patch_size ordering.")
#             except Exception:
#                 # last absolute fallback: cubic root evenly distribute
#                 cube = int(round(N ** (1.0 / 3.0)))
#                 import math
#                 H = cube
#                 W = cube
#                 D = int(math.ceil(N / (H * W)))
#                 grid_shape = (H, W, D)

#         # Ensure integers
#         H, W, D = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])


#         # ---- 3) compute slice features (image_2d expected to be precomputed features or BiomedCLIP outputs) ----
#         # If visual_encoder_2D is provided and image_2d is raw, produce slice features
#         if visual_encoder_2D is not None and image_2d is not None and image_2d.dim() == 5:
#             # image_2d expected shape (B, S, C, H2, W2) or (B, C, S, H2, W2) — try handle common cases
#             # In the original pipeline image_2d is already BioMedCLIP slice features in many places.
#             x_2D = image_2d
#             if x_2D.dim() == 5 and x_2D.size(1) != 3:
#                 # convert (B, S, C, H, W) -> (B*S, C, H, W)
#                 x_2D_proc = x_2D.reshape(-1, *x_2D.shape[-3:])
#                 slice_features = visual_encoder_2D(x_2D_proc)
#                 slice_features = slice_features.view(B, -1, slice_features.shape[-1])
#             elif x_2D.dim() == 4:
#                 # already (B*S, C, H, W) and was preprocessed outside
#                 sf = visual_encoder_2D(x_2D)
#                 slice_features = sf.view(B, -1, sf.shape[-1])
#             else:
#                 # if image_2d already feature tensor (B, S, D)
#                 slice_features = image_2d.view(B, image_2d.size(1), -1)
#         else:
#             # assume image_2d is already slice features (B, S, D)
#             if image_2d is not None:
#                 slice_features = image_2d.view(B, image_2d.size(1), -1)
#             else:
#                 # fallback zero features
#                 slice_features = torch.zeros(B, grid_shape[0], C, device=x.device)

#         # ---- 4) ThreeDSGAT ----
#         x_pre_sgat = x.clone()
#         # x_sgat = self.sgat(x_pre_sgat, grid_shape)  # returns (B, N, C)
#         x_sgat = self.sgat(x)
#         x_post_sgat = x_sgat.clone()

#         # ---- 5) slice-guided attention / patch scoring ----
#         # Prepare semantic features (B, S, D)
#         semantic_features = slice_features.to(x.device)
#         patch_score_raw, att_map = self.slice_guided_attention(x_sgat, semantic_features, semantic_features)
#         # project + normalize
#         patch_score = self.patch_score_proj(patch_score_raw).view(B, N)
#         patch_score = self.patch_score_norm(patch_score)  # sigmoid in [0,1]

#         # ---- 6) KD projection (map slice feats -> patch-level) ----
#         # If kd_projection expects slice count, pass that info
#         try:
#             patch_kd_feats = self.kd_projection(x_sgat, semantic_features, grid_shape)
#         except Exception:
#             # fallback: expand nearest slices to patches
#             S = semantic_features.size(1)
#             # map patch z to slice index (linear mapping)
#             Hg, Wg, Dg = grid_shape
#             z_idx = (torch.arange(N, device=x.device) // (Wg * Dg))  # might not be correct if ordering differs
#             z_idx = (z_idx.float() / (z_idx.max().clamp(min=1).float()) * (S - 1)).round().long().clamp(0, S-1)
#             # build per-patch slice feats
#             batch_kd = []
#             for b in range(B):
#                 batch_kd.append(semantic_features[b, z_idx])
#             patch_kd_feats = torch.stack(batch_kd, dim=0)

#         # ---- 7) fusion (residual) ----
#         lambda_kd = 0.2
#         x_fused = x_sgat + lambda_kd * patch_kd_feats

#         # ---- 8) weighting and transformer blocks ----
#         x_weighted = x_fused * patch_score.unsqueeze(-1)

#         if hasattr(self, "cls_token"):
#             cls_token = self.cls_token.expand(x_weighted.shape[0], -1, -1)
#             x_weighted = torch.cat((cls_token, x_weighted), dim=1)

#         hidden_states_out_weighted = []
#         for blk in self.blocks:
#             x_weighted = blk(x_weighted)
#             hidden_states_out_weighted.append(x_weighted)
#         x_weighted = self.norm(x_weighted)

#         # return dictionary (keep backward compatibility where needed)
#         return {
#             "cls_feats": x_weighted[:, 0],
#             "patch_feats": x_post_sgat,
#             "patch_scores": patch_score,
#             "att_map": att_map,
#             "patch_kd_feats": patch_kd_feats,
#             "x_pre_sgat": x_pre_sgat,
#             "x_post_sgat": x_post_sgat,
#             "hidden_states": hidden_states_out_weighted,
#         }


# vit.py
# vit.py

from __future__ import annotations
from collections.abc import Sequence
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
from monai.networks.blocks.transformerblock import TransformerBlock

# -------------------------------------------------------------------------
# 1. Slice-Guided Attention (SGA) - [特征融合模块]
# -------------------------------------------------------------------------
class SliceGuidedAttention(nn.Module):
    """
    SGA 模块: Cross-Attention
    Query: 3D 结构特征 (Student)
    Key/Value: 2D 切片特征 (Teacher)
    作用: 利用 2D 高清语义修补 3D 特征的细节。
    """
    def __init__(self, emb_dim=768, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        
        # 使用 PyTorch 原生 MultiheadAttention (C++优化，速度快)
        # batch_first=True: input shape [Batch, Seq_len, Feature]
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(dropout)
        
        # FFN: 进一步混合特征
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 4, emb_dim),
            nn.Dropout(dropout)
        )
        self.norm_ffn = nn.LayerNorm(emb_dim)

    def forward(self, x_3d, x_2d):
        """
        Args:
            x_3d (Query): [B, N_patches, C]
            x_2d (Key/Value): [B, N_slices, C]
        """
        # Cross Attention: 3D 查询 2D
        # attn_weights: [B, N_patches, N_slices] (可用于可视化模型关注了哪张切片)
        attn_output, attn_weights = self.attn(query=x_3d, key=x_2d, value=x_2d)
        
        # Residual + Norm
        x = self.norm(x_3d + self.dropout(attn_output))
        
        # FFN + Residual + Norm
        x = self.norm_ffn(x + self.ffn(x))
        
        return x, attn_weights

# -------------------------------------------------------------------------
# 2. 3D Spatial Graph Attention (3D-SGAT) - [空间编码模块]
# -------------------------------------------------------------------------
class ThreeDSGAT(nn.Module):
    """
    3D-SGAT: 基于 Grid Graph 的局部空间注意力。
    强制模型关注 3D 空间邻域，捕捉解剖拓扑结构。
    """
    def __init__(self, hidden_size, img_size, patch_size, num_heads=8, neighbor_mode="6", distance_gamma=1.0, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.distance_gamma = distance_gamma
        self.dropout = nn.Dropout(dropout)

        self.Wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=True)
        self.norm = nn.LayerNorm(hidden_size)

        # 1. 计算 Grid 形状 (H, W, D)
        # 注意: Monai PatchEmbedding 输出的 Patch 顺序通常是 H->W->D
        self.grid_shape = (
            img_size[0] // patch_size[0], # H (Sagittal)
            img_size[1] // patch_size[1], # W (Coronal)
            img_size[2] // patch_size[2]  # D (Axial/Depth)
        )
        H, W, D = self.grid_shape
        N = H * W * D
        self.num_patches = N

        # 2. 预计算坐标网格
        # indexing='ij' -> (H, W, D)
        h_coords = torch.arange(H)
        w_coords = torch.arange(W)
        d_coords = torch.arange(D)
        gh, gw, gd = torch.meshgrid(h_coords, w_coords, d_coords, indexing='ij')
        coords = torch.stack([gh.reshape(-1), gw.reshape(-1), gd.reshape(-1)], dim=-1) # (N, 3)
        self.register_buffer("coords", coords.long())

        # 3. 定义邻居偏移量
        rels = []
        for dh in (-1, 0, 1):
            for dw in (-1, 0, 1):
                for dd in (-1, 0, 1):
                    if dh == 0 and dw == 0 and dd == 0: continue
                    rels.append((dh, dw, dd))
        rels = torch.tensor(rels, dtype=torch.long)
        
        if neighbor_mode == "6":
            # 6邻域 (上下左右前后)
            rels = torch.tensor([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]], dtype=torch.long)
        
        K = rels.shape[0]
        self.K = K
        
        # 4. 计算邻居索引矩阵 [N, K]
        nbr_coords = coords.unsqueeze(1) + rels.unsqueeze(0) # (N, K, 3)
        
        # 边界检查
        in_bounds = (
            (nbr_coords[...,0] >= 0) & (nbr_coords[...,0] < H) &
            (nbr_coords[...,1] >= 0) & (nbr_coords[...,1] < W) &
            (nbr_coords[...,2] >= 0) & (nbr_coords[...,2] < D)
        )
        
        # Flatten index: id = h*(W*D) + w*D + d
        idx_flat = (nbr_coords[...,0] * (W * D) + nbr_coords[...,1] * D + nbr_coords[...,2])
        self_idx = torch.arange(N).unsqueeze(1).expand(-1, K)
        
        # 越界处理：指向自己 (Self-loop)
        idx_flat_valid = torch.where(in_bounds, idx_flat, self_idx)
        self.register_buffer("neighbor_idx", idx_flat_valid.long())

        # 预计算距离权重
        dist = torch.norm((coords.unsqueeze(1).float() - nbr_coords.float()), dim=-1)
        self.register_buffer("neighbor_dist", dist.float())

    def forward(self, x):
        B, N, C = x.shape
        
        # Q Projection
        q = self.Wq(x).view(B, N, self.num_heads, self.head_dim) # [B, N, Heads, Dim]
        
        # Gather Neighbor Features (K, V)
        # neighbor_idx: [N, K] -> [1, N, K] -> [B, N, K]
        neighbor_idx = self.neighbor_idx.unsqueeze(0).expand(B, -1, -1)
        
        # Advanced Indexing Gather: [B, N, C] -> [B, N, K, C]
        # 获取每个 Patch 的 K 个邻居的特征
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, N, self.K)
        nbr_feats = x[batch_idx, neighbor_idx] 
        
        k = self.Wk(nbr_feats).view(B, N, self.K, self.num_heads, self.head_dim)
        v = self.Wv(nbr_feats).view(B, N, self.K, self.num_heads, self.head_dim)
        
        # Attention Score: Center(Q) vs Neighbors(K)
        # q: [B, N, 1, H, D]
        # k: [B, N, K, H, D]
        # Einsum: bnhd, bnkhd -> bnhk (Batch, Node, Head, K_neighbors)
        attn_scores = torch.einsum("bnhd,bnkhd->bnhk", q, k) / math.sqrt(self.head_dim)
        
        # Distance Weighting (距离越近权重越大)
        # spatial_weight = torch.exp(- (self.distance_gamma * self.neighbor_dist.unsqueeze(0).unsqueeze(2))) # [1, N, 1, K]
        # attn_scores = attn_scores * spatial_weight.to(attn_scores.device)
        distance_penalty = self.distance_gamma * self.neighbor_dist.unsqueeze(0).unsqueeze(2)
        attn_scores = attn_scores - distance_penalty.to(attn_scores.device)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Aggregation (Weighted Sum of Neighbors)
        agg = torch.einsum("bnhk,bnkhd->bnhd", attn_weights, v).reshape(B, N, C)
        
        out = self.Wo(agg)
        out = self.norm(x + self.dropout(out)) # Residual connection
        
        return out

# -------------------------------------------------------------------------
# 3. ViT Stage 2 (Main Encoder) - [总控模块]
# -------------------------------------------------------------------------
class ViT_stage2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout_rate: float = 0.0,
        pos_embed: str = "perceptron",
        spatial_dims: int = 3,
        **kwargs
    ) -> None:
        super().__init__()
        
        self.hidden_size = hidden_size
        self.img_size = img_size if isinstance(img_size, Sequence) else (img_size,)*3
        self.patch_size = patch_size if isinstance(patch_size, Sequence) else (patch_size,)*3

        # 1. Monai 基础 Patch Embedding
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=self.img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            pos_emded=pos_embed,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )

        # 2. 核心模块: 3D-SGAT (提取纯 3D 结构特征)
        self.sgat = ThreeDSGAT(
            hidden_size=hidden_size,
            img_size=self.img_size,
            patch_size=self.patch_size,
            num_heads=num_heads,
            dropout=dropout_rate
        )

        # 3. 蒸馏投影层 (Adapter)
        # 将 SGAT 输出映射到 Teacher 的特征空间 (如果需要)
        self.kd_proj = nn.Linear(hidden_size, hidden_size)

        # 4. 特征融合: Slice-Guided Attention
        self.sga = SliceGuidedAttention(
            emb_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout_rate
        )

        # 5. Transformer Blocks (提取深层语义)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def _expand_slices_to_patches(self, slice_feats, num_patches):
        """
        [关键逻辑]: 将 2D Slice 特征扩展到 3D Patch 网格上，用于计算 Dense Distillation Loss。
        逻辑: 计算每个 3D Patch 属于哪个 Z 轴深度，然后赋予它对应 Z 轴的 2D Slice 特征。
        """
        B, S, C = slice_feats.shape
        H, W, D = self.sgat.grid_shape # (H, W, D)
        device = slice_feats.device
        
        # 生成 Patch 对应的 Z 轴索引
        # Monai PatchEmbedding 展平顺序是 H -> W -> D
        # 所以 id = h * (W*D) + w * D + d
        # d = id % D
        patch_indices = torch.arange(num_patches, device=device)
        # d_indices = patch_indices % D # [N]
        
        # # 将深度 d (0 ~ D-1) 映射到 slice 索引 (0 ~ S-1)
        # # 使用线性映射
        # slice_indices = (d_indices.float() / (D - 1) * (S - 1)).round().long()
        # slice_indices = slice_indices.clamp(0, S - 1)

        h_indices = patch_indices // (W * D) 
        slice_indices = (h_indices.float() / (H - 1) * (S - 1)).round().long().clamp(0, S-1)
                
        # Gather: 对每个 Batch，根据索引取出对应的 Slice 特征
        # [B, S, C] -> [B, N, C]
        expanded_feats = []
        for b in range(B):
            expanded_feats.append(slice_feats[b, slice_indices])
        
        return torch.stack(expanded_feats, dim=0)

    def forward(self, x, slice_features=None, **kwargs):
        """
        Args:
            x: 3D Volume [B, C, D, H, W]
            slice_features: 2D Teacher Features [B, S, 768] (来自 Dataset/Teacher)
        """
        B = x.shape[0]
        
        # 1. Patch Embedding -> [B, N, C]
        x = self.patch_embedding(x)
        
        # 2. 3D-SGAT -> [B, N, C]
        # 获得纯粹的 3D 结构特征 (Student)
        f_3d_raw = self.sgat(x)
        
        kd_dict = {}
        f_fused = f_3d_raw
        att_map = None

        if slice_features is not None:
            # --- 解耦路径 A: 知识蒸馏 (Knowledge Distillation) ---
            # 这一步发生在融合之前，强制 f_3d_raw 独立学习
            
            # Student: 投影后的 3D 特征
            f_student_kd = self.kd_proj(f_3d_raw)
            
            # Teacher: 扩展到 3D 网格的 2D 特征
            # 如果 slice_features 来自 KD 专用采样 (image_2d_kd)，则使用它
            # 为了简化接口，这里假设传入的 slice_features 就是用于 KD 的
            f_teacher_kd = self._expand_slices_to_patches(slice_features, f_3d_raw.shape[1])
            
            kd_dict = {
                "student": f_student_kd,
                "teacher": f_teacher_kd.detach() # Teacher 梯度截断
            }

            # --- 解耦路径 B: 特征融合 (Fusion) ---
            # 使用 SGA 融合 2D 信息
            # Query = f_3d_raw, Key/Value = slice_features
            f_fused, att_map = self.sga(f_3d_raw, slice_features)

        # 3. Transformer Layers (Global Reasoning)
        # 添加 CLS Token
        cls_token = self.cls_token.expand(B, -1, -1)
        x_in = torch.cat((cls_token, f_fused), dim=1)
        
        for blk in self.blocks:
            x_in = blk(x_in)
            
        x_out = self.norm(x_in)

        return {
            "cls_feats": x_out[:, 0],       # [B, C] -> 用于 Image-Text CL Loss
            "patch_feats": x_out,         # [B, N, C] -> 融合后的特征 (可选用于后续任务)
            "kd_data": kd_dict,             # 包含 student 和 teacher 特征 -> 用于 KD Loss
            "att_map": att_map,             # 可视化 Attention Map
            "f_3d_raw": f_3d_raw            # SGAT 原始输出 -> 用于 Topology Loss
        }

class ViT(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )  # Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=4, p2=16, p3=16) + Linear(in_features=1024, out_features=768, bias=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )


        self.patch_score_proj = nn.Linear(hidden_size, 1)
        self.patch_score_norm = nn.Sigmoid()
        self.slice_guided_attention = regular_attention()
        self.norm = nn.LayerNorm(hidden_size)
        self.norm_masked = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x, k=None, visual_encoder_2D=None, text_features=None, image_path=None): 
        batch, slice_num = x.size(0), x.size(2)
        hidden_states_out_masked = None
        x_masked = None

        abaltion_score_method = "CrossAttention" 
        abaltion_score_feature = "2DSlice"

        if visual_encoder_2D is not None and not abaltion_score_method == "Random": 
            x_2D = F.interpolate(x.clone(), size=(32, 224, 224), mode='trilinear', align_corners=False) 
            x_2D = x_2D.expand(-1, 3, -1, -1, -1).permute(0, 2, 1, 3, 4)
            x_2D = x_2D.reshape(-1, *x_2D.shape[-3:])
            slice_features = visual_encoder_2D(x_2D)

        if x.device.type == "cuda":
            self.patch_embedding = self.patch_embedding.cuda()
        x = self.patch_embedding(x.clone())
        if k != None:
            batch_size, num_patches, slice_num, dim = x.size(0), x.size(1), 32, x.size(2)
            unmasked_number = int(num_patches * (1-k))
            if abaltion_score_method == "CrossAttention":
                if visual_encoder_2D is not None and abaltion_score_feature == "2DSlice":
                    semantic_features = slice_features.view(batch, slice_num, -1)

            if abaltion_score_method == "CrossAttention":
                patch_score = self.slice_guided_attention(x, semantic_features, semantic_features)
                patch_score = self.patch_score_proj(patch_score)
                scores_flat = self.patch_score_norm(patch_score.view(batch_size, num_patches))

            image_feats_spacial = x.clone()
            image_feats_spacial = image_feats_spacial * scores_flat.unsqueeze(-1)
            topk_scores, topk_indices = torch.topk(scores_flat, k=unmasked_number, dim=1, largest=True, sorted=False)
            sorted_topk_indices, _ = torch.sort(topk_indices, dim=1)
            topk_features = torch.gather(image_feats_spacial.clone(), dim=1, index=sorted_topk_indices.unsqueeze(-1).expand(-1, -1, 768)) 
            x_masked = topk_features

            if hasattr(self, "cls_token"):
                cls_token = self.cls_token.expand(x_masked.shape[0], -1, -1)
                x_masked = torch.cat((cls_token, x_masked), dim=1)

            hidden_states_out_masked = []
            for blk in self.blocks:  # 12 x TransformerBlock
                x_masked = blk(x_masked)
                hidden_states_out_masked.append(x_masked)
            x_masked = self.norm_masked(x_masked)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:  # 12 x TransformerBlock
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        if k != None:
            return x, hidden_states_out, x_masked, hidden_states_out_masked
        else:
            return x, hidden_states_out


class ViT_stage1(nn.Module):
    """
    Vision Transformer (ViT), based on: "Dosovitskiy et al.,
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>"

    ViT supports Torchscript but only works for Pytorch after 1.8.
    """

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        Args:
            in_channels (int): dimension of input channels.
            img_size (Union[Sequence[int], int]): dimension of input image.
            patch_size (Union[Sequence[int], int]): dimension of patch size.
            hidden_size (int, optional): dimension of hidden layer. Defaults to 768.
            mlp_dim (int, optional): dimension of feedforward layer. Defaults to 3072.
            num_layers (int, optional): number of transformer blocks. Defaults to 12.
            num_heads (int, optional): number of attention heads. Defaults to 12.
            pos_embed (str, optional): position embedding layer type. Defaults to "conv".
            classification (bool, optional): bool argument to determine if classification is used. Defaults to False.
            num_classes (int, optional): number of classes if classification is used. Defaults to 2.
            dropout_rate (float, optional): faction of the input units to drop. Defaults to 0.0.
            spatial_dims (int, optional): number of spatial dimensions. Defaults to 3.
            post_activation (str, optional): add a final acivation function to the classification head
                when `classification` is True. Default to "Tanh" for `nn.Tanh()`.
                Set to other values to remove this function.
            qkv_bias (bool, optional): apply bias to the qkv linear layer in self attention block. Defaults to False.
            save_attn (bool, optional): to make accessible the attention in self attention block. Defaults to False.

        Examples::

            # for single channel input with image size of (96,96,96), conv position embedding and segmentation backbone
            >>> net = ViT(in_channels=1, img_size=(96,96,96), pos_embed='conv')

            # for 3-channel with image size of (128,128,128), 24 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(128,128,128), pos_embed='conv', classification=True)

            # for 3-channel with image size of (224,224), 12 layers and classification backbone
            >>> net = ViT(in_channels=3, img_size=(224,224), pos_embed='conv', classification=True, spatial_dims=2)

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )  # Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=4, p2=16, p3=16) + Linear(in_features=1024, out_features=768, bias=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x, k=None, visual_encoder_2D=None, text_features=None, image_path=None):
        batch, slice_num = x.size(0), x.size(2)
        hidden_states_out_masked = None

        if x.device.type == "cuda":
            self.patch_embedding = self.patch_embedding.cuda()
        x = self.patch_embedding(x.clone())

        batch_size, num_patches, dim = x.size(0), x.size(1), x.size(2)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out_masked = []
        for blk in self.blocks:  # 12 x TransformerBlock
            x = blk(x)
            hidden_states_out_masked.append(x)
        x = self.norm(x)

        return x, hidden_states_out_masked

class ViT4LLM_v3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )  # Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=4, p2=16, p3=16) + Linear(in_features=1024, out_features=768, bias=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.patch_score_proj = nn.Linear(hidden_size, 1)
        self.patch_score_norm = nn.Sigmoid()
        self.slice_guided_attention = regular_attention()
        self.norm = nn.LayerNorm(hidden_size)
        self.norm_masked = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.visual_encoder_2D = model.visual.trunk
        self.visual_encoder_2D.requires_grad_(False)

    def forward(self, x):
        batch, slice_num = x.size(0), x.size(2)

        x_2D = F.interpolate(x.clone(), size=(32, 224, 224), mode='trilinear', align_corners=False)
        x_2D = x_2D.expand(-1, 3, -1, -1, -1).permute(0, 2, 1, 3, 4)
        x_2D = x_2D.reshape(-1, *x_2D.shape[-3:])
        slice_features = self.visual_encoder_2D(x_2D)

        if x.device.type == "cuda":
            self.patch_embedding = self.patch_embedding.cuda()
        x = self.patch_embedding(x.clone()) 

        batch_size, num_patches, slice_num, dim = x.size(0), x.size(1), 32, x.size(2)

        semantic_features = slice_features.view(batch, slice_num, -1)

        patch_score = self.slice_guided_attention(x, semantic_features, semantic_features) 

        patch_score = self.patch_score_proj(patch_score) 
        scores_flat = self.patch_score_norm(patch_score.view(batch_size, num_patches))

        image_feats_spacial = x.clone()
        image_feats_spacial = image_feats_spacial * scores_flat.unsqueeze(-1)
        x_masked = image_feats_spacial

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x_masked.shape[0], -1, -1)
            x_masked = torch.cat((cls_token, x_masked), dim=1)

        hidden_states_out_masked = []
        for blk in self.blocks:  # 12 x TransformerBlock
            x_masked = blk(x_masked)
            hidden_states_out_masked.append(x_masked)
        x_masked = self.norm_masked(x_masked)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:  # 12 x TransformerBlock
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        return x, hidden_states_out, x_masked, hidden_states_out_masked 


class ViT4LLM(nn.Module):

    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, x):
        x = self.patch_embedding(x)
        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        hidden_states_out = []
        for blk in self.blocks:
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)
        return x, hidden_states_out


class ViT3DTower(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower = ViT4LLM(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )


    def forward(self, images):
        last_feature, hidden_states = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size

class ViT3DTower_v3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        try:
            remain_2d3d_ViT_type = config.remain_2d3d_ViT_type
            print("Load VisualPacker_3d_phi_v3 seetings.")
            print("remain_2d3d_ViT_type: ", remain_2d3d_ViT_type)
        except:
            remain_2d3d_ViT_type = 'both'
            print("Load VisualPacker_3d_phi_v3 seetings.")
            print("Do not set remain_2d3d_ViT_type, use default: ", remain_2d3d_ViT_type)
        self.remain_2d3d_ViT_type = remain_2d3d_ViT_type

        self.vision_tower = ViT4LLM_v3(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images):
        last_feature, hidden_states, weighted_last_feature, weighted_hidden_states = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature 
            weighted_image_features = weighted_last_feature 
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
            weighted_image_features = weighted_hidden_states[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch': 
            image_features = image_features[:, 1:] 
            weighted_image_features = weighted_image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
            weighted_image_features = weighted_image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        if self.remain_2d3d_ViT_type == "both":
            return image_features, weighted_image_features
        elif self.remain_2d3d_ViT_type == "3dvit":
            return image_features
        elif self.remain_2d3d_ViT_type == "2d3dvit":
            return weighted_image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size



class ViT4LLM_v3_med2e3(nn.Module):
    def __init__(
        self,
        in_channels: int,
        img_size: Sequence[int] | int,
        patch_size: Sequence[int] | int,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "conv",
        classification: bool = False,
        num_classes: int = 2,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        post_activation="Tanh",
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.classification = classification
        self.patch_embedding = PatchEmbeddingBlock(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            pos_embed=pos_embed,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
        )  # Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=4, p2=16, p3=16) + Linear(in_features=1024, out_features=768, bias=True)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_size, mlp_dim, num_heads, dropout_rate, qkv_bias, save_attn)
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_size)
        if self.classification:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
        model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        self.visual_encoder_2D = model.visual.trunk
        self.visual_encoder_2D.requires_grad_(False)

    def forward(self, x):
        batch, slice_num = x.size(0), x.size(2)

        x_2D = F.interpolate(x.clone(), size=(32, 224, 224), mode='trilinear', align_corners=False)
        x_2D = x_2D.expand(-1, 3, -1, -1, -1).permute(0, 2, 1, 3, 4)
        x_2D = x_2D.reshape(-1, *x_2D.shape[-3:])
        slice_features = self.visual_encoder_2D(x_2D)

        if x.device.type == "cuda":
            self.patch_embedding = self.patch_embedding.cuda()
        x = self.patch_embedding(x.clone())

        batch_size, num_patches, dim = x.size(0), x.size(1), x.size(2)
        slice_features = slice_features.view(batch, slice_num, -1)

        if hasattr(self, "cls_token"):
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        hidden_states_out = []
        for blk in self.blocks:  # 12 x TransformerBlock
            x = blk(x)
            hidden_states_out.append(x)
        x = self.norm(x)

        return x, hidden_states_out, slice_features


class ViT3DTower_med2e3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        try:
            remain_2d3d_ViT_type = config.remain_2d3d_ViT_type
            print("Load VisualPacker_3d_phi_v3 seetings.")
            print("remain_2d3d_ViT_type: ", remain_2d3d_ViT_type)
        except:
            remain_2d3d_ViT_type = 'both'
            print("Load VisualPacker_3d_phi_v3 seetings.")
            print("Do not set remain_2d3d_ViT_type, use default: ", remain_2d3d_ViT_type)
        self.remain_2d3d_ViT_type = remain_2d3d_ViT_type

        self.vision_tower = ViT4LLM_v3_med2e3(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images):
        last_feature, hidden_states, slice_last_feature = self.vision_tower(images)
        if self.select_layer == -1:
            image_features = last_feature 
            slice_last_feature = slice_last_feature 
        elif self.select_layer < -1:
            image_features = hidden_states[self.select_feature]
            slice_last_feature = slice_last_feature[self.select_feature]
        else:
            raise ValueError(f'Unexpected select layer: {self.select_layer}')

        if self.select_feature == 'patch':
            image_features = image_features[:, 1:] 
            slice_last_feature = slice_last_feature
        elif self.select_feature == 'cls_patch':
            image_features = image_features
            slice_last_feature = slice_last_feature
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features, slice_last_feature

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def hidden_size(self):
        return self.vision_tower.hidden_size


class ViT3DTower_dual_encoders(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        try:
            remain_2d3d_ViT_type = config.remain_2d3d_ViT_type
            print("Load dual_encoder seetings.")
            print("remain_2d3d_ViT_type: ", remain_2d3d_ViT_type)
        except:
            remain_2d3d_ViT_type = 'dual_vits'
            print("Load dual_encoder seetings.")
            print("Do not set remain_2d3d_ViT_type, use default: ", remain_2d3d_ViT_type)
        self.remain_2d3d_ViT_type = remain_2d3d_ViT_type

        self.vision_tower_stage1 = ViT_stage1(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

        self.vision_tower_stage2 = ViT_stage2(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images, images_2d):

        last_feature_stage1, hidden_states_stage1 = self.vision_tower_stage1(images)
        # last_feature_stage2, hidden_states_stage2 = self.vision_tower_stage2(images, images_2d)

        # esh do

        vision_out = self.vision_tower_stage2(images, slice_features=images_2d)


        last_feature_stage2 = vision_out["cls_feats"]
        hidden_states_stage2 = vision_out["patch_feats"] 



        image_features_stage1 = last_feature_stage1
        image_features_stage2 = last_feature_stage2

        if self.select_feature == 'patch':
            image_features_stage1 = image_features_stage1[:, 1:]
            image_features_stage2 = image_features_stage2[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features_stage1 = image_features_stage1
            image_features_stage2 = image_features_stage2
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        if self.remain_2d3d_ViT_type == "dual_vits":
            return image_features_stage1, image_features_stage2
        elif self.remain_2d3d_ViT_type == "3d_vit":
            return image_features_stage1
        elif self.remain_2d3d_ViT_type == "2e3_vit":
            return image_features_stage2

    @property
    def dtype(self):
        return self.vision_tower_stage1.dtype

    @property
    def device(self):
        return self.vision_tower_stage1.device

    @property
    def hidden_size(self):
        return self.vision_tower_stage1.hidden_size

class ViT3DTower_3dvit_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower_stage1 = ViT_stage1(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images, images_2d):
        
        last_feature_stage1, hidden_states_stage1 = self.vision_tower_stage1(images)

        image_features_stage1 = last_feature_stage1

        if self.select_feature == 'patch':
            image_features_stage1 = image_features_stage1[:, 1:] 
        elif self.select_feature == 'cls_patch':
            image_features_stage1 = image_features_stage1
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features_stage1

    @property
    def dtype(self):
        return self.vision_tower_stage1.dtype

    @property
    def device(self):
        return self.vision_tower_stage1.device

    @property
    def hidden_size(self):
        return self.vision_tower_stage1.hidden_size


class ViT3DTower_2e3vit_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower_stage2 = ViT_stage2(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images, images_2d):

        last_feature_stage2, hidden_states_stage2 = self.vision_tower_stage2(images, images_2d)

        image_features_stage2 = last_feature_stage2

        if self.select_feature == 'patch': 
            image_features_stage2 = image_features_stage2[:, 1:] 
        elif self.select_feature == 'cls_patch':
            image_features_stage2 = image_features_stage2
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features_stage2

    @property
    def dtype(self):
        return self.vision_tower_stage2.dtype

    @property
    def device(self):
        return self.vision_tower_stage2.device

    @property
    def hidden_size(self):
        return self.vision_tower_stage2.hidden_size


class ViT3DTower_reproduce_med2e3_encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.select_layer = config.vision_select_layer
        self.select_feature = config.vision_select_feature

        self.vision_tower_reproduce_med2e3 = ViT_stage1(
            in_channels=self.config.image_channel,
            img_size=self.config.image_size,
            patch_size=self.config.patch_size,
            pos_embed="perceptron",
            spatial_dims=len(self.config.patch_size),
            classification=True,
        )

    def forward(self, images, images_2d):
        
        last_feature, hidden_states = self.vision_tower_reproduce_med2e3(images)

        image_features = last_feature

        if self.select_feature == 'patch': 
            image_features = image_features[:, 1:] 
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')

        return image_features, images_2d 

    @property
    def dtype(self):
        return self.vision_tower_reproduce_med2e3.dtype

    @property
    def device(self):
        return self.vision_tower_reproduce_med2e3.device

    @property
    def hidden_size(self):
        return self.vision_tower_reproduce_med2e3.hidden_size
