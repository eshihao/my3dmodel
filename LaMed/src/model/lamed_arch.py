# from abc import ABC, abstractmethod

# import torch
# import torch.nn as nn

# from .multimodal_encoder.builder import build_vision_tower
# from .multimodal_projector.builder import build_mm_projector
# from .segmentation_module.builder import build_segmentation_module
# from LaMed.src.model.loss import BCELoss, BinaryDiceLoss


# class LamedMetaModel:
#     def __init__(self, config):
#         super(LamedMetaModel, self).__init__(config)

#         self.config = config
#         self.seg_enable = False

#         if hasattr(config, "vision_tower"):
#             self.vision_tower = build_vision_tower(config)
#             self.mm_projector = build_mm_projector(config)

#         if hasattr(config, "segmentation_module") and config.segmentation_module is not None:
#             self.seg_enable = True
#             self.seg_module = build_segmentation_module(config)

#             self.seg_projector = nn.Sequential(
#                 nn.Linear(config.hidden_size, config.hidden_size),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(config.hidden_size, config.mm_hidden_size),
#                 nn.Dropout(0.1),
#             )

#             self.dice_loss = BinaryDiceLoss()
#             self.bce_loss = BCELoss()

#     def get_vision_tower(self):
#         vision_tower = getattr(self, 'vision_tower', None)
#         return vision_tower

#     def initialize_vision_modules(self, model_args):
#         self.config.image_channel = model_args.image_channel
#         self.config.image_size = model_args.image_size
#         self.config.patch_size = model_args.patch_size

#         self.config.vision_tower = model_args.vision_tower
#         self.config.vision_select_layer = model_args.vision_select_layer
#         self.config.vision_select_feature = model_args.vision_select_feature

#         self.config.mm_projector_type = model_args.mm_projector_type
#         try:
#             self.config.remain_2d3d_ViT_type = model_args.remain_2d3d_ViT_type
#         except:
#             self.config.remain_2d3d_ViT_type = "both"
#             print("Do not set remain_2d3d_ViT_type, use default: ", self.config.remain_2d3d_ViT_type)
            
        
#         self.config.proj_layer_type = model_args.proj_layer_type
#         self.config.proj_layer_num = model_args.proj_layer_num
#         self.config.proj_pooling_type = model_args.proj_pooling_type
#         self.config.proj_pooling_size = model_args.proj_pooling_size

#         # vision tower
#         if self.get_vision_tower() is None:
#             self.vision_tower = build_vision_tower(self.config)
#             self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)

#         self.config.mm_hidden_size = self.vision_tower.hidden_size

#         # mm_projector
#         if getattr(self, 'mm_projector', None) is None:
#             self.mm_projector = build_mm_projector(self.config)
#             if self.config.vision_tower == 'vit_stage2_dual_encoders':
#                 if model_args.use_parallel_projector == True:
#                     self.mm_projector2 = build_mm_projector(self.config)
#                     print("Using parallel mm_projector and mm_projector2 together, to process visual tokens.")
#                 else:
#                     print("Using shared mm_projector to process two sources of visual features.")

#         if model_args.pretrain_mm_mlp_adapter is not None:
#             mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
#             def get_w(weights, keyword):
#                 return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
#             self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)

#     def initialize_seg_modules(self, model_args):
#         self.config.segmentation_module = model_args.segmentation_module

#         # segmentation_module
#         if getattr(self, 'seg_module', None) is None:
#             self.seg_module = build_segmentation_module(self.config)
#             self.seg_projector = nn.Sequential(
#                 nn.Linear(self.config.hidden_size, self.config.hidden_size),
#                 nn.ReLU(inplace=True),
#                 nn.Linear(self.config.hidden_size, self.config.mm_hidden_size),
#                 nn.Dropout(0.1),
#             )
#             self.seg_enable = True

#         if model_args.pretrain_seg_module is not None:
#             seg_module_weights = torch.load(model_args.pretrain_seg_module, map_location='cpu')
#             new_state_dict = {}
#             for key, value in seg_module_weights.items():
#                 if key.startswith('model.text_encoder.') or key.startswith('text_encoder.'):
#                     continue
#                 if key.startswith('model.'):
#                     new_key = key[len('model.'):]
#                     new_state_dict[new_key] = value
#             self.seg_module.load_state_dict(new_state_dict, strict=True)

#         self.dice_loss = BinaryDiceLoss()
#         self.bce_loss = BCELoss()

# class LamedMetaForCausalLM(ABC):
#     @abstractmethod
#     def get_model(self):
#         pass

#     def get_vision_tower(self):
#         return self.get_model().get_vision_tower()

#     def encode_images(self, images, text_feature=None, images_2d=None):
#         image_features = self.get_model().get_vision_tower()(images, images_2d)
#         if isinstance(image_features, tuple):
#             if image_features[-1].shape[1] == 2048:
#                 image_features_ori, image_features_sec = image_features[0], image_features[1]
#                 image_features_ori = self.get_model().mm_projector(image_features_ori)
#                 try:
#                     image_features_sec = self.get_model().mm_projector2(image_features_sec)
#                 except:
#                     image_features_sec = self.get_model().mm_projector(image_features_sec)
#                 image_features = torch.cat([image_features_ori, image_features_sec], dim=1)

#             elif image_features[-1].shape[1] == 32:
#                 image_features_ori, image_features_sec = image_features[0], image_features[1]
#                 image_features = self.get_model().mm_projector(image_features_ori, image_features_sec, text_feature)

#         else:
#             image_features = self.get_model().mm_projector(image_features)

#         return image_features

#     def prepare_inputs_for_multimodal(
#         self, input_ids, position_ids, attention_mask, past_key_values, labels,
#         images, images_2d,
#     ):
#         vision_tower = self.get_vision_tower()
#         if vision_tower is None or images is None or input_ids.shape[1] == 1:
#             return input_ids, position_ids, attention_mask, past_key_values, None, labels
#         else:
#             inputs_embeds = self.get_model().embed_tokens(input_ids)
#             image_features = self.encode_images(images, inputs_embeds, images_2d)
#             inputs_embeds = torch.cat(
#                 (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]), dim=1)
#         return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

#     def initialize_vision_tokenizer(self, model_args, tokenizer):
#         num_new_tokens = model_args.num_new_tokens

#         self.resize_token_embeddings(len(tokenizer))

#         if num_new_tokens > 0:
#             input_embeddings = self.get_input_embeddings().weight.data
#             output_embeddings = self.get_output_embeddings().weight.data

#             input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
#                 dim=0, keepdim=True)
#             output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
#                 dim=0, keepdim=True)

#             input_embeddings[-num_new_tokens:] = input_embeddings_avg
#             output_embeddings[-num_new_tokens:] = output_embeddings_avg

#             if model_args.tune_mm_mlp_adapter:
#                 for p in self.get_input_embeddings().parameters():
#                     p.requires_grad = True
#                 for p in self.get_output_embeddings().parameters():
#                     p.requires_grad = False
#             else:
#                 # we add 4 new tokens
#                 # if new tokens need input, please train input_embeddings
#                 for p in self.get_input_embeddings().parameters():
#                     p.requires_grad = True
#                 # if new tokens need predict, please train output_embeddings
#                 for p in self.get_output_embeddings().parameters():
#                     p.requires_grad = True

#         if model_args.pretrain_mm_mlp_adapter:
#             mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
#             embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

#             if input_embeddings.shape == embed_tokens_weight.shape:
#                 input_embeddings = embed_tokens_weight
#             elif embed_tokens_weight.shape[0] == num_new_tokens:
#                 input_embeddings[-num_new_tokens:] = embed_tokens_weight
#             else:
#                 raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")



from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_mm_projector
from .segmentation_module.builder import build_segmentation_module
from LaMed.src.model.loss import BCELoss, BinaryDiceLoss


class LamedMetaModel:
    def __init__(self, config):
        super(LamedMetaModel, self).__init__(config)

        self.config = config
        self.seg_enable = False

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_mm_projector(config)

        if hasattr(config, "segmentation_module") and config.segmentation_module is not None:
            self.seg_enable = True
            self.seg_module = build_segmentation_module(config)

            self.seg_projector = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(config.hidden_size, config.mm_hidden_size),
                nn.Dropout(0.1),
            )

            self.dice_loss = BinaryDiceLoss()
            self.bce_loss = BCELoss()

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower

    def initialize_vision_modules(self, model_args):
        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        self.config.patch_size = model_args.patch_size

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        try:
            self.config.remain_2d3d_ViT_type = model_args.remain_2d3d_ViT_type
        except:
            self.config.remain_2d3d_ViT_type = "both"
            print("Do not set remain_2d3d_ViT_type, use default: ", self.config.remain_2d3d_ViT_type)
            
        
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size

        # vision tower
        if self.get_vision_tower() is None:
            self.vision_tower = build_vision_tower(self.config)
            self.vision_tower.requires_grad_(not model_args.freeze_vision_tower)

        self.config.mm_hidden_size = self.vision_tower.hidden_size

        # mm_projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(self.config)
            if self.config.vision_tower == 'vit_stage2_dual_encoders':
                if model_args.use_parallel_projector == True:
                    self.mm_projector2 = build_mm_projector(self.config)
                    print("Using parallel mm_projector and mm_projector2 together, to process visual tokens.")
                else:
                    print("Using shared mm_projector to process two sources of visual features.")

        if model_args.pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            
            # [FIX] 修复权重加载逻辑，防止 IndexError
            def get_w(weights, keyword):
                search_key = keyword + '.'
                return {k.split(search_key)[1]: v for k, v in weights.items() if search_key in k}
                
            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'), strict=True)
            # 如果有 mm_projector2，通常也需要加载，但这取决于你的训练策略
            # 如果 strict=True 报错缺失 keys，可能需要在这里处理 mm_projector2

    def initialize_seg_modules(self, model_args):
        self.config.segmentation_module = model_args.segmentation_module

        # segmentation_module
        if getattr(self, 'seg_module', None) is None:
            self.seg_module = build_segmentation_module(self.config)
            self.seg_projector = nn.Sequential(
                nn.Linear(self.config.hidden_size, self.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.config.hidden_size, self.config.mm_hidden_size),
                nn.Dropout(0.1),
            )
            self.seg_enable = True

        if model_args.pretrain_seg_module is not None:
            seg_module_weights = torch.load(model_args.pretrain_seg_module, map_location='cpu')
            new_state_dict = {}
            for key, value in seg_module_weights.items():
                if key.startswith('model.text_encoder.') or key.startswith('text_encoder.'):
                    continue
                if key.startswith('model.'):
                    new_key = key[len('model.'):]
                    new_state_dict[new_key] = value
            self.seg_module.load_state_dict(new_state_dict, strict=True)

        self.dice_loss = BinaryDiceLoss()
        self.bce_loss = BCELoss()

class LamedMetaForCausalLM(ABC):
    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images, text_feature=None, images_2d=None):
        # 注意：这里调用的是 self.get_model().get_vision_tower()
        # 请确保 lamed_phi3.py 中重写了 encode_images 以适配你的 Vision Tower 签名
        # 如果你已经在 lamed_phi3.py 中重写了，这里的代码实际上不会被调用
        vision_tower = self.get_model().get_vision_tower()
        
        # 尝试适配不同的 forward 参数
        try:
            if images_2d is not None:
                image_features = vision_tower(images, images_2d)
            else:
                image_features = vision_tower(images)
        except TypeError:
             # 如果不支持 images_2d 位置参数，尝试关键字或单参数
             try:
                 image_features = vision_tower(images, slice_features=images_2d)
             except TypeError:
                 image_features = vision_tower(images)

        # 处理返回值 (Tuple/Dict/Tensor)
        if isinstance(image_features, (list, tuple)):
            # 兼容旧逻辑
            if len(image_features) > 1 and image_features[-1].shape[1] == 2048:
                image_features_ori, image_features_sec = image_features[0], image_features[1]
                image_features_ori = self.get_model().mm_projector(image_features_ori)
                try:
                    image_features_sec = self.get_model().mm_projector2(image_features_sec)
                except:
                    image_features_sec = self.get_model().mm_projector(image_features_sec)
                image_features = torch.cat([image_features_ori, image_features_sec], dim=1)
            else:
                # 默认取第一个
                image_features = self.get_model().mm_projector(image_features[0])
        elif isinstance(image_features, dict):
            # 优先取 patch_feats 或 last_hidden_state
            feat = image_features.get('patch_feats', image_features.get('last_hidden_state'))
            if feat is None:
                feat = list(image_features.values())[0]
            image_features = self.get_model().mm_projector(feat)
        else:
            image_features = self.get_model().mm_projector(image_features)

        return image_features

    def prepare_inputs_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, images_2d,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        else:
            inputs_embeds = self.get_model().embed_tokens(input_ids)
            # 调用 encode_images (优先使用 lamed_phi3.py 中的实现)
            image_features = self.encode_images(images, inputs_embeds, images_2d)
            
            # 拼接 Embedding
            inputs_embeds = torch.cat(
                (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]), dim=1)
        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = True

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            
            # 有时权重里没有 embed_tokens，加个 try-catch
            if 'model.embed_tokens.weight' in mm_projector_weights:
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings = embed_tokens_weight
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")