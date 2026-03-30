# from typing import List, Optional, Tuple, Union, Any

# import torch
# import torch.nn as nn

# from transformers import AutoConfig, AutoModelForCausalLM, \
#                          Phi3Config, Phi3Model, Phi3ForCausalLM

# from transformers.modeling_outputs import CausalLMOutputWithPast
# from transformers.generation.utils import GenerateOutput

# from ..lamed_arch import LamedMetaModel, LamedMetaForCausalLM


# class LamedPhi3Config(Phi3Config):
#     model_type = "lamed_phi3"
#     # model_type = "lamed_phi4"


# class LamedPhi3Model(LamedMetaModel, Phi3Model):
#     config_class = LamedPhi3Config
#     def __init__(self, config: Phi3Config):
#         super(LamedPhi3Model, self).__init__(config)


# class LamedPhi3ForCausalLM(LamedMetaForCausalLM, Phi3ForCausalLM):
#     config_class = LamedPhi3Config

#     def __init__(self, config):
#         super(LamedPhi3ForCausalLM, self).__init__(config)
#         self.model = LamedPhi3Model(config)
#         self.vocab_size = config.vocab_size
#         self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

#         # Initialize weights and apply final processing
#         self.post_init()

#     def get_model(self):
#         return self.model

#     def forward(
#             self,
#             images: Optional[torch.FloatTensor] = None,
#             input_ids: torch.LongTensor = None,
#             labels: Optional[torch.LongTensor] = None,
#             attention_mask: Optional[torch.Tensor] = None,
#             images_2d: Optional[torch.Tensor] = None,
#             segs: Optional[torch.FloatTensor] = None,

#             position_ids: Optional[torch.LongTensor] = None,
#             past_key_values: Optional[List[torch.FloatTensor]] = None,
#             inputs_embeds: Optional[torch.FloatTensor] = None,
#             use_cache: Optional[bool] = None,
#             output_attentions: Optional[bool] = None,
#             output_hidden_states: Optional[bool] = None,
#             return_dict: Optional[bool] = None,
#             cache_position: Optional[torch.LongTensor] = None,
#             logits_to_keep: Union[int, torch.Tensor] = 0,

#     ) -> Union[Tuple, CausalLMOutputWithPast]:

#         input_ids_pre = input_ids

#         if inputs_embeds is None:
#             (
#                 input_ids,
#                 position_ids,
#                 attention_mask,
#                 past_key_values,
#                 inputs_embeds,
#                 labels
#             ) = self.prepare_inputs_for_multimodal(
#                 input_ids,
#                 position_ids,
#                 attention_mask,
#                 past_key_values,
#                 labels,
#                 images,
#                 images_2d,
#             )

#         try:
#             seg_ids = torch.nonzero(torch.sum(segs, dim=(1, 2, 3, 4))).flatten().tolist()
#         except:
#             seg_ids = []

#         if self.get_model().seg_enable and seg_ids:
#             outputs = super().forward(
#                                     input_ids=input_ids,
#                                     inputs_embeds=inputs_embeds,
#                                     attention_mask=attention_mask,
#                                     labels=labels,
#                                     output_hidden_states=True,

#                                     position_ids=position_ids,
#                                     past_key_values=past_key_values,
#                                     use_cache=use_cache,
#                                     output_attentions=output_attentions,
#                                     return_dict=return_dict
#                                 )

#             output_hidden_states = outputs.hidden_states

#             last_hidden_state = output_hidden_states[-1]

#             seg_token_mask = input_ids_pre[:, 1:] == self.config.seg_token_id
#             seg_token_mask = torch.cat(
#                 [
#                     seg_token_mask,
#                     torch.zeros((seg_token_mask.shape[0], 1), dtype=seg_token_mask.dtype).cuda(),
#                 ],
#                 dim=1,
#             )

#             seg_prompts = []
#             for i in seg_ids:
#                 if torch.sum(seg_token_mask[i]) == 1:
#                     seg_token = last_hidden_state[i][seg_token_mask[i]]
#                     seg_prompt = self.get_model().seg_projector(seg_token)
#                 elif torch.sum(seg_token_mask[i]) > 1:
#                     seg_tokens = last_hidden_state[i][seg_token_mask[i]]
#                     seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
#                     seg_prompt = self.get_model().seg_projector(seg_token)
#                 else:
#                     seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
#                                              device=last_hidden_state.device)
#                 seg_prompts.append(seg_prompt)

#             seg_prompts = torch.cat(seg_prompts, dim=0)
#             logits = self.get_model().seg_module(images[seg_ids], text_emb=seg_prompts)
#             loss_dice = self.get_model().dice_loss(logits, segs[seg_ids])
#             loss_bce = self.get_model().bce_loss(logits, segs[seg_ids])
#             seg_loss = loss_dice + loss_bce
#             outputs.loss = outputs.loss + seg_loss
#             return outputs
#         else:
#             return super().forward(
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 position_ids=position_ids,
#                 past_key_values=past_key_values,
#                 inputs_embeds=inputs_embeds,
#                 labels=labels,
#                 use_cache=use_cache,
#                 output_attentions=output_attentions,
#                 output_hidden_states=output_hidden_states,
#                 return_dict=return_dict
#             )


#     @torch.no_grad()
#     def generate(
#         self,
#         images: Optional[torch.Tensor] = None,
#         inputs: Optional[torch.Tensor] = None,
#         seg_enable: bool = False,
#         **kwargs,
#     ) -> Union[GenerateOutput, torch.LongTensor, Any]:
#         position_ids = kwargs.pop("position_ids", None)
#         attention_mask = kwargs.pop("attention_mask", None)
#         images_2d = kwargs.pop("image_2d", None)
#         if "inputs_embeds" in kwargs:
#             raise NotImplementedError("`inputs_embeds` is not supported")

#         if images is not None:
#             (
#                 inputs,
#                 position_ids,
#                 attention_mask,
#                 _,
#                 inputs_embeds,
#                 _
#             ) = self.prepare_inputs_for_multimodal(
#                 inputs,
#                 position_ids,
#                 attention_mask,
#                 None,
#                 None,
#                 images,
#                 images_2d,
#             )
#         else:
#             inputs_embeds = self.get_model().embed_tokens(inputs)

#         if seg_enable:
#             outputs = super().generate(
#                 inputs_embeds=inputs_embeds,
#                 output_hidden_states=True,
#                 return_dict_in_generate=True,
#                 **kwargs
#             )

#             output_hidden_states = outputs.hidden_states
#             output_ids = outputs.sequences

#             seg_token_mask = output_ids[:, 1:] == self.config.seg_token_id

#             last_tensors = [tuple[-1] for tuple in output_hidden_states]
#             last_hidden_state = torch.cat(last_tensors[1:], dim=1)

#             seg_prompts = []
#             noseg_ids = []
#             for i in range(len(seg_token_mask)):
#                 if torch.sum(seg_token_mask[i]) == 1:
#                     seg_token = last_hidden_state[i][seg_token_mask[i]]
#                     seg_prompt = self.get_model().seg_projector(seg_token)
#                 elif torch.sum(seg_token_mask[i]) > 1:
#                     seg_tokens = last_hidden_state[i][seg_token_mask[i]]
#                     seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
#                     seg_prompt = self.get_model().seg_projector(seg_token)
#                 else:
#                     noseg_ids.append(i)
#                     seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype,
#                                              device=last_hidden_state.device)
#                 seg_prompts.append(seg_prompt)

#             seg_prompts = torch.cat(seg_prompts, dim=0)
#             logits = self.get_model().seg_module(images, seg_prompts)
#             logits[noseg_ids] = -torch.inf

#             return output_ids, logits
#         else:
#             output_ids = super().generate(
#                 inputs_embeds=inputs_embeds,
#                 attention_mask=attention_mask,
#                 **kwargs
#             )
#             return output_ids


#     def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
#                                       inputs_embeds=None, **kwargs):
#         images = kwargs.pop("images", None)
#         inputs = super().prepare_inputs_for_generation(
#             input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
#         )
#         if images is not None:
#             inputs['images'] = images
#         return inputs


# AutoConfig.register("lamed_phi3", LamedPhi3Config)
# AutoModelForCausalLM.register(LamedPhi3Config, LamedPhi3ForCausalLM)
from typing import List, Optional, Tuple, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, \
                         Phi3Config, Phi3Model, Phi3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from ..lamed_arch import LamedMetaModel, LamedMetaForCausalLM

class LamedPhi3Config(Phi3Config):
    model_type = "lamed_phi3"

class LamedPhi3Model(LamedMetaModel, Phi3Model):
    config_class = LamedPhi3Config
    def __init__(self, config: Phi3Config):
        super(LamedPhi3Model, self).__init__(config)

class LamedPhi3ForCausalLM(LamedMetaForCausalLM, Phi3ForCausalLM):
    config_class = LamedPhi3Config

    def __init__(self, config):
        super(LamedPhi3ForCausalLM, self).__init__(config)
        self.model = LamedPhi3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_model(self):
        return self.model

    # ========================================================================
    # [ULTIMATE SPEED FIX] 强制 BF16 + no_grad
    # ========================================================================
    def encode_images(self, images, inputs_embeds=None, images_2d=None):
        vision_tower = self.get_model().get_vision_tower()
        
        # 1. 强制 Vision Tower 不计算梯度 (提速 50%+)
        # 2. 强制转换输入为 BF16 (3090 核心加速)
        with torch.no_grad():
            dtype = self.model.dtype # 获取 LLM 的精度 (通常是 bf16)
            
            # 确保输入是 bf16
            images = images.to(dtype=dtype)
            if images_2d is not None:
                images_2d = images_2d.to(dtype=dtype)

            # [核心修复]：显式使用关键字参数 images_2d=images_2d 传递给视觉塔
            try:
                if images_2d is not None:
                    vision_out = vision_tower(images, images_2d=images_2d)
                else:
                    vision_out = vision_tower(images)
            except TypeError as e:
                # Fallback logic 兼容极老版本的命名
                try:
                    vision_out = vision_tower(images, slice_features=images_2d)
                except:
                    vision_out = vision_tower(images)
            
            # Extract Feature Tensor
            # dual encoders: keep two branches to avoid silently dropping one path.
            image_features = None
            image_features_sec = None

            if isinstance(vision_out, torch.Tensor):
                image_features = vision_out
            elif isinstance(vision_out, dict):
                for key in ["patch_feats", "x_3d", "last_hidden_state", "image_features"]:
                    if key in vision_out:
                        image_features = vision_out[key]
                        break
                if image_features is None:
                    image_features = list(vision_out.values())[0]
            elif isinstance(vision_out, (tuple, list)):
                image_features = vision_out[0]
                if len(vision_out) > 1:
                    image_features_sec = vision_out[1]

            def _align_patch_num(feat, expected_n=2048):
                if feat is None:
                    return None
                _, n, _ = feat.shape
                if n != expected_n:
                    feat = feat.transpose(1, 2)
                    feat = F.interpolate(feat, size=expected_n, mode='linear', align_corners=False)
                    feat = feat.transpose(1, 2)
                return feat

            image_features = _align_patch_num(image_features, expected_n=2048)
            image_features_sec = _align_patch_num(image_features_sec, expected_n=2048)

            if image_features is None:
                b = images.shape[0]
                image_features = torch.zeros((b, 2048, 768), device=images.device, dtype=dtype)

        # 3. Projector (Requires Grad!)
        image_features = image_features.to(dtype)
        image_features = self.get_model().mm_projector(image_features)

        # If dual branch exists, project it with mm_projector2 (or fallback projector),
        # then fuse with averaging to keep token length unchanged.
        if image_features_sec is not None:
            image_features_sec = image_features_sec.to(dtype)
            if hasattr(self.get_model(), "mm_projector2"):
                image_features_sec = self.get_model().mm_projector2(image_features_sec)
            else:
                image_features_sec = self.get_model().mm_projector(image_features_sec)
            image_features = 0.5 * (image_features + image_features_sec)
        
        return image_features

    def forward(
            self,
            images: Optional[torch.FloatTensor] = None,
            input_ids: torch.LongTensor = None,
            labels: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            images_2d: Optional[torch.Tensor] = None,
            segs: Optional[torch.FloatTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        input_ids_pre = input_ids

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_2d,
            )

        # SegVol Logic
        seg_ids = []
        if segs is not None:
            try:
                if segs.sum() > 0:
                    seg_ids = torch.nonzero(torch.sum(segs, dim=(1, 2, 3, 4))).flatten().tolist()
            except:
                pass

        if hasattr(self.get_model(), "seg_module") and self.get_model().seg_module is not None and len(seg_ids) > 0:
            outputs = super().forward(
                                    input_ids=input_ids,
                                    inputs_embeds=inputs_embeds,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    output_hidden_states=True,
                                    position_ids=position_ids,
                                    past_key_values=past_key_values,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    return_dict=return_dict
                                )

            output_hidden_states = outputs.hidden_states
            last_hidden_state = output_hidden_states[-1]

            seg_token_mask = input_ids_pre[:, 1:] == self.config.seg_token_id
            padding = torch.zeros((seg_token_mask.shape[0], 1), dtype=seg_token_mask.dtype, device=seg_token_mask.device)
            seg_token_mask = torch.cat([seg_token_mask, padding], dim=1)

            seg_prompts = []
            for i in seg_ids:
                if torch.sum(seg_token_mask[i]) == 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype, device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            if len(seg_prompts) > 0:
                seg_prompts = torch.cat(seg_prompts, dim=0)
                logits = self.get_model().seg_module(images[seg_ids], text_emb=seg_prompts)
                loss_dice = self.get_model().dice_loss(logits, segs[seg_ids])
                loss_bce = self.get_model().bce_loss(logits, segs[seg_ids])
                seg_loss = loss_dice + loss_bce
                outputs.loss = outputs.loss + seg_loss
            return outputs
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

    @torch.no_grad()
    def generate(
        self,
        images: Optional[torch.Tensor] = None,
        inputs: Optional[torch.Tensor] = None,
        seg_enable: bool = False,
        images_2d: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor, Any]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                images_2d,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if seg_enable:
            outputs = super().generate(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                return_dict_in_generate=True,
                **kwargs
            )
            output_hidden_states = outputs.hidden_states
            output_ids = outputs.sequences
            seg_token_mask = output_ids[:, 1:] == self.config.seg_token_id

            last_tensors = [tuple[-1] for tuple in output_hidden_states]
            last_hidden_state = torch.cat(last_tensors[1:], dim=1)

            seg_prompts = []
            noseg_ids = []
            for i in range(len(seg_token_mask)):
                if torch.sum(seg_token_mask[i]) == 1:
                    seg_token = last_hidden_state[i][seg_token_mask[i]]
                    seg_prompt = self.get_model().seg_projector(seg_token)
                elif torch.sum(seg_token_mask[i]) > 1:
                    seg_tokens = last_hidden_state[i][seg_token_mask[i]]
                    seg_token = torch.mean(seg_tokens, dim=0, keepdim=True)
                    seg_prompt = self.get_model().seg_projector(seg_token)
                else:
                    noseg_ids.append(i)
                    seg_prompt = torch.zeros([1, self.config.mm_hidden_size], dtype=last_hidden_state.dtype, device=last_hidden_state.device)
                seg_prompts.append(seg_prompt)

            seg_prompts = torch.cat(seg_prompts, dim=0)
            logits = self.get_model().seg_module(images, seg_prompts)
            if len(noseg_ids) > 0:
                logits[noseg_ids] = -torch.inf

            return output_ids, logits
        else:
            return super().generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        images_2d = kwargs.pop("images_2d", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs['images'] = images
        if images_2d is not None:
            inputs['images_2d'] = images_2d
        return inputs

AutoConfig.register("lamed_phi3", LamedPhi3Config)
AutoModelForCausalLM.register(LamedPhi3Config, LamedPhi3ForCausalLM)
