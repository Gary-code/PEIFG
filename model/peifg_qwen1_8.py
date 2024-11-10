from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from peifg.utils.constants import *
from peifg.model.plug.blip_process import BlipImageEvalProcessor
from peifg.model.llm.qwen.modeling_qwen import QWenLMHeadModel, QWenModel
from peifg.model.llm.qwen.configuration_qwen import QWenConfig
from peifg.model.vision_encoder.sam import build_sam_vit_b
from peifg.model.qformer import Qformer
from peifg.model.prompt import Prompt

class peifgConfig(QWenConfig):
    model_type = "peifg"


class peifgQwenModel(QWenModel):
    # SAM + CLIP-L + QWen(LLM)
    config_class = peifgConfig

    def __init__(self, config: QWenConfig):
        super(peifgQwenModel, self).__init__(config)
        # self.language_model = QWenModel(config)
        self.vision_tower = CLIPVisionModel.from_pretrained('./pre-trained/clip-vit-large-patch14')
        self.vision_tower_high = build_sam_vit_b("./pre-trained/SAM")  # stage 1 pre-trained SAM
        self.qformer = Qformer.from_pretrained("./pre-trained/instructblip-flan-t5-xl", torch_dtype=torch.float16, ignore_mismatched_sizes=True)
        self.mm_projector =  nn.Linear(1024, 1024)
        self.mm_projector =  nn.Linear(1024, 1024) 
        self.img_projector = nn.Linear(2048, 1408)  # project img to Qformer
        # TODO bias True, self.expert_projector will be meta
        self.expert_projector = nn.Linear(768, 2048, bias=False) # project expert to LLM
        # TODO modify the pool size or top_key
        self.expert_prompt = Prompt(length=5, embed_dim=768, pool_size=10, top_k=3) # prompt 
        # TODO No SAM or CLIP linear
        self.img_projector_NO_SAM_Or_CLIP = nn.Linear(1024, 2048, bias=False)
        

    def initialize_vision_modules(
        self, 
        vision_tower,
        pretrained_stage1_model=None,
        freeze_vision_tower=False,
        use_im_start_end=False,
        vision_select_layer=-1,
        dtype=torch.float16,
        device='auto',
        is_train=False
    ):

        # 224*224
        image_processor = CLIPImageProcessor.from_pretrained('./pre-trained/clip-vit-large-patch14') 
        # 1024*1024
        image_processor_high = BlipImageEvalProcessor(image_size=1024)
      
        self.vision_tower = self.vision_tower.to(dtype=dtype, device=device)  # CLIP-L
        self.vision_tower_high = build_sam_vit_b("./pre-trained/SAM")
        self.vision_tower_high = self.vision_tower_high.to(dtype=dtype, device=device)  # SAM
        self.qformer = self.qformer.to(dtype=dtype, device=device) # qformer
        self.img_projector = self.img_projector.to(dtype=dtype, device=device)
        self.expert_projector = self.expert_projector.to(dtype=dtype, device=device)
        self.expert_prompt = self.expert_prompt.to(dtype=dtype, device=device) # prompt 
        self.mm_projector = self.mm_projector.to(dtype=dtype, device=device)
        self.mm_projector = self.mm_projector.to(dtype=dtype, device=device)
        self.img_projector_NO_SAM_Or_CLIP = self.img_projector_NO_SAM_Or_CLIP.to(dtype=dtype, device=device)
        if is_train:
            self.expert_prompt.prompt = nn.init.uniform_(self.expert_prompt.prompt, -1, 1)



        image_token_len = 256

        self.config.vision_tower = vision_tower
        self.config.image_token_len = image_token_len
        self.config.use_im_start_end = True
        self.config.vision_select_layer = vision_select_layer
        self.config.freeze_vision_tower = freeze_vision_tower
        
        return dict(
            image_processor=image_processor,
            image_processor_high=image_processor_high,
            image_token_len=image_token_len,
            # vision_config=vision_config
        )

    def embed_tokens(self, x):
        return self.wte(x)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        query_input_ids: torch.LongTensor = None,
        query_attention_mask: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:



        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)


        vision_tower = getattr(self, 'vision_tower', None)
        vision_tower_high = getattr(self, 'vision_tower_high', None)
        # TODO test for RL
        expert_corr_loss = None
        expert_reduce_sim = None
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:

            use_im_start_end = getattr(self.config, "use_im_start_end", -1)

            vision_select_layer = getattr(self.config, "vision_select_layer", -1)
            im_patch_token = getattr(self.config, "im_patch_token", -1)
            im_start_token = getattr(self.config, "im_start_token", -1)
            im_end_token = getattr(self.config, "im_end_token", -1)
            freeze_vision_tower = getattr(self.config, "freeze_vision_tower", False)
            
            im_patch_token = 151859   
            im_start_token = 151857  # 图像开始token
            im_end_token = 151858    # 图像结束token
            
            # expert token
            ex_pad_token = 151851
            ex_start_token = 151855
            ex_end_token = 151856


            image_features = []
            image_features_1 = []
            image_features_2 = []
            
            for image in images:
                # with torch.set_grad_enabled(False):
                image_forward_out = vision_tower(image[0], output_hidden_states=True)
                select_hidden_state = image_forward_out.hidden_states[vision_select_layer]
                image_feature = select_hidden_state[:, 1:]  # 256*1024
                cnn_feature = vision_tower_high(image[1])
                cnn_feature = cnn_feature.flatten(2).permute(0, 2, 1) # 256*1024
                image_features_1.append(image_feature)
                image_features_2.append(cnn_feature)


            if type(images) is list:
                image_features_1 = [self.mm_projector(image_feature) for image_feature in image_features_1]
                image_features_2 = [self.mm_projector(image_feature) for image_feature in image_features_2]                
                image_features = [torch.cat((image_feature[0], image_feature[1]), dim=-1) for image_feature in zip(image_features_1, image_features_2)] # (bs, 256, 2048)
                # TODO NO SAM or CLIP, image_featrue_1 for clip, image_featrue_1 for SAM
                # image_features = [self.img_projector_NO_SAM_Or_CLIP(image_feature) for image_feature in image_features_2] # (bs, 256, 2048)
                
                # qformer
                image_features_qformer = [self.img_projector(image_feature) for image_feature in image_features]
                image_stack_feature = torch.cat(image_features_qformer, dim=0)
                qformer_outputs = self.qformer(image_embeds=image_stack_feature, qformer_input_ids=query_input_ids, qformer_attention_mask=query_attention_mask)
                
                # TODO no expert
                # expert_corr_loss = None
                # expert_reduce_sim = None
                
                # obtain expert prompt
                expert_prompt_dict = self.expert_prompt(qformer_outputs['pooler_output'])
                expert_prompt_embeds = self.expert_projector(expert_prompt_dict['prompted_embedding'])  # bs, topK * length, 768 -> 2048
                expert_reduce_sim = expert_prompt_dict['reduce_sim']
                expert_corr_loss = expert_prompt_dict['corr_loss']
            else:
                raise NotImplementedError
            dummy_image_features_1 = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features_2 = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features_1 = self.mm_projector(dummy_image_features_1)
            dummy_image_features_2 = self.mm_projector(dummy_image_features_2)
            dummy_image_features = torch.cat((dummy_image_features_1, dummy_image_features_2), dim=-1)
            use_im_start_end = True
            new_input_embeds = []
            # TODO no expert
            # for cur_input_ids, cur_input_embeds, cur_image_features in zip(input_ids, inputs_embeds, image_features):
            for cur_input_ids, cur_input_embeds, cur_image_features, cur_expert_prompt_embeds in zip(input_ids, inputs_embeds, image_features, expert_prompt_embeds):
                if (cur_input_ids == im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue

                if use_im_start_end:
                    if (cur_input_ids == im_start_token).sum() != (cur_input_ids == im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    
                    image_start_tokens = torch.where(cur_input_ids == im_start_token)[0]
                    for image_start_token_pos, per_cur_image_features in zip(image_start_tokens, cur_image_features):
                        # replace the image token
                        per_cur_image_features = per_cur_image_features.to(device=cur_input_embeds.device)
                        num_patches = per_cur_image_features.shape[0]

                        if cur_input_ids[image_start_token_pos + num_patches + 1] != im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        
                        cur_input_embeds = torch.cat(
                            (
                                cur_input_embeds[:image_start_token_pos+1], 
                                per_cur_image_features, 
                                cur_input_embeds[image_start_token_pos + num_patches + 1: image_start_token_pos + num_patches + 4],
                                cur_expert_prompt_embeds,
                                cur_input_embeds[image_start_token_pos + num_patches + 3 + cur_expert_prompt_embeds.shape[0] + 1: ]
                            ), 
                            dim=0
                        )
                        

                    new_input_embeds.append(cur_input_embeds)
                else:
                    raise NotImplementedError

            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        language_model_output =  super(peifgQwenModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        if input_ids.shape[1] != 1:
            return language_model_output, expert_reduce_sim, expert_corr_loss
        else:
            return language_model_output, None, None

class peifgQwenForCausalLM(QWenLMHeadModel):
    config_class = peifgConfig


    def __init__(self, config):
        super(QWenLMHeadModel, self).__init__(config)
        self.transformer = peifgQwenModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        query_input_ids: torch.LongTensor = None,
        query_attention_mask: torch.LongTensor = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        transformer_outputs, reduce_sim, corr_loss = self.transformer(
            input_ids=input_ids,
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict
            
        )

        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        # logits

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            
        # for n,p in self.transformer.named_parameters():
            

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        # TODO no expert
        # if loss is not None:
        #         loss = loss
        
        if loss is not None:
            loss = loss - 0.1 * reduce_sim + 0.1 * corr_loss

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
                "images": kwargs.get("images", None),
                "query_input_ids": kwargs.get("query_input_ids", None),
                "query_attention_mask": kwargs.get("query_attention_mask", None)
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(
        self, 
        tokenizer, 
        freeze_lm_model=False, 
        pretrained_stage1_model=None,
        device="cuda"
    ):
        config = self.get_model().config

        self.resize_token_embeddings(len(tokenizer))

        config.im_patch_token = 151859

        config.use_im_start_end = True

        if config.use_im_start_end:
            self.resize_token_embeddings(len(tokenizer))

            config.im_start_token, config.im_end_token = 151857, 151858



AutoConfig.register("peifg", peifgConfig)
AutoModelForCausalLM.register(peifgConfig, peifgQwenForCausalLM)
