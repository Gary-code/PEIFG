import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm
from transformers import (InstructBlipQFormerModel, InstructBlipPreTrainedModel, InstructBlipForConditionalGeneration)

class Qformer(InstructBlipPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        
        self.qformer = InstructBlipQFormerModel(config.qformer_config)
       
        self.query_tokens = nn.Parameter(torch.zeros(1, config.num_query_tokens, config.qformer_config.hidden_size))
        
    def forward(self, image_embeds, qformer_input_ids, qformer_attention_mask=None):
        """
       image: bs, 256, 1408
        """
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        
        return query_outputs