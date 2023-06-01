from typing import Optional, Tuple, Union

import torch 
import torch.nn as nn 
from .gpt import CustomGPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import CausalLMOutputWithCrossAttentions


class SAILGPT2LMHeadModel(CustomGPT2LMHeadModel):
    def __init__(self, 
                 n_layer,
                 n_head,
                 n_prototype,
                 attention_only,
        ):
        
        super().__init__(n_layer=n_layer,
                         n_head=n_head, 
                         n_prototype=n_prototype, 
                         attention_only=attention_only)
    
        
    # https://github.com/huggingface/transformers/blob/e45e756d22206ca8fa9fb057c8c3d8fa79bf81c6/src/transformers/models/gpt2/modeling_gpt2.py#LL1053C9-L1053C9
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
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
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            

            if kwargs.get("save_loss_type") == "last_loss":
                your_loss = None 
                self.loss = your_loss
            
            elif kwargs.get("save_token_loss") is not None:
                #---- Save loss for positions 
                
                ce_loss = torch.nn.CrossEntropyLoss(reduction='none')(shift_logits.view(-1, shift_logits.size(-1)), 
                                                                      shift_labels.view(-1))
                ce_loss = ce_loss.view(shift_labels.size(0), -1).sum(dim=0) # sum over batch 
                self.saved_token_loss = ce_loss
                
            
            
            
            
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
