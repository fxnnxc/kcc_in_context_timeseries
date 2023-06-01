from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, CausalLMOutputWithCrossAttentions
from transformers import GPT2LMHeadModel, GPT2Config
from typing import Optional, Tuple, Union

import torch.nn as nn 
import torch 

# ------- GPT LM Model ----------
# https://github.com/huggingface/transformers/blob/e45e756d22206ca8fa9fb057c8c3d8fa79bf81c6/src/transformers/models/gpt2/modeling_gpt2.py#L957
class CustomGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, 
                 n_layer,
                 n_head,
                 n_prototype=50257,
                 attention_only=False, 
                 **kwargs):
        config = GPT2Config(
            vocab_size=n_prototype,
            n_layer=n_layer,
            n_head=n_head,
            **kwargs
        )
        print(config)
        self.config = config
        super().__init__(config)    
        # --- Change the Block as Attention Only Block 
        if attention_only:
            for layer in range(n_layer):    
                self.transformer.h[layer] = GPT2AttentionOnlyBlock(config)

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



# ----- Hard copy of GPT2Block for ATTNENTION only Model 
# https://github.com/huggingface/transformers/blob/e45e756d22206ca8fa9fb057c8c3d8fa79bf81c6/src/transformers/models/gpt2/modeling_gpt2.py#L362
class GPT2AttentionOnlyBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config, layer_idx=layer_idx)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

    def forward(
                self,
                hidden_states,
                layer_past= None,
                attention_mask= None,
                head_mask= None,
                encoder_hidden_states = None,
                encoder_attention_mask = None,
                use_cache = False,
                output_attentions = False,
            ):
        
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)






    
if __name__ == "__main__":
    gpt  = CustomGPT2LMHeadModel(
        n_layer=1,
        n_head=4, 
        attention_only=True
    )
    print(gpt)
    
    gpt  = CustomGPT2LMHeadModel(
        n_layer=1,
        n_head=4, 
        attention_only=False 
    )
    print(gpt)
    
    gpt  = CustomGPT2LMHeadModel(
        n_layer=0,
        n_head=4, 
        attention_only=False 
    )
    print(gpt)