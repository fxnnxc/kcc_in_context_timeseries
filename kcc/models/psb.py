from typing import Optional, Tuple, Union

import torch 
import torch.nn as nn 
from transformers.models.gpt2.modeling_gpt2 import (CausalLMOutputWithCrossAttentions, 
                                                    GPT2PreTrainedModel, 
                                                    GPT2Block, 
                                                    BaseModelOutputWithPastAndCrossAttentions,
                                                    GPT2LMHeadModel,
                                                    GPT2Config,
                                                    GPT2Attention
)       

import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from transformers.utils import ModelOutput


@dataclass
class PSBModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    scale:Optional[Tuple[torch.FloatTensor]] =None
    bias:Optional[Tuple[torch.FloatTensor]] =None
    scale_loss:Optional[Tuple[torch.FloatTensor]] =None
    bias_loss:Optional[Tuple[torch.FloatTensor]] =None
    ce_loss:Optional[Tuple[torch.FloatTensor]] =None


class PSBGPT2Model(GPT2PreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)
        
        self.scale_weight = nn.Linear(1, self.embed_dim)
        self.bias_weight = nn.Linear(1, self.embed_dim)
        

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            input_scale = None, 
            input_bias =None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                batch_size = input_ids.shape[0]
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
                batch_size = inputs_embeds.shape[0]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            if token_type_ids is not None:
                token_type_ids = token_type_ids.view(-1, input_shape[-1])
            if position_ids is not None:
                position_ids = position_ids.view(-1, input_shape[-1])

            if past_key_values is None:
                past_length = 0
                past_key_values = tuple([None] * len(self.h))
            else:
                past_length = past_key_values[0][0].size(-2)
            if position_ids is None:
                position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
                position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

            # GPT2Attention mask.
            if attention_mask is not None:
                if batch_size <= 0:
                    raise ValueError("batch_size has to be defined and > 0")
                attention_mask = attention_mask.view(batch_size, -1)
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

            # If a 2D or 3D attention mask is provided for the cross-attention
            # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.add_cross_attention and encoder_hidden_states is not None:
                encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
            else:
                encoder_attention_mask = None

            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # head_mask has shape n_layer x batch x n_heads x N x N
            head_mask = self.get_head_mask(head_mask, self.config.n_layer)

            if inputs_embeds is None:
                inputs_embeds = self.wte(input_ids)
            position_embeds = self.wpe(position_ids)
            hidden_states = inputs_embeds + position_embeds

            if token_type_ids is not None:
                token_type_embeds = self.wte(token_type_ids)
                hidden_states = hidden_states + token_type_embeds

            # ------------------------
            scale = self.scale_weight(input_scale.float())
            bias  = self.bias_weight(input_bias.float())
            
            hidden_states +=  (scale + bias)
            
            # ------------------------


            hidden_states = self.drop(hidden_states)

            output_shape = input_shape + (hidden_states.size(-1),)

            if self.gradient_checkpointing and self.training:
                if use_cache:
                    pass 
                    # logger.warning_once(
                    #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    # )
                    # use_cache = False

            presents = () if use_cache else None
            all_self_attentions = () if output_attentions else None
            all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
            all_hidden_states = () if output_hidden_states else None
            for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
                # Model parallel
                if self.model_parallel:
                    torch.cuda.set_device(hidden_states.device)
                    # Ensure layer_past is on same device as hidden_states (might not be correct)
                    if layer_past is not None:
                        layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                    # Ensure that attention_mask is always on the same device as hidden_states
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(hidden_states.device)
                    if isinstance(head_mask, torch.Tensor):
                        head_mask = head_mask.to(hidden_states.device)
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            # None for past_key_value
                            return module(*inputs, use_cache, output_attentions)

                        return custom_forward

                    outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        None,
                        attention_mask,
                        head_mask[i],
                        encoder_hidden_states,
                        encoder_attention_mask,
                    )
                else:
                    outputs = block(
                        hidden_states,
                        layer_past=layer_past,
                        attention_mask=attention_mask,
                        head_mask=head_mask[i],
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )

                hidden_states = outputs[0]
                if use_cache is True:
                    presents = presents + (outputs[1],)

                if output_attentions:
                    all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                    if self.config.add_cross_attention:
                        all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

                # Model Parallel: If it's the last layer for that device, put things on the next device
                if self.model_parallel:
                    for k, v in self.device_map.items():
                        if i == v[-1] and "cuda:" + str(k) != self.last_device:
                            hidden_states = hidden_states.to("cuda:" + str(k + 1))

            hidden_states = self.ln_f(hidden_states)

            hidden_states = hidden_states.view(output_shape)
            # Add last hidden state
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                return tuple(
                    v
                    for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                    if v is not None
                )

            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=presents,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            )


# ------- GPT LM Model -----------------------------------------------
# https://github.com/huggingface/transformers/blob/e45e756d22206ca8fa9fb057c8c3d8fa79bf81c6/src/transformers/models/gpt2/modeling_gpt2.py#L957
class PSBGPT2LMHeadModel(GPT2LMHeadModel):
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
        self.transformer = PSBGPT2Model(self.config)
        self.bias_head   = nn.Linear(self.transformer.embed_dim, 1)
        self.scale_head  = nn.Linear(self.transformer.embed_dim, 1)
        
        if attention_only:
            for layer in range(n_layer):    
                self.transformer.h[layer] = GPT2AttentionOnlyBlock(config)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            input_scale=None,
            input_bias=None,
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
                input_scale, 
                input_bias, 
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

            lm_logits  = self.lm_head(hidden_states)
            scale_pred = self.scale_head(hidden_states)
            bias_pred  = self.bias_head(hidden_states)

            loss = None
            ce_loss = None 
            bias_loss = None 
            scale_loss = None 
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(lm_logits.device)
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                ce_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                if kwargs.get("save_loss_type") == "last_loss":
                    your_loss = None 
                    self.loss = your_loss
                
                elif kwargs.get("save_token_loss") is not None:
                    #---- Save loss for positions 
                    
                    token_ce_loss = torch.nn.CrossEntropyLoss(reduction='none')(shift_logits.view(-1, shift_logits.size(-1)), 
                                                                        shift_labels.view(-1))
                    token_ce_loss = token_ce_loss.view(shift_labels.size(0), -1).sum(dim=0) # sum over batch 
                    self.saved_token_loss = token_ce_loss
                    
                bias_loss = nn.MSELoss()(input_bias, bias_pred)
                scale_loss = nn.MSELoss()(input_bias, scale_pred)
                loss = ce_loss + bias_loss + scale_loss 
                

            # ------



            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output

            return PSBModelOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=transformer_outputs.past_key_values,
                hidden_states=transformer_outputs.hidden_states,
                attentions=transformer_outputs.attentions,
                cross_attentions=transformer_outputs.cross_attentions,
                scale=scale_pred,
                bias=bias_pred,
                ce_loss=ce_loss,
                bias_loss = bias_loss,
                scale_loss =scale_loss
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




