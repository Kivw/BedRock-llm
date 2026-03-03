from collections.abc import Callable
from typing import Optional

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring,can_return_tuple
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_bedrockv3 import BedRockV3Config


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        return hidden_states


class SwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        x = self.up_proj(x)
        return self.down_proj(gate * x)


class RotaryEmbedding(nn.Module):
    def __init__(self, config: BedRockV3Config, device=None):
        super().__init__()

        self.max_seq_len = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        # 随着rope的发展，出现了多种变体，这里采用原始论文中不加scaling的版本
        self.rope_type = 'default' # 必须要有这个属性，在post_init中会检查
        inv_freq, _  = self.compute_default_rope_parameters(config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: Optional[BedRockV3Config] = None,  # BedrockV3Config类型的可选参数，用于配置模型参数
        device: Optional["torch.device"] = None,  # torch.device类型的可选参数，指定计算设备
        seq_len: Optional[int] = None  # 整数类型的可选参数，指定序列长度
    ):
        '''
        计算ROPE中的theta参数，即旋转角度中用到的频率
        '''
    
        base = config.rope_parameters['rope_theta']
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads

        attention_factor = 1.0  # Unused in this type of RoPE，必须要有，后续检查中期望返回两个值

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float64) / dim)
        ) # [dim/2]
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        '''
        依据频率和位置信息计算旋转角度并得到旋转角度的三角函数值
        '''
        # self.inv_freq[None, :, None]->[1,dim/2,1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device) # [batch,dim/2,1]
        position_ids_expanded = position_ids[:, None, :].float() # [batch,1,seq_len]

        device_type = x.device.type if isinstance(x.device, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False): # force float32
            # 依据频率和位置信息计算旋转角度
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1,2).contiguous() # [batch,dim/2,1] @ [batch,1,seq_len] -> [batch,dim/2,seq_len]-> [batch,seq_len,dim/2]
            emb = torch.cat((freqs, freqs), dim=-1) # [batch,seq_len,dim] 将dim分成前后两部分，第一个维度和第dim/2+1个维度组合进行旋转
            cos = emb.cos() # [batch,seq_len,dim]
            sin = emb.sin() # [batch,seq_len,dim]

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2] # 前一半维度
    x2 = x[..., x.shape[-1] // 2 :] # 后一半维度
    return torch.cat((-x2, x1), dim=-1) # 将后一半维度取负并与前一半维度调换位置拼接

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim) # [batch,1,seq_len,dim]
    sin = sin.unsqueeze(unsqueeze_dim) # [batch,1,seq_len,dim]
    q_embed = (q * cos) + (rotate_half(q) * sin) # 这里q*cos，在q的dim维度上全部乘以cos, rotate_half(q) * sin,在q的dim维度上全部乘以sin,然后相加对应公式f_q(x_m, m)
    k_embed = (k * cos) + (rotate_half(k) * sin) # 这里k*cos，在k的dim维度上全部乘以cos, rotate_half(k) * sin,在k的dim维度上全部乘以sin,然后相加对应公式f_k(x_m, m)
    return q_embed, k_embed
        
def repeat_kv(hidden_states: torch.Tensor, n_rep: int):
    """
    This is the equivalent of torch.repeat_interleave(hidden_states, n_rep, dim=1), the hidden states go from (batch, num_key_value_heads, seqlen, head_dim)
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups) # (batch, num_attention_heads, seqlen, head_dim)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling # [batch, num_attention_heads, seqlen, seqlen]
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states) # [batch, num_attention_heads, seqlen, head_dim]
    attn_output = attn_output.transpose(1, 2).contiguous() # [batch, seqlen, num_attention_heads, head_dim]

    return attn_output, attn_weights



class Attention(nn.Module):
    def __init__(self, config: BedRockV3Config, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads # 多少个组 Key/Value
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs]
    ) -> tuple[torch.Tensor, Cache | None]:

        input_shape = hidden_states.shape[:-1] # (batch, seqlen)
        hidden_shape = (*input_shape, -1, self.head_dim) # (batch, seqlen, -1, head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1,2).contiguous() # (batch, num_attention_heads, seqlen, head_dim)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1,2).contiguous()
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1,2).contiguous()

        cos, sin = position_embeddings # [batch,seq_len,dim]
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE modules, cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            # The expected shape for each tensor in the `CacheLayer`s is `[batch_size, num_heads, seq_len, head_dim]`.
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)# 这里更新key_states和value_states
        
        attention_interface = Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            # sliding_window=self.sliding_window,  # we delete this funciton
            **kwargs,
        )
        
        attn_output = attn_output.reshape(*input_shape, -1).contiguous() # (batch, seqlen, dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


# ========================以上部分为核心组件======================

class BedRockDecoderLayer(GradientCheckpointingLayer):
    '''
    这里继承了GradientCheckpointingLayer，用于梯度检查点，可以在训练时节省显存.
    梯度检查通过在forward时只保存结果不保存中间激活值，激活值在反向传播时重新计算的方法节省显存。
    '''
    def __init__(self, config: BedRockV3Config, layer_idx: int):
        super().__init__()

        self.hidden_size = config.hidden_size
        self.self_attn = Attention(config=config, layer_idx=layer_idx)

        self.mlp = SwiGLU(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values:  Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs]
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states) # pre-norm
        
        # self attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids, # flash
            past_key_values=past_key_values,
            use_cache=use_cache, # flash
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
        )
        hidden_states = residual + hidden_states
        #  fully connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states) # pre-norm
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


@auto_docstring # 自动生成文档
class BedRockPreTrainedModel(PreTrainedModel):
    config: BedRockV3Config
    base_model_prefix = "model" # HF uses this to locate the backbone model (self.model) for loading, device mapping, and weight tying
    supports_gradient_checkpointing = True # enables HuggingFace gradient checkpointing support to reduce memory usage by recomputing activations during backward pass
    _no_split_modules = ["BedRockDecoderLayer"] #  used by HuggingFace device_map to prevent splitting decoder layers across multiple devices (ensures layer integrity)
    _skip_keys_device_placement = ["past_key_values"] # 告诉 HuggingFace 在 device placement 时跳过 past_key_values，不自动移动它们的设备（KV cache 由模型手动管理）

    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True 
    _supports_attention_backend = True  # 声明模型支持 HuggingFace 的可切换 attention 后端（如 FlashAttention 或 PyTorch native），用于优化推理和训练性能
    _can_record_outputs = {
        "hidden_states": BedRockDecoderLayer, # 当 output_hidden_states=True 时，HF 会记录 BedRockDecoderLayer 的输出隐藏状态
        "attentions": Attention, # 当 output_attentions=True 时，HF 会记录 Attention 的注意力矩阵
    }

@auto_docstring
class BedRockV3Model(BedRockPreTrainedModel):
    def __init__(self, config: BedRockV3Config):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [BedRockDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = RotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init() # Initialize weights and apply final processing

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs], # 允许传递其他参数
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):# ^ 是异或运算符，表示当且仅当 input_ids 和 inputs_embeds 其中一个为 None 时，条件成立
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds") 

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) # 如果没有提供输入嵌入，则使用嵌入层将输入 ID 转换为嵌入

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ) # 如果使用缓存，并且没有提供 past_key_values，则根据输入嵌入的形状生成 cache_position
        
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) # [1, seq_len]
        
        # 构造causal mask
        if not isinstance(causal_mask_mapping := attention_mask, dict): # := 是海象运算符，赋值并返回该值
            # Prepare mask arguments
            mask_kwargs = {
                "config" : self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # create the masks # 实际上，模型一开始传入的attention mask只包含了填充mask,这里将其转换为causal mask和padding mask的结合
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.layers[:self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping["full_attention"],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_positions=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

@auto_docstring
class BedRockV3ForCausalLM(BedRockPreTrainedModel, GenerationMixin):
    config_class = BedRockV3Config
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"} # 共享embedding layer和lm_head的权重,既可以节省参数量又可以提升效果
    _tp_plan = {"lm_head": "colwise_gather_output"} # Tensor Parallel 分割和 gather
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])} # Pipeline Parallel 输入输出接口

        
    def __init__(self, config):
        super().__init__(config)
        self.model = BedRockV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    @can_return_tuple # 
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Unpack[TransformersKwargs],
    ):
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state

        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

__all__ = [
    "BedRockV3Model",
    "BedRockV3ForCausalLM",
]