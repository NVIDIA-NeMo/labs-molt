# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom Qwen2 model implementation for NeMo Automodel.

This module provides a self-contained Qwen2 implementation with separate
HuggingFace-style q/k/v and gate/up projections.

Example (YAML):

```yaml
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: Qwen/Qwen2.5-7B
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import torch
import torch.nn as nn
from transformers import Qwen2Config
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    compute_lm_head_logits,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)

# Use shared rope_utils (same implementation as Llama, supports both config formats)
from nemo_automodel.components.models.llama.rope_utils import (
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_fused,
    apply_rotary_pos_emb_quack,
)
from nemo_automodel.shared.import_utils import get_check_model_inputs_decorator, safe_import_from

__all__ = ["Qwen2ForCausalLM"]

check_model_inputs = get_check_model_inputs_decorator()


class Qwen2Attention(nn.Module):
    """Multi-headed attention with separate QKV projections — HuggingFace default layout."""

    def __init__(self, config: Qwen2Config, layer_idx: int, backend: "BackendConfig" | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.backend = backend or BackendConfig()
        self.rope_fusion = self.backend.rope_fusion
        self.rope_backend = self.backend.rope
        self._quack_apply_rotary_emb = None
        if self.rope_backend == "quack":
            available, apply_rotary_emb = safe_import_from(
                "quack.rotary",
                "apply_rotary_emb",
                msg="rope='quack' requires the 'quack-kernels' package. Install nemo-automodel[cuda].",
            )
            if not available:
                raise ImportError("rope='quack' requires the 'quack-kernels' package. Install nemo-automodel[cuda].")
            self._quack_apply_rotary_emb = apply_rotary_emb

        # Separate projections -- same layout as HuggingFace default Qwen2
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)

        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        if self.backend.attn == "te":
            # BSHD continues through the HuggingFace attention interface; this
            # module is selected only for packed THD batches.
            self._te_thd_only = True
            self.attn_module, self.attn_func = initialize_attn_module_and_func(
                attn_impl="te",
                num_attention_heads=config.num_attention_heads,
                num_qk_channels=self.head_dim,
                num_v_channels=self.head_dim,
                softmax_scale=self.scaling,
                num_gqa_groups=config.num_key_value_heads,
                attention_dropout=config.attention_dropout,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Qwen2 attention over padded BSHD or packed THD states.

        Args:
            hidden_states: Hidden states ``[B, S, H]`` or packed local states
                ``[T, H]``. ``B`` is batch, ``S`` is sequence, ``T`` is local
                total tokens, and ``H`` is hidden size.
            position_embeddings: RoPE tensors ``(cos, sin)`` or fused
                ``(cos, sin, freqs_cis)``. Local tensors use ``[B, S, D]`` or
                ``[T, D]`` and the fused raw table uses ``[S, 1, 1, D]``.
            attention_mask: Padded attention mask for BSHD; THD uses cumulative
                document lengths from ``kwargs``.
            past_key_values: Optional BSHD KV cache; unsupported for THD.
            cache_position: Optional BSHD cache positions ``[S]``.
            **kwargs: THD requires ``qkv_format='thd'`` and ``cu_seqlens``
                ``[N + 1]``; CP additionally supplies ``cp_size`` and ``cp_rank``.

        Returns:
            Attention output shaped like ``hidden_states`` and optional BSHD
            attention weights. THD returns ``None`` for the weights.
        """
        is_thd = kwargs.get("qkv_format") == "thd"
        if is_thd and hidden_states.ndim != 2:
            raise ValueError(f"THD attention requires hidden_states [T, H], got {tuple(hidden_states.shape)}.")

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        if is_thd:
            query_states = q.view(hidden_shape)
            key_states = k.view(hidden_states.shape[0], -1, self.head_dim)
            value_states = v.view(hidden_states.shape[0], -1, self.head_dim)
        else:
            query_states = q.view(hidden_shape).transpose(1, 2)
            key_states = k.view(*input_shape, -1, self.head_dim).transpose(1, 2)
            value_states = v.view(*input_shape, -1, self.head_dim).transpose(1, 2)

        if self.rope_backend == "quack" and query_states.is_cuda:
            cos, sin = position_embeddings[:2]
            query_states, key_states = apply_rotary_pos_emb_quack(
                query_states,
                key_states,
                cos,
                sin,
                self._quack_apply_rotary_emb,
            )
        elif self.rope_fusion and len(position_embeddings) == 3:
            cos, sin, freqs_cis = position_embeddings
            query_states, key_states = apply_rotary_pos_emb_fused(
                query_states,
                key_states,
                freqs_cis,
                cu_seqlens=kwargs.get("cu_seqlens"),
                cp_size=kwargs.get("cp_size", 1),
                cp_rank=kwargs.get("cp_rank", 0),
            )
        else:
            cos, sin = position_embeddings[:2]
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if is_thd:
            if past_key_values is not None:
                raise ValueError("Packed THD attention does not support past_key_values.")
            attn_module = getattr(self, "attn_module", None)
            if attn_module is None:
                raise ValueError("Packed THD attention requires backend.attn='te'.")
            window_size = (-1, 0) if self.sliding_window is None else (self.sliding_window - 1, 0)
            query_states, key_states, value_states, te_kwargs = preprocess_args_and_kwargs_for_attn(
                query_states,
                key_states,
                value_states,
                attention_mask,
                "te",
                window_size=window_size,
                **kwargs,
            )
            attn_output = attn_module(query_states, key_states, value_states, **te_kwargs)
            attn_output = postprocess_output_for_attn(attn_output, "te")
            return self.o_proj(attn_output.flatten(1)), None

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Select attention interface based on config
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # Qwen2 feature
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen2SeparateMLP(nn.Module):
    """SwiGLU MLP with separate gate_proj and up_proj -- identical to HuggingFace default."""

    def __init__(self, config: Qwen2Config):
        super().__init__()
        from transformers.activations import ACT2FN

        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        mlp_bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(GradientCheckpointingLayer):
    """Single Qwen2 decoder layer with RMSNorm, attention, and MLP."""

    def __init__(
        self,
        config: Qwen2Config,
        layer_idx: int,
        backend: BackendConfig,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = Qwen2Attention(config=config, layer_idx=layer_idx, backend=backend)

        self.mlp = Qwen2SeparateMLP(config=config)

        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2PreTrainedModel(PreTrainedModel):
    """Abstract class for Qwen2 pretrained models."""

    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen2DecoderLayer,
        "attentions": Qwen2Attention,
    }


class Qwen2Model(Qwen2PreTrainedModel):
    """Qwen2 transformer model (embeddings + decoder layers + norm)."""

    def __init__(
        self,
        config: Qwen2Config,
        backend: BackendConfig,
    ):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen2DecoderLayer(
                    config=config,
                    layer_idx=layer_idx,
                    backend=backend,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, device=None
        )
        self.rotary_emb = Qwen2RotaryEmbedding(config=config, rope_fusion=backend.rope_fusion)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """Run the Qwen2 decoder in padded BSHD or packed THD layout.

        Args:
            input_ids: Token IDs ``[B, S]`` or packed local IDs ``[T]``.
            attention_mask: Optional padded mask ``[B, S]``. THD uses packed
                document boundaries instead.
            position_ids: Position IDs ``[B, S]`` or packed local IDs ``[T]``.
            past_key_values: Optional BSHD generation cache; unsupported for THD.
            inputs_embeds: Alternative hidden inputs ``[B, S, H]`` or ``[T, H]``.
            use_cache: Whether to update the BSHD KV cache.
            output_attentions: Whether to request attention outputs.
            output_hidden_states: Whether to retain per-layer hidden states.
            return_dict: Whether to return ``BaseModelOutputWithPast``.
            cache_position: Optional BSHD cache positions ``[S]``.
            **kwargs: THD metadata including ``qkv_format``, ``cu_seqlens``,
                ``max_seqlen``, ``cp_size``, and ``cp_rank``.

        Returns:
            Decoder output with final states ``[B, S, H]`` or packed local
            states ``[T, H]`` and requested hidden-state tensors in that layout.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        is_thd = kwargs.get("qkv_format") == "thd"
        if is_thd and inputs_embeds.ndim != 2:
            raise ValueError(f"THD model input must be [T, H], got {tuple(inputs_embeds.shape)}.")
        if is_thd:
            use_cache = False
            if past_key_values is not None:
                raise ValueError("Packed THD training does not support past_key_values.")

        if not is_thd and use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if not is_thd and cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            if is_thd:
                if kwargs.get("cp_size", 1) > 1:
                    raise ValueError("THD context parallelism requires explicit position_ids.")
                position_ids = torch.arange(inputs_embeds.shape[0], device=inputs_embeds.device)
            else:
                position_ids = cache_position.unsqueeze(0)

        # Create masks (Qwen2 supports sliding window attention)
        if is_thd:
            causal_mask_mapping = {"full_attention": None, "sliding_attention": None}
        elif not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(
            hidden_states,
            position_ids,
            qkv_format="thd" if is_thd else "bshd",
            cp_size=kwargs.get("cp_size", 1),
        )

        all_hidden_states = () if output_hidden_states else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values if use_cache else None,
                    all_hidden_states,
                ]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class Qwen2ForCausalLM(HFCheckpointingMixin, Qwen2PreTrainedModel):
    """Qwen2 model with causal language modeling head.

    Uses separate q/k/v and gate/up projections -- HuggingFace layout.
    """

    tie_word_embeddings_support: TieSupport = TieSupport.BOTH
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = True
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = False

    def __init__(
        self,
        config: Qwen2Config,
        backend: BackendConfig | None = None,
    ):
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)
        self.backend = backend or BackendConfig()
        self.model = Qwen2Model(config=config, backend=self.backend)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Tie weights if specified in config (standard for Qwen2/Llama)
        # Must be done after post_init() to ensure embed_tokens is initialized
        if getattr(config, "tie_word_embeddings", True):
            self.tie_weights()

        # Convert to configured dtype if specified
        if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
            self.to(dtype=config.torch_dtype)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"[Qwen2ForCausalLM] Attention implementation: {self.config._attn_implementation}")
            print(f"[Qwen2ForCausalLM] torch_dtype: {self.config.torch_dtype}")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def tie_weights(self, *_args: object, **_kwargs: object) -> None:
        # Transformers v5 does not reliably tie this custom model from the
        # dict-shaped _tied_weights_keys alone; honor the config flag explicitly
        # (mirrors LlamaForCausalLM).
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        """Run causal LM projection for BSHD or packed THD inputs.

        Args:
            input_ids: Token IDs ``[B, S]`` or packed local IDs ``[T]``.
            attention_mask: Optional padded mask. THD uses ``cu_seqlens``.
            position_ids: Position IDs ``[B, S]`` or packed local IDs ``[T]``.
            past_key_values: Optional BSHD KV cache; unsupported for THD.
            inputs_embeds: Optional hidden inputs ``[B, S, H]`` or ``[T, H]``.
            labels: Optional labels ``[B, S]`` or packed ``[T]``.
            use_cache: Whether to update the BSHD KV cache.
            output_attentions: Whether to request attention outputs.
            output_hidden_states: Whether to return per-layer hidden states.
            return_dict: Whether to return ``CausalLMOutputWithPast``.
            cache_position: Optional BSHD cache positions ``[S]``.
            logits_to_keep: Positions to project from hidden size ``H`` to
                vocabulary size ``V``.
            **kwargs: THD metadata. ``cu_seqlens`` is ``[N + 1]`` and CP adds
                ``cp_size`` and ``cp_rank``.

        Returns:
            Causal LM output with BSHD logits ``[B, S, V]``. Packed local
            logits are restored to ``[1, T, V]``.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_thd = kwargs.get("qkv_format") == "thd"
        if is_thd:
            if position_ids is None:
                raise ValueError("Packed THD input requires position_ids.")
            kwargs.pop("padding_mask", None)
            if input_ids is not None and input_ids.ndim > 1:
                input_ids = input_ids.squeeze(0)
            if position_ids.ndim > 1:
                position_ids = position_ids.squeeze(0)
            for key, value in kwargs.items():
                if not isinstance(value, torch.Tensor):
                    continue
                if key == "max_seqlen":
                    kwargs[key] = value.item()
                    continue
                if value.ndim > 1:
                    value = value.squeeze(0)
                if key in ("cu_seqlens", "cu_seqlens_padded"):
                    value = value[value != -1000].contiguous()
                kwargs[key] = value
            if inputs_embeds is not None and inputs_embeds.ndim > 2:
                inputs_embeds = inputs_embeds.squeeze(0)
            if labels is not None and labels.ndim > 1:
                labels = labels.squeeze(0)
            attention_mask = None

        # Always use return_dict internally so we can reliably access fields.
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        logits = compute_lm_head_logits(self.lm_head, hidden_states, logits_to_keep, is_thd=is_thd).logits

        loss = None
        if labels is not None:
            if is_thd and labels.ndim == 1:
                labels = labels.unsqueeze(0)
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        out = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
        if return_dict:
            return out
        return out.to_tuple()


ModelClass = Qwen2ForCausalLM
