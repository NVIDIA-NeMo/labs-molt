# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Dense Qwen3 implementation with packed THD context parallelism."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import Qwen3Config
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention as HFQwen3Attention,
)
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer as HFQwen3DecoderLayer,
)
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import BackendConfig, compute_lm_head_logits
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import TieSupport
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.llama.rope_utils import (
    Qwen2RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_fused,
)
from nemo_automodel.shared.import_utils import get_check_model_inputs_decorator

__all__ = ["Qwen3ForCausalLM"]

check_model_inputs = get_check_model_inputs_decorator()


class Qwen3Attention(HFQwen3Attention):
    """HuggingFace Qwen3 attention extended with a packed THD TE path."""

    def __init__(self, config: Qwen3Config, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__(config=config, layer_idx=layer_idx)
        self.backend = backend
        self.rope_fusion = backend.rope_fusion
        if backend.attn == "te":
            # Ordinary BSHD inputs retain the HuggingFace attention interface.
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run Qwen3 attention over padded BSHD or packed THD states.

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
            **kwargs: THD requires ``qkv_format='thd'`` and ``cu_seqlens``
                ``[N + 1]``; CP additionally supplies ``cp_size`` and ``cp_rank``.

        Returns:
            Attention output shaped like ``hidden_states`` and optional BSHD
            attention weights. THD returns ``None`` for the weights.
        """
        is_thd = kwargs.get("qkv_format") == "thd"
        if not is_thd:
            return super().forward(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings[:2],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                **kwargs,
            )
        if hidden_states.ndim != 2:
            raise ValueError(f"THD attention requires hidden_states [T, H], got {tuple(hidden_states.shape)}.")
        if past_key_values is not None:
            raise ValueError("Packed THD attention does not support past_key_values.")
        attn_module = getattr(self, "attn_module", None)
        if attn_module is None:
            raise ValueError("Packed THD attention requires backend.attn='te'.")

        token_count = hidden_states.shape[0]
        query_states = self.q_norm(self.q_proj(hidden_states).view(token_count, -1, self.head_dim))
        key_states = self.k_norm(self.k_proj(hidden_states).view(token_count, -1, self.head_dim))
        value_states = self.v_proj(hidden_states).view(token_count, -1, self.head_dim)

        if self.rope_fusion and len(position_embeddings) == 3:
            query_states, key_states = apply_rotary_pos_emb_fused(
                query_states,
                key_states,
                position_embeddings[2],
                cu_seqlens=kwargs.get("cu_seqlens"),
                cp_size=kwargs.get("cp_size", 1),
                cp_rank=kwargs.get("cp_rank", 0),
            )
        else:
            query_states, key_states = apply_rotary_pos_emb(
                query_states,
                key_states,
                position_embeddings[0],
                position_embeddings[1],
            )

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


class Qwen3DecoderLayer(HFQwen3DecoderLayer):
    """Qwen3 decoder layer using the packed-aware attention implementation."""

    def __init__(self, config: Qwen3Config, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__(config=config, layer_idx=layer_idx)
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx, backend=backend)
        self.attention_type = config.layer_types[layer_idx]


class Qwen3PreTrainedModel(PreTrainedModel):
    """Base class for the dense Qwen3 implementation."""

    config_class = Qwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


class Qwen3Model(Qwen3PreTrainedModel):
    """Dense Qwen3 decoder supporting padded BSHD and packed THD layouts."""

    def __init__(self, config: Qwen3Config, backend: BackendConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config=config, layer_idx=idx, backend=backend)
                for idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config, rope_fusion=backend.rope_fusion)
        self.gradient_checkpointing = False
        self.has_sliding_layers = "sliding_attention" in self.config.layer_types
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
        """Run the Qwen3 decoder in padded BSHD or packed THD layout.

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
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
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
                value
                for value in [hidden_states, past_key_values if use_cache else None, all_hidden_states]
                if value is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class Qwen3ForCausalLM(HFCheckpointingMixin, Qwen3PreTrainedModel, GenerationMixin):
    """Dense Qwen3 causal LM with packed THD context parallelism."""

    tie_word_embeddings_support: TieSupport = TieSupport.BOTH
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _keep_in_fp32_modules = ["rotary_emb"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = True
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = False

    def __init__(self, config: Qwen3Config, backend: BackendConfig | None = None) -> None:
        super().__init__(config)
        self.backend = backend or BackendConfig()
        self.model = Qwen3Model(config=config, backend=self.backend)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        if config.dtype is not None:
            cast_model_to_dtype(self, config.dtype)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def tie_weights(self, *_args: object, **_kwargs: object) -> None:
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
        logits_to_keep: int | torch.Tensor = 0,
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
        logits = compute_lm_head_logits(self.lm_head, outputs.last_hidden_state, logits_to_keep, is_thd=is_thd).logits

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


ModelClass = Qwen3ForCausalLM
