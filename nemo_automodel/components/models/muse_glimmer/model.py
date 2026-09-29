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

"""Native AutoModel implementation of the complete dense MuseGlimmer VLM.

The parameter hierarchy intentionally matches the checkpoint's Hugging Face
implementation. Text attention is backend-native: PyTorch SDPA is used for the
ordinary backend and Transformer Engine's ``DotProductAttention`` is constructed
directly for TE BSHD/THD execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    round_robin_local_indices,
    shard_batch_aux_only,
    shard_sequence_for_cp_round_robin,
)
from nemo_automodel.components.distributed.context_parallel.utils import cp_dispatcher_suspended
from nemo_automodel.components.models.common import BackendConfig, compute_lm_head_logits
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.muse_glimmer.config import MuseGlimmerConfig
from nemo_automodel.components.models.muse_glimmer.parallelization import register_muse_glimmer_parallel_strategy
from nemo_automodel.components.models.muse_glimmer.state_dict_adapter import MuseGlimmerStateDictAdapter
from nemo_automodel.components.models.muse_glimmer.vision import MuseGlimmerVisionAdapter, MuseGlimmerVisionEncoder


class MuseGlimmerRMSNorm(nn.Module):
    """RMSNorm whose checkpoint weight stores an offset from one."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.rsqrt(output.pow(2).mean(-1, keepdim=True) + self.eps)
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)


class MuseGlimmerFinalRMSNorm(nn.Module):
    """RMSNorm whose checkpoint weight stores the actual output gain."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.pow(output.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        output = output * self.weight.float()
        return output.type_as(x)


class MuseGlimmerScalelessRMSNorm(nn.Module):
    """Parameter-free RMSNorm used for embedding and Q/K normalization."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x.float()
        output = output * torch.pow(output.pow(2).mean(-1, keepdim=True) + self.eps, -0.5)
        return output.type_as(x)


class MuseGlimmerRotaryEmbedding(nn.Module):
    """MuseGlimmer split-half rotary embedding matching the canonical HF implementation."""

    def __init__(self, dim: int, max_position_embeddings: int, theta: float = 500_000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("freqs", freqs, persistent=True)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0)
        inv_freq = self.freqs[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def apply_rotary_emb(
    x: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Apply split-half RoPE to native BSHD or THD tensors."""
    cos, sin = position_embeddings
    if x.ndim == 4:
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
    elif x.ndim == 3:
        if cos.shape[0] != 1 or sin.shape[0] != 1:
            raise ValueError("THD MuseGlimmer RoPE accepts a single flattened position stream.")
        cos = cos.squeeze(0).unsqueeze(1)
        sin = sin.squeeze(0).unsqueeze(1)
    else:
        raise ValueError(f"MuseGlimmer RoPE expects BSHD or THD input, got shape {tuple(x.shape)}.")
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos) + (rotated * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat B,Hkv,S,D keys/values to query-head count."""
    batch, num_kv_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class MuseGlimmerMLP(nn.Module):
    """Bias-free SwiGLU language MLP."""

    def __init__(self, config: MuseGlimmerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MuseGlimmerAttention(nn.Module):
    """MuseGlimmer GQA with Q/K RMSNorm, optional RoPE, output gate, SDPA, and TE."""

    def __init__(self, config: MuseGlimmerConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        self.config = config
        self.backend = backend
        self.layer_idx = layer_idx
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        query_dim = self.num_heads * self.head_dim
        kv_dim = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, query_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(query_dim, config.hidden_size, bias=False)

        self.use_output_gate = config.use_attn_output_gate
        if self.use_output_gate:
            self.output_gate_proj = nn.Linear(config.hidden_size, query_dim, bias=False)

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.qk_norm = MuseGlimmerScalelessRMSNorm(self.head_dim, config.rms_norm_eps)
            self.scale_query_by = config.scale_query_by

        self.use_rope = config.no_rope_layers[layer_idx] == 1
        self.layer_type = config.layer_types[layer_idx]
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        if backend.attn == "te":
            self.attn_module, self.attn_func = initialize_attn_module_and_func(
                attn_impl="te",
                num_attention_heads=config.num_attention_heads,
                num_qk_channels=self.head_dim,
                num_v_channels=self.head_dim,
                softmax_scale=self.scaling,
                num_gqa_groups=config.num_key_value_heads,
                attention_dropout=config.attention_dropout,
            )
        elif backend.attn not in ("sdpa", "eager"):
            raise ValueError(
                f"Native MuseGlimmer supports backend.attn='sdpa', 'eager', or 'te', got {backend.attn!r}."
            )

    def _shape_qkv(
        self,
        hidden_states: torch.Tensor,
        is_thd: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        if is_thd:
            token_count = hidden_states.shape[0]
            return (
                query.view(token_count, self.num_heads, self.head_dim),
                key.view(token_count, self.num_key_value_heads, self.head_dim),
                value.view(token_count, self.num_key_value_heads, self.head_dim),
            )
        batch, seq_len, _ = hidden_states.shape
        return (
            query.view(batch, seq_len, self.num_heads, self.head_dim),
            key.view(batch, seq_len, self.num_key_value_heads, self.head_dim),
            value.view(batch, seq_len, self.num_key_value_heads, self.head_dim),
        )

    def _te_window_size(self, is_thd: bool, max_seqlen: int | None) -> tuple[int, int]:
        if self.sliding_window is None:
            return (-1, 0)
        if is_thd and max_seqlen is not None and max_seqlen <= self.sliding_window:
            # Full causal is exactly equivalent and avoids an unnecessary local
            # window restriction for short documents packed into a long stream.
            return (-1, 0)
        return (self.sliding_window - 1, 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        is_thd = kwargs.get("qkv_format") == "thd"
        if is_thd and hidden_states.ndim != 2:
            raise ValueError(f"MuseGlimmer THD attention expects [T,H], got {tuple(hidden_states.shape)}.")
        if not is_thd and hidden_states.ndim != 3:
            raise ValueError(f"MuseGlimmer BSHD attention expects [B,S,H], got {tuple(hidden_states.shape)}.")

        query, key, value = self._shape_qkv(hidden_states, is_thd)
        if self.use_qk_norm:
            query = self.qk_norm(query) * self.scale_query_by
            key = self.qk_norm(key)
        if self.use_rope:
            query = apply_rotary_emb(query, position_embeddings)
            key = apply_rotary_emb(key, position_embeddings)

        use_te = self.backend.attn == "te" and past_key_values is None
        if is_thd:
            if not use_te:
                raise ValueError(
                    "Packed MuseGlimmer THD attention requires backend.attn='te' and does not support KV cache."
                )
            # TE's max_seqlen includes the padding folded into the final
            # document slot. MuseGlimmer's sliding-window equivalence depends on the
            # longest real document, which the model CP hook preserves.
            max_seqlen = kwargs.get("_muse_glimmer_thd_max_real_seqlen", kwargs.get("max_seqlen"))
            if isinstance(max_seqlen, torch.Tensor):
                max_seqlen = int(max_seqlen.item())
            query, key, value, te_kwargs = preprocess_args_and_kwargs_for_attn(
                query,
                key,
                value,
                None,
                "te",
                window_size=self._te_window_size(True, max_seqlen),
                **kwargs,
            )
            attn_output = self.attn_module(query, key, value, **te_kwargs)
            attn_output = postprocess_output_for_attn(attn_output, "te")
            attn_output = attn_output.view(*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        elif use_te:
            query, key, value, te_kwargs = preprocess_args_and_kwargs_for_attn(
                query,
                key,
                value,
                attention_mask,
                "te",
                window_size=self._te_window_size(False, None),
                **kwargs,
            )
            attn_output = self.attn_module(query, key, value, **te_kwargs)
            attn_output = postprocess_output_for_attn(attn_output, "te")
            attn_output = attn_output.view(*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        else:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if past_key_values is not None:
                key, value = past_key_values.update(
                    key,
                    value,
                    self.layer_idx,
                    {"cache_position": cache_position},
                )
            key = repeat_kv(key, self.num_key_value_groups)
            value = repeat_kv(value, self.num_key_value_groups)
            seq_len = query.shape[-2]
            attn_output = (
                F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=bool(kwargs.get("is_causal", attention_mask is None and seq_len > 1)),
                )
                .transpose(1, 2)
                .contiguous()
            )

        if self.use_output_gate:
            output_gate = self.output_gate_proj(hidden_states).view(
                *hidden_states.shape[:-1], self.num_heads, self.head_dim
            )
            attn_output = torch.sigmoid(output_gate) * attn_output
        return self.o_proj(attn_output.reshape(*hidden_states.shape[:-1], -1))


class MuseGlimmerDecoderLayer(GradientCheckpointingLayer):
    """One MuseGlimmer language decoder layer."""

    def __init__(self, config: MuseGlimmerConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        self.input_layernorm = MuseGlimmerRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = MuseGlimmerAttention(config, layer_idx, backend)
        self.post_attn_norm = MuseGlimmerRMSNorm(config.hidden_size, config.post_norm_eps)
        self.post_attention_layernorm = MuseGlimmerRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MuseGlimmerMLP(config)
        self.post_ffn_norm = MuseGlimmerRMSNorm(config.hidden_size, config.post_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        residual = hidden_states
        attn_output = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            attention_mask,
            past_key_values,
            cache_position,
            **kwargs,
        )
        hidden_states = residual + self.post_attn_norm(attn_output)
        residual = hidden_states
        ffn_output = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + self.post_ffn_norm(ffn_output)


class MuseGlimmerPreTrainedModel(PreTrainedModel):
    """Hugging Face-compatible base class for native MuseGlimmer."""

    config_class = MuseGlimmerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MuseGlimmerDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _supports_cache_class = True
    _can_compile_fullgraph = True


def _select_cp_positions(
    position_ids: torch.Tensor | None,
    *,
    full_seq_len: int,
    local_indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Align full or already-sharded position IDs with model-owned CP embeddings."""
    if position_ids is None:
        full = torch.arange(full_seq_len, device=device, dtype=torch.long)
        return full.unsqueeze(0).expand(batch_size, -1).index_select(1, local_indices)
    seq_dim = 2 if position_ids.ndim == 3 else 1
    if position_ids.shape[seq_dim] == local_indices.numel():
        return position_ids
    if position_ids.shape[seq_dim] == full_seq_len:
        return position_ids.index_select(seq_dim, local_indices.to(position_ids.device)).contiguous()
    raise ValueError(
        "MuseGlimmer CP position_ids must be full-sequence or local-sequence length, "
        f"got {tuple(position_ids.shape)} for full={full_seq_len}, local={local_indices.numel()}."
    )


class MuseGlimmerModel(MuseGlimmerPreTrainedModel):
    """Complete MuseGlimmer vision-language backbone."""

    def __init__(self, config: MuseGlimmerConfig, backend: BackendConfig) -> None:
        super().__init__(config)
        self.backend = backend
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_norm = (
            MuseGlimmerScalelessRMSNorm(config.hidden_size, config.rms_norm_eps)
            if config.normalize_tok_embeddings
            else None
        )

        self.has_vision = config.has_vision
        if self.has_vision:
            self.vision_encoder = MuseGlimmerVisionEncoder(config)
            self.vision_adapter = MuseGlimmerVisionAdapter(config)
            self.vision_projection = nn.Linear(config.vision_adapter_dim, config.hidden_size, bias=False)
            self.perception_emb_norm = (
                MuseGlimmerScalelessRMSNorm(config.hidden_size, config.rms_norm_eps)
                if config.normalize_tok_embeddings
                else None
            )

        self.rotary_emb = MuseGlimmerRotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        )
        self.layers = nn.ModuleList(
            [MuseGlimmerDecoderLayer(config, layer_idx, backend) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = MuseGlimmerFinalRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.gradient_checkpointing = False
        self.cp_mesh = None
        self.post_init()

    def _embed_vision(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        vision_mask: torch.Tensor | None,
        *,
        global_vision_mask: torch.Tensor | None = None,
        thd_local_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.has_vision or (pixel_values is None and pixel_values_videos is None):
            return hidden_states
        hidden_states = hidden_states.clone()

        media = (
            (pixel_values, image_grid_thw, self.config.patch_token_id),
            (pixel_values_videos, video_grid_thw, self.config.video_token_id),
        )
        for values, grid_thw, token_id in media:
            if values is None:
                continue
            if grid_thw is None:
                raise ValueError("Canonical MuseGlimmer vision inputs require the matching grid_thw tensor.")
            with cp_dispatcher_suspended(self.cp_mesh):
                vision_features = self.vision_encoder(values, grid_thw)
            vision_features = self.vision_adapter(vision_features)
            vision_features = self.vision_projection(vision_features)
            if self.perception_emb_norm is not None:
                vision_features = self.perception_emb_norm(vision_features)

            media_mask = input_ids == token_id
            if vision_mask is not None and pixel_values_videos is None:
                media_mask = vision_mask
            selected_tokens = int(media_mask.sum().item())
            if global_vision_mask is not None or thd_local_indices is not None:
                if pixel_values_videos is not None:
                    raise NotImplementedError("Packed MuseGlimmer VLM currently accepts one media type per batch.")
                if global_vision_mask is None or thd_local_indices is None:
                    raise ValueError(
                        "Packed MuseGlimmer VLM context parallelism requires both the global vision mask "
                        "and TE local-token indices."
                    )
                global_vision_mask = global_vision_mask.reshape(-1).to(device=input_ids.device, dtype=torch.bool)
                thd_local_indices = thd_local_indices.reshape(-1).to(device=input_ids.device, dtype=torch.long)
                if thd_local_indices.numel() != media_mask.numel():
                    raise ValueError(
                        "MuseGlimmer packed VLM local-token indices must align with the local input stream, "
                        f"got {thd_local_indices.numel()} indices for {media_mask.numel()} tokens."
                    )
                global_feature_count = int(global_vision_mask.sum().item())
                if global_feature_count != vision_features.shape[0]:
                    raise ValueError(
                        f"MuseGlimmer produced {vision_features.shape[0]} visual features for "
                        f"{global_feature_count} global placeholder tokens."
                    )
                feature_index_by_token = global_vision_mask.to(torch.long).cumsum(0) - 1
                local_feature_indices = feature_index_by_token.index_select(0, thd_local_indices)
                local_feature_indices = local_feature_indices[media_mask.reshape(-1)]
                if local_feature_indices.numel() and int(local_feature_indices.min().item()) < 0:
                    raise ValueError("MuseGlimmer packed VLM selected a non-visual global token as a visual feature.")
                vision_features = vision_features.index_select(0, local_feature_indices.to(vision_features.device))
            elif selected_tokens != vision_features.shape[0]:
                raise ValueError(
                    f"MuseGlimmer produced {vision_features.shape[0]} visual features for {selected_tokens} placeholder tokens."
                )
            hidden_states[media_mask] = vision_features.to(hidden_states.dtype)
        return hidden_states

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds.")
        is_thd = kwargs.get("qkv_format") == "thd"
        global_vision_mask = kwargs.pop("_muse_glimmer_global_vision_mask", None)
        thd_local_indices = kwargs.pop("_thd_local_indices", None)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else False

        if inputs_embeds is None:
            embedding_ids = input_ids
            if self.has_vision:
                multimodal_mask = (input_ids == self.config.patch_token_id) | (input_ids == self.config.video_token_id)
                embedding_ids = input_ids.clone()
                embedding_ids[multimodal_mask] = 0
            inputs_embeds = self.embed_tokens(embedding_ids)
        if self.embed_norm is not None:
            inputs_embeds = self.embed_norm(inputs_embeds)

        if input_ids is not None:
            inputs_embeds = self._embed_vision(
                input_ids,
                inputs_embeds,
                pixel_values,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
                vision_mask,
                global_vision_mask=global_vision_mask,
                thd_local_indices=thd_local_indices,
            )

        if is_thd:
            if past_key_values is not None or use_cache:
                raise ValueError("Packed MuseGlimmer THD training does not support a KV cache.")
            if inputs_embeds.ndim != 2:
                raise ValueError(f"Packed MuseGlimmer inputs must flatten to [T,H], got {tuple(inputs_embeds.shape)}.")
            if position_ids is None:
                raise ValueError("Packed MuseGlimmer THD inputs require explicit position_ids.")
        else:
            if inputs_embeds.ndim != 3:
                raise ValueError(f"MuseGlimmer BSHD inputs must have shape [B,S,H], got {tuple(inputs_embeds.shape)}.")

        full_seq_len = inputs_embeds.shape[-2]
        batch_size = 1 if is_thd else inputs_embeds.shape[0]
        if not is_thd and self.cp_mesh is not None and self.cp_mesh.size() > 1:
            if past_key_values is not None:
                raise NotImplementedError("MuseGlimmer context-parallel forward does not support a KV cache.")
            inputs_embeds, local_indices, _ = shard_sequence_for_cp_round_robin(
                self.cp_mesh,
                inputs_embeds,
                seq_dim=1,
            )
            position_ids = _select_cp_positions(
                position_ids,
                full_seq_len=full_seq_len,
                local_indices=local_indices,
                batch_size=batch_size,
                device=inputs_embeds.device,
            )
            use_cache = False
            attention_mask = None

        if not is_thd:
            seq_len = inputs_embeds.shape[1]
            if use_cache and past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            if cache_position is None:
                past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(past_len, past_len + seq_len, device=inputs_embeds.device)
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)
        elif position_ids.ndim > 1:
            position_ids = position_ids.squeeze(0)

        hidden_states = inputs_embeds.to(self.layers[0].input_layernorm.weight.dtype)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        causal_mask = None
        sliding_mask = None
        if not is_thd and (self.backend.attn != "te" or past_key_values is not None):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": hidden_states,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": None,
            }
            causal_mask = create_causal_mask(**mask_kwargs)
            sliding_mask = create_sliding_window_causal_mask(**mask_kwargs)

        all_hidden_states = () if output_hidden_states else None
        for layer_idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if is_thd:
                layer_mask = None
            elif self.backend.attn == "te" and past_key_values is None:
                layer_mask = attention_mask
            else:
                layer_mask = sliding_mask if self.config.layer_types[layer_idx] == "sliding_attention" else causal_mask
            hidden_states = layer(
                hidden_states,
                position_embeddings,
                layer_mask,
                past_key_values,
                cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        past = past_key_values if use_cache else None
        if not return_dict:
            return tuple(value for value in (hidden_states, past, all_hidden_states) if value is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class MuseGlimmerForConditionalGeneration(HFCheckpointingMixin, MuseGlimmerPreTrainedModel, GenerationMixin):
    """Native complete MuseGlimmer VLM with causal language-modeling head."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _tied_weights_keys = []
    _tp_plan = {"lm_head": "colwise_rep"}
    _keep_in_fp32_modules = ["rotary_emb"]
    supports_thd = True

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Parallel and input-layout capabilities."""

        supports_tp: bool = True
        supports_cp: bool = True
        supports_pp: bool = False
        supports_ep: bool = False
        supports_thd: bool = True

    @classmethod
    def from_config(
        cls,
        config: MuseGlimmerConfig,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> "MuseGlimmerForConditionalGeneration":
        del kwargs
        return cls(config, backend=backend)

    def __init__(self, config: MuseGlimmerConfig, backend: BackendConfig | None = None) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)
        self.config = config
        self.backend = backend or BackendConfig()
        self.model = MuseGlimmerModel(config, self.backend)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.cp_mesh = None
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = MuseGlimmerStateDictAdapter(config)
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: nn.Module) -> None:
        self.model = decoder

    def get_decoder(self) -> nn.Module:
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        vision_mask=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        if past_key_values is None or past_key_values.get_seq_length() == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_thw"] = image_grid_thw
            model_inputs["pixel_values_videos"] = pixel_values_videos
            model_inputs["video_grid_thw"] = video_grid_thw
            model_inputs["vision_mask"] = vision_mask
        return model_inputs

    def _set_te_cp_transport(self, comm_type: str) -> None:
        """Switch native TE modules between ordinary and packed CP transport."""
        for layer in self.model.layers:
            te_module = getattr(layer.self_attn, "attn_module", None)
            if te_module is None:
                continue
            cp_group = getattr(te_module, "cp_group", None)
            if cp_group is None or getattr(te_module, "cp_comm_type", None) == comm_type:
                continue
            te_module.set_context_parallel_group(
                cp_group,
                getattr(te_module, "cp_global_ranks", None),
                getattr(te_module, "cp_stream", None),
                cp_comm_type=comm_type,
            )

    def _validate_thd_documents(self, batch: dict[str, Any]) -> int:
        """Record real lengths and validate the TE p2p path when CP is active."""
        seq_lens = batch.get("seq_lens")
        if not isinstance(seq_lens, torch.Tensor):
            raise ValueError("Packed MuseGlimmer context parallelism requires per-document seq_lens.")
        real_seq_lens = seq_lens[seq_lens > 0]
        if real_seq_lens.numel() == 0:
            return 0
        max_seqlen = int(real_seq_lens.max().item())
        if self.cp_mesh is not None and max_seqlen > self.config.sliding_window:
            raise ValueError(
                "Packed MuseGlimmer TE context parallelism requires each document to fit within "
                f"the model sliding window ({self.config.sliding_window} tokens), but found "
                f"a {max_seqlen}-token document. The packed stream may still be longer; set "
                "packed_sequence.max_length to the sliding-window size."
            )
        return max_seqlen

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Select native MuseGlimmer CP preparation for BSHD or packed TE THD."""
        del num_chunks
        if batch.get("qkv_format") == "thd":
            max_real_seqlen = self._validate_thd_documents(batch)
            self._set_te_cp_transport("p2p")
            model_inputs = {"_muse_glimmer_thd_max_real_seqlen": max_real_seqlen}
            pixel_values = batch.get("pixel_values")
            if pixel_values is None:
                return model_inputs
            input_ids = batch.get("input_ids")
            if input_ids is None:
                raise ValueError("Packed MuseGlimmer VLM context parallelism requires input_ids.")
            vision_mask = batch.get("vision_mask")
            if vision_mask is None:
                vision_mask = (input_ids == self.config.patch_token_id) | (input_ids == self.config.video_token_id)
            model_inputs["_muse_glimmer_global_vision_mask"] = vision_mask
            return model_inputs
        self._set_te_cp_transport("all_gather")
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=shard_batch_aux_only,
                local_token_global_indices=round_robin_local_indices,
            )
        }

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        vision_mask: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        del output_attentions
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else not self.training
        is_thd = kwargs.get("qkv_format") == "thd"

        if is_thd:
            kwargs.pop("padding_mask", None)
            if input_ids is not None and input_ids.ndim > 1:
                input_ids = input_ids.squeeze(0)
            if inputs_embeds is not None and inputs_embeds.ndim > 2:
                inputs_embeds = inputs_embeds.squeeze(0)
            if position_ids is not None and position_ids.ndim > 1:
                position_ids = position_ids.squeeze(0)
            if labels is not None and labels.ndim > 1:
                labels = labels.squeeze(0)
            for key, value in tuple(kwargs.items()):
                if not isinstance(value, torch.Tensor):
                    continue
                if key == "max_seqlen":
                    kwargs[key] = int(value.item())
                    continue
                if value.ndim > 1:
                    value = value.squeeze(0)
                if key in ("cu_seqlens", "cu_seqlens_padded"):
                    value = value[value != -1000].contiguous()
                kwargs[key] = value
            attention_mask = None
            use_cache = False

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            vision_mask=vision_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        projected = compute_lm_head_logits(
            self.lm_head,
            hidden_states,
            logits_to_keep,
            is_thd=is_thd,
        )
        logits = projected.logits
        output_multiplier = float(self.config.output_multiplier)
        soft_cap = self.config.output_soft_cap_temp
        logits = logits * output_multiplier
        if soft_cap is not None:
            logits = logits / soft_cap
            logits = torch.tanh(logits)
            logits = logits * soft_cap

        loss = None
        if labels is not None:
            if is_thd and labels.ndim == 1:
                labels = labels.unsqueeze(0)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )

        # Fused/checkpointed linear CE consumes the final hidden states directly.
        # Scaling here makes F.linear(hidden * multiplier, lm_head.weight) match
        # MuseGlimmer's native output multiplier before optional logit soft-capping.
        loss_hidden_states = hidden_states * output_multiplier
        if is_thd and loss_hidden_states.ndim == 2:
            loss_hidden_states = loss_hidden_states.unsqueeze(0)
        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=loss_hidden_states,
            attentions=None,
        )
        return output if return_dict else output.to_tuple()


register_muse_glimmer_parallel_strategy()
ModelClass = MuseGlimmerForConditionalGeneration
