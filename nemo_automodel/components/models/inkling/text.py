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

"""Native Inkling text backbone, attention, masks, and decoding cache."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .configuration import InklingTextConfig
from .layers import InklingDenseMLP, InklingMoE, InklingShortConvolution


class _InklingCacheLayer:
    """Dynamic key/value and short-convolution state for one decoder layer."""

    def __init__(self, *, sliding_window: int | None, number_of_conv_states: int) -> None:
        self.sliding_window = sliding_window
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
        self.cumulative_length = 0
        self.conv_states: dict[int, torch.Tensor | None] = dict.fromkeys(range(number_of_conv_states))
        self._has_previous_conv_state: dict[int, bool] = dict.fromkeys(range(number_of_conv_states), False)
        self.record_past = False

    def get_seq_length(self) -> int:
        """Return the number of tokens observed by this layer."""
        if self.sliding_window is not None:
            return self.cumulative_length
        return 0 if self.keys is None else self.keys.shape[-2]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """Return key/value length and absolute offset before appending a query."""
        if self.sliding_window is None:
            return self.get_seq_length() + query_length, 0
        is_full = self.cumulative_length >= self.sliding_window
        key_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        key_length = self.sliding_window - 1 + query_length if is_full else self.cumulative_length + query_length
        return key_length, key_offset

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Append key/value states and return the states visible to this query.

        Args:
            key_states: Tensor of shape ``[batch, key_value_heads, sequence, head_dim]``.
            value_states: Tensor of shape ``[batch, key_value_heads, sequence, head_dim]``.

        Returns:
            A pair of tensors with shape ``[batch, key_value_heads, visible_sequence, head_dim]``.
        """
        if self.keys is None:
            self.keys = key_states[:, :, :0, :]
            self.values = value_states[:, :, :0, :]
        full_keys = torch.cat((self.keys, key_states), dim=-2)
        full_values = torch.cat((self.values, value_states), dim=-2)
        if self.sliding_window is None:
            self.keys = full_keys
            self.values = full_values
        else:
            self.cumulative_length += key_states.shape[-2]
            retained = self.sliding_window - 1
            self.keys = full_keys[:, :, -retained:, :]
            self.values = full_values[:, :, -retained:, :]
        return full_keys, full_values

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        state_idx: int,
        conv_kernel_size: int,
    ) -> torch.Tensor:
        """Append short-convolution inputs and update the fixed-size cache.

        Args:
            conv_states: Tensor of shape ``[batch, hidden, sequence]``.
            state_idx: Index of one of the four convolution states owned by the layer.
            conv_kernel_size: Number of trailing tokens retained in the cache.

        Returns:
            Tensor of shape ``[batch, hidden, cached_sequence + sequence]`` used by
            the current convolution.
        """
        cached = self.conv_states[state_idx]
        if cached is None:
            cached = torch.zeros(
                *conv_states.shape[:-1],
                conv_kernel_size,
                dtype=conv_states.dtype,
                device=conv_states.device,
            )
            self.conv_states[state_idx] = cached

        if not self._has_previous_conv_state[state_idx]:
            full_states = conv_states
            self._has_previous_conv_state[state_idx] = True
            if full_states.shape[-1] < conv_kernel_size:
                full_states = F.pad(full_states, (conv_kernel_size - full_states.shape[-1], 0))
        else:
            full_states = torch.cat((cached, conv_states), dim=-1)

        cached.copy_(full_states[..., -conv_kernel_size:])
        return full_states

    def has_previous_state(self, state_idx: int) -> bool:
        """Return whether a convolution state has seen at least one token."""
        return self._has_previous_conv_state[state_idx]

    def reorder(self, beam_indices: torch.Tensor) -> None:
        """Reorder cached batch entries during beam search.

        Args:
            beam_indices: Long tensor of shape ``[new_batch]`` indexing the old batch axis.
        """
        if self.keys is not None:
            self.keys = self.keys.index_select(0, beam_indices.to(self.keys.device))
            self.values = self.values.index_select(0, beam_indices.to(self.values.device))
        for state_idx, state in self.conv_states.items():
            if state is not None:
                self.conv_states[state_idx] = state.index_select(0, beam_indices.to(state.device))


class InklingDynamicCache:
    """Model-owned dynamic cache for Inkling attention and short convolutions."""

    def __init__(self, config: InklingTextConfig) -> None:
        self.layers = [
            _InklingCacheLayer(
                sliding_window=config.sliding_window_size if layer_type == "hybrid_sliding" else None,
                number_of_conv_states=config.number_of_conv_states,
            )
            for layer_type in config.layer_types
        ]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the number of tokens observed at a decoder layer."""
        return self.layers[layer_idx].get_seq_length()

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """Return the absolute starting position of the next query."""
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        """Return the key/value length and absolute offset for a decoder layer."""
        return self.layers[layer_idx].get_mask_sizes(query_length)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Append key/value states at one decoder layer.

        Args:
            key_states: Tensor of shape ``[batch, key_value_heads, sequence, head_dim]``.
            value_states: Tensor of shape ``[batch, key_value_heads, sequence, head_dim]``.
            layer_idx: Decoder-layer index.

        Returns:
            A pair of tensors with shape ``[batch, key_value_heads, visible_sequence, head_dim]``.
        """
        return self.layers[layer_idx].update(key_states, value_states)

    def update_conv_state(
        self,
        conv_states: torch.Tensor,
        layer_idx: int,
        state_idx: int,
        conv_kernel_size: int,
    ) -> torch.Tensor:
        """Append and cache inputs for one short convolution.

        Args:
            conv_states: Tensor of shape ``[batch, hidden, sequence]``.
            layer_idx: Decoder-layer index.
            state_idx: Convolution index inside that layer.
            conv_kernel_size: Number of trailing tokens retained.

        Returns:
            Tensor of shape ``[batch, hidden, cached_sequence + sequence]``.
        """
        return self.layers[layer_idx].update_conv_state(conv_states, state_idx, conv_kernel_size)

    def has_previous_state(self, layer_idx: int, state_idx: int) -> bool:
        """Return whether one short-convolution cache has been populated."""
        return self.layers[layer_idx].has_previous_state(state_idx)

    def reorder_cache(self, beam_indices: torch.Tensor) -> None:
        """Reorder every layer cache during beam search.

        Args:
            beam_indices: Long tensor of shape ``[new_batch]`` indexing the old batch axis.
        """
        for layer in self.layers:
            layer.reorder(beam_indices)


class InklingRMSNorm(nn.Module):
    """RMS normalization with fp32 variance accumulation."""

    def __init__(self, hidden_size: int, eps: float, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize the final hidden axis.

        Args:
            hidden_states: Tensor of shape ``[..., hidden]`` with arbitrary leading dimensions.

        Returns:
            Tensor with the same shape and dtype as ``hidden_states``.
        """
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized.to(input_dtype)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize the learned scale to one."""
        nn.init.ones_(self.weight)


class InklingRelativeLogits(nn.Module):
    """Project token-conditioned relative-position profiles into attention bias."""

    def __init__(self, d_rel: int, rel_extent: int, *, dtype: torch.dtype) -> None:
        super().__init__()
        self.rel_extent = rel_extent
        self.proj = nn.Parameter(torch.empty(d_rel, rel_extent, dtype=dtype))

    def forward(
        self,
        relative_states: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Materialize the relative-position bias.

        Args:
            relative_states: Tensor of shape ``[batch, query, heads, relative_dim]``.
            query_positions: Long tensor of shape ``[query]`` containing absolute positions.
            key_positions: Long tensor of shape ``[key]`` containing absolute positions.

        Returns:
            Tensor of shape ``[batch, heads, query, key]``.
        """
        relative_logits = (relative_states @ self.proj).transpose(1, 2)
        distance = (query_positions[:, None] - key_positions[None, :])[None, None, :, :]
        gather_index = distance.clamp(0, self.rel_extent - 1).expand(*relative_logits.shape[:2], -1, -1)
        position_bias = relative_logits.gather(-1, gather_index)
        return position_bias.masked_fill((distance < 0) | (distance >= self.rel_extent), 0.0)

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        """Initialize the relative-position profile bank."""
        nn.init.normal_(self.proj, mean=0.0, std=init_std)


def _repeat_key_value(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Repeat grouped key/value heads to match query heads.

    Args:
        hidden_states: Tensor of shape ``[batch, key_value_heads, sequence, head_dim]``.
        repeats: Number of query heads sharing each key/value head.

    Returns:
        Tensor of shape ``[batch, key_value_heads * repeats, sequence, head_dim]``.
    """
    if repeats == 1:
        return hidden_states
    batch, key_value_heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(batch, key_value_heads, repeats, sequence, head_dim)
    return expanded.reshape(batch, key_value_heads * repeats, sequence, head_dim)


def _build_attention_mask(
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values: InklingDynamicCache | None,
    layer_idx: int,
    sliding_window: int | None,
) -> torch.Tensor:
    """Build an additive causal mask for Inkling eager attention.

    Args:
        inputs_embeds: Tensor of shape ``[batch, query, hidden]``.
        attention_mask: Optional tensor of shape ``[batch, total_sequence]`` where nonzero
            entries are valid keys, or a prepared tensor of shape ``[batch, 1, query, key]``.
        past_key_values: Optional model-owned decoding cache.
        layer_idx: Decoder-layer index used to size a hybrid cache.
        sliding_window: Optional number of visible tokens for local attention.

    Returns:
        Additive tensor of shape ``[batch, 1, query, key]`` in ``inputs_embeds.dtype``.
    """
    if attention_mask is not None and attention_mask.ndim == 4:
        return attention_mask.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)

    batch, query_length = inputs_embeds.shape[:2]
    if past_key_values is None:
        key_length = attention_mask.shape[-1] if attention_mask is not None else query_length
        query_offset = 0
        key_offset = 0
    else:
        key_length, key_offset = past_key_values.get_mask_sizes(query_length, layer_idx)
        query_offset = past_key_values.get_query_offset(layer_idx)

    query_positions = torch.arange(query_length, device=inputs_embeds.device) + query_offset
    key_positions = torch.arange(key_length, device=inputs_embeds.device) + key_offset
    allowed = key_positions[None, :] <= query_positions[:, None]
    if sliding_window is not None:
        allowed &= key_positions[None, :] > query_positions[:, None] - sliding_window

    allowed = allowed[None, None, :, :].expand(batch, 1, -1, -1)
    if attention_mask is not None:
        key_valid = attention_mask.to(device=inputs_embeds.device, dtype=torch.bool)
        key_valid = key_valid[:, key_offset : key_offset + key_length]
        allowed = allowed & key_valid[:, None, None, :]

    additive_mask = torch.zeros(allowed.shape, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
    return additive_mask.masked_fill(~allowed, torch.finfo(inputs_embeds.dtype).min)


def _build_conv_mask(attention_mask: torch.Tensor | None, query_length: int) -> torch.Tensor | None:
    """Extract a local two-dimensional padding mask for short convolutions.

    Args:
        attention_mask: Optional tensor of shape ``[batch, total_sequence]``.
        query_length: Length of the current local sequence.

    Returns:
        Optional boolean tensor of shape ``[batch, query_length]`` where ``True``
        marks a valid token.
    """
    if attention_mask is None or attention_mask.ndim != 2 or torch.all(attention_mask == 1):
        return None
    return attention_mask[:, -query_length:].to(dtype=torch.bool).contiguous()


class InklingAttention(nn.Module):
    """Inkling grouped-query attention with learned relative logits."""

    def __init__(self, config: InklingTextConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.is_sliding = config.layer_types[layer_idx] == "hybrid_sliding"
        self.head_dim = config.swa_head_dim if self.is_sliding else config.head_dim
        self.num_heads = config.swa_num_attention_heads if self.is_sliding else config.num_attention_heads
        self.num_key_value_heads = config.swa_num_key_value_heads if self.is_sliding else config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.sliding_window = config.sliding_window_size if self.is_sliding else None
        self.rel_extent = config.sliding_window_size if self.is_sliding else config.rel_extent
        self.scaling = 1.0 / self.head_dim
        self.attention_dropout = config.attention_dropout
        configured_attention = getattr(config, "_attn_implementation", None)
        self.attention_backend = configured_attention if configured_attention in {"eager", "sdpa"} else backend.attn
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False, dtype=model_dtype)
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=model_dtype
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False, dtype=model_dtype
        )
        self.r_proj = nn.Linear(config.hidden_size, self.num_heads * config.d_rel, bias=False, dtype=model_dtype)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False, dtype=model_dtype)
        self.k_sconv = InklingShortConvolution(
            self.num_key_value_heads * self.head_dim, config.conv_kernel_size, layer_idx, conv_idx=0
        )
        self.v_sconv = InklingShortConvolution(
            self.num_key_value_heads * self.head_dim, config.conv_kernel_size, layer_idx, conv_idx=1
        )
        self.q_norm = InklingRMSNorm(self.head_dim, config.rms_norm_eps, dtype=model_dtype)
        self.k_norm = InklingRMSNorm(self.head_dim, config.rms_norm_eps, dtype=model_dtype)
        self.rel_logits_proj = InklingRelativeLogits(config.d_rel, self.rel_extent, dtype=model_dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        conv_mask: torch.Tensor | None = None,
        past_key_values: InklingDynamicCache | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, hidden]``.
            attention_mask: Additive tensor of shape ``[batch, 1, query, key]``.
            conv_mask: Optional boolean tensor of shape ``[batch, sequence]`` where
                ``True`` marks a valid token.
            past_key_values: Optional model-owned decoding cache.
            **kwargs: Backend-compatible attention arguments; currently ignored by eager attention.

        Returns:
            A pair containing the output tensor of shape ``[batch, sequence, hidden]``
            and optional attention probabilities of shape ``[batch, heads, query, key]``.
            The SDPA backend returns ``None`` for probabilities.
        """
        input_shape = hidden_states.shape[:-1]
        query_shape = (*input_shape, self.num_heads, self.head_dim)
        key_value_shape = (*input_shape, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
        key_states = self.k_sconv(
            self.k_proj(hidden_states), past_key_values=past_key_values, conv_mask=conv_mask, **kwargs
        )
        value_states = self.v_sconv(
            self.v_proj(hidden_states), past_key_values=past_key_values, conv_mask=conv_mask, **kwargs
        )
        key_states = self.k_norm(key_states.view(key_value_shape)).transpose(1, 2)
        value_states = value_states.view(key_value_shape).transpose(1, 2)

        # The short-convolution parameters intentionally remain fp32 under FSDP.
        # Some FSDP policies therefore return their activations in fp32 even
        # though the attention projections run in bf16.  SDPA requires q/k/v to
        # share a dtype, so restore the projection compute dtype before caching
        # or dispatching to either attention backend.
        attention_dtype = query_states.dtype
        key_states = key_states.to(dtype=attention_dtype)
        value_states = value_states.to(dtype=attention_dtype)

        query_length = query_states.shape[2]
        if past_key_values is None:
            key_length = key_states.shape[2]
            query_offset = key_offset = 0
        else:
            key_length, key_offset = past_key_values.get_mask_sizes(query_length, self.layer_idx)
            query_offset = past_key_values.get_query_offset(self.layer_idx)
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_positions = torch.arange(key_length, device=hidden_states.device) + key_offset
        query_positions = torch.arange(query_length, device=hidden_states.device) + query_offset
        relative_states = self.r_proj(hidden_states).view(*input_shape, self.num_heads, -1)
        position_bias = self.rel_logits_proj(relative_states, query_positions, key_positions)

        if not self.is_sliding and self.config.log_scaling_n_floor is not None:
            effective_n = (query_positions + 1).float()
            tau = 1.0 + self.config.log_scaling_alpha * torch.log(
                (effective_n / self.config.log_scaling_n_floor).clamp(min=1.0)
            )
            tau = tau.view(1, 1, -1, 1)
            query_states = (query_states.float() * tau).to(query_states.dtype)
            position_bias = (position_bias.float() * tau).to(position_bias.dtype)

        key_states = _repeat_key_value(key_states, self.num_key_value_groups)
        value_states = _repeat_key_value(value_states, self.num_key_value_groups)
        combined_mask = position_bias + attention_mask
        if self.attention_backend == "sdpa":
            attention_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=combined_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.scaling,
            )
            attention_probs = None
        else:
            attention_scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
            attention_scores = attention_scores + combined_mask
            attention_probs = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attention_probs = F.dropout(attention_probs, p=self.attention_dropout, training=self.training)
            attention_output = torch.matmul(attention_probs, value_states)
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = self.o_proj(attention_output.reshape(*input_shape, -1))
        return attention_output, attention_probs

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        """Initialize projections, norms, relative logits, and convolution weights."""
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.r_proj, self.o_proj):
            nn.init.normal_(projection.weight, mean=0.0, std=init_std)
        self.q_norm.init_weights()
        self.k_norm.init_weights()
        self.rel_logits_proj.init_weights(init_std)
        self.k_sconv.init_weights(init_std)
        self.v_sconv.init_weights(init_std)


class InklingDecoderLayer(nn.Module):
    """One native Inkling decoder layer."""

    def __init__(
        self,
        config: InklingTextConfig,
        layer_idx: int,
        backend: BackendConfig,
        moe_config: MoEConfig,
    ) -> None:
        super().__init__()
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.self_attn = InklingAttention(config, layer_idx, backend)
        self.mlp = (
            InklingMoE(config, backend, moe_config=moe_config)
            if config.mlp_layer_types[layer_idx] == "sparse"
            else InklingDenseMLP(config)
        )
        self.input_layernorm = InklingRMSNorm(config.hidden_size, config.rms_norm_eps, dtype=model_dtype)
        self.post_attention_layernorm = InklingRMSNorm(config.hidden_size, config.rms_norm_eps, dtype=model_dtype)
        self.layer_type = config.layer_types[layer_idx]
        self.attention_type = "full_attention" if self.layer_type == "hybrid" else "sliding_attention"
        self.attn_sconv = InklingShortConvolution(config.hidden_size, config.conv_kernel_size, layer_idx, conv_idx=2)
        self.mlp_sconv = InklingShortConvolution(config.hidden_size, config.conv_kernel_size, layer_idx, conv_idx=3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        conv_mask: torch.Tensor | None = None,
        past_key_values: InklingDynamicCache | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply attention and feed-forward residual blocks.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, hidden]``.
            attention_mask: Additive tensor of shape ``[batch, 1, query, key]``.
            conv_mask: Optional boolean tensor of shape ``[batch, sequence]`` where
                ``True`` marks a valid token.
            past_key_values: Optional model-owned decoding cache.
            **kwargs: Additional eager-attention arguments.

        Returns:
            Tensor of shape ``[batch, sequence, hidden]``.
        """
        residual = hidden_states
        normalized_hidden_states = self.input_layernorm(hidden_states).to(dtype=residual.dtype)
        hidden_states, _ = self.self_attn(
            normalized_hidden_states,
            attention_mask=attention_mask,
            conv_mask=conv_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
        hidden_states = self.attn_sconv(hidden_states, past_key_values=past_key_values, conv_mask=conv_mask, **kwargs)
        hidden_states = hidden_states.to(dtype=residual.dtype)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states).to(dtype=residual.dtype)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_sconv(hidden_states, past_key_values=past_key_values, conv_mask=conv_mask, **kwargs)
        hidden_states = hidden_states.to(dtype=residual.dtype)
        return residual + hidden_states

    @torch.no_grad()
    def init_weights(self, init_std: float, buffer_device: torch.device) -> None:
        """Initialize all parameters owned by the decoder layer."""
        self.self_attn.init_weights(init_std)
        self.input_layernorm.init_weights()
        self.post_attention_layernorm.init_weights()
        self.attn_sconv.init_weights(init_std)
        self.mlp_sconv.init_weights(init_std)
        self.mlp.init_weights(buffer_device, init_std) if isinstance(self.mlp, InklingMoE) else self.mlp.init_weights(
            init_std
        )


class InklingTextModel(nn.Module):
    """Native Inkling text backbone with pipeline-stage support."""

    def __init__(
        self,
        config: InklingTextConfig,
        backend: BackendConfig,
        moe_config: MoEConfig,
    ) -> None:
        super().__init__()
        self.config = config
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            dtype=model_dtype,
        )
        self.layers = nn.ModuleList(
            [
                InklingDecoderLayer(config, layer_idx, backend, moe_config)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = InklingRMSNorm(config.hidden_size, config.rms_norm_eps, dtype=model_dtype)
        self.embed_norm = InklingRMSNorm(config.hidden_size, config.rms_norm_eps, dtype=model_dtype)

    def get_input_embeddings(self) -> nn.Module:
        """Return the token-embedding module."""
        return self.embed_tokens

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        """Replace the token-embedding module."""
        self.embed_tokens = embeddings

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: InklingDynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        """Run the text backbone.

        Args:
            input_ids: Optional tensor of shape ``[batch, sequence]``. Pipeline stages
                without embeddings may instead receive floating hidden states of shape
                ``[batch, sequence, hidden]`` through this argument.
            attention_mask: Optional padding tensor of shape ``[batch, total_sequence]``
                or a mapping from attention type to additive masks of shape
                ``[batch, 1, query, key]``.
            position_ids: Optional long tensor of shape ``[batch, sequence]``.
            past_key_values: Optional model-owned decoding cache.
            inputs_embeds: Optional tensor of shape ``[batch, sequence, hidden]``.
            use_cache: Whether to allocate and return a decoding cache.
            **kwargs: Additional eager-attention arguments.

        Returns:
            A model output whose ``last_hidden_state`` has shape ``[batch, sequence, hidden]``.
        """
        if inputs_embeds is None:
            if self.embed_tokens is None:
                if input_ids is None or not input_ids.dtype.is_floating_point:
                    raise ValueError("Pipeline stages without embeddings require hidden states as input")
                inputs_embeds = input_ids
                input_ids = None
            else:
                if input_ids is None:
                    raise ValueError("You must provide either input_ids or inputs_embeds")
                inputs_embeds = self.embed_norm(self.embed_tokens(input_ids))
        elif input_ids is not None:
            raise ValueError("You must provide exactly one of input_ids or inputs_embeds")

        use_cache = getattr(self.config, "use_cache", False) if use_cache is None else use_cache
        if use_cache and past_key_values is None:
            past_key_values = InklingDynamicCache(self.config)

        if position_ids is None:
            past_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_tokens
            position_ids = position_ids.unsqueeze(0)

        if isinstance(attention_mask, dict):
            masks = attention_mask
            conv_mask = masks.get("linear_attention")
        else:
            conv_mask = _build_conv_mask(attention_mask, inputs_embeds.shape[1])
            full_layer_idx = next(
                (idx for idx, layer_type in enumerate(self.config.layer_types) if layer_type == "hybrid"),
                0,
            )
            sliding_layer_idx = next(
                (idx for idx, layer_type in enumerate(self.config.layer_types) if layer_type == "hybrid_sliding"),
                0,
            )
            masks = {
                "full_attention": _build_attention_mask(
                    inputs_embeds,
                    attention_mask,
                    past_key_values,
                    full_layer_idx,
                    None,
                ),
                "sliding_attention": _build_attention_mask(
                    inputs_embeds,
                    attention_mask,
                    past_key_values,
                    sliding_layer_idx,
                    self.config.sliding_window_size,
                ),
                "linear_attention": conv_mask,
            }

        hidden_states = inputs_embeds
        layers = self.layers.values() if isinstance(self.layers, nn.ModuleDict) else self.layers
        for decoder_layer in layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=masks[decoder_layer.attention_type],
                conv_mask=conv_mask,
                past_key_values=past_key_values,
                **kwargs,
            )

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device) -> None:
        """Initialize the text backbone for checkpoint-free construction."""
        init_std = self.config.initializer_range
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=init_std)
        if self.embed_tokens.padding_idx is not None:
            self.embed_tokens.weight[self.embed_tokens.padding_idx].zero_()
        self.embed_norm.init_weights()
        self.norm.init_weights()
        layers = self.layers.values() if isinstance(self.layers, nn.ModuleDict) else self.layers
        for layer in layers:
            layer.init_weights(init_std, buffer_device)


__all__ = ["InklingDynamicCache", "InklingRMSNorm", "InklingTextModel"]
