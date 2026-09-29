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

"""Native feed-forward and short-convolution layers for Inkling.

The equations follow the published Transformers 5.14 Inkling implementation,
while the routed experts use NeMo AutoModel's grouped-GEMM and expert-parallel
implementations.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _mask_padding_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    """Zero padded tokens before a short convolution.

    Args:
        hidden_states: Tensor of shape ``[batch, sequence, hidden]``.
        attention_mask: Optional boolean tensor of shape ``[batch, sequence]`` where
            ``True`` marks a valid token.

    Returns:
        Tensor of shape ``[batch, sequence, hidden]``. The output aliases the input
        when no masking is required.
    """
    if attention_mask is None or attention_mask.shape[1] <= 1 or attention_mask.shape[0] <= 1:
        return hidden_states
    return (hidden_states * attention_mask[:, :, None]).to(hidden_states.dtype)


def _causal_conv1d_update(
    hidden_states: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Update cached depthwise-convolution state for one decoding step.

    Args:
        hidden_states: Tensor of shape ``[batch, hidden, sequence]`` containing new tokens.
        conv_state: Mutable cache tensor of shape ``[batch, hidden, kernel]``.
        weight: Depthwise weights of shape ``[hidden, 1, kernel]``.

    Returns:
        Tensor of shape ``[batch, hidden, sequence]``. ``conv_state`` is updated in place.
    """
    sequence = hidden_states.shape[-1]
    state_length = conv_state.shape[-1]
    full_states = torch.cat((conv_state, hidden_states), dim=-1).to(weight.dtype)
    conv_state.copy_(full_states[:, :, -state_length:])
    output = F.conv1d(full_states, weight, padding=0, groups=weight.shape[0])
    return output[:, :, -sequence:].to(hidden_states.dtype)


def _causal_conv1d(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply a causal depthwise convolution to a full sequence.

    Args:
        hidden_states: Tensor of shape ``[batch, hidden, sequence]``.
        weight: Depthwise weights of shape ``[hidden, 1, kernel]``.

    Returns:
        Tensor of shape ``[batch, hidden, sequence]``.
    """
    sequence = hidden_states.shape[-1]
    output = F.conv1d(
        hidden_states.to(weight.dtype),
        weight=weight,
        padding=weight.shape[-1] - 1,
        groups=weight.shape[0],
    )
    return output[:, :, :sequence].to(hidden_states.dtype)


class _InklingShortConvolutionFP32(nn.Module):
    """Run one short convolution inside a dtype-uniform fp32 FSDP unit."""

    def __init__(self, hidden_size: int, conv_kernel_size: int, layer_idx: int, conv_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.conv_idx = conv_idx
        self.conv_kernel_size = conv_kernel_size
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, conv_kernel_size, dtype=torch.float32))

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        conv_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply the causal short convolution with a residual connection.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, hidden]``. Internally
                transposed to ``[batch, hidden, sequence]`` for the depthwise
                ``conv1d`` and transposed back before the residual add.
            past_key_values: Optional cache holding the per-layer conv state; updated
                in place during incremental decoding.
            conv_mask: Optional boolean padding mask of shape ``[batch, sequence]``
                that zeroes padded positions before the convolution.

        Returns:
            torch.Tensor: Tensor of shape ``[batch, sequence, hidden]`` (input plus
            convolution output).
        """
        residual = hidden_states
        hidden_states = _mask_padding_states(hidden_states, conv_mask)
        seq_len = hidden_states.shape[1]
        hidden_states = hidden_states.transpose(1, 2)
        weight = self.weight

        use_precomputed_states = past_key_values is not None and past_key_values.has_previous_state(
            self.layer_idx, self.conv_idx
        )
        if use_precomputed_states and seq_len == 1 and not past_key_values.layers[self.layer_idx].record_past:
            conv_state = past_key_values.layers[self.layer_idx].conv_states[self.conv_idx]
            hidden_states = _causal_conv1d_update(hidden_states, conv_state, weight)
        else:
            if past_key_values is not None:
                hidden_states = past_key_values.update_conv_state(
                    hidden_states,
                    self.layer_idx,
                    state_idx=self.conv_idx,
                    conv_kernel_size=self.conv_kernel_size,
                )
            hidden_states = _causal_conv1d(hidden_states, weight)
            # Initial cached prefills shorter than the kernel are left-padded by
            # the cache, while later multi-token updates prepend cached state.
            # In both cases return only the tokens from this invocation.
            if past_key_values is not None:
                hidden_states = hidden_states[:, :, -seq_len:]

        return hidden_states.transpose(1, 2) + residual


class InklingShortConvolution(nn.Module):
    """Keep short-convolution weights in a model-owned fp32 holder."""

    def __init__(self, hidden_size: int, conv_kernel_size: int, layer_idx: int, conv_idx: int) -> None:
        super().__init__()
        self._fp32_params = _InklingShortConvolutionFP32(hidden_size, conv_kernel_size, layer_idx, conv_idx)

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        """Initialize the fp32 depthwise-convolution weights."""
        nn.init.normal_(self._fp32_params.weight, mean=0.0, std=init_std)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: Any | None = None,
        conv_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run the short convolution in fp32 and cast the result back.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, hidden]``.
            past_key_values: Optional cache holding the per-layer conv state.
            conv_mask: Optional boolean padding mask of shape ``[batch, sequence]``.

        Returns:
            torch.Tensor: Tensor of shape ``[batch, sequence, hidden]`` in the input
            dtype.
        """
        input_dtype = hidden_states.dtype
        output = self._fp32_params(
            hidden_states.float(),
            past_key_values=past_key_values,
            conv_mask=conv_mask,
            **kwargs,
        )
        return output.to(dtype=input_dtype)


def build_inkling_moe_config(text_config, backend: BackendConfig) -> MoEConfig:
    """Build the routed-expert :class:`MoEConfig` from an ``InklingTextConfig``.

    Only the fields consumed by :class:`GroupedExperts` matter here (routing and
    shared experts are handled by :class:`InklingGate` / :class:`InklingSharedExperts`),
    so ``n_shared_experts`` is set to 0 and the gate-specific fields are left at
    neutral defaults.

    Args:
        text_config: Local ``InklingTextConfig``.
        backend: Backend configuration selecting the expert compute/dispatch kernels.

    Returns:
        MoEConfig: Configuration for the routed grouped experts.
    """
    model_dtype = get_dtype(getattr(text_config, "torch_dtype", None), torch.bfloat16)
    return MoEConfig(
        dim=text_config.hidden_size,
        inter_dim=text_config.intermediate_size,
        moe_inter_dim=text_config.moe_intermediate_size,
        n_routed_experts=text_config.n_routed_experts,
        n_shared_experts=0,
        n_activated_experts=text_config.num_experts_per_tok,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=text_config.route_scale,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        dtype=model_dtype,
    )


class InklingCorrectionBias(nn.Module):
    """Own and apply Inkling's trained fp32 router correction bias."""

    def __init__(self, num_experts: int) -> None:
        super().__init__()
        self.e_score_correction_bias = nn.Parameter(torch.empty(num_experts, dtype=torch.float32))

    def forward(self, routed_scores: torch.Tensor) -> torch.Tensor:
        """Add the correction bias in the routing score dtype."""
        return routed_scores + self.e_score_correction_bias.to(routed_scores.dtype)


class InklingGate(nn.Module):
    """Top-k router matching ``transformers`` ``InklingTopkRouter``.

    The gate weight bank additionally holds ``n_shared_experts`` rows whose logits
    participate in the softmax normalization; the resulting shared weights
    (``gammas``) scale the always-on shared experts. Selection uses sigmoid scores
    plus a per-expert correction bias, while the returned weights come from a
    log-sigmoid softmax over the selected routed logits concatenated with the
    shared logits, scaled by ``route_scale * global_scale``.
    """

    def __init__(self, config, gate_precision: torch.dtype | None = torch.float32) -> None:
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.n_total_experts = self.num_experts + self.n_shared_experts
        self.hidden_dim = config.hidden_size
        self.route_scale = config.route_scale
        self.top_k = config.num_experts_per_tok
        self.gate_precision = gate_precision

        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.weight = nn.Parameter(torch.empty(self.n_total_experts, config.hidden_size, dtype=model_dtype))
        self.global_scale = nn.Parameter(torch.ones(1, dtype=model_dtype))
        # Keep the trained correction bias in a callable fp32 FSDP unit. Calling
        # the holder is required for its unshard/reshard hooks to run.
        self._fp32_params = InklingCorrectionBias(self.num_experts)

    @property
    def e_score_correction_bias(self) -> nn.Parameter:
        """Expose the correction bias under the Transformers-compatible name."""
        return self._fp32_params.e_score_correction_bias

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Route tokens to experts.

        Args:
            hidden_states: Input tokens of shape ``[num_tokens, hidden_dim]``.

        Returns:
            tuple: ``(topk_weights, topk_indices, shared_gammas)`` where
            ``topk_weights`` and ``topk_indices`` have shape ``[num_tokens, top_k]``
            and ``shared_gammas`` has shape ``[num_tokens, n_shared_experts]``.
        """
        compute_dtype = self.gate_precision
        flat = hidden_states.reshape(-1, self.hidden_dim)
        weight = self.weight
        if compute_dtype is not None:
            flat = flat.to(compute_dtype)
            weight = weight.to(compute_dtype)
        router_logits = F.linear(flat, weight)

        scores = router_logits.sigmoid()
        routed_scores = scores[..., : -self.n_shared_experts]
        scores_for_choice = self._fp32_params(routed_scores)
        topk_indices = torch.topk(scores_for_choice, self.top_k, dim=-1, sorted=False)[1]

        routed_logits = router_logits[..., : -self.n_shared_experts]
        shared_logits = router_logits[..., -self.n_shared_experts :]
        topk_logits = torch.cat([routed_logits.gather(-1, topk_indices), shared_logits], dim=-1)
        topk_log_probs = F.logsigmoid(topk_logits)
        topk_weights = torch.exp(topk_log_probs - torch.logsumexp(topk_log_probs, dim=-1, keepdim=True))

        topk_weights = topk_weights * self.route_scale * self.global_scale.to(topk_weights.dtype)

        shared_gammas = topk_weights[..., -self.n_shared_experts :].contiguous()
        topk_weights = topk_weights[..., : self.top_k].contiguous()
        return topk_weights, topk_indices, shared_gammas

    def update_bias(self) -> None:
        """No-op: Inkling's correction bias is a trained parameter, not an aux-loss update."""
        return


def inkling_swiglu(hidden_states: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
    """Apply SwiGLU to Inkling's interleaved ``[gate, up]`` projection layout.

    Args:
        hidden_states: Tensor of shape ``[..., 2 * intermediate]`` with arbitrary
            leading dimensions, whose last axis interleaves gate and up channels
            as ``[g0, u0, g1, u1, ...]`` (``::2`` selects gate, ``1::2`` selects up).
        routing_weights: Tensor broadcastable to ``[..., intermediate]`` that scales
            the activated output.

    Returns:
        torch.Tensor: Activated tensor of shape ``[..., intermediate]`` in
        ``hidden_states``' dtype.
    """
    gate = hidden_states[..., ::2]
    up = hidden_states[..., 1::2]
    return (F.silu(gate) * up * routing_weights).to(hidden_states.dtype)


class InklingDenseMLP(nn.Module):
    """Dense Inkling MLP retaining the checkpoint's fused interleaved layout."""

    def __init__(self, config) -> None:
        super().__init__()
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.hidden_size, 2 * config.intermediate_size, dtype=model_dtype)
        )
        self.down_proj = nn.Parameter(torch.empty(config.intermediate_size, config.hidden_size, dtype=model_dtype))
        self.global_scale = nn.Parameter(torch.ones(1, dtype=model_dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the dense MLP over the fused interleaved gate/up projection.

        Args:
            hidden_states: Tensor of shape ``[..., hidden]`` with arbitrary leading
                dimensions.

        Returns:
            torch.Tensor: Tensor of shape ``[..., hidden]``. ``gate_up_proj`` maps
            it to ``[..., 2 * intermediate]`` with gate/up channels interleaved as
            ``[g0, u0, g1, u1, ...]`` (``::2`` gate, ``1::2`` up) before SwiGLU.
        """
        gate_up = hidden_states @ self.gate_up_proj
        activated = F.silu(gate_up[..., ::2]) * gate_up[..., 1::2]
        return (activated @ self.down_proj) * self.global_scale

    @torch.no_grad()
    def init_weights(self, init_std: float = 0.02) -> None:
        nn.init.normal_(self.gate_up_proj, mean=0.0, std=init_std)
        nn.init.normal_(self.down_proj, mean=0.0, std=init_std)
        nn.init.ones_(self.global_scale)


class InklingSharedExperts(nn.Module):
    """Always-on shared experts, matching ``transformers`` ``InklingSharedExperts``.

    The fused gate/up projection stays in the checkpoint's interleaved layout so
    distributed loading only needs transpose views. The per-token ``gammas``
    (from :class:`InklingGate`) weight each expert's output before summation.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.n_shared_experts = config.n_shared_experts
        intermediate_dim = config.moe_intermediate_size
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.gate_up_proj = nn.Parameter(
            torch.empty(config.n_shared_experts, config.hidden_size, 2 * intermediate_dim, dtype=model_dtype)
        )
        self.down_proj = nn.Parameter(
            torch.empty(config.n_shared_experts, intermediate_dim, config.hidden_size, dtype=model_dtype)
        )

    def forward(self, hidden_states: torch.Tensor, gammas: torch.Tensor) -> torch.Tensor:
        """Apply the shared experts.

        Args:
            hidden_states: Input tokens of shape ``[num_tokens, hidden_dim]``.
            gammas: Per-token shared weights of shape ``[num_tokens, n_shared_experts]``.

        Returns:
            torch.Tensor: Output of shape ``[num_tokens, hidden_dim]``.
        """
        input_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(1, -1, input_shape[-1]).expand(self.n_shared_experts, -1, -1)
        gammas = gammas.reshape(-1, self.n_shared_experts, 1).transpose(0, 1)

        gate_up = torch.bmm(hidden_states, self.gate_up_proj)
        activated = inkling_swiglu(gate_up, gammas.to(gate_up.dtype))
        down = torch.bmm(activated, self.down_proj)

        out = down.float().sum(dim=0).to(hidden_states.dtype)
        return out.view(input_shape)


class InklingMoE(MoE):
    """Inkling MoE feed-forward: custom router + shared experts + grouped routed experts.

    Subclasses :class:`~nemo_automodel.components.moe.layers.MoE` so the distributed
    parallelizer (which discovers MoE blocks via ``isinstance`` and shards
    ``self.experts`` across the expert-parallel mesh) treats it uniformly, but the
    generic ``__init__`` / ``forward`` are replaced to match Inkling's routing.
    """

    def __init__(self, config, backend: BackendConfig, moe_config: MoEConfig | None = None) -> None:
        self.moe_config = moe_config or build_inkling_moe_config(config, backend)
        super().__init__(self.moe_config, backend)

        gate_precision = backend.gate_precision if backend.gate_precision is not None else torch.float32
        self.gate = InklingGate(config, gate_precision=gate_precision)
        self.shared_experts = InklingSharedExperts(config)
        self.shared_expert_gate = None

        # Inkling stores routed gate/up channels as [g0, u0, g1, u1, ...].
        # All expert implementations expose one of these activation attributes.
        if hasattr(self.experts, "expert_activation_grouped"):
            self.experts.expert_activation_grouped = inkling_swiglu
        if hasattr(self.experts, "expert_activation"):
            self.experts.expert_activation = inkling_swiglu

    def forward(self, hidden_states: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Route tokens through the shared and routed experts and recombine.

        Args:
            hidden_states: Input of shape ``[..., hidden_dim]`` (the trailing
                dimension is the model dim; leading dims are flattened to tokens).
            padding_mask: Optional boolean mask of padding positions; padded tokens
                are excluded from routed-expert dispatch.

        Returns:
            torch.Tensor: Output with the same shape as ``hidden_states``.
        """
        shape = hidden_states.size()
        x = hidden_states.view(-1, self.dim)
        if padding_mask is not None:
            token_mask = (~padding_mask).flatten()
        else:
            token_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        topk_weights, topk_indices, shared_gammas = self.gate(x)
        routed = self.experts(x, token_mask, topk_weights, topk_indices)
        shared = self.shared_experts(x, shared_gammas)
        return (routed + shared).view(shape)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize MoE parameters in-place (used for from-scratch training)."""
        with buffer_device:
            for p in (self.gate.weight,):
                nn.init.normal_(p, mean=0.0, std=init_std)
            for p in (self.shared_experts.gate_up_proj, self.shared_experts.down_proj):
                nn.init.normal_(p, mean=0.0, std=init_std)
            nn.init.ones_(self.gate.global_scale)
            nn.init.zeros_(self.gate.e_score_correction_bias)
        self.experts.init_weights(buffer_device, init_std)
