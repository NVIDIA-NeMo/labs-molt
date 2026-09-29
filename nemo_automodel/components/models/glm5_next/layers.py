# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native GLM-5.3 text layers.

The released model alternates Kimi Delta Attention (KDA) and GLM-style
KPool-compressed Dynamic Sparse Attention (DSA), with manifold-constrained
Hyper-Connections (mHC) around both sublayers.  FLA owns the production KDA
kernel; small pure-Torch fallbacks keep CPU construction and unit tests useful.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.cudnn_sparse_attention import (
    cudnn_sparse_attention,
    is_cudnn_sparse_attention_available,
)
from nemo_automodel.components.models.glm5_next.config import Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.cp import (
    Glm5NextPackedContext,
    all_gather_backward_anchor,
    all_gather_sequence,
    build_fla_cp_context,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.shared.import_utils import safe_import_from
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

_FLA_MSG = "GLM-5.3 KDA requires the flash-linear-attention/fla extra for GPU training."
_SHORT_CONV_OK, _fla_causal_conv1d = safe_import_from("fla.modules.conv", "causal_conv1d", msg=_FLA_MSG)
_CHUNK_KDA_OK, _chunk_kda = safe_import_from("fla.ops.kda", "chunk_kda", msg=_FLA_MSG)
_RECURRENT_KDA_OK, _recurrent_kda = safe_import_from("fla.ops.kda", "fused_recurrent_kda", msg=_FLA_MSG)
_KDA_GATE_OK, _fused_kda_gate = safe_import_from("fla.ops.kda.gate", "fused_kda_gate", msg=_FLA_MSG)


class Glm5NextRMSNorm(nn.Module):
    """RMSNorm with fp32 variance accumulation.

    Input and output have shape ``[batch, sequence, hidden]``; the leading axes
    may be replaced by any token layout as long as hidden is last.
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize ``[..., hidden]`` and preserve the input dtype."""
        input_dtype = hidden_states.dtype
        states = hidden_states.float()
        states = states * torch.rsqrt(states.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * states.to(input_dtype)

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


class Glm5NextUnweightedRMSNorm(nn.Module):
    """Parameter-free fp32 RMS normalization used by mHC."""

    def __init__(self, eps: float) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize ``[..., hc_streams * hidden]`` without a learned weight."""
        return hidden_states * torch.rsqrt(hidden_states.float().square().mean(-1, keepdim=True) + self.eps).to(
            hidden_states.dtype
        )


class Glm5NextHyperConnectionFp32Params(nn.Module):
    """Own mHC parameters that must remain fp32 under FSDP mixed precision."""

    def __init__(self, mix_size: int) -> None:
        super().__init__()
        self.base = nn.Parameter(torch.empty(mix_size, dtype=torch.float32))
        self.scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def forward(
        self,
        pre_w: torch.Tensor,
        post_w: torch.Tensor,
        comb_w: torch.Tensor,
        hc: int,
        eps: float,
        sinkhorn_iters: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build FP32 mHC weights while this holder's FSDP unit is unsharded."""
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)
        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb


class Glm5NextHyperConnection(nn.Module):
    """Manifold-constrained mixer for ``hc_mult`` residual streams.

    ``hidden_streams`` is ``[batch, sequence, hc_mult, hidden]``. The returned
    tensors are ``post [batch, sequence, hc_mult]``, ``comb [batch, sequence,
    hc_mult, hc_mult]`` and ``collapsed [batch, sequence, hidden]``.
    """

    def __init__(self, config: Glm5NextTextConfig) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.input_norm = Glm5NextUnweightedRMSNorm(config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size, dtype=dtype))
        self._fp32_params = Glm5NextHyperConnectionFp32Params(mix)

    @property
    def base(self) -> nn.Parameter:
        """Expose the checkpoint's flat mHC base parameter."""
        return self._fp32_params.base

    @property
    def scale(self) -> nn.Parameter:
        """Expose the checkpoint's flat mHC scale parameter."""
        return self._fp32_params.scale

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build Sinkhorn mixing weights and collapse streams for one sublayer."""
        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre, post, comb = self._fp32_params(
            pre_w,
            post_w,
            comb_w,
            hc,
            self.hc_eps,
            self.hc_sinkhorn_iters,
        )
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize mHC parameters on ``buffer_device``."""
        with buffer_device:
            nn.init.normal_(self.fn, mean=0.0, std=init_std)
            self.base.zero_()
            self.scale.fill_(1.0)


class _TorchShortConvolution(nn.Module):
    """Depthwise causal Conv1d matching FLA ``ShortConvolution`` state keys."""

    def __init__(self, hidden_size: int, kernel_size: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(hidden_size, 1, kernel_size, dtype=dtype))

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        """Convolve ``[batch, sequence, channels]`` and reset at packed boundaries."""
        if _SHORT_CONV_OK and x.is_cuda:
            return _fla_causal_conv1d(
                x=x,
                weight=self.weight.squeeze(1),
                bias=None,
                initial_state=kwargs.get("cache"),
                output_final_state=kwargs.get("output_final_state", False),
                activation="silu",
                cu_seqlens=cu_seqlens,
                cp_context=kwargs.get("cp_context"),
            )
        if cu_seqlens is None:
            y = F.conv1d(x.transpose(1, 2), self.weight, groups=x.shape[-1], padding=self.kernel_size - 1)
            return F.silu(y[..., : x.shape[1]].transpose(1, 2)), None
        output = torch.zeros_like(x)
        boundaries = cu_seqlens.flatten().tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                continue
            segment = x[:, start:end]
            y = F.conv1d(segment.transpose(1, 2), self.weight, groups=x.shape[-1], padding=self.kernel_size - 1)
            output[:, start:end] = F.silu(y[..., : end - start].transpose(1, 2))
        return output, None

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.weight, -0.01, 0.01)


class _TorchRMSNormGated(Glm5NextRMSNorm):
    """CPU fallback for FLA's gated per-head RMSNorm."""

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalize ``[batch, sequence, heads, head_dim]`` then sigmoid-gate it."""
        return super().forward(hidden_states) * torch.sigmoid(gate.float()).to(hidden_states.dtype)


def _short_conv(hidden_size: int, kernel_size: int, dtype: torch.dtype) -> nn.Module:
    return _TorchShortConvolution(hidden_size, kernel_size, dtype)


def _rms_norm_gated(hidden_size: int, eps: float, dtype: torch.dtype) -> nn.Module:
    return _TorchRMSNormGated(hidden_size, eps, dtype)


class Glm5NextKDAFp32Params(nn.Module):
    """Own recurrent-decay parameters that must remain fp32 under FSDP."""

    def __init__(self, num_heads: int, projection_size: int) -> None:
        super().__init__()
        # Keep the native layout identical to the released checkpoint.  Besides
        # avoiding a state-dict reshape, this matters under FSDP: checkpoint
        # planning sees a DTensor sharded along dimension zero and cannot remove
        # that dimension before the parameter is materialized for forward.
        self.A_log = nn.Parameter(torch.empty(num_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.empty(projection_size, dtype=torch.float32))

    def forward(self, gate: torch.Tensor, head_dim: int, lower_bound: float | None) -> torch.Tensor:
        """Return log-decay gates ``[batch, sequence, heads, head_dim]``."""
        gate = gate.reshape(*gate.shape[:-1], -1, head_dim)
        if _KDA_GATE_OK and gate.is_cuda:
            return _fused_kda_gate(
                gate,
                self.A_log.contiguous(),
                dt_bias=self.dt_bias.contiguous(),
                lower_bound=lower_bound,
            )
        gate = gate.float() + self.dt_bias.view(1, 1, -1, head_dim)
        decay = self.A_log.view(1, 1, -1, 1).exp()
        return lower_bound * torch.sigmoid(decay * gate) if lower_bound is not None else -decay * F.softplus(gate)


def _torch_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Differentiable reference KDA for CPU/small tensors.

    All q/k/v/g tensors are ``[batch, sequence, heads, head_dim]`` and beta is
    ``[batch, sequence, heads]``. Packed boundaries reset the recurrent state.
    """
    batch, sequence, heads, head_dim = q.shape
    output = torch.zeros_like(v)
    boundaries = [0, sequence] if cu_seqlens is None else cu_seqlens.flatten().tolist()
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        state = torch.zeros(batch, heads, head_dim, head_dim, dtype=torch.float32, device=q.device)
        for token in range(start, end):
            q_t = q[:, token].float() * (head_dim**-0.5)
            k_t, v_t = k[:, token].float(), v[:, token].float()
            state = state * g[:, token].exp().unsqueeze(-1)
            prediction = torch.einsum("bhd,bhdv->bhv", k_t, state)
            error = (v_t - prediction) * beta[:, token].float().unsqueeze(-1)
            state = state + torch.einsum("bhd,bhv->bhdv", k_t, error)
            output[:, token] = torch.einsum("bhd,bhdv->bhv", q_t, state).to(output.dtype)
    return output


class Glm5NextLinearAttention(nn.Module):
    """Kimi Delta Attention with released GLM-5.3 checkpoint parameter names."""

    def __init__(self, config: Glm5NextTextConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = config.linear_head_dim
        self.num_heads = config.linear_num_heads
        self.projection_size = self.head_dim * self.num_heads
        self.conv_size = config.linear_conv_kernel_dim
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.q_proj = nn.Linear(config.hidden_size, self.projection_size, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(config.hidden_size, self.projection_size, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(config.hidden_size, self.projection_size, bias=False, dtype=dtype)
        self.q_conv1d = _short_conv(self.projection_size, self.conv_size, dtype)
        self.k_conv1d = _short_conv(self.projection_size, self.conv_size, dtype)
        self.v_conv1d = _short_conv(self.projection_size, self.conv_size, dtype)
        self._fp32_params = Glm5NextKDAFp32Params(self.num_heads, self.projection_size)
        self.f_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.f_b_proj = nn.Linear(self.head_dim, self.projection_size, bias=False, dtype=dtype)
        self.b_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False, dtype=dtype)
        self.g_a_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.g_b_proj = nn.Linear(self.head_dim, self.projection_size, bias=False, dtype=dtype)
        self.o_norm = _rms_norm_gated(self.head_dim, config.rms_norm_eps, dtype)
        self.o_proj = nn.Linear(self.projection_size, config.hidden_size, bias=False, dtype=dtype)
        self._cp_mesh = None

    @property
    def A_log(self) -> nn.Parameter:
        """Expose the checkpoint's flat ``A_log`` parameter name."""
        return self._fp32_params.A_log

    @property
    def dt_bias(self) -> nn.Parameter:
        """Expose the checkpoint's flat ``dt_bias`` parameter name."""
        return self._fp32_params.dt_bias

    def setup_cp_attention(self, cp_mesh) -> None:
        """Attach the one-dimensional contiguous CP mesh."""
        self._cp_mesh = cp_mesh

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        packed_context: Glm5NextPackedContext | None = None,
        padding_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Run KDA over ``[batch, local_sequence, hidden]`` without crossing documents."""
        if packed_context is not None and packed_context.cp_enabled:
            if self._cp_mesh is None:
                raise RuntimeError("GLM-5.3 KDA received a CP batch before apply_cp attached its mesh")
            group = self._cp_mesh.get_group()
            outputs = [
                self._core(
                    hidden_states[row : row + 1],
                    cp_context=build_fla_cp_context(packed_context, row, group, self.conv_size),
                )
                for row in range(hidden_states.shape[0])
            ]
            output = torch.cat(outputs, dim=0)
        elif packed_context is not None:
            outputs = []
            for row in range(hidden_states.shape[0]):
                cu_seqlens, _ = packed_context.row_cu_seqlens(row)
                outputs.append(self._core(hidden_states[row : row + 1], cu_seqlens=cu_seqlens))
            output = torch.cat(outputs, dim=0)
        else:
            output = self._core(hidden_states)
        if padding_mask is not None:
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0)
        return output

    def _core(
        self,
        hidden_states: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        cp_context: Any = None,
    ) -> torch.Tensor:
        """Project and execute KDA for one packed row or a regular batch."""
        kernel_kwargs: dict[str, Any] = {} if cp_context is None else {"cp_context": cp_context}
        conv_kwargs = dict(cache=None, output_final_state=False, cu_seqlens=cu_seqlens, **kernel_kwargs)
        q, _ = self.q_conv1d(x=self.q_proj(hidden_states), **conv_kwargs)
        k, _ = self.k_conv1d(x=self.k_proj(hidden_states), **conv_kwargs)
        v, _ = self.v_conv1d(x=self.v_proj(hidden_states), **conv_kwargs)
        shape = (*hidden_states.shape[:-1], self.num_heads, self.head_dim)
        q, k, v = q.view(shape).contiguous(), k.view(shape).contiguous(), v.view(shape).contiguous()
        gate = self.f_b_proj(self.f_a_proj(hidden_states)).contiguous()
        gate = self._fp32_params(gate, self.head_dim, self.config.linear_lower_bound).contiguous()
        beta = self.b_proj(hidden_states).float().sigmoid().contiguous()
        if _CHUNK_KDA_OK and hidden_states.is_cuda:
            kernel = _chunk_kda if cp_context is not None or hidden_states.shape[1] > 64 else _recurrent_kda
            kernel_options: dict[str, Any] = {
                "use_qk_l2norm_in_kernel": True,
                "transpose_state_layout": True,
            }
            if kernel is _chunk_kda:
                kernel_options["safe_gate"] = self.config.linear_lower_bound is not None
            output, _ = kernel(
                q=q,
                k=k,
                v=v,
                g=gate,
                beta=beta,
                initial_state=None,
                output_final_state=cp_context is None,
                cu_seqlens=cu_seqlens,
                **kernel_options,
                **kernel_kwargs,
            )
        else:
            q = (q.float() / torch.sqrt(q.float().square().sum(-1, keepdim=True) + 1e-6)).to(q.dtype)
            k = (k.float() / torch.sqrt(k.float().square().sum(-1, keepdim=True) + 1e-6)).to(k.dtype)
            output = _torch_recurrent_kda(q, k, v, gate, beta, cu_seqlens)
        final_gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(shape)
        output = self.o_norm(output, final_gate).reshape(*hidden_states.shape[:-1], -1).contiguous()
        return self.o_proj(output)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize KDA while preserving fp32 recurrent parameters."""
        with buffer_device:
            if self.config.linear_lower_bound is not None:
                self.A_log.zero_()
            else:
                self.A_log.uniform_(1, 16).log_()
            self.dt_bias.uniform_(math.log(1e-3), math.log(1e-1))
            dt = self.dt_bias.exp().clamp_min(1e-4)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
            for module in (
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.f_a_proj,
                self.f_b_proj,
                self.b_proj,
                self.g_a_proj,
                self.g_b_proj,
                self.o_proj,
            ):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            for conv in (self.q_conv1d, self.k_conv1d, self.v_conv1d):
                conv.reset_parameters()
            if hasattr(self.o_norm, "reset_parameters"):
                self.o_norm.reset_parameters()


class Glm5NextKPoolIndexer(nn.Module):
    """KPool-compressed DSA indexer for training without a KV cache."""

    def __init__(self, config: Glm5NextTextConfig, layer_idx: int, dtype: torch.dtype) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.index_kpool = config.index_kpool
        self.always_select_tail = config.index_kpool_always_select_tail
        self.wq_b = nn.Linear(config.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=dtype)
        self.wk = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6, dtype=dtype)
        self.weights_proj = nn.Linear(config.hidden_size, self.n_heads, bias=False, dtype=dtype)
        self.index_kpool_compress_ape = nn.Parameter(torch.zeros(self.index_kpool, self.head_dim, dtype=dtype))
        self.index_kpool_compress_gate = nn.Parameter(torch.zeros(self.head_dim, config.hidden_size, dtype=dtype))
        self.softmax_scale = self.head_dim**-0.5

    @torch.no_grad()
    def prepare_pools(self, full_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compress one document's keys into KPool candidates.

        Args:
            full_hidden: Hidden states with shape ``[1, keys, hidden]`` for one
                unpadded document.

        Returns:
            Pool keys with shape ``[complete_pools, index_head_dim]`` and raw
            token indices with shape ``[complete_pools, index_kpool]``.
        """
        keys = self.k_norm(self.wk(full_hidden)).squeeze(0)
        gates = F.linear(full_hidden.squeeze(0), self.index_kpool_compress_gate)
        length = keys.shape[0]
        complete_pools = length // self.index_kpool
        if complete_pools:
            width = complete_pools * self.index_kpool
            grouped_keys = keys[:width].view(complete_pools, self.index_kpool, self.head_dim)
            grouped_gates = gates[:width].view(complete_pools, self.index_kpool, self.head_dim)
            logits = grouped_gates.float() + self.index_kpool_compress_ape.float().unsqueeze(0)
            pool_keys = (logits.softmax(dim=1).to(keys.dtype) * grouped_keys).sum(dim=1)
            pool_indices = torch.arange(width, device=keys.device).view(complete_pools, self.index_kpool)
        else:
            pool_keys = keys.new_empty((0, self.head_dim))
            pool_indices = torch.empty((0, self.index_kpool), dtype=torch.long, device=keys.device)
        return pool_keys, pool_indices

    @torch.no_grad()
    def select(
        self,
        query_hidden: torch.Tensor,
        query_resid: torch.Tensor,
        query_positions: torch.Tensor,
        pool_keys: torch.Tensor,
        pool_indices: torch.Tensor,
        key_length: int,
    ) -> torch.Tensor:
        """Select raw key indices for one query chunk.

        Args:
            query_hidden: Hidden states with shape ``[1, queries, hidden]``.
            query_resid: Low-rank query states with shape
                ``[1, queries, q_lora_rank]``.
            query_positions: Document-local positions with shape ``[queries]``.
            pool_keys: Prepared KPool keys with shape
                ``[complete_pools, index_head_dim]``.
            pool_indices: Prepared raw token indices with shape
                ``[complete_pools, index_kpool]``.
            key_length: Number of tokens in the unpadded document.

        Returns:
            Int32 raw indices with shape
            ``[1, queries, index_topk + index_kpool - 1]`` when tail selection
            is enabled, otherwise ``[1, queries, index_topk]``.
        """
        complete_pools = pool_keys.shape[0]

        queries = self.wq_b(query_resid).view(1, -1, self.n_heads, self.head_dim)
        scores = torch.einsum("bqhd,pd->bqhp", queries.float(), pool_keys.float())
        scores = F.relu(scores * self.softmax_scale)
        weights = self.weights_proj(query_hidden).float() * (self.n_heads**-0.5)
        scores = torch.einsum("bqh,bqhp->bqp", weights, scores)
        if complete_pools:
            pool_end = pool_indices[:, -1]
            visible = pool_end.view(1, 1, -1) <= query_positions.view(1, -1, 1)
            scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
            select_k = min(self.index_topk // self.index_kpool, complete_pools)
            selected = scores.topk(select_k, dim=-1).indices
            selected_valid = visible.expand_as(scores).gather(-1, selected)
            raw = pool_indices[selected].flatten(-2)
            raw = raw.masked_fill(~selected_valid.unsqueeze(-1).expand_as(pool_indices[selected]).flatten(-2), -1)
        else:
            raw = torch.empty((1, query_hidden.shape[1], 0), dtype=torch.long, device=query_hidden.device)

        output_width = self.index_topk
        if self.always_select_tail and self.index_kpool > 1:
            tail_count = (query_positions + 1).remainder(self.index_kpool)
            tail_start = query_positions + 1 - tail_count
            offsets = torch.arange(self.index_kpool - 1, device=query_hidden.device)
            tail = tail_start[:, None] + offsets
            tail = tail.masked_fill(offsets[None] >= tail_count[:, None], -1).unsqueeze(0)
            raw = torch.cat((raw, tail), dim=-1)
            output_width += self.index_kpool - 1
        return F.pad(raw, (0, max(output_width - raw.shape[-1], 0)), value=-1)[..., :output_width].to(torch.int32)

    @torch.no_grad()
    def forward(
        self,
        full_hidden: torch.Tensor,
        query_hidden: torch.Tensor,
        query_resid: torch.Tensor,
        query_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare one document and select indices for a query chunk.

        ``full_hidden`` is ``[1, keys, hidden]``; query tensors are ``[1,
        queries, ...]`` and positions are document-local ``[queries]``.
        Returns ``[1, queries, index_topk + kpool - 1]`` int32 indices.
        """
        pool_keys, pool_indices = self.prepare_pools(full_hidden)
        return self.select(
            query_hidden,
            query_resid,
            query_positions,
            pool_keys,
            pool_indices,
            full_hidden.shape[1],
        )

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize indexer projections and KPool parameters."""
        with buffer_device:
            for module in (self.wq_b, self.wk, self.weights_proj):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            self.k_norm.reset_parameters()
            self.index_kpool_compress_ape.zero_()
            self.index_kpool_compress_gate.fill_(1.0)


class Glm5NextSparseAttention(nn.Module):
    """NoPE MLA whose visibility is selected by the GLM KPool indexer."""

    query_chunk_size = 32

    def __init__(self, config: Glm5NextTextConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.backend = backend
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.qk_head_dim = config.qk_nope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.scaling = self.qk_head_dim**-0.5
        if backend.attn == "cudnn" and self.kv_lora_rank != 512:
            raise ValueError(f"GLM-5.3 cuDNN sparse attention requires kv_lora_rank=512, got {self.kv_lora_rank}.")
        if backend.attn == "cudnn" and config.attention_dropout != 0.0:
            raise ValueError(
                "GLM-5.3 cuDNN sparse attention does not support attention dropout; "
                f"got attention_dropout={config.attention_dropout}."
            )
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.q_a_proj = nn.Linear(config.hidden_size, self.q_lora_rank, bias=config.attention_bias, dtype=dtype)
        self.q_a_layernorm = Glm5NextRMSNorm(self.q_lora_rank, config.rms_norm_eps, dtype)
        self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False, dtype=dtype)
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, self.kv_lora_rank, bias=config.attention_bias, dtype=dtype
        )
        self.kv_a_layernorm = Glm5NextRMSNorm(self.kv_lora_rank, config.rms_norm_eps, dtype)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_head_dim + self.v_head_dim),
            bias=False,
            dtype=dtype,
        )
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False, dtype=dtype)
        self.indexer = Glm5NextKPoolIndexer(config, layer_idx, dtype)
        self._cp_mesh = None

    def setup_cp_attention(self, cp_mesh) -> None:
        """Attach the CP mesh used for differentiable full-sequence gathering."""
        self._cp_mesh = cp_mesh

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        packed_context: Glm5NextPackedContext,
        padding_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        """Run document-isolated sparse attention on ``[batch, local_sequence, hidden]``."""
        if packed_context is None:
            raise ValueError("GLM-5.3 sparse attention requires a packed document context")
        if packed_context.cp_enabled:
            if self._cp_mesh is None:
                raise RuntimeError("GLM-5.3 DSA received a CP batch before apply_cp attached its mesh")
            full_hidden = all_gather_sequence(hidden_states, self._cp_mesh.get_group(), dim=1)
        else:
            full_hidden = hidden_states
        output = torch.zeros_like(hidden_states)
        local_start = packed_context.seq_start
        local_end = local_start + hidden_states.shape[1]
        for row in range(hidden_states.shape[0]):
            doc_ids = packed_context.doc_ids[row]
            starts = torch.nonzero(doc_ids[1:] != doc_ids[:-1], as_tuple=False).flatten().add(1).tolist()
            boundaries = [0, *starts, doc_ids.numel()]
            for doc_start, doc_end in zip(boundaries[:-1], boundaries[1:]):
                if int(doc_ids[doc_start]) <= 0:
                    continue
                query_start, query_end = max(doc_start, local_start), min(doc_end, local_end)
                if query_end <= query_start:
                    continue
                doc = full_hidden[row : row + 1, doc_start:doc_end]
                local_query_start = query_start - doc_start
                local_query_end = query_end - doc_start
                doc_output = self._forward_document(doc, local_query_start, local_query_end)
                out_start = query_start - local_start
                output[row : row + 1, out_start : out_start + doc_output.shape[1]] = doc_output
        if packed_context.cp_enabled:
            # A short packed sample can leave this contiguous CP interval with
            # no valid queries. Keep the differentiable all-gather connected to
            # the local output so every CP rank launches its backward AllReduce.
            output = output + all_gather_backward_anchor(full_hidden)
        if padding_mask is not None:
            output = output.masked_fill(padding_mask.unsqueeze(-1), 0)
        return output

    def _forward_document(self, full_hidden: torch.Tensor, query_start: int, query_end: int) -> torch.Tensor:
        """Execute sparse attention for a local query interval of one full document.

        Args:
            full_hidden: Unpadded document states with shape ``[1, key_tokens, hidden]``.
            query_start: Inclusive document-local index of the first local query.
            query_end: Exclusive document-local index of the final local query.

        Returns:
            Projected attention output with shape
            ``[1, query_end - query_start, hidden]``.
        """
        length = full_hidden.shape[1]
        latent = self.kv_a_layernorm(self.kv_a_proj_with_mqa(full_hidden))
        pool_keys, pool_indices = self.indexer.prepare_pools(full_hidden)
        if self.backend.attn == "cudnn":
            return self._forward_document_cudnn(
                full_hidden,
                latent,
                pool_keys,
                pool_indices,
                query_start,
                query_end,
            )

        expanded = self.kv_b_proj(latent).view(1, length, self.num_heads, self.qk_head_dim + self.v_head_dim)
        key, value = expanded.split([self.qk_head_dim, self.v_head_dim], dim=-1)
        key, value = key.transpose(1, 2), value.transpose(1, 2)
        chunks = []
        for start in range(query_start, query_end, self.query_chunk_size):
            end = min(start + self.query_chunk_size, query_end)
            query_hidden = full_hidden[:, start:end]
            q_resid = self.q_a_layernorm(self.q_a_proj(query_hidden))
            query = self.q_b_proj(q_resid).view(1, end - start, self.num_heads, self.qk_head_dim).transpose(1, 2)
            positions = torch.arange(start, end, device=full_hidden.device)
            indices = self.indexer.select(
                query_hidden,
                q_resid,
                positions,
                pool_keys,
                pool_indices,
                length,
            )
            valid = indices.ge(0) & indices.lt(length)
            safe = indices.clamp(0, length - 1)
            selected_counts = torch.zeros((1, end - start, length), dtype=torch.int32, device=full_hidden.device)
            selected_counts.scatter_add_(-1, safe.long(), valid.to(torch.int32))
            selected = selected_counts.ne(0)
            attn = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=selected.unsqueeze(1),
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                scale=self.scaling,
            )
            chunks.append(attn.transpose(1, 2).reshape(1, end - start, -1))
        return self.o_proj(torch.cat(chunks, dim=1))

    def _forward_document_cudnn(
        self,
        full_hidden: torch.Tensor,
        latent: torch.Tensor,
        pool_keys: torch.Tensor,
        pool_indices: torch.Tensor,
        query_start: int,
        query_end: int,
    ) -> torch.Tensor:
        """Run absorbed latent attention through FlashMLA/cuDNN for one document.

        Args:
            full_hidden: Unpadded document states with shape ``[1, key_tokens, hidden]``.
            latent: Normalized shared latent K/V with shape
                ``[1, key_tokens, 512]``.
            pool_keys: KPool-compressed index keys with shape
                ``[complete_pools, index_head_dim]``.
            pool_indices: Document-local token indices with shape
                ``[complete_pools, index_kpool]``.
            query_start: Inclusive document-local index of the first local query.
            query_end: Exclusive document-local index of the final local query.

        Returns:
            Projected attention output with shape
            ``[1, query_end - query_start, hidden]``.
        """
        if not is_cudnn_sparse_attention_available():
            raise RuntimeError(
                "backend.attn='cudnn' requires the optional cuDNN sparse-attention "
                "and FlashMLA runtimes, but they are unavailable in this environment."
            )

        weight = self.kv_b_proj.weight.view(
            self.num_heads,
            self.qk_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        w_kc, w_vc = weight.split([self.qk_head_dim, self.v_head_dim], dim=1)
        absorbed_queries = []
        selected_indices = []
        length = full_hidden.shape[1]
        for start in range(query_start, query_end, self.query_chunk_size):
            end = min(start + self.query_chunk_size, query_end)
            query_hidden = full_hidden[:, start:end]
            q_resid = self.q_a_layernorm(self.q_a_proj(query_hidden))
            query = self.q_b_proj(q_resid).view(1, end - start, self.num_heads, self.qk_head_dim)
            absorbed_queries.append(torch.einsum("bqhd,hdc->bqhc", query, w_kc.to(query.dtype)).squeeze(0))
            positions = torch.arange(start, end, device=full_hidden.device)
            indices = self.indexer.select(
                query_hidden,
                q_resid,
                positions,
                pool_keys,
                pool_indices,
                length,
            )
            selected_indices.append(indices.squeeze(0).unsqueeze(1))

        latent_output = cudnn_sparse_attention(
            torch.cat(absorbed_queries, dim=0).contiguous(),
            latent.squeeze(0).unsqueeze(1).contiguous(),
            torch.cat(selected_indices, dim=0).contiguous(),
            self.scaling,
            all_rows_nonempty=self.indexer.always_select_tail,
        )
        attention_output = torch.einsum("qhc,hvc->qhv", latent_output, w_vc.to(latent_output.dtype))
        return self.o_proj(attention_output.reshape(1, query_end - query_start, -1))

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize MLA and indexer parameters."""
        with buffer_device:
            for module in (self.q_a_proj, self.q_b_proj, self.kv_a_proj_with_mqa, self.kv_b_proj, self.o_proj):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
            self.q_a_layernorm.reset_parameters()
            self.kv_a_layernorm.reset_parameters()
        self.indexer.init_weights(buffer_device, init_std)


class Glm5NextDecoderLayer(nn.Module):
    """One mHC decoder block with KDA/DSA and dense/MoE feed-forward."""

    def __init__(
        self,
        config: Glm5NextTextConfig,
        layer_idx: int,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.block_type = config.layer_types[layer_idx]
        self.is_linear_attn = self.block_type == "linear_attention"
        self.is_moe_layer = config.mlp_layer_types[layer_idx] == "sparse"
        self.self_attn = (
            Glm5NextLinearAttention(config, layer_idx)
            if self.is_linear_attn
            else Glm5NextSparseAttention(config, layer_idx, backend)
        )
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.mlp = (
            MoE(moe_config, backend)
            if self.is_moe_layer
            else MLP(
                config.hidden_size,
                config.intermediate_size,
                backend.linear,
                dtype=dtype,
                swiglu_limit=config.swiglu_limit,
            )
        )
        self.input_layernorm = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.post_attention_layernorm = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.attn_hc = Glm5NextHyperConnection(config)
        self.ffn_hc = Glm5NextHyperConnection(config)

    def forward(
        self,
        hidden_streams: torch.Tensor,
        *,
        packed_context: Glm5NextPackedContext,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Transform residual streams ``[batch, local_sequence, hc_mult, hidden]``."""
        dtype = hidden_streams.dtype
        residual = hidden_streams
        post, comb, collapsed = self.attn_hc(hidden_streams)
        update = self.self_attn(
            self.input_layernorm(collapsed),
            packed_context=packed_context,
            padding_mask=padding_mask,
            **kwargs,
        )
        hidden_streams = post.to(dtype).unsqueeze(-1) * update.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )
        residual = hidden_streams
        post, comb, collapsed = self.ffn_hc(hidden_streams)
        update = self.post_attention_layernorm(collapsed)
        update = self.mlp(update, padding_mask) if self.is_moe_layer else self.mlp(update)
        return post.to(dtype).unsqueeze(-1) * update.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), residual
        )

    def update_moe_gate_bias(self) -> None:
        """Update the correction bias for this layer's learned MoE router."""
        if self.is_moe_layer:
            moe = unwrap_checkpoint_wrapper(self.mlp)
            if isinstance(moe, MoE) and moe.gate.bias_update_factor > 0:
                moe.gate.update_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float) -> None:
        """Initialize all decoder children."""
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.attn_hc.init_weights(buffer_device, init_std)
        self.ffn_hc.init_weights(buffer_device, init_std)
        self.self_attn.init_weights(buffer_device, init_std)
        self.mlp.init_weights(buffer_device, init_std)


__all__ = [
    "Glm5NextDecoderLayer",
    "Glm5NextHyperConnection",
    "Glm5NextKPoolIndexer",
    "Glm5NextLinearAttention",
    "Glm5NextRMSNorm",
    "Glm5NextSparseAttention",
]
