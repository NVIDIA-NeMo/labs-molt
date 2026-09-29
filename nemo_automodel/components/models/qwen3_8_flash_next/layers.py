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

"""Qwen3.8-Flash-Next model-local layers.

The HyperConnection equations in this module follow the Qwen3.8-Flash-Next reference
implementation. They are intentionally not shared with DeepSeek-V4: that model
uses a different Sinkhorn-based HyperConnection parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.distributed.blockdiag_cp import BlockdiagCpModelState
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk
from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet
from nemo_automodel.components.models.qwen3_8_flash_next.cp import (
    Qwen3_8_FlashNextCPContext,
    qwen3_8_flash_next_cp_all_gather,
)
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import (
    Qwen3_8_FlashNextQSAIndexer,
    qsa_gqa_attention,
)
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextAttention
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _rms_norm_gated_fp32(
    hidden_states: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    use_sigmoid: bool,
) -> torch.Tensor:
    """FP32 RMSNorm + output gate as one fusable elementwise chain."""
    input_dtype = hidden_states.dtype
    normalized = hidden_states.float()
    variance = normalized.square().mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    normalized = weight.float() * normalized
    gate_fp32 = gate.float()
    activated_gate = torch.sigmoid(gate_fp32) if use_sigmoid else F.silu(gate_fp32)
    return (normalized * activated_gate).to(input_dtype)


def _grouped_rms_norm_fp32(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
) -> torch.Tensor:
    """FP32 grouped RMSNorm as one fusable elementwise chain."""
    input_dtype = hidden_states.dtype
    grouped = hidden_states.float().unflatten(-1, (-1, group_size))
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = (grouped * torch.rsqrt(variance + eps)).flatten(-2)
    return (normalized * (1.0 + weight.float())).to(input_dtype)


# Compiled variants are used on CUDA only; CPU tests and the numerical oracle
# keep the eager chain so they run without inductor warm-up.
_rms_norm_gated_fp32_compiled = torch.compile(_rms_norm_gated_fp32, dynamic=True)
_grouped_rms_norm_fp32_compiled = torch.compile(_grouped_rms_norm_fp32, dynamic=True)


class Qwen3_8_FlashNextRMSNormGated(nn.Module):
    """Qwen3.8-Flash-Next GDN output normalization with its checkpoint-selected gate.

    Transformers' Qwen3.5 GatedDeltaNet hard-codes a SiLU output gate.  Qwen3.8-Flash-Next
    keeps the same projections and delta-rule core but sets
    ``output_gate_type='sigmoid'``.  Keeping this model-local module avoids
    changing the shared Qwen3.5 execution contract.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        activation: str = "sigmoid",
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if activation not in ("sigmoid", "silu"):
            raise ValueError(f"Unsupported Qwen3.8-Flash-Next GDN output gate {activation!r}")
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        self.variance_epsilon = eps
        self.activation = activation

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Normalize in fp32, then apply the configured gate in fp32.

        Args:
            hidden_states: Tensor of shape ``[..., hidden_size]`` containing
                GatedDeltaNet values.
            gate: Tensor of shape ``[..., hidden_size]`` containing the
                elementwise output-gate logits.

        Returns:
            Tensor of shape ``[..., hidden_size]`` in the input dtype.
        """
        # SGLang's layernorm_gated kernel keeps xhat*weight and gate multiply
        # in fp32 and casts only at the output store.
        norm_fn = _rms_norm_gated_fp32_compiled if hidden_states.is_cuda else _rms_norm_gated_fp32
        return norm_fn(hidden_states, gate, self.weight, self.variance_epsilon, self.activation == "sigmoid")

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Match the multiplicative RMSNorm checkpoint convention."""
        self.weight.fill_(1.0)


class Qwen3_8_FlashNextGatedDeltaNet(CPAwareGatedDeltaNet):
    """CP-aware GatedDeltaNet with Qwen3.8-Flash-Next's sigmoid output gate."""

    def __init__(self, config: object, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        output_gate_type = str(getattr(config, "output_gate_type", "sigmoid"))
        self._packed_global_cu_seqlens: torch.Tensor | None = None
        self.norm = Qwen3_8_FlashNextRMSNormGated(
            self.head_v_dim,
            eps=float(getattr(config, "rms_norm_eps")),
            activation=output_gate_type,
            dtype=get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16),
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs: object) -> torch.Tensor:
        """Route packed boundaries around the inherited non-CP varlen path.

        Without CP, ``cu_seqlens`` flows to the inherited FLA varlen forward
        unchanged. With an active CP mesh the boundaries describe the global
        packed row, so they are stashed for :meth:`_forward_with_cp` and the
        inherited dispatcher sees no ``cu_seqlens``.
        """
        cu_seqlens = kwargs.pop("cu_seqlens", None)
        cp_active = self._cp_mesh is not None and self._cp_mesh.size() > 1
        if not cp_active or cu_seqlens is None:
            if cu_seqlens is not None and kwargs.get("attention_mask") is None:
                # The inherited packed conv path reads per-token document IDs
                # from ``attention_mask``. Synthesize them from the boundaries
                # for cu_seqlens-only packed batches.
                boundaries = cu_seqlens.reshape(-1).to(device=hidden_states.device, dtype=torch.long)
                document_ids = torch.repeat_interleave(
                    torch.arange(boundaries.numel() - 1, device=hidden_states.device),
                    boundaries.diff(),
                )
                kwargs["attention_mask"] = document_ids.unsqueeze(0).to(torch.int32)
            return super().forward(hidden_states, cu_seqlens=cu_seqlens, **kwargs)
        self._packed_global_cu_seqlens = cu_seqlens
        try:
            return super().forward(hidden_states, **kwargs)
        finally:
            self._packed_global_cu_seqlens = None

    def _forward_with_cp(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        seq_index: torch.Tensor | None,
        blockdiag_state: BlockdiagCpModelState | None = None,
    ) -> torch.Tensor:
        """Run the inherited FLA CP core in explicit contiguous sequence order.

        Args:
            hidden_states: Local contiguous states of shape ``[batch,
                local_sequence, hidden]``. Rank ``r`` owns global positions
                ``[r * local_sequence, (r + 1) * local_sequence)``.
            position_ids: Local global positions of shape ``[batch,
                local_sequence]`` or repeated text-RoPE axes of shape ``[axes,
                batch, local_sequence]``.
            seq_index: Optional local positions of shape ``[local_sequence]``
                or ``[batch, local_sequence]``; ignored because Qwen3.8-Flash-Next's
                model-owned layout is always contiguous.
            blockdiag_state: External packed CP metadata. Qwen3.8-Flash-Next builds its
                own state and rejects an externally supplied one.

        Returns:
            Local GDN output of shape ``[batch, local_sequence, hidden]`` in
            unchanged contiguous order.
        """
        del seq_index
        if blockdiag_state is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next GDN context parallelism builds its own packed state; "
                "external blockdiag state is unsupported"
            )
        if self._cp_mesh is None or self._cp_mesh.size() <= 1:
            raise RuntimeError("Qwen3.8-Flash-Next contiguous GDN CP requires an active CP mesh")
        if position_ids is None:
            raise ValueError("Qwen3.8-Flash-Next contiguous GDN CP requires global position_ids")

        cp_group = self._cp_mesh.get_group()
        global_sequence_length = hidden_states.shape[1] * self._cp_mesh.size()
        packed_boundaries = getattr(self, "_packed_global_cu_seqlens", None)
        if packed_boundaries is None:
            # One contiguous document spanning the padded global sequence.
            boundary_values = [0, global_sequence_length]
        else:
            boundary_values = [int(value) for value in packed_boundaries.reshape(-1).tolist()]
            if boundary_values[-1] < global_sequence_length:
                # The CP padding tail is its own segment so its recurrence
                # cannot contaminate the final document.
                boundary_values.append(global_sequence_length)
        cu_seqlens_cpu = torch.tensor(boundary_values, dtype=torch.long, device="cpu")
        cu_seqlens = cu_seqlens_cpu.to(hidden_states.device)
        contiguous_state = BlockdiagCpModelState(
            group=cp_group,
            packed_cu_seqlens=cu_seqlens,
            packed_cu_seqlens_cpu=cu_seqlens_cpu,
        )
        return super()._forward_with_cp(
            hidden_states,
            position_ids=position_ids,
            seq_index=None,
            blockdiag_state=contiguous_state,
        )


class Qwen3_8_FlashNextGroupedRMSNorm(nn.Module):
    """Gemma-style RMS normalization over fixed-width HyperConnection branches.

    Args:
        hidden_size: Flattened feature width.
        group_size: Number of features normalized together. A flattened input
            of shape ``[..., hidden_size]`` is viewed as
            ``[..., hidden_size // group_size, group_size]``.
        eps: Variance epsilon.

    Tensor layout:
        Input and output have shape ``[..., hidden_size]``. The learned weight
        has shape ``[hidden_size]`` and is applied as ``1 + weight``.
    """

    def __init__(self, hidden_size: int, group_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        if hidden_size <= 0 or group_size <= 0 or hidden_size % group_size != 0:
            raise ValueError(
                "Qwen3_8_FlashNextGroupedRMSNorm requires positive, divisible widths; "
                f"got hidden_size={hidden_size}, group_size={group_size}"
            )
        self.hidden_size = hidden_size
        self.group_size = group_size
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Normalize each branch independently.

        Args:
            hidden_states: Flattened branch states of shape
                ``[..., hidden_size]``.

        Returns:
            Normalized states of shape ``[..., hidden_size]``.
        """
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected hidden width {self.hidden_size}, got {hidden_states.shape[-1]}")
        norm_fn = _grouped_rms_norm_fp32_compiled if hidden_states.is_cuda else _grouped_rms_norm_fp32
        return norm_fn(hidden_states, self.weight, self.group_size, self.eps)

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Reset the additive Gemma-style scale to zero."""
        self.weight.zero_()


@dataclass(frozen=True)
class Qwen3_8_FlashNextHyperConnectionResidual:
    """Residual tensors retained between a HyperConnection read and write.

    Attributes:
        hidden_states: Flattened HC streams of shape ``[..., hc_count * hidden_size]``.
        normalized_states: Branch-normalized streams with the same shape.
    """

    hidden_states: torch.Tensor
    normalized_states: torch.Tensor


class Qwen3_8_FlashNextHyperConnection(nn.Module):
    """Qwen3.8-Flash-Next gated HyperConnection read/write transform.

    A read normalizes ``hc_count`` streams independently, predicts a feature
    gate, and averages the gated streams to one block input. A write predicts
    one injection gate per stream and adds the block output back to every
    stream. The final decoder mixer uses only the read side.

    Args:
        hidden_size: Width of one HC stream.
        hc_count: Number of streams.
        lowrank_size: Bottleneck width used to predict read gates.
        rms_norm_eps: Variance epsilon for branch normalization.
        backend: Linear backend configuration.
        use_combine: Whether to instantiate the write-side injection weight.
        dtype: Parameter dtype override. If omitted, the backend model dtype is
            resolved by the caller and should be passed explicitly.

    Tensor layout:
        Reads ``[..., hc_count * hidden_size]`` and returns
        ``[..., hidden_size]``. Writes a block tensor ``[..., hidden_size]``
        back to a residual tensor ``[..., hc_count * hidden_size]``.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_count: int,
        lowrank_size: int,
        rms_norm_eps: float,
        backend: BackendConfig,
        *,
        use_combine: bool = True,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        super().__init__()
        if hc_count <= 1:
            raise ValueError(f"Qwen3.8-Flash-Next requires hc_count > 1, got {hc_count}")
        if hidden_size <= 0 or lowrank_size <= 0:
            raise ValueError(f"hidden_size and lowrank_size must be positive, got {hidden_size}, {lowrank_size}")
        self.hidden_size = hidden_size
        self.hc_count = hc_count
        self.flat_hidden_size = hidden_size * hc_count
        self.lowrank_size = lowrank_size
        self.use_combine = use_combine
        parameter_dtype = get_dtype(dtype, torch.bfloat16)

        self.hc_norm = Qwen3_8_FlashNextGroupedRMSNorm(
            self.flat_hidden_size,
            group_size=hidden_size,
            eps=rms_norm_eps,
        )
        self.input_mix_weight_down = initialize_linear_module(
            backend.linear,
            self.flat_hidden_size,
            lowrank_size,
            bias=False,
            dtype=parameter_dtype,
        )
        self.input_mix_weight_up = initialize_linear_module(
            backend.linear,
            lowrank_size,
            self.flat_hidden_size,
            bias=False,
            dtype=parameter_dtype,
        )
        self.block_inject_weight = (
            initialize_linear_module(
                backend.linear,
                self.flat_hidden_size,
                hc_count,
                bias=False,
                dtype=parameter_dtype,
            )
            if use_combine
            else None
        )

    def mix(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, Qwen3_8_FlashNextHyperConnectionResidual]:
        """Collapse HC streams to the input of an attention or MoE block.

        Args:
            hidden_states: Flattened HC streams of shape
                ``[..., hc_count * hidden_size]``.

        Returns:
            A pair containing the mixed block input of shape
            ``[..., hidden_size]`` and the residual tensors needed by
            :meth:`combine`.
        """
        if hidden_states.shape[-1] != self.flat_hidden_size:
            raise ValueError(f"Expected HC width {self.flat_hidden_size}, got {hidden_states.shape[-1]}")
        normalized = self.hc_norm(hidden_states)
        # FSDP may keep the residual stream in fp32 while materializing the
        # projection weights in bf16 for compute. Match the projection dtype at
        # the linear boundary, then combine its gates with the original
        # normalized stream so the caller's residual dtype is preserved.
        projection_input = normalized.to(dtype=self.input_mix_weight_down.weight.dtype)
        gates = F.silu(self.input_mix_weight_down(projection_input) / self.hc_count)
        gates = torch.sigmoid(self.input_mix_weight_up(gates))
        mixed = (
            gates.unflatten(-1, (self.hc_count, self.hidden_size))
            * normalized.unflatten(-1, (self.hc_count, self.hidden_size))
        ).mean(dim=-2)
        residual = Qwen3_8_FlashNextHyperConnectionResidual(hidden_states, normalized)
        return mixed, residual

    def combine(
        self,
        block_output: torch.Tensor,
        residual: Qwen3_8_FlashNextHyperConnectionResidual,
    ) -> torch.Tensor:
        """Inject one block output into every HC stream.

        Args:
            block_output: Attention or MoE output of shape
                ``[..., hidden_size]``.
            residual: Flattened pre-block streams and their normalized values,
                each shaped ``[..., hc_count * hidden_size]``.

        Returns:
            Updated flattened streams of shape
            ``[..., hc_count * hidden_size]``.
        """
        if self.block_inject_weight is None:
            raise RuntimeError("This final-mixer HyperConnection has no combine side")
        if block_output.shape[-1] != self.hidden_size:
            raise ValueError(f"Expected block output width {self.hidden_size}, got {block_output.shape[-1]}")
        if residual.hidden_states.shape[-1] != self.flat_hidden_size:
            raise ValueError(f"Expected residual width {self.flat_hidden_size}, got {residual.hidden_states.shape[-1]}")
        projection_input = residual.normalized_states.to(dtype=self.block_inject_weight.weight.dtype)
        injection_gate = 2.0 * torch.sigmoid(self.block_inject_weight(projection_input) / self.hc_count)
        streams = residual.hidden_states.unflatten(-1, (self.hc_count, self.hidden_size))
        injection = block_output.unsqueeze(-2) * injection_gate.unsqueeze(-1)
        return (streams + injection).flatten(-2)

    @torch.no_grad()
    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize HC weights for training from scratch.

        Args:
            init_std: Standard deviation for all HC linear weights.
        """
        self.hc_norm.reset_parameters()
        nn.init.trunc_normal_(self.input_mix_weight_down.weight, mean=0.0, std=init_std)
        nn.init.trunc_normal_(self.input_mix_weight_up.weight, mean=0.0, std=init_std)
        if self.block_inject_weight is not None:
            nn.init.trunc_normal_(self.block_inject_weight.weight, mean=0.0, std=init_std)


class Qwen3_8_FlashNextQSAAttention(Qwen3NextAttention):
    """Qwen3.8-Flash-Next gated attention with compressed-block QSA routing.

    The main query/key/value, output gate, and output projection retain the
    Qwen3-Next equations.  A separate frozen indexer returns logical token IDs,
    then the model-owned QSA dispatcher evaluates only those IDs. CUDA BF16
    training selects FlexAttention or SM90 CuTe through backend.attn; CPU and
    explicit reference backends use the PyTorch oracle. Main Q/K/V remain
    differentiable.
    """

    def __init__(self, config: object, layer_idx: int, backend: BackendConfig) -> None:
        # QSA owns its sparse-attention dispatch. The inherited constructor is
        # reused only for projections/norms, but its generic attention factory
        # does not implement ``flex`` or ``cute``. Give it an isolated SDPA
        # copy, then discard the unused callable and retain the real backend.
        parent_backend = replace(backend, attn="sdpa")
        super().__init__(config, layer_idx, parent_backend)
        self.backend = backend
        self.attn_module = None
        self.attn_func = None
        self._cp_mesh: DeviceMesh | None = None
        self.indexer = Qwen3_8_FlashNextQSAIndexer(config, backend)

    def setup_cp_attention(self, cp_mesh: DeviceMesh) -> None:
        """Install the contiguous CP mesh used for QSA K/V exchange."""
        self._cp_mesh = cp_mesh

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cp_context: Qwen3_8_FlashNextCPContext | None = None,
        **attn_kwargs: object,
    ) -> torch.Tensor:
        """Select compressed blocks and run model-owned sparse GQA.

        Args:
            x: Block input of shape ``[batch, sequence, hidden_size]``.
            freqs_cis: Rotary values ``[batch, sequence, rotary_dim]`` whose
                final axis stores concatenated cosine and sine values.
            attention_mask: Optional token mask of shape ``[batch, sequence]``
                or backend-specific causal attention mask.
            cp_context: Optional contiguous CP metadata. When present, ``x``
                contains local queries while compressed/main K/V are gathered
                to global rank order.
            **attn_kwargs: Backend attention metadata. Packed layouts provide
                global document boundaries through ``cu_seqlens`` or
                ``cp_context``.

        Returns:
            Attention output with the same shape as ``x``.
        """
        if x.ndim != 3:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires rank-3 [batch, sequence, hidden] inputs, including a "
                f"batch-one packed THD row; got {tuple(x.shape)}"
            )
        if x.shape[1] == 0:
            raise ValueError("Qwen3.8-Flash-Next QSA requires a non-empty sequence")
        packed_cu_seqlens = attn_kwargs.get("cu_seqlens")
        cp_active = self._cp_mesh is not None and self._cp_mesh.size() > 1
        context_boundaries = getattr(cp_context, "global_cu_seqlens", None) if cp_context is not None else None
        if context_boundaries is not None:
            if packed_cu_seqlens is not None:
                raise ValueError(
                    "Qwen3.8-Flash-Next packed CP takes document boundaries from its CP context; "
                    "passing cu_seqlens directly is ambiguous"
                )
            packed_cu_seqlens = context_boundaries
        elif packed_cu_seqlens is not None and (cp_active or cp_context is not None):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA packed CP requires global document boundaries in the CP context"
            )
        if packed_cu_seqlens is not None:
            if x.shape[0] != 1:
                raise ValueError(
                    f"Qwen3.8-Flash-Next packed QSA requires a THD batch of one row, got batch={x.shape[0]}"
                )
            if not isinstance(packed_cu_seqlens, torch.Tensor):
                raise TypeError(f"cu_seqlens must be a tensor of document boundaries, got {type(packed_cu_seqlens)}")
        if cp_active and cp_context is None:
            raise RuntimeError(
                "Qwen3.8-Flash-Next QSA has an active CP mesh but did not receive model-owned batch metadata"
            )
        if cp_context is not None:
            # QSA discards the inherited dense attention module so the shared
            # parallelizer installs this model-owned mesh. The batch context is
            # the authoritative transport metadata; validate both agree.
            if self._cp_mesh is not None:
                cp_group = self._cp_mesh.get_group()
                if cp_context.size != self._cp_mesh.size() or cp_context.rank != dist.get_rank(cp_group):
                    raise RuntimeError("Qwen3.8-Flash-Next QSA CP context does not match the installed CP mesh")

        selected_token_ids = self.indexer(
            x,
            freqs_cis=freqs_cis,
            attention_mask=None if packed_cu_seqlens is not None else attention_mask,
            cp_context=cp_context,
            cu_seqlens=packed_cu_seqlens,
        )

        batch_size, sequence_length, _ = x.shape
        # Keep the indexer's fixed-width output hookable for parity while
        # avoiding 2,051 padded gathers on short sequences.  Valid IDs are
        # contiguous and a causal row can never contain more than S tokens
        # (or, when packed, more than the longest document).
        if packed_cu_seqlens is not None:
            max_visible_tokens = int(packed_cu_seqlens.diff().max())
        elif cp_context is not None:
            max_visible_tokens = cp_context.global_sequence_length
        else:
            max_visible_tokens = sequence_length
        attention_width = min(max_visible_tokens, selected_token_ids.shape[-1])
        selected_token_ids = selected_token_ids[..., :attention_width]
        query = self.q_proj(x).view(batch_size, sequence_length, -1, self.head_dim * 2)
        key = self.k_proj(x).view(batch_size, sequence_length, -1, self.head_dim)
        value = self.v_proj(x).view(batch_size, sequence_length, -1, self.head_dim)
        query, gate = torch.chunk(query, 2, dim=-1)
        gate = gate.reshape(*x.shape[:-1], -1)
        query = self.q_norm(query)
        key = self.k_norm(key)
        query, key = apply_rotary_emb_qk(
            query,
            key,
            freqs_cis,
            format="bshd",
            rope_fusion=False,
        )
        if cp_context is not None:
            key = qwen3_8_flash_next_cp_all_gather(key, cp_context, sequence_dim=1, differentiable=True)
            value = qwen3_8_flash_next_cp_all_gather(value, cp_context, sequence_dim=1, differentiable=True)

        # One dispatcher serves dense, packed, and CP layouts: routes are
        # global K/V coordinates and both CUDA backends accept S_q != S_kv, so
        # packed rows need no dedicated kernel path. CPU keeps the oracle.
        attn_output = qsa_gqa_attention(
            query,
            key,
            value,
            selected_token_ids,
            backend=self.backend.attn,
            softmax_scale=self.scaling,
        )
        attn_output = attn_output.reshape(*x.shape[:-1], -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize attention and indexer weights.

        Args:
            buffer_device: Device retained for the common attention initializer contract.
            init_std: Projection initialization standard deviation.
        """
        super().init_weights(buffer_device, init_std=init_std)
        self.indexer.init_weights(init_std=init_std)


class Qwen3_8_FlashNextDecoderLayer(nn.Module):
    """One Qwen3.8-Flash-Next decoder layer with two learned HyperConnections.

    Args:
        layer_idx: Zero-based decoder index.
        config: Qwen3.8-Flash-Next text configuration.
        moe_config: Native MoE configuration.
        backend: Attention, linear, and expert backend configuration.
        ple: Optional Engram-derived PLE module. The checkpoint installs it
            only on decoder index 1.

    Tensor layout:
        The first layer accepts token embeddings ``[batch, sequence, hidden]``
        and expands them to flattened HC state
        ``[batch, sequence, hc_count * hidden]``. Every layer returns that
        flattened HC layout.
    """

    def __init__(
        self,
        layer_idx: int,
        config: object,
        moe_config: object,
        backend: BackendConfig,
        *,
        ple: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        layer_types = getattr(config, "layer_types", None)
        if layer_types is None:
            block_types = getattr(config, "layers_block_type")
            layer_types = ["full_attention" if value == "attention" else value for value in block_types]
        self.layer_type = str(layer_types[layer_idx])
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.hc_count = int(getattr(config, "hc_count"))
        self.ple = ple
        # PLE's owner-sharded lookup uses mutable distributed collectives.
        # PyTorch's selective TorchDispatch checkpoint context cannot safely
        # cache those side effects, so the MoE parallelizer leaves only this
        # decoder block eager while checkpointing every other Qwen3.8-Flash-Next block.
        self._nemo_disable_activation_checkpointing = ple is not None
        if self.layer_type == "linear_attention":
            self.linear_attn = Qwen3_8_FlashNextGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3_8_FlashNextQSAAttention(config, layer_idx, backend)
        else:
            raise ValueError(f"Unsupported Qwen3.8-Flash-Next layer type {self.layer_type!r}")

        self.mlp = MoE(moe_config, backend)
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        hc_kwargs = {
            "hidden_size": self.hidden_size,
            "hc_count": self.hc_count,
            "lowrank_size": int(getattr(config, "hc_lowrank")),
            "rms_norm_eps": float(getattr(config, "rms_norm_eps")),
            "backend": backend,
            "dtype": dtype,
        }
        self.attn_hyper_connection = Qwen3_8_FlashNextHyperConnection(**hc_kwargs)
        self.mlp_hyper_connection = Qwen3_8_FlashNextHyperConnection(**hc_kwargs)

    def _expand_initial_streams(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Expand a one-stream decoder input into the persistent HC layout.

        Args:
            hidden_states: Tensor of shape ``[batch, sequence, hidden_size]``
                or ``[batch, sequence, hc_count * hidden_size]``.

        Returns:
            Tensor of shape
            ``[batch, sequence, hc_count * hidden_size]``.
        """
        if hidden_states.shape[-1] == self.hidden_size * self.hc_count:
            return hidden_states
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                "Qwen3.8-Flash-Next decoder input must contain one stream or all HC streams; "
                f"got width {hidden_states.shape[-1]}"
            )
        return torch.cat([hidden_states] * self.hc_count, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cp_context: Qwen3_8_FlashNextCPContext | None = None,
        **attn_kwargs: object,
    ) -> torch.Tensor:
        """Run PLE, attention/GDN, and top-10 MoE updates.

        Args:
            hidden_states: One-stream input ``[batch, sequence, hidden]`` on
                the first layer, otherwise flattened HC streams
                ``[batch, sequence, hc_count * hidden]``.
            input_ids: Raw tokenizer IDs of shape ``[batch, sequence]`` used by
                the PLE hash path.
            freqs_cis: Composed rotary values ``[batch, sequence, rotary_dim]``
                whose final axis stores concatenated cosine and sine values.
            attention_mask: Optional token mask of shape ``[batch, sequence]``
                or backend-specific causal attention mask.
            padding_mask: Optional ``[batch, sequence]`` mask where ``True``
                marks padding for MoE dispatch.
            position_ids: Optional positions of shape ``[batch, sequence]`` or
                ``[axes, batch, sequence]``.
            cp_context: Optional contiguous CP metadata. Tensor-bearing fields
                contain replicated global raw IDs/padding of shape ``[batch,
                global_sequence]`` and identify this rank's local interval.
            **attn_kwargs: Attention backend metadata.

        Returns:
            Flattened HC streams of shape
            ``[batch, sequence, hc_count * hidden]``.
        """
        if attention_mask is not None and padding_mask is None and attention_mask.ndim <= 2:
            padding_mask = attention_mask.bool().logical_not()

        hidden_states = self._expand_initial_streams(hidden_states)
        if self.ple is not None:
            hidden_states = hidden_states + self.ple(hidden_states, input_ids, cp_context=cp_context)

        attn_input, attn_residual = self.attn_hyper_connection.mix(hidden_states)
        if self.layer_type == "linear_attention":
            packed_cu_seqlens = attn_kwargs.get("cu_seqlens")
            if packed_cu_seqlens is None and cp_context is not None:
                packed_cu_seqlens = getattr(cp_context, "global_cu_seqlens", None)
            if isinstance(packed_cu_seqlens, torch.Tensor):
                packed_cu_seqlens = packed_cu_seqlens.to(torch.long)
            attn_output = self.linear_attn(
                hidden_states=attn_input,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cu_seqlens=packed_cu_seqlens,
            )
        else:
            qsa_mask = attention_mask
            if qsa_mask is None and padding_mask is not None:
                qsa_mask = padding_mask.logical_not()
            attn_output = self.self_attn(
                x=attn_input,
                attention_mask=qsa_mask,
                freqs_cis=freqs_cis,
                cp_context=cp_context,
                **attn_kwargs,
            )
        hidden_states = self.attn_hyper_connection.combine(attn_output, attn_residual)

        mlp_input, mlp_residual = self.mlp_hyper_connection.mix(hidden_states)
        mlp_module = unwrap_checkpoint_wrapper(self.mlp)
        if not isinstance(mlp_module, MoE):
            raise TypeError(f"Qwen3.8-Flash-Next requires an MoE block, got {type(mlp_module).__name__}")
        mlp_output = self.mlp(mlp_input, padding_mask)
        return self.mlp_hyper_connection.combine(mlp_output, mlp_residual)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize this decoder layer for training from scratch.

        Args:
            buffer_device: Device used by attention/MoE initializers.
            init_std: Standard deviation for dense projection weights.
        """
        if self.layer_type == "full_attention":
            self.self_attn.init_weights(buffer_device, init_std=init_std)
        else:
            self.linear_attn.dt_bias.fill_(1.0)
            self.linear_attn.A_log.uniform_(0, 16).log_()
            for linear in (
                self.linear_attn.in_proj_qkv,
                self.linear_attn.in_proj_z,
                self.linear_attn.in_proj_b,
                self.linear_attn.in_proj_a,
                self.linear_attn.out_proj,
            ):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
            if hasattr(self.linear_attn.norm, "reset_parameters"):
                self.linear_attn.norm.reset_parameters()
            else:
                self.linear_attn.norm.weight.zero_()
        self.attn_hyper_connection.init_weights(init_std=init_std)
        self.mlp_hyper_connection.init_weights(init_std=init_std)
        self.mlp.init_weights(buffer_device)
