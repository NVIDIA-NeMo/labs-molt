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

"""Context-Parallel-aware wrapper for Qwen3.5 MoE GatedDeltaNet linear attention.

When a CP mesh is attached (via ``apply_cp``), the forward pass:
  1. Recovers dense sequence order from PyTorch's load-balanced CP layout using
     a local ``seq_index`` when provided, otherwise deriving it from the CP
     DualChunkSwap layout.
  2. Runs the causal conv1d and FLA gated delta rule on that dense ordering.
  3. Restores the output back to the original load-balanced CP layout.

When no CP mesh is set, the module delegates to the original HF forward.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.autograd import Function
from torch.distributed.device_mesh import DeviceMesh
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeGatedDeltaNet,
    causal_conv1d_fn,
    causal_conv1d_update,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from nemo_automodel.components.models.qwen3_5.packing import prepare_gated_delta_packed_metadata
from nemo_automodel.shared.utils import dtype_from_str

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.blockdiag_cp import BlockdiagCpModelState


class _AllGatherConcatFn(Function):
    """All-gather + concat with autograd-safe backward.

    The forward concatenates equal-sized local shards from all ranks along `dim`.
    Backward all-reduces the concatenated gradient across ranks, then slices out
    the local shard for the current rank.
    """

    @staticmethod
    def forward(ctx, local_tensor: torch.Tensor, group: dist.ProcessGroup, dim: int):
        dim = dim if dim >= 0 else local_tensor.ndim + dim
        world_size = dist.get_world_size(group)
        gathered = [torch.empty_like(local_tensor) for _ in range(world_size)]
        dist.all_gather(gathered, local_tensor.contiguous(), group=group)

        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.dim = dim
        ctx.local_dim_size = local_tensor.size(dim)
        return torch.cat(gathered, dim=dim)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_full = grad_output.contiguous()
        dist.all_reduce(grad_full, op=dist.ReduceOp.SUM, group=ctx.group)
        start = ctx.rank * ctx.local_dim_size
        grad_local = grad_full.narrow(ctx.dim, start, ctx.local_dim_size).contiguous()
        return grad_local, None, None


class _SSMGateParam:
    """Get-only (non-data) descriptor exposing an ``SSMGate`` param as an attribute.

    Lets ``self.A_log`` / ``self.dt_bias`` resolve to the fp32 ``SSMGate`` holder
    (``self._fp32_params``) without a ``__getattr__`` monkeypatch. Being a non-data
    descriptor, it does not intercept assignment, so HF's ``__init__`` doing
    ``self.A_log = nn.Parameter(...)`` still routes through ``nn.Module.__setattr__``
    into ``_parameters`` (where it lives until ``install_ssm_gate`` moves it).
    """

    def __init__(self, name: str):
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        return getattr(obj._fp32_params, self.name)


class CPAwareGatedDeltaNet(Qwen3_5MoeGatedDeltaNet):
    """Drop-in replacement for ``Qwen3_5MoeGatedDeltaNet`` with FLA Context Parallelism.

    The SSM-gating params (``A_log``/``dt_bias``) are moved into a fp32 ``SSMGate``
    submodule (``_fp32_params``) at construction so they keep fp32 storage (master
    weights) even under a bf16 bulk dtype, and so FSDP can shard them in their own
    dtype-uniform fp32 group. ``A_log``/``dt_bias`` remain readable as attributes via
    get-only descriptors that resolve to the submodule — no ``__getattr__`` patch.

    ``_cp_mesh`` is set externally by the parallelizer to enable context parallelism.
    """

    _cp_mesh: DeviceMesh | None
    # Get-only (non-data) descriptors: reads resolve to the fp32 ``SSMGate`` holder,
    # while writes during HF ``__init__`` (``self.A_log = nn.Parameter(...)``) still
    # land in ``_parameters`` (handled by ``nn.Module.__setattr__``) before we move
    # them into the holder.
    A_log = _SSMGateParam("A_log")
    dt_bias = _SSMGateParam("dt_bias")

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self._cp_mesh = None
        # Transformers 5.15 moved kernel callables from instance attributes to decorated
        # module-level fallbacks. Prefer the installed kernels, matching the pre-5.15
        # behavior and the FLA implementation used by the context-parallel path.
        try:
            from causal_conv1d import causal_conv1d_fn as fast_causal_conv1d_fn
            from causal_conv1d import causal_conv1d_update as fast_causal_conv1d_update
        except ImportError:
            fast_causal_conv1d_fn = causal_conv1d_fn
            fast_causal_conv1d_update = causal_conv1d_update

        try:
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
        except ImportError:
            chunk_gated_delta_rule = torch_chunk_gated_delta_rule
            fused_recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule

        self.causal_conv1d_fn = fast_causal_conv1d_fn
        self.causal_conv1d_update = fast_causal_conv1d_update
        self.chunk_gated_delta_rule = chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule
        # HF created bare ``A_log``/``dt_bias`` in ``_parameters``; move them into a
        # native fp32 ``SSMGate`` submodule (built directly, not relocated at runtime).
        install_ssm_gate(self, fp32_dtype=_resolve_ssm_dtype(config))

    def _compute_gate(self, a: torch.Tensor) -> torch.Tensor:
        """Compute the gating value ``g`` via the fp32 ``SSMGate`` submodule.

        Computing inside the submodule's forward keeps FSDP's unshard/reshard
        lifecycle natural for the isolated fp32 group.
        """
        return self._fp32_params(a)

    def _forward_no_cp(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """HF GatedDeltaNet forward with FSDP-safe fp32 gate computation.

        Args:
            hidden_states: Hidden states of shape [batch, sequence, hidden].
            cache_params: Optional Hugging Face recurrent cache.
            cache_position: Optional token positions of shape [sequence]; unused
                by the Transformers 5.5-compatible path.
            attention_mask: Optional validity or indexed document mask of shape
                [batch, sequence].
            cu_seqlens: Optional cumulative document lengths of shape
                [documents + 1] on the compute device.
            cu_seqlens_cpu: Optional CPU mirror of ``cu_seqlens`` with shape
                [documents + 1] for FLA host-side chunk planning.
            indices: Optional flattened valid-token indices of shape [tokens] on
                the compute device.

        Returns:
            GatedDeltaNet output of shape [batch, sequence, hidden].
        """
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

        batch_size, seq_len, hidden_dim = hidden_states.shape

        use_precomputed_states = (
            cache_params is not None and cache_params.has_previous_state(self.layer_idx) and seq_len == 1
        )

        # Resolve packing kwargs. Fallback to mask-derivation only when neither
        # was passed in (bypasses the decoder-layer subclass).
        if not use_precomputed_states and cu_seqlens is None and indices is None:
            packed_metadata = prepare_gated_delta_packed_metadata(attention_mask, None)
            if packed_metadata is not None:
                cu_seqlens = packed_metadata.cu_seqlens
                cu_seqlens_cpu = packed_metadata.cu_seqlens_cpu
                indices = packed_metadata.indices

        if cu_seqlens is not None and cu_seqlens_cpu is None:
            cu_seqlens_cpu = cu_seqlens.detach().cpu()
        if cu_seqlens_cpu is not None and cu_seqlens_cpu.device.type != "cpu":
            raise ValueError("cu_seqlens_cpu must reside on CPU for FLA host-side chunk planning.")

        is_packed = (not use_precomputed_states) and cu_seqlens is not None
        # Only unpad when there is actually padding to remove. For B=1 packs
        # without padding, ``indices`` covers ``[0, B*T)`` and we keep the
        # ``[B, T, ...]`` layout (bit-for-bit identical to the prior fast path).
        needs_unpad = is_packed and indices is not None and indices.numel() != batch_size * seq_len

        # Padding-token zero-out: skip under packing because either we unpad
        # (which drops padding entirely) or there is no padding to begin with.
        # Outside packing the original behavior is preserved.
        if not is_packed:
            # transformers 5.15 dropped apply_mask_to_padding_states' shape guards, so it
            # now multiplies by ``attention_mask[:, :, None]`` unconditionally. The helper
            # documents a 2D padding mask; the 4D packed causal mask (materialized at
            # cp_size<=1) broadcasts to 5D and raises. Only pass a 2D mask, matching the
            # pre-5.15 behavior of skipping it for anything else.
            if attention_mask is None or attention_mask.dim() == 2:
                hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        # Unpad on entry: ``[B, T, H] -> [1, total_valid, H]``. All projections,
        # conv1d, FLA and norm below run in this dense layout.
        if needs_unpad:
            hidden_states = hidden_states.reshape(batch_size * seq_len, hidden_dim)[indices].unsqueeze(0)

        mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        z = self.in_proj_z(hidden_states)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        eff_batch, eff_seq_len = mixed_qkv.shape[0], mixed_qkv.shape[2]
        z = z.reshape(eff_batch, eff_seq_len, -1, self.head_v_dim)

        if use_precomputed_states:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if cache_params is not None:
                conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                # ``seq_idx`` for causal_conv1d_fn marks per-token segment ids.
                # Source is the indexed mask, gathered at the same ``indices``
                # used for unpadding so it lines up with the unpadded layout.
                if not is_packed:
                    seq_idx_for_conv = None
                elif needs_unpad:
                    seq_idx_for_conv = attention_mask.reshape(-1)[indices].unsqueeze(0).to(torch.int32).contiguous()
                else:
                    seq_idx_for_conv = attention_mask.to(torch.int32).contiguous()
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=seq_idx_for_conv,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :eff_seq_len])

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)

        query = query.reshape(eff_batch, eff_seq_len, -1, self.head_k_dim)
        key = key.reshape(eff_batch, eff_seq_len, -1, self.head_k_dim)
        value = value.reshape(eff_batch, eff_seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = self._compute_gate(a)

        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if not use_precomputed_states:
            if is_packed:
                # FLA requires ``q.shape[0] == 1`` when cu_seqlens is supplied.
                # When already unpadded eff_batch==1; otherwise flatten now.
                if not needs_unpad:
                    query = query.reshape(1, batch_size * seq_len, *query.shape[2:])
                    key = key.reshape(1, batch_size * seq_len, *key.shape[2:])
                    value = value.reshape(1, batch_size * seq_len, *value.shape[2:])
                    g = g.reshape(1, batch_size * seq_len, *g.shape[2:])
                    beta = beta.reshape(1, batch_size * seq_len, *beta.shape[2:])
                core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=cache_params is not None,
                    use_qk_l2norm_in_kernel=True,
                    cu_seqlens=cu_seqlens,
                    cu_seqlens_cpu=cu_seqlens_cpu,
                )
                if not needs_unpad:
                    core_attn_out = core_attn_out.reshape(batch_size, seq_len, *core_attn_out.shape[2:])
            else:
                core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    initial_state=None,
                    output_final_state=cache_params is not None,
                    use_qk_l2norm_in_kernel=True,
                )
        else:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(eff_batch, eff_seq_len, -1)
        output = self.out_proj(core_attn_out)

        # Repad on exit: scatter ``[1, total_valid, H]`` back into ``[B, T, H]``.
        if needs_unpad:
            output = output.squeeze(0)
            padded = torch.zeros(
                batch_size * seq_len,
                output.shape[-1],
                dtype=output.dtype,
                device=output.device,
            )
            padded.index_copy_(0, indices, output)
            output = padded.reshape(batch_size, seq_len, -1)

        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position=None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        qkv_format: str | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cu_seqlens_cpu: torch.Tensor | None = None,
        indices: torch.Tensor | None = None,
        seq_index: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run GatedDeltaNet with dense, round-robin CP, or packed contiguous CP.

        Args:
            hidden_states: Hidden states of shape ``[batch, sequence, hidden]``;
                under CP, ``sequence`` is this rank's local shard.
            cache_params: Optional Hugging Face recurrent cache.
            cache_position: Optional token positions of shape ``[sequence]``.
            attention_mask: Optional validity or document mask of shape
                ``[batch, sequence]``.
            position_ids: Optional positions of shape ``[batch, sequence]`` or
                ``[axes, batch, sequence]``.
            qkv_format: Optional attention layout tag; unused by this module.
            cu_seqlens: Optional packed boundaries of shape
                [documents + 1] for the non-CP path.
            cu_seqlens_cpu: Optional CPU mirror of ``cu_seqlens`` with shape
                [documents + 1] for FLA host-side chunk planning.
            indices: Optional valid-token indices of shape ``[tokens]`` for the
                non-CP packed path.
            seq_index: Optional global token positions of shape ``[sequence]``
                or ``[batch, sequence]`` for round-robin CP restoration.

        Returns:
            Hidden states of shape ``[batch, sequence, hidden]`` in the same
            local sequence layout as ``hidden_states``.
        """
        # Fast path: no CP → run HF forward with fp32-safe gate computation.
        if self._cp_mesh is None or self._cp_mesh.size() <= 1:
            return self._forward_no_cp(
                hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                indices=indices,
            )

        from nemo_automodel.components.distributed.blockdiag_cp import current_blockdiag_cp_state

        return self._forward_with_cp(
            hidden_states,
            position_ids=position_ids,
            seq_index=seq_index,
            blockdiag_state=current_blockdiag_cp_state(),
        )

    # ------------------------------------------------------------------
    # Conv1d boundary communication
    # ------------------------------------------------------------------
    def _conv1d_with_cp(
        self,
        mixed_qkv: torch.Tensor,
        cp_context,
    ) -> torch.Tensor:
        """Run causal conv1d via FLA's CP-aware conv implementation.

        Args:
            mixed_qkv: [B, D, S_local] tensor (channels-first for conv).
            cp_context: FLA CP context built by ``build_cp_context``.

        Returns:
            [B, D, S_local] conv output with correct boundary handling.
        """
        from fla.modules.convolution import causal_conv1d as fla_causal_conv1d

        conv_in = mixed_qkv.transpose(1, 2).contiguous()  # [B, S_local, D]
        conv_outs = []
        for bi in range(conv_in.shape[0]):
            out_bi, _ = fla_causal_conv1d(
                x=conv_in[bi : bi + 1],
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                cp_context=cp_context,
            )
            conv_outs.append(out_bi)

        return torch.cat(conv_outs, dim=0).transpose(1, 2).contiguous()

    def _extract_local_seq_index(
        self,
        seq_index: torch.Tensor | None,
        seq_len: int,
    ) -> torch.Tensor | None:
        if seq_index is None:
            return None

        if seq_index.ndim == 1:
            local_positions = seq_index
        elif seq_index.ndim == 2:
            local_positions = seq_index[0]
        else:
            return None

        if local_positions.shape[-1] == seq_len:
            return local_positions.to(dtype=torch.long)

        return None

    def _build_dual_chunk_local_positions(
        self,
        *,
        seq_len: int,
        cp_size: int,
        cp_rank: int,
        device: torch.device,
    ) -> torch.Tensor:
        if seq_len % 2 != 0:
            raise RuntimeError(
                f"Qwen3.5 CP linear-attn layer {self.layer_idx} expected an even local sequence length "
                "from DualChunkSwap CP layout."
            )

        chunk_len = seq_len // 2
        first_chunk = cp_rank
        second_chunk = 2 * cp_size - 1 - cp_rank
        return torch.cat(
            (
                torch.arange(
                    first_chunk * chunk_len,
                    (first_chunk + 1) * chunk_len,
                    device=device,
                    dtype=torch.long,
                ),
                torch.arange(
                    second_chunk * chunk_len,
                    (second_chunk + 1) * chunk_len,
                    device=device,
                    dtype=torch.long,
                ),
            ),
            dim=0,
        )

    def _all_gather_concat(
        self,
        tensor: torch.Tensor,
        cp_group: dist.ProcessGroup,
        *,
        dim: int,
        differentiable: bool = False,
    ) -> torch.Tensor:
        if differentiable:
            return _AllGatherConcatFn.apply(tensor, cp_group, dim)

        cp_world = dist.get_world_size(cp_group)
        gathered = [torch.empty_like(tensor) for _ in range(cp_world)]
        dist.all_gather(gathered, tensor.contiguous(), group=cp_group)
        return torch.cat(gathered, dim=dim)

    def _undo_attention_load_balancing(
        self,
        hidden_states: torch.Tensor,
        original_positions: torch.Tensor,
        cp_group: dist.ProcessGroup,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cp_rank = dist.get_rank(cp_group)
        seq_len = hidden_states.shape[1]

        cp_order_hidden = self._all_gather_concat(hidden_states, cp_group, dim=1, differentiable=True)
        cp_order_positions = self._all_gather_concat(original_positions, cp_group, dim=0)

        sort_order = torch.argsort(cp_order_positions)
        sorted_positions = cp_order_positions.index_select(0, sort_order)
        expected_positions = torch.arange(
            sorted_positions.numel(),
            device=sorted_positions.device,
            dtype=sorted_positions.dtype,
        )
        if not torch.equal(sorted_positions, expected_positions):
            raise RuntimeError(
                f"Qwen3.5 CP linear-attn layer {self.layer_idx} requires dense global token positions "
                "covering 0..S-1 after gathering CP shards."
            )
        full_hidden = cp_order_hidden.index_select(1, sort_order)

        start = cp_rank * seq_len
        end = start + seq_len
        return full_hidden[:, start:end], sorted_positions

    def _redo_attention_load_balancing(
        self,
        output: torch.Tensor,
        original_positions: torch.Tensor,
        sorted_positions: torch.Tensor,
        cp_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        full_output = self._all_gather_concat(output, cp_group, dim=1, differentiable=True)
        restore_indices = torch.searchsorted(sorted_positions, original_positions)
        restored_positions = sorted_positions.index_select(0, restore_indices)
        if not torch.equal(restored_positions, original_positions):
            raise RuntimeError(
                f"Failed to restore Qwen3.5 CP linear-attn output on layer {self.layer_idx}: "
                "sorted positions do not cover the local CP layout."
            )
        return full_output.index_select(1, restore_indices)

    # ------------------------------------------------------------------
    # CP-aware forward
    # ------------------------------------------------------------------
    def _forward_with_cp(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        seq_index: torch.Tensor | None,
        blockdiag_state: BlockdiagCpModelState | None = None,
    ) -> torch.Tensor:
        """Run FLA GatedDeltaNet over a context-parallel sequence shard.

        Args:
            hidden_states: Local hidden states of shape ``[batch, local_sequence,
                hidden]``. The sequence axis uses contiguous rank order when
                ``blockdiag_state`` is active, otherwise PyTorch CP's
                head-tail round-robin order.
            position_ids: Optional local positions of shape ``[batch,
                local_sequence]`` or mRoPE positions of shape ``[axes, batch,
                local_sequence]``.
            seq_index: Optional global token indices of shape
                ``[local_sequence]`` or ``[batch, local_sequence]`` for the
                round-robin layout.
            blockdiag_state: Optional packed CP state whose cumulative-length
                tensors have shape ``[num_documents + 1]`` and reset conv/GDN
                recurrence at document boundaries.

        Returns:
            Local GatedDeltaNet output of shape ``[batch, local_sequence,
            hidden]`` in the same sequence layout as ``hidden_states``.
        """
        from fla.ops.cp import build_cp_context
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_chunk_gated_delta_rule

        batch_size, seq_len, _ = hidden_states.shape

        cp_size = self._cp_mesh.size()
        cp_group = blockdiag_state.group if blockdiag_state is not None else self._cp_mesh.get_group()
        group_size = dist.get_world_size(cp_group)
        if group_size != cp_size:
            raise RuntimeError(
                f"Qwen3.5-MoE GDN CP process-group size ({group_size}) does not match cp_mesh.size() ({cp_size})"
            )
        cp_rank = dist.get_rank(cp_group)

        uses_contiguous_layout = blockdiag_state is not None
        local_positions = None
        if not uses_contiguous_layout:
            local_positions = self._extract_local_seq_index(seq_index, seq_len)
            if local_positions is None:
                local_positions = self._build_dual_chunk_local_positions(
                    seq_len=seq_len,
                    cp_size=cp_size,
                    cp_rank=cp_rank,
                    device=hidden_states.device,
                )

        # ---- Build FLA CP context (once, reused for every sequence) ----
        # After undoing the load-balanced attention layout, each rank again owns a
        # contiguous chunk of a dense global sequence of length seq_len * cp_size.
        global_seq_len = seq_len * cp_size
        if uses_contiguous_layout:
            cu_seqlens = blockdiag_state.packed_cu_seqlens
            cp_context = build_cp_context(
                cu_seqlens=cu_seqlens,
                group=cp_group,
                conv1d_kernel_size=self.conv_kernel_size,
                cu_seqlens_cpu=blockdiag_state.packed_cu_seqlens_cpu,
            )
        else:
            cu_seqlens_single = torch.tensor(
                [0, global_seq_len],
                dtype=torch.long,
                device=hidden_states.device,
            )
            cp_context = build_cp_context(
                cu_seqlens=cu_seqlens_single,
                group=cp_group,
                conv1d_kernel_size=self.conv_kernel_size,
            )
        # Attention runs on a load-balanced CP layout, but conv + recurrent state
        # propagation require rank-order sequential tokens.
        sorted_positions = None
        if not uses_contiguous_layout:
            hidden_states, sorted_positions = self._undo_attention_load_balancing(
                hidden_states,
                local_positions,
                cp_group,
            )

        # ---- Projections (batched, pointwise) ----
        mixed_qkv = self.in_proj_qkv(hidden_states)  # [B, S_local, conv_dim]
        z = self.in_proj_z(hidden_states)  # [B, S_local, value_dim]
        b = self.in_proj_b(hidden_states)  # [B, S_local, num_v_heads]
        a = self.in_proj_a(hidden_states)  # [B, S_local, num_v_heads]

        # ---- Causal Conv1d with cross-rank boundary exchange ----
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, D, S_local]
        mixed_qkv = self._conv1d_with_cp(mixed_qkv, cp_context)  # [B, D, S_local]
        mixed_qkv = mixed_qkv.transpose(1, 2)  # [B, S_local, D]

        # ---- Split QKV ----
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )

        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)

        # ---- Gate & beta ----
        beta = b.sigmoid()
        g = self._compute_gate(a)

        # GVA: repeat q/k heads to match v heads
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        # ---- Chunk GDN with CP (per-sequence) ----
        # cp_context is built for a single sequence; reuse for each batch element.
        attn_outs = []
        for bi in range(batch_size):
            out_bi, _ = fla_chunk_gated_delta_rule(
                query[bi : bi + 1],
                key[bi : bi + 1],
                value[bi : bi + 1],
                g=g[bi : bi + 1],
                beta=beta[bi : bi + 1],
                initial_state=None,
                output_final_state=False,
                use_qk_l2norm_in_kernel=True,
                cp_context=cp_context,
            )
            attn_outs.append(out_bi)
        core_attn_out = torch.cat(attn_outs, dim=0)  # [B, S_local, H_v, D_v]

        # ---- Gated RMSNorm + output projection ----
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        output = self.out_proj(core_attn_out)
        if not uses_contiguous_layout:
            output = self._redo_attention_load_balancing(
                output,
                local_positions,
                sorted_positions,
                cp_group=cp_group,
            )
        return output


# SSM-gating params kept in fp32 storage (regardless of the model's bulk dtype)
# and isolated in the ``_fp32_params`` SSMGate submodule for FSDP.
_FP32_PARAM_NAMES = ("A_log", "dt_bias")


class SSMGate(torch.nn.Module):
    """Owns the fp32 SSM-gating params (``A_log``/``dt_bias``) and computes the gate.

    Keeping these in a dedicated submodule lets FSDP shard them in their own
    dtype-uniform fp32 group (true master weights), and computing the gate inside
    ``forward`` keeps FSDP's unshard/reshard lifecycle natural.
    """

    def __init__(self, num_v_heads: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.A_log = torch.nn.Parameter(torch.empty(num_v_heads, dtype=dtype))
        self.dt_bias = torch.nn.Parameter(torch.empty(num_v_heads, dtype=dtype))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)


def install_ssm_gate(mod, fp32_dtype=torch.float32):
    """Move ``mod``'s HF-created bare ``A_log``/``dt_bias`` into a fp32 ``SSMGate``.

    HF's GatedDeltaNet ``__init__`` creates ``A_log``/``dt_bias`` as bare params in
    ``mod._parameters``. This relocates them into an :class:`SSMGate` submodule
    registered as ``_fp32_params`` (casting to ``fp32_dtype``), so they keep fp32
    storage under a bf16 bulk dtype and get their own dtype-uniform FSDP group.
    Attribute access (``self.A_log``/``self.dt_bias``) continues to work via the
    :class:`_SSMGateParam` descriptors on ``CPAwareGatedDeltaNet`` — no
    ``__getattr__`` patch. Returns the gate submodule.
    """
    num_v_heads = mod._parameters["A_log"].shape[0]
    gate = SSMGate(num_v_heads, dtype=fp32_dtype)
    for pname in _FP32_PARAM_NAMES:
        param = mod._parameters.pop(pname)
        if param.dtype != fp32_dtype:
            param.data = param.data.to(fp32_dtype)
        setattr(gate, pname, param)  # overwrite the freshly-built empty param
    mod.add_module("_fp32_params", gate)
    return gate


def _resolve_ssm_dtype(config):
    """Resolve the fp32 storage dtype for the SSM-gating params from ``config``.

    Honors ``mamba_ssm_dtype`` when present; otherwise defaults to AutoModel's
    fp32 training-storage contract for ``A_log``/``dt_bias``.
    """
    ssm_dtype = getattr(config, "mamba_ssm_dtype", None)
    if isinstance(ssm_dtype, str):
        ssm_dtype = dtype_from_str(ssm_dtype)
    return ssm_dtype or torch.float32
