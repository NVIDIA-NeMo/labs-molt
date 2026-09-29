# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Context parallelism for Mamba/SSM layers using a hidden-parallel strategy.

Instead of splitting the sequence across CP ranks (as attention CP does), this module
uses an all-to-all redistribution so that each CP rank processes the *full* sequence
but only a *subset* of heads (d_inner / cp_size).  The data flow is::

    [B, L_local, D]  -->  all-to-all  -->  [B, L_global, D/cp]
        -->  conv1d + SSM kernel  -->
    [B, L_global, D/cp]  -->  all-to-all  -->  [B, L_local, D]

This module is intentionally **not** a subclass of ``nn.Module`` because it owns
no trainable parameters.  It holds *references* to the Mamba mixer's parameters
and slices them in the forward path so that gradients flow back to the full
(unsliced) parameters.
"""

import torch
import torch.distributed
import torch.nn as nn

# ---------------------------------------------------------------------------
# Autograd-aware all-to-all primitive
# ---------------------------------------------------------------------------


class _AllToAll(torch.autograd.Function):
    """Autograd wrapper around ``torch.distributed.all_to_all_single``.

    For equal-sized splits the all-to-all operation is its own inverse,
    so the backward pass is simply another all-to-all on the same group.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: torch.distributed.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        output = torch.empty_like(input_)
        torch.distributed.all_to_all_single(output, input_, group=group)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        group = ctx.group
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(grad_output)
        torch.distributed.all_to_all_single(grad_input, grad_output, group=group)
        return grad_input, None


def _all_to_all(input_: torch.Tensor, group: torch.distributed.ProcessGroup) -> torch.Tensor:
    """Functional entry-point for the autograd-aware all-to-all."""
    return _AllToAll.apply(input_, group)


# ---------------------------------------------------------------------------
# Sequence-sharded <-> Hidden-sharded layout transformations (batch-first)
# ---------------------------------------------------------------------------


def _all_to_all_cp2hp(
    input_: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    batch_size: int,
) -> torch.Tensor:
    """Transform from sequence-sharded to hidden-sharded layout (batch-first).

    Args:
        input_: Tensor of shape ``[B, L_local, H]`` (BSHD) or ``[T, H]`` (THD)
            where H is the full hidden dimension on this rank.
        cp_group: Context-parallel process group.
        batch_size: Batch size ``B`` (needed to recover dimensions after reshape).

    Returns:
        Tensor of shape ``[B, L_global, H / cp_size]`` (BSHD) or ``[T, H / cp_size]`` (THD).
    """
    is_2d = input_.dim() == 2
    if is_2d:
        input_ = input_.unsqueeze(0)

    cp_size = cp_group.size()
    B, L_local, H = input_.shape
    H_local = H // cp_size

    # [B*L_local, cp, H_local] -> [cp, B*L_local, H_local] -> flatten for all-to-all
    send_tensor = (
        input_.reshape(B * L_local, cp_size, H_local)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(cp_size * B * L_local, H_local)
    )

    recv_tensor = _all_to_all(send_tensor, cp_group)

    # [cp, B, L_local, H_local] -> [B, cp*L_local, H_local]
    result = (
        recv_tensor.reshape(cp_size, B, L_local, H_local)
        .permute(1, 0, 2, 3)
        .contiguous()
        .reshape(B, cp_size * L_local, H_local)
    )

    if is_2d:
        result = result.squeeze(0)
    return result


def _all_to_all_hp2cp(
    input_: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
    batch_size: int,
) -> torch.Tensor:
    """Transform from hidden-sharded to sequence-sharded layout (batch-first).

    This is the inverse of :func:`_all_to_all_cp2hp`.

    Args:
        input_: Tensor of shape ``[B, L_global, H_local]`` (BSHD) or ``[T, H_local]`` (THD)
            where ``H_local = H / cp_size``.
        cp_group: Context-parallel process group.
        batch_size: Batch size ``B``.

    Returns:
        Tensor of shape ``[B, L_local, H]`` (BSHD) or ``[T, H]`` (THD)
        where ``L_local = L_global / cp_size`` and ``H = H_local * cp_size``.
    """
    is_2d = input_.dim() == 2
    if is_2d:
        input_ = input_.unsqueeze(0)

    cp_size = cp_group.size()
    B, L_global, H_local = input_.shape
    L_local = L_global // cp_size

    # [B, cp, L_local, H_local] -> [cp, B, L_local, H_local] -> flatten for all-to-all
    send_tensor = (
        input_.reshape(B, cp_size, L_local, H_local)
        .permute(1, 0, 2, 3)
        .contiguous()
        .reshape(cp_size * B * L_local, H_local)
    )

    recv_tensor = _all_to_all(send_tensor, cp_group)

    # [cp, B*L_local, H_local] -> [B*L_local, cp, H_local] -> [B, L_local, H]
    result = (
        recv_tensor.reshape(cp_size, B * L_local, H_local)
        .permute(1, 0, 2)
        .contiguous()
        .reshape(B, L_local, cp_size * H_local)
    )

    if is_2d:
        result = result.squeeze(0)
    return result


def _reorder_chunks(
    input_: torch.Tensor,
    order: list[int],
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reorder equal-sized chunks of a tensor according to *order*.

    Args:
        input_: ``[B, L, H]`` (BSHD) or ``[T, H]`` (THD).
        order: Permutation indices (length must equal the number of chunks).
        cu_seqlens: If provided, reorder per-sequence on dim=0 (THD).
    """
    num_chunks = len(order)

    if cu_seqlens is not None:
        parts = []
        for i in range(len(cu_seqlens) - 1):
            start, end = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
            chunks = input_[start:end].split((end - start) // num_chunks, dim=0)
            parts.append(torch.cat([chunks[j] for j in order], dim=0))
        return torch.cat(parts, dim=0)

    chunks = torch.chunk(input_, chunks=num_chunks, dim=1)
    return torch.cat([chunks[i] for i in order], dim=1)


def _deinterleave_packed_seqs(
    input_: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cp_size: int,
) -> torch.Tensor:
    """Rearrange tokens from rank-major to sequence-major order after all-to-all.

    After ``_all_to_all_cp2hp`` on packed 2-D data the token layout along
    the sequence dimension is::

        [rank0_seq0 | rank0_seq1 | ... | rank1_seq0 | rank1_seq1 | ...]

    This function rearranges to::

        [rank0_seq0 | rank1_seq0 | ... | rank0_seq1 | rank1_seq1 | ...]

    so that each sequence's tokens are contiguous (required by the
    ``_undo_attention_load_balancing`` reorder that follows).

    Args:
        input_: 2-D tensor ``[T_global, H]``.
        cu_seqlens: **Local** (pre-all-to-all) cumulative sequence lengths.
        cp_size: Context-parallel world size.

    Returns:
        Rearranged 2-D tensor with the same shape.
    """
    num_seqs = len(cu_seqlens) - 1
    if num_seqs <= 1:
        return input_

    # Each rank contributes a contiguous block of T_local tokens.
    T_local = input_.shape[0] // cp_size

    parts: list[torch.Tensor] = []
    for s in range(num_seqs):
        start_s = cu_seqlens[s].item()
        end_s = cu_seqlens[s + 1].item()
        for r in range(cp_size):
            offset = r * T_local
            parts.append(input_[offset + start_s : offset + end_s])
    return torch.cat(parts, dim=0)


def _reinterleave_packed_seqs(
    input_: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cp_size: int,
) -> torch.Tensor:
    """Inverse of :func:`_deinterleave_packed_seqs`.

    Rearranges from sequence-major back to rank-major order before the
    inverse all-to-all in ``post_conv_ssm``.
    """
    num_seqs = len(cu_seqlens) - 1
    if num_seqs <= 1:
        return input_

    # Build per-rank blocks by gathering each rank's portion of every sequence.
    seq_lens = [(cu_seqlens[s + 1] - cu_seqlens[s]).item() for s in range(num_seqs)]
    # In the deinterleaved (sequence-major) layout each sequence occupies
    # seq_len_local * cp_size contiguous tokens.
    rank_blocks: list[list[torch.Tensor]] = [[] for _ in range(cp_size)]
    offset = 0
    for s in range(num_seqs):
        for r in range(cp_size):
            rank_blocks[r].append(input_[offset : offset + seq_lens[s]])
            offset += seq_lens[s]
    return torch.cat([torch.cat(rb, dim=0) for rb in rank_blocks], dim=0)


def _undo_attention_load_balancing(
    input_: torch.Tensor,
    cp_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reorder from DualChunkSwap to sequential for SSM processing."""
    num_chunks = 2 * cp_size
    order = [2 * i for i in range(cp_size)] + [num_chunks - 1 - 2 * i for i in range(cp_size)]
    return _reorder_chunks(input_, order, cu_seqlens)


def _redo_attention_load_balancing(
    input_: torch.Tensor,
    cp_size: int,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reorder from sequential back to DualChunkSwap for attention.

    Inverse of :func:`_undo_attention_load_balancing`.
    """
    num_chunks = 2 * cp_size
    order = [None] * num_chunks
    order[::2] = range(cp_size)
    order[1::2] = reversed(range(cp_size, num_chunks))
    return _reorder_chunks(input_, order, cu_seqlens)


# ---------------------------------------------------------------------------
# MambaContextParallel – orchestrates CP for a single Mamba mixer layer
# ---------------------------------------------------------------------------


class MambaContextParallel:
    """Hidden-parallel context parallelism for a Mamba2 mixer layer.

    This class does **not** own trainable parameters.  It stores a *reference*
    to the mixer module and accesses its parameters (conv1d, dt_bias, A_log, D)
    on the fly so that gradients propagate to the original (full) parameters
    and FSDP-managed DTensor replacements are picked up correctly.

    DualChunkSwap reordering is always undone before the SSM kernel and redone
    after, because both TE CP (p2p) and PyTorch's ``context_parallel(allgather)``
    reorder sequence chunks for load balancing.

    Args:
        cp_group: Context-parallel process group.
        num_heads: Total number of SSM heads (before any parallelism).
        head_dim: Dimension per head.
        n_groups: Number of SSM groups (for grouped B/C states).
        d_state: SSM state dimension.
        mixer: Reference to the Mamba mixer module (owns conv1d, dt_bias, A_log, D).
    """

    def __init__(
        self,
        cp_group: torch.distributed.ProcessGroup,
        num_heads: int,
        head_dim: int,
        n_groups: int,
        d_state: int,
        mixer: nn.Module,
    ) -> None:
        self.cp_group = cp_group
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.n_groups = n_groups
        self.d_state = d_state
        self._mixer = mixer

        self.cp_size = cp_group.size()
        self.cp_rank = cp_group.rank()

        self.d_inner = num_heads * head_dim

        # --- Validate and compute per-rank sizes ---

        # Each CP rank must get at least one head.
        assert num_heads % self.cp_size == 0, f"num_heads ({num_heads}) must be divisible by cp_size ({self.cp_size})"
        self.num_heads_local = num_heads // self.cp_size
        self.d_inner_local = self.num_heads_local * head_dim

        # Groups: when n_groups < cp_size we need to replicate B/C states.
        if n_groups < self.cp_size:
            assert self.cp_size % n_groups == 0, (
                f"cp_size ({self.cp_size}) must be divisible by n_groups ({n_groups}) when n_groups < cp_size"
            )
            self.group_repeat_count = self.cp_size // n_groups
            self.n_groups_local = 1
        else:
            assert n_groups % self.cp_size == 0, f"n_groups ({n_groups}) must be divisible by cp_size ({self.cp_size})"
            self.group_repeat_count = 1
            self.n_groups_local = n_groups // self.cp_size

    # ------------------------------------------------------------------ #
    #  Activation transforms (before / after conv+SSM)                    #
    # ------------------------------------------------------------------ #

    def pre_conv_ssm(
        self,
        projected_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Redistribute from sequence-sharded to hidden-sharded layout, undoing DualChunkSwap."""
        if self.cp_size == 1:
            return projected_states

        B = 1 if cu_seqlens is not None else projected_states.shape[0]
        groups_state_size = self.n_groups * self.d_state

        z, x, B_state, C_state, dt = torch.split(
            projected_states,
            [self.d_inner, self.d_inner, groups_state_size, groups_state_size, self.num_heads],
            dim=-1,
        )

        # Replicate B and C group states when n_groups < cp_size so that
        # replicas land on consecutive CP ranks with their associated heads.
        if self.group_repeat_count > 1:
            B_state = self._repeat_group_state(B_state)
            C_state = self._repeat_group_state(C_state)

        z = _all_to_all_cp2hp(z, self.cp_group, B)
        x = _all_to_all_cp2hp(x, self.cp_group, B)
        B_state = _all_to_all_cp2hp(B_state, self.cp_group, B)
        C_state = _all_to_all_cp2hp(C_state, self.cp_group, B)
        dt = _all_to_all_cp2hp(dt, self.cp_group, B)

        result = torch.cat([z, x, B_state, C_state, dt], dim=-1)
        # After all-to-all the tensor has global sequence length (L_local * cp_size
        # per sequence).  Undo DualChunkSwap so the SSM kernel sees tokens in
        # their true sequential order.
        if cu_seqlens is not None:
            # cu_seqlens from the batch is GLOBAL (pre-TE-partitioning).
            # After TE CP sharding each sequence has L/cp_size local tokens,
            # so derive local cu_seqlens for the deinterleave which operates
            # on the all-to-all output (cp_size local chunks per sequence).
            cu_seqlens_local = cu_seqlens // self.cp_size
            # For packed data (multiple sequences), the all-to-all produces a
            # rank-major layout where sequences from different ranks are
            # interleaved.  Deinterleave so each sequence is contiguous
            # before applying the DualChunkSwap undo.
            result = _deinterleave_packed_seqs(result, cu_seqlens_local, self.cp_size)
            cu_seqlens_global = cu_seqlens  # already global
        else:
            cu_seqlens_global = None
        result = _undo_attention_load_balancing(result, self.cp_size, cu_seqlens=cu_seqlens_global)

        return result

    def post_conv_ssm(
        self,
        output: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Redistribute SSM output from hidden-sharded back to sequence-sharded layout."""
        if self.cp_size == 1:
            return output

        B = 1 if cu_seqlens is not None else output.shape[0]
        # Redo DualChunkSwap before the inverse all-to-all so that the
        # sequence-parallel layout matches what attention layers expect.
        if cu_seqlens is not None:
            # cu_seqlens from the batch is GLOBAL (pre-TE-partitioning).
            cu_seqlens_global = cu_seqlens  # already global
            cu_seqlens_local = cu_seqlens // self.cp_size
        else:
            cu_seqlens_global = None
            cu_seqlens_local = None
        output = _redo_attention_load_balancing(output, self.cp_size, cu_seqlens=cu_seqlens_global)
        if cu_seqlens_local is not None:
            # Reverse the deinterleaving done in pre_conv_ssm so that the
            # inverse all-to-all restores the original per-rank layout.
            output = _reinterleave_packed_seqs(output, cu_seqlens_local, self.cp_size)
        return _all_to_all_hp2cp(output, self.cp_group, B)

    # ------------------------------------------------------------------ #
    #  Parameter slicing (returns views so grads flow to full params)      #
    # ------------------------------------------------------------------ #

    def get_conv1d_weight(self) -> torch.Tensor:
        """Slice ``conv1d.weight`` for the current CP rank.

        Weight shape: ``[conv_dim, 1, kernel_size]`` where
        ``conv_dim = d_inner + 2 * n_groups * d_state``.
        Returns ``[conv_dim_local, kernel_size]`` (squeezed for causal_conv1d kernel).
        """
        return self._slice_conv_param(self._mixer.conv1d.weight).squeeze(1)

    def get_conv1d_bias(self) -> torch.Tensor:
        """Slice ``conv1d.bias`` for the current CP rank.

        Bias shape: ``[conv_dim]``.  Returns ``[conv_dim_local]``.
        """
        if self._mixer.conv1d.bias is None:
            return None
        return self._slice_conv_param(self._mixer.conv1d.bias)

    def get_dt_bias(self, dt_bias: torch.Tensor | None = None) -> torch.Tensor:
        """Slice ``dt_bias`` for the current CP rank."""
        return self._slice_vector_param(self._mixer.dt_bias if dt_bias is None else dt_bias)

    def get_A_log(self, A_log: torch.Tensor | None = None) -> torch.Tensor:
        """Slice ``A_log`` for the current CP rank."""
        return self._slice_vector_param(self._mixer.A_log if A_log is None else A_log)

    def get_D(self, D: torch.Tensor | None = None) -> torch.Tensor:
        """Slice ``D`` for the current CP rank."""
        return self._slice_vector_param(self._mixer.D if D is None else D)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _repeat_group_state(self, state: torch.Tensor) -> torch.Tensor:
        """Repeat group states for CP ranks when n_groups < cp_size.

        ``[B, L, n_groups * d_state]`` -> ``[B, L, n_groups * repeat * d_state]``
        Also supports THD 2D input ``[T, n_groups * d_state]``.
        """
        is_2d = state.dim() == 2
        if is_2d:
            state = state.unsqueeze(0)
        result = (
            state.reshape(*state.shape[:-1], self.n_groups, self.d_state)
            .unsqueeze(-2)
            .expand(-1, -1, -1, self.group_repeat_count, -1)
            .reshape(*state.shape[:-1], self.n_groups * self.group_repeat_count * self.d_state)
        )
        return result.squeeze(0) if is_2d else result

    def _slice_vector_param(self, param: torch.Tensor) -> torch.Tensor:
        """Slice a per-head vector parameter for the current CP rank."""
        start = self.cp_rank * self.num_heads_local
        return param[start : start + self.num_heads_local]

    def _slice_conv_param(self, param: torch.Tensor) -> torch.Tensor:
        """Slice a conv1d parameter (weight or bias) along its channel dimension.

        Parameter slicing is done in the forward path so that gradients
        backpropagate to the original (full) parameters.
        """
        groups_state_size = self.n_groups * self.d_state
        x, B_param, C_param = torch.split(
            param,
            [self.d_inner, groups_state_size, groups_state_size],
            dim=0,
        )

        x_start = self.cp_rank * self.d_inner_local
        x_sliced = x[x_start : x_start + self.d_inner_local]

        bc_size = self.n_groups_local * self.d_state
        bc_start = (self.cp_rank // self.group_repeat_count) * bc_size
        B_sliced = B_param[bc_start : bc_start + bc_size]
        C_sliced = C_param[bc_start : bc_start + bc_size]

        return torch.cat([x_sliced, B_sliced, C_sliced], dim=0).contiguous()
