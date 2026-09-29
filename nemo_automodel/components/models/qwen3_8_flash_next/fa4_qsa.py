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

"""SM90 QSA using PyTorch preprocessing and FlashAttention 4.

Optional FA4/CuTe dependencies are loaded on the first CUDA call. The only
CuTe code owned here is the mask callback compiled into FA4's kernels.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable

import torch
import torch.nn.functional as F

from nemo_automodel.shared.import_utils import safe_import


@functools.cache
def _load_fa4() -> tuple[Callable, type, Callable]:
    """Cache optional FA4 entry points and the mask callback, never tensors."""
    dependencies = [
        safe_import(name)
        for name in (
            "flash_attn.cute.interface",
            "flash_attn.cute.block_sparsity",
            "cutlass",
            "cutlass.cute",
            "flash_attn.cute.utils",
        )
    ]
    if not all(available for available, _ in dependencies):
        raise ImportError(
            "FA4 QSA requires FlashAttention's flash_attn.cute SM90 kernels, "
            "nvidia-cutlass-dsl==4.6.2 and compatible TVM FFI."
        )
    interface, sparsity, cutlass, cute, utils = [module for _, module in dependencies]

    @cute.jit
    def mask_mod(batch, head, query_idx, key_idx, seqlen_info, aux_tensors):
        """Read QSA membership inside FA4; this is not a standalone kernel.

        Args:
            batch: Scalar SSA batch coordinate.
            head: Scalar SSA query-head coordinate; membership is head-independent.
            query_idx: Scalar SSA query position.
            key_idx: Scalar SSA physical K/V position.
            seqlen_info: FA4 query/key sequence lengths.
            aux_tensors: One uint8 tensor [batch, queries, keys], with padded
                row strides; one means selected and zero means excluded.

        Returns:
            Scalar boolean SSA membership, false outside the logical sequences.
        """
        membership = aux_tensors[0]
        qi = cutlass.min(query_idx[0], seqlen_info.seqlen_q - 1)
        ki = cutlass.min(key_idx[0], seqlen_info.seqlen_k - 1)
        selected = membership[batch[0], qi, ki] != 0
        valid = selected & (query_idx[0] < seqlen_info.seqlen_q) & (key_idx[0] < seqlen_info.seqlen_k)
        return utils.scalar_to_ssa(valid, cutlass.Boolean)

    return interface.flash_attn_func, sparsity.BlockSparseTensorsTorch, mask_mod


def _compact_blocks(kinds: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Compact partial/full block columns for each row.

    Args:
        kinds: Integer tensor [batch, block_rows, block_columns], with values
            zero (absent), one (partial) or two (full).

    Returns:
        Int32 partial counts [batch, 1, block_rows], partial column indices
        [batch, 1, block_rows, block_columns], then full counts and indices
        with the same layouts. Only each count's index prefix is consumed.
        All returned tensors are contiguous and independently allocated.
    """
    columns = torch.arange(kinds.shape[-1], device=kinds.device, dtype=torch.int32)
    partial, full = kinds == 1, kinds == 2
    pc = partial.sum(-1, dtype=torch.int32).unsqueeze(1)
    fc = full.sum(-1, dtype=torch.int32).unsqueeze(1)
    pi = torch.where(partial, columns, kinds.shape[-1]).sort(-1).values.unsqueeze(1)
    fi = torch.where(full, columns, kinds.shape[-1]).sort(-1).values.unsqueeze(1)
    return tuple(t.contiguous() for t in (pc, pi, fc, fi))


def _block_kinds(membership: torch.Tensor, block_keys: int) -> torch.Tensor:
    """Classify query/key blocks without host synchronization.

    Args:
        membership: Uint8 tensor [batch, queries, keys].
        block_keys: Number of physical keys in one block; queries use 128.

    Returns:
        Int32 [batch, ceil(queries/128), ceil(keys/block_keys)] tile classes:
        absent=0, partial=1, full=2. Tail padding is never classified as full.
    """
    batch, queries, keys = membership.shape
    nq, nk = (queries + 127) // 128, (keys + block_keys - 1) // block_keys
    padded = F.pad(membership, (0, nk * block_keys - keys, 0, nq * 128 - queries))
    tiles = padded.reshape(batch, nq, 128, nk, block_keys)
    present = (tiles != 0).any(dim=(2, 4))
    full = (tiles == 1).all(dim=(2, 4))
    return present.to(torch.int32) + full.to(torch.int32)


@torch.compile(dynamic=False)
def _preprocess(routes: torch.Tensor, kv_length: int) -> tuple[torch.Tensor, ...]:
    """Build a byte membership table and FA4 forward/reverse block lists.

    Args:
        routes: Signed int32/int64 tensor [batch, queries, routes_per_query].
            Duplicate IDs collapse; invalid IDs, including large int64 values,
            are discarded before indexing.
        kv_length: Positive number of physical key/value positions.

    Returns:
        Uint8 membership [batch, queries, kv_length], with each row padded to
        a multiple of 32 bytes, followed by four forward and four reverse
        tensors in the layout documented by _compact_blocks. Forward rows are
        ceil(queries/128) and columns ceil(kv_length/80); reverse rows are
        ceil(kv_length/64) and columns ceil(queries/128).
        No input is mutated, and no device scalar is read on the host.
    """
    batch, queries, _ = routes.shape
    valid = (routes >= 0) & (routes < kv_length)
    indices = torch.where(valid, routes, kv_length).long()
    width = ((kv_length + 1 + 31) // 32) * 32
    storage = torch.zeros((batch, queries, width), device=routes.device, dtype=torch.uint8)
    # All duplicate writes store one. Invalid IDs write a separate padding
    # column, so they cannot overwrite a real token's membership.
    storage.scatter_(-1, indices, 1)
    membership = storage[..., :kv_length]
    forward = _compact_blocks(_block_kinds(membership, 80))
    backward = _compact_blocks(_block_kinds(membership, 64).transpose(1, 2))
    return membership, *forward, *backward


def fa4_sparse_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Evaluate exact route-set GQA using FA4 on SM90.

    Duplicate routes select a token once. Invalid IDs are ignored. Empty query
    rows have zero output and query gradients. Routes encode causality,
    documents and padding; no additional triangular mask is imposed.
    FA4 owns first-order autograd; higher-order gradients and deterministic
    backward are unsupported. Only discrete preprocessing is torch-compiled.

    Args:
        query: BF16 CUDA tensor [batch, queries, query_heads, 256].
            Arbitrary strides are accepted; non-unit final strides are copied.
        key: BF16 CUDA tensor [batch, keys, kv_heads, 256]. May contain gathered
            global K/V while query is a local CP slice. query_heads must be a
            positive multiple of kv_heads.
        value: BF16 CUDA tensor with key's shape and device.
        selected_token_ids: Signed int32/int64 CUDA tensor
            [batch, queries, routes_per_query] in physical K/V coordinates.
            Noncontiguous inputs are accepted. Dimensions must be nonempty.
        softmax_scale: Finite positive score multiplier; defaults to 1/sqrt(256).

    Returns:
        Independent BF16 CUDA tensor [batch, queries, query_heads, 256].
        Inputs are not mutated.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("FA4 QSA expects Q/K/V in [batch, sequence, heads, head_dim] layout")
    if key.shape != value.shape or key.shape[0] != query.shape[0]:
        raise ValueError("FA4 QSA requires matching K/V shapes and Q/K/V batch sizes")
    if any(size <= 0 for tensor in (query, key, value) for size in tensor.shape):
        raise ValueError("FA4 QSA requires nonempty Q/K/V dimensions")
    if query.shape[-1] != 256 or key.shape[-1] != 256:
        raise ValueError("FA4 QSA requires head_dim=256")
    if query.shape[2] % key.shape[2] != 0:
        raise ValueError("FA4 QSA requires query_heads divisible by kv_heads")
    if selected_token_ids.ndim != 3 or selected_token_ids.shape[:2] != query.shape[:2]:
        raise ValueError("FA4 QSA routes must have shape [batch, query_sequence, routes]")
    if selected_token_ids.shape[-1] == 0:
        raise ValueError("FA4 QSA requires a nonempty route dimension; use -1 for empty rows")
    if selected_token_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("FA4 QSA routes must be signed int32 or int64")
    if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
        raise TypeError("FA4 QSA requires BF16 query, key, and value")
    if not query.is_cuda or any(t.device != query.device for t in (key, value, selected_token_ids)):
        raise ValueError("FA4 QSA requires Q/K/V/routes on the same CUDA device")
    if torch.cuda.get_device_capability(query.device) != (9, 0):
        raise RuntimeError("FA4 QSA currently supports SM90 GPUs only")
    padded_keys = ((key.shape[1] + 32) // 32) * 32
    if query.shape[0] * query.shape[1] * padded_keys >= 2**31:
        raise ValueError("FA4 QSA byte mask exceeds int32 indexing limits")
    scale = 256**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("FA4 QSA softmax_scale must be finite and positive")
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("FA4 QSA backward uses atomic reductions and does not support deterministic algorithms")
    with torch.cuda.device(query.device):
        attention, block_lists, mask_mod = _load_fa4()
        raw = _preprocess(selected_token_ids, key.shape[1])
        forward_blocks = block_lists(*raw[1:5], block_size=(128, 80))
        backward_blocks = block_lists(*raw[5:9], block_size=(128, 64))
        # No tensor Python attributes: FA4 can also consume checkpoint-restored
        # auxiliary tensors, whose storage/layout survive without custom tags.
        output, _ = attention(
            *(t.contiguous() if t.stride(-1) != 1 else t for t in (query, key, value)),
            softmax_scale=scale,
            pack_gqa=False,
            mask_mod=mask_mod,
            aux_tensors=[raw[0]],
            block_sparse_tensors=forward_blocks,
            block_sparse_tensors_bwd=backward_blocks,
        )
        return output
