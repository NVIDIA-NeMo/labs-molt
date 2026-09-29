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

"""Shared FlashMLA-forward/cuDNN-backward sparse latent attention."""

from __future__ import annotations

import math
from typing import Any

import torch

from nemo_automodel.shared.import_utils import safe_import_from

_HAS_CUDNN_DSA, _CUDNN_DSA = safe_import_from(
    "cudnn",
    "DSA",
    msg=(
        "cuDNN sparse-attention kernels are unavailable. Install "
        "nvidia-cudnn-frontend[cutedsl] to use backend.attn='cudnn'."
    ),
)
_HAS_FLASH_MLA, _FLASH_MLA_SPARSE_FWD = safe_import_from(
    "flash_mla",
    "flash_mla_sparse_fwd",
    msg="FlashMLA sparse prefill is unavailable. Install the FlashMLA nv_dev package.",
)

_SUPPORTED_ATTENTION_HEAD_DIMS = (512, 576)
_VALUE_HEAD_DIM = 512
_FLASH_MLA_TOPK_ALIGNMENT = 512


def is_cudnn_sparse_attention_available() -> bool:
    """Return whether the cuDNN backward and FlashMLA forward runtimes import."""
    return bool(_HAS_CUDNN_DSA and _HAS_FLASH_MLA)


def _require_available() -> None:
    """Raise when either optional sparse-attention runtime is unavailable."""
    if not is_cudnn_sparse_attention_available():
        raise RuntimeError(
            "cuDNN sparse attention requires both nvidia-cudnn-frontend[cutedsl] "
            "and FlashMLA with flash_mla_sparse_fwd."
        )


def _require_cuda_tensors(operation: str, *tensors: torch.Tensor) -> tuple[int, int]:
    """Validate that arbitrary-layout input tensors share one SM90+ CUDA device.

    Args:
        operation: Name included in validation errors.
        *tensors: Tensors with arbitrary shapes that must share one CUDA device.

    Returns:
        CUDA compute capability as ``(major, minor)``.
    """
    if not tensors or any(not tensor.is_cuda for tensor in tensors):
        raise RuntimeError(f"{operation} requires CUDA tensors.")
    device = tensors[0].device
    if any(tensor.device != device for tensor in tensors[1:]):
        raise ValueError(f"{operation} requires every tensor on the same CUDA device.")
    major, minor = torch.cuda.get_device_capability(device)
    if major < 9:
        raise RuntimeError(f"{operation} requires SM90 or later, got SM{major}{minor}.")
    return major, minor


def _compact_and_sort_indices(indices: torch.Tensor, key_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalize sparse indices into an ascending valid prefix.

    Args:
        indices: Integer tensor of shape ``[query_tokens, sparse_width]`` with
            global K/V coordinates and negative invalid entries.
        key_count: Number of rows in the flattened K/V tensor.

    Returns:
        A contiguous int32 index tensor of shape ``[query_tokens, sparse_width]``
        with an ascending valid prefix and ``-1`` suffix, and a contiguous int32
        tensor of shape ``[query_tokens]`` containing valid-prefix lengths.
    """
    valid = (indices >= 0) & (indices < key_count)
    topk_length = valid.sum(dim=-1, dtype=torch.int32)
    positions = torch.arange(indices.size(-1), device=indices.device).view(1, -1)
    compact_order = torch.where(valid, positions, torch.full_like(positions, indices.size(-1))).argsort(dim=-1)
    indices = torch.gather(indices, -1, compact_order)
    compact_valid = torch.gather(valid, -1, compact_order)
    indices = indices.masked_fill(~compact_valid, -1)

    prefix = positions < topk_length.unsqueeze(-1)
    sort_key = torch.where(prefix, indices, torch.full_like(indices, key_count))
    sort_order = sort_key.argsort(dim=-1)
    indices = torch.gather(indices, -1, sort_order)
    sorted_valid = torch.gather(prefix.expand_as(indices), -1, sort_order)
    return indices.masked_fill(~sorted_valid, -1).to(torch.int32).contiguous(), topk_length.contiguous()


def _padded_head_count(num_heads: int, major: int) -> int:
    """Return the FlashMLA-supported query-head count for one SM generation."""
    if major >= 10:
        for padded in (64, 128):
            if num_heads == padded or (num_heads < padded and padded % num_heads == 0):
                return padded
        alignment = 128
    else:
        alignment = 64
    if num_heads % alignment == 0:
        return num_heads
    if num_heads < alignment and alignment % num_heads == 0:
        return alignment
    raise ValueError(f"FlashMLA sparse prefill requires the query-head count to divide {alignment}, got H={num_heads}.")


def _pad_attention_heads(
    q: torch.Tensor, attn_sink: torch.Tensor, padded_heads: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad query and attention-sink head axes for FlashMLA.

    Args:
        q: Query tensor of shape ``[query_tokens, heads, head_dim]``.
        attn_sink: FP32 attention-sink tensor of shape ``[heads]``.
        padded_heads: FlashMLA-compatible output head count.

    Returns:
        Query tensor of shape ``[query_tokens, padded_heads, head_dim]`` and
        attention-sink tensor of shape ``[padded_heads]``. Inputs are returned
        unchanged when ``heads == padded_heads``.
    """
    if q.shape[1] == padded_heads:
        return q, attn_sink
    q_padded = q.new_zeros((q.shape[0], padded_heads, q.shape[2]))
    q_padded[:, : q.shape[1]] = q
    sink_padded = attn_sink.new_full((padded_heads,), float("-inf"))
    sink_padded[: q.shape[1]] = attn_sink
    return q_padded, sink_padded


class _CudnnSparseAttention(torch.autograd.Function):
    """Pair FlashMLA forward with cuDNN backward for latent THD attention."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        kv_latent: torch.Tensor,
        topk_indices: torch.Tensor,
        softmax_scale: float,
        padded_heads: int,
        topk_length: torch.Tensor | None,
        all_rows_nonempty: bool,
        valid_row_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        """Run FlashMLA forward and save tensors required by cuDNN backward.

        Args:
            ctx: Autograd context used to save forward tensors and scalar metadata.
            q: CUDA BF16 query tensor of shape ``[query_tokens, heads, head_dim]``.
            kv_latent: CUDA BF16 latent K/V tensor of shape ``[key_tokens, 1, head_dim]``.
            topk_indices: CUDA int32 tensor of shape ``[query_tokens, 1, sparse_width]``
                containing global K/V coordinates and invalid entries marked ``-1``.
            softmax_scale: Scale applied to query-key scores.
            padded_heads: FlashMLA-compatible padded head count.
            topk_length: Optional int32 valid-prefix lengths of shape ``[query_tokens]``.
            all_rows_nonempty: Whether every query has a positive valid prefix.
            valid_row_indices: Optional int64 indices of nonempty queries with shape
                ``[valid_query_tokens]``.

        Returns:
            CUDA BF16 latent values of shape ``[query_tokens, heads, 512]``.
        """
        kv = kv_latent.squeeze(1).contiguous()
        if topk_length is None:
            indices, topk_length = _compact_and_sort_indices(topk_indices.squeeze(1), kv.shape[0])
        else:
            indices = topk_indices.squeeze(1).contiguous()
        padded_topk = math.ceil(indices.shape[-1] / _FLASH_MLA_TOPK_ALIGNMENT) * _FLASH_MLA_TOPK_ALIGNMENT
        if padded_topk != indices.shape[-1]:
            indices = torch.nn.functional.pad(indices, (0, padded_topk - indices.shape[-1]), value=-1)

        attn_sink = torch.full((q.shape[1],), float("-inf"), dtype=torch.float32, device=q.device)
        q_kernel, sink_kernel = _pad_attention_heads(q.contiguous(), attn_sink, padded_heads)
        out_kernel, _max_logits, lse_kernel = _FLASH_MLA_SPARSE_FWD(
            q_kernel,
            kv.unsqueeze(1),
            indices.unsqueeze(1),
            softmax_scale,
            d_v=_VALUE_HEAD_DIM,
            attn_sink=sink_kernel,
            topk_length=topk_length,
            indexer_topk=0,
        )
        out = out_kernel[:, : q.shape[1]].contiguous()
        lse = lse_kernel[:, : q.shape[1]].contiguous()
        if not all_rows_nonempty:
            out.masked_fill_(topk_length.eq(0).view(-1, 1, 1), 0)
        cached_valid_rows = (
            valid_row_indices if valid_row_indices is not None else torch.empty(0, dtype=torch.int64, device=q.device)
        )
        ctx.save_for_backward(q, kv, out, lse, attn_sink, indices.clamp_min(0), topk_length, cached_valid_rows)
        ctx.softmax_scale = softmax_scale
        ctx.padded_heads = padded_heads
        ctx.all_rows_nonempty = all_rows_nonempty
        ctx.has_cached_valid_rows = valid_row_indices is not None
        return out

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor | None, ...]:
        """Map output gradients to query and gathered latent-KV layouts.

        Args:
            ctx: Autograd context populated by :meth:`forward`.
            grad_output: CUDA BF16 output gradient of shape
                ``[query_tokens, heads, 512]``.

        Returns:
            Gradients for the eight forward inputs: query tensor of shape
            ``[query_tokens, heads, head_dim]``, latent K/V tensor of shape
            ``[key_tokens, 1, head_dim]``, then ``None`` for scalar and metadata
            inputs.
        """
        q, kv, out, lse, attn_sink, indices, topk_length, cached_valid_rows = ctx.saved_tensors
        valid_row_indices = None
        if not ctx.all_rows_nonempty:
            valid_row_indices = (
                cached_valid_rows
                if ctx.has_cached_valid_rows
                else torch.nonzero(topk_length > 0, as_tuple=False).flatten()
            )

        q_input = q
        out_input = out
        grad_input = grad_output
        lse_input = lse
        indices_kernel = indices
        topk_length_kernel = topk_length
        used_dummy = False
        if valid_row_indices is not None:
            if valid_row_indices.numel() == 0:
                used_dummy = True
                q_input = torch.zeros_like(q[:1])
                out_input = torch.zeros_like(out[:1])
                grad_input = torch.zeros_like(grad_output[:1])
                lse_input = torch.zeros_like(lse[:1])
                indices_kernel = torch.zeros_like(indices[:1])
                topk_length_kernel = torch.ones_like(topk_length[:1])
            else:
                q_input = q.index_select(0, valid_row_indices)
                out_input = out.index_select(0, valid_row_indices)
                grad_input = grad_output.index_select(0, valid_row_indices)
                lse_input = lse.index_select(0, valid_row_indices)
                indices_kernel = indices.index_select(0, valid_row_indices)
                topk_length_kernel = topk_length.index_select(0, valid_row_indices)

        q_kernel, sink_kernel = _pad_attention_heads(q_input, attn_sink, ctx.padded_heads)
        if ctx.padded_heads == q.shape[1]:
            out_kernel = out_input
            grad_kernel = grad_input.contiguous()
            lse_kernel = lse_input
        else:
            out_kernel = out_input.new_zeros((out_input.shape[0], ctx.padded_heads, out_input.shape[2]))
            out_kernel[:, : out_input.shape[1]] = out_input
            grad_kernel = grad_input.new_zeros((grad_input.shape[0], ctx.padded_heads, grad_input.shape[2]))
            grad_kernel[:, : grad_input.shape[1]] = grad_input
            lse_kernel = lse_input.new_zeros((lse_input.shape[0], ctx.padded_heads))
            lse_kernel[:, : lse_input.shape[1]] = lse_input

        result = _CUDNN_DSA.sparse_attention_backward_wrapper(
            q_kernel.contiguous(),
            kv,
            out_kernel.contiguous(),
            grad_kernel.contiguous(),
            lse_kernel.contiguous(),
            sink_kernel,
            indices_kernel,
            softmax_scale=ctx.softmax_scale,
            topk_length=topk_length_kernel,
        )
        if valid_row_indices is None:
            grad_q = result["dq"][:, : q.shape[1]].contiguous()
        else:
            grad_q_valid = result["dq"][:0, : q.shape[1]] if used_dummy else result["dq"][:, : q.shape[1]]
            grad_q = torch.zeros_like(q)
            grad_q.index_copy_(0, valid_row_indices, grad_q_valid)
        grad_kv = result["dkv"].unsqueeze(1).contiguous()
        return grad_q, grad_kv, None, None, None, None, None, None


def cudnn_sparse_attention(
    q: torch.Tensor,
    kv_latent: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_scale: float,
    topk_length: torch.Tensor | None = None,
    all_rows_nonempty: bool = False,
    valid_row_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse latent attention with FlashMLA forward and cuDNN backward.

    Args:
        q: Contiguous CUDA BF16 query tensor of shape
            ``[query_tokens, heads, head_dim]``, where ``head_dim`` is 512 or 576.
        kv_latent: Contiguous CUDA BF16 latent K/V tensor of shape
            ``[key_tokens, 1, head_dim]``.
        topk_indices: Contiguous CUDA int32 tensor of shape
            ``[query_tokens, 1, sparse_width]`` with global K/V coordinates and
            invalid entries marked ``-1``.
        softmax_scale: Scale forwarded unchanged to FlashMLA and cuDNN backward.
        topk_length: Optional contiguous CUDA int32 valid-prefix lengths of shape
            ``[query_tokens]``. When supplied, ``topk_indices`` must already contain
            a compact, ascending valid prefix.
        all_rows_nonempty: Whether every query has a positive valid-prefix length.
        valid_row_indices: Optional contiguous CUDA int64 indices of nonempty queries
            with shape ``[valid_query_tokens]``.

    Returns:
        Contiguous CUDA BF16 latent output tensor of shape
        ``[query_tokens, heads, 512]``.

    Raises:
        RuntimeError: If optional kernels, CUDA, or SM90+ are unavailable.
        TypeError: If compute tensors are not BF16 or indices are not int32.
        ValueError: If tensor layouts, dimensions, sparse width, or scale are invalid.
    """
    _require_available()
    major, _ = _require_cuda_tensors("cuDNN DSA sparse attention", q, kv_latent, topk_indices)
    if q.dtype != torch.bfloat16 or kv_latent.dtype != torch.bfloat16:
        raise TypeError(f"q and kv_latent must be bfloat16, got {q.dtype} and {kv_latent.dtype}.")
    if topk_indices.dtype != torch.int32:
        raise TypeError(f"topk_indices must be int32, got {topk_indices.dtype}.")
    if q.ndim != 3 or q.shape[-1] not in _SUPPORTED_ATTENTION_HEAD_DIMS:
        raise ValueError(
            f"q must have shape [query_tokens, heads, head_dim] with head_dim in "
            f"{_SUPPORTED_ATTENTION_HEAD_DIMS}, got {tuple(q.shape)}."
        )
    head_dim = q.shape[-1]
    if kv_latent.ndim != 3 or kv_latent.shape[1:] != (1, head_dim):
        raise ValueError(f"kv_latent must have shape [key_tokens, 1, {head_dim}], got {tuple(kv_latent.shape)}.")
    if topk_indices.ndim != 3 or topk_indices.shape[:2] != (q.shape[0], 1):
        raise ValueError(
            f"topk_indices must have shape [query_tokens, 1, sparse_width], got {tuple(topk_indices.shape)}."
        )
    if topk_indices.shape[-1] <= 0:
        raise ValueError(f"sparse_width must be positive, got {topk_indices.shape[-1]}.")
    if kv_latent.shape[0] >= torch.iinfo(torch.int32).max:
        raise ValueError("The flattened K/V token count must fit in an int32 global index.")
    if not isinstance(softmax_scale, (float, int)) or not math.isfinite(float(softmax_scale)):
        raise TypeError("softmax_scale must be a finite Python float.")
    if float(softmax_scale) <= 0.0:
        raise ValueError(f"softmax_scale must be positive, got {softmax_scale}.")
    if not isinstance(all_rows_nonempty, bool):
        raise TypeError(f"all_rows_nonempty must be a bool, got {type(all_rows_nonempty).__name__}.")
    if all_rows_nonempty and valid_row_indices is not None:
        raise ValueError("valid_row_indices must be None when all_rows_nonempty is true.")
    if topk_length is not None:
        if topk_length.shape != (q.shape[0],) or topk_length.dtype != torch.int32 or topk_length.device != q.device:
            raise ValueError(
                "topk_length must be an int32 tensor on the query device with shape "
                f"{(q.shape[0],)}, got shape={tuple(topk_length.shape)}, "
                f"dtype={topk_length.dtype}, device={topk_length.device}."
            )
        if not topk_length.is_contiguous():
            raise ValueError("topk_length must be contiguous.")
    if valid_row_indices is not None:
        if topk_length is None:
            raise ValueError("valid_row_indices requires precomputed topk_length metadata.")
        if (
            valid_row_indices.ndim != 1
            or valid_row_indices.dtype != torch.int64
            or valid_row_indices.device != q.device
            or not valid_row_indices.is_contiguous()
        ):
            raise ValueError("valid_row_indices must be a contiguous int64 tensor on the query device.")
        if valid_row_indices.numel() > q.shape[0]:
            raise ValueError("valid_row_indices cannot contain more entries than query rows.")

    padded_heads = _padded_head_count(q.shape[1], major)
    return _CudnnSparseAttention.apply(
        q.contiguous(),
        kv_latent.contiguous(),
        topk_indices.contiguous(),
        float(softmax_scale),
        padded_heads,
        topk_length,
        all_rows_nonempty,
        valid_row_indices,
    )


__all__ = ["cudnn_sparse_attention", "is_cudnn_sparse_attention_available"]
