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

"""Qwen3.8-Flash-Next QSA routing, FlexAttention sparse GQA, and its PyTorch oracle.

The supplied Qwen3.8-Flash-Next reference is an inference implementation: its indexer
returns integer top-k IDs and defines neither an auxiliary indexer loss nor a
straight-through gradient.  This module therefore freezes the indexer weights
explicitly.  Routing is still recomputed from the current hidden states on
every forward, while gradients flow through the main attention Q/K/V path.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb
from nemo_automodel.components.models.qwen3_8_flash_next.cp import (
    Qwen3_8_FlashNextCPContext,
    qwen3_8_flash_next_cp_all_gather,
)
from nemo_automodel.components.models.qwen3_8_flash_next.fa4_qsa import fa4_sparse_gqa_attention
from nemo_automodel.components.models.qwen3_8_flash_next.flex_qsa import flex_sparse_gqa_attention
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextRMSNorm
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

# The gathered implementation is a numerical oracle and CPU fallback, not the
# production CUDA path. Keep its workspace bounded without exposing a model or
# recipe knob that could be mistaken for part of the QSA architecture.
_PYTORCH_ORACLE_QUERY_CHUNK_SIZE = 16


def right_padded_sequence_lengths(
    attention_mask: torch.Tensor | None,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Validate a non-packed right-tail mask and return logical lengths.

    Args:
        attention_mask: ``None`` or a binary tensor ``[B, S]``.  Each row must
            be exactly ``1 ** L + 0 ** (S - L)``; left/interior padding and
            packed document-ID masks are rejected.
        batch_size: Expected batch dimension ``B``.
        sequence_length: Expected physical sequence dimension ``S``.
        device: Device on which to return the lengths.

    Returns:
        Logical sequence lengths as int64 ``[B]``.
    """
    if attention_mask is None:
        return torch.full((batch_size,), sequence_length, dtype=torch.long, device=device)
    if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch_size, sequence_length):
        raise NotImplementedError(
            "Qwen3.8-Flash-Next QSA currently requires a non-packed [batch, sequence] right-tail attention mask; "
            f"got {tuple(attention_mask.shape)}"
        )
    mask = attention_mask.to(device=device)
    if mask.dtype != torch.bool:
        binary = (mask == 0) | (mask == 1)
        if not bool(binary.all()):
            raise ValueError("Qwen3.8-Flash-Next QSA attention_mask must contain only binary 0/1 values")
    valid = mask.bool()
    lengths = valid.sum(dim=-1, dtype=torch.long)
    expected = torch.arange(sequence_length, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    if not bool(torch.equal(valid, expected)):
        raise NotImplementedError(
            "Qwen3.8-Flash-Next QSA currently supports only right-tail padding (1...10...0); "
            "left padding, interior padding, and packed masks are not supported"
        )
    return lengths


def apply_qsa_rope(states: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply the Qwen3.8-Flash-Next attention RoPE to indexer states.

    Args:
        states: Index query or compressed-key states ``[B, N, H, D_index]``.
        freqs_cis: Model-composed rotary values ``[B, N, D_rope]`` laid out as
            ``cat(cos[..., D_rope/2], sin[..., D_rope/2])``.  ``D_rope`` may be
            smaller than ``D_index``; the remaining index dimensions pass
            through unchanged.

    Returns:
        Rotated states in the same ``[B, N, H, D_index]`` layout and dtype.
    """
    if states.ndim != 4:
        raise ValueError(f"QSA RoPE states must be [B, N, H, D], got {tuple(states.shape)}")
    if freqs_cis.ndim != 3 or freqs_cis.shape[:2] != states.shape[:2]:
        raise ValueError(
            "QSA RoPE frequencies must be [B, N, D_rope] and match the state token axes; "
            f"got states={tuple(states.shape)}, freqs={tuple(freqs_cis.shape)}"
        )
    rotary_width = freqs_cis.shape[-1]
    if rotary_width <= 0 or rotary_width % 2 != 0 or rotary_width > states.shape[-1]:
        raise ValueError(
            "QSA rotary width must be positive, even, and no wider than the index head; "
            f"got rotary_width={rotary_width}, index_head_dim={states.shape[-1]}"
        )
    cos, sin = freqs_cis.split(rotary_width // 2, dim=-1)
    return apply_rotary_emb(states, cos, sin)


_QSA_SCORE_BUDGET_BYTES = 256 * 1024 * 1024


def _qsa_query_chunk_rows(query_chunk_size: int | None, *, num_heads: int, num_blocks: int, rows: int) -> int:
    """Rows scored per chunk: the explicit setting, or as many as fit the fp32 score budget.

    Args:
        query_chunk_size: Explicit maximum rows per chunk, or ``None`` for the memory-based default.
        num_heads: Index query heads (``H_index``); the score tensor is ``[rows, H_index, num_blocks]`` fp32.
        num_blocks: Compressed key blocks visible to this batch row.
        rows: Number of valid query rows to cover.

    Returns:
        A positive chunk size no larger than ``rows``.
    """
    if query_chunk_size is not None:
        return query_chunk_size
    bytes_per_row = max(num_heads * num_blocks * 4, 1)
    return max(1, min(rows, _QSA_SCORE_BUDGET_BYTES // bytes_per_row))


@torch.no_grad()
def select_qsa_token_ids(
    index_queries: torch.Tensor,
    compressed_keys: torch.Tensor,
    sequence_lengths: torch.Tensor,
    *,
    token_budget: int,
    compress_ratio: int,
    query_chunk_size: int | None = None,
    query_position_offset: int = 0,
    global_sequence_length: int | None = None,
) -> torch.Tensor:
    """Score compressed blocks and expand gold QSA top-k IDs.

    For query position ``t``, only ``floor((t + 1) / compress_ratio)`` complete
    blocks are visible.  Each block score is
    ``sum_h(relu(dot(q[t,h], k[block,0]))) / sqrt(D)``.  The best
    ``token_budget / compress_ratio`` blocks are expanded to token IDs, then
    the 0--``compress_ratio - 1`` tokens in the current incomplete causal tail
    are appended.  Invalid slots are ``-1``.

    The selection is vectorized over query rows: the only host synchronization
    is one read of ``sequence_lengths``; chunk boundaries, the first sparse row,
    and the tail slots follow from positions known on the host, so no
    data-dependent branch waits on the device.

    Args:
        index_queries: Normalized and rotated local index queries
            ``[B, S_query, H_index, D_index]``.
        compressed_keys: FP32-mean-pooled, normalized, rotated index keys
            ``[B, floor(S_global / compress_ratio), 1, D_index]``.
        sequence_lengths: Right-padded logical lengths ``[B]``.
        token_budget: Maximum number of tokens contributed by complete blocks.
        compress_ratio: Number of consecutive tokens represented by one block.
        query_chunk_size: Maximum query rows scored together. ``None`` sizes
            chunks so the temporary FP32 score tensor ``[rows, H_index,
            blocks]`` stays within 256 MiB (one chunk for 4k-16k sequences).
            Chunking bounds memory without changing top-k semantics.
        query_position_offset: Global position represented by local query row
            zero. It is zero without CP and ``cp_rank * S_query`` for the
            contiguous CP layout.
        global_sequence_length: Padded global physical sequence length. It
            defaults to the local query length for the non-CP path.

    Returns:
        Global logical token IDs ``[B, S_query, token_budget +
        compress_ratio - 1]`` in int32. Valid IDs are contiguous at the start
        of each row.
    """
    if index_queries.ndim != 4:
        raise ValueError(f"index_queries must be [B, S, H, D], got {tuple(index_queries.shape)}")
    if compressed_keys.ndim != 4 or compressed_keys.shape[2] != 1:
        raise ValueError(f"compressed_keys must be [B, P, 1, D], got {tuple(compressed_keys.shape)}")
    batch_size, query_sequence_length, num_heads, head_dim = index_queries.shape
    if compressed_keys.shape[0] != batch_size or compressed_keys.shape[-1] != head_dim:
        raise ValueError("QSA query/key batch and head dimensions must match")
    if sequence_lengths.shape != (batch_size,):
        raise ValueError(f"sequence_lengths must be [{batch_size}], got {tuple(sequence_lengths.shape)}")
    if token_budget <= 0 or compress_ratio <= 1 or token_budget % compress_ratio != 0:
        raise ValueError(
            "QSA requires a positive token_budget divisible by compress_ratio > 1; "
            f"got token_budget={token_budget}, compress_ratio={compress_ratio}"
        )
    if query_chunk_size is not None and query_chunk_size <= 0:
        raise ValueError(f"query_chunk_size must be positive, got {query_chunk_size}")
    if num_heads <= 0 or head_dim <= 0:
        raise ValueError("QSA index queries require positive head count and head dimension")
    if query_position_offset < 0:
        raise ValueError(f"query_position_offset must be non-negative, got {query_position_offset}")
    if global_sequence_length is None:
        global_sequence_length = query_sequence_length
    if global_sequence_length < query_position_offset + query_sequence_length:
        raise ValueError(
            "QSA global sequence does not cover every local query; "
            f"got global={global_sequence_length}, offset={query_position_offset}, local={query_sequence_length}"
        )

    # The single host synchronization of this routine.
    lengths = [int(length) for length in sequence_lengths.tolist()]
    if any(length < 0 or length > global_sequence_length for length in lengths):
        raise ValueError(f"QSA logical lengths must lie in [0, {global_sequence_length}]")
    num_blocks = compressed_keys.shape[1]
    required_blocks = max((length // compress_ratio for length in lengths), default=0)
    if required_blocks > num_blocks:
        raise ValueError(
            "compressed_keys do not contain every complete logical block; "
            f"need at least {required_blocks}, got {num_blocks}"
        )

    device = index_queries.device
    block_budget = token_budget // compress_ratio
    final_width = token_budget + compress_ratio - 1
    selected_tokens = torch.full(
        (batch_size, query_sequence_length, final_width),
        -1,
        dtype=torch.int32,
        device=device,
    )
    block_offsets = torch.arange(compress_ratio, device=device, dtype=torch.long)
    tail_offsets = torch.arange(compress_ratio - 1, device=device, dtype=torch.long)
    score_scale = math.sqrt(head_dim)
    # Gold fast_topk preserves causal block order while all visible blocks fit
    # the budget; score-ordered top-k starts at the first position whose
    # visible block count exceeds it (t=2051 for c4/budget=2048).  Sparse rows
    # are therefore a contiguous suffix of every chunk.
    first_sparse_position = (block_budget + 1) * compress_ratio - 1

    for batch_idx, logical_length in enumerate(lengths):
        available_blocks = logical_length // compress_ratio
        keys = compressed_keys[batch_idx, :available_blocks, 0].float()
        local_valid_length = min(max(logical_length - query_position_offset, 0), query_sequence_length)
        if local_valid_length == 0:
            continue
        topk_width = min(block_budget, available_blocks)
        chunk_rows = _qsa_query_chunk_rows(
            query_chunk_size, num_heads=num_heads, num_blocks=max(available_blocks, 1), rows=local_valid_length
        )
        for query_start in range(0, local_valid_length, chunk_rows):
            query_end = min(query_start + chunk_rows, local_valid_length)
            rows = query_end - query_start
            first_position = query_position_offset + query_start
            query_positions = torch.arange(first_position, first_position + rows, device=device)
            visible_blocks = torch.div(query_positions + 1, compress_ratio, rounding_mode="floor")
            result = torch.full((rows, final_width), -1, dtype=torch.int32, device=device)

            if topk_width:
                candidate_blocks = torch.arange(topk_width, device=device)
                top_blocks = candidate_blocks.unsqueeze(0).expand(rows, -1)
                valid_blocks = candidate_blocks.unsqueeze(0) < visible_blocks.unsqueeze(1)
                sparse_start = min(max(first_sparse_position - first_position, 0), rows)
                if sparse_start < rows:
                    sparse_queries = index_queries[batch_idx, query_start + sparse_start : query_end].float()
                    scores = torch.einsum("qhd,pd->qhp", sparse_queries, keys)
                    scores = torch.relu(scores).sum(dim=1) / score_scale
                    block_ids = torch.arange(available_blocks, device=device)
                    sparse_visible = visible_blocks[sparse_start:]
                    scores = scores.masked_fill(block_ids.unsqueeze(0) >= sparse_visible.unsqueeze(1), -torch.inf)
                    sparse_top_blocks = torch.topk(scores, k=block_budget, dim=-1).indices
                    top_blocks = torch.cat([top_blocks[:sparse_start], sparse_top_blocks], dim=0)
                    valid_blocks = torch.cat(
                        [valid_blocks[:sparse_start], torch.ones_like(valid_blocks[sparse_start:])], dim=0
                    )
                expanded = top_blocks.unsqueeze(-1) * compress_ratio + block_offsets
                expanded = torch.where(valid_blocks.unsqueeze(-1), expanded, torch.full_like(expanded, -1))
                result[:, : topk_width * compress_ratio] = expanded.reshape(rows, -1).to(torch.int32)

            # Tail slots start right after the valid blocks; every slot they
            # address still holds -1, so writing -1 for absent tail tokens is a
            # no-op and no data-dependent indexing is needed.
            tail_start = visible_blocks * compress_ratio
            tail_count = query_positions + 1 - tail_start
            valid_block_count = torch.clamp(visible_blocks, max=block_budget)
            tail_values = tail_start.unsqueeze(1) + tail_offsets.unsqueeze(0)
            tail_valid = tail_offsets.unsqueeze(0) < tail_count.unsqueeze(1)
            destination = valid_block_count.unsqueeze(1) * compress_ratio + tail_offsets.unsqueeze(0)
            tail_written = torch.where(tail_valid, tail_values, torch.full_like(tail_values, -1)).to(torch.int32)
            result.scatter_(1, destination, tail_written)

            selected_tokens[batch_idx, query_start:query_end] = result

    return selected_tokens


def _gathered_qsa_gqa_attention_chunk(
    grouped_query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Evaluate one bounded-workspace chunk of the PyTorch QSA oracle.

    ``grouped_query`` uses ``[B, Q, Hkv, G, D]``; K/V remain
    ``[B, S_global, Hkv, D]`` and IDs use ``[B, Q, K]``.

    Args:
        grouped_query: Local grouped queries ``[B, Q, Hkv, G, D]``.
        key: Global keys ``[B, S_global, Hkv, D]``.
        value: Global values ``[B, S_global, Hkv, D]``.
        selected_token_ids: Global logical IDs ``[B, Q, K]``; ``-1`` marks
            invalid fixed-width slots.
        softmax_scale: Score multiplier.

    Returns:
        Chunk output ``[B, Q, Hq, D]`` in the query dtype.
    """
    batch_size = grouped_query.shape[0]
    token_ids = selected_token_ids.long()
    valid = token_ids >= 0
    safe_ids = token_ids.clamp_min(0)
    batch_ids = torch.arange(batch_size, device=grouped_query.device).view(batch_size, 1, 1)

    # Advanced indexing keeps the physical KV-head axis intact:
    # [B, Q, K, Hkv, D] -> [B, Q, Hkv, K, D].
    gathered_key = key[batch_ids, safe_ids].permute(0, 1, 3, 2, 4)
    gathered_value = value[batch_ids, safe_ids].permute(0, 1, 3, 2, 4)
    scores = torch.einsum(
        "bqhgd,bqhkd->bqhgk",
        grouped_query.float(),
        gathered_key.float(),
    )
    scores = scores * softmax_scale
    score_valid = valid[:, :, None, None, :]
    scores = scores.masked_fill(~score_valid, -torch.inf)

    # Padding queries have no IDs. Give softmax a finite row, then erase every
    # probability so both output and Q/K/V gradient are exactly zero.
    has_tokens = valid.any(dim=-1)
    scores = torch.where(has_tokens[:, :, None, None, None], scores, torch.zeros_like(scores))
    probabilities = torch.softmax(scores, dim=-1).masked_fill(~score_valid, 0.0)
    chunk_output = torch.einsum(
        "bqhgk,bqhkd->bqhgd",
        probabilities,
        gathered_value.float(),
    )
    return chunk_output.flatten(2, 3).to(grouped_query.dtype)


def gathered_qsa_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Run the differentiable PyTorch QSA oracle without expanding K/V heads.

    This implementation is retained for CPU execution and numerical parity.
    CUDA training with ``backend.attn='flex'`` dispatches to FlexAttention
    instead. The oracle uses a private fixed query chunk solely to bound
    temporary gathered K/V storage; it has no public model configuration.

    Args:
        query: Main normalized/rotated local queries ``[B, S_query, Hq, D]``.
        key: Main normalized/rotated global keys ``[B, S_global, Hkv, D]``.
        value: Main global values ``[B, S_global, Hkv, D]``.
        selected_token_ids: Indexer output ``[B, S_query, K]``. IDs are global
            logical positions in the same batch row; ``-1`` marks fixed-width
            padding.
        softmax_scale: Score multiplier, defaulting to ``1 / sqrt(D)``.

    Returns:
        Sparse attention output ``[B, S_query, Hq, D]``. Rows with no selected
        IDs (right-tail padding queries) are exactly zero.
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("QSA query, key, and value must use [B, S, H, D] layout")
    if key.shape != value.shape:
        raise ValueError(f"QSA key and value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}")
    batch_size, query_sequence_length, num_query_heads, head_dim = query.shape
    kv_sequence_length = key.shape[1]
    if key.shape[0] != batch_size or key.shape[-1] != head_dim:
        raise ValueError("QSA main Q/K/V batch and head dimensions must match")
    num_kv_heads = key.shape[2]
    if num_kv_heads <= 0 or num_query_heads % num_kv_heads != 0:
        raise ValueError(f"QSA requires Hq divisible by Hkv, got Hq={num_query_heads}, Hkv={num_kv_heads}")
    if selected_token_ids.ndim != 3 or selected_token_ids.shape[:2] != (batch_size, query_sequence_length):
        raise ValueError(
            f"selected_token_ids must be [B, S, K] matching the queries; got {tuple(selected_token_ids.shape)}"
        )
    if query_sequence_length == 0:
        return torch.empty_like(query)
    if bool(((selected_token_ids < -1) | (selected_token_ids >= kv_sequence_length)).any()):
        raise ValueError("QSA selected token IDs must be -1 or a valid logical sequence position")

    scale = head_dim**-0.5 if softmax_scale is None else softmax_scale
    query_groups = num_query_heads // num_kv_heads
    grouped_query = query.unflatten(2, (num_kv_heads, query_groups))
    outputs: list[torch.Tensor] = []

    for query_start in range(0, query_sequence_length, _PYTORCH_ORACLE_QUERY_CHUNK_SIZE):
        query_end = min(query_start + _PYTORCH_ORACLE_QUERY_CHUNK_SIZE, query_sequence_length)
        chunk_query = grouped_query[:, query_start:query_end]
        chunk_token_ids = selected_token_ids[:, query_start:query_end]
        chunk_output = _gathered_qsa_gqa_attention_chunk(
            chunk_query,
            key,
            value,
            chunk_token_ids,
            softmax_scale=scale,
        )
        outputs.append(chunk_output)

    return torch.cat(outputs, dim=1)


def qsa_gqa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_token_ids: torch.Tensor,
    *,
    backend: str,
    softmax_scale: float | None = None,
) -> torch.Tensor:
    """Dispatch QSA to the selected CUDA kernel or the PyTorch CPU oracle.

    CPU execution always uses the oracle so model construction, checkpoint
    inspection, and distributed CPU parity tests need no compiled kernels.
    CUDA execution is strict: unsupported backends or dtypes are reported
    rather than silently falling back to the gathered implementation.

    Args:
        query: Tensor [batch, local_queries, query_heads, head_dim].
        key: Tensor [batch, global_keys, kv_heads, head_dim].
        value: Tensor with key's layout. Q/K/V share one dtype and device.
        selected_token_ids: Signed IDs [batch, local_queries, routes] in global
            K/V coordinates; invalid CUDA IDs are padding and duplicates collapse.
        backend: CUDA backend, "flex" or "cute". FA4 requires SM90 BF16 D256.
        softmax_scale: Optional positive QK score multiplier.

    Returns:
        Independent tensor [batch, local_queries, query_heads, head_dim] with
        query's dtype/device. Empty route rows produce zero.
    """
    if not query.is_cuda:
        return gathered_qsa_gqa_attention(
            query,
            key,
            value,
            selected_token_ids,
            softmax_scale=softmax_scale,
        )
    if backend == "cute":
        return fa4_sparse_gqa_attention(query, key, value, selected_token_ids, softmax_scale=softmax_scale)
    if backend != "flex":
        raise RuntimeError(
            f"Qwen3.8-Flash-Next CUDA QSA requires backend.attn='flex' or 'cute', got {backend!r}; "
            "call gathered_qsa_gqa_attention directly for a numerical oracle"
        )
    if any(tensor.dtype != torch.bfloat16 for tensor in (query, key, value)):
        raise RuntimeError("Qwen3.8-Flash-Next flex QSA requires CUDA BF16 query, key, and value tensors")
    return flex_sparse_gqa_attention(
        query,
        key,
        value,
        selected_token_ids,
        softmax_scale=softmax_scale,
    )


class Qwen3_8_FlashNextQSAIndexer(nn.Module):
    """Frozen, hookable Qwen3.8-Flash-Next compressed-block indexer.

    ``forward`` has no cache or mutable routing state and returns the complete
    logical-ID tensor ``[B, S, indexer_budget + compress_ratio - 1]``.  A normal
    PyTorch forward hook can therefore capture the exact routing artifact used
    by the subsequent sparse attention.

    The fused projection produces raw query/key layout
    ``[B, S, (H_index + 1) * D_index]``.  Queries become
    ``[B, S, H_index, D_index]``.  Raw keys become ``[B, S, 1, D_index]`` and
    complete consecutive groups are averaged in FP32 into
    ``[B, floor(S / c), 1, D_index]`` before K RMSNorm and group-start RoPE.
    """

    def __init__(self, config: object, backend: BackendConfig) -> None:
        super().__init__()
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.num_query_heads = int(getattr(config, "indexer_n_heads"))
        self.num_key_heads = int(getattr(config, "indexer_kv_heads"))
        self.head_dim = int(getattr(config, "indexer_head_dim"))
        self.token_budget = int(getattr(config, "indexer_budget"))
        self.compress_ratio = int(getattr(config, "indexer_compress_ratio"))
        query_chunk_size = getattr(config, "qsa_indexer_query_chunk_size", None)
        self.query_chunk_size = None if query_chunk_size is None else int(query_chunk_size)
        if self.num_query_heads <= 0 or self.head_dim <= 0:
            raise ValueError("QSA index head count and dimension must be positive")
        if self.num_key_heads != 1:
            raise ValueError(f"Qwen3.8-Flash-Next QSA requires one index KV head, got {self.num_key_heads}")
        if self.compress_ratio <= 1 or self.token_budget <= 0 or self.token_budget % self.compress_ratio != 0:
            raise ValueError("QSA indexer_budget must be positive and divisible by indexer_compress_ratio > 1")
        if self.query_chunk_size is not None and self.query_chunk_size <= 0:
            raise ValueError("qsa_indexer_query_chunk_size must be positive or None (size by score memory)")

        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.index_qk_proj = initialize_linear_module(
            backend.linear,
            self.hidden_size,
            (self.num_query_heads + self.num_key_heads) * self.head_dim,
            bias=False,
            dtype=dtype,
        )
        eps = float(getattr(config, "rms_norm_eps"))
        self.q_layernorm = Qwen3NextRMSNorm(self.head_dim, eps=eps)
        self.k_layernorm = Qwen3NextRMSNorm(self.head_dim, eps=eps)

        # The supplied gold path emits discrete top-k IDs and contains no
        # indexer auxiliary loss or STE.  Make that training contract explicit
        # instead of silently leaving parameters trainable with grad=None.
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        cp_context: Qwen3_8_FlashNextCPContext | None = None,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return selected logical token IDs for every physical query row.

        Args:
            hidden_states: Decoder block input ``[B, S, hidden_size]``.
            freqs_cis: Composed Qwen3.8-Flash-Next rotary values ``[B, S, D_rope]`` stored
                as concatenated cosine/sine halves.
            attention_mask: Optional binary right-tail mask ``[B, S]``.
            cp_context: Optional contiguous CP metadata. When present,
                ``hidden_states`` and ``freqs_cis`` contain the local query
                shard while raw logical lengths and selected IDs use global
                sequence coordinates.
            cu_seqlens: Optional packed-document boundaries ``[num_docs + 1]``
                for a THD batch of one row. Routes then stay confined to each
                query's document and use global flattened token IDs.

        Returns:
            int32 logical IDs ``[B, S, token_budget + compress_ratio - 1]``.
            Padding-query rows contain only ``-1``. With ``cu_seqlens`` the
            IDs are global flattened positions in ``[0, T)``.
        """
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(f"QSA hidden_states must be [B, S, {self.hidden_size}], got {tuple(hidden_states.shape)}")
        if cu_seqlens is not None:
            if attention_mask is not None:
                raise ValueError("QSA packed routing takes document boundaries from cu_seqlens, not attention_mask")
            return self._forward_packed(
                hidden_states,
                freqs_cis=freqs_cis,
                cu_seqlens=cu_seqlens,
                cp_context=cp_context,
            )
        batch_size, sequence_length, _ = hidden_states.shape
        if cp_context is None:
            sequence_lengths = right_padded_sequence_lengths(
                attention_mask,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=hidden_states.device,
            )
            query_position_offset = 0
            global_sequence_length = sequence_length
        else:
            if cp_context.global_input_ids.shape[0] != batch_size:
                raise ValueError(
                    "QSA CP batch size disagrees with the global raw-ID context; "
                    f"got local={batch_size}, global={cp_context.global_input_ids.shape[0]}"
                )
            if sequence_length != cp_context.local_sequence_length:
                raise ValueError(
                    "QSA local sequence length disagrees with its CP context; "
                    f"got local={sequence_length}, context={cp_context.local_sequence_length}"
                )
            sequence_lengths = cp_context.global_sequence_lengths.to(hidden_states.device)
            query_position_offset = cp_context.local_sequence_start
            global_sequence_length = cp_context.global_sequence_length
            if sequence_length % self.compress_ratio != 0:
                raise ValueError(
                    "QSA contiguous CP requires every local sequence shard to be divisible by the compression ratio; "
                    f"got local={sequence_length}, ratio={self.compress_ratio}"
                )

        projected = self.index_qk_proj(hidden_states)
        query_width = self.num_query_heads * self.head_dim
        raw_query = projected[..., :query_width].unflatten(-1, (self.num_query_heads, self.head_dim))
        raw_key = projected[..., query_width:].unflatten(-1, (self.num_key_heads, self.head_dim))
        index_query = apply_qsa_rope(self.q_layernorm(raw_query), freqs_cis)

        num_blocks = sequence_length // self.compress_ratio
        grouped_key = raw_key[:, : num_blocks * self.compress_ratio].unflatten(1, (num_blocks, self.compress_ratio))
        compressed_key = grouped_key.float().mean(dim=2).to(raw_key.dtype)
        compressed_freqs = freqs_cis[:, : num_blocks * self.compress_ratio : self.compress_ratio]
        compressed_key = apply_qsa_rope(self.k_layernorm(compressed_key), compressed_freqs)
        if cp_context is not None:
            compressed_key = qwen3_8_flash_next_cp_all_gather(
                compressed_key,
                cp_context,
                sequence_dim=1,
                differentiable=False,
            )

        return select_qsa_token_ids(
            index_query,
            compressed_key,
            sequence_lengths,
            token_budget=self.token_budget,
            compress_ratio=self.compress_ratio,
            query_chunk_size=self.query_chunk_size,
            query_position_offset=query_position_offset,
            global_sequence_length=global_sequence_length,
        )

    def _forward_packed(
        self,
        hidden_states: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cp_context: Qwen3_8_FlashNextCPContext | None = None,
    ) -> torch.Tensor:
        """Select document-confined routes for one packed THD row.

        Each document is unpacked into its own right-padded pseudo-batch row so
        the per-row selection runs unchanged: compression groups restart at
        every document start and visibility is bounded by the document length.
        Local per-document IDs are then offset by the document start and packed
        back into the flattened layout.

        Args:
            hidden_states: Packed decoder block input ``[1, T, hidden_size]``.
            freqs_cis: Document-relative rotary values ``[1, T, D_rope]``.
            cu_seqlens: Strictly increasing boundaries ``[num_docs + 1]``
                starting at zero and ending at ``T``.

        Returns:
            int32 global flattened IDs ``[1, T, token_budget +
            compress_ratio - 1]`` with ``-1`` padding.
        """
        if hidden_states.shape[0] != 1:
            raise ValueError(f"QSA packed routing requires a THD batch of one row, got batch={hidden_states.shape[0]}")
        total_tokens = hidden_states.shape[1]
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError(f"cu_seqlens must be a rank-1 boundary tensor, got shape {tuple(cu_seqlens.shape)}")
        boundaries = cu_seqlens.to(device=hidden_states.device, dtype=torch.long)
        document_lengths = boundaries.diff()
        if cp_context is not None:
            if total_tokens != cp_context.local_sequence_length:
                raise ValueError(
                    "QSA packed CP local sequence length disagrees with its CP context; "
                    f"got local={total_tokens}, context={cp_context.local_sequence_length}"
                )
            boundary_cap = cp_context.global_sequence_length
        else:
            boundary_cap = total_tokens
        if (
            int(boundaries[0]) != 0
            or int(boundaries[-1]) > boundary_cap
            or (cp_context is None and int(boundaries[-1]) != total_tokens)
            or bool((document_lengths <= 0).any())
        ):
            raise ValueError(
                "cu_seqlens must start at 0, end at the packed token count, and be strictly increasing; "
                f"got first={int(boundaries[0])}, last={int(boundaries[-1])}, T={boundary_cap}"
            )
        if cp_context is not None:
            return self._forward_packed_cp(
                hidden_states,
                freqs_cis=freqs_cis,
                boundaries=boundaries,
                cp_context=cp_context,
            )

        projected = self.index_qk_proj(hidden_states)
        query_width = self.num_query_heads * self.head_dim
        raw_query = projected[..., :query_width].unflatten(-1, (self.num_query_heads, self.head_dim))
        raw_key = projected[..., query_width:].unflatten(-1, (self.num_key_heads, self.head_dim))

        # Unpack the single packed row into right-padded per-document rows.
        document_starts = boundaries[:-1]
        max_length = int(document_lengths.max())
        local_positions = torch.arange(max_length, device=hidden_states.device)
        row_positions = document_starts[:, None] + local_positions[None, :]
        row_valid = local_positions[None, :] < document_lengths[:, None]
        safe_positions = row_positions.clamp(max=total_tokens - 1)
        query_rows = raw_query[0, safe_positions]
        key_rows = raw_key[0, safe_positions]
        freqs_rows = freqs_cis[0, safe_positions]

        index_query = apply_qsa_rope(self.q_layernorm(query_rows), freqs_rows)
        num_blocks = max_length // self.compress_ratio
        grouped_key = key_rows[:, : num_blocks * self.compress_ratio].unflatten(1, (num_blocks, self.compress_ratio))
        compressed_key = grouped_key.float().mean(dim=2).to(key_rows.dtype)
        compressed_freqs = freqs_rows[:, : num_blocks * self.compress_ratio : self.compress_ratio]
        compressed_key = apply_qsa_rope(self.k_layernorm(compressed_key), compressed_freqs)

        document_routes = select_qsa_token_ids(
            index_query,
            compressed_key,
            document_lengths,
            token_budget=self.token_budget,
            compress_ratio=self.compress_ratio,
            query_chunk_size=self.query_chunk_size,
        )
        global_routes = torch.where(
            document_routes >= 0,
            document_routes + document_starts[:, None, None].to(torch.int32),
            document_routes.new_full((), -1),
        )
        packed_routes = global_routes.new_full((1, total_tokens, global_routes.shape[-1]), -1)
        packed_routes[0, row_positions[row_valid]] = global_routes[row_valid]
        return packed_routes

    def _forward_packed_cp(
        self,
        hidden_states: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        boundaries: torch.Tensor,
        cp_context: Qwen3_8_FlashNextCPContext,
    ) -> torch.Tensor:
        """Select document-confined routes for one rank's packed query shard.

        The frozen indexer needs no gradients, so raw index keys and rotary
        values are gathered globally with plain collectives and every document
        is compressed identically on all ranks. Documents intersecting the
        local shard are scored one at a time with the existing per-row
        selection; queries in the CP padding tail (beyond the final document
        boundary) keep all ``-1`` routes.

        Args:
            hidden_states: Local packed shard ``[1, S_local, hidden_size]``.
            freqs_cis: Local document-relative rotary values ``[1, S_local, D_rope]``.
            boundaries: Validated global boundaries ``[num_docs + 1]``.
            cp_context: Contiguous CP metadata for the packed global row.

        Returns:
            int32 global flattened IDs ``[1, S_local, token_budget +
            compress_ratio - 1]`` with ``-1`` padding.
        """
        projected = self.index_qk_proj(hidden_states)
        query_width = self.num_query_heads * self.head_dim
        raw_query = projected[..., :query_width].unflatten(-1, (self.num_query_heads, self.head_dim))
        raw_key = projected[..., query_width:].unflatten(-1, (self.num_key_heads, self.head_dim))
        index_query = apply_qsa_rope(self.q_layernorm(raw_query), freqs_cis)

        global_raw_key = qwen3_8_flash_next_cp_all_gather(
            raw_key.flatten(2, 3),
            cp_context,
            sequence_dim=1,
            differentiable=False,
        ).unflatten(-1, (self.num_key_heads, self.head_dim))
        global_freqs = qwen3_8_flash_next_cp_all_gather(
            freqs_cis,
            cp_context,
            sequence_dim=1,
            differentiable=False,
        )

        local_start = cp_context.local_sequence_start
        local_end = cp_context.local_sequence_end
        route_width = self.token_budget + self.compress_ratio - 1
        packed_routes = torch.full(
            (1, hidden_states.shape[1], route_width),
            -1,
            dtype=torch.int32,
            device=hidden_states.device,
        )
        for document_start, document_end in zip(boundaries.tolist(), boundaries[1:].tolist()):
            overlap_start = max(document_start, local_start)
            overlap_end = min(document_end, local_end)
            if overlap_start >= overlap_end:
                continue
            document_length = document_end - document_start
            query_slice = index_query[:, overlap_start - local_start : overlap_end - local_start]

            document_key = global_raw_key[:, document_start:document_end]
            document_freqs = global_freqs[:, document_start:document_end]
            num_blocks = document_length // self.compress_ratio
            grouped_key = document_key[:, : num_blocks * self.compress_ratio].unflatten(
                1, (num_blocks, self.compress_ratio)
            )
            compressed_key = grouped_key.float().mean(dim=2).to(document_key.dtype)
            compressed_freqs = document_freqs[:, : num_blocks * self.compress_ratio : self.compress_ratio]
            compressed_key = apply_qsa_rope(self.k_layernorm(compressed_key), compressed_freqs)

            document_routes = select_qsa_token_ids(
                query_slice,
                compressed_key,
                boundaries.new_tensor([document_length], dtype=torch.long),
                token_budget=self.token_budget,
                compress_ratio=self.compress_ratio,
                query_chunk_size=self.query_chunk_size,
                query_position_offset=overlap_start - document_start,
                global_sequence_length=document_length,
            )
            global_routes = torch.where(
                document_routes >= 0,
                document_routes + document_start,
                document_routes.new_full((), -1),
            )
            packed_routes[:, overlap_start - local_start : overlap_end - local_start] = global_routes
        return packed_routes

    @torch.no_grad()
    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize frozen indexer parameters for scratch-model construction."""
        nn.init.trunc_normal_(self.index_qk_proj.weight, mean=0.0, std=init_std)
        self.q_layernorm.reset_parameters()
        self.k_layernorm.reset_parameters()


__all__ = [
    "Qwen3_8_FlashNextQSAIndexer",
    "apply_qsa_rope",
    "gathered_qsa_gqa_attention",
    "qsa_gqa_attention",
    "right_padded_sequence_lengths",
    "select_qsa_token_ids",
]
