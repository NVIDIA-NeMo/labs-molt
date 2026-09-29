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

"""Differentiable CSA2 attention from the official DeepSeek V4.1 inference model.

Full layers publish compressed KV, index keys, and selected positions. Reindex
layers replace only the selection; Reuse layers consume it unchanged. Immutable
per-forward state keeps source gradients intact through activation recomputation.
Weights are dequantized for training while the released FP8 window KV and FP4
compressed KV/indexer representations retain their quantize/dequantize boundaries. The released indexer
weights are frozen because the hard top-k operation provides no language-model
gradient and the inference release does not implement indexer distillation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import partial

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig, initialize_rms_norm_module
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4FP32Parameter, DeepseekV4GroupedLinear
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.cp import gather_sequence
from nemo_automodel.components.models.deepseek_v41.indexer import indexer_scores
from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41RMSNorm
from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache
from nemo_automodel.shared.utils import dtype_from_str


@dataclass(frozen=True)
class DeepseekV41AttentionState:
    """CSA2 state owned by one full-sequence model forward.

    Attributes:
        compressed_kv: Rotated KV tensor of shape [batch, compressed, head_dim].
        index_keys: Rotated index keys of shape [batch, compressed, index_head_dim].
        topk_indices: Compressed-relative indices of shape [batch, sequence, topk],
            with -1 denoting absent positions.
        candidates: Optional candidate mask of shape [batch, sequence, compressed].
        compressed_valid: Optional valid-group mask of shape [batch, compressed].
        compressed_seq_ids: Optional document IDs [batch, compressed], zero for padding.
        compression_ratio: Number of tokens represented by a compressed position.

    Under CP, compressed and index-key axes span the global sequence while
    sequence axes in topk_indices and candidates contain only local queries.

    Tensor fields retain autograd history and are never modified by consumers.
    A model creates an empty state for every forward, including every microbatch.
    """

    compressed_kv: torch.Tensor | None = None
    index_keys: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None
    candidates: torch.Tensor | None = None
    compressed_valid: torch.Tensor | None = None
    compression_ratio: int = 0
    compressed_seq_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class DeepseekV41AttentionOutput:
    """Attention result and the shared state for the next layer.

    Attributes:
        hidden_states: Tensor of shape [batch, sequence, hidden].
        state: Shared tensor layouts documented by DeepseekV41AttentionState.
    """

    hidden_states: torch.Tensor
    state: DeepseekV41AttentionState


class _RotaryEmbedding(nn.Module):
    """Adjacent-pair RoPE with the official frequency-only YaRN adjustment."""

    def __init__(self, config: DeepseekV41TextConfig, *, compressed: bool) -> None:
        super().__init__()
        self.dim = config.qk_rope_head_dim
        self.theta = config.compress_rope_theta if compressed else config.rope_theta
        scaling = config.rope_scaling if compressed else None
        self.factor = float(scaling.get("factor", 1.0)) if scaling else 1.0
        self.original_length = int(scaling.get("original_max_position_embeddings", 0)) if scaling else 0
        self.beta_fast = float(scaling.get("beta_fast", 32)) if scaling else 32.0
        self.beta_slow = float(scaling.get("beta_slow", 1)) if scaling else 1.0

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """Construct FP32 phase angles without a cast-sensitive frequency buffer.

        Args:
            positions: Integer tensor of shape [batch, sequence].

        Returns:
            FP32 angles of shape [batch, sequence, rotary_pairs], where
            rotary_pairs is half the rotary head dimension.
        """
        frequencies = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, device=positions.device, dtype=torch.float32) / self.dim)
        )
        if self.original_length > 0:
            low = max(
                math.floor(
                    self.dim
                    * math.log(self.original_length / (self.beta_fast * 2 * math.pi))
                    / (2 * math.log(self.theta))
                ),
                0,
            )
            high = min(
                math.ceil(
                    self.dim
                    * math.log(self.original_length / (self.beta_slow * 2 * math.pi))
                    / (2 * math.log(self.theta))
                ),
                self.dim - 1,
            )
            ramp = (
                (torch.arange(self.dim // 2, device=positions.device, dtype=torch.float32) - low)
                / max(high - low, 1e-3)
            ).clamp(0, 1)
            frequencies = frequencies / self.factor * ramp + frequencies * (1 - ramp)
        return positions.float().unsqueeze(-1) * frequencies


def _apply_rope(values: torch.Tensor, angles: torch.Tensor, *, inverse: bool = False) -> torch.Tensor:
    """Rotate the final channels without changing the input storage.

    Args:
        values: Tensor of shape [batch, sequence, channels] or
            [batch, sequence, heads, channels].
        angles: FP32 tensor of shape [batch, sequence, rotary_pairs]. The last
            2 * rotary_pairs channels of values use adjacent-pair rotation.
        inverse: Conjugate the rotation for the attention output.

    Returns:
        Tensor with the shape and dtype of values, in independent storage.
    """
    rotary_dim = angles.shape[-1] * 2
    pairs = torch.view_as_complex(values[..., -rotary_dim:].float().unflatten(-1, (-1, 2)).contiguous())
    # Preserve the released complex-multiply rounding. Separate real-valued
    # multiplies change some BF16 values and can cross subsequent FP4/FP8 bins.
    frequencies = torch.polar(torch.ones_like(angles), angles)
    if values.ndim == 4:
        frequencies = frequencies.unsqueeze(2)
    if inverse:
        frequencies = frequencies.conj()
    rotated = torch.view_as_real(pairs * frequencies).flatten(-2).to(values.dtype)
    return torch.cat((values[..., :-rotary_dim], rotated), dim=-1)


class _CompressorLinear(nn.Linear):
    """Keep pooling weights in FP32 while selecting compute from the input dtype."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Project [..., input] into [..., output] with the input's compute dtype."""
        return F.linear(values, self.weight.to(values.dtype))


class _Compressor(nn.Module):
    """Non-overlapping channelwise softmax pooling; ratio one is a projection."""

    def __init__(
        self, config: DeepseekV41TextConfig, *, ratio: int, dtype: torch.dtype, rms_norm: str = "torch_fp32"
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.wkv = _CompressorLinear(config.hidden_size, config.head_dim, bias=False, dtype=torch.float32)
        norm = (
            partial(initialize_rms_norm_module, "te", device=self.wkv.weight.device)
            if rms_norm == "te"
            else DeepseekV41RMSNorm
        )
        self.norm = norm(config.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        if ratio > 1:
            self.wgate = nn.Linear(config.hidden_size, config.head_dim, bias=False, dtype=torch.float32)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return complete compressed groups before rotary embedding.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, floor(sequence / ratio), head_dim] in the
            input dtype. Incomplete trailing groups contribute no output.
        """
        if self.ratio == 1:
            return self.norm(self.wkv(hidden_states))
        length = hidden_states.shape[1] // self.ratio * self.ratio
        values = self.wkv(hidden_states[:, :length].float())
        scores = self.wgate(hidden_states[:, :length].float())
        values = values.unflatten(1, (-1, self.ratio))
        scores = scores.unflatten(1, (-1, self.ratio))
        pooled = (values * scores.softmax(dim=2)).sum(dim=2).to(hidden_states.dtype)
        return self.norm(pooled)


def _select_candidate_blocks(
    scores: torch.Tensor, visible_lengths: torch.Tensor, *, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """Keep high-scoring blocks and always retain the latest visible block.

    Args:
        scores: Causally masked scores of shape [batch, sequence, compressed].
        visible_lengths: Integer tensor of shape [batch, sequence, 1] counting
            visible compressed positions for each query.
        topk_blocks: Maximum retained blocks per query.
        block_size: Compressed positions per block.

    Returns:
        Boolean candidate mask of shape [batch, sequence, compressed].
    """
    width = scores.shape[-1]
    block_scores = F.pad(scores, (0, -width % block_size), value=-torch.inf)
    block_scores = block_scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    last_block = (visible_lengths - 1) // block_size
    block_ids = torch.arange(block_scores.shape[-1], device=scores.device)
    block_scores = block_scores.masked_fill(block_ids == last_block, torch.inf)
    selected = block_scores.topk(min(topk_blocks, block_scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(block_scores, dtype=torch.bool).scatter(-1, selected.indices, selected.values > -torch.inf)
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class _Indexer(nn.Module):
    """Frozen released CSA2 indexer with shared keys and hierarchical selection."""

    def __init__(
        self,
        config: DeepseekV41TextConfig,
        *,
        layer_idx: int,
        dtype: torch.dtype,
        rms_norm: str = "torch_fp32",
        attn_backend: str = "eager",
    ) -> None:
        super().__init__()
        self.attn_backend = attn_backend
        self.owns_keys = layer_idx in config.kv_source_layer_ids
        self.is_candidate_source = layer_idx == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_idx
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.candidate_topk_blocks = config.candidate_topk_blocks
        self.candidate_block_size = config.candidate_block_size
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False, dtype=dtype)
        if self.owns_keys:
            self.wk = nn.Linear(config.head_dim, self.head_dim, bias=False, dtype=dtype)
            norm = (
                partial(initialize_rms_norm_module, "te", device=self.wk.weight.device)
                if rms_norm == "te"
                else DeepseekV41RMSNorm
            )
            self.k_norm = norm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        query_latent: torch.Tensor,
        latent: torch.Tensor | None,
        angles: torch.Tensor,
        compressed_angles: torch.Tensor,
        state: DeepseekV41AttentionState,
        position_ids: torch.Tensor | None = None,
        cp_group: dist.ProcessGroup | None = None,
        packed_seq_ids: torch.Tensor | None = None,
    ) -> DeepseekV41AttentionState:
        """Produce index keys when owned, then replace the current selection.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            query_latent: Tensor of shape [batch, sequence, q_lora_rank].
            latent: Optional unrotated KV of shape [batch, compressed, head_dim].
            angles: FP32 rotary angles of shape [batch, sequence, rotary_pairs].
            compressed_angles: FP32 angles of shape [batch, compressed, rotary_pairs].
            state: Shared tensors with layouts in DeepseekV41AttentionState.
            position_ids: Optional physical global positions [batch, local_sequence].
            packed_seq_ids: Optional document IDs [batch, local_sequence], zero for padding.
            cp_group: Optional CP group; latent and compressed_angles contain
                local complete groups, while state index keys span all ranks.

        Returns:
            New state with index_keys [batch, compressed, index_head_dim],
            topk_indices [batch, sequence, topk], and optional candidates
            [batch, sequence, compressed]. Compressed KV storage is preserved.
        """
        keys = state.index_keys
        if self.owns_keys:
            if latent is None:
                raise ValueError("A Full CSA2 layer must provide its unrotated compressed latent")
            keys = quantize_cache(
                _apply_rope(self.k_norm(self.wk(latent)), compressed_angles), format="mxfp4", block_size=32
            )
            keys = gather_sequence(keys, cp_group)
        if keys is None:
            raise ValueError("A Reindex CSA2 layer requires index keys from a preceding Full layer")
        batch, sequence, _ = hidden_states.shape
        width = keys.shape[1]
        if width == 0:
            return replace(
                state,
                index_keys=keys,
                topk_indices=torch.empty(batch, sequence, 0, dtype=torch.long, device=hidden_states.device),
                candidates=(
                    torch.empty(batch, sequence, 0, dtype=torch.bool, device=hidden_states.device)
                    if self.is_candidate_source
                    else state.candidates
                ),
            )
        queries = self.wq_b(query_latent).unflatten(-1, (self.num_heads, self.head_dim))
        queries = quantize_cache(_apply_rope(queries, angles), format="mxfp4", block_size=32)
        weights = self.weights_proj(hidden_states) * (self.head_dim**-0.5 * self.num_heads**-0.5)
        if position_ids is None:
            position_ids = torch.arange(sequence, device=hidden_states.device).unsqueeze(0)
        lengths = (position_ids + 1) // state.compression_ratio
        visible_lengths = lengths.unsqueeze(-1).expand(batch, -1, -1)
        allowed = torch.arange(width, device=hidden_states.device) < visible_lengths
        if state.compressed_valid is not None:
            allowed = allowed & state.compressed_valid.unsqueeze(1)
        if packed_seq_ids is not None:
            if state.compressed_seq_ids is None:
                raise ValueError("Packed CSA2 state requires compressed document IDs")
            allowed = allowed & (packed_seq_ids.unsqueeze(-1) == state.compressed_seq_ids.unsqueeze(1))
            allowed = allowed & (packed_seq_ids.unsqueeze(-1) > 0)
        if self.attn_backend == "tilelang":
            scores = indexer_scores(queries, keys, weights, allowed)
        else:
            # Preserve the reference's BF16 matmul result and reduction boundaries.
            scores = torch.einsum("bshd,btd->bsht", queries, keys)
            scores = (scores.relu() * weights.unsqueeze(-1)).sum(dim=2)
            scores = scores.masked_fill(~allowed, -torch.inf)
        if packed_seq_ids is not None:
            return self._select_packed(scores, packed_seq_ids, replace(state, index_keys=keys))
        candidates = state.candidates
        if self.is_candidate_source:
            candidates = (
                _select_candidate_blocks(
                    scores,
                    visible_lengths,
                    topk_blocks=self.candidate_topk_blocks,
                    block_size=self.candidate_block_size,
                )
                & allowed
            )
        elif self.uses_candidates:
            if candidates is None or candidates.shape != scores.shape:
                raise ValueError("A hierarchical Reindex layer requires candidates from its source layer")
            scores = scores.masked_fill(~candidates, -torch.inf)
        selected = scores.topk(min(self.topk, width), dim=-1, sorted=False)
        indices = selected.indices.sort(dim=-1).values
        # Invalid entries remain -1 even when fewer candidates than top-k exist.
        valid = torch.isfinite(scores.gather(-1, indices))
        indices = torch.where(valid, indices, -1)
        return replace(state, index_keys=keys, topk_indices=indices, candidates=candidates)

    def _select_packed(
        self, scores: torch.Tensor, sequence_ids: torch.Tensor, state: DeepseekV41AttentionState
    ) -> DeepseekV41AttentionState:
        """Run selection on each document's own key interval.

        topk tie-breaking depends on the physical row width. Selecting over an entire
        pack, even with other documents masked out, can therefore change a document's
        attention. Local intervals also reset candidate blocks at document boundaries.

        Args:
            scores: Causal, document-masked scores [batch, local_sequence, global_compressed].
            sequence_ids: Document IDs [batch, local_sequence], zero for padding.
            state: Global compressed metadata and keys described by DeepseekV41AttentionState.

        Returns:
            Immutable state with global-key topk indices [batch, local_sequence, topk]
            and source candidates [batch, local_sequence, global_compressed].
        """
        indices = torch.full(
            (*scores.shape[:2], min(self.topk, scores.shape[-1])),
            -1,
            device=scores.device,
            dtype=torch.long,
        )
        candidates = torch.zeros_like(scores, dtype=torch.bool) if self.is_candidate_source else state.candidates
        if self.uses_candidates and (candidates is None or candidates.shape != scores.shape):
            raise ValueError("A hierarchical Reindex layer requires candidates from its source layer")
        for row in range(scores.shape[0]):
            for document in sequence_ids[row].unique().tolist():
                if document == 0:
                    continue
                queries = torch.where(sequence_ids[row] == document)[0]
                keys = torch.where((state.compressed_seq_ids[row] == document) & state.compressed_valid[row])[0]
                if keys.numel() == 0:
                    continue
                document_scores = scores[row, queries[:, None], keys]
                if self.is_candidate_source:
                    visible = torch.isfinite(document_scores).sum(-1, keepdim=True)
                    keep = _select_candidate_blocks(
                        document_scores.unsqueeze(0),
                        visible.unsqueeze(0),
                        topk_blocks=self.candidate_topk_blocks,
                        block_size=self.candidate_block_size,
                    ).squeeze(0) & torch.isfinite(document_scores)
                    candidates[row, queries[:, None], keys] = keep
                elif self.uses_candidates:
                    document_scores = document_scores.masked_fill(~candidates[row, queries[:, None], keys], -torch.inf)
                selected = document_scores.topk(min(self.topk, keys.numel()), dim=-1, sorted=False)
                ordered = selected.indices.sort(-1).values
                valid = torch.isfinite(document_scores.gather(-1, ordered))
                indices[row, queries, : ordered.shape[-1]] = torch.where(valid, keys[ordered], -1)
        return replace(state, topk_indices=indices, candidates=candidates)


class DeepseekV41Attention(nn.Module):
    """Full-sequence CSA2 with local KV, shared compressed KV, and an attention sink.

    The training implementation supports eager, SDPA and TileLang attention with
    torch linear layers and eager FP32 or TE RMSNorm. Left padding, KV-cache
    decoding and tensor sharding remain unsupported. Context parallelism keeps
    contiguous local queries and exchanges window KV and shared compressed KV.
    Every local sequence must contain complete compression groups.
    """

    def __init__(self, config: DeepseekV41TextConfig, layer_idx: int, backend: BackendConfig) -> None:
        super().__init__()
        if backend.attn not in ("eager", "sdpa", "tilelang"):
            raise ValueError("DeepSeek V4.1 attention supports backend.attn='eager', 'sdpa', or 'tilelang'")
        if backend.attn == "tilelang" and config.attention_dropout:
            raise ValueError("The TileLang sparse attention backend requires attention_dropout=0")
        if backend.linear != "torch" or backend.rms_norm not in ("torch_fp32", "te"):
            raise ValueError("DeepSeek V4.1 attention requires torch linear layers and torch_fp32 or te RMSNorm")
        self.backend = backend
        self.cp_group: dist.ProcessGroup | None = None
        self.layer_idx = layer_idx
        self.compress_ratio = config.compress_ratios[layer_idx]
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_groups = config.o_groups
        self.window_size = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_kv_source = layer_idx in config.kv_source_layer_ids
        self.is_index_source = layer_idx in config.index_source_layer_ids
        dtype = dtype_from_str(config.torch_dtype, torch.bfloat16)
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False, dtype=dtype)
        # TE defaults to CUDA when device is omitted, including under a meta
        # construction context. Use the projection's actual device explicitly.
        norm = (
            partial(initialize_rms_norm_module, "te", device=self.wq_a.weight.device)
            if backend.rms_norm == "te"
            else DeepseekV41RMSNorm
        )
        self.q_norm = norm(config.q_lora_rank, eps=config.rms_norm_eps, dtype=dtype)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.wkv = nn.Linear(config.hidden_size, self.head_dim, bias=False, dtype=dtype)
        self.kv_norm = norm(self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.wo_a = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // self.num_groups,
            self.num_groups * config.o_lora_rank,
            self.num_groups,
        ).to(dtype=dtype)
        self.wo_b = nn.Linear(self.num_groups * config.o_lora_rank, config.hidden_size, bias=False, dtype=dtype)
        self.sinks_param = DeepseekV4FP32Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.rotary_emb = _RotaryEmbedding(config, compressed=bool(self.compress_ratio))
        self.compressor = (
            _Compressor(config, ratio=self.compress_ratio, dtype=dtype, rms_norm=backend.rms_norm)
            if self.is_kv_source
            else None
        )
        self.indexer = (
            _Indexer(config, layer_idx=layer_idx, dtype=dtype, rms_norm=backend.rms_norm, attn_backend=backend.attn)
            if self.is_index_source
            else None
        )

    def setup_cp_attention(self, cp_mesh: DeviceMesh) -> None:
        """Configure the model-owned KV transport through the shared CP hook."""
        self.cp_group = cp_mesh.get_group()

    @property
    def attn_sink(self) -> nn.Parameter:
        """Return the checkpoint's per-head FP32 attention sink parameter."""
        return self.sinks_param.weight

    def reset_parameters(self, init_std: float = 0.02) -> None:
        """Initialize every attention parameter after construction or meta materialization.

        Args:
            init_std: Standard deviation of projection weights.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
        for norm in (self.q_norm, self.kv_norm):
            nn.init.ones_(norm.weight)
        if self.compressor is not None:
            nn.init.ones_(self.compressor.norm.weight)
        if self.indexer is not None and self.indexer.owns_keys:
            nn.init.ones_(self.indexer.k_norm.weight)
        nn.init.zeros_(self.attn_sink)

    def _project_output(self, attended: torch.Tensor, angles: torch.Tensor, valid_tokens: torch.Tensor) -> torch.Tensor:
        """Undo RoPE on [batch, sequence, heads, head_dim] and project to hidden width.

        ``angles`` has shape [batch, sequence, rotary_pairs]; ``valid_tokens``
        is boolean [batch, sequence]. Return [batch, sequence, hidden] with
        padded queries zeroed, preserving the attention output dtype.
        """
        attended = _apply_rope(attended, angles, inverse=True)
        grouped = attended.reshape(*attended.shape[:2], self.num_groups, -1)
        output = self.wo_b(self.wo_a(grouped).flatten(2))
        return output.masked_fill(~valid_tokens.unsqueeze(-1), 0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        state: DeepseekV41AttentionState,
        attention_mask: torch.Tensor | None = None,
        cp_group: dist.ProcessGroup | None = None,
        packed_seq_ids: torch.Tensor | None = None,
    ) -> DeepseekV41AttentionOutput:
        """Apply attention and publish immutable state for the next layer.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            position_ids: Integer tensor of shape [batch, sequence] or [1, sequence],
                containing contiguous global positions for unpacked input and
                document-local positions for packed input.
            packed_seq_ids: Optional document IDs [batch, local_sequence], zero for padding.
                Document starts must be aligned to compression-group boundaries.
            state: Per-forward tensors documented in DeepseekV41AttentionState.
                Consumers must receive the state from their preceding layer.
            attention_mask: Optional binary right-padding mask of shape
                [batch, local_sequence], with one for tokens and zero for padding.
            cp_group: Optional CP group, overriding setup_cp_attention. Hidden
                states and positions are local shards; shared KV state is global.

        Returns:
            Output with hidden_states [batch, sequence, hidden] and the new state,
            whose tensors use the layouts in DeepseekV41AttentionState. Input
            tensors and the input state are never mutated.
        """
        batch, sequence, _ = hidden_states.shape
        if sequence == 0:
            raise ValueError("DeepSeek V4.1 attention requires a nonempty full sequence")
        cp_group = self.cp_group if cp_group is None else cp_group
        cp_size = 1 if cp_group is None else dist.get_world_size(cp_group)
        cp_rank = 0 if cp_group is None else dist.get_rank(cp_group)
        if cp_size > 1 and self.compress_ratio and sequence % self.compress_ratio:
            raise ValueError("DeepSeek V4.1 CP shards must contain complete compression groups")
        global_sequence = sequence * cp_size
        positions = torch.arange(sequence, device=hidden_states.device) + cp_rank * sequence
        if position_ids.shape not in ((1, sequence), (batch, sequence)):
            raise ValueError("DeepSeek V4.1 position_ids must have shape [batch, sequence] or [1, sequence]")
        if packed_seq_ids is None and not torch.equal(position_ids, positions.expand_as(position_ids)):
            raise ValueError("DeepSeek V4.1 attention supports only contiguous zero-based full-sequence position_ids")
        if packed_seq_ids is not None and packed_seq_ids.shape != (batch, sequence):
            raise ValueError("packed_seq_ids must have shape [batch, local_sequence]")
        valid_tokens = torch.ones(batch, sequence, dtype=torch.bool, device=hidden_states.device)
        if attention_mask is not None:
            if attention_mask.shape != (batch, sequence):
                raise ValueError("DeepSeek V4.1 attention_mask must have shape [batch, sequence]")
            if not torch.all((attention_mask == 0) | (attention_mask == 1)):
                raise ValueError("DeepSeek V4.1 attention_mask must contain only zero and one")
            valid_tokens = attention_mask.bool()
            if packed_seq_ids is None and torch.any(valid_tokens[:, 1:] & ~valid_tokens[:, :-1]):
                raise ValueError("DeepSeek V4.1 compression supports right padding only")
        global_valid = gather_sequence(valid_tokens, cp_group)
        if packed_seq_ids is None and torch.any(global_valid[:, 1:] & ~global_valid[:, :-1]):
            raise ValueError("DeepSeek V4.1 compression supports right padding only across CP ranks")
        global_seq_ids = None if packed_seq_ids is None else gather_sequence(packed_seq_ids, cp_group)
        angles = self.rotary_emb(position_ids)
        query_latent = self.q_norm(self.wq_a(hidden_states))
        query = _apply_rope(self.wq_b(query_latent).unflatten(-1, (self.num_heads, self.head_dim)), angles)
        kv = quantize_cache(_apply_rope(self.kv_norm(self.wkv(hidden_states)), angles), format="fp8", block_size=32)
        kv = gather_sequence(kv, cp_group)
        next_state = state
        if self.compress_ratio:
            width = global_sequence // self.compress_ratio
            local_width = sequence // self.compress_ratio
            compressed_angles = self.rotary_emb(
                position_ids[:, : local_width * self.compress_ratio : self.compress_ratio]
            )
            latent = None
            if self.compressor is not None:
                latent = self.compressor(hidden_states)
                compressed_valid = (
                    valid_tokens[:, : local_width * self.compress_ratio]
                    .unflatten(1, (local_width, self.compress_ratio))
                    .all(dim=-1)
                )
                next_state = DeepseekV41AttentionState(
                    compressed_kv=gather_sequence(
                        quantize_cache(_apply_rope(latent, compressed_angles), format="nvfp4", block_size=16), cp_group
                    ),
                    compressed_valid=gather_sequence(compressed_valid, cp_group),
                    compression_ratio=self.compress_ratio,
                    compressed_seq_ids=(
                        None
                        if global_seq_ids is None
                        else global_seq_ids[:, : width * self.compress_ratio : self.compress_ratio]
                    ),
                )
            elif next_state.compressed_kv is None or next_state.compression_ratio != self.compress_ratio:
                raise ValueError("A CSA2 consumer requires a preceding Full layer with the same compression ratio")
            if self.indexer is not None:
                next_state = self.indexer(
                    hidden_states,
                    query_latent=query_latent,
                    latent=latent,
                    angles=angles,
                    compressed_angles=compressed_angles,
                    state=next_state,
                    position_ids=positions.unsqueeze(0).expand(batch, -1),
                    cp_group=cp_group,
                    packed_seq_ids=packed_seq_ids,
                )
            if next_state.topk_indices is None or next_state.compressed_kv is None:
                raise ValueError("A Reuse CSA2 layer requires compressed KV and indices from its source")
            if next_state.compressed_kv.shape[:2] != (batch, width) or next_state.topk_indices.shape[:2] != (
                batch,
                sequence,
            ):
                raise ValueError("CSA2 state belongs to a different batch or sequence")
            kv = torch.cat((kv, next_state.compressed_kv), dim=1)
        if self.backend.attn == "tilelang":
            # Preserve the released sparse slot order and reuse V4's trainable
            # online-softmax kernel, including its BF16 probability boundary.
            starts = (positions - self.window_size + 1).clamp_min(0).expand(batch, -1)
            if packed_seq_ids is not None:
                starts = torch.maximum(starts, positions.unsqueeze(0) - position_ids)
            slots = starts.unsqueeze(-1) + torch.arange(min(global_sequence, self.window_size), device=positions.device)
            visible = (slots <= positions.view(1, -1, 1)) & (slots < global_sequence)
            safe_slots = slots.clamp_max(global_sequence - 1)
            visible = visible & global_valid.gather(1, safe_slots.flatten(1)).view_as(slots)
            if packed_seq_ids is not None:
                key_seq_ids = global_seq_ids.gather(1, safe_slots.flatten(1)).view_as(slots)
                visible = visible & (key_seq_ids == packed_seq_ids.unsqueeze(-1))
            indices = torch.where(visible, slots, -1)
            if self.compress_ratio:
                selected = next_state.topk_indices
                indices = torch.cat((indices, torch.where(selected >= 0, selected + global_sequence, -1)), dim=-1)
            indices = indices.masked_fill(~valid_tokens.unsqueeze(-1), -1)
            indices = F.pad(indices, (0, -indices.shape[-1] % 64), value=-1)
            attended = dsv4_sparse_attention(
                query,
                kv,
                self.sinks_param(query),
                indices,
                self.head_dim**-0.5,
                backend="tilelang",
                reference_rounding=True,
            )
            return DeepseekV41AttentionOutput(self._project_output(attended, angles, valid_tokens), next_state)
        # Dense masks are needed only by the eager/SDPA fallback; TileLang uses sparse indices.
        key_positions = torch.arange(global_sequence, device=positions.device)
        local_allowed = (key_positions.unsqueeze(0) <= positions.unsqueeze(1)) & (
            key_positions.unsqueeze(0) > positions.unsqueeze(1) - self.window_size
        )
        allowed = local_allowed.unsqueeze(0) & global_valid.unsqueeze(1)
        if packed_seq_ids is not None:
            allowed = allowed & (packed_seq_ids.unsqueeze(-1) == global_seq_ids.unsqueeze(1))
        if self.compress_ratio:
            # A separate sentinel column makes -1 masking safe when valid index 0
            # also occurs in the row; boolean scatter must never overwrite it.
            compressed_allowed = torch.zeros(batch, sequence, width + 1, dtype=torch.bool, device=hidden_states.device)
            indices = next_state.topk_indices
            compressed_allowed = compressed_allowed.scatter(-1, torch.where(indices >= 0, indices, width), True)
            allowed = torch.cat((allowed, compressed_allowed[..., :width]), dim=-1)
        # A zero-valued extra key contributes exp(attn_sink) only to the softmax
        # denominator. It keeps fully padded query rows numerically well-defined.
        kv = torch.cat((kv, kv.new_zeros(batch, 1, self.head_dim)), dim=1)
        bias = torch.zeros(batch, 1, sequence, allowed.shape[-1], device=hidden_states.device, dtype=torch.float32)
        bias = bias.masked_fill(~allowed.unsqueeze(1), -torch.inf).expand(-1, self.num_heads, -1, -1)
        sink = self.sinks_param(query).view(1, self.num_heads, 1, 1).expand(batch, -1, sequence, -1)
        bias = torch.cat((bias, sink), dim=-1)
        if self.backend.attn == "sdpa":
            attended = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                kv.unsqueeze(1),
                kv.unsqueeze(1),
                attn_mask=bias,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.head_dim**-0.5,
            ).transpose(1, 2)
        else:
            logits = torch.einsum("bshd,btd->bhst", query.float(), kv.float()) * self.head_dim**-0.5
            probabilities = (logits + bias).softmax(dim=-1)
            probabilities = F.dropout(probabilities, p=self.attention_dropout, training=self.training)
            attended = torch.einsum("bhst,btd->bshd", probabilities, kv.float()).to(query.dtype)
        return DeepseekV41AttentionOutput(self._project_output(attended, angles, valid_tokens), next_state)
