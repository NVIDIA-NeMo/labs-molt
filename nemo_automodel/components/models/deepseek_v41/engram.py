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

"""DeepSeek-V4.1 compressed-token Engram hashes and residual memory lookup.

The tokenizer normalization, per-layer hash seeds, and signed square-root gate
follow DeepSeek's released ``inference/engram.py`` and ``inference/model.py``.
V4.1 omits the short convolution used by earlier Engram architectures.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.tensor import DTensor
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextEngramTableConfig
from nemo_automodel.shared.import_utils import safe_import
from nemo_automodel.shared.utils import dtype_from_str

if TYPE_CHECKING:
    from .config import DeepseekV41TextConfig

_, tokenizers = safe_import("tokenizers")


def _compressed_token_map(tokenizer: PreTrainedTokenizerFast) -> tuple[tuple[int, ...], int]:
    """Normalize decoded vocabulary entries using the released token-ID contract."""
    sentinel = "\ue000"
    normalizers = tokenizers.normalizers
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(tokenizers.Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(tokenizers.Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_id: dict[str, int] = {}
    token_map = []
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        if key not in key_to_id:
            key_to_id[key] = len(key_to_id)
        token_map.append(key_to_id[key])
    return tuple(token_map), len(key_to_id)


def _next_prime(start: int, seen: set[int]) -> int:
    """Find the first unused prime above ``start`` without an optional dependency."""
    candidate = start + 1
    while True:
        is_prime = candidate >= 2 and (
            candidate == 2
            or (candidate % 2 != 0 and all(candidate % divisor for divisor in range(3, math.isqrt(candidate) + 1, 2)))
        )
        if is_prime and candidate not in seen:
            return candidate
        candidate += 1


class DeepseekV41NgramHash(nn.Module):
    """Build all Engram layers' hashes from complete, uncached input sequences.

    Args:
        config: Text configuration containing the released Engram dimensions.
        tokenizer: Fast tokenizer whose normalized vocabulary must have exactly
            ``config.engram_compressed_vocab_size`` entries.
    """

    def __init__(self, config: DeepseekV41TextConfig, tokenizer: PreTrainedTokenizerFast) -> None:
        super().__init__()
        self.max_ngram_size = config.engram_max_ngram_size
        self.n_heads = config.engram_n_heads
        self.layer_ids = tuple(config.engram_layer_ids)
        if not self.layer_ids or self.max_ngram_size < 2 or self.n_heads < 1 or config.engram_vocab_size < 2:
            raise ValueError(
                "Engram hashing requires enabled layers, n-gram order >= 2, heads >= 1, and bucket size >= 2"
            )
        if len(self.layer_ids) != len(config.engram_num_embeddings) or len(set(self.layer_ids)) != len(self.layer_ids):
            raise ValueError("Engram requires one table row count for each distinct layer ID")
        self._token_map_values, vocab_size = _compressed_token_map(tokenizer)
        if vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(
                "Engram compressed tokenizer vocabulary mismatch: "
                f"got {vocab_size}, expected {config.engram_compressed_vocab_size}; hash multipliers depend on this size"
            )
        if not 0 <= config.engram_pad_token_id < len(self._token_map_values):
            raise ValueError(f"Engram pad token ID {config.engram_pad_token_id} lies outside the tokenizer vocabulary")
        self.pad_id = self._token_map_values[config.engram_pad_token_id]
        primes, offsets, multipliers = [], [], []
        seen_primes: set[int] = set()
        multiplier_bound = max(1, (np.iinfo(np.int64).max // vocab_size) // 2)
        for layer_id, num_embeddings in zip(self.layer_ids, config.engram_num_embeddings):
            layer_primes = []
            for _ in range(self.max_ngram_size - 1):
                current = config.engram_vocab_size - 1
                for _ in range(self.n_heads):
                    current = _next_prime(current, seen_primes)
                    seen_primes.add(current)
                    layer_primes.append(current)
            if sum(layer_primes) > num_embeddings:
                raise ValueError(
                    f"Engram layer {layer_id} requires {sum(layer_primes)} rows for its hash buckets, "
                    f"but engram_num_embeddings specifies {num_embeddings}"
                )
            primes.append(tuple(layer_primes))
            offsets.append(tuple(int(value) for value in np.cumsum([0, *layer_primes[:-1]])))
            generator = np.random.default_rng(10007 * layer_id)
            values = generator.integers(0, multiplier_bound, size=(self.max_ngram_size,), dtype=np.int64)
            multipliers.append(tuple(int(value) for value in values * 2 + 1))
        self._prime_values = tuple(primes)
        self._offset_values = tuple(offsets)
        self._multiplier_values = tuple(multipliers)
        self.register_buffer("token_map", torch.tensor(self._token_map_values, dtype=torch.long), persistent=False)
        self.register_buffer("primes", torch.tensor(self._prime_values, dtype=torch.long), persistent=False)
        self.register_buffer("offsets", torch.tensor(self._offset_values, dtype=torch.long), persistent=False)
        self.register_buffer("multipliers", torch.tensor(self._multiplier_values, dtype=torch.long), persistent=False)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Restore derived integer buffers after meta-device materialization."""
        for name, values in (
            ("token_map", self._token_map_values),
            ("primes", self._prime_values),
            ("offsets", self._offset_values),
            ("multipliers", self._multiplier_values),
        ):
            buffer = self.get_buffer(name)
            buffer.copy_(torch.tensor(values, dtype=torch.long, device=buffer.device))

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Hash sequences without crossing document, image, or padding boundaries.

        Args:
            input_ids: Integer tensor of shape [batch, sequence] containing raw
                tokenizer IDs for complete sequences.
            sequence_ids: Optional integer document IDs [batch, sequence]; zero marks padding.
            token_mask: Optional boolean tensor of shape [batch, sequence].
                False marks image or padding tokens and blocks all lookback
                through those positions. The caller also masks their residual gate.

        Returns:
            Integer tensor of shape [batch, sequence, engram_layers, hash_heads],
            where hash_heads orders n-gram sizes from two through max_ngram_size,
            with n_heads separate prime buckets for each size.
        """
        if input_ids.ndim != 2 or input_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram input_ids must be an int32/int64 tensor of shape [batch, sequence]")
        if input_ids.numel() and bool(((input_ids < 0) | (input_ids >= self.token_map.numel())).any()):
            raise ValueError("Engram input_ids contains a token ID outside the tokenizer vocabulary")
        batch, sequence = input_ids.shape
        compressed = self.token_map[input_ids.long()]
        if token_mask is not None:
            if token_mask.shape != input_ids.shape or token_mask.dtype != torch.bool:
                raise ValueError("Engram token_mask must be bool with the same [batch, sequence] shape as input_ids")
            compressed = torch.where(token_mask, compressed, -1)
        positions = torch.arange(sequence, device=input_ids.device).expand(batch, sequence)
        if sequence_ids is not None and sequence_ids.shape != input_ids.shape:
            raise ValueError("Engram sequence_ids must have shape [batch, sequence]")
        blocked = torch.zeros_like(positions, dtype=torch.bool)
        tokens = []
        for shift in range(self.max_ngram_size):
            source_positions = positions - shift
            source = compressed.gather(1, source_positions.clamp_min(0))
            blocked = blocked | (source_positions < 0) | (source == -1)
            if sequence_ids is not None:
                source_ids = sequence_ids.gather(1, source_positions.clamp_min(0))
                blocked = blocked | (source_ids != sequence_ids) | (sequence_ids == 0)
            tokens.append(torch.where(blocked, self.pad_id, source))
        products = torch.stack(tokens, dim=-1).unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for shift in range(1, self.max_ngram_size):
            rolling = torch.bitwise_xor(rolling, products[..., shift])
            head_start = (shift - 1) * self.n_heads
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, head_start : head_start + self.n_heads])
        return torch.cat(hashes, dim=-1) + self.offsets


class DeepseekV41Engram(nn.Module):
    """Read a row-owner-sharded Engram table and update all HC residual streams.

    Args:
        config: Text configuration containing logical table sizes and HC width.
        layer_idx: Zero-based decoder layer ID, present in engram_layer_ids.
        backend: Linear backend for the fused key/value projection.
        process_group: Runtime row-owner group. None creates the complete table
            and is appropriate only when the configuration fits on one device.
    """

    def __init__(
        self,
        config: DeepseekV41TextConfig,
        layer_idx: int,
        backend: BackendConfig,
        *,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.layer_hash_index = tuple(config.engram_layer_ids).index(layer_idx)
        self.num_embeddings = config.engram_num_embeddings[self.layer_hash_index]
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.hash_heads = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.eps = config.rms_norm_eps
        self.initializer_range = config.initializer_range
        owner_size = 1 if process_group is None else dist.get_world_size(process_group)
        padded_rows = ((self.num_embeddings + owner_size - 1) // owner_size) * owner_size
        dtype = dtype_from_str(config.dtype, torch.bfloat16)
        self.embed = Qwen3_8_FlashNextEngramTableConfig(
            num_embeddings=padded_rows,
            embedding_dim=config.engram_head_dim,
            initializer_range=config.initializer_range,
        ).build(process_group=process_group, dtype=dtype)
        self.wkv = initialize_linear_module(
            backend.linear,
            self.hash_heads * config.engram_head_dim,
            self.hidden_size * (self.hc_mult + 1),
            bias=False,
            dtype=dtype,
        )
        self.q_weight = nn.Parameter(torch.ones(self.hc_mult, self.hidden_size, dtype=dtype))
        self.k_weight = nn.Parameter(torch.ones(self.hc_mult, self.hidden_size, dtype=dtype))
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize the table, projection, and learned branch normalization weights."""
        self.embed.reset_parameters()
        # Physical owner padding is absent from the released checkpoint. Keep
        # it zero so fresh initialization and strict checkpoint resume agree.
        local_weight = self.embed.weight.to_local() if isinstance(self.embed.weight, DTensor) else self.embed.weight
        valid_rows = max(0, min(local_weight.shape[0], self.num_embeddings - self.embed.global_row_start))
        local_weight[valid_rows:].zero_()
        nn.init.normal_(self.wkv.weight, mean=0.0, std=self.initializer_range)
        nn.init.ones_(self.q_weight)
        nn.init.ones_(self.k_weight)
        self.embed.mark_sharding_contract()

    def forward(
        self,
        hidden_states: torch.Tensor,
        hash_ids: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inject the normalized, signed-square-root-gated memory residual.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hc_mult, hidden].
            hash_ids: Integer tensor of shape [batch, sequence, hash_heads]
                containing logical table rows for this Engram layer. Under CP
                both inputs contain only this rank's local sequence positions.
            token_mask: Optional bool tensor of shape [batch, sequence], with
                False for image/padding positions that must remain unchanged.

        Returns:
            Tensor of shape [batch, sequence, hc_mult, hidden] in the input
            dtype. The output neither aliases nor mutates hidden_states.
        """
        if hidden_states.ndim != 4 or hidden_states.shape[-2:] != (self.hc_mult, self.hidden_size):
            raise ValueError("Engram hidden_states must have shape [batch, sequence, hc_mult, hidden_size]")
        if hash_ids.shape != (*hidden_states.shape[:2], self.hash_heads):
            raise ValueError("Engram hash_ids must have shape [batch, sequence, hash_heads] matching hidden_states")
        valid = hash_ids.dtype in (torch.int32, torch.int64)
        if valid and hash_ids.numel():
            valid = bool(((hash_ids >= 0) & (hash_ids < self.num_embeddings)).all())
        validity = torch.tensor(int(valid), device=hash_ids.device, dtype=torch.int32)
        if self.embed.process_group is not None:
            dist.all_reduce(validity, op=dist.ReduceOp.MIN, group=self.embed.process_group)
        if not bool(validity):
            raise ValueError(
                f"Engram hash_ids must be integer logical row IDs in [0, {self.num_embeddings}) on every rank"
            )
        if token_mask is not None and (token_mask.shape != hidden_states.shape[:2] or token_mask.dtype != torch.bool):
            raise ValueError("Engram token_mask must be bool with shape [batch, sequence]")
        embeddings = self.embed(hash_ids).flatten(-2).to(hidden_states.dtype)
        key, value = self.wkv(embeddings).split((self.hc_mult * self.hidden_size, self.hidden_size), dim=-1)
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        hidden = hidden_states.float()
        weights = self.q_weight.float() * self.k_weight.float()
        rstd = torch.rsqrt(hidden.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hidden * weights * key).sum(-1) * rstd * self.hidden_size**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        if token_mask is not None:
            gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
        return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(hidden_states.dtype)
