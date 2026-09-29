# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Hugging Face checkpoint configuration for DeepSeek-V4.1-Flash."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from transformers import AutoTokenizer, PretrainedConfig, PreTrainedTokenizerFast

if TYPE_CHECKING:

    pass


class DeepseekV41TextConfig(PretrainedConfig):
    """Declarative configuration of the CSA2, single-pass mHC and Engram backbone.

    Defaults match the released Flash text configuration. Reducing
    ``num_hidden_layers`` retains the complete source-layer and compression
    schedules: a pretrained prefix must keep its original sharing and hashing
    identities. Explicit smaller schedules support independent tiny models.
    Unknown checkpoint metadata is retained by the Hugging Face base class.
    """

    model_type = "deepseek_v41_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 5120,
        moe_intermediate_size: int = 2304,
        num_hidden_layers: int = 40,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        q_lora_rank: int = 1280,
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        hidden_act: str = "silu",
        swiglu_limit: float = 10.0,
        rms_norm_eps: float = 1e-20,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_scaling: dict[str, Any] | None = None,
        n_routed_experts: int = 384,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        scoring_func: str = "sqrtsoftplus",
        topk_method: str = "noaux_tc",
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 1.5,
        sliding_window: int = 128,
        compress_ratios: list[int] | None = None,
        compress_rope_theta: float = 160000.0,
        kv_source_layer_ids: list[int] | None = None,
        index_source_layer_ids: list[int] | None = None,
        index_n_heads: int = 32,
        index_head_dim: int = 128,
        index_topk: int = 512,
        candidate_source_layer_id: int = 20,
        candidate_topk_blocks: int = 2048,
        candidate_block_size: int = 8,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        hc_eps: float = 1e-6,
        engram_layer_ids: list[int] | None = None,
        engram_num_embeddings: list[int] | None = None,
        engram_max_ngram_size: int = 4,
        engram_vocab_size: int = 16000000,
        engram_n_heads: int = 8,
        engram_head_dim: int = 256,
        engram_pad_token_id: int = 2,
        engram_compressed_vocab_size: int = 99092,
        num_nextn_predict_layers: int = 3,
        dspark_block_size: int = 5,
        dspark_noise_token_id: int = 128799,
        dspark_target_layer_ids: list[int] | None = None,
        dspark_markov_rank: int = 256,
        dspark_n_routed_experts: int = 128,
        dspark_num_experts_per_tok: int = 3,
        dtype: str = "bfloat16",
        pad_token_id: int | None = 2,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = (
            {
                "rope_type": "yarn",
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "original_max_position_embeddings": 65536,
            }
            if rope_scaling is None
            else dict(rope_scaling)
        )
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        self.sliding_window = sliding_window
        # The released schedule includes three SWA-only DSpark layers.
        self.compress_ratios = (
            [0, 0] + [2] * 18 + [1] * 20 + [0] * 3 if compress_ratios is None else list(compress_ratios)
        )
        self.compress_rope_theta = compress_rope_theta
        self.kv_source_layer_ids = [2, 8, 14, 20] if kv_source_layer_ids is None else list(kv_source_layer_ids)
        self.index_source_layer_ids = (
            [2, 8, 14, 20, 24, 28, 32, 36] if index_source_layer_ids is None else list(index_source_layer_ids)
        )
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
        self.candidate_source_layer_id = candidate_source_layer_id
        self.candidate_topk_blocks = candidate_topk_blocks
        self.candidate_block_size = candidate_block_size
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.engram_layer_ids = [1, 14] if engram_layer_ids is None else list(engram_layer_ids)
        self.engram_num_embeddings = (
            ([384006168, 384016682] if self.engram_layer_ids else [])
            if engram_num_embeddings is None
            else list(engram_num_embeddings)
        )
        self.engram_max_ngram_size = engram_max_ngram_size
        self.engram_vocab_size = engram_vocab_size
        self.engram_n_heads = engram_n_heads
        self.engram_head_dim = engram_head_dim
        self.engram_pad_token_id = engram_pad_token_id
        self.engram_compressed_vocab_size = engram_compressed_vocab_size
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.dspark_block_size = dspark_block_size
        self.dspark_noise_token_id = dspark_noise_token_id
        self.dspark_target_layer_ids = (
            [37, 38, 39] if dspark_target_layer_ids is None else list(dspark_target_layer_ids)
        )
        self.dspark_markov_rank = dspark_markov_rank
        self.dspark_n_routed_experts = dspark_n_routed_experts
        self.dspark_num_experts_per_tok = dspark_num_experts_per_tok
        self._validate_dimensions()
        self._validate_layer_schedule()
        # Serialized Transformers configs can contain rope_parameters in
        # addition to rope_scaling. Its YaRN validator reads the model's
        # dimensions during base initialization, so establish them first.
        super().__init__(
            dtype=kwargs.pop("torch_dtype", dtype),
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def _validate_dimensions(self) -> None:
        for name, value in (
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("moe_intermediate_size", self.moe_intermediate_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("head_dim", self.head_dim),
            ("q_lora_rank", self.q_lora_rank),
            ("o_lora_rank", self.o_lora_rank),
            ("o_groups", self.o_groups),
            ("n_routed_experts", self.n_routed_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
            ("sliding_window", self.sliding_window),
            ("index_n_heads", self.index_n_heads),
            ("index_head_dim", self.index_head_dim),
            ("index_topk", self.index_topk),
            ("hc_mult", self.hc_mult),
            ("hc_sinkhorn_iters", self.hc_sinkhorn_iters),
            ("max_position_embeddings", self.max_position_embeddings),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.num_key_value_heads != 1:
            raise ValueError("DeepSeek-V4.1 requires num_key_value_heads=1 for shared latent KV")
        if self.num_attention_heads % self.o_groups:
            raise ValueError("num_attention_heads must be divisible by o_groups")
        if (
            type(self.qk_rope_head_dim) is not int
            or self.qk_rope_head_dim <= 0
            or self.qk_rope_head_dim % 2
            or self.qk_rope_head_dim > min(self.head_dim, self.index_head_dim)
        ):
            raise ValueError("qk_rope_head_dim must be positive, even, and no larger than head_dim or index_head_dim")
        if self.num_experts_per_tok > self.n_routed_experts:
            raise ValueError("num_experts_per_tok must not exceed n_routed_experts")
        if self.n_shared_experts != 1:
            raise ValueError("DeepSeek-V4.1 requires n_shared_experts=1")
        for name, value in (
            ("rms_norm_eps", self.rms_norm_eps),
            ("hc_eps", self.hc_eps),
            ("rope_theta", self.rope_theta),
            ("compress_rope_theta", self.compress_rope_theta),
            ("routed_scaling_factor", self.routed_scaling_factor),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value!r}")
        if not 0 <= self.attention_dropout < 1:
            raise ValueError("attention_dropout must lie in [0, 1)")
        if self.num_nextn_predict_layers < 0 or self.dspark_block_size < 0:
            raise ValueError("num_nextn_predict_layers and dspark_block_size must be non-negative")
        if self.engram_layer_ids:
            for name, value in (
                ("engram_n_heads", self.engram_n_heads),
                ("engram_head_dim", self.engram_head_dim),
                ("engram_vocab_size", self.engram_vocab_size),
                ("engram_compressed_vocab_size", self.engram_compressed_vocab_size),
            ):
                if type(value) is not int or value <= 0:
                    raise ValueError(f"{name} must be a positive integer, got {value!r}")
            if self.engram_max_ngram_size < 2:
                raise ValueError("engram_max_ngram_size must be at least 2 when Engram is enabled")
            if not 0 <= self.engram_pad_token_id < self.vocab_size:
                raise ValueError("engram_pad_token_id must be within the token vocabulary")

    def _validate_layer_schedule(self) -> None:
        if len(self.compress_ratios) < self.num_hidden_layers:
            raise ValueError("compress_ratios must cover every active backbone layer")
        if any(type(ratio) is not int or ratio < 0 for ratio in self.compress_ratios):
            raise ValueError("compress_ratios must contain non-negative integers")
        for name, layer_ids in (
            ("kv_source_layer_ids", self.kv_source_layer_ids),
            ("index_source_layer_ids", self.index_source_layer_ids),
            ("engram_layer_ids", self.engram_layer_ids),
        ):
            if any(type(layer_id) is not int or layer_id < 0 for layer_id in layer_ids):
                raise ValueError(f"{name} must contain non-negative integer layer IDs")
            if layer_ids != sorted(set(layer_ids)):
                raise ValueError(f"{name} must contain strictly increasing layer IDs")
            if any(layer_id >= len(self.compress_ratios) for layer_id in layer_ids):
                raise ValueError(f"{name} must refer to layers covered by compress_ratios")
        if not set(self.kv_source_layer_ids).issubset(self.index_source_layer_ids):
            raise ValueError("Every KV source must also occur in index_source_layer_ids")
        owner_ratio = None
        for layer_id, ratio in enumerate(self.compress_ratios[: self.num_hidden_layers]):
            is_source = layer_id in self.kv_source_layer_ids
            if ratio == 0:
                if is_source or layer_id in self.index_source_layer_ids:
                    raise ValueError(f"SWA-only layer {layer_id} cannot be a KV or index source")
                continue
            if is_source:
                owner_ratio = ratio
            if owner_ratio != ratio:
                raise ValueError(
                    f"Compressed layer {layer_id} needs a preceding KV source with the same compression ratio"
                )
        candidate = self.candidate_source_layer_id
        if type(candidate) is not int or candidate < -1:
            raise ValueError("candidate_source_layer_id must be -1 (disabled) or a non-negative layer ID")
        if candidate >= 0:
            if candidate not in self.kv_source_layer_ids:
                raise ValueError("candidate_source_layer_id must identify a Full-mode KV source")
            if self.candidate_block_size <= 0 or self.candidate_topk_blocks <= 0:
                raise ValueError("candidate_block_size and candidate_topk_blocks must be positive when enabled")
        if len(self.engram_layer_ids) != len(self.engram_num_embeddings):
            raise ValueError("engram_num_embeddings must contain one table row count per engram_layer_ids entry")
        if any(type(rows) is not int or rows <= 0 for rows in self.engram_num_embeddings):
            raise ValueError("engram_num_embeddings must contain positive integer row counts")


class DeepseekV41VisionConfig(PretrainedConfig):
    """Configuration of the released 2D-RoPE vision encoder and image sizing."""

    model_type = "deepseek_v41_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        num_hidden_layers: int = 32,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        intermediate_size: int = 2816,
        patch_size: int = 14,
        rope_theta: float = 10000.0,
        downsample_ratio: int = 3,
        max_image_tokens: int = 1024,
        min_pixels: int = 295936,
        max_wh_ratio: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.patch_size = patch_size
        self.rope_theta = rope_theta
        self.downsample_ratio = downsample_ratio
        self.max_image_tokens = max_image_tokens
        self.min_pixels = min_pixels
        self.max_wh_ratio = max_wh_ratio
        for name, value in (
            ("hidden_size", hidden_size),
            ("num_attention_heads", num_attention_heads),
            ("intermediate_size", intermediate_size),
            ("patch_size", patch_size),
            ("downsample_ratio", downsample_ratio),
            ("max_image_tokens", max_image_tokens),
            ("min_pixels", min_pixels),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"vision {name} must be a positive integer, got {value!r}")
        if type(num_hidden_layers) is not int or num_hidden_layers < 0:
            raise ValueError("vision num_hidden_layers must be a non-negative integer")
        if hidden_size % num_attention_heads or hidden_size // num_attention_heads % 4:
            raise ValueError("vision hidden_size must yield an integer head dimension divisible by 4 for 2D RoPE")
        if not math.isfinite(rope_theta) or rope_theta <= 0:
            raise ValueError("vision rope_theta must be finite and positive")
        if max_wh_ratio is not None and (not math.isfinite(max_wh_ratio) or max_wh_ratio < 1):
            raise ValueError("vision max_wh_ratio must be None or finite and at least 1")


class DeepseekV41Config(PretrainedConfig):
    """Nested checkpoint configuration for ``DeepseekV41ForCausalLM``.

    Hugging Face dictionaries are materialized at this boundary; model
    components receive the typed ``text_config`` and ``vision_config``.
    Quantization metadata, when present in a checkpoint, is preserved by the
    base class rather than imposed on checkpoint-free BF16 configurations.
    """

    model_type = "deepseek_v41"
    sub_configs = {"text_config": DeepseekV41TextConfig, "vision_config": DeepseekV41VisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: dict[str, Any] | DeepseekV41TextConfig | None = None,
        vision_config: dict[str, Any] | DeepseekV41VisionConfig | None = None,
        image_token_id: int = 129264,
        dtype: str = "bfloat16",
        pad_token_id: int | None = 2,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        tie_word_embeddings: bool = False,
        architectures: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(text_config, dict):
            text_config = DeepseekV41TextConfig(**text_config)
        elif text_config is None:
            text_config = DeepseekV41TextConfig()
        elif not isinstance(text_config, DeepseekV41TextConfig):
            raise TypeError("text_config must be a DeepseekV41TextConfig, dictionary, or None")
        if isinstance(vision_config, dict):
            vision_config = DeepseekV41VisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = DeepseekV41VisionConfig()
        elif not isinstance(vision_config, DeepseekV41VisionConfig):
            raise TypeError("vision_config must be a DeepseekV41VisionConfig, dictionary, or None")
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        super().__init__(
            dtype=kwargs.pop("torch_dtype", dtype),
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            architectures=["DeepseekV41ForCausalLM"] if architectures is None else list(architectures),
            **kwargs,
        )

    def build_tokenizer(self) -> PreTrainedTokenizerFast:
        """Load the checkpoint's fast tokenizer for deterministic Engram hashing.

        Returns:
            The original fast tokenizer from this configuration's checkpoint
            source, using its resolved commit when available.

        Raises:
            ValueError: The configuration has no checkpoint source. Callers
                constructing a tiny model can supply a tokenizer directly.
            TypeError: The checkpoint resolves to a slow tokenizer.
        """
        if not self._name_or_path:
            raise ValueError("Engram tokenizer construction requires a checkpoint source or an explicit tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(
            self._name_or_path,
            revision=self._commit_hash,
            trust_remote_code=False,
            use_fast=True,
        )
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise TypeError("DeepSeek-V4.1 Engram requires the checkpoint's original fast tokenizer")
        return tokenizer


