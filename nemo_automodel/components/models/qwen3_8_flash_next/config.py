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

"""Checkpoint-compatible configuration classes for Qwen3.8-Flash-Next."""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class Qwen3_8_FlashNextTextConfig(PretrainedConfig):
    """Configuration for the Qwen3.8-Flash-Next HyperConnection, PLE, hybrid-MoE text backbone."""

    model_type = "qwen3_8_flash_next_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 248320,
        hidden_size: int = 2560,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 48,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 2,
        head_dim: int = 256,
        hidden_act: str = "silu",
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        max_position_embeddings: int = 262144,
        tie_word_embeddings: bool = False,
        dtype: str = "bfloat16",
        rope_theta: float | None = None,
        rope_scaling: dict[str, Any] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        partial_rotary_factor: float | None = None,
        full_attention_interval: int = 4,
        layer_types: list[str] | None = None,
        # GatedDeltaNet.
        output_gate_type: str = "sigmoid",
        linear_conv_kernel_dim: int = 4,
        linear_key_head_dim: int = 128,
        linear_value_head_dim: int = 128,
        linear_num_key_heads: int = 16,
        linear_num_value_heads: int = 48,
        mamba_ssm_dtype: str = "float32",
        # MoE.
        decoder_sparse_step: int = 1,
        moe_intermediate_size: int = 640,
        shared_expert_intermediate_size: int = 640,
        num_experts: int = 512,
        num_experts_per_tok: int = 10,
        norm_topk_prob: bool = True,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        mlp_only_layers: list[int] | None = None,
        # HyperConnections and PLE.
        hc_count: int = 4,
        hc_lowrank: int = 320,
        ple_layer_ids: list[int] | None = None,
        ple_embed_dim: int | None = None,
        ple_conv_kernel_size: int = 4,
        ngram_size: int = 3,
        heads_per_ngram: int = 8,
        ngram_vocab_size_base: int = 20000000,
        make_ngram_vocab_size_divisible_by: int = 128,
        split_ngram_parts: int = 128,
        # QSA indexer.
        indexer_budget: int = 2048,
        indexer_compress_ratio: int = 4,
        indexer_head_dim: int = 128,
        indexer_kv_heads: int = 1,
        indexer_n_heads: int = 4,
        qsa_indexer_query_chunk_size: int | None = None,
        # Multi-token prediction.
        mtp: dict[str, Any] | None = None,
        mtp_num_hidden_layers: int = 1,
        mtp_use_dedicated_embeddings: bool = False,
        pad_token_id: int | None = None,
        bos_token_id: int = 248044,
        eos_token_id: int = 248044,
        **kwargs: Any,
    ) -> None:
        if hc_count <= 1:
            raise ValueError(f"Qwen3.8-Flash-Next requires hc_count > 1, got {hc_count}.")

        if rope_parameters is not None:
            rope_parameters = dict(rope_parameters)
            if rope_scaling is None:
                rope_scaling = dict(rope_parameters)
            if rope_theta is None and "rope_theta" in rope_parameters:
                rope_theta = float(rope_parameters["rope_theta"])
            if partial_rotary_factor is None and "partial_rotary_factor" in rope_parameters:
                partial_rotary_factor = float(rope_parameters["partial_rotary_factor"])
        if rope_theta is None:
            rope_theta = 10000.0
        if partial_rotary_factor is None:
            partial_rotary_factor = 0.25
        if rope_scaling is None:
            rope_scaling = {}
        else:
            rope_scaling = dict(rope_scaling)
        if rope_parameters is None:
            rope_parameters = dict(rope_scaling)

        if layer_types is not None:
            layer_types = list(layer_types)

        if ple_layer_ids is None:
            ple_layer_ids = []
        else:
            ple_layer_ids = list(ple_layer_ids)

        if mlp_only_layers is None:
            mlp_only_layers = []
        else:
            mlp_only_layers = list(mlp_only_layers)

        if mtp is None:
            mtp = {
                "hybrid": True,
                "layer_types": ["full_attention"] * mtp_num_hidden_layers,
                "mtp_use_hidden_state_from_layer": None,
                "num_hidden_layers": mtp_num_hidden_layers,
                "rope_theta": rope_theta,
            }
        else:
            mtp = dict(mtp)

        # Match the SGLang reference ordering: initialize the generic config
        # before assigning Qwen3.8-Flash-Next's custom RoPE and hybrid-layer fields.
        # This also avoids stale Transformers releases trying to validate
        # Qwen3.8-Flash-Next's mRoPE-only keys as generic default-RoPE keys.
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            dtype=dtype,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.rope_parameters = rope_parameters
        self.partial_rotary_factor = partial_rotary_factor
        self.full_attention_interval = full_attention_interval
        self.layer_types = layer_types

        self.output_gate_type = output_gate_type
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.mamba_ssm_dtype = mamba_ssm_dtype

        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = mlp_only_layers

        self.hc_count = hc_count
        self.hc_mult = hc_count
        self.hc_lowrank = hc_lowrank
        self.ple_layer_ids = ple_layer_ids
        self.ple_embed_dim = ple_embed_dim or hidden_size
        self.ple_conv_kernel_size = ple_conv_kernel_size
        self.ngram_size = ngram_size
        self.heads_per_ngram = heads_per_ngram
        self.ngram_vocab_size_base = ngram_vocab_size_base
        self.make_ngram_vocab_size_divisible_by = make_ngram_vocab_size_divisible_by
        self.split_ngram_parts = split_ngram_parts

        self.indexer_budget = indexer_budget
        self.indexer_compress_ratio = indexer_compress_ratio
        self.indexer_head_dim = indexer_head_dim
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_n_heads = indexer_n_heads
        # Query rows the indexer scores per chunk. None sizes chunks by a 256 MiB fp32
        # score budget (whole row for 4k-16k sequences); an int forces that many rows.
        self.qsa_indexer_query_chunk_size = qsa_indexer_query_chunk_size

        self.mtp = mtp
        self.mtp_num_hidden_layers = mtp_num_hidden_layers
        self.mtp_use_dedicated_embeddings = mtp_use_dedicated_embeddings

    @property
    def layers_block_type(self) -> list[str]:
        """Return SGLang-compatible mixer names for each decoder layer."""
        if self.layer_types is not None:
            return ["attention" if layer_type == "full_attention" else layer_type for layer_type in self.layer_types]
        return [
            "attention" if (layer_idx + 1) % self.full_attention_interval == 0 else "linear_attention"
            for layer_idx in range(self.num_hidden_layers)
        ]

    @property
    def linear_layer_ids(self) -> list[int]:
        """Return zero-based GatedDeltaNet decoder-layer indices."""
        return [
            layer_idx for layer_idx, layer_type in enumerate(self.layers_block_type) if layer_type == "linear_attention"
        ]

    @property
    def full_attention_layer_ids(self) -> list[int]:
        """Return zero-based QSA full-attention decoder-layer indices."""
        return [layer_idx for layer_idx, layer_type in enumerate(self.layers_block_type) if layer_type == "attention"]

    @property
    def short_conv_layer_ids(self) -> list[int]:
        """Return zero-based decoder-layer indices that contain PLE short convolution."""
        return sorted({int(layer_id) - 1 for layer_id in self.ple_layer_ids})

    @property
    def short_conv_state_shape(self) -> tuple[int, int] | None:
        """Return PLE convolution cache shape as ``[channels, history]``."""
        if not self.short_conv_layer_ids:
            return None
        history = (self.ple_conv_kernel_size - 1) * self.ngram_size
        channels = self.hidden_size * self.hc_count
        return channels, history

    @property
    def ngram_context_len(self) -> int:
        """Return the number of preceding token IDs required by PLE hashing."""
        if not self.ple_layer_ids:
            return 0
        return max(int(self.ngram_size) - 1, 0)


class Qwen3_8_FlashNextVisionConfig(PretrainedConfig):
    """Configuration for the Qwen3.8-Flash-Next Qwen3-VL-style vision tower."""

    model_type = "qwen3_8_flash_next"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth: int = 27,
        hidden_act: str = "gelu_pytorch_tanh",
        hidden_size: int = 1152,
        in_channels: int = 3,
        initializer_range: float = 0.02,
        intermediate_size: int = 4304,
        num_heads: int = 16,
        num_position_embeddings: int = 2304,
        out_hidden_size: int = 2560,
        patch_size: int = 16,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        deepstack_visual_indexes: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.depth = depth
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_position_embeddings = num_position_embeddings
        self.out_hidden_size = out_hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.deepstack_visual_indexes = [] if deepstack_visual_indexes is None else list(deepstack_visual_indexes)


class Qwen3_8_FlashNextConfig(PretrainedConfig):
    """Top-level configuration for Qwen3.8-Flash-Next conditional generation checkpoints."""

    model_type = "qwen3_8_flash_next"
    architectures = ["Qwen3_8_FlashNextForConditionalGeneration"]
    sub_configs = {"text_config": Qwen3_8_FlashNextTextConfig, "vision_config": Qwen3_8_FlashNextVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: dict[str, Any] | Qwen3_8_FlashNextTextConfig | None = None,
        vision_config: dict[str, Any] | Qwen3_8_FlashNextVisionConfig | None = None,
        image_token_id: int = 248056,
        video_token_id: int = 248057,
        vision_start_token_id: int = 248053,
        vision_end_token_id: int = 248054,
        language_model_only: bool = False,
        tie_word_embeddings: bool = False,
        rope_parameters: dict[str, Any] | None = None,
        architectures: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        # The nested text config is authoritative. Older exports may carry a
        # stale duplicate of this field at the top level.
        if text_config is not None:
            kwargs.pop("split_ngram_parts", None)

        # Preserve SGLang's compatibility path for older text-only checkpoints
        # whose text attributes lived at the top level.
        text_kwargs = (
            dict(kwargs) if text_config is None and "hidden_size" in kwargs and "num_hidden_layers" in kwargs else {}
        )

        if isinstance(vision_config, dict):
            vision_config = Qwen3_8_FlashNextVisionConfig(**vision_config)
        elif vision_config is None:
            vision_config = Qwen3_8_FlashNextVisionConfig()

        if isinstance(text_config, dict):
            text_config = Qwen3_8_FlashNextTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3_8_FlashNextTextConfig(**text_kwargs)

        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        self.language_model_only = language_model_only
        self.rope_parameters = dict(rope_parameters or getattr(text_config, "rope_parameters", {}))
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            architectures=architectures or ["Qwen3_8_FlashNextForConditionalGeneration"],
            **kwargs,
        )


class Qwen3_8_FlashNextLegacyTextConfig(Qwen3_8_FlashNextTextConfig):
    """Text config alias for checkpoint dumps that predate the model rename."""

    model_type = "qwen4_exp_text"

    def __init__(self, **kwargs: Any) -> None:
        """Pass through so transformers keeps the parent's field handling.

        ``PreTrainedConfig.__init_subclass__`` dataclass-wraps subclasses and
        replaces a missing ``__init__`` with a generated one that skips the
        parent's sub-config materialization.
        """
        super().__init__(**kwargs)


class Qwen3_8_FlashNextLegacyConfig(Qwen3_8_FlashNextConfig):
    """Config alias for checkpoint dumps that predate the model rename.

    Released checkpoints store ``model_type: qwen4_exp`` and architecture
    ``Qwen4ExpForConditionalGeneration``. The dumps are immutable, so the
    legacy identifiers resolve to the renamed classes here instead.
    """

    model_type = "qwen4_exp"

    def __init__(self, **kwargs: Any) -> None:
        """Pass through so transformers keeps the parent's sub-config conversion."""
        super().__init__(**kwargs)
