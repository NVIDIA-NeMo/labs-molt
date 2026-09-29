# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""AutoModel-owned configuration for GLM-5.3-Flash.

The released checkpoint requires Transformers 5.16, while AutoModel's current
runtime baseline predates the upstream ``glm5_next`` config.  These classes keep
the checkpoint field protocol stable and allow ``AutoConfig`` to resolve the
model without remote code or a dependency bump.
"""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig


def _json_safe_value(value: Any) -> Any:
    """Return a JSON-serializable representation of a config value."""
    if value.__class__.__module__ == "torch" and value.__class__.__name__ == "dtype":
        return str(value).removeprefix("torch.")
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    return value


class Glm5NextTextConfig(PretrainedConfig):
    """Configuration for the GLM-5.3 hybrid KDA/KPool-DSA text backbone."""

    model_type = "glm5_next_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 4096,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 45,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        n_shared_experts: int = 1,
        n_routed_experts: int = 288,
        routed_scaling_factor: float = 2.5,
        kv_lora_rank: int = 512,
        q_lora_rank: int = 1536,
        qk_rope_head_dim: int = 0,
        qk_nope_head_dim: int = 256,
        v_head_dim: int = 256,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        mlp_layer_types: list[str] | None = None,
        layer_types: list[str] | None = None,
        indexer_types: list[str] | None = None,
        index_topk_pattern: str | list[str] | None = None,
        index_topk_freq: int = 1,
        index_skip_topk_offset: int = 2,
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        index_kpool: int = 16,
        index_kpool_always_select_tail: bool = True,
        hidden_act: str = "silu",
        swiglu_limit: float = 10.0,
        linear_head_dim: int = 128,
        linear_num_heads: int = 64,
        linear_conv_kernel_dim: int = 4,
        linear_lower_bound: float | None = -5.0,
        linear_attn_config: dict[str, Any] | None = None,
        hc_mult: int = 4,
        hc_eps: float = 1e-6,
        hc_sinkhorn_iters: int = 20,
        max_position_embeddings: int = 1048576,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = False,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        output_router_logits: bool = False,
        router_aux_loss_coef: float = 0.001,
        num_nextn_predict_layers: int = 1,
        pad_token_id: int | None = 154820,
        bos_token_id: int | None = None,
        eos_token_id: int | list[int] | None = None,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        # Transformers aliases ``head_dim`` to the RoPE-only width for DSA.
        self.head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        if mlp_layer_types is None:
            dense = min(3, num_hidden_layers)
            mlp_layer_types = ["dense"] * dense + ["sparse"] * (num_hidden_layers - dense)
        if len(mlp_layer_types) != num_hidden_layers:
            raise ValueError("mlp_layer_types must have one entry per decoder layer")
        self.mlp_layer_types = list(mlp_layer_types)

        if layer_types is None:
            layer_types = [
                "deepseek_sparse_attention" if layer_idx % 4 == 3 else "linear_attention"
                for layer_idx in range(num_hidden_layers)
            ]
        layer_types = [
            "deepseek_sparse_attention" if layer_type == "full_attention" else layer_type for layer_type in layer_types
        ]
        if len(layer_types) != num_hidden_layers:
            raise ValueError("layer_types must have one entry per decoder layer")
        unknown_layer_types = set(layer_types) - {"linear_attention", "deepseek_sparse_attention"}
        if unknown_layer_types:
            raise ValueError(f"Unsupported GLM-5.3 attention layer types: {sorted(unknown_layer_types)}")
        self.layer_types = list(layer_types)

        if indexer_types is None:
            if index_topk_pattern is not None:
                indexer_types = (
                    [{"F": "full", "S": "shared"}[char] for char in index_topk_pattern]
                    if isinstance(index_topk_pattern, str)
                    else list(index_topk_pattern)
                )
            else:
                freq = max(int(index_topk_freq), 1)
                indexer_types = [
                    "full" if max(layer_idx - index_skip_topk_offset + 1, 0) % freq == 0 else "shared"
                    for layer_idx in range(num_hidden_layers)
                ]
        if len(indexer_types) != num_hidden_layers:
            raise ValueError("indexer_types must have one entry per decoder layer")
        self.indexer_types = list(indexer_types)
        self.index_topk_pattern = index_topk_pattern
        self.index_topk_freq = index_topk_freq
        self.index_skip_topk_offset = index_skip_topk_offset
        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_kpool = index_kpool
        self.index_kpool_always_select_tail = index_kpool_always_select_tail

        if linear_attn_config is not None:
            linear_head_dim = linear_attn_config.get("head_dim", linear_head_dim)
            linear_num_heads = linear_attn_config.get("num_heads", linear_num_heads)
            linear_conv_kernel_dim = linear_attn_config.get("short_conv_kernel_size", linear_conv_kernel_dim)
            linear_lower_bound = linear_attn_config.get("gate_lower_bound", linear_lower_bound)
            if linear_attn_config.get("safe_gate", True) and linear_lower_bound is None:
                linear_lower_bound = -5.0
        self.linear_head_dim = linear_head_dim
        self.linear_num_heads = linear_num_heads
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_lower_bound = linear_lower_bound
        self.linear_attn_config = {
            "head_dim": linear_head_dim,
            "num_heads": linear_num_heads,
            "short_conv_kernel_size": linear_conv_kernel_dim,
            "gate_lower_bound": linear_lower_bound,
            "safe_gate": linear_lower_bound is not None,
        }

        self.hc_mult = hc_mult
        self.hc_eps = hc_eps
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.num_nextn_predict_layers = num_nextn_predict_layers

        if num_attention_heads != num_key_value_heads:
            raise ValueError("GLM-5.3 DSA requires num_attention_heads == num_key_value_heads")
        if q_lora_rank is None:
            raise ValueError("GLM-5.3 DSA requires q_lora_rank")
        if qk_rope_head_dim != 0:
            raise ValueError("GLM-5.3 DSA is NoPE and requires qk_rope_head_dim=0")
        if index_kpool < 1 or index_topk % index_kpool:
            raise ValueError("index_kpool must be positive and divide index_topk")

        # ``head_dim`` is independently emitted by the checkpoint and must not
        # overwrite the NoPE width above through a future attribute alias.
        kwargs.pop("head_dim", None)
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def num_local_experts(self) -> int:
        """Alias used by Hugging Face expert implementations."""
        return self.n_routed_experts

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_value(super().to_dict())


class Glm5NextVisionConfig(PretrainedConfig):
    """Configuration for the GLM-5.3 image/video encoder."""

    model_type = "glm5_next_vision"

    def __init__(
        self,
        depth: int = 24,
        hidden_size: int = 1024,
        hidden_act: str = "silu",
        attention_bias: bool = True,
        attention_dropout: float = 0.0,
        num_heads: int = 16,
        in_channels: int = 3,
        image_size: int = 448,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 2,
        out_hidden_size: int = 4096,
        intermediate_size: int = 4096,
        projection_intermediate_size: int = 10240,
        initializer_range: float = 0.02,
        swiglu_limit: float = 10.0,
        **kwargs: Any,
    ) -> None:
        self.depth = depth
        self.num_hidden_layers = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.num_heads = num_heads
        self.num_attention_heads = num_heads
        self.in_channels = in_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.rms_norm_eps = rms_norm_eps
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.intermediate_size = intermediate_size
        self.projection_intermediate_size = projection_intermediate_size
        self.initializer_range = initializer_range
        self.swiglu_limit = swiglu_limit
        super().__init__(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_value(super().to_dict())


class Glm5NextConfig(PretrainedConfig):
    """Top-level GLM-5.3 vision-language configuration."""

    model_type = "glm5_next"
    sub_configs = {"text_config": Glm5NextTextConfig, "vision_config": Glm5NextVisionConfig}
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config: dict[str, Any] | Glm5NextTextConfig | None = None,
        vision_config: dict[str, Any] | Glm5NextVisionConfig | None = None,
        image_token_id: int = 154854,
        video_token_id: int = 154855,
        image_start_token_id: int = 154830,
        image_end_token_id: int = 154831,
        video_start_token_id: int = 154832,
        video_end_token_id: int = 154833,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        if text_config is None:
            text_config = Glm5NextTextConfig()
        elif isinstance(text_config, dict):
            text_config = Glm5NextTextConfig(**text_config)
        if vision_config is None:
            vision_config = Glm5NextVisionConfig()
        elif isinstance(vision_config, dict):
            vision_config = Glm5NextVisionConfig(**vision_config)
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.image_start_token_id = image_start_token_id
        self.image_end_token_id = image_end_token_id
        self.video_start_token_id = video_start_token_id
        self.video_end_token_id = video_end_token_id
        self.hidden_size = text_config.hidden_size
        self.vocab_size = text_config.vocab_size
        self.max_position_embeddings = text_config.max_position_embeddings
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

    def get_text_config(self, decoder: bool = False) -> Glm5NextTextConfig:
        """Return the decoder config using the Transformers multimodal protocol."""
        del decoder
        return self.text_config

    def to_dict(self) -> dict[str, Any]:
        return _json_safe_value(super().to_dict())
