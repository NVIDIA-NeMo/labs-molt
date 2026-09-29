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

"""Inkling configuration classes for Automodel's registry-first config lookup."""

from __future__ import annotations

from typing import Any

from transformers.configuration_utils import PretrainedConfig


class InklingTextConfig(PretrainedConfig):
    """Configuration for the Inkling text backbone."""

    model_type = "inkling_text"
    base_config_key = "text_config"
    attribute_map = {
        "embedding_multiplier": "logits_mup_width_multiplier",
        "sliding_window": "sliding_window_size",
        "num_local_experts": "n_routed_experts",
        "sconv_kernel_size": "conv_kernel_size",
        "model_max_length": "max_position_embeddings",
    }

    def __init__(
        self,
        vocab_size: int = 201024,
        unpadded_vocab_size: int | None = None,
        hidden_size: int = 6144,
        num_hidden_layers: int = 66,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 8,
        head_dim: int = 128,
        swa_num_attention_heads: int = 64,
        swa_num_key_value_heads: int = 16,
        swa_head_dim: int = 128,
        sliding_window_size: int = 512,
        d_rel: int = 16,
        rel_extent: int = 1024,
        log_scaling_n_floor: int | None = None,
        log_scaling_alpha: float = 0.1,
        local_layer_ids: list[int] | None = None,
        layer_types: list[str] | None = None,
        max_position_embeddings: int = 131072,
        rms_norm_eps: float = 1e-6,
        conv_kernel_size: int = 4,
        mlp_layer_types: list[str] | None = None,
        intermediate_size: int = 24576,
        hidden_act: str = "silu",
        moe_intermediate_size: int | None = None,
        n_routed_experts: int = 256,
        num_experts_per_tok: int = 6,
        n_shared_experts: int = 2,
        shared_expert_sink: bool = True,
        route_scale: float = 8.0,
        logits_mup_width_multiplier: float = 24.0,
        rms_norm_eps_moe_gate: float = 1e-6,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        initializer_range: float = 0.02,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        num_mtp_layers: int | None = None,
        chain_hidden_post_norm: bool = False,
        mtp_hidden_states_first: bool = True,
        mtp_local_layer_ids: list[int] | None = None,
        dense_mlp_idx: int = 0,
        dense_intermediate_size: int | None = None,
        **kwargs: Any,
    ) -> None:
        if layer_types is None:
            local_layers = (
                set(local_layer_ids)
                if local_layer_ids is not None
                else {i for i in range(num_hidden_layers) if (i + 1) % 6}
            )
            layer_types = ["hybrid_sliding" if i in local_layers else "hybrid" for i in range(num_hidden_layers)]
        if mlp_layer_types is None:
            mlp_layer_types = ["dense" if i < dense_mlp_idx else "sparse" for i in range(num_hidden_layers)]
        if moe_intermediate_size is None:
            # Raw checkpoints use intermediate_size for experts and a separate dense width.
            moe_intermediate_size = intermediate_size if dense_intermediate_size is not None else 3072
        if dense_intermediate_size is not None:
            intermediate_size = dense_intermediate_size

        self.vocab_size = vocab_size
        self.unpadded_vocab_size = unpadded_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.swa_num_attention_heads = swa_num_attention_heads
        self.swa_num_key_value_heads = swa_num_key_value_heads
        self.swa_head_dim = swa_head_dim
        self.sliding_window_size = sliding_window_size
        self.d_rel = d_rel
        self.rel_extent = rel_extent
        self.log_scaling_n_floor = log_scaling_n_floor
        self.log_scaling_alpha = log_scaling_alpha
        self.local_layer_ids = local_layer_ids
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.conv_kernel_size = conv_kernel_size
        self.mlp_layer_types = mlp_layer_types
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.moe_intermediate_size = moe_intermediate_size
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.shared_expert_sink = shared_expert_sink
        self.route_scale = route_scale
        self.logits_mup_width_multiplier = logits_mup_width_multiplier
        self.rms_norm_eps_moe_gate = rms_norm_eps_moe_gate
        self.attention_dropout = attention_dropout
        self.use_cache = use_cache
        self.initializer_range = initializer_range
        self.num_mtp_layers = num_mtp_layers
        self.chain_hidden_post_norm = chain_hidden_post_norm
        self.mtp_hidden_states_first = mtp_hidden_states_first
        self.mtp_local_layer_ids = mtp_local_layer_ids
        self.number_of_conv_states = 4
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        self.layer_types = layer_types

    @property
    def mtp_layer_types(self) -> list[str] | None:
        if self.num_mtp_layers is None:
            return None
        if self.mtp_local_layer_ids is None:
            return ["hybrid"] * self.num_mtp_layers
        return ["hybrid_sliding" if i in self.mtp_local_layer_ids else "hybrid" for i in range(self.num_mtp_layers)]

    @property
    def mtp_mlp_layer_types(self) -> list[str] | None:
        if self.num_mtp_layers is None:
            return None
        return ["dense"] * self.num_mtp_layers


class InklingAudioConfig(PretrainedConfig):
    """Configuration for Inkling's audio tower."""

    model_type = "inkling_audio"
    base_config_key = "audio_config"
    attribute_map = {
        "num_codebooks": "n_mel_bins",
        "codebook_size": "mel_vocab_size",
        "hidden_size": "text_hidden_size",
        "text_hidden_size": "decoder_dmodel",
    }

    def __init__(
        self,
        n_mel_bins: int = 80,
        mel_vocab_size: int = 256,
        text_hidden_size: int = 6144,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        self.n_mel_bins = n_mel_bins
        self.mel_vocab_size = mel_vocab_size
        self.text_hidden_size = text_hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class InklingVisionConfig(PretrainedConfig):
    """Configuration for Inkling's vision tower."""

    model_type = "inkling_vision"
    base_config_key = "vision_config"
    attribute_map = {
        "num_hidden_layers": "n_layers",
        "num_channels": "n_channels",
        "text_hidden_size": "decoder_dmodel",
    }

    def __init__(
        self,
        text_hidden_size: int = 6144,
        patch_size: int = 40,
        temporal_patch_size: int = 2,
        num_channels: int = 3,
        hidden_size: int = 1024,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        **kwargs: Any,
    ) -> None:
        self.text_hidden_size = text_hidden_size
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        super().__init__(**kwargs)


class InklingConfig(PretrainedConfig):
    """Top-level multimodal Inkling configuration."""

    model_type = "inkling_mm_model"
    sub_configs = {
        "text_config": InklingTextConfig,
        "audio_config": InklingAudioConfig,
        "vision_config": InklingVisionConfig,
    }

    def __init__(
        self,
        text_config: InklingTextConfig | dict[str, Any] | None = None,
        audio_config: InklingAudioConfig | dict[str, Any] | None = None,
        vision_config: InklingVisionConfig | dict[str, Any] | None = None,
        image_token_id: int = 200054,
        audio_token_id: int = 200053,
        image_bos_token_id: int = 200005,
        audio_bos_token_id: int = 200020,
        mtp_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        mtp_config = mtp_config or {}
        if isinstance(text_config, dict):
            text_config = dict(text_config)
            text_config.setdefault("num_mtp_layers", mtp_config.get("num_nextn_predict_layers"))
            text_config.setdefault("chain_hidden_post_norm", mtp_config.get("chain_hidden_post_norm", False))
            text_config.setdefault("mtp_local_layer_ids", mtp_config.get("local_layer_ids"))

        self.text_config = self._coerce_sub_config(text_config, InklingTextConfig)
        self.audio_config = self._coerce_sub_config(audio_config, InklingAudioConfig)
        self.vision_config = self._coerce_sub_config(vision_config, InklingVisionConfig)
        self.vision_config.text_hidden_size = self.text_config.hidden_size
        self.audio_config.text_hidden_size = self.text_config.hidden_size
        self.image_token_id = 200054 if image_token_id is None else image_token_id
        self.audio_token_id = 200053 if audio_token_id is None else audio_token_id
        self.image_bos_token_id = 200005 if image_bos_token_id is None else image_bos_token_id
        self.audio_bos_token_id = 200020 if audio_bos_token_id is None else audio_bos_token_id
        self.mtp_config = mtp_config
        super().__init__(**kwargs)

    @staticmethod
    def _coerce_sub_config(config, config_cls):
        if isinstance(config, dict):
            return config_cls(**config)
        if config is None:
            return config_cls()
        if isinstance(config, config_cls):
            return config
        raise TypeError(f"Expected {config_cls.__name__}, dict, or None; got {type(config).__name__}")


__all__ = ["InklingConfig", "InklingTextConfig", "InklingAudioConfig", "InklingVisionConfig"]
