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

"""Configuration for the native AutoModel MuseGlimmer implementation."""

from __future__ import annotations

import math
from typing import Any

from transformers import PretrainedConfig


class MuseGlimmerTextConfig(PretrainedConfig):
    """Canonical nested configuration for the MuseGlimmer language backbone."""

    model_type = "muse_glimmer_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 202048,
        hidden_size: int = 6656,
        intermediate_size: int = 19968,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        hidden_activation: str = "silu",
        max_position_embeddings: int = 131072,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        post_norm_eps: float = 1e-8,
        use_cache: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        sliding_window: int = 2048,
        layer_types: list[str] | None = None,
        layer_rope_theta: list[float] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        qk_scale_factor: float = 3.87,
        output_multiplier: float = 0.19611613513818404,
        final_logit_softcapping: float | None = 20.0,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 200000,
        eos_token_id: int = 200001,
        pad_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        if layer_rope_theta is None:
            layer_rope_theta = [
                500000.0 if (num_hidden_layers - index - 1) % 4 else 0.0 for index in range(num_hidden_layers)
            ]
        if layer_types is None:
            layer_types = ["sliding_attention" if theta else "full_attention" for theta in layer_rope_theta]
        if rope_parameters is None:
            rope_parameters = {"rope_theta": 500000.0, "rope_type": "default"}

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_activation = hidden_activation
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.post_norm_eps = post_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.layer_types = layer_types
        self.layer_rope_theta = layer_rope_theta
        self.rope_parameters = rope_parameters
        self.qk_scale_factor = qk_scale_factor
        self.output_multiplier = output_multiplier
        self.final_logit_softcapping = final_logit_softcapping
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )


class MuseGlimmerVisionConfig(PretrainedConfig):
    """Canonical nested configuration for the MuseGlimmer vision tower."""

    model_type = "muse_glimmer_vision"

    def __init__(
        self,
        hidden_size: int = 1536,
        intermediate_size: int = 8960,
        num_hidden_layers: int = 50,
        num_attention_heads: int = 16,
        hidden_act: str = "gelu",
        patch_size: int = 14,
        patch_temporal: int = 2,
        merge_size: int = 2,
        pos_emb_height: int = 32,
        pos_emb_width: int = 32,
        max_position_embeddings: int = 1024,
        layer_norm_eps: float = 1e-5,
        layer_types: list[str] | None = None,
        rope_parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if layer_types is None:
            layer_types = [
                "full_attention" if (index + 1) % 4 == 0 or index == num_hidden_layers - 1 else "window_attention"
                for index in range(num_hidden_layers)
            ]
        if rope_parameters is None:
            rope_parameters = {"rope_theta": 10000.0, "rope_type": "default"}

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.patch_size = patch_size
        self.patch_temporal = patch_temporal
        self.merge_size = merge_size
        self.pos_emb_height = pos_emb_height
        self.pos_emb_width = pos_emb_width
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        if len(layer_types) != num_hidden_layers or any(
            layer_type not in {"window_attention", "full_attention"} for layer_type in layer_types
        ):
            raise ValueError(
                "MuseGlimmer vision layer_types must contain one 'window_attention' or 'full_attention' entry per layer."
            )
        self.rope_parameters = rope_parameters
        super().__init__(**kwargs)
        # Transformers 5.8's generic language-model validator does not yet know
        # the canonical vision-only ``window_attention`` value. Assign it after
        # base validation so serialization still preserves the checkpoint field.
        self.layer_types = layer_types


class MuseGlimmerConfig(PretrainedConfig):
    """Configuration accepting both legacy flat and canonical nested MuseGlimmer checkpoints."""

    model_type = "muse_glimmer"
    sub_configs = {"text_config": MuseGlimmerTextConfig, "vision_config": MuseGlimmerVisionConfig}

    def __init__(
        self,
        hidden_size: int = 6656,
        num_hidden_layers: int = 52,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 2,
        head_dim: int = 128,
        intermediate_size: int = 19968,
        vocab_size: int = 202048,
        rms_norm_eps: float = 1e-5,
        post_norm_eps: float = 1e-8,
        rope_theta: float = 500_000.0,
        max_position_embeddings: int = 16384,
        use_qk_norm: bool = True,
        qk_scale_factor: float = 43.7840518911,
        use_attn_output_gate: bool = True,
        output_multiplier: float = 0.19611613513818404,
        output_soft_cap_temp: float | None = 20.0,
        normalize_tok_embeddings: bool = True,
        sliding_window: int = 2048,
        sliding_window_pattern: list[int] | None = None,
        every_n_layers_nope: int = 4,
        no_rope_layers: list[int] | None = None,
        layer_types: list[str] | None = None,
        vision_latent_dim: int = 1536,
        vision_output_dim: int = 6144,
        vision_layers: int = 50,
        vision_heads: int = 16,
        vision_mlp_ratio: float = 8960 / 1536,
        vision_patch_size: int = 14,
        vision_patch_temporal: int = 2,
        vision_downsample_factor: int = 2,
        vision_sparse_attention_factor: int = 4,
        vision_pos_emb_grid_h: int = 32,
        vision_pos_emb_grid_w: int = 32,
        vision_adapter_dim: int = 4096,
        patch_token_id: int = 200092,
        image_token_id: int | None = None,
        video_token_id: int = 200091,
        vid_start_id: int = 200082,
        vid_end_id: int = 200083,
        vid_frame_sep_id: int = 200087,
        video_num_frames: int = 96,
        video_sampling_fps: float = 2.0,
        has_vision: bool = True,
        hidden_act: str = "silu",
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        mlp_bias: bool = False,
        tie_word_embeddings: bool = False,
        bos_token_id: int = 200000,
        eos_token_id: int = 200001,
        text_config: dict[str, Any] | MuseGlimmerTextConfig | None = None,
        vision_config: dict[str, Any] | MuseGlimmerVisionConfig | None = None,
        out_hidden_size: int | None = None,
        projector_hidden_size: int | None = None,
        projector_hidden_act: str = "gelu",
        **kwargs: Any,
    ) -> None:
        canonical_layout = text_config is not None
        if isinstance(text_config, dict):
            text_config = MuseGlimmerTextConfig(**text_config)
        if isinstance(vision_config, dict):
            vision_config = MuseGlimmerVisionConfig(**vision_config)

        if canonical_layout:
            if not isinstance(text_config, MuseGlimmerTextConfig):
                raise TypeError("text_config must be a mapping or MuseGlimmerTextConfig.")
            hidden_size = text_config.hidden_size
            num_hidden_layers = text_config.num_hidden_layers
            num_attention_heads = text_config.num_attention_heads
            num_key_value_heads = text_config.num_key_value_heads
            head_dim = text_config.head_dim
            intermediate_size = text_config.intermediate_size
            vocab_size = text_config.vocab_size
            rms_norm_eps = text_config.rms_norm_eps
            post_norm_eps = text_config.post_norm_eps
            rope_theta = float(text_config.rope_parameters.get("rope_theta", rope_theta))
            max_position_embeddings = text_config.max_position_embeddings
            qk_scale_factor = text_config.qk_scale_factor
            output_multiplier = text_config.output_multiplier
            output_soft_cap_temp = text_config.final_logit_softcapping
            sliding_window = text_config.sliding_window
            layer_types = list(text_config.layer_types)
            no_rope_layers = [int(theta != 0) for theta in text_config.layer_rope_theta]
            hidden_act = text_config.hidden_activation
            attention_dropout = text_config.attention_dropout
            attention_bias = text_config.attention_bias
            tie_word_embeddings = text_config.tie_word_embeddings
            bos_token_id = text_config.bos_token_id
            eos_token_id = text_config.eos_token_id

            if vision_config is not None:
                vision_latent_dim = vision_config.hidden_size
                vision_layers = vision_config.num_hidden_layers
                vision_heads = vision_config.num_attention_heads
                vision_mlp_ratio = vision_config.intermediate_size / vision_config.hidden_size
                vision_patch_size = vision_config.patch_size
                vision_patch_temporal = vision_config.patch_temporal
                vision_downsample_factor = vision_config.merge_size
                vision_pos_emb_grid_h = vision_config.pos_emb_height
                vision_pos_emb_grid_w = vision_config.pos_emb_width
            vision_output_dim = out_hidden_size if out_hidden_size is not None else vision_output_dim
            vision_adapter_dim = projector_hidden_size if projector_hidden_size is not None else vision_adapter_dim
            if image_token_id is not None:
                patch_token_id = image_token_id

        kwargs.setdefault("architectures", ["MuseGlimmerForConditionalGeneration"])
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.post_norm_eps = post_norm_eps
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.use_qk_norm = use_qk_norm
        self.qk_scale_factor = qk_scale_factor
        self.scale_query_by = qk_scale_factor if canonical_layout else qk_scale_factor / math.sqrt(head_dim)
        self.use_attn_output_gate = use_attn_output_gate
        self.output_multiplier = output_multiplier
        self.output_soft_cap_temp = output_soft_cap_temp
        self.normalize_tok_embeddings = normalize_tok_embeddings
        self.sliding_window = sliding_window
        self.every_n_layers_nope = every_n_layers_nope
        self.hidden_act = hidden_act
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.mlp_bias = mlp_bias

        self.vision_latent_dim = vision_latent_dim
        self.vision_output_dim = vision_output_dim
        self.vision_layers = vision_layers
        self.vision_heads = vision_heads
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_patch_size = vision_patch_size
        self.vision_patch_temporal = vision_patch_temporal
        self.vision_downsample_factor = vision_downsample_factor
        self.vision_sparse_attention_factor = vision_sparse_attention_factor
        self.vision_pos_emb_grid_h = vision_pos_emb_grid_h
        self.vision_pos_emb_grid_w = vision_pos_emb_grid_w
        self.vision_adapter_dim = vision_adapter_dim
        self.patch_token_id = patch_token_id
        self.image_token_id = patch_token_id
        self.video_token_id = video_token_id
        self.vid_start_id = vid_start_id
        self.vid_end_id = vid_end_id
        self.vid_frame_sep_id = vid_frame_sep_id
        self.video_num_frames = video_num_frames
        self.video_sampling_fps = video_sampling_fps
        self.has_vision = has_vision
        self.out_hidden_size = vision_output_dim
        self.projector_hidden_size = vision_adapter_dim
        self.projector_hidden_act = projector_hidden_act

        if sliding_window_pattern is None:
            sliding_window_pattern = [sliding_window, sliding_window, sliding_window, 0]
        self.sliding_window_pattern = sliding_window_pattern

        if no_rope_layers is None:
            no_rope_layers = [
                0 if (num_hidden_layers - layer_idx - 1) % every_n_layers_nope == 0 else 1
                for layer_idx in range(num_hidden_layers)
            ]
        self.no_rope_layers = no_rope_layers

        if layer_types is None:
            nope_freq = every_n_layers_nope or 1
            layer_types = []
            for layer_idx in range(num_hidden_layers):
                count_backward = layer_idx + nope_freq - num_hidden_layers % nope_freq
                pattern_value = sliding_window_pattern[count_backward % len(sliding_window_pattern)]
                layer_types.append("sliding_attention" if pattern_value > 0 else "full_attention")
        self.layer_types = layer_types

        if len(self.no_rope_layers) != num_hidden_layers:
            raise ValueError(
                f"no_rope_layers must contain {num_hidden_layers} entries, got {len(self.no_rope_layers)}."
            )
        if len(self.layer_types) != num_hidden_layers:
            raise ValueError(f"layer_types must contain {num_hidden_layers} entries, got {len(self.layer_types)}.")

        if text_config is None:
            text_config = MuseGlimmerTextConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                hidden_activation=hidden_act,
                max_position_embeddings=max_position_embeddings,
                rms_norm_eps=rms_norm_eps,
                post_norm_eps=post_norm_eps,
                attention_bias=attention_bias,
                attention_dropout=attention_dropout,
                sliding_window=sliding_window,
                layer_types=layer_types,
                layer_rope_theta=[rope_theta if use_rope else 0.0 for use_rope in no_rope_layers],
                rope_parameters={"rope_theta": rope_theta, "rope_type": "default"},
                qk_scale_factor=self.scale_query_by,
                output_multiplier=output_multiplier,
                final_logit_softcapping=output_soft_cap_temp,
                tie_word_embeddings=tie_word_embeddings,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
            )
        if vision_config is None:
            vision_config = MuseGlimmerVisionConfig(
                hidden_size=vision_latent_dim,
                intermediate_size=round(vision_mlp_ratio * vision_latent_dim),
                num_hidden_layers=vision_layers,
                num_attention_heads=vision_heads,
                patch_size=vision_patch_size,
                patch_temporal=vision_patch_temporal,
                merge_size=vision_downsample_factor,
                pos_emb_height=vision_pos_emb_grid_h,
                pos_emb_width=vision_pos_emb_grid_w,
            )
        self.text_config = text_config
        self.vision_config = vision_config

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
