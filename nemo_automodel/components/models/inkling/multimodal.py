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

"""Native Inkling vision, audio, and multimodal composition modules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .configuration import InklingAudioConfig, InklingConfig, InklingVisionConfig
from .text import InklingDynamicCache, InklingRMSNorm, InklingTextModel


@dataclass
class InklingModelOutputWithPast(BaseModelOutputWithPast):
    """Inkling backbone output including projected image features."""

    image_hidden_states: torch.FloatTensor | None = None


class InklingAudioEmbeddings(nn.Module):
    """Embed and sum the independently quantized dMel codebooks."""

    def __init__(self, config: InklingAudioConfig) -> None:
        super().__init__()
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.embed_audio_tokens = nn.Embedding(
            config.n_mel_bins * config.mel_vocab_size,
            config.text_hidden_size,
            dtype=model_dtype,
        )
        self.register_buffer(
            "audio_tokens_offsets",
            torch.arange(config.n_mel_bins) * config.mel_vocab_size,
            persistent=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed discretized mel bins.

        Args:
            input_ids: Long tensor of shape ``[frames, mel_bins]``.

        Returns:
            Tensor of shape ``[frames, hidden]``.
        """
        embeddings = self.embed_audio_tokens(input_ids + self.audio_tokens_offsets)
        return embeddings.sum(dim=-2)


class InklingAudioModel(nn.Module):
    """Native Inkling audio tower."""

    def __init__(self, config: InklingAudioConfig) -> None:
        super().__init__()
        self.config = config
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.embed_audio_tokens = InklingAudioEmbeddings(config)
        self.norm = InklingRMSNorm(config.text_hidden_size, config.rms_norm_eps, dtype=model_dtype)

    def forward(self, audio_input_ids: torch.Tensor) -> BaseModelOutputWithPooling:
        """Encode audio tokens.

        Args:
            audio_input_ids: Long tensor of shape ``[frames, mel_bins]``.

        Returns:
            An output whose hidden state and pooler output have shape ``[frames, hidden]``.
        """
        hidden_states = self.norm(self.embed_audio_tokens(audio_input_ids))
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=hidden_states)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize the audio embedding and final normalization."""
        nn.init.normal_(self.embed_audio_tokens.embed_audio_tokens.weight, std=self.config.initializer_range)
        self.norm.init_weights()


class InklingVisionEncoderLayer(nn.Module):
    """Fold time and space into channels, then project one HMLP stage."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        temporal_fold: int,
        spatial_fold: int,
        add_norm: bool,
        *,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False, dtype=dtype)
        self.layer_norm = InklingRMSNorm(output_dim, 1e-6, dtype=dtype) if add_norm else None
        self.spatial_fold = spatial_fold
        self.temporal_fold = temporal_fold

    def _fold_timespace_to_depth(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Fold time and two spatial axes into the channel dimension.

        Args:
            hidden_states: Tensor of shape ``[patches, time, height, width, channels]``.

        Returns:
            Tensor of shape ``[patches, time / temporal_fold, height / spatial_fold,
            width / spatial_fold, channels * temporal_fold * spatial_fold**2]``.
        """
        patches, time, height, width, channels = hidden_states.shape
        new_time = time // self.temporal_fold
        new_height = height // self.spatial_fold
        new_width = width // self.spatial_fold
        hidden_states = hidden_states.reshape(
            patches,
            new_time,
            self.temporal_fold,
            new_height,
            self.spatial_fold,
            new_width,
            self.spatial_fold,
            channels,
        )
        hidden_states = hidden_states.permute(0, 1, 3, 5, 2, 4, 6, 7)
        return hidden_states.reshape(
            patches,
            new_time,
            new_height,
            new_width,
            self.temporal_fold * self.spatial_fold * self.spatial_fold * channels,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply one hierarchical vision projection.

        Args:
            hidden_states: Tensor of shape ``[patches, time, height, width, channels]``.

        Returns:
            Tensor of shape ``[patches, folded_time, folded_height, folded_width, output_channels]``.
        """
        if self.spatial_fold > 1 or self.temporal_fold > 1:
            hidden_states = self._fold_timespace_to_depth(hidden_states)
        hidden_states = self.projection(hidden_states)
        if self.layer_norm is not None:
            hidden_states = F.gelu(self.layer_norm(hidden_states))
        return hidden_states

    @torch.no_grad()
    def init_weights(self, init_std: float) -> None:
        """Initialize the projection and optional normalization."""
        nn.init.normal_(self.projection.weight, mean=0.0, std=init_std)
        if self.layer_norm is not None:
            self.layer_norm.init_weights()


def _prime_factors(number: int) -> list[int]:
    """Return prime factors in ascending order."""
    factors = []
    while number % 2 == 0:
        factors.append(2)
        number //= 2
    for divisor in range(3, math.isqrt(number) + 1, 2):
        while number % divisor == 0:
            factors.append(divisor)
            number //= divisor
    if number > 1:
        factors.append(number)
    return factors


def _minimum_cost_scale_indices(cost_matrix: torch.Tensor) -> torch.LongTensor:
    """Select distinct ordered scales with minimum absolute-log cost.

    Args:
        cost_matrix: Tensor of shape ``[layers_plus_one, available_scales]``.

    Returns:
        Long tensor of shape ``[layers_plus_one]`` indexing distinct scales.
    """
    rows, columns = cost_matrix.shape
    costs = torch.full_like(cost_matrix, float("inf"))
    parents = torch.full((rows, columns), -1, dtype=torch.long, device=cost_matrix.device)
    costs[0] = cost_matrix[0]
    for row in range(1, rows):
        for column in range(row, columns):
            previous_cost, previous_column = torch.min(costs[row - 1, :column], dim=0)
            costs[row, column] = cost_matrix[row, column] + previous_cost
            parents[row, column] = previous_column

    indices = torch.empty(rows, dtype=torch.long, device=cost_matrix.device)
    indices[-1] = torch.argmin(costs[-1])
    for row in range(rows - 1, 0, -1):
        indices[row - 1] = parents[row, indices[row]]
    return indices


def _plan_output_scales(config: InklingVisionConfig) -> torch.LongTensor:
    """Plan the time, height, width, and channel sizes of every HMLP stage."""
    # This is construction-time integer bookkeeping, not model state. Keep it
    # on CPU even when the module tree is being materialized under ``device=meta``.
    spatial = torch.cumprod(torch.tensor(_prime_factors(config.patch_size)[::-1], device="cpu"), dim=0)
    temporal = torch.cumprod(
        torch.tensor(_prime_factors(config.temporal_patch_size)[::-1], device="cpu"),
        dim=0,
    )
    spatial_channels = torch.ceil(spatial**2 * config.num_channels / 64).int() * 64
    temporal_channels = torch.ceil(spatial[-1] ** 2 * config.num_channels * temporal).int() * 64

    base = torch.tensor([[1, 1, 1, config.num_channels]], device="cpu")
    spatial_scales = torch.stack((torch.ones_like(spatial), spatial, spatial, spatial_channels), dim=1)
    temporal_scales = torch.stack(
        (
            temporal,
            torch.full_like(temporal, spatial[-1]),
            torch.full_like(temporal, spatial[-1]),
            temporal_channels,
        ),
        dim=1,
    )
    scales = torch.cat((base, spatial_scales, temporal_scales), dim=0)
    reductions = torch.prod(scales[:, :-1], dim=1).float()
    total_elements = config.patch_size**2 * config.temporal_patch_size * config.num_channels
    ideal = torch.linspace(0, math.log(total_elements), config.num_hidden_layers + 1, device="cpu")
    cost_matrix = torch.abs(ideal.unsqueeze(1) - torch.log(reductions).unsqueeze(0))
    if config.num_hidden_layers >= scales.shape[0]:
        indices = torch.argmin(cost_matrix, dim=1)
    else:
        indices = _minimum_cost_scale_indices(cost_matrix)
    indices[0] = 0
    indices[-1] = scales.shape[0] - 1
    return scales[indices]


class InklingVisionModel(nn.Module):
    """Native hierarchical MLP vision tower."""

    def __init__(self, config: InklingVisionConfig) -> None:
        super().__init__()
        self.config = config
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        scales = _plan_output_scales(config)
        self.register_buffer("scales", scales, persistent=False)
        self.encoder_layers = nn.ModuleList()
        for layer_idx, (start_scale, end_scale) in enumerate(zip(scales[:-1], scales[1:])):
            shuffle_multiplier = int(
                (end_scale[0] // start_scale[0]) * (end_scale[1] // start_scale[1]) * (end_scale[2] // start_scale[2])
            )
            output_dim = config.text_hidden_size if layer_idx == config.num_hidden_layers - 1 else int(end_scale[3])
            self.encoder_layers.append(
                InklingVisionEncoderLayer(
                    input_dim=int(start_scale[3]) * shuffle_multiplier,
                    output_dim=output_dim,
                    spatial_fold=int(end_scale[1] // start_scale[1]),
                    temporal_fold=int(end_scale[0] // start_scale[0]),
                    add_norm=layer_idx != config.num_hidden_layers - 1,
                    dtype=model_dtype,
                )
            )
        self.final_norm = InklingRMSNorm(config.text_hidden_size, config.rms_norm_eps, dtype=model_dtype)

    def forward(self, pixel_values: torch.Tensor, **kwargs: Any) -> BaseModelOutputWithPooling:
        """Encode preprocessed image/video patches.

        Args:
            pixel_values: Tensor of shape ``[patches, time, height, width, channels]``.
            **kwargs: Reserved for the common multimodal calling convention.

        Returns:
            An output whose pooler output has shape ``[patches, sequence, text_hidden]``.
        """
        del kwargs
        hidden_states = pixel_values
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.final_norm(hidden_states).reshape(pixel_values.shape[0], -1, self.config.text_hidden_size)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states, pooler_output=hidden_states)

    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize every vision projection and normalization."""
        for layer in self.encoder_layers:
            layer.init_weights(self.config.initializer_range)
        self.final_norm.init_weights()


class InklingModel(nn.Module):
    """Compose native text, vision, and audio Inkling towers."""

    def __init__(
        self,
        config: InklingConfig,
        backend: BackendConfig,
        moe_config: MoEConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.language_model = InklingTextModel(config.text_config, backend, moe_config)
        self.audio_tower = InklingAudioModel(config.audio_config)
        self.vision_tower = InklingVisionModel(config.vision_config)

    def get_input_embeddings(self) -> nn.Module:
        """Return the text token-embedding module."""
        return self.language_model.get_input_embeddings()

    def get_image_features(self, pixel_values: torch.Tensor, **kwargs: Any) -> BaseModelOutputWithPooling:
        """Project image/video patches into text hidden space.

        Args:
            pixel_values: Tensor of shape ``[patches, time, height, width, channels]``.
            **kwargs: Reserved for the common multimodal calling convention.

        Returns:
            An output whose pooler output has shape ``[patches, sequence, text_hidden]``.
        """
        return self.vision_tower(pixel_values, **kwargs)

    def get_audio_features(
        self,
        audio_input_ids: torch.Tensor,
        audio_input_ids_mask: torch.Tensor | None = None,
    ) -> BaseModelOutputWithPooling:
        """Project discretized audio into text hidden space.

        Args:
            audio_input_ids: Long tensor of shape ``[audios, frames, mel_bins]``.
            audio_input_ids_mask: Optional boolean tensor of shape ``[audios, frames]``.

        Returns:
            An output whose hidden state has shape ``[valid_frames, text_hidden]``.
        """
        if audio_input_ids_mask is not None:
            audio_input_ids = audio_input_ids[audio_input_ids_mask.bool()]
        else:
            audio_input_ids = audio_input_ids.reshape(-1, audio_input_ids.shape[-1])
        return self.audio_tower(audio_input_ids)

    def _get_placeholder_mask(
        self,
        input_ids: torch.Tensor | None,
        inputs_embeds: torch.Tensor,
        features: torch.Tensor,
        token_id: int,
    ) -> torch.Tensor:
        """Match multimodal features to placeholder tokens.

        Args:
            input_ids: Optional long tensor of shape ``[batch, sequence]``.
            inputs_embeds: Tensor of shape ``[batch, sequence, hidden]``.
            features: Tensor of shape ``[placeholders, hidden]``.
            token_id: Vocabulary ID used as the placeholder.

        Returns:
            Boolean tensor of shape ``[batch, sequence, hidden]``.
        """
        if input_ids is None:
            token = torch.tensor(token_id, dtype=torch.long, device=inputs_embeds.device)
            special_mask = (inputs_embeds == self.get_input_embeddings()(token)).all(dim=-1)
        else:
            special_mask = input_ids == token_id
        token_count = int(special_mask.sum().item())
        expanded_mask = special_mask.unsqueeze(-1).expand_as(inputs_embeds)
        if inputs_embeds[expanded_mask].numel() != features.numel():
            raise ValueError(
                "Multimodal features and placeholder tokens do not match: "
                f"tokens={token_count}, features={features.shape[0]}"
            )
        return expanded_mask

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: InklingDynamicCache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> InklingModelOutputWithPast:
        """Merge multimodal features and run the text backbone.

        Args:
            input_ids: Optional long tensor of shape ``[batch, sequence]``.
            pixel_values: Optional tensor of shape ``[patches, time, height, width, channels]``.
            audio_input_ids: Optional long tensor of shape ``[audios, frames, mel_bins]``.
            audio_input_ids_mask: Optional boolean tensor of shape ``[audios, frames]``.
            attention_mask: Optional padding tensor of shape ``[batch, total_sequence]``
                or mapping of prepared attention masks.
            position_ids: Optional long tensor of shape ``[batch, sequence]``.
            past_key_values: Optional model-owned decoding cache.
            inputs_embeds: Optional tensor of shape ``[batch, sequence, hidden]``.
            use_cache: Whether to allocate and return a decoding cache.
            **kwargs: Additional text-attention arguments.

        Returns:
            An output whose last hidden state has shape ``[batch, sequence, hidden]``.
        """
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must provide either input_ids or inputs_embeds")
            inputs_embeds = self.language_model.embed_norm(self.get_input_embeddings()(input_ids))
        elif input_ids is not None:
            raise ValueError("You must provide exactly one of input_ids or inputs_embeds")

        image_features = None
        if pixel_values is not None:
            if self.vision_tower is None:
                raise ValueError("pixel_values were provided to a pipeline stage without the vision tower")
            image_features = self.get_image_features(pixel_values).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = self._get_placeholder_mask(
                input_ids, inputs_embeds, image_features, self.config.image_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        if audio_input_ids is not None:
            if self.audio_tower is None:
                raise ValueError("audio_input_ids were provided to a pipeline stage without the audio tower")
            audio_features = self.get_audio_features(audio_input_ids, audio_input_ids_mask).last_hidden_state
            audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
            audio_mask = self._get_placeholder_mask(
                input_ids, inputs_embeds, audio_features, self.config.audio_token_id
            )
            inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )
        return InklingModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            image_hidden_states=image_features,
        )

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device) -> None:
        """Initialize every tower for checkpoint-free construction."""
        self.language_model.init_weights(buffer_device)
        self.audio_tower.init_weights()
        self.vision_tower.init_weights()


__all__ = ["InklingModel", "InklingModelOutputWithPast"]
