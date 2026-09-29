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

"""Dense Qwen3-VL integration with in-forward context-parallel multimodal sharding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
)
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLForConditionalGeneration as HFQwen3VLForConditionalGeneration,
)

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    round_robin_local_indices,
    shard_batch_aux_only,
    shard_sequence_for_cp_round_robin,
)
from nemo_automodel.components.distributed.context_parallel.utils import cp_dispatcher_suspended
from nemo_automodel.components.distributed.cp_vision_frame_shard import maybe_distribute_visual
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.qwen3_vl.parallelization import register_qwen3_vl_parallel_strategy


class Qwen3VLForConditionalGeneration(HFCheckpointingMixin, HFQwen3VLForConditionalGeneration):
    """Dense Qwen3-VL with a DeepStack-aware context-parallel forward path."""

    tie_word_embeddings_support: TieSupport = TieSupport.BOTH
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for dense Qwen3-VL."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = False
        supports_ep: bool = False
        supports_thd: bool = False
        supports_cp_vision_frame_sharding: bool = True

    def __init__(self, config: Qwen3VLConfig) -> None:
        """Construct the Hugging Face-compatible dense Qwen3-VL model.

        Args:
            config: Qwen3-VL configuration containing the text and vision sub-configs.
        """
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__(config)

    def tie_weights(self, *_args: object, **_kwargs: object) -> None:
        """Tie the language head to the active text embedding when configured."""
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def _prepare_visual_inputs_for_cp(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Encode and scatter Qwen3-VL visual features before sequence sharding.

        Images and videos share one vision-tower call when both are present. This
        preserves collective ordering under FSDP and lets CP vision frame sharding
        partition their combined frame stream once.

        Args:
            input_ids: Token ids of shape ``[batch, sequence]`` containing image
                and video placeholder ids.
            inputs_embeds: Text embeddings of shape ``[batch, sequence, hidden]``.
            pixel_values: Optional image patch rows of shape
                ``[image_patch_rows, patch_dim]`` in image-entry order.
            pixel_values_videos: Optional video patch rows of shape
                ``[video_patch_rows, patch_dim]`` in video-entry/frame order.
            image_grid_thw: Optional image grid tensor of shape ``[num_images, 3]``.
            video_grid_thw: Optional video grid tensor of shape ``[num_videos, 3]``.

        Returns:
            A tuple containing updated embeddings of shape ``[batch, sequence,
            hidden]``, an optional boolean visual-position mask of shape ``[batch,
            sequence]``, and optional DeepStack tensors each shaped
            ``[visual_tokens, hidden]`` in sequence visual-token order.

        Raises:
            ValueError: If patch rows are provided without matching grid metadata,
                or grid metadata is provided without patch rows.
        """
        if (pixel_values is None) != (image_grid_thw is None):
            raise ValueError("Qwen3-VL image pixel_values and image_grid_thw must be provided together")
        if (pixel_values_videos is None) != (video_grid_thw is None):
            raise ValueError("Qwen3-VL video pixel_values_videos and video_grid_thw must be provided together")

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        has_images = pixel_values is not None
        has_videos = pixel_values_videos is not None

        if has_images and has_videos:
            image_pixels = pixel_values.type(self.model.visual.dtype)
            video_pixels = pixel_values_videos.type(self.model.visual.dtype)
            merged_output = maybe_distribute_visual(
                self.model.visual,
                torch.cat((image_pixels, video_pixels), dim=0),
                torch.cat((image_grid_thw, video_grid_thw), dim=0),
            )
            spatial_merge_size_sq = self.model.visual.spatial_merge_size**2
            num_image_tokens = int((image_grid_thw.prod(-1) // spatial_merge_size_sq).sum().item())
            image_embeds = merged_output.pooler_output[:num_image_tokens]
            video_embeds = merged_output.pooler_output[num_image_tokens:]
            merged_deepstack = merged_output.deepstack_features or []
            deepstack_image_embeds = [features[:num_image_tokens] for features in merged_deepstack]
            deepstack_video_embeds = [features[num_image_tokens:] for features in merged_deepstack]
        elif has_images:
            image_output = maybe_distribute_visual(
                self.model.visual,
                pixel_values.type(self.model.visual.dtype),
                image_grid_thw,
            )
            image_embeds = image_output.pooler_output
            deepstack_image_embeds = image_output.deepstack_features or []
        elif has_videos:
            video_output = maybe_distribute_visual(
                self.model.visual,
                pixel_values_videos.type(self.model.visual.dtype),
                video_grid_thw,
            )
            video_embeds = video_output.pooler_output
            deepstack_video_embeds = video_output.deepstack_features or []

        if has_images:
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if has_videos:
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            deepstack_visual_embeds = []
            for image_features, video_features in zip(deepstack_image_embeds, deepstack_video_embeds):
                joint_features = image_features.new_zeros(visual_pos_masks.sum(), image_features.shape[-1])
                joint_features[image_mask_joint] = image_features
                joint_features[video_mask_joint] = video_features
                deepstack_visual_embeds.append(joint_features)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds
        else:
            visual_pos_masks = None
            deepstack_visual_embeds = None

        return inputs_embeds, visual_pos_masks, deepstack_visual_embeds

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Return Qwen3-VL's aux-only sharder and full-sequence mRoPE positions.

        Embedding, vision encoding, DeepStack construction, and differentiable
        sequence sharding run inside :meth:`forward` per microbatch. This hook
        only computes metadata and selects the round-robin CP strategy, so it
        does not touch model weights or create an autograd graph shared across
        microbatches.

        Args:
            batch: Full-sequence input mapping. ``input_ids`` has shape
                ``[batch, sequence]``; optional grid metadata has shape
                ``[num_media, 3]``.
            num_chunks: Accepted for the common model-hook contract; unused by
                the round-robin Qwen3-VL strategy.

        Returns:
            Mapping containing a :class:`ContextParallelSharder`, optional
            full-sequence mRoPE ``position_ids``, and a ``None`` replacement for
            ``mm_token_type_ids`` after that metadata has been consumed.

        Raises:
            ValueError: If ``input_ids`` is missing or multimodal position
                metadata is incomplete.
        """
        del num_chunks
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("Qwen3-VL context parallelism requires input_ids")

        position_ids = batch.get("position_ids")
        if position_ids is None:
            position_ids = self.model.compute_3d_position_ids(
                input_ids=input_ids,
                inputs_embeds=None,
                image_grid_thw=batch.get("image_grid_thw"),
                video_grid_thw=batch.get("video_grid_thw"),
                attention_mask=batch.get("attention_mask"),
                mm_token_type_ids=batch.get("mm_token_type_ids"),
            )

        prepared: dict[str, Any] = {
            "cp_sharder": ContextParallelSharder(
                shard_batch=shard_batch_aux_only,
                local_token_global_indices=round_robin_local_indices,
            ),
            "mm_token_type_ids": None,
        }
        if position_ids is not None:
            prepared["position_ids"] = position_ids
        return prepared

    def _shard_multimodal_inputs_for_cp(
        self,
        inputs_embeds: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        deepstack_visual_embeds: list[torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor] | None]:
        """Shard token embeddings and ragged DeepStack inputs in matching order.

        Args:
            inputs_embeds: Full-sequence embeddings ``[batch, sequence, hidden]``.
            visual_pos_masks: Optional full-sequence visual mask
                ``[batch, sequence]``.
            deepstack_visual_embeds: Optional tensors shaped
                ``[visual_tokens, hidden]``, one per DeepStack injection layer.

        Returns:
            Local embeddings ``[batch, local_sequence, hidden]``, the matching
            local visual mask, and local ragged DeepStack tensors.

        Raises:
            TypeError: If ``visual_pos_masks`` is not boolean.
            ValueError: If mask and DeepStack shapes are inconsistent.
        """
        cp_mesh = getattr(self, "cp_mesh", None)
        local_inputs, _, _ = shard_sequence_for_cp_round_robin(cp_mesh, inputs_embeds, seq_dim=1)
        if visual_pos_masks is None:
            return local_inputs, None, None
        if visual_pos_masks.dtype != torch.bool:
            raise TypeError("Qwen3-VL visual_pos_masks must be a boolean tensor")
        if visual_pos_masks.shape != inputs_embeds.shape[:2]:
            raise ValueError(
                "Qwen3-VL visual_pos_masks must match inputs_embeds on batch and sequence axes, got "
                f"{tuple(visual_pos_masks.shape)} and {tuple(inputs_embeds.shape[:2])}"
            )

        deepstack_visual_embeds = deepstack_visual_embeds or []
        num_visual_tokens = int(visual_pos_masks.sum().item())
        sequence_aligned: list[torch.Tensor] = []
        for layer_idx, embeds in enumerate(deepstack_visual_embeds):
            if embeds.ndim != 2:
                raise ValueError(
                    f"Qwen3-VL DeepStack tensor {layer_idx} must have shape "
                    f"[visual_tokens, hidden], got {tuple(embeds.shape)}"
                )
            if embeds.shape[0] != num_visual_tokens:
                raise ValueError(
                    f"Qwen3-VL DeepStack tensor {layer_idx} has {embeds.shape[0]} visual tokens "
                    f"but visual_pos_masks selects {num_visual_tokens}"
                )
            aligned = embeds.new_zeros(*visual_pos_masks.shape, embeds.shape[-1])
            aligned[visual_pos_masks] = embeds
            sequence_aligned.append(aligned)

        local_mask, _, _ = shard_sequence_for_cp_round_robin(
            cp_mesh,
            visual_pos_masks,
            seq_dim=1,
            pad_value=False,
        )
        local_deepstack = []
        for aligned in sequence_aligned:
            local_aligned, _, _ = shard_sequence_for_cp_round_robin(cp_mesh, aligned, seq_dim=1)
            local_deepstack.append(local_aligned[local_mask])
        return local_inputs, local_mask, local_deepstack

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> tuple | Qwen3VLCausalLMOutputWithPast:
        """Run Qwen3-VL, including its in-forward CP multimodal path.

        Named tensor arguments preserve the Hugging Face
        ``Qwen3VLForConditionalGeneration.forward`` layouts. The internal CP path
        consumes ``inputs_embeds`` of shape ``[batch, local_sequence, hidden]``,
        ``visual_pos_masks`` of shape ``[batch, local_sequence]``, and DeepStack
        tensors of shape ``[local_visual_tokens, hidden]``.
        """
        cp_mesh = getattr(self, "cp_mesh", None)
        cp_size = cp_mesh.size() if cp_mesh is not None else 1
        if cp_size <= 1:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                use_cache=use_cache,
                cache_position=cache_position,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )

        if input_ids is None:
            if inputs_embeds is None:
                raise ValueError("Qwen3-VL context parallelism requires input_ids or inputs_embeds")
            if pixel_values is not None or pixel_values_videos is not None:
                raise ValueError("Qwen3-VL media inputs under context parallelism require input_ids")
            visual_pos_masks = None
            deepstack_visual_embeds = None
        else:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            with cp_dispatcher_suspended(cp_mesh):
                inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._prepare_visual_inputs_for_cp(
                    input_ids,
                    inputs_embeds,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                )

        inputs_embeds, visual_pos_masks, deepstack_visual_embeds = self._shard_multimodal_inputs_for_cp(
            inputs_embeds,
            visual_pos_masks,
            deepstack_visual_embeds,
        )
        text_outputs = self.model.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )
        hidden_states = text_outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=text_outputs.past_key_values,
            hidden_states=text_outputs.hidden_states,
            attentions=text_outputs.attentions,
            rope_deltas=self.model.rope_deltas,
        )


register_qwen3_vl_parallel_strategy()
ModelClass = Qwen3VLForConditionalGeneration
