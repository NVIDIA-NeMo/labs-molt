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

"""Standalone NeMo AutoModel implementation of the Inkling multimodal MoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .configuration import InklingConfig
from .layers import build_inkling_moe_config
from .multimodal import InklingModel
from .state_dict_adapter import InklingStateDictAdapter
from .text import InklingDynamicCache


@dataclass
class InklingCausalLMOutputWithPast(ModelOutput):
    """Inkling logits, optional loss/cache, and multimodal hidden states."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: InklingDynamicCache | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    image_hidden_states: torch.FloatTensor | None = None


class InklingForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Native Inkling VLM with expert-parallel feed-forwards."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    # The adapter covers every checkpoint tensor. Avoid initializing the 975B
    # sharded model immediately before loading it.
    _skip_init_weights_on_load: bool = True

    # Keep the multimodal forward under PP so stage 0 can consume media chunks.
    _pp_keep_self_forward: bool = True

    # Short convolutions and router correction bias use callable fp32 holders.
    _keep_in_fp32_modules_strict = ["_fp32_params"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = True
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: InklingConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> "InklingForConditionalGeneration":
        """Construct an Inkling model from its local AutoModel config."""
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> "InklingForConditionalGeneration":
        """Construct the native model tree for checkpoint loading.

        The NeMo AutoModel bridge and checkpointer own weight loading. This method
        resolves only the local Inkling config and native module structure.
        """
        config = kwargs.pop("config", None)
        if config is None:
            config = InklingConfig.from_pretrained(pretrained_model_name_or_path)
        torch_dtype = kwargs.pop("torch_dtype", getattr(config, "torch_dtype", torch.bfloat16))
        if isinstance(torch_dtype, str):
            torch_dtype = get_dtype(torch_dtype, torch.bfloat16)
        config.torch_dtype = torch_dtype
        config._name_or_path = pretrained_model_name_or_path
        model = cls.from_config(config, *model_args, **kwargs)
        model.name_or_path = pretrained_model_name_or_path
        return model

    def __init__(
        self,
        config: InklingConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        del kwargs
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.config = config
        self.backend = backend or BackendConfig()
        if self.backend.gate_precision is None:
            self.backend.gate_precision = torch.float32

        top_dtype = getattr(config, "torch_dtype", None)
        if top_dtype is not None:
            for sub_config in (config.text_config, config.audio_config, config.vision_config):
                sub_config.torch_dtype = top_dtype

        text_config = config.text_config
        self.moe_config = moe_config or build_inkling_moe_config(text_config, self.backend)
        self.model = InklingModel(config, self.backend, self.moe_config)
        self.model.moe_config = self.moe_config

        model_dtype = get_dtype(getattr(text_config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False, dtype=model_dtype)
        self.vocab_size = text_config.vocab_size
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = InklingStateDictAdapter(
                text_config,
                self.moe_config,
                self.backend,
                dtype=model_dtype,
            )
        cast_model_to_dtype(self, model_dtype)

    def get_input_embeddings(self) -> nn.Module:
        """Return the text token-embedding module."""
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, embeddings: nn.Module) -> None:
        """Replace the text token-embedding module."""
        self.model.language_model.set_input_embeddings(embeddings)

    def get_output_embeddings(self) -> nn.Module | None:
        """Return the language-model output projection."""
        return self.lm_head

    def set_output_embeddings(self, embeddings: nn.Module) -> None:
        """Replace the language-model output projection."""
        self.lm_head = embeddings

    def get_image_features(self, pixel_values: torch.Tensor, **kwargs: Any) -> Any:
        """Encode image/video patches through the native vision tower.

        Args:
            pixel_values: Tensor of shape ``[patches, time, height, width, channels]``.
            **kwargs: Reserved for the common multimodal calling convention.

        Returns:
            An output whose pooler output has shape ``[patches, text_hidden]``.
        """
        return self.model.get_image_features(pixel_values, **kwargs)

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module,
    ) -> list[list[str]]:
        """Keep Inkling's post-embedding norm on the first pipeline stage."""
        del text_model
        module_names_per_stage[0].append(f"{layers_prefix}embed_norm")
        return module_names_per_stage

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return PP input/output metadata using Inkling's unpadded vocabulary."""
        text_config = self.config.text_config
        hidden_shape = (microbatch_size, seq_len, text_config.hidden_size)
        vocab_size = text_config.unpadded_vocab_size or text_config.vocab_size
        if is_first:
            inputs_meta = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        else:
            inputs_meta = (torch.empty(*hidden_shape, device="meta", dtype=dtype),)
        if self.lm_head is None:
            outputs_meta = (torch.empty(*hidden_shape, device="meta", dtype=dtype),)
        else:
            outputs_meta = (
                torch.empty(microbatch_size, seq_len, vocab_size, device="meta", dtype=self.lm_head.weight.dtype),
            )
        return inputs_meta, outputs_meta

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        pixel_values: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: InklingDynamicCache | None = None,
        audio_input_ids: torch.LongTensor | None = None,
        audio_input_ids_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> InklingCausalLMOutputWithPast | torch.Tensor:
        """Run native Inkling multimodal conditional generation.

        Args:
            input_ids: Optional long tensor of shape ``[batch, sequence]``. Non-first
                pipeline stages may receive hidden states of shape ``[batch, sequence,
                hidden]`` through this argument.
            pixel_values: Optional tensor of shape ``[patches, time, height, width, channels]``.
            attention_mask: Optional padding tensor of shape ``[batch, total_sequence]``.
            position_ids: Optional long tensor of shape ``[batch, sequence]``.
            past_key_values: Optional model-owned decoding cache.
            audio_input_ids: Optional long tensor of shape ``[audios, frames, mel_bins]``.
            audio_input_ids_mask: Optional boolean tensor of shape ``[audios, frames]``.
            inputs_embeds: Optional tensor of shape ``[batch, sequence, hidden]``.
            labels: Optional long tensor of shape ``[batch, sequence]``.
            use_cache: Whether to allocate and return a decoding cache.
            logits_to_keep: Number or indices of trailing logits to compute.
            **kwargs: Additional text-attention arguments.

        Returns:
            An output whose logits have shape ``[batch, kept_sequence, vocab]`` during
            normal execution, or a raw tensor with that shape on pipeline stages.
        """
        language_model = self.model.language_model
        pipeline_mode = isinstance(language_model.layers, nn.ModuleDict)
        effective_use_cache = False if use_cache is None and self.training else use_cache
        if inputs_embeds is None and input_ids is not None and input_ids.dtype.is_floating_point:
            inputs_embeds = input_ids
            input_ids = None

        is_first_stage = language_model.embed_tokens is not None
        if pixel_values is None and is_first_stage:
            chunks = getattr(self, "_vlm_pixel_values_chunks", None)
            chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
            if chunks is not None and chunk_idx is not None and chunk_idx < len(chunks):
                pixel_values = chunks[chunk_idx]
                self._vlm_chunk_idx = chunk_idx + 1

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            audio_input_ids=audio_input_ids,
            audio_input_ids_mask=audio_input_ids_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=effective_use_cache,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        if self.lm_head is None:
            return hidden_states

        hidden_states = hidden_states / self.config.text_config.logits_mup_width_multiplier
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        unpadded_vocab_size = self.config.text_config.unpadded_vocab_size
        if unpadded_vocab_size is not None and unpadded_vocab_size < logits.shape[-1]:
            logits = logits[..., :unpadded_vocab_size].contiguous()
        if pipeline_mode:
            return logits

        loss = None
        if labels is not None:
            if logits.shape[1] != labels.shape[1]:
                raise ValueError("labels require logits_to_keep=0 so sequence lengths match")
            loss = F.cross_entropy(
                logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        return InklingCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            image_hidden_states=outputs.image_hidden_states,
        )

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize every parameter for checkpoint-free construction."""
        del dtype
        buffer_device = buffer_device or next(self.parameters()).device
        self.model.init_weights(buffer_device)
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.text_config.initializer_range)

    def update_moe_gate_bias(self) -> None:
        """Keep Inkling's trained router correction bias unchanged."""
        return


ModelClass = InklingForConditionalGeneration

__all__ = ["InklingForConditionalGeneration", "ModelClass"]
