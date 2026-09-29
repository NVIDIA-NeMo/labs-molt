# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native AutoModel implementation of GLM-5.3-Flash."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    contiguous_local_indices,
)
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype, compute_lm_head_logits
from nemo_automodel.components.models.glm5_next.config import Glm5NextConfig, Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.cp import (
    Glm5NextPackedContext,
    doc_ids_from_cu_seqlens,
    shard_batch_for_glm5_next_cp,
)
from nemo_automodel.components.models.glm5_next.layers import (
    Glm5NextDecoderLayer,
    Glm5NextRMSNorm,
)
from nemo_automodel.components.models.glm5_next.vision import Glm5NextVisionModel, Glm5NextVisionOutput
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def build_glm5_next_moe_config(
    config: Glm5NextTextConfig,
    dtype: torch.dtype,
    overrides: dict[str, Any] | None = None,
) -> MoEConfig:
    """Translate the GLM router/expert contract to AutoModel's grouped MoE."""
    values = dict(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=config.n_shared_experts,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=config.n_group,
        n_limited_groups=config.topk_group,
        train_gate=True,
        gate_bias_update_factor=1e-3,
        # HF uses the correction bias only to select experts; routing weights
        # are gathered from the unbiased sigmoid scores.
        score_func="sigmoid_with_bias",
        route_scale=config.routed_scaling_factor,
        aux_loss_coeff=0.0,
        norm_topk_prob=config.norm_topk_prob,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        apply_router_weight_after_down=True,
        swiglu_limit=config.swiglu_limit,
        softmax_before_topk=False,
        router_weights_fp32=True,
        router_weight_uses_score_correction_bias=False,
        shared_expert_gate=False,
        shared_expert_inter_dim=config.moe_intermediate_size,
        force_e_score_correction_bias=True,
        dtype=dtype,
    )
    if overrides:
        values.update(overrides)
    return MoEConfig(**values)


def _packed_context_from_inputs(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    padding_mask: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    packed_seq_ids: torch.Tensor | None,
) -> Glm5NextPackedContext:
    """Build one global document map for a non-CP forward."""
    sequence = input_ids.shape[1]
    if packed_seq_ids is not None:
        doc_ids = packed_seq_ids.to(torch.int32)
        if doc_ids.ndim == 1:
            doc_ids = doc_ids.unsqueeze(0)
    elif cu_seqlens is not None:
        doc_ids = doc_ids_from_cu_seqlens(cu_seqlens, sequence)
    elif attention_mask is not None and attention_mask.ndim == 2:
        doc_ids = attention_mask.to(torch.int32)
    else:
        doc_ids = torch.ones(input_ids.shape[:2], dtype=torch.int32, device=input_ids.device)
        if padding_mask is not None:
            doc_ids.masked_fill_(padding_mask.bool(), 0)
    return Glm5NextPackedContext(doc_ids=doc_ids, original_seq_len=sequence)


class Glm5NextTextModel(nn.Module):
    """Embedding, mHC decoder stack, mean stream collapse and final RMSNorm."""

    def __init__(
        self,
        config: Glm5NextTextConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Pass either moe_config or moe_overrides, not both")
        self.config = config
        self.backend = backend
        dtype = get_dtype(getattr(config, "torch_dtype", getattr(config, "dtype", None)), torch.bfloat16)
        self.moe_config = moe_config or build_glm5_next_moe_config(config, dtype, moe_overrides)
        self.padding_idx = config.pad_token_id
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            dtype=dtype,
        )
        self.layers = nn.ModuleDict(
            {
                str(layer_idx): Glm5NextDecoderLayer(config, layer_idx, self.moe_config, backend)
                for layer_idx in range(config.num_hidden_layers)
            }
        )
        self.norm = Glm5NextRMSNorm(config.hidden_size, config.rms_norm_eps, dtype)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        glm5_next_packed_context: Glm5NextPackedContext,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run ``[batch, local_sequence]`` ids/embeddings through the text model."""
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids and inputs_embeds")
        hidden = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        hidden = hidden.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        for layer in self.layers.values():
            hidden = layer(
                hidden,
                packed_context=glm5_next_packed_context,
                padding_mask=padding_mask,
                **kwargs,
            )
        return self.norm(hidden.mean(dim=2))

    def update_moe_gate_bias(self) -> None:
        """Update every sparse layer's no-aux-loss routing correction bias."""
        for layer in self.layers.values():
            layer.update_moe_gate_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device) -> None:
        """Initialize a checkpoint-free text model on ``buffer_device``."""
        init_std = self.config.initializer_range
        with buffer_device:
            nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=init_std)
            if self.padding_idx is not None:
                self.embed_tokens.weight[self.padding_idx].zero_()
            self.norm.reset_parameters()
        for layer in self.layers.values():
            layer.init_weights(buffer_device, init_std)


class Glm5NextModel(nn.Module):
    """Checkpoint-layout container for ``visual`` and ``language_model``."""

    def __init__(
        self,
        config: Glm5NextConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.visual = Glm5NextVisionModel(config.vision_config)
        self.language_model = Glm5NextTextModel(
            config.text_config,
            backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> Glm5NextVisionOutput:
        """Encode image patches and split-free concatenated features."""
        return self.visual(pixel_values.to(self.visual.dtype), image_grid_thw)


class Glm5NextForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Trainable GLM-5.3 VLM with EP and contiguous packed CP support."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _skip_init_weights_on_load = True
    _owns_cp_attention = True
    _owns_packed_attention = True
    _packed_cp_attn_backends = ("sdpa", "cudnn")
    _keep_in_fp32_modules_strict = [
        "_fp32_params",
        "e_score_correction_bias",
        "rotary_pos_emb",
    ]
    cp_mesh = None

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Parallel axes intentionally supported for the released checkpoint."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = False
        supports_ep: bool = True
        supports_thd: bool = True

    @classmethod
    def from_config(
        cls,
        config: Glm5NextConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> "Glm5NextForConditionalGeneration":
        """Construct from an already resolved native config."""
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> "Glm5NextForConditionalGeneration":
        """Resolve the local config; checkpoint loading is owned by AutoModel."""
        config = Glm5NextConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Glm5NextConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.config = config
        self.backend = copy.copy(backend) if backend is not None else BackendConfig()
        if self.backend.gate_precision is None:
            self.backend.gate_precision = torch.float32
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = Glm5NextModel(
            config,
            self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        text_config = config.text_config
        dtype = get_dtype(
            getattr(text_config, "torch_dtype", getattr(text_config, "dtype", None)),
            torch.bfloat16,
        )
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            text_config.hidden_size,
            text_config.vocab_size,
            bias=False,
            dtype=dtype,
        )
        self.vocab_size = text_config.vocab_size
        if self.backend.enable_hf_state_dict_adapter:
            from nemo_automodel.components.models.glm5_next.state_dict_adapter import Glm5NextStateDictAdapter

            self.state_dict_adapter = Glm5NextStateDictAdapter(
                config,
                self.model.language_model.moe_config,
                self.backend,
                dtype=dtype,
            )

    @property
    def language_model(self) -> Glm5NextTextModel:
        """Expose the text module through the multimodal discovery protocol."""
        return self.model.language_model

    def get_input_embeddings(self) -> nn.Module:
        """Return the token embedding table."""
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Replace the token embedding table."""
        self.model.language_model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        """Return the untied language-model head."""
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        """Replace the language-model head."""
        self.lm_head = value

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor) -> Glm5NextVisionOutput:
        """Return raw and merged features for flattened image patches."""
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def _embed_and_splice(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        """Embed the full sequence and replace image placeholder positions."""
        embeddings = self.get_input_embeddings()(input_ids)
        if pixel_values is None:
            return embeddings
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required when pixel_values are provided")
        features = self.get_image_features(pixel_values, image_grid_thw).pooler_output
        mask = input_ids == self.config.image_token_id
        expected = int(mask.sum().item())
        if features.shape[0] != expected:
            raise ValueError(f"GLM-5.3 produced {features.shape[0]} image tokens for {expected} placeholders")
        embeddings = embeddings.clone()
        embeddings[mask] = features.to(device=embeddings.device, dtype=embeddings.dtype)
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Run image splice, contiguous CP slicing, text decoding and lm head."""
        del position_ids
        if pixel_values_videos is not None:
            raise NotImplementedError("GLM-5.3 AutoModel onboarding currently supports images, not video training")
        is_thd = kwargs.get("qkv_format") == "thd"
        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if inputs_embeds is not None and inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if padding_mask is not None and padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(0)
        if input_ids is None and inputs_embeds is None:
            raise ValueError("input_ids or inputs_embeds is required")

        context = kwargs.pop("glm5_next_packed_context", None)
        context_ids = input_ids
        if context_ids is None:
            context_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)
        if context is None:
            context = _packed_context_from_inputs(
                context_ids,
                attention_mask,
                padding_mask,
                kwargs.get("cu_seqlens"),
                kwargs.get("_packed_seq_ids"),
            )

        if inputs_embeds is None:
            inputs_embeds = self._embed_and_splice(input_ids, pixel_values, image_grid_thw)
        elif pixel_values is not None:
            raise ValueError("pixel_values cannot be combined with precomputed inputs_embeds")

        padded_length = context.doc_ids.shape[1]
        if inputs_embeds.shape[1] < padded_length:
            pad = inputs_embeds.new_zeros(
                inputs_embeds.shape[0], padded_length - inputs_embeds.shape[1], inputs_embeds.shape[2]
            )
            inputs_embeds = torch.cat((inputs_embeds, pad), dim=1)
        local_length = context.local_seq_len
        if inputs_embeds.shape[1] != local_length or context.seq_start:
            inputs_embeds = inputs_embeds[:, context.seq_start : context.seq_start + local_length].contiguous()
        if padding_mask is None:
            padding_mask = context.local_doc_ids <= 0

        hidden = self.model.language_model(
            inputs_embeds=inputs_embeds,
            glm5_next_packed_context=context,
            padding_mask=padding_mask,
            **kwargs,
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else bool(getattr(self.config.text_config, "output_hidden_states", False))
        )
        return compute_lm_head_logits(
            self.lm_head,
            hidden,
            logits_to_keep,
            is_thd=is_thd,
            output_hidden_states=output_hidden_states,
        )

    def prepare_model_inputs_for_cp(self, batch: dict[str, Any], *, num_chunks: int = 1) -> dict[str, Any]:
        """Install GLM's contiguous packed sharder while leaving media and ids global."""
        del batch, num_chunks
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=partial(shard_batch_for_glm5_next_cp, shard_primary=False),
                local_token_global_indices=contiguous_local_indices,
            )
        }

    def update_moe_gate_bias(self) -> None:
        """Update no-aux-loss router correction biases after an optimizer step."""
        self.model.language_model.update_moe_gate_bias()

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize all tensors for checkpoint-free construction."""
        buffer_device = buffer_device or torch.device(
            f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        )
        self.model.language_model.init_weights(buffer_device)
        self.model.visual.init_weights(buffer_device, self.config.text_config.initializer_range)
        final_std = self.config.text_config.hidden_size**-0.5
        with buffer_device:
            nn.init.trunc_normal_(self.lm_head.weight, mean=0.0, std=final_std, a=-3 * final_std, b=3 * final_std)
        cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params",))


ModelClass = Glm5NextForConditionalGeneration

__all__ = [
    "Glm5NextForConditionalGeneration",
    "Glm5NextModel",
    "Glm5NextTextModel",
    "build_glm5_next_moe_config",
]
