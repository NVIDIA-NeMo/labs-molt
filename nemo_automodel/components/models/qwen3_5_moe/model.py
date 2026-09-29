# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Qwen3.5-MoE (VL) NeMo Automodel support."""

import copy
import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.shared.import_utils import UnavailableError, UnavailableMeta


def _make_missing(name: str):
    return UnavailableMeta(name, (), {"_msg": "transformers.models.qwen3_5_moe is not available."})


try:
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeConfig,
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForConditionalGeneration as HFQwen3_5MoeForConditionalGeneration,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedDeltaNet,
        Qwen3_5MoeModelOutputWithPast,
        Qwen3_5MoeTextRotaryEmbedding,
        Qwen3_5MoeVisionRotaryEmbedding,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeModel as HFQwen3_5MoeModel,
    )

    _QWEN3_5_MOE_HF_AVAILABLE = True
except ModuleNotFoundError:
    _QWEN3_5_MOE_HF_AVAILABLE = False
    Qwen3_5MoeConfig = _make_missing("Qwen3_5MoeConfig")
    Qwen3_5MoeTextConfig = _make_missing("Qwen3_5MoeTextConfig")
    HFQwen3_5MoeForConditionalGeneration = _make_missing("Qwen3_5MoeForConditionalGeneration")
    Qwen3_5MoeGatedDeltaNet = _make_missing("Qwen3_5MoeGatedDeltaNet")
    Qwen3_5MoeModelOutputWithPast = _make_missing("Qwen3_5MoeModelOutputWithPast")
    Qwen3_5MoeTextRotaryEmbedding = _make_missing("Qwen3_5MoeTextRotaryEmbedding")
    Qwen3_5MoeVisionRotaryEmbedding = _make_missing("Qwen3_5MoeVisionRotaryEmbedding")
    HFQwen3_5MoeModel = _make_missing("Qwen3_5MoeModel")

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    contiguous_local_indices,
    round_robin_local_indices,
    shard_batch_aux_only,
    shard_sequence_for_cp_contiguous,
    shard_sequence_for_cp_round_robin,
)
from nemo_automodel.components.distributed.cp_vision_frame_shard import (
    cp_vision_frame_sharding_active,
    maybe_distribute_visual,
)
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.mtp import (
    MTPConfig,
    MTPContextParallelInputs,
    MTPModule,
    prepare_mtp_context_parallel_inputs,
    roll_tensor,
)
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype, compute_lm_head_logits
from nemo_automodel.components.models.qwen3_5.packing import (
    GatedDeltaPackedMetadata,
    prepare_gated_delta_packed_metadata,
)
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextAttention, Qwen3NextRMSNorm
from nemo_automodel.components.models.qwen3_next.model import Block
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .cp_linear_attn import CPAwareGatedDeltaNet
from .state_dict_adapter import Qwen3_5MoeStateDictAdapter


@dataclass
class Qwen3_5MoeCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Qwen3.5-MoE output extended with MTP auxiliary hidden states."""

    mtp_per_depth_h: list[torch.Tensor] | None = None
    mtp_loss_scaling_factor: float | None = None


class _Qwen3_5MoeAttention(Qwen3NextAttention):
    """Qwen3.5-MoE full attention with packed block-diagonal CP dispatch."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._base_attn_func = self.attn_func
        self.attn_func = self._dispatch_attention

    def _dispatch_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Route preprocessed QKV through packed CP or the configured backend.

        Args:
            query: Query states of shape ``[batch, heads, local_sequence, head_dim]``.
            key: Key states of shape ``[batch, kv_heads, local_sequence, head_dim]``.
            value: Value states with the same layout as ``key``.
            **attn_kwargs: Keyword arguments produced by the parent attention's
                backend preprocessing.

        Returns:
            Attention output of shape
            ``[batch, heads, local_sequence, head_dim]``.
        """
        from nemo_automodel.components.distributed.blockdiag_cp import (
            cp_blockdiag_sdpa,
            current_blockdiag_cp_state,
        )

        if current_blockdiag_cp_state() is not None:
            if self.backend.attn != "sdpa":
                raise ValueError("Qwen3.5-MoE packed context parallelism requires model.backend.attn='sdpa'")
            return cp_blockdiag_sdpa(query, key, value, **attn_kwargs)
        return self._base_attn_func(query, key, value, **attn_kwargs)


class Qwen3_5MoeBlock(Block):
    """Block that uses the Qwen3.5-MoE native GatedDeltaNet (separate in_proj_qkv,
    in_proj_z, in_proj_b, in_proj_a)"""

    def __init__(self, layer_idx, config, moe_config, backend):
        super().__init__(layer_idx, config, moe_config, backend)
        # Replace the Qwen3Next fused GatedDeltaNet with CP-aware variant
        if self.layer_type == "linear_attention":
            self.linear_attn = CPAwareGatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = _Qwen3_5MoeAttention(config, layer_idx, backend)

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        packed_gdn_metadata: GatedDeltaPackedMetadata | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Run one Qwen3.5-MoE decoder block.

        Args:
            x: Hidden states of shape [batch, sequence, hidden].
            freqs_cis: Rotary frequencies of shape [axes, batch, sequence,
                head_dim].
            attention_mask: Optional validity, indexed document, or backend mask
                of shape [batch, sequence] or [batch, 1, sequence, sequence].
            padding_mask: Optional padding mask of shape [batch, sequence].
            position_ids: Optional positions of shape [batch, sequence] or
                [axes, batch, sequence].
            packed_gdn_metadata: Optional model-forward-owned packing metadata;
                tensor layouts are documented by :class:`GatedDeltaPackedMetadata`.
            **attn_kwargs: Backend-specific attention arguments.

        Returns:
            Hidden states of shape [batch, sequence, hidden].
        """
        if self.layer_type != "linear_attention":
            attn_kwargs = dict(attn_kwargs)
            attn_kwargs.pop("seq_index", None)
            return super().forward(
                x,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                position_ids=position_ids,
                **attn_kwargs,
            )

        linear_attn_mask = attention_mask
        from nemo_automodel.components.distributed.blockdiag_cp import current_blockdiag_cp_state

        if current_blockdiag_cp_state() is not None:
            packed_gdn_metadata = None
            linear_attn_mask = None
        elif packed_gdn_metadata is None:
            packed_gdn_metadata = prepare_gated_delta_packed_metadata(
                attention_mask,
                attn_kwargs.get("_packed_seq_ids"),
            )

        if packed_gdn_metadata is not None:
            linear_attn_mask = packed_gdn_metadata.document_ids

        if linear_attn_mask is not None and padding_mask is None:
            padding_mask = linear_attn_mask.bool().logical_not()

        normed_x = self.input_layernorm(x)
        attn_out = self.linear_attn(
            hidden_states=normed_x,
            attention_mask=linear_attn_mask,
            position_ids=position_ids,
            seq_index=attn_kwargs.get("seq_index"),
            cu_seqlens=packed_gdn_metadata.cu_seqlens if packed_gdn_metadata is not None else None,
            cu_seqlens_cpu=packed_gdn_metadata.cu_seqlens_cpu if packed_gdn_metadata is not None else None,
            indices=packed_gdn_metadata.indices if packed_gdn_metadata is not None else None,
        )
        x = x + attn_out

        mlp_out = self._mlp(x=self.post_attention_layernorm(x), padding_mask=padding_mask)
        return x + mlp_out

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        if self.layer_type == "full_attention":
            self.self_attn.init_weights(buffer_device)
        elif self.layer_type == "linear_attention":
            self.linear_attn.dt_bias.data.fill_(1.0)
            self.linear_attn.A_log.data.uniform_(0, 16).log_()
            linear_list = [
                self.linear_attn.in_proj_qkv,
                self.linear_attn.in_proj_z,
                self.linear_attn.in_proj_b,
                self.linear_attn.in_proj_a,
                self.linear_attn.out_proj,
            ]
            for linear in linear_list:
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            if hasattr(self.linear_attn.norm, "reset_parameters"):
                self.linear_attn.norm.reset_parameters()
            else:
                # HF Qwen3_5MoeRMSNormGated has no reset_parameters; manually reset weight to ones
                self.linear_attn.norm.weight.data.fill_(1.0)
        self.mlp.init_weights(buffer_device)


def _resolve_mtp_num_layers(config: Any, override: int | None = None) -> int:
    if override is not None:
        return int(override)
    value = getattr(config, "num_nextn_predict_layers", None)
    if value is None:
        value = getattr(config, "mtp_num_hidden_layers", 0)
    return int(value or 0)


def _default_init_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return torch.device("cpu")


def _qwen3_5_moe_backend(backend: BackendConfig | None = None) -> BackendConfig:
    """Return a Qwen3.5-MoE backend with TE fused RoPE disabled.

    The Qwen3.5 full-attention blocks reuse Qwen3-Next attention, and VLM/packed
    execution can present THD-shaped q/k tensors. TE fused RoPE expects 4D inputs
    in this path, so use non-fused RoPE while preserving the rest of the backend.
    """
    resolved = copy.copy(backend) if backend is not None else BackendConfig()
    resolved.rope_fusion = False
    return resolved


def build_mtp_config_from_hf(
    config: Any,
    *,
    loss_scaling_factor: float = 0.1,
    num_nextn_predict_layers: int | None = None,
) -> MTPConfig:
    """Build Qwen3.5-MoE MTP runtime config from HF-style config fields."""
    num_layers = _resolve_mtp_num_layers(config, num_nextn_predict_layers)
    return MTPConfig(
        num_layers=num_layers,
        layer_pattern="*" if num_layers > 0 else "",
        loss_scaling_factor=loss_scaling_factor,
    )


def _make_mtp_block_config(config: Qwen3_5MoeTextConfig, layer_idx: int) -> Qwen3_5MoeTextConfig:
    mtp_config = copy.copy(config)
    layer_types = list(getattr(config, "layer_types", []) or [])
    if len(layer_types) <= layer_idx:
        fill = layer_types[-1] if layer_types else "full_attention"
        layer_types.extend([fill] * (layer_idx + 1 - len(layer_types)))
    layer_types[layer_idx] = "full_attention"
    mtp_config.layer_types = layer_types
    mtp_config.num_hidden_layers = max(int(getattr(config, "num_hidden_layers", 0) or 0), layer_idx + 1)
    return mtp_config


def _split_qwen3_5_moe_position_ids(
    position_ids: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    cache_position: torch.Tensor | None = None,
) -> torch.Tensor:
    if position_ids is None:
        if cache_position is None:
            cache_position = torch.arange(0, seq_len, device=device)
        position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        position_ids = position_ids[1:]
    return position_ids


def _freqs_cis_from_rotary(
    rotary_emb: nn.Module,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    cos, sin = rotary_emb(hidden_states, position_ids)
    head_dim = cos.shape[-1] // 2
    return torch.cat((cos[..., :head_dim], sin[..., :head_dim]), dim=-1)


def _rolled_embed_inputs(inputs_embeds: torch.Tensor, num_depths: int) -> tuple[torch.Tensor, ...]:
    embed_inputs = []
    cur = inputs_embeds
    for _ in range(num_depths):
        cur = roll_tensor(cur, shifts=-1, dim=-2)
        embed_inputs.append(cur)
    return tuple(embed_inputs)


def _mask_mtp_embed_inputs(
    embed_inputs: tuple[torch.Tensor, ...],
    valid_masks: tuple[torch.BoolTensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Zero invalid future-token embeddings in the local CP layout.

    Args:
        embed_inputs: Per-depth embeddings of shape ``[batch, sequence, hidden]``.
        valid_masks: Per-depth masks of shape ``[batch, sequence]`` in the same
            local CP token layout as ``embed_inputs``.

    Returns:
        Per-depth embeddings of shape ``[batch, sequence, hidden]`` with invalid
        trailing or packed-boundary positions set to zero.
    """
    if len(embed_inputs) != len(valid_masks):
        raise ValueError(
            f"MTP embed depth count {len(embed_inputs)} does not match valid-mask count {len(valid_masks)}"
        )
    masked = []
    for depth, (embed_input, valid_mask) in enumerate(zip(embed_inputs, valid_masks), start=1):
        if valid_mask.shape != embed_input.shape[:-1]:
            raise ValueError(
                f"MTP depth {depth} valid-mask shape {tuple(valid_mask.shape)} "
                f"does not match embedding token shape {tuple(embed_input.shape[:-1])}"
            )
        masked.append(embed_input.masked_fill(~valid_mask.bool().unsqueeze(-1), 0))
    return tuple(masked)


class Qwen3_5MoeMTPSublayer(Qwen3_5MoeBlock):
    """One full-attention Qwen3.5-MoE MTP sublayer."""

    def __init__(
        self,
        layer_idx: int,
        config: Qwen3_5MoeTextConfig,
        moe_config: MoEConfig,
        backend: BackendConfig,
        *,
        has_fusion: bool = False,
        has_final_norm: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(layer_idx, _make_mtp_block_config(config, layer_idx), moe_config, backend)
        self.has_fusion = has_fusion
        self.has_final_norm = has_final_norm
        if has_fusion:
            self.enorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hnorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.eh_proj = initialize_linear_module(
                backend.linear,
                2 * config.hidden_size,
                config.hidden_size,
                bias=False,
                dtype=dtype,
            )
        if has_final_norm:
            self.final_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        embed_input: torch.Tensor | None = None,
        rotary_emb: nn.Module,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if self.has_fusion:
            if embed_input is None:
                raise ValueError("first Qwen3.5-MoE MTP sublayer requires embed_input")
            e = self.enorm(embed_input)
            h = self.hnorm(hidden_states)
            hidden_states = self.eh_proj(torch.cat([e, h], dim=-1))

        freqs_cis = _freqs_cis_from_rotary(rotary_emb, hidden_states, position_ids)
        hidden_states = super().forward(
            x=hidden_states,
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            position_ids=position_ids,
            **attn_kwargs,
        )
        if self.has_final_norm:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device) -> None:
        super().init_weights(buffer_device)
        if self.has_fusion:
            self.enorm.reset_parameters()
            self.hnorm.reset_parameters()
            with buffer_device:
                nn.init.trunc_normal_(self.eh_proj.weight, mean=0.0, std=0.02)
        if self.has_final_norm:
            self.final_layernorm.reset_parameters()


def build_qwen3_5_moe_mtp(
    config: Qwen3_5MoeTextConfig,
    mtp_config: MTPConfig,
    backend: BackendConfig,
    moe_config: MoEConfig,
    dtype: torch.dtype,
) -> MTPModule:
    """Construct Qwen3.5-MoE MTP blocks."""
    base_layer_idx = int(config.num_hidden_layers)

    def factory(*, global_idx, depth, sublayer_idx, block_type, has_fusion, has_final_norm):
        del depth, sublayer_idx, block_type
        return Qwen3_5MoeMTPSublayer(
            base_layer_idx + global_idx,
            config,
            moe_config,
            backend,
            has_fusion=has_fusion,
            has_final_norm=has_final_norm,
            dtype=dtype,
        )

    return MTPModule(
        mtp_config=mtp_config,
        block_types_per_sublayer=["full_attention"],
        sublayer_factory=factory,
    )


class Fp32SafeQwen3_5MoeTextRotaryEmbedding(Qwen3_5MoeTextRotaryEmbedding):
    """Ensure inv_freq stays in float32 across ``.to(dtype)`` calls."""

    def _apply(self, fn: Any, recurse: bool = True):
        inv_freq_fp32 = self.inv_freq.detach().clone().to(torch.float32)
        result = super()._apply(fn, recurse=recurse)
        self.register_buffer(
            "inv_freq",
            inv_freq_fp32.to(device=self.inv_freq.device),
            persistent=False,
        )
        return result


class Fp32SafeQwen3_5MoeVisionRotaryEmbedding(Qwen3_5MoeVisionRotaryEmbedding):
    """Ensure the vision rotary inv_freq buffer remains float32."""

    def _apply(self, fn: Any, recurse: bool = True):
        inv_freq_fp32 = self.inv_freq.detach().clone().to(torch.float32)
        result = super()._apply(fn, recurse=recurse)
        self.register_buffer(
            "inv_freq",
            inv_freq_fp32.to(device=self.inv_freq.device),
            persistent=False,
        )
        return result


# ---------------------------------------------------------------------------
# VL composite model (wraps HF Qwen3_5MoeModel to expose backend language_model)
# ---------------------------------------------------------------------------
class Qwen3_5MoeModel(HFQwen3_5MoeModel):
    """Thin wrapper that exposes ``language_model`` internals as properties
    expected by the NeMo training loop (e.g. ``model.layers``)."""

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def embed_tokens(self):
        return self.language_model.embed_tokens

    @property
    def norm(self):
        return self.language_model.norm

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        cache_position=None,
        **kwargs,
    ):
        # If we have visual pixel values and a vision encoder, go through the full HF
        # VL forward (vision encoding + multimodal scatter + text).
        if (pixel_values is not None or pixel_values_videos is not None) and self.visual is not None:
            embed_tokens = self.get_input_embeddings()
            if inputs_embeds is None:
                if embed_tokens is not None:
                    inputs_embeds = embed_tokens(input_ids)
                elif (
                    input_ids is not None
                    and isinstance(input_ids, torch.Tensor)
                    and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32)
                ):
                    # Pipeline-parallel: input_ids may already be embeddings
                    inputs_embeds = input_ids
                    input_ids = None
                else:
                    raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")
            media_tensor = pixel_values if pixel_values is not None else pixel_values_videos
            if isinstance(media_tensor, torch.Tensor) and hasattr(self.visual, "rotary_pos_emb"):
                self.visual.rotary_pos_emb.to(media_tensor.device)
            return super().forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                **kwargs,
            )

        # Text-only path: call the NeMo backend language model directly.
        if inputs_embeds is None and (
            input_ids is not None
            and isinstance(input_ids, torch.Tensor)
            and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            # Pipeline-parallel: input_ids may already be embeddings.
            inputs_embeds = input_ids
            input_ids = None

        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        outputs = self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        return outputs


# ---------------------------------------------------------------------------
# Text decoder backend (replaces HF Qwen3_5MoeTextModel with NeMo blocks)
# ---------------------------------------------------------------------------
class Qwen3_5MoeTextModelBackend(nn.Module):
    """Qwen3.5-MoE text decoder rebuilt on top of the Qwen3-Next Block."""

    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.config = config
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")

        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        # Resolve model dtype once; thread explicitly to every sub-module so
        # fp32 master weights work even when construction is not wrapped in
        # local_torch_dtype().
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        # --------------- MoE config ---------------
        # Qwen3.5-MoE has MoE on every layer, with a shared expert + sigmoid gate.
        # No ``decoder_sparse_step`` — defaults to 1 so every layer is MoE.
        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.hidden_size,  # unused — no dense MLP layers
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.num_experts,
            n_shared_experts=1,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="softmax",
            route_scale=1.0,
            aux_loss_coeff=getattr(config, "router_aux_loss_coef", 0.001),
            norm_topk_prob=True,  # Qwen3.5-MoE always normalises topk weights
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=True,
            shared_expert_gate=True,
            shared_expert_inter_dim=config.shared_expert_intermediate_size,
            dtype=model_dtype,
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        # --------------- Layers ---------------
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=model_dtype)

        # Use Qwen3_5MoeBlock — same as Qwen3Next Block but with native GatedDeltaNet
        self.layers = nn.ModuleDict(
            {
                str(layer_id): Qwen3_5MoeBlock(layer_id, config, self.moe_config, backend)
                for layer_id in range(config.num_hidden_layers)
            }
        )

        # Use Qwen3NextRMSNorm (1+weight formula)
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # M-RoPE (interleaved) — use HF implementation, kept in fp32
        self.rotary_emb = Fp32SafeQwen3_5MoeTextRotaryEmbedding(config=config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        **attn_kwargs: Any,
    ) -> Qwen3_5MoeModelOutputWithPast:
        """Decode text tokens with model-forward-owned packed GDN metadata.

        Args:
            input_ids: Optional token IDs of shape [batch, sequence].
            inputs_embeds: Optional token embeddings of shape [batch, sequence,
                hidden].
            attention_mask: Optional validity, indexed document, or backend mask
                of shape [batch, sequence] or [batch, 1, sequence, sequence].
            position_ids: Optional positions of shape [batch, sequence] or
                [axes, batch, sequence].
            cache_position: Optional token positions of shape [sequence].
            padding_mask: Optional padding mask of shape [batch, sequence].
            past_key_values: Unsupported recurrent or KV cache.
            use_cache: Whether to use a cache; only ``False`` or ``None`` is
                supported.
            **attn_kwargs: Backend-specific attention arguments, including optional
                ``_packed_seq_ids`` of shape [batch, sequence].

        Returns:
            Model output whose ``last_hidden_state`` has shape [batch, sequence,
            hidden].
        """
        if past_key_values is not None or use_cache:
            raise NotImplementedError("KV cache is not supported for the Qwen3.5-MoE backend implementation.")

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            if self.embed_tokens is None:
                raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        # --- M-RoPE position handling (3-D: temporal / height / width) ---
        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        # Qwen3.5-MoE uses [4, bs, seq] position_ids where dim-0 is [text, T, H, W].
        # We strip the text positions (dim 0) and keep [T, H, W] for M-RoPE.
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        # When context parallelism is active the attention_mask is NOT sharded
        # along the sequence dimension (it keeps shape [B, S_global] while
        # hidden_states are [B, S_local]).  Both TE ring-attention and FLA CP
        # do not support padding masks, so we null them out.
        if getattr(self, "_cp_enabled", False):
            attention_mask = None
            from nemo_automodel.components.distributed.blockdiag_cp import current_blockdiag_cp_state

            if current_blockdiag_cp_state() is None:
                padding_mask = None

        if padding_mask is None and attention_mask is not None:
            if attention_mask.ndim <= 2:
                # 1D/2D mask (standard or indexed packing mask): invert directly
                padding_mask = attention_mask.bool().logical_not()
            else:
                # 4D mask [B, 1, S, S] (e.g. from sdpa packing collater):
                # extract per-token padding from the diagonal (a token is padded
                # if it cannot attend to itself).
                padding_mask = attention_mask[:, 0].diagonal(dim1=-2, dim2=-1).bool().logical_not()

        hidden_states = inputs_embeds

        # Compute M-RoPE (cos, sin) via HF rotary emb, then convert to freqs_cis
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        head_dim = cos.shape[-1] // 2
        freqs_cis = torch.cat((cos[..., :head_dim], sin[..., :head_dim]), dim=-1)
        packed_gdn_metadata = None
        if not getattr(self, "_cp_enabled", False):
            packed_gdn_metadata = prepare_gated_delta_packed_metadata(
                attention_mask,
                attn_kwargs.get("_packed_seq_ids"),
            )

        # --- Decoder layers (Qwen3Next Block, unmodified) ---
        for decoder_layer in self.layers.values():
            hidden_states = decoder_layer(
                x=hidden_states,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                position_ids=position_ids,
                packed_gdn_metadata=packed_gdn_metadata,
                **attn_kwargs,
            )

        if self.norm is not None:
            hidden_states = self.norm(hidden_states)

        return Qwen3_5MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            rope_deltas=None,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or _default_init_device()

        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            self.rotary_emb.device = buffer_device

        for layer in self.layers.values():
            layer.init_weights(buffer_device=buffer_device)


# ---------------------------------------------------------------------------
# Top-level text-only causal language model
# ---------------------------------------------------------------------------
class Qwen3_5MoeForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Text-only Qwen3.5-MoE causal language model."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _pp_keep_self_forward: bool = True

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True
        supports_mtp_cp: bool = True

    @classmethod
    def from_config(
        cls,
        config: Qwen3_5MoeTextConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ):
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ):
        if not _QWEN3_5_MOE_HF_AVAILABLE:
            raise UnavailableError("transformers.models.qwen3_5_moe is not available.")
        config = Qwen3_5MoeTextConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Qwen3_5MoeTextConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        mtp_loss_scaling_factor: float = 0.1,
        num_nextn_predict_layers: int | None = None,
        **kwargs: Any,
    ) -> None:
        if not _QWEN3_5_MOE_HF_AVAILABLE:
            raise UnavailableError("transformers.models.qwen3_5_moe is not available.")
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__()
        moe_overrides = kwargs.pop("moe_overrides", None)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

        self.config = config
        self.backend = _qwen3_5_moe_backend(backend)
        self.model = Qwen3_5MoeTextModelBackend(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )

        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
        )
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id if config.pad_token_id is not None else -1

        self.mtp_config = build_mtp_config_from_hf(
            config,
            loss_scaling_factor=mtp_loss_scaling_factor,
            num_nextn_predict_layers=num_nextn_predict_layers,
        )
        self.mtp = (
            build_qwen3_5_moe_mtp(config, self.mtp_config, self.backend, self.model.moe_config, dtype=dtype)
            if self.mtp_config.enabled
            else None
        )
        self.moe_config = self.model.moe_config

        keep_fp32 = list(getattr(self, "_keep_in_fp32_modules", None) or [])
        if "_fp32_params" not in keep_fp32:
            keep_fp32.append("_fp32_params")
        self._keep_in_fp32_modules = keep_fp32

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen3_5MoeStateDictAdapter(
                config,
                self.model.moe_config,
                self.backend,
                dtype=dtype,
                pretrained_model_name_or_path=getattr(config, "_name_or_path", None)
                or getattr(config, "name_or_path", None),
                text_only=True,
            )

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def prepare_mtp_inputs_for_cp(self, batch: dict[str, Any], *, ignore_index: int = -100) -> MTPContextParallelInputs:
        """Prepare text-model future-token streams before CP sharding.

        Args:
            batch: Full-sequence batch whose ``input_ids``, ``labels``, and
                optional ``position_ids`` tensors have shape ``[batch, sequence]``.
            ignore_index: Fill value for invalid targets at trailing and packed
                document-boundary positions.

        Returns:
            Global per-depth token IDs, position IDs, targets, and validity masks.
            Token tensors have shape ``[batch, sequence]``.
        """
        return prepare_mtp_context_parallel_inputs(
            batch,
            num_depths=self.mtp_config.num_layers,
            ignore_index=ignore_index,
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        mtp_per_depth_input_ids: tuple[torch.LongTensor, ...] | None = None,
        mtp_per_depth_position_ids: tuple[torch.LongTensor, ...] | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        cache_position: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Qwen3_5MoeCausalLMOutputWithPast:
        """Run the text-only causal LM forward pass.

        Args:
            input_ids: Token ids of shape ``[batch, sequence]``. On a non-first
                pipeline stage where ``embed_tokens`` is absent, this argument
                instead carries hidden states of shape
                ``[batch, sequence, hidden]`` and is routed as ``inputs_embeds``.
            attention_mask: Optional mask of shape ``[batch, sequence]`` or
                ``[batch, 1, sequence, sequence]``.
            position_ids: Optional M-RoPE positions of shape
                ``[axes, batch, sequence]`` or ``[batch, sequence]``.
            past_key_values: Optional KV-cache state; caching is not supported
                by this backend and a non-``None`` value raises.
            inputs_embeds: Optional hidden inputs of shape
                ``[batch, sequence, hidden]``.
            labels: Optional labels of shape ``[batch, sequence]``. Accepted for
                Hugging Face compatibility; loss is computed outside this model.
            mtp_per_depth_input_ids: Optional globally shifted then CP-sharded
                token IDs, one tensor of shape ``[batch, sequence]`` per depth.
            mtp_per_depth_position_ids: Optional globally shifted then CP-sharded
                positions, one tensor of shape ``[batch, sequence]`` per depth.
            use_cache: Whether to use a KV cache; ``True`` is unsupported.
            logits_to_keep: ``0`` to project every position, a positive integer
                to keep the last positions, or indices of shape
                ``[kept_sequence]``.
            output_hidden_states: Whether to include final decoder states in the
                output for the fused loss path.
            cache_position: Optional token positions of shape ``[sequence]``.
            padding_mask: Optional padding mask of shape ``[batch, sequence]``
                where ``True`` marks padding.
            **kwargs: Additional attention and packed-sequence metadata forwarded
                to the decoder and optional MTP layers.

        Returns:
            A causal-LM output whose logits have shape
            ``[batch, kept_sequence, vocab]`` (or
            ``[batch, sequence, hidden]`` on a non-final pipeline stage), whose
            requested hidden states have shape ``[batch, sequence, hidden]``,
            and whose optional per-depth MTP tensors each have shape
            ``[batch, sequence, hidden]``.
        """
        del labels
        if inputs_embeds is None and self.model.embed_tokens is None and input_ids is not None:
            inputs_embeds = input_ids
            input_ids = None

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False if use_cache is None else use_cache,
            cache_position=cache_position,
            padding_mask=padding_mask,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        lm_output = compute_lm_head_logits(
            self.lm_head,
            hidden_states,
            logits_to_keep,
            output_hidden_states=output_hidden_states,
        )

        mtp_per_depth_h: list[torch.Tensor] | None = None
        if self.mtp is not None and self.training:
            source_embeds = inputs_embeds if inputs_embeds is not None else self.model.embed_tokens(input_ids)
            mtp_position_ids = _split_qwen3_5_moe_position_ids(
                position_ids,
                batch_size=source_embeds.shape[0],
                seq_len=source_embeds.shape[1],
                device=source_embeds.device,
                cache_position=cache_position,
            )
            if mtp_per_depth_input_ids is not None:
                if mtp_per_depth_position_ids is None:
                    raise ValueError("Qwen3.5-MoE MTP per-depth token IDs require per-depth position IDs")
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    input_ids_per_depth=mtp_per_depth_input_ids,
                    embed_fn=self.model.embed_tokens,
                    position_ids_per_depth=mtp_per_depth_position_ids,
                    attention_mask=attention_mask,
                    padding_mask=padding_mask,
                    rotary_emb=self.model.rotary_emb,
                    **kwargs,
                )
            elif int(kwargs.get("cp_size", 1) or 1) > 1:
                raise ValueError("Context-parallel Qwen3.5-MoE MTP requires precomputed per-depth inputs and positions")
            elif input_ids is None:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    embed_inputs=_rolled_embed_inputs(source_embeds, self.mtp.num_depths),
                    position_ids=mtp_position_ids,
                    attention_mask=attention_mask,
                    padding_mask=padding_mask,
                    rotary_emb=self.model.rotary_emb,
                    **kwargs,
                )
            else:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    input_ids=input_ids,
                    embed_fn=self.model.embed_tokens,
                    position_ids=mtp_position_ids,
                    attention_mask=attention_mask,
                    padding_mask=padding_mask,
                    rotary_emb=self.model.rotary_emb,
                    **kwargs,
                )

        return Qwen3_5MoeCausalLMOutputWithPast(
            logits=lm_output.logits,
            hidden_states=lm_output.hidden_states,
            mtp_per_depth_h=mtp_per_depth_h,
            mtp_loss_scaling_factor=(self.mtp_config.loss_scaling_factor if mtp_per_depth_h is not None else None),
        )

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or _default_init_device()
        self.model.init_weights(buffer_device=buffer_device)
        final_out_std = self.config.hidden_size**-0.5
        cutoff_factor = 3
        with buffer_device:
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )
            if self.mtp is not None:
                for sublayer in self.mtp.layers:
                    sublayer.init_weights(buffer_device=buffer_device)
            self.model.rotary_emb.device = buffer_device
        cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params",))


# ---------------------------------------------------------------------------
# Top-level conditional generation model
# ---------------------------------------------------------------------------
class Qwen3_5MoeForConditionalGeneration(HFCheckpointingMixin, HFQwen3_5MoeForConditionalGeneration, MoEFSDPSyncMixin):
    """Qwen3.5-MoE VL conditional generation model using NeMo backend components.

    Inherits the HF model to reuse:
      * Vision encoder (``Qwen3_5MoeVisionModel``)
      * VL forward logic (image/video scatter, M-RoPE position computation)
      * ``prepare_inputs_for_generation`` / ``_expand_inputs_for_generation``

    Replaces:
      * ``model.language_model`` with ``Qwen3_5MoeTextModelBackend``
      * ``lm_head`` with NeMo backend linear
    """

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    _keep_in_fp32_modules_strict = ["_fp32_params"]
    # Packed CP uses the model-owned block-diagonal SDPA dispatch. Generic
    # hybrid CP also supports TE for unpacked sequences, but TE has no route
    # through this packed attention contract.
    _packed_cp_attn_backends = ("sdpa",)

    # forward() pulls per-microbatch pixel_values from _vlm_pixel_values_chunks;
    # patch_hf_model_for_pp must not replace it under PP.
    _pp_keep_self_forward: bool = True
    # CP submesh, installed by Qwen3_5ParallelizationStrategy when context
    # parallelism is active; None means the forward embeds and shards nothing for CP.
    cp_mesh = None

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        # CP is supported via the CP-aware GatedDeltaNet linear attention + ring
        # SDPA on the full-attention layers; the forward embeds and round-robin
        # shards its own sequence (see prepare_model_inputs_for_cp / forward).
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True
        supports_thd: bool = False
        supports_mtp_cp: bool = True
        supports_cp_vision_frame_sharding: bool = True

    @classmethod
    def from_config(
        cls,
        config: Qwen3_5MoeConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        if not _QWEN3_5_MOE_HF_AVAILABLE:
            raise UnavailableError("transformers.models.qwen3_5_moe is not available.")
        config = Qwen3_5MoeConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        mtp_loss_scaling_factor: float = 0.1,
        num_nextn_predict_layers: int | None = None,
        **kwargs,
    ):
        if not _QWEN3_5_MOE_HF_AVAILABLE:
            raise UnavailableError("transformers.models.qwen3_5_moe is not available.")
        backend = _qwen3_5_moe_backend(backend)

        # _init_model() only overrides the top-level hf_config.torch_dtype; for
        # VL configs the nested text_config / vision_config keep their original
        # dtype (typically bf16 from the checkpoint's config.json). Propagate
        # the user-requested dtype to every nested sub-config that exposes a
        # torch_dtype attribute, before constructing the HF parent (whose
        # vision encoder / multimodal code may read sub-config torch_dtype) and
        # our text backend.
        top_dtype = getattr(config, "torch_dtype", None)
        if top_dtype is not None:
            for sub_cfg in vars(config).values():
                if sub_cfg is not config and hasattr(sub_cfg, "torch_dtype"):
                    sub_cfg.torch_dtype = top_dtype

        reject_unsupported_tie_word_embeddings(type(self), config)
        # Initialize HF parent (creates self.model, self.lm_head, vision encoder, etc.)
        super().__init__(config)

        self.backend = backend

        # Swap HF model wrapper with our NeMo-aware version
        self.model.__class__ = Qwen3_5MoeModel

        # Replace HF text decoder with our NeMo backend
        text_config = config.text_config if hasattr(config, "text_config") else config
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model.language_model = Qwen3_5MoeTextModelBackend(
            text_config, backend=self.backend, moe_config=moe_config, moe_overrides=moe_overrides
        )

        # Replace lm_head with NeMo backend linear
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            text_config.hidden_size,
            text_config.vocab_size,
            bias=False,
            dtype=get_dtype(getattr(text_config, "torch_dtype", None), torch.bfloat16),
        )

        dtype = get_dtype(text_config.torch_dtype, torch.bfloat16)
        self.mtp_config = build_mtp_config_from_hf(
            text_config,
            loss_scaling_factor=mtp_loss_scaling_factor,
            num_nextn_predict_layers=num_nextn_predict_layers,
        )
        self.mtp = (
            build_qwen3_5_moe_mtp(
                text_config,
                self.mtp_config,
                self.backend,
                self.model.language_model.moe_config,
                dtype=dtype,
            )
            if self.mtp_config.enabled
            else None
        )

        # Expose moe_config for FSDP sync mixin
        self.model.moe_config = self.model.language_model.moe_config

        # Keep the SSM-gating params (A_log/dt_bias) — isolated in each
        # linear_attn ``_fp32_params`` holder at construction — in fp32 storage
        # even when the model's bulk dtype is bf16. cast_model_to_dtype() (called
        # from initialize_weights) honors this AutoModel training-storage contract.
        keep_fp32 = list(getattr(self, "_keep_in_fp32_modules", None) or [])
        if "_fp32_params" not in keep_fp32:
            keep_fp32.append("_fp32_params")
        self._keep_in_fp32_modules = keep_fp32

        self.vocab_size = text_config.vocab_size
        pad_token_id = getattr(text_config, "pad_token_id", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else -1

        # State dict adapter for checkpoint conversion
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen3_5MoeStateDictAdapter(
                text_config,
                self.model.language_model.moe_config,
                self.backend,
                dtype=dtype,
                pretrained_model_name_or_path=getattr(config, "_name_or_path", None)
                or getattr(config, "name_or_path", None),
            )

        # Wrap vision rotary embedding with fp32-safe version
        vision_model = getattr(self.model, "visual")
        rotary = vision_model.rotary_pos_emb
        dim = rotary.inv_freq.shape[0] * 2
        fp32_safe_rotary = Fp32SafeQwen3_5MoeVisionRotaryEmbedding(dim)
        fp32_safe_rotary.register_buffer(
            "inv_freq",
            rotary.inv_freq.detach().clone().to(torch.float32, copy=True),
            persistent=False,
        )
        fp32_safe_rotary.to(rotary.inv_freq.device)
        vision_model.rotary_pos_emb = fp32_safe_rotary

    def _encode_vision_for_cp(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        is_video: bool,
    ) -> torch.Tensor:
        """Encode one modality into flat, entry-ordered visual tokens under CP.

        Args:
            pixel_values: Patch rows of shape ``[total_patch_rows, patch_dim]``
                for all image or video entries, frame-contiguous in entry order.
            grid_thw: Per-entry temporal/height/width grids of shape
                ``[num_entries, 3]``.
            is_video: Whether the replicated fallback uses the video feature
                helper instead of the image feature helper.

        Returns:
            Flat merged-token embeddings of shape ``[visual_tokens, hidden]``
            in original entry order. When CP vision frame sharding is active, each
            rank computes a frame partition and the differentiable gather
            reconstructs this replicated output.
        """
        if cp_vision_frame_sharding_active():
            return maybe_distribute_visual(
                self.model.visual, pixel_values.type(self.model.visual.dtype), grid_thw
            ).pooler_output
        if is_video:
            outputs = self.model.get_video_features(pixel_values, grid_thw, return_dict=True)
        else:
            outputs = self.model.get_image_features(pixel_values, grid_thw, return_dict=True)
        return torch.cat(outputs.pooler_output, dim=0)

    def get_pipeline_kwargs_chunk_dims(self, kwargs: dict[str, Any]) -> dict[str, int]:
        """Return model-owned PP microbatch split dims for keyword inputs.

        Qwen3.5-MoE mRoPE position ids can arrive as ``[T/H/W, batch, seq]`` or
        ``[text/T/H/W, batch, seq]``. The PP schedule must split that tensor on
        the batch axis, while all standard batch-major kwargs keep the framework
        default.
        """
        position_ids = kwargs.get("position_ids")
        if isinstance(position_ids, torch.Tensor) and position_ids.ndim == 3 and position_ids.shape[0] in (3, 4):
            return {"position_ids": 1}
        return {}

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Return a sharder-only CP backend plus the full-sequence mRoPE positions.

        Embedding and the VLM->LM multimodal scatter now run inside ``forward``
        per microbatch (see :meth:`_embed_and_splice_for_cp`), so this hook only
        computes the mRoPE ``position_ids`` on the full (unsharded) sequence via
        ``get_rope_index`` and returns them with a
        :class:`ContextParallelSharder`. Packed SDPA batches select contiguous
        block-diagonal CP; ordinary batches retain the round-robin sharder.
        ``input_ids`` and media stay in the batch for the forward, while
        ``mm_token_type_ids`` is consumed here (only ``get_rope_index`` needs it).

        Args:
            batch: The full-sequence batch. Tensor fields consumed here are
                ``input_ids`` of shape ``[batch, sequence]``, optional
                ``attention_mask`` of shape ``[batch, sequence]`` or
                ``[batch, 1, sequence, sequence]``, optional ``position_ids``
                of shape ``[axes, batch, sequence]``, and image/video grids of
                shape ``[num_entries, 2 or 3]``.
            num_chunks: Accepted for hook-signature parity; unused.

        Returns:
            A mapping containing the selected ``cp_sharder``, full-sequence
            ``position_ids`` of shape ``[axes, batch, sequence]``, a consumed
            ``mm_token_type_ids=None``, and any promoted ``image_grid_thw`` of
            shape ``[num_entries, 3]``.
        """
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("Qwen3.5-MoE CP pre-embedding requires input_ids.")
        attention_mask = batch.get("attention_mask")
        position_ids = batch.get("position_ids")
        image_grid_thw = batch.get("image_grid_thw")
        image_grid_hws = batch.get("image_grid_hws")
        video_grid_thw = batch.get("video_grid_thw")
        mm_token_type_ids = batch.get("mm_token_type_ids")

        # Normalize a [N, 2] H/W grid to the [N, 3] T/H/W grid get_rope_index and
        # the forward's embed path expect; write it back so the forward reads it.
        promoted: dict[str, Any] = {}
        if image_grid_thw is None and image_grid_hws is not None and image_grid_hws.numel() > 0:
            if image_grid_hws.shape[-1] == 2:
                ones = torch.ones(
                    image_grid_hws.shape[0],
                    1,
                    dtype=image_grid_hws.dtype,
                    device=image_grid_hws.device,
                )
                image_grid_thw = torch.cat([ones, image_grid_hws], dim=-1)
            else:
                image_grid_thw = image_grid_hws
            promoted = {"image_grid_thw": image_grid_thw, "image_grid_hws": None}

        if position_ids is None:
            rope_kwargs = {
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "attention_mask": attention_mask,
            }
            if "mm_token_type_ids" in inspect.signature(self.model.get_rope_index).parameters:
                if mm_token_type_ids is None:
                    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
                    image_token_id = getattr(self.config, "image_token_id", None)
                    video_token_id = getattr(self.config, "video_token_id", None)
                    if image_token_id is not None:
                        mm_token_type_ids = mm_token_type_ids.masked_fill(input_ids == image_token_id, 1)
                    if video_token_id is not None:
                        mm_token_type_ids = mm_token_type_ids.masked_fill(input_ids == video_token_id, 2)
                rope_kwargs["mm_token_type_ids"] = mm_token_type_ids.to(device=input_ids.device)
            position_ids, rope_deltas = self.model.get_rope_index(input_ids, **rope_kwargs)
            self.model.rope_deltas = rope_deltas

        packed_seq_ids = batch.get("_packed_seq_ids")
        uses_blockdiag_cp = isinstance(packed_seq_ids, torch.Tensor) or (
            isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 4
        )
        if uses_blockdiag_cp and self.backend.attn != "sdpa":
            raise ValueError("Qwen3.5-MoE packed context parallelism requires model.backend.attn='sdpa'")
        if uses_blockdiag_cp:
            from nemo_automodel.components.distributed.blockdiag_cp import make_cp_blockdiag_batch_and_ctx

            cp_sharder = ContextParallelSharder(
                shard_batch=partial(make_cp_blockdiag_batch_and_ctx, shard_primary=False),
                local_token_global_indices=contiguous_local_indices,
            )
        else:
            cp_sharder = ContextParallelSharder(
                shard_batch=shard_batch_aux_only,
                local_token_global_indices=round_robin_local_indices,
            )

        return {
            "cp_sharder": cp_sharder,
            "position_ids": position_ids,
            "mm_token_type_ids": None,
            **promoted,
        }

    def prepare_mtp_inputs_for_cp(self, batch: dict[str, Any], *, ignore_index: int = -100) -> MTPContextParallelInputs:
        """Prepare Qwen VLM future-token streams before CP sharding.

        Call this after :meth:`prepare_model_inputs_for_cp` has materialized
        full-sequence mRoPE positions.

        Args:
            batch: Full-sequence batch whose ``input_ids`` and ``labels`` tensors
                have shape ``[batch, sequence]`` and whose ``position_ids`` tensor
                has shape ``[axes, batch, sequence]``.
            ignore_index: Fill value for invalid targets at trailing and packed
                document-boundary positions.

        Returns:
            Global per-depth token IDs, mRoPE position IDs, targets, and validity
            masks. Token tensors have shape ``[batch, sequence]`` and position
            tensors have shape ``[axes, batch, sequence]``.
        """
        return prepare_mtp_context_parallel_inputs(
            batch,
            num_depths=self.mtp_config.num_layers,
            ignore_index=ignore_index,
        )

    def _embed_and_splice_for_cp(
        self,
        input_ids: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        """Embed token ids and splice image/video features into the embeddings.

        The VLM->LM multimodal scatter runs on the full (unsharded) sequence
        inside the forward before the CP sequence shard, identical to the
        pre-CP-refactor pre-embed.

        Args:
            input_ids: Token ids ``[batch, sequence]`` (full, unsharded).
            pixel_values: Optional packed image patches for HF vision encoding.
            pixel_values_videos: Optional packed video patches.
            image_grid_thw: Per-image ``[num_images, 3]`` T/H/W grid.
            video_grid_thw: Per-video ``[num_videos, 3]`` T/H/W grid.

        Returns:
            ``inputs_embeds`` of shape ``[batch, sequence, hidden]`` with image /
            video features scattered into their placeholder positions.
        """
        inputs_embeds = self.get_input_embeddings()(input_ids)

        # The HF visual tower's bidirectional attention is not CP-sharded; when this
        # splice runs in-forward under an active CP ring context it must suspend the
        # ring dispatcher, or torch's load-balanced ring SDPA all-gathers the vision
        # Q/K/V and rejects the non-causal attention. No-op when CP is inactive.
        from nemo_automodel.components.distributed.context_parallel.utils import (
            cp_dispatcher_suspended,  # noqa: PLC0415
        )

        with cp_dispatcher_suspended(self.cp_mesh):
            if pixel_values is not None:
                if hasattr(self.model.visual, "rotary_pos_emb"):
                    self.model.visual.rotary_pos_emb.to(pixel_values.device)
                image_embeds = self._encode_vision_for_cp(pixel_values, image_grid_thw, is_video=False).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                image_mask, _ = self.model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                if hasattr(self.model.visual, "rotary_pos_emb"):
                    self.model.visual.rotary_pos_emb.to(pixel_values_videos.device)
                video_embeds = self._encode_vision_for_cp(pixel_values_videos, video_grid_thw, is_video=True).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                _, video_mask = self.model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    video_features=video_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        return inputs_embeds

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Per-stage input/output meta tensors for the PP schedule's shape inference.

        Matches the framework default except that under context parallelism the
        first stage embeds the full sequence and shards it in forward, so every
        stage output and later-stage input carries the LOCAL (padded-to-``2*cp``
        then ``//cp``) sequence length while the first-stage input stays
        full-length. At ``cp_size == 1`` this reduces to the default symmetric
        shapes.
        """
        text_config = getattr(self.config, "text_config", self.config)
        hidden_size = text_config.hidden_size
        vocab_size = text_config.vocab_size

        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        local_seq_len = seq_len
        if cp_size > 1:
            padded_seq_len = seq_len + (-seq_len) % (2 * cp_size)
            local_seq_len = padded_seq_len // cp_size

        if is_first:
            inputs_meta = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        else:
            inputs_meta = (torch.empty(microbatch_size, local_seq_len, hidden_size, device="meta", dtype=dtype),)

        has_lm_head = getattr(self, "lm_head", None) is not None
        emits_hidden_states = getattr(self, "_pp_return_hidden_states", False) is True
        if has_lm_head and not emits_hidden_states:
            outputs_meta = (torch.empty(microbatch_size, local_seq_len, vocab_size, device="meta", dtype=dtype),)
        else:
            outputs_meta = (torch.empty(microbatch_size, local_seq_len, hidden_size, device="meta", dtype=dtype),)
        return inputs_meta, outputs_meta

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        mtp_per_depth_input_ids: tuple[torch.LongTensor, ...] | None = None,
        mtp_per_depth_position_ids: tuple[torch.LongTensor, ...] | None = None,
        mtp_per_depth_valid_masks: tuple[torch.BoolTensor, ...] | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: bool | None = None,
        **kwargs: Any,
    ):
        """Run Qwen3.5-MoE VLM generation with model-owned CP embedding.

        Args:
            input_ids: Token ids of shape ``[batch, sequence]``. Under CP this
                remains full-length until multimodal features are spliced.
            position_ids: Optional mRoPE positions of shape
                ``[axes, batch, sequence]``; under CP, ``sequence`` is local.
            attention_mask: Optional mask of shape ``[batch, sequence]`` or
                ``[batch, 1, sequence, sequence]``.
            padding_mask: Optional local padding mask of shape
                ``[batch, sequence]`` where ``True`` marks padding.
            inputs_embeds: Optional embeddings of shape
                ``[batch, sequence, hidden]``.
            cache_position: Optional token positions of shape ``[sequence]``.
            mtp_per_depth_input_ids: Optional globally shifted then CP-sharded
                token IDs, one tensor of shape ``[batch, sequence]`` per depth.
            mtp_per_depth_position_ids: Optional globally shifted then CP-sharded
                mRoPE positions, one tensor of shape
                ``[axes, batch, sequence]`` per depth.
            mtp_per_depth_valid_masks: Optional CP-sharded masks, one tensor of
                shape ``[batch, sequence]`` per depth, where ``True`` marks a
                valid same-document future token.
            logits_to_keep: Number of trailing logits to retain, or indices of
                shape ``[kept_sequence]``.
            output_hidden_states: Whether to expose final decoder states for the
                fused loss path.
            **kwargs: Model inputs including image/video patch rows of shape
                ``[total_patch_rows, patch_dim]``, their grids of shape
                ``[num_entries, 3]``, and optional packed document ids of shape
                ``[batch, sequence]``.

        Returns:
            A causal-LM output whose logits have shape
            ``[batch, sequence, vocab]`` (or ``[batch, sequence, hidden]`` on a
            non-final pipeline stage), whose requested hidden states have shape
            ``[batch, sequence, hidden]``, and whose optional MTP tensors each
            have shape ``[batch, sequence, hidden]``. Under CP, ``sequence`` is
            the local padded shard.
        """
        # Resolve from the text/decoder sub-config for this VL model.
        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(text_config, "output_hidden_states", False)
        )

        # PP VLM support: retrieve pixel_values from stored chunks if not passed
        pixel_values = kwargs.get("pixel_values", None)
        pixel_values_videos = kwargs.get("pixel_values_videos", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        image_token_id = self.config.image_token_id
        vision_start_token_id = self.config.vision_start_token_id
        has_media_tokens = input_ids is not None and (
            (input_ids == image_token_id).any() or (input_ids == vision_start_token_id).any()
        )

        chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
        consumed_vlm_chunk = False

        if pixel_values is None and has_media_tokens:
            image_chunks = getattr(self, "_vlm_pixel_values_chunks", None)
            if image_chunks is not None and chunk_idx < len(image_chunks):
                pixel_values = image_chunks[chunk_idx]
                image_grid_chunks = getattr(self, "_vlm_image_grid_hws_chunks", None)
                if image_grid_chunks is not None and chunk_idx < len(image_grid_chunks):
                    image_grid_hws = image_grid_chunks[chunk_idx]
                    if image_grid_hws is not None and image_grid_hws.numel() > 0:
                        if image_grid_hws.shape[-1] == 2:
                            ones = torch.ones(
                                image_grid_hws.shape[0], 1, dtype=image_grid_hws.dtype, device=image_grid_hws.device
                            )
                            image_grid_thw = torch.cat([ones, image_grid_hws], dim=-1)
                        else:
                            image_grid_thw = image_grid_hws
                kwargs["pixel_values"] = pixel_values
                kwargs["image_grid_thw"] = image_grid_thw
                consumed_vlm_chunk = True

        if pixel_values_videos is None and has_media_tokens:
            video_chunks = getattr(self, "_vlm_pixel_values_videos_chunks", None)
            if video_chunks is not None and chunk_idx < len(video_chunks):
                video_chunk = video_chunks[chunk_idx]
                if video_chunk.numel() > 0:
                    pixel_values_videos = video_chunk
                    video_grid_chunks = getattr(self, "_vlm_video_grid_thw_chunks", None)
                    if video_grid_chunks is not None and chunk_idx < len(video_grid_chunks):
                        video_grid_thw = video_grid_chunks[chunk_idx]
                    kwargs["pixel_values_videos"] = pixel_values_videos
                    kwargs["video_grid_thw"] = video_grid_thw
                consumed_vlm_chunk = True

        if consumed_vlm_chunk:
            self._vlm_chunk_idx = chunk_idx + 1

        if "qkv_format" in kwargs and kwargs["qkv_format"] == "thd":
            input_ids, position_ids, padding_mask, kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, kwargs
            )
            attention_mask = None

        # Context-parallel: embed + vision-splice the full sequence, then select
        # the packed contiguous shard or the ordinary round-robin chunk pair.
        # Auxiliary streams and mRoPE positions already use the matching layout.
        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        mtp_embed_inputs = None
        if (
            cp_size <= 1
            and self.mtp is not None
            and self.training
            and inputs_embeds is None
            and input_ids is not None
            and not torch.is_floating_point(input_ids)
            and has_media_tokens
            and (
                (isinstance(pixel_values, torch.Tensor) and pixel_values.numel() > 0)
                or (isinstance(pixel_values_videos, torch.Tensor) and pixel_values_videos.numel() > 0)
            )
        ):
            # MTP predicts from the same fused text/vision embedding stream as
            # the backbone. Re-embedding media placeholder IDs would silently
            # discard the continuous image/video features on the CP1 reference.
            inputs_embeds = self._embed_and_splice_for_cp(
                input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
            input_ids = None
            for media_key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
                kwargs.pop(media_key, None)
        if cp_size > 1 and inputs_embeds is None and input_ids is not None and not torch.is_floating_point(input_ids):
            is_first_stage = getattr(self.model.language_model, "embed_tokens", None) is not None
            is_last_stage = getattr(self, "lm_head", None) is not None
            if is_first_stage:
                if not is_last_stage and (pixel_values is not None or pixel_values_videos is not None):
                    raise NotImplementedError(
                        "Qwen3.5-MoE does not support image/video microbatch chunking under combined "
                        "pipeline + context parallelism; use a text-only batch for cp>1 and pp>1."
                    )
                inputs_embeds = self._embed_and_splice_for_cp(
                    input_ids,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                )
                from nemo_automodel.components.distributed.blockdiag_cp import current_blockdiag_cp_state

                uses_blockdiag_cp = current_blockdiag_cp_state() is not None
                if self.mtp is not None and self.training:
                    if (
                        mtp_per_depth_input_ids is None
                        or mtp_per_depth_position_ids is None
                        or mtp_per_depth_valid_masks is None
                    ):
                        raise ValueError(
                            "Context-parallel Qwen3.5-MoE MTP requires precomputed per-depth inputs, positions, and masks"
                        )
                    rolled_embed_inputs = _rolled_embed_inputs(inputs_embeds, self.mtp.num_depths)
                    if uses_blockdiag_cp:
                        mtp_embed_inputs = tuple(
                            shard_sequence_for_cp_contiguous(self.cp_mesh, embed_input, seq_dim=1)[0]
                            for embed_input in rolled_embed_inputs
                        )
                    else:
                        mtp_embed_inputs = tuple(
                            shard_sequence_for_cp_round_robin(self.cp_mesh, embed_input, seq_dim=1)[0]
                            for embed_input in rolled_embed_inputs
                        )
                    mtp_embed_inputs = _mask_mtp_embed_inputs(mtp_embed_inputs, mtp_per_depth_valid_masks)

                if uses_blockdiag_cp:
                    inputs_embeds, _, _ = shard_sequence_for_cp_contiguous(self.cp_mesh, inputs_embeds, seq_dim=1)
                else:
                    inputs_embeds, _, _ = shard_sequence_for_cp_round_robin(self.cp_mesh, inputs_embeds, seq_dim=1)
                input_ids = None
                # The media was consumed into inputs_embeds; drop it so self.model
                # does not re-splice into the already-sharded embeddings.
                for media_key in ("pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"):
                    kwargs.pop(media_key, None)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        lm_output = compute_lm_head_logits(
            self.lm_head, hidden_states, logits_to_keep, output_hidden_states=output_hidden_states
        )

        mtp_per_depth_h: list[torch.Tensor] | None = None
        if self.mtp is not None and self.training:
            language_model = self.model.language_model
            source_embeds = inputs_embeds if inputs_embeds is not None else language_model.embed_tokens(input_ids)
            mtp_position_ids = _split_qwen3_5_moe_position_ids(
                position_ids,
                batch_size=source_embeds.shape[0],
                seq_len=source_embeds.shape[1],
                device=source_embeds.device,
                cache_position=cache_position,
            )
            mtp_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key
                not in {
                    "pixel_values",
                    "pixel_values_videos",
                    "image_grid_thw",
                    "video_grid_thw",
                    "mm_token_type_ids",
                }
            }
            if mtp_embed_inputs is not None:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    embed_inputs=mtp_embed_inputs,
                    position_ids_per_depth=mtp_per_depth_position_ids,
                    attention_mask=attention_mask,
                    padding_mask=padding_mask,
                    rotary_emb=language_model.rotary_emb,
                    **mtp_kwargs,
                )
            elif cp_size > 1:
                raise ValueError("Context-parallel Qwen3.5-MoE MTP requires globally prepared per-depth inputs")
            elif input_ids is None:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    embed_inputs=_rolled_embed_inputs(source_embeds, self.mtp.num_depths),
                    position_ids=mtp_position_ids,
                    attention_mask=attention_mask,
                    padding_mask=padding_mask,
                    rotary_emb=language_model.rotary_emb,
                    **mtp_kwargs,
                )
            else:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    input_ids=input_ids,
                    embed_fn=language_model.embed_tokens,
                    position_ids=mtp_position_ids,
                    attention_mask=attention_mask,
                    padding_mask=padding_mask,
                    rotary_emb=language_model.rotary_emb,
                    **mtp_kwargs,
                )
            return Qwen3_5MoeCausalLMOutputWithPast(
                logits=lm_output.logits,
                hidden_states=lm_output.hidden_states,
                mtp_per_depth_h=mtp_per_depth_h,
                mtp_loss_scaling_factor=self.mtp_config.loss_scaling_factor,
            )

        return lm_output

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        buffer_device = buffer_device or _default_init_device()
        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config

        with buffer_device:
            language_model = self.model.language_model
            try:
                language_model.init_weights(buffer_device=buffer_device)
            except TypeError:
                language_model.init_weights()
            final_out_std = text_config.hidden_size**-0.5
            cutoff_factor = 3
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )
            mtp = getattr(self, "mtp", None)
            if mtp is not None:
                for sublayer in mtp.layers:
                    sublayer.init_weights(buffer_device=buffer_device)

        # Skip the SSM-gating holders so they keep fp32 storage (master weights):
        # cast_model_to_dtype cannot reliably restore fp32 once FSDP2-sharded, so it
        # detaches them and never casts them. The MoE parallelizer's dtype-aware
        # sharder places each holder in its own fp32 FSDP group.
        cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params",))

        with buffer_device:
            self.model.language_model.rotary_emb.device = buffer_device


if _QWEN3_5_MOE_HF_AVAILABLE:
    ModelClass = Qwen3_5MoeForConditionalGeneration
