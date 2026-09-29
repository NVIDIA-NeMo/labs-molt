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

"""Qwen3.5 dense causal LM with Megatron-style MTP support."""

from __future__ import annotations

import copy
import inspect
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    create_causal_mask,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration as HFQwen3_5ForConditionalGeneration,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Model as HFQwen3_5Model,
)

from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    round_robin_local_indices,
    shard_batch_aux_only,
    shard_sequence_for_cp_round_robin,
)
from nemo_automodel.components.distributed.cp_vision_frame_shard import (
    cp_vision_frame_sharding_active,
    maybe_distribute_visual,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.mtp import MTPConfig, MTPModule, roll_tensor
from nemo_automodel.components.models.common.packing import is_indexed_packed_mask
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.qwen3_5.packing import (
    GatedDeltaPackedMetadata,
    prepare_gated_delta_packed_metadata,
)
from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet
from nemo_automodel.components.models.qwen3_next.layers import Qwen3NextRMSNorm
from nemo_automodel.components.models.qwen3_next.model import Block
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .state_dict_adapter import Qwen3_5DenseStateDictAdapter


@dataclass
class Qwen3_5CausalLMOutputWithPast(CausalLMOutputWithPast):
    """Qwen3.5 causal-LM output extended with MTP auxiliary hidden states."""

    rope_deltas: torch.Tensor | None = None
    mtp_per_depth_h: list[torch.Tensor] | None = None
    mtp_loss_scaling_factor: float | None = None


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


def _qwen3_5_backend(backend: BackendConfig | None = None) -> BackendConfig:
    """Return a Qwen3.5 backend with TE fused RoPE disabled.

    Qwen3.5 VLM training can feed full-attention layers in packed/THD shape via
    the shared Qwen3-Next attention block. TE fused RoPE expects 4D inputs there,
    so keep the non-fused RoPE path while preserving the rest of the backend
    selection (TE Linear, attention backend, etc.).
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
    """Build Qwen3.5 MTP runtime config from HF-style config fields."""
    num_layers = _resolve_mtp_num_layers(config, num_nextn_predict_layers)
    return MTPConfig(
        num_layers=num_layers,
        layer_pattern="*" if num_layers > 0 else "",
        loss_scaling_factor=loss_scaling_factor,
    )


def _make_full_attention_config(config: Qwen3_5TextConfig, layer_idx: int) -> Qwen3_5TextConfig:
    mtp_config = copy.copy(config)
    layer_types = list(getattr(config, "layer_types", []) or [])
    if len(layer_types) <= layer_idx:
        fill = layer_types[-1] if layer_types else "full_attention"
        layer_types.extend([fill] * (layer_idx + 1 - len(layer_types)))
    layer_types[layer_idx] = "full_attention"
    mtp_config.layer_types = layer_types
    mtp_config.num_hidden_layers = max(int(getattr(config, "num_hidden_layers", 0) or 0), layer_idx + 1)
    # The MTP sublayer re-enters the HF Qwen3.5 decoder self-attention over the
    # SAME packed/varlen batch as the backbone. The HF flash-attention-2 varlen
    # path cannot view the batched [B, S, H, D] MTP query against the packed
    # cu_seqlens (RuntimeError: shape '[B, ...]' is invalid ...), so route the
    # MTP self-attention through SDPA -- which consumes a 4D block-causal mask
    # exactly like the native backbone -- instead of FA2. See NVBugs 6330129.
    mtp_config._attn_implementation = "sdpa"
    return mtp_config


def _split_qwen3_5_position_ids(
    position_ids: torch.Tensor | None,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    past_key_values: Any | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        base = torch.arange(seq_len, device=device) + past_seen_tokens
        position_ids = base.view(1, 1, -1).expand(4, batch_size, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        return position_ids[1:], position_ids[0]
    return position_ids, None


def _mtp_block_causal_mask(packing_mask: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
    """Build a 4D block-causal attention mask from an indexed packing mask.

    ``packing_mask`` is ``[B, S]`` with the 1-based document index per token
    (0 = padding). The returned bool mask ``[B, 1, S, S]`` (``True`` = attend)
    keeps attention causal *and* within each packed document, matching the
    backbone's packed-sequence semantics. Used for the MTP sublayers, which run
    SDPA self-attention over the same packed batch (NVBugs 6330129).
    """
    mask = packing_mask.to(device=inputs_embeds.device)
    seq_len = mask.shape[-1]
    same_doc = mask.unsqueeze(2) == mask.unsqueeze(1)  # [B, S, S]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=mask.device).tril()
    not_padding = (mask > 0).unsqueeze(2) & (mask > 0).unsqueeze(1)
    return (same_doc & causal.unsqueeze(0) & not_padding).unsqueeze(1)


def _rolled_embed_inputs(inputs_embeds: torch.Tensor, num_depths: int) -> tuple[torch.Tensor, ...]:
    embed_inputs = []
    cur = inputs_embeds
    for _ in range(num_depths):
        cur = roll_tensor(cur, shifts=-1, dim=-2)
        embed_inputs.append(cur)
    return tuple(embed_inputs)


class Qwen3_5DenseMTPSublayer(Qwen3_5DecoderLayer):
    """One full-attention Qwen3.5 dense MTP sublayer."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_idx: int,
        *,
        has_fusion: bool = False,
        has_final_norm: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__(_make_full_attention_config(config, layer_idx), layer_idx)
        # Transformers 5.15 renamed this discriminator to ``block_type``.
        self.layer_type = self.block_type
        self.has_fusion = has_fusion
        self.has_final_norm = has_final_norm
        if has_fusion:
            self.enorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.hnorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.eh_proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False, dtype=dtype)
        if has_final_norm:
            self.final_layernorm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        embed_input: torch.Tensor | None = None,
        rotary_emb: nn.Module,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        if self.has_fusion:
            if embed_input is None:
                raise ValueError("first Qwen3.5 MTP sublayer requires embed_input")
            e = self.enorm(embed_input)
            h = self.hnorm(hidden_states)
            hidden_states = self.eh_proj(torch.cat([e, h], dim=-1))

        if position_ids is None:
            seq_len = hidden_states.shape[-2]
            base = torch.arange(seq_len, device=hidden_states.device).view(1, -1)
            position_ids = base.expand(hidden_states.shape[0], -1)

        position_embeddings = rotary_emb(hidden_states, position_ids)
        hidden_states = super().forward(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids[0] if position_ids.ndim == 3 and position_ids.shape[0] == 4 else None,
            past_key_values=past_key_values,
            **kwargs,
        )
        if self.has_final_norm:
            hidden_states = self.final_layernorm(hidden_states)
        return hidden_states

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        init_std = float(getattr(self.self_attn.config, "initializer_range", 0.02))
        target_device = buffer_device or torch.device("cpu")
        with target_device:
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, Qwen3_5RMSNorm):
                    nn.init.zeros_(module.weight)


def build_qwen3_5_dense_mtp(
    config: Qwen3_5TextConfig,
    mtp_config: MTPConfig,
    dtype: torch.dtype,
) -> MTPModule:
    """Construct dense Qwen3.5 MTP blocks.

    Qwen3.5 MTP follows Megatron Bridge: each depth is one full-attention
    Qwen3.5 decoder block, regardless of the backbone's GatedDeltaNet layers.
    """
    base_layer_idx = int(config.num_hidden_layers)

    def factory(*, global_idx, depth, sublayer_idx, block_type, has_fusion, has_final_norm):
        del depth, sublayer_idx, block_type
        return Qwen3_5DenseMTPSublayer(
            config,
            layer_idx=base_layer_idx + global_idx,
            has_fusion=has_fusion,
            has_final_norm=has_final_norm,
            dtype=dtype,
        )

    return MTPModule(
        mtp_config=mtp_config,
        block_types_per_sublayer=["full_attention"],
        sublayer_factory=factory,
    )


class Fp32SafeQwen3_5TextRotaryEmbedding(Qwen3_5TextRotaryEmbedding):
    """Ensure inv_freq stays in float32 across ``.to(dtype)`` calls."""

    def _apply(self, fn: Any, recurse: bool = True):
        inv_freq_fp32 = self.inv_freq.detach().clone().to(torch.float32)
        result = super()._apply(fn, recurse=recurse)
        self.register_buffer("inv_freq", inv_freq_fp32.to(device=self.inv_freq.device), persistent=False)
        return result


def _dense_moe_config(config: Qwen3_5TextConfig, dtype: torch.dtype) -> MoEConfig:
    """Trivial MoEConfig for the dense Qwen3.5 backbone.

    The dense model has no experts (``num_experts`` is 0/absent), so ``Block``
    builds a dense ``MLP`` and never consults this config; it is only required to
    satisfy ``Block.__init__``'s signature.
    """
    inter = config.intermediate_size
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=inter,
        moe_inter_dim=inter,
        n_routed_experts=0,
        n_shared_experts=0,
        n_activated_experts=0,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        norm_topk_prob=True,
        dtype=dtype,
    )


class Qwen3_5DenseBlock(Block):
    """Qwen3.5 dense decoder block on top of the Qwen3-Next ``Block``.

    Identical to ``Qwen3_5MoeBlock`` except the MLP degrades to a dense ``MLP``
    (no experts). The CP-aware GatedDeltaNet is built natively for
    linear-attention layers, and the forward threads NEAT-packing kwargs.
    """

    def __init__(self, layer_idx, config, moe_config, backend):
        super().__init__(layer_idx, config, moe_config, backend)
        if self.layer_type == "linear_attention":
            self.linear_attn = CPAwareGatedDeltaNet(config, layer_idx)

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
        """Run one dense Qwen3.5 decoder block.

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
            for linear in (
                self.linear_attn.in_proj_qkv,
                self.linear_attn.in_proj_z,
                self.linear_attn.in_proj_b,
                self.linear_attn.in_proj_a,
                self.linear_attn.out_proj,
            ):
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            if hasattr(self.linear_attn.norm, "reset_parameters"):
                self.linear_attn.norm.reset_parameters()
            else:
                self.linear_attn.norm.weight.data.fill_(1.0)
        self.mlp.init_weights(buffer_device)


class Qwen3_5DenseTextBackbone(nn.Module):
    """Qwen3.5 dense text decoder rebuilt on the Qwen3-Next ``Block``.

    Native counterpart of ``Qwen3_5MoeTextModelBackend`` for the dense model:
    reuses the same blocks/GatedDeltaNet/norm/rotary so dense and MoE share one
    code path, with the fp32 ``SSMGate`` built at construction (no runtime patch).
    """

    def __init__(self, config: Qwen3_5TextConfig, backend: BackendConfig):
        super().__init__()
        self.config = config
        self.backend = backend
        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        moe_config = _dense_moe_config(config, model_dtype)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, dtype=model_dtype)
        self.layers = nn.ModuleDict(
            {str(i): Qwen3_5DenseBlock(i, config, moe_config, backend) for i in range(config.num_hidden_layers)}
        )
        self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Fp32SafeQwen3_5TextRotaryEmbedding(config=config)

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
        output_hidden_states: bool | None = None,
        **attn_kwargs: Any,
    ) -> BaseModelOutputWithPast:
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
            output_hidden_states: Accepted for Hugging Face compatibility and
                ignored.
            **attn_kwargs: Backend-specific attention arguments, including optional
                ``_packed_seq_ids`` of shape [batch, sequence].

        Returns:
            Model output whose ``last_hidden_state`` has shape [batch, sequence,
            hidden].
        """
        del output_hidden_states  # accepted for HF-forward compatibility; ignored
        if past_key_values is not None or use_cache:
            raise NotImplementedError("KV cache is not supported for the Qwen3.5 dense backend implementation.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        # [4, bs, seq] position_ids (dim-0 = [text, T, H, W]); keep [T, H, W] for M-RoPE.
        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        if getattr(self, "_cp_enabled", False):
            attention_mask = None
            padding_mask = None

        if padding_mask is None and attention_mask is not None:
            if attention_mask.ndim <= 2:
                padding_mask = attention_mask.bool().logical_not()
            else:
                padding_mask = attention_mask[:, 0].diagonal(dim1=-2, dim2=-1).bool().logical_not()

        hidden_states = inputs_embeds
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        head_dim = cos.shape[-1] // 2
        freqs_cis = torch.cat((cos[..., :head_dim], sin[..., :head_dim]), dim=-1)
        packed_gdn_metadata = None
        if not getattr(self, "_cp_enabled", False):
            packed_gdn_metadata = prepare_gated_delta_packed_metadata(
                attention_mask,
                attn_kwargs.get("_packed_seq_ids"),
            )

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
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None)

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


class Qwen3_5Model(HFQwen3_5Model):
    """Thin VLM wrapper exposing ``language_model`` internals as properties and
    routing the forward: HF vision+scatter path when media is present, else the
    NeMo dense backbone directly. Mirrors ``Qwen3_5MoeModel``."""

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
        # Media present + vision encoder: full HF VL forward (vision encode +
        # multimodal scatter), which then calls self.language_model (NeMo backbone).
        if (pixel_values is not None or pixel_values_videos is not None) and self.visual is not None:
            embed_tokens = self.get_input_embeddings()
            input_ids_for_super = input_ids
            inputs_embeds_for_super = inputs_embeds
            if inputs_embeds_for_super is None:
                if input_ids is not None and isinstance(input_ids, torch.Tensor) and torch.is_floating_point(input_ids):
                    inputs_embeds_for_super = input_ids
                    input_ids_for_super = None
                elif embed_tokens is None:
                    raise ValueError("inputs_embeds must be provided for pipeline stages without embed_tokens")
            else:
                input_ids_for_super = None
            media_tensor = pixel_values if pixel_values is not None else pixel_values_videos
            if isinstance(media_tensor, torch.Tensor) and hasattr(self.visual, "rotary_pos_emb"):
                self.visual.rotary_pos_emb.to(media_tensor.device)
            return super().forward(
                input_ids=input_ids_for_super,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds_for_super,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                **kwargs,
            )

        # Text-only path: call the NeMo backend directly.
        if (
            inputs_embeds is None
            and input_ids is not None
            and isinstance(input_ids, torch.Tensor)
            and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            inputs_embeds = input_ids
            input_ids = None
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        return self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )


class Qwen3_5ForCausalLM(HFCheckpointingMixin, nn.Module):
    """Qwen3.5 dense causal LM with optional Megatron-style MTP head."""

    tie_word_embeddings_support: TieSupport = TieSupport.BOTH
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = True
        supports_cp: bool = False
        supports_pp: bool = True
        supports_ep: bool = False

    @classmethod
    def from_config(
        cls,
        config: Qwen3_5TextConfig,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ):
        return cls(config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ):
        config = Qwen3_5TextConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        backend: BackendConfig | None = None,
        *,
        mtp_loss_scaling_factor: float = 0.1,
        num_nextn_predict_layers: int | None = None,
        **kwargs: Any,
    ) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        super().__init__()
        del kwargs
        self.config = config
        self.backend = _qwen3_5_backend(backend)

        self.model = Qwen3_5DenseTextBackbone(config, self.backend)
        dtype = next(self.model.parameters()).dtype
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)
        if getattr(config, "tie_word_embeddings", False):
            self.tie_weights()

        # Keep the SSM-gating params (in each linear_attn ``_fp32_params`` holder)
        # in fp32 storage even under a bf16 bulk dtype.
        keep_fp32 = list(getattr(self, "_keep_in_fp32_modules", None) or [])
        if "_fp32_params" not in keep_fp32:
            keep_fp32.append("_fp32_params")
        self._keep_in_fp32_modules = keep_fp32

        self.mtp_config = build_mtp_config_from_hf(
            config,
            loss_scaling_factor=mtp_loss_scaling_factor,
            num_nextn_predict_layers=num_nextn_predict_layers,
        )
        self.mtp = build_qwen3_5_dense_mtp(config, self.mtp_config, dtype=dtype) if self.mtp_config.enabled else None

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen3_5DenseStateDictAdapter(route_linear_attn_fp32_params=True)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def tie_weights(self, *_args: object, **_kwargs: object) -> None:
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs: Any,
    ) -> Qwen3_5CausalLMOutputWithPast:
        del labels
        effective_use_cache = False if use_cache is None else use_cache
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=effective_use_cache,
            output_hidden_states=True,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        mtp_per_depth_h: list[torch.Tensor] | None = None
        if self.mtp is not None and self.training:
            source_embeds = inputs_embeds if inputs_embeds is not None else self.model.embed_tokens(input_ids)
            rotary_position_ids, text_position_ids = _split_qwen3_5_position_ids(
                position_ids,
                batch_size=source_embeds.shape[0],
                seq_len=source_embeds.shape[1],
                device=source_embeds.device,
                past_key_values=past_key_values,
            )
            # For packed sequences (indexed mask), feed the MTP SDPA sublayers a
            # 4D block-causal mask so attention stays within each document, the
            # same way the backbone treats packing. Otherwise use the standard
            # causal mask. See NVBugs 6330129.
            packing_mask = attention_mask if is_indexed_packed_mask(attention_mask) else kwargs.get("_packed_seq_ids")
            if is_indexed_packed_mask(packing_mask):
                causal_mask = _mtp_block_causal_mask(packing_mask, source_embeds)
            else:
                causal_mask = create_causal_mask(
                    config=self.model.config,
                    inputs_embeds=source_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                )
            if input_ids is None:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    embed_inputs=_rolled_embed_inputs(source_embeds, self.mtp.num_depths),
                    position_ids=rotary_position_ids,
                    attention_mask=causal_mask,
                    rotary_emb=self.model.rotary_emb,
                    **kwargs,
                )
            else:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    input_ids=input_ids,
                    embed_fn=self.model.embed_tokens,
                    position_ids=rotary_position_ids,
                    attention_mask=causal_mask,
                    rotary_emb=self.model.rotary_emb,
                    **kwargs,
                )

        return Qwen3_5CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None) or (hidden_states,),
            attentions=getattr(outputs, "attentions", None),
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
        init_std = float(getattr(self.config, "initializer_range", 0.02))
        # The backbone (embed/norm/layers, incl. GatedDeltaNet-specific init) owns
        # its own init_weights; init only the non-backbone modules (lm_head, MTP)
        # generically so the GatedDeltaNet/SSMGate init is not clobbered.
        self.model.init_weights(buffer_device=buffer_device)
        with buffer_device:
            for name, module in self.named_modules():
                if name == "model" or name.startswith("model."):
                    continue
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, mean=0.0, std=init_std)
                    if module.padding_idx is not None:
                        module.weight[module.padding_idx].zero_()
                elif isinstance(module, (Qwen3_5RMSNorm, Qwen3NextRMSNorm)):
                    nn.init.zeros_(module.weight)
        cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params",))


class Qwen3_5ForConditionalGeneration(HFCheckpointingMixin, HFQwen3_5ForConditionalGeneration):
    """Qwen3.5/Qwen3.6 dense VLM with optional Megatron-style MTP head.

    The base VLM stays on the upstream HF implementation so image/video feature
    insertion, M-RoPE position handling, and generation helpers remain intact.
    MTP is added as an auxiliary train-time module over the final language
    hidden states, matching the dense text-only MTP architecture.
    """

    # forward() pulls per-microbatch pixel_values from _vlm_pixel_values_chunks;
    # patch_hf_model_for_pp must not replace it under PP.
    _pp_keep_self_forward: bool = True
    # CP submesh, installed by the parallelizer's apply_cp when context parallelism
    # is active; None means the forward embeds and shards nothing for CP.
    cp_mesh = None

    tie_word_embeddings_support: TieSupport = TieSupport.BOTH
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}

    # ``cast_model_to_dtype`` snapshots/restores matched fp32 buffers around its
    # bulk cast, and ``cast_frozen_modules_to_compute_dtype`` exempts matched
    # names, so this keeps the vision tower's rotary ``inv_freq`` buffer at its
    # exact fp32 values under shared bf16/fp16 casts (including the frozen-vision
    # recipes' cast). Class-level: ``super().__init__()`` runs ``post_init`` ->
    # ``initialize_weights`` -> the first cast before the constructor body.
    _keep_in_fp32_modules: list[str] = ["rotary_pos_emb"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = True
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = False
        supports_thd: bool = False
        supports_cp_vision_frame_sharding: bool = True

    @classmethod
    def from_config(
        cls,
        config: Qwen3_5Config,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ):
        return cls(config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ):
        config = Qwen3_5Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Qwen3_5Config,
        backend: BackendConfig | None = None,
        *,
        mtp_loss_scaling_factor: float = 0.1,
        num_nextn_predict_layers: int | None = None,
        **kwargs: Any,
    ) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        del kwargs
        super().__init__(config)
        self.backend = _qwen3_5_backend(backend)

        text_config = config.text_config
        # Replace the HF text decoder with the native NeMo backbone (built on the
        # shared Block + CPAwareGatedDeltaNet), and class-swap the inner VLM model to
        # the routing wrapper. The HF vision tower + image/video scatter + mRoPE +
        # generation helpers stay intact (inherited). The backbone's ModuleDict layers
        # are pipeline-split-safe, so no Qwen3_5TextModelPP shim is needed.
        self.model.__class__ = Qwen3_5Model
        self.model.language_model = Qwen3_5DenseTextBackbone(text_config, self.backend)

        # Keep the SSM-gating params (per-layer ``_fp32_params`` holder) in fp32
        # storage even under a bf16 bulk dtype.
        keep_fp32 = list(getattr(self, "_keep_in_fp32_modules", None) or [])
        if "_fp32_params" not in keep_fp32:
            keep_fp32.append("_fp32_params")
        self._keep_in_fp32_modules = keep_fp32

        param_dtype = next(self.model.language_model.parameters()).dtype
        dtype = get_dtype(getattr(text_config, "torch_dtype", None), param_dtype)
        # ``super().__init__`` ran HF ``post_init`` (-> ``initialize_weights``) and may
        # have cast the inherited ``lm_head`` to a different bulk dtype before the
        # native backbone was swapped in; realign it to the backbone dtype so the
        # final hidden states and ``lm_head`` agree.
        if self.lm_head is not None and self.lm_head.weight.dtype != dtype:
            self.lm_head = self.lm_head.to(dtype)
        # HF post_init tied lm_head to the pre-swap embedding; the language_model
        # swap above orphaned that alias, so re-tie to the active embedding when the
        # config requests it (no-op otherwise).
        self.tie_weights()
        self.mtp_config = build_mtp_config_from_hf(
            text_config,
            loss_scaling_factor=mtp_loss_scaling_factor,
            num_nextn_predict_layers=num_nextn_predict_layers,
        )
        self.mtp = (
            build_qwen3_5_dense_mtp(text_config, self.mtp_config, dtype=dtype) if self.mtp_config.enabled else None
        )
        if self.mtp is not None:
            cast_model_to_dtype(self.mtp, dtype)
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen3_5DenseStateDictAdapter(route_linear_attn_fp32_params=True)

    def tie_weights(self, *_args: object, **_kwargs: object) -> None:
        """Tie ``lm_head`` to the active VLM text embedding when requested."""
        if getattr(self.config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def _pop_staged_vlm_media(
        self,
        input_ids: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        pixel_values = kwargs.get("pixel_values", None)
        pixel_values_videos = kwargs.get("pixel_values_videos", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        has_image_tokens = (
            bool((input_ids == image_token_id).any().item())
            if input_ids is not None and image_token_id is not None
            else False
        )
        has_video_tokens = (
            bool((input_ids == video_token_id).any().item())
            if input_ids is not None and video_token_id is not None
            else False
        )
        has_vision_start_tokens = (
            bool((input_ids == vision_start_token_id).any().item())
            if input_ids is not None and vision_start_token_id is not None
            else False
        )
        has_media_tokens = input_ids is not None and (has_image_tokens or has_video_tokens or has_vision_start_tokens)
        if input_ids is not None:
            if pixel_values is not None and image_token_id is not None and not has_image_tokens:
                pixel_values = None
                image_grid_thw = None
            if pixel_values_videos is not None and video_token_id is not None and not has_video_tokens:
                pixel_values_videos = None
                video_grid_thw = None
        if not has_media_tokens:
            return pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw

        chunk_idx = getattr(self, "_vlm_chunk_idx", 0)
        consumed_vlm_chunk = False

        if pixel_values is None:
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
                consumed_vlm_chunk = True

        if pixel_values_videos is None:
            video_chunks = getattr(self, "_vlm_pixel_values_videos_chunks", None)
            if video_chunks is not None and chunk_idx < len(video_chunks):
                video_chunk = video_chunks[chunk_idx]
                if video_chunk.numel() > 0:
                    pixel_values_videos = video_chunk
                    video_grid_chunks = getattr(self, "_vlm_video_grid_thw_chunks", None)
                    if video_grid_chunks is not None and chunk_idx < len(video_grid_chunks):
                        video_grid_thw = video_grid_chunks[chunk_idx]
                consumed_vlm_chunk = True

        if consumed_vlm_chunk:
            self._vlm_chunk_idx = chunk_idx + 1

        return pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw

    def _encode_vision_for_cp(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        *,
        is_video: bool,
    ) -> torch.Tensor:
        """Encode one modality's patches into flat, entry-ordered visual tokens under CP.

        Under context parallelism the vision tower otherwise runs redundantly on the full,
        un-sharded patch set on every CP rank. When the CP vision-frame sharding group is published
        (see :func:`~nemo_automodel.components.distributed.cp_vision_frame_shard.set_cp_vision_group`),
        route the single ``self.model.visual`` call through
        :func:`~nemo_automodel.components.distributed.cp_vision_frame_shard.maybe_distribute_visual`,
        which shards the frames across the CP group and all-gathers the per-frame embeds in
        original entry order. Its ``pooler_output`` is the flat, already-concatenated tensor,
        identical to ``torch.cat(get_image_features(...).pooler_output, dim=0)`` (HF
        ``get_image_features`` splits that same flat tensor per entry). When sharding is
        disabled / inactive this is the exact replicated ``get_image_features`` /
        ``get_video_features`` path.

        Args:
            pixel_values: Patch rows of shape ``[total_patch_rows, patch_dim]`` for ALL entries
                of one modality, frame-contiguous in entry order.
            grid_thw: ``[num_entries, 3]`` tensor of per-entry ``(t, h, w)`` grid sizes.
            is_video: Select the video (``True``) vs image (``False``) replicated fallback.

        Returns:
            ``[total_patch_rows / spatial_merge_size**2, hidden]`` flat merged-token embeddings
            in original entry order (vision hidden size), on the vision tower's output device.
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

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Return a sharder-only CP backend plus the full-sequence mRoPE positions.

        Embedding and the VLM->LM multimodal scatter now run inside ``forward``
        per microbatch (see :meth:`_embed_and_splice_for_cp`), so this hook only
        (a) computes the mRoPE ``position_ids`` on the *full* (unsharded) sequence
        via ``get_rope_index`` and returns them for :func:`shard_batch_aux_only`
        to round-robin-shard on the mRoPE axis, and (b) returns the
        :class:`ContextParallelSharder`. ``input_ids`` and the media inputs are
        left in the batch for the forward; ``mm_token_type_ids`` is consumed here
        (only ``get_rope_index`` needs it) so the sharded forward never sees a
        full-length copy.

        Args:
            batch: The batch dict (with ``input_ids`` and optional multimodal
                keys). ``input_ids`` is ``[batch, sequence]``.
            num_chunks: Accepted for hook-signature parity; unused (round-robin CP).
        """
        input_ids = batch.get("input_ids")
        if input_ids is None:
            raise ValueError("Qwen3.5 dense CP pre-embedding requires input_ids.")
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

        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=shard_batch_aux_only,
                local_token_global_indices=round_robin_local_indices,
            ),
            "position_ids": position_ids,
            "mm_token_type_ids": None,
            **promoted,
        }

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
        inside the forward before the CP sequence shard, so it is identical to
        the pre-CP-refactor pre-embed. Vision features may be frame-sharded
        across the CP group before ``get_placeholder_mask`` scatters them into
        the full ``input_ids`` sequence.

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
                image_embeds = self._encode_vision_for_cp(
                    pixel_values,
                    image_grid_thw,
                    is_video=False,
                ).to(inputs_embeds.device, inputs_embeds.dtype)
                image_mask, _ = self.model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_embeds,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                if hasattr(self.model.visual, "rotary_pos_emb"):
                    self.model.visual.rotary_pos_emb.to(pixel_values_videos.device)
                video_embeds = self._encode_vision_for_cp(
                    pixel_values_videos,
                    video_grid_thw,
                    is_video=True,
                ).to(inputs_embeds.device, inputs_embeds.dtype)
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

        Matches the framework default (first stage consumes full token ids
        ``[mb, seq]``; later stages consume hidden states; the last stage owning
        ``lm_head`` emits logits, earlier stages emit hidden states) except that
        under context parallelism the first stage embeds the full sequence and
        shards it in forward, so every stage output and later-stage input carries
        the LOCAL (padded-to-``2*cp`` then ``//cp``) sequence length while the
        first-stage input stays full-length. At ``cp_size == 1`` this reduces to
        the default symmetric shapes.
        """
        text_config = self.config.text_config
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
        logits_to_keep: int | torch.Tensor = 0,
        padding_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Qwen3_5CausalLMOutputWithPast:
        effective_use_cache = False if use_cache is None and self.training else use_cache
        kwargs = dict(kwargs)
        if effective_use_cache is not None:
            kwargs["use_cache"] = effective_use_cache
        kwargs["pixel_values"] = pixel_values
        kwargs["pixel_values_videos"] = pixel_values_videos
        kwargs["image_grid_thw"] = image_grid_thw
        kwargs["video_grid_thw"] = video_grid_thw
        pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw = self._pop_staged_vlm_media(
            input_ids, kwargs
        )

        if "qkv_format" in kwargs and kwargs["qkv_format"] == "thd":
            input_ids, position_ids, padding_mask, kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, kwargs
            )
            attention_mask = None
            if padding_mask is not None:
                kwargs["padding_mask"] = padding_mask

        model_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key not in {"pixel_values", "pixel_values_videos", "image_grid_thw", "video_grid_thw"}
        }

        # --- Pipeline-parallel stage dispatch ---
        # Under PP the model is split: stage 0 keeps embed_tokens (+ vision tower),
        # the last stage keeps lm_head (+ final norm), middle stages keep neither.
        # Detect a split stage and (a) on non-first stages feed the upstream hidden
        # states straight into the text backbone, (b) on non-last stages return raw
        # hidden states for the next stage, (c) run lm_head only where it survives.
        # MTP needs embed_tokens (absent past stage 0) so it is skipped under PP.
        language_model = self.model.language_model
        is_first_stage = getattr(language_model, "embed_tokens", None) is not None
        is_last_stage = getattr(self, "lm_head", None) is not None

        # Context-parallel: embed + vision-splice the full sequence, then keep this
        # rank's round-robin chunk pair (aux streams + mRoPE aligned by
        # shard_batch_aux_only). The local shard matches the old dispatch-level
        # pre-embed and stays differentiable (gradients reach embeddings/vision).
        cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
        if (
            cp_size > 1
            and is_first_stage
            and inputs_embeds is None
            and input_ids is not None
            and not torch.is_floating_point(input_ids)
        ):
            if not is_last_stage and (pixel_values is not None or pixel_values_videos is not None):
                raise NotImplementedError(
                    "Qwen3.5 does not support image/video microbatch chunking under combined pipeline + "
                    "context parallelism; use a text-only batch for cp>1 and pp>1."
                )
            inputs_embeds = self._embed_and_splice_for_cp(
                input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
            )
            inputs_embeds, _, _ = shard_sequence_for_cp_round_robin(self.cp_mesh, inputs_embeds, seq_dim=1)
            input_ids = None
            pixel_values = None
            pixel_values_videos = None

        if not (is_first_stage and is_last_stage):
            if is_first_stage:
                outputs = self.model(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    mm_token_type_ids=mm_token_type_ids,
                    **model_kwargs,
                )
                hidden_states = outputs.last_hidden_state
            else:
                # The PP schedule passes the upstream stage's hidden states in the
                # input_ids slot (a float tensor) or as inputs_embeds.
                hs = inputs_embeds
                if hs is None and input_ids is not None and torch.is_floating_point(input_ids):
                    hs = input_ids
                text_out = language_model(
                    inputs_embeds=hs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    **model_kwargs,
                )
                hidden_states = getattr(text_out, "last_hidden_state", text_out)
            if not is_last_stage:
                return hidden_states
            return self.lm_head(hidden_states)

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            **model_kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        mtp_per_depth_h: list[torch.Tensor] | None = None
        if self.mtp is not None and self.training:
            language_model = self.model.language_model
            source_embeds = inputs_embeds if inputs_embeds is not None else language_model.embed_tokens(input_ids)
            rotary_position_ids, text_position_ids = _split_qwen3_5_position_ids(
                position_ids,
                batch_size=source_embeds.shape[0],
                seq_len=source_embeds.shape[1],
                device=source_embeds.device,
                past_key_values=past_key_values,
            )
            # For packed sequences (indexed mask), feed the MTP SDPA sublayers a
            # 4D block-causal mask so attention stays within each document, the
            # same way the backbone treats packing. Otherwise use the standard
            # causal mask. See NVBugs 6330129.
            packing_mask = (
                attention_mask if is_indexed_packed_mask(attention_mask) else model_kwargs.get("_packed_seq_ids")
            )
            if is_indexed_packed_mask(packing_mask):
                causal_mask = _mtp_block_causal_mask(packing_mask, source_embeds)
            else:
                causal_mask = create_causal_mask(
                    config=language_model.config,
                    inputs_embeds=source_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                )
            mtp_kwargs = {
                key: value
                for key, value in model_kwargs.items()
                if key
                not in {
                    "cache_position",
                    "cu_seqlens",
                    "cu_seqlens_padded",
                    "max_seqlen",
                    "mm_token_type_ids",
                    "padding_mask",
                    "qkv_format",
                }
            }
            if input_ids is None:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    embed_inputs=_rolled_embed_inputs(source_embeds, self.mtp.num_depths),
                    position_ids=rotary_position_ids,
                    attention_mask=causal_mask,
                    rotary_emb=language_model.rotary_emb,
                    **mtp_kwargs,
                )
            else:
                mtp_per_depth_h = self.mtp(
                    hidden_states,
                    input_ids=input_ids,
                    embed_fn=language_model.embed_tokens,
                    position_ids=rotary_position_ids,
                    attention_mask=causal_mask,
                    rotary_emb=language_model.rotary_emb,
                    **mtp_kwargs,
                )

        return Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=getattr(outputs, "hidden_states", None) or (hidden_states,),
            attentions=getattr(outputs, "attentions", None),
            rope_deltas=getattr(outputs, "rope_deltas", None),
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
        # Initialize the native text backbone (embed/norm/layers incl. the
        # GatedDeltaNet/SSMGate-specific init). The HF vision tower + lm_head were
        # initialized by ``super().__init__`` (or loaded from a checkpoint).
        # HF's ``post_init`` (in ``super().__init__``) routes through
        # ``init_weights -> initialize_weights`` *before* the backbone is swapped in,
        # so ``language_model`` may still be the HF text model whose ``init_weights``
        # takes no ``buffer_device``; fall back to the no-arg HF signature then.
        language_model = self.model.language_model
        try:
            language_model.init_weights(buffer_device=buffer_device)
        except TypeError:
            language_model.init_weights()
        mtp = getattr(self, "mtp", None)
        if mtp is not None:
            with buffer_device:
                for sublayer in mtp.layers:
                    sublayer.init_weights(buffer_device=buffer_device)
        # Keep the fp32 SSM-gating params fp32 (skip them in the dtype cast); each
        # ``_fp32_params`` holder is sharded as its own fp32 FSDP group.
        cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params",))


ModelClass = Qwen3_5ForCausalLM
