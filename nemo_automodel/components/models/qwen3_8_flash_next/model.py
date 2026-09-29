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

"""Trainable Qwen3.8-Flash-Next conditional-generation model.

This implementation uses the checkpoint's compressed-block QSA router and
FlexAttention or optional SM90 CuTe sparse GQA for CUDA BF16 long-sequence SFT,
with a PyTorch oracle for CPU and numerical parity. Pipeline and tensor
parallelism remain unsupported. Context parallelism uses a model-owned contiguous
sequence shard for QSA, GDN, and PLE, and composes with sequence packing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

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
from nemo_automodel.components.models.qwen3_5_moe.model import Fp32SafeQwen3_5MoeTextRotaryEmbedding
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoEConfig
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .backend import Qwen3_8_FlashNextBackendConfig
from .config import Qwen3_8_FlashNextConfig, Qwen3_8_FlashNextTextConfig
from .cp import (
    Qwen3_8_FlashNextCPContext,
    packed_boundaries_from_seq_lens,
    shard_batch_for_qwen3_8_flash_next_cp,
)
from .engram import (
    QWEN3_8_FLASH_NEXT_NGRAM_PADDED_ROWS,
    Qwen3_8_FlashNextEngramTableConfig,
    Qwen3_8_FlashNextNGramEmbedding,
    Qwen3_8_FlashNextPLELayer,
)
from .layers import Qwen3_8_FlashNextDecoderLayer, Qwen3_8_FlashNextHyperConnection
from .state_dict_adapter import Qwen3_8_FlashNextStateDictAdapter


@dataclass
class Qwen3_8_FlashNextCausalLMOutput(CausalLMOutputWithPast):
    """Causal-LM output with optional per-layer HC states for parity capture."""


def _qwen3_8_flash_next_backend(backend: BackendConfig | None = None) -> BackendConfig:
    """Return a typed backend with the model's unfused rotary path."""
    resolved = copy.copy(backend) if backend is not None else Qwen3_8_FlashNextBackendConfig()
    resolved.rope_fusion = False
    return resolved


def _resolve_model_dtype(config: object) -> torch.dtype:
    value = getattr(config, "dtype", None)
    if value is None:
        value = getattr(config, "torch_dtype", None)
    return get_dtype(value, torch.bfloat16)


def _default_owner_group() -> dist.ProcessGroup | None:
    if not dist.is_available() or not dist.is_initialized():
        return None
    return dist.group.WORLD


class Qwen3_8_FlashNextTextModelBackend(nn.Module):
    """Qwen3.8-Flash-Next text decoder with four HC streams and one PLE layer.

    Args:
        config: Text architecture configuration.
        backend: Native attention, linear, and MoE backend configuration.
        moe_config: Optional native MoE configuration override.
        moe_overrides: Optional fields merged into the default MoE config.
        engram_process_group: Group that owns contiguous PLE table row shards.
            With the supported no-PP/no-TP topology this is the EP/world group.
        engram_table_config: Optional tiny-table override for unit tests. The
            released checkpoint uses ``[320001536, 160]`` globally.

    Tensor layout:
        Token embeddings start as ``[batch, sequence, hidden_size]``. Decoder
        layers retain flattened HC state
        ``[batch, sequence, hc_count * hidden_size]``. The final learned HC read
        collapses it back to ``[batch, sequence, hidden_size]``.
    """

    def __init__(
        self,
        config: Qwen3_8_FlashNextTextConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict[str, Any] | None = None,
        engram_process_group: dist.ProcessGroup | None = None,
        engram_table_config: Qwen3_8_FlashNextEngramTableConfig | None = None,
    ) -> None:
        super().__init__()
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides")
        if len(config.ple_layer_ids) > 1:
            raise NotImplementedError(
                f"This checkpoint contract supports one PLE table; got ple_layer_ids={config.ple_layer_ids}"
            )
        self.config = config
        self.backend = backend
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.model_dtype = _resolve_model_dtype(config)

        moe_defaults: dict[str, Any] = {
            "dim": config.hidden_size,
            "inter_dim": config.hidden_size,
            "moe_inter_dim": config.moe_intermediate_size,
            "n_routed_experts": config.num_experts,
            "n_shared_experts": 1,
            "n_activated_experts": config.num_experts_per_tok,
            "n_expert_groups": 0,
            "n_limited_groups": 0,
            "train_gate": True,
            "gate_bias_update_factor": 0.0,
            "score_func": "softmax",
            "route_scale": 1.0,
            "aux_loss_coeff": config.router_aux_loss_coef,
            "norm_topk_prob": config.norm_topk_prob,
            "expert_bias": False,
            "router_bias": False,
            "expert_activation": "swiglu",
            "softmax_before_topk": True,
            "shared_expert_gate": True,
            "shared_expert_inter_dim": config.shared_expert_intermediate_size,
            "dtype": self.model_dtype,
        }
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
            dtype=self.model_dtype,
        )
        self.layers = nn.ModuleDict()
        for layer_idx in range(config.num_hidden_layers):
            ple = None
            if (layer_idx + 1) in config.ple_layer_ids:
                ngram_heads = (config.ngram_size - 1) * config.heads_per_ngram
                if config.ple_embed_dim % ngram_heads != 0:
                    raise ValueError(
                        "ple_embed_dim must divide evenly over all n-gram heads; "
                        f"got {config.ple_embed_dim} and {ngram_heads} heads"
                    )
                table_config = engram_table_config or Qwen3_8_FlashNextEngramTableConfig(
                    num_embeddings=QWEN3_8_FLASH_NEXT_NGRAM_PADDED_ROWS,
                    embedding_dim=config.ple_embed_dim // ngram_heads,
                    initializer_range=config.initializer_range,
                )
                owner_group = engram_process_group if engram_process_group is not None else _default_owner_group()
                if owner_group is None and engram_table_config is None:
                    raise RuntimeError(
                        "The released Qwen3.8-Flash-Next PLE table must be constructed after "
                        "torch.distributed initialization so its 320M rows are owner-sharded. "
                        "Pass an explicit tiny engram_table_config only for a single-rank test."
                    )
                table = table_config.build(
                    process_group=owner_group,
                    dtype=self.model_dtype,
                )
                ple_embedding = Qwen3_8_FlashNextNGramEmbedding(
                    table,
                    ngram_size=config.ngram_size,
                    heads_per_ngram=config.heads_per_ngram,
                    eos_token_id=config.eos_token_id,
                )
                ple = Qwen3_8_FlashNextPLELayer(
                    ple_embedding,
                    hidden_size=config.hidden_size,
                    hc_count=config.hc_count,
                    ple_embed_dim=config.ple_embed_dim,
                    conv_kernel_size=config.ple_conv_kernel_size,
                    rms_norm_eps=config.rms_norm_eps,
                    backend=backend,
                    dtype=self.model_dtype,
                )
            self.layers[str(layer_idx)] = Qwen3_8_FlashNextDecoderLayer(
                layer_idx,
                config,
                self.moe_config,
                backend,
                ple=ple,
            )

        self.hyper_connection_mixer = Qwen3_8_FlashNextHyperConnection(
            hidden_size=config.hidden_size,
            hc_count=config.hc_count,
            lowrank_size=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            backend=backend,
            use_combine=False,
            dtype=self.model_dtype,
        )
        self.rotary_emb = Fp32SafeQwen3_5MoeTextRotaryEmbedding(config=config)

    def get_input_embeddings(self) -> nn.Module:
        """Return the raw-token embedding table."""
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Replace the raw-token embedding table.

        Args:
            value: Module mapping ``[batch, sequence]`` IDs to
                ``[batch, sequence, hidden_size]`` embeddings.
        """
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = None,
        output_hidden_states: bool | None = None,
        _qwen3_8_flash_next_cp_context: Qwen3_8_FlashNextCPContext | None = None,
        **attn_kwargs: Any,
    ) -> BaseModelOutputWithPast:
        """Run the HC decoder and final HC mixer.

        Args:
            input_ids: Raw tokenizer IDs of shape ``[batch, sequence]``. They
                remain required when ``inputs_embeds`` is supplied because PLE
                hashes the raw IDs.
            inputs_embeds: Optional precomputed token/vision embeddings of shape
                ``[batch, sequence, hidden_size]``.
            attention_mask: Optional mask of shape ``[batch, sequence]`` or a
                backend-specific attention mask.
            position_ids: Optional positions of shape ``[batch, sequence]`` or
                ``[axes, batch, sequence]``.
            padding_mask: Optional mask ``[batch, sequence]`` where ``True`` is padding.
            past_key_values: KV/SSM cache; unsupported by this training backend.
            use_cache: Cache request; ``True`` is unsupported.
            output_hidden_states: Include embedding, per-layer HC, and final
                collapsed states for parity diagnostics.
            _qwen3_8_flash_next_cp_context: Internal contiguous CP metadata. Its replicated
                raw-ID and padding tensors have shape ``[batch,
                global_sequence]``; model activations remain local
                ``[batch, sequence, hidden]`` tensors.
            **attn_kwargs: Attention backend metadata.

        Returns:
            A base-model output whose ``last_hidden_state`` has shape
            ``[batch, sequence, hidden_size]``.
        """
        if past_key_values is not None or use_cache:
            raise NotImplementedError("Qwen3.8-Flash-Next training does not support recurrent or KV caches")
        loader_seq_lens = attn_kwargs.pop("seq_lens", None)
        attn_kwargs.pop("seq_lens_padded", None)
        if attn_kwargs.get("qkv_format") in (None, "thd"):
            attn_kwargs.pop("qkv_format", None)
        if loader_seq_lens is not None and attn_kwargs.get("cu_seqlens") is None:
            packed_reference = input_ids if input_ids is not None else inputs_embeds
            if packed_reference is None:
                raise ValueError("packed seq_lens require input_ids or inputs_embeds for the physical row length")
            attn_kwargs["cu_seqlens"] = packed_boundaries_from_seq_lens(
                loader_seq_lens,
                total_tokens=packed_reference.shape[1],
            )
        if getattr(self, "_cp_enabled", False) and _qwen3_8_flash_next_cp_context is None:
            raise RuntimeError(
                "Qwen3.8-Flash-Next context parallelism is enabled, but the model-owned contiguous batch context is missing"
            )
        if _qwen3_8_flash_next_cp_context is not None:
            packed_keys = (
                "cu_seqlens",
                "cu_seqlens_q",
                "cu_seqlens_kv",
                "cu_seqlens_padded",
                "seq_lens",
                "seq_lens_padded",
                "packed_seq_ids",
                "_packed_seq_ids",
                "indices",
                "max_seqlen",
                "max_seqlen_q",
                "max_seqlen_kv",
            )
            if attn_kwargs.get("qkv_format") == "thd" or any(attn_kwargs.get(key) is not None for key in packed_keys):
                raise NotImplementedError(
                    "Qwen3.8-Flash-Next packed context parallelism requires document boundaries to be normalized "
                    "into the model-owned CP context by its batch sharder"
                )
            if attention_mask is not None and attention_mask.ndim != 2:
                raise NotImplementedError(
                    "Qwen3.8-Flash-Next context parallelism requires a non-packed [batch, local_sequence] attention mask"
                )
        if input_ids is None:
            if self.config.ple_layer_ids:
                raise ValueError("input_ids are required because Qwen3.8-Flash-Next PLE hashes raw token IDs")
            if inputs_embeds is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if input_ids is None:
            input_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

        batch_size, sequence_length, _ = inputs_embeds.shape
        if _qwen3_8_flash_next_cp_context is not None:
            if sequence_length != _qwen3_8_flash_next_cp_context.local_sequence_length:
                raise ValueError(
                    "Qwen3.8-Flash-Next local input length disagrees with its CP context; "
                    f"got local={sequence_length}, context={_qwen3_8_flash_next_cp_context.local_sequence_length}"
                )
            if position_ids is None:
                raise ValueError(
                    "Qwen3.8-Flash-Next context parallelism requires global position_ids from its batch sharder"
                )
            candidate_positions = (
                position_ids[1:] if position_ids.ndim == 3 and position_ids.shape[0] == 4 else position_ids
            )
            if candidate_positions.ndim == 3 and candidate_positions.shape[0] > 1:
                repeated_text_positions = candidate_positions[:1].expand_as(candidate_positions)
                if not bool(torch.equal(candidate_positions, repeated_text_positions)):
                    raise NotImplementedError(
                        "Qwen3.8-Flash-Next CP supports ordinary text positions only; genuinely different multi-axis mRoPE "
                        "streams and packed THD are unsupported"
                    )
        if position_ids is None:
            positions = torch.arange(sequence_length, device=inputs_embeds.device)
            position_ids = positions.view(1, 1, -1).expand(3, batch_size, -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
            position_ids = position_ids[1:]

        if padding_mask is None and attention_mask is not None and attention_mask.ndim <= 2:
            padding_mask = attention_mask.bool().logical_not()

        cos, sin = self.rotary_emb(inputs_embeds, position_ids)
        rotary_width = cos.shape[-1] // 2
        freqs_cis = torch.cat((cos[..., :rotary_width], sin[..., :rotary_width]), dim=-1)

        hidden_states = inputs_embeds
        captured_states: list[torch.Tensor] | None = [hidden_states] if output_hidden_states else None
        for layer in self.layers.values():
            hidden_states = layer(
                hidden_states,
                input_ids=input_ids,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                position_ids=position_ids,
                cp_context=_qwen3_8_flash_next_cp_context,
                **attn_kwargs,
            )
            if captured_states is not None:
                captured_states.append(hidden_states)

        hidden_states, _ = self.hyper_connection_mixer.mix(hidden_states)
        if captured_states is not None:
            captured_states.append(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=tuple(captured_states) if captured_states is not None else None,
        )

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device) -> None:
        """Initialize decoder weights for training from scratch.

        Args:
            buffer_device: Device used by layer initializers.
        """
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.initializer_range)
        for layer in self.layers.values():
            layer.init_weights(buffer_device, init_std=self.config.initializer_range)
            if layer.ple is not None:
                layer.ple.ple_embedding.ngram_embedding.reset_parameters()
                layer.ple.init_weights(self.config.initializer_range)
        self.hyper_connection_mixer.init_weights(init_std=self.config.initializer_range)


class Qwen3_8_FlashNextModel(nn.Module):
    """Language-only Qwen3.8-Flash-Next decoder wrapper."""

    def __init__(
        self,
        config: Qwen3_8_FlashNextConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict[str, Any] | None = None,
        engram_process_group: dist.ProcessGroup | None = None,
        engram_table_config: Qwen3_8_FlashNextEngramTableConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.language_model = Qwen3_8_FlashNextTextModelBackend(
            config.text_config,
            backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
            engram_process_group=engram_process_group,
            engram_table_config=engram_table_config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values: object | None = None,
        output_hidden_states: bool | None = None,
        _qwen3_8_flash_next_cp_context: Qwen3_8_FlashNextCPContext | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        """Run the language-only Qwen3.8-Flash-Next decoder.

        Args:
            input_ids: Raw IDs of shape ``[batch, sequence]``; required for
                PLE hashing.
            attention_mask: Optional token-validity mask of shape
                ``[batch, sequence]`` or backend-specific attention mask.
            position_ids: Optional positions of shape ``[batch, sequence]`` or
                ``[axes, batch, sequence]``.
            inputs_embeds: Optional embeddings of shape
                ``[batch, sequence, hidden_size]``.
            past_key_values: Cache state; unsupported for training.
            output_hidden_states: Capture decoder HC states.
            _qwen3_8_flash_next_cp_context: Internal contiguous CP metadata with replicated
                raw-ID/padding tensors of shape ``[batch, global_sequence]``.
            **kwargs: Text-attention backend arguments.

        Returns:
            Base-model output whose final text states have shape
            ``[batch, sequence, hidden_size]``.
        """
        return self.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_hidden_states=output_hidden_states,
            _qwen3_8_flash_next_cp_context=_qwen3_8_flash_next_cp_context,
            **kwargs,
        )


class Qwen3_8_FlashNextForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    """Trainable language-only Qwen3.8-Flash-Next causal-LM wrapper."""

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _keep_in_fp32_modules_strict = ["_fp32_params"]
    _owns_cp_attention = True
    # Packed (THD) training and packed CP are owned by the model's
    # route-indexed QSA path for the listed CUDA backends.
    _packed_cp_attn_backends = ("flex", "cute")

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Validated distributed features for the QSA sparse-SFT path."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = False
        supports_ep: bool = True
        supports_mtp_cp: bool = False

    @classmethod
    def from_config(
        cls,
        config: Qwen3_8_FlashNextConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs: Any,
    ) -> Qwen3_8_FlashNextForConditionalGeneration:
        """Construct from a parsed Qwen3.8-Flash-Next configuration."""
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> Qwen3_8_FlashNextForConditionalGeneration:
        """Construct architecture from a local/HF config before checkpoint load."""
        config = Qwen3_8_FlashNextConfig.from_pretrained(pretrained_model_name_or_path)
        config.language_model_only = True
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Qwen3_8_FlashNextConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        *,
        engram_process_group: dist.ProcessGroup | None = None,
        engram_table_config: Qwen3_8_FlashNextEngramTableConfig | None = None,
        **kwargs: Any,
    ) -> None:
        reject_unsupported_tie_word_embeddings(type(self), config)
        if not config.language_model_only:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next AutoModel support is currently language-only; set "
                "config.language_model_only=True. Vision and video training have not been validated."
            )
        super().__init__()
        moe_overrides = kwargs.pop("moe_overrides", None)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")
        self.config = config
        self.backend = _qwen3_8_flash_next_backend(backend)
        self.model = Qwen3_8_FlashNextModel(
            config,
            self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
            engram_process_group=engram_process_group,
            engram_table_config=engram_table_config,
        )
        dtype = _resolve_model_dtype(config.text_config)
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
            dtype=dtype,
        )
        self.vocab_size = config.text_config.vocab_size
        self.pad_token_id = config.text_config.pad_token_id if config.text_config.pad_token_id is not None else -1
        self.moe_config = self.model.language_model.moe_config
        self.mtp = None

        keep_fp32 = list(getattr(self, "_keep_in_fp32_modules", None) or [])
        if "_fp32_params" not in keep_fp32:
            keep_fp32.append("_fp32_params")
        self._keep_in_fp32_modules = keep_fp32

        if self.backend.enable_hf_state_dict_adapter:
            if len(config.text_config.ple_layer_ids) != 1:
                raise ValueError(
                    "Qwen3.8-Flash-Next checkpoint loading requires exactly one PLE layer; "
                    f"got ple_layer_ids={config.text_config.ple_layer_ids}"
                )
            ple_idx = int(config.text_config.ple_layer_ids[0]) - 1
            ple = self.model.language_model.layers[str(ple_idx)].ple
            if ple is None:
                raise RuntimeError(f"Configured PLE layer {ple_idx} was not constructed")
            self.state_dict_adapter = Qwen3_8_FlashNextStateDictAdapter(
                config.text_config,
                self.moe_config,
                self.backend,
                ple.ple_embedding.ngram_embedding,
                dtype=dtype,
                pretrained_model_name_or_path=getattr(config, "_name_or_path", None)
                or getattr(config, "name_or_path", None),
            )

    def get_input_embeddings(self) -> nn.Module:
        """Return the text token embedding module."""
        return self.model.language_model.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Replace the text token embedding module."""
        self.model.language_model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        """Return the untied LM projection."""
        return self.lm_head

    def set_output_embeddings(self, value: nn.Module) -> None:
        """Replace the untied LM projection."""
        self.lm_head = value

    def _nemo_prepare_model_owned_dtensors(self, fsdp_mesh: DeviceMesh) -> set[nn.Parameter]:
        """Convert PLE owner shards to global DTensors before FSDP wrapping.

        Args:
            fsdp_mesh: Flattened data/context shard mesh whose rank order is
                identical to the PLE owner process group.

        Returns:
            The exact registered parameter identities that FSDP must ignore
            because the model consumes their local shards directly.
        """
        parameters: set[nn.Parameter] = set()
        for layer in self.model.language_model.layers.values():
            if layer.ple is None:
                continue
            table = layer.ple.ple_embedding.ngram_embedding
            parameter = table.parallelize_weight(fsdp_mesh)
            if id(parameter) not in {id(registered) for registered in self.parameters()}:
                raise RuntimeError("The parallelized Engram DTensor is not registered on the Qwen3.8-Flash-Next model")
            parameters.add(parameter)
        return parameters

    def prepare_model_inputs_for_cp(self, batch: dict[str, Any], *, num_chunks: int = 1) -> dict[str, Any]:
        """Return Qwen3.8-Flash-Next's contiguous model-owned CP batch sharder.

        Args:
            batch: Full-sequence batch. Token-aligned tensors have shape
                ``[batch, global_sequence, ...]`` and remain unchanged until
                the returned sharder is invoked.
            num_chunks: Accepted for the framework hook contract; Qwen3.8-Flash-Next
                shards each non-packed batch directly and does not use it.

        Returns:
            Mapping containing one unresolved ``cp_sharder``. Once invoked it
            returns local contiguous token tensors of shape ``[batch,
            padded_global_sequence / cp_size, ...]``.
        """
        del batch, num_chunks
        return {
            "cp_sharder": ContextParallelSharder(
                shard_batch=partial(
                    shard_batch_for_qwen3_8_flash_next_cp,
                    pad_multiple=self.config.text_config.indexer_compress_ratio,
                ),
                local_token_global_indices=contiguous_local_indices,
            )
        }

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        past_key_values: object | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        output_hidden_states: bool | None = None,
        _qwen3_8_flash_next_cp_context: Qwen3_8_FlashNextCPContext | None = None,
        **kwargs: Any,
    ) -> Qwen3_8_FlashNextCausalLMOutput:
        """Run language-only generation and project final HC-mixed states.

        Args:
            input_ids: Raw tokenizer IDs of shape ``[batch, sequence]``.
            attention_mask: Optional token-validity mask of shape
                ``[batch, sequence]`` or backend-specific attention mask.
            position_ids: Optional positions of shape ``[batch, sequence]`` or
                ``[axes, batch, sequence]``.
            inputs_embeds: Optional embeddings of shape
                ``[batch, sequence, hidden_size]``.
            labels: Optional labels of shape ``[batch, sequence]``. They are
                accepted for recipe compatibility; loss is computed externally.
            past_key_values: Cache state; unsupported.
            use_cache: Cache request; ``True`` is unsupported.
            logits_to_keep: ``0`` for all positions, a positive trailing count,
                or an integer tensor of shape ``[kept_sequence]`` containing
                explicit sequence indices.
            output_hidden_states: Explicit ``True`` returns the embedding,
                every per-layer HC state, and the final state. When omitted,
                ``config.output_hidden_states=True`` returns only the final
                state for fused linear cross entropy without retaining every
                decoder activation.
            _qwen3_8_flash_next_cp_context: Internal contiguous CP metadata with replicated
                raw-ID/padding tensors of shape ``[batch, global_sequence]``.
            **kwargs: Text-attention backend metadata.

        Returns:
            Causal-LM output with logits
            ``[batch, kept_sequence, vocab_size]``.
        """
        del labels
        if use_cache:
            raise NotImplementedError("Qwen3.8-Flash-Next training does not support caches")
        capture_all_hidden_states = output_hidden_states is True
        return_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            output_hidden_states=capture_all_hidden_states,
            _qwen3_8_flash_next_cp_context=_qwen3_8_flash_next_cp_context,
            **kwargs,
        )
        lm_output = compute_lm_head_logits(
            self.lm_head,
            outputs.last_hidden_state,
            logits_to_keep,
            output_hidden_states=return_hidden_states,
        )
        return Qwen3_8_FlashNextCausalLMOutput(
            logits=lm_output.logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states if capture_all_hidden_states else lm_output.hidden_states,
        )

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize checkpoint-free model weights.

        Args:
            buffer_device: Target device for backend initializers.
            dtype: Final model parameter dtype, excluding intrinsic fp32 GDN state.
        """
        if buffer_device is None:
            buffer_device = (
                torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            )
        self.model.language_model.init_weights(buffer_device)
        std = self.config.text_config.hidden_size**-0.5
        nn.init.trunc_normal_(self.lm_head.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params",))
        for layer in self.model.language_model.layers.values():
            if layer.ple is not None:
                layer.ple.ple_embedding.ngram_embedding.mark_sharding_contract()


ModelClass = Qwen3_8_FlashNextForConditionalGeneration
