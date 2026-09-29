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

from dataclasses import dataclass
from typing import Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel._transformers.model_capabilities import ModelCapabilities
from nemo_automodel.components.models.common import (
    BackendConfig,
    HFCheckpointingMixin,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.mtp import (
    MTPContextParallelInputs,
    prepare_mtp_context_parallel_inputs,
)
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import (
    cast_model_to_dtype,
    compute_lm_head_logits,
    generation_config_from_model_config,
    restore_pretrained_generation_config,
)
from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3Block
from nemo_automodel.components.models.nemotron_v3.mtp import (
    _resolve_block_types_per_sublayer,
    build_mtp_config_from_hf,
    build_nemotron_v3_mtp,
)
from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


@dataclass
class NemotronHCausalLMOutputWithPast(CausalLMOutputWithPast):
    """``CausalLMOutputWithPast`` plus declared MTP fields.

    The MTP per-depth hidden states and scaling factor must be regular
    dataclass fields (rather than dynamically-set attributes) so they survive
    output-restructuring layers like FSDP2's mixed-precision output cast,
    which rebuild ``ModelOutput`` instances from declared fields only.
    """

    mtp_per_depth_h: list[torch.Tensor] | None = None
    mtp_loss_scaling_factor: float | None = None


class NemotronV3Model(nn.Module):
    """NemotronV3 base model (without LM head).

    This is a hybrid architecture with Mamba2, Attention, MLP, and MoE layers.
    """

    _keep_in_fp32_modules_strict = ["e_score_correction_bias", "_fp32_params"]

    def __init__(
        self,
        config,
        backend: BackendConfig | None = None,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        """Initialize NemotronV3Model.

        Args:
            config: NemotronH config with model parameters
            backend: Backend configuration for MoE and other components
            moe_config: MoE configuration (optional, will create default if None)
            moe_overrides: Optional dict of overrides to apply to the default MoE config
        """
        super().__init__()
        self.config = config
        self.backend = backend or BackendConfig()
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")
        # Dense Nemotron-H variants (e.g. the Nano 4B / 9B / 12B BF16 models) have no MoE
        # layers, and their HF config omits the n_routed_experts / moe_* fields, so eagerly
        # building a MoEConfig would raise AttributeError. Only construct it when the model
        # actually has MoE layers (or one was supplied explicitly). A dense model carries
        # moe_config=None, which NemotronV3Block only ever dereferences in its "moe" branch
        # (never reached for mamba/attention/mlp layers).
        layer_block_types = getattr(config, "layers_block_type", None) or []
        has_moe = (
            moe_config is not None
            or moe_overrides is not None
            or "moe" in layer_block_types
            or getattr(config, "n_routed_experts", None) is not None
        )
        if not has_moe:
            self.moe_config = None
        else:
            moe_defaults = dict(
                n_routed_experts=config.n_routed_experts,
                n_shared_experts=1,  # NemotronV3 has 1 shared expert
                n_activated_experts=config.num_experts_per_tok,
                n_expert_groups=config.n_group,
                n_limited_groups=config.topk_group,
                train_gate=True,
                gate_bias_update_factor=0.0,
                aux_loss_coeff=0.0,  # No aux loss for NemotronV3
                score_func="sigmoid",  # NemotronV3 uses sigmoid scoring
                route_scale=config.routed_scaling_factor,
                dim=config.hidden_size,
                inter_dim=config.intermediate_size,  # For shared expert
                moe_inter_dim=config.moe_intermediate_size,  # For routed experts
                norm_topk_prob=config.norm_topk_prob,
                router_bias=False,
                expert_bias=config.mlp_bias,
                expert_activation="relu2",  # NemotronV3 uses ReLU² activation
                dtype=config.torch_dtype,
                shared_expert_gate=False,
                shared_expert_inter_dim=config.moe_shared_expert_intermediate_size,
                shared_expert_activation="relu2",  # Use ReLU² for shared experts
                force_e_score_correction_bias=True,  # NemotronV3 checkpoint has this buffer
                moe_latent_size=getattr(config, "moe_latent_size", None),
            )
            if moe_overrides:
                moe_defaults.update(moe_overrides)
            self.moe_config = moe_config or MoEConfig(**moe_defaults)

        # Embeddings
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)

        # Transformer layers (hybrid: mamba, attention, mlp, moe)
        self.layers = nn.ModuleDict()
        for idx in range(config.num_hidden_layers):
            self.layers[str(idx)] = NemotronV3Block(
                config, layer_idx=idx, moe_config=self.moe_config, backend=self.backend
            )

        # Final norm
        self.norm = initialize_rms_norm_module(
            self.backend.rms_norm,
            config.hidden_size,
            eps=config.layer_norm_epsilon,
            dtype=dtype,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        causal_mask_mapping: dict[str, torch.Tensor] | None = None,
        inputs_embeds: torch.Tensor | None = None,
        past_key_values=None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Forward pass through the model.  Supports BSHD ``[B, S, H]`` and THD ``[T, H]``.

        Pipeline-parallel awareness: when ``self.embed_tokens is None`` (non-first
        PP stage), ``input_ids`` is interpreted as the upstream hidden-state
        tensor and routed through the ``inputs_embeds`` branch. When
        ``self.norm is None`` (non-last PP stage), the final norm is skipped.
        """
        # Get embeddings (PP-aware: a trimmed mid-stage has embed_tokens=None
        # and receives the prior stage's hidden states as input_ids).
        if inputs_embeds is None:
            if getattr(self, "embed_tokens", None) is not None:
                if input_ids is None:
                    raise ValueError("input_ids must be provided if inputs_embeds is not provided")
                hidden_states = self.embed_tokens(input_ids)
            else:
                if input_ids is None or input_ids.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                    raise ValueError("Non-first PP stage expects an upstream hidden-state tensor")
                hidden_states = input_ids
        else:
            hidden_states = inputs_embeds

        # TE natively supports THD; squeeze to [T, H] so attention layers
        # pick the THD branch. SDPA/flex only support 4D BSHD.
        _attn_impl = getattr(getattr(self, "backend", None), "attn", None)
        squeezed_for_thd = False
        if (
            kwargs.get("qkv_format") == "thd"
            and _attn_impl == "te"
            and hidden_states.dim() == 3
            and hidden_states.shape[0] == 1
        ):
            hidden_states = hidden_states.squeeze(0)
            squeezed_for_thd = True

        is_thd = hidden_states.dim() == 2

        # Non-THD-collater path doesn't emit cu_seqlens — recover it from a
        # 2D indexed attention_mask so mamba's seq_idx derivation has input.
        # Gated on B==1: cu_seqlens describes sub-sequence boundaries within a
        # single flattened token stream. At B>1 the rows are independent
        # sequences (the batch dim already separates them), so cumsum(0) would
        # be a wrong cross-row global offset. Leave cu_seqlens unset there and
        # let each consumer treat every row as one sequence (mamba seq_idx
        # falls back to per-row, the PP seq_idx tail to its no-mask sentinel).
        if (
            "cu_seqlens" not in kwargs
            and attention_mask is not None
            and attention_mask.dim() == 2
            and attention_mask.shape[0] == 1
            and attention_mask.dtype != torch.bool
        ):
            seq_lens = attention_mask.sum(dim=-1).to(torch.int32)
            kwargs["cu_seqlens"] = F.pad(seq_lens.cumsum(0).to(torch.int32), (1, 0))

        # Per-microbatch arrives as [1, K] (THD collator stacks per-MB and PP
        # tensor_splits dim 0). Squeeze to 1D, strip the -1000 right-pad
        # sentinels from pad_and_stack, and clone to a fresh tensor so the
        # values survive TE backward (PP may free view-backed kwarg storage).
        _SEQLEN_SENTINEL = -1000
        for _k in ("cu_seqlens", "cu_seqlens_padded"):
            _v = kwargs.get(_k)
            if isinstance(_v, torch.Tensor) and _v.dim() == 2 and _v.shape[0] == 1:
                _v = _v.squeeze(0)
            if isinstance(_v, torch.Tensor) and _v.dim() == 1:
                if (_v == _SEQLEN_SENTINEL).any():
                    _v = _v[_v != _SEQLEN_SENTINEL]
                kwargs[_k] = _v.contiguous().clone()
        _ms = kwargs.get("max_seqlen")
        if isinstance(_ms, torch.Tensor) and _ms.dim() >= 1 and _ms.numel() == 1:
            kwargs["max_seqlen"] = _ms.flatten()[0].clone()

        causal_mask = causal_mask_mapping.get("full_attention") if causal_mask_mapping is not None else None

        # Neat-packed SDPA path: seq_idx came from _packed_seq_ids upstream
        # and attention_mask is the 4D block-causal bool mask. Mamba uses
        # seq_idx (no 2D mask); attention uses the 4D mask directly.
        _neat_packed = "seq_idx" in kwargs and attention_mask is not None and attention_mask.dim() == 4

        for layer in self.layers.values():
            if is_thd:
                mask = None
            elif _neat_packed:
                mask = attention_mask if layer.block_type == "attention" else None
            elif layer.block_type == "attention":
                # Fall back to 2D attention_mask when causal_mask is None
                # (e.g. TE+CP, where the CP split drops the precomputed 4D mask).
                mask = causal_mask if causal_mask is not None else attention_mask
            elif layer.block_type == "mamba":
                # Mamba layers use 2D padding mask during prefill, None during decode
                mask = None if (past_key_values is not None and past_key_values.has_previous_state) else attention_mask
            else:
                # MLP/MoE layers don't use mask
                mask = None

            hidden_states = layer(
                hidden_states,
                attention_mask=mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                **kwargs,
            )

        # Norm is None on non-last PP stages (splitter trims it).
        if getattr(self, "norm", None) is not None:
            hidden_states = self.norm(hidden_states)

        if squeezed_for_thd:
            hidden_states = hidden_states.unsqueeze(0)

        return hidden_states

    @torch.no_grad()
    def initialize_weights(self, buffer_device: torch.device | None = None) -> None:
        """Initialize model weights according to NemotronV3 spec.

        After PP splitting, ``embed_tokens`` may be ``None`` on non-first
        stages and ``norm`` may be ``None`` on non-last stages; guard each.

        Args:
            buffer_device: Device to use for buffer initialization
        """
        with buffer_device:
            if getattr(self, "embed_tokens", None) is not None:
                nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=self.config.initializer_range)
            if getattr(self, "norm", None) is not None:
                self.norm.reset_parameters()

        for block in self.layers.values():
            block.init_weights(buffer_device=buffer_device)


class NemotronHForCausalLM(HFCheckpointingMixin, GenerationMixin, nn.Module, MoEFSDPSyncMixin):
    """NemotronV3 model with language modeling head.

    Supports ``.generate()`` from ``transformers.generation.GenerationMixin`` with O(1)
    per-step KV caching for attention layers and recurrent state caching for Mamba2 layers.
    """

    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    # Hybrid Mamba2/Attention uses NemotronHybridCache, not DynamicCache.
    _is_stateful: bool = True
    main_input_name: str = "input_ids"
    _keep_in_fp32_modules_strict = ["e_score_correction_bias", "_fp32_params"]

    # Skip patch_hf_model_for_pp; our forward already handles PP routing.
    _pp_keep_self_forward: bool = True

    def get_experts_implementation(self) -> dict[str, str | None]:
        """Return the configured expert backend for Transformers generation hooks."""
        return {"": getattr(self.config, "_experts_implementation", None)}

    def set_experts_implementation(self, experts_implementation: str | dict[str, str | None]) -> None:
        """Set the expert backend used by Transformers generation hooks."""
        implementation = (
            experts_implementation
            if isinstance(experts_implementation, str)
            else experts_implementation.get("", getattr(self.config, "_experts_implementation", None))
        )
        self.config._experts_implementation = implementation

    @classmethod
    def get_capabilities(cls, config) -> ModelCapabilities:
        """Return parallelism capabilities for a specific Nemotron-H config.

        NemotronHForCausalLM serves both the MoE ("v3") and the dense Nemotron-H
        variants. Expert parallelism only applies when the config actually has MoE
        layers; a dense config has no experts to parallelize, so supports_ep must
        be False for it (see #2004). The other flags are the same for both.
        """
        is_moe = getattr(config, "n_routed_experts", None) is not None or "moe" in (
            getattr(config, "layers_block_type", None) or []
        )
        return ModelCapabilities(supports_cp=True, supports_pp=True, supports_ep=is_moe, supports_mtp_cp=True)

    @classmethod
    def from_config(
        cls,
        config,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        """Create model from config.

        Args:
            config: NemotronH config
            backend: Backend configuration
            **kwargs: Additional arguments

        Returns:
            NemotronHForCausalLM instance
        """
        return cls(config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        """Load pretrained model.

        Args:
            pretrained_model_name_or_path: Path or name of pretrained model
            *model_args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            NemotronHForCausalLM instance
        """
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        model = cls.from_config(config, *model_args, **kwargs)
        # The checkpoint's own generation settings win over the config-derived
        # defaults, as in PreTrainedModel.from_pretrained.
        restore_pretrained_generation_config(model, pretrained_model_name_or_path)
        return model

    def __init__(
        self,
        config,
        backend: BackendConfig | None = None,
        *,
        mtp_loss_scaling_factor: float = 0.1,
        num_nextn_predict_layers: int | None = None,
        mtp_use_repeated_layer: bool = False,
        **kwargs,
    ):
        """Initialize NemotronV3ForCausalLM.

        Args:
            config: NemotronH config.
            backend: Backend configuration.
            mtp_loss_scaling_factor: Auxiliary-loss weight for the MTP head
                (default ``0.1``). Programmatic override only — not exposed
                as a YAML knob to keep recipe configs auto-detected.
            num_nextn_predict_layers: Optional override for the HF config's
                ``num_nextn_predict_layers`` field (i.e. the MTP forward
                iteration count). When ``None``, the value from ``config`` is
                used. Set explicitly when the trained model used weight-tied
                MTP (``mtp_use_repeated_layer=True``) and the HF export only
                retains the physical depth count.
            mtp_use_repeated_layer: When ``True``, build a single physical
                MTP depth and reuse it across all iterations. Mirrors
                Megatron's ``--mtp-use-repeated-layer``. Defaults to ``False``.
            **kwargs: Additional arguments. Recognized keys:
                ``moe_config``, ``moe_overrides``.
        """
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = backend or BackendConfig()

        # Base model
        moe_overrides = kwargs.pop("moe_overrides", None)
        moe_config = kwargs.pop("moe_config", None)
        self.model = NemotronV3Model(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        self.output_hidden_states = config.to_dict().get("output_hidden_states", False)

        # LM head
        dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=dtype,
        )

        self.mtp_config = build_mtp_config_from_hf(
            config,
            loss_scaling_factor=mtp_loss_scaling_factor,
            num_nextn_predict_layers=num_nextn_predict_layers,
            use_repeated_layer=mtp_use_repeated_layer,
        )
        if self.mtp_config.enabled:
            block_types = _resolve_block_types_per_sublayer(config)
            self.mtp = build_nemotron_v3_mtp(
                config,
                mtp_config=self.mtp_config,
                backend=self.backend,
                moe_config=self.model.moe_config,
                dtype=dtype,
                block_types=block_types,
            )
        else:
            self.mtp = None

        # When True, the MTP heads also run under eval mode (``self.training``
        # False) so validation can measure per-head token acceptance. Defaults
        # to False so generation/decoding never pays the MTP cost; the training
        # harness toggles it around the validation forward only.
        self.compute_mtp_in_eval = False

        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = NemotronV3StateDictAdapter(
                config=config,
                moe_config=self.model.moe_config,
                backend=self.backend,
                dtype=dtype,
            )

        # Required by GenerationMixin.generate(). Seeded from the model config like
        # PreTrainedModel does, so eos/bos/pad are set: the consolidated export writes
        # this object to generation_config.json, and a blank one there leaves the
        # exported model with no stop token.
        self.generation_config = generation_config_from_model_config(config)

    @property
    def device(self) -> torch.device:
        """Return the device of the first model parameter (required by GenerationMixin)."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the first model parameter (used by cache construction)."""
        return next(self.parameters()).dtype

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable activation checkpointing on each transformer (and MTP) block.

        Wraps every decoder block (and MTP block, when present) with a
        non-reentrant checkpoint wrapper so that block activations are recomputed
        during the backward pass instead of being stored. This is the single-GPU
        entry point: ``FSDP2Manager.parallelize`` calls it when ``world_size == 1``
        (the expert-parallel path performs the equivalent wrapping inside the MoE
        parallelizer's ``apply_ac``). Without it, the hybrid Mamba2/Attention MoE
        keeps every block's activations live, which is what pushes single-GPU LoRA
        SFT over a single 80GB device. Idempotent.

        Args:
            gradient_checkpointing_kwargs: Accepted for HF API compatibility;
                currently unused (NO_REENTRANT wrapping is always used).
        """
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            checkpoint_wrapper,
        )

        if getattr(self, "_gradient_checkpointing", False):
            return

        def _wrap(block):
            return checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT, preserve_rng_state=True)

        containers = [self.model.layers]
        if self.mtp is not None and getattr(self.mtp, "layers", None) is not None:
            containers.append(self.mtp.layers)
        for layers in containers:
            for layer_id, block in list(layers.named_children()):
                layers.register_module(layer_id, _wrap(block))
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        """Unwrap any checkpoint-wrapped blocks (inverse of ``gradient_checkpointing_enable``)."""
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

        if not getattr(self, "_gradient_checkpointing", False):
            return
        containers = [self.model.layers]
        if self.mtp is not None and getattr(self.mtp, "layers", None) is not None:
            containers.append(self.mtp.layers)
        for layers in containers:
            for layer_id, block in list(layers.named_children()):
                if isinstance(block, CheckpointWrapper):
                    layers.register_module(layer_id, block._checkpoint_wrapped_module)
        self._gradient_checkpointing = False

    def _is_pipeline_parallel_stage(self) -> bool:
        """True when this module instance has been trimmed to a PP stage subset.

        Detection mirrors ``DeepseekV4ForCausalLM._is_pipeline_parallel_stage``:
        any of (a) ``lm_head`` is None, (b) inner ``embed_tokens`` is None,
        (c) ``model.layers`` count diverges from ``config.num_hidden_layers``
        is sufficient — the PP splitter nulls these attributes when trimming.

        The checks use ``hasattr`` to distinguish "splitter nulled the
        attribute" (attribute present, value is None) from "caller replaced
        ``self.model`` with a stub that doesn't declare the attribute"
        (attribute absent). Tests that swap in stub inner modules should not
        be misclassified as PP stages.
        """
        if self.lm_head is None:
            return True
        if hasattr(self.model, "embed_tokens") and self.model.embed_tokens is None:
            return True
        if hasattr(self.model, "layers"):
            try:
                return len(self.model.layers) != int(self.config.num_hidden_layers)
            except TypeError:
                return False
        return False

    def _build_mtp_embed_inputs_for_pp(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Build the per-depth rolled-token embeddings on the first PP stage.

        The first PP stage owns ``embed_tokens`` and is the only rank that can
        produce the future-token embeddings consumed by the MTP head on the
        final stage. The tuple flows alongside ``hidden_states`` through every
        intermediate stage as additional positional outputs (see ``forward``).

        Args:
            input_ids: Token ids ``[B, S]`` (int).

        Returns:
            Tuple of length ``self.mtp_config.num_layers`` containing
            ``[B, S, hidden]`` embeddings for depths 1..D (i.e. for predicting
            tokens shifted left by 1..D positions).
        """
        if getattr(self.model, "embed_tokens", None) is None:
            raise ValueError("First PP stage must own embed_tokens to build MTP embeddings")
        if input_ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError("First PP stage must receive token ids to build MTP embeddings")

        from nemo_automodel.components.models.common.mtp import roll_tensor  # noqa: PLC0415

        cur_input_ids = input_ids
        embeds: list[torch.Tensor] = []
        for _ in range(self.mtp_config.num_layers):
            cur_input_ids = roll_tensor(cur_input_ids, shifts=-1, dim=-1)
            embeds.append(self.model.embed_tokens(cur_input_ids))
        return tuple(embeds)

    def prepare_mtp_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        ignore_index: int = -100,
    ) -> MTPContextParallelInputs | None:
        """Prepare global MTP futures before context-parallel sharding.

        The recipe calls this hook while ``batch`` is still in global sequence
        order. It shifts future-token IDs, positions, and loss targets without
        crossing packed-sequence boundaries. The recipe then shards each tensor
        with the same :class:`ContextParallelSharder` instance as the main
        inputs, avoiding both rank-local rolls and an all-gather.

        Args:
            batch: Unsharded model batch containing ``input_ids`` and ``labels``
                with shape ``[batch, sequence]``. Missing ``position_ids`` are
                synthesized in global order. ``seq_idx``, ``_packed_seq_ids``,
                raw ``seq_lens_padded``, or ``cu_seqlens`` may describe
                packed-sequence boundaries.
            ignore_index: Fill value for invalid MTP loss targets.

        Returns:
            Globally ordered tensors for every enabled MTP depth, or ``None``
            when MTP is disabled.
        """
        if not self.mtp_config.enabled:
            return None

        return prepare_mtp_context_parallel_inputs(
            batch,
            num_depths=self.mtp_config.num_layers,
            ignore_index=ignore_index,
        )

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Pin the MTP head to the last PP stage's FQN list.

        Called by ``split_model_into_stages`` (functional.py:494-502) after the
        default per-stage FQN auto-generation. The auto-generator includes
        ``embed_tokens`` on the first stage and ``norm``/``lm_head`` on the
        last stage but doesn't know about ``model.mtp``; this hook appends it.
        """
        del layers_prefix, text_model  # unused — no per-stage rotary to replicate
        stage_modules = [list(m) for m in module_names_per_stage]
        if self.mtp is not None and stage_modules:
            last = stage_modules[-1]
            if "mtp" not in last:
                last.append("mtp")
        return stage_modules

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return analytical (inputs_meta, outputs_meta) for a PP stage.

        Inter-stage tensors are plain ``[B, S, H]`` (no HC stream). With MTP
        enabled, every transfer carries ``1 + D`` tensors so the variadic
        forward signature is exercised on every microbatch.
        """
        hidden_shape = (microbatch_size, seq_len, self.config.hidden_size)
        mtp_depth = int(getattr(self.mtp_config, "num_layers", 0) or 0)

        def meta(shape: tuple[int, ...], d: torch.dtype = dtype) -> torch.Tensor:
            return torch.empty(*shape, device="meta", dtype=d)

        def append_mtp(primary: torch.Tensor) -> tuple[torch.Tensor, ...]:
            if mtp_depth == 0:
                return (primary,)
            return (primary, *(meta(hidden_shape) for _ in range(mtp_depth)))

        if is_first:
            inputs_meta: tuple[torch.Tensor, ...] = (
                torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),
            )
        else:
            inputs_meta = append_mtp(meta(hidden_shape))

        if self.lm_head is not None:
            primary_out = meta((microbatch_size, seq_len, self.config.vocab_size))
        else:
            primary_out = meta(hidden_shape)
        outputs_meta = append_mtp(primary_out)
        # Last stage appends an int32 [B, S] seq_idx so the loss fn can mask
        # MTP label rolls across sub-seq boundaries — bonded to its microbatch
        # via the PP output-tuple contract (schedule-agnostic).
        if self.lm_head is not None and mtp_depth > 0:
            outputs_meta = (*outputs_meta, meta((microbatch_size, seq_len), d=torch.int32))
        return inputs_meta, outputs_meta

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        *mtp_embed_inputs: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_mask_mapping: dict[str, torch.Tensor] | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        mtp_per_depth_input_ids: tuple[torch.LongTensor, ...] | None = None,
        mtp_per_depth_position_ids: tuple[torch.LongTensor, ...] | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        compute_logits: bool = True,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        """Forward pass with optional loss computation.

        Supports both BSHD format (``input_ids`` shape ``[B, S]``) and THD format
        (``input_ids`` shape ``[T]`` after ``squeeze_input_for_thd``). When
        ``kwargs["qkv_format"] == "thd"`` AND the attention backend is TE,
        inputs are squeezed to THD before the base-model forward and logits
        are unsqueezed back to ``[1, T, V]`` on exit. SDPA / flex stay in BSHD.

        Pipeline-parallel awareness: when run as a PP stage, ``input_ids`` is
        the upstream stage's hidden-state tensor on non-first stages, and
        ``*mtp_embed_inputs`` carries ``num_nextn_predict_layers`` future-token
        embeddings produced by the first stage. See the Returns section below
        for the per-stage tuple contract. The single-rank (no-PP) path returns
        :class:`NemotronHCausalLMOutputWithPast` unchanged.

        Args:
            input_ids: Input token IDs. BSHD: ``[B, S]``; THD: ``[1, T]``
                (squeezed internally). On non-first PP stages this slot
                instead carries the upstream stage's hidden-state tensor.
            *mtp_embed_inputs: Pre-computed future-token embeddings produced
                by the first PP stage and forwarded between stages as
                positional args. Empty on the single-rank (no-PP) path.
            attention_mask: 2D padding mask ``[B, S]``.
            causal_mask_mapping: Dict with precomputed 4D causal masks
                (key ``"full_attention"`` is consumed).
            inputs_embeds: Pre-computed input embeddings (optional).
            labels: Token IDs for loss computation ``[B, S]`` (optional;
                under PP, loss is computed by ``PipelineCausalLMLoss``).
            past_key_values: Optional ``NemotronHybridCache`` for incremental decoding.
            use_cache: Whether to return ``past_key_values`` for subsequent steps.
            cache_position: Token position indices for cache updates.
            position_ids: Position IDs (forwarded into MTP sublayer kwargs).
            mtp_per_depth_input_ids: Optional globally shifted future-token IDs,
                one tensor per MTP depth. Each tensor has shape [batch,
                sequence], or [tokens] for THD, matching the local CP layout.
                When supplied, MTP embeds these tensors directly instead of
                rolling rank-local ``input_ids``.
            mtp_per_depth_position_ids: Optional pre-computed future-token
                position IDs, one tensor per MTP depth. Each tensor has shape
                [batch, sequence], or [tokens] for THD, matching the local CP
                layout of ``position_ids``. When supplied, MTP consumes these
                tensors directly instead of rolling rank-local positions.
            padding_mask: Padding mask ``[B, S]`` used by the THD squeeze helper
                and as the MoE / mamba 2D mask source.
            logits_to_keep: If > 0, only compute logits for the last
                ``logits_to_keep`` token positions.
            output_hidden_states: Whether to return hidden states.
            return_dict: Accepted for API compatibility (always returns a
                ``NemotronHCausalLMOutputWithPast`` off-PP).
            compute_logits: Whether to materialize logits. Fused-linear loss callers
                may disable this off-PP when they consume final hidden states directly.
            **kwargs: Additional arguments forwarded to the base model
                (e.g. ``qkv_format``, ``cu_seqlens``, ``cu_seqlens_padded``,
                ``max_seqlen``, ``seq_idx``, ``cp_rank``, ``cp_size``,
                ``_packed_seq_ids``).

        Returns:
            Off-PP: :class:`NemotronHCausalLMOutputWithPast` with ``logits``,
            optional ``loss``, ``past_key_values``, ``hidden_states``, and the
            MTP per-depth hidden states / loss-scaling factor when MTP is on.

            Under PP, returns a positional tuple instead:
              * mid stages: ``(hidden_states, *mtp_embed_inputs)`` arity ``1 + D``.
              * last stage: ``(logits, *mtp_per_depth_h, seq_idx)`` arity
                ``1 + D + 1`` when MTP is enabled, else ``logits`` alone.
        """
        is_pp_stage = self._is_pipeline_parallel_stage()
        is_first_stage = getattr(self.model, "embed_tokens", None) is not None
        has_lm_head = self.lm_head is not None
        mtp_depth = int(getattr(self.mtp_config, "num_layers", 0) or 0)
        pp_mtp_enabled = is_pp_stage and self.mtp_config.enabled
        # CP shards are not contiguous sequence slices. MTP therefore cannot
        # derive future-token embeddings or positions by rolling rank-local
        # tensors; callers must provide both globally shifted tensors.
        mtp_active = self.mtp is not None and (self.training or (self.compute_mtp_in_eval and not use_cache))
        cp_size = int(kwargs.get("cp_size", 1) or 1)
        if cp_size > 1 and mtp_active:
            precomputed_token_sources = int(bool(mtp_embed_inputs)) + int(mtp_per_depth_input_ids is not None)
            if precomputed_token_sources != 1 or mtp_per_depth_position_ids is None:
                raise ValueError(
                    "Context-parallel MTP requires exactly one of precomputed per-depth embeddings or input IDs, "
                    "together with per-depth position IDs"
                )

        # Neat-packed SDPA: convert _packed_seq_ids (1-based [B,S] int, 0=pad)
        # to mamba's seq_idx and derive a 2D padding_mask. The neat collater
        # already supplies a 4D attention_mask, but mamba's mixer multiplies
        # hidden_states by a 2D mask, so we need both.
        _packed_seq_ids = kwargs.pop("_packed_seq_ids", None)
        if isinstance(_packed_seq_ids, torch.Tensor) and "seq_idx" not in kwargs:
            kwargs["seq_idx"] = _packed_seq_ids.to(torch.int32).contiguous()
            if padding_mask is None:
                padding_mask = _packed_seq_ids == 0

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        # Stash pre-squeeze [B, S] input_ids: the MTP embed tuple must be
        # built AFTER self.model() runs (FSDP2 root lazy-init requires the
        # root forward first) but with the pre-squeeze shape so emitted
        # tensors match the [B, S, H] contract from get_pipeline_stage_metas.
        pre_squeeze_input_ids = (
            input_ids
            if (
                pp_mtp_enabled
                and is_first_stage
                and not has_lm_head
                and not mtp_embed_inputs
                and input_ids is not None
                and input_ids.dtype in (torch.int32, torch.int64, torch.long)
            )
            else None
        )

        # Squeezing to [T, H] only helps TE; SDPA/flex need 4D BSHD. Keep
        # is_thd true regardless so the post-forward unsqueeze still fires.
        _attn_impl = getattr(getattr(self, "backend", None), "attn", None)
        is_thd = kwargs.get("qkv_format") == "thd"
        squeeze_for_thd = is_thd and _attn_impl == "te"
        if squeeze_for_thd and is_first_stage:
            input_ids, position_ids, padding_mask, kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, kwargs
            )
            attention_mask = None
            causal_mask_mapping = None

        # MoE needs a padding_mask; derive from attention_mask when missing.
        if padding_mask is None and attention_mask is not None and attention_mask.dim() == 2:
            padding_mask = attention_mask.bool().logical_not()

        # On non-first PP stages, the upstream hidden-state tensor arrives in
        # the input_ids slot; the inner model routes it via inputs_embeds.
        hidden_states = self.model(
            input_ids,
            attention_mask=attention_mask,
            causal_mask_mapping=causal_mask_mapping,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            cache_position=cache_position,
            **kwargs,
        )

        # Root forward has run; FSDP2 lazy-init is satisfied. Build MTP embed
        # tuple from pre-squeeze [B, S] ids so emitted shapes match
        # get_pipeline_stage_metas.
        if pre_squeeze_input_ids is not None:
            mtp_embed_inputs = self._build_mtp_embed_inputs_for_pp(pre_squeeze_input_ids)

        if past_key_values is not None:
            past_key_values.has_previous_state = True

        if compute_logits:
            logits = compute_lm_head_logits(self.lm_head, hidden_states, logits_to_keep).logits
        else:
            if labels is not None or is_pp_stage:
                raise ValueError("compute_logits=False is only supported off-PP without labels.")
            logits = hidden_states.new_empty((*hidden_states.shape[:-1], 0))

        loss = None
        # PP path defers loss to PipelineCausalLMLoss; only compute here off-PP.
        if labels is not None and has_lm_head and not is_pp_stage:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        # MTP head: last PP stage in training only. Other stages / eval emit
        # placeholder empties below so the tuple arity stays 1 + D.
        mtp_per_depth_h: list[torch.Tensor] | None = None
        # Run the MTP heads in training, or in eval when explicitly requested for
        # validation acceptance metrics. Never on the cached generation path —
        # decoding feeds one token at a time and must not pay the MTP cost.
        if mtp_active:
            mtp_attention_mask = (
                causal_mask_mapping.get("full_attention") if causal_mask_mapping is not None else attention_mask
            )
            # Forward THD-packing context to the MTP sublayers; without it
            # they'd auto-detect bshd from the 3D shape and bleed attention
            # across sub-seq boundaries. cp_rank/cp_size must also be
            # forwarded — the CP forward pre-hook only attaches to the
            # backbone attention, not to MTP sublayers that share the class.
            mtp_kwargs = {
                "position_ids": position_ids,
                "attention_mask": mtp_attention_mask,
            }
            for _k in (
                "qkv_format",
                "cu_seqlens",
                "cu_seqlens_padded",
                "max_seqlen",
                "max_seqlen_q",
                "max_seqlen_kv",
                "seq_idx",
                "cp_rank",
                "cp_size",
            ):
                if _k in kwargs:
                    _v = kwargs[_k]
                    # Same per-microbatch [1, K] → 1D normalization as in
                    # NemotronV3Model.forward; re-applied for the MTP chain.
                    if _k in ("cu_seqlens", "cu_seqlens_padded", "seq_idx") and isinstance(_v, torch.Tensor):
                        if _v.dim() == 2 and _v.shape[0] == 1:
                            _v = _v.squeeze(0)
                        if _v.dim() == 1:
                            # Strip pad_and_stack's -1000 sentinels; TE would
                            # interpret pairs like (1024, -1000) as a negative-
                            # length sub-seq and OOB-write during backward.
                            if (_v == -1000).any():
                                _v = _v[_v != -1000]
                            _v = _v.contiguous().clone()
                    mtp_kwargs[_k] = _v
            # Squeeze to 2D ``[T, H]`` when THD-packed so the attention layer
            # selects its THD branch.  Also squeeze the propagated MTP embed
            # tensors for the same reason.
            mtp_hidden = hidden_states
            mtp_embeds_for_call = tuple(mtp_embed_inputs) if mtp_embed_inputs else ()
            mtp_input_ids_for_call = mtp_per_depth_input_ids
            mtp_position_ids_for_call = mtp_per_depth_position_ids
            # SALM / multimodal: when inputs_embeds (pre-fused audio+text) are present
            # and no PP embeddings were provided, pre-roll them per depth so audio
            # positions carry the correct audio embedding rather than embed_fn(padding_id).
            if not mtp_embeds_for_call and inputs_embeds is not None:
                from nemo_automodel.components.models.common.mtp import roll_tensor  # noqa: PLC0415

                cur_emb = inputs_embeds
                salm_embed_inputs: list[torch.Tensor] = []
                for _ in range(self.mtp_config.num_layers):
                    cur_emb = roll_tensor(cur_emb, shifts=-1, dim=-2)
                    salm_embed_inputs.append(cur_emb)
                mtp_embeds_for_call = tuple(salm_embed_inputs)
            if squeeze_for_thd:
                if mtp_hidden.dim() == 3 and mtp_hidden.shape[0] == 1:
                    mtp_hidden = mtp_hidden.squeeze(0)
                mtp_embeds_for_call = tuple(
                    e.squeeze(0) if (e.dim() == 3 and e.shape[0] == 1) else e for e in mtp_embeds_for_call
                )
                if mtp_input_ids_for_call is not None:
                    mtp_input_ids_for_call = tuple(
                        ids.squeeze(0) if (ids.dim() == 2 and ids.shape[0] == 1) else ids
                        for ids in mtp_input_ids_for_call
                    )
                if mtp_position_ids_for_call is not None:
                    mtp_position_ids_for_call = tuple(
                        p.squeeze(0) if (p.dim() == 2 and p.shape[0] == 1) else p for p in mtp_position_ids_for_call
                    )
            if mtp_position_ids_for_call is not None:
                mtp_kwargs["position_ids_per_depth"] = mtp_position_ids_for_call

            if mtp_embeds_for_call and mtp_input_ids_for_call is not None:
                raise ValueError("MTP per-depth input IDs cannot be combined with pre-computed embeddings")

            if mtp_embeds_for_call:
                # Final PP stage: embeddings produced upstream.
                mtp_per_depth_h = self.mtp(
                    hidden_states=mtp_hidden,
                    embed_inputs=mtp_embeds_for_call,
                    **mtp_kwargs,
                )
            elif mtp_input_ids_for_call is not None:
                # Context parallel: IDs were shifted globally, then sharded.
                mtp_per_depth_h = self.mtp(
                    hidden_states=mtp_hidden,
                    input_ids_per_depth=mtp_input_ids_for_call,
                    embed_fn=self.model.embed_tokens,
                    **mtp_kwargs,
                )
            else:
                # Non-PP single-rank: roll input_ids locally.
                mtp_per_depth_h = self.mtp(
                    hidden_states=mtp_hidden,
                    input_ids=input_ids,
                    embed_fn=self.model.embed_tokens,
                    **mtp_kwargs,
                )
            if squeeze_for_thd and mtp_per_depth_h is not None:
                mtp_per_depth_h = [h.unsqueeze(0) if h.dim() == 2 else h for h in mtp_per_depth_h]
        elif pp_mtp_enabled and has_lm_head:
            # Eval, or no MTP on this rank — emit empties to keep tuple arity.
            mtp_per_depth_h = [hidden_states.new_empty(hidden_states.shape) for _ in range(mtp_depth)]

        # Restore batch dim only when inner forward returned 2D — inputs_embeds
        # path already produces [1, T, V] and would become [1, 1, T, V] here.
        if is_thd and logits.dim() == 2:
            logits = logits.unsqueeze(0)

        # PP return contract:
        #   mid stages: (hidden_states, *mtp_embed_inputs)              arity 1+D
        #   last stage: (logits,        *mtp_per_depth_h,  seq_idx)     arity 1+D+1
        # The seq_idx tail binds the per-microbatch sub-seq layout to its loss
        # call via the PP runtime's forward(mb_i)→loss(mb_i) contract.
        if is_pp_stage:
            if pp_mtp_enabled:
                if not has_lm_head:
                    return (logits, *mtp_embed_inputs)
                assert mtp_per_depth_h is not None
                # seq_idx tail sources, in order of preference:
                #   1. kwargs["seq_idx"] (neat-path).
                #   2. derived from kwargs["cu_seqlens"] (THD/TE path).
                #   3. all-1 sentinel — loss-fn cross-boundary mask is a no-op.
                if logits.dim() == 3:
                    _B, _S = logits.shape[:2]
                elif logits.dim() == 2:
                    _B, _S = 1, logits.shape[0]
                else:
                    _B, _S = 1, hidden_states.shape[-2]

                _seq_idx_tail = kwargs.get("seq_idx", None)
                if not isinstance(_seq_idx_tail, torch.Tensor):
                    _cu = kwargs.get("cu_seqlens", None)
                    if isinstance(_cu, torch.Tensor):
                        _cu1d = _cu.squeeze(0) if (_cu.dim() == 2 and _cu.shape[0] == 1) else _cu
                        if _cu1d.dim() == 1:
                            _positions = torch.arange(_S, device=_cu1d.device)
                            # ``right=True`` so a position equal to a boundary
                            # (first token of sub-seq k, position == cu_seqlens[k])
                            # maps to k, not k-1 — matches mtp.py and layers.py.
                            _seq_idx_1d = torch.searchsorted(_cu1d[1:].contiguous(), _positions, right=True).to(
                                torch.int32
                            )
                            _seq_idx_tail = _seq_idx_1d.unsqueeze(0).expand(_B, _S).contiguous()
                if not isinstance(_seq_idx_tail, torch.Tensor):
                    _seq_idx_tail = torch.ones((_B, _S), dtype=torch.int32, device=logits.device)
                else:
                    if _seq_idx_tail.dim() == 1:
                        _seq_idx_tail = _seq_idx_tail.unsqueeze(0)
                    if _seq_idx_tail.dtype != torch.int32:
                        _seq_idx_tail = _seq_idx_tail.to(torch.int32)
                return (logits, *mtp_per_depth_h, _seq_idx_tail)
            return logits

        # Non-PP: dataclass return for the recipe's MTP loss reader.
        return NemotronHCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=(hidden_states,) if output_hidden_states else None,
            attentions=None,
            mtp_per_depth_h=mtp_per_depth_h,
            mtp_loss_scaling_factor=(self.mtp_config.loss_scaling_factor if mtp_per_depth_h is not None else None),
        )

    @staticmethod
    def _make_causal_mask(
        query_len: int,
        kv_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """Build a 4D SDPA-compatible causal mask.

        Prefill (query_len == kv_len): standard lower-triangular causal mask.
        Decode (query_len == 1): all-zeros row allowing attention to all cached positions.
        """
        if query_len == 1:
            # Decode: attend to all positions
            return torch.zeros(batch_size, 1, 1, kv_len, dtype=dtype, device=device)
        # Prefill: lower-triangular causal mask
        mask = torch.zeros(batch_size, 1, query_len, kv_len, dtype=dtype, device=device)
        mask.masked_fill_(
            torch.triu(torch.ones(query_len, kv_len, device=device), diagonal=1).bool(),
            float("-inf"),
        )
        return mask

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        past_key_values: Any | None = None,
        cache_position: torch.LongTensor | None = None,
        use_cache: bool | None = True,
        **kwargs,
    ) -> dict:
        """Prepare model inputs for each generation step.

        On the first call (prefill), creates a :class:`NemotronHybridCache` and
        forwards the full prompt.  On subsequent calls (decode), only the newly
        generated token is forwarded.

        Args:
            input_ids: Accumulated token ids [batch_size, current_seq_len].
            attention_mask: Padding mask [batch_size, current_seq_len].
            inputs_embeds: Pre-computed embeddings for the first step (optional).
            past_key_values: NemotronHybridCache from the previous step (None on first call).
            cache_position: Token position indices.
            use_cache: Whether to use caching (default True).
            **kwargs: Remaining model kwargs.

        Returns:
            Dict of keyword arguments to pass to :meth:`forward`.
        """
        from nemo_automodel.components.models.nemotron_v3.cache import NemotronHybridCache

        batch_size = input_ids.shape[0]

        # Create cache on first call, or replace non-NemotronHybridCache
        # (transformers v5.5+ GenerationMixin may pre-create a DynamicCache)
        if past_key_values is None or not isinstance(past_key_values, NemotronHybridCache):
            past_key_values = NemotronHybridCache(self.config, batch_size, self.dtype, self.device)
            # First call: cache_position covers the full prompt
            if cache_position is None:
                prompt_len = inputs_embeds.shape[1] if inputs_embeds is not None else input_ids.shape[1]
                cache_position = torch.arange(prompt_len, device=input_ids.device)

        # After prefill, send only the new token
        if past_key_values.has_previous_state:
            input_ids = input_ids[:, -1:]
            if cache_position is None:
                kv_len = past_key_values.get_seq_length()
                cache_position = torch.tensor([kv_len], device=input_ids.device)
            elif cache_position.ndim == 1 and cache_position.numel() > 1:
                # GenerationMixin may forward the full prompt positions on decode
                # even though only the last token is being decoded. Nemotron-v3's
                # Mamba cache update expects a single decode position here.
                cache_position = cache_position[-1:]

        # On the first step, prefer inputs_embeds when available
        if inputs_embeds is not None and not past_key_values.has_previous_state:
            model_inputs = {"input_ids": None, "inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        # Build causal mask for attention layers
        query_len = (
            input_ids.shape[1] if model_inputs["inputs_embeds"] is None else model_inputs["inputs_embeds"].shape[1]
        )
        kv_len = past_key_values.get_seq_length() + query_len
        causal_mask = self._make_causal_mask(query_len, kv_len, batch_size, self.dtype, self.device)

        model_inputs["causal_mask_mapping"] = {"full_attention": causal_mask}
        model_inputs["past_key_values"] = past_key_values
        model_inputs["cache_position"] = cache_position
        model_inputs["use_cache"] = use_cache
        model_inputs["attention_mask"] = attention_mask
        return model_inputs

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize model weights.

        PP-aware: skips ``lm_head`` and ``mtp`` initialization when those have
        been trimmed to ``None`` on a non-owning stage. ``self.model`` itself
        also internally guards ``embed_tokens`` and ``norm``.

        Args:
            buffer_device: Device to use for buffer initialization
            dtype: Target dtype for model weights
        """
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.initialize_weights(buffer_device=buffer_device)
            if self.lm_head is not None:
                nn.init.normal_(self.lm_head.weight, mean=0.0, std=self.config.initializer_range)
            if self.mtp is not None:
                for sublayer in self.mtp.layers:
                    sublayer.init_weights(buffer_device=buffer_device)

        cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params",))


ModelClass = NemotronHForCausalLM
