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
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig

from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.models.common import (
    BackendConfig,
    get_rope_config,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import cast_model_to_dtype, compute_lm_head_logits
from nemo_automodel.components.models.gpt_oss.rope_utils import RotaryEmbedding, position_ids_to_freqs_cis
from nemo_automodel.components.models.qwen3_next.layers import (
    Qwen3NextAttention,
    Qwen3NextFp32GatedDeltaNet,
    Qwen3NextRMSNorm,
)
from nemo_automodel.components.models.qwen3_next.state_dict_adapter import Qwen3NextStateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class Block(nn.Module):
    def __init__(self, layer_idx: int, config: Qwen3NextConfig, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            # fp32-aware GatedDeltaNet: keeps the intrinsically-fp32 A_log/dt_bias decay
            # gate in fp32 under FSDP mixed precision (see Qwen3NextFp32GatedDeltaNet).
            self.linear_attn = Qwen3NextFp32GatedDeltaNet(config, layer_idx)
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(config, layer_idx, backend)

        is_moe_layer = (
            (layer_idx not in getattr(config, "mlp_only_layers", []))
            and (getattr(config, "num_experts", 0) > 0)
            and ((layer_idx + 1) % getattr(config, "decoder_sparse_step", 1) == 0)
        )
        if is_moe_layer:
            self.mlp = MoE(moe_config, backend)
        else:
            # Thread dtype from config.torch_dtype so the dense MLP dtype stays
            # aligned with the rest of the model (fp32 under fp32 master weights).
            self.mlp = MLP(
                config.hidden_size,
                config.intermediate_size,
                backend.linear,
                dtype=get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16),
            )

        self.input_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        if self.layer_type == "linear_attention":
            attn_out = self.linear_attn(
                hidden_states=self.input_layernorm(x),
                attention_mask=attention_mask,
            )
        elif self.layer_type == "full_attention":
            attn_out = self.self_attn(
                x=self.input_layernorm(x),
                attention_mask=attention_mask,
                freqs_cis=freqs_cis,
                **attn_kwargs,
            )
        x = x + attn_out

        mlp_out = self._mlp(x=self.post_attention_layernorm(x), padding_mask=padding_mask)
        x = x + mlp_out
        return x

    def _mlp(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        # ``self.mlp`` may be wrapped by activation checkpointing (submodule-level
        # AC), so inspect the underlying module to pick the dense (no padding_mask)
        # vs MoE (padding_mask) call signature, but invoke the wrapped module.
        mlp = unwrap_checkpoint_wrapper(self.mlp)
        if isinstance(mlp, MLP):
            return self.mlp(x)
        else:
            assert isinstance(mlp, MoE)
            return self.mlp(x, padding_mask)

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        # Initialize attention weights
        if self.layer_type == "full_attention":
            self.self_attn.init_weights(buffer_device)
        elif self.layer_type == "linear_attention":
            self.linear_attn.dt_bias.data.fill_(1.0)
            self.linear_attn.A_log.data.uniform_(0, 16).log_()
            linear_list = [self.linear_attn.in_proj_qkvz, self.linear_attn.in_proj_ba, self.linear_attn.out_proj]
            for linear in linear_list:
                nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
            # transformers 5.15 dropped the FusedRMSNormGated branch, so this is always
            # Qwen3NextRMSNormGated, which has no reset_parameters(). Its weight is an
            # RMSNorm gain, initialized to ones.
            nn.init.ones_(self.linear_attn.norm.weight)
        self.mlp.init_weights(buffer_device)


class Qwen3NextModel(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
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

        # Resolve model dtype from config.torch_dtype once; thread it
        # explicitly into every sub-module so fp32 master weights work even
        # when construction is not wrapped in local_torch_dtype().
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        # Map HF Qwen3Next MoE config -> our MoE wrapper
        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.num_experts,
            n_shared_experts=1,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="softmax",  # Qwen3Next uses softmax topk routing
            route_scale=1.0,
            aux_loss_coeff=config.router_aux_loss_coef,
            norm_topk_prob=config.norm_topk_prob,
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

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=model_dtype)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype
        )

        # Rotary embedding cache compatible with our rope_utils functions
        self.max_seq_len = config.max_position_embeddings
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        base, rope_scaling, partial_rotary_factor = get_rope_config(config)

        self.rotary_emb = RotaryEmbedding(
            head_dim=self.head_dim,
            base=base,
            dtype=torch.float32,
            initial_context_length=rope_scaling.get("original_max_position_embeddings", 4096),
            scaling_factor=rope_scaling.get("factor", 1.0),
            ntk_alpha=rope_scaling.get("beta_slow", 1.0),
            ntk_beta=rope_scaling.get("beta_fast", 32.0),
            partial_rotary_factor=partial_rotary_factor,
            device=torch.device(f"cuda:{torch.cuda.current_device()}"),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if position_ids is None:
            position_ids = (
                torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
            )

        # Compute freqs_cis from RotaryEmbedding inv_freq and current position_ids; then concat [cos, sin]
        freqs_cis = position_ids_to_freqs_cis(
            self.rotary_emb,
            position_ids,
            qkv_format=attn_kwargs.get("qkv_format", "bshd"),
            for_fused_rope=self.backend.rope_fusion,
            cp_size=attn_kwargs.get("cp_size", 1),
        )

        h = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids

        for layer in self.layers.values():
            h = layer(
                x=h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                position_ids=position_ids,
                **attn_kwargs,
            )

        h = self.norm(h) if self.norm else h
        return h

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")

        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()
            # Ensure rotary embedding uses correct device after dtype move
            self.rotary_emb.device = buffer_device

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)


class Qwen3NextForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY

    _keep_in_fp32_modules_strict = ["_fp32_params"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = False
        supports_pp: bool = False
        supports_ep: bool = True

    @classmethod
    def from_config(
        cls,
        config: Qwen3NextConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        config = Qwen3NextConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: Qwen3NextConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = Qwen3NextModel(config, backend=self.backend, moe_config=moe_config, moe_overrides=moe_overrides)
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear, config.hidden_size, config.vocab_size, bias=False, dtype=model_dtype
        )
        keep_fp32 = list(getattr(self, "_keep_in_fp32_modules", None) or [])
        if "_fp32_params" not in keep_fp32:
            keep_fp32.append("_fp32_params")
        self._keep_in_fp32_modules = keep_fp32
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Qwen3NextStateDictAdapter(
                self.config, self.model.moe_config, self.backend, dtype=model_dtype
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: bool | None = None,
        **attn_kwargs: Any,
    ) -> CausalLMOutputWithPast:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        is_thd = attn_kwargs.get("qkv_format") == "thd"
        if is_thd:
            input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, attn_kwargs
            )
            attention_mask = None

        hidden = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            **attn_kwargs,
        )

        return compute_lm_head_logits(
            self.lm_head, hidden, logits_to_keep, is_thd=is_thd, output_hidden_states=output_hidden_states
        )

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            final_out_std = self.config.hidden_size**-0.5
            cutoff_factor = 3
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )

        cast_model_to_dtype(self, dtype, skip_modules=("_fp32_params",))
        with buffer_device:
            # Ensure rotary embedding uses correct device after dtype move
            self.model.rotary_emb.device = buffer_device


ModelClass = Qwen3NextForCausalLM
