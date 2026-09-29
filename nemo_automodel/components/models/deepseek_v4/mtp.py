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

"""DeepSeek V4 Multi-Token Prediction (MTP) blocks.

The released DSV4-Flash checkpoint stores MTP under ``mtp.{depth}.*``.  Each
MTP depth mirrors the reference ``MTPBlock``:

  - fuse the future-token embedding and the backbone HC stream with
    ``e_proj(embed) + h_proj(hidden)``;
  - run one HC-enabled DSV4 attention + MoE block;
  - collapse the HC stream with an MTP-local ``hc_head`` and ``norm`` before
    the shared LM head computes the auxiliary CE loss.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module, initialize_rms_norm_module
from nemo_automodel.components.models.common.mtp import MTPConfig, roll_tensor
from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4Attention,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE


class DeepseekV4MTPBlock(nn.Module):
    """One DSV4 MTP depth.

    Args:
        config: Main DSV4 config.
        layer_idx: Global layer index used by the attention implementation.
        moe_config: Shared MoE config.
        backend: BackendConfig for kernels/modules.
        dtype: Model dtype.
        rotary_emb: Shared main rotary embedding module.
        rotary_emb_compress: Shared compressor rotary embedding module.
    """

    def __init__(
        self,
        config,
        layer_idx: int,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype,
        rotary_emb,
        rotary_emb_compress,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        H = config.hidden_size
        eps = config.rms_norm_eps

        self.e_proj = initialize_linear_module(backend.linear, H, H, bias=False, dtype=dtype)
        self.h_proj = initialize_linear_module(backend.linear, H, H, bias=False, dtype=dtype)
        self.enorm = initialize_rms_norm_module(backend.rms_norm, H, eps=eps, dtype=dtype)
        self.hnorm = initialize_rms_norm_module(backend.rms_norm, H, eps=eps, dtype=dtype)

        mtp_attn_cfg = copy.copy(config)
        ratios = getattr(config, "compress_ratios", None)
        if ratios is None:
            mtp_attn_cfg.compress_ratios = None
        else:
            ratios = list(ratios)
            if layer_idx >= len(ratios):
                ratios.extend([0] * (layer_idx + 1 - len(ratios)))
            mtp_attn_cfg.compress_ratios = ratios
        self.self_attn = DeepseekV4Attention(mtp_attn_cfg, layer_idx=layer_idx, backend=backend)
        self.mlp = MoE(moe_config, backend)
        if int(getattr(config, "vision_n_layers", 0) or 0) > 0 and not backend.fake_balanced_gate:
            # Imported lazily because model.py imports this module lazily for the
            # same circular-dependency reason. Released vision checkpoints carry
            # ``bias_vl`` for every MTP gate even though MTP is inference-only in
            # the current visual path.
            from nemo_automodel.components.models.deepseek_v4.model import (  # noqa: PLC0415
                DeepseekV4VisionGate,
            )

            self.mlp.gate = DeepseekV4VisionGate(
                config,
                moe_config,
                gate_precision=backend.gate_precision,
                hash_routing=False,
            )
        self.input_layernorm = initialize_rms_norm_module(backend.rms_norm, H, eps=eps, dtype=dtype)
        self.post_attention_layernorm = initialize_rms_norm_module(backend.rms_norm, H, eps=eps, dtype=dtype)

        hc_kwargs = dict(
            hc_mult=config.hc_mult,
            hidden_size=H,
            hc_sinkhorn_iters=int(getattr(config, "hc_sinkhorn_iters", 20) or 20),
            hc_eps=float(config.hc_eps),
            rms_norm_eps=float(eps),
        )
        self.attn_hc = DeepseekV4HyperConnection(**hc_kwargs)
        self.ffn_hc = DeepseekV4HyperConnection(**hc_kwargs)
        self.hc_head = DeepseekV4HyperHead(
            hc_mult=config.hc_mult,
            hidden_size=H,
            hc_eps=float(config.hc_eps),
            rms_norm_eps=float(eps),
        )
        self.norm = initialize_rms_norm_module(backend.rms_norm, H, eps=eps, dtype=dtype)

        object.__setattr__(self, "_rotary_emb", rotary_emb)
        object.__setattr__(self, "_rotary_emb_compress", rotary_emb_compress)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        embed_input: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        **attn_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one MTP depth.

        Args:
            hidden_states: HC stream ``[B, S, hc_mult, H]``.
            embed_input: Future-token embeddings ``[B, S, H]``.

        Returns:
            Tuple of ``(next_hc_stream, prediction_hidden)`` where
            ``prediction_hidden`` is ``[B, S, H]`` and should be projected by
            the shared LM head for the MTP loss.
        """
        if hidden_states.dim() != 4:
            raise ValueError(f"DSV4 MTP expects HC hidden state [B,S,hc,H], got {tuple(hidden_states.shape)}")

        e = self.e_proj(self.enorm(embed_input)).unsqueeze(2)
        h = self.h_proj(self.hnorm(hidden_states))
        hidden_states = e + h

        if position_ids is None:
            seq_len = embed_input.shape[1]
            position_ids = (
                torch.arange(seq_len, device=embed_input.device).unsqueeze(0).expand(embed_input.shape[0], -1)
            )
        position_embeddings = self._rotary_emb(embed_input, position_ids)
        position_embeddings_compress = self._rotary_emb_compress(embed_input, position_ids)

        pre, post, comb = self.attn_hc(hidden_states)
        collapsed = (pre.unsqueeze(-1) * hidden_states).sum(dim=2).to(hidden_states.dtype)
        attn_out, _ = self.self_attn(
            hidden_states=self.input_layernorm(collapsed),
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_embeddings_compress=position_embeddings_compress,
            rotary_compress=self._rotary_emb_compress,
            position_ids=position_ids,
            **attn_kwargs,
        )
        dtype = hidden_states.dtype
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(
            comb.transpose(-1, -2).to(dtype), hidden_states
        )

        pre, post, comb = self.ffn_hc(hidden_states)
        collapsed = (pre.unsqueeze(-1) * hidden_states).sum(dim=2).to(hidden_states.dtype)
        set_routing_context = getattr(self.mlp.gate, "set_routing_context", None)
        if set_routing_context is not None:
            set_routing_context(input_ids, None)
        mlp_out = self.mlp(self.post_attention_layernorm(collapsed), padding_mask)
        hidden_states = post.to(dtype).unsqueeze(-1) * mlp_out.unsqueeze(-2) + torch.matmul(
            comb.transpose(-1, -2).to(dtype), hidden_states
        )

        prediction_hidden = self.norm(self.hc_head(hidden_states))
        return hidden_states, prediction_hidden

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        init_std = float(getattr(self.config, "initializer_range", 0.02))
        self.enorm.reset_parameters()
        self.hnorm.reset_parameters()
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.norm.reset_parameters()
        target_device = buffer_device or torch.device("cpu")
        with target_device:
            nn.init.trunc_normal_(self.e_proj.weight, mean=0.0, std=init_std)
            nn.init.trunc_normal_(self.h_proj.weight, mean=0.0, std=init_std)
        self.self_attn.init_weights(target_device, init_std=init_std)
        self.mlp.init_weights(target_device, init_std=init_std)
        init_dsv4_weights = getattr(self.mlp.gate, "init_dsv4_weights", None)
        if init_dsv4_weights is not None:
            init_dsv4_weights()
        self.attn_hc.init_weights(init_std)
        self.ffn_hc.init_weights(init_std)
        self.hc_head.init_weights(init_std)


class DeepseekV4MTPModule(nn.Module):
    """DSV4 MTP stack, one :class:`DeepseekV4MTPBlock` per prediction depth."""

    def __init__(
        self,
        config,
        mtp_config: MTPConfig,
        backend: BackendConfig,
        moe_config: MoEConfig,
        dtype: torch.dtype,
        rotary_emb,
        rotary_emb_compress,
    ):
        super().__init__()
        if not mtp_config.enabled:
            raise ValueError("DeepseekV4MTPModule constructed with disabled MTPConfig")
        self.mtp_config = mtp_config
        base_layer_idx = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [
                DeepseekV4MTPBlock(
                    config=config,
                    layer_idx=base_layer_idx + depth,
                    moe_config=moe_config,
                    backend=backend,
                    dtype=dtype,
                    rotary_emb=rotary_emb,
                    rotary_emb_compress=rotary_emb_compress,
                )
                for depth in range(mtp_config.num_layers)
            ]
        )

    @property
    def num_depths(self) -> int:
        return self.mtp_config.num_layers

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.LongTensor | None = None,
        embed_fn=None,
        embed_inputs: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
        position_ids: torch.LongTensor | None = None,
        **block_kwargs,
    ) -> list[torch.Tensor]:
        per_depth_h: list[torch.Tensor] = []
        cur_input_ids = input_ids
        if embed_inputs is not None and len(embed_inputs) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} MTP embedding tensors, got {len(embed_inputs)}")
        if embed_inputs is None and (cur_input_ids is None or embed_fn is None):
            raise ValueError("MTP requires either embed_inputs or both input_ids and embed_fn")

        for depth, block in enumerate(self.layers):
            if embed_inputs is None:
                cur_input_ids = roll_tensor(cur_input_ids, shifts=-1, dim=-1)
                decoder_input = embed_fn(cur_input_ids)
            else:
                decoder_input = embed_inputs[depth]
            kwargs = dict(block_kwargs)
            if position_ids is not None:
                kwargs["position_ids"] = position_ids
            hidden_states, prediction_hidden = block(
                hidden_states,
                embed_input=decoder_input,
                input_ids=cur_input_ids,
                **kwargs,
            )
            per_depth_h.append(prediction_hidden)
        return per_depth_h


def build_mtp_config_from_hf(config, *, loss_scaling_factor: float = 0.1) -> MTPConfig:
    """Build an MTPConfig from a DeepseekV4Config."""
    num_layers = int(getattr(config, "num_nextn_predict_layers", 0) or 0)
    return MTPConfig(
        num_layers=num_layers, layer_pattern="*" if num_layers > 0 else "", loss_scaling_factor=loss_scaling_factor
    )


def build_deepseek_v4_mtp(
    config,
    mtp_config: MTPConfig,
    backend: BackendConfig,
    moe_config: MoEConfig,
    dtype: torch.dtype,
    rotary_emb,
    rotary_emb_compress,
) -> DeepseekV4MTPModule:
    """Construct DSV4 MTP blocks."""
    return DeepseekV4MTPModule(
        config=config,
        mtp_config=mtp_config,
        backend=backend,
        moe_config=moe_config,
        dtype=dtype,
        rotary_emb=rotary_emb,
        rotary_emb_compress=rotary_emb_compress,
    )
