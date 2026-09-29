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

"""DeepSeek V4 Model.

Key architectural points (from official inference/model.py):

HC (Hyper-Connections):
  Every transformer block maintains hc_mult=4 copies of the hidden state.
  The embedding output is expanded: [B,S,dim] -> [B,S,hc_mult,dim].
  hc_pre  reduces [B,S,hc_mult,dim] -> [B,S,dim] before attn/ffn.
  hc_post expands [B,S,dim] -> [B,S,hc_mult,dim] after attn/ffn.
  Full HC requires the hc_split_sinkhorn CUDA kernel.
  Current fallback: mean-pooling for hc_pre, broadcast add for hc_post.

HC parameters (ALL layers, stored in float32):
  hc_attn_fn    : [mix_hc, hc_mult*dim]  where mix_hc = (2+hc_mult)*hc_mult = 24
  hc_attn_base  : [mix_hc]
  hc_attn_scale : [3]
  hc_ffn_fn     : [mix_hc, hc_mult*dim]
  hc_ffn_base   : [mix_hc]
  hc_ffn_scale  : [3]

Gate hash layers (layer_idx < num_hash_layers):
  Instead of score-based routing, the gate uses a fixed token-id -> expert-id
  lookup table (tid2eid: [vocab_size, n_activated_experts]).

All layers use MoE FFN (no dense layers).
Compress-ratio sliding-window attention is not yet implemented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Union

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.common.utils import (
    _has_dtensor_params,
    cast_model_to_dtype,
    compute_lm_head_logits,
)
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.cp import (
    build_dsv4_cp_causal_padding_mask,
    build_dsv4_cp_packed_causal_padding_mask,
    build_packed_seq_ids,
    dsv4_cp_enabled,
    dsv4_cp_local_seq_multiple,
    dsv4_cp_size,
    make_dsv4_contiguous_shard_cp_batch_and_ctx,
)
from nemo_automodel.components.models.deepseek_v4.fsdp import fully_shard_deepseek_v4
from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4Attention,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4RotaryEmbedding,
    _dsv4_sinkhorn_backend,
    build_causal_padding_mask,
    build_packed_causal_padding_mask,
)
from nemo_automodel.components.models.deepseek_v4.processing import (
    COMPRESS_PAD_TO,
    IMAGE,
    IMAGE_END,
    IMAGE_PAD,
    IMAGE_START,
    build_image_block,
)
from nemo_automodel.components.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter
from nemo_automodel.components.models.deepseek_v4.vision import (
    DeepseekV4VisionAligner,
    DeepseekV4VisionTransformer,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import Gate, MoE
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


@dataclass
class DeepseekV4CausalLMOutput(CausalLMOutputWithPast):
    """Output of DeepseekV4ForCausalLM.forward.

    Subclasses ``transformers.modeling_outputs.CausalLMOutputWithPast`` so the
    standard ``logits`` / ``hidden_states`` fields are present (the recipe's
    fused cross-entropy path requires ``"hidden_states" in out`` and reads the
    final hidden states off the output) while the DSV4-specific MTP fields are
    carried as declared dataclass fields. As required by ``ModelOutput``, every
    field after the first declares a ``None`` default.

    Attributes:
        logits: ``[B, S, vocab_size]`` next-token prediction logits.
        hidden_states: Final pre-lm_head hidden states ``[B, S, hidden]``
            (or ``[T, hidden]`` for packed THD), populated only when
            ``output_hidden_states`` is set; otherwise ``None``.
        mtp_per_depth_h: Per-depth MTP hidden states (training mode only).
            List of length ``num_nextn_predict_layers``, each ``[B, S, hidden]``.
            ``None`` when MTP is disabled or in eval mode.
        mtp_loss_scaling_factor: Coefficient for the MTP auxiliary loss.
    """

    mtp_per_depth_h: list[torch.Tensor] | None = None
    mtp_loss_scaling_factor: float | None = None


def _seq_lens_from_cu_seqlens(cu_seqlens: torch.Tensor, name: str) -> torch.Tensor:
    """Convert standard THD cumulative offsets to DSV4's per-row lengths."""
    if not isinstance(cu_seqlens, torch.Tensor) or cu_seqlens.dim() not in (1, 2):
        raise ValueError(f"`{name}` must be a rank-1 or rank-2 tensor.")
    if cu_seqlens.shape[-1] < 2:
        raise ValueError(f"`{name}` must contain at least the initial and final offsets.")

    seq_lens = torch.diff(cu_seqlens, dim=-1)
    return seq_lens.unsqueeze(0) if seq_lens.dim() == 1 else seq_lens


def _normalize_thd_packing_metadata(attn_kwargs: dict[str, Any]) -> None:
    """Accept standard THD offsets at the DSV4 model boundary.

    DSV4 internally uses ``seq_lens`` to build document-aware masks. Packed
    callers commonly provide the equivalent ``cu_seqlens`` representation, so
    normalize it here when context parallelism has not already produced native
    padded-BSHD lengths.
    """
    if attn_kwargs.get("qkv_format") != "thd":
        return

    if attn_kwargs.get("seq_lens") is None and attn_kwargs.get("cu_seqlens") is not None:
        attn_kwargs["seq_lens"] = _seq_lens_from_cu_seqlens(attn_kwargs["cu_seqlens"], "cu_seqlens")

    if attn_kwargs.get("seq_lens_padded") is None:
        cu_seqlens_padded = attn_kwargs.get("cu_seqlens_padded")
        if cu_seqlens_padded is not None:
            attn_kwargs["seq_lens_padded"] = _seq_lens_from_cu_seqlens(cu_seqlens_padded, "cu_seqlens_padded")
        elif attn_kwargs.get("seq_lens") is not None:
            attn_kwargs["seq_lens_padded"] = attn_kwargs["seq_lens"]


def apply_deepseek_v4_image_visibility(
    attention_mask: torch.Tensor,
    vision_token_types: torch.Tensor,
) -> torch.Tensor:
    """Make every token inside an image span mutually visible.

    Args:
        attention_mask: Additive causal/sliding mask with layout
            ``[batch, 1, sequence, sequence]``.
        vision_token_types: Pseudo-token types with layout ``[batch, sequence]``
            and ``-1`` at text positions.

    Returns:
        Additive mask with the same layout as ``attention_mask``. Text
        visibility remains causal/sliding; query/key pairs in the same complete
        ``IMAGE_START``...``IMAGE_END`` span are unmasked bidirectionally.
    """
    if vision_token_types.dim() == 1:
        vision_token_types = vision_token_types.unsqueeze(0)
    if attention_mask.dim() != 4 or attention_mask.shape[0] != vision_token_types.shape[0]:
        raise ValueError("Image visibility requires [batch, 1, sequence, sequence] attention masks")
    if attention_mask.shape[-2:] != (vision_token_types.shape[1], vision_token_types.shape[1]):
        raise ValueError("Image visibility does not support a sharded or non-square attention mask")

    is_start = vision_token_types == IMAGE_START
    is_end = vision_token_types == IMAGE_END
    starts_seen = is_start.cumsum(dim=1)
    ends_seen = is_end.cumsum(dim=1)
    if (ends_seen > starts_seen).any() or not torch.equal(starts_seen[:, -1], ends_seen[:, -1]):
        raise ValueError("Visual pseudo tokens contain an incomplete or misordered image span")
    valid = (starts_seen > ends_seen) | is_end
    same_span = starts_seen.unsqueeze(2) == starts_seen.unsqueeze(1)
    visible = valid.unsqueeze(2) & valid.unsqueeze(1) & same_span
    return attention_mask.masked_fill(visible.unsqueeze(1), 0.0)


class DeepseekV4Block(nn.Module):
    """Single transformer block for DeepSeek V4.

    Uses HuggingFace transformers PR 45616's HyperConnection decoder-layer
    pattern: two ``DeepseekV4HyperConnection`` modules own the collapse /
    expand mixer weights at the attention and FFN sites respectively.
    Checkpoint's flat ``hc_attn_*`` / ``hc_ffn_*`` keys are routed into
    ``attn_hc.*`` / ``ffn_hc.*`` by the state-dict adapter.
    """

    def __init__(
        self,
        layer_idx: int,
        config: DeepseekV4Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hc_mult = config.hc_mult

        model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
        self.self_attn = DeepseekV4Attention(config, layer_idx=layer_idx, backend=backend)
        self.mlp = MoE(moe_config, backend)
        # Hash routing: the first ``num_hash_layers`` layers use a fixed
        # tid2eid lookup table instead of the score-based generic Gate.
        # Swap after MoE construction so the rest of MoE (experts, shared
        # experts, etc.) keeps its standard layout.
        self.is_hash_routing_layer = layer_idx < int(getattr(config, "num_hash_layers", 0) or 0)
        self.is_vision_routing = int(getattr(config, "vision_n_layers", 0) or 0) > 0
        if self.is_vision_routing and not backend.fake_balanced_gate:
            self.mlp.gate = DeepseekV4VisionGate(
                config,
                moe_config,
                gate_precision=backend.gate_precision,
                hash_routing=self.is_hash_routing_layer,
            )
        elif self.is_hash_routing_layer and not backend.fake_balanced_gate:
            self.mlp.gate = DeepseekV4HashGate(config, moe_config)
        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype
        )

        # HC (Hyper-Connection) mixers — one per sub-site (attention + FFN).
        # Each owns learnable ``fn`` (fp32 packed-linear), ``base`` (fp32 bias),
        # ``scale`` (fp32 per-head gain) parameters.  ``_keep_in_fp32_modules_strict``
        # on ``DeepseekV4ForCausalLM`` keeps all nine HC param tensors in fp32
        # at runtime via submodule-name matching.
        hc_kwargs = dict(
            hc_mult=config.hc_mult,
            hidden_size=config.hidden_size,
            hc_sinkhorn_iters=int(getattr(config, "hc_sinkhorn_iters", 20) or 20),
            hc_eps=float(config.hc_eps),
            rms_norm_eps=float(config.rms_norm_eps),
            sinkhorn_backend=_dsv4_sinkhorn_backend(backend),
        )
        self.attn_hc = DeepseekV4HyperConnection(**hc_kwargs)
        self.ffn_hc = DeepseekV4HyperConnection(**hc_kwargs)

    def forward(
        self,
        x: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor | None = None,
        position_embeddings_compress: tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_compress: nn.Module | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        vision_token_types: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Transform one HC block.

        Args:
            x: HC streams with layout ``[batch, sequence, hc_mult, hidden]``.
            position_embeddings: Main RoPE tensors with layout compatible with
                ``[batch, sequence, qk_rope_head_dim]``.
            position_ids: Token positions with layout ``[batch, sequence]``.
            position_embeddings_compress: Optional compressor RoPE tensors.
            rotary_compress: Optional compressor rotary module.
            attention_mask: Additive attention mask with layout
                ``[batch, 1, sequence, sequence]``.
            padding_mask: Boolean padding mask with layout ``[batch, sequence]``.
            input_ids: Token IDs with layout ``[batch, sequence]``.
            vision_token_types: Visual pseudo-token types with layout
                ``[batch, sequence]`` and ``-1`` at text positions.

        Returns:
            HC streams with layout ``[batch, sequence, hc_mult, hidden]``.
        """
        # x throughout this layer: [B, S, hc_mult, hidden] (HC multi-copy state)
        # padding_mask is only used by the MoE module; only derive it from a 2D
        # raw attention_mask (1=valid, 0=pad).  When attention_mask is the 4D
        # additive mask built upstream, the caller is expected to supply
        # padding_mask separately (or leave it None for the no-pad case).
        if attention_mask is not None and padding_mask is None and attention_mask.dim() == 2:
            padding_mask = attention_mask.bool().logical_not()

        def attention_site(hidden_streams: torch.Tensor) -> torch.Tensor:
            pre, post, comb = self.attn_hc(hidden_streams)
            collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
            attn_out, _ = self.self_attn(
                hidden_states=self.input_layernorm(collapsed),
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_embeddings_compress=position_embeddings_compress,
                rotary_compress=rotary_compress,
                position_ids=position_ids,
                vision_token_types=vision_token_types,
                **attn_kwargs,
            )
            dtype = hidden_streams.dtype
            # Expand: native DSV4 uses comb[j, h] * residual[j], i.e. comb.T @ residual.
            return post.to(dtype).unsqueeze(-1) * attn_out.unsqueeze(-2) + torch.matmul(
                comb.transpose(-1, -2).to(dtype), hidden_streams
            )

        def ffn_prepare(hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            pre, post, comb = self.ffn_hc(hidden_streams)
            collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
            return collapsed, post, comb

        x = attention_site(x)
        collapsed, post, comb = ffn_prepare(x)

        # Hash-routing layers need the current batch's input_ids to do the
        # tid2eid lookup; stash it on the gate just before the MoE call.
        gate = getattr(self.mlp, "gate", None)
        if isinstance(gate, DeepseekV4VisionGate):
            gate.set_routing_context(input_ids, vision_token_types)
        elif self.is_hash_routing_layer and isinstance(gate, DeepseekV4HashGate):
            gate.set_input_ids(input_ids)
        mlp_out = self.mlp(self.post_attention_layernorm(collapsed), padding_mask)
        dtype = x.dtype
        return post.to(dtype).unsqueeze(-1) * mlp_out.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2).to(dtype), x)

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.input_layernorm.reset_parameters()
        self.post_attention_layernorm.reset_parameters()
        self.self_attn.init_weights(buffer_device, init_std=init_std)
        self.mlp.init_weights(buffer_device, init_std=init_std)
        if isinstance(self.mlp.gate, DeepseekV4VisionGate):
            self.mlp.gate.init_dsv4_weights()
        if isinstance(self.mlp.gate, DeepseekV4HashGate):
            self.mlp.gate.init_weights(init_std=init_std)
        self.attn_hc.init_weights(init_std)
        self.ffn_hc.init_weights(init_std)


class DeepseekV4HashGate(nn.Module):
    """Hash gate for first num_hash_layers: routes tokens via a fixed lookup table.

    Instead of computing routing scores, the gate uses tid2eid[token_id] to
    pre-assign expert indices.  The routing weight is still computed from the
    gate weight but the *selection* is deterministic per token id.

    tid2eid shape: [vocab_size, n_activated_experts]  (int64 runtime, non-trainable)

    Signature matches ``components.moe.layers.Gate`` — ``forward(x, token_mask,
    cp_mesh)`` returning ``(weights, indices, aux_loss)`` — so the generic MoE
    module can call it interchangeably.  The per-forward ``input_ids`` needed
    for the tid2eid lookup is stashed on the module by the enclosing Block via
    :meth:`set_input_ids` immediately before the MoE call.
    """

    def __init__(self, config: DeepseekV4Config, moe_config: MoEConfig):
        super().__init__()
        self.topk = moe_config.n_activated_experts
        self.n_experts = moe_config.n_routed_experts
        self.score_func = moe_config.score_func
        self.route_scale = moe_config.route_scale
        self.norm_topk_prob = moe_config.norm_topk_prob

        # Routing score weight (used to compute weights, not for selection)
        self.weight = nn.Parameter(torch.zeros(self.n_experts, config.hidden_size))
        # Token-id -> expert-id lookup table.  Registered as a persistent
        # buffer (not a Parameter) because FSDP's param-sharding path rejects
        # int tensors via .requires_grad_(), and the table is non-trainable
        # anyway.  DeepEP expects runtime expert indices to be int64; the
        # checkpoint adapter may load the on-disk I32 table into this buffer.
        self.register_buffer(
            "tid2eid",
            torch.zeros(config.vocab_size, self.topk, dtype=torch.int64),
            persistent=True,
        )
        # Kept for API compat with the generic Gate (e.g. optimizer sync paths
        # that probe for .bias) — hash layers have no learnable bias.
        self.bias = None
        # Ephemeral per-forward input_ids set by the Block (not a parameter /
        # buffer; cleared after each forward to avoid holding references).
        self._pending_input_ids: torch.Tensor | None = None

    def set_input_ids(self, input_ids: torch.Tensor | None) -> None:
        """Stash the current batch's input_ids for the next ``forward`` call."""
        self._pending_input_ids = input_ids

    def update_bias(self) -> None:
        """No-op for compat with callers that walk MoE gates and call update_bias."""

    def init_weights(self, init_std: float = 0.02) -> None:
        """Initialize the trainable gate and a valid deterministic hash table.

        Args:
            init_std: Standard deviation for the routing weight initialization.
        """
        nn.init.normal_(self.weight, mean=0.0, std=init_std)
        with torch.no_grad():
            token_ids = torch.arange(self.tid2eid.shape[0], device=self.tid2eid.device).unsqueeze(1)
            expert_offsets = torch.arange(self.topk, device=self.tid2eid.device).unsqueeze(0)
            self.tid2eid.copy_((token_ids * self.topk + expert_offsets) % self.n_experts)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        cp_mesh: "DeviceMesh | None" = None,  # noqa: F821 — MoE passes it but we do not need it
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        import torch.nn.functional as F

        input_ids = self._pending_input_ids
        # Clear immediately so a stale cached tensor cannot leak to a later
        # forward that forgets to set it.
        self._pending_input_ids = None

        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "sqrtsoftplus":
            scores = F.softplus(scores).sqrt()
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = scores.softmax(dim=-1)

        if input_ids is not None:
            indices = self.tid2eid[input_ids.flatten().to(torch.int64)]
        else:
            # Fallback to score-based topk — keeps the module usable in tests or
            # PP stages where input_ids is not threaded through.
            indices = scores.topk(self.topk, dim=-1)[1]

        weights = scores.gather(1, indices.long())
        if self.score_func != "softmax":
            denom = weights.sum(dim=-1, keepdim=True) + 1e-20
            weights = weights / denom
        weights = weights * self.route_scale
        return weights.type_as(x), indices, None


class DeepseekV4VisionGate(Gate):
    """DSV4 gate with separate visual bias and optional text hash routing.

    The released vision checkpoint routes visual pseudo tokens by score in all
    layers. In the first hash layers only text tokens use ``tid2eid``; visual
    tokens use ``scores + bias_vl``. Later layers select text experts with the
    normal correction bias and visual experts with ``bias_vl``.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        moe_config: MoEConfig,
        *,
        gate_precision: torch.dtype | None,
        hash_routing: bool,
    ) -> None:
        super().__init__(moe_config, gate_precision=gate_precision)
        self.vocab_size = int(config.vocab_size)
        self.hash_routing = hash_routing
        # Selection biases do not participate in autograd (top-k indices are
        # discrete), matching Automodel's normal score-correction bias. Keep
        # this checkpoint tensor as an fp32 buffer so FSDP does not create a
        # mixed-storage gate parameter group.
        self.register_buffer(
            "bias_vl",
            torch.zeros(self.n_experts, dtype=torch.float32),
            persistent=True,
        )
        if hash_routing:
            self.register_buffer(
                "tid2eid",
                torch.zeros(self.vocab_size, self.topk, dtype=torch.int64),
                persistent=True,
            )
        else:
            self.tid2eid = None
        self._pending_input_ids: torch.Tensor | None = None
        self._pending_vision_token_types: torch.Tensor | None = None

    def set_routing_context(
        self,
        input_ids: torch.Tensor | None,
        vision_token_types: torch.Tensor | None,
    ) -> None:
        """Set token metadata consumed by the next gate call.

        Args:
            input_ids: Token IDs with layout ``[batch, sequence]``.
            vision_token_types: Visual types with layout ``[batch, sequence]``
                and ``-1`` for text tokens.
        """
        self._pending_input_ids = input_ids
        self._pending_vision_token_types = vision_token_types

    @staticmethod
    def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
        """Return a local ``[experts]`` tensor from a tensor or DTensor."""
        return tensor.to_local() if isinstance(tensor, DTensor) else tensor

    def _route_scores(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select experts from router logits of layout ``[tokens, experts]``."""
        if self.score_func != "sqrtsoftplus":
            raise ValueError(f"DeepSeek-V4 Vision requires sqrtsoftplus routing, got {self.score_func}")

        input_ids = self._pending_input_ids
        vision_token_types = self._pending_vision_token_types
        self._pending_input_ids = None
        self._pending_vision_token_types = None

        original_scores = torch.sqrt(torch.nn.functional.softplus(scores.float()))
        num_tokens = original_scores.shape[0]
        if vision_token_types is not None:
            vision_mask = vision_token_types.flatten() >= 0
        elif input_ids is not None:
            vision_mask = input_ids.flatten() >= self.vocab_size
        else:
            vision_mask = torch.zeros(num_tokens, dtype=torch.bool, device=original_scores.device)
        if vision_mask.numel() != num_tokens:
            raise ValueError(f"Routing metadata has {vision_mask.numel()} tokens but gate received {num_tokens}")

        visual_bias = self._local_tensor(self.bias_vl).to(device=original_scores.device)
        if self.hash_routing and input_ids is not None:
            flat_ids = input_ids.flatten().to(device=original_scores.device, dtype=torch.long)
            if flat_ids.numel() != num_tokens:
                raise ValueError(f"input_ids has {flat_ids.numel()} tokens but gate received {num_tokens}")
            safe_ids = torch.where(vision_mask, torch.zeros_like(flat_ids), flat_ids)
            if ((safe_ids < 0) | (safe_ids >= self.vocab_size)).any():
                raise ValueError("Text token IDs are outside the DeepSeek-V4 vocabulary")
            hash_indices = self.tid2eid[safe_ids]
            visual_indices = (original_scores + visual_bias).topk(self.topk, dim=-1)[1]
            indices = torch.where(vision_mask.unsqueeze(-1), visual_indices.to(hash_indices.dtype), hash_indices)
        elif self.hash_routing:
            indices = (original_scores + visual_bias).topk(self.topk, dim=-1)[1]
        else:
            correction_bias = self._local_score_correction_bias()
            if correction_bias is None:
                correction_bias = torch.zeros_like(visual_bias)
            else:
                correction_bias = correction_bias.to(device=original_scores.device)
            selection_bias = torch.where(vision_mask.unsqueeze(-1), visual_bias, correction_bias)
            indices = (original_scores + selection_bias).topk(self.topk, dim=-1)[1]

        weights = original_scores.gather(1, indices.long())
        if self.norm_topk_prob and self.topk > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
            original_scores = original_scores / (original_scores.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * self.route_scale
        return weights, indices, original_scores

    def init_dsv4_weights(self) -> None:
        """Initialize visual bias and a deterministic checkpoint-free hash map."""
        with torch.no_grad():
            self._local_tensor(self.bias_vl).zero_()
            if self.tid2eid is not None:
                token_ids = torch.arange(self.tid2eid.shape[0], device=self.tid2eid.device).unsqueeze(1)
                expert_offsets = torch.arange(self.topk, device=self.tid2eid.device).unsqueeze(0)
                self.tid2eid.copy_((token_ids * self.topk + expert_offsets) % self.n_experts)


class DeepseekV4Model(nn.Module):
    def __init__(
        self,
        config: DeepseekV4Config,
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

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.moe_intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.num_experts_per_tok,
            # V4 has no group-limited routing (noaux_tc with no n_group/topk_group)
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=1e-3,
            score_func="sqrtsoftplus",
            route_scale=config.routed_scaling_factor,
            aux_loss_coeff=0,
            norm_topk_prob=config.norm_topk_prob,
            dtype=get_dtype(config.torch_dtype, torch.bfloat16),
            # V4 Flash routed experts use clamped SwiGLU (gate.max=limit,
            # up.±limit) in FP32 — see reference model.py Expert.forward.
            swiglu_limit=float(getattr(config, "swiglu_limit", 0.0) or 0.0),
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, dtype=get_dtype(config.torch_dtype, torch.bfloat16)
        )
        self.vision_enabled = int(getattr(config, "vision_n_layers", 0) or 0) > 0
        if self.vision_enabled:
            model_dtype = get_dtype(config.torch_dtype, torch.bfloat16)
            self.vision = DeepseekV4VisionTransformer(config)
            self.aligner = DeepseekV4VisionAligner(config)
            self.image_start = nn.Parameter(torch.empty(config.hidden_size, dtype=model_dtype))
            self.image_end = nn.Parameter(torch.empty(config.hidden_size, dtype=model_dtype))
            self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=model_dtype))
            self.image_pad = nn.Parameter(torch.empty(config.hidden_size, dtype=model_dtype))
        else:
            self.vision = None
            self.aligner = None
            self.register_parameter("image_start", None)
            self.register_parameter("image_end", None)
            self.register_parameter("image_newline", None)
            self.register_parameter("image_pad", None)
        self.layers = nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = DeepseekV4Block(layer_id, config, self.moe_config, backend)

        # Final HC collapse: sigmoid-weighted sum across hc_mult streams before
        # the shared RMSNorm + lm_head.  Ported from HF PR 45616's
        # ``DeepseekV4HyperHead``.  Owns ``hc_fn`` / ``hc_base`` / ``hc_scale``
        # — all kept in fp32 via ``_keep_in_fp32_modules_strict`` (see
        # ``DeepseekV4ForCausalLM``).
        self.hc_head = DeepseekV4HyperHead(
            hc_mult=config.hc_mult,
            hidden_size=config.hidden_size,
            hc_eps=float(config.hc_eps),
            rms_norm_eps=float(config.rms_norm_eps),
        )

        self.norm = initialize_rms_norm_module(
            backend.rms_norm,
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=get_dtype(config.torch_dtype, torch.bfloat16),
        )

        self.max_seq_len = config.max_position_embeddings
        # Two rotary embeddings (HF PR 45616 pattern): main rope for the token
        # attention path, compressor rope for the long-range pooled KV branch.
        # HF partial_rotary_factor = qk_rope_head_dim / head_dim so cos/sin
        # come out sized to qk_rope_head_dim.
        partial_rotary_factor = float(config.qk_rope_head_dim) / float(config.head_dim)
        # Reference (``dsv4flash/inference/model.py:519-525``) only applies YaRN
        # to the compress-rope path: when compress_ratio>0 it uses
        # ``original_seq_len=args.original_seq_len`` and theta=compress_rope_theta;
        # otherwise ``original_seq_len=0`` (YaRN disabled) and theta=rope_theta.
        rope_scaling = getattr(config, "rope_scaling", None)
        self.rotary_emb = DeepseekV4RotaryEmbedding(
            rope_theta=float(config.rope_theta),
            head_dim=int(config.head_dim),
            partial_rotary_factor=partial_rotary_factor,
            rope_scaling=None,
        )
        self.rotary_emb_compress = DeepseekV4RotaryEmbedding(
            rope_theta=float(getattr(config, "compress_rope_theta", 160000.0) or 160000.0),
            head_dim=int(config.head_dim),
            partial_rotary_factor=partial_rotary_factor,
            rope_scaling=rope_scaling,
        )

    def encode_image(self, patches: torch.Tensor, n_vit_h: int, n_vit_w: int) -> torch.Tensor:
        """Encode one image from ViT patches into LLM-width grid features.

        Args:
            patches: Image patches with layout
                ``[n_vit_h * n_vit_w, 3, patch_size, patch_size]``.
            n_vit_h: Number of patch rows.
            n_vit_w: Number of patch columns.

        Returns:
            Aligned features with layout
            ``[ceil(n_vit_h / ratio) * ceil(n_vit_w / ratio), hidden]``.
        """
        if self.vision is None or self.aligner is None:
            raise ValueError("Image inputs require a DeepSeek-V4 vision checkpoint")
        return self.aligner(self.vision(patches, n_vit_h, n_vit_w), n_vit_h, n_vit_w)

    def merge_image_embeddings(
        self,
        inputs_embeds: torch.Tensor,
        *,
        pixel_values: torch.Tensor,
        image_grid_hws: torch.Tensor,
        vision_token_types: torch.Tensor,
        n_images_per_sample: torch.Tensor | None,
    ) -> torch.Tensor:
        """Replace pseudo-token embeddings with encoded images and sentinels.

        Args:
            inputs_embeds: Text embeddings with layout ``[batch, sequence, hidden]``.
            pixel_values: Concatenated patches with layout
                ``[all_patches, 3, patch_size, patch_size]``.
            image_grid_hws: ViT grid sizes with layout ``[all_images, 2]``.
            vision_token_types: Pseudo-token types with layout
                ``[batch, sequence]`` and ``-1`` for text.
            n_images_per_sample: Optional counts with layout ``[batch]``.

        Returns:
            Embeddings with the same layout as ``inputs_embeds``.
        """
        if image_grid_hws.dim() != 2 or image_grid_hws.shape[-1] != 2:
            raise ValueError(f"image_grid_hws must have shape [images, 2], got {tuple(image_grid_hws.shape)}")
        if vision_token_types.shape != inputs_embeds.shape[:2]:
            raise ValueError("vision_token_types must match the embedding batch and sequence dimensions")

        image_starts = [
            (vision_token_types[sample_idx] == IMAGE_START).nonzero(as_tuple=False).flatten()
            for sample_idx in range(inputs_embeds.shape[0])
        ]
        inferred_counts = torch.tensor(
            [positions.numel() for positions in image_starts], dtype=torch.long, device=vision_token_types.device
        )
        if n_images_per_sample is not None:
            counts = n_images_per_sample.to(device=inferred_counts.device, dtype=torch.long)
            if counts.shape != inferred_counts.shape or not torch.equal(counts, inferred_counts):
                raise ValueError(
                    f"n_images_per_sample={counts.tolist()} does not match pseudo-token spans={inferred_counts.tolist()}"
                )
        if int(inferred_counts.sum().item()) != image_grid_hws.shape[0]:
            raise ValueError(
                f"Found {int(inferred_counts.sum().item())} image spans but got {image_grid_hws.shape[0]} grids"
            )

        sentinels = torch.stack([self.image_start, self.image_pad, self.image_pad, self.image_newline, self.image_end])
        flat_indices: list[torch.Tensor] = []
        flat_values: list[torch.Tensor] = []
        patch_offset = 0
        image_index = 0
        sequence_length = inputs_embeds.shape[1]
        for sample_idx, starts in enumerate(image_starts):
            sample_types = vision_token_types[sample_idx]
            for start_tensor in starts:
                start = int(start_tensor.item())
                block_start = start
                while (
                    block_start > 0
                    and start - block_start < COMPRESS_PAD_TO - 1
                    and int(sample_types[block_start - 1].item()) == IMAGE_PAD
                ):
                    block_start -= 1
                ends = (sample_types[start:] == IMAGE_END).nonzero(as_tuple=False).flatten()
                if ends.numel() == 0:
                    raise ValueError("Visual pseudo-token block is missing IMAGE_END")
                block_end = start + int(ends[0].item())

                n_vit_h = int(image_grid_hws[image_index, 0].item())
                n_vit_w = int(image_grid_hws[image_index, 1].item())
                n_patches = n_vit_h * n_vit_w
                patch_slice = pixel_values[patch_offset : patch_offset + n_patches]
                if patch_slice.shape[0] != n_patches:
                    raise ValueError(f"Image {image_index} requires {n_patches} patches but the batch ended early")
                aligned = self.encode_image(patch_slice, n_vit_h, n_vit_w)
                n_llm_h = math.ceil(n_vit_h / int(self.config.vision_downsample_ratio))
                n_llm_w = math.ceil(n_vit_w / int(self.config.vision_downsample_ratio))
                expected_types, perm = build_image_block(n_llm_h, n_llm_w, block_start)
                block_types = sample_types[block_start : block_end + 1]
                expected_types = expected_types.to(device=block_types.device)
                if not torch.equal(block_types, expected_types):
                    raise ValueError("Visual pseudo-token block does not match the DeepSeek-V4 N-layout")

                block = sentinels[block_types]
                image_positions = (block_types == IMAGE).nonzero(as_tuple=False).flatten()
                perm = perm.to(device=aligned.device)
                block = block.index_copy(0, image_positions, aligned[perm])
                positions = torch.arange(block_start, block_end + 1, device=inputs_embeds.device)
                flat_indices.append(positions + sample_idx * sequence_length)
                flat_values.append(block)
                patch_offset += n_patches
                image_index += 1

        if patch_offset != pixel_values.shape[0]:
            raise ValueError(f"Consumed {patch_offset} image patches but received {pixel_values.shape[0]}")
        if not flat_indices:
            return inputs_embeds
        indices = torch.cat(flat_indices)
        values = torch.cat(flat_values).to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        merged = inputs_embeds.flatten(0, 1).index_copy(0, indices, values)
        return merged.view_as(inputs_embeds)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_hws: torch.Tensor | None = None,
        n_images_per_sample: torch.Tensor | None = None,
        vision_token_types: torch.Tensor | None = None,
        return_hc_hidden: bool = False,
        **attn_kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run the DSV4 text backbone with an optional image embedding bridge.

        Args:
            input_ids: Token IDs with layout ``[batch, sequence]`` on the first
                PP stage, or HC activations on later stages.
            inputs_embeds: Optional embeddings with layout
                ``[batch, sequence, hidden]``.
            position_ids: Positions with layout ``[batch, sequence]``.
            attention_mask: Valid-token mask with layout ``[batch, sequence]``.
            padding_mask: Padding mask with layout ``[batch, sequence]``.
            pixel_values: Concatenated image patches with layout
                ``[all_patches, 3, patch_size, patch_size]``.
            image_grid_hws: Patch grids with layout ``[all_images, 2]``.
            n_images_per_sample: Image counts with layout ``[batch]``.
            vision_token_types: Pseudo-token types with layout
                ``[batch, sequence]`` and ``-1`` at text positions.
            return_hc_hidden: Whether to also return the uncollapsed HC stream.

        Returns:
            Hidden states with layout ``[batch, sequence, hidden]`` and,
            optionally, HC states with layout
            ``[batch, sequence, hc_mult, hidden]``.
        """
        # PP-aware forward (same pattern as DeepseekV3Model.forward).
        # Stage 0 of pipeline parallelism owns ``embed_tokens`` and receives
        # raw token ids; subsequent stages have ``embed_tokens=None`` and
        # receive the previous stage's hidden state in the ``input_ids`` slot
        # (already 4D ``[B, S, hc_mult, hidden]`` because ``DeepseekV4Block``
        # preserves the HC stream axis).  Detect via ``self.embed_tokens is None``
        # rather than via dtype, since the stage trimming pass nulls the
        # attribute when the layer is dropped.
        on_first_stage = self.embed_tokens is not None

        if on_first_stage:
            if input_ids is None and inputs_embeds is None:
                raise ValueError("First PP stage requires input_ids or inputs_embeds")
            if vision_token_types is None and input_ids is not None:
                vision_token_types = torch.where(
                    input_ids >= int(self.config.vocab_size),
                    input_ids - int(self.config.vocab_size),
                    torch.full_like(input_ids, -1),
                )
            if vision_token_types is not None and vision_token_types.dim() == 1:
                vision_token_types = vision_token_types.unsqueeze(0)
            has_visual_tokens = vision_token_types is not None and bool((vision_token_types >= 0).any().item())
            if not has_visual_tokens:
                vision_token_types = None
            if inputs_embeds is None:
                if (input_ids < 0).any():
                    raise ValueError("DeepSeek-V4 input_ids cannot be negative")
                safe_input_ids = torch.where(
                    input_ids < int(self.config.vocab_size), input_ids, torch.zeros_like(input_ids)
                )
                inputs_embeds = self.embed_tokens(safe_input_ids)
            if has_visual_tokens:
                if not self.vision_enabled:
                    raise ValueError("Visual pseudo tokens require a DeepSeek-V4 vision checkpoint")
                if pixel_values is None or image_grid_hws is None:
                    raise ValueError("Visual pseudo tokens require pixel_values and image_grid_hws")
                if inputs_embeds.dim() != 3:
                    raise ValueError("DeepSeek-V4 visual inputs do not support packed THD embeddings")
                inputs_embeds = self.merge_image_embeddings(
                    inputs_embeds,
                    pixel_values=pixel_values,
                    image_grid_hws=image_grid_hws,
                    vision_token_types=vision_token_types,
                    n_images_per_sample=n_images_per_sample,
                )
            elif pixel_values is not None or image_grid_hws is not None:
                raise ValueError("Image tensors were provided without visual pseudo tokens")
            # Packed-sequence (THD) inputs arrive with the batch axis collapsed
            # (``process_input_for_thd`` flattens ids to ``[T]`` -> embeds ``[T, dim]``).
            # Restore the leading batch dim so the hc_mult expand below sees the
            # expected ``[B, S, dim]`` rank, mirroring the output-side THD
            # ``unsqueeze(0)`` in ``compute_lm_head_logits(is_thd=True)``.
            if inputs_embeds.dim() == 2:
                inputs_embeds = inputs_embeds.unsqueeze(0)
            # Expand embeddings to hc_mult copies: [B,S,dim] -> [B,S,hc_mult,dim]
            h = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
            shape_ref = inputs_embeds  # 3D ref for rotary / mask sizing
        else:
            # Mid-stage: ``input_ids`` is actually the upstream activation.
            # Either positional (4D float) or via ``inputs_embeds=`` kwarg.
            h = input_ids if input_ids is not None else inputs_embeds
            if h is None:
                raise ValueError("Non-first PP stage expects an inter-stage activation")
            # h is [B, S, hc_mult, hidden]; shape_ref needs 3D [B, S, hidden].
            shape_ref = h.flatten(start_dim=2)[:, :, : self.config.hidden_size]

        if position_ids is not None and position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        if position_ids is None:
            seq_len = shape_ref.shape[1]
            position_ids = torch.arange(seq_len, device=shape_ref.device).unsqueeze(0).expand(shape_ref.shape[0], -1)
        elif position_ids.shape[0] == 1 and shape_ref.shape[0] > 1:
            position_ids = position_ids.expand(shape_ref.shape[0], -1).contiguous()

        # (cos, sin) pairs for the main attention path and the compressor path.
        # Rotary modules live on every stage (PP keep-list ensures it).
        position_embeddings = self.rotary_emb(shape_ref, position_ids)
        position_embeddings_compress = self.rotary_emb_compress(shape_ref, position_ids)

        # Build the 4D additive causal+padding+SWA mask.  Same band-diagonal
        # pattern HF's ``create_sliding_window_causal_mask`` produces; every
        # layer in the released DSV4-Flash was trained under it.
        _normalize_thd_packing_metadata(attn_kwargs)
        sliding_window = int(getattr(self.config, "sliding_window", 0) or 0) or None
        packed_seq_lens = None
        if attn_kwargs.get("qkv_format") == "thd":
            # THD packing uses seq_lens_padded to keep pack/CP padding inside a
            # valid block. Using only seq_lens leaves trailing pad query rows
            # with no legal keys, which the sparse TileLang path cannot execute.
            packed_seq_lens = attn_kwargs.get("seq_lens_padded")
            if packed_seq_lens is None:
                packed_seq_lens = attn_kwargs.get("seq_lens")
        cp_group = attn_kwargs.get("_dsv4_cp_group")
        cp_active = dsv4_cp_enabled(cp_group)
        packed_seq_ids = attn_kwargs.get("packed_seq_ids")
        if packed_seq_ids is None and packed_seq_lens is not None and not cp_active:
            packed_seq_ids = build_packed_seq_ids(
                packed_seq_lens,
                seq_len=shape_ref.shape[1],
                device=shape_ref.device,
            )
            attn_kwargs["packed_seq_ids"] = packed_seq_ids
        elif packed_seq_ids is not None:
            packed_seq_ids = packed_seq_ids.to(device=shape_ref.device, dtype=torch.long)
            if packed_seq_ids.dim() == 1:
                packed_seq_ids = packed_seq_ids.unsqueeze(0)
            attn_kwargs["packed_seq_ids"] = packed_seq_ids

        if cp_active and packed_seq_ids is not None:
            cp_padding_mask = padding_mask
            if cp_padding_mask is None and attention_mask is not None and attention_mask.dim() == 2:
                cp_padding_mask = attention_mask.bool().logical_not()
            attention_mask_4d = build_dsv4_cp_packed_causal_padding_mask(
                position_ids=position_ids,
                packed_seq_ids=packed_seq_ids,
                dtype=shape_ref.dtype,
                device=shape_ref.device,
                cp_group=cp_group,
                padding_mask=cp_padding_mask,
                sliding_window=sliding_window,
            )
        elif cp_active:
            cp_padding_mask = padding_mask
            if cp_padding_mask is None and attention_mask is not None and attention_mask.dim() == 2:
                cp_padding_mask = attention_mask.bool().logical_not()
            attention_mask_4d = build_dsv4_cp_causal_padding_mask(
                position_ids=position_ids,
                key_len=shape_ref.shape[1] * dsv4_cp_size(cp_group),
                dtype=shape_ref.dtype,
                device=shape_ref.device,
                cp_group=cp_group,
                padding_mask=cp_padding_mask,
                sliding_window=sliding_window,
            )
        elif packed_seq_lens is not None:
            attention_mask_4d = build_packed_causal_padding_mask(
                packed_seq_lens,
                seq_len=shape_ref.shape[1],
                dtype=shape_ref.dtype,
                device=shape_ref.device,
                sliding_window=sliding_window,
            )
        else:
            attention_mask_4d = build_causal_padding_mask(
                attention_mask,
                seq_len=shape_ref.shape[1],
                dtype=shape_ref.dtype,
                device=shape_ref.device,
                batch_size=shape_ref.shape[0],
                sliding_window=sliding_window,
            )

        has_visual_tokens = vision_token_types is not None and bool((vision_token_types >= 0).any().item())
        if has_visual_tokens:
            if cp_active or packed_seq_lens is not None:
                raise ValueError(
                    "DeepSeek-V4 visual image spans currently require unpacked CP=1 execution; "
                    "multi-axis visual-span masking is not supported"
                )
            attention_mask_4d = apply_deepseek_v4_image_visibility(attention_mask_4d, vision_token_types)
        else:
            # Keep text-only batches on the original causal sparse-attention width.
            vision_token_types = None

        # ``input_ids`` is only meaningful for hash-routing layers, which live
        # on stage 0 (num_hash_layers <= layers per stage 0).  Mid-stages pass
        # None — hash layers shouldn't be present there.
        layer_input_ids = input_ids if on_first_stage else None

        for layer in self.layers.values():
            if layer is None:  # PP-trimmed slot
                continue
            h = layer(
                x=h,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                position_embeddings_compress=position_embeddings_compress,
                rotary_compress=self.rotary_emb_compress,
                attention_mask=attention_mask_4d,
                padding_mask=padding_mask
                if padding_mask is not None
                else (
                    attention_mask.bool().logical_not()
                    if attention_mask is not None and attention_mask.dim() == 2
                    else None
                ),
                input_ids=layer_input_ids,
                vision_token_types=vision_token_types,
                **attn_kwargs,
            )

        mtp_hc_hidden = h if return_hc_hidden else None

        # Reduce hc_mult copies -> [B,S,dim] via the learned HC head, then
        # apply the shared RMSNorm.  Both modules live ONLY on the last PP
        # stage (intermediate stages keep h at 4D so the next stage can
        # consume it).  Matches HF PR 45616's ``DeepseekV4Model.forward``.
        if getattr(self, "hc_head", None) is not None:
            h = self.hc_head(h)
        if getattr(self, "norm", None) is not None:
            h = self.norm(h)
        if return_hc_hidden:
            if mtp_hc_hidden is None:
                raise ValueError("return_hc_hidden requested before HC stream was available")
            return h, mtp_hc_hidden
        return h

    def update_moe_gate_bias(self) -> None:
        with torch.no_grad():
            for block in self.layers.values():
                if isinstance(block.mlp, MoE):
                    block.mlp.gate.update_bias()

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        init_std = float(getattr(self.config, "initializer_range", 0.02))
        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.vision is not None:
                self.vision.init_weights(init_std)
                self.aligner.init_weights(init_std)
                nn.init.normal_(self.image_start, mean=0.0, std=init_std)
                nn.init.normal_(self.image_end, mean=0.0, std=init_std)
                nn.init.normal_(self.image_newline, mean=0.0, std=init_std)
                nn.init.normal_(self.image_pad, mean=0.0, std=init_std)
            if self.norm is not None:
                self.norm.reset_parameters()
            if self.hc_head is not None:
                self.hc_head.init_weights(init_std)
        for layer in self.layers.values():
            layer.init_weights(buffer_device=buffer_device, init_std=init_std)


class DeepseekV4ForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    _nemo_fully_shard = staticmethod(fully_shard_deepseek_v4)

    # Keep HC mixers and the MoE gate's correction bias in fp32 regardless of
    # the outer cast policy.  Matches HF PR 45616's
    # ``DeepseekV4PreTrainedModel._keep_in_fp32_modules_strict`` (lines 890-900
    # of modular_deepseek_v4.py) plus the existing ``e_score_correction_bias``
    # entry that is specific to KAutomodel's shared Gate buffer.
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _keep_in_fp32_modules_strict = [
        "attn_hc.fn",
        "attn_hc.base",
        "attn_hc.scale",
        "ffn_hc.fn",
        "ffn_hc.base",
        "ffn_hc.scale",
        "hc_head.hc_fn",
        "hc_head.hc_base",
        "hc_head.hc_scale",
        "self_attn.sinks",
        "self_attn.compressor.wkv",
        "self_attn.compressor.wgate",
        "self_attn.compressor.ape",
        "self_attn.compressor.indexer.wkv",
        "self_attn.compressor.indexer.wgate",
        "self_attn.compressor.indexer.ape",
        "e_score_correction_bias",
        "bias_vl",
        "norm1.weight",
        "norm2.weight",
        "vision.norm.weight",
        "lm_head",
        # RoPE inv_freq (matches rotary_emb + rotary_emb_compress) must stay fp32: the
        # bf16 cast in initialize_weights would otherwise round it and degrade rotary
        # precision vs HF (see llama/rope_utils.py).
        "rotary_emb",
    ]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        # CP is supported with the Miles-style TileLang attention path; the runtime
        # gate in ``_transformers/capabilities.py`` restricts it to ``attn='tilelang'``.
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True
        supports_thd: bool = True

    @classmethod
    def from_config(
        cls,
        config: DeepseekV4Config,
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
        config = DeepseekV4Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: DeepseekV4Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        self.backend = backend or BackendConfig()
        moe_overrides = kwargs.pop("moe_overrides", None)
        mtp_loss_scaling_factor = kwargs.pop("mtp_loss_scaling_factor", 0.1)
        self.model = DeepseekV4Model(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        self.lm_head = initialize_linear_module(
            self.backend.linear,
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=torch.float32,
        )
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = DeepSeekV4StateDictAdapter(
                self.config,
                self.model.moe_config,
                self.backend,
                dtype=get_dtype(config.torch_dtype, torch.bfloat16),
            )

        # MTP construction (import inside __init__ to avoid circular imports).
        from nemo_automodel.components.models.deepseek_v4.mtp import (  # noqa: PLC0415
            build_deepseek_v4_mtp,
            build_mtp_config_from_hf,
        )

        self.mtp_config = build_mtp_config_from_hf(config, loss_scaling_factor=mtp_loss_scaling_factor)
        if self.mtp_config.enabled:
            self.mtp = build_deepseek_v4_mtp(
                config=config,
                mtp_config=self.mtp_config,
                backend=self.backend,
                moe_config=self.model.moe_config,
                dtype=get_dtype(config.torch_dtype, torch.bfloat16),
                rotary_emb=self.model.rotary_emb,
                rotary_emb_compress=self.model.rotary_emb_compress,
            )
        else:
            self.mtp = None

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Keep DSV4 non-layer PP dependencies with the stages that need them."""

        text_model = text_model or self.model
        stage_modules = [list(modules) for modules in module_names_per_stage]

        def append_once(modules: list[str], fqn: str) -> None:
            if fqn not in modules:
                modules.append(fqn)

        if getattr(text_model, "rotary_emb_compress", None) is not None:
            for modules in stage_modules:
                append_once(modules, f"{layers_prefix}rotary_emb_compress")
        if getattr(text_model, "vision", None) is not None:
            for name in ("vision", "aligner", "image_start", "image_end", "image_newline", "image_pad"):
                append_once(stage_modules[0], f"{layers_prefix}{name}")
        if getattr(text_model, "hc_head", None) is not None:
            append_once(stage_modules[-1], f"{layers_prefix}hc_head")
        if self.mtp is not None:
            append_once(stage_modules[-1], "mtp")

        return stage_modules

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Return PP input/output meta tensors for DSV4's HC and MTP contract."""

        hidden_shape = (microbatch_size, seq_len, self.config.hidden_size)
        hc_hidden_shape = (microbatch_size, seq_len, self.config.hc_mult, self.config.hidden_size)
        mtp_depth = int(getattr(self.mtp_config, "num_layers", 0) or 0)

        def meta(shape: tuple[int, ...]) -> torch.Tensor:
            return torch.empty(*shape, device="meta", dtype=dtype)

        def append_mtp_metas(primary: torch.Tensor) -> tuple[torch.Tensor, ...]:
            mtp_metas = (meta(hidden_shape) for _ in range(mtp_depth))
            return (primary, *mtp_metas)

        if is_first:
            inputs_meta = (torch.empty(microbatch_size, seq_len, device="meta", dtype=torch.long),)
        else:
            inputs_meta = append_mtp_metas(meta(hc_hidden_shape if self.config.hc_mult > 1 else hidden_shape))

        if self.lm_head is not None:
            output_meta = meta((microbatch_size, seq_len, self.config.vocab_size))
        elif getattr(self.model, "norm", None) is not None:
            output_meta = meta(hidden_shape)
        else:
            output_meta = meta(hc_hidden_shape if self.config.hc_mult > 1 else hidden_shape)

        return inputs_meta, append_mtp_metas(output_meta)

    def _is_pipeline_parallel_stage(self) -> bool:
        if self.lm_head is None:
            return True
        if getattr(self.model, "embed_tokens", None) is None:
            return True
        try:
            return len(self.model.layers) != int(self.config.num_hidden_layers)
        except TypeError:
            return False

    def _build_mtp_embed_inputs_for_pp(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if getattr(self.model, "embed_tokens", None) is None:
            raise ValueError("First PP stage must own embed_tokens to build MTP embeddings")
        if input_ids.dtype not in (torch.int32, torch.int64, torch.long):
            raise ValueError("First PP stage must receive token ids to build MTP embeddings")

        from nemo_automodel.components.models.common.mtp import roll_tensor  # noqa: PLC0415

        cur_input_ids = input_ids
        embeds = []
        for _ in range(self.mtp_config.num_layers):
            cur_input_ids = roll_tensor(cur_input_ids, shifts=-1, dim=-1)
            embeds.append(self.model.embed_tokens(cur_input_ids))
        return tuple(embeds)

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Model-owned context-parallel batch prep (Miles-style contiguous shard).

        Returns a ``ContextParallelSharder`` (under the ``"cp_sharder"`` batch key) so
        the CP dispatch delegates CP sharding back to this
        model, with the config-derived per-rank shard multiple bound. DSV4
        embeds internally, so (unlike VLM models) this does not pre-embed --
        it leaves ``input_ids`` for the sharding callable.
        """
        from functools import partial  # noqa: PLC0415

        from nemo_automodel.components.distributed.context_parallel.sharder import (  # noqa: PLC0415
            ContextParallelSharder,
            contiguous_local_indices,
        )

        cp_sharder = ContextParallelSharder(
            shard_batch=partial(
                make_dsv4_contiguous_shard_cp_batch_and_ctx,
                pad_multiple=dsv4_cp_local_seq_multiple(self.config),
                sync_packed_length=self.backend.dispatcher == "hybridep",
            ),
            local_token_global_indices=contiguous_local_indices,
        )
        return {"cp_sharder": cp_sharder}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *mtp_embed_inputs: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_hws: torch.Tensor | None = None,
        n_images_per_sample: torch.Tensor | None = None,
        vision_token_types: torch.Tensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: bool | None = None,
        **attn_kwargs: Any,
    ) -> "DeepseekV4CausalLMOutput" | tuple[torch.Tensor, ...] | torch.Tensor:
        """Run causal language modeling with optional DSV4 visual inputs.

        Args:
            input_ids: Token IDs with layout ``[batch, sequence]``.
            mtp_embed_inputs: Optional PP-propagated MTP embeddings, each with
                layout ``[batch, sequence, hidden]``.
            position_ids: Position IDs with layout ``[batch, sequence]``.
            attention_mask: Valid-token mask with layout ``[batch, sequence]``.
            padding_mask: Padding mask with layout ``[batch, sequence]``.
            pixel_values: Concatenated patches with layout
                ``[all_patches, 3, patch_size, patch_size]``.
            image_grid_hws: ViT grids with layout ``[all_images, 2]``.
            n_images_per_sample: Image counts with layout ``[batch]``.
            vision_token_types: Visual types with layout ``[batch, sequence]``
                and ``-1`` for text.
            logits_to_keep: Number or positions of logits to retain.
            output_hidden_states: Whether to expose final states.

        Returns:
            ``DeepseekV4CausalLMOutput`` outside PP, or the PP stage tensor
            contract when pipeline parallelism is active.
        """
        if output_hidden_states is None:
            output_hidden_states = getattr(getattr(self, "config", None), "output_hidden_states", False)

        is_pp_stage = self._is_pipeline_parallel_stage()
        pp_mtp_enabled = is_pp_stage and self.mtp_config.enabled

        thd_mode = "qkv_format" in attn_kwargs and attn_kwargs["qkv_format"] == "thd"

        if vision_token_types is None and input_ids is not None and getattr(self.model, "vision", None) is not None:
            vision_token_types = torch.where(
                input_ids >= self.config.vocab_size,
                input_ids - self.config.vocab_size,
                torch.full_like(input_ids, -1),
            )

        # PP VLM batches keep variable-length patch tensors off the pipeline
        # schedule and stage them per microbatch on the first model part. Pull
        # the matching chunk back here, where the DSV4-owned embedding bridge
        # consumes it. Text-only microbatches consume their empty media slots.
        # Later PP stages have no embed_tokens and must neither
        # read nor advance the shared media cursor.
        on_first_stage = getattr(self.model, "embed_tokens", None) is not None
        if pixel_values is None and on_first_stage and getattr(self, "_vlm_pixel_values_chunks", None) is not None:
            chunk_idx = int(getattr(self, "_vlm_chunk_idx", 0) or 0)
            if chunk_idx >= len(self._vlm_pixel_values_chunks):
                raise RuntimeError(
                    f"DeepSeek-V4 PP media cursor {chunk_idx} exceeds "
                    f"{len(self._vlm_pixel_values_chunks)} staged chunks"
                )
            has_visual_tokens = vision_token_types is not None and bool((vision_token_types >= 0).any().item())
            if has_visual_tokens:
                pixel_values = self._vlm_pixel_values_chunks[chunk_idx]
                grid_chunks = getattr(self, "_vlm_image_grid_hws_chunks", None)
                if grid_chunks is None or chunk_idx >= len(grid_chunks):
                    raise RuntimeError("DeepSeek-V4 PP media is missing image_grid_hws for the current chunk")
                image_grid_hws = grid_chunks[chunk_idx]
            self._vlm_chunk_idx = chunk_idx + 1

        use_mtp = self.mtp is not None and self.training
        if use_mtp and vision_token_types is not None and bool((vision_token_types >= 0).any().item()):
            raise ValueError(
                "DSV4 visual fine-tuning currently requires num_nextn_predict_layers=0; "
                "visual pseudo-token shifting for MTP is not implemented"
            )
        model_out = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            pixel_values=pixel_values,
            image_grid_hws=image_grid_hws,
            n_images_per_sample=n_images_per_sample,
            vision_token_types=vision_token_types,
            return_hc_hidden=use_mtp,
            **attn_kwargs,
        )
        if use_mtp:
            hidden_states, mtp_hc_hidden = model_out
        else:
            hidden_states = model_out
            mtp_hc_hidden = None

        # Final hidden states (input to lm_head). Capture the FULL-sequence
        # tensor before any logits_to_keep slicing so the fused cross-entropy
        # path can recompute logits over every position.
        final_hidden_states = hidden_states

        # deepseek runs the lm_head in fp32: project in fp32 and cast the logits
        # back to the hidden dtype via the shared helper.
        logits = compute_lm_head_logits(
            self.lm_head, hidden_states, logits_to_keep, is_thd=thd_mode, fp32_lm_head=True
        ).logits

        if pp_mtp_enabled and self.lm_head is None:
            if not mtp_embed_inputs:
                mtp_embed_inputs = self._build_mtp_embed_inputs_for_pp(input_ids)
            return (logits, *mtp_embed_inputs)

        mtp_per_depth_h = None
        if use_mtp:
            if is_pp_stage and not mtp_embed_inputs:
                raise ValueError("Final PP stage requires propagated MTP embeddings")
            # MTP consumes the pre-final-head HC stream [B, S, hc_mult, hidden]
            # and returns collapsed per-depth [B, S, hidden] tensors for CE.
            seq_len = hidden_states.shape[1]
            batch_size = hidden_states.shape[0]
            if position_ids is None:
                position_ids = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
            sliding_window = int(getattr(self.config, "sliding_window", 0) or 0) or None
            cp_group = attn_kwargs.get("_dsv4_cp_group")
            if dsv4_cp_enabled(cp_group):
                cp_padding_mask = padding_mask
                if cp_padding_mask is None and attention_mask is not None and attention_mask.dim() == 2:
                    cp_padding_mask = attention_mask.bool().logical_not()
                mtp_attn_mask = build_dsv4_cp_causal_padding_mask(
                    position_ids=position_ids,
                    key_len=seq_len * dsv4_cp_size(cp_group),
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                    cp_group=cp_group,
                    padding_mask=cp_padding_mask,
                    sliding_window=sliding_window,
                )
            else:
                mtp_attn_mask = build_causal_padding_mask(
                    attention_mask,
                    seq_len=seq_len,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                    batch_size=batch_size,
                    sliding_window=sliding_window,
                )
            mtp_kwargs = {
                "hidden_states": mtp_hc_hidden,
                "position_ids": position_ids,
                "attention_mask": mtp_attn_mask,
                "padding_mask": padding_mask,
            }
            if cp_group is not None:
                mtp_kwargs["_dsv4_cp_group"] = cp_group
            if mtp_embed_inputs:
                mtp_kwargs["embed_inputs"] = tuple(mtp_embed_inputs)
            else:
                mtp_kwargs["input_ids"] = input_ids
                mtp_kwargs["embed_fn"] = self.model.embed_tokens
            mtp_per_depth_h = self.mtp(**mtp_kwargs)
        elif pp_mtp_enabled and self.lm_head is not None:
            mtp_per_depth_h = [hidden_states.new_empty(hidden_states.shape) for _ in range(self.mtp_config.num_layers)]

        if is_pp_stage:
            if pp_mtp_enabled:
                if self.training and self.mtp is None:
                    raise ValueError("Final PP stage has MTP enabled but does not own the MTP module")
                return (logits, *mtp_per_depth_h)
            return logits

        out_hidden_states = None
        if output_hidden_states:
            out_hidden_states = final_hidden_states
            # Mirror the THD logits unsqueeze so hidden_states and logits share
            # a leading [1, T, ...] layout for packed sequences.
            if thd_mode and out_hidden_states.dim() == 2:
                out_hidden_states = out_hidden_states.unsqueeze(0)

        return DeepseekV4CausalLMOutput(
            logits=logits,
            hidden_states=out_hidden_states,
            mtp_per_depth_h=mtp_per_depth_h,
            mtp_loss_scaling_factor=self.mtp_config.loss_scaling_factor,
        )

    def update_moe_gate_bias(self) -> None:
        self.model.update_moe_gate_bias()

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
        if self.mtp is not None:
            for sublayer in self.mtp.layers:
                sublayer.init_weights(buffer_device=buffer_device)
        # After FSDP2 wrapping, parameter dtypes must already be correct from
        # construction-time metadata. A blanket ``model.to(bf16)`` would
        # downcast fp32 DTensors before checkpoint load can fill them.
        if _has_dtensor_params(self):
            return
        cast_model_to_dtype(self, dtype)


ModelClass = DeepseekV4ForCausalLM
