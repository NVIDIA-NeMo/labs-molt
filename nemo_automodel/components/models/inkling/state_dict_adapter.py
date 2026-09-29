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

"""State-dict adapter for the native Thinking Machines Inkling checkpoint."""

import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.config import MoEConfig

_RAW_EXPERT_RE = re.compile(r"model\.llm\.layers\.(\d+)\.mlp\.experts\.(w13_weight|w2_weight)$")
_HF_EXPERT_RE = re.compile(r"(?:model\.)?language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
_HF_DENSE_RE = re.compile(r"((?:model\.)?language_model\.layers\.\d+\.mlp)\.(gate_proj|up_proj|down_proj)\.weight$")
_HF_SHARED_RE = re.compile(
    r"((?:model\.)?language_model\.layers\.\d+\.mlp\.shared_experts)\."
    r"(gate_proj|up_proj|down_proj)$"
)

_RAW_TRANSPOSED_SUFFIXES = (
    ".mlp.experts.w13_weight",
    ".mlp.experts.w2_weight",
    ".mlp.shared_experts.shared_w13_weight",
    ".mlp.shared_experts.shared_w2_weight",
    ".mlp.w13_dn.weight",
    ".mlp.w2_md.weight",
)


def _interleave(tensor: torch.Tensor, dim: int, *, inverse: bool = False) -> torch.Tensor:
    """Match Transformers' Inkling ``Interleave`` checkpoint conversion."""
    dim = dim % tensor.ndim
    shape = list(tensor.shape)
    if inverse:
        shape[dim : dim + 1] = [2, shape[dim] // 2]
    else:
        shape[dim : dim + 1] = [shape[dim] // 2, 2]
    return tensor.reshape(shape).transpose(dim, dim + 1).reshape(tensor.shape).contiguous()


def _native_to_raw_key(key: str) -> str:
    if key == "model.language_model.embed_tokens.weight":
        return "model.llm.embed.weight"
    if key == "model.language_model.embed_norm.weight":
        return "model.llm.embed_norm.weight"
    if key == "model.language_model.norm.weight":
        return "model.llm.norm.weight"
    if key == "lm_head.weight":
        return "model.llm.unembed.weight"
    if key == "model.audio_tower.embed_audio_tokens.embed_audio_tokens.weight":
        return "model.audio.encoder.weight"
    if key == "model.audio_tower.norm.weight":
        return "model.audio.final_norm.weight"

    vision_projection = re.fullmatch(r"model\.vision_tower\.encoder_layers\.(\d+)\.projection\.weight", key)
    if vision_projection:
        return f"model.visual.layers.linear_{vision_projection.group(1)}.weight"
    vision_norm = re.fullmatch(r"model\.vision_tower\.encoder_layers\.(\d+)\.layer_norm\.weight", key)
    if vision_norm:
        return f"model.visual.layers.norm_{vision_norm.group(1)}.weight"
    if key == "model.vision_tower.final_norm.weight":
        return "model.visual.final_norm.weight"

    key = key.replace("model.language_model.layers.", "model.llm.layers.")
    key = key.replace(".self_attn.q_proj.weight", ".attn.wq_du.weight")
    key = key.replace(".self_attn.k_proj.weight", ".attn.wk_dv.weight")
    key = key.replace(".self_attn.v_proj.weight", ".attn.wv_dv.weight")
    key = key.replace(".self_attn.r_proj.weight", ".attn.wr_du.weight")
    key = key.replace(".self_attn.o_proj.weight", ".attn.wo_ud.weight")
    key = key.replace(".self_attn.q_norm.weight", ".attn.q_norm.weight")
    key = key.replace(".self_attn.k_norm.weight", ".attn.k_norm.weight")
    key = key.replace(".self_attn.k_sconv._fp32_params.weight", ".attn.k_sconv.weight")
    key = key.replace(".self_attn.v_sconv._fp32_params.weight", ".attn.v_sconv.weight")
    key = key.replace(".self_attn.rel_logits_proj.proj", ".attn.rel_logits_proj.proj")
    key = key.replace(".attn_sconv._fp32_params.weight", ".attn_sconv.weight")
    key = key.replace(".mlp_sconv._fp32_params.weight", ".mlp_sconv.weight")
    key = key.replace(".input_layernorm.weight", ".attn_norm.weight")
    key = key.replace(".post_attention_layernorm.weight", ".mlp_norm.weight")
    key = key.replace(".mlp.experts.gate_and_up_projs", ".mlp.experts.w13_weight")
    key = key.replace(".mlp.experts.down_projs", ".mlp.experts.w2_weight")
    key = key.replace(".mlp.shared_experts.gate_up_proj", ".mlp.shared_experts.shared_w13_weight")
    key = key.replace(".mlp.shared_experts.down_proj", ".mlp.shared_experts.shared_w2_weight")
    key = key.replace(".mlp.gate_up_proj", ".mlp.w13_dn.weight")
    key = key.replace(".mlp.down_proj", ".mlp.w2_md.weight")
    key = key.replace(".mlp.gate._fp32_params.e_score_correction_bias", ".mlp.gate.bias")
    return key


def _raw_to_native_key(key: str) -> str:
    if key == "model.llm.embed.weight":
        return "model.language_model.embed_tokens.weight"
    if key == "model.llm.embed_norm.weight":
        return "model.language_model.embed_norm.weight"
    if key == "model.llm.norm.weight":
        return "model.language_model.norm.weight"
    if key == "model.llm.unembed.weight":
        return "lm_head.weight"
    if key == "model.audio.encoder.weight":
        return "model.audio_tower.embed_audio_tokens.embed_audio_tokens.weight"
    if key == "model.audio.final_norm.weight":
        return "model.audio_tower.norm.weight"

    vision_projection = re.fullmatch(r"model\.visual\.layers\.linear_(\d+)\.weight", key)
    if vision_projection:
        return f"model.vision_tower.encoder_layers.{vision_projection.group(1)}.projection.weight"
    vision_norm = re.fullmatch(r"model\.visual\.layers\.norm_(\d+)\.weight", key)
    if vision_norm:
        return f"model.vision_tower.encoder_layers.{vision_norm.group(1)}.layer_norm.weight"
    if key == "model.visual.final_norm.weight":
        return "model.vision_tower.final_norm.weight"

    key = key.replace("model.llm.layers.", "model.language_model.layers.")
    key = key.replace(".attn.wq_du.weight", ".self_attn.q_proj.weight")
    key = key.replace(".attn.wk_dv.weight", ".self_attn.k_proj.weight")
    key = key.replace(".attn.wv_dv.weight", ".self_attn.v_proj.weight")
    key = key.replace(".attn.wr_du.weight", ".self_attn.r_proj.weight")
    key = key.replace(".attn.wo_ud.weight", ".self_attn.o_proj.weight")
    key = key.replace(".attn.q_norm.weight", ".self_attn.q_norm.weight")
    key = key.replace(".attn.k_norm.weight", ".self_attn.k_norm.weight")
    key = key.replace(".attn.k_sconv.weight", ".self_attn.k_sconv._fp32_params.weight")
    key = key.replace(".attn.v_sconv.weight", ".self_attn.v_sconv._fp32_params.weight")
    key = key.replace(".attn.rel_logits_proj.proj", ".self_attn.rel_logits_proj.proj")
    key = key.replace(".attn_sconv.weight", ".attn_sconv._fp32_params.weight")
    key = key.replace(".mlp_sconv.weight", ".mlp_sconv._fp32_params.weight")
    key = key.replace(".attn_norm.weight", ".input_layernorm.weight")
    key = key.replace(".mlp_norm.weight", ".post_attention_layernorm.weight")
    key = key.replace(".mlp.experts.w13_weight", ".mlp.experts.gate_and_up_projs")
    key = key.replace(".mlp.experts.w2_weight", ".mlp.experts.down_projs")
    key = key.replace(".mlp.shared_experts.shared_w13_weight", ".mlp.shared_experts.gate_up_proj")
    key = key.replace(".mlp.shared_experts.shared_w2_weight", ".mlp.shared_experts.down_proj")
    key = key.replace(".mlp.w13_dn.weight", ".mlp.gate_up_proj")
    key = key.replace(".mlp.w2_md.weight", ".mlp.down_proj")
    key = key.replace(".mlp.gate.bias", ".mlp.gate._fp32_params.e_score_correction_bias")
    return key


class InklingStateDictAdapter(StateDictAdapter):
    """Convert Inkling weights between native AutoModel and raw checkpoint layouts."""

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Expose raw Inkling checkpoint keys, using views for large matrices."""
        checkpoint_state = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(fqn, tensor, exclude_key_regex=exclude_key_regex):
                checkpoint_state[key] = value
        return checkpoint_state

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert one native tensor to the raw Inkling checkpoint layout."""
        raw_key = _native_to_raw_key(fqn)
        if kwargs.get("exclude_key_regex") and re.match(kwargs["exclude_key_regex"], raw_key):
            return []
        if raw_key.endswith(_RAW_TRANSPOSED_SUFFIXES):
            tensor = tensor.transpose(-1, -2)
        return [(raw_key, tensor)]

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Load raw Inkling keys or an already-converted Transformers state dict."""
        n_experts = self.moe_config.n_routed_experts
        start_expert, end_expert, rank = 0, n_experts, None
        if device_mesh is not None:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            rank = (
                state_dict_utils.get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )

        state_dict: dict[str, Any] = {}
        dense_parts: dict[str, dict[str, torch.Tensor]] = {}
        shared_parts: dict[str, dict[str, torch.Tensor]] = {}

        def convert_raw_matrix(raw_key: str, value: Any) -> Any:
            is_expert = _RAW_EXPERT_RE.fullmatch(raw_key) is not None
            if state_dict_utils.is_dtensor(value):
                return value.transpose(-1, -2)
            if is_expert:
                value = value[start_expert:end_expert]
            value = value.transpose(-1, -2).contiguous().to(self.dtype)
            if is_expert and device_mesh is not None:
                value = state_dict_utils.create_dtensor_from_local(value, device_mesh, rank)
            return value

        for key, value in hf_state_dict.items():
            if key.startswith("model.mtp.") or key.startswith("mtp.") or key.endswith("_scale_inv"):
                continue

            if key.startswith(("model.llm.", "model.audio.", "model.visual.")):
                native_key = _raw_to_native_key(key)
                state_dict[native_key] = (
                    convert_raw_matrix(key, value) if key.endswith(_RAW_TRANSPOSED_SUFFIXES) else value
                )
                continue

            expert_match = _HF_EXPERT_RE.fullmatch(key)
            if expert_match:
                layer_num, projection = expert_match.groups()
                raw_key = f"model.llm.layers.{layer_num}.mlp.experts."
                if projection == "gate_up_proj":
                    raw_key += "w13_weight"
                    value = _interleave(value, 1, inverse=True)
                else:
                    raw_key += "w2_weight"
                native_key = _raw_to_native_key(raw_key)
                state_dict[native_key] = convert_raw_matrix(raw_key, value)
                continue

            dense_match = _HF_DENSE_RE.fullmatch(key)
            if dense_match:
                prefix, projection = dense_match.groups()
                if projection == "down_proj":
                    state_dict[f"{prefix}.down_proj"] = value.transpose(-1, -2).contiguous().to(self.dtype)
                else:
                    dense_parts.setdefault(prefix, {})[projection] = value
                continue

            shared_match = _HF_SHARED_RE.fullmatch(key)
            if shared_match:
                prefix, projection = shared_match.groups()
                if projection == "down_proj":
                    state_dict[f"{prefix}.down_proj"] = value.transpose(-1, -2).contiguous().to(self.dtype)
                else:
                    shared_parts.setdefault(prefix, {})[projection] = value
                continue

            native_key = key.replace(
                ".mlp.gate.e_score_correction_bias",
                ".mlp.gate._fp32_params.e_score_correction_bias",
            )
            native_key = native_key.replace(".k_sconv.conv1d.weight", ".k_sconv._fp32_params.weight")
            native_key = native_key.replace(".v_sconv.conv1d.weight", ".v_sconv._fp32_params.weight")
            native_key = native_key.replace(".attn_sconv.conv1d.weight", ".attn_sconv._fp32_params.weight")
            native_key = native_key.replace(".mlp_sconv.conv1d.weight", ".mlp_sconv._fp32_params.weight")
            state_dict[native_key] = value

        for prefix, parts in dense_parts.items():
            if set(parts) != {"gate_proj", "up_proj"}:
                raise RuntimeError(f"Incomplete dense Inkling MLP weights for {prefix}: {sorted(parts)}")
            concatenated = torch.cat((parts["gate_proj"], parts["up_proj"]), dim=0)
            raw = _interleave(concatenated, 0, inverse=True)
            state_dict[f"{prefix}.gate_up_proj"] = raw.transpose(-1, -2).contiguous().to(self.dtype)

        for prefix, parts in shared_parts.items():
            if set(parts) != {"gate_proj", "up_proj"}:
                raise RuntimeError(f"Incomplete shared Inkling expert weights for {prefix}: {sorted(parts)}")
            concatenated = torch.cat((parts["gate_proj"], parts["up_proj"]), dim=1)
            raw = _interleave(concatenated, 1, inverse=True)
            state_dict[f"{prefix}.gate_up_proj"] = raw.transpose(-1, -2).contiguous().to(self.dtype)

        return state_dict
