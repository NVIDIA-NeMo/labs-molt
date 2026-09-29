# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checkpoint conversion for the released GLM-5.3-Flash VLM.

The checkpoint already uses the native vision/text prefix layout. Conversion is
needed for grouped EP experts, flat mHC/KDA parameters, the extra MTP layer, and
128x128 block-scaled FP8 training weights.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm5_next.config import Glm5NextConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.components.moe.state_dict_utils import is_dtensor

_BLOCK_SIZE = 128
_FP8_WEIGHT = re.compile(
    r"^model\.language_model\.layers\.\d+\.(?:"
    r"self_attn\.(?:q_a_proj|q_b_proj|kv_a_proj_with_mqa)"
    r"|mlp\.(?:gate|up|down)_proj"
    r"|mlp\.experts\.\d+\.(?:gate|up|down)_proj"
    r"|mlp\.shared_experts\.(?:gate|up|down)_proj"
    r")\.weight$"
)
_SPARSE_O_WEIGHT = re.compile(r"^model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.weight$")
_HC_KEY = re.compile(r"^(model\.language_model\.layers\.\d+)\.hc_(attn|ffn)_(fn|base|scale)$")
_NATIVE_HC_KEY = re.compile(
    r"^(model\.language_model\.layers\.\d+)\.(attn_hc|ffn_hc)(?:\._fp32_params)?\.(fn|base|scale)$"
)
_KDA_PARAMETER = re.compile(r"^(model\.language_model\.layers\.\d+\.self_attn)\.(A_log|dt_bias)$")
_NATIVE_KDA_PARAMETER = re.compile(r"^(model\.language_model\.layers\.\d+\.self_attn)\._fp32_params\.(A_log|dt_bias)$")


def _scale_shape(weight: torch.Tensor) -> tuple[int, int]:
    return (
        (weight.shape[-2] + _BLOCK_SIZE - 1) // _BLOCK_SIZE,
        (weight.shape[-1] + _BLOCK_SIZE - 1) // _BLOCK_SIZE,
    )


def _scale_placeholder(weight: torch.Tensor) -> torch.Tensor:
    """Create the global FP8 block-scale load destination for a 2-D weight."""
    local = weight.to_local() if is_dtensor(weight) else weight
    return torch.ones(_scale_shape(weight), dtype=torch.float32, device=local.device)


def _local_shard_offsets(tensor: torch.Tensor) -> tuple[int, ...]:
    """Return the global start coordinate of a DTensor's contiguous local shard."""
    from torch.distributed.tensor import Shard

    offsets = [0] * tensor.ndim
    current_shape = list(tensor.shape)
    for mesh_dim, placement in enumerate(tensor.placements):
        if not isinstance(placement, Shard) or placement.dim >= tensor.ndim:
            continue
        shard_dim = placement.dim
        local_size, relative_offset = Shard.local_shard_size_and_offset(
            current_shape[shard_dim],
            tensor.device_mesh.size(mesh_dim),
            tensor.device_mesh.get_local_rank(mesh_dim=mesh_dim),
        )
        offsets[shard_dim] += int(relative_offset)
        current_shape[shard_dim] = int(local_size)
    return tuple(offsets)


def _apply_local_block_scales(
    local_weight: torch.Tensor,
    local_scale: torch.Tensor,
    local_offsets: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply global-grid block scales to one possibly misaligned local shard."""
    rows, cols = local_weight.shape
    row_offset = local_offsets[0] % _BLOCK_SIZE
    col_offset = local_offsets[1] % _BLOCK_SIZE
    expected = (
        (row_offset + rows + _BLOCK_SIZE - 1) // _BLOCK_SIZE,
        (col_offset + cols + _BLOCK_SIZE - 1) // _BLOCK_SIZE,
    )
    if tuple(local_scale.shape) != expected:
        raise ValueError(
            f"FP8 scale shape {tuple(local_scale.shape)} does not cover local weight "
            f"{tuple(local_weight.shape)} at global offset {local_offsets} (expected {expected})"
        )
    expanded = local_scale.float().repeat_interleave(_BLOCK_SIZE, 0).repeat_interleave(_BLOCK_SIZE, 1)
    scale = expanded[row_offset : row_offset + rows, col_offset : col_offset + cols]
    return (local_weight.float() * scale).to(dtype)


def dequantize_block_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize an e4m3 weight with 128x128 fp32 inverse scales."""
    weight_is_dtensor = is_dtensor(weight)
    scale_is_dtensor = is_dtensor(scale_inv)
    local_weight = weight.to_local() if weight_is_dtensor else weight
    local_scale = scale_inv.to_local() if scale_is_dtensor else scale_inv
    local_offsets = _local_shard_offsets(weight) if weight_is_dtensor else (0, 0)
    block_starts = tuple(offset // _BLOCK_SIZE for offset in local_offsets)
    block_ends = tuple(
        (offset + size + _BLOCK_SIZE - 1) // _BLOCK_SIZE for offset, size in zip(local_offsets, local_weight.shape)
    )
    expected_local_scale = tuple(end - start for start, end in zip(block_starts, block_ends))

    if scale_is_dtensor:
        scale_offsets = _local_shard_offsets(scale_inv)
        if scale_offsets != block_starts or tuple(local_scale.shape) != expected_local_scale:
            raise ValueError(
                "FP8 scale DTensor shard does not cover the corresponding weight shard: "
                f"scale offset/shape={scale_offsets}/{tuple(local_scale.shape)}, "
                f"expected={block_starts}/{expected_local_scale}"
            )
    elif tuple(scale_inv.shape) == _scale_shape(weight):
        local_scale = scale_inv[
            block_starts[0] : block_ends[0],
            block_starts[1] : block_ends[1],
        ]
    elif tuple(local_scale.shape) != expected_local_scale:
        raise ValueError(
            f"FP8 scale shape {tuple(local_scale.shape)} does not match global weight "
            f"{tuple(weight.shape)} or its local block coverage {expected_local_scale}"
        )

    output = _apply_local_block_scales(
        local_weight,
        local_scale.to(local_weight.device),
        local_offsets,
        dtype,
    )
    if weight_is_dtensor:
        from torch.distributed.tensor import DTensor

        return DTensor.from_local(
            output,
            weight.device_mesh,
            weight.placements,
            shape=weight.shape,
            stride=weight.stride(),
        )
    return output


def _hf_to_native_key(key: str) -> str:
    match = _HC_KEY.match(key)
    if match:
        site = "attn_hc" if match.group(2) == "attn" else "ffn_hc"
        holder = "._fp32_params" if match.group(3) in ("base", "scale") else ""
        return f"{match.group(1)}.{site}{holder}.{match.group(3)}"
    match = _KDA_PARAMETER.match(key)
    if match:
        return f"{match.group(1)}._fp32_params.{match.group(2)}"
    return key


def _native_to_hf_key(key: str) -> str:
    match = _NATIVE_HC_KEY.match(key)
    if match:
        site = "attn" if match.group(2) == "attn_hc" else "ffn"
        return f"{match.group(1)}.hc_{site}_{match.group(3)}"
    match = _NATIVE_KDA_PARAMETER.match(key)
    if match:
        return f"{match.group(1)}.{match.group(2)}"
    return key


class Glm5NextStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Convert GLM split experts and FP8 weights to trainable grouped BF16."""

    # Released checkpoints require allocating FP8 dequantization and grouped-expert rebuilding.
    _supports_low_memory_dcp_load = False

    def __init__(
        self,
        config: Glm5NextConfig,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

    @property
    def _hf_prefix(self) -> str:
        return "model.language_model."

    @property
    def _expert_path_segment(self) -> str:
        return "mlp.experts"

    def _dequantize(self, state_dict: dict[str, Any]) -> None:
        scale_keys = []
        for key, value in list(state_dict.items()):
            scale_key = key + "_scale_inv"
            if key.endswith(".weight") and scale_key in state_dict:
                state_dict[key] = dequantize_block_fp8(value, state_dict[scale_key], dtype=self.dtype)
                scale_keys.append(scale_key)
        for key in scale_keys:
            state_dict.pop(key, None)

    def _is_fp8_weight(self, key: str) -> bool:
        """Match the checkpoint's quantized matrices, including DSA-only output projections."""
        if _FP8_WEIGHT.match(key):
            return True
        match = _SPARSE_O_WEIGHT.match(key)
        if match is None:
            return False
        return self.config.text_config.layer_types[int(match.group(1))] != "linear_attention"

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Dequantize, drop MTP, route flat parameters and aggregate experts."""
        del kwargs
        layer_limit = self.config.text_config.num_hidden_layers
        mtp_prefix = f"model.language_model.layers.{layer_limit}."
        for key in list(hf_state_dict):
            if key.startswith(mtp_prefix):
                hf_state_dict.pop(key)
        self._dequantize(hf_state_dict)
        for key in list(hf_state_dict):
            value = hf_state_dict.pop(key)
            native_key = _hf_to_native_key(key)
            if native_key.endswith("._fp32_params.A_log"):
                value = value.float()
            elif native_key.endswith("._fp32_params.dt_bias") or native_key.endswith(".e_score_correction_bias"):
                value = value.float()
            hf_state_dict[native_key] = value
        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Expand grouped experts and restore released checkpoint names."""
        output: dict[str, Any] = {}
        for key, value in state_dict.items():
            for hf_key, hf_value in self.convert_single_tensor_to_hf(
                key,
                value,
                exclude_key_regex=exclude_key_regex,
                quantization=quantization,
                **kwargs,
            ):
                output[hf_key] = hf_value
        return output

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Convert one native tensor, including split expert and FP8 load targets."""
        exclude = kwargs.get("exclude_key_regex")
        quantization = kwargs.get("quantization", False)
        expert = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        result = expert if expert is not None else [(fqn, tensor)]
        converted: list[tuple[str, Any]] = []
        for key, value in result:
            key = _native_to_hf_key(key)
            if exclude and re.match(exclude, key):
                continue
            if quantization and self._is_fp8_weight(key):
                fp8_value = value.to(torch.float8_e4m3fn)
                converted.append((key, fp8_value))
                converted.append((key + "_scale_inv", _scale_placeholder(value)))
            else:
                converted.append((key, value))
        return converted


__all__ = ["Glm5NextStateDictAdapter", "dequantize_block_fp8"]
