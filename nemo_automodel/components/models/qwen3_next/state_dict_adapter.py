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

import logging
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.gated_delta_net_fp32 import (
    forced_gated_delta_net_fp32_dtype_mapping,
    route_fp32_holder_key,
    strip_fp32_holder_key,
    upcast_gated_delta_net_fp32_state_tensor,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

logger = logging.getLogger(__name__)


class Qwen3NextStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Converts between HF Qwen3Next checkpoints and our grouped-experts native format.

    Qwen3Next HF experts use keys:
      model.layers.{L}.mlp.experts.{E}.gate_proj.weight
      model.layers.{L}.mlp.experts.{E}.up_proj.weight
      model.layers.{L}.mlp.experts.{E}.down_proj.weight

    Our native format groups them into:
      model.layers.{L}.mlp.experts.gate_and_up_projs  # [n_experts, dim, 2*moe_inter_dim]
      model.layers.{L}.mlp.experts.down_projs         # [n_experts, moe_inter_dim, dim]

    Qwen3Next HF shared experts use keys:
      model.layers.{L}.mlp.shared_expert.gate_proj.weight
      model.layers.{L}.mlp.shared_expert.up_proj.weight
      model.layers.{L}.mlp.shared_expert.down_proj.weight

    Our native format uses:
      model.layers.{L}.mlp.shared_experts.gate_proj.weight  # Note: plural "shared_experts"
      model.layers.{L}.mlp.shared_experts.up_proj.weight
      model.layers.{L}.mlp.shared_experts.down_proj.weight
    """

    _supports_low_memory_dcp_load = True

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True

        # Key mapping from HF Qwen3Next format to internal format
        self.hf_to_internal_map = {
            ".mlp.shared_expert.": ".mlp.shared_experts.",
        }

        # Reverse mapping for to_hf conversion
        self.internal_to_hf_map = {v: k for k, v in self.hf_to_internal_map.items()}

    def _apply_key_mapping(self, state_dict: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
        """Apply key substring mappings to state dict keys.

        Args:
            state_dict: State dict to apply mappings to
            mapping: Dictionary mapping substrings to replace them with

        Returns:
            New state dict with mapped keys
        """
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for pattern, replacement in mapping.items():
                if pattern in key:
                    new_key = new_key.replace(pattern, replacement)
                    break
            new_state_dict[new_key] = value
        return new_state_dict

    def to_hf(
        self, state_dict: dict[str, Any], exclude_key_regex: str | None = None, quantization: bool = False, **kwargs
    ) -> dict[str, Any]:
        hf_state_dict = {}
        for fqn, tensor in state_dict.items():
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            )
            for key, value in converted_tensors:
                hf_state_dict[key] = value

        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        # Detect whether HF checkpoints use the "model." prefix
        for key in hf_state_dict.keys():
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break

        # First apply key mappings for shared experts (shared_expert -> shared_experts)
        hf_state_dict = self._apply_key_mapping(hf_state_dict, self.hf_to_internal_map)
        fp32_routed_state_dict = {}
        for key, value in hf_state_dict.items():
            native_key = route_fp32_holder_key(key)
            fp32_routed_state_dict[native_key] = upcast_gated_delta_net_fp32_state_tensor(native_key, value)
        hf_state_dict = fp32_routed_state_dict

        # Then convert routed experts from split to grouped format
        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from native format to HuggingFace format.

        Args:
            fqn: Fully qualified name of the tensor in native format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = expert_result
        else:
            result = [(fqn, tensor)]

        mapped_result = []
        for key, value in result:
            new_key = key
            for pattern, replacement in self.internal_to_hf_map.items():
                if pattern in key:
                    new_key = new_key.replace(pattern, replacement)
                    break
            # Hide the GatedDeltaNet fp32 holder wrapping so saved checkpoints keep the
            # bare HF key (``...linear_attn.A_log`` instead of
            # ``...linear_attn._fp32_params.A_log``) and stay directly HF-loadable.
            new_key = strip_fp32_holder_key(new_key)
            value = upcast_gated_delta_net_fp32_state_tensor(new_key, value)
            mapped_result.append((new_key, value))

        if exclude_key_regex:
            import re

            mapped_result = [(k, v) for k, v in mapped_result if not re.match(exclude_key_regex, k)]

        return mapped_result

    def forced_hf_dtype_mapping(self, state_dict: dict[str, Any]) -> dict[str, str]:
        """Return HF export dtype overrides for intrinsically-fp32 GDN tensors."""
        return forced_gated_delta_net_fp32_dtype_mapping(state_dict)
