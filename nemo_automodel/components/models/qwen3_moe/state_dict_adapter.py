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
import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

logger = logging.getLogger(__name__)

# Native LoRA suffixes for grouped MoE expert tensors
_LORA_EXPERT_SUFFIXES = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")


class Qwen3MoeStateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """Converts between HF Qwen3-MoE checkpoints and our grouped-experts native format.

    Qwen3-MoE HF experts use keys:
      model.layers.{L}.mlp.experts.{E}.gate_proj.weight
      model.layers.{L}.mlp.experts.{E}.up_proj.weight
      model.layers.{L}.mlp.experts.{E}.down_proj.weight

    Our native format groups them into:
      model.layers.{L}.mlp.experts.gate_and_up_projs  # [n_experts, dim, 2*moe_inter_dim]
      model.layers.{L}.mlp.experts.down_projs         # [n_experts, moe_inter_dim, dim]
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

    @property
    def _v5_peft_target_parameters(self) -> tuple[str, ...]:
        """Qwen3 MoE is validated for fused PEFT v5 ParamWrapper export."""
        return ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")

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

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from native format to HuggingFace format.

        When ``v4_compatible=False`` (the default), LoRA expert tensors are
        emitted in PEFT v0.18+ ParamWrapper format so that
        ``PeftModel.from_pretrained()`` can load them directly; the layout
        follows peft >= 0.19.1 unless ``legacy_paramwrapper_layout=True``
        asks for the pre-flip layout (huggingface/peft#3165).  When
        ``v4_compatible=True``, the legacy per-expert split is used instead
        (via the parent mixin).

        Args:
            fqn: Fully qualified name of the tensor in native format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        v4_compatible = kwargs.get("v4_compatible", False)

        # Check if this is a LoRA expert tensor eligible for ParamWrapper conversion
        if not v4_compatible:
            expert_segment = self._expert_path_segment
            for suffix in _LORA_EXPERT_SUFFIXES:
                if fqn.endswith(f".{suffix}") and f".{expert_segment}.{suffix}" in fqn:
                    result = self._convert_lora_to_paramwrapper(
                        fqn, tensor, legacy_layout=kwargs.get("legacy_paramwrapper_layout", False)
                    )
                    if exclude_key_regex:
                        result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
                    return result

        # Non-LoRA keys or legacy mode: fall through to parent mixin
        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        if expert_result is not None:
            result = expert_result
        else:
            result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        return result

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to native format, handling ParamWrapper LoRA keys.

        Before delegating to the parent ``_from_hf_w_merged_experts`` (which
        handles legacy per-expert LoRA format), this method scans for
        ParamWrapper-format LoRA keys and converts them back to the native
        grouped format expected by ``GroupedExpertsLoRA``.
        """
        # Detect whether HF checkpoints use the "model." prefix
        for key in hf_state_dict.keys():
            if ".mlp.experts." in key and key.endswith(".weight"):
                self._uses_model_prefix = key.startswith("model.")
                break

        # Convert any ParamWrapper-format LoRA keys to native grouped format
        hf_state_dict = self._convert_paramwrapper_to_native(hf_state_dict)

        return self._from_hf_w_merged_experts(hf_state_dict, device_mesh)
