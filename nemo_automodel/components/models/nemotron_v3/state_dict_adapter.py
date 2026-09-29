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

_MAMBA_FP32_PARAMS_TO_BARE = re.compile(r"(\.mixer)\._fp32_params\.")
_MAMBA_FP32_PARAM_NAMES = ("A_log", "dt_bias", "D")


def _strip_mamba_fp32_holder_key(key: str) -> str:
    return _MAMBA_FP32_PARAMS_TO_BARE.sub(r"\1.", key)


def _route_mamba_fp32_holder_key(key: str) -> str:
    if "._fp32_params." in key or ".mixer." not in key:
        return key
    head, tail = key.rsplit(".mixer.", 1)
    if tail not in _MAMBA_FP32_PARAM_NAMES:
        return key
    return f"{head}.mixer._fp32_params.{tail}"


def _is_mamba_fp32_state_key(key: str) -> bool:
    if ".mixer." not in key:
        return False
    _, tail = key.rsplit(".mixer.", 1)
    if tail in _MAMBA_FP32_PARAM_NAMES:
        return True
    if tail.startswith("_fp32_params."):
        return tail[len("_fp32_params.") :] in _MAMBA_FP32_PARAM_NAMES
    return False


def _upcast_mamba_fp32_state_tensor(key: str, value: Any) -> Any:
    if _is_mamba_fp32_state_key(key) and isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
        return value.to(torch.float32)
    return value


class NemotronV3StateDictAdapter(MoESplitExpertsStateDictMixin, StateDictAdapter):
    """State dict adapter for NemotronV3 models.

    Converts between HuggingFace checkpoint format and internal NeMo format.

    HF format uses 'backbone' prefix:
        - backbone.embed_tokens.weight
        - backbone.layers.{}.norm.weight
        - backbone.layers.{}.mixer.* (mamba/attention/moe components)
        - backbone.norm_f.weight
        - lm_head.weight

    Internal format uses 'model' prefix:
        - model.embed_tokens.weight
        - model.layers.{}.norm.weight
        - model.layers.{}.mixer.* (mamba/attention/moe components)
        - model.norm.weight
        - lm_head.weight

    For MoE layers:
        - HF: Split per-expert weights (experts.{}.up_proj.weight, experts.{}.down_proj.weight)
        - Internal: Merged expert weights (experts.gate_and_up_projs, experts.down_projs)

    NemotronV3 uses ReLU² activation (non-gated), so gate_and_up_projs has
    shape [n_experts, dim, inter_dim] instead of [n_experts, dim, 2*inter_dim].

    Note: NemotronV3 uses 'mixer' instead of 'mlp' in layer paths.
    """

    _supports_low_memory_dcp_load = True

    def __init__(
        self,
        config,
        moe_config: MoEConfig | None,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.config = config
        # moe_config is None for dense Nemotron-H variants (no MoE layers); the
        # expert merge/split paths below are only reached for ``.mixer.experts.`` keys,
        # which dense checkpoints never contain.
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = False

        # Mapping for expert weights (HF split → internal merged)
        self.from_hf_map = {
            "model.layers.{}.mixer.experts.{}.up_proj.weight": "model.layers.{}.mixer.experts.gate_and_up_projs",
            "model.layers.{}.mixer.experts.{}.down_proj.weight": "model.layers.{}.mixer.experts.down_projs",
        }

    @property
    def _hf_prefix(self) -> str:
        """Return the source checkpoint's public Nemotron-H model prefix."""
        return "model." if self._uses_model_prefix else "backbone."

    @property
    def _expert_path_segment(self) -> str:
        """NemotronV3 uses 'mixer.experts' instead of 'mlp.experts'."""
        return "mixer.experts"

    @property
    def _v5_peft_target_parameters(self) -> tuple[str, ...]:
        """Nemotron V3 exposes fused non-gated expert parameters in Transformers v5."""
        return ("mixer.experts.up_proj", "mixer.experts.down_proj")

    def _native_key_to_hf(self, key: str, *, v4_compatible: bool = False) -> str:
        """Normalize a native Nemotron V3 key to its public HF namespace."""
        key = _strip_mamba_fp32_holder_key(key)
        # Base checkpoints may still use the remote-code backbone namespace.
        # Default PEFT exports target the built-in Transformers v5 model, whose
        # modules live under model regardless of the base checkpoint's keys.
        hf_prefix = "model." if ".lora_" in key and not v4_compatible else self._hf_prefix
        key = re.sub(
            r"^(?P<outer>base_model\.model\.)?model\.",
            lambda match: f"{match.group('outer') or ''}{hf_prefix}",
            key,
        )
        hf_root = re.escape(self._hf_prefix.rstrip("."))
        key = re.sub(rf"^{hf_root}\.norm\.weight$", f"{self._hf_prefix}norm_f.weight", key)
        key = re.sub(rf"^{hf_root}\.embed_tokens\.weight$", f"{self._hf_prefix}embeddings.weight", key)
        return key

    def map_peft_target_module_to_hf(self, module_name: str, *, v4_compatible: bool = False) -> str:
        """Map PEFT targets to the same namespace as the exported adapter weights.

        Args:
            module_name: A target-module name in native layout.
            v4_compatible: Preserve the source checkpoint's legacy namespace.

        Returns:
            Target-module name for the selected HF export format.
        """
        if v4_compatible:
            return self._native_key_to_hf(module_name, v4_compatible=True)
        return _strip_mamba_fp32_holder_key(module_name)

    def _hf_key_to_native(self, key: str) -> str:
        """Normalize a public HF Nemotron V3 key to its native namespace."""
        # A legacy adapter can use a different namespace from its base model.
        # Normalize each input independently of the namespace retained for export.
        hf_root = "(?:backbone|model)"
        key = re.sub(
            rf"^(?P<outer>base_model\.model\.)?{hf_root}\.norm_f\.weight$",
            lambda match: f"{match.group('outer') or ''}model.norm.weight",
            key,
        )
        # Accept singular ``embedding`` (transformers save_pretrained / RL toy
        # assets) and plural ``embeddings`` (released nvidia checkpoints). Export
        # stays plural via ``_native_key_to_hf``. Matches nvidia remote-code load_hook.
        key = re.sub(
            rf"^(?P<outer>base_model\.model\.)?{hf_root}\.embeddings?\.weight$",
            lambda match: f"{match.group('outer') or ''}model.embed_tokens.weight",
            key,
        )
        return re.sub(
            rf"^(?P<outer>base_model\.model\.)?{hf_root}\.",
            lambda match: f"{match.group('outer') or ''}model.",
            key,
        )

    def to_hf(self, state_dict: dict[str, Any], exclude_key_regex: str | None = None, **kwargs) -> dict[str, Any]:
        """Convert from internal model state dict to HuggingFace format.

        Args:
            state_dict: Internal format state dict
            exclude_key_regex: Optional regex pattern to exclude keys
            **kwargs: Additional arguments

        Returns:
            HuggingFace format state dict
        """
        hf_state_dict = {}
        for fqn in list(state_dict.keys()):
            tensor = state_dict.pop(fqn)
            converted_tensors = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, **kwargs
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
        """Convert HF checkpoint to internal format.

        - Rename backbone → model
        - Rename norm_f → norm
        - Aggregate per-expert weights into grouped tensors
        - If device_mesh is provided, only load experts needed for the current rank
        - Process MTP keys (``mtp.layers.{i}.*``) separately, reusing the
          same MoE expert-merge logic for the MoE sublayer of each MTP depth.

        Args:
            hf_state_dict: HuggingFace format state dict
            device_mesh: Optional device mesh for distributed expert loading
            **kwargs: Additional arguments

        Returns:
            Internal format state dict
        """
        # Drop checkpoint keys for backbone layers past ``num_hidden_layers``
        # (e.g. when loading the first N layers of a larger checkpoint for a
        # downsized smoke run). The matcher tolerates both ``backbone.layers.{i}``
        # and ``model.layers.{i}`` since the prefix is normalized after this.
        num_layers = int(getattr(self.config, "num_hidden_layers", 0) or 0)
        if num_layers > 0:
            layer_idx_pattern = re.compile(r"^(?:backbone|model)\.layers\.(\d+)\.")
            for key in list(hf_state_dict.keys()):
                m = layer_idx_pattern.match(key)
                if m is not None and int(m.group(1)) >= num_layers:
                    hf_state_dict.pop(key)

        # Separate MTP keys; they live in their own top-level namespace and
        # are not subject to the backbone/model rename.
        mtp_state_dict: dict[str, Any] = {}
        backbone_state_dict: dict[str, Any] = {}
        for key in list(hf_state_dict.keys()):
            value = hf_state_dict.pop(key)
            if key.startswith("mtp."):
                mtp_state_dict[key] = value
            else:
                backbone_state_dict[key] = value

        # Detect whether the source checkpoint uses the remote-code ``backbone``
        # namespace or Transformers v5's native ``model`` namespace. MTP keys
        # never carry either prefix. Adapter-only restores must not change the
        # base checkpoint's export namespace: legacy PEFT keys can use native
        # ``model`` names even when the base checkpoint uses ``backbone``.
        for key in backbone_state_dict.keys():
            if any(part.startswith("lora_") for part in key.split(".")):
                continue
            bare_key = key.removeprefix("base_model.model.")
            if bare_key.startswith("backbone."):
                self._uses_model_prefix = False
                break
            if bare_key.startswith("model."):
                self._uses_model_prefix = True
                break

        # First, rename backbone → model and norm_f → norm
        renamed_state_dict = {}
        for key in list(backbone_state_dict.keys()):
            value = backbone_state_dict.pop(key)
            new_key = self._hf_key_to_native(key)

            new_key = _route_mamba_fp32_holder_key(new_key)
            renamed_state_dict[new_key] = _upcast_mamba_fp32_state_tensor(new_key, value)

        # Then merge experts using the mixin method. Dense Nemotron-H variants have no
        # experts (moe_config is None) and no '.mixer.experts.' keys, so the merge is a
        # pure pass-through — skip it; the mixin would otherwise dereference
        # moe_config.n_routed_experts.
        if self.moe_config is None:
            merged = renamed_state_dict
        else:
            merged = self._from_hf_w_merged_experts(renamed_state_dict, device_mesh)

        # Re-route MTP keys through the standard merge with prefix stripped.
        if mtp_state_dict:
            stripped: dict[str, Any] = {}
            for key, value in mtp_state_dict.items():
                stripped_key = key[len("mtp.") :] if key.startswith("mtp.") else key
                stripped_key = _route_mamba_fp32_holder_key(stripped_key)
                stripped[stripped_key] = _upcast_mamba_fp32_state_tensor(stripped_key, value)
            # reset_view_loaded_keys=False: this is the second merge of a single from_hf (after the
            # backbone merge above), so accumulate MTP view-loaded keys onto the backbone's record.
            prior_view_keys = set(self.view_loaded_native_keys)
            merged_mtp = (
                stripped
                if self.moe_config is None
                else self._from_hf_w_merged_experts(stripped, device_mesh, reset_view_loaded_keys=False)
            )
            for key, value in merged_mtp.items():
                merged[f"mtp.{key}"] = value
            # The merge loop records view-loaded keys in mtp.-stripped form (it only ever sees
            # stripped keys); re-prefix them so the checkpoint loader's key-diff matches them
            # against the model's real mtp.* parameter names instead of flagging them as
            # missing/unexpected.
            new_view_keys = self.view_loaded_native_keys - prior_view_keys
            self._view_loaded_native_keys = prior_view_keys | {f"mtp.{key}" for key in new_view_keys}

        return merged

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single tensor from internal format to HuggingFace format.

        Args:
            fqn: Fully qualified name of the tensor in internal format
            tensor: The tensor to convert
            **kwargs: Additional arguments for conversion

        Returns:
            List of (fqn, tensor) tuples in HuggingFace format
        """
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        v4_compatible = kwargs.get("v4_compatible", False)

        # MTP keys live in their own ``mtp.*`` namespace; route them through
        # the standard expert-split path with the prefix overridden so
        # emitted HF keys stay under ``mtp.`` instead of ``backbone.``.
        if fqn.startswith("mtp."):
            fqn = _strip_mamba_fp32_holder_key(fqn)
            expert_split = (
                None
                if self.moe_config is None
                else self._convert_single_merged_expert_to_hf_split_experts(
                    fqn,
                    tensor,
                    prefix_override="mtp.",
                    **kwargs,
                )
            )
            result = expert_split if expert_split is not None else [(fqn, tensor)]
            result = [(key, _upcast_mamba_fp32_state_tensor(key, value)) for key, value in result]
            if exclude_key_regex:
                result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]
            return result

        # Try to convert merged expert weights to split experts. Dense variants have no
        # experts (moe_config is None), so skip straight to the standard rename path.
        expert_result = (
            None
            if self.moe_config is None
            else self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
        )
        if expert_result is not None:
            # The shared expert converter preserves the native input prefix for
            # LoRA keys. Route every result through Nemotron's adapter-specific
            # normalization for the selected PEFT export format.
            result = [(self._native_key_to_hf(key, v4_compatible=v4_compatible), value) for key, value in expert_result]
        else:
            new_fqn = self._native_key_to_hf(fqn, v4_compatible=v4_compatible)
            result = [(new_fqn, _upcast_mamba_fp32_state_tensor(new_fqn, tensor))]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        return result

    def forced_hf_dtype_mapping(self, state_dict: dict[str, Any]) -> dict[str, str]:
        """Return HF export dtype overrides for tensors that are intrinsically fp32."""
        forced: dict[str, str] = {}
        for fqn, value in state_dict.items():
            if not isinstance(value, torch.Tensor) or not value.dtype.is_floating_point:
                continue
            if _is_mamba_fp32_state_key(fqn) or "e_score_correction_bias" in fqn:
                forced[fqn] = "F32"
        return forced
