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

"""State-dict adapter for Qwen3.5-MoE.

HF Qwen3.5-MoE stores expert weights as **aggregated 3-D tensors**:

    model.language_model.layers.{L}.mlp.experts.gate_up_proj   # [n_experts, 2*moe_inter, hidden]
    model.language_model.layers.{L}.mlp.experts.down_proj      # [n_experts, hidden, moe_inter]

NeMo uses a different naming convention **and transposed layout** (x @ weight):

    model.language_model.layers.{L}.mlp.experts.gate_and_up_projs  # [n_experts, hidden, 2*moe_inter]
    model.language_model.layers.{L}.mlp.experts.down_projs         # [n_experts, moe_inter, hidden]

Both expert tensors require `.transpose(1, 2)` when converting between formats.

Additionally, the shared expert uses singular in HF and plural in NeMo:

    HF:   .mlp.shared_expert.{gate,up,down}_proj.weight
    NeMo: .mlp.shared_experts.{gate,up,down}_proj.weight

All other keys (attention, linear_attn/GatedDeltaNet, norms, embeddings, vision
encoder) pass through unchanged. The HF VLM checkpoint stores the language
model head as ``model.lm_head`` while Automodel registers it on the outer model
as ``lm_head``.
"""

import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
from safetensors import SafetensorError, safe_open
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.gated_delta_net_fp32 import (
    forced_gated_delta_net_fp32_dtype_mapping,
    route_fp32_holder_key,
    strip_fp32_holder_key,
    upcast_gated_delta_net_fp32_state_tensor,
)
from nemo_automodel.components.models.qwen3_5.state_dict_adapter import (
    map_qwen3_5_mtp_from_hf_key,
    map_qwen3_5_mtp_to_hf_key,
)
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.layers import MoEConfig

_FP8_BLOCK_SIZE = 128

_BASE_SPLIT_FP8_EXPERT_KEY = re.compile(
    r"(?:model\.)?(?:language_model\.)?layers\.\d+\.mlp\.experts\.\d+"
    r"\.(?:gate_proj|up_proj|down_proj)\.weight$"
)
_BASE_GROUPED_EXPERT_KEY = re.compile(
    r"(?:model\.)?(?:language_model\.)?layers\.\d+\.mlp\.experts\.(?:gate_up_proj|down_proj)$"
)
_BASE_SPLIT_FP8_EXPERT_PARAM = re.compile(
    r"(?:model\.)?(?:language_model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight(?P<scale>_scale_inv)?$"
)
_MTP_SPLIT_EXPERT_KEY = re.compile(r"mtp\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight")
_MTP_GROUPED_EXPERT_KEY = re.compile(r"mtp\.layers\.\d+\.mlp\.experts\.(?:gate_up_proj|down_proj)")


def _block_scale_placeholder(weight: Any) -> torch.Tensor:
    """Create a regular 128x128 block-scale load target for a 2-D weight.

    Args:
        weight: FP8 weight Tensor or DTensor of shape [out, in].

    Returns:
        Tensor of shape [ceil(out / 128), ceil(in / 128)] with one scale for each 128x128 block.
    """
    shape = weight.shape
    local = weight.to_local() if state_dict_utils.is_dtensor(weight) else weight
    return torch.ones(
        (
            (shape[0] + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
            (shape[1] + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
        ),
        dtype=torch.bfloat16,
        device=local.device,
    )


def _dequantize_block_fp8(weight: Any, scale_inv: Any, dtype: torch.dtype) -> torch.Tensor:
    """Dequantize one local 2-D 128x128 block-scaled FP8 expert weight.

    Args:
        weight: FP8 weight Tensor or DTensor of shape [out, in].
        scale_inv: Block-scale Tensor or DTensor of shape [ceil(out / 128), ceil(in / 128)], with one scale for
            each 128x128 block.
        dtype: Output dtype.

    Returns:
        Dequantized Tensor of shape [out, in] in ``dtype``.
    """
    local_weight = weight.to_local() if state_dict_utils.is_dtensor(weight) else weight
    local_scale = scale_inv.to_local() if state_dict_utils.is_dtensor(scale_inv) else scale_inv
    rows, cols = local_weight.shape
    expected_shape = (
        (rows + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
        (cols + _FP8_BLOCK_SIZE - 1) // _FP8_BLOCK_SIZE,
    )
    if tuple(local_scale.shape) != expected_shape:
        raise ValueError(
            f"FP8 scale shape {tuple(local_scale.shape)} does not match weight shape "
            f"{tuple(local_weight.shape)} (expected {expected_shape})"
        )
    expanded_scale = (
        local_scale.float().repeat_interleave(_FP8_BLOCK_SIZE, dim=0).repeat_interleave(_FP8_BLOCK_SIZE, dim=1)
    )
    return (local_weight.float() * expanded_scale[:rows, :cols]).to(dtype)


def _slice_ep_shard_dim1(
    local_tensor: torch.Tensor,
    ep_shard_rank: int,
    ep_shard_size: int,
    tensor_name: str,
) -> torch.Tensor:
    """Slice dim 1 of a local expert tensor for the ``ep_shard`` mesh dimension."""
    if ep_shard_size == 1:
        return local_tensor
    if local_tensor.shape[1] % ep_shard_size != 0:
        raise ValueError(
            f"{tensor_name} dim 1 ({local_tensor.shape[1]}) is not divisible by ep_shard_size={ep_shard_size}"
        )
    chunk = local_tensor.shape[1] // ep_shard_size
    return local_tensor[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]


def _filter_excluded(result: list[tuple[str, Any]], exclude_key_regex: str | None) -> list[tuple[str, Any]]:
    """Remove exported HF entries matched by ``exclude_key_regex``."""
    if not exclude_key_regex:
        return result
    return [(key, value) for key, value in result if not re.match(exclude_key_regex, key)]


def _infer_base_expert_hf_layout(checkpoint_keys: Iterable[str]) -> str | None:
    """Infer the decoder expert layout from checkpoint key names."""
    split_key = None
    grouped_key = None
    checkpoint_keys = set(checkpoint_keys)
    for key in checkpoint_keys:
        if split_key is None and _BASE_SPLIT_FP8_EXPERT_KEY.fullmatch(key) and f"{key}_scale_inv" in checkpoint_keys:
            split_key = key
        elif grouped_key is None and _BASE_GROUPED_EXPERT_KEY.fullmatch(key):
            grouped_key = key
        if split_key is not None and grouped_key is not None:
            raise ValueError(
                "Checkpoint contains both split-FP8 and grouped decoder expert keys; "
                f"cannot infer one layout (examples: {split_key!r}, {grouped_key!r})"
            )
    if split_key is not None:
        return "split-fp8"
    if grouped_key is not None:
        return "grouped"
    return None


def _infer_mtp_expert_hf_layout(checkpoint_keys: Iterable[str]) -> str | None:
    """Infer the MTP expert layout from checkpoint key names."""
    split_key = None
    grouped_key = None
    for key in checkpoint_keys:
        if split_key is None and _MTP_SPLIT_EXPERT_KEY.fullmatch(key):
            split_key = key
        elif grouped_key is None and _MTP_GROUPED_EXPERT_KEY.fullmatch(key):
            grouped_key = key
        if split_key is not None and grouped_key is not None:
            raise ValueError(
                "Checkpoint contains both split and grouped MTP expert keys; "
                f"cannot infer one layout (examples: {split_key!r}, {grouped_key!r})"
            )
    if split_key is not None:
        return "split"
    if grouped_key is not None:
        return "grouped"
    return None


def _get_local_safetensors_keys(model_path: str | None) -> set[str]:
    """Read key metadata from a local HF safetensors checkpoint without loading tensors."""
    if not model_path:
        return set()
    checkpoint_dir = Path(model_path)
    if not checkpoint_dir.is_dir():
        return set()

    checkpoint_keys: set[str] = set()
    for index_path in sorted(checkpoint_dir.glob("*.safetensors.index.json")):
        try:
            with index_path.open() as index_file:
                index = json.load(index_file)
        except (OSError, json.JSONDecodeError):
            continue
        weight_map = index.get("weight_map")
        if isinstance(weight_map, dict):
            checkpoint_keys.update(key for key in weight_map if isinstance(key, str))
    if checkpoint_keys:
        return checkpoint_keys

    for shard_path in sorted(checkpoint_dir.glob("*.safetensors")):
        try:
            with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
                checkpoint_keys.update(shard.keys())
        except (OSError, SafetensorError):
            continue
    return checkpoint_keys


def _strip_fp32_params(key: str) -> str:
    """Strip the fp32 holder segment from GDN state-dict keys."""
    return strip_fp32_holder_key(key)


def _route_fp32_params(key: str) -> str:
    """Route bare GDN fp32 params into the holder used by the native module."""
    return route_fp32_holder_key(key)


class Qwen3_5MoeStateDictAdapter(StateDictAdapter):
    """Converts between HF Qwen3.5-MoE checkpoints and the NeMo native format.

    HF Qwen3.5-MoE stores expert weights as **aggregated 3-D tensors**:

        model.language_model.layers.{L}.mlp.experts.gate_up_proj   # [n_experts, 2*moe_inter, hidden]
        model.language_model.layers.{L}.mlp.experts.down_proj      # [n_experts, hidden, moe_inter]

    NeMo uses a different naming convention **and transposed layout** (x @ weight):

        model.language_model.layers.{L}.mlp.experts.gate_and_up_projs  # [n_experts, hidden, 2*moe_inter]
        model.language_model.layers.{L}.mlp.experts.down_projs         # [n_experts, moe_inter, hidden]

    Both expert tensors require `.transpose(1, 2)` when converting between formats.

    Loading paths:
      DCP path:  to_hf renames+transposes native→HF, DCP loads into DTensors,
                 and from_hf renames+transposes HF→native. Split MTP experts
                 are locally reassembled before restoring the complete EP mesh.
      Init path: from_hf receives plain tensors from safetensors, slices to local EP
                 shard, transposes, and wraps in DTensor via create_dtensor_from_local.

    Additionally, the shared expert uses singular in HF and plural in NeMo:

        HF:   .mlp.shared_expert.{gate,up,down}_proj.weight
        NeMo: .mlp.shared_experts.{gate,up,down}_proj.weight
    """

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
        pretrained_model_name_or_path: str | None = None,
        mtp_expert_hf_layout: str | None = None,
        text_only: bool = False,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.mtp_expert_hf_layout = mtp_expert_hf_layout
        self.text_only = text_only
        self._expert_hf_layout: str | None = None
        self._inferred_mtp_expert_hf_layout: str | None = None
        self._local_checkpoint_layout_checked = False
        self._uses_model_prefix = True

        self.hf_to_internal_map = {
            ".mlp.shared_expert.": ".mlp.shared_experts.",
        }
        self.internal_to_hf_map = {v: k for k, v in self.hf_to_internal_map.items()}

    def _get_expert_hf_layout(self, checkpoint_keys: Iterable[str] | None = None) -> str:
        """Resolve and remember whether decoder experts use grouped BF16 or split block-FP8 HF keys."""
        if checkpoint_keys is not None:
            inferred_layout = _infer_base_expert_hf_layout(checkpoint_keys)
            if inferred_layout is not None:
                self._expert_hf_layout = inferred_layout
                return inferred_layout

        if self._expert_hf_layout is not None:
            return self._expert_hf_layout

        model_paths = [self.pretrained_model_name_or_path]
        for attr in ("_name_or_path", "name_or_path"):
            value = getattr(self.config, attr, None)
            if isinstance(value, str) and value:
                model_paths.append(value)
        for model_path in model_paths:
            inferred_layout = _infer_base_expert_hf_layout(_get_local_safetensors_keys(model_path))
            if inferred_layout is not None:
                self._expert_hf_layout = inferred_layout
                return inferred_layout

        # Preserve the established Qwen3.5/Qwen3.6 grouped BF16 contract when
        # checkpoint metadata is unavailable or inconclusive.
        self._expert_hf_layout = "grouped"
        return self._expert_hf_layout

    def _get_mtp_expert_hf_layout(self, checkpoint_keys: Iterable[str] | None = None) -> str:
        """Resolve and remember whether MTP experts use split or grouped HF keys."""
        layout = self.mtp_expert_hf_layout
        if layout is None:
            config_layout = getattr(self.config, "mtp_expert_hf_layout", None)
            layout = config_layout if isinstance(config_layout, str) else None

        configured_layout = None
        if layout is not None:
            normalized = layout.lower().replace("_", "-")
            if normalized in {"split", "per-expert", "per-expert-safetensors"}:
                configured_layout = "split"
            elif normalized in {"grouped", "group"}:
                configured_layout = "grouped"
            else:
                raise ValueError(f"Unsupported MTP expert HF layout: {layout!r}")
        if configured_layout is not None:
            return configured_layout

        inferred_layout = _infer_mtp_expert_hf_layout(checkpoint_keys) if checkpoint_keys is not None else None
        if checkpoint_keys is None and not self._local_checkpoint_layout_checked:
            self._local_checkpoint_layout_checked = True
            model_paths = [self.pretrained_model_name_or_path]
            for attr in ("_name_or_path", "name_or_path"):
                value = getattr(self.config, attr, None)
                if isinstance(value, str) and value:
                    model_paths.append(value)
            for model_path in model_paths:
                inferred_layout = _infer_mtp_expert_hf_layout(_get_local_safetensors_keys(model_path))
                if inferred_layout is not None:
                    break
        if inferred_layout is not None:
            self._inferred_mtp_expert_hf_layout = inferred_layout

        if self._inferred_mtp_expert_hf_layout is not None:
            return self._inferred_mtp_expert_hf_layout

        model_names = [self.pretrained_model_name_or_path]
        for attr in ("_name_or_path", "name_or_path"):
            value = getattr(self.config, attr, None)
            if isinstance(value, str) and value:
                model_names.append(value)

        for model_name in model_names:
            if not model_name:
                continue
            normalized_name = model_name.lower()
            if "qwen3.5-35b-a3b" in normalized_name:
                return "split"
            if "qwen3.6-35b-a3b" in normalized_name:
                return "grouped"

        return "grouped"

    def _apply_key_mapping(self, state_dict: dict[str, Any], mapping: dict[str, str]) -> dict[str, Any]:
        """Apply key substring mappings to state dict keys."""
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
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Rename native keys to HF keys and transpose expert tensors."""
        mtp_expert_hf_layout = self._get_mtp_expert_hf_layout()
        hf_state_dict: dict[str, Any] = {}
        for fqn, tensor in state_dict.items():
            for key, value in self.convert_single_tensor_to_hf(
                fqn,
                tensor,
                exclude_key_regex=exclude_key_regex,
                mtp_expert_hf_layout=mtp_expert_hf_layout,
                quantization=quantization,
            ):
                hf_state_dict[key] = value
        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Rename HF keys to native keys and transpose expert tensors.

        DTensors (DCP path): rename + transpose; split MTP experts are locally
        reassembled without applying EP-shard slicing again.
        Plain tensors (init path): slice to local EP shard, transpose, create DTensor.
        """
        checkpoint_keys = set(hf_state_dict)
        self._get_expert_hf_layout(checkpoint_keys)
        self._get_mtp_expert_hf_layout(checkpoint_keys)
        self._uses_model_prefix = any(key.startswith("model.") for key in hf_state_dict if not key.startswith("mtp."))
        model_prefix = "model." if self._uses_model_prefix else ""

        n_experts = self.moe_config.n_routed_experts

        # Pre-compute EP slicing params (only used for plain tensor path)
        start_expert, end_expert, rank = 0, n_experts, None
        ep_shard_rank, ep_shard_size = 0, 1
        if device_mesh is not None:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            rank = (
                state_dict_utils.get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
            if "ep_shard" in device_mesh.mesh_dim_names:
                ep_shard_sub = state_dict_utils.get_submesh(device_mesh, ("ep_shard",))
                if ep_shard_sub.size() > 1:
                    ep_shard_rank = ep_shard_sub.get_local_rank()
                    ep_shard_size = ep_shard_sub.size()

        state_dict: dict[str, Any] = {}
        base_expert_parts: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
        mtp_expert_parts: dict[str, dict[str, dict[int, torch.Tensor]]] = {}

        def store_native_key(native_key: str, tensor: Any) -> None:
            native_key = route_fp32_holder_key(native_key)
            state_dict[native_key] = upcast_gated_delta_net_fp32_state_tensor(native_key, tensor)

        for key, value in hf_state_dict.items():
            base_split_match = _BASE_SPLIT_FP8_EXPERT_PARAM.match(key)
            if base_split_match:
                layer_num = base_split_match.group(1)
                expert_num = int(base_split_match.group(2))
                projection = base_split_match.group(3)
                if not state_dict_utils.should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                    continue
                projection_parts = base_expert_parts.setdefault(
                    layer_num,
                    {"gate_proj": {}, "up_proj": {}, "down_proj": {}},
                )
                expert_parts = projection_parts[projection].setdefault(expert_num, {})
                expert_parts["scale" if base_split_match.group("scale") else "weight"] = value
                continue

            mapped_mtp_key = map_qwen3_5_mtp_from_hf_key(key)
            if mapped_mtp_key != key:
                store_native_key(mapped_mtp_key, value)
                continue

            match = re.match(
                r"(?:model\.)?(?:language_model\.)?layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$",
                key,
            )
            mtp_match = re.match(r"mtp\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$", key)
            if match or mtp_match:
                active_match = match or mtp_match
                layer_num = active_match.group(1)
                which = active_match.group(2)
                if mtp_match:
                    native_key = f"mtp.layers.{layer_num}.mlp.experts."
                else:
                    language_model_prefix = "" if self.text_only else "language_model."
                    native_key = f"{model_prefix}{language_model_prefix}layers.{layer_num}.mlp.experts."
                native_key += "gate_and_up_projs" if which == "gate_up_proj" else "down_projs"

                if state_dict_utils.is_dtensor(value):
                    # DCP path: already sharded DTensor — rename + transpose.
                    state_dict[native_key] = value.transpose(1, 2)
                else:
                    # Init path: plain tensor — slice to local EP shard, transpose.
                    local_tensor = value[start_expert:end_expert].transpose(1, 2).to(self.dtype)
                    if ep_shard_size > 1:
                        assert local_tensor.shape[1] % ep_shard_size == 0
                        chunk = local_tensor.shape[1] // ep_shard_size
                        local_tensor = local_tensor[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]
                    state_dict[native_key] = state_dict_utils.create_dtensor_from_local(local_tensor, device_mesh, rank)
                continue

            mtp_split_match = re.match(
                r"mtp\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$",
                key,
            )
            if mtp_split_match:
                layer_num = mtp_split_match.group(1)
                expert_num = int(mtp_split_match.group(2))
                which = mtp_split_match.group(3)
                if not state_dict_utils.should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                    continue
                parts = mtp_expert_parts.setdefault(layer_num, {"gate_proj": {}, "up_proj": {}, "down_proj": {}})
                parts[which][expert_num] = value
                continue

            # Skip quantization scale keys
            if key.endswith("_scale_inv"):
                continue

            # --- Shared expert key mapping (shared_expert → shared_experts) ---
            mapped_key = key
            for pattern, replacement in self.hf_to_internal_map.items():
                if pattern in mapped_key:
                    mapped_key = mapped_key.replace(pattern, replacement)
                    break

            if mapped_key.startswith("mtp."):
                store_native_key(mapped_key, value)
            elif mapped_key.startswith("model.lm_head."):
                store_native_key(mapped_key.removeprefix("model."), value)
            elif mapped_key.startswith("lm_head."):
                store_native_key(mapped_key, value)
            elif key.startswith("model."):
                store_native_key(mapped_key, value)
            else:
                store_native_key(
                    f"{model_prefix}{mapped_key}" if not mapped_key.startswith("model.") else mapped_key, value
                )

        language_model_prefix = "" if self.text_only else "language_model."
        for layer_num, parts in base_expert_parts.items():
            expert_ids = sorted(set(parts["gate_proj"]) | set(parts["up_proj"]) | set(parts["down_proj"]))
            gate_up_tensors = []
            down_tensors = []
            for expert_id in expert_ids:
                projections = {}
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    projection_parts = parts[projection].get(expert_id, {})
                    if "weight" not in projection_parts or "scale" not in projection_parts:
                        raise RuntimeError(
                            f"Missing FP8 {projection} weight/scale for layer {layer_num}, expert {expert_id}"
                        )
                    projections[projection] = _dequantize_block_fp8(
                        projection_parts["weight"],
                        projection_parts["scale"],
                        self.dtype,
                    )
                gate_t = projections["gate_proj"].transpose(0, 1)
                up_t = projections["up_proj"].transpose(0, 1)
                down_t = projections["down_proj"].transpose(0, 1)
                gate_up_tensors.append(torch.cat((gate_t, up_t), dim=1))
                down_tensors.append(down_t)

            gate_up_tensor = torch.stack(gate_up_tensors, dim=0)
            down_tensor = torch.stack(down_tensors, dim=0)
            if ep_shard_size > 1:
                gate_up_tensor = _slice_ep_shard_dim1(
                    gate_up_tensor,
                    ep_shard_rank,
                    ep_shard_size,
                    "Base gate_and_up_projs",
                )
                down_tensor = _slice_ep_shard_dim1(
                    down_tensor,
                    ep_shard_rank,
                    ep_shard_size,
                    "Base down_projs",
                )

            native_prefix = f"{model_prefix}{language_model_prefix}layers.{layer_num}.mlp.experts."
            state_dict[f"{native_prefix}gate_and_up_projs"] = state_dict_utils.create_dtensor_from_local(
                gate_up_tensor,
                device_mesh,
                rank,
            )
            state_dict[f"{native_prefix}down_projs"] = state_dict_utils.create_dtensor_from_local(
                down_tensor,
                device_mesh,
                rank,
            )

        for layer_num, parts in mtp_expert_parts.items():
            expert_ids = sorted(set(parts["gate_proj"]) | set(parts["up_proj"]) | set(parts["down_proj"]))
            gate_up_tensors = []
            down_tensors = []
            for expert_id in expert_ids:
                if expert_id not in parts["gate_proj"] or expert_id not in parts["up_proj"]:
                    raise RuntimeError(f"Missing gate/up MTP expert weights for layer {layer_num}, expert {expert_id}")
                if expert_id not in parts["down_proj"]:
                    raise RuntimeError(f"Missing down MTP expert weight for layer {layer_num}, expert {expert_id}")
                gate_t = parts["gate_proj"][expert_id].transpose(0, 1)
                up_t = parts["up_proj"][expert_id].transpose(0, 1)
                down_t = parts["down_proj"][expert_id].transpose(0, 1)
                gate_up_tensors.append(torch.cat((gate_t, up_t), dim=1))
                down_tensors.append(down_t)

            gate_up_tensor = torch.stack(gate_up_tensors, dim=0).to(self.dtype)
            down_tensor = torch.stack(down_tensors, dim=0).to(self.dtype)
            split_tensors_are_dtensors = state_dict_utils.is_dtensor(gate_up_tensor)
            if split_tensors_are_dtensors != state_dict_utils.is_dtensor(down_tensor):
                raise TypeError("MTP split gate/up and down expert weights must use the same tensor type")

            if split_tensors_are_dtensors:
                # DCP split-expert values retain the non-EP placements after
                # ``split_experts_weights_dtensor_aware`` removes the expert mesh
                # dimension. Recover their already-sharded local storage before
                # rebuilding the native tensor on the complete expert mesh.
                gate_up_tensor = gate_up_tensor.to_local()
                down_tensor = down_tensor.to_local()
            elif ep_shard_size > 1:
                gate_up_tensor = _slice_ep_shard_dim1(
                    gate_up_tensor,
                    ep_shard_rank,
                    ep_shard_size,
                    "MTP gate_and_up_projs",
                )
                down_tensor = _slice_ep_shard_dim1(
                    down_tensor,
                    ep_shard_rank,
                    ep_shard_size,
                    "MTP down_projs",
                )

            state_dict[f"mtp.layers.{layer_num}.mlp.experts.gate_and_up_projs"] = (
                state_dict_utils.create_dtensor_from_local(gate_up_tensor, device_mesh, rank)
            )
            state_dict[f"mtp.layers.{layer_num}.mlp.experts.down_projs"] = state_dict_utils.create_dtensor_from_local(
                down_tensor, device_mesh, rank
            )

        return state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Rename a single native key to HF format and transpose expert tensors."""
        exclude_key_regex = kwargs.get("exclude_key_regex")
        quantization = kwargs.get("quantization", False)

        base_gate_up_match = re.match(r"(.+layers\.(\d+)\.mlp\.experts)\.gate_and_up_projs$", fqn)
        base_down_match = re.match(r"(.+layers\.(\d+)\.mlp\.experts)\.down_projs$", fqn)
        if quantization and self._get_expert_hf_layout() == "split-fp8":
            if base_gate_up_match:
                expert_prefix = base_gate_up_match.group(1)
                splits, expert_ids = state_dict_utils.split_experts_weights_dtensor_aware(
                    tensor,
                    self.moe_config.n_routed_experts,
                )
                result = []
                inter_dim = self.moe_config.moe_inter_dim
                for expert_tensor, expert_id in zip(splits, expert_ids):
                    gate = expert_tensor[:, :inter_dim].transpose(0, 1).to(dtype=torch.float8_e4m3fn)
                    up = expert_tensor[:, inter_dim:].transpose(0, 1).to(dtype=torch.float8_e4m3fn)
                    gate_key = f"{expert_prefix}.{expert_id}.gate_proj.weight"
                    up_key = f"{expert_prefix}.{expert_id}.up_proj.weight"
                    result.extend(
                        (
                            (gate_key, gate),
                            (f"{gate_key}_scale_inv", _block_scale_placeholder(gate)),
                            (up_key, up),
                            (f"{up_key}_scale_inv", _block_scale_placeholder(up)),
                        )
                    )
                return _filter_excluded(result, exclude_key_regex)
            if base_down_match:
                expert_prefix = base_down_match.group(1)
                splits, expert_ids = state_dict_utils.split_experts_weights_dtensor_aware(
                    tensor,
                    self.moe_config.n_routed_experts,
                )
                result = []
                for expert_tensor, expert_id in zip(splits, expert_ids):
                    down = expert_tensor.transpose(0, 1).to(dtype=torch.float8_e4m3fn)
                    down_key = f"{expert_prefix}.{expert_id}.down_proj.weight"
                    result.extend(
                        (
                            (down_key, down),
                            (f"{down_key}_scale_inv", _block_scale_placeholder(down)),
                        )
                    )
                return _filter_excluded(result, exclude_key_regex)

        new_fqn = fqn
        value = tensor
        mtp_gate_up_match = re.match(r"mtp\.layers\.(\d+)\.mlp\.experts\.gate_and_up_projs$", fqn)
        mtp_down_match = re.match(r"mtp\.layers\.(\d+)\.mlp\.experts\.down_projs$", fqn)
        mtp_expert_hf_layout = kwargs.get("mtp_expert_hf_layout")
        if mtp_expert_hf_layout is None:
            mtp_expert_hf_layout = self._get_mtp_expert_hf_layout()
        if mtp_expert_hf_layout == "split":
            if mtp_gate_up_match:
                layer_num = mtp_gate_up_match.group(1)
                splits, expert_ids = state_dict_utils.split_experts_weights_dtensor_aware(
                    tensor, self.moe_config.n_routed_experts
                )
                result = []
                inter_dim = self.moe_config.moe_inter_dim
                for expert_tensor, expert_id in zip(splits, expert_ids):
                    gate = expert_tensor[:, :inter_dim].transpose(0, 1)
                    up = expert_tensor[:, inter_dim:].transpose(0, 1)
                    if not state_dict_utils.is_dtensor(gate):
                        gate = gate.contiguous()
                    if not state_dict_utils.is_dtensor(up):
                        up = up.contiguous()
                    result.append((f"mtp.layers.{layer_num}.mlp.experts.{expert_id}.gate_proj.weight", gate))
                    result.append((f"mtp.layers.{layer_num}.mlp.experts.{expert_id}.up_proj.weight", up))
                return _filter_excluded(result, exclude_key_regex)
            if mtp_down_match:
                layer_num = mtp_down_match.group(1)
                splits, expert_ids = state_dict_utils.split_experts_weights_dtensor_aware(
                    tensor, self.moe_config.n_routed_experts
                )
                result = []
                for expert_tensor, expert_id in zip(splits, expert_ids):
                    down = expert_tensor.transpose(0, 1)
                    if not state_dict_utils.is_dtensor(down):
                        down = down.contiguous()
                    result.append((f"mtp.layers.{layer_num}.mlp.experts.{expert_id}.down_proj.weight", down))
                return _filter_excluded(result, exclude_key_regex)

        # Qwen3.6 stores MTP experts in the same grouped layout as decoder experts,
        # while Qwen3.5 stores MTP experts as per-expert tensors handled above.
        if ".mlp.experts.gate_and_up_projs" in fqn:
            new_fqn = fqn.replace(".mlp.experts.gate_and_up_projs", ".mlp.experts.gate_up_proj")
            value = tensor.transpose(1, 2)
        elif ".mlp.experts.down_projs" in fqn:
            new_fqn = fqn.replace(".mlp.experts.down_projs", ".mlp.experts.down_proj")
            value = tensor.transpose(1, 2)

        # Apply shared_experts → shared_expert reverse mapping
        for pattern, replacement in self.internal_to_hf_map.items():
            if pattern in new_fqn:
                new_fqn = new_fqn.replace(pattern, replacement)
                break

        # Hide the GatedDeltaNet fp32 holder wrapping so saved checkpoints keep the
        # bare HF key (``...linear_attn.A_log`` instead of
        # ``...linear_attn._fp32_params.A_log``) and stay directly HF-loadable.
        new_fqn = strip_fp32_holder_key(new_fqn)

        new_fqn = map_qwen3_5_mtp_to_hf_key(new_fqn)
        value = upcast_gated_delta_net_fp32_state_tensor(new_fqn, value)

        if exclude_key_regex and re.match(exclude_key_regex, new_fqn):
            return []
        return [(new_fqn, value)]

    def forced_hf_dtype_mapping(self, state_dict: dict[str, Any]) -> dict[str, str]:
        """Return HF export dtype overrides for intrinsically-fp32 GDN tensors."""
        return forced_gated_delta_net_fp32_dtype_mapping(state_dict)
