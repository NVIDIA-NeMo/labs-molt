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

"""State-dict adapter for Gemma4 MoE.

HF Gemma4 MoE (eevee-4 26B-A4B) stores expert weights as 3-D tensors:

    layers.{L}.moe.gate_up_proj       # [n_experts, 2*expert_inter_size, hidden_size]
    layers.{L}.moe.down_proj          # [n_experts, hidden_size, expert_inter_size]
    layers.{L}.moe.per_expert_scale   # [n_experts]

NeMo uses transposed layout with concatenated gate+up:

    layers.{L}.moe.experts.gate_and_up_projs  # [n_experts, hidden_size, 2*expert_inter_size]
    layers.{L}.moe.experts.down_projs         # [n_experts, expert_inter_size, hidden_size]

Additionally, the Gemma4 router is mapped to the NeMo Gemma4Gate:

    HF:   .router.proj.weight / .router.scale
    NeMo: .moe.gate.proj.weight / .moe.gate.scale

The per_expert_scale is absorbed into down_projs during from_hf.  When
saving back to HF, per_expert_scale is emitted as ones (scale already baked
into the weights).
"""

import re
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.layers import MoEConfig


class Gemma4MoEStateDictAdapter(StateDictAdapter):
    """Converts between HF Gemma4 MoE checkpoints and the NeMo format.

    Handles:
      1. Expert weight concatenation (gate_proj + up_proj -> gate_and_up_projs)
      2. per_expert_scale absorption into down_projs
      3. Router key remapping (router.* -> moe.gate.*)
      4. Expert-parallel sharding when a device mesh is provided
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
        self._loading_into_model_weights = False

    # ------------------------------------------------------------------
    # HF -> NeMo
    # ------------------------------------------------------------------
    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert Hugging Face Gemma4 weights into native model layout.

        Args:
            hf_state_dict: Hugging Face state mapping. Expert gate/up tensors have shape
                ``[experts, 2 * expert_hidden, hidden]`` and down tensors have shape
                ``[experts, hidden, expert_hidden]``. During a direct checkpoint load, those tensors use the model's
                existing weight memory with the last two dimensions transposed.
            device_mesh: Optional expert-parallel mesh. Distributed conversion slices the global expert axis and may
                shard the native feature axis according to the mesh.
            **kwargs: Additional adapter-interface arguments.

        Returns:
            Native state mapping. Expert gate/up tensors have shape ``[local_experts, hidden, 2 * expert_hidden]``
            and down tensors have shape ``[local_experts, expert_hidden, hidden]``. For a direct sharded load, outputs
            are DTensors with global expert shape and ``Shard(0)`` on the ``ep`` mesh; each local tensor uses the
            corresponding model weight memory. Normal conversion outputs use newly allocated memory.
        """
        loading_into_model_weights = self._loading_into_model_weights
        self._loading_into_model_weights = False
        self._uses_model_prefix = any(key.startswith("model.") for key in hf_state_dict)
        model_prefix = "model." if self._uses_model_prefix else ""

        n_experts = self.moe_config.n_routed_experts
        if device_mesh is not None:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            rank = (
                state_dict_utils.get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
        else:
            start_expert, end_expert = 0, n_experts
            rank = None

        # Collect MoE expert tensors per layer for combined processing
        expert_buffers: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        state_dict: dict[str, Any] = {}

        for key, value in hf_state_dict.items():
            # --- Router keys: router.{proj.weight,scale} -> moe.gate.{proj.weight,scale} ---
            router_match = re.search(r"(layers\.\d+)\.router\.(proj\.weight|scale|per_expert_scale)$", key)
            if router_match:
                layer_path = router_match.group(1)
                router_attr = router_match.group(2)
                if router_attr == "per_expert_scale":
                    expert_buffers[layer_path]["per_expert_scale"] = value
                else:
                    new_key = key.replace(f"{layer_path}.router.{router_attr}", f"{layer_path}.moe.gate.{router_attr}")
                    state_dict[new_key] = value
                continue

            # --- Expert weight keys ---
            expert_match = re.search(r"(layers\.\d+)\.(?:moe|experts)\.(gate_up_proj|down_proj|per_expert_scale)$", key)
            if expert_match:
                layer_path = expert_match.group(1)
                weight_name = expert_match.group(2)
                expert_buffers[layer_path][weight_name] = value
                continue

            # --- Pass-through keys ---
            state_dict[key] = value

        # Process collected expert weights per layer
        _REQUIRED_EXPERT_KEYS = {"gate_up_proj", "down_proj", "per_expert_scale"}
        for layer_path, tensors in expert_buffers.items():
            missing = _REQUIRED_EXPERT_KEYS - tensors.keys()
            if missing:
                raise RuntimeError(
                    f"Incomplete expert weights for {layer_path}: missing {missing}. "
                    f"Available keys: {list(tensors.keys())}"
                )

            gate_up_proj = tensors["gate_up_proj"]  # [E, 2*inter, hidden]
            down_proj = tensors["down_proj"]  # [E, hidden, inter]
            per_expert_scale = tensors["per_expert_scale"]  # [E]

            # Transpose gate_up_proj from HF [E, 2*inter, hidden] to NeMo [E, hidden, 2*inter]
            if loading_into_model_weights and state_dict_utils.is_dtensor(gate_up_proj):
                if not state_dict_utils.is_dtensor(down_proj) or not state_dict_utils.is_dtensor(per_expert_scale):
                    raise RuntimeError(
                        f"Inconsistent sharded load destinations for {layer_path}: gate/up, down, and scale must "
                        "all be DTensors."
                    )

                gate_and_up_local = gate_up_proj.transpose(-2, -1)
                down_local_tensor = down_proj.to_local()
                scale_local_tensor = per_expert_scale.to_local()
                if down_local_tensor.shape[0] != scale_local_tensor.shape[0]:
                    raise RuntimeError(
                        f"Sharded down projection for {layer_path} has {down_local_tensor.shape[0]} local experts, "
                        f"but its scale destination has {scale_local_tensor.shape[0]}."
                    )
                # The loaded down tensor already uses the model's weight memory. Apply its local expert scales to that
                # memory directly, without creating another expert tensor or communicating across the EP mesh.
                with torch.no_grad():
                    down_local_tensor.mul_(scale_local_tensor[:, None, None])
                down_local = down_proj.transpose(-2, -1)
            elif loading_into_model_weights:
                # DCP loaded these checkpoint tensors into the model's weight memory. Restore the native dimension
                # order, then apply the small per-expert scale values to the loaded down weights.
                gate_and_up_local = gate_up_proj.transpose(-2, -1)
                down_proj.mul_(per_expert_scale[:, None, None])
                down_local = down_proj.transpose(-2, -1)
            else:
                gate_and_up = gate_up_proj.transpose(-2, -1)  # [E, hidden, 2*inter]

                # Transpose down_proj from HF [E, hidden, inter] to NeMo [E, inter, hidden]
                # and absorb per_expert_scale
                down = down_proj.transpose(-2, -1) * per_expert_scale[:, None, None]  # [E, inter, hidden]

                # Slice for EP
                gate_and_up_local = gate_and_up[start_expert:end_expert].to(self.dtype)
                down_local = down[start_expert:end_expert].to(self.dtype)

            if loading_into_model_weights and state_dict_utils.is_dtensor(gate_and_up_local):
                # These transposed DTensors still use the original model weight memory. Their Shard(0) placement
                # already selects this rank's experts, so slicing or wrapping again would corrupt the global offsets
                # DCP used for the read.
                prefix = f"{model_prefix}language_model.{layer_path}"
                state_dict[f"{prefix}.moe.experts.gate_and_up_projs"] = gate_and_up_local
                state_dict[f"{prefix}.moe.experts.down_projs"] = down_local
                continue

            # Slice for EP_SHARD across the feature dimension before wrapping as DTensor.
            if device_mesh is not None and "ep_shard" in device_mesh.mesh_dim_names:
                ep_shard_mesh = state_dict_utils.get_submesh(device_mesh, ("ep_shard",))
                ep_shard_size = ep_shard_mesh.size()
                if ep_shard_size > 1:
                    ep_shard_rank = ep_shard_mesh.get_local_rank()
                    gate_shard_size = gate_and_up_local.shape[1] // ep_shard_size
                    gate_start = ep_shard_rank * gate_shard_size
                    gate_end = gate_start + gate_shard_size
                    gate_and_up_local = gate_and_up_local[:, gate_start:gate_end, :]

                    down_shard_size = down_local.shape[1] // ep_shard_size
                    down_start = ep_shard_rank * down_shard_size
                    down_end = down_start + down_shard_size
                    down_local = down_local[:, down_start:down_end, :]

            prefix = f"{model_prefix}language_model.{layer_path}"
            state_dict[f"{prefix}.moe.experts.gate_and_up_projs"] = state_dict_utils.create_dtensor_from_local(
                gate_and_up_local, device_mesh, rank
            )
            state_dict[f"{prefix}.moe.experts.down_projs"] = state_dict_utils.create_dtensor_from_local(
                down_local, device_mesh, rank
            )

        return state_dict

    # ------------------------------------------------------------------
    # NeMo -> HF
    # ------------------------------------------------------------------
    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert native Gemma4 weights to Hugging Face keys and layouts.

        Args:
            state_dict: Native state mapping. Expert gate/up tensors have shape
                ``[local_experts, hidden, 2 * expert_hidden]`` and down tensors have shape
                ``[local_experts, expert_hidden, hidden]``.
            exclude_key_regex: Optional pattern selecting keys to omit.
            quantization: Whether checkpoint initialization requires a precision conversion. Quantized loads do not
                load directly into the model's existing weight memory.
            **kwargs: Adapter-interface arguments. ``device_mesh`` describes expert sharding.
                ``for_checkpoint_load=True`` requests destinations for DCP to load.

        Returns:
            Hugging Face state mapping. Expert gate/up tensors have shape
            ``[experts, 2 * expert_hidden, hidden]`` and down tensors have shape ``[experts, hidden, expert_hidden]``.
            During a direct load, model-sized outputs use the existing model weight memory with transposed dimensions.
            EP outputs retain ``Shard(0)`` on the global experts axis, and each rank creates only its small local
            ``per_expert_scale`` slice.
        """
        self._uses_model_prefix = any(key.startswith("model.") for key in state_dict)
        prefix = "model." if self._uses_model_prefix else ""
        device_mesh: DeviceMesh | None = kwargs.get("device_mesh")
        n_experts = self.moe_config.n_routed_experts
        expert_tensors = [tensor for fqn, tensor in state_dict.items() if ".moe.experts." in fqn]
        single_device_can_load_into_model = (
            device_mesh is None
            and bool(expert_tensors)
            and all(
                isinstance(tensor, torch.Tensor) and not tensor.is_meta and not state_dict_utils.is_dtensor(tensor)
                for tensor in expert_tensors
            )
        )
        ep_can_load_into_model = (
            device_mesh is not None
            and bool(expert_tensors)
            and all(self._supports_ep_load_destination(tensor, n_experts) for tensor in expert_tensors)
        )
        load_into_model_storage = (
            bool(kwargs.get("for_checkpoint_load", False))
            and not quantization
            and (single_device_can_load_into_model or ep_can_load_into_model)
        )
        self._loading_into_model_weights = load_into_model_storage

        hf_state_dict: dict[str, Any] = {}

        for fqn, tensor in state_dict.items():
            # --- Router keys ---
            gate_match = re.search(r"(layers\.\d+)\.moe\.gate\.(proj\.weight|scale)$", fqn)
            if gate_match:
                layer_path = gate_match.group(1)
                gate_attr = gate_match.group(2)
                hf_key = fqn.replace(f"{layer_path}.moe.gate.{gate_attr}", f"{layer_path}.router.{gate_attr}")
                hf_state_dict[hf_key] = tensor
                continue

            # --- Expert: gate_and_up_projs -> experts.gate_up_proj ---
            if ".moe.experts.gate_and_up_projs" in fqn:
                layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
                layer_prefix = f"{prefix}language_model.layers.{layer_num}"
                if load_into_model_storage:
                    # This transposed tensor uses the native [E, hidden, 2*inter] model weight memory, so loading the
                    # HF [E, 2*inter, hidden] checkpoint into it updates the model directly.
                    hf_state_dict[f"{layer_prefix}.experts.gate_up_proj"] = tensor.transpose(-2, -1)
                else:
                    global_tensor = self._gather_expert_tensor(tensor, device_mesh, n_experts)
                    hf_state_dict[f"{layer_prefix}.experts.gate_up_proj"] = global_tensor.transpose(-2, -1).contiguous()
                continue

            # --- Expert: down_projs -> experts.down_proj + router.per_expert_scale ---
            if ".moe.experts.down_projs" in fqn:
                layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
                layer_prefix = f"{prefix}language_model.layers.{layer_num}"
                if load_into_model_storage:
                    # Load the raw HF down weight into the model's existing memory. ``from_hf`` then applies the small
                    # per-expert scale values to those loaded weights without creating another full-sized tensor.
                    hf_state_dict[f"{layer_prefix}.experts.down_proj"] = tensor.transpose(-2, -1)
                    if state_dict_utils.is_dtensor(tensor):
                        local_tensor = tensor.to_local()
                        local_scale = torch.empty(local_tensor.shape[0], dtype=self.dtype, device=local_tensor.device)
                        hf_state_dict[f"{layer_prefix}.router.per_expert_scale"] = DTensor.from_local(
                            local_scale,
                            tensor.device_mesh,
                            tensor.placements,
                            shape=torch.Size([n_experts]),
                            stride=(1,),
                        )
                    else:
                        hf_state_dict[f"{layer_prefix}.router.per_expert_scale"] = torch.empty(
                            n_experts, dtype=self.dtype, device=tensor.device
                        )
                else:
                    global_tensor = self._gather_expert_tensor(tensor, device_mesh, n_experts)
                    hf_state_dict[f"{layer_prefix}.experts.down_proj"] = global_tensor.transpose(-2, -1).contiguous()
                    hf_state_dict[f"{layer_prefix}.router.per_expert_scale"] = torch.ones(n_experts, dtype=self.dtype)
                continue

            # --- Pass-through ---
            hf_state_dict[fqn] = tensor

        if exclude_key_regex:
            hf_state_dict = {k: v for k, v in hf_state_dict.items() if not re.match(exclude_key_regex, k)}

        return hf_state_dict

    @staticmethod
    def _supports_ep_load_destination(tensor: Any, n_experts: int) -> bool:
        """Return whether a grouped expert DTensor can receive its HF checkpoint slice in place.

        Args:
            tensor: Native expert DTensor with global shape ``[experts, ...]`` and local shape
                ``[local_experts, ...]``. The ``ep`` mesh dimension must use ``Shard(0)``; every other mesh dimension
                must replicate the tensor. Inner-axis expert sharding is intentionally unsupported by this path.
            n_experts: Total number of routed experts in the checkpoint.

        Returns:
            ``True`` when transposing the final local tensor preserves an HF-layout ``Shard(0)`` destination that
            DCP can fill without gathering global experts.
        """
        if (
            not state_dict_utils.is_dtensor(tensor)
            or tensor.is_meta
            or tensor.shape[0] != n_experts
            or "ep" not in tensor.device_mesh.mesh_dim_names
        ):
            return False

        return all(
            (
                isinstance(placement, Shard)
                and placement.dim == 0
                and tensor.device_mesh.mesh_dim_names[mesh_dim] == "ep"
            )
            or (isinstance(placement, Replicate) and tensor.device_mesh.mesh_dim_names[mesh_dim] != "ep")
            for mesh_dim, placement in enumerate(tensor.placements)
        )

    def _gather_expert_tensor(
        self,
        tensor: torch.Tensor,
        device_mesh: DeviceMesh | None,
        n_experts: int,
    ) -> torch.Tensor:
        """Gather EP-sharded expert tensor across ranks into a full tensor."""
        if device_mesh is None:
            if state_dict_utils.is_dtensor(tensor):
                return tensor.to_local()
            return tensor

        if state_dict_utils.is_dtensor(tensor):
            split_weights, expert_ids = state_dict_utils.split_experts_weights_dtensor_aware(tensor, n_experts)
            # Individual expert weights may still be DTensors sharded over ep_shard.
            # Materialize them before object gather/copy into the CPU HF tensor.
            local_weights = [
                (weight.full_tensor() if state_dict_utils.is_dtensor(weight) else weight).to(self.dtype).cpu()
                for weight in split_weights
            ]
        else:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expert_ids = list(range(start_expert, end_expert))
            local_weights = [tensor[i].to(self.dtype).cpu() for i in range(tensor.shape[0])]

        global_tensor = torch.zeros(
            (n_experts, local_weights[0].shape[0], local_weights[0].shape[1]),
            dtype=self.dtype,
            device="cpu",
        )

        if dist.is_initialized() and "ep" in device_mesh.mesh_dim_names:
            try:
                ep_dim = device_mesh.mesh_dim_names.index("ep")
                ep_group = device_mesh.get_group(ep_dim)
            except Exception:
                ep_group = None

            if ep_group is not None:
                payload = (expert_ids, local_weights)
                gathered: list[tuple[list[int], list[torch.Tensor]]] = [None] * dist.get_world_size(ep_group)
                dist.all_gather_object(gathered, payload, group=ep_group)
                for ids, weights in gathered:
                    for eid, w in zip(ids, weights):
                        global_tensor[eid].copy_(w.to(self.dtype).cpu())
            else:
                for weight, expert_id in zip(local_weights, expert_ids):
                    global_tensor[expert_id].copy_(weight)
        else:
            for weight, expert_id in zip(local_weights, expert_ids):
                global_tensor[expert_id].copy_(weight)

        del local_weights, expert_ids
        return global_tensor

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single native tensor back to HF format.

        Handles per-tensor conversion for weight streaming (IPC refit) required in RL training:
        - Router keys: moe.gate.{proj.weight,scale} -> router.{proj.weight,scale}
        - Expert gate_and_up_projs: transpose [E, hidden, 2*inter] -> [E, 2*inter, hidden]
          and rename to experts.gate_up_proj
        - Expert down_projs: transpose [E, inter, hidden] -> [E, hidden, inter],
          rename to experts.down_proj, and emit router.per_expert_scale as ones
        """
        exclude_key_regex = kwargs.get("exclude_key_regex")
        if exclude_key_regex and re.match(exclude_key_regex, fqn):
            return []

        # --- Router keys: moe.gate.{attr} -> router.{attr} ---
        gate_match = re.search(r"(layers\.\d+)\.moe\.gate\.(proj\.weight|scale)$", fqn)
        if gate_match:
            layer_path = gate_match.group(1)
            gate_attr = gate_match.group(2)
            hf_key = fqn.replace(f"{layer_path}.moe.gate.{gate_attr}", f"{layer_path}.router.{gate_attr}")
            return [(hf_key, tensor)]

        # --- Expert: gate_and_up_projs -> experts.gate_up_proj (transposed) ---
        if ".moe.experts.gate_and_up_projs" in fqn:
            hf_key = fqn.replace(".moe.experts.gate_and_up_projs", ".experts.gate_up_proj")
            return [(hf_key, tensor.transpose(-2, -1).contiguous())]

        # --- Expert: down_projs -> experts.down_proj (transposed) + per_expert_scale ---
        if ".moe.experts.down_projs" in fqn:
            hf_key = fqn.replace(".moe.experts.down_projs", ".experts.down_proj")
            transposed = tensor.transpose(-2, -1).contiguous()
            layer_match = re.search(r"(.*layers\.\d+)\.", fqn)
            scale_key = f"{layer_match.group(1)}.router.per_expert_scale"
            n_experts = tensor.shape[0]
            return [
                (hf_key, transposed),
                (scale_key, torch.ones(n_experts, dtype=tensor.dtype)),
            ]

        # --- Pass-through for all other keys ---
        return [(fqn, tensor)]
