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

import gc
import os
import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    get_submesh,
    is_dtensor,
    should_load_expert_for_rank,
    split_experts_weights_dtensor_aware,
)

# Native LoRA suffixes for grouped MoE expert tensors
_LORA_EXPERT_SUFFIXES = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")


def get_world_size_safe() -> int:
    """Return the distributed world size before or after process-group initialization.

    Returns:
        The initialized process-group size, or ``WORLD_SIZE`` before initialization.
    """
    if torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


# peft flipped the in/out interpretation of non-transposed 3-D expert
# parameters in ParamWrapper in 0.19.1 (huggingface/peft#3165), which
# transposes every fused expert LoRA weight it creates. Export emits the
# corrected (>= 0.19.1) layout by default; the pre-flip layout stays
# available through the explicit ``legacy_paramwrapper_layout`` checkpoint
# option for consumers pinned to an older peft.
_PARAMWRAPPER_LAYOUT_METADATA_KEY = "paramwrapper_layout"
_PARAMWRAPPER_LAYOUT_MODERN = "peft-0.19.1"
_PARAMWRAPPER_LAYOUT_LEGACY = "peft-0.18"


class MoESplitExpertsStateDictMixin:
    """Mixin class providing MoE state dict conversion utilities.

    This mixin provides methods for:
    - Expert parallelism calculations (ranges, assignment)
    - Format conversion between HuggingFace and native formats
    - Both GroupedExperts and DeepEP format support
    - DTensor-aware expert loading and conversion

    Can be used by any MoE model that needs expert parallelism and format conversion.
    """

    # These attributes must be set by subclasses in their __init__ method:
    # - self.moe_config: MoE configuration object with expert settings
    # - self.config: Model configuration object
    # - self.backend: Backend configuration object

    @property
    def supports_low_memory_dcp_load(self) -> bool:
        """Whether DCP needs at most small temporary tensors for this MoE checkpoint."""
        if not self._supports_low_memory_dcp_load or self.moe_config is None:
            return self._supports_low_memory_dcp_load
        return self._expert_checkpoint_tensors_use_model_storage and not self.moe_config.expert_bias

    @property
    def _expert_checkpoint_tensors_use_model_storage(self) -> bool:
        """Whether grouped expert checkpoint tensors use the model's weight memory.

        This covers only the shared expert conversion. A concrete adapter must also verify that its non-expert
        checkpoint tensors require at most small temporary tensors before enabling low-memory DCP loading.
        """
        return self._grouped_expert_storage_is_model_weight and self.backend.dispatcher != "mok"

    @property
    def _grouped_expert_storage_is_model_weight(self) -> bool:
        """Whether the runtime grouped expert tensors use the model's parameter storage.

        This mirrors the expert implementation selected by ``MoE.__init__``. EP dispatchers use ordinary
        ``GroupedExperts`` at world size one, ``GroupedExpertsDeepEP`` for the grouped backends at larger world sizes,
        and ``GroupedExpertsTE`` otherwise. Non-EP dispatchers construct ordinary grouped experts.
        """
        if self.backend.dispatcher == "mok":
            return self.backend.experts != "te"
        if self.backend.dispatcher not in {"deepep", "hybridep", "uccl_ep"}:
            return True
        if get_world_size_safe() == 1:
            return True
        return self.backend.experts in {"gmm", "torch_mm", "torch_mm_mxfp8"}

    @property
    def _is_gated_moe(self) -> bool:
        """Check if the MoE uses gated activation (e.g., SwiGLU) or non-gated (e.g., ReLU²)."""
        from nemo_automodel.components.moe.experts import is_gated_activation

        return is_gated_activation(self.moe_config.expert_activation)

    def _register_inplace_loaded_key(self, fqn: str, prefix_override: str | None) -> None:
        """Mark ``fqn`` as loaded via in-place views so ``_from_hf_w_merged_experts`` skips its rebuild.

        The tracked key must match the native_key that the from_hf merge loop
        reconstructs from the HF per-expert keys. For backbone tensors the
        native_key equals ``fqn``; for MTP tensors (``prefix_override="mtp."``)
        the HF keys live under the ``mtp.`` namespace and from_hf processes
        them with that prefix stripped, so the tracked key is also the
        ``mtp.``-less form. The user of this set (``_from_hf_w_merged_experts``)
        receives the matching stripped key when called via the adapter's
        per-namespace dispatch.
        """
        if prefix_override is not None and prefix_override.endswith("."):
            tracked = fqn[len(prefix_override) :] if fqn.startswith(prefix_override) else fqn
        else:
            tracked = fqn
        if not hasattr(self, "_inplace_loaded_native_keys") or self._inplace_loaded_native_keys is None:
            self._inplace_loaded_native_keys = set()
        self._inplace_loaded_native_keys.add(tracked)

    @property
    def view_loaded_native_keys(self) -> set[str]:
        """Native keys loaded in-place via strided views during the most recent ``from_hf``.

        MoE experts with a plain local split are loaded by DCP writing the checkpoint tensors
        straight through non-contiguous strided views into the model's grouped expert storage.
        Such keys are intentionally absent from the dict ``from_hf`` returns (the data is already
        in the model) but are NOT missing. ``_from_hf_w_merged_experts`` records them here so the
        checkpoint loader can exclude them from false "missing" key-diff warnings. The record is
        reset at the start of each load by ``_from_hf_w_merged_experts(reset_view_loaded_keys=True)``.
        """
        return getattr(self, "_view_loaded_native_keys", None) or set()

    @property
    def _hf_prefix(self) -> str:
        """Prefix for HuggingFace format keys. Override in subclass."""
        return "model." if self._uses_model_prefix else ""

    @property
    def _expert_path_segment(self) -> str:
        """Path segment for experts (e.g., 'mlp.experts' or 'mixer.experts'). Override in subclass."""
        return "mlp.experts"

    @property
    def _v5_peft_target_parameters(self) -> tuple[str, ...]:
        """Fused expert parameters validated for PEFT v5 ParamWrapper export.

        Adapters opt in by overriding this property. Keeping the default empty
        preserves the legacy per-expert export for model families whose HF
        naming, activation layout, or checkpoint post-processing has not been
        validated against ParamWrapper yet.
        """
        return ()

    def _v5_peft_hf_expert_path_segment(self) -> str:
        """Return the common HF module path that owns the fused expert parameters."""
        target_parameters = self._v5_peft_target_parameters
        if not target_parameters:
            return self._expert_path_segment

        expert_segments = {target.rsplit(".", 1)[0] for target in target_parameters}
        if len(expert_segments) != 1:
            raise ValueError(
                f"PEFT v5 expert target parameters must share one parent module, got {sorted(target_parameters)}"
            )
        return next(iter(expert_segments))

    def _validate_expert_availability(
        self,
        hf_state_dict: dict[str, Any],
        n_experts: int,
        device_mesh: Optional["DeviceMesh"] = None,
    ) -> None:
        """Validate that all required experts are available in the HF state dict before loading.
        Only validates experts needed for the current rank and layers present in the state dict.
        Expert groups already loaded through registered in-place views validate the rank-local IDs recorded by
        ``_split_experts_weights``, including when EP is disabled and no MoE mesh is passed to this method.

        Args:
            hf_state_dict: HuggingFace state mapping. Expert gate/up values are tensors of shape
                [expert_hidden, hidden], and down values have shape [hidden, expert_hidden]. This method validates
                their keys only.
            n_experts: Total number of experts.
            device_mesh: Optional device mesh whose ``ep`` dimension partitions the experts axis.

        Raises:
            RuntimeError: If required expert weights are missing from the checkpoint.
        """
        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            required_experts = list(range(start_expert, end_expert))
            rank = (
                get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
            rank_info = f" (rank {rank})"
        else:
            required_experts = list(range(n_experts))
            rank_info = ""

        expert_segment = self._expert_path_segment

        # Detect actual prefix from keys (handles both HF format and pre-renamed internal format)
        key_prefix = ""
        for key in hf_state_dict.keys():
            if f".{expert_segment}." in key and "layers." in key:
                key_prefix = key[: key.index("layers.")]
                break

        # Build list of all possible prefixes
        prefixes = ["model.language_model.", "model.", "language_model.", ""]
        if key_prefix and key_prefix not in prefixes:
            prefixes.insert(0, key_prefix)

        layers_with_experts: dict[int, set[str]] = {}
        # Create pattern with all prefixes
        escaped_prefixes = [re.escape(p) for p in prefixes]
        prefix_pattern = "(?P<prefix>" + "|".join(escaped_prefixes) + ")"
        pattern = (
            rf"{prefix_pattern}layers\.(\d+)\.{re.escape(expert_segment)}\.\d+\.(gate_proj|up_proj|down_proj)\.weight"
        )
        for key in hf_state_dict.keys():
            match = re.match(pattern, key)
            if match:
                prefix = match.group("prefix") or ""
                layer_num = int(match.group(2))
                layers_with_experts.setdefault(layer_num, set()).add(prefix)

        if not layers_with_experts:
            return

        missing_weights = []
        projection_types = ["gate_proj", "up_proj", "down_proj"] if self._is_gated_moe else ["up_proj", "down_proj"]
        inplace_loaded_keys: set[str] = getattr(self, "_inplace_loaded_native_keys", None) or set()
        inplace_expert_ids = getattr(self, "_last_expert_ids", None) or required_experts
        total_required = 0

        for layer_num, prefixes in layers_with_experts.items():
            for prefix in prefixes:
                for proj_type in projection_types:
                    native_projection = "down_projs" if proj_type == "down_proj" else "gate_and_up_projs"
                    native_key = f"{prefix}layers.{layer_num}.{expert_segment}.{native_projection}"
                    experts_to_validate = inplace_expert_ids if native_key in inplace_loaded_keys else required_experts
                    total_required += len(experts_to_validate)
                    for expert_id in experts_to_validate:
                        expected_key = f"{prefix}layers.{layer_num}.{expert_segment}.{expert_id}.{proj_type}.weight"
                        if expected_key not in hf_state_dict:
                            missing_weights.append(expected_key)

        if missing_weights:
            missing_count = len(missing_weights)
            raise RuntimeError(
                f"Expert weights missing from checkpoint{rank_info}: {missing_count}/{total_required} required weights not found. "
                f"Cannot load experts - checkpoint may be incomplete or corrupted. "
                f"Layers with experts: {sorted(layers_with_experts)}, Required experts: {required_experts}. "
                f"First few missing keys: {missing_weights[:5]}"
                + (f" (and {missing_count - 5} more)" if missing_count > 5 else "")
            )

    def _split_experts_weights(self, weight: torch.Tensor, n_experts: int) -> list[torch.Tensor]:
        """Split grouped expert weights into per-expert tensors.

        Args:
            weight: Tensor of shape [experts, ...], with arbitrary trailing dimensions. An EP DTensor uses an
                ``ep`` mesh dimension. A non-EP DTensor may use any FSDP placement on a mesh without ``ep``.
            n_experts: Global number of experts in ``weight``.

        Returns:
            Per-expert tensors of shape [...]. A DTensor sharded on the expert axis returns only the experts local
            to this rank; other DTensors return every expert and preserve adjusted placements.
        """
        if is_dtensor(weight):
            split_weights, expert_ids = split_experts_weights_dtensor_aware(weight, n_experts)
            self._last_expert_ids = expert_ids
            return split_weights
        else:
            if weight.shape[0] != n_experts:
                raise ValueError(f"Expected first dimension to be {n_experts}, got {weight.shape[0]}")

            split_weights = []
            expert_ids = []
            for i in range(n_experts):
                expert_weight = weight[i]  # Shape: [...] (expert dimension removed)
                split_weights.append(expert_weight)
                expert_ids.append(i)

            self._last_expert_ids = expert_ids
            return split_weights

    def _concatenate_expert_weights(
        self, expert_weights_by_layer: dict[str, Any], n_experts: int
    ) -> torch.Tensor | None:
        """Concatenate the weights of separate experts into GroupedExpert weights.

        Args:
            expert_weights_by_layer: Nested dict structure containing expert weights
            n_experts: Total number of experts expected

        Returns:
            Stacked tensor if all experts are available for a layer, None otherwise
        """
        for layer, abstract_keys in list(expert_weights_by_layer.items()):
            for abstract_key, experts in list(abstract_keys.items()):
                if len(experts) == n_experts:
                    sorted_expert_ids = sorted(experts.keys())
                    sorted_experts = [experts[i] for i in sorted_expert_ids]
                    stacked_tensor = torch.stack(sorted_experts, dim=0)

                    del expert_weights_by_layer[layer][abstract_key]
                    if not expert_weights_by_layer[layer]:
                        del expert_weights_by_layer[layer]

                    return stacked_tensor

        return None

    def _convert_lora_to_paramwrapper(
        self, fqn: str, tensor: torch.Tensor, legacy_layout: bool = False
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert a single grouped MoE LoRA tensor to PEFT ParamWrapper format.

        ParamWrapper format stores fused 3-D expert LoRA parameters as 2-D
        tensors with the expert dimension folded into the rank dimension.
        peft flipped the in/out interpretation of these parameters in 0.19.1
        (huggingface/peft#3165); the corrected layout is emitted by default,
        and *legacy_layout* selects the pre-flip layout for consumers pinned
        to peft <= 0.19.0.

        Shape mapping (automodel native -> ParamWrapper), default (peft >= 0.19.1):

        down_proj (outer wrapper, NO ``base_layer`` prefix — processed first alphabetically):
          - ``lora_down_A``  (E, I, r) -> ``lora_A.weight``  (r*E, I)  permute+reshape
          - ``lora_down_B``  (E, r, H) -> ``lora_B.weight``  (H, r*E)  permute+reshape

        input projection (``gate_up_proj`` or ``up_proj``; inner wrapper, HAS
        ``base_layer.`` prefix):
          - ``lora_gate_and_up_A``  (E, H, r) -> ``base_layer.lora_A.weight``  (r*E, H)  permute+reshape
          - ``lora_gate_and_up_B``  (E, r, U) -> ``base_layer.lora_B.weight``  (U, r*E)  permute+reshape

        and with ``legacy_layout=True`` (peft <= 0.19.0, the pre-flip layout):
          - ``lora_down_B``  (E, r, H) -> ``lora_A.weight``  (r*E, H)  reshape
          - ``lora_down_A``  (E, I, r) -> ``lora_B.weight``  (I, r*E)  permute+reshape
          - ``lora_gate_and_up_B``  (E, r, U) -> ``base_layer.lora_A.weight``  (r*E, U)  reshape
          - ``lora_gate_and_up_A``  (E, H, r) -> ``base_layer.lora_B.weight``  (H, r*E)  permute+reshape

        Returns:
            List containing one ``(fqn, tensor)`` tuple in ParamWrapper format.
        """
        match = re.search(r"(.*)layers\.(\d+)\.", fqn)
        if not match:
            return [(fqn, tensor)]

        prefix = match.group(1)
        layer_num = match.group(2)
        expert_segment = self._v5_peft_hf_expert_path_segment()
        suffix = fqn.rsplit(".", 1)[-1]
        swapped = not legacy_layout

        # PEFT ParamWrapper nesting: target_parameters are sorted alphabetically
        # and wrapped in order. The FIRST wrapped becomes the OUTER ParamWrapper.
        # "down_proj" < "gate_up_proj", so down_proj is outer (no base_layer prefix)
        # and gate_up_proj is inner (has base_layer prefix).
        if suffix == "lora_gate_and_up_B":
            if swapped:
                # (E, r, U) -> permute(2,1,0) -> (U, r, E) -> (U, r*E)
                out = tensor.permute(2, 1, 0).contiguous().reshape(tensor.shape[2], -1)
                pw_suffix = "base_layer.lora_B.weight"
            else:
                # (E, r, U) -> (r*E, U)
                out = tensor.reshape(-1, tensor.shape[2]).contiguous()
                pw_suffix = "base_layer.lora_A.weight"
        elif suffix == "lora_gate_and_up_A":
            if swapped:
                # (E, H, r) -> permute(0,2,1) -> (E, r, H) -> (r*E, H)
                out = tensor.permute(0, 2, 1).contiguous().reshape(-1, tensor.shape[1])
                pw_suffix = "base_layer.lora_A.weight"
            else:
                # (E, H, r) -> permute(1,2,0) -> (H, r, E) -> (H, r*E)
                out = tensor.permute(1, 2, 0).contiguous().reshape(tensor.shape[1], -1)
                pw_suffix = "base_layer.lora_B.weight"
        elif suffix == "lora_down_B":
            if swapped:
                # (E, r, H) -> permute(2,1,0) -> (H, r, E) -> (H, r*E)
                out = tensor.permute(2, 1, 0).contiguous().reshape(tensor.shape[2], -1)
                pw_suffix = "lora_B.weight"
            else:
                # (E, r, H) -> (r*E, H)
                out = tensor.reshape(-1, tensor.shape[2]).contiguous()
                pw_suffix = "lora_A.weight"
        elif suffix == "lora_down_A":
            if swapped:
                # (E, I, r) -> permute(0,2,1) -> (E, r, I) -> (r*E, I)
                out = tensor.permute(0, 2, 1).contiguous().reshape(-1, tensor.shape[1])
                pw_suffix = "lora_A.weight"
            else:
                # (E, I, r) -> permute(1,2,0) -> (I, r, E) -> (I, r*E)
                out = tensor.permute(1, 2, 0).contiguous().reshape(tensor.shape[1], -1)
                pw_suffix = "lora_B.weight"
        else:
            return [(fqn, tensor)]

        out_fqn = f"{prefix}layers.{layer_num}.{expert_segment}.{pw_suffix}"
        return [(out_fqn, out)]

    def _convert_paramwrapper_to_native(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert PEFT ParamWrapper LoRA keys to native grouped MoE LoRA format.

        This is the reverse of ``_convert_lora_to_paramwrapper``. It detects
        ParamWrapper-format keys and converts them back to the 3-D grouped
        tensors expected by GroupedExpertsLoRA. Because peft flipped the
        ParamWrapper layout in 0.19.1 (huggingface/peft#3165), the file's
        layout is resolved once per call: from the checkpoint's metadata stamp
        when the loader provided one (``_paramwrapper_layout_hint``), otherwise
        from the tensors whose shapes distinguish the two generations, so
        adapters exported under either peft load correctly. When neither
        settles it, a clear error is raised instead of guessing. The stamp
        must be one of the two known layout ids and must agree with whatever
        the shapes say; a stamp that fails either check is rejected, not trusted.

        Reverse transforms (down_proj is outer, the input projection is inner),
        peft >= 0.19.1 layout:
          - ``experts.lora_A.weight``            (r*E, I)  -> (E, I, r)    = lora_down_A
          - ``experts.lora_B.weight``            (H, r*E)  -> (E, r, H)    = lora_down_B
          - ``experts.base_layer.lora_A.weight`` (r*E, H)  -> (E, H, r)    = lora_gate_and_up_A
          - ``experts.base_layer.lora_B.weight`` (U, r*E)  -> (E, r, U)    = lora_gate_and_up_B

        peft <= 0.19.0 layout:
          - ``experts.lora_A.weight``            (r*E, H)  -> (E, r, H)    = lora_down_B
          - ``experts.lora_B.weight``            (I, r*E)  -> (E, I, r)    = lora_down_A
          - ``experts.base_layer.lora_A.weight`` (r*E, U)  -> (E, r, U)    = lora_gate_and_up_B
          - ``experts.base_layer.lora_B.weight`` (H, r*E)  -> (E, H, r)    = lora_gate_and_up_A
        """
        hf_expert_segment = re.escape(self._v5_peft_hf_expert_path_segment())
        n_experts = self.moe_config.n_routed_experts
        moe_inter = self.moe_config.moe_inter_dim
        gate_up_width = 2 * moe_inter if self._is_gated_moe else moe_inter
        # The model dim tells us when the two layouts have identical shapes;
        # not every moe_config carries it.
        dim = getattr(self.moe_config, "dim", None)
        down_ambiguous = dim is not None and dim == moe_inter
        gate_up_ambiguous = dim is not None and dim == gate_up_width

        # Detect ParamWrapper keys
        pw_pattern = re.compile(
            rf"(?P<prefix>.*)layers\.(?P<layer>\d+)\.{hf_expert_segment}\."
            rf"(?P<pw_suffix>(?:base_layer\.)?lora_[AB]\.weight)$"
        )

        # A checkpoint is written under exactly one layout, so any tensor whose
        # shapes distinguish the two generations identifies the layout for the
        # whole file, including the tensors whose own shapes cannot (e.g.
        # MiniMax M2's gate_up pair, where hidden == 2 * moe intermediate).
        matches: list[tuple[str, str, Any, str]] = []
        votes: set[bool] = set()
        for key, tensor in state_dict.items():
            m = pw_pattern.match(key)
            if m is None:
                continue
            pw_suffix = m.group("pw_suffix")
            # Preserve the full prefix from the input key (e.g. "base_model.model.model.")
            # so downstream prefix stripping (_drop_outer_prefix) works correctly.
            base_key = f"{m.group('prefix')}layers.{m.group('layer')}.{self._expert_path_segment}"
            matches.append((key, base_key, tensor, pw_suffix))
            if pw_suffix == "lora_A.weight" and not down_ambiguous:
                votes.add(tensor.shape[1] == moe_inter)
            elif pw_suffix == "lora_B.weight" and not down_ambiguous:
                votes.add(tensor.shape[0] != moe_inter)
            elif pw_suffix == "base_layer.lora_A.weight" and not gate_up_ambiguous:
                votes.add(tensor.shape[1] != gate_up_width)
            elif pw_suffix == "base_layer.lora_B.weight" and not gate_up_ambiguous:
                votes.add(tensor.shape[0] == gate_up_width)

        if not matches:
            return state_dict

        # Metadata first: the save path stamps the layout into
        # automodel_peft_config.json and the checkpoint loader hands it to us
        # through this attribute; shape detection is the fallback for adapters
        # saved before the stamp existed. The two must agree: an unknown stamp,
        # or one the tensor shapes rule out, is an error rather than a guess.
        if len(votes) > 1:
            raise ValueError(
                "ParamWrapper LoRA tensors in this adapter disagree about their layout "
                "(peft flipped it in 0.19.1, huggingface/peft#3165). The adapter file "
                "does not match this model's dimensions, or it is corrupt."
            )
        shape_vote = next(iter(votes)) if votes else None
        layout_hint = getattr(self, "_paramwrapper_layout_hint", None)
        if layout_hint is not None:
            if layout_hint not in (_PARAMWRAPPER_LAYOUT_MODERN, _PARAMWRAPPER_LAYOUT_LEGACY):
                raise ValueError(
                    f"Unknown peft ParamWrapper layout {layout_hint!r} in this adapter's "
                    f"automodel_peft_config.json; expected {_PARAMWRAPPER_LAYOUT_MODERN!r} or "
                    f"{_PARAMWRAPPER_LAYOUT_LEGACY!r}. The metadata is corrupt, or the adapter "
                    "was exported by a newer automodel than this one."
                )
            swapped = layout_hint == _PARAMWRAPPER_LAYOUT_MODERN
            if shape_vote is not None and shape_vote != swapped:
                raise ValueError(
                    f"This adapter's automodel_peft_config.json says its ParamWrapper layout is "
                    f"{layout_hint!r}, but the LoRA tensor shapes only fit the "
                    f"{(_PARAMWRAPPER_LAYOUT_LEGACY if swapped else _PARAMWRAPPER_LAYOUT_MODERN)!r} layout "
                    "(peft flipped it in 0.19.1, huggingface/peft#3165). The metadata and the weights "
                    "do not belong together, or the adapter does not match this model's dimensions."
                )
        elif shape_vote is not None:
            swapped = shape_vote
        else:
            raise ValueError(
                "Cannot tell which peft ParamWrapper layout this adapter uses: the "
                "model's dimensions make the peft 0.18 and peft >= 0.19.1 layouts the "
                "same shape (peft flipped the layout in 0.19.1, huggingface/peft#3165) "
                "and the checkpoint carries no layout metadata. Re-export the adapter "
                "with a current automodel (which stamps the layout into "
                "automodel_peft_config.json), or load it with the matching peft directly."
            )

        consumed_keys: set[str] = set()
        new_entries: dict[str, torch.Tensor] = {}

        for key, base_key, tensor, pw_suffix in matches:
            # down_proj is outer (no base_layer), gate_up_proj is inner (base_layer)
            if pw_suffix == "lora_A.weight":
                r = tensor.shape[0] // n_experts
                if swapped:
                    # swapped: (r*E, I) -> (E, r, I) -> permute(0,2,1) -> (E, I, r) = lora_down_A
                    out = tensor.reshape(n_experts, r, moe_inter).permute(0, 2, 1).contiguous()
                    new_entries[f"{base_key}.lora_down_A"] = out
                else:
                    # legacy: (r*E, H) -> (E, r, H) = lora_down_B
                    out = tensor.reshape(n_experts, r, tensor.shape[1]).contiguous()
                    new_entries[f"{base_key}.lora_down_B"] = out

            elif pw_suffix == "lora_B.weight":
                r = tensor.shape[1] // n_experts
                if swapped:
                    # swapped: (H, r*E) -> (H, r, E) -> permute(2,1,0) -> (E, r, H) = lora_down_B
                    out = tensor.reshape(tensor.shape[0], r, n_experts).permute(2, 1, 0).contiguous()
                    new_entries[f"{base_key}.lora_down_B"] = out
                else:
                    # legacy: (I, r*E) -> (I, r, E) -> permute(2,0,1) -> (E, I, r) = lora_down_A
                    out = tensor.reshape(moe_inter, r, n_experts).permute(2, 0, 1).contiguous()
                    new_entries[f"{base_key}.lora_down_A"] = out

            elif pw_suffix == "base_layer.lora_A.weight":
                r = tensor.shape[0] // n_experts
                if swapped:
                    # swapped: (r*E, H) -> (E, r, H) -> permute(0,2,1) -> (E, H, r) = lora_gate_and_up_A
                    out = tensor.reshape(n_experts, r, tensor.shape[1]).permute(0, 2, 1).contiguous()
                    new_entries[f"{base_key}.lora_gate_and_up_A"] = out
                else:
                    # legacy: (r*E, U) -> (E, r, U) = lora_gate_and_up_B
                    out = tensor.reshape(n_experts, r, gate_up_width).contiguous()
                    new_entries[f"{base_key}.lora_gate_and_up_B"] = out

            elif pw_suffix == "base_layer.lora_B.weight":
                r = tensor.shape[1] // n_experts
                if swapped:
                    # swapped: (U, r*E) -> (U, r, E) -> permute(2,1,0) -> (E, r, U) = lora_gate_and_up_B
                    out = tensor.reshape(gate_up_width, r, n_experts).permute(2, 1, 0).contiguous()
                    new_entries[f"{base_key}.lora_gate_and_up_B"] = out
                else:
                    # legacy: (H, r*E) -> (H, r, E) -> permute(2,0,1) -> (E, H, r) = lora_gate_and_up_A
                    out = tensor.reshape(tensor.shape[0], r, n_experts).permute(2, 0, 1).contiguous()
                    new_entries[f"{base_key}.lora_gate_and_up_A"] = out

            else:
                continue

            consumed_keys.add(key)

        result = {k: v for k, v in state_dict.items() if k not in consumed_keys}
        result.update(new_entries)
        return result

    def _convert_lora_expert_to_hf(
        self,
        fqn: str,
        tensor: torch.Tensor,
        n_experts: int,
        inter_dim: int,
        expert_segment: str,
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert a grouped MoE expert LoRA tensor to per-expert HF PEFT format.

        Handles the four LoRA parameter types produced by GroupedExpertsLoRA /
        GroupedExpertsDeepEPLoRA and converts them to per-expert ``lora_A.weight``
        / ``lora_B.weight`` keys that HF PEFT understands.

        The prefix (e.g. ``base_model.model.model.``) is preserved from the
        incoming *fqn* so that both PEFT and FFT save paths work correctly.
        """
        match = re.search(r"(.*)layers\.(\d+)\.", fqn)
        if not match:
            return None
        fqn_prefix = match.group(1)
        layer_num = match.group(2)
        suffix = fqn.rsplit(".", 1)[-1]

        splits = self._split_experts_weights(tensor, n_experts)
        result: list[tuple[str, torch.Tensor]] = []

        for i, w in enumerate(splits):
            expert_id = self._last_expert_ids[i]
            base = f"{fqn_prefix}layers.{layer_num}.{expert_segment}.{expert_id}"

            if suffix == "lora_gate_and_up_A":
                # [dim, lora_dim] -> [lora_dim, dim] (nn.Linear convention)
                w_t = w.transpose(0, 1).contiguous()
                if self._is_gated_moe:
                    result.append((f"{base}.gate_proj.lora_A.weight", w_t))
                    result.append((f"{base}.up_proj.lora_A.weight", w_t.clone()))
                else:
                    result.append((f"{base}.up_proj.lora_A.weight", w_t))

            elif suffix == "lora_gate_and_up_B":
                # [lora_dim, 2*inter] (gated) or [lora_dim, inter] (non-gated)
                if self._is_gated_moe:
                    w_gate = w[:, :inter_dim].transpose(0, 1).contiguous()
                    w_up = w[:, inter_dim:].transpose(0, 1).contiguous()
                    result.append((f"{base}.gate_proj.lora_B.weight", w_gate))
                    result.append((f"{base}.up_proj.lora_B.weight", w_up))
                else:
                    result.append((f"{base}.up_proj.lora_B.weight", w.transpose(0, 1).contiguous()))

            elif suffix == "lora_down_A":
                # [inter_dim, lora_dim] -> [lora_dim, inter_dim]
                result.append((f"{base}.down_proj.lora_A.weight", w.transpose(0, 1).contiguous()))

            elif suffix == "lora_down_B":
                # [lora_dim, dim] -> [dim, lora_dim]
                result.append((f"{base}.down_proj.lora_B.weight", w.transpose(0, 1).contiguous()))

        return result

    def _recombine_lora_expert_keys(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Recombine per-expert HF LoRA keys back to grouped MoE LoRA format.

        This is the reverse of ``_convert_lora_expert_to_hf``.  It detects
        per-expert LoRA keys (e.g.
        ``layers.0.mlp.experts.0.gate_proj.lora_A.weight``) and recombines
        them into the grouped tensors expected by GroupedExpertsLoRA /
        GroupedExpertsDeepEPLoRA (e.g. ``layers.0.mlp.experts.lora_gate_and_up_A``).
        """
        expert_segment = re.escape(self._expert_path_segment)
        n_experts = self.moe_config.n_routed_experts

        lora_pattern = re.compile(
            rf"(?P<prefix>.*)layers\.(?P<layer>\d+)\.{expert_segment}\."
            rf"(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)\.(?P<lora>lora_[AB])\.weight"
        )

        # Group: (prefix, layer, proj, lora) -> {expert_id: tensor}
        lora_groups: dict[tuple, dict[int, torch.Tensor]] = {}
        consumed_keys: set[str] = set()

        for key, value in state_dict.items():
            m = lora_pattern.match(key)
            if m:
                group_key = (m.group("prefix"), m.group("layer"), m.group("proj"), m.group("lora"))
                lora_groups.setdefault(group_key, {})[int(m.group("expert"))] = value
                consumed_keys.add(key)

        if not consumed_keys:
            return state_dict

        result = {k: v for k, v in state_dict.items() if k not in consumed_keys}
        processed: set[tuple] = set()

        for (prefix, layer, proj, lora), experts in lora_groups.items():
            group_id = (prefix, layer, proj, lora)
            if group_id in processed:
                continue
            processed.add(group_id)

            if len(experts) != n_experts:
                for eid, t in experts.items():
                    orig_seg = self._expert_path_segment
                    result[f"{prefix}layers.{layer}.{orig_seg}.{eid}.{proj}.{lora}.weight"] = t
                continue

            sorted_ids = sorted(experts.keys())
            base_key = f"{prefix}layers.{layer}.{self._expert_path_segment}"

            if proj in ("gate_proj", "up_proj") and lora == "lora_A":
                if self._is_gated_moe and proj == "up_proj":
                    continue  # gate_proj.lora_A already produces this (deduplicate)
                tensors = [experts[eid].transpose(0, 1).contiguous() for eid in sorted_ids]
                result[f"{base_key}.lora_gate_and_up_A"] = torch.stack(tensors, dim=0)
                if self._is_gated_moe:
                    processed.add((prefix, layer, "up_proj", "lora_A"))

            elif proj in ("gate_proj", "up_proj") and lora == "lora_B":
                if self._is_gated_moe and proj == "up_proj":
                    continue  # handled by gate_proj.lora_B below
                if self._is_gated_moe:
                    up_key = (prefix, layer, "up_proj", "lora_B")
                    up_experts = lora_groups.get(up_key, {})
                    if len(up_experts) != n_experts:
                        for eid, t in experts.items():
                            orig_seg = self._expert_path_segment
                            result[f"{prefix}layers.{layer}.{orig_seg}.{eid}.{proj}.{lora}.weight"] = t
                        continue
                    gate_ts = [experts[eid].transpose(0, 1).contiguous() for eid in sorted_ids]
                    up_ts = [up_experts[eid].transpose(0, 1).contiguous() for eid in sorted_ids]
                    combined = torch.cat([torch.stack(gate_ts, dim=0), torch.stack(up_ts, dim=0)], dim=-1)
                    result[f"{base_key}.lora_gate_and_up_B"] = combined
                    processed.add(up_key)
                else:
                    tensors = [experts[eid].transpose(0, 1).contiguous() for eid in sorted_ids]
                    result[f"{base_key}.lora_gate_and_up_B"] = torch.stack(tensors, dim=0)

            elif proj == "down_proj":
                native_suffix = "lora_down_A" if lora == "lora_A" else "lora_down_B"
                tensors = [experts[eid].transpose(0, 1).contiguous() for eid in sorted_ids]
                result[f"{base_key}.{native_suffix}"] = torch.stack(tensors, dim=0)

        return result

    def _to_hf_w_split_experts(self, state_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Convert DeepEP format to HuggingFace format.

        Handles ``gate_and_up_projs`` / ``down_projs`` -> individual expert
        weights. Forwards ``**kwargs`` to
        ``_convert_single_merged_expert_to_hf_split_experts`` for adapter
        compatibility (e.g. ``exclude_key_regex``).
        """
        hf_state_dict: dict[str, Any] = {}

        for fqn, tensor in state_dict.items():
            converted = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **kwargs)
            if converted is not None:
                for key, value in converted:
                    hf_state_dict[key] = value
            else:
                hf_state_dict[fqn] = tensor

        return hf_state_dict

    def _from_hf_w_merged_experts(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        reset_view_loaded_keys: bool = True,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to native format.

        For gated activations (SwiGLU, Quick-GEGLU):
            Creates combined gate_and_up_projs [n_experts, dim, 2*inter_dim] and
            transposed down_projs tensors.

        For non-gated activations (ReLU²):
            Creates gate_and_up_projs [n_experts, dim, inter_dim] and transposed down_projs tensors.

        Args:
            hf_state_dict: State mapping consumed by this method. Per-expert gate and up tensors have shape
                [expert_hidden, hidden], while down tensors have shape [hidden, expert_hidden]. DTensor values
                use the same global layouts and are localized before merging.
            device_mesh: Optional device mesh whose expert-parallel dimension selects the local experts. The
                returned grouped expert tensors use the placements created by ``create_dtensor_from_local``.
            reset_view_loaded_keys: Clear the in-place (strided-view) loaded-key record at the
                start of this call. A single ``from_hf`` may invoke this method more than once
                (e.g. backbone then MTP merge); the later call(s) pass ``False`` so the view-loaded
                keys accumulate across one logical load. Resetting here (rather than in the loader)
                keeps the whole view-key lifecycle inside the adapter and ensures each load starts
                clean (no leak from a prior load such as an init-time partial load).

        Returns:
            Native state mapping. Gated input projections have shape
            [local_experts, hidden, 2 * expert_hidden], non-gated input projections have shape
            [local_experts, hidden, expert_hidden], and down projections have shape
            [local_experts, expert_hidden, hidden].
        """
        expert_segment = self._expert_path_segment
        if self.moe_config is not None and self.moe_config.expert_bias:
            split_bias_pattern = re.compile(
                rf"(?:^|\.){re.escape(expert_segment)}\.\d+\.(?:gate_proj|up_proj|down_proj)\.bias$"
            )
            unsupported_bias_key = next((key for key in hf_state_dict if split_bias_pattern.search(key)), None)
            if unsupported_bias_key is not None:
                raise NotImplementedError(
                    "Loading Hugging Face per-expert bias tensors with expert_bias=True is not implemented; "
                    f"refusing key {unsupported_bias_key!r}."
                )

        if reset_view_loaded_keys:
            self._view_loaded_native_keys = set()

        n_experts = self.moe_config.n_routed_experts
        is_gated = self._is_gated_moe

        self._validate_expert_availability(hf_state_dict, n_experts, device_mesh)

        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expected_experts_per_rank = end_expert - start_expert
            rank = (
                get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
        else:
            start_expert, end_expert = 0, n_experts
            expected_experts_per_rank = n_experts
            rank = None

        state_dict: dict[str, Any] = {}
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]] = {}

        # Handle both formats:
        # - model.layers.{L}.{expert_segment}.{E}.gate_proj.weight (with model prefix)
        # - language_model.layers.{L}.{expert_segment}.{E}.gate_proj.weight (with language_model prefix)
        # - layers.{L}.{expert_segment}.{E}.gate_proj.weight (without model prefix)
        expert_pattern = re.compile(
            rf"(?P<prefix>(?:model\.)?(?:language_model\.)?)layers\.(\d+)\.{re.escape(expert_segment)}\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
        )

        inplace_loaded_keys: set = getattr(self, "_inplace_loaded_native_keys", None) or set()
        consumed_inplace_keys: set = set()

        for key in list(hf_state_dict.keys()):
            value = hf_state_dict.pop(key)
            if f".{expert_segment}." in key and key.endswith(".weight"):
                m = expert_pattern.match(key)
                if m is None:
                    state_dict[key] = value
                    continue

                prefix = m.group("prefix") or ""
                layer_num, expert_num, which = m.group(2), m.group(3), m.group(4)
                expert_num = int(expert_num)

                if which in ["gate_proj", "up_proj"]:
                    native_key = f"{prefix}layers.{layer_num}.{expert_segment}.gate_and_up_projs"
                else:  # down_proj
                    native_key = f"{prefix}layers.{layer_num}.{expert_segment}.down_projs"

                # Skip rebuild: DCP wrote through the view; model already holds the data.
                if native_key in inplace_loaded_keys:
                    consumed_inplace_keys.add(native_key)
                    del value
                    continue

                if not should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                    continue

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}

                if native_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][native_key] = {}

                if which in ["gate_proj", "up_proj"]:
                    # Non-gated models only use up_proj, skip gate_proj
                    if not is_gated and which == "gate_proj":
                        continue

                    # Store weight: gated uses dict for gate+up, non-gated stores tensor directly
                    if is_gated:
                        if expert_num not in expert_weights_by_layer[layer_num][native_key]:
                            expert_weights_by_layer[layer_num][native_key][expert_num] = {}
                        expert_weights_by_layer[layer_num][native_key][expert_num][which] = value
                    else:
                        expert_weights_by_layer[layer_num][native_key][expert_num] = value

                    # Check if all experts are complete
                    all_complete = len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank
                    if is_gated:
                        all_complete = all_complete and all(
                            isinstance(d, dict) and "gate_proj" in d and "up_proj" in d
                            for d in expert_weights_by_layer[layer_num][native_key].values()
                        )

                    if all_complete:
                        expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())
                        expert_parts = []
                        for expert_id in expert_ids:
                            expert_data = expert_weights_by_layer[layer_num][native_key][expert_id]

                            if is_gated:
                                gate_weight = expert_data["gate_proj"]
                                up_weight = expert_data["up_proj"]
                                if is_dtensor(gate_weight):
                                    gate_weight = gate_weight.to_local()
                                if is_dtensor(up_weight):
                                    up_weight = up_weight.to_local()
                                gate_t = gate_weight.transpose(0, 1)
                                up_t = up_weight.transpose(0, 1)
                                expert_parts.append((gate_t, up_t))
                            else:
                                up_weight = expert_data
                                if is_dtensor(up_weight):
                                    up_weight = up_weight.to_local()
                                expert_parts.append((up_weight.transpose(0, 1),))

                        merged = self._direct_fill_grouped_expert_tensor(expert_parts)
                        state_dict[native_key] = create_dtensor_from_local(merged, device_mesh, rank)
                        merged_on_cuda = merged.is_cuda

                        # Release the per-expert sources before processing the next projection or layer so they
                        # do not accumulate alongside the grouped output. Only trim the CUDA allocator when the
                        # merge itself used CUDA storage; the single-device fallback merges on the host.
                        del expert_parts, merged
                        del expert_weights_by_layer[layer_num][native_key]
                        if not expert_weights_by_layer[layer_num]:
                            del expert_weights_by_layer[layer_num]
                        gc.collect(0)
                        if merged_on_cuda:
                            torch.cuda.empty_cache()

                else:  # down_proj
                    expert_weights_by_layer[layer_num][native_key][expert_num] = value

                    if len(expert_weights_by_layer[layer_num][native_key]) == expected_experts_per_rank:
                        expert_ids = sorted(expert_weights_by_layer[layer_num][native_key].keys())

                        expert_parts = []
                        for expert_id in expert_ids:
                            down_weight = expert_weights_by_layer[layer_num][native_key][expert_id]  # [dim, inter_dim]

                            # Extract local tensor if input is already a DTensor
                            if is_dtensor(down_weight):
                                down_weight = down_weight.to_local()

                            down_t = down_weight.transpose(0, 1)  # [inter_dim, dim]
                            expert_parts.append((down_t,))

                        merged = self._direct_fill_grouped_expert_tensor(expert_parts)
                        state_dict[native_key] = create_dtensor_from_local(merged, device_mesh, rank)
                        merged_on_cuda = merged.is_cuda

                        # See gate/up branch above for the cleanup rationale.
                        del expert_parts, merged
                        del expert_weights_by_layer[layer_num][native_key]
                        if not expert_weights_by_layer[layer_num]:
                            del expert_weights_by_layer[layer_num]
                        gc.collect(0)
                        if merged_on_cuda:
                            torch.cuda.empty_cache()

            else:
                if not key.endswith("_scale_inv"):
                    state_dict[key] = value

        # Drop consumed entries so a subsequent from_hf (e.g. MTP merge after backbone) starts clean.
        if consumed_inplace_keys:
            self._inplace_loaded_native_keys -= consumed_inplace_keys
            # These native keys were loaded in-place via strided views into model storage (DCP
            # writes the checkpoint tensors straight through them), so they are intentionally
            # absent from the returned state_dict but are NOT missing. Record them so the
            # checkpoint loader's key-diff can exclude them and only flag genuinely unloaded params.
            self._view_loaded_native_keys = (
                getattr(self, "_view_loaded_native_keys", None) or set()
            ) | consumed_inplace_keys

        # Recombine any per-expert HF LoRA keys back to grouped format
        state_dict = self._recombine_lora_expert_keys(state_dict)

        # Convert any ParamWrapper-format LoRA keys to native grouped format
        state_dict = self._convert_paramwrapper_to_native(state_dict)

        return state_dict

    def _direct_fill_grouped_expert_tensor(self, expert_parts: list[tuple[torch.Tensor, ...]]) -> torch.Tensor:
        """Merge experts directly into one final grouped tensor.

        Args:
            expert_parts: Projection tensors for each local expert. Multiple tensors in a tuple are joined along
                the last dimension.

        Returns:
            One contiguous grouped tensor in ``self.dtype``.
        """
        if not expert_parts or not expert_parts[0]:
            raise ValueError("At least one expert projection tensor is required")

        first_expert = expert_parts[0]
        leading_shape = first_expert[0].shape[:-1]
        grouped = torch.empty(
            (len(expert_parts), *leading_shape, sum(part.shape[-1] for part in first_expert)),
            dtype=self.dtype,
            device=first_expert[0].device,
        )
        if len(first_expert) == 1:
            torch.stack([parts[0] for parts in expert_parts], dim=0, out=grouped)
        else:
            for expert_index, parts in enumerate(expert_parts):
                torch.cat(parts, dim=-1, out=grouped[expert_index])

        return grouped

    def _convert_single_merged_expert_to_hf_split_experts(
        self,
        fqn: str,
        tensor: torch.Tensor,
        *,
        prefix_override: str | None = None,
        for_checkpoint_load: bool = False,
        **kwargs,
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert one grouped expert tensor to Hugging Face's per-expert layout.

        During checkpoint loading, DCP can write into contiguous or non-contiguous views. A view into model weight
        memory updates the model directly. A view into temporary TE or MoK grouped storage is rebuilt into the model
        by ``from_hf`` after the read. Save/export conversion still creates contiguous tensors for serialization.

        Args:
            fqn: Fully qualified name of the tensor in native format.
            tensor: Native grouped-expert tensor. Gate/up tensors have shape
                ``[experts, hidden, 2 * expert_hidden]`` (or
                ``[experts, hidden, expert_hidden]`` for non-gated MoE), and
                down tensors have shape ``[experts, expert_hidden, hidden]``.
            prefix_override: When provided, replaces ``self._hf_prefix`` in
                emitted HF keys. Used to route conversions through namespaces
                outside the main backbone, e.g. ``"mtp."`` for the MTP head.
            for_checkpoint_load: Return views that DCP will completely overwrite. Save/export callers leave this
                disabled so converted tensors preserve their current values in contiguous storage.
            **kwargs: Absorbed for forward-compatibility with base callers
                that forward arbitrary state-dict kwargs (e.g. ``exclude_key_regex``).

        Returns:
            List of ``(fqn, tensor)`` tuples in Hugging Face format, or ``None``
            if this is not an expert tensor. Save/export tensors are contiguous;
            checkpoint-load tensors may be non-contiguous views into CUDA model
            storage so DCP can overwrite the native grouped weight in place.
        """
        n_experts = self.moe_config.n_routed_experts
        inter_dim = self.moe_config.moe_inter_dim
        prefix = prefix_override if prefix_override is not None else self._hf_prefix
        expert_segment = self._expert_path_segment

        # MoE expert LoRA keys do not depend on the runtime backend or its checkpoint storage aliases.
        # When v4_compatible=True, emit per-expert split keys. Otherwise, adapters that explicitly opt in emit the
        # fused ParamWrapper format; unvalidated adapters retain the legacy per-expert format.
        v4_compatible = kwargs.get("v4_compatible", False)
        for suffix in _LORA_EXPERT_SUFFIXES:
            if f".{expert_segment}.{suffix}" in fqn and fqn.endswith(f".{suffix}"):
                if not v4_compatible and self._v5_peft_target_parameters:
                    return self._convert_lora_to_paramwrapper(
                        fqn, tensor, legacy_layout=kwargs.get("legacy_paramwrapper_layout", False)
                    )
                return self._convert_lora_expert_to_hf(fqn, tensor, n_experts, inter_dim, expert_segment)

        # Quantization casts each split with ``value.to(float8_e4m3fn)``, creating storage separate from the model's
        # grouped weights. DCP must not treat that cast as model weight memory, or the model would keep its random
        # expert values. Quantized loads therefore rebuild the grouped expert tensor after the read.
        quantization = kwargs.get("quantization", False)
        reused_checkpoint_load_views = False

        # GroupedExpertsTE (backend.experts == "te") exposes both virtual grouped tensors as
        # torch.stack copies, so neither can be loaded through in-place views. GroupedExpertsMoK
        # keeps separate contiguous gate/up parameters and exposes gate_and_up_projs through a torch.cat copy. Its
        # down_projs transpose still uses the model's weight memory. Treat the two native tensors independently so
        # MoK rebuilds gate/up after the read while DCP loads down directly into the model.
        backend = getattr(self, "backend", None)
        grouped_storage_is_model_weight = self._grouped_expert_storage_is_model_weight
        gate_up_storage_is_model_weight = (
            grouped_storage_is_model_weight and getattr(backend, "dispatcher", None) != "mok"
        )
        down_storage_is_model_weight = grouped_storage_is_model_weight

        from nemo_automodel.components.moe.state_dict_utils import (
            is_dtensor,
            validate_dtensor_expert_sharding,
        )

        def checkpoint_load_destination(view: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
            """Return the checkpoint-layout view that DCP will fill."""
            nonlocal reused_checkpoint_load_views

            if for_checkpoint_load and not quantization and not is_dtensor(source):
                reused_checkpoint_load_views = True
                return view

            return view.contiguous()

        if f".{expert_segment}.gate_and_up_projs" in fqn and fqn.endswith(".gate_and_up_projs"):
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)

            if is_dtensor(tensor):
                validate_dtensor_expert_sharding(tensor, n_experts, f"gate_and_up_projs layer {layer_num}")

            splits = self._split_experts_weights(tensor, n_experts)

            inplace_ok = for_checkpoint_load and (
                (is_dtensor(tensor) or (isinstance(tensor, torch.Tensor) and tensor.is_cuda and not tensor.is_meta))
                and len(splits) > 0
                and not is_dtensor(splits[0])
                and not quantization
                and gate_up_storage_is_model_weight
            )
            if inplace_ok and for_checkpoint_load:
                self._register_inplace_loaded_key(fqn, prefix_override)

            result = []
            for i, w in enumerate(splits):
                expert_id = self._last_expert_ids[i]
                if self._is_gated_moe:
                    # Gated: split into gate_proj and up_proj
                    if inplace_ok:
                        w_gate = w[:, :inter_dim].transpose(0, 1)
                        w_up = w[:, inter_dim:].transpose(0, 1)
                    else:
                        w_gate_view = w[:, :inter_dim].transpose(0, 1)
                        w_up_view = w[:, inter_dim:].transpose(0, 1)
                        w_gate = checkpoint_load_destination(w_gate_view, w)
                        w_up = checkpoint_load_destination(w_up_view, w)
                    result.append((f"{prefix}layers.{layer_num}.{expert_segment}.{expert_id}.gate_proj.weight", w_gate))
                    result.append((f"{prefix}layers.{layer_num}.{expert_segment}.{expert_id}.up_proj.weight", w_up))
                else:
                    # Non-gated: only up_proj (tensor is [dim, inter_dim], not [dim, 2*inter_dim])
                    if inplace_ok:
                        w_up = w.transpose(0, 1)
                    else:
                        w_up = checkpoint_load_destination(w.transpose(0, 1), w)
                    result.append((f"{prefix}layers.{layer_num}.{expert_segment}.{expert_id}.up_proj.weight", w_up))
            # These split views were created during this conversion. Check only newly created Python objects before
            # releasing unused CUDA blocks; scanning every long-lived model object for each layer makes exports slow.
            del splits
            if not inplace_ok and isinstance(tensor, torch.Tensor) and not tensor.is_meta and torch.cuda.is_available():
                if tensor.is_cuda and not reused_checkpoint_load_views:
                    gc.collect(0)
                    torch.cuda.empty_cache()
            return result

        elif (
            f".{expert_segment}.down_projs" in fqn
            and fqn.endswith(".down_projs")
            and tensor.ndim == 3
            and tensor.shape[1] == inter_dim
        ):
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)

            if is_dtensor(tensor):
                validate_dtensor_expert_sharding(tensor, n_experts, f"down_projs (DeepEP) layer {layer_num}")

            splits = self._split_experts_weights(tensor, n_experts)
            inplace_ok = for_checkpoint_load and (
                (is_dtensor(tensor) or (isinstance(tensor, torch.Tensor) and tensor.is_cuda and not tensor.is_meta))
                and len(splits) > 0
                and not is_dtensor(splits[0])
                and not quantization
                and down_storage_is_model_weight
            )
            if inplace_ok and for_checkpoint_load:
                self._register_inplace_loaded_key(fqn, prefix_override)

            result = []
            for i, w in enumerate(splits):
                expert_id = self._last_expert_ids[i]
                if inplace_ok:
                    w_down = w.transpose(0, 1)
                else:
                    w_down = checkpoint_load_destination(w.transpose(0, 1), w)
                result.append(
                    (
                        f"{prefix}layers.{layer_num}.{expert_segment}.{expert_id}.down_proj.weight",
                        w_down,
                    )
                )
            # See gate_and_up branch above for the cleanup rationale.
            del splits
            if not inplace_ok and isinstance(tensor, torch.Tensor) and not tensor.is_meta and torch.cuda.is_available():
                if tensor.is_cuda and not reused_checkpoint_load_views:
                    gc.collect(0)
                    torch.cuda.empty_cache()
            return result

        return None
