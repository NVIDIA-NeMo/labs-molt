# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import glob
import json
import os
import shutil
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.components.checkpoint._backports.hf_utils import (
    FQN_TO_DTYPE_MAPPING_FILENAME,
    FQN_TO_FILE_INDEX_MAPPING_FILENAME,
)
from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.moe.state_dict_mixin import (
    _PARAMWRAPPER_LAYOUT_LEGACY,
    _PARAMWRAPPER_LAYOUT_METADATA_KEY,
    _PARAMWRAPPER_LAYOUT_MODERN,
    MoESplitExpertsStateDictMixin,
)
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

if TYPE_CHECKING:
    from peft import PeftConfig
    from torch.distributed import ProcessGroup


_MOE_LORA_PARAMETER_SUFFIXES = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")


def _is_group_rank_0(process_group: "ProcessGroup | None") -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank(group=process_group) == 0


def _group_barrier(process_group: "ProcessGroup | None") -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier(group=process_group)


def _unwrap_ddp_model(model: nn.Module) -> nn.Module:
    """Return the module that owns export metadata hidden by DDP."""
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _save_generated_hf_assets(
    model_part: nn.Module,
    metadata_reference_path: str | None,
    hf_metadata_dir: str,
    tokenizer,
    v4_compatible: bool,
    model_config=None,
    save_custom_model_code: bool = True,
) -> None:
    """Run the existing generated Hugging Face metadata export path."""
    if save_custom_model_code:
        _maybe_save_custom_model_code(metadata_reference_path, hf_metadata_dir, model_part=model_part)

    config = model_config if model_config is not None else getattr(model_part, "config", None)
    if config is not None:
        config_name = "config.json"
        if v4_compatible and _config_exists(metadata_reference_path, config_name):
            _save_original_config_json(metadata_reference_path, hf_metadata_dir, config_name)
            config_name = "config.v5.json"

        _maybe_strip_quantization_config(model_part, config=config)
        with open(os.path.join(hf_metadata_dir, config_name), "w") as f:
            if hasattr(config, "to_json_string"):
                # Use ``use_diff=False`` so the full config (not the
                # diff against class defaults) is serialized. For
                # remote-code configs registered via
                # ``register_for_auto_class`` (e.g. DeciLM /
                # Llama-Nemotron-Super-49B ``model_type='nemotron-nas'``),
                # ``to_diff_dict`` sees the class-level ``model_type``
                # attribute as equal to the class default and drops
                # it from the serialized JSON. Reloading via
                # ``AutoConfig.from_pretrained`` on the resulting
                # consolidated directory then raises
                # ``Unrecognized model ... Should have a 'model_type'
                # key``. Writing the full dict guarantees
                # ``model_type``, ``architectures`` and ``auto_map``
                # land in the saved config regardless of class defaults.
                f.write(config.to_json_string(use_diff=False))
            else:
                # Diffusers models use FrozenDict for config instead of PretrainedConfig
                json.dump(dict(config), f, indent=2, default=str)

    if getattr(model_part, "generation_config", None) is not None:
        config_name = "generation_config.json"
        if v4_compatible and _config_exists(metadata_reference_path, config_name):
            _save_original_config_json(metadata_reference_path, hf_metadata_dir, config_name)
            config_name = "generation_config.v5.json"
        with open(os.path.join(hf_metadata_dir, config_name), "w") as f:
            f.write(model_part.generation_config.to_json_string())

    if tokenizer is not None:
        tokenizer.save_pretrained(hf_metadata_dir)


class _ConsolidatedHFMetadataExporter(Protocol):
    """Model-provided writer for consolidated Hugging Face metadata."""

    def validate(self, *, tokenizer: object, original_model_path: str | None) -> None:
        """Validate exporter inputs before any distributed rank writes files."""
        ...

    def save(
        self,
        *,
        hf_metadata_dir: str,
        tokenizer: object,
        original_model_path: str | None,
    ) -> None:
        """Write model-specific metadata into the shared Hugging Face metadata directory."""
        ...


class ConsolidatedHFAddon:
    """
    Addon that writes consolidated Hugging Face metadata alongside sharded weights.

    Models can provide a custom consolidated metadata exporter; all other models
    retain the generated config, custom-code, and tokenizer path. Rank 0 writes
    the artifacts, then synchronizes ranks.
    """

    def pre_save(self, **kwargs) -> None:
        """
        Pre-save hook to emit consolidated HF artifacts.

        Expected kwargs:
            model_state (ModelState): Wrapper holding the model parts.
            hf_metadata_dir (str): Target directory for HF metadata artifacts.
            tokenizer (PreTrainedTokenizerBase | None): Optional tokenizer to save.
            fqn_to_dtype_mapping (dict[str, str] | None): Original HF safetensors dtype map.
            original_model_path (str | None): Authoritative source checkpoint snapshot.
        """
        model_state = kwargs["model_state"]
        hf_metadata_dir = kwargs["hf_metadata_dir"]
        fqn_to_file_index_mapping = kwargs["fqn_to_file_index_mapping"]
        fqn_to_dtype_mapping = kwargs.get("fqn_to_dtype_mapping", None)
        tokenizer = kwargs.get("tokenizer", None)
        model_part = model_state.model[0]  # ModelState already converts to list if needed
        original_model_path = kwargs["original_model_path"]
        process_group = kwargs.get("process_group")

        export_model = _unwrap_ddp_model(model_part)
        get_metadata_exporter = getattr(export_model, "_get_consolidated_hf_metadata_exporter", None)
        metadata_exporter: _ConsolidatedHFMetadataExporter | None = (
            get_metadata_exporter(
                tokenizer=tokenizer,
                original_model_path=original_model_path,
            )
            if callable(get_metadata_exporter)
            else None
        )
        if metadata_exporter is not None:
            metadata_exporter.validate(
                tokenizer=tokenizer,
                original_model_path=original_model_path,
            )

        # Perform save operations on rank 0
        if _is_group_rank_0(process_group):
            if metadata_exporter is not None:
                metadata_exporter.save(
                    hf_metadata_dir=hf_metadata_dir,
                    tokenizer=tokenizer,
                    original_model_path=original_model_path,
                )
            else:
                _save_generated_hf_assets(
                    model_part,
                    original_model_path,
                    hf_metadata_dir,
                    tokenizer,
                    v4_compatible=kwargs.get("v4_compatible", False),
                )

            # save the fqn_to_file_index_mapping file
            with open(os.path.join(hf_metadata_dir, FQN_TO_FILE_INDEX_MAPPING_FILENAME), "w") as f:
                json.dump(fqn_to_file_index_mapping, f, indent=2, sort_keys=True)
            if fqn_to_dtype_mapping:
                with open(os.path.join(hf_metadata_dir, FQN_TO_DTYPE_MAPPING_FILENAME), "w") as f:
                    json.dump(fqn_to_dtype_mapping, f, indent=2, sort_keys=True)
        _group_barrier(process_group)

    def post_save(self, **kwargs) -> None:
        """
        Copy the saved HF metadata to the consolidated directory.

        The reason we keep it this way is because the HF metadata needs to stay
        available for offline consolidation and re-export, otherwise any changes
        made to the config during training will be lost.

        Expected kwargs:
            consolidated_path (str): Target directory for consolidated artifacts.
            hf_metadata_dir (str): Target directory for HF metadata artifacts.
        """
        consolidated_path = kwargs["consolidated_path"]
        hf_metadata_path = kwargs["hf_metadata_path"]
        process_group = kwargs.get("process_group")
        if not consolidated_path:
            # in this case we are just saving the sharded HF safetensors
            return

        if _is_group_rank_0(process_group):
            # Copy each public metadata item into consolidated_path while keeping
            # .hf_metadata intact for the offline consolidation helper.
            for item_name in os.listdir(hf_metadata_path):
                if item_name in {FQN_TO_FILE_INDEX_MAPPING_FILENAME, FQN_TO_DTYPE_MAPPING_FILENAME}:
                    continue  # internal helper metadata, not part of the HF output
                src_path = os.path.join(hf_metadata_path, item_name)
                dst_path = os.path.join(consolidated_path, item_name)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dst_path)
        _group_barrier(process_group)


class PeftAddon:
    """
    Addon that writes PEFT-specific metadata and tokenizer alongside adapter weights.

    On rank 0, this saves `adapter_config.json`, `automodel_peft_config.json`,
    the tokenizer (if provided), and synchronizes all ranks afterward.
    """

    def pre_save(self, **kwargs) -> None:
        """
        Pre-save hook to emit PEFT artifacts.

        Expected kwargs:
            model_path (str): Directory in which to save PEFT files.
            tokenizer (PreTrainedTokenizerBase | None): Optional tokenizer to save.
            model_state (ModelState): Wrapper holding the model parts.
            peft_config (PeftConfig): PEFT configuration for serialization.
        """
        model_path = kwargs["model_path"]
        tokenizer = kwargs.get("tokenizer", None)
        model_state = kwargs["model_state"]
        peft_config = kwargs["peft_config"]
        original_model_path = kwargs["original_model_path"]
        v4_compatible = kwargs.get("v4_compatible", False)
        process_group = kwargs.get("process_group")
        hf_peft_config = _get_hf_peft_config(peft_config, model_state, v4_compatible=v4_compatible)
        automodel_peft_metadata = _get_automodel_peft_metadata(peft_config)
        adapter = getattr(_unwrap_ddp_model(model_state.model[0]), "state_dict_adapter", None)
        layout_stamp = _get_paramwrapper_layout_stamp(
            adapter, v4_compatible, kwargs.get("legacy_paramwrapper_layout", False)
        )
        if layout_stamp is not None:
            automodel_peft_metadata[_PARAMWRAPPER_LAYOUT_METADATA_KEY] = layout_stamp
        if _is_group_rank_0(process_group):
            # if the HF model has custom model code, we need to save it as part of the checkpoint
            model_part = model_state.model[0] if model_state is not None else None
            _maybe_save_custom_model_code(original_model_path, model_path, model_part=model_part)
            # save the tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(model_path)
            # save in HF format. Only keys that are needed for PEFT module loading will be saved here.
            with open(os.path.join(model_path, "adapter_config.json"), "w") as f:
                json.dump(hf_peft_config, f, indent=2, sort_keys=True)
            # save the full PEFT config for inference loading inside Automodel.
            with open(os.path.join(model_path, "automodel_peft_config.json"), "w") as f:
                json.dump(automodel_peft_metadata, f, indent=2, sort_keys=True)
        _group_barrier(process_group)

    def post_save(self, **kwargs) -> None:
        pass


def _get_hf_peft_config(peft_config: "PeftConfig", model_state: ModelState, v4_compatible: bool = False) -> dict:
    """
    Get the minimal PEFT config in the format expected by Hugging Face.

    Args:
        peft_config: Source PEFT configuration.
        model_state: Model wrapper used to infer target modules and model task.
        v4_compatible: When True, use legacy per-expert expansion format.

    Returns:
        A dictionary containing the minimal HF-compatible PEFT configuration
        (e.g., task type, LoRA rank/alpha, and discovered target modules).
    """
    MODEL_TYPE_TO_PEFT_TASK_TYPE = {"CausalLM": "CAUSAL_LM"}
    # Walk ALL local model parts, not just model[0]: under pipeline parallelism
    # ModelState.model is the list of this rank's virtual stages (each holding a
    # subset of layers). Using only model[0] would capture just the first virtual
    # stage's layers, so adapter_config.json would under-list target_modules even
    # after the cross-PP union. _extract_target_modules / _extract_target_parameters
    # union across the parts internally. The model-class properties (config,
    # architecture) are identical across parts, so read those from model[0].
    model_parts = model_state.model
    model_part = model_parts[0]
    pp_group = getattr(model_state, "pp_group", None)
    target_modules = _extract_target_modules(model_parts, v4_compatible=v4_compatible, pp_group=pp_group)
    target_parameters = _extract_target_parameters(model_parts, v4_compatible=v4_compatible, pp_group=pp_group)
    try:
        arch_name = model_part.config.architectures[0]
        # "LlamaForCausalLM".split("For") → ["Llama", "CausalLM"]
        # "LlamaBidirectionalModel".split("For") → ["LlamaBidirectionalModel"]
        parts = arch_name.split("For")
        model_task = parts[-1] if len(parts) > 1 else "FeatureExtraction"
    except (AttributeError, IndexError, TypeError):
        model_task = "N/A"

    try:
        name_or_path = model_part.config.name_or_path
    except (AttributeError, TypeError):
        name_or_path = "N/A"

    try:
        task_type = MODEL_TYPE_TO_PEFT_TASK_TYPE[model_task]
    except KeyError:
        task_type = "CAUSAL_LM"

    config = {
        "task_type": task_type,
        "peft_type": "LORA",
        "r": peft_config.dim,
        "lora_alpha": peft_config.alpha,
        "use_dora": peft_config.use_dora,
        "target_modules": target_modules,
        "bias": "none",
        "base_model_name_or_path": name_or_path,
    }
    if target_parameters:
        config["target_parameters"] = target_parameters
    return config


def _get_paramwrapper_layout_stamp(adapter, v4_compatible: bool, legacy_paramwrapper_layout: bool) -> "str | None":
    """Which fused expert LoRA layout this PEFT save uses, for the metadata stamp.

    peft flipped the ParamWrapper layout in 0.19.1 (huggingface/peft#3165);
    stamping the layout into automodel_peft_config.json lets the loader
    resolve it from metadata instead of tensor shapes. Returns None for
    adapters that don't emit fused (v5 ParamWrapper) expert LoRA at all.
    """
    if v4_compatible or not getattr(adapter, "_v5_peft_target_parameters", ()):
        return None
    return _PARAMWRAPPER_LAYOUT_LEGACY if legacy_paramwrapper_layout else _PARAMWRAPPER_LAYOUT_MODERN


def _get_automodel_peft_metadata(peft_config: "PeftConfig") -> dict:
    """
    Get the PEFT metadata in the format expected by Automodel.

    Args:
        peft_config: Source PEFT configuration.

    Returns:
        A dict containing Automodel-specific PEFT metadata fields filtered from
        the full PEFT configuration.
    """
    PEFT_KEYS = {"dim", "alpha"}
    result = {}
    for k, v in peft_config.to_dict().items():
        if k in PEFT_KEYS:
            continue
        if isinstance(v, torch.dtype):
            v = str(v)
        result[k] = v
    return result


def _has_trainable_moe_lora_parameters(
    model: "nn.Module | list[nn.Module]",
    pp_group: "torch.distributed.ProcessGroup | None" = None,
) -> bool:
    """Return whether any pipeline stage owns trainable grouped-expert LoRA parameters."""
    model_parts = model if isinstance(model, (list, tuple)) else [model]
    parameter_endings = tuple(f".{suffix}" for suffix in _MOE_LORA_PARAMETER_SUFFIXES)
    local_has_moe_lora = any(
        param.requires_grad and name.endswith(parameter_endings)
        for model_part in model_parts
        for name, param in model_part.named_parameters()
    )

    if pp_group is not None and torch.distributed.get_world_size(group=pp_group) > 1:
        world = torch.distributed.get_world_size(group=pp_group)
        gathered = [False] * world
        torch.distributed.all_gather_object(gathered, local_has_moe_lora, group=pp_group)
        return any(gathered)

    return local_has_moe_lora


def _extract_target_parameters(
    model: "nn.Module | list[nn.Module]",
    v4_compatible: bool = False,
    pp_group: "torch.distributed.ProcessGroup | None" = None,
) -> list[str]:
    """Extract ``target_parameters`` for PEFT v0.18+ ParamWrapper format.

    Returns fused expert parameter paths for adapters that explicitly opt in
    to PEFT v5 ParamWrapper export and have trainable grouped-expert LoRA
    parameters, or an empty list otherwise.

    ``model`` may be a single module or a list of local PP parts. When a PP
    group is provided, expert-LoRA presence is unioned across its ranks so all
    stages emit identical metadata.
    """
    model_part = model[0] if isinstance(model, (list, tuple)) else model
    if v4_compatible:
        return []
    adapter = getattr(model_part, "state_dict_adapter", None)
    if (
        adapter is not None
        and isinstance(adapter, MoESplitExpertsStateDictMixin)
        and adapter._v5_peft_target_parameters
        and _has_trainable_moe_lora_parameters(model, pp_group=pp_group)
    ):
        return list(adapter._v5_peft_target_parameters)
    return []


def _extract_target_modules(
    model: "nn.Module | list[nn.Module]",
    v4_compatible: bool = False,
    pp_group: "torch.distributed.ProcessGroup | None" = None,
) -> list[str]:
    """
    Extract the target modules from the model used by LoRA/PEFT layers.

    Combined-projection module names (e.g. ``qkv_proj``, ``gate_up_proj``) are
    expanded to the individual HF projection names for adapter_config.json
    compatibility with vLLM, TensorRT-LLM, and HF PEFT.

    For MoE expert LoRA, grouped 3-D adapter parameters are expanded to
    per-expert HF projection names when in v4-compatible mode (where
    per-expert ``target_modules`` are used).  In v5 mode (``v4_compatible=False``)
    the expansion is skipped because ``target_parameters`` provides the
    fused ParamWrapper paths instead.

    Strips ``_orig_mod.`` (torch.compile) and ``_checkpoint_wrapped_module.``
    (activation checkpointing) prefixes from module names.

    ``model`` may be a single module or a list of modules. Under pipeline
    parallelism the caller passes the full list of this rank's virtual stages,
    so names are unioned across all local parts (using only the first part would
    miss the other stages' layers). When ``pp_group`` is also provided, the
    discovered names are additionally unioned across the PP group so the result
    covers every rank's layers too.
    """
    model_parts = model if isinstance(model, (list, tuple)) else [model]
    # Mapping from combined projection names to their HF-compatible split names.
    _COMBINED_TO_SPLIT = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    final_target_modules = set()
    for _mp in model_parts:
        for name, _ in _mp.named_modules():
            if "lora" in name.lower():
                target_name = name.rsplit(".", 1)[0]
                if target_name.startswith("_orig_mod."):
                    target_name = target_name[len("_orig_mod.") :]
                target_name = canonical_parameter_fqn(target_name)

                # Expand combined projection names to individual HF projection names
                last_component = target_name.rsplit(".", 1)[-1]
                if last_component in _COMBINED_TO_SPLIT:
                    parent = target_name.rsplit(".", 1)[0] if "." in target_name else ""
                    for split_name in _COMBINED_TO_SPLIT[last_component]:
                        expanded = f"{parent}.{split_name}" if parent else split_name
                        final_target_modules.add(expanded)
                else:
                    final_target_modules.add(target_name)

    # MoE expert LoRA: adapter weights are nn.Parameter (not nn.Module) so
    # they don't appear in named_modules(). Expand to per-expert HF names,
    # unless the adapter explicitly opts into v5 fused target_parameters.
    # Unvalidated MoE adapters keep the legacy expansion as their default.
    adapter = getattr(model_parts[0], "state_dict_adapter", None)
    _has_split_expert_mixin = isinstance(adapter, MoESplitExpertsStateDictMixin)
    _uses_v5_target_parameters = bool(
        _has_split_expert_mixin
        and not v4_compatible
        and adapter._v5_peft_target_parameters
        and _has_trainable_moe_lora_parameters(model_parts)
    )
    if _has_split_expert_mixin and not _uses_v5_target_parameters:
        seen_expert_groups: set[tuple[str, str]] = set()
        for _mp in model_parts:
            for name, param in _mp.named_parameters():
                if not param.requires_grad:
                    continue
                for lora_suffix in _MOE_LORA_PARAMETER_SUFFIXES:
                    if name.endswith(f".{lora_suffix}"):
                        expert_path = name[: -len(f".{lora_suffix}")]
                        if expert_path.startswith("_orig_mod."):
                            expert_path = expert_path[len("_orig_mod.") :]
                        expert_path = canonical_parameter_fqn(expert_path)

                        group = "gate_and_up" if "gate_and_up" in lora_suffix else "down"
                        if (expert_path, group) in seen_expert_groups:
                            break
                        seen_expert_groups.add((expert_path, group))

                        n_experts = param.shape[0]
                        for expert_id in range(n_experts):
                            if group == "gate_and_up":
                                final_target_modules.add(f"{expert_path}.{expert_id}.gate_proj")
                                final_target_modules.add(f"{expert_path}.{expert_id}.up_proj")
                            else:
                                final_target_modules.add(f"{expert_path}.{expert_id}.down_proj")
                        break

    adapter = getattr(model_parts[0], "state_dict_adapter", None)
    # Adapters whose HF checkpoint layout renames modules (e.g. Kimi K3's
    # mlp.experts.{E}.gate_proj -> block_sparse_moe.experts.{E}.w1) convert the
    # state-dict keys on save, so adapter_config.json's target_modules must get
    # the same treatment or PEFT can't resolve them against the real model.
    # The base method defaults to the identity; legacy adapters that don't
    # subclass StateDictAdapter (e.g. kimivl) skip it, as before.
    if isinstance(adapter, StateDictAdapter):
        final_target_modules = {
            adapter.map_peft_target_module_to_hf(name, v4_compatible=v4_compatible) for name in final_target_modules
        }

    # Under pipeline parallelism each rank only holds the local stage's layers,
    # so named_modules() above yields layer-specific target names for that stage
    # only. Union the sets across the PP group so adapter_config.json lists every
    # trained layer, matching the gathered adapter weights (see
    # ModelState.state_dict). Without this the config would advertise only ~1/pp
    # of the layers.
    if pp_group is not None and torch.distributed.get_world_size(group=pp_group) > 1:
        world = torch.distributed.get_world_size(group=pp_group)
        gathered: list[list[str]] = [None] * world
        torch.distributed.all_gather_object(gathered, sorted(final_target_modules), group=pp_group)
        for part in gathered:
            if part:
                final_target_modules.update(part)

    return sorted(final_target_modules)


def _maybe_strip_quantization_config(model_part: nn.Module, config=None) -> None:
    """Remove ``quantization_config`` from the HF config when no parameters are quantized.

    Models loaded from quantized checkpoints (e.g. mxfp4 GPT-OSS) carry a
    ``quantization_config`` on their ``config`` object.  After dequantization
    all parameters are standard floating-point, but the stale config entry would
    still be written to the saved ``config.json``.  This strips it so the output
    checkpoint is a clean bf16 checkpoint, consistent with e.g.
    ``unsloth/gpt-oss-20b-BF16``.
    """
    if config is None:
        config = getattr(model_part, "config", None)
    if config is None or not hasattr(config, "quantization_config"):
        return

    _QUANTIZED_DTYPES = frozenset({torch.uint8, torch.int8})
    if any(p.dtype in _QUANTIZED_DTYPES for p in model_part.parameters()):
        return

    delattr(config, "quantization_config")


def _config_exists(original_model_path: str | None, config_name: str) -> bool:
    if original_model_path is None or not os.path.isdir(original_model_path):
        return False
    src = os.path.join(original_model_path, config_name)
    return os.path.isfile(src)


def _save_original_config_json(original_model_path: str, hf_metadata_dir: str, config_name: str) -> None:
    """Copy the original pretrained ``config.json`` with ``quantization_config`` stripped.

    This is used in v4-compatible mode so that downstream consumers (e.g. vLLM)
    that expect a transformers-v4-style config receive the file verbatim from the
    original checkpoint, minus any quantization metadata (since saved weights are
    always bf16).
    """
    src = os.path.join(original_model_path, config_name)
    if not os.path.isfile(src):
        return
    with open(src) as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    dst = os.path.join(hf_metadata_dir, config_name)
    with open(dst, "w") as f:
        json.dump(cfg, f, indent=2)


# Symbols some trust_remote_code models import at module load that newer transformers
# removed. Map symbol -> (module, fallback literal) so a consolidated checkpoint reloads
# under a deploy container whose transformers dropped them (e.g. DeciLM/Nemotron-Super
# imports NEED_SETUP_CACHE_CLASSES_MAPPING, gone since transformers 4.57).
_REMOVED_TRANSFORMERS_SYMBOLS = {
    "NEED_SETUP_CACHE_CLASSES_MAPPING": ("transformers.generation.utils", "{}"),
}


def _apply_transformers_compat_guards(py_path: str) -> None:
    """Guard imports of transformers symbols removed in newer versions.

    For each copied ``.py`` that does ``from <module> import ... <symbol> ...`` where
    ``<symbol>`` was removed upstream, insert a preamble defining the symbol on
    ``<module>`` if absent, so the subsequent import resolves. Files that don't
    reference such symbols are left byte-for-byte unchanged.
    """
    try:
        with open(py_path, encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return

    # Seed with symbols already guarded (cross-call idempotency); also dedups multiple
    # import lines of the same symbol within one file.
    guarded = {sym for sym in _REMOVED_TRANSFORMERS_SYMBOLS if f"_nemo_compat_mod.{sym}" in text}

    out: list[str] = []
    changed = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("from ") and " import " in stripped:
            for symbol, (module, fallback) in _REMOVED_TRANSFORMERS_SYMBOLS.items():
                if symbol not in guarded and symbol in stripped and f"from {module} import" in stripped:
                    indent = line[: len(line) - len(stripped)]
                    out.append(
                        f"{indent}import {module} as _nemo_compat_mod  # noqa\n"
                        f'{indent}if not hasattr(_nemo_compat_mod, "{symbol}"):\n'
                        f"{indent}    _nemo_compat_mod.{symbol} = {fallback}\n"
                        f"{indent}del _nemo_compat_mod\n"
                    )
                    guarded.add(symbol)
                    changed = True
                    break
        out.append(line)

    if changed:
        with open(py_path, "w", encoding="utf-8") as f:
            f.writelines(out)


def _maybe_save_custom_model_code(
    original_model_path: str | None,
    hf_metadata_dir: str,
    model_part: nn.Module | None = None,
) -> None:
    """
    Save the custom model code if it exists. This function preserves the original directory structure.

    When ``original_model_path`` is a local dir, copy its ``.py`` files. When it is an HF
    hub id (e.g. ``nvidia/Nemotron-Flash-1B``) and the loaded model has ``auto_map`` custom
    code, copy the ``.py`` files from the cached ``transformers_modules`` directory so the
    consolidated checkpoint carries ``modeling_*.py`` locally and reloads without needing
    ``trust_remote_code=True``.
    """
    copied: set[str] = set()

    def _copy_py_tree(src_dir: str) -> None:
        for src_path in glob.glob(os.path.join(src_dir, "**", "*.py"), recursive=True):
            if os.path.basename(src_path) == "__init__.py":
                continue
            rel_path = os.path.relpath(src_path, src_dir)
            dst_path = os.path.join(hf_metadata_dir, rel_path)
            if dst_path in copied:
                continue
            os.makedirs(os.path.dirname(dst_path) or hf_metadata_dir, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            _apply_transformers_compat_guards(dst_path)
            copied.add(dst_path)

    if original_model_path is not None:
        if os.path.isfile(original_model_path):
            dst_path = os.path.join(hf_metadata_dir, os.path.basename(original_model_path))
            os.makedirs(hf_metadata_dir, exist_ok=True)
            shutil.copy2(original_model_path, dst_path)
            copied.add(dst_path)
        elif os.path.isdir(original_model_path):
            _copy_py_tree(original_model_path)

    # Fallback: HF hub id path — resolve custom code via the model class's module file.
    # Needed for trust_remote_code models (e.g. Nemotron-Flash) so reloads from the
    # consolidated dir have has_local_code=True and don't require trust_remote_code.
    if model_part is not None and not copied:
        custom_dirs: set[str] = set()
        for cls in _iter_custom_code_classes(model_part):
            try:
                import inspect

                src_file = inspect.getfile(cls)
            except (TypeError, OSError):
                continue
            module_name = getattr(cls, "__module__", "") or ""
            if not module_name.startswith("transformers_modules."):
                continue
            custom_dirs.add(os.path.dirname(src_file))
        for src_dir in custom_dirs:
            _copy_py_tree(src_dir)


def _iter_custom_code_classes(model_part: nn.Module):
    """Yield classes referenced by ``config.auto_map`` (and the model's own class).

    Walks the full MRO so wrappers like FSDP2 (which add mixins / rename the
    top-level class) don't hide the original ``transformers_modules.*`` class.
    """
    seen: set[type] = set()
    custom_pkg = ""
    for base in type(model_part).__mro__:
        mod = getattr(base, "__module__", "") or ""
        if mod.startswith("transformers_modules."):
            if base not in seen:
                seen.add(base)
                yield base
            # Record the package path to resolve auto_map entries relative to it.
            if not custom_pkg:
                custom_pkg = ".".join(mod.split(".")[:-1])

    config = getattr(model_part, "config", None)
    auto_map = getattr(config, "auto_map", None) if config is not None else None
    if not isinstance(auto_map, dict) or not custom_pkg:
        return
    import importlib

    for value in auto_map.values():
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for ref in candidates:
            if not isinstance(ref, str) or "." not in ref:
                continue
            module_path, class_name = ref.rsplit(".", 1)
            # auto_map entries are like "modeling_nemotron_flash.NemotronFlashForCausalLM";
            # the module lives under the same transformers_modules package as the model class.
            full_module = f"{custom_pkg}.{module_path}"
            try:
                mod = importlib.import_module(full_module)
                target = getattr(mod, class_name, None)
            except Exception:
                continue
            if isinstance(target, type) and target not in seen:
                seen.add(target)
                yield target
