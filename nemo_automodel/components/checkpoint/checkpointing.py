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

import gc
import glob
import json
import logging
import os
import pickle
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.distributed.checkpoint as dcp

try:
    import multistorageclient as msc

    MSC_AVAILABLE = True
except ImportError:
    msc = None
    MSC_AVAILABLE = False

# Safe import of HF_HUB_CACHE from huggingface_hub.constants
try:
    from huggingface_hub.constants import HF_HUB_CACHE
except ImportError:
    HF_HUB_CACHE = None

from safetensors.torch import load as safetensors_load
from safetensors.torch import load_file, save_file
from safetensors.torch import save as safetensors_save
from torch import nn
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
from torch.distributed.checkpoint.storage import StorageReader, StorageWriter
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.nn.parallel import DistributedDataParallel
from torch.serialization import MAP_LOCATION, FileLike

from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from nemo_automodel.components.checkpoint._backports.filesystem import FileSystemReader, SerializationFormat
from nemo_automodel.components.checkpoint._backports.hf_storage import (
    _HuggingFaceStorageReader,
    _HuggingFaceStorageWriter,
    get_fqn_to_dtype_mapping,
    get_fqn_to_file_index_mapping,
)
from nemo_automodel.components.checkpoint.addons import ConsolidatedHFAddon, PeftAddon
from nemo_automodel.components.checkpoint.conversion_mapping import (
    get_combined_key_mapping,
    requires_tensor_merging,
)
from nemo_automodel.components.checkpoint.state_dict_adapter import CheckpointLoadPart, StateDictAdapter
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.components.checkpoint.utils import (
    ensure_tied_lm_head,
    estimate_state_dict_bytes,
    estimate_tensor_bytes,
    format_bytes,
    format_output_file_count,
    get_safetensors_index_total_size,
    get_tied_lm_head_source_names,
    get_world_size_safe,
    is_cloud_path,
    is_rank_0,
)
from nemo_automodel.shared.embedding_padding import zero_embedding_row_
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

if TYPE_CHECKING:
    from peft import PeftConfig
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase


from nemo_automodel.components.checkpoint.config import CheckpointingConfig, SaveConsolidatedMode, _is_geq_torch_2_9

_CONSOLIDATED_SIZE_WARNING_THRESHOLD_BYTES = 50 * 1024**3
_DEFAULT_HF_CONSOLIDATED_SHARD_SIZE_BYTES = 5 * 1024**3

logger = logging.getLogger(__name__)


def _format_restricted_load_error(f: FileLike) -> str:
    return (
        f"Refusing to load torch artifact from {f!r} with pickle-based torch.load. "
        "The artifact is not compatible with torch.load(weights_only=True), and loading it with "
        "weights_only=False can execute code. Migrate the artifact in a restricted environment."
    )


def load_torch_ckpt(
    f: FileLike,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: bool | None = None,
    mmap: bool | None = None,
    **pickle_load_args: Any,
) -> Any:
    """Load a torch checkpoint with restricted unpickling by default.

    Args:
        f: File path or binary file object accepted by ``torch.load``.
        map_location: Device remapping accepted by ``torch.load``.
        pickle_module: Module used to unpickle metadata and objects.
        weights_only: When ``False``, explicitly opt into unrestricted pickle loading.
            ``None`` and ``True`` use restricted loading.
        mmap: Whether to memory-map tensor storages from a file path.
        **pickle_load_args: Additional arguments forwarded to the unpickler.

    Returns:
        The deserialized checkpoint.

    Raises:
        RuntimeError: If restricted loading rejects the artifact.
    """
    if weights_only is False:
        logger.warning(
            "Loading torch artifact from %r with weights_only=False. This can execute code; "
            "only load checkpoints from a trusted source.",
            f,
        )
        # B614 is suppressed only for explicit caller opt-in to trusted legacy checkpoints.
        # Remove this branch when pickle-based checkpoint compatibility is no longer supported.
        return torch.load(  # nosec B614
            f,
            map_location=map_location,
            pickle_module=pickle_module,
            weights_only=False,
            mmap=mmap,
            **pickle_load_args,
        )

    try:
        return torch.load(
            f,
            map_location=map_location,
            pickle_module=pickle_module,
            weights_only=True,
            mmap=mmap,
            **pickle_load_args,
        )
    except pickle.UnpicklingError as err:
        raise RuntimeError(_format_restricted_load_error(f)) from err


def _unwrap_ddp_model(model: nn.Module) -> nn.Module:
    """Return the module that owns model metadata hidden by DDP."""
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _should_dequantize_base_checkpoint(model: nn.Module, requested: bool | None) -> bool:
    """Return whether this load requires checkpoint dequantization.

    ``requested`` permits dequantization unless it is explicitly ``False``.
    The conversion is needed only when the source model config declares a
    quantization method; a stale ``True`` setting must not route BF16 weights
    through the quantized full-CPU loading path.

    Args:
        model: Model whose source checkpoint metadata is being loaded.
        requested: Configured dequantization preference.

    Returns:
        Whether the source checkpoint declares quantized weights that should
        be converted while loading.
    """
    if requested is False:
        return False

    quantization_config = getattr(getattr(_unwrap_ddp_model(model), "config", None), "quantization_config", None)
    if isinstance(quantization_config, dict):
        quantization_method = quantization_config.get("quant_method")
    else:
        quantization_method = getattr(quantization_config, "quant_method", None)
    return quantization_method is not None


def _get_shared_parameter_names(model_parts: list[nn.Module]) -> list[list[str]]:
    """Find checkpoint names referring to the same live parameter.

    This primarily safeguards HF initialization for encoder-decoder models
    sharing embeddings across both stacks. ModelState already handles ordinary
    input-embedding/LM-head tying; safetensors may omit additional aliases that
    the loader must restore without accepting genuinely missing parameters.

    Args:
        model_parts: Model or pipeline parts after state-dict normalization.

    Returns:
        Groups of canonical names sharing one parameter object. Equal-valued
        independent parameters are not aliases, including after sharding.
    """
    names_by_parameter: dict[int, list[str]] = {}
    for part in model_parts:
        for name, parameter in _unwrap_ddp_model(part).named_parameters(remove_duplicate=False):
            names_by_parameter.setdefault(id(parameter), []).append(canonical_parameter_fqn(name))
    return [names for names in names_by_parameter.values() if len(names) > 1]


def _normalize_dtype_mapping_to_state_dict_keys(
    fqn_to_dtype_mapping: dict[str, str], state_dict_keys: list[str], base_model_prefix: str | None = None
) -> dict[str, str]:
    """Align original HF dtype metadata with the keys that will be exported."""
    state_dict_key_set = set(state_dict_keys)
    normalized: dict[str, str] = {}
    prefix = base_model_prefix.strip(".") if base_model_prefix else None

    for fqn, dtype_str in fqn_to_dtype_mapping.items():
        if fqn in state_dict_key_set:
            normalized[fqn] = dtype_str
            continue

        if prefix and not fqn.startswith(f"{prefix}."):
            prefixed_fqn = f"{prefix}.{fqn}"
            if prefixed_fqn in state_dict_key_set:
                normalized[prefixed_fqn] = dtype_str

    return normalized


def _apply_adapter_forced_dtype_mapping(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    fqn_to_dtype_mapping: dict[str, str],
) -> dict[str, str]:
    """Let model adapters override original HF dtype metadata for export-only keys."""
    model = _unwrap_ddp_model(model)
    adapter = getattr(model, "state_dict_adapter", None)
    forced_dtype_mapping = getattr(adapter, "forced_hf_dtype_mapping", None)
    if not callable(forced_dtype_mapping):
        return fqn_to_dtype_mapping

    forced = forced_dtype_mapping(state_dict)
    if not forced:
        return fqn_to_dtype_mapping

    normalized = dict(fqn_to_dtype_mapping)
    state_dict_key_set = set(state_dict)
    for fqn, dtype_str in forced.items():
        if fqn in state_dict_key_set:
            normalized[fqn] = dtype_str
    return normalized


def _ensure_msc_available() -> None:
    """Raise an error if MSC is not installed but a cloud path is used."""
    if not MSC_AVAILABLE:
        raise ImportError(
            "multistorageclient is required for cloud storage paths. "
            "Install it with: pip install multi-storage-client "
            "--index-url https://pypi.nvidia.com"
        )


def _adapter_path(checkpoint_dir: str) -> str:
    """Return the PEFT adapter safetensors path inside a checkpoint dir (local or ``msc://``)."""
    if is_cloud_path(checkpoint_dir):
        return checkpoint_dir.rstrip("/") + "/adapter_model.safetensors"
    return os.path.join(checkpoint_dir, "adapter_model.safetensors")


def _save_safetensors(state_dict: dict[str, torch.Tensor], path: str) -> None:
    """Write a safetensors file to a local path or an ``msc://`` cloud path.

    For cloud paths the tensors are serialized to bytes and streamed to the MSC
    file handle, since ``save_file`` only accepts a local filesystem path.
    """
    if is_cloud_path(path):
        _ensure_msc_available()
        with msc.open(path, "wb") as f:
            f.write(safetensors_save(state_dict))
    else:
        save_file(state_dict, path)


def _load_safetensors(path: str) -> dict[str, torch.Tensor]:
    """Read a safetensors file from a local path or an ``msc://`` cloud path."""
    if is_cloud_path(path):
        _ensure_msc_available()
        with msc.open(path, "rb") as f:
            return safetensors_load(f.read())
    return load_file(path)


def _maybe_msc_reader(path: str, storage_reader: StorageReader | None) -> StorageReader | None:
    """Return an MSC filesystem reader for ``msc://`` paths, else the given reader."""
    if storage_reader is None and is_cloud_path(path):
        _ensure_msc_available()
        return msc.torch.MultiStorageFileSystemReader(path)
    return storage_reader


def _maybe_msc_writer(path: str, storage_writer: StorageWriter | None) -> StorageWriter | None:
    """Return an MSC filesystem writer for ``msc://`` paths, else the given writer."""
    if storage_writer is None and is_cloud_path(path):
        _ensure_msc_available()
        return msc.torch.MultiStorageFileSystemWriter(path)
    return storage_writer


def _is_safetensors_checkpoint(path: str) -> bool:
    """Return True if path looks like a safetensors checkpoint (so we can preserve dtype); else DCP or other."""
    if os.path.isfile(path):
        return path.endswith(".safetensors")
    if not os.path.isdir(path):
        return False
    if os.path.isfile(os.path.join(path, "model.safetensors.index.json")):
        return True
    return len(glob.glob(os.path.join(path, "*.safetensors"))) > 0


def _summarize_state_dict_key_diff(
    expected_keys: set[str],
    loaded_keys: set[str],
    *,
    limit: int = 10,
) -> dict[str, Any]:
    """Summarize state-dict key mismatches for checkpoint load diagnostics."""
    missing = sorted(expected_keys - loaded_keys)
    unexpected = sorted(loaded_keys - expected_keys)
    return {
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "missing_examples": missing[:limit],
        "unexpected_examples": unexpected[:limit],
    }


def _get_checkpoint_metadata(
    path: str,
    storage_reader: StorageReader | None = None,
) -> Metadata:
    """Read checkpoint metadata, including saved tensor sizes and dtypes."""
    reader = storage_reader if storage_reader is not None else FileSystemReader(path)
    return reader.read_metadata()


if _is_geq_torch_2_9():
    from torch.distributed.checkpoint.staging import DefaultStager
    from torch.distributed.checkpoint.state_dict_saver import AsyncCheckpointerType, AsyncSaveResponse


@dataclass
class _AsyncSaveContext:
    """
    Internal container for async checkpointing state.

    One instance is maintained for the model save and one for the optimizer save
    to keep staging/upload futures and the associated process group and stager
    together in a single place.
    """

    stager: Any | None
    process_group: Any | None  # torch.distributed.ProcessGroup
    future: Any | None  # AsyncSaveResponse
    staging_active: bool = False


class _ModelSavePlanner(dcp.DefaultSavePlanner):
    """Keep model save plans in one checkpointer-scoped cache namespace."""

    def __init__(self, cache_namespace: str) -> None:
        super().__init__(enable_plan_caching=True)
        self._cached_plans_key = f"{cache_namespace}:model"


class _OptimizerSavePlanner(dcp.DefaultSavePlanner):
    """Keep optimizer save plans in one checkpointer-scoped cache namespace."""

    def __init__(self, cache_namespace: str) -> None:
        super().__init__(enable_plan_caching=True)
        self._cached_plans_key = f"{cache_namespace}:optimizer"


def _new_gloo_process_group(
    process_group: torch.distributed.ProcessGroup | None,
    timeout: timedelta | None = None,
) -> torch.distributed.ProcessGroup:
    """Create a Gloo group with the same membership as ``process_group``.

    Args:
        process_group: Source process group whose membership should be preserved.
        timeout: Optional timeout for operations executed on the new group.

    Returns:
        The newly created Gloo process group.
    """
    if process_group is None:
        if timeout is not None:
            return torch.distributed.new_group(backend="gloo", timeout=timeout)
        return torch.distributed.new_group(backend="gloo")
    ranks = torch.distributed.get_process_group_ranks(process_group)
    if timeout is not None:
        return torch.distributed.new_group(
            ranks=ranks,
            backend="gloo",
            timeout=timeout,
            use_local_synchronization=True,
        )
    return torch.distributed.new_group(
        ranks=ranks,
        backend="gloo",
        use_local_synchronization=True,
    )


def _should_write_hf_metadata(config: CheckpointingConfig) -> bool:
    """Whether to write HF metadata/artifacts for a checkpoint."""
    return config.model_save_format == SerializationFormat.SAFETENSORS and not config.is_peft


def _should_write_consolidated_safetensors(config: CheckpointingConfig, is_final_checkpoint: bool = False) -> bool:
    """Whether to output consolidated HF weights along with sharded weights."""
    if not _should_write_hf_metadata(config):
        return False
    if config.save_consolidated == SaveConsolidatedMode.EVERY:
        return True
    return config.save_consolidated == SaveConsolidatedMode.FINAL and is_final_checkpoint


def _get_original_hf_index_total_size(config: CheckpointingConfig) -> int | None:
    """Return the original HF safetensors index total size, if available."""
    try:
        reference_path = _get_hf_safetensors_reference_path(
            config.model_cache_dir,
            config.model_repo_id,
        )
    except (FileNotFoundError, ValueError):
        return None
    return get_safetensors_index_total_size(reference_path)


def _warn_if_inline_consolidation_enabled(config: CheckpointingConfig) -> None:
    """Educate users about the cost of inline HF consolidation."""
    if config.save_consolidated != SaveConsolidatedMode.EVERY or not _should_write_hf_metadata(config):
        return
    if not is_rank_0():
        return
    logger.warning(
        "checkpoint.save_consolidated=every exports HuggingFace safetensors during every checkpoint save "
        "and can leave GPUs idle during consolidation and filesystem writes. Recommended: "
        "checkpoint.save_consolidated=final, or checkpoint.save_consolidated=false and run "
        "bash <checkpoint>/model/consolidate.sh after training.",
    )


def _warn_if_large_inline_consolidation(
    config: CheckpointingConfig,
    state_dict: dict[str, torch.Tensor],
    fqn_to_index_mapping: dict[str, int] | None,
    is_final_checkpoint: bool = False,
) -> None:
    """Warn when inline consolidated export is large enough to waste GPU allocation time."""
    if not _should_write_consolidated_safetensors(config, is_final_checkpoint):
        return
    if config.save_consolidated != SaveConsolidatedMode.EVERY:
        return
    # Only rank 0 emits this warning, so bail out before estimating. Neither
    # helper below is collective -- one is a local file read, the other walks
    # the state dict without materializing tensors -- so skipping them on the
    # other ranks cannot deadlock.
    if not is_rank_0():
        return
    estimated_bytes = _get_original_hf_index_total_size(config)
    is_hf_index_estimate = estimated_bytes is not None
    if estimated_bytes is None:
        estimated_bytes = estimate_state_dict_bytes(state_dict)
    if estimated_bytes is None or estimated_bytes < _CONSOLIDATED_SIZE_WARNING_THRESHOLD_BYTES:
        return
    world_size = get_world_size_safe()
    output_file_count = len(set(fqn_to_index_mapping.values())) if fqn_to_index_mapping else 1
    output_file_summary = format_output_file_count(output_file_count)
    if is_hf_index_estimate:
        logger.warning(
            "checkpoint.save_consolidated=every is exporting ~%s of HF safetensors during checkpoint save "
            "(size from HF index; %s, world_size=%d). This can idle GPU ranks; prefer "
            "save_consolidated=final, or save_consolidated=false and run "
            "bash <checkpoint>/model/consolidate.sh after training.",
            format_bytes(estimated_bytes),
            output_file_summary,
            world_size,
        )
    else:
        logger.warning(
            "checkpoint.save_consolidated=every may be exporting a large HF checkpoint; this rank's local "
            "estimate is ~%s (full size may differ under distributed parallelism; %s, world_size=%d). Prefer "
            "save_consolidated=final, or save_consolidated=false and run "
            "bash <checkpoint>/model/consolidate.sh after training.",
            format_bytes(estimated_bytes),
            output_file_summary,
            world_size,
        )


class Checkpointer:
    """
    High-level checkpoint manager built on torch.distributed.checkpoint (DCP).

    Supports:
    - HF sharded safetensors via custom storage reader/writer
    - Optional consolidated export (config, generation config, tokenizer)
    - PEFT adapter save/load handling
    - Async save for torch >= 2.9.0

    Also provides DP- and global-rank-aware helpers for saving/loading
    auxiliary state and utilities to initialize from a base HF checkpoint.
    """

    def __init__(
        self,
        config: CheckpointingConfig,
        dp_rank: int,
        tp_rank: int,
        pp_rank: int,
        moe_mesh: DeviceMesh | None = None,
        process_group: torch.distributed.ProcessGroup | None = None,
        pp_group: Optional["torch.distributed.ProcessGroup"] = None,
    ) -> None:
        """
        Initialize the checkpointer.

        Args:
            config: Checkpointing configuration.
            dp_rank: Data parallel rank for the current process.
            tp_rank: Tensor parallel rank for the current process.
            pp_rank: Pipeline parallel rank for the current process.
            moe_mesh: Optional device mesh used for MoE when adapting state dicts.
            process_group: Process group used for distributed checkpoint collectives.
            pp_group: Optional pipeline-parallel process group. Passed to
                ``ModelState`` so PEFT adapters are gathered across PP stages at
                save time (complete adapter under ``pp_size > 1``).
        """
        self.config = config
        self.moe_mesh = moe_mesh
        self.pp_group = pp_group
        self.dp_rank = dp_rank
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.process_group = process_group
        self._planner_cache_namespace = uuid.uuid4().hex

        # async specific variables
        self._model_ctx = _AsyncSaveContext(stager=None, process_group=None, future=None, staging_active=False)
        self._optim_ctx = _AsyncSaveContext(stager=None, process_group=None, future=None, staging_active=False)
        self._consolidation_process_group = None
        if self.config.is_async:
            self._model_ctx.stager = DefaultStager()
            self._optim_ctx.stager = DefaultStager()
            self._model_ctx.process_group = _new_gloo_process_group(process_group)
            self._optim_ctx.process_group = _new_gloo_process_group(process_group)
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size(group=process_group) > 1
            and _should_write_hf_metadata(self.config)
            and self.config.save_consolidated != SaveConsolidatedMode.FALSE
            and not self.config.single_rank_consolidation
        ):
            # Every rank evaluates the same config-owned condition and must create
            # process groups in the same order.
            self._consolidation_process_group = _new_gloo_process_group(
                process_group,
                timeout=timedelta(minutes=self.config.consolidation_timeout_minutes),
            )
        self._consolidation_thread: threading.Thread | None = None
        self._consolidation_error: BaseException | None = None

        self._addons = []
        if _should_write_hf_metadata(self.config):
            self._addons.append(ConsolidatedHFAddon())
        if self.config.is_peft:
            self._addons.append(PeftAddon())
        _warn_if_inline_consolidation_enabled(self.config)

    @torch.no_grad()
    def save_model(
        self,
        model: nn.Module,
        weights_path: str,
        peft_config: Optional["PeftConfig"] = None,
        tokenizer: Optional["PreTrainedTokenizerBase"] = None,
        is_final_checkpoint: bool = False,
    ) -> None:
        """
        Save model weights to `weights_path/model`.

        Behavior:
        - PEFT: write `adapter_model.safetensors` and metadata on rank 0.
        - Safetensors + consolidation: emit HF artifacts under
          `weights_path/model/consolidated` and build a consolidated index.
        - Otherwise: use DCP with a Hugging Face or default storage writer to save shards.

        Args:
            model: Model to checkpoint.
            weights_path: Base directory for checkpoints.
            peft_config: Optional PEFT configuration when saving adapters.
            tokenizer: Optional tokenizer to save with consolidated artifacts.
            is_final_checkpoint: Whether this save is the final scheduled training checkpoint.
        """
        # Create the model directories
        model_dir = os.path.join(weights_path, "model")
        should_write_consolidated = _should_write_consolidated_safetensors(self.config, is_final_checkpoint)
        consolidated_dir = os.path.join(model_dir, "consolidated") if should_write_consolidated else None
        hf_metadata_dir = os.path.join(model_dir, ".hf_metadata") if _should_write_hf_metadata(self.config) else None
        _ensure_shared_dirs(model_dir, consolidated_dir, hf_metadata_dir, process_group=self.process_group)

        # Because this call lies outside of the dcp save call, we need to consolidate on all ranks on the main process
        # of all ranks, which lies on the critical path. Therefore, we can only do this outside of async mode.
        # In async mode the same distributed consolidation is deferred to a background thread on every rank that
        # waits for the async upload to finish, instead of the storage writer's single-rank finish() consolidation.
        # If single_rank_consolidation is set, we skip distributed consolidation and let rank 0 handle it
        # via the storage writer's finish() method - useful for Unity Catalog Volumes.
        consolidate_on_all_ranks = (
            should_write_consolidated and not self.config.is_async and not self.config.single_rank_consolidation
        )
        defer_consolidation = (
            should_write_consolidated and self.config.is_async and not self.config.single_rank_consolidation
        )
        consolidation_process_group = (
            self._consolidation_process_group if self._consolidation_process_group is not None else self.process_group
        )

        model_state = ModelState(
            model,
            self.config.is_peft,
            cpu_offload=self.config.cpu_offload,
            pp_group=self.pp_group,
        )
        state_dict = model_state.state_dict()

        # Convert to HF format if using custom model implementations.
        state_dict = _maybe_adapt_state_dict_to_hf(
            model_state.model[0],
            state_dict,
            quantization=False,
            device_mesh=self.moe_mesh,
            v4_compatible=self.config.v4_compatible,
            legacy_paramwrapper_layout=self.config.legacy_paramwrapper_layout,
        )
        if self.config.model_save_format == SerializationFormat.SAFETENSORS:
            # Module metadata (e.g. Transformer Engine state) is not part of HF weights.
            state_dict = {key: value for key, value in state_dict.items() if not key.endswith("_extra_state")}
        # MoE adapters return non-contiguous views; safetensors.save rejects those.
        _materialize_to_hf_views_for_save(state_dict)
        # Build the consolidated model.safetensors.index.json if needed
        fqn_to_file_index_mapping = self._maybe_build_consolidated_index(model_state, state_dict)
        fqn_to_dtype_mapping = self._maybe_build_original_dtype_mapping(model_state, state_dict)
        _warn_if_large_inline_consolidation(
            self.config,
            state_dict,
            fqn_to_file_index_mapping,
            is_final_checkpoint,
        )

        # Run pre-saves for addons e.g., PEFT or consolidated HF safetensors
        for addon in self._addons:
            addon.pre_save(
                model_state=model_state,
                model_path=model_dir,
                consolidated_path=consolidated_dir,
                hf_metadata_dir=hf_metadata_dir,
                tokenizer=tokenizer,
                peft_config=peft_config,
                fqn_to_file_index_mapping=fqn_to_file_index_mapping,
                fqn_to_dtype_mapping=fqn_to_dtype_mapping,
                original_model_path=self._get_original_model_path(model_state),
                v4_compatible=self.config.v4_compatible,
                legacy_paramwrapper_layout=self.config.legacy_paramwrapper_layout,
                process_group=consolidation_process_group,
            )
        self._maybe_write_offline_consolidation_script(model_dir)

        storage_writer = self._get_storage_writer(
            consolidated_dir,
            fqn_to_file_index_mapping,
            fqn_to_dtype_mapping,
            model_dir,
            consolidate_on_all_ranks or defer_consolidation,
        )
        self._model_ctx.future = self._do_save(state_dict, model_dir, storage_writer)

        for addon in self._addons:
            addon.post_save(
                consolidated_path=consolidated_dir,
                hf_metadata_path=hf_metadata_dir,
                process_group=consolidation_process_group,
            )

        if consolidate_on_all_ranks:
            consolidate_safetensors_files_on_every_rank(
                input_dir=model_dir,
                output_dir=consolidated_dir,
                fqn_to_index_mapping=fqn_to_file_index_mapping,
                num_threads=5,
                use_staging=self.config.staging_dir is not None,
                staging_dir=self.config.staging_dir,
                fqn_to_dtype_mapping=fqn_to_dtype_mapping,
                process_group=consolidation_process_group,
            )
            if is_rank_0():
                logger.info("Successfully exported consolidated HF safetensors to %s.", consolidated_dir)
        elif defer_consolidation:
            self._schedule_deferred_consolidation(
                self._model_ctx.future,
                model_dir,
                consolidated_dir,
                fqn_to_file_index_mapping,
                fqn_to_dtype_mapping,
                consolidation_process_group,
            )
        self._maybe_log_final_offline_consolidation_hint(model_dir, is_final_checkpoint)

    @torch.no_grad()
    def save_optimizer(
        self,
        optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer],
        model: nn.Module | list[nn.Module],
        weights_path: str,
        scheduler: Any | None = None,
        *,
        optimizer_part_ids: list[int] | None = None,
    ) -> None:
        """
        Save optimizer (and optional scheduler) state to `weights_path/optim` using DCP.

        Args:
            optimizer: Optimizer or per-model-part optimizers whose state will be saved.
            model: Model or pipeline model parts providing partitioning context.
            weights_path: Base directory for checkpoints.
            scheduler: Optional LR scheduler to include.
            optimizer_part_ids: Global pipeline-stage indices corresponding to
                per-model-part optimizers.
        """
        optimizer_path = os.path.join(weights_path, "optim")
        _ensure_shared_dirs(optimizer_path, process_group=self.process_group)
        optimizer_state = OptimizerState(
            model,
            optimizer,
            scheduler,
            is_peft=self.config.is_peft,
            cpu_offload=self.config.cpu_offload,
            has_expert_parallelism=self.moe_mesh is not None,
            optimizer_part_ids=optimizer_part_ids,
        )
        state_dict = optimizer_state.state_dict()
        self._optim_ctx.future = self._do_save(state_dict, optimizer_path)

    def load_optimizer(
        self,
        optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer],
        model: nn.Module | list[nn.Module],
        weights_path: str,
        scheduler: Any | None = None,
        *,
        optimizer_part_ids: list[int] | None = None,
    ) -> None:
        """
        Load optimizer (and optional scheduler) state from `weights_path/optim` using DCP.

        Args:
            optimizer: Optimizer or per-model-part optimizers to populate.
            model: Model or pipeline model parts providing partitioning context.
            weights_path: Base directory for checkpoints.
            scheduler: Optional LR scheduler to populate.
            optimizer_part_ids: Global pipeline-stage indices corresponding to
                per-model-part optimizers.
        """
        optimizer_state = OptimizerState(
            model,
            optimizer,
            scheduler,
            is_peft=self.config.is_peft,
            cpu_offload=self.config.cpu_offload,
            has_expert_parallelism=self.moe_mesh is not None,
            optimizer_part_ids=optimizer_part_ids,
        )
        state_dict = optimizer_state.state_dict()
        self._do_load(state_dict, os.path.join(weights_path, "optim"))
        optimizer_state.load_state_dict(state_dict)

    def _load_model_in_parts(
        self,
        model_state: ModelState,
        load_parts: Iterator[CheckpointLoadPart],
        model_state_dict: dict[str, torch.Tensor],
        model_path: str,
        storage_reader: StorageReader,
    ) -> None:
        """Load, convert, and release one checkpoint part at a time.

        Args:
            model_state: Wrapper for the model whose final parameter storage is populated.
            load_parts: Adapter-owned sequence of checkpoint destinations and finish callbacks.
            model_state_dict: Native model names mapped to final model tensors. Every key must be completed exactly
                once across ``load_parts``.
            model_path: Hugging Face safetensors checkpoint directory.
            storage_reader: Reader already bound to ``model_path``. Its parsed metadata is reused across parts.

        Raises:
            RuntimeError: If the checkpoint is missing a requested tensor or the parts do not cover all model tensors.
            TypeError: If the adapter yields an object other than :class:`CheckpointLoadPart`.
            ValueError: If a part is empty or repeats checkpoint or model keys.
        """
        started = time.monotonic()
        checkpoint_keys = set(_get_checkpoint_metadata(model_path, storage_reader).state_dict_metadata)
        metadata_seconds = time.monotonic() - started
        expected_model_keys = set(model_state_dict)
        requested_checkpoint_keys: set[str] = set()
        completed_model_keys: set[str] = set()
        requested_bytes = 0
        max_temporary_bytes = 0
        read_seconds = 0.0
        finish_seconds = 0.0
        part_count = 0
        process_group_kwargs = {"process_group": self.process_group} if self.process_group is not None else {}

        for part in load_parts:
            if not isinstance(part, CheckpointLoadPart):
                raise TypeError(f"Checkpoint adapter yielded {type(part).__name__}, expected CheckpointLoadPart")
            if not part.checkpoint_tensors:
                raise ValueError("Checkpoint adapter yielded a load part with no checkpoint tensors")
            if not part.model_keys:
                raise ValueError("Checkpoint adapter yielded a load part with no completed model tensors")

            part_checkpoint_keys = set(part.checkpoint_tensors)
            unknown_temporary_keys = sorted(part.temporary_checkpoint_keys - part_checkpoint_keys)
            if unknown_temporary_keys:
                raise ValueError(
                    f"Checkpoint adapter reported {len(unknown_temporary_keys)} temporary tensors absent from its "
                    f"load destinations (examples={unknown_temporary_keys[:5]})"
                )
            duplicate_checkpoint_keys = sorted(part_checkpoint_keys & requested_checkpoint_keys)
            if duplicate_checkpoint_keys:
                raise ValueError(
                    f"Checkpoint adapter requested {len(duplicate_checkpoint_keys)} tensors more than once "
                    f"(examples={duplicate_checkpoint_keys[:5]})"
                )
            missing_checkpoint_keys = sorted(part_checkpoint_keys - checkpoint_keys)
            if missing_checkpoint_keys:
                raise RuntimeError(
                    f"Checkpoint {model_path} is missing {len(missing_checkpoint_keys)} tensors required by load "
                    f"part {part_count + 1} (examples={missing_checkpoint_keys[:5]})"
                )

            duplicate_model_keys = sorted(part.model_keys & completed_model_keys)
            if duplicate_model_keys:
                raise ValueError(
                    f"Checkpoint adapter completed {len(duplicate_model_keys)} model tensors more than once "
                    f"(examples={duplicate_model_keys[:5]})"
                )
            unexpected_model_keys = sorted(part.model_keys - expected_model_keys)
            if unexpected_model_keys:
                raise ValueError(
                    f"Checkpoint adapter reported {len(unexpected_model_keys)} unknown model tensors "
                    f"(examples={unexpected_model_keys[:5]})"
                )

            part_bytes = sum(estimate_tensor_bytes(tensor) for tensor in part.checkpoint_tensors.values())
            requested_bytes += part_bytes
            temporary_bytes = sum(
                estimate_tensor_bytes(
                    tensor.to_local() if type(tensor).__name__ == "DTensor" else tensor  # noqa: PLC2801
                )
                for checkpoint_key, tensor in part.checkpoint_tensors.items()
                if checkpoint_key in part.temporary_checkpoint_keys
            )
            max_temporary_bytes = max(max_temporary_bytes, temporary_bytes)
            requested_checkpoint_keys |= part_checkpoint_keys

            read_started = time.monotonic()
            # The reader already points at model_path. Omitting checkpoint_id avoids resetting it, so safetensors
            # metadata parsed before the first part can be reused by every subsequent DCP plan.
            dcp.load(part.checkpoint_tensors, storage_reader=storage_reader, **process_group_kwargs)
            read_seconds += time.monotonic() - read_started

            finish_started = time.monotonic()
            part.finish()
            finish_seconds += time.monotonic() - finish_started
            completed_model_keys |= part.model_keys
            part_count += 1
            del part

        if part_count == 0:
            raise RuntimeError("Checkpoint adapter returned an empty load-part sequence")
        missing_model_keys = sorted(expected_model_keys - completed_model_keys)
        if missing_model_keys:
            raise RuntimeError(
                f"Checkpoint load parts omitted {len(missing_model_keys)} model tensors "
                f"(examples={missing_model_keys[:5]})"
            )

        if model_state.uses_tied_lm_head and not model_state.is_peft:
            ensure_tied_lm_head(model_state.model[0])

        total_seconds = time.monotonic() - started
        requested_gb = requested_bytes / (1 << 30)
        max_temporary_gb = max_temporary_bytes / (1 << 30)
        logger.info(
            "load_model: loaded a %.2f GB checkpoint in %d parts over %.2fs "
            "(%.2f GB/s overall | largest temporary allocation on this rank %.2f GB, metadata %.2fs, "
            "storage read %.2fs, finish %.2fs)",
            requested_gb,
            part_count,
            total_seconds,
            requested_gb / max(total_seconds, 1e-9),
            max_temporary_gb,
            metadata_seconds,
            read_seconds,
            finish_seconds,
        )

    @torch.no_grad()
    def load_model(
        self,
        model: nn.Module,
        model_path: str,
        is_init_step: bool = False,
        use_checkpoint_id: bool = True,
        key_mapping: dict[str, str] | None = None,
        allow_checkpoint_key_subset: bool = False,
    ) -> None:
        """
        Load model weights from `model_path`.

        Behavior:
        - For PEFT (non-init): rank 0 reads `adapter_model.safetensors`, then broadcasts.
        - Otherwise: use DCP with a Hugging Face or default storage reader to populate the state dict.
        - If the model exposes a `state_dict_adapter`, convert to/from HF format as needed.
        - For models requiring tensor merging (e.g., Mixtral), uses transformers' conversion mapping.

        Args:
            model: Model or parallelized model parts to load into.
            model_path: Path to the model checkpoint directory or HF snapshot.
            is_init_step: If True, treat load as initialization from a base checkpoint.
            use_checkpoint_id: Pass `checkpoint_id` to DCP if True; disable when using direct HF paths.
            key_mapping: Optional key remapping when reading from HF checkpoints.
            allow_checkpoint_key_subset: If True, keep the model's current initialization for
                parameters that are absent from the checkpoint instead of requiring an exact key match.
        """
        # Validate checkpoint directory
        if not os.path.exists(model_path) and not is_cloud_path(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist")
        model_state = ModelState(
            model,
            is_peft=self.config.is_peft,
            is_init_step=is_init_step,
            skip_task_head_prefixes=getattr(self.config, "skip_task_head_prefixes_for_base_model", None),
            cpu_offload=self.config.cpu_offload,
            has_expert_parallelism=self.moe_mesh is not None,
        )
        should_dequantize_base_checkpoint = bool(
            is_init_step
            and _should_dequantize_base_checkpoint(model_state.model[0], self.config.dequantize_base_checkpoint)
        )

        has_state_dict_adapter = hasattr(_unwrap_ddp_model(model_state.model[0]), "state_dict_adapter")

        # Every model loads through DCP (rank-local adapter conversion for custom models). A quantized
        # adapter may describe small, self-contained groups that DCP loads and converts in sequence.
        is_safetensors = _is_safetensors_checkpoint(model_path)
        state_dict_adapter = getattr(_unwrap_ddp_model(model_state.model[0]), "state_dict_adapter", None)
        uses_standard_hf_state_dict = state_dict_adapter is None

        part_loaded_model_state_dict: dict[str, torch.Tensor] | None = None
        checkpoint_load_parts: Iterator[CheckpointLoadPart] | None = None
        # Adapter-owned parts name their DCP destinations with exact checkpoint keys, so any generic Transformers
        # key_mapping is redundant for this path and must not prevent the adapter from describing bounded groups.
        if (
            is_init_step
            and is_safetensors
            and should_dequantize_base_checkpoint
            and isinstance(state_dict_adapter, StateDictAdapter)
            and len(model_state.model) == 1
            and not allow_checkpoint_key_subset
        ):
            candidate_state_dict = model_state.state_dict()
            candidate_parts = state_dict_adapter.iter_checkpoint_load_parts(
                candidate_state_dict,
                device_mesh=self.moe_mesh,
            )
            if candidate_parts is not None:
                part_loaded_model_state_dict = candidate_state_dict
                checkpoint_load_parts = candidate_parts

        if checkpoint_load_parts is not None and part_loaded_model_state_dict is not None:
            storage_reader = self._get_storage_reader(
                model_path,
                key_mapping=None,
                is_init_step=True,
                is_safetensors=True,
            )
            if storage_reader is None:
                raise RuntimeError(
                    f"No safetensors storage reader is available for part-by-part loading from {model_path}"
                )
            self._load_model_in_parts(
                model_state,
                checkpoint_load_parts,
                part_loaded_model_state_dict,
                model_path,
                storage_reader,
            )
            return

        # Standard loading path (DCP copies into model's existing tensors; dtypes follow the model)
        direct_load_started = time.monotonic()
        state_dict = model_state.state_dict()
        expected_keys = set(state_dict.keys())
        # When the model has a state_dict_adapter, it handles all key transformations
        # (to_hf/from_hf). Passing key_mapping to the storage reader would double-transform
        # keys: the storage reader renames checkpoint keys in metadata, and then to_hf also
        # renames model keys, producing a mismatch in the DCP planner.
        reader_key_mapping = None if has_state_dict_adapter else key_mapping
        storage_reader = self._get_storage_reader(
            model_path, reader_key_mapping, is_init_step=is_init_step, is_safetensors=is_safetensors
        )

        # MoE adapters return views into model storage; DCP writes safetensors
        # data straight through them and from_hf skips the rebuild.
        state_dict = _maybe_adapt_state_dict_to_hf(
            model_state.model[0],
            state_dict,
            # Training checkpoints are saved from the dequantized native model.
            # Only base-checkpoint initialization needs FP8 scale destinations.
            quantization=should_dequantize_base_checkpoint,
            device_mesh=self.moe_mesh,
            for_checkpoint_load=True,
        )
        destinations_ready = time.monotonic()
        requested_bytes = sum(
            tensor.nelement() * tensor.element_size()
            for tensor in state_dict.values()
            if isinstance(tensor, torch.Tensor)
        )

        compat_tied_lm_head_source_key: str | None = None
        lm_head_param_name = getattr(model_state, "lm_head_param_name", None)
        should_try_tied_lm_head_compat = (
            getattr(model_state, "uses_tied_lm_head", False)
            and not getattr(model_state, "has_local_tied_lm_head", False)
            and isinstance(lm_head_param_name, str)
            and lm_head_param_name in state_dict
        )
        checkpoint_metadata = {}
        checkpoint_metadata_keys: set[str] = set()
        extra_state_keys = sorted(key for key in state_dict if key.endswith("_extra_state"))
        preserved_extra_state = {}
        shared_parameter_names = (
            _get_shared_parameter_names(model_state.model) if is_init_step and uses_standard_hf_state_dict else []
        )
        if should_try_tied_lm_head_compat or allow_checkpoint_key_subset or extra_state_keys or shared_parameter_names:
            checkpoint_metadata = _get_checkpoint_metadata(model_path, storage_reader).state_dict_metadata
            checkpoint_metadata_keys = set(checkpoint_metadata)
        if extra_state_keys:
            # Serialized module metadata can grow after training (e.g. TE FP8 scaling history).
            # Allocate its saved representation; parameter and buffer destinations retain strict shape checks.
            for key in extra_state_keys:
                value = state_dict[key]
                saved = checkpoint_metadata.get(key)
                if isinstance(value, torch.Tensor) and isinstance(saved, TensorStorageMetadata):
                    if value.shape != saved.size or value.dtype != saved.properties.dtype:
                        state_dict[key] = value.new_empty(saved.size, dtype=saved.properties.dtype)
            # DCP flattens dictionary metadata into dotted child keys.
            missing_extra_state_keys = [
                key
                for key in extra_state_keys
                if key not in checkpoint_metadata_keys
                and not any(name.startswith(f"{key}.") for name in checkpoint_metadata_keys)
            ]
            if missing_extra_state_keys:
                for key in missing_extra_state_keys:
                    preserved_extra_state[key] = state_dict.pop(key)
                logging.warning(
                    "Checkpoint %s is missing %d requested module _extra_state keys. Keeping current module "
                    "extra state for those entries (examples=%s).",
                    model_path,
                    len(missing_extra_state_keys),
                    missing_extra_state_keys[:10],
                )
        if should_try_tied_lm_head_compat:
            if lm_head_param_name not in checkpoint_metadata_keys:
                for source_name in get_tied_lm_head_source_names(model_state.model[0], lm_head_param_name):
                    if source_name not in checkpoint_metadata_keys or source_name in state_dict:
                        continue
                    compat_tied_lm_head_source_key = source_name
                    state_dict[source_name] = state_dict.pop(lm_head_param_name)
                    logging.warning(
                        "Checkpoint %s is missing %s. Loading tied source %s into lm_head "
                        "(HF tied-embedding checkpoints omit lm_head, and pre-fix DCP "
                        "checkpoints with PP also omit it).",
                        model_path,
                        lm_head_param_name,
                        source_name,
                    )
                    break
                if compat_tied_lm_head_source_key is None:
                    logging.warning(
                        "Checkpoint %s is missing %s and no tied source key was found. "
                        "Keeping the current lm_head initialization for compatibility.",
                        model_path,
                        lm_head_param_name,
                    )
                    state_dict.pop(lm_head_param_name, None)

        # HF safetensors can omit any alias of a shared parameter, not just the LM head. Only omit a destination
        # when the same live parameter has a saved source; genuinely missing parameters must still fail DCP planning.
        shared_alias_sources: dict[str, str] = {}
        for names in shared_parameter_names:
            source_name = next((name for name in names if name in checkpoint_metadata_keys), None)
            if source_name is None:
                continue
            for name in names:
                if name in state_dict and name not in checkpoint_metadata_keys:
                    state_dict.setdefault(source_name, state_dict.pop(name))
                    shared_alias_sources[name] = source_name

        if allow_checkpoint_key_subset:
            missing_checkpoint_keys = sorted(key for key in state_dict if key not in checkpoint_metadata_keys)
            # A subset load tolerates a few absent keys (e.g. heads excluded at save time);
            # a checkpoint sharing NO keys with the model is a wrong or unreadable checkpoint
            # and must not become a silent no-op load.
            if missing_checkpoint_keys and len(missing_checkpoint_keys) == len(state_dict):
                raise RuntimeError(
                    f"Checkpoint {model_path} contains none of the {len(state_dict)} requested model keys "
                    f"(checkpoint has {len(checkpoint_metadata_keys)} keys, examples="
                    f"{sorted(checkpoint_metadata_keys)[:5]}). Refusing to continue with "
                    "allow_checkpoint_key_subset=True because the load would be a no-op."
                )
            # Subset loading means the checkpoint keys must be a SUBSET of the model keys
            # (the model may carry extra heads kept at init). Keys the checkpoint has but the
            # model lacks signal that the wrong architecture was built (e.g. a VLM checkpoint
            # loaded into an LLM model would drop the vision tower), so refuse rather than
            # silently export a partial model. lm_head is never a false positive here: it is
            # only popped from state_dict when it is also absent from the checkpoint.
            unmatched_checkpoint_keys = sorted(
                key for key in checkpoint_metadata_keys if key not in state_dict and not key.endswith("_extra_state")
            )
            if unmatched_checkpoint_keys:
                raise RuntimeError(
                    f"Checkpoint {model_path} has {len(unmatched_checkpoint_keys)} keys absent from the built "
                    f"model (examples={unmatched_checkpoint_keys[:5]}). The model was likely constructed with the "
                    "wrong architecture for this checkpoint (e.g. exporting a VLM checkpoint with an LLM recipe). "
                    "Rebuild the model with the recipe that produced the checkpoint."
                )
            if missing_checkpoint_keys:
                for key in missing_checkpoint_keys:
                    state_dict.pop(key, None)
                logging.warning(
                    "Checkpoint %s is missing %d requested model keys. Keeping the current model "
                    "initialization for those parameters because allow_checkpoint_key_subset=True "
                    "(examples=%s).",
                    model_path,
                    len(missing_checkpoint_keys),
                    missing_checkpoint_keys[:10],
                )

        state_dict = self._do_load(state_dict, model_path, storage_reader, is_init_step=is_init_step)
        storage_read_complete = time.monotonic()

        if compat_tied_lm_head_source_key is not None and isinstance(lm_head_param_name, str):
            state_dict[lm_head_param_name] = state_dict.pop(compat_tied_lm_head_source_key)

        for alias_name, source_name in shared_alias_sources.items():
            state_dict[alias_name] = state_dict[source_name]
        # A checkpoint may keep only an alias omitted from the original destinations (e.g. a local tied LM head).
        # It was needed for the read, but restore the original key set for installation and mismatch reporting.
        for source_name in set(shared_alias_sources.values()) - expected_keys:
            state_dict.pop(source_name)

        state_dict = _maybe_adapt_state_dict_from_hf(
            model_state.model[0],
            state_dict,
            moe_mesh=self.moe_mesh,
            paramwrapper_layout_hint=_read_paramwrapper_layout_metadata(model_path),
        )
        adapter_complete = time.monotonic()
        expected_keys_for_diff = {k for k in expected_keys if not k.endswith("_extra_state")}
        loaded_keys_for_diff = {k for k in state_dict if not k.endswith("_extra_state")}
        # MoE experts load in-place via strided views into model storage (DCP writes through
        # them), so they are absent from the returned state_dict but ARE loaded. The adapter
        # tracks them (reset + populated entirely inside from_hf); count them as loaded for the
        # diff to avoid false "missing" warnings while genuinely unloaded params are still flagged.
        _adapter = getattr(_unwrap_ddp_model(model_state.model[0]), "state_dict_adapter", None)
        loaded_keys_for_diff |= getattr(_adapter, "view_loaded_native_keys", None) or set()
        if allow_checkpoint_key_subset:
            # Keys deliberately kept at init were already warned about above; keep
            # reporting unexpected keys, which nothing else surfaces.
            expected_keys_for_diff &= loaded_keys_for_diff
        key_diff = _summarize_state_dict_key_diff(expected_keys_for_diff, loaded_keys_for_diff)
        if key_diff["missing_count"] or key_diff["unexpected_count"]:
            safe_moe_tp_requires_complete_checkpoint = any(
                getattr(part, "_nemo_moe_tp_requires_pretrained_weights", False) for part in model_state.model
            )
            if safe_moe_tp_requires_complete_checkpoint:
                raise RuntimeError(
                    "Safe custom-MoE tensor parallelism requires a complete base checkpoint; "
                    f"missing={key_diff['missing_count']} unexpected={key_diff['unexpected_count']} "
                    f"(missing examples={key_diff['missing_examples']}, "
                    f"unexpected examples={key_diff['unexpected_examples']}). Randomly initialized "
                    "replicated parameters would differ across TP ranks."
                )
            logging.warning(
                "Checkpoint key mismatch for %s: missing=%d unexpected=%d "
                "(missing examples=%s, unexpected examples=%s)",
                type(model_state.model[0]).__name__,
                key_diff["missing_count"],
                key_diff["unexpected_count"],
                key_diff["missing_examples"],
                key_diff["unexpected_examples"],
            )
        # Omitted module metadata stays local while strict installation still checks real weights.
        state_dict.update(preserved_extra_state)
        model_state.load_state_dict(
            state_dict,
            strict=not (len(model_state.model) > 1 or has_state_dict_adapter or allow_checkpoint_key_subset),
            broadcast_from_rank0=self.process_group is None and torch.distributed.is_initialized(),
        )
        install_complete = time.monotonic()
        requested_gb = requested_bytes / (1 << 30)
        direct_load_seconds = install_complete - direct_load_started
        logging.info(
            "load_model: %.2f GB loaded in %.2fs "
            "(%.2f GB/s overall | destinations %.2fs, storage read %.2fs, adapt %.2fs, install %.2fs)",
            requested_gb,
            direct_load_seconds,
            requested_gb / max(direct_load_seconds, 1e-9),
            destinations_ready - direct_load_started,
            storage_read_complete - destinations_ready,
            adapter_complete - storage_read_complete,
            install_complete - adapter_complete,
        )

        del state_dict
        gc.collect()

    @staticmethod
    def initialize_model_weights(
        model: torch.nn.Module, device: torch.device, peft_init_method: str | None = None
    ) -> None:
        """
        Materialize meta-device parameters and initialize model weights.

        Moves empty parameter shells to the target device, resets HF initialization
        flags, calls the model's weight initialization method, and initializes any
        PEFT adapters.

        Args:
            model: Model whose weights should be initialized.
            device: Target device for materialized parameters.
            peft_init_method: Initialization method for PEFT adapters (e.g. "xavier").
        """
        # Only materialize parameters that are actually on the meta device.
        # When the caller sets is_meta_device=True but the model was already
        # constructed on a real device (e.g. ContextManagers was patched to
        # a no-op), calling to_empty_parameters_only would replace valid
        # weights with uninitialized CUDA memory.
        has_meta_params = any(p.device.type == "meta" for p in model.parameters())
        if has_meta_params:
            to_empty_parameters_only(model, device=device)

        # Buffers (e.g. RoPE inv_freq) may still be on meta device.  Move them
        # to *device* with uninitialized storage so that the subsequent
        # initialize_weights() call can overwrite them with proper values
        # (HF's _init_weights recomputes non-persistent buffers from config).
        # Without this, meta buffers would survive until a later model.to_empty()
        # call, which fills them with recycled GPU memory — values that may
        # differ between successive model builds in the same process.
        for module in model.modules():
            for key in list(module._buffers):
                buf = module._buffers[key]
                if buf is not None and buf.device.type == "meta":
                    module._buffers[key] = torch.empty_like(buf, device=device)

        # HF models set _is_hf_initialized to True after initialization.
        # But because we initialize on meta device, these are erroneously set to True.
        # We need to set them to False and call initialize_weights to re-initialize the weights.

        # Some models cannot call initialize_weights when sharded with DTensors:
        # - Gemma3ForConditionalGeneration / Gemma3ForCausalLM: _init_weights() calls
        #   init.zeros_(module.weight[module.padding_idx]) on the embedding layer, which
        #   triggers DTensor redistribute and fails with sharded (TP) embeddings.
        # - NemotronHForCausalLM: the HF remote code's _init_weights uses dt_bias.copy_()
        #   which fails with DTensors. This applies to the HF-remote-code path only
        #   (detected via the model.backbone attribute), for both:
        #   - dense/v2 (no n_routed_experts) under force_hf=True, and
        #   - v3 (MoE, has n_routed_experts) under force_hf=True.
        #   With force_hf=False, both dense and v3 use our custom implementation
        #   (model.model with ModuleDict layers), which runs its own initialize_weights
        #   correctly, so the skip must NOT apply there.
        try:
            model_class = model.config.architectures[0]
        except Exception:
            model_class = ""
        is_nemotron_v2 = (
            model_class == "NemotronHForCausalLM"
            and not getattr(model.config, "n_routed_experts", None)
            and hasattr(model, "backbone")  # HF remote-code path only; custom dense runs its own init
        )
        is_nemotron_v3_hf = (
            model_class == "NemotronHForCausalLM"
            and getattr(model.config, "n_routed_experts", None)  # is Nemotron V3
            and hasattr(model, "backbone")  # is HF remote code
        )
        # HF's _init_weights calls init.zeros_(weight[padding_idx]) on nn.Embedding
        # layers. When the weight is a DTensor the integer index triggers a
        # redistribute (an all-gather of the whole embedding) and fails for TP
        # shards. Clear padding_idx for the duration of initialize_weights() so
        # that op is skipped, then zero the row on the rank-local shard. Skipping
        # the whole initialization instead left every from_config parameter as the
        # uninitialized memory to_empty() handed out (all-zero or garbage models).
        padded_embeddings = [
            mod
            for mod in model.modules()
            if isinstance(mod, nn.Embedding) and isinstance(mod.weight, DTensor) and mod.padding_idx is not None
        ]
        # Models that know the upcoming load will fully populate every tensor
        # (e.g. Devstral FP8 via its state_dict_adapter) can opt out of HF's
        # random init. Skipping also sidesteps stage-divergent DTensor
        # collectives inside `initialize_weights()` that would hang PP setups.
        owns_weight_load = bool(getattr(model, "_skip_init_weights_on_load", False))
        skip_initialize_weights = (
            model_class
            in [
                "Gemma3ForConditionalGeneration",
                "Gemma3ForCausalLM",
            ]
            or is_nemotron_v2
            or is_nemotron_v3_hf
            or owns_weight_load
        )
        if not skip_initialize_weights:
            for _, module in model.named_modules():
                if hasattr(module, "_is_hf_initialized"):
                    module._is_hf_initialized = False

            if hasattr(model, "initialize_weights"):
                # Infer the target dtype from existing (floating-point)
                # parameters so that a model constructed in fp32 (e.g. for fp32
                # master weights under FSDP2) is not silently cast back to
                # bf16 inside model.initialize_weights() -> cast_model_to_dtype().
                param_dtype = None
                for p in model.parameters():
                    if p.is_floating_point():
                        param_dtype = p.dtype
                        break
                saved_padding_idx = [(mod, mod.padding_idx) for mod in padded_embeddings]
                for mod, _ in saved_padding_idx:
                    mod.padding_idx = None
                try:
                    try:
                        if param_dtype is not None:
                            model.initialize_weights(dtype=param_dtype)
                        else:
                            model.initialize_weights()
                    except TypeError:
                        # Model's initialize_weights() does not accept a dtype kwarg.
                        model.initialize_weights()
                finally:
                    for mod, padding_idx in saved_padding_idx:
                        mod.padding_idx = padding_idx
                for mod, padding_idx in saved_padding_idx:
                    zero_embedding_row_(mod.weight, padding_idx)
            else:
                logging.warning(
                    "Warning: Model does not have initialize_weights method."
                    " Requires custom initialization to be implemented."
                )

        # Custom models constructed on meta tensors are materialized and
        # initialized here, after __init__ has already returned. Re-apply tied
        # embeddings at this point so random from-config initialization does not
        # leave lm_head.weight split from embed_tokens.weight.
        ensure_tied_lm_head(model)

        if peft_init_method is not None:
            _init_peft_adapters(model, peft_init_method)

    def load_base_model(
        self,
        model: torch.nn.Module,
        device: torch.device,
        root_dir: str,
        model_name: str | None,
        load_base_model: bool = True,
    ) -> None:
        """
        Load a model from the base Hugging Face checkpoint in parallel.

        Args:
            model: Model to load state into
            device: Device to load model onto
            root_dir: Root directory of the model cache or snapshots
            model_name: Name of the model or an absolute path to a snapshot
            load_base_model: If True, restore from HF base checkpoint
        """
        model_type = getattr(getattr(model, "config", None), "model_type", None)

        if load_base_model:
            assert model_name is not None, "model_name is required when loading base model"
            # Get combined key mapping from model attribute and model-type specific conversions
            model_key_mapping = getattr(model, "_checkpoint_conversion_mapping", None)
            key_mapping = get_combined_key_mapping(model_type, model_key_mapping)
            # NemotronH remote code (trust_remote_code) uses backbone.* params matching checkpoint keys
            # skip backbone.*→model.* conversion to avoid key mismatch
            if model_type == "nemotron_h" and hasattr(model, "backbone"):
                key_mapping = None
            self.load_model(
                model,
                model_path=model_name
                if os.path.exists(model_name)
                else _get_hf_safetensors_reference_path(root_dir, model_name),
                is_init_step=True,
                key_mapping=key_mapping,
            )

        _reinit_non_persistent_buffers(model, device, model_type=model_type)

        self.config.original_model_root_dir = root_dir
        ensure_tied_lm_head(model)

    def maybe_wait_for_staging(self) -> None:
        """
        Wait for the staging to finish if it is enabled.
        """
        if self._model_ctx.staging_active and self._model_ctx.future is not None:
            self._model_ctx.future.staging_completion.result()
            self._model_ctx.staging_active = False
        if self._optim_ctx.staging_active and self._optim_ctx.future is not None:
            self._optim_ctx.future.staging_completion.result()
            self._optim_ctx.staging_active = False

    def async_wait(self) -> None:
        """
        Wait for the async save (and any deferred consolidation) to finish.
        """
        if self._model_ctx.future is not None:
            self._model_ctx.future.upload_completion.result()
            self._model_ctx.future = None
            self._release_async_stager(self._model_ctx)
        if self._optim_ctx.future is not None:
            self._optim_ctx.future.upload_completion.result()
            self._optim_ctx.future = None
            self._release_async_stager(self._optim_ctx)
        self._join_deferred_consolidation()

    @staticmethod
    def _release_async_stager(context: _AsyncSaveContext) -> None:
        """Close a completed async stager so the next save uses a fresh instance."""
        if context.stager is not None:
            context.stager.close()
            context.stager = None

    def _schedule_deferred_consolidation(
        self,
        future: "AsyncSaveResponse | None",
        model_dir: str,
        consolidated_dir: str,
        fqn_to_index_mapping: dict[str, int] | None,
        fqn_to_dtype_mapping: dict[str, str] | None,
        process_group: "torch.distributed.ProcessGroup | None",
    ) -> None:
        """
        Consolidate HF safetensors on a background thread once the async upload completes.

        Every rank schedules the same distributed consolidation used in sync mode, so the
        shards written by the async save are merged in parallel across ranks without
        blocking the training loop. Collectives inside the consolidation run on the
        dedicated Gloo group created at init.

        Args:
            future: Async save response whose ``upload_completion`` gates the consolidation.
            model_dir: Directory holding the sharded safetensors written by the async save.
            consolidated_dir: Output directory for the consolidated HF safetensors.
            fqn_to_index_mapping: Mapping from tensor FQN to consolidated output file index.
            fqn_to_dtype_mapping: Optional mapping from tensor FQN to original HF dtype string.
            process_group: Group the consolidation collectives run on; the same one the
                synchronous path and the save addons use.
        """
        self._join_deferred_consolidation()

        def _consolidate() -> None:
            try:
                if future is not None:
                    future.upload_completion.result()
                consolidate_safetensors_files_on_every_rank(
                    input_dir=model_dir,
                    output_dir=consolidated_dir,
                    fqn_to_index_mapping=fqn_to_index_mapping,
                    num_threads=5,
                    use_staging=self.config.staging_dir is not None,
                    staging_dir=self.config.staging_dir,
                    fqn_to_dtype_mapping=fqn_to_dtype_mapping,
                    process_group=process_group,
                )
                if is_rank_0():
                    logger.info("Successfully exported consolidated HF safetensors to %s.", consolidated_dir)
            except BaseException as e:  # Re-raised on the main thread in async_wait.
                self._consolidation_error = e

        self._consolidation_thread = threading.Thread(
            target=_consolidate, name="hf-safetensors-consolidation", daemon=True
        )
        self._consolidation_thread.start()

    def _join_deferred_consolidation(self) -> None:
        """Wait for a pending background consolidation and surface its error, if any."""
        thread = self._consolidation_thread
        if thread is not None:
            thread.join()
            self._consolidation_thread = None
        if self._consolidation_error is not None:
            error = self._consolidation_error
            self._consolidation_error = None
            raise error

    def save_on_dp_ranks(self, state: Any, state_name: str, path: str) -> None:
        """Save state shared by all tensor- and pipeline-parallel peers.

        This helper is intended for data-parallel-scoped state such as a
        stateful dataloader. Only the TP0/PP0 peer writes each DP rank's state.

        Args:
            state: Stateful object to save
            state_name: Name of the stateful object
            path: Path to save stateful object
        """
        state_dir = os.path.join(path, state_name)
        _ensure_dirs(state_dir, process_group=self.process_group)
        if self.tp_rank == 0 and self.pp_rank == 0:
            torch.save(state.state_dict(), os.path.join(state_dir, f"{state_name}_dp_rank_{self.dp_rank}.pt"))

    def load_on_dp_ranks(self, state: Any, state_name: str, path: str) -> None:
        """Load state shared by all tensor- and pipeline-parallel peers.

        This helper is intended for data-parallel-scoped state such as a
        stateful dataloader. All TP/PP peers in a DP rank load the same state.

        Args:
            state: Stateful object to load
            state_name: Name of the stateful object
            path: Path to load stateful object
        """
        state_dir = os.path.join(path, state_name)
        state.load_state_dict(
            load_torch_ckpt(
                os.path.join(state_dir, f"{state_name}_dp_rank_{self.dp_rank}.pt"),
                weights_only=not self.config.allow_legacy_pickle_restore,
            )
        )

    def save_on_global_ranks(self, state: Any, state_name: str, path: str) -> None:
        """Save state that is unique to every global process rank.

        Args:
            state: Stateful object to save.
            state_name: Name of the stateful object.
            path: Path to save the stateful object.
        """
        state_dir = os.path.join(path, state_name)
        _ensure_dirs(state_dir, process_group=self.process_group)
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        torch.save(state.state_dict(), os.path.join(state_dir, f"{state_name}_global_rank_{global_rank}.pt"))

    def load_on_global_ranks(self, state: Any, state_name: str, path: str) -> None:
        """Load state unique to this global rank, with legacy DP fallback.

        Args:
            state: Stateful object to load.
            state_name: Name of the stateful object.
            path: Path containing the stateful object.
        """
        state_dir = os.path.join(path, state_name)
        global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        state_file = os.path.join(state_dir, f"{state_name}_global_rank_{global_rank}.pt")
        if not os.path.exists(state_file):
            state_file = os.path.join(state_dir, f"{state_name}_dp_rank_{self.dp_rank}.pt")
            if os.path.exists(state_file) and global_rank == 0:
                logger.warning(
                    "Loading legacy per-DP %s state from %s. Exact rank-local restoration is not guaranteed under "
                    "tensor or pipeline parallelism.",
                    state_name,
                    state_file,
                )
        state.load_state_dict(
            load_torch_ckpt(
                state_file,
                weights_only=not self.config.allow_legacy_pickle_restore,
            )
        )

    def save_distributed_state(self, state: Any, state_name: str, path: str) -> None:
        """Save a custom stateful object through DCP on all ranks.

        This is intended for auxiliary objects whose state dict contains
        sharded tensors, for example BAGEL EMA shadows under FSDP2. Rank-0
        ``torch.save`` would only persist rank 0's local shard; DCP sees the
        DTensor metadata and writes all shards correctly.
        """
        state_dir = os.path.join(path, state_name)
        _ensure_shared_dirs(state_dir, process_group=self.process_group)
        state_dict = state.state_dict()
        planner = dcp.DefaultSavePlanner(enable_plan_caching=True)
        process_group = getattr(self, "process_group", None)
        process_group_kwargs = {"process_group": process_group} if process_group is not None else {}
        dcp.save(state_dict, checkpoint_id=state_dir, planner=planner, **process_group_kwargs)

    def load_distributed_state(self, state: Any, state_name: str, path: str) -> None:
        """Load a custom stateful object previously saved with DCP."""
        state_dir = os.path.join(path, state_name)
        state_dict = state.state_dict()
        process_group = getattr(self, "process_group", None)
        process_group_kwargs = {"process_group": process_group} if process_group is not None else {}
        dcp.load(state_dict, checkpoint_id=state_dir, **process_group_kwargs)
        state.load_state_dict(state_dict)

    def close(self) -> None:
        """
        Close the checkpointer.
        """
        self.maybe_wait_for_staging()
        self.async_wait()
        if self._model_ctx.stager is not None:
            self._model_ctx.stager.close()
        if self._optim_ctx.stager is not None:
            self._optim_ctx.stager.close()
        consolidation_process_group = self._consolidation_process_group
        self._consolidation_process_group = None
        if torch.distributed.is_initialized():
            for context in (self._model_ctx, self._optim_ctx):
                if context.process_group is not None:
                    torch.distributed.destroy_process_group(context.process_group)
                    context.process_group = None
            if consolidation_process_group is not None:
                torch.distributed.destroy_process_group(consolidation_process_group)

    def finalize(self) -> None:
        """Publish any final async checkpoint and close owned resources."""
        try:
            if self.config.enabled:
                self.async_wait()
        finally:
            self.close()

    def _do_load(
        self,
        state_dict: dict[str, torch.Tensor],
        path: str,
        storage_reader: StorageReader | None = None,
        is_init_step: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Load a state dictionary from `path` using DCP or PEFT special-case logic.

        Args:
            state_dict: Mutable state dict to populate with tensors.
            path: Checkpoint directory path.
            storage_reader: Optional HF storage reader for safetensors.
            is_init_step: True if loading from a base checkpoint during initialization.

        Returns:
            The populated state dictionary (may be replaced for PEFT).
        """
        # Both model and optimizer loading is done in this function.
        is_model = _is_model_checkpoint_path(path)
        # PEFT loading is broadcasted from rank0 so it is a special case
        if self.config.is_peft and is_model and (not is_init_step):
            state_dict = _load_safetensors(_adapter_path(path))
        else:
            storage_reader = _maybe_msc_reader(path, storage_reader)
            process_group = getattr(self, "process_group", None)
            process_group_kwargs = {"process_group": process_group} if process_group is not None else {}
            dcp.load(state_dict, checkpoint_id=path, storage_reader=storage_reader, **process_group_kwargs)
        return state_dict

    def _do_save(
        self, state_dict: dict[str, torch.Tensor], path: str, storage_writer: StorageWriter | None = None
    ) -> Optional["AsyncSaveResponse"]:
        """
        Save a state dictionary to `path` using DCP or PEFT special-case logic.

        - For PEFT model saves: only rank 0 writes `adapter_model.safetensors`.
        - If async mode is enabled, schedule an asynchronous save.

        Args:
            state_dict: State dict to be serialized.
            path: Checkpoint directory path.
            storage_writer: Optional HF storage writer for safetensors sharding.

        Returns:
            Optional Future object if async mode is enabled.
        """
        # Both model and optimizer saving is done in this function.
        is_model = _is_model_checkpoint_path(path)
        # PEFT saving is done on rank0 so it is a special case
        if self.config.is_peft and is_model:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank(group=self.process_group) == 0:
                _save_safetensors(state_dict, _adapter_path(path))
            if torch.distributed.is_initialized():
                torch.distributed.barrier(group=self.process_group)
            return

        ret = None
        planner_cls = _ModelSavePlanner if is_model else _OptimizerSavePlanner
        planner = planner_cls(self._planner_cache_namespace)

        # Routes to MSC storage write for cloud paths
        storage_writer = _maybe_msc_writer(path, storage_writer)

        if self.config.is_async:
            ctx = self._model_ctx if is_model else self._optim_ctx
            if ctx.stager is None:
                ctx.stager = DefaultStager()
            ret = dcp.async_save(
                state_dict,
                checkpoint_id=path,
                storage_writer=storage_writer,
                process_group=ctx.process_group,
                async_stager=ctx.stager,
                async_checkpointer_type=AsyncCheckpointerType.PROCESS,
                planner=planner,
            )
            ctx.staging_active = True
        else:
            process_group = getattr(self, "process_group", None)
            process_group_kwargs = {"process_group": process_group} if process_group is not None else {}
            dcp.save(
                state_dict,
                checkpoint_id=path,
                storage_writer=storage_writer,
                planner=planner,
                **process_group_kwargs,
            )
        return ret

    def _maybe_write_offline_consolidation_script(self, model_dir: str) -> None:
        """Write a conservative helper script for offline HF safetensors consolidation."""
        if not _should_write_hf_metadata(self.config) or not is_rank_0():
            return

        script_path = os.path.join(model_dir, "consolidate.sh")
        output_dir = os.path.join(model_dir, "consolidated")
        contents = f"""#!/usr/bin/env bash
set -euo pipefail

# Offline HF safetensors consolidation helper.
# Defaults are conservative for login nodes and small CPU machines. For large
# checkpoints, run this on a CPU compute node and increase parallelism. Work is
# split across NPROC_PER_NODE worker processes, and each process uses NUM_THREADS
# writer threads; keep NPROC_PER_NODE * NUM_THREADS within your CPU allocation.
# Example for an 80-core CPU node:
#   NPROC_PER_NODE=16 NUM_THREADS=5 bash "$0"
# Slurm example:
#   sbatch --cpus-per-task=80 --wrap='NPROC_PER_NODE=16 NUM_THREADS=5 bash /path/to/consolidate.sh'
# Optional: set CAST_DTYPE=bf16 to request a floating-point dtype cast during export.
NPROC_PER_NODE="${{NPROC_PER_NODE:-1}}"
NUM_THREADS="${{NUM_THREADS:-5}}"
CAST_DTYPE="${{CAST_DTYPE:-}}"
PYTHON="${{PYTHON:-python3}}"
TORCHRUN="${{TORCHRUN:-torchrun}}"
CONSOLIDATION_TOOL="${{CONSOLIDATION_TOOL:-tools/offline_hf_consolidation.py}}"
CAST_DTYPE_ARGS=()
if [[ -n "${{CAST_DTYPE}}" ]]; then
  CAST_DTYPE_ARGS=(--cast-dtype "${{CAST_DTYPE}}")
fi
if [[ ! -f "${{CONSOLIDATION_TOOL}}" ]]; then
  echo "Could not find offline consolidation tool at ${{CONSOLIDATION_TOOL}}." >&2
  echo "Run from the AutoModel repo root or set CONSOLIDATION_TOOL=/path/to/tools/offline_hf_consolidation.py." >&2
  exit 1
fi

if [[ "${{NPROC_PER_NODE}}" -gt 1 ]]; then
  "${{TORCHRUN}}" --nproc-per-node="${{NPROC_PER_NODE}}" "${{CONSOLIDATION_TOOL}}" \\
    --backend gloo \\
    --num-threads "${{NUM_THREADS}}" \\
    --model-name "{self.config.model_repo_id}" \\
    --input-dir "{model_dir}" \\
    --output-dir "{output_dir}" \\
    "${{CAST_DTYPE_ARGS[@]}}"
else
  "${{PYTHON}}" "${{CONSOLIDATION_TOOL}}" \\
    --backend gloo \\
    --num-threads "${{NUM_THREADS}}" \\
    --model-name "{self.config.model_repo_id}" \\
    --input-dir "{model_dir}" \\
    --output-dir "{output_dir}" \\
    "${{CAST_DTYPE_ARGS[@]}}"
fi
"""
        with open(script_path, "w") as f:
            f.write(contents)
        os.chmod(script_path, 0o755)
        logger.debug(
            "Wrote offline HF safetensors consolidation helper script to %s.",
            script_path,
        )

    def _maybe_log_final_offline_consolidation_hint(self, model_dir: str, is_final_checkpoint: bool = False) -> None:
        """Log the final-checkpoint helper hint when consolidated export was disabled."""
        if (
            not is_final_checkpoint
            or self.config.save_consolidated != SaveConsolidatedMode.FALSE
            or not _should_write_hf_metadata(self.config)
            or not is_rank_0()
        ):
            return

        logger.info(
            "Final checkpoint was saved with checkpoint.save_consolidated=false. "
            "To export Hugging Face consolidated weights if needed, run bash %s.",
            os.path.join(model_dir, "consolidate.sh"),
        )

    def _maybe_build_consolidated_index(
        self, model_state: ModelState, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, int] | None:
        """
        Build FQN to shard index mapping for consolidated HF export.

        Uses the base checkpoint index (if present), removes non-persistent keys,
        and assigns new keys to the last shard by default.

        Args:
            model_state: Wrapper exposing the primary model part.
            state_dict: Current pipeline stage's subset of the exported state dict. Each value is a tensor of
                arbitrary shape representing its full logical tensor, including when its per-rank storage is sharded.

        Returns:
            Mapping from FQN to shard index, or None when not consolidating.
        """
        if not _should_write_hf_metadata(self.config):
            return None
        model = model_state.model[0]
        excluded_keys: set[str] = set()
        # we first need to find the FQN -> .safetensors mapping
        reference_path = _get_hf_safetensors_reference_path(
            self.config.model_cache_dir,
            self.config.model_repo_id,
        )
        if reference_path:
            # HF VLM models may contain a special checkpoint mapping attribute
            fqn_to_file_index_mapping = get_fqn_to_file_index_mapping(
                reference_path, getattr(model, "_checkpoint_conversion_mapping", None)
            )
            model_part = model_state.model[0]
            config = getattr(model_part, "config", None)
            model_type = getattr(config, "model_type", None)
            pre_shard_hf_state_dict_keys = (
                getattr(model, "_pre_shard_hf_state_dict_keys", None) or self.config.model_state_dict_keys
            )
            fallback_key_sizes = None
            if pre_shard_hf_state_dict_keys is None:
                fallback_key_sizes = _collect_global_tensor_sizes(state_dict, self.pp_group)
                pre_shard_hf_state_dict_keys = list(fallback_key_sizes)
            if model_type and requires_tensor_merging(model_type) and not hasattr(model_part, "state_dict_adapter"):
                # in this case, Transformers performed weight conversion so we will save the converted format in the checkpoint
                num_shards = max(fqn_to_file_index_mapping.values()) if fqn_to_file_index_mapping else 1
                fqn_to_file_index_mapping = _equally_divide_layers(num_shards, pre_shard_hf_state_dict_keys)
            else:
                # Decide whether the size metadata collective is needed from inputs that are
                # identical on every PP rank. Rank-local exclusions below must not control
                # collective participation, or one stage could wait forever for another.
                if set(fqn_to_file_index_mapping).isdisjoint(pre_shard_hf_state_dict_keys):
                    fallback_key_sizes = fallback_key_sizes or _collect_global_tensor_sizes(state_dict, self.pp_group)
                # some HF models like Moonlight-16B have non-persistent buffers in the base checkpoint
                # however, HF initializes buffers with persistent=False, so we need to make sure these
                # buffer keys are not saved during checkpointing
                # The `_pre_shard_hf_state_dict_keys` attribute is set during parallelization: in
                # `apply_model_infrastructure` (_transformers/infrastructure.py) for LLM/VLM models and in
                # `_apply_parallelization` (_diffusers/auto_diffusion_pipeline.py) for diffusion pipelines.
                keys_to_remove = list(set(fqn_to_file_index_mapping.keys()) - set(pre_shard_hf_state_dict_keys))
                # Only drop lm_head from the save map when it is actually an alias
                # of the embedding (e.g. single-rank tied case). PP last stages have
                # `uses_tied_lm_head=True` but must still persist their own lm_head.
                if getattr(model_state, "has_local_tied_lm_head", False):
                    keys_to_remove.append(model_state.lm_head_param_name)
                excluded_keys.update(keys_to_remove)
                for key in keys_to_remove:
                    fqn_to_file_index_mapping.pop(key, None)
                if not fqn_to_file_index_mapping:
                    fallback_keys = [
                        key
                        for key in (pre_shard_hf_state_dict_keys or list(state_dict.keys()))
                        if key not in excluded_keys
                    ]
                    fqn_to_file_index_mapping = _divide_keys_by_size(
                        fallback_keys,
                        state_dict,
                        _DEFAULT_HF_CONSOLIDATED_SHARD_SIZE_BYTES,
                        key_size_mapping=fallback_key_sizes,
                    )
                    if is_rank_0():
                        logger.info(
                            "Original HF shard mapping for %s contained no exported model keys; using size-based "
                            "consolidated shard mapping instead.",
                            self.config.model_repo_id,
                        )
        else:
            pre_shard_hf_state_dict_keys = getattr(model, "_pre_shard_hf_state_dict_keys", None)
            if pre_shard_hf_state_dict_keys is None:
                pre_shard_hf_state_dict_keys = self.config.model_state_dict_keys
            global_key_sizes = _collect_global_tensor_sizes(state_dict, self.pp_group)
            fallback_keys = pre_shard_hf_state_dict_keys or list(global_key_sizes)
            fqn_to_file_index_mapping = _divide_keys_by_size(
                fallback_keys,
                state_dict,
                _DEFAULT_HF_CONSOLIDATED_SHARD_SIZE_BYTES,
                key_size_mapping=global_key_sizes,
            )
            num_shards = max(fqn_to_file_index_mapping.values()) if fqn_to_file_index_mapping else 1
            if is_rank_0():
                logger.info(
                    "No original HF safetensors reference path found for %s; using size-based consolidated shard "
                    "mapping with target shard size %s and %d output shard(s).",
                    self.config.model_repo_id,
                    format_bytes(_DEFAULT_HF_CONSOLIDATED_SHARD_SIZE_BYTES),
                    num_shards,
                )

        # Add any missing keys from the global pre-shard HF state dict and the current state dict.
        # These will go to the same file as the last file (or file 1 for single-file models).
        # The global keys keep mappings complete under PP, while the current keys preserve
        # parameters registered after parallelization, such as test- or application-owned weights.
        # Use default of 1 only when the exported state dict itself has no mapped tensor keys.
        default_index = max(fqn_to_file_index_mapping.values()) if fqn_to_file_index_mapping else 1

        # add any additional keys that are not in the base checkpoint
        additional_keys = dict.fromkeys([*(pre_shard_hf_state_dict_keys or ()), *state_dict])
        for fqn in additional_keys:
            if fqn not in excluded_keys:
                fqn_to_file_index_mapping[fqn] = fqn_to_file_index_mapping.get(fqn, default_index)
        return fqn_to_file_index_mapping

    def _maybe_build_original_dtype_mapping(
        self, model_state: ModelState, state_dict: dict[str, torch.Tensor]
    ) -> dict[str, str] | None:
        """
        Build FQN to target safetensors dtype mapping for consolidated export.

        Original HF safetensors headers provide the baseline mapping when available.
        Model-owned adapter overrides are applied even for config-only runs so
        intrinsically fp32 tensors retain their required export dtype.
        """
        if not _should_write_hf_metadata(self.config):
            return None

        model = _unwrap_ddp_model(model_state.model[0])
        normalized_dtype_mapping: dict[str, str] = {}
        reference_path = _get_hf_safetensors_reference_path(
            self.config.model_cache_dir,
            self.config.model_repo_id,
        )
        if reference_path:
            dtype_mapping = get_fqn_to_dtype_mapping(
                reference_path, getattr(model, "_checkpoint_conversion_mapping", None)
            )
            if dtype_mapping:
                normalized_dtype_mapping = _normalize_dtype_mapping_to_state_dict_keys(
                    dtype_mapping, list(state_dict.keys()), getattr(model, "base_model_prefix", None)
                )
        normalized_dtype_mapping = _apply_adapter_forced_dtype_mapping(model, state_dict, normalized_dtype_mapping)
        return normalized_dtype_mapping or None

    def _get_storage_writer(
        self,
        consolidated_output_path: str | None,
        fqn_to_index_mapping: dict[str, int] | None,
        fqn_to_dtype_mapping: dict[str, str] | None,
        model_path: str,
        consolidation_handled_externally: bool = False,
    ) -> StorageWriter | None:
        """
        Construct a Hugging Face storage writer for sharded safetensors.

        Args:
            consolidated_output_path: Optional path for consolidated artifacts.
            fqn_to_index_mapping: Optional mapping from FQN to shard index.
            fqn_to_dtype_mapping: Optional mapping from FQN to original HF safetensors dtype string.
            model_path: Path where the model checkpoint is saved.
            consolidation_handled_externally: If True, consolidation happens outside the writer
                (inline on all ranks in sync mode, or on a background thread in async mode), so
                the writer's own finish() consolidation is disabled.

        Returns:
            Configured storage writer or None for non-safetensors.
        """
        if self.config.model_save_format == SerializationFormat.SAFETENSORS:
            return _HuggingFaceStorageWriter(
                path=model_path,
                save_sharded=True,
                consolidated_output_path=consolidated_output_path if not consolidation_handled_externally else None,
                fqn_to_index_mapping=fqn_to_index_mapping,
                fqn_to_dtype_mapping=fqn_to_dtype_mapping,
                staging_dir=self.config.staging_dir,
            )

    def _get_storage_reader(
        self,
        model_path: str,
        key_mapping: dict[str, str] | None,
        is_init_step: bool = False,
        is_safetensors: bool | None = None,
    ) -> StorageReader | None:
        """
        Construct a Hugging Face storage reader when loading safetensors or during init.

        Prefers the upstream ``torch.distributed.checkpoint.hf_storage.HuggingFaceStorageReader``
        when no ``key_mapping`` is needed, since it uses safetensors' native ``get_slice()`` for
        efficient partial reads (only the bytes for the local DTensor shard are read from disk).
        Falls back to the backported reader when ``key_mapping`` is required or when the upstream
        reader is not available.

        Args:
            model_path: Path to the model checkpoint directory or HF snapshot.
            key_mapping: Optional key remapping for conversion.
            is_init_step: If True, always produce a reader for base HF load.
            is_safetensors: Whether `model_path` holds a safetensors checkpoint; computed
                from the directory contents when not supplied.

        Returns:
            Configured storage reader, or None for the default DCP FileSystemReader.
        """
        # The configured save format does not always match what is on disk at `model_path`
        # (e.g. exporting a torch_save DCP checkpoint with a safetensors-configured
        # checkpointer). A safetensors reader on a non-safetensors directory returns EMPTY
        # metadata instead of raising, so trust the directory contents for non-init loads.
        if is_safetensors is None:
            is_safetensors = _is_safetensors_checkpoint(model_path)
        if not is_init_step and not is_safetensors:
            return None
        # The upstream HuggingFaceStorageReader delegates dtype decoding to
        # safetensors.torch._TYPES, which does not yet recognize the FP8
        # scale dtypes emitted by some quantized HF checkpoints (e.g.
        # DeepSeek V4's F8_E8M0 scales → KeyError('F8_E8M0') inside
        # read_metadata → DCP ends up with metadata=None on every rank).
        # The in-tree backport's DTYPE_MAP was extended for F8_E8M0/F8_E5M2,
        # so prefer it for base-model HF loads. Mid-training DCP loads may
        # still use the faster upstream reader.
        if key_mapping is None and not is_init_step:
            try:
                from torch.distributed.checkpoint.hf_storage import (
                    HuggingFaceStorageReader as _UpstreamHFReader,
                )

                return _UpstreamHFReader(path=model_path)
            except ImportError:
                pass
        return _HuggingFaceStorageReader(path=model_path, key_mapping=key_mapping)

    def _get_original_model_path(self, model_state: ModelState) -> str | None:
        """
        Get the path to the original model from the Hugging Face checkpoint.
        """
        if not hasattr(model_state.model[0], "name_or_path") and not hasattr(
            getattr(model_state.model[0], "config", None), "name_or_path"
        ):
            return None

        pretrained_model_name_or_path = getattr(model_state.model[0], "name_or_path", None) or getattr(
            getattr(model_state.model[0], "config", None), "name_or_path", None
        )
        # Randomly initialized HF models often have an empty `name_or_path`. In that case,
        # there is no "original" HF snapshot to reference for metadata.
        if not pretrained_model_name_or_path:
            return None

        if os.path.isdir(pretrained_model_name_or_path):
            return pretrained_model_name_or_path

        # `original_model_root_dir` exists on the config but may be None. In that case,
        # fall back to the standard HF hub cache root.
        cache_dir = getattr(self.config, "original_model_root_dir", None) or HF_HUB_CACHE
        return _get_hf_safetensors_reference_path(cache_dir, pretrained_model_name_or_path)


def _get_hf_safetensors_reference_path(cache_dir: str | Path | None, repo_id: str | None) -> str | None:
    """Return the local HF safetensors reference directory for a model.

    Prefer the snapshot directory containing `model.safetensors.index.json` for
    sharded checkpoints. If no index exists but a snapshot directory is present,
    return that directory as the single-file safetensors reference path. Return
    None when `repo_id` is None or the repo has no cached snapshot directory.

    For example, if the located file is

        /opt/models/models--meta-llama--Llama-3.2-3B/snapshots/13afe.../model.safetensors.index.json

    this function will return the directory path

        /opt/models/models--meta-llama--Llama-3.2-3B/snapshots/13afe...

    This will error if the model hasn't been downloaded or if the cache directory is incorrect.

    Args:
        cache_dir: Path to cache directory
        repo_id: Hugging Face repository ID

    Returns:
        Path to the snapshot/model directory containing safetensors weights, or
        None when no Hugging Face repo ID or cached snapshot is available.
    """
    # repo_id can be None if the model is not Hugging Face Hub yet
    if repo_id is None:
        return None

    if os.path.exists(repo_id):
        return repo_id

    cache_dir = cache_dir or HF_HUB_CACHE
    if cache_dir is None:
        # Defensive guard: HF_HUB_CACHE is expected to always be a string/path.
        raise ValueError("Hugging Face cache directory is not set (cache_dir=None).")
    repo_dir = f"models--{repo_id.replace('/', '--')}"
    snapshots_root = Path(cache_dir) / repo_dir / "snapshots"

    # Look for an index file inside any snapshot directory.
    pattern = snapshots_root / "*" / "model.safetensors.index.json"
    matches = glob.glob(str(pattern))
    if matches:
        # Return the directory path that contains the index file.
        return str(Path(matches[0]).parent)

    # Fall back: if no index file, return the first available snapshot directory (if any).
    # This is the case for single-file models.
    snapshot_dirs = [p for p in glob.glob(str(snapshots_root / "*")) if Path(p).is_dir()]
    if snapshot_dirs:
        try:
            return snapshot_dirs[0]
        except IndexError:
            raise FileNotFoundError(f"No snapshot directories found in {snapshots_root}")
    return None


def to_empty_parameters_only(
    model: nn.Module, *, device: torch.device, recurse: bool = True, dtype: torch.dtype | None = None
) -> nn.Module:
    """
    Move parameters to the specified device without copying storage, skipping buffers.

    Mirrors torch.nn.Module.to_empty but applies only to parameters, not buffers.

    Args:
        model: The module to transform
        device: Target device
        recurse: Whether to recurse into child modules

    Returns:
        The same module instance
    """
    return _apply(model, lambda t: torch.empty_like(t, device=device, dtype=dtype), recurse=recurse)


def _create_dirs(*dirs: str | None) -> None:
    """Create local directory paths and ignore cloud paths."""
    for directory in dirs:
        if directory and not is_cloud_path(directory):
            try:
                os.makedirs(directory, exist_ok=True)
            except FileExistsError:
                # virtiofs & co.: a racing rank's mkdir can surface as EEXIST while isdir() lags
                if not os.path.isdir(directory):
                    raise


def _ensure_dirs(*dirs: str | None, process_group: torch.distributed.ProcessGroup | None = None) -> None:
    """
    Create directories on all ranks and synchronize across ranks.

    Args:
        *dirs: One or more directory paths that should exist.
        process_group: Ranks that must observe the directories before continuing.
    """
    _create_dirs(*dirs)
    if torch.distributed.is_initialized():
        torch.distributed.barrier(group=process_group)


def _ensure_shared_dirs(*dirs: str | None, process_group: torch.distributed.ProcessGroup | None = None) -> None:
    """Create shared DCP directories on group rank zero and synchronize the group.

    Unlike auxiliary per-rank state, DCP checkpoint directories must be visible
    to every rank through the same filesystem.

    Args:
        *dirs: One or more shared directory paths that should exist.
        process_group: Ranks that must observe the directories before continuing.
    """
    is_dist_initialized = torch.distributed.is_initialized()
    if not is_dist_initialized or torch.distributed.get_rank(group=process_group) == 0:
        _create_dirs(*dirs)
    if is_dist_initialized:
        torch.distributed.barrier(group=process_group)


def _is_model_checkpoint_path(path: str) -> bool:
    """Return whether a checkpoint path identifies model weights."""
    return Path(path.rstrip("/")).name == "model" or os.path.isfile(_adapter_path(path))


def _init_peft_adapters(model: nn.Module, peft_init_method: str) -> None:
    """
    Initialize the PEFT adapters with the scaled weights.

    Args:
        model: Model to initialize PEFT adapters for
        peft_init_method: Method to initialize PEFT adapters e.g. "xavier". See `LinearLoRA` for more details.
    """
    for module in model.modules():
        if hasattr(module, "init_lora_weights"):
            try:
                module.init_lora_weights(peft_init_method)
            except Exception as e:
                logging.warning(f"Failed to initialize weights for PEFT adapter `{module.__class__.__name__}`: {e}")


_MODELS_REQUIRING_BUFFER_REINIT: frozenset[str] = frozenset(
    {
        "bailing_moe",
        "gemma3",
        "nemotron-nas",
    }
)


def _reinit_non_persistent_buffers(model: nn.Module, device: torch.device, model_type: str | None = None) -> None:
    """
    Recompute non-persistent buffers that are not saved in checkpoints.

    Non-persistent buffers are not saved in checkpoints, so after meta-device
    materialization they contain uninitialized CUDA memory.  When
    ``initialize_weights()`` is skipped (e.g. for Gemma3 to avoid DTensor
    issues), these buffers must be recomputed explicitly.

    Only runs for models listed in ``_MODELS_REQUIRING_BUFFER_REINIT`` to
    avoid unexpected side-effects on arbitrary HF Hub models.

    Handles four patterns:

    1. **Standard RoPE** — single ``inv_freq`` buffer with ``rope_init_fn`` and
       optional legacy ``rope_kwargs`` (e.g. Nemotron-NAS, Ling).
    2. **Per-layer-type RoPE** — ``{layer_type}_inv_freq`` buffers via
       ``compute_default_rope_parameters`` (e.g. Gemma3RotaryEmbedding).
    3. **Scaled embedding** — ``embed_scale`` buffer on ``ScaledWordEmbedding``
       modules (Gemma family), recomputed from ``scalar_embed_scale``.
    4. **Vision position IDs** — ``position_ids`` buffer on vision embedding
       modules (SigLIP), recomputed from ``num_positions``.

    Args:
        model: Model to reinitialize non-persistent buffers for.
        device: Device to create the new buffers on.
        model_type: The ``config.model_type`` string.  If not in
            ``_MODELS_REQUIRING_BUFFER_REINIT`` the function is a no-op.
    """
    if model_type not in _MODELS_REQUIRING_BUFFER_REINIT:
        return

    for name, module in model.named_modules():
        # Pattern 1: legacy standard RoPE. Ling's checkpoint code computes this
        # buffer only in __init__, so HF meta loading leaves it uninitialized.
        if hasattr(module, "rope_init_fn") and hasattr(module, "inv_freq"):
            try:
                inv_freq, _ = module.rope_init_fn(module.config, device, **getattr(module, "rope_kwargs", {}))
                module.inv_freq = inv_freq
                if hasattr(module, "original_inv_freq"):
                    module.original_inv_freq = inv_freq.clone()
                logging.debug(f"Reinitialized RoPE inv_freq for {name} on device {device}")
            except Exception as e:
                logging.warning(f"Failed to reinitialize RoPE inv_freq for {name}: {e}")

        # Pattern 2: per-layer-type RoPE (Gemma3RotaryEmbedding and similar)
        elif hasattr(module, "layer_types") and hasattr(module, "rope_type") and hasattr(module, "config"):
            rope_config = getattr(module, "config", None)
            rope_parameters = getattr(rope_config, "rope_parameters", None)
            if rope_parameters is None:
                continue
            for layer_type in getattr(module, "layer_types", []):
                inv_freq_attr = f"{layer_type}_inv_freq"
                if not hasattr(module, inv_freq_attr):
                    continue
                try:
                    rope_init_fn = getattr(module, "compute_default_rope_parameters", None)
                    if rope_init_fn is None:
                        continue
                    rope_type = module.rope_type.get(layer_type, "default")
                    if rope_type != "default":
                        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

                        rope_init_fn = ROPE_INIT_FUNCTIONS[rope_type]
                    curr_inv_freq, curr_attention_scaling = rope_init_fn(rope_config, device, layer_type=layer_type)
                    setattr(module, inv_freq_attr, curr_inv_freq)
                    orig_attr = f"{layer_type}_original_inv_freq"
                    if hasattr(module, orig_attr):
                        setattr(module, orig_attr, curr_inv_freq.clone())
                    setattr(module, f"{layer_type}_attention_scaling", curr_attention_scaling)
                    logging.debug(f"Reinitialized RoPE {inv_freq_attr} for {name} on device {device}")
                except Exception as e:
                    logging.warning(f"Failed to reinitialize RoPE {inv_freq_attr} for {name}: {e}")

        # Pattern 3: ScaledWordEmbedding embed_scale (Gemma family)
        if hasattr(module, "scalar_embed_scale") and "embed_scale" in getattr(module, "_buffers", {}):
            try:
                module.embed_scale = torch.tensor(module.scalar_embed_scale, device=device)
                logging.debug(f"Reinitialized embed_scale={module.scalar_embed_scale} for {name} on device {device}")
            except Exception as e:
                logging.warning(f"Failed to reinitialize embed_scale for {name}: {e}")

        # Pattern 4: Vision embedding position_ids (SigLIP and similar)
        if hasattr(module, "num_positions") and "position_ids" in getattr(module, "_buffers", {}):
            try:
                module.position_ids = torch.arange(module.num_positions, device=device).expand((1, -1))
                logging.debug(f"Reinitialized position_ids (num_positions={module.num_positions}) for {name}")
            except Exception as e:
                logging.warning(f"Failed to reinitialize position_ids for {name}: {e}")


def _apply(module, fn, recurse=True) -> nn.Module:
    """
    Apply a transformation function to parameters (and gradients) only.

    Mirrors `nn.Module.to_empty` for parameters while skipping buffers. Respects
    future flags controlling in-place vs swap behavior and safely handles
    wrapper subclasses.

    Args:
        module: Module whose parameters are to be transformed.
        fn: Callable applied to each parameter (and its gradient).
        recurse: Whether to recurse into child modules.

    Returns:
        The same module instance after transformation.
    """
    from torch.utils._python_dispatch import is_traceable_wrapper_subclass

    if recurse:
        for child in module.children():
            _apply(child, fn, recurse=recurse)

    def compute_should_use_set_data(tensor, tensor_applied):
        if torch._has_compatible_shallow_copy_type(tensor, tensor_applied):
            # If the new tensor has compatible tensor type as the existing tensor,
            # the current behavior is to change the tensor in-place using `.data =`,
            # and the future behavior is to overwrite the existing tensor. However,
            # changing the current behavior is a BC-breaking change, and we want it
            # to happen in future releases. So for now we introduce the
            # `torch.__future__.get_overwrite_module_params_on_conversion()`
            # global flag to let the user control whether they want the future
            # behavior of overwriting the existing tensor or not.
            return not torch.__future__.get_overwrite_module_params_on_conversion()
        else:
            return False

    should_use_swap_tensors = torch.__future__.get_swap_module_params_on_conversion()
    for key, param in module._parameters.items():
        if param is None:
            continue
        # Tensors stored in modules are graph leaves, and we don't want to
        # track autograd history of `param_applied`, so we have to use
        # `with torch.no_grad():`
        with torch.no_grad():
            param_applied = fn(param)
        p_should_use_set_data = compute_should_use_set_data(param, param_applied)

        # subclasses may have multiple child tensors so we need to use swap_tensors
        p_should_use_swap_tensors = should_use_swap_tensors or is_traceable_wrapper_subclass(param_applied)

        param_grad = param.grad
        if p_should_use_swap_tensors:
            try:
                if param_grad is not None:
                    # Accessing param.grad makes its at::Tensor's use_count 2, which will prevent swapping.
                    # Decrement use count of the gradient by setting to None
                    param.grad = None
                param_applied = torch.nn.Parameter(param_applied, requires_grad=param.requires_grad)
                torch.utils.swap_tensors(param, param_applied)
            except Exception as e:
                if param_grad is not None:
                    param.grad = param_grad
                raise RuntimeError(f"_apply(): Couldn't swap {module._get_name()}.{key}") from e
            out_param = param
        elif p_should_use_set_data:
            param.data = param_applied
            out_param = param
        else:
            assert isinstance(param, torch.nn.Parameter)
            assert param.is_leaf
            out_param = torch.nn.Parameter(param_applied, param.requires_grad)
            module._parameters[key] = out_param

        if param_grad is not None:
            with torch.no_grad():
                grad_applied = fn(param_grad)
            g_should_use_set_data = compute_should_use_set_data(param_grad, grad_applied)
            if p_should_use_swap_tensors:
                grad_applied.requires_grad_(param_grad.requires_grad)
                try:
                    torch.utils.swap_tensors(param_grad, grad_applied)
                except Exception as e:
                    raise RuntimeError(f"_apply(): Couldn't swap {module._get_name()}.{key}.grad") from e
                out_param.grad = param_grad
            elif g_should_use_set_data:
                assert out_param.grad is not None
                out_param.grad.data = grad_applied
            else:
                assert param_grad.is_leaf
                out_param.grad = grad_applied.requires_grad_(param_grad.requires_grad)

    return module


def _apply_key_mapping(
    state_dict: dict[str, torch.Tensor],
    key_mapping: dict[str, str],
) -> dict[str, torch.Tensor]:
    """
    Rename state-dict keys using regex-based ``key_mapping``.

    This mirrors the renaming logic used by the DCP / HuggingFace storage
    reader but operates directly on an in-memory state dict.  It is needed
    when loading safetensors checkpoints outside of DCP so that HF checkpoint
    keys (e.g. ``language_model.model.X``) are translated to the model's
    parameter FQNs (e.g. ``model.language_model.X``).

    Args:
        state_dict: Original state dict whose keys may need renaming.
        key_mapping: ``{regex_pattern: replacement}`` pairs applied in order.

    Returns:
        A new dict with renamed keys.
    """
    from nemo_automodel.components.checkpoint._backports.hf_storage import (
        _get_key_renaming_mapping,
    )

    return {_get_key_renaming_mapping(k, key_mapping): v for k, v in state_dict.items()}


def _maybe_adapt_state_dict_to_hf(
    model_part: nn.Module, state_dict: dict[str, torch.Tensor], quantization: bool = False, **kwargs
) -> dict[str, torch.Tensor]:
    """
    Custom models use state dict adapters to convert the state dict to the Hugging Face format.
    """
    adapter = getattr(_unwrap_ddp_model(model_part), "state_dict_adapter", None)
    if adapter:
        return adapter.to_hf(state_dict, exclude_key_regex=r".*_extra_state.*", quantization=quantization, **kwargs)
    return state_dict


def _materialize_to_hf_views_for_save(state_dict: dict[str, torch.Tensor]) -> None:
    """Replace non-contiguous tensor values in ``state_dict`` with contiguous copies in place.

    MoE adapters return non-contiguous strided views into the model's grouped
    expert storage for the optimized load path; ``safetensors.torch.save``
    (which the DCP HF storage writer calls) rejects non-contiguous tensors,
    so we materialize one tensor at a time here with ``empty_cache`` between
    iterations. Per-tensor transient is bounded to a single expert weight
    instead of allocating the full grouped set up front.
    """
    if not state_dict:
        return
    cuda_available = torch.cuda.is_available()
    for key, value in list(state_dict.items()):
        if isinstance(value, torch.Tensor) and not value.is_contiguous():
            state_dict[key] = value.contiguous()
            del value
            if cuda_available:
                torch.cuda.empty_cache()


def _equally_divide_layers(num_shards: int, keys: list[str]) -> dict[str, int]:
    """
    Equally divide the state dict keys into num_shards shards.
    """
    if num_shards <= 0:
        raise ValueError(f"num_shards must be > 0, got {num_shards}")

    num_layers = len(keys)
    if num_layers == 0:
        return {}

    layers_per_shard, remainder = divmod(num_layers, num_shards)
    fqn_to_index_mapping: dict[str, int] = {}
    start = 0
    for shard_index in range(1, num_shards + 1):
        extra = 1 if shard_index <= remainder else 0
        end = start + layers_per_shard + extra
        for key in keys[start:end]:
            fqn_to_index_mapping[key] = shard_index
        start = end
    return fqn_to_index_mapping


def _divide_keys_by_size(
    keys: list[str],
    state_dict: dict[str, torch.Tensor],
    target_shard_bytes: int,
    key_size_mapping: dict[str, int] | None = None,
) -> dict[str, int]:
    """Assign keys to deterministic size-based shards.

    Args:
        keys: Ordered tensor names to assign.
        state_dict: Mapping of tensor names to tensors of arbitrary shape. Each value represents its full logical
            tensor, including when its per-rank storage is sharded, and is read only for its logical byte size when
            ``key_size_mapping`` is not provided.
        target_shard_bytes: Positive target size for each shard in bytes.
        key_size_mapping: Optional mapping of tensor names to logical byte sizes, including tensors not present in
            the rank-local ``state_dict``.

    Returns:
        Mapping from every input key to a positive, one-based shard index.

    Raises:
        ValueError: If ``target_shard_bytes`` is not positive.
    """
    if target_shard_bytes <= 0:
        raise ValueError(f"target_shard_bytes must be > 0, got {target_shard_bytes}")

    fqn_to_index_mapping: dict[str, int] = {}
    current_shard = 1
    current_shard_bytes = 0

    for key in keys:
        tensor = state_dict.get(key)
        tensor_bytes = (
            key_size_mapping.get(key, 0)
            if key_size_mapping is not None
            else estimate_tensor_bytes(tensor)
            if tensor is not None
            else 0
        )
        if current_shard_bytes > 0 and current_shard_bytes + tensor_bytes > target_shard_bytes:
            current_shard += 1
            current_shard_bytes = 0

        fqn_to_index_mapping[key] = current_shard
        current_shard_bytes += tensor_bytes

    return fqn_to_index_mapping


def _collect_global_tensor_sizes(
    state_dict: dict[str, torch.Tensor],
    process_group: torch.distributed.ProcessGroup | None,
) -> dict[str, int]:
    """Collect logical tensor sizes across pipeline stages without moving tensor data.

    Args:
        state_dict: Current pipeline stage's key subset, mapping names to tensors of arbitrary shape. Each value must
            report the full logical element count through ``numel()``; for example, a DTensor reports its global
            logical size rather than its per-rank TP/FSDP shard size. Tensors remain on their existing devices and
            their placements are not changed.
        process_group: Pipeline-parallel process group whose ranks collectively own the logical state dict, or
            ``None`` for a local-only size mapping.

    Returns:
        Mapping from tensor names to logical byte sizes, merged across all ranks in ``process_group``.

    Raises:
        RuntimeError: If a participating pipeline rank does not provide its size mapping.
        ValueError: If pipeline ranks report different logical sizes for the same tensor name.
    """
    local_sizes = {key: estimate_tensor_bytes(tensor) for key, tensor in state_dict.items()}
    if (
        process_group is None
        or not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_world_size(group=process_group) == 1
    ):
        return local_sizes

    world_size = torch.distributed.get_world_size(group=process_group)
    gathered_sizes: list[dict[str, int] | None] = [None] * world_size
    torch.distributed.all_gather_object(gathered_sizes, local_sizes, group=process_group)

    global_sizes: dict[str, int] = {}
    for rank, rank_sizes in enumerate(gathered_sizes):
        if rank_sizes is None:
            raise RuntimeError(f"Pipeline rank {rank} did not provide tensor sizes for consolidated export")
        for key, tensor_bytes in rank_sizes.items():
            if key in global_sizes and global_sizes[key] != tensor_bytes:
                raise ValueError(
                    f"Conflicting logical sizes for {key!r} across pipeline ranks: "
                    f"{global_sizes[key]} and {tensor_bytes} bytes"
                )
            global_sizes[key] = tensor_bytes
    return global_sizes


def _maybe_adapt_state_dict_from_hf(
    model_part: nn.Module,
    state_dict: dict[str, torch.Tensor],
    moe_mesh: DeviceMesh | None = None,
    paramwrapper_layout_hint: str | None = None,
) -> dict[str, torch.Tensor]:
    """
    Custom models use state dict adapters to convert the state dict from the Hugging Face format to the native format.

    ``paramwrapper_layout_hint`` carries the fused expert LoRA layout recorded in
    the checkpoint's automodel_peft_config.json (see _read_paramwrapper_layout_metadata),
    so the adapter resolves the peft ParamWrapper layout from metadata instead of shapes.
    """
    adapter = getattr(_unwrap_ddp_model(model_part), "state_dict_adapter", None)
    if adapter:
        ep_mesh_dims = [dim for dim in moe_mesh.mesh_dim_names if dim != "pp"] if moe_mesh is not None else []
        ep_mesh = moe_mesh[tuple(ep_mesh_dims)] if ep_mesh_dims else moe_mesh
        adapter._paramwrapper_layout_hint = paramwrapper_layout_hint
        try:
            return adapter.from_hf(state_dict, device_mesh=ep_mesh)
        finally:
            adapter._paramwrapper_layout_hint = None
    return state_dict


def _read_paramwrapper_layout_metadata(model_path: str | os.PathLike) -> str | None:
    """Return the fused expert LoRA layout stamped into a PEFT checkpoint, if any.

    The PEFT save path records which peft ParamWrapper layout the adapter was
    exported in (peft flipped it in 0.19.1, huggingface/peft#3165) inside
    automodel_peft_config.json. Absent or unstamped checkpoints return None and
    the adapter falls back to shape detection.
    """
    metadata_path = os.path.join(model_path, "automodel_peft_config.json")
    try:
        with open(metadata_path) as f:
            return json.load(f).get("paramwrapper_layout")
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError):
        return None
