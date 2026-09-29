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

import json
import logging
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_utils import _get_resolved_checkpoint_files, load_state_dict

from nemo_automodel.shared.tied_weights import (
    ensure_tied_lm_head as ensure_tied_lm_head,
)
from nemo_automodel.shared.tied_weights import (
    get_input_embeddings_weight_and_name,
    get_lm_head_weight_and_name,
    is_tied_word_embeddings,
)
from nemo_automodel.shared.tied_weights import (
    has_local_tied_lm_head as has_local_tied_lm_head,
)

logger = logging.getLogger(__name__)
_AUTOMODEL_CHECKPOINT_RE = re.compile(r"^epoch_\d+_step_(\d+)$")
_CHECKPOINT_STEP_RE = re.compile(r"step_(\d+)$")
_INCOMPLETE_CHECKPOINT_MARKER = ".incomplete"


def get_rank_safe() -> int:
    """Return the current distributed rank, defaulting to 0 when not initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_world_size_safe() -> int:
    """Return the current distributed world size, defaulting to 1 when not initialized."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_rank_0() -> bool:
    """Return True on the main rank."""
    return get_rank_safe() == 0


def estimate_tensor_bytes(tensor: torch.Tensor) -> int:
    """Estimate logical bytes in a tensor without materializing it."""
    return int(tensor.numel()) * int(tensor.element_size())


def estimate_state_dict_bytes(state_dict: dict[str, torch.Tensor]) -> int | None:
    """Estimate logical bytes in a state dict without materializing tensors."""
    total = 0
    try:
        for tensor in state_dict.values():
            total += estimate_tensor_bytes(tensor)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return total


def get_safetensors_index_total_size(index_path: str | None) -> int | None:
    """Return the total checkpoint size recorded in a Hugging Face safetensors index."""
    if index_path is None:
        return None
    index_file = Path(index_path)
    if index_file.is_dir():
        index_file = index_file / "model.safetensors.index.json"
    try:
        with open(index_file) as f:
            index = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    total_size = index.get("metadata", {}).get("total_size")
    return total_size if isinstance(total_size, int) else None


def format_bytes(num_bytes: int) -> str:
    """Format bytes as a human-readable GiB value."""
    return f"{num_bytes / 1024**3:.1f} GiB"


def format_output_file_count(count: int) -> str:
    """Format the output shard count for user-facing log messages."""
    return f"{count} output {'file' if count == 1 else 'files'}"


def _checkpoint_step_num(path: Path) -> int:
    """Return the trailing checkpoint step number, or -1 when the name is not a checkpoint."""
    match = _CHECKPOINT_STEP_RE.search(path.name)
    return int(match.group(1)) if match else -1


def _list_existing_checkpoints(ckpt_root: Path) -> list[Path]:
    """Return existing checkpoint directories whose names end in ``step_<N>``."""
    if not ckpt_root.exists():
        return []
    checkpoints = [path for path in ckpt_root.glob("*step_*") if path.is_dir() and not path.is_symlink()]
    return sorted((path for path in checkpoints if _checkpoint_step_num(path) >= 0), key=_checkpoint_step_num)


def is_cloud_path(path: str | Path) -> bool:
    """Check if path is a cloud storage path (MSC)."""
    return os.fspath(path).startswith("msc://")


def mark_checkpoint_incomplete(checkpoint_dir: str | Path) -> None:
    """Mark a checkpoint directory as still being written.

    A save that is interrupted (wall-clock limit, preemption, OOM) leaves the
    directory behind with only some of its components written. The marker makes
    that state observable so a resumed run neither loads nor retains it; it is
    removed by :func:`clear_checkpoint_incomplete` once the checkpoint is
    published.

    Args:
        checkpoint_dir: Directory of the checkpoint being written.
    """
    (Path(checkpoint_dir) / _INCOMPLETE_CHECKPOINT_MARKER).touch(exist_ok=True)


def clear_checkpoint_incomplete(checkpoint_dir: str | Path) -> None:
    """Clear the marker written by :func:`mark_checkpoint_incomplete`.

    Args:
        checkpoint_dir: Directory of the checkpoint that finished writing.
    """
    (Path(checkpoint_dir) / _INCOMPLETE_CHECKPOINT_MARKER).unlink(missing_ok=True)


def _resolve_checkpoint_pointer_target(ckpt_root: Path, raw_target: str) -> Path | None:
    """Resolve a checkpoint pointer target relative to ckpt_root."""
    if not raw_target:
        return None
    target = Path(raw_target)
    if not target.is_absolute():
        target = ckpt_root / target
    return Path(os.path.abspath(target))


def _checkpoint_contains_target(checkpoint: Path, target: Path) -> bool:
    """Return whether target points at or inside checkpoint."""
    checkpoint_abs = Path(os.path.abspath(checkpoint))
    target_abs = Path(os.path.abspath(target))
    return target_abs == checkpoint_abs or checkpoint_abs in target_abs.parents


def _is_checkpoint_pointer_text_file(path: Path, mode: int) -> bool:
    """Return whether path looks like a symlink fallback checkpoint pointer."""
    return stat.S_ISREG(mode) and path.suffix == ".txt" and path.stem.isupper()


def resolve_trust_remote_code(pretrained_model_name_or_path):
    """
    Whitelist NVIDIA models to allow remote code execution.

    Args:
        pretrained_model_name_or_path (str): The name or path of the pretrained model.

    Returns:
        bool: True if the model should be loaded with trust_remote_code, False otherwise.
    """
    if not pretrained_model_name_or_path:
        return False
    # pretrained_model_name_or_path can be something like nvidia/NVIDIA-Nemotron-Nano-9B-v2
    return not os.path.isdir(pretrained_model_name_or_path) and pretrained_model_name_or_path.startswith("nvidia/")


def get_tied_lm_head_source_names(model: nn.Module, lm_head_param_name: str | None = None) -> list[str]:
    """Return candidate checkpoint keys that can source a tied LM head.

    Args:
        model: Model or pipeline stage to inspect.
        lm_head_param_name: Optional normalized LM head FQN.

    Returns:
        Ordered list of possible source FQNs.
    """
    candidate_source_names: list[str] = []
    tied_keys = getattr(model, "_tied_weights_keys", None)
    # ``_tied_weights_keys`` has two shapes in practice:
    #   - dict: NeMo custom models set an explicit target->source map
    #     (e.g. ``{"lm_head.weight": "model.embed_tokens.weight"}``);
    #   - list/tuple/set of str: HF upstream lists only the *target* FQNs
    #     that are tied to the input embedding. The source is resolved via
    #     ``get_input_embeddings()`` below, so for the list shape we only use
    #     the dict for target-name matching and rely on the fallbacks to find
    #     the source.
    if isinstance(tied_keys, dict):
        for target_name, source_name in tied_keys.items():
            if not isinstance(target_name, str) or not isinstance(source_name, str):
                continue
            if (
                lm_head_param_name is None
                or target_name == lm_head_param_name
                or target_name.endswith("lm_head.weight")
            ):
                candidate_source_names.append(source_name)

    _, input_embeddings_param_name = get_input_embeddings_weight_and_name(model)
    if input_embeddings_param_name is not None:
        candidate_source_names.append(input_embeddings_param_name)

    candidate_source_names.extend(
        [
            "model.language_model.embed_tokens.weight",
            "language_model.embed_tokens.weight",
            "model.embed_tokens.weight",
            "embed_tokens.weight",
        ]
    )

    seen_source_names: set[str] = set()
    deduped_source_names: list[str] = []
    for source_name in candidate_source_names:
        if source_name in seen_source_names:
            continue
        seen_source_names.add(source_name)
        deduped_source_names.append(source_name)
    return deduped_source_names


def materialize_missing_tied_lm_head(
    state_dict: dict[str, Any],
    model: nn.Module,
    *,
    allow_current_lm_head_fallback: bool = False,
) -> bool:
    """Populate a missing tied ``lm_head.weight`` from its embedding source.

    Hugging Face checkpoints for tied-embedding models often omit
    ``lm_head.weight`` entirely. That is fine for unsplit models where
    ``tie_weights()`` can restore the alias, but it breaks pipeline-parallel last
    stages which own ``lm_head`` but not ``embed_tokens``.

    Args:
        state_dict: Checkpoint state dict to mutate in place.
        model: Target model or pipeline stage.
        allow_current_lm_head_fallback: If ``True``, fall back to the current
            ``lm_head`` tensor when the tied source cannot be found in
            ``state_dict``. This preserves legacy resume behavior for older
            checkpoints that were saved without a local ``lm_head.weight``.

    Returns:
        ``True`` if a missing ``lm_head.weight`` was materialized, else ``False``.
    """
    if not is_tied_word_embeddings(model):
        return False

    lm_head_weight, lm_head_param_name = get_lm_head_weight_and_name(model)
    if lm_head_weight is None or lm_head_param_name is None or lm_head_param_name in state_dict:
        return False

    for source_name in get_tied_lm_head_source_names(model, lm_head_param_name):
        tensor = state_dict.get(source_name)
        if isinstance(tensor, torch.Tensor):
            state_dict[lm_head_param_name] = tensor.detach()
            return True

    if allow_current_lm_head_fallback:
        state_dict[lm_head_param_name] = lm_head_weight.detach()
        return True

    return False


def _get_checkpoint_tensor_dtypes(
    pretrained_model_name_or_path: str,
    hf_config: Any,
    load_kwargs: Mapping[str, object] | None = None,
) -> dict[str, torch.dtype]:
    """Inspect checkpoint tensors and return their exact dtypes by key.

    This reads checkpoint metadata only by loading tensors on the ``meta``
    device, so it preserves the per-tensor dtype information without
    materializing full checkpoint weights in memory.
    """
    load_kwargs = dict(load_kwargs or {})

    provided_state_dict = load_kwargs.get("state_dict")
    if isinstance(provided_state_dict, Mapping):
        return {name: tensor.dtype for name, tensor in provided_state_dict.items() if isinstance(tensor, torch.Tensor)}

    if load_kwargs.get("gguf_file") is not None:
        return {}

    trust_remote_code = load_kwargs.get(
        "trust_remote_code",
        resolve_trust_remote_code(pretrained_model_name_or_path),
    )
    download_kwargs = {
        key: load_kwargs[key]
        for key in (
            "cache_dir",
            "force_download",
            "proxies",
            "local_files_only",
            "token",
            "revision",
            "subfolder",
            "commit_hash",
        )
        if key in load_kwargs
    }
    checkpoint_files, _ = _get_resolved_checkpoint_files(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        variant=load_kwargs.get("variant"),
        gguf_file=load_kwargs.get("gguf_file"),
        use_safetensors=load_kwargs.get("use_safetensors"),
        user_agent={"file_type": "model", "framework": "pytorch"},
        is_remote_code=bool(trust_remote_code),
        transformers_explicit_filename=getattr(hf_config, "transformers_weights", None),
        download_kwargs=download_kwargs,
    )
    if not checkpoint_files:
        return {}

    checkpoint_dtypes: dict[str, torch.dtype] = {}
    weights_only = bool(load_kwargs.get("weights_only", True))
    for checkpoint_file in checkpoint_files:
        state_dict = load_state_dict(checkpoint_file, map_location="meta", weights_only=weights_only)
        checkpoint_dtypes.update(
            {name: tensor.dtype for name, tensor in state_dict.items() if isinstance(tensor, torch.Tensor)}
        )
    return checkpoint_dtypes
