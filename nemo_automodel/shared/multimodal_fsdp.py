# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared helpers for FSDP treatment of multimodal submodules."""

from collections.abc import Callable, Iterator
from typing import Literal, cast

import torch.nn as nn

FrozenMultimodalSharding = Literal["root", "per_layer", "replicate"]

# Multimodal submodule names, split by role because the two groups have
# different consumers: ``_has_trainable_multimodal_tower`` in
# ``components/moe/parallelizer.py`` gates on towers alone, while FSDP policy
# resolution uses the union. Keep this the single definition -- a second copy
# drifts silently, and the drift only shows up as a sharding difference.
MULTIMODAL_TOWER_NAMES = (
    "visual",
    "vision",
    "vision_tower",
    "vision_model",
    "vit_model",
    "image_encoder",
    "vision_encoder",
    "audio_tower",
    "audio_encoder",
    "audio_model",
)

# Projector/embedder modules mapping tower output into the text hidden space.
# ``embed_vision`` and ``embed_audio`` are sibling ``Gemma4MultimodalEmbedder``
# instances on ``Gemma4Model``; listing only one gives two instances of the same
# class different FSDP treatment under every policy.
MULTIMODAL_PROJECTOR_NAMES = (
    "embed_vision",
    "embed_audio",
    "mm_projector",
    "aligner",
    "multi_modal_projector",
    "multimodal_projector",
    "vision_projector",
    "vit_large_projector",
    "audio_projector",
)

MULTIMODAL_MODULE_NAMES = MULTIMODAL_TOWER_NAMES + MULTIMODAL_PROJECTOR_NAMES

VALID_FROZEN_MULTIMODAL_SHARDING: tuple[FrozenMultimodalSharding, ...] = ("root", "per_layer", "replicate")


def normalize_frozen_multimodal_sharding(value: str) -> FrozenMultimodalSharding:
    """Validate and normalize the frozen multimodal FSDP policy."""
    if not isinstance(value, str):
        raise ValueError(f"distributed.multimodal.frozen_sharding must be a string. Got {type(value).__name__}.")
    normalized = value.lower().replace("-", "_")
    if normalized not in VALID_FROZEN_MULTIMODAL_SHARDING:
        valid = ", ".join(VALID_FROZEN_MULTIMODAL_SHARDING)
        raise ValueError(f"distributed.multimodal.frozen_sharding must be one of: {valid}. Got {value!r}.")
    return cast(FrozenMultimodalSharding, normalized)


def is_multimodal_module_name(name: str) -> bool:
    """Return True when ``name`` identifies a known multimodal tower/projector."""
    return name in MULTIMODAL_MODULE_NAMES


def module_parameters(module: nn.Module) -> list[nn.Parameter]:
    """Return the module's recursive parameters."""
    return list(module.parameters())


def module_is_fully_frozen(module: nn.Module) -> bool:
    """Return whether ``module`` owns parameters and none require gradients."""
    params = module_parameters(module)
    return bool(params) and not any(param.requires_grad for param in params)


def _is_module_container(module: nn.Module) -> bool:
    return isinstance(module, (nn.ModuleList, nn.ModuleDict))


def _container_items(module: nn.Module) -> list[tuple[object, nn.Module]]:
    if isinstance(module, nn.ModuleDict):
        return list(module.items())
    return list(enumerate(module))


def _shard_layer_containers_recursively(
    module: nn.Module,
    shard_module: Callable[[nn.Module], object],
) -> bool:
    sharded_child = False
    for _, child in module.named_children():
        if _is_module_container(child):
            for _, item in _container_items(child):
                if _is_module_container(item):
                    sharded_child |= _shard_layer_containers_recursively(item, shard_module)
                else:
                    shard_module(item)
                    sharded_child = True
        else:
            sharded_child |= _shard_layer_containers_recursively(child, shard_module)
    return sharded_child


def shard_multimodal_module(module: nn.Module, shard_module: Callable[[nn.Module], object]) -> None:
    """Shard a multimodal module at layer-container granularity when possible."""
    if not _shard_layer_containers_recursively(module, shard_module):
        shard_module(module)


def iter_multimodal_modules(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    """Yield maximal multimodal submodules by qualified name.

    ``named_modules`` is depth-first pre-order, so a parent is always seen
    before its descendants; already-selected prefixes are skipped to keep each
    tower/projector maximal (e.g. ``vision_tower`` rather than the
    ``vision_tower.vision_model`` nested inside it).

    The attribute-scan fallback only serves ``tests/unit_tests/moe`` model
    doubles, which are plain classes rather than ``nn.Module`` subclasses. It
    uses different (non-maximal, top-two-levels-only) selection than the real
    path, so tests that exercise it are not testing production behavior.
    Removing it requires converting ~30 doubles across that file to real
    modules; tracked separately rather than widening this change.
    """
    named_modules = getattr(model, "named_modules", None)
    if callable(named_modules):
        selected_names: list[str] = []
        for name, module in named_modules():
            if not name:
                continue
            if any(name == selected or name.startswith(selected + ".") for selected in selected_names):
                continue
            leaf_name = name.rsplit(".", 1)[-1]
            if is_multimodal_module_name(leaf_name):
                selected_names.append(name)
                yield name, module
        return

    seen_ids: set[int] = set()
    owners = [("", model)]
    inner_model = getattr(model, "model", None)
    if inner_model is not None and inner_model is not model:
        owners.append(("model", inner_model))

    for owner_name, owner in owners:
        for attr_name in MULTIMODAL_MODULE_NAMES:
            module = getattr(owner, attr_name, None)
            if module is None or id(module) in seen_ids:
                continue
            seen_ids.add(id(module))
            module_name = f"{owner_name}.{attr_name}" if owner_name else attr_name
            yield module_name, module


def ignored_params_for_root(root: nn.Module, ignored_params: set[nn.Parameter]) -> set[nn.Parameter] | None:
    """Return ignored parameters that are owned by an FSDP root.

    Args:
        root: Module that will become the FSDP root.
        ignored_params: Parameters of arbitrary shapes that should remain
            replicated rather than becoming part of the FSDP root.

    Returns:
        The subset of ``ignored_params`` owned by ``root``, preserving each
        parameter's original shape, or ``None`` when the subset is empty.
    """
    if not ignored_params:
        return None
    root_params = set(module_parameters(root))
    ignored_in_root = root_params & ignored_params
    return ignored_in_root or None
