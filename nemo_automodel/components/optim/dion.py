# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations

import logging
import math
from typing import Any, Protocol

import torch.nn as nn

from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh


class _DionFamilyConfig(Protocol):
    """Structural type for the dion-family optimizer configs build_dion_optimizer reads."""

    lr: float
    weight_decay: float
    scalar_opt: str
    scalar_betas: tuple[float, float]
    scalar_eps: float


_import_error: Exception | None = None
try:
    from dion import Dion, Dion2, Muon, NorMuon
except Exception as e:  # pragma: no cover - handled at runtime
    Dion = Dion2 = Muon = NorMuon = None
    _import_error = e

logger = logging.getLogger(__name__)


def _separate_param_groups(
    model: nn.Module,
    base_lr: float,
    scalar_opt: str,
    weight_decay: float,
    scalar_betas: tuple[float, float] | None = None,
    scalar_eps: float | None = None,
    scalar_lr: float | None = None,
    embed_lr: float | None = None,
    lm_head_lr: float | None = None,
) -> list[dict[str, Any]]:
    """
    Separate model parameters into groups for Dion/Muon optimizers.

    Args:
        model: The model to optimize.
        base_lr: Base learning rate for matrix params (Muon algorithm).
        scalar_opt: Optimizer algorithm for scalar params ("adamw" or "lion").
        weight_decay: Weight decay for vector params.
        scalar_betas: (beta1, beta2) for scalar optimizer.
        scalar_eps: Epsilon for scalar optimizer.
        scalar_lr: Learning rate for scalar (vector/bias) params. Defaults to base_lr.
        embed_lr: Learning rate for embedding params. Defaults to scalar_lr or base_lr.
        lm_head_lr: Learning rate for lm_head. Defaults to base_lr / sqrt(d_in).
    """
    matrix_params = []
    vector_params = []
    embed_params = []
    lm_head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        module = None
        try:
            module_name = name.rsplit(".", 1)[0]
            module = dict(model.named_modules()).get(module_name, None)
        except Exception:
            module = None

        if isinstance(module, nn.Embedding):
            embed_params.append(param)
            continue

        if "lm_head" in name:
            lm_head_params.append(param)
            continue

        if param.ndim == 2:
            matrix_params.append(param)
        else:
            vector_params.append(param)

    scalar_kwargs = {}
    if scalar_betas is not None:
        scalar_kwargs["beta1"] = scalar_betas[0]
        scalar_kwargs["beta2"] = scalar_betas[1]
    if scalar_eps is not None:
        scalar_kwargs["epsilon"] = scalar_eps

    effective_scalar_lr = scalar_lr if scalar_lr is not None else base_lr
    effective_embed_lr = embed_lr if embed_lr is not None else effective_scalar_lr

    param_groups: list[dict[str, Any]] = [
        dict(params=matrix_params),
        dict(
            params=vector_params,
            algorithm=scalar_opt,
            lr=effective_scalar_lr,
            weight_decay=weight_decay,
            **scalar_kwargs,
        ),
        dict(params=embed_params, algorithm=scalar_opt, lr=effective_embed_lr, weight_decay=0.0, **scalar_kwargs),
    ]

    if lm_head_params:
        # Use explicit lm_head_lr or scale by sqrt(d_in) as recommended in Dion docs
        if lm_head_lr is not None:
            effective_lm_head_lr = lm_head_lr
        else:
            first = lm_head_params[0]
            d_in = first.shape[-1] if first.ndim >= 2 else max(1, first.numel())
            effective_lm_head_lr = base_lr / math.sqrt(float(d_in))
        param_groups.append(
            dict(
                params=lm_head_params, algorithm=scalar_opt, lr=effective_lm_head_lr, weight_decay=0.0, **scalar_kwargs
            )
        )

    return param_groups


def _get_dion_mesh(device_mesh: Any) -> Any:
    if device_mesh is None:
        return None
    if not hasattr(device_mesh, "ndim") or device_mesh.ndim == 1:
        return device_mesh
    try:
        dion_mesh = get_fsdp_dp_mesh(device_mesh)
        logger.info(f"[Dion] Resolved FSDP DP mesh from device_mesh: {dion_mesh}")
        return dion_mesh
    except (AttributeError, KeyError, RuntimeError, TypeError) as e:
        logger.debug(f"[Dion] Could not resolve FSDP DP mesh: {e}")
    return device_mesh


def build_dion_optimizer(
    config: "_DionFamilyConfig",
    model: nn.Module,
    *,
    device_mesh: Any | None = None,
    mesh_kwarg: str | None = "distributed_mesh",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Build the parameter groups and resolve the device mesh for a Dion-family
    optimizer.

    This does not instantiate the optimizer; it returns ``(param_groups,
    mesh_kwargs)`` so the caller (a typed config in
    :mod:`nemo_automodel.components.optim.optimizer`) can assemble its own
    constructor kwargs and instantiate the optimizer itself.  ``mesh_kwargs`` is a
    dict that maps ``mesh_kwarg`` to the resolved mesh (or is empty when there is
    no mesh), ready to splat into the optimizer constructor.

    The parameter-grouping settings are read off ``config``: ``lr``,
    ``weight_decay``, ``scalar_opt``, ``scalar_betas``, ``scalar_eps``
    (required), and the optional ``scalar_lr``, ``embed_lr``, ``lm_head_lr`` and
    ``no_compile``.

    Args:
        config: The dion-family config (see :class:`_DionFamilyConfig`) to read settings from.
        model: Model whose parameters are to be optimized.
        device_mesh: Optional DeviceMesh for FSDP/TP. When non-empty it is
            resolved to a 1-D Dion submesh.
        mesh_kwarg: Name of the constructor argument that receives the resolved
            mesh (``"distributed_mesh"`` for Muon/Dion2/NorMuon,
            ``"outer_shard_mesh"`` for legacy Dion).  Set to ``None`` to never
            include the mesh.

    Returns:
        A ``(param_groups, mesh_kwargs)`` tuple: the per-group parameter dicts and
        the mesh constructor kwargs (``{mesh_kwarg: mesh}`` or ``{}``).
    """
    if _import_error:
        raise RuntimeError("Failed to import Dion. Please install Dion.") from _import_error

    if getattr(config, "no_compile", False):
        import torch._dynamo

        torch._dynamo.config.disable = True
        logger.info("[Dion] no_compile=True: torch._dynamo fully disabled (optimizer runs in eager mode)")

    scalar_betas = config.scalar_betas
    if scalar_betas is not None:
        scalar_betas = tuple(scalar_betas) or None

    base_lr = float(config.lr)
    weight_decay = float(config.weight_decay)

    param_groups = _separate_param_groups(
        model,
        base_lr,
        config.scalar_opt,
        weight_decay,
        scalar_betas=scalar_betas,
        scalar_eps=config.scalar_eps,
        scalar_lr=getattr(config, "scalar_lr", None),
        embed_lr=getattr(config, "embed_lr", None),
        lm_head_lr=getattr(config, "lm_head_lr", None),
    )

    dion_mesh = _get_dion_mesh(device_mesh)
    mesh_kwargs: dict[str, Any] = {}
    if mesh_kwarg is not None and dion_mesh is not None:
        mesh_kwargs[mesh_kwarg] = dion_mesh

    logger.info(f"[Dion] Built {len(param_groups)} param groups:")
    for i, pg in enumerate(param_groups):
        algo = pg.get("algorithm", "dion2 (default)")
        n_params = len(pg["params"])
        n_elements = sum(p.numel() for p in pg["params"])
        lr_override = pg.get("lr", "default")
        logger.info(f"  Group {i}: algo={algo}, params={n_params}, elements={n_elements:,}, lr={lr_override}")

    return param_groups, mesh_kwargs
