# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import logging
import math
import os
from typing import Any

import torch.nn as nn

logger = logging.getLogger(__name__)

# AutoModel stores MoE experts as a single *grouped* parameter of shape
# (num_experts, in, out) -> ndim == 3. AutoModel's stock grouping
# (`_separate_param_groups`) only sends ndim == 2 tensors to Muon, so on an MoE
# model the experts -- the bulk of the trainable params -- silently fall back to
# AdamW, defeating the point of Muon. We classify params ourselves so the
# grouped expert weights get Muon too, matching the Megatron `dist_muon` /
# Moonlight reference (Muon on linear + expert matrices; AdamW on embeddings,
# the output head, norms and biases). Set MUON_EXPERTS_ADAMW=1 to revert to
# AutoModel's experts-on-AdamW behavior (escape hatch for the EP path).
_EXPERT_MODULE_NAMES = ("GroupedExperts", "GroupedExpertsDeepEP", "GroupedExpertsTE")


def _classify_params(model: nn.Module, experts_to_muon: bool):
    """Split trainable params into (matrix->Muon, vector->AdamW, embed, lm_head).

    Mirrors AutoModel's `_separate_param_groups` roles with one correction:
    grouped MoE expert weights (ndim == 3, living in a ``GroupedExperts*``
    module) go to Muon instead of AdamW. Conv kernels and other non-expert 3D+
    tensors keep AutoModel's behavior (AdamW), so e.g. the linear-attention
    ``conv1d.weight`` is left alone.
    """
    named_modules = dict(model.named_modules())
    matrix, vector, embed, lm_head = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        module = named_modules.get(name.rsplit(".", 1)[0]) if "." in name else None
        # Match via the MRO: under multi-node EP, FSDP2 renames the experts module to a
        # dynamic subclass (FSDPGroupedExperts*), so a plain __name__ check would miss it
        # and drop the 3D expert weights back to AdamW. The original class stays in the MRO.
        is_expert = any(base.__name__ in _EXPERT_MODULE_NAMES for base in type(module).__mro__)

        if isinstance(module, nn.Embedding):
            embed.append(param)
        elif "lm_head" in name:
            lm_head.append(param)
        elif name.endswith("bias") or param.ndim <= 1:
            # biases (incl. 2D grouped expert biases like *_proj_bias), norms,
            # scalar gains -> AdamW
            vector.append(param)
        elif experts_to_muon and is_expert and param.ndim >= 3:
            matrix.append(param)  # grouped expert weight -> Muon  [the fix]
        elif param.ndim == 2:
            matrix.append(param)  # plain 2D linear weight -> Muon (AutoModel parity)
        else:
            vector.append(param)  # conv kernels / other 3D+ -> AdamW (AutoModel parity)
    return matrix, vector, embed, lm_head


def build_automodel_muon_optimizer(
    model: nn.Module,
    muon_cfg: dict[str, Any],
    adam_cfg: dict[str, Any],
    distributed_mesh=None,
):
    """Build AutoModel's Dion-family Muon optimizer from Molt CLI args.

    Unlike AutoModel's `build_dion_optimizer`, we build the param groups here so
    grouped MoE experts are optimized by Muon (see `_classify_params`). The
    Muon matrix group uses ``--muon.*``; the AdamW sub-group (vector/bias, embed,
    lm_head) uses ``--<role>.adam.*`` -- the two weight decays are independent,
    matching the CLI help.
    """
    # AutoModel renamed optim/utils.py -> optim/dion.py; support both pins.
    try:
        from nemo_automodel.components.optim import dion as automodel_optim
    except ImportError:
        from nemo_automodel.components.optim import utils as automodel_optim

    target = getattr(automodel_optim, "Muon", None)
    if target is None:
        raise RuntimeError(
            "AutoModel/Dion Muon is unavailable because the optional `dion` package is not installed. "
            "Install `dion` for --optim muon, or use --optim adam."
        ) from getattr(automodel_optim, "_import_error", None)

    muon_lr = muon_cfg["lr"]
    muon_weight_decay = muon_cfg.get("weight_decay")
    if muon_weight_decay is None:
        muon_weight_decay = adam_cfg["weight_decay"]
    adam_lr = adam_cfg["lr"]
    adam_weight_decay = adam_cfg["weight_decay"]
    betas = tuple(adam_cfg["betas"])
    eps = adam_cfg["eps"]

    # ns_steps is NOT a Dion knob: the Newton-Schulz iteration count is fixed
    # (PolarExpress / the 5-coefficient table). Reject a non-default value so the
    # CLI flag can't masquerade as an active tuning knob.
    if muon_cfg.get("ns_steps", 5) != 5:
        raise ValueError(
            "AutoModel/Dion Muon has no configurable Newton-Schulz step count; "
            "keep --muon.ns_steps=5 (the flag is a no-op)."
        )

    experts_to_muon = os.environ.get("MUON_EXPERTS_ADAMW", "") not in ("1", "true", "True")
    matrix, vector, embed, lm_head = _classify_params(model, experts_to_muon)

    scalar_kwargs = dict(beta1=betas[0], beta2=betas[1], epsilon=eps)
    # Group 0 (matrix) has no explicit lr/wd/algorithm; it inherits lr=muon_lr,
    # weight_decay=muon_weight_decay, mu and algorithm="muon" from the Muon
    # constructor defaults.
    param_groups = [
        dict(params=matrix),
        dict(params=vector, algorithm="adamw", lr=adam_lr, weight_decay=adam_weight_decay, **scalar_kwargs),
        dict(params=embed, algorithm="adamw", lr=adam_lr, weight_decay=0.0, **scalar_kwargs),
    ]
    if lm_head:
        first = lm_head[0]
        d_in = first.shape[-1] if first.ndim >= 2 else max(1, first.numel())
        lm_head_lr = muon_lr / math.sqrt(float(d_in))  # AutoModel's lm_head heuristic
        param_groups.append(dict(params=lm_head, algorithm="adamw", lr=lm_head_lr, weight_decay=0.0, **scalar_kwargs))
    param_groups = [g for g in param_groups if g["params"]]

    kwargs = dict(
        lr=muon_lr,
        mu=muon_cfg["momentum"],
        betas=betas,
        weight_decay=muon_weight_decay,
        epsilon=eps,
        nesterov=muon_cfg["nesterov"],
    )
    valid = inspect.signature(target).parameters
    if "distributed_mesh" in valid:
        # AutoModel knows how to pull the 1D dp_shard_cp submesh out of the full
        # device mesh; reuse it instead of duplicating the extraction.
        kwargs["distributed_mesh"] = automodel_optim._get_dion_mesh(distributed_mesh)
    kwargs = {k: v for k, v in kwargs.items() if k in valid}

    optimizer = target(param_groups, **kwargs)

    for i, group in enumerate(optimizer.param_groups):
        algo = group.get("algorithm", "muon")
        n_params = len(group["params"])
        n_elements = sum(p.numel() for p in group["params"])
        logger.info(
            f"[Muon] group {i}: algo={algo}, lr={group.get('lr')}, wd={group.get('weight_decay')}, "
            f"params={n_params}, elements={n_elements:,}"
        )
    return optimizer
