# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

from typing import Optional, Union

import torch
import torch.nn as nn
from torch.distributed.tensor import DTensor


def resolve_ac_mode(value: Union[bool, str, None]) -> Union[bool, str]:
    """Normalize the gradient_checkpoint CLI value into AutoModel's
    ActivationCheckpointingMode (``bool | "selective"``).

    The flag is ``nargs="?"`` with ``const="full"``: a bare
    ``--…gradient_checkpoint`` -> ``"full"``, or an explicit mode string.
    ``"full"``/``"true"`` -> ``True`` (full-block AC, the value every
    AutoModel MoE/deepep recipe uses), ``"selective"`` -> per-op AC, and
    falsy words -> ``False``. Used by BOTH the MoE path (actor.py ->
    DistributedSetup) and the dense/HF path (strategy.py -> FSDP2Config) so they
    never disagree on the AC mode.
    """
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "selective":
            return "selective"
        return v not in ("", "false", "none", "off", "0")
    return bool(value)


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """

    log_ratio = torch.nan_to_num(
        log_probs.float() - log_probs_base.float(),
        nan=0.0,
        posinf=30.0,
        neginf=-30.0,
    )

    if kl_estimator == "k1":
        # Signed log-ratio p - q, returned unclamped: the nan_to_num above already
        # bounds true infinities to ±30, and on-policy distillation consumes this as
        # a dense per-token reward (advantage = -kl_coef * kl) that must not be capped
        # on the most-divergent tokens (matches slime, which never clamps it). Only the
        # non-negative loss-side estimators (k2/k3) get the ±10 bound below.
        return log_ratio

    if kl_estimator == "k2":
        # Non-negative KL approximation: (p - q)^2 / 2
        # http://joschu.net/blog/kl-approx.html
        # Approximately equivalent to one-step KL penalty with k1
        # used in https://arxiv.org/pdf/2310.10505.
        log_ratio = log_ratio**2 / 2.0
    elif kl_estimator == "k3":
        # Non-negative KL approximation: exp(q - p) - 1 - (q - p)
        # http://joschu.net/blog/kl-approx.html
        log_ratio = (-log_ratio).exp() - 1 + log_ratio
    else:
        raise ValueError(f"Unknown kl_estimator: {kl_estimator}")

    return log_ratio.clamp(min=-10, max=10)


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature != 1.0:
        # Non-inplace: callers may keep the tensor in `output["logits"]` for
        # downstream consumers (allgather/entropy paths share the buffer).
        logits = logits / temperature

    batch_dim = logits.shape[:-1]
    last_dim = logits.shape[-1]
    flat_logits = logits.reshape(-1, last_dim)
    flat_labels = labels.reshape(-1)

    # Fast path: fused triton CE kernel only supports fp32/fp64.
    # https://github.com/OpenRLHF/OpenRLHF/pull/718#issuecomment-2641081881
    if logits.dtype in [torch.float32, torch.float64]:
        try:
            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

            output = cross_entropy_loss(flat_logits, flat_labels)
            return (-output[0]).view(*batch_dim)
        except ImportError:
            pass

    # Chunked fp32 logsumexp+gather. Bounds peak memory at
    # chunk_size * vocab * 4 bytes (256 * 152K * 4 ≈ 156 MiB) and avoids the
    # [B*S, V] fp32 spike that OOMs at long sequences with large vocab models
    # like Qwen3.6 (152K vocab) when callers pass bf16 logits. Empirically a
    # 1024 chunk OOMs on 80GB H100 once optimizer+activations are loaded.
    n_rows = flat_logits.shape[0]
    out = torch.empty(n_rows, device=logits.device, dtype=torch.float32)
    chunk_size = 256
    for s_idx in range(0, n_rows, chunk_size):
        end_idx = min(s_idx + chunk_size, n_rows)
        chunk = flat_logits[s_idx:end_idx].float()
        gathered = chunk.gather(dim=-1, index=flat_labels[s_idx:end_idx].unsqueeze(-1)).squeeze(-1)
        lse = torch.logsumexp(chunk, dim=-1)
        out[s_idx:end_idx] = gathered - lse
    return out.view(*batch_dim)


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(dim=dim)
    valid = torch.where(mask.bool(), tensor, torch.zeros_like(tensor))
    denom = mask.sum(dim=dim).clamp_min(1)
    return valid.sum(dim=dim) / denom


@torch.compile
def compute_entropy(logits: torch.Tensor):
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)
    return entropy


def split_moe_aux_loss(output, enabled: bool):
    """Return ``(aux_loss_for_optimization, aux_loss_for_logging)``.

    HF MoE models return an unscaled ``output.aux_loss`` that Molt adds to
    the trainer loss. NeMo AutoModel custom MoE injects aux-loss gradients via
    ``MoEAuxLossAutoScaler`` during backward, so those outputs are marked and the
    first element is zeroed (already in the gradient) while logging still sees it.
    """
    if not enabled:
        return 0.0, 0.0

    if isinstance(output, dict):
        aux_loss = output.get("aux_loss", 0.0)
        in_backward = bool(output.get("_molt_aux_loss_in_backward", False))
    else:
        aux_loss = getattr(output, "aux_loss", 0.0)
        in_backward = bool(getattr(output, "_molt_aux_loss_in_backward", False))

    if aux_loss is None:
        aux_loss = 0.0
    return (0.0 if in_backward else aux_loss), aux_loss


def move_model_to_cpu_for_offload(model: nn.Module, distributed_config):
    """Move params + buffers to CPU when FSDP offload is on (else a no-op)."""
    if getattr(distributed_config, "offload_policy", None) is None:
        return model
    for buffer in model.buffers():
        buffer.data = buffer.data.to("cpu")
    return model.to("cpu")


def _iter_nemo_moe_gates(model: nn.Module):
    for module in model.modules():
        cls = type(module)
        if (
            cls.__name__ == "Gate"
            and cls.__module__.startswith("nemo_automodel.components.moe")
            and hasattr(module, "aux_loss_coeff")
        ):
            yield module


def configure_nemo_moe_aux_loss(model: nn.Module, aux_loss_coef: float) -> bool:
    """Use NeMo's MoE aux-loss autograd path with Molt's CLI coefficient."""
    coef = float(aux_loss_coef or 0.0)
    gates = list(_iter_nemo_moe_gates(model))
    if not gates:
        return False

    for gate in gates:
        gate.aux_loss_coeff = coef
        if coef > 0:
            gate._track_load_balance = True

    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "router_aux_loss_coef"):
        config.router_aux_loss_coef = coef
    for module in model.modules():
        moe_config = getattr(module, "moe_config", None)
        if moe_config is not None and hasattr(moe_config, "aux_loss_coeff"):
            moe_config.aux_loss_coeff = coef

    active = coef > 0
    model._molt_aux_loss_in_backward = active
    return active


def attach_nemo_moe_aux_loss(output, model: nn.Module):
    if not model.training or not getattr(model, "_molt_aux_loss_in_backward", False):
        return output

    aux_losses = []
    for gate in _iter_nemo_moe_gates(model):
        aux_loss = getattr(gate, "_last_aux_loss", None)
        if aux_loss is None:
            continue
        if isinstance(aux_loss, DTensor):
            aux_loss = aux_loss.to_local()
        aux_losses.append(aux_loss.detach().float())
    if not aux_losses:
        return output

    aux_loss = torch.stack(aux_losses).sum()
    if isinstance(output, dict):
        output["aux_loss"] = aux_loss
        output["_molt_aux_loss_in_backward"] = True
    else:
        setattr(output, "aux_loss", aux_loss)
        setattr(output, "_molt_aux_loss_in_backward", True)
    return output
