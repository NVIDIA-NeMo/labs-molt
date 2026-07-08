# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM weight refit for the FSDP2/AutoModel backend.

One module owns the two refit-time decisions the sender makes per parameter:

* *which* state-dict entries to push (``should_refit_state_dict_entry`` /
  ``state_dict_parameter_trainability``) — skip frozen VLM visual params and the
  non-trainable buffers vLLM already has from the base checkpoint;
* *how* to materialize each one (``gather_full_param``) — under FSDP2, params are
  ``DTensor`` instances whose ``.full_tensor()`` gathers the unsharded tensor
  across both FSDP shard and TP shard dims in one call.

The receiver-side plumbing in ``trainer/workers/policy_actor.py`` (a packed NCCL
broadcast of many weights at once, paired with the vLLM ``update_weights_packed``
RPC) is unchanged — only the *gather* step swaps in.
"""

from typing import Dict, Optional, Tuple

import torch
from torch.distributed.tensor import DTensor


def gather_full_param(param: torch.Tensor, dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, torch.Size]:
    """Materialize the full unsharded tensor for an FSDP2/TP-sharded parameter.

    Returns ``(full_tensor, full_shape)`` where ``full_tensor`` is on the local
    device with all mesh dims gathered. For non-DTensor params (e.g., the value
    head we don't shard, or buffers), returns ``(param.data, param.shape)``.

    Caller invokes this on each rank; ``full_tensor`` is replicated. Memory cost
    is the size of the full tensor on every participating rank — acceptable for
    weight refit (one-shot per training step). For very large models the async RL
    path uses per-tensor streaming with a ping-pong buffer to bound peak memory.
    """
    if isinstance(param, DTensor):
        full = param.full_tensor()
    else:
        full = param.data
    if dtype is not None and full.is_floating_point():
        full = full.to(dtype=dtype)
    return full, full.shape


def state_dict_parameter_trainability(model: torch.nn.Module) -> Dict[str, bool]:
    """Return state_dict parameter names with duplicate/tied aliases preserved."""
    try:
        named_parameters = model.named_parameters(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with older torch.
        named_parameters = model.named_parameters()
    return {name: param.requires_grad for name, param in named_parameters}


def should_refit_state_dict_entry(
    name: str,
    tensor: torch.Tensor,
    parameter_trainability: Dict[str, bool],
    *,
    is_vlm: bool,
) -> bool:
    if not torch.is_tensor(tensor):
        return False

    requires_grad = parameter_trainability.get(name)
    if requires_grad is None:
        # Non-trainable buffer → don't sync (general rule, no per-name allow-list):
        # vLLM already holds the correct value from its base-checkpoint load, and a
        # buffer is not updated by gradients during RL. Force-syncing one would also
        # DOWNCAST any fp32-kept buffer (e.g. the MoE router-correction
        # e_score_correction_bias) to the bf16 packed-sync dtype, corrupting vLLM's
        # fp32 copy and shifting expert routing vs the train engine. (molt is RL-only
        # and never calls the aux-loss-free `update_bias`, so these buffers are frozen.)
        return False

    # Keep the previous VLM behavior: frozen visual params are already present
    # in vLLM from the base checkpoint, so only language params that can change
    # during training need to be refit. `remove_duplicate=False` above keeps
    # tied language aliases such as `lm_head.weight` visible here.
    if is_vlm and not requires_grad:
        return False
    return True
