# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""vLLM weight refit for the FSDP2/AutoModel backend.

Owns *how* to materialize each pushed parameter (``gather_full_param``): under
FSDP2, params are ``DTensor`` instances whose ``.full_tensor()`` gathers the
unsharded tensor across both FSDP shard and TP shard dims in one call.

The sender (``trainer/workers/policy_actor.py``) pushes every ``state_dict``
entry — vLLM's ``load_weights`` matches by name and ignores what it doesn't have,
so the "which weights to accept" decision lives on the vLLM side. The one
exception is LoRA: vLLM has no adapter parameter to match, so the adapters are
merged into their base weight here instead (``lora_refit_merges``).
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
    full = param.full_tensor() if isinstance(param, DTensor) else param.data
    if dtype is not None and full.is_floating_point():
        full = full.to(dtype=dtype)
    return full, full.shape


def lora_refit_merges(
    model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]:
    """Map each LoRA-adapted base weight's ``state_dict`` key to ``(first, second, scale)``.

    vLLM holds no adapter parameter, and its ``AutoWeightsLoader`` *raises* on an unknown
    name rather than dropping it, so refit pushes the merged effective weight
    ``W + scale * (first @ second)`` under the base FQN. The operands are pre-ordered so the
    product already carries the base weight's shape: a dense ``nn.Linear`` holds
    ``[out, in]`` (``lora_B @ lora_A``), grouped experts hold ``[E, in, out]``
    (``lora_A @ lora_B``, batched).

    Walking modules instead of parsing names is what gets ``scale`` right — it is
    ``alpha/rank`` *per module*, and AutoModel's ``moe_rank_scaling`` gives the experts a
    different rank than the dense layers.
    """
    merges = {}
    for name, module in model.named_modules():
        prefix = f"{name}." if name else ""
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            merges[prefix + "weight"] = (module.lora_B.weight, module.lora_A.weight, module.scale)
        elif hasattr(module, "lora_gate_and_up_A"):
            merges[prefix + "gate_and_up_projs"] = (module.lora_gate_and_up_A, module.lora_gate_and_up_B, module.scale)
            merges[prefix + "down_projs"] = (module.lora_down_A, module.lora_down_B, module.scale)

    # Fail loud on a module-FQN / state_dict-key mismatch: the base weights are frozen under
    # LoRA, so an unmerged refit is silent — the engine keeps serving the initial policy and
    # only shows up much later as vllm_kl climbing with training.
    missing = sorted(merges.keys() - state_dict.keys())
    if missing:
        raise RuntimeError(
            f"Refit: {len(missing)} LoRA-adapted base weights are absent from the model "
            f"state_dict (e.g. {missing[:3]}), so their adapters would never reach vLLM and the "
            "rollout engine would keep serving the unadapted base policy."
        )
    return merges
