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

"""CPU optimizer offload (the ``--fsdp.offload optimizer`` mode).

Run the AdamW step on CPU so the fp32 master and Adam moments never occupy GPU
during the step — this shrinks the optimizer-step peak, the binding one for
long-context MoE RL. Params stay on GPU for the forward, so unlike FSDP param
offload it is MoE-safe. This is the Megatron HybridDeviceOptimizer essence
minus the d2h/h2d overlap, which is pointless when rollout >> the optimizer
step; a blocking, single-allocator-pool implementation gets the memory win for
free and avoids cross-stream fragmentation.
"""

import torch
import torch.optim as optim
from torch.distributed.tensor import DTensor


def local_shard(t: torch.Tensor) -> torch.Tensor:
    """The local-shard storage of a DTensor, else the plain tensor's storage. Paired
    with ``set_local_shard``: the to-CPU/back-to-GPU swap must capture the storage object
    (a plain param needs ``.data``, not the Parameter wrapper) so the GPU shard stays live
    for restore."""
    return t._local_tensor if isinstance(t, DTensor) else t.data


def set_local_shard(t: torch.Tensor, shard: torch.Tensor) -> None:
    """Point a DTensor at a new local shard in place (keeping mesh/placements), or swap a
    plain tensor's storage."""
    if isinstance(t, DTensor):
        t._local_tensor = shard
    else:
        t.data = shard


class CpuOptimizerOffloader:
    """Wrap an AdamW so ``step()`` runs on CPU while params stay resident on GPU.

    Exposes the ``step()``/``zero_grad()`` surface the AutoModel Engine drives, so the
    Engine needs no offload awareness; checkpointing keeps using the wrapped optimizer.
    The optimizer stays built over the model's GPU DTensor params (so its moments remain
    model-matched sharded DTensors — required by AutoModel's OptimizerState + DCP
    checkpoint; a separate-CPU-master optimizer would break resume). Per step we point
    each param's and grad's local shard at a persistent fp32 CPU master / reused CPU grad
    buffer, run the wrapped ``step()`` (so the Adam moments are created and kept on CPU),
    then copy the updated master H2D back into the retained GPU shard. The fp32 master is
    persistent, so updates don't round away even if the GPU param is bf16. AdamW only
    (Muon's Newton-Schulz is impractical on CPU). Numerically equivalent to a GPU step,
    not bit-identical (CPU vs GPU rounding of sqrt/division).

    Buffers are keyed by ``id(param)``: params live for the worker's lifetime (one model,
    built once), so the id is stable and never reused. A WeakKeyDictionary can't be used —
    weakref equality delegates to the tensor's elementwise ``__eq__``, which raises an
    ambiguous-bool error on lookup.
    """

    def __init__(self, optimizer: optim.Optimizer):
        self.optimizer = optimizer
        self._cpu_master = {}  # id(param) -> persistent fp32 CPU master shard
        self._cpu_grad = {}  # id(param) -> reused fp32 CPU grad buffer (D2H target)

    @torch.no_grad()
    def step(self) -> None:
        """Run the optimizer step on CPU over every param that received a (already
        GPU-clipped) grad. The GPU-shard restore runs in a ``finally`` so a raised
        ``step()`` can't leave a param CPU-resident for the next forward."""
        params = [p for group in self.optimizer.param_groups for p in group["params"] if p.grad is not None]
        gpu_shards = []
        for p in params:
            key = id(p)
            if key not in self._cpu_master:
                # Lazily allocate the persistent CPU buffers, re-deriving the master from
                # the current GPU shard — so on resume (model weights loaded before the
                # first step) the master reflects the restored weights.
                self._cpu_master[key] = local_shard(p).detach().to("cpu", torch.float32)
                self._cpu_grad[key] = torch.empty_like(self._cpu_master[key])
            self._cpu_grad[key].copy_(local_shard(p.grad))  # D2H grad (blocking)
            gpu_shards.append((p, local_shard(p)))
            set_local_shard(p, self._cpu_master[key])  # param -> CPU fp32 master
            set_local_shard(p.grad, self._cpu_grad[key])  # grad  -> CPU buffer
        try:
            self.optimizer.step()  # CPU AdamW: updates the masters + CPU Adam moments
        finally:
            for p, gpu_shard in gpu_shards:
                gpu_shard.copy_(self._cpu_master[id(p)])  # H2D updated weights (cast if bf16)
                set_local_shard(p, gpu_shard)  # restore the GPU shard for the next forward

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


@torch.no_grad()
def offload_moments_to_cpu(optimizer: optim.Optimizer) -> None:
    """Page the Adam moments to CPU after a checkpoint resume. DCP's
    set_optimizer_state_dict restores them onto the model param's (GPU) device, but the
    CPU step needs them on CPU (co-located with the swapped-out param/grad). Call right
    after loading the optimizer, before the first forward; a no-op in steady state (the
    step creates and keeps the moments on CPU)."""
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor) and local_shard(v).device.type != "cpu":
                set_local_shard(v, local_shard(v).to("cpu"))
