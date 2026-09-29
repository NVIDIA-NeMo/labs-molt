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

"""Low-level linear graph-shaping helpers shared by tensor-parallel components.

Inductor's async tensor-parallel pass (``torch._inductor.config._micro_pipeline_tp``)
fuses collectives with matmuls by pattern-matching the reshape-mm-reshape graph
that ``F.linear`` produces for 3-D input.  The compile-safe ``torch.bmm`` path
used by ``TPLinear``/``LinearLoRA`` never matches, so async-TP fusion silently
fails to fire.  These helpers detect async-TP tracing and emit the native
linear graph in that mode only.
"""

import torch
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Shard


def _is_async_tp_linear_enabled() -> bool:
    """Return whether Dynamo is tracing with Inductor's async-TP pass enabled.

    True only when both conditions hold: the caller is being traced by
    torch.compile and ``torch._inductor.config._micro_pipeline_tp`` is set
    (see ``enable_async_tensor_parallel`` in parallelizer.py).  Always False
    in eager mode.
    """
    if not torch.compiler.is_compiling():
        return False
    inductor = getattr(torch, "_inductor", None)
    config = getattr(inductor, "config", None)
    return bool(getattr(config, "_micro_pipeline_tp", False))


def _async_tp_linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """Emit the native linear graph recognized by async-TP fusion.

    Args:
        x: Input activations of shape ``[..., in_features]``; any number of
            leading dimensions is accepted (typically ``[B, S, in_features]``
            with ``B`` = batch, ``S`` = sequence).  May be a DTensor with
            tensor-parallel placements.
        weight: Weight of shape ``[out_features, in_features]``; may be a
            DTensor sharded for colwise (``Shard(0)``) or rowwise
            (``Shard(1)``) tensor parallelism.
        bias: Optional bias of shape ``[out_features]``.

    Returns:
        Output of shape ``[..., out_features]`` with the same leading
        dimensions as ``x``.
    """
    # Keep bias outside F.linear so the collective sees the matmul result as
    # its direct producer/consumer.  PyTorch lowers 3-D F.linear to the
    # reshape-mm-reshape pattern consumed by micro_pipeline_tp.
    output = F.linear(x, weight)
    return output + bias if bias is not None else output


def tp_linear_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    mm_for_2d_compile: bool,
) -> torch.Tensor:
    """Dispatch a TP-safe linear between F.linear, async-TP shaping, and bmm/mm.

    Shared by ``TPLinear.forward`` and ``LinearLoRA.forward``.  Eager
    unsharded/last-dimension-sharded inputs use ``F.linear``; async-TP tracing emits the
    fusable native linear graph via ``_async_tp_linear``; every remaining
    compile path and dim-0/1-sharded DTensor input keeps the DTensor-safe
    matmul fallback (``aten.view`` cannot flatten a sharded dimension).

    Args:
        x: Input activations of shape ``[B, S, in_features]`` (``B`` = batch,
            ``S`` = sequence) or ``[N, in_features]`` (``N`` = flattened
            tokens).  May be a DTensor: a 3-D DTensor sharded on dim 0 or 1
            (e.g. ``Shard(1)`` from sequence parallelism) always takes the
            ``torch.bmm`` path — the async-TP fast path assumes the input is
            replicated or sharded only on the feature dim, so fusion simply
            does not fire for such layers.
        weight: Weight of shape ``[out_features, in_features]``; may be a
            DTensor sharded for colwise (``Shard(0)``) or rowwise (``Shard(1)``)
            tensor parallelism.
        bias: Optional bias of shape ``[out_features]``; replicated if DTensor.
        mm_for_2d_compile: Numerics for 2-D ``x`` traced under torch.compile
            without async-TP: ``torch.mm`` plus explicit bias add when True
            (``TPLinear``), ``F.linear`` when False (``LinearLoRA``).

    Returns:
        Output of shape ``[..., out_features]`` with the same leading
        dimensions as ``x``.
    """
    # bmm avoids aten.view which cannot flatten a sharded dimension.
    # F.linear calls view([b,s,h]->[b*s,h]) which fails when dim 0/1 is sharded
    # (sequence parallelism) or during AOT-autograd tracing with compile.
    x_needs_bmm = (
        isinstance(x, DTensor)
        and x.dim() == 3
        and any(isinstance(p, Shard) and p.dim % x.dim() < x.dim() - 1 for p in x.placements)
    )
    if _is_async_tp_linear_enabled() and not x_needs_bmm:
        return _async_tp_linear(x, weight, bias)
    if not torch.compiler.is_compiling() and not x_needs_bmm:
        return F.linear(x, weight, bias)
    if x.dim() == 3:
        out = torch.bmm(x, weight.t().unsqueeze(0).expand(x.shape[0], -1, -1))
    elif mm_for_2d_compile:
        out = torch.mm(x, weight.t())
    else:
        return F.linear(x, weight, bias)
    return out + bias if bias is not None else out
