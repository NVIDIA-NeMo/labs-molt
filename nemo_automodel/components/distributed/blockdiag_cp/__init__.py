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

"""Block-diagonal (per-document) varlen context parallelism for packed sequences.

Packed-sequence training concatenates many documents into one long sequence; a
correct attention mask is block-causal per document. The stock DTensor
``context_parallel`` path assumes a single causal document, so packed VLM/LLM
batches need a CP implementation that reshards by contiguous chunks and rebuilds
per-document masking on every rank.

This package provides that implementation, split by responsibility:

- :mod:`.state`   -- runtime knob normalization + activation-checkpoint-safe step state.
- :mod:`.kernels` -- dense-mask and varlen (FlashAttention / TransformerEngine) kernels.
- :mod:`.exchange` -- differentiable K/V collectives (all-gather, left-halo, all-to-all-v).
- :mod:`.runtime` -- the SDPA entry point and collective-safe path selection.
- :mod:`.batch`   -- batch padding/sharding and the per-step train context.
- :mod:`.packed`  -- the cp_size==1 packed-sequence varlen SDPA hook.

Integration follows the model-owned CP convention of
:func:`nemo_automodel.components.distributed.cp_utils.make_cp_batch_and_ctx`: a model
returns a
:class:`~nemo_automodel.components.distributed.context_parallel.sharder.ContextParallelSharder`
whose batch verb is :func:`make_cp_blockdiag_batch_and_ctx`, and routes its softmax
attention through :func:`cp_blockdiag_sdpa` while the returned context is active.
For ``cp_size == 1`` packed runs, the model scopes the varlen SDPA patch to its
attention forwards with :func:`attach_cp1_packed_varlen_hooks` and arms the
per-forward state with :func:`enable_cp1_packed_varlen` /
:func:`disable_cp1_packed_varlen`.

Only these integration entry points are exported; everything else (knob
normalization, varlen metadata precompute, fire counters, kernels) is an internal
detail of the package's modules. Model wiring lands in follow-up PRs.
"""

from nemo_automodel.components.distributed.blockdiag_cp.batch import make_cp_blockdiag_batch_and_ctx
from nemo_automodel.components.distributed.blockdiag_cp.packed import (
    attach_cp1_packed_varlen_hooks,
    cp1_packed_varlen_backend,
    disable_cp1_packed_varlen,
    enable_cp1_packed_varlen,
)
from nemo_automodel.components.distributed.blockdiag_cp.runtime import cp_blockdiag_sdpa
from nemo_automodel.components.distributed.blockdiag_cp.state import (
    BlockdiagCpModelState,
    configure_cp_varlen,
    current_blockdiag_cp_state,
)

__all__ = [
    "attach_cp1_packed_varlen_hooks",
    "BlockdiagCpModelState",
    "configure_cp_varlen",
    "current_blockdiag_cp_state",
    "cp_blockdiag_sdpa",
    "cp1_packed_varlen_backend",
    "disable_cp1_packed_varlen",
    "enable_cp1_packed_varlen",
    "make_cp_blockdiag_batch_and_ctx",
]
