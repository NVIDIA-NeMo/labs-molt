# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""fp32 SDPA attention for Gemma4 (fix for #2208: fused bf16 SDPA NaNs on Hopper)."""

import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward


def sdpa_fp32_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run SDPA with fp32 query/key/value and cast the output back to the input dtype."""
    in_dtype = query.dtype
    if attention_mask is not None and torch.is_tensor(attention_mask) and attention_mask.dtype.is_floating_point:
        attention_mask = attention_mask.float()
    out, weights = sdpa_attention_forward(module, query.float(), key.float(), value.float(), attention_mask, **kwargs)
    return out.to(in_dtype), (weights.to(in_dtype) if torch.is_tensor(weights) else weights)


def enable_gemma4_sdpa_fp32() -> None:
    """Install :func:`sdpa_fp32_attention_forward` as the process-wide ``"sdpa"`` implementation."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    # Newer transformers exposes an ``AttentionInterface`` with ``register()`` and
    # internal ``_global_mapping``/``_local_mapping`` dicts; older versions are a
    # plain ``dict`` that supports item assignment. Update every surface the
    # attention lookup may read from so the fix works across versions.
    try:
        ALL_ATTENTION_FUNCTIONS.register("sdpa", sdpa_fp32_attention_forward)
    except Exception:
        pass
    try:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_fp32_attention_forward
    except Exception:
        pass
    for attr in ("_global_mapping", "_local_mapping"):
        mapping = getattr(ALL_ATTENTION_FUNCTIONS, attr, None)
        if isinstance(mapping, dict) and "sdpa" in mapping:
            mapping["sdpa"] = sdpa_fp32_attention_forward
