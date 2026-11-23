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

"""The trainer's half of a vLLM weight sync: which tensors go to the engines, and under which names.

``ActorWeights`` is the ``WeightSource`` molt hands to vLLM's weight-transfer engine (vLLM docs:
training/weight_updates). It walks the actor's state dict, gathers each FSDP2/TP/EP-sharded entry to
its full tensor and names it as in the checkpoint. vLLM's ``load_weights`` matches by name and ignores
what it does not have, so which weights to accept is decided on the vLLM side.
"""

from functools import cached_property
from typing import Iterator, List, Tuple

import torch
from torch.distributed.tensor import DTensor
from vllm.distributed.weight_transfer import ParamMeta, WeightSource


def gather_full_param(param: torch.Tensor) -> torch.Tensor:
    """The full tensor of an FSDP2/TP/EP-sharded parameter, gathered across every mesh dim in one call.

    A collective: every rank calls it in the same order and ends up holding the full tensor. A tensor
    that is not sharded is returned as is.
    """
    return param.full_tensor() if isinstance(param, DTensor) else param


def _skip_tied_lm_head(config, name: str) -> bool:
    """Whether to skip broadcasting this weight to vLLM as a redundant tied lm_head.

    Tied-embedding models share lm_head with embed_tokens; vLLM receives the shared
    weight through embed_tokens, so a separate lm_head.weight update is redundant and
    would form an empty "loaded 0 of N" refit batch. Exact-match only (``lm_head.weight``)
    so nested/auxiliary heads like ``mtp.lm_head.weight`` are still sent.
    """
    return bool(getattr(config, "tie_word_embeddings", False)) and name == "lm_head.weight"


def _lora_merge_map(model: torch.nn.Module, sd: dict) -> dict:
    """Map each LoRA base weight to ``(A key, B key, scale)`` for the vLLM refit merge.

    Keyed by state-dict names: activation checkpointing wraps layers in ``CheckpointWrapper``,
    which appears in ``named_modules()`` but is stripped from ``state_dict()`` keys, so a
    module-name map would never match and the adapters would never reach vLLM.
    """
    from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

    # scale comes from the patched module itself (LinearLoRA.scale = alpha / dim).
    scales = {
        canonical_parameter_fqn(name): float(mod.scale)
        for name, mod in model.named_modules()
        if getattr(mod, "lora_A", None) is not None and getattr(mod, "lora_B", None) is not None
    }
    merge = {}
    for name in sd:
        if not name.endswith(".lora_A.weight"):
            continue
        base = name[: -len(".lora_A.weight")]
        b_name, w_name = f"{base}.lora_B.weight", f"{base}.weight"
        if base not in scales or b_name not in sd or w_name not in sd:
            raise RuntimeError(
                f"Refit: LoRA adapter {name!r} has no matching module or base/B weight; its delta "
                "would be dropped and vLLM would keep the frozen base. Refusing to diverge silently."
            )
        merge[w_name] = (name, b_name, scales[base])
    return merge


# State-dict entries that are not weights of their own: TE's fp8 bookkeeping, and LoRA adapters,
# which reach vLLM merged into their base weight.
_NOT_WEIGHTS = ("_extra_state", ".lora_A.weight", ".lora_B.weight")


class ActorWeights(WeightSource):
    """The actor's weights as a weight sync sends them: full tensors under checkpoint names.

    Every state-dict entry is sent. A ``requires_grad`` filter is deliberately absent: its names differ
    from the state dict's (AutoModel / FSDP naming), and such a filter once skipped nearly every trained
    weight and left the engines on the base checkpoint. Every tensor travels in its own dtype: vLLM
    casts on load, so a weight kept in fp32 (an MoE router) is not rounded to bf16 on the way.
    """

    def __init__(self, model: torch.nn.Module, ep_size: int = 1):
        self.model = model
        self.ep_size = ep_size

    @cached_property
    def metadata(self) -> List[ParamMeta]:
        # Read off one pass over the weights, so it cannot disagree with what iter_full sends: vLLM
        # sizes its receive buffers from this list, and a disagreement hangs the transfer.
        return [ParamMeta(name, weight.dtype, tuple(weight.shape)) for name, weight in self.iter_full()]

    def iter_full(self) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield ``(checkpoint name, full tensor)``. Gathering a sharded weight is a collective, so every
        trainer rank iterates this in step."""
        state = self.model.state_dict()
        lora = _lora_merge_map(self.model, state)
        adapter = getattr(self.model, "state_dict_adapter", None)
        config = getattr(self.model, "config", None)
        for name, tensor in state.items():
            if not torch.is_tensor(tensor) or name.endswith(_NOT_WEIGHTS):
                continue
            # An expert that is not a DTensor holds this rank's experts only (BackendConfig
            # experts="te"): gathering it would silently send rank 0's slice.
            if self.ep_size > 1 and "expert" in name and not isinstance(tensor, DTensor):
                raise RuntimeError(
                    f"Refit: expert weight {name!r} is not a DTensor under ep_size={self.ep_size}; only "
                    "rank 0's experts would reach vLLM. Build the MoE with BackendConfig "
                    "experts='torch_mm' (the default)."
                )
            weight = gather_full_param(tensor)
            if name in lora:
                a_name, b_name, scale = lora[name]
                # W + scale·B@A: one rounding away from the trainer's W·x + scale·B(A·x).
                delta = gather_full_param(state[b_name]).float() @ gather_full_param(state[a_name]).float()
                weight = weight + scale * delta.to(weight.dtype)
            # AutoModel's adapter owns the checkpoint naming and may return a grouped weight as several
            # tensors. No options: some adapters (Inkling 2, Mistral Large 4) reject those they do not know.
            if adapter is None:
                named = [(name, weight)]
            elif hasattr(adapter, "convert_single_tensor_to_hf"):
                named = adapter.convert_single_tensor_to_hf(name, weight)
            else:  # an adapter with the whole-dict API only (KimiVL)
                named = adapter.to_hf({name: weight}).items()
            for hf_name, hf_weight in named:
                # A tied lm_head reaches vLLM through embed_tokens.
                if hf_weight.is_floating_point() and not _skip_tied_lm_head(config, hf_name):
                    yield hf_name, hf_weight
