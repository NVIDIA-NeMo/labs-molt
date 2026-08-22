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

"""Value model (critic) for PPO.

Shares the actor's entire construction and forward machinery via ``BaseModel``
(FSDP2 / TP / EP / CP, packing, VLM prep, dtype) and differs only in the final
projection: the vocabulary head is replaced by a scalar value head, so the model
emits one V(s) per token instead of logits. AutoModel custom MoE forwards return a
raw ``head(hidden)`` tensor and do not surface hidden states, so making the head
one-wide is what turns that tensor into the per-token value directly.
"""

import torch
import torch.nn as nn
from nemo_automodel import PreFSDPHookResult

from molt.trainer.fsdp.packing import unshard_dtensor

from .base import BaseModel


class _ValueHead(nn.Linear):
    """Scalar value projection over the backbone's last hidden state.

    Replaces the vocab ``lm_head`` so the model's "logits" are per-token values.
    Under TP the hidden state arrives as a DTensor on the TP mesh, so we materialize
    its full (un-TP-sharded) view via ``unshard_dtensor`` before the plain head:

    - For the common ``ColwiseParallel`` head (e.g. HF Qwen3 ``colwise_gather_output``)
      the head input is *replicated*, so this is a no-op gather.
    - For a sequence-/hidden-sharded input (SequenceParallel) it all-gathers, so the
      head still sees the full hidden_size and computes correct values — a bare
      ``to_local()`` would have silently used only this rank's shard.

    This mirrors the policy loss callback's ``unshard_dtensor(logits)`` and
    collapses only the TP dimension; Engine owns CP output restoration.
    AutoModel installs this head before FSDP, so its parameters participate in the
    same reduction, clipping, optimizer, and checkpoint lifecycle as the backbone.
    ``unshard_dtensor`` is a no-op at TP=1 (input already plain).
    """

    def __init__(self, hidden_size: int, initializer_range: float = 0.02, device=None):
        self.initializer_range = initializer_range
        # The head stays fp32; its scalar output makes the extra compute negligible.
        super().__init__(hidden_size, 1, bias=False, device=device, dtype=torch.float32)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self.initializer_range)

    def forward(self, hidden_states):
        # Gather any TP/SP sharding (no-op for a replicated DTensor or a plain tensor).
        hidden_states = unshard_dtensor(hidden_states)
        # No forward autocast anymore, so align the bf16 backbone hidden to the
        # fp32 value-head weight explicitly instead of relying on autocast.
        return super().forward(hidden_states.to(self.weight.dtype))


def _resolve_hidden_size(model) -> int:
    """Hidden size for the value head, read off the built model rather than its config —
    config-independent, so it sidesteps where (and how) VLMs nest the language-model dims
    (``text_config``, ``llm_config`` for Nemotron-Omni, dict vs object; several of our
    models expose no top-level ``hidden_size`` at all).

    Primary source is the ``lm_head`` we're about to replace: its ``in_features`` is
    exactly the post-norm hidden the value head consumes — correct even under a factorized
    input embedding (``embedding_size != hidden_size``). Fall back to the token-embedding
    dim (== hidden for every decoder we run) for the rare model exposing no output head."""
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if head is not None:
        dim = getattr(head, "in_features", None) or head.weight.shape[-1]
        if dim:
            return int(dim)
    emb = model.get_input_embeddings()
    return int(getattr(emb, "embedding_dim", None) or emb.weight.shape[-1])


def _install_value_head(model) -> PreFSDPHookResult:
    """Replace ``model``'s task head in place before AutoModel applies FSDP."""

    old_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else model.lm_head
    source_weight = getattr(old_head, "weight", None)
    if source_weight is None:
        source_weight = model.get_input_embeddings().weight
    value_head = _ValueHead(
        _resolve_hidden_size(model),
        initializer_range=getattr(model.config, "initializer_range", 0.02),
        device=source_weight.device,
    )
    # The value head is NOT tied to the vocabulary embeddings. Keeping this flag
    # set would make checkpoint export deduplicate or later re-tie incompatible
    # [1,H] and [vocab,H] tensors.
    for cfg in (model.config, getattr(model.config, "text_config", None)):
        if cfg is not None:
            cfg.tie_word_embeddings = False
    if hasattr(model, "set_output_embeddings"):
        model.set_output_embeddings(value_head)
    else:
        model.lm_head = value_head
    return PreFSDPHookResult(task_module=value_head)


class Critic(BaseModel):
    """``BaseModel`` with the vocab head swapped for a scalar value head.

    For checkpoint-backed construction the vocabulary projection is replaced in
    AutoModel's ``pre_fsdp_hook``. The scalar head is consequently part of the model
    before FSDP discovers parameters, so normal FSDP gradient reduction, clipping,
    checkpointing, and optimizer handling all include it.
    """

    def __init__(self, *args, **kwargs):
        pretrain_or_model = args[0] if args else kwargs.get("pretrain_or_model")
        if isinstance(pretrain_or_model, str):
            if kwargs.get("pre_fsdp_hook") is not None:
                raise TypeError("Critic owns pre_fsdp_hook; callers must not override it")
            kwargs["pre_fsdp_hook"] = _install_value_head
            super().__init__(*args, **kwargs)
        else:
            # Preserve the lightweight pre-instantiated-model path used by unit
            # tests and inference utilities. Such a model has not been built by
            # this wrapper, so installing the head here does not move a parameter
            # across an existing AutoModel FSDP boundary.
            super().__init__(*args, **kwargs)
            _install_value_head(self.model)
