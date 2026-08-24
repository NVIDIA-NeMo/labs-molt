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

from typing import Optional

import torch
import torch.nn as nn

from .base import BaseModel
from .utils import unshard_dtensor


class _ValueHead(nn.Linear):
    """Scalar value projection over the backbone's last hidden state.

    Replaces the vocab ``lm_head`` so the model's "logits" are per-token values.
    Under TP/SP the hidden state can arrive as a sharded DTensor, so it is
    materialized via ``unshard_dtensor`` first (a bare ``to_local()`` would
    silently use only this rank's shard; at TP=1 this is a no-op). Installed
    before FSDP, so its parameters share the backbone's reduction, clipping,
    optimizer, and checkpoint lifecycle.
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
    """Hidden size for the value head, read off the built model, not its config
    (VLMs nest the language-model dims in arch-specific places). The ``lm_head``
    being replaced gives the exact post-norm hidden the head consumes; fall back
    to the token-embedding dim for models exposing no output head."""
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if head is not None:
        dim = getattr(head, "in_features", None) or head.weight.shape[-1]
        if dim:
            return int(dim)
    emb = model.get_input_embeddings()
    return int(getattr(emb, "embedding_dim", None) or emb.weight.shape[-1])


def _install_value_head(model) -> nn.Module:
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
    return value_head


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
            # Pre-instantiated models (tests, inference utilities) are not FSDP
            # wrapped by us, so the head can be installed after construction.
            super().__init__(*args, **kwargs)
            _install_value_head(self.model)

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cp_context_stack=None,
        routed_experts: Optional[torch.Tensor] = None,
        mm_train_inputs=None,
        **mm_inputs,
    ) -> dict[str, torch.Tensor]:
        """Predict dense token values and, when requested, the RL action span.

        Args:
            sequences: Padded token IDs with shape ``[batch, sequence]``.
            action_mask: Optional mask with shape ``[batch, actions]``.
            attention_mask: Valid-token mask matching ``sequences``.
            routed_experts: Optional rollout routes with shape
                ``[batch, global_layers, topk, sequence]``.
            mm_train_inputs: One processor result per VLM sample.

        Returns:
            ``token_values`` with shape ``[batch, sequence - 1]`` and, when an
            action mask is supplied, ``action_values`` matching that mask.
        """
        if mm_train_inputs is not None:
            if mm_inputs:
                raise ValueError("pass either mm_train_inputs or expanded media tensors, not both")
            mm_inputs = mm_train_inputs
        logits, _targets, cp_forward, indices, batch, seqlen = self._forward_backbone(
            sequences,
            attention_mask,
            position_ids,
            cp_context_stack,
            mm_inputs,
            routed_experts=routed_experts,
        )
        values = unshard_dtensor(logits).squeeze(-1).float()
        values = self._restore_full_sequence(
            values, cp_forward=cp_forward, batch=batch, seqlen=seqlen, indices=indices
        )[:, :-1]
        result = {"token_values": values}
        if action_mask is not None:
            result["action_values"] = values[:, -action_mask.shape[1] :] * action_mask.float()
        return result
