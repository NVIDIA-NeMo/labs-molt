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
import torch.distributed as dist
import torch.nn as nn

from .base import BaseModel
from .utils import unshard_dtensor


class _ValueHead(nn.Module):
    """Scalar value projection over the backbone's last hidden state.

    Replaces the vocab ``lm_head`` so the model's "logits" are per-token values.
    Under TP/SP the hidden state can arrive as a sharded DTensor, so it is
    materialized via ``unshard_dtensor`` first (a bare ``to_local()`` would
    silently use only this rank's shard; at TP=1 this is a no-op). The head
    stays a plain replicated module added after the FSDP wrap, so its weight
    gradient is a plain tensor the critic trainer DP-all-reduces.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # fp32 to match the fp32 master-weight convention; the head is tiny.
        self.proj = nn.Linear(hidden_size, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden_states):
        hidden_states = unshard_dtensor(hidden_states)
        return self.proj(hidden_states.to(self.proj.weight.dtype))


class Critic(BaseModel):
    """``BaseModel`` with the vocab head swapped for a scalar value head.

    The value head is a plain (replicated) module added *after* the FSDP wrap.
    Every rank initializes it and rank 0's weights are broadcast so the replicas
    are identical; FSDP does not cover its gradient, so the critic trainer
    DP-all-reduces it each optimizer step (``value_head_parameters`` +
    ``FsdpStrategy.sync_replicated_grads``). Under TP/EP the head consumes the
    replicated post-norm hidden state, so only the DP(+CP) reduction matters.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Hidden size comes off the built model, not its config (VLMs nest the
        # language-model dims in arch-specific places): the lm_head being
        # replaced gives the exact post-norm hidden the value head consumes.
        old_head = self.model.get_output_embeddings() if hasattr(self.model, "get_output_embeddings") else None
        if old_head is not None and getattr(old_head, "in_features", None):
            hidden_size = int(old_head.in_features)
        elif getattr(old_head, "weight", None) is not None:
            hidden_size = int(old_head.weight.shape[-1])
        else:
            emb = self.model.get_input_embeddings()
            hidden_size = int(getattr(emb, "embedding_dim", None) or emb.weight.shape[-1])

        value_head = _ValueHead(hidden_size)
        # HF's standard fresh-head init, so |V| ~ O(1) from step 1.
        nn.init.normal_(value_head.proj.weight, mean=0.0, std=getattr(self.model.config, "initializer_range", 0.02))
        device = next((p.device for p in self.model.parameters() if p.device.type != "meta"), None)
        if device is not None:
            value_head = value_head.to(device)
        # Make every rank's replica identical (rank 0 wins); a head defined inside
        # the model would get this for free from the pre-FSDP seeded init.
        if dist.is_initialized() and value_head.proj.weight.is_cuda:
            dist.broadcast(value_head.proj.weight.data, src=0)
        # The value head is NOT a tied vocab head: clear tie_word_embeddings so the
        # checkpointer saves it as its own tensor and nothing later re-ties the
        # [1, hidden] head to the [vocab, hidden] embeddings.
        for cfg in (self.model.config, getattr(self.model.config, "text_config", None)):
            if cfg is not None and getattr(cfg, "tie_word_embeddings", False):
                cfg.tie_word_embeddings = False
        if hasattr(self.model, "set_output_embeddings"):
            self.model.set_output_embeddings(value_head)
        else:
            self.model.lm_head = value_head
        self.value_head = value_head

    def value_head_parameters(self):
        """Params the critic trainer must DP-all-reduce (FSDP does not cover them)."""
        return list(self.value_head.parameters())

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
