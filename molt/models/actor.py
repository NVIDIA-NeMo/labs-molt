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
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

from typing import Optional

import torch
from nemo_automodel.components.loss import token_entropy, token_log_probs

from .base import BaseModel


class Actor(BaseModel):
    """Policy model with the OpenRLHF-style ``actor(tokens)`` interface."""

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cp_context_stack=None,
        return_entropy: bool = False,
        routed_experts: Optional[torch.Tensor] = None,
        mm_train_inputs=None,
        **mm_inputs,
    ) -> dict[str, torch.Tensor]:
        """Score every next token and, when requested, the RL action span.

        Args:
            sequences: Padded token IDs with shape ``[batch, sequence]``.
            action_mask: Optional mask with shape ``[batch, actions]``. When
                present, ``action_log_probs`` contains the final ``actions``
                positions and is zero outside this mask.
            attention_mask: Valid-token mask matching ``sequences``.
            return_entropy: Also return exact per-token entropy.
            routed_experts: Optional rollout routes with shape
                ``[batch, global_layers, topk, sequence]``.
            mm_train_inputs: One processor result per VLM sample.

        Returns:
            Dict with dense ``log_probs`` of shape ``[batch, sequence - 1]``,
            optional ``entropy`` of the same shape, and optional
            ``action_log_probs`` matching ``action_mask``.
        """
        if mm_train_inputs is not None:
            if mm_inputs:
                raise ValueError("pass either mm_train_inputs or expanded media tensors, not both")
            mm_inputs = mm_train_inputs
        logits, targets, cp_forward, indices, batch, seqlen = self._forward_backbone(
            sequences,
            attention_mask,
            position_ids,
            cp_context_stack,
            mm_inputs,
            routed_experts=routed_experts,
        )
        log_probs = token_log_probs(logits, targets, temperature=self.temperature)
        log_probs = self._restore_full_sequence(
            log_probs, cp_forward=cp_forward, batch=batch, seqlen=seqlen, indices=indices
        )
        result = {"log_probs": log_probs[:, :-1]}

        if return_entropy:
            entropy = token_entropy(logits, temperature=self.temperature)
            entropy = self._restore_full_sequence(
                entropy, cp_forward=cp_forward, batch=batch, seqlen=seqlen, indices=indices
            )
            result["entropy"] = entropy[:, :-1]

        if action_mask is not None:
            result["action_log_probs"] = result["log_probs"][:, -action_mask.shape[1] :] * action_mask.float()
        return result
