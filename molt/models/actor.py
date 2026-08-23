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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from nemo_automodel.components.datasets.datum import Datum
from nemo_automodel.components.loss import token_entropy, token_log_probs

from .base import BaseModel

if TYPE_CHECKING:
    from molt.trainer.algorithm.experience import Experience


class Actor(BaseModel):
    """Policy wrapper that owns model construction and rollout temperature.

    AutoModel ``Engine`` owns all training and collection-time forwards.
    """

    def make_policy_datums(
        self,
        experience: "Experience",
        *,
        old_action_log_probs: torch.Tensor | None,
        advantages: torch.Tensor,
        base_action_log_probs: torch.Tensor | None,
        rollout_log_probs: torch.Tensor | None,
        include_sequence_ids: bool = False,
        routed_experts: torch.Tensor | None = None,
    ) -> list[Datum]:
        """Build one policy microbatch with its PPO token inputs."""
        side_inputs = {
            "weights": experience.action_mask.float(),
            "advantages": advantages,
        }
        if old_action_log_probs is not None:
            side_inputs["old_action_log_probs"] = old_action_log_probs
        if base_action_log_probs is not None:
            side_inputs["base_action_log_probs"] = base_action_log_probs
        if rollout_log_probs is not None:
            side_inputs["rollout_log_probs"] = rollout_log_probs
        return self._make_datums(
            experience,
            side_inputs=side_inputs,
            include_sequence_ids=include_sequence_ids,
            routed_experts=routed_experts,
        )

    def compute_action_log_probs(self, output: Any, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Compute the realized next-token log probabilities for one model batch."""
        if torch.is_tensor(output):
            logits = output
        elif isinstance(output, Mapping):
            logits = output["logits"]
        else:
            logits = output.logits
        return token_log_probs(logits, inputs["target_tokens"], temperature=self.temperature)

    def compute_entropy(self, output: Any, _inputs: Mapping[str, torch.Tensor] | None = None) -> torch.Tensor:
        """Compute one entropy value for each token in a model batch."""
        if torch.is_tensor(output):
            logits = output
        elif isinstance(output, Mapping):
            logits = output["logits"]
        else:
            logits = output.logits
        return token_entropy(logits, temperature=self.temperature)
