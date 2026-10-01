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
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from molt.trainer.fsdp.packing import log_probs_from_vocab_parallel_logits, unshard_dtensor

from .base import BaseModel, _AttrDict
from .utils import compute_entropy, log_probs_from_logits


class Actor(BaseModel):
    """Policy model wrapper for RLHF.

    Reuses ``BaseModel`` for construction and the shared ``_forward_backbone``
    (packing / VLM / CP prep + model call); ``forward`` turns the model logits into
    per-token log-probs, optionally entropy, and the action-span log-probs.
    """

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        cp_context_stack=None,
        return_entropy=False,
        routed_experts: Optional[torch.Tensor] = None,
        top_ids: Optional[torch.Tensor] = None,
        **mm_inputs,
    ) -> _AttrDict:
        """Run the policy forward and return one named output dict.

        Always returns an ``_AttrDict`` (key- and attribute-accessible) with:

        - ``logits``:           model logits (TP-sharded DTensor unless gathered).
        - ``log_probs``:        ``[B, S-1]`` log-probs of the realized next tokens.
        - ``action_log_probs``: ``[B, num_actions]`` masked to the generated span —
                                only when ``action_mask`` is given (RL / reference).
        - ``entropy``:          ``[B, S-1]`` — only when ``return_entropy`` (RL).
        - ``action_top_log_probs``: ``[B, num_actions, k]`` log-probs of the sampler's top-k
                                candidate ids ``top_ids`` (``[B, k, S-1]``, seq last) — only when
                                given (score centering).
        - ``aux_loss``:         MoE load-balancing loss — only for NeMo custom MoE.
        """
        output, rolled_sequences, cp_forward, indices, batch, seqlen = self._forward_backbone(
            sequences, attention_mask, position_ids, cp_context_stack, mm_inputs, routed_experts=routed_experts
        )
        logits = output["logits"]
        full_logits = None

        if return_entropy:
            entropy_logits = unshard_dtensor(logits).to(torch.float32)
            if not cp_forward:
                full_logits = entropy_logits
                output["logits"] = full_logits
            # Compute entropy on the same temperature-scaled distribution as the
            # policy log-probs below (which divide by self.temperature). Without
            # this the entropy-regularization term operates on a different (T=1)
            # distribution than the policy whenever rollout.temperature != 1.0.
            # Standard practice scales logits before both log-probs *and*
            # entropy. Divide a copy so output["logits"]/full_logits stay raw
            # (log_probs_from_logits applies its own temperature division).
            entropy_src = entropy_logits / self.temperature if self.temperature != 1.0 else entropy_logits
            entropy = compute_entropy(entropy_src)
            # entropy is seq-local even under TP+CP (unshard_dtensor only
            # collapses the TP vocab dim), so restore the full sequence axis the
            # same way as log_probs below.
            entropy = self._restore_full_sequence(
                entropy, cp_forward=cp_forward, batch=batch, seqlen=seqlen, indices=indices
            )
            output["entropy"] = entropy[:, :-1]

        top_log_probs = None
        if top_ids is not None and (cp_forward or isinstance(logits, DTensor)):
            raise NotImplementedError("top-k log-probs (score centering) require fsdp.cp_size == tp_size == 1")
        if isinstance(logits, DTensor):
            log_probs = log_probs_from_vocab_parallel_logits(
                logits,
                rolled_sequences,
                temperature=self.temperature,
            )
        else:
            # Pass bf16 directly: log_probs_from_logits chunks the fp32 upcast
            # internally to avoid the [B*S, V] memory spike that OOMs on
            # large-vocab models (Qwen3.6: 152K vocab × 65K tokens = 37 GiB).
            log_probs_input = logits if (cp_forward or full_logits is None) else full_logits
            if top_ids is None:
                log_probs = log_probs_from_logits(log_probs_input, rolled_sequences, temperature=self.temperature)
            else:
                # Score centering: the sampler's top-k ids ride along as extra targets of the SAME
                # pass (one fp32 upcast + logsumexp per chunk; a second call would double that cost).
                # (B, S, k) aligned with rolled_sequences: row t holds the candidates for token t+1.
                ids = F.pad(top_ids.permute(0, 2, 1), (0, 0, 0, 1))
                if indices is not None:  # packed forward: same real-token order (and pad) as the labels
                    ids = ids.reshape(-1, ids.shape[-1]).index_select(0, indices).unsqueeze(0)
                    ids = F.pad(ids, (0, 0, 0, rolled_sequences.shape[1] - ids.shape[1]))
                targets = torch.cat([rolled_sequences.unsqueeze(-1), ids], dim=-1)
                all_log_probs = log_probs_from_logits(log_probs_input, targets, temperature=self.temperature)
                log_probs, top_log_probs = all_log_probs[..., 0], all_log_probs[..., 1:]

        log_probs = self._restore_full_sequence(
            log_probs, cp_forward=cp_forward, batch=batch, seqlen=seqlen, indices=indices
        )

        # Drop the final column: logits[t] predicts token[t+1], so the last
        # position has no target. log_probs / action_log_probs / entropy are all
        # shifted [:, :-1] and stay mutually aligned.
        output["log_probs"] = log_probs[:, :-1]

        # RL / reference (action_mask given) additionally expose the action-span
        # log-probs, zeroed outside the generated tokens.
        if action_mask is not None:
            output["action_log_probs"] = output["log_probs"][:, -action_mask.shape[1] :] * action_mask.float()
            if top_log_probs is not None:
                top_log_probs = self._restore_full_sequence(
                    top_log_probs, cp_forward=cp_forward, batch=batch, seqlen=seqlen, indices=indices
                )
                output["action_top_log_probs"] = top_log_probs[:, :-1][:, -action_mask.shape[1] :]

        return output
