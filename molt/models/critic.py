# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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

from molt.trainer.fsdp.packing import unshard_dtensor

from .base import BaseModel, _AttrDict


class _ValueHead(nn.Module):
    """Scalar value projection over the backbone's last hidden state.

    Replaces the vocab ``lm_head`` so the model's "logits" are per-token values.
    Under TP the hidden state arrives as a DTensor on the TP mesh, so we materialize
    its full (un-TP-sharded) view via ``unshard_dtensor`` before the plain head:

    - For the common ``ColwiseParallel`` head (e.g. HF Qwen3 ``colwise_gather_output``)
      the head input is *replicated*, so this is a no-op gather.
    - For a sequence-/hidden-sharded input (SequenceParallel) it all-gathers, so the
      head still sees the full hidden_size and computes correct values — a bare
      ``to_local()`` would have silently used only this rank's shard.

    This mirrors the actor's ``unshard_dtensor(logits)`` (it collapses only the TP
    dim; CP sequence sharding is restored later by ``_restore_full_sequence``). The
    head stays a plain replicated module, so its weight gradient is a plain tensor the
    critic trainer DP-all-reduces — not a DTensor grad on a plain param the
    optimizer / grad-clip would mishandle. No-op at TP=1 (input already plain).
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # fp32 to match the fp32 master-weight convention; bf16 compute comes from
        # the same forward autocast the backbone uses.
        self.proj = nn.Linear(hidden_size, 1, bias=False, dtype=torch.float32)

    def forward(self, hidden_states):
        # Gather any TP/SP sharding (no-op for a replicated DTensor or a plain tensor).
        hidden_states = unshard_dtensor(hidden_states)
        return self.proj(hidden_states)


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


class Critic(BaseModel):
    """``BaseModel`` with the vocab head swapped for a scalar value head.

    ``__init__`` builds the model exactly like ``Actor`` (same parallelism), then
    replaces the vocab projection with ``Linear(hidden, 1)``. ``forward`` reuses the
    shared ``_forward_backbone`` and reads the one-wide head output as V(s).

    The value head is a plain (replicated) module added *after* ``from_pretrained``'s
    FSDP wrap — a head defined inside the model would be initialized and sharded
    uniformly, but ours is initialized independently on each rank. We therefore
    broadcast rank 0's weights to all ranks so the replica is identical, and the
    critic trainer all-reduces its gradient across the DP group each step (see
    ``value_head_parameters`` + ``FsdpStrategy.sync_replicated_grads``). Under TP/EP
    the head sees the (replicated) post-norm hidden state and ``_ValueHead`` localizes
    it, so the head stays plain and its grad needs no TP/EP reduction (identical on
    those ranks); only the DP(+CP) reduction matters and is done by the trainer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        hidden = _resolve_hidden_size(self.model)
        value_head = _ValueHead(hidden)
        # Standard PPO value-head init: std = 1 / (hidden + 1). A fresh-HF-head std
        # (initializer_range ~0.02) is ~54x too large here (hidden=2688: 0.02 vs
        # 1/2689=3.7e-4) and makes initial |V(s)| ~ O(1), so step-1 value_loss starts
        # ~2 and the critic grad ~67; the small std keeps initial V(s)~0 so
        # value_loss/grad start near zero and the critic learns from a clean baseline.
        init_std = 1.0 / (hidden + 1)
        nn.init.normal_(value_head.proj.weight, mean=0.0, std=init_std)
        device = next((p.device for p in self.model.parameters() if p.device.type != "meta"), None)
        if device is not None:
            value_head = value_head.to(device)
        # Unified init: make every rank's replica identical (rank 0 wins). A head
        # defined inside the model would get this for free (init before FSDP wrap).
        if dist.is_initialized() and value_head.proj.weight.is_cuda:
            dist.broadcast(value_head.proj.weight.data, src=0)
        # The value head is NOT a tied vocab head. Clear tie_word_embeddings so the
        # checkpointer saves lm_head as its own tensor (a tied head is deduplicated
        # against the embeddings and would be dropped from the critic checkpoint),
        # and so nothing later re-ties the [1, hidden] head to the [vocab, hidden]
        # embeddings. Set it on the text sub-config too for nested VLM models.
        for cfg in (self.model.config, getattr(self.model.config, "text_config", None)):
            if cfg is not None and getattr(cfg, "tie_word_embeddings", False):
                cfg.tie_word_embeddings = False
        # set_output_embeddings routes to the real head even for nested VLM models
        # (e.g. language_model.lm_head); fall back to a direct attribute otherwise.
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
        **mm_inputs,
    ) -> _AttrDict:
        """Return per-token values.

        - ``values``:        ``[B, S-1]`` V(s) on the dense next-token step axis.
        - ``action_values``: ``[B, num_actions]`` masked to the generated span,
                             only when ``action_mask`` is given.
        """
        output, _rolled, cp_forward, indices, batch, seqlen = self._forward_backbone(
            sequences, attention_mask, position_ids, cp_context_stack, mm_inputs
        )
        # Head is one-wide, so the model's "logits" are per-token values [B, S, 1].
        values = unshard_dtensor(output["logits"]).squeeze(-1).float()
        values = self._restore_full_sequence(
            values, cp_forward=cp_forward, batch=batch, seqlen=seqlen, indices=indices
        )
        # logits[t] scores state s_t / predicts t+1; drop the final column to align
        # with action_log_probs (both live on the [:, :-1] next-token axis).
        values = values[:, :-1]
        out = _AttrDict(values=values)
        if action_mask is not None:
            out["action_values"] = values[:, -action_mask.shape[1] :] * action_mask.float()
        return out
