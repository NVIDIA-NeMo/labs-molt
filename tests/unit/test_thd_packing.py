# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collate-time THD packing parity with the forward-time packer it replaces."""

import pytest
import torch

from molt.trainer.algorithm.experience import Experience, make_experience_batch, make_packed_experience_batch
from molt.trainer.fsdp.packing import pack_padded_batch


def _loose_items():
    """Two per-sample records, as NaiveReplayBuffer stores them (padding removed)."""
    return [
        Experience(
            sequences=torch.tensor([10, 11, 12]),
            attention_mask=torch.tensor([1, 1, 1]),
            action_mask=torch.tensor([True, True]),
            advantages=torch.tensor([0.5, 0.5]),
            action_log_probs=torch.tensor([-0.1, -0.2]),
        ),
        Experience(
            sequences=torch.tensor([20, 21]),
            attention_mask=torch.tensor([1, 1]),
            action_mask=torch.tensor([True]),
            advantages=torch.tensor([0.9]),
            action_log_probs=torch.tensor([-0.3]),
        ),
    ]


def test_packs_tokens_like_the_forward_packer():
    packed = make_packed_experience_batch(_loose_items())

    # The path it replaces: rebuild the padded batch, then pack inside the forward.
    padded = make_experience_batch(_loose_items())
    ref_ids, ref_pos, ref_rolled, _, ref_kwargs = pack_padded_batch(
        padded.sequences, padded.attention_mask, style="automodel"
    )

    assert torch.equal(packed["input_ids"], ref_ids)
    assert torch.equal(packed["position_ids"], ref_pos)
    assert torch.equal(packed["cu_seqlens"], ref_kwargs["cu_seqlens"])
    assert packed["max_seqlen"] == ref_kwargs["max_seqlen"]
    assert packed["packed_seq_lens"] == [3, 2]
    # Per-sample and padded-row rolls agree wherever a target exists; each
    # sample's final position has none and is masked out.
    scored = packed["action_mask"][0].bool()
    assert scored.tolist() == [True, True, False, True, False]
    assert torch.equal(packed["labels"][0][scored], ref_rolled[0][scored])


def test_side_inputs_ride_the_flat_token_axis():
    packed = make_packed_experience_batch(_loose_items())

    assert packed["advantages"].shape == packed["input_ids"].shape
    assert packed["advantages"][0].tolist() == pytest.approx([0.5, 0.5, 0.0, 0.9, 0.0])
    assert packed["action_log_probs"][0].tolist() == pytest.approx([-0.1, -0.2, 0.0, -0.3, 0.0])


def test_token_mean_loss_is_layout_invariant():
    """A masked sum over the flat axis equals the padded one — why packing is free."""
    packed = make_packed_experience_batch(_loose_items())
    padded = make_experience_batch(_loose_items())

    flat_sum = (packed["advantages"] * packed["action_mask"]).sum()
    padded_sum = (padded.advantages * padded.action_mask).sum()

    assert torch.allclose(flat_sum, padded_sum)


def test_multi_dim_per_token_field_is_rejected():
    items = _loose_items()
    items[0].values = torch.zeros(2, 3, 2)  # R3-style [layers, topk, T]
    items[1].values = torch.zeros(2, 3, 1)
    with pytest.raises(ValueError, match="1-D per-token fields only"):
        make_packed_experience_batch(items)
