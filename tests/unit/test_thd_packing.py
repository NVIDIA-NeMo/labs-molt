# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collate-time THD packing parity with the forward-time packer it replaces."""

import pytest
import torch

from molt.trainer.algorithm.experience import Experience, make_experience_batch


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


def test_packs_tokens_into_one_flat_row():
    packed = make_experience_batch(_loose_items(), packed=True)

    assert packed.sequences.tolist() == [[10, 11, 12, 20, 21]]
    # Positions restart per sequence -- what makes varlen attention split the pack.
    assert packed.position_ids.tolist() == [[0, 1, 2, 0, 1]]
    assert packed.packed_seq_lens == [3, 2]


def test_side_inputs_ride_the_flat_token_axis():
    packed = make_experience_batch(_loose_items(), packed=True)

    # Action-side fields are [T-1] and gain one trailing masked slot per sample,
    # so they stay elementwise-aligned with the tokens.
    assert packed.advantages.shape == packed.sequences.shape
    assert packed.advantages[0].tolist() == pytest.approx([0.5, 0.5, 0.0, 0.9, 0.0])
    assert packed.action_log_probs[0].tolist() == pytest.approx([-0.1, -0.2, 0.0, -0.3, 0.0])
    assert packed.action_mask[0].tolist() == [True, True, False, True, False]


def test_token_mean_loss_is_layout_invariant():
    """A masked sum over the flat axis equals the padded one — why packing is free."""
    packed = make_experience_batch(_loose_items(), packed=True)
    padded = make_experience_batch(_loose_items())

    flat_sum = (packed.advantages * packed.action_mask).sum()
    padded_sum = (padded.advantages * padded.action_mask).sum()

    assert torch.allclose(flat_sum, padded_sum)


def test_routed_experts_pack_on_the_sequence_axis():
    """R3 routing ids are [layers, topk, T] with the sequence last, so they concatenate."""
    items = _loose_items()
    items[0].routed_experts = torch.zeros(2, 1, 3, dtype=torch.long)
    items[1].routed_experts = torch.ones(2, 1, 2, dtype=torch.long)

    packed = make_experience_batch(items, packed=True)

    assert packed.routed_experts.shape == (1, 2, 1, 5)
    assert packed.routed_experts[0, 0, 0].tolist() == [0, 0, 0, 1, 1]
