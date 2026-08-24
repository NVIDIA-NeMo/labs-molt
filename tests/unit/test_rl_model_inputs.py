# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from nemo_automodel.engine import Engine

from molt.models.actor import Actor
from molt.models.critic import Critic, _ValueHead
from molt.models.loss import PolicyLoss
from molt.trainer.fsdp.packing import pack_padded_batch, unpack_to_padded
from molt.utils.vlm_utils import merge_mm_train_inputs


class _TinyPolicyModel(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)
        self.last_inputs = None

    def forward(self, input_ids, **kwargs):
        self.last_inputs = kwargs
        return self.output(self.embedding(input_ids))


def test_actor_returns_dense_log_probs_aligned_to_the_action_span():
    torch.manual_seed(7)
    backbone = _TinyPolicyModel()
    actor = Actor(backbone, temperature=0.7)
    sequences = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.ones_like(sequences)
    action_mask = torch.tensor([[False, True, True]])

    with torch.no_grad():
        output = actor(
            sequences,
            action_mask,
            attention_mask=attention_mask,
            return_entropy=True,
        )

    logits = backbone(sequences).float() / 0.7
    next_tokens = sequences.roll(-1, dims=1)
    expected = F.log_softmax(logits, dim=-1).gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1)[:, :-1]
    torch.testing.assert_close(output.log_probs, expected)
    torch.testing.assert_close(output.action_log_probs, expected[:, -3:] * action_mask)
    assert output.entropy.shape == output.log_probs.shape
    assert not output.action_log_probs.requires_grad


def test_packed_vlm_forward_restores_each_sample_to_dense_coordinates():
    torch.manual_seed(9)
    backbone = _TinyPolicyModel(vocab_size=128)
    actor = Actor(backbone)
    actor.is_vlm = True
    actor.packing_layout = "thd"
    actor._vlm_config = backbone.config
    actor._image_token_id = 99
    actor._video_token_id = None
    sequences = torch.tensor([[10, 99, 12, 13, 0], [20, 21, 22, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]])
    media = [
        {
            "pixel_values": torch.ones(1, 3, 2, 2),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
        },
        None,
    ]

    output = actor(sequences, attention_mask=attention_mask, mm_train_inputs=media)

    logits = backbone.output(backbone.embedding(sequences)).float()
    targets = sequences.roll(-1, dims=1)
    expected = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)[:, :-1]
    torch.testing.assert_close(output.log_probs, expected * attention_mask[:, 1:])
    assert backbone.last_inputs["qkv_format"] == "thd"
    assert torch.equal(backbone.last_inputs["seq_lens"], torch.tensor([[3, 2]], dtype=torch.int32))
    assert backbone.last_inputs["pixel_values"].shape == (1, 3, 2, 2)


def test_packed_vlm_accepts_a_text_only_batch_without_media_inputs():
    backbone = _TinyPolicyModel(vocab_size=128)
    actor = Actor(backbone)
    actor.is_vlm = True
    actor.packing_layout = "thd"
    actor._vlm_config = backbone.config
    actor._image_token_id = 99
    actor._video_token_id = None
    sequences = torch.tensor([[10, 11, 12]])

    output = actor(sequences, attention_mask=torch.ones_like(sequences), mm_train_inputs=None)

    assert output.log_probs.shape == (1, 2)
    assert "pixel_values" not in backbone.last_inputs


def test_padded_vlm_merges_ragged_pixel_lists():
    first = torch.ones(3, 2, 3)
    second = torch.ones(3, 4, 2)

    merged = merge_mm_train_inputs(
        [{"pixel_values": [first]}, {"pixel_values": [second]}],
        torch.device("cpu"),
    )

    assert isinstance(merged["pixel_values"], list)
    assert [tuple(value.shape) for value in merged["pixel_values"]] == [(3, 2, 3), (3, 4, 2)]
    assert all(value.dtype == torch.bfloat16 for value in merged["pixel_values"])


def test_padded_vlm_uses_mm_token_types_for_video_inputs():
    backbone = _TinyPolicyModel(vocab_size=128)
    actor = Actor(backbone)
    actor.is_vlm = True
    actor._vlm_config = backbone.config
    actor._image_token_id = 99
    actor._video_token_id = 98
    sequences = torch.tensor([[10, 98, 11]])
    media = [
        {
            "pixel_values_videos": torch.ones(1, 3, 2, 2),
            "video_grid_thw": torch.tensor([[1, 2, 2]]),
        }
    ]

    actor(sequences, attention_mask=torch.ones_like(sequences), mm_train_inputs=media)

    assert torch.equal(backbone.last_inputs["mm_token_type_ids"], torch.tensor([[0, 2, 0]], dtype=torch.int32))
    assert "token_type_ids" not in backbone.last_inputs


def test_padded_vlm_cp_leaves_missing_position_ids_to_the_model_sharder(monkeypatch):
    """Omni-style padded VLM hooks build positions after the logical batch is formed."""

    class _IdentitySharder:
        def __init__(self, _model, _mesh, batch, **_kwargs):
            assert "position_ids" not in batch

        def shard(self, batch):
            return nullcontext, batch

        def gather_token_tensor(self, values, **_kwargs):
            return values

    import nemo_automodel.components.distributed.context_parallel as cp

    monkeypatch.setattr(cp, "ContextParallelSharder", _IdentitySharder)
    backbone = _TinyPolicyModel(vocab_size=128)
    backbone.prepare_model_inputs_for_cp = lambda batch, **_kwargs: batch
    actor = Actor(backbone)
    actor.is_vlm = True
    actor.cp_size = 2
    actor.device_mesh = object()

    output = actor(
        torch.tensor([[10, 11, 12]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        mm_train_inputs=[None],
    )

    assert output.log_probs.shape == (1, 2)


def test_policy_loss_runs_through_engine_backward_and_accumulated_step():
    torch.manual_seed(11)
    actor = Actor(_TinyPolicyModel())
    optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.05)
    actor.model = Engine(
        actor.model,
        optimizer=optimizer,
        gradient_accumulation_steps=2,
        max_grad_norm=1.0,
    )
    policy_loss = PolicyLoss()
    before = actor.module.output.weight.detach().clone()
    microbatches = (
        (
            torch.tensor([[1, 2, 3, 4, 5]]),
            torch.tensor([[False, True, True, False]]),
            torch.tensor([[0.0, 1.0, -0.5, 0.0]]),
        ),
        (
            torch.tensor([[6, 7, 8, 9], [10, 11, 12, 13]]),
            torch.tensor([[True, True, False], [False, True, True]]),
            torch.tensor([[0.7, 1.2, 0.0], [0.0, -0.4, 0.8]]),
        ),
    )
    window_tokens = sum(action_mask.sum() for _, action_mask, _ in microbatches)

    for index, (sequences, action_mask, advantages) in enumerate(microbatches):
        output = actor(sequences, action_mask, attention_mask=torch.ones_like(sequences))
        loss, *_ = policy_loss(
            output.action_log_probs,
            output.action_log_probs.detach(),
            advantages,
            action_mask=action_mask,
            batch_num_tokens=window_tokens,
        )
        actor.model.backward(loss, scale_wrt_gas=False)
        actor.model.step()
        if index == 0:
            assert torch.equal(actor.module.output.weight, before)

    assert not torch.equal(actor.module.output.weight, before)
    assert float(actor.model.get_global_grad_norm()) > 0


@pytest.mark.parametrize("layout", ["thd", "indexed_mask"])
def test_packing_round_trips_dense_token_coordinates(layout):
    sequences = torch.tensor([[0, 1, 2, 0], [3, 4, 5, 6]])
    attention_mask = torch.tensor([[0, 1, 1, 0], [1, 1, 1, 1]])
    packed, position_ids, targets, indices, attention = pack_padded_batch(
        sequences,
        attention_mask,
        layout=layout,
        pad_to_tokens=8 if layout == "thd" else None,
    )

    assert torch.equal(packed[:, :4], torch.tensor([[1, 3, 4, 5]]))
    assert torch.equal(position_ids[:, :4], torch.tensor([[0, 0, 1, 2]]))
    assert torch.equal(targets[:, :4], torch.tensor([[2, 4, 5, 6]]))
    restored = unpack_to_padded(torch.arange(10, 10 + packed.numel()).reshape(1, -1), indices, 2, 4)
    assert torch.equal(restored, torch.tensor([[0, 10, 0, 0], [11, 12, 13, 0]]))

    if layout == "thd":
        assert attention["qkv_format"] == "thd"
        assert torch.equal(attention["cu_seqlens"], torch.tensor([0, 1, 4], dtype=torch.int32))
        assert torch.equal(attention["cu_seqlens_padded"], torch.tensor([0, 1, 8], dtype=torch.int32))
        assert torch.equal(attention["padding_mask"], torch.tensor([[False] * 4 + [True] * 4]))
    else:
        assert torch.equal(attention["attention_mask"], torch.tensor([[1, 2, 2, 2]]))


def test_thd_packing_aligns_each_document_without_mapping_synthetic_tokens():
    sequences = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    attention_mask = sequences.ne(0)
    packed, _positions, targets, physical_to_dense, attention = pack_padded_batch(
        sequences,
        attention_mask,
        layout="thd",
        sequence_alignment=4,
    )

    assert torch.equal(packed, torch.tensor([[1, 2, 0, 0, 4, 0, 0, 0]]))
    assert torch.equal(targets, torch.tensor([[2, 3, 0, 0, 5, 0, 0, 0]]))
    assert torch.equal(physical_to_dense, torch.tensor([0, 1, -1, -1, 4, -1, -1, -1]))
    assert torch.equal(attention["seq_lens"], torch.tensor([[2, 1]], dtype=torch.int32))
    assert torch.equal(attention["seq_lens_padded"], torch.tensor([[4, 4]], dtype=torch.int32))


def test_routing_replay_uses_the_same_physical_map_as_packed_tokens():
    actor = Actor(_TinyPolicyModel())
    actor._routing_replay_adapter = object()
    routes = torch.arange(2 * 1 * 2 * 4).reshape(2, 1, 2, 4)
    physical_to_dense = torch.tensor([0, 1, -1, -1, 4, -1, -1, -1])

    prepared = actor._prepare_routed_experts(routes, physical_to_dense, cp_forward=False)

    assert prepared.shape == (1, 8, 1, 2)
    assert torch.equal(prepared[0, 0], routes[0, :, :, 0])
    assert torch.equal(prepared[0, 1], routes[0, :, :, 1])
    assert torch.equal(prepared[0, 4], routes[1, :, :, 0])
    assert (prepared[0, [2, 3, 5, 6, 7]] == -1).all()


def test_hybridep_equalization_uses_ep_then_ep_shard_and_rejects_unequal_padded_batches(monkeypatch):
    actor = Actor(_TinyPolicyModel())
    actor._hybridep_equalization_groups = ("ep", "ep_shard")
    calls = []

    def max_width(extrema, *, op, group):
        calls.append((op, group))
        extrema[2] = 8

    monkeypatch.setattr(torch.distributed, "all_reduce", max_width)
    assert actor._hybridep_target_width(torch.ones(1, 4), packed=True) == 8
    assert [group for _op, group in calls] == ["ep", "ep_shard"]

    def unequal_batch(extrema, *, op, group):
        extrema[1] = 2

    monkeypatch.setattr(torch.distributed, "all_reduce", unequal_batch)
    with pytest.raises(NotImplementedError, match="same batch size"):
        actor._hybridep_target_width(torch.ones(1, 4), packed=False)


class _TinyValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(initializer_range=0.01, tie_word_embeddings=True)
        self.config.text_config = SimpleNamespace(tie_word_embeddings=True)
        self.embed_tokens = nn.Embedding(16, 4)
        self.lm_head = nn.Linear(4, 16, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, head):
        self.lm_head = head

    def forward(self, input_ids, **_kwargs):
        return self.lm_head(self.embed_tokens(input_ids))


def test_critic_installs_value_head_and_returns_dense_action_values():
    critic = Critic(_TinyValueModel())
    with torch.no_grad():
        critic.model.embed_tokens.weight.zero_()
        critic.model.embed_tokens.weight[:, 0] = torch.arange(16)
        critic.model.lm_head.weight.zero_()
        critic.model.lm_head.weight[0, 0] = 1

    sequences = torch.tensor([[1, 2, 3, 4], [5, 6, 0, 0]])
    action_mask = torch.tensor([[False, True, True], [True, False, False]])
    output = critic(sequences, action_mask, attention_mask=(sequences != 0).long())

    assert isinstance(critic.model.lm_head, _ValueHead)
    assert critic.model.lm_head.weight.dtype == torch.float32
    assert not critic.model.config.tie_word_embeddings
    assert not critic.model.config.text_config.tie_word_embeddings
    assert torch.equal(output.token_values, torch.tensor([[1.0, 2.0, 3.0], [5.0, 6.0, 0.0]]))
    assert torch.equal(output.action_values, torch.tensor([[0.0, 2.0, 3.0], [5.0, 0.0, 0.0]]))


def test_value_head_upcasts_hidden_states_and_follows_meta_device():
    head = _ValueHead(4)
    assert head(torch.ones(2, 4, dtype=torch.bfloat16)).dtype == torch.float32

    critic = Critic(_TinyValueModel().to(device="meta"))
    assert critic.model.lm_head.weight.device.type == "meta"
