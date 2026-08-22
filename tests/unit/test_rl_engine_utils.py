# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch
import torch.nn as nn

from nemo_automodel.components.datasets.datum import LossInputLayout
from nemo_automodel.engine import Engine, LossFnOutputBatch, PerTokenOutput, collate_prebatched

from molt.models.critic import _ValueHead, _install_value_head
from molt.models.loss import PolicyLoss
from molt.trainer.workers.engine_utils import prepare_rl_engine_datum
from molt.trainer.workers.policy_actor import PolicyTrainer


def _experience():
    return SimpleNamespace(
        sequences=torch.tensor([[10, 11, 12, 13, 0], [20, 21, 22, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        action_mask=torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
        values=torch.tensor([[0.0, 0.1, 0.2, 0.0], [0.3, 0.4, 0.0, 0.0]]),
        returns=torch.tensor([[0.0, 1.1, 1.2, 0.0], [1.3, 1.4, 0.0, 0.0]]),
        mm_train_inputs=[],
    )


def _wrapper(**kwargs):
    defaults = {"model": nn.Linear(1, 1), "is_vlm": False}
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_prepare_padded_rl_datum_uses_shifted_prediction_axis():
    experience = _experience()
    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
        packing_samples=False,
    )

    assert prepared.dense_shape == (2, 4)
    assert prepared.packed_indices is None
    assert torch.equal(prepared.datum.model_inputs["input_ids"], experience.sequences[:, :-1])
    assert torch.equal(
        prepared.datum.model_inputs["attention_mask"],
        torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    )
    assert torch.equal(
        prepared.datum.model_inputs["position_ids"],
        torch.tensor([[0, 1, 2, 1], [0, 1, 1, 1]]),
    )
    assert torch.equal(prepared.datum.loss_fn_inputs["target_tokens"], experience.sequences[:, 1:])
    assert set(prepared.datum.loss_fn_input_layouts.values()) == {LossInputLayout.PER_TOKEN}


def test_prepare_padded_rl_datum_masks_left_padding_without_activating_its_last_slot():
    experience = _experience()
    experience.sequences = torch.tensor([[0, 0, 10, 11, 12]])
    experience.attention_mask = torch.tensor([[0, 0, 1, 1, 1]])
    experience.action_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
    experience.values = torch.zeros(1, 4)
    experience.returns = torch.ones(1, 4)

    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
        packing_samples=False,
    )

    assert torch.equal(prepared.datum.model_inputs["attention_mask"], torch.tensor([[0, 0, 1, 1]]))
    assert torch.equal(prepared.datum.model_inputs["position_ids"], torch.tensor([[1, 1, 0, 1]]))


def test_prepare_packed_rl_datum_uses_automodel_thd_and_restores_dense_output():
    experience = _experience()
    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
        packing_samples=True,
    )

    model_inputs = prepared.datum.model_inputs
    assert model_inputs["qkv_format"] == "thd"
    assert torch.equal(model_inputs["input_ids"], torch.tensor([[10, 11, 12, 20, 21]]))
    assert torch.equal(model_inputs["seq_lens"], torch.tensor([[3, 2]]))
    assert torch.equal(prepared.datum.loss_fn_inputs["weights"], torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]]))

    restored = prepared.restore_token_output(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))
    assert torch.equal(restored, torch.tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 0.0, 0.0]]))


class _RouteAdapter:
    def prepare_routed_experts(self, routed_experts):
        return routed_experts.permute(0, 3, 1, 2).contiguous()


def test_prepare_rl_datum_declares_sequence_boundaries_and_routing_side_channel():
    experience = _experience()
    routes = torch.arange(2 * 3 * 2 * 5, dtype=torch.int16).reshape(2, 3, 2, 5)
    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(_routing_replay_adapter=_RouteAdapter()),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
        packing_samples=True,
        include_sequence_ids=True,
        routed_experts=routes,
    )

    assert prepared.datum.loss_fn_input_layouts["sequence_ids"] is LossInputLayout.PER_TOKEN
    assert prepared.datum.loss_fn_input_layouts["routed_experts"] is LossInputLayout.PER_TOKEN
    assert prepared.datum.loss_fn_input_layouts["num_sequences"] is LossInputLayout.REPLICATED
    assert prepared.datum.loss_fn_input_pad_values["sequence_ids"] == -1
    assert prepared.datum.loss_fn_input_pad_values["routed_experts"] == -1
    assert torch.equal(prepared.datum.loss_fn_inputs["sequence_ids"], torch.tensor([[0, 0, 0, 1, 1]]))
    assert prepared.datum.loss_fn_inputs["routed_experts"].shape == (1, 5, 3, 2)
    assert prepared.datum.loss_fn_inputs["num_sequences"].item() == 2


def test_prepare_vlm_rl_datum_keeps_media_and_builds_token_types():
    experience = _experience()
    experience.sequences[0, 1] = 99
    experience.mm_train_inputs = [
        {"pixel_values": torch.ones(1, 3, 2, 2), "image_grid_thw": torch.tensor([[1, 2, 2]])},
        None,
    ]
    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(is_vlm=True, _image_token_id=99, _video_token_id=None),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
        packing_samples=False,
    )

    assert "position_ids" not in prepared.datum.model_inputs
    assert prepared.datum.model_inputs["pixel_values"].shape == (1, 3, 2, 2)
    assert prepared.datum.model_inputs["mm_token_type_ids"][0, 1] == 1


class _HeadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(initializer_range=0.01, tie_word_embeddings=True)
        self.config.text_config = SimpleNamespace(tie_word_embeddings=True)
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, head):
        self.lm_head = head


def test_install_value_head_replaces_task_head_before_fsdp():
    model = _HeadModel()

    assert _install_value_head(model) is None

    assert isinstance(model.lm_head, _ValueHead)
    assert model.lm_head.proj.weight.shape == (1, 4)
    assert not model.config.tie_word_embeddings
    assert not model.config.text_config.tie_word_embeddings


class _ScalarValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, input_ids):
        return input_ids.float().unsqueeze(-1) * self.scale


def test_prebatched_rl_datum_runs_one_engine_backward_window():
    experience = _experience()
    model = _ScalarValueModel()
    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(model=model),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
        packing_samples=False,
    )
    engine = Engine(model, device="cpu", microbatch_size=1, collate_fn=collate_prebatched)

    def loss_fn(output, loss_inputs):
        values = output.squeeze(-1)
        weights = loss_inputs["weights"]
        loss_matrix = 0.5 * (values - loss_inputs["returns"]).pow(2)
        return (loss_matrix * weights).sum(), LossFnOutputBatch(
            per_token={"action_values": PerTokenOutput(values * weights)}
        )

    result = engine.forward_backward([prepared.datum], loss_fn)
    expected_values = experience.sequences[:, :-1].float() * 0.1
    expected_sum = (0.5 * (expected_values - experience.returns).pow(2) * experience.action_mask).sum()

    assert torch.allclose(result.loss_sum, expected_sum.double())
    assert torch.allclose(result.loss, expected_sum.double() / experience.action_mask.sum())
    assert torch.equal(result.loss_fn_outputs[0]["action_values"], expected_values * experience.action_mask)
    assert model.scale.grad is not None


class _TinyPolicyModel(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8):
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        del attention_mask, position_ids
        return self.output(self.embedding(input_ids))


def test_policy_callback_runs_engine_backward_and_optimizer_step(monkeypatch):
    # The production helper selects a CUDA-only fused CE kernel when flash-attn
    # is installed. Keep this Engine contract test CPU-only.
    monkeypatch.setattr(
        "molt.trainer.workers.policy_actor.log_probs_from_logits",
        lambda logits, labels, temperature: (
            torch.log_softmax(logits / temperature, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        ),
    )
    experience = _experience()
    experience.action_log_probs = torch.zeros_like(experience.values)
    experience.advantages = torch.tensor([[0.0, 1.0, -0.5, 0.0], [0.7, 1.2, 0.0, 0.0]])
    experience.base_action_log_probs = torch.zeros_like(experience.values)
    experience.rollout_log_probs = None
    experience.routed_experts = None

    model = _TinyPolicyModel()
    wrapper = _wrapper(
        model=model,
        packing_samples=False,
        temperature=1.0,
        _routing_replay_adapter=None,
    )
    trainer = object.__new__(PolicyTrainer)
    trainer.actor = wrapper
    trainer.actor_loss_fn = PolicyLoss()
    trainer._sequence_group = None
    trainer.args = SimpleNamespace(
        actor=SimpleNamespace(entropy_coef=0.01),
        algo=SimpleNamespace(kl=SimpleNamespace(use_loss=True, init_coef=0.1, estimator="k2")),
    )
    prepared = prepare_rl_engine_datum(
        experience,
        wrapper,
        loss_fields={
            "old_action_log_probs": experience.action_log_probs,
            "advantages": experience.advantages,
            "base_action_log_probs": experience.base_action_log_probs,
            "rollout_log_probs": experience.rollout_log_probs,
        },
        packing_samples=False,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    engine = Engine(
        model,
        device="cpu",
        microbatch_size=1,
        collate_fn=collate_prebatched,
        optimizers=optimizer,
        max_grad_norm=1.0,
    )
    before = model.output.weight.detach().clone()

    result = engine.forward_backward(
        [prepared.datum], lambda output, inputs: trainer._engine_loss(output, inputs, kl_ctl=0.1)
    )
    optim_result = engine.optim_step()

    assert torch.isfinite(result.loss)
    assert result.weight_sum.item() == experience.action_mask.sum().item()
    assert result.loss_fn_outputs[0]["action_log_probs"].shape == experience.action_mask.shape
    assert result.loss_fn_outputs[0]["entropy"].shape == experience.action_mask.shape
    assert float(optim_result.grad_norm) > 0
    assert not torch.equal(model.output.weight, before)
