# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from types import MethodType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
from nemo_automodel.components.datasets.datum import LossInputLayout, collate_datums, collate_vlm_datums
from nemo_automodel.engine import Engine, LossOutput

from molt.models.actor import Actor
from molt.models.base import BaseModel
from molt.models.critic import Critic, _install_value_head, _ValueHead
from molt.models.loss import PolicyLoss
from molt.trainer.algorithm.experience import Experience
from molt.trainer.workers.policy_actor import PolicyTrainer


def _experience():
    return Experience(
        sequences=torch.tensor([[10, 11, 12, 13, 0], [20, 21, 22, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
        action_mask=torch.tensor([[0, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool),
        values=torch.tensor([[0.0, 0.1, 0.2, 0.0], [0.3, 0.4, 0.0, 0.0]]),
        returns=torch.tensor([[0.0, 1.1, 1.2, 0.0], [1.3, 1.4, 0.0, 0.0]]),
    )


def _wrapper(**kwargs):
    defaults = {
        "model": nn.Linear(1, 1),
        "is_vlm": False,
        "packing_layout": None,
        "routing_replay_context": None,
    }
    defaults.update(kwargs)
    wrapper = SimpleNamespace(**defaults)
    wrapper._make_datums = MethodType(BaseModel._make_datums, wrapper)
    return wrapper


@pytest.mark.parametrize("packing_layout", [None, "thd", "indexed_mask"])
def test_text_collator_delegates_physical_layout_to_automodel(packing_layout):
    collate = BaseModel.datum_collator(_wrapper(packing_layout=packing_layout), None)

    assert collate.func is collate_datums
    assert collate.keywords == {
        "packed": packing_layout == "thd",
        "packing_layout": "indexed_mask" if packing_layout == "indexed_mask" else None,
    }


def test_vlm_collator_delegates_media_and_packing_to_automodel():
    processor = SimpleNamespace(image_processor=object())
    wrapper = _wrapper(is_vlm=True, packing_layout="thd")

    collate = BaseModel.datum_collator(wrapper, processor)

    assert collate.func is collate_vlm_datums
    assert collate.keywords["processor"] is processor
    assert collate.keywords["packed"] is True
    assert collate.keywords["packing_layout"] is None


class _MropeModel(nn.Module):
    def get_rope_index(self):
        raise AssertionError("capability probing must not execute the position builder")


def test_multi_axis_mrope_packed_vlm_cp_fails_fast():
    processor = SimpleNamespace(image_processor=object())
    wrapper = _wrapper(is_vlm=True, packing_layout="thd", model=_MropeModel())

    with pytest.raises(NotImplementedError, match="multi-axis mRoPE"):
        BaseModel.datum_collator(wrapper, processor, cp_size=2)


def test_make_text_datums_uses_the_next_token_axis():
    experience = _experience()
    datums = Critic.make_value_datums(_wrapper(), experience)

    assert len(datums) == 2
    assert torch.equal(datums[0].model_inputs["input_ids"], torch.tensor([10, 11, 12]))
    assert torch.equal(datums[1].model_inputs["input_ids"], torch.tensor([20, 21]))
    assert torch.equal(datums[0].loss_fn_inputs["target_tokens"], torch.tensor([11, 12, 13]))
    assert torch.equal(datums[1].loss_fn_inputs["target_tokens"], torch.tensor([21, 22]))
    assert set(datums[0].loss_fn_input_layouts.values()) == {LossInputLayout.PER_TOKEN}

    model_inputs, loss_inputs = collate_datums(datums, packed=False)
    assert torch.equal(model_inputs["input_ids"], torch.tensor([[10, 11, 12], [20, 21, 0]]))
    assert torch.equal(model_inputs["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
    assert torch.equal(loss_inputs["target_tokens"], torch.tensor([[11, 12, 13], [21, 22, 0]]))
    assert torch.equal(loss_inputs["weights"], torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]))


def test_collection_datums_need_targets_but_no_fake_loss_weights():
    datums = BaseModel.make_scoring_datums(_wrapper(), _experience())

    assert set(datums[0].loss_fn_inputs) == {"target_tokens"}
    assert "weights" not in datums[0].loss_fn_inputs


def test_left_padding_is_removed_and_outputs_return_to_action_coordinates():
    experience = _experience()
    experience.sequences = torch.tensor([[0, 0, 10, 11, 12]])
    experience.attention_mask = torch.tensor([[0, 0, 1, 1, 1]])
    experience.action_mask = torch.tensor([[0, 0, 1, 1]], dtype=torch.bool)
    experience.values = torch.zeros(1, 4)
    experience.returns = torch.ones(1, 4)

    datum = Critic.make_value_datums(_wrapper(), experience)[0]

    assert torch.equal(datum.model_inputs["input_ids"], torch.tensor([10, 11]))
    assert torch.equal(datum.loss_fn_inputs["target_tokens"], torch.tensor([11, 12]))
    restored = experience.align_action_outputs([torch.tensor([3.0, 4.0])])
    assert torch.equal(restored, torch.tensor([[0.0, 0.0, 3.0, 4.0]]))


def test_output_restoration_masks_non_action_tokens_and_keeps_trailing_features():
    experience = _experience()
    outputs = [torch.arange(6).reshape(3, 2), torch.arange(4).reshape(2, 2)]

    restored = experience.align_action_outputs(outputs)

    assert restored.shape == (2, 4, 2)
    assert torch.equal(restored[0, 0], torch.zeros(2, dtype=restored.dtype))
    assert torch.equal(restored[0, 1:3], outputs[0][1:3])
    assert torch.equal(restored[1, :2], outputs[1])
    assert torch.equal(restored[:, 3], torch.zeros(2, 2, dtype=restored.dtype))


class _RouteAdapter:
    def prepare_routed_experts(self, routed_experts):
        return routed_experts.permute(0, 3, 1, 2).contiguous()


def test_make_datums_declares_grouping_and_routing_side_channels():
    experience = _experience()
    experience.advantages = experience.returns
    routes = torch.arange(2 * 3 * 2 * 5, dtype=torch.int16).reshape(2, 3, 2, 5)
    experience.routed_experts = routes
    datums = Actor.make_policy_datums(
        _wrapper(routing_replay_context=_RouteAdapter()),
        experience,
        include_sequence_ids=True,
    )

    for datum in datums:
        assert datum.loss_fn_input_layouts["sequence_ids"] is LossInputLayout.PER_TOKEN
        assert datum.loss_fn_input_layouts["routed_experts"] is LossInputLayout.PER_TOKEN
        assert datum.loss_fn_input_layouts["num_sequences"] is LossInputLayout.REPLICATED
        assert datum.loss_fn_input_pad_values["sequence_ids"] == -1
        assert datum.loss_fn_input_pad_values["routed_experts"] == -1

    _, loss_inputs = collate_datums(datums, packed=False)
    assert torch.equal(loss_inputs["sequence_ids"], torch.tensor([[0, 0, 0], [1, 1, -1]]))
    assert loss_inputs["routed_experts"].shape == (2, 3, 3, 2)
    assert bool((loss_inputs["routed_experts"][1, 2] == -1).all())
    assert loss_inputs["num_sequences"].item() == 2


def test_make_vlm_datums_keeps_processor_ready_tokens_and_media():
    experience = _experience()
    experience.sequences[0, 1] = 99
    experience.mm_train_inputs = [
        {"pixel_values": torch.ones(1, 3, 2, 2), "image_grid_thw": torch.tensor([[1, 2, 2]])},
        None,
    ]
    datums = Critic.make_value_datums(_wrapper(is_vlm=True), experience)

    assert torch.equal(datums[0].model_inputs["input_ids"], torch.tensor([10, 99, 12, 13]))
    assert torch.equal(datums[1].model_inputs["input_ids"], torch.tensor([20, 21, 22]))
    assert datums[0].model_inputs["pixel_values"].shape == (1, 3, 2, 2)
    assert "pixel_values" not in datums[1].model_inputs
    assert "labels" not in datums[0].loss_fn_inputs
    assert torch.equal(datums[0].loss_fn_inputs["returns"], experience.returns[0, :3])
    assert torch.equal(datums[1].loss_fn_inputs["returns"], experience.returns[1, :2])

    processor = SimpleNamespace(
        image_processor=SimpleNamespace(merge_size=2),
        tokenizer=SimpleNamespace(pad_token_id=0),
        image_token_id=99,
    )
    model_inputs, loss_inputs = collate_vlm_datums(datums, processor=processor)
    assert torch.equal(model_inputs["input_ids"], torch.tensor([[10, 99, 12], [20, 21, 22]]))
    assert torch.equal(model_inputs["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
    assert torch.equal(model_inputs["mm_token_type_ids"], torch.tensor([[0, 1, 0], [0, 0, 0]]))
    assert torch.equal(loss_inputs["target_tokens"], torch.tensor([[99, 12, 13], [21, 22, 0]]))
    assert torch.equal(loss_inputs["weights"], torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]))
    assert torch.equal(loss_inputs["returns"], torch.tensor([[0.0, 1.1, 1.2], [1.3, 1.4, 0.0]]))


def test_vlm_collection_collation_needs_no_labels_or_weights():
    experience = _experience()
    datums = BaseModel.make_scoring_datums(
        _wrapper(is_vlm=True),
        experience,
    )
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(merge_size=2),
        tokenizer=SimpleNamespace(pad_token_id=0),
        image_token_id=99,
    )

    model_inputs, task_inputs = collate_vlm_datums(datums, processor=processor)

    assert torch.equal(model_inputs["input_ids"], torch.tensor([[10, 11, 12], [20, 21, 22]]))
    assert torch.equal(model_inputs["attention_mask"], torch.tensor([[1, 1, 1], [1, 1, 0]]))
    assert torch.equal(task_inputs["target_tokens"], torch.tensor([[11, 12, 13], [21, 22, 0]]))
    assert "labels" not in task_inputs
    assert "weights" not in task_inputs


class _ScalarValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, input_ids, **_kwargs):
        return input_ids.float().unsqueeze(-1) * self.scale


def _token_values(output, _inputs):
    return output.squeeze(-1)


def test_sample_datums_share_one_engine_optimizer_window():
    experience = _experience()
    model = _ScalarValueModel()
    datums = Critic.make_value_datums(_wrapper(model=model), experience)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    engine = Engine(
        model,
        device="cpu",
        collate_fn=partial(collate_datums, packed=False),
        optimizers=optimizer,
        max_grad_norm=1.0,
    )

    def value_loss(output, inputs):
        values = _token_values(output, inputs)
        loss_sum = (0.5 * (values - inputs["returns"]).pow(2) * inputs["weights"]).sum()
        return LossOutput(loss_sum=loss_sum, token_outputs={"action_values": values})

    before = model.scale.detach().clone()
    result = engine.forward_backward([datums], value_loss)
    restored = experience.align_action_outputs(result.token_outputs[0]["action_values"])
    step = engine.step()

    expected = experience.sequences[:, :-1].float() * before
    assert torch.equal(restored, expected * experience.action_mask)
    assert result.weight_sum.item() == experience.action_mask.sum().item()
    assert float(step.grad_norm) > 0
    assert not torch.equal(model.scale, before)


def test_packed_vlm_collection_restores_each_sample_to_replay_coordinates():
    experience = _experience()
    experience.sequences[0, 1] = 99
    experience.mm_train_inputs = [
        {"pixel_values": torch.ones(1, 3, 2, 2), "image_grid_thw": torch.tensor([[1, 2, 2]])},
        None,
    ]
    routes = torch.arange(2 * 1 * 2 * 5, dtype=torch.int16).reshape(2, 1, 2, 5)
    datums = BaseModel.make_scoring_datums(
        _wrapper(is_vlm=True, routing_replay_context=_RouteAdapter()),
        experience,
        routed_experts=routes,
    )
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(merge_size=2),
        tokenizer=SimpleNamespace(pad_token_id=0),
        image_token_id=99,
    )
    model = _ScalarValueModel()
    model.backend = SimpleNamespace(attn="te")
    engine = Engine(
        model,
        device="cpu",
        collate_fn=partial(collate_vlm_datums, processor=processor, packed=True, sequence_alignment=4),
    )

    seen = {}

    def token_values(output, inputs):
        seen["routes"] = inputs["routed_experts"]
        return _token_values(output, inputs)

    outputs = engine.forward(datums, token_values)
    restored = experience.align_action_outputs(outputs)

    torch.testing.assert_close(restored, torch.tensor([[0.0, 9.9, 1.2, 0.0], [2.0, 2.1, 0.0, 0.0]]))
    assert seen["routes"].shape == (8, 1, 2)
    assert bool((seen["routes"] == -1).any())


def test_dynamic_vlm_batches_keep_their_own_output_boundaries():
    first = Experience(
        sequences=torch.tensor([[10, 99, 12, 13, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0]]),
        action_mask=torch.tensor([[0, 1, 1, 0]], dtype=torch.bool),
        values=torch.zeros(1, 4),
        returns=torch.ones(1, 4),
        mm_train_inputs=[{"pixel_values": torch.ones(1, 3, 2, 2), "image_grid_thw": torch.tensor([[1, 2, 2]])}],
    )
    second = _experience()
    second.mm_train_inputs = [None, None]
    wrapper = _wrapper(is_vlm=True)
    datum_batches = [
        Critic.make_value_datums(wrapper, first),
        Critic.make_value_datums(wrapper, second),
    ]
    processor = SimpleNamespace(
        image_processor=SimpleNamespace(merge_size=2),
        tokenizer=SimpleNamespace(pad_token_id=0),
        image_token_id=99,
    )
    model = _ScalarValueModel()
    model.backend = SimpleNamespace(attn="te")
    engine = Engine(
        model,
        device="cpu",
        collate_fn=partial(collate_vlm_datums, processor=processor, packed=True),
    )

    def value_loss(output, inputs):
        values = _token_values(output, inputs)
        return LossOutput(
            loss_sum=(values.square() * inputs["weights"]).sum(),
            token_outputs={"action_values": values},
        )

    result = engine.forward_backward(datum_batches, value_loss)
    first_values = first.align_action_outputs(result.token_outputs[0]["action_values"])
    second_values = second.align_action_outputs(result.token_outputs[1]["action_values"])

    assert [len(batch["action_values"]) for batch in result.token_outputs] == [1, 2]
    torch.testing.assert_close(first_values, torch.tensor([[0.0, 9.9, 1.2, 0.0]]))
    torch.testing.assert_close(second_values, torch.tensor([[0.0, 1.1, 1.2, 0.0], [2.0, 2.1, 0.0, 0.0]]))
    expected_loss = (
        (first_values.square() * first.action_mask).sum() + (second_values.square() * second.action_mask).sum()
    ) / (first.action_mask.sum() + second.action_mask.sum())
    torch.testing.assert_close(result.loss, expected_loss.to(result.loss))


class _TinyPolicyModel(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, **_kwargs):
        return self.output(self.embedding(input_ids))


def test_policy_datums_run_real_logprob_entropy_backward_and_step():
    experience = _experience()
    experience.action_log_probs = torch.zeros_like(experience.values)
    experience.advantages = torch.tensor([[0.0, 1.0, -0.5, 0.0], [0.7, 1.2, 0.0, 0.0]])
    experience.base_action_log_probs = torch.zeros_like(experience.values)
    experience.rollout_log_probs = None

    model = _TinyPolicyModel()
    actor = _wrapper(model=model, temperature=1.0)
    actor.compute_action_log_probs = MethodType(Actor.compute_action_log_probs, actor)
    actor.compute_entropy = MethodType(Actor.compute_entropy, actor)
    datums = Actor.make_policy_datums(actor, experience)
    trainer = object.__new__(PolicyTrainer)
    trainer.actor = actor
    trainer.actor_loss_fn = PolicyLoss(loss_agg_mode="token-sum")
    trainer._sequence_group = None
    trainer.args = SimpleNamespace(
        actor=SimpleNamespace(entropy_coef=0.01),
        algo=SimpleNamespace(kl=SimpleNamespace(use_loss=True, init_coef=0.1, estimator="k2")),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    engine = Engine(
        model,
        device="cpu",
        collate_fn=partial(collate_datums, packed=False),
        optimizers=optimizer,
        max_grad_norm=1.0,
    )

    collected = experience.align_action_outputs(engine.forward(datums, actor.compute_action_log_probs))
    assert collected.shape == experience.action_mask.shape
    assert bool((collected[~experience.action_mask] == 0).all())
    assert all(parameter.grad is None for parameter in model.parameters())

    def policy_loss(output, inputs):
        return trainer._policy_objective(output, inputs, kl_ctl=0.1)

    before = model.output.weight.detach().clone()
    result = engine.forward_backward([datums], policy_loss)
    action_log_probs = experience.align_action_outputs(result.token_outputs[0]["action_log_probs"])
    entropy = experience.align_action_outputs(result.token_outputs[0]["entropy"])
    step = engine.step()

    assert torch.isfinite(result.loss)
    assert result.weight_sum.item() == experience.action_mask.sum().item()
    assert action_log_probs.shape == experience.action_mask.shape
    assert entropy.shape == experience.action_mask.shape
    assert float(step.grad_norm) > 0
    assert not torch.equal(model.output.weight, before)


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

    result = _install_value_head(model)

    assert result is model.lm_head
    assert isinstance(model.lm_head, _ValueHead)
    assert model.lm_head.weight.shape == (1, 4)
    assert model.lm_head.weight.dtype == torch.float32
    assert set(model.state_dict()) == {"embed_tokens.weight", "lm_head.weight"}
    assert not model.config.tie_word_embeddings
    assert not model.config.text_config.tie_word_embeddings


def test_value_head_uses_model_initializer_and_upcasts_hidden_states():
    torch.manual_seed(123)
    expected = torch.empty(1, 4)
    nn.init.normal_(expected, mean=0.0, std=0.01)

    torch.manual_seed(123)
    head = _ValueHead(4, initializer_range=0.01)

    assert torch.equal(head.weight, expected)
    assert head(torch.ones(2, 4, dtype=torch.bfloat16)).dtype == torch.float32


def test_install_value_head_follows_meta_output_head_device():
    model = _HeadModel().to(device="meta")

    result = _install_value_head(model)

    assert result is model.lm_head
    assert model.lm_head.weight.device.type == "meta"


def test_actor_log_probs_use_automodel_token_operation(monkeypatch):
    logits = object()
    targets = torch.tensor([[2, 3]])
    expected = torch.tensor([[-0.2, -0.3]])

    def token_log_probs(received_logits, received_targets, *, temperature):
        assert received_logits is logits
        assert received_targets is targets
        assert temperature == 0.7
        return expected

    monkeypatch.setattr("molt.models.actor.token_log_probs", token_log_probs)
    actor = SimpleNamespace(temperature=0.7)

    result = Actor.compute_action_log_probs(actor, {"logits": logits}, {"target_tokens": targets})

    assert result is expected


def test_actor_entropy_uses_automodel_token_operation(monkeypatch):
    logits = object()
    expected = torch.tensor([[0.4, 0.5]])

    def token_entropy(received_logits, *, temperature):
        assert received_logits is logits
        assert temperature == 0.7
        return expected

    monkeypatch.setattr("molt.models.actor.token_entropy", token_entropy)
    actor = SimpleNamespace(temperature=0.7)

    result = Actor.compute_entropy(actor, {"logits": logits})

    assert result is expected


def test_policy_loss_returns_named_action_outputs_and_entropy():
    logits = object()
    log_probs = torch.tensor([[-0.2, -0.3]])
    entropy = torch.tensor([[0.4, 0.5]])

    trainer = object.__new__(PolicyTrainer)
    trainer.actor = SimpleNamespace(
        temperature=0.7,
        compute_action_log_probs=lambda _output, _inputs: log_probs,
        compute_entropy=lambda _output, _inputs: entropy,
    )
    trainer.actor_loss_fn = lambda *_args, **_kwargs: (log_probs.new_zeros(()),)
    trainer._sequence_group = None
    trainer.args = SimpleNamespace(
        actor=SimpleNamespace(entropy_coef=0.1),
        algo=SimpleNamespace(kl=SimpleNamespace(use_loss=False, init_coef=0.0)),
    )
    loss_inputs = {
        "weights": torch.ones_like(log_probs),
        "advantages": torch.ones_like(log_probs),
    }

    result = trainer._policy_objective({"logits": logits}, loss_inputs, kl_ctl=0.0)

    torch.testing.assert_close(result.loss_sum, -entropy.sum() * 0.1)
    assert result.token_outputs["action_log_probs"] is log_probs
    assert result.token_outputs["entropy"] is entropy
