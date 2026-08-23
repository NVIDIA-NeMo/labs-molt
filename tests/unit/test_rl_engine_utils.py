# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import partial
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from nemo_automodel.components.datasets.datum import LossInputLayout, collate_datums, collate_vlm_datums
from nemo_automodel.engine import Engine, LossFnOutputBatch, PerTokenOutput

from molt.models.critic import _install_value_head, _ValueHead
from molt.models.loss import PolicyLoss
from molt.trainer.workers.engine_utils import (
    action_log_probs_from_output,
    prepare_rl_engine_datum,
    resolve_rl_engine_collation,
    run_rl_engine_forward,
)
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
    defaults = {"model": nn.Linear(1, 1), "is_vlm": False, "packing_layout": None}
    if kwargs.get("packing_samples"):
        defaults["packing_layout"] = "thd"
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class _FakeDeviceMesh:
    def __init__(self, cp_size):
        self.mesh_dim_names = ("cp",) if cp_size > 1 else ()
        self._cp_mesh = SimpleNamespace(size=lambda: cp_size)

    def __getitem__(self, name):
        assert name == "cp"
        return self._cp_mesh


def _vlm_collation_strategy(*, dynamic=False, cp_size=1):
    return SimpleNamespace(
        device_mesh=_FakeDeviceMesh(cp_size),
        args=SimpleNamespace(train=SimpleNamespace(dynamic_batch_enable=dynamic)),
    )


def test_vlm_engine_collation_uses_automodel_and_preserves_fixed_sample_batching():
    processor = SimpleNamespace(image_processor=object())
    wrapper = _wrapper(is_vlm=True, packing_samples=True)

    collate_fn, microbatch_size = resolve_rl_engine_collation(
        wrapper, processor, _vlm_collation_strategy(), micro_train_batch_size=3
    )

    assert collate_fn.func is collate_vlm_datums
    assert collate_fn.keywords["packed"] is True
    assert collate_fn.keywords["packing_layout"] is None
    assert microbatch_size == 3


def test_dynamic_vlm_engine_collation_defaults_to_one_datum():
    processor = SimpleNamespace(image_processor=object())
    wrapper = _wrapper(is_vlm=True, packing_samples=False)

    _collate_fn, microbatch_size = resolve_rl_engine_collation(
        wrapper, processor, _vlm_collation_strategy(dynamic=True), micro_train_batch_size=3
    )

    assert microbatch_size == 1


class _MropeModel(nn.Module):
    def get_rope_index(self):
        raise AssertionError("capability probing must not execute the position builder")


def test_multi_axis_mrope_packed_vlm_cp_fails_fast():
    processor = SimpleNamespace(image_processor=object())
    wrapper = _wrapper(is_vlm=True, packing_samples=True, model=_MropeModel())

    with pytest.raises(NotImplementedError, match="multi-axis mRoPE"):
        resolve_rl_engine_collation(wrapper, processor, _vlm_collation_strategy(cp_size=2), micro_train_batch_size=2)


@pytest.mark.parametrize("packed", [False, True])
def test_text_engine_collation_delegates_padding_and_packing_to_automodel(packed):
    wrapper = _wrapper(packing_samples=packed)

    collate_fn, microbatch_size = resolve_rl_engine_collation(
        wrapper, None, _vlm_collation_strategy(), micro_train_batch_size=3
    )

    assert collate_fn.func is collate_datums
    assert collate_fn.keywords == {
        "packed": packed,
        "packing_layout": None,
    }
    assert microbatch_size == 3


@pytest.mark.parametrize("is_vlm", [False, True])
def test_engine_collation_selects_indexed_mask_layout(is_vlm):
    processor = SimpleNamespace(image_processor=object()) if is_vlm else None
    wrapper = _wrapper(is_vlm=is_vlm, packing_samples=True, packing_layout="indexed_mask")

    collate_fn, microbatch_size = resolve_rl_engine_collation(
        wrapper, processor, _vlm_collation_strategy(), micro_train_batch_size=3
    )

    expected = collate_vlm_datums if is_vlm else collate_datums
    assert collate_fn.func is expected
    assert collate_fn.keywords["packed"] is False
    assert collate_fn.keywords["packing_layout"] == "indexed_mask"
    assert microbatch_size == 3


def test_prepare_padded_rl_datum_uses_shifted_prediction_axis():
    experience = _experience()
    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
    )

    assert prepared.dense_shape == (2, 4)
    assert prepared.num_datums == 2
    assert torch.equal(prepared.datums[0].model_inputs["input_ids"], torch.tensor([10, 11, 12]))
    assert torch.equal(prepared.datums[1].model_inputs["input_ids"], torch.tensor([20, 21]))
    assert torch.equal(prepared.datums[0].loss_fn_inputs["target_tokens"], torch.tensor([11, 12, 13]))
    assert torch.equal(prepared.datums[1].loss_fn_inputs["target_tokens"], torch.tensor([21, 22]))
    assert set(prepared.datums[0].loss_fn_input_layouts.values()) == {LossInputLayout.PER_TOKEN}

    model_inputs, loss_inputs = collate_datums(list(prepared.datums), packed=False)
    assert torch.equal(model_inputs["input_ids"], torch.tensor([[10, 11, 12], [20, 21, 0]]))
    assert torch.equal(
        model_inputs["attention_mask"],
        torch.tensor([[1, 1, 1], [1, 1, 0]]),
    )
    assert torch.equal(
        loss_inputs["target_tokens"],
        torch.tensor([[11, 12, 13], [21, 22, 0]]),
    )
    assert torch.equal(loss_inputs["weights"], torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 0.0]]))


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
    )

    assert torch.equal(prepared.datums[0].model_inputs["input_ids"], torch.tensor([10, 11]))
    assert torch.equal(prepared.datums[0].loss_fn_inputs["target_tokens"], torch.tensor([11, 12]))
    assert torch.equal(prepared.datums[0].loss_fn_inputs["weights"], torch.tensor([1.0, 1.0]))
    restored = prepared.restore_token_outputs([torch.tensor([3.0, 4.0])])
    assert torch.equal(restored, torch.tensor([[0.0, 0.0, 3.0, 4.0]]))


def test_prepare_packed_rl_datum_uses_automodel_thd_and_restores_dense_output():
    experience = _experience()
    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
    )

    model_inputs, loss_inputs = collate_datums(list(prepared.datums), packed=True)
    assert model_inputs["qkv_format"] == "thd"
    assert torch.equal(model_inputs["input_ids"], torch.tensor([[10, 11, 12, 20, 21]]))
    assert torch.equal(model_inputs["seq_lens"], torch.tensor([[3, 2]]))
    assert torch.equal(loss_inputs["weights"], torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]]))

    restored = prepared.restore_token_outputs([torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])])
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
        include_sequence_ids=True,
        routed_experts=routes,
    )

    for datum in prepared.datums:
        assert datum.loss_fn_input_layouts["sequence_ids"] is LossInputLayout.PER_TOKEN
        assert datum.loss_fn_input_layouts["routed_experts"] is LossInputLayout.PER_TOKEN
        assert datum.loss_fn_input_layouts["num_sequences"] is LossInputLayout.REPLICATED
        assert datum.loss_fn_input_pad_values["sequence_ids"] == -1
        assert datum.loss_fn_input_pad_values["routed_experts"] == -1

    _, loss_inputs = collate_datums(list(prepared.datums), packed=False)
    assert torch.equal(loss_inputs["sequence_ids"], torch.tensor([[0, 0, 0], [1, 1, -1]]))
    assert loss_inputs["routed_experts"].shape == (2, 3, 3, 2)
    assert bool((loss_inputs["routed_experts"][1, 2] == -1).all())
    assert loss_inputs["num_sequences"].item() == 2


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
    )

    assert prepared.num_datums == 2
    assert torch.equal(prepared.datums[0].model_inputs["input_ids"], torch.tensor([10, 99, 12, 13]))
    assert torch.equal(prepared.datums[1].model_inputs["input_ids"], torch.tensor([20, 21, 22]))
    assert "position_ids" not in prepared.datums[0].model_inputs
    assert prepared.datums[0].model_inputs["pixel_values"].shape == (1, 3, 2, 2)
    assert prepared.datums[0].model_inputs["mm_token_type_ids"][1] == 1
    assert "pixel_values" not in prepared.datums[1].model_inputs
    assert torch.equal(prepared.datums[0].loss_fn_inputs["returns"], experience.returns[0, :3])
    assert torch.equal(prepared.datums[1].loss_fn_inputs["returns"], experience.returns[1, :2])


def test_packed_vlm_rl_datums_collate_side_channels_and_restore_dense_outputs():
    experience = _experience()
    experience.sequences[0, 1] = 99
    experience.mm_train_inputs = [
        {"pixel_values": torch.ones(1, 3, 2, 2), "image_grid_thw": torch.tensor([[1, 2, 2]])},
        None,
    ]
    wrapper = _wrapper(
        is_vlm=True,
        _image_token_id=99,
        _video_token_id=None,
        _routing_replay_adapter=_RouteAdapter(),
    )
    routes = torch.arange(2 * 1 * 2 * 5, dtype=torch.int16).reshape(2, 1, 2, 5)
    prepared = prepare_rl_engine_datum(
        experience,
        wrapper,
        loss_fields={"old_values": experience.values, "returns": experience.returns},
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
        microbatch_size=1,
        collate_fn=partial(collate_vlm_datums, processor=processor, packed=True, sequence_alignment=4),
    )

    seen = {"calls": 0}

    def token_output(output, loss_inputs):
        seen["calls"] += 1
        values = output.squeeze(-1)
        seen["routes"] = loss_inputs["routed_experts"]
        return values

    restored = run_rl_engine_forward(engine, prepared, "values", token_output)

    torch.testing.assert_close(restored, torch.tensor([[0.0, 9.9, 1.2, 0.0], [2.0, 2.1, 0.0, 0.0]]))
    assert seen["calls"] == 1
    assert seen["routes"].shape == (8, 1, 2)
    assert bool((seen["routes"] == -1).any())


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


class _ScalarValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, input_ids):
        return input_ids.float().unsqueeze(-1) * self.scale


def test_per_sample_rl_datums_run_one_engine_backward_window():
    experience = _experience()
    model = _ScalarValueModel()
    prepared = prepare_rl_engine_datum(
        experience,
        _wrapper(model=model),
        loss_fields={"old_values": experience.values, "returns": experience.returns},
    )
    engine = Engine(model, device="cpu", microbatch_size=2, collate_fn=partial(collate_datums, packed=False))

    collected = run_rl_engine_forward(
        engine,
        prepared,
        "action_values",
        lambda output, _inputs: output.squeeze(-1),
    )
    expected_values = experience.sequences[:, :-1].float() * 0.1
    assert torch.equal(collected, expected_values * experience.action_mask)
    assert model.scale.grad is None

    def loss_fn(output, loss_inputs):
        values = output.squeeze(-1)
        weights = loss_inputs["weights"]
        loss_matrix = 0.5 * (values - loss_inputs["returns"]).pow(2)
        return (loss_matrix * weights).sum(), LossFnOutputBatch(
            per_token={"action_values": PerTokenOutput(values * weights)}
        )

    result = engine.forward_backward(
        prepared.datums,
        loss_fn,
        microbatch_sizes=(prepared.num_datums,),
    )
    expected_sum = (0.5 * (expected_values - experience.returns).pow(2) * experience.action_mask).sum()
    restored_values = prepared.restore_token_outputs([record["action_values"] for record in result.loss_fn_outputs])

    assert torch.allclose(result.loss_sum, expected_sum.double())
    assert torch.allclose(result.loss, expected_sum.double() / experience.action_mask.sum())
    assert torch.equal(restored_values, expected_values * experience.action_mask)
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


def test_action_log_probs_delegate_vocab_sharded_logits_to_automodel(monkeypatch):
    class FakeDTensor:
        pass

    logits = FakeDTensor()
    targets = torch.tensor([[2, 3]])
    expected = torch.tensor([[-0.2, -0.3]])

    def log_probs(received_logits, received_targets, *, temperature):
        assert received_logits is logits
        assert received_targets is targets
        assert temperature == 0.7
        return expected

    monkeypatch.setattr("molt.trainer.workers.engine_utils.DTensor", FakeDTensor)
    monkeypatch.setattr("molt.trainer.workers.engine_utils.vocab_parallel_log_probs", log_probs)

    result = action_log_probs_from_output({"logits": logits}, {"target_tokens": targets}, temperature=0.7)

    assert result is expected


def test_policy_entropy_delegates_vocab_sharded_logits_to_automodel(monkeypatch):
    class FakeDTensor:
        pass

    logits = FakeDTensor()
    log_probs = torch.tensor([[-0.2, -0.3]])
    expected_entropy = torch.tensor([[0.4, 0.5]])

    def entropy(received_logits, *, temperature):
        assert received_logits is logits
        assert temperature == 0.7
        return expected_entropy

    monkeypatch.setattr("molt.trainer.workers.policy_actor.DTensor", FakeDTensor)
    monkeypatch.setattr("molt.trainer.workers.policy_actor.vocab_parallel_entropy", entropy)
    monkeypatch.setattr(
        "molt.trainer.workers.policy_actor.action_log_probs_from_output",
        lambda _output, _inputs, _temperature: log_probs,
    )

    trainer = object.__new__(PolicyTrainer)
    trainer.actor = SimpleNamespace(temperature=0.7)
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

    numerator, outputs = trainer.compute_policy_loss({"logits": logits}, loss_inputs, kl_ctl=0.0)

    torch.testing.assert_close(numerator, -expected_entropy.sum() * 0.1)
    torch.testing.assert_close(outputs.per_token["action_log_probs"].tensor, log_probs)
    assert outputs.per_token["entropy"].tensor is expected_entropy


def test_policy_loss_runs_engine_backward_and_optimizer_step(monkeypatch):
    # The production helper selects a CUDA-only fused CE kernel when flash-attn
    # is installed. Keep this Engine contract test CPU-only.
    monkeypatch.setattr(
        "molt.trainer.workers.engine_utils.log_probs_from_logits",
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
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    engine = Engine(
        model,
        device="cpu",
        microbatch_size=2,
        collate_fn=partial(collate_datums, packed=False),
        optimizers=optimizer,
        max_grad_norm=1.0,
    )
    before = model.output.weight.detach().clone()

    collected = run_rl_engine_forward(
        engine,
        prepared,
        "action_log_probs",
        lambda output, inputs: action_log_probs_from_output(output, inputs, wrapper.temperature),
    )
    assert collected.shape == experience.action_mask.shape
    assert torch.equal(collected[~experience.action_mask], torch.zeros_like(collected[~experience.action_mask]))
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not optimizer.state

    result = engine.forward_backward(
        prepared.datums,
        lambda output, inputs: trainer.compute_policy_loss(output, inputs, kl_ctl=0.1),
        microbatch_sizes=(prepared.num_datums,),
    )
    optim_result = engine.optim_step()
    action_log_probs = prepared.restore_token_outputs(
        [record["action_log_probs"] for record in result.loss_fn_outputs]
    )
    entropy = prepared.restore_token_outputs([record["entropy"] for record in result.loss_fn_outputs])

    assert torch.isfinite(result.loss)
    assert result.weight_sum.item() == experience.action_mask.sum().item()
    assert action_log_probs.shape == experience.action_mask.shape
    assert entropy.shape == experience.action_mask.shape
    assert float(optim_result.grad_norm) > 0
    assert not torch.equal(model.output.weight, before)
