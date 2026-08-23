# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from nemo_automodel.components.datasets.datum import Datum, LossInputLayout, collate_datums, collate_vlm_datums

from molt.datasets.sft_dataset import SFTDataset
from molt.trainer.sft_trainer import SFTTrainer


class _Strategy:
    def __init__(self, accumulated_gradient=2):
        self.args = SimpleNamespace(
            optim="adam",
            model=SimpleNamespace(aux_loss_coef=0.0, freeze_visual_encoder=False),
            fsdp=SimpleNamespace(packing_samples=False),
            train=SimpleNamespace(micro_batch_size=2),
            logger=SimpleNamespace(
                wandb=SimpleNamespace(key=None, org=None, project=None, group=None, run_name="test"),
                tensorboard_dir=None,
            ),
        )
        self.accumulated_gradient = accumulated_gradient
        self.device_mesh = self.moe_mesh = None
        self.events = []
        self.messages = []
        self.reductions = []

    def is_rank_0(self):
        return True

    def print(self, *args, **kwargs):
        self.messages.append(" ".join(str(arg) for arg in args))

    def all_reduce(self, data, op="mean"):
        self.reductions.append((data.detach().clone(), op))
        return data

    def _maybe_debug_grad_stats(self, model, name):
        self.events.append("debug")


class _RawLM(torch.nn.Module):
    """Position-sensitive raw backbone returning ``[batch, sequence, vocab]`` logits."""

    def __init__(self, tensor_output=False):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.3, 0.1, 0.7]))
        self.register_buffer("position_scale", torch.tensor([0.00, 0.02, -0.01, 0.03]))
        self.config = SimpleNamespace(pad_token_id=0)
        self.tensor_output = tensor_output
        self.seen_position_ids = []

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        assert attention_mask is not None
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        self.seen_position_ids.append(position_ids.detach().clone())
        logits = self.logits.expand(*input_ids.shape, -1)
        logits = logits + position_ids.unsqueeze(-1) * self.position_scale
        return logits if self.tensor_output else SimpleNamespace(logits=logits)


class _Actor(torch.nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.model = model or _RawLM()

    def forward(self, *args, **kwargs):
        raise AssertionError("Engine SFT must bypass the outer Actor wrapper")


class _Scheduler:
    def __init__(self, optimizer, events):
        self.optimizer = optimizer
        self.events = events
        self.step_calls = 0

    def step(self):
        self.events.append("scheduler")
        self.step_calls += 1


class _Loader(list):
    sampler = None


class _Writer:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, value, step))

    def close(self):
        pass


def _batch(tokens, attention_mask, weights):
    dataset = object.__new__(SFTDataset)
    dataset.processor = None
    assert all(all(value == 1 for value in row) for row in attention_mask)
    return [dataset._make_datum(row_tokens, row_weights, None) for row_tokens, row_weights in zip(tokens, weights)]


def _args(batch_size=3):
    return SimpleNamespace(
        train=SimpleNamespace(batch_size=batch_size),
        eval=SimpleNamespace(steps=-1),
        ckpt=SimpleNamespace(save_steps=-1),
        logger=SimpleNamespace(logging_steps=1),
    )


def _reference_loss(model, batches):
    numerator = torch.zeros(())
    denominator = torch.zeros(())
    for batch in batches:
        model_inputs, loss_inputs = collate_datums(batch)
        labels = loss_inputs["labels"]
        output = model(
            model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            position_ids=model_inputs.get("position_ids"),
        )
        logits = output if torch.is_tensor(output) else output.logits
        numerator = numerator + F.cross_entropy(
            logits.flatten(0, 1),
            labels.flatten(),
            ignore_index=-100,
            reduction="sum",
        )
        denominator = denominator + loss_inputs["weights"].sum()
    return numerator / denominator


def _vlm_datum(input_ids, weights, **media_inputs):
    """Build an unshifted processor-ready item for the AM VLM collater.

    Args:
        input_ids: Token IDs with shape ``[sequence]``.
        weights: Boolean supervision values with shape ``[sequence]``.
        **media_inputs: Processor tensors with shape ``[media, ...]``.

    Returns:
        A Datum retaining the token and media tensor shapes.
    """
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    weights = torch.tensor(weights, dtype=torch.bool)
    labels = input_ids.clone().masked_fill(~weights, -100)
    return Datum(
        model_inputs={
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            **media_inputs,
        },
        loss_fn_inputs={"labels": labels, "weights": weights},
        loss_fn_input_layouts={
            "labels": LossInputLayout.PER_TOKEN,
            "weights": LossInputLayout.PER_TOKEN,
        },
        loss_fn_input_pad_values={"labels": -100},
    )


def test_trainer_uses_automodel_vlm_collater_and_preserves_processor_fields():
    processor = SimpleNamespace(image_processor=object(), tokenizer=SimpleNamespace(pad_token_id=9))
    strategy = _Strategy(accumulated_gradient=1)
    actor = _Actor()
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.05)
    trainer = SFTTrainer(actor, strategy, optimizer, [], None, _Scheduler(optimizer, []), tokenizer=processor)
    datums = [
        _vlm_datum(
            [1, 2, 3],
            [0, 1, 1],
            pixel_values=torch.tensor([[1.0, 2.0]]),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
            image_flags=torch.tensor([[1]]),
            imgs_sizes=torch.tensor([[32, 64]]),
        ),
        _vlm_datum([4, 5], [0, 1]),
    ]

    assert trainer.engine.collate_fn.func is collate_vlm_datums
    model_inputs, loss_inputs = trainer.engine.collate_fn(datums)

    torch.testing.assert_close(model_inputs["input_ids"], torch.tensor([[1, 2], [4, 5]]))
    torch.testing.assert_close(model_inputs["attention_mask"], torch.tensor([[1, 1], [1, 0]]))
    torch.testing.assert_close(loss_inputs["labels"], torch.tensor([[2, 3], [5, -100]]))
    torch.testing.assert_close(loss_inputs["weights"], torch.tensor([[True, True], [True, False]]))
    torch.testing.assert_close(model_inputs["image_flags"], torch.tensor([[1]]))
    torch.testing.assert_close(model_inputs["imgs_sizes"], torch.tensor([[32, 64]]))
    assert model_inputs["pixel_values"].dtype == torch.bfloat16
    assert loss_inputs.item_to_datum == (0, 1)
    assert loss_inputs.layouts == {
        "labels": LossInputLayout.PER_TOKEN,
        "weights": LossInputLayout.PER_TOKEN,
    }
    assert loss_inputs.pad_values == {"labels": -100}


def test_vlm_datum_adapter_rejects_weight_on_unpredictable_first_token():
    processor = SimpleNamespace(image_processor=object(), tokenizer=SimpleNamespace(pad_token_id=0))
    with pytest.raises(ValueError, match="target position zero"):
        collate_vlm_datums([_vlm_datum([1, 2], [1, 1])], processor=processor)


def test_trainer_selects_automodel_vlm_packing_collater():
    processor = SimpleNamespace(image_processor=object(), tokenizer=SimpleNamespace(pad_token_id=0))
    strategy = _Strategy(accumulated_gradient=1)
    strategy.args.fsdp.packing_samples = True
    actor = _Actor()
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.05)
    trainer = SFTTrainer(actor, strategy, optimizer, [], None, _Scheduler(optimizer, []), tokenizer=processor)

    model_inputs, loss_inputs = trainer.engine.collate_fn(
        [_vlm_datum([1, 2, 3], [0, 1, 1]), _vlm_datum([4, 5], [0, 1])]
    )

    assert trainer.engine.collate_fn.func is collate_vlm_datums
    assert trainer.engine.collate_fn.keywords["packed"] is True
    assert model_inputs["qkv_format"] == "thd"
    assert model_inputs["input_ids"].shape == (1, 3)
    assert loss_inputs["weights"].sum().item() == 3


def test_trainer_rejects_mrope_vlm_packing_with_context_parallelism():
    class _CpMesh:
        mesh_dim_names = ("cp",)

        def __getitem__(self, _name):
            return SimpleNamespace(size=lambda: 2)

    class _MropeLM(_RawLM):
        def get_rope_index(self, input_ids, **_kwargs):
            positions = torch.arange(input_ids.shape[-1]).expand(3, input_ids.shape[0], -1)
            return positions, torch.zeros(1)

    processor = SimpleNamespace(image_processor=object(), tokenizer=SimpleNamespace(pad_token_id=0))
    strategy = _Strategy(accumulated_gradient=1)
    strategy.args.fsdp.packing_samples = True
    strategy.device_mesh = _CpMesh()
    actor = _Actor(_MropeLM())
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.05)

    with pytest.raises(NotImplementedError, match="multi-axis mRoPE"):
        SFTTrainer(actor, strategy, optimizer, [], None, _Scheduler(optimizer, []), tokenizer=processor)


def test_trainer_selects_automodel_text_packing_collater():
    strategy = _Strategy(accumulated_gradient=1)
    strategy.args.fsdp.packing_samples = True
    actor = _Actor()
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.05)
    trainer = SFTTrainer(actor, strategy, optimizer, [], None, _Scheduler(optimizer, []))

    model_inputs, loss_inputs = trainer.engine.collate_fn(
        _batch(
            [[0, 1, 2], [3, 2, 1, 0]],
            [[1, 1, 1], [1, 1, 1, 1]],
            [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0]],
        )
    )

    assert model_inputs["qkv_format"] == "thd"
    assert model_inputs["input_ids"].shape == (1, 5)
    assert loss_inputs["labels"].shape == (1, 5)
    assert loss_inputs["weights"].sum().item() == 4


@pytest.mark.parametrize("tensor_output", [False, True])
def test_engine_only_sft_matches_full_window_masked_update(tensor_output):
    strategy = _Strategy()
    actor = _Actor(_RawLM(tensor_output=tensor_output))
    initial = actor.model.state_dict()
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.05)
    scheduler = _Scheduler(optimizer, strategy.events)
    batches = [
        _batch(
            [[0, 1, 2], [3, 2, 1, 0, 3]],
            [[1, 1, 1], [1, 1, 1, 1, 1]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0, 0.0]],
        ),
        _batch([[1, 3, 2, 0]], [[1, 1, 1, 1]], [[1.0, 1.0, 1.0, 0.0]]),
    ]
    trainer = SFTTrainer(
        actor,
        strategy,
        optimizer,
        _Loader(batches),
        None,
        scheduler,
        max_norm=0,
        max_epochs=1,
    )
    assert trainer.engine.max_grad_norm is None
    assert strategy.messages[-1] == "[SFT] backend=engine"

    real_forward_backward = trainer.engine.forward_backward
    real_optim_step = trainer.engine.optim_step
    engine_grads = []

    def tracked_forward_backward(*args, **kwargs):
        strategy.events.append("forward_backward")
        return real_forward_backward(*args, **kwargs)

    def tracked_optim_step(*args, **kwargs):
        engine_grads.append(actor.model.logits.grad.detach().clone())
        strategy.events.append("optim_step")
        return real_optim_step(*args, **kwargs)

    trainer.engine.forward_backward = tracked_forward_backward
    trainer.engine.optim_step = tracked_optim_step
    logged = []
    trainer.save_logs_and_checkpoints = lambda a, step, bar, logs=None, states=None: logged.append(
        (step, dict(logs), dict(states))
    )

    reference = _RawLM(tensor_output=tensor_output)
    reference.load_state_dict(initial)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    expected_loss = _reference_loss(reference, batches)
    expected_loss.backward()
    reference_optimizer.step()

    trainer.fit(_args(), num_update_steps_per_epoch=1)

    assert strategy.events == ["forward_backward", "debug", "optim_step", "scheduler"]
    assert scheduler.step_calls == 1
    assert strategy.reductions == []
    assert len(logged) == 1
    step, logs, states = logged[0]
    assert step == 1
    assert states == {"consumed_samples": 3}
    assert logs["sft_loss"] == pytest.approx(expected_loss.item())
    assert logs["grad_norm"] == 0.0
    torch.testing.assert_close(engine_grads[0], reference.logits.grad)
    torch.testing.assert_close(actor.model.logits, reference.logits)
    torch.testing.assert_close(
        actor.model.seen_position_ids[0],
        torch.tensor([[0, 1, 1, 1], [0, 1, 2, 3]]),
    )


def test_sft_loss_does_not_double_count_native_moe_aux_loss():
    trainer = object.__new__(SFTTrainer)
    trainer.loss_fn = lambda logits, labels: F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")
    logits = torch.tensor([[[0.2, -0.3, 0.1], [0.7, 0.0, -0.4]]], requires_grad=True)
    aux_loss = torch.tensor(2.0, requires_grad=True)

    loss = trainer.compute_sft_loss(
        SimpleNamespace(logits=logits, aux_loss=aux_loss),
        {"labels": torch.tensor([[0, 1]])},
    )
    loss.backward()

    assert logits.grad is not None
    assert aux_loss.grad is None


def test_eval_uses_engine_forward_and_one_dataset_reduction():
    strategy = _Strategy(accumulated_gradient=1)
    actor = _Actor()
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.05)
    scheduler = _Scheduler(optimizer, strategy.events)
    batches = [
        _batch([[0, 1, 2]], [[1, 1, 1]], [[1.0, 1.0, 0.0]]),
        _batch([[3, 2, 1, 0]], [[1, 1, 1, 1]], [[1.0, 1.0, 1.0, 0.0]]),
    ]
    trainer = SFTTrainer(actor, strategy, optimizer, [], _Loader(batches), scheduler)
    writer = _Writer()
    trainer._tensorboard = writer
    real_forward = trainer.engine.forward
    forward_calls = 0

    def tracked_forward(*args, **kwargs):
        nonlocal forward_calls
        forward_calls += 1
        return real_forward(*args, **kwargs)

    trainer.engine.forward = tracked_forward
    expected = _reference_loss(actor.model, batches).item()
    actor.model.seen_position_ids.clear()

    trainer.evaluate(_Loader(batches), steps=7)

    assert forward_calls == 2
    assert len(strategy.reductions) == 1
    reduced, op = strategy.reductions[0]
    assert reduced.shape == (2,)
    assert op == "sum"
    assert len(writer.scalars) == 1
    assert writer.scalars[0][0] == "eval/eval sft_loss"
    assert writer.scalars[0][1] == pytest.approx(expected)
    assert writer.scalars[0][2] == 7
    assert actor.training is True
    assert actor.model.logits.grad is None


def test_eval_rejects_zero_supervised_tokens_and_restores_train_mode():
    strategy = _Strategy(accumulated_gradient=1)
    actor = _Actor()
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.05)
    trainer = SFTTrainer(actor, strategy, optimizer, [], None, _Scheduler(optimizer, []))
    batch = _batch([[0, 1, 2]], [[1, 1, 1]], [[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="no supervised tokens"):
        trainer.evaluate(_Loader([batch]))

    assert actor.training is True


def test_scheduler_does_not_advance_when_engine_optim_step_fails():
    strategy = _Strategy(accumulated_gradient=1)
    actor = _Actor()
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.05)
    scheduler = _Scheduler(optimizer, strategy.events)
    trainer = SFTTrainer(
        actor,
        strategy,
        optimizer,
        _Loader([_batch([[0, 1, 2]], [[1, 1, 1]], [[1.0, 1.0, 0.0]])]),
        None,
        scheduler,
        max_epochs=1,
    )

    def fail_optim_step():
        raise RuntimeError("optimizer failed")

    trainer.engine.optim_step = fail_optim_step
    with pytest.raises(RuntimeError, match="optimizer failed"):
        trainer.fit(_args(batch_size=1), num_update_steps_per_epoch=1)

    assert scheduler.step_calls == 0
