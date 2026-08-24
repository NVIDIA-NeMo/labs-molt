# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from nemo_automodel.engine import Engine

from molt.trainer.sft_trainer import SFTTrainer


class _Strategy:
    def __init__(self, accumulated_gradient=2):
        self.args = SimpleNamespace(
            logger=SimpleNamespace(
                wandb=SimpleNamespace(key=None, org=None, project=None, group=None, run_name="test"),
                tensorboard_dir=None,
            )
        )
        self.accumulated_gradient = accumulated_gradient
        self.cp_size = 1
        self.dp_size = 1
        self.events = []
        self.messages = []
        self.reductions = []

    def is_rank_0(self):
        return True

    def print(self, *args, **kwargs):
        self.messages.append(" ".join(str(arg) for arg in args))

    def global_token_count(self, value):
        return value.float()

    def all_reduce(self, data, op="mean"):
        self.reductions.append((data.detach().clone(), op))
        return data

    def _maybe_debug_grad_stats(self, model, name):
        self.events.append("debug")


class _RawLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.3, 0.1, 0.7]))

    def forward(self, input_ids, attention_mask=None):
        assert attention_mask is not None
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, -1))


class _Actor(torch.nn.Module):
    def __init__(self, engine):
        super().__init__()
        self.model = engine

    def forward(self, input_ids, attention_mask=None, **_kwargs):
        logits = self.model(input_ids, attention_mask=attention_mask).logits
        log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
        log_probs = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        return {"log_probs": log_probs}


class _Scheduler:
    def __init__(self, events):
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


def _batch(tokens, loss_mask):
    input_ids = torch.tensor(tokens, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "loss_mask": torch.tensor(loss_mask, dtype=torch.bool),
        "mm_train_inputs": None,
    }


def _trainer(batches, *, accumulated_gradient=2, eval_batches=None):
    strategy = _Strategy(accumulated_gradient)
    raw_model = _RawLM()
    optimizer = torch.optim.SGD(raw_model.parameters(), lr=0.05)
    scheduler = _Scheduler(strategy.events)
    engine = Engine(
        raw_model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        max_grad_norm=None,
        gradient_accumulation_steps=accumulated_gradient,
    )
    actor = _Actor(engine)
    trainer = SFTTrainer(actor, strategy, optimizer, _Loader(batches), eval_batches, scheduler, max_epochs=1)
    return trainer, actor, raw_model, optimizer, scheduler, strategy


def _args(batch_size=3):
    return SimpleNamespace(
        train=SimpleNamespace(batch_size=batch_size),
        eval=SimpleNamespace(steps=-1),
        ckpt=SimpleNamespace(save_steps=-1),
        logger=SimpleNamespace(logging_steps=1),
    )


def _reference_loss(model, batches):
    loss_sum = torch.zeros(())
    token_sum = torch.zeros(())
    for batch in batches:
        logits = model(batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        log_probs = F.log_softmax(logits[:, :-1], dim=-1)
        log_probs = log_probs.gather(-1, batch["input_ids"][:, 1:].unsqueeze(-1)).squeeze(-1)
        mask = batch["loss_mask"][:, :-1]
        loss_sum += torch.where(mask, -log_probs, 0.0).sum()
        token_sum += mask.sum()
    return loss_sum / token_sum


def test_sft_matches_one_full_window_update():
    batches = [
        _batch([[0, 1, 2], [3, 2, 1]], [[False, True, False], [True, True, False]]),
        _batch([[1, 3, 2]], [[True, True, False]]),
    ]
    trainer, actor, raw_model, _, scheduler, strategy = _trainer(batches)
    initial = raw_model.state_dict()
    logged = []
    trainer.save_logs_and_checkpoints = lambda args, step, bar, logs=None, states=None: logged.append(
        (step, dict(logs), dict(states))
    )

    reference = _RawLM()
    reference.load_state_dict(initial)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    expected_loss = _reference_loss(reference, batches)
    expected_loss.backward()
    reference_optimizer.step()

    trainer.fit(_args(), num_update_steps_per_epoch=1)

    assert strategy.events == ["debug", "scheduler"]
    assert scheduler.step_calls == 1
    assert strategy.messages[-1] == "[SFT] backend=engine"
    assert logged[0][0] == 1
    assert logged[0][2] == {"consumed_samples": 3}
    assert logged[0][1]["sft_loss"] == pytest.approx(expected_loss.item())
    torch.testing.assert_close(raw_model.logits, reference.logits)
    assert actor.model.get_global_grad_norm() is not None


def test_eval_reports_one_global_token_mean_and_restores_mode():
    batches = [
        _batch([[0, 1, 2]], [[True, True, False]]),
        _batch([[3, 2, 1]], [[True, False, False]]),
    ]
    trainer, actor, raw_model, _, _, strategy = _trainer([], accumulated_gradient=1, eval_batches=_Loader(batches))
    writer = _Writer()
    trainer._tensorboard = writer
    expected = _reference_loss(raw_model, batches).item()

    trainer.evaluate(_Loader(batches), steps=7)

    assert len(strategy.reductions) == 1
    assert strategy.reductions[0][0].shape == (2,)
    assert strategy.reductions[0][1] == "sum"
    assert writer.scalars[0][0] == "eval/eval sft_loss"
    assert writer.scalars[0][1] == pytest.approx(expected)
    assert writer.scalars[0][2] == 7
    assert actor.training is True
    assert raw_model.logits.grad is None


def test_eval_rejects_an_all_masked_dataset_and_restores_mode():
    batch = _batch([[0, 1, 2]], [[False, False, False]])
    trainer, actor, _, _, _, _ = _trainer([], accumulated_gradient=1)
    actor.eval()

    with pytest.raises(ValueError, match="no supervised tokens"):
        trainer.evaluate(_Loader([batch]))

    assert actor.training is False
