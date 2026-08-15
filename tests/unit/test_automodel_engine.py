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

"""The first MOLT policy slice through AutoModel's real Datum and Engine APIs."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.datasets.datum import Datum
from nemo_automodel.components.training.engine import Engine

from molt.models import PolicyLoss, datum_policy_loss


class TinyLM(nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids, **_):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def _datum(input_ids, weights, old_logprobs, advantages):
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    return Datum(
        input_ids=input_ids,
        loss_fn_inputs={
            "target_tokens": torch.roll(input_ids, shifts=-1),
            "weights": torch.tensor(weights, dtype=torch.float),
            "logprobs": torch.tensor(old_logprobs, dtype=torch.float),
            "advantages": torch.tensor(advantages, dtype=torch.float),
        },
    )


def _datum_window():
    return [
        [
            _datum([1, 2, 3, 4, 5], [0, 1, 0, 1, 0], [-2.2, -2.0, -1.8, -2.1, 0], [0, 1, 0, -0.4, 0]),
            _datum([6, 7, 8], [1, 1, 0], [-2.4, -2.3, 0], [0.3, 0.8, 0]),
        ],
        [_datum([9, 10, 11, 12], [1, 0, 1, 0], [-2.1, -1.7, -2.5, 0], [-0.2, 0, 0.6, 0])],
    ]


def _manual_window_loss(model, datum_window, policy_loss):
    numerator = torch.zeros(())
    denominator = 0.0
    for microbatch in datum_window:
        for datum in microbatch:
            logits = model(datum.input_ids.unsqueeze(0)).logits.squeeze(0).float()
            targets = datum.loss_fn_inputs["target_tokens"]
            new_logprobs = F.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            weights = datum.loss_fn_inputs["weights"]
            per_token_loss, *_ = policy_loss(
                new_logprobs,
                datum.loss_fn_inputs["logprobs"],
                datum.loss_fn_inputs["advantages"],
                action_mask=weights,
                reduce=False,
            )
            numerator = numerator + (per_token_loss * weights).sum()
            denominator += float(weights.sum())
    return numerator / denominator


@pytest.mark.parametrize("packed", [False, True])
def test_real_engine_matches_manual_policy_loss_and_gradients(packed):
    torch.manual_seed(7)
    engine_model = TinyLM()
    manual_model = TinyLM()
    manual_model.load_state_dict(engine_model.state_dict())
    engine = Engine(
        Engine.Config(pack_datums=packed, defer_fsdp_grad_sync=False, max_grad_norm=10.0),
        model_parts=[engine_model],
    )
    policy_loss = PolicyLoss()
    metric_sink = []

    output = engine.forward_backward(
        _datum_window(),
        loss_fn=datum_policy_loss,
        loss_kwargs={"policy_loss": policy_loss, "metric_sink": metric_sink},
    )
    expected = _manual_window_loss(manual_model, _datum_window(), policy_loss)
    expected.backward()

    torch.testing.assert_close(output.loss, expected.detach())
    assert [row.numel() for row in output.logprobs] == [5, 3, 4]
    assert len(metric_sink) == 3
    for engine_param, manual_param in zip(engine_model.parameters(), manual_model.parameters()):
        torch.testing.assert_close(engine_param.grad, manual_param.grad)


def test_datum_policy_loss_rejects_output_count_mismatch():
    datums = _datum_window()[0]
    output = SimpleNamespace(logprobs=[torch.zeros(datums[0].seq_len)])
    with pytest.raises(ValueError, match="1 logprob rows for 2 Datums"):
        datum_policy_loss(output, datums, policy_loss=PolicyLoss())


def test_unreduced_datum_loss_matches_legacy_window_reduction():
    datum_window = _datum_window()
    policy_loss = PolicyLoss()
    total_tokens = sum(float(datum.loss_fn_inputs["weights"].sum()) for mb in datum_window for datum in mb)

    datum_rows = []
    datum_objective = torch.zeros(())
    offset = 0.0
    for microbatch in datum_window:
        rows = []
        for datum in microbatch:
            row = (torch.linspace(-3.0, -1.0, datum.seq_len) + offset).requires_grad_()
            rows.append(row)
            datum_rows.append(row)
            offset += 0.1
        losses = datum_policy_loss(SimpleNamespace(logprobs=rows), microbatch, policy_loss=policy_loss)
        datum_objective = (
            datum_objective
            + sum((loss * datum.loss_fn_inputs["weights"]).sum() for loss, datum in zip(losses, microbatch))
            / total_tokens
        )
    datum_grads = torch.autograd.grad(datum_objective, datum_rows)

    legacy_rows = [row.detach().clone().requires_grad_() for row in datum_rows]
    legacy_objective = torch.zeros(())
    row_offset = 0
    for microbatch in datum_window:
        rows = legacy_rows[row_offset : row_offset + len(microbatch)]
        row_offset += len(microbatch)
        width = max(datum.seq_len for datum in microbatch)
        new_logprobs = torch.stack([F.pad(row, (0, width - row.numel())) for row in rows])
        old_logprobs = torch.stack(
            [F.pad(datum.loss_fn_inputs["logprobs"], (0, width - datum.seq_len)) for datum in microbatch]
        )
        advantages = torch.stack(
            [F.pad(datum.loss_fn_inputs["advantages"], (0, width - datum.seq_len)) for datum in microbatch]
        )
        weights = torch.stack(
            [F.pad(datum.loss_fn_inputs["weights"], (0, width - datum.seq_len)) for datum in microbatch]
        )
        reduced, *_ = policy_loss(
            new_logprobs,
            old_logprobs,
            advantages,
            action_mask=weights,
            batch_num_tokens=torch.tensor(total_tokens),
        )
        legacy_objective = legacy_objective + reduced
    legacy_grads = torch.autograd.grad(legacy_objective, legacy_rows)

    torch.testing.assert_close(datum_objective, legacy_objective)
    for datum_grad, legacy_grad in zip(datum_grads, legacy_grads):
        torch.testing.assert_close(datum_grad, legacy_grad)


def test_experience_to_datums_preserves_ragged_next_token_alignment():
    pytest.importorskip("ray")
    from molt.trainer.algorithm.experience import experience_to_datums

    experience = SimpleNamespace(
        sequences=torch.tensor([[1, 2, 3, 4, 5, 0], [6, 0, 8, 0, 0, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0]]),
        action_mask=torch.tensor([[0, 1, 0, 1, 0], [1, 0, 0, 0, 0]], dtype=torch.bool),
        action_log_probs=torch.arange(10, dtype=torch.float).reshape(2, 5),
        advantages=torch.arange(10, 20, dtype=torch.float).reshape(2, 5),
        rollout_log_probs=None,
        routed_experts=None,
        mm_train_inputs=[],
    )

    datums = experience_to_datums(experience)

    assert [datum.input_ids.tolist() for datum in datums] == [[1, 2, 3, 4, 5], [6, 0, 8]]
    assert [datum.loss_fn_inputs["target_tokens"].tolist() for datum in datums] == [
        [2, 3, 4, 5, 1],
        [0, 8, 6],
    ]
    assert [datum.loss_fn_inputs["weights"].tolist() for datum in datums] == [
        [0, 1, 0, 1, 0],
        [1, 0, 0],
    ]
    assert datums[0].loss_fn_inputs["logprobs"].tolist() == [0, 1, 2, 3, 0]
    assert datums[1].loss_fn_inputs["advantages"].tolist() == [15, 16, 0]

    noncontiguous = SimpleNamespace(**vars(experience))
    noncontiguous.attention_mask = experience.attention_mask.clone()
    noncontiguous.attention_mask[1] = torch.tensor([1, 0, 1, 0, 0, 0])
    with pytest.raises(ValueError, match="contiguous right padding"):
        experience_to_datums(noncontiguous)

    wrong_shape = SimpleNamespace(**vars(experience))
    wrong_shape.advantages = experience.advantages[:, :-1]
    with pytest.raises(ValueError, match="advantages must have shape"):
        experience_to_datums(wrong_shape)


def test_policy_trainer_submits_one_complete_window_and_steps_once(monkeypatch):
    pytest.importorskip("ray")
    from molt.trainer.algorithm.experience import Experience
    from molt.trainer.workers.policy_actor import PolicyTrainer

    def make_experience(tokens, action_mask, reward):
        sequence = torch.tensor([tokens])
        steps = sequence.shape[1] - 1
        return Experience(
            sequences=sequence,
            attention_mask=torch.ones_like(sequence),
            action_mask=torch.tensor([action_mask], dtype=torch.bool),
            action_log_probs=torch.zeros(1, steps),
            rollout_log_probs=torch.zeros(1, steps),
            advantages=torch.ones(1, steps),
            response_length=torch.tensor([sum(action_mask)]),
            total_length=torch.tensor([len(tokens)]),
            info={"reward": torch.tensor([reward])},
        )

    experiences = [
        make_experience([1, 2, 3, 4], [1, 0, 1], 0.5),
        make_experience([5, 6, 7], [1, 0], 1.0),
    ]

    class ReplayBuffer:
        sample_batch_size = 1
        cpu_offload = True

        def __len__(self):
            return len(experiences)

        def __getitem__(self, index):
            return experiences[index]

        @staticmethod
        def collate_fn(batch):
            return batch[0]

    class Scheduler:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

        @staticmethod
        def get_last_lr():
            return [0.01]

    args = SimpleNamespace(
        train=SimpleNamespace(dynamic_batch_enable=False, force_on_policy=False),
        fsdp=SimpleNamespace(cp_size=1, tp_size=1),
    )
    scheduler = Scheduler()

    class Strategy:
        accumulated_gradient = 2

        def __init__(self):
            self.args = args
            self.optimizer_steps = 0
            self.saw_nonzero_grad = False
            self.last_grad_norm = 0.0

        @staticmethod
        def is_rank_0():
            return False

        @staticmethod
        def print(*_):
            return None

        def optimizer_step(self, optimizer, model, scheduler_arg, **_kwargs):
            grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            assert grads
            self.saw_nonzero_grad = any(torch.count_nonzero(grad).item() for grad in grads)
            assert self.saw_nonzero_grad
            self.last_grad_norm = float(torch.linalg.vector_norm(torch.stack([grad.norm() for grad in grads])))
            self.optimizer_steps += 1
            optimizer.step()
            scheduler_arg.step()
            optimizer.zero_grad(set_to_none=True)

        def get_grad_norm(self, _model):
            return self.last_grad_norm

        @staticmethod
        def all_reduce(value, op="mean"):
            return value

        @staticmethod
        def compute_perf_metrics(*_):
            return {}

        @staticmethod
        def global_token_count(_):
            raise AssertionError("the Engine branch must own the window denominator")

    class Actor(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = TinyLM()
            self.train_calls = 0

        def train(self, mode=True):
            self.train_calls += 1
            return super().train(mode)

    class RecordingEngine:
        def __init__(self, model):
            self.calls = []
            self.engine = Engine(
                Engine.Config(pack_datums=False, defer_fsdp_grad_sync=False),
                model_parts=[model],
            )

        def forward_backward(self, datum_window, loss_fn, *, loss_kwargs):
            self.calls.append(datum_window)
            return self.engine.forward_backward(datum_window, loss_fn=loss_fn, loss_kwargs=loss_kwargs)

    trainer = object.__new__(PolicyTrainer)
    trainer.strategy = Strategy()
    trainer.args = args
    trainer.max_epochs = 1
    trainer.replay_buffer = ReplayBuffer()
    trainer.dataloader_pin_memory = False
    trainer.actor = Actor()
    trainer.actor_optim = torch.optim.SGD(trainer.actor.parameters(), lr=0.05)
    trainer.actor_scheduler = scheduler
    trainer.actor_loss_fn = PolicyLoss()
    trainer._automodel_engine = RecordingEngine(trainer.actor.model)
    trainer._mfu = None
    parameters_before = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
    reference_model = TinyLM()
    reference_model.load_state_dict(trainer.actor.model.state_dict())
    reference_optim = torch.optim.SGD(reference_model.parameters(), lr=0.05)

    monkeypatch.delenv("MOLT_DUMP_ROLLOUT_LOGPROBS", raising=False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)

    trainer.policy_train(kl_ctl=0.0)

    assert len(trainer._automodel_engine.calls) == 1
    assert [len(microbatch) for microbatch in trainer._automodel_engine.calls[0]] == [1, 1]
    assert trainer.strategy.optimizer_steps == 1
    assert trainer.strategy.saw_nonzero_grad
    assert scheduler.steps == 1
    assert trainer.actor.train_calls == 1
    assert any(not torch.equal(before, after) for before, after in zip(parameters_before, trainer.actor.parameters()))
    reference_loss = _manual_window_loss(reference_model, trainer._automodel_engine.calls[0], trainer.actor_loss_fn)
    reference_loss.backward()
    reference_optim.step()
    for actual, expected in zip(trainer.actor.model.parameters(), reference_model.parameters()):
        torch.testing.assert_close(actual, expected)

    parameters_after_step = [parameter.detach().clone() for parameter in trainer.actor.parameters()]
    experiences[0].action_mask.zero_()
    with pytest.raises(ValueError, match="action tokens in every Experience"):
        trainer.policy_train(kl_ctl=0.0)

    assert len(trainer._automodel_engine.calls) == 1
    assert trainer.strategy.optimizer_steps == 1
    assert scheduler.steps == 1
    for after_step, parameter in zip(parameters_after_step, trainer.actor.parameters()):
        torch.testing.assert_close(after_step, parameter)
