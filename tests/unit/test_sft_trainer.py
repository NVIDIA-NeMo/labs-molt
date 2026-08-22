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

from contextlib import ExitStack
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from molt.trainer import sft_trainer as sft_trainer_module
from molt.trainer.sft_trainer import SFTTrainer


class _Strategy:
    def __init__(self):
        self.args = SimpleNamespace(
            model=SimpleNamespace(aux_loss_coef=0.0),
            logger=SimpleNamespace(
                wandb=SimpleNamespace(key=None, org=None, project=None, group=None, run_name="test"),
                tensorboard_dir=None,
            ),
        )
        self.cp_size = 1
        self.dp_size = 1
        self.messages = []

    def is_rank_0(self):
        return True

    def print(self, *args, **kwargs):
        self.messages.append(" ".join(str(arg) for arg in args))

    def global_token_count(self, mask):
        return mask.sum()

    def all_reduce(self, data, op="mean"):
        return data

    def backward(self, *args, **kwargs):
        raise AssertionError("eval microbatch path must not run backward")

    def optimizer_step(self, *args, **kwargs):
        raise AssertionError("eval microbatch path must not step the optimizer")

    def get_grad_norm(self, model):
        return 0.0


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))
        self.seen_mm_inputs = None
        self.seen_cp_context_stack = "unset"
        self.cp_context_closed = False

    def forward(
        self,
        input_ids,
        attention_mask=None,
        cp_context_stack=None,
        return_entropy=False,
        **mm_inputs,
    ):
        # Mirror Actor.forward's signature: cp_context_stack is a named param, so
        # it must not leak into mm_inputs.
        self.seen_mm_inputs = mm_inputs
        self.seen_cp_context_stack = cp_context_stack
        if cp_context_stack is not None:
            # Stand in for the CP train context the real Actor parks here; the
            # trainer must close it (after backward) to fire this callback.
            cp_context_stack.callback(lambda: setattr(self, "cp_context_closed", True))
        log_probs = self.param + torch.zeros(input_ids.shape[0], input_ids.shape[1] - 1, device=input_ids.device)
        # Single named output dict (matches the real Actor.forward).
        return {"log_probs": log_probs}


class _TrainStrategy(_Strategy):
    """Strategy that permits backward/optimizer_step and records the CP loss scale."""

    def __init__(self, cp_size=1):
        super().__init__()
        self.cp_size = cp_size
        self.dp_size = 1
        self.dp_cp_size = cp_size
        self.backward_calls = 0
        self.optimizer_steps = 0

    def backward(self, loss, model, optimizer, **kwargs):
        self.backward_calls += 1
        loss.backward()

    def optimizer_step(self, optimizer, model, scheduler, **kwargs):
        self.optimizer_steps += 1


def _make_trainer(model=None, strategy=None):
    strategy = strategy or _Strategy()
    model = model or _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optimizer,
        train_dataloader=[],
        eval_dataloader=None,
        scheduler=scheduler,
    )


def test_sft_eval_microbatch_path_preserves_loss_mask_and_mm_inputs():
    model = _Model()
    trainer = _make_trainer(model)

    batch = (
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, dtype=torch.long),
        torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
        {"pixel_values": torch.ones(1, 3)},
    )

    prepared, batch_num_tokens = trainer._prepare_accum_window([batch], torch.device("cpu"))
    inputs, attention_mask, shifted_loss_mask, mm_inputs = prepared[0]

    torch.testing.assert_close(inputs, torch.tensor([[1, 2, 3, 4]]))
    torch.testing.assert_close(attention_mask, torch.ones(1, 4, dtype=torch.long))
    torch.testing.assert_close(shifted_loss_mask, torch.tensor([[0.0, 1.0, 1.0]]))
    torch.testing.assert_close(batch_num_tokens, torch.tensor(2.0))
    assert "pixel_values" in mm_inputs

    logs, loss = trainer._run_microbatch(prepared[0], batch_num_tokens, accum_steps=1, backward=False)

    assert loss == 0.0
    assert logs == {"sft_loss": 0.0}
    assert "pixel_values" in model.seen_mm_inputs
    # cp_context_stack is a named Actor arg, so it must not leak into mm_inputs.
    assert "cp_context_stack" not in model.seen_mm_inputs


def _single_batch():
    return (
        torch.tensor([[1, 2, 3, 4]]),
        torch.ones(1, 4, dtype=torch.long),
        torch.tensor([[0.0, 1.0, 1.0, 0.0]]),
        {},
    )


def test_sft_noncp_microbatch_passes_no_cp_context():
    model = _Model()
    trainer = _make_trainer(model, _TrainStrategy(cp_size=1))

    prepared, batch_num_tokens = trainer._prepare_accum_window([_single_batch()], torch.device("cpu"))
    # Unified prepared format is always a tuple now (no dict CP branch).
    assert isinstance(prepared[0], tuple)
    trainer._run_microbatch(prepared[0], batch_num_tokens, accum_steps=1, backward=True)

    assert model.seen_cp_context_stack is None


def test_sft_cp_microbatch_delegates_to_actor_and_closes_context():
    # CP is owned by the Actor: the trainer must pass a real ExitStack (so the
    # Actor can park its CP train context on it) and close it after backward.
    model = _Model()
    strategy = _TrainStrategy(cp_size=2)
    trainer = _make_trainer(model, strategy)

    prepared, _ = trainer._prepare_accum_window([_single_batch()], torch.device("cpu"))
    # No trainer-side CP padding/sharding anymore: the dense sequence flows to the Actor.
    inputs, _, shifted_loss_mask, _ = prepared[0]
    assert inputs.shape[1] == 4
    assert shifted_loss_mask.shape[1] == 3

    trainer._run_microbatch(prepared[0], torch.tensor(2.0), accum_steps=1, backward=True)

    assert isinstance(model.seen_cp_context_stack, ExitStack)
    assert model.cp_context_closed is True  # closed after backward
    assert strategy.backward_calls == 1
    assert strategy.optimizer_steps == 1


def test_sft_cp_loss_value_scale_uses_dp_size_not_dp_cp():
    # The loss-VALUE scale passed to the loss fn must be dp_size (CP ranks share
    # the sample and each computes the full gathered loss), NOT dp_cp_size — so
    # the reported/logged loss (all_reduce-mean over the world) stays the true
    # global token-mean. The CP *gradient* compensation for FSDP averaging over
    # the extra dp_cp dim is applied separately in FsdpStrategy.backward
    # (loss *= cp_size); see test_backward_applies_cp_size_grad_compensation.
    model = _Model()
    strategy = _TrainStrategy(cp_size=2)  # dp_size=1, dp_cp_size=2
    trainer = _make_trainer(model, strategy)

    captured = {}
    real_loss_fn = trainer.loss_fn

    def _recording_loss_fn(*args, **kwargs):
        captured.update(kwargs)
        return real_loss_fn(*args, **kwargs)

    trainer.loss_fn = _recording_loss_fn

    prepared, batch_num_tokens = trainer._prepare_accum_window([_single_batch()], torch.device("cpu"))
    trainer._run_microbatch(prepared[0], batch_num_tokens, accum_steps=1, backward=True)

    assert captured["dp_size"] == strategy.dp_size == 1


def _bare_strategy(cp_size, accumulated_gradient=1):
    # Build an FsdpStrategy without the full distributed bring-up: only the
    # attributes FsdpStrategy.backward touches are needed.
    from molt.trainer.fsdp.strategy import FsdpStrategy

    strat = FsdpStrategy.__new__(FsdpStrategy)
    strat.cp_size = cp_size
    strat.accumulated_gradient = accumulated_gradient
    strat.dp_size = 1
    strat.dp_cp_size = cp_size
    strat.moe_mesh = None
    return strat


def test_backward_does_not_scale_main_loss_by_cp_size():
    # The CP gradient factor comes from the sharder's gather_token_tensor, whose
    # differentiable all-gather SUMS grads across CP (local grad = cp_size× the
    # replicated-loss grad); that cancels FSDP's mean over dp_cp. So backward()
    # must NOT itself multiply the main loss by cp_size — a plain leaf param's
    # grad is cp_size-invariant. (The ungathered MoE aux loss keeps its cp factor
    # via main_loss_backward_scale, exercised only when moe_mesh is set.)
    model = torch.nn.Linear(1, 1)

    for cp_size in (1, 2, 4):
        strat = _bare_strategy(cp_size)
        w = torch.tensor([1.0], requires_grad=True)
        loss = (w * 3.0).sum()
        strat.backward(loss, model, optimizer=None)
        assert w.grad.item() == 3.0, f"cp_size={cp_size}: {w.grad.item()} != 3.0 (no cp scaling)"


def test_backward_divides_main_loss_by_accumulated_gradient():
    # Gradient accumulation still averages the per-microbatch loss over the window;
    # cp_size does not enter (see above). w.grad = 3 / accum, not 3 * cp_size / accum.
    model = torch.nn.Linear(1, 1)
    strat = _bare_strategy(cp_size=2, accumulated_gradient=4)
    w = torch.tensor([1.0], requires_grad=True)
    loss = (w * 3.0).sum()
    strat.backward(loss, model, optimizer=None)
    assert w.grad.item() == 3.0 / 4


def test_fit_logs_the_window_token_mean_not_a_microbatch_fraction():
    # Every microbatch loss is divided by the WHOLE window's token count, so one alone is
    # a 1/accum_steps fraction. The step metric must be their sum (the quantity AutoModel's
    # train_ft reports as "loss", and the units eval already uses), while the per-microbatch
    # value stays a plain per-token mean so the progress bar shows a real loss.
    accum, batch_size, seqlen = 4, 2, 9
    torch.manual_seed(0)
    log_probs = [-torch.rand(batch_size, seqlen - 1).abs() - 0.1 for _ in range(accum)]

    class _Scripted(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(()))
            self.idx = 0

        def forward(self, input_ids, attention_mask=None, cp_context_stack=None, **mm_inputs):
            out = log_probs[self.idx] + self.param
            self.idx += 1
            return {"log_probs": out}

    class _Loader(list):
        sampler = None  # fit() only isinstance-checks this against DistributedSampler

    strategy = _TrainStrategy()
    strategy.accumulated_gradient = accum
    seen = []
    # Snapshot: fit() overwrites logs_dict["sft_loss"] in place at the window boundary.
    strategy.all_reduce = lambda data, op="mean": (seen.append(dict(data) if isinstance(data, dict) else data), data)[
        1
    ]

    trainer = _make_trainer(_Scripted(), strategy)
    ones = torch.ones(batch_size, seqlen, dtype=torch.long)
    trainer.train_dataloader = _Loader([(ones, ones, torch.ones(batch_size, seqlen), {})] * accum)
    trainer.epochs = 1
    logged = []
    trainer.save_logs_and_checkpoints = lambda a, gs, bar, logs=None, states=None: logged.append(dict(logs))

    trainer.fit(
        SimpleNamespace(
            train=SimpleNamespace(batch_size=batch_size * accum),
            eval=SimpleNamespace(steps=-1),
            ckpt=SimpleNamespace(save_steps=-1),
            logger=SimpleNamespace(logging_steps=1),
        ),
        consumed_samples=0,
        num_update_steps_per_epoch=1,
    )

    window_token_mean = sum((-lp).sum().item() for lp in log_probs) / (accum * batch_size * (seqlen - 1))
    assert len(logged) == 1
    assert abs(logged[0]["sft_loss"] - window_token_mean) < 1e-6, logged[0]["sft_loss"]
    assert "loss_mean" not in logged[0]  # one loss key, not a correct one next to a wrong one
    # Each microbatch reported its OWN per-token mean, on the same scale as the step metric.
    per_microbatch = [d["sft_loss"] for d in seen if isinstance(d, dict) and "sft_loss" in d]
    assert per_microbatch == [(-lp).mean().item() for lp in log_probs]


class _RawLM(torch.nn.Module):
    """Tiny raw backbone returning logits with shape ``[batch, sequence, vocab]``."""

    def __init__(self, tensor_output=False):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.3, 0.1, 0.7]))
        self.config = SimpleNamespace(pad_token_id=0)
        self.tensor_output = tensor_output

    def forward(self, input_ids, attention_mask=None, position_ids=None):
        assert position_ids is not None
        del attention_mask, position_ids
        logits = self.logits.expand(*input_ids.shape, -1)
        return logits if self.tensor_output else {"logits": logits}


class _EngineActor(torch.nn.Module):
    """Minimal Actor boundary; Engine must call ``model`` rather than this wrapper."""

    is_vlm = False
    packing_samples = False

    def __init__(self, model=None):
        super().__init__()
        self.model = model or _RawLM()

    def forward(self, *args, **kwargs):
        raise AssertionError("Engine SFT must bypass the outer Actor wrapper")


class _EngineStrategy(_TrainStrategy):
    def __init__(self):
        super().__init__()
        self.args.optim = "adam"
        self.accumulated_gradient = 2
        self.cpu_offload = False
        self.offload_optimizer = False
        self.tp_size = self.cp_size = self.ep_size = self.pp_size = 1
        self.sequence_parallel = False
        self.device_mesh = self.moe_mesh = None
        self.events = []

    def global_token_count(self, mask):
        raise AssertionError("Engine owns the global loss denominator")

    def all_reduce(self, data, op="mean"):
        raise AssertionError("Engine results must not be reduced again by the trainer")

    def backward(self, *args, **kwargs):
        raise AssertionError("Engine owns backward")

    def optimizer_step(self, *args, **kwargs):
        raise AssertionError("Engine owns the optimizer step")

    def _maybe_debug_grad_stats(self, model, name):
        self.events.append("debug")


class _CountingScheduler:
    def __init__(self, optimizer, events):
        self.optimizer = optimizer
        self.events = events
        self.step_calls = 0

    def step(self):
        self.events.append("scheduler")
        self.step_calls += 1

    def get_last_lr(self):
        return [self.optimizer.param_groups[0]["lr"]]


class _Loader(list):
    sampler = None


def _engine_batch(tokens, weights):
    tokens = torch.tensor([tokens])
    return tokens, torch.ones_like(tokens), torch.tensor([weights]), {}


@pytest.mark.parametrize("tensor_output", [False, True])
def test_engine_sft_matches_full_window_weighted_update(tensor_output):
    strategy = _EngineStrategy()
    actor = _EngineActor(_RawLM(tensor_output=tensor_output))
    initial = actor.model.state_dict()
    optimizer = torch.optim.AdamW(actor.parameters(), lr=0.05, weight_decay=0.0)
    scheduler = _CountingScheduler(optimizer, strategy.events)
    batches = [
        _engine_batch([0, 1, 2, 3], [0.0, 0.5, 0.0, 9.0]),
        _engine_batch([3, 2, 1, 0], [1.0, 1.0, 0.0, 7.0]),
    ]

    trainer = SFTTrainer(
        model=actor,
        strategy=strategy,
        optim=optimizer,
        train_dataloader=_Loader(batches),
        eval_dataloader=None,
        scheduler=scheduler,
        max_norm=0,
        max_epochs=1,
    )
    assert trainer.engine is not None
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
    trainer.save_logs_and_checkpoints = lambda a, gs, bar, logs=None, states=None: logged.append(
        (gs, dict(logs), dict(states))
    )

    reference = _RawLM()
    reference.load_state_dict(initial)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.05, weight_decay=0.0)
    numerator = torch.zeros(())
    denominator = torch.zeros(())
    for inputs, attention_mask, weights, _ in batches:
        logits = reference.logits.expand(*inputs.shape, -1)
        token_nll = F.cross_entropy(
            logits[:, :-1].flatten(0, 1),
            inputs[:, 1:].flatten(),
            reduction="none",
        ).view_as(inputs[:, 1:])
        weights = weights.clone()
        weights[:, -1] = 0
        numerator = numerator + (token_nll * weights[:, :-1]).sum()
        denominator = denominator + weights[:, :-1].sum()
    expected_loss = numerator / denominator
    expected_loss.backward()
    reference_optimizer.step()

    trainer.fit(
        SimpleNamespace(
            train=SimpleNamespace(batch_size=2),
            eval=SimpleNamespace(steps=-1),
            ckpt=SimpleNamespace(save_steps=-1),
            logger=SimpleNamespace(logging_steps=1),
        ),
        num_update_steps_per_epoch=1,
    )

    assert strategy.events == ["forward_backward", "debug", "optim_step", "scheduler"]
    assert scheduler.step_calls == 1
    assert len(logged) == 1
    global_step, logs, client_states = logged[0]
    assert global_step == 1
    assert client_states == {"consumed_samples": 2}
    assert logs["sft_loss"] == pytest.approx(expected_loss.item())
    assert logs["grad_norm"] == 0.0  # Engine reports zero when clipping is disabled.
    torch.testing.assert_close(engine_grads[0], reference.logits.grad)
    torch.testing.assert_close(actor.model.logits, reference.logits)


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("vlm", "VLM input preparation"),
        ("packing", "packed input preparation"),
        ("optimizer_offload", "CPU offload"),
        ("full_offload", "CPU offload"),
        ("tensor_parallel", "model parallelism"),
        ("context_parallel", "model parallelism"),
        ("expert_parallel", "model parallelism"),
        ("pipeline_parallel", "model parallelism"),
        ("sequence_parallel", "sequence parallelism"),
        ("muon", "non-Adam optimizer"),
        ("per_microbatch_sync", "per-microbatch gradient sync"),
        ("aux", "explicit auxiliary-loss logging"),
    ],
)
def test_unsupported_sft_configuration_keeps_legacy_backend(case, reason, monkeypatch):
    strategy = _EngineStrategy()
    actor = _EngineActor()
    if case == "vlm":
        actor.is_vlm = True
    elif case == "packing":
        actor.packing_samples = True
    elif case == "optimizer_offload":
        strategy.offload_optimizer = True
    elif case == "full_offload":
        strategy.cpu_offload = True
    elif case == "tensor_parallel":
        strategy.tp_size = 2
    elif case == "context_parallel":
        strategy.cp_size = 2
    elif case == "expert_parallel":
        strategy.ep_size = 2
    elif case == "pipeline_parallel":
        strategy.pp_size = 2
    elif case == "sequence_parallel":
        strategy.sequence_parallel = True
    elif case == "muon":
        strategy.args.optim = "muon"
    elif case == "per_microbatch_sync":
        monkeypatch.setenv("MOLT_DEFER_GRAD_SYNC", "0")
    elif case == "aux":
        strategy.args.model.aux_loss_coef = 0.1

    optimizer = torch.optim.AdamW(actor.parameters())
    trainer = SFTTrainer(actor, strategy, optimizer, [], None, _CountingScheduler(optimizer, []))

    assert trainer.engine is None
    assert reason in strategy.messages[-1]


def test_missing_engine_module_keeps_legacy_backend(monkeypatch):
    strategy = _EngineStrategy()
    actor = _EngineActor()
    optimizer = torch.optim.AdamW(actor.parameters())
    real_import_module = sft_trainer_module.import_module

    def missing_engine(name):
        if name == "nemo_automodel.engine":
            raise ModuleNotFoundError("No module named 'nemo_automodel.engine'", name=name)
        return real_import_module(name)

    monkeypatch.setattr(sft_trainer_module, "import_module", missing_engine)
    trainer = SFTTrainer(actor, strategy, optimizer, [], None, _CountingScheduler(optimizer, []))

    assert trainer.engine is None
    assert "Engine API unavailable" in strategy.messages[-1]


def test_broken_engine_dependency_does_not_silently_fall_back(monkeypatch):
    strategy = _EngineStrategy()
    actor = _EngineActor()
    optimizer = torch.optim.AdamW(actor.parameters())

    def broken_engine(name):
        raise ModuleNotFoundError("No module named 'engine_dependency'", name="engine_dependency")

    monkeypatch.setattr(sft_trainer_module, "import_module", broken_engine)
    with pytest.raises(ModuleNotFoundError, match="engine_dependency"):
        SFTTrainer(actor, strategy, optimizer, [], None, _CountingScheduler(optimizer, []))
