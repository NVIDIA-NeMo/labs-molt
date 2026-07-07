from contextlib import ExitStack
from types import SimpleNamespace

import torch

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

    def is_rank_0(self):
        return True

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


def test_backward_applies_cp_size_grad_compensation():
    # FSDP averages grads over dp_cp = dp_size * cp_size, while the loss is
    # normalized for dp_size only. backward() must multiply the backward loss by
    # cp_size so the post-FSDP gradient is correct; the gather's autograd slices
    # (does not sum) grads across CP. Plain nn.Module => no FSDP grad sync, so we
    # observe the raw cp_size factor on a leaf param's grad.
    model = torch.nn.Linear(1, 1)

    for cp_size, expected in ((1, 3.0), (2, 6.0), (4, 12.0)):
        strat = _bare_strategy(cp_size)
        w = torch.tensor([1.0], requires_grad=True)
        loss = (w * 3.0).sum()
        strat.backward(loss, model, optimizer=None)
        assert w.grad.item() == expected, f"cp_size={cp_size}: {w.grad.item()} != {expected}"
