# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from importlib.util import find_spec
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("flash_attn")
pytest.importorskip("liger_kernel")
pytest.importorskip("nemo_automodel")
from transformers import Qwen3Config, Qwen3ForCausalLM

from molt.models import Actor, Critic, PolicyLoss
from molt.trainer.fsdp import FsdpStrategy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger Qwen3 integration requires CUDA")


def _save_tiny_qwen3(path):
    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
    )
    Qwen3ForCausalLM(config).save_pretrained(path)


def _strategy():
    world_size = int(os.environ["WORLD_SIZE"])
    args = SimpleNamespace(
        local_rank=int(os.environ["LOCAL_RANK"]),
        fsdp=SimpleNamespace(
            tp_size=1,
            cp_size=1,
            ep_size=1,
            pp_size=1,
            param_dtype="bf16",
            offload="none",
            sequence_parallel=False,
        ),
        actor=SimpleNamespace(gradient_checkpoint="full"),
        train=SimpleNamespace(dynamic_batch_enable=False),
    )
    strategy = FsdpStrategy(
        seed=7,
        full_determinism=False,
        micro_train_batch_size=2,
        train_batch_size=2 * world_size,
        args=args,
    )
    strategy.setup_distributed()
    return strategy


def _load(model_cls, checkpoint, strategy, packing_samples, use_liger_kernel):
    return model_cls(
        str(checkpoint),
        attn_implementation="te" if packing_samples else "sdpa",
        param_dtype="bf16",
        device_mesh=strategy.device_mesh,
        moe_mesh=strategy.moe_mesh,
        distributed_config=strategy.distributed_config,
        moe_config=strategy.moe_config,
        activation_checkpointing="full",
        packing_samples=packing_samples,
        use_liger_kernel=use_liger_kernel,
    )


def _batch():
    device = torch.device("cuda", torch.cuda.current_device())
    sequences = torch.tensor(
        [[1, 5, 6, 7, 8, 9, 10, 2], [0, 0, 1, 11, 12, 13, 14, 2]],
        device=device,
    )
    attention_mask = sequences.ne(0)
    action_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.bool, device=device)
    advantages = torch.tensor([[0.5, -0.25, 1.0, 0.75], [0.0, 0.0, -0.5, 0.25]], device=device)
    return sequences, attention_mask, action_mask, advantages


def _policy_step(model, batch, *, dp_size=1):
    sequences, attention_mask, action_mask, advantages = batch
    output = model(sequences, action_mask, attention_mask, return_entropy=True)
    old_log_probs = torch.zeros_like(output.action_log_probs)
    num_tokens = action_mask.sum() * dp_size
    loss, reported, *_ = PolicyLoss()(
        output.action_log_probs,
        old_log_probs,
        advantages,
        action_mask=action_mask,
        dp_size=dp_size,
        batch_num_tokens=num_tokens,
    )
    return output, loss, reported


def _local(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def _assert_finite(tensors):
    assert all(torch.isfinite(_local(tensor)).all() for tensor in tensors)


@pytest.mark.parametrize("packing_samples", [False, True], ids=["unpacked", "packed"])
def test_single_rank_liger_qwen3_parity(tmp_path, packing_samples):
    if os.environ.get("WORLD_SIZE") != "1" or "LOCAL_RANK" not in os.environ:
        pytest.skip("single-rank case runs under torchrun --nproc-per-node=1")
    if packing_samples and find_spec("transformer_engine") is None:
        pytest.skip("packed native Qwen3 requires transformer-engine")
    checkpoint = tmp_path / "tiny-qwen3"
    _save_tiny_qwen3(checkpoint)
    strategy = _strategy()
    baseline = _load(Actor, checkpoint, strategy, packing_samples, False)
    liger = _load(Actor, checkpoint, strategy, packing_samples, True)
    assert any(
        name.endswith(".mlp") and module.forward.__module__.startswith("liger_kernel.")
        for name, module in liger.named_modules()
    )
    batch = _batch()

    assert baseline.state_dict().keys() == liger.state_dict().keys()
    baseline_output, baseline_loss, baseline_reported = _policy_step(baseline, batch)
    liger_output, liger_loss, liger_reported = _policy_step(liger, batch)
    _assert_finite(
        [
            baseline_output.logits,
            baseline_output.action_log_probs,
            baseline_output.entropy,
            baseline_loss,
            liger_output.logits,
            liger_output.action_log_probs,
            liger_output.entropy,
            liger_loss,
        ]
    )
    torch.testing.assert_close(_local(liger_output.logits), _local(baseline_output.logits), atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(
        liger_output.action_log_probs, baseline_output.action_log_probs, atol=1e-1, rtol=1e-2
    )
    torch.testing.assert_close(liger_output.entropy, baseline_output.entropy, atol=1e-1, rtol=1e-2)
    torch.testing.assert_close(liger_reported, baseline_reported, atol=1e-2, rtol=5e-2)

    action_mask = batch[2]
    actor_reference_kl = (liger_output.action_log_probs - baseline_output.action_log_probs).masked_select(action_mask)
    _assert_finite([actor_reference_kl])

    models_and_losses = [(baseline, baseline_loss), (liger, liger_loss)]
    initial_baseline = {name: _local(param.detach()).clone() for name, param in baseline.named_parameters()}
    for model, loss in models_and_losses:
        loss.backward()
        _assert_finite(param.grad for param in model.parameters() if param.grad is not None)
    for model, _loss in models_and_losses:
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        _assert_finite(model.parameters())

    baseline_params = dict(baseline.named_parameters())
    liger_params = dict(liger.named_parameters())
    assert any(
        not torch.equal(_local(baseline_params[name]), value) for name, value in initial_baseline.items()
    ), "optimizer step did not update the baseline"
    for name in baseline_params:
        torch.testing.assert_close(
            _local(liger_params[name]), _local(baseline_params[name]), atol=1e-2, rtol=1e-2
        )

    critic = _load(Critic, checkpoint, strategy, packing_samples, True)
    critic_output = critic(batch[0], batch[2], batch[1])
    assert critic_output.action_values.shape == batch[2].shape
    _assert_finite([critic_output.action_values])


@pytest.mark.parametrize("packing_samples", [False, True], ids=["unpacked", "packed"])
def test_fsdp2_liger_qwen3_step_matches_baseline_loss(tmp_path, packing_samples):
    if os.environ.get("WORLD_SIZE") != "2" or "LOCAL_RANK" not in os.environ:
        pytest.skip("FSDP2 case runs under torchrun --nproc-per-node=2")
    if packing_samples and find_spec("transformer_engine") is None:
        pytest.skip("packed native Qwen3 requires transformer-engine")
    checkpoint = tmp_path / f"tiny-qwen3-rank-{os.environ['RANK']}"
    _save_tiny_qwen3(checkpoint)
    strategy = _strategy()
    baseline = _load(Actor, checkpoint, strategy, packing_samples, False)
    liger = _load(Actor, checkpoint, strategy, packing_samples, True)
    batch = _batch()

    _baseline_output, baseline_loss, baseline_reported = _policy_step(baseline, batch, dp_size=2)
    _liger_output, liger_loss, liger_reported = _policy_step(liger, batch, dp_size=2)
    for reported in (baseline_reported, liger_reported):
        dist.all_reduce(reported)
        reported /= dist.get_world_size()
    torch.testing.assert_close(liger_reported, baseline_reported, atol=1e-2, rtol=5e-2)

    for model, loss in ((baseline, baseline_loss), (liger, liger_loss)):
        loss.backward()
        _assert_finite(param.grad for param in model.parameters() if param.grad is not None)
        torch.optim.AdamW(model.parameters(), lr=1e-3).step()
        _assert_finite(model.parameters())
