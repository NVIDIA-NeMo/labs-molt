# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from types import SimpleNamespace

import pytest
import torch

from molt.trainer.algorithm.experience import Experience
from molt.trainer.algorithm.replay_buffer import NaiveReplayBuffer
from molt.trainer.rollout import experience_maker as experience_maker_module
from molt.trainer.rollout.experience_maker import RemoteExperienceMaker
from molt.trainer.workers.actor_group import _make_forward_batch, _split_forward_batch


def _args(
    *,
    packing=False,
    dynamic=True,
    train_budget=20,
    rollout_budget=20,
    batch_size=1,
    force_on_policy=False,
    cp_size=1,
    tp_size=1,
    pad_multiple=1,
):
    return SimpleNamespace(
        fsdp=SimpleNamespace(packing_samples=packing, cp_size=cp_size, tp_size=tp_size, ep_size=1),
        train=SimpleNamespace(
            dynamic_batch_enable=dynamic,
            dynamic_batch_pad_to_multiple=pad_multiple,
            max_tokens_per_gpu=train_budget,
            batch_size=batch_size,
            micro_batch_size=1,
            force_on_policy=force_on_policy,
        ),
        rollout=SimpleNamespace(max_tokens_per_gpu=rollout_budget, n_samples_per_prompt=1),
        algo=SimpleNamespace(
            advantage=SimpleNamespace(estimator="reinforce"),
            kl=SimpleNamespace(init_coef=1.0, use_loss=True, estimator="k1"),
        ),
        reward=SimpleNamespace(clip_range=None),
    )


def _sample(length, index, *, lazy=False):
    return Experience(
        sequences=None if lazy else torch.full((1, length), index + 1, dtype=torch.long),
        attention_mask=None if lazy else torch.ones(1, length, dtype=torch.long),
        action_mask=torch.ones(1, max(length - 1, 0), dtype=torch.bool),
        rewards=torch.tensor([float(index)]),
        total_length=torch.tensor([length]),
        index=[index],
        group_ids=[f"group-{index}"],
        rollout_ids=[f"rollout-{index}"],
        heavy_ref=f"ref-{index}" if lazy else None,
    )


class _FakeStrategy:
    def __init__(self, args, reduced_counts=None):
        self.args = args
        self.reduced_counts = reduced_counts
        self.messages = []

    def all_reduce(self, values, op):
        assert op == "max"
        if self.reduced_counts is None:
            return values
        return torch.tensor(self.reduced_counts, dtype=values.dtype, device=values.device)

    def print(self, message):
        self.messages.append(message)


def _setup_replay(monkeypatch, lengths, args, reduced_counts=None):
    model_parallel_size = args.fsdp.cp_size * args.fsdp.tp_size
    monkeypatch.setattr("molt.trainer.algorithm.replay_buffer.dist.get_world_size", lambda: model_parallel_size)
    monkeypatch.setattr(
        "molt.trainer.algorithm.replay_buffer.torch.cuda.current_device",
        lambda: torch.device("cpu"),
    )
    replay = NaiveReplayBuffer(sample_batch_size=1, cpu_offload=False, dynamic_batch=True)
    for index, length in enumerate(lengths):
        replay.append(_sample(length, index))
    strategy = _FakeStrategy(args, reduced_counts)
    replay.setup_dynamic_batch(strategy)
    return replay, strategy


class _ForwardGroup:
    duplicate_actors = 1

    def __init__(self, effective_actors, marker):
        self.effective_actors = effective_actors
        self.marker = marker
        self.calls = []

    def async_run_method_batch(self, method_name, **kwargs):
        self.calls.append((method_name, kwargs))
        key = "experiences" if method_name == "forward_batch" else "experience"
        items = kwargs[key]
        chunk_size = len(items) // self.effective_actors
        results = []
        for rank in range(self.effective_actors):
            chunk = items[rank * chunk_size : (rank + 1) * chunk_size]
            if method_name == "forward_batch":
                results.append(
                    [
                        [
                            torch.full_like(experience.action_mask, self.marker, dtype=torch.float32)
                            for experience in batch
                        ]
                        for batch in chunk
                    ]
                )
            else:
                results.append(
                    [
                        torch.full_like(experience.action_mask, self.marker, dtype=torch.float32)
                        for experience in chunk
                    ]
                )
        return results

    def async_run_method(self, method_name):
        assert method_name == "empty_cache"
        return [None] * self.effective_actors


def _maker(args, actor, reference=None, critic=None):
    return RemoteExperienceMaker(
        actor_model_group=actor,
        initial_model_group=reference,
        critic_model_group=critic,
        kl_controller=None,
        strategy=SimpleNamespace(args=args),
        tokenizer=None,
    )


def test_packed_replay_partition_order_is_unchanged(monkeypatch):
    replay, _ = _setup_replay(
        monkeypatch,
        [5, 3, 7, 2, 6],
        _args(packing=True, train_budget=10, batch_size=5),
    )

    assert replay.dynamic_indices == [[3, 4], [0, 1], [2]]
    assert replay.dynamic_optimizer_step == [0, 0, 1]


def test_padded_replay_prices_dense_footprint_and_cp_alignment(monkeypatch):
    padded, _ = _setup_replay(
        monkeypatch,
        [100, 1, 1, 1, 1],
        _args(train_budget=104, batch_size=5),
    )
    packed, _ = _setup_replay(
        monkeypatch,
        [100, 1, 1, 1, 1],
        _args(packing=True, train_budget=104, batch_size=5),
    )
    cp_aligned, _ = _setup_replay(
        monkeypatch,
        [5, 5],
        _args(train_budget=7, batch_size=2, cp_size=2),
    )

    assert padded.dynamic_indices == [[0], [1, 2, 3, 4]]
    assert packed.dynamic_indices == [[0, 1, 2, 3, 4]]
    assert cp_aligned.dynamic_indices == [[0], [1]]


def test_padded_replay_bucket_cost_matches_materialized_shape(monkeypatch):
    exact, _ = _setup_replay(
        monkeypatch,
        [1001, 700],
        _args(train_budget=2047, batch_size=2),
    )
    bucketed, _ = _setup_replay(
        monkeypatch,
        [1001, 700],
        _args(train_budget=2047, batch_size=2, pad_multiple=1024),
    )

    assert exact.dynamic_indices == [[0, 1]]
    assert bucketed.dynamic_indices == [[0], [1]]

    batch = bucketed.collate_fn([[bucketed.items[index] for index in bucketed.dynamic_indices[0]]])
    assert batch.sequences.shape == (1, 1024)
    assert batch.attention_mask.shape == (1, 1024)
    assert batch.action_mask.shape == (1, 1023)


def test_padded_replay_matches_cross_rank_count_and_preserves_windows(monkeypatch):
    replay, _ = _setup_replay(
        monkeypatch,
        [8, 7, 2, 1],
        _args(train_budget=16, batch_size=4),
        reduced_counts=[3],
    )
    assert replay.dynamic_indices == [[0], [1], [2, 3]]

    lengths = [10, 9, 2, 1, 8, 7, 2, 1]
    fixed, _ = _setup_replay(
        monkeypatch,
        lengths,
        _args(train_budget=20, batch_size=4),
    )
    on_policy, _ = _setup_replay(
        monkeypatch,
        lengths,
        _args(train_budget=20, batch_size=4, force_on_policy=True),
    )

    assert fixed.dynamic_indices == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert fixed.dynamic_optimizer_step == [0, 1, 0, 1]
    assert on_policy.dynamic_optimizer_step == [0, 0, 1]
    assert sorted(index for batch in on_policy.dynamic_indices for index in batch) == list(range(8))


def test_padded_replay_warns_for_over_budget_singletons(monkeypatch):
    replay, strategy = _setup_replay(
        monkeypatch,
        [12, 11, 1],
        _args(train_budget=10, batch_size=3),
    )

    assert replay.dynamic_indices == [[0], [1], [2]]
    assert len(strategy.messages) == 1
    assert "2 sample(s)" in strategy.messages[0]


def test_padded_forward_schedule_is_shared_and_controller_keeps_lazy_samples(monkeypatch):
    monkeypatch.setattr(experience_maker_module.ray, "get", lambda value: value)
    args = _args(rollout_budget=20)
    actor = _ForwardGroup(2, 3)
    reference = _ForwardGroup(2, 2)
    critic = _ForwardGroup(2, 1)
    samples = [_sample(length, index, lazy=True) for index, length in enumerate([10, 9, 8, 7, 6, 5])]
    refs = [sample.heavy_ref for sample in samples]

    result = _maker(args, actor, reference, critic).make_experience(samples)

    schedules = []
    for group in (actor, reference, critic):
        method_name, kwargs = group.calls[0]
        assert method_name == "forward_batch"
        schedules.append([[sample.index[0] for sample in batch] for batch in kwargs["experiences"]])
    assert schedules[0] == schedules[1] == schedules[2]
    assert schedules[0] == [[0, 1], [2], [3, 4], [5]]
    assert sorted(index for batch in schedules[0] for index in batch) == list(range(6))

    for index, experience in enumerate(result):
        assert experience.heavy_ref == refs[index]
        assert experience.sequences is None
        assert experience.values.shape == experience.action_mask.shape
        assert experience.base_action_log_probs.shape == experience.action_mask.shape
        assert experience.action_log_probs.shape == experience.action_mask.shape
        assert experience.values.unique().item() == 1
        assert experience.base_action_log_probs.unique().item() == 2
        assert experience.action_log_probs.unique().item() == 3


@pytest.mark.parametrize(("dynamic", "packing"), [(False, False), (True, True)])
def test_static_and_packed_forwards_remain_per_sample(monkeypatch, dynamic, packing):
    monkeypatch.setattr(experience_maker_module.ray, "get", lambda value: value)
    actor = _ForwardGroup(1, 3)
    samples = [_sample(5, 0), _sample(3, 1)]

    _maker(_args(dynamic=dynamic, packing=packing), actor).make_experience(samples)

    method_name, kwargs = actor.calls[0]
    assert method_name == "forward"
    assert kwargs["experience"] == samples


def test_padded_forward_warns_for_over_budget_singletons(monkeypatch):
    monkeypatch.setattr(experience_maker_module.ray, "get", lambda value: value)
    warnings = []
    monkeypatch.setattr(experience_maker_module.logger, "warning", warnings.append)
    actor = _ForwardGroup(1, 3)

    _maker(_args(rollout_budget=10), actor).make_experience(
        [_sample(12, 0), _sample(11, 1), _sample(1, 2)]
    )

    assert len(warnings) == 1
    assert "2 rollout sample(s)" in warnings[0]


def test_padded_forward_schedule_uses_shape_bucket(monkeypatch):
    monkeypatch.setattr(experience_maker_module.ray, "get", lambda value: value)
    actor = _ForwardGroup(1, 3)

    _maker(_args(rollout_budget=2047, pad_multiple=1024), actor).make_experience(
        [_sample(1001, 0), _sample(700, 1)]
    )

    method_name, kwargs = actor.calls[0]
    assert method_name == "forward_batch"
    assert [[sample.index[0] for sample in batch] for batch in kwargs["experiences"]] == [[0], [1]]


def test_worker_batch_reload_preserves_routing_vlm_and_crops_results(monkeypatch):
    first = _sample(4, 0)
    second = _sample(2, 1)
    first.routed_experts = torch.tensor([[[[1, 2, 3, 4]]]])
    first.mm_train_inputs = [{"pixel_values": torch.ones(1, 1, 2, 2)}]
    second.mm_train_inputs = [{"pixel_values": torch.ones(1, 1, 1, 1) * 2}]
    first.heavy_ref = "first"
    second.heavy_ref = "second"
    reloaded = []

    def reload(experience):
        reloaded.append(experience.heavy_ref)
        experience.heavy_ref = None
        return experience

    monkeypatch.setattr(Experience, "reload", reload)

    batch = _make_forward_batch([first, second], pad_to_multiple=8)
    outputs = _split_forward_batch(torch.arange(6).view(2, 3), [first, second])

    assert reloaded == ["first", "second"]
    assert batch.sequences.shape == (2, 8)
    assert batch.action_mask.shape == (2, 7)
    assert batch.routed_experts.shape == (2, 1, 1, 8)
    assert batch.routed_experts[0, 0, 0].tolist() == [1, 2, 3, 4, -1, -1, -1, -1]
    assert batch.routed_experts[1, 0, 0].tolist() == [-1] * 8
    assert len(batch.mm_train_inputs) == 2
    assert outputs[0].shape == (1, 3)
    assert outputs[1].shape == (1, 1)
    assert outputs[1].item() == 3


@pytest.mark.skipduringci
@pytest.mark.skipif(
    os.environ.get("MOLT_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available(),
    reason="set MOLT_RUN_GPU_TESTS=1 on a CUDA worker",
)
def test_flex_attention_padding_preserves_real_outputs_and_gradients():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    torch.manual_seed(0)
    device = torch.device("cuda")
    length, padded_length = 1001, 1024
    qkv = torch.randn(3, 1, 2, length, 64, device=device, dtype=torch.bfloat16)

    def causal_mask(_batch, _head, query, key):
        return query >= key

    def run(values, sequence_length):
        query, key, value = [tensor.detach().clone().requires_grad_() for tensor in values]
        block_mask = create_block_mask(
            causal_mask,
            B=1,
            H=2,
            Q_LEN=sequence_length,
            KV_LEN=sequence_length,
            device=device,
        )
        output = torch.compile(flex_attention, dynamic=False)(query, key, value, block_mask=block_mask)
        output[..., :length, :].float().sum().backward()
        return output.detach(), [tensor.grad.detach() for tensor in (query, key, value)]

    exact_output, exact_grads = run(qkv, length)
    padded_qkv = torch.nn.functional.pad(qkv, (0, 0, 0, padded_length - length))
    padded_output, padded_grads = run(padded_qkv, padded_length)

    torch.testing.assert_close(padded_output[..., :length, :], exact_output, rtol=2e-2, atol=2e-2)
    for padded_grad, exact_grad in zip(padded_grads, exact_grads):
        torch.testing.assert_close(padded_grad[..., :length, :], exact_grad, rtol=2e-2, atol=2e-2)
