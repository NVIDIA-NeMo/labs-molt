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

from types import SimpleNamespace

import pytest
import torch

import molt.trainer.algorithm.experience as experience_mod
from molt.trainer.algorithm.experience import (
    Experience,
    balance_experiences,
    get_model_parallel_size,
    make_experience_batch,
    split_experience_batch,
)
from molt.trainer.rollout.experience_maker import RemoteExperienceMaker


def _args(cp=1, tp=1, ep=1, actor_gpus=1):
    return SimpleNamespace(
        actor=SimpleNamespace(num_nodes=1, num_gpus_per_node=actor_gpus),
        fsdp=SimpleNamespace(cp_size=cp, tp_size=tp, ep_size=ep),
    )


def test_model_parallel_size_excludes_ep():
    assert get_model_parallel_size(_args(cp=2, tp=3, ep=4)) == 6


def _len_sample(length, idx):
    return Experience(total_length=torch.tensor([length]), rollout_ids=[f"r{idx}"])


def test_balance_experiences_excludes_ep_and_returns_flat_samples():
    # DP size = actor_gpus // (cp*tp) = 4 (EP is excluded from the divisor). 6 per-sample
    # experiences -> keep (6 // 4) * 4 = 4 (drop the 2-sample remainder). If EP were wrongly in
    # the divisor, dp would be 2 and all 6 would be kept.
    samples = [_len_sample(6 - i, i) for i in range(6)]
    balanced = balance_experiences(samples, _args(ep=2, actor_gpus=4))

    assert len(balanced) == 4  # flat, equal count per rank (1×4), remainder dropped — not concatenated
    ids = {id(s) for s in samples}
    assert all(id(b) in ids for b in balanced)  # returns the input samples themselves (no concat)


def test_balance_experiences_equal_count_and_length_sorted():
    # 10 samples across 4 DP ranks: the 2-sample remainder is dropped so every rank receives the
    # SAME count (2) — unequal counts desync num_steps and deadlock the world all_reduce. The flat
    # result is contiguous per-rank blocks; within each block samples are sorted by length descending
    # (k-th microbatch size-matched across ranks so no straggler trips the NCCL watchdog).
    samples = [_len_sample(10 - i, i) for i in range(10)]
    balanced = balance_experiences(samples, _args(actor_gpus=4))

    assert len(balanced) == 8  # 4 ranks × 2 samples; trailing 2-sample remainder dropped
    assert len({id(b) for b in balanced}) == 8  # no duplicates
    for rank in range(4):
        block = balanced[rank * 2 : rank * 2 + 2]
        block_lengths = [int(b.total_length.item()) for b in block]
        assert block_lengths == sorted(block_lengths, reverse=True)


def _vision_inputs():
    return [{"pixel_values": torch.ones(1), "image_grid_thw": torch.ones(1)}]


@pytest.mark.parametrize("extra_text", [False, True])
def test_balance_experiences_aligns_media_across_context_parallel_groups(monkeypatch, extra_text):
    samples = [_len_sample(n, i) for i, n in enumerate((170, 160, 154, 150, 66, 50))]
    for sample in (samples[0], samples[2], samples[3], samples[5]):
        sample.mm_train_inputs = _vision_inputs()
    monkeypatch.setattr(experience_mod.ray, "put", lambda _: object())
    samples[2].offload()
    assert samples[2].mm_train_inputs is None
    if extra_text:
        samples.append(_len_sample(20, 6))

    balanced = balance_experiences(samples, _args(cp=8, actor_gpus=16))
    assert {id(s) for s in balanced} == {id(s) for s in samples[:6]}
    signatures = [s.media_keys for s in balanced]
    vision = ("image_grid_thw", "pixel_values")
    assert signatures[:3] == signatures[3:] == [(), vision, vision]


def test_balance_experiences_orders_text_before_vision_for_single_dp_rank():
    # Even with effective_num == 1 (e.g. DP=1, CP=8), FSDP2 still shards across dp_cp with deferred
    # gradient sync: keep all samples and order text before vision so vision syncs on the last step.
    samples = [_len_sample(n, i) for i, n in enumerate((170, 80, 120))]
    samples[0].mm_train_inputs = _vision_inputs()
    samples[2].mm_train_inputs = _vision_inputs()

    balanced = balance_experiences(samples, _args(cp=8, actor_gpus=8))
    vision = ("image_grid_thw", "pixel_values")
    assert len(balanced) == 3
    assert [s.media_keys for s in balanced] == [(), vision, vision]


def test_balance_experiences_raises_when_no_media_group_fills_dp_ranks():
    samples = [_len_sample(n, i) for i, n in enumerate((100, 90, 80, 70))]
    samples[0].mm_train_inputs = _vision_inputs()
    samples[1].mm_train_inputs = _vision_inputs()

    with pytest.raises(ValueError, match="same media inputs"):
        balance_experiences(samples, _args(actor_gpus=4))


def test_balance_experiences_balances_total_length_across_media_groups():
    samples = [_len_sample(n, i) for i, n in enumerate((100, 10, 500, 100))]
    samples[2].mm_train_inputs = _vision_inputs()
    samples[3].mm_train_inputs = _vision_inputs()

    balanced = balance_experiences(samples, _args(actor_gpus=2))
    vision = ("image_grid_thw", "pixel_values")
    assert [s.media_keys for s in balanced] == [(), vision, (), vision]
    rank_totals = sorted(sum(int(s.total_length.item()) for s in balanced[r * 2 : (r + 1) * 2]) for r in range(2))
    # Pairing heaviest vision with lightest text gives (200, 510) rather than (110, 600).
    assert rank_totals == [200, 510]


def test_dynamic_batch_aligns_mixed_media_across_dp_ranks(monkeypatch):
    from molt.trainer.algorithm.replay_buffer import NaiveReplayBuffer

    def _item(length, idx, with_vision):
        return Experience(
            sequences=torch.ones(1, length, dtype=torch.long) * idx,
            attention_mask=torch.ones(1, length, dtype=torch.long),
            action_mask=torch.ones(1, length - 1, dtype=torch.bool),
            total_length=torch.tensor([length]),
            mm_train_inputs=_vision_inputs() if with_vision else [None],
            rollout_ids=[f"r{idx}"],
        )

    # Without media grouping, KK on rank 0 pairs [T(80), T(20)] and [V(80), V(20)] (1 text MB + 1 vision MB),
    # while rank 1 pairs [T(70), V(30)] and [V(65), T(35)] (2 vision MBs) -> FSDP2 deadlock on MB 0.
    rank_items = [
        [_item(80, 0, False), _item(20, 1, False), _item(80, 2, True), _item(20, 3, True)],
        [_item(70, 4, False), _item(35, 5, False), _item(65, 6, True), _item(30, 7, True)],
    ]
    args = SimpleNamespace(
        fsdp=SimpleNamespace(cp_size=1, tp_size=1),
        train=SimpleNamespace(force_on_policy=True, max_tokens_per_gpu=100, batch_size=8),
    )
    monkeypatch.setattr("molt.trainer.algorithm.replay_buffer.dist.get_world_size", lambda: 2)
    monkeypatch.setattr("molt.trainer.algorithm.replay_buffer.torch.cuda.current_device", lambda: torch.device("cpu"))

    # Two-phase capture so both ranks share the real elementwise max all_reduce.
    captured = []

    def _run_with_all_reduce(rank_idx, max_counts=None):
        buf = NaiveReplayBuffer(sample_batch_size=1, cpu_offload=False, dynamic_batch=True)
        for sample in rank_items[rank_idx]:
            buf.append(sample)

        def _all_reduce(tensor, op="max"):
            assert op == "max"
            if max_counts is None:
                captured.append(tensor.clone())
                return tensor
            return max_counts.clone()

        buf.setup_dynamic_batch(SimpleNamespace(args=args, all_reduce=_all_reduce, print=lambda *a: None))
        return buf

    _run_with_all_reduce(0)
    _run_with_all_reduce(1)
    assert len(captured[0]) == len(captured[1])
    shared_max = torch.maximum(captured[0], captured[1])

    buf0 = _run_with_all_reduce(0, shared_max)
    buf1 = _run_with_all_reduce(1, shared_max)

    mbs0 = [buf0.collate_fn([buf0[i]]) for i in range(len(buf0))]
    mbs1 = [buf1.collate_fn([buf1[i]]) for i in range(len(buf1))]
    vision = ("image_grid_thw", "pixel_values")
    sig0 = [mb.media_keys for mb in mbs0]
    sig1 = [mb.media_keys for mb in mbs1]

    assert sig0 == sig1
    assert sig0[0] == () and sig0[-1] == vision
    assert buf0.dynamic_optimizer_step == buf1.dynamic_optimizer_step
    assert buf0.dynamic_optimizer_step[-1] == 1 and sum(buf0.dynamic_optimizer_step) == 1
    assert sorted(idx for mb in buf0.dynamic_indices for idx in mb) == [0, 1, 2, 3]
    assert sorted(idx for mb in buf1.dynamic_indices for idx in mb) == [0, 1, 2, 3]


def test_policy_train_disables_shuffle_only_for_mixed_media(monkeypatch):
    from molt.trainer.algorithm.replay_buffer import NaiveReplayBuffer
    from molt.trainer.workers import policy_actor

    def _sample(idx, with_vision):
        return Experience(
            sequences=torch.full((1, 4), idx, dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            action_mask=torch.ones(1, 3, dtype=torch.bool),
            total_length=torch.tensor([4]),
            mm_train_inputs=_vision_inputs() if with_vision else [None],
            rollout_ids=[f"r{idx}"],
        )

    def _run(samples):
        trainer = policy_actor.PolicyTrainer.__new__(policy_actor.PolicyTrainer)
        trainer.args = SimpleNamespace(
            fsdp=SimpleNamespace(cp_size=1, tp_size=1),
            train=SimpleNamespace(dynamic_batch_enable=False, force_on_policy=True, max_epochs=1),
        )
        trainer.strategy = SimpleNamespace(
            accumulated_gradient=1,
            is_rank_0=lambda: False,
            global_token_count=lambda x: x,
            compute_perf_metrics=lambda *a, **kw: {},
            print=lambda *a: None,
        )
        trainer.actor_loss_fn = SimpleNamespace(loss_agg_mode="token-mean")
        trainer.replay_buffer = NaiveReplayBuffer(sample_batch_size=1, cpu_offload=False, dynamic_batch=False)
        for s in samples:
            trainer.replay_buffer.append(s)
        trainer.dataloader_pin_memory = False
        trainer.max_epochs = 1
        trainer._mfu = None
        seen = []
        trainer.training_step = lambda exp, *a, **kw: seen.append((int(exp.sequences[0, 0]), exp.media_keys)) or {}
        trainer._record_status = lambda *a, **kw: None
        monkeypatch.setattr(policy_actor.torch.cuda, "current_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(policy_actor.torch.cuda, "reset_peak_memory_stats", lambda *a: None)
        torch.manual_seed(0)
        trainer.policy_train(kl_ctl=0.0)
        return seen

    vision = ("image_grid_thw", "pixel_values")
    mixed = [_sample(i, i >= 3) for i in range(6)]
    assert _run(mixed) == [(i, vision if i >= 3 else ()) for i in range(6)]

    # Homogeneous media still shuffles on cp=1, tp=1.
    homogeneous = [_sample(i, True) for i in range(6)]
    seen_ids = [idx for idx, _ in _run(homogeneous)]
    assert sorted(seen_ids) == list(range(6)) and seen_ids != list(range(6))


def _grpo_maker():
    """A real RemoteExperienceMaker: only `strategy.args` and the KL controller are stubbed, so a
    new step on the compute_advantages path does not need another binding here."""
    args = SimpleNamespace(
        reward=SimpleNamespace(clip_range=None, overlong_buffer_len=None, stop_properly_penalty_coef=None),
        algo=SimpleNamespace(
            advantage=SimpleNamespace(
                estimator="grpo", gamma=1.0, lam=1.0, length_adaptive_alpha=None, no_whiten=False
            )
        ),
        rollout=SimpleNamespace(n_samples_per_prompt=4),
    )
    return RemoteExperienceMaker(None, None, SimpleNamespace(value=0.0), SimpleNamespace(args=args), None)


def _sample(idx, group_id, reward, length=6):
    # Equal length so the concatenated path stacks without padding — the two paths are
    # then directly comparable row-by-row.
    return Experience(
        action_mask=torch.ones(1, length, dtype=torch.bool),
        kl=torch.zeros(1, length),
        rewards=torch.tensor([float(reward)]),
        index=[idx],
        group_ids=[group_id],
        rollout_ids=[f"r{idx}"],
        info={},
    )


def test_distributed_advantages_match_materialized():
    # The distributed path runs compute_advantages_and_returns on per-sample "light"
    # Experiences; the materialized path runs it on the concatenated batch. Advantages are
    # a function of (reward, group, mask) only, so the per-sample results must be identical
    # — this is what lets the trainer skip gathering the heavy batch on the controller.
    rewards = [1.0, 0.0, 0.5, 0.25, 0.9, 0.1, 0.4, 0.6]
    groups = ["g0"] * 4 + ["g1"] * 4

    per_sample = [_sample(i, groups[i], rewards[i]) for i in range(8)]
    RemoteExperienceMaker.compute_advantages_and_returns(_grpo_maker(), per_sample)

    concat = make_experience_batch([_sample(i, groups[i], rewards[i]) for i in range(8)])
    RemoteExperienceMaker.compute_advantages_and_returns(_grpo_maker(), [concat])

    for i in range(8):
        assert torch.allclose(per_sample[i].advantages[0], concat.advantages[i])
        assert torch.allclose(per_sample[i].returns[0], concat.returns[i])


def test_text_info_survives_batching_and_splitting():
    items = [
        Experience(sequences=torch.tensor([1, 2]), info={"feedback": "wrong unit"}),
        Experience(sequences=torch.tensor([3]), info={"feedback": "syntax error"}),
    ]

    batch = make_experience_batch(items)
    assert batch.info["feedback"] == ["wrong unit", "syntax error"]
    assert [item.info["feedback"] for item in split_experience_batch(batch)] == ["wrong unit", "syntax error"]


def test_text_info_missing_on_some_samples_batches_as_none():
    # Env feedback is usually set on failures only; whichever sample comes first, the batch keeps
    # the key with None for the samples that lack it (no KeyError, no silently dropped key).
    with_text = Experience(sequences=torch.tensor([1, 2]), info={"feedback": "wrong unit"})
    without = Experience(sequences=torch.tensor([3]), info={})
    assert make_experience_batch([with_text, without]).info["feedback"] == ["wrong unit", None]
    batch = make_experience_batch([without, with_text])
    assert batch.info["feedback"] == [None, "wrong unit"]
    assert [item.info["feedback"] for item in split_experience_batch(batch)] == [None, "wrong unit"]


def test_experience_offload_reload_roundtrip(monkeypatch):
    # offload() moves the heavy fields into the object store (leaving a ref) and keeps the
    # lightweight ones in place; reload() restores them exactly. The controller only ever reads
    # the light fields, so a batch of these handles never fetches an image.
    store = {}

    def fake_put(obj):
        store[len(store)] = obj
        return len(store) - 1

    monkeypatch.setattr(experience_mod.ray, "put", fake_put)
    monkeypatch.setattr(experience_mod.ray, "get", lambda key: store[key])

    exp = Experience(
        sequences=torch.arange(6).view(1, 6),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        action_mask=torch.ones(1, 5, dtype=torch.bool),
        rewards=torch.tensor([1.0]),
        mm_train_inputs=[{"pixel_values": torch.zeros(2, 3)}],
    )
    exp.offload()
    assert exp.heavy_ref is not None
    assert exp.sequences is None and exp.attention_mask is None and exp.mm_train_inputs is None
    assert exp.action_mask is not None and exp.rewards is not None  # light fields untouched

    # offload() is idempotent: a second call must not re-put the now-nulled fields and clobber the
    # ref (the reload below would then restore Nones and lose the sample's tensors).
    first_ref = exp.heavy_ref
    exp.offload()
    assert exp.heavy_ref == first_ref

    exp.reload()
    assert exp.heavy_ref is None
    assert torch.equal(exp.sequences, torch.arange(6).view(1, 6))
    assert torch.equal(exp.mm_train_inputs[0]["pixel_values"], torch.zeros(2, 3))
    exp.reload()  # idempotent — second call is a no-op


def _kl_args(ref, use_loss):
    return SimpleNamespace(
        algo=SimpleNamespace(
            kl=SimpleNamespace(ref=ref, use_loss=use_loss, estimator="k1", init_coef=0.1),
            advantage=SimpleNamespace(estimator="reinforce"),
        ),
        train=SimpleNamespace(force_on_policy=False),
    )


def _make_experience_with_rollout_reference(monkeypatch, use_loss, rollout_log_probs):
    """make_experience with --algo.kl.ref rollout: no reference group exists; the actor forward is
    stubbed, and asking for a reference forward is the failure the test guards against."""
    maker = RemoteExperienceMaker(object(), None, None, SimpleNamespace(args=_kl_args("rollout", use_loss)), None)
    action_log_probs = torch.tensor([[-1.0, -2.0, -0.5]])

    def _fake_dispatch(experiences, group, result_attr):
        assert result_attr == "action_log_probs", f"unexpected reference forward for {result_attr}"
        for experience in experiences:
            experience.action_log_probs = action_log_probs

    monkeypatch.setattr(maker, "_dispatch_forward", _fake_dispatch)
    # The runners offload every sample before it reaches the controller (router.run_group), and the
    # controller never reloads it, so a stub ref is enough.
    monkeypatch.setattr(experience_mod.ray, "put", lambda obj: "heavy-ref")
    exp = Experience(action_mask=torch.ones(1, 3, dtype=torch.bool), rollout_log_probs=rollout_log_probs, info={})
    exp.offload()
    return maker.make_experience([exp])[0], action_log_probs


def test_rollout_reference_kl_uses_the_rollout_log_probs_without_a_reference_model(monkeypatch):
    rollout = torch.tensor([[-1.5, -1.0, -0.5]])

    exp, action = _make_experience_with_rollout_reference(monkeypatch, use_loss=False, rollout_log_probs=rollout)
    torch.testing.assert_close(exp.kl, action - rollout)  # k1 reward signal against the rollout policy
    assert exp.base_action_log_probs is None  # KL-as-reward: the loss needs no reference log-probs

    exp, _ = _make_experience_with_rollout_reference(monkeypatch, use_loss=True, rollout_log_probs=rollout)
    torch.testing.assert_close(exp.base_action_log_probs, rollout)  # KL-in-loss: policy_train reads them


def test_rollout_reference_kl_requires_rollout_log_probs(monkeypatch):
    with pytest.raises(ValueError, match="rollout log-probs"):
        _make_experience_with_rollout_reference(monkeypatch, use_loss=True, rollout_log_probs=None)


def test_balance_experiences_aligns_unequal_actor_and_ref_dp(monkeypatch):
    # Disaggregated teacher/ref with ref_dp=4 != actor_dp=2 (lcm=4):
    # 1) 3T+5V must drop to 4V (divisible by BOTH actor_dp=2 and ref_dp=4, not 6).
    args = SimpleNamespace(
        actor=SimpleNamespace(num_nodes=2, num_gpus_per_node=8),
        ref=SimpleNamespace(num_nodes=4, num_gpus_per_node=8),
        fsdp=SimpleNamespace(cp_size=8, tp_size=1, ep_size=8),
        algo=SimpleNamespace(
            advantage=SimpleNamespace(estimator="grpo"),
            kl=SimpleNamespace(init_coef=0.1, ref="model"),
        ),
    )
    vision = ("image_grid_thw", "pixel_values")
    samples_3t5v = [_len_sample(100 + i, i) for i in range(8)]
    for s in samples_3t5v[3:]:
        s.mm_train_inputs = _vision_inputs()
    bal_3t5v = balance_experiences(samples_3t5v, args)
    assert len(bal_3t5v) == 4 and len(bal_3t5v) % 4 == 0
    assert [s.media_keys for s in bal_3t5v] == [vision] * 4

    # 2) 4T+4V: actor sees [T, T, V, V | T, T, V, V], while _dispatch_forward re-aligns
    # for ref_dp=4 so every ref DP rank gets [T, V] without mutating actor order.
    samples_4t4v = [_len_sample(100 + i, i) for i in range(8)]
    for s in samples_4t4v[4:]:
        s.mm_train_inputs = _vision_inputs()
    bal_4t4v = balance_experiences(samples_4t4v, args)
    assert [s.media_keys for s in bal_4t4v] == [(), (), vision, vision, (), (), vision, vision]

    ref_rank_chunks = []

    class _FakeGroup:
        def __init__(self, dp):
            self.dp_size = dp
            self.duplicate_actors = 1

        def async_run_method_batch(self, method_name, experience):
            assert method_name == "forward"
            chunk_size = len(experience) // self.dp_size
            assert len(experience) % self.dp_size == 0
            refs = []
            for r in range(self.dp_size):
                chunk = experience[r * chunk_size : (r + 1) * chunk_size]
                ref_rank_chunks.append([e.media_keys for e in chunk])
                refs.append([torch.tensor([float(e.total_length.item())]) for e in chunk])
            return refs

        def async_run_method(self, method_name):
            return []

    monkeypatch.setattr("molt.trainer.rollout.experience_maker.ray.get", lambda x: x)
    actor_group = _FakeGroup(dp=2)
    ref_group = _FakeGroup(dp=4)
    maker = RemoteExperienceMaker(actor_group, ref_group, None, SimpleNamespace(args=args), None)
    before_ids = [id(e) for e in bal_4t4v]
    maker._dispatch_forward(bal_4t4v, ref_group, "base_action_log_probs")

    # Every ref rank got identical media order [(), vision], outputs landed on the right
    # Experience objects, and the actor's experience ordering stayed untouched.
    assert ref_rank_chunks == [[(), vision]] * 4
    assert [id(e) for e in bal_4t4v] == before_ids
    for exp in bal_4t4v:
        assert exp.base_action_log_probs.item() == float(exp.total_length.item())

    # 3) Coprime DP sizes (actor_dp=2, ref_dp=3 -> lcm=6): 7T+6V keeps 6T+6V = 12.
    coprime_args = SimpleNamespace(
        actor=SimpleNamespace(num_nodes=1, num_gpus_per_node=2),
        ref=SimpleNamespace(num_nodes=1, num_gpus_per_node=3),
        fsdp=SimpleNamespace(cp_size=1, tp_size=1, ep_size=1),
        algo=SimpleNamespace(
            advantage=SimpleNamespace(estimator="grpo"),
            kl=SimpleNamespace(init_coef=0.1, ref="model"),
        ),
    )
    coprime_samples = [_len_sample(100 + i, i) for i in range(13)]
    for s in coprime_samples[7:]:
        s.mm_train_inputs = _vision_inputs()
    bal_coprime = balance_experiences(coprime_samples, coprime_args)
    assert len(bal_coprime) == 12
    assert [s.media_keys for s in bal_coprime[:6]] == [(), (), (), vision, vision, vision]
    assert [s.media_keys for s in bal_coprime[6:]] == [(), (), (), vision, vision, vision]


def test_balance_experiences_preserves_whole_grpo_prompt_groups():
    def _grouped(length, gid, rid, with_vision):
        return Experience(
            total_length=torch.tensor([length]),
            group_ids=[gid],
            rollout_ids=[rid],
            mm_train_inputs=_vision_inputs() if with_vision else [None],
        )

    # dp=4: 2 text groups x 6 (12) + 3 vision groups x 6 (18).
    # Naive slicing would keep 16 vision samples (splitting gv2 to 4/6). Whole-group
    # selection must keep 2 whole vision groups (12) so every GRPO group stays size 6.
    samples = [
        *[_grouped(100 + i, f"gt{g}", f"rt{g}_{i}", False) for g in range(2) for i in range(6)],
        *[_grouped(200 + i, f"gv{g}", f"rv{g}_{i}", True) for g in range(3) for i in range(6)],
    ]
    bal = balance_experiences(samples, _args(actor_gpus=4))
    assert len(bal) == 24
    counts = {}
    for s in bal:
        counts[s.group_ids[0]] = counts.get(s.group_ids[0], 0) + 1
    assert all(cnt == 6 for cnt in counts.values())
    assert set(counts) == {"gt0", "gt1", "gv0", "gv1"}

    # Variable-size multi-turn groups [7, 9, 8, 6] on dp=4: keeps [7, 9, 8] = 24 whole.
    mt_samples = [_grouped(50 + i, f"g{g}", f"r{g}_{i}", True) for g, sz in enumerate((7, 9, 8, 6)) for i in range(sz)]
    bal_mt = balance_experiences(mt_samples, _args(actor_gpus=4))
    assert len(bal_mt) == 24
    assert {s.group_ids[0] for s in bal_mt} == {"g0", "g1", "g2"}

    # 2 alignable text groups (12) + 1 unalignable vision group (6) on dp=4:
    # drops the 1 vision group whole rather than bisecting it.
    unalignable_v = [
        *[_grouped(100 + i, f"gt{g}", f"rt{g}_{i}", False) for g in range(2) for i in range(6)],
        *[_grouped(200 + i, "gv0", f"rv0_{i}", True) for i in range(6)],
    ]
    bal_uv = balance_experiences(unalignable_v, _args(actor_gpus=4))
    assert len(bal_uv) == 12
    assert {s.group_ids[0] for s in bal_uv} == {"gt0", "gt1"}

    # Defined small-batch fallbacks:
    # 1) A single prompt group of 6 single-turn rollouts on dp=4 cannot form a whole-group
    #    multiple of 4, so it falls back to 4 whole rollouts.
    single_prompt = [_grouped(100 + i, "g0", f"r{i}", True) for i in range(6)]
    bal_sp = balance_experiences(single_prompt, _args(actor_gpus=4))
    assert len(bal_sp) == 4

    # 2) A single multi-turn prompt group with rollout sizes [2, 2, 1] on dp=4 falls back to
    #    whole rollouts ([2, 2] = 4) rather than splitting a rollout.
    mt_single_prompt = [
        _grouped(100, "g0", "r0", True),
        _grouped(101, "g0", "r0", True),
        _grouped(102, "g0", "r1", True),
        _grouped(103, "g0", "r1", True),
        _grouped(104, "g0", "r2", True),
    ]
    bal_mt_sp = balance_experiences(mt_single_prompt, _args(actor_gpus=4))
    assert len(bal_mt_sp) == 4
    assert {s.rollout_ids[0] for s in bal_mt_sp} == {"r0", "r1"}

    # 3) A mixed small batch where no media bucket reaches divisor (2T + 2V on dp=4) fails loudly.
    small_mixed = [
        _grouped(100, "gt0", "rt0", False),
        _grouped(101, "gt1", "rt1", False),
        _grouped(102, "gv0", "rv0", True),
        _grouped(103, "gv1", "rv1", True),
    ]
    with pytest.raises(ValueError, match="same media inputs"):
        balance_experiences(small_mixed, _args(actor_gpus=4))


def test_policy_train_trimmed_batch_runs_one_step_and_keeps_tail_vision_and_multi_encoder_sync(monkeypatch):
    from molt.trainer.algorithm.replay_buffer import NaiveReplayBuffer
    from molt.trainer.workers import policy_actor

    audio = [{"input_features": torch.ones(1)}]

    def _exp(idx, mm):
        return Experience(
            sequences=torch.full((1, 4), idx, dtype=torch.long),
            attention_mask=torch.ones(1, 4, dtype=torch.long),
            action_mask=torch.ones(1, 3, dtype=torch.bool),
            total_length=torch.tensor([4]),
            mm_train_inputs=mm,
            rollout_ids=[f"r{idx}"],
        )

    def _run(samples, *, accum_steps, mbs=1, force_on_policy=False):
        trainer = policy_actor.PolicyTrainer.__new__(policy_actor.PolicyTrainer)
        trainer.args = SimpleNamespace(
            fsdp=SimpleNamespace(cp_size=1, tp_size=1),
            train=SimpleNamespace(dynamic_batch_enable=False, force_on_policy=force_on_policy, max_epochs=1),
        )
        trainer.strategy = SimpleNamespace(
            accumulated_gradient=accum_steps,
            is_rank_0=lambda: False,
            global_token_count=lambda x: x,
            compute_perf_metrics=lambda *a, **kw: {},
            print=lambda *a: None,
        )
        trainer.actor_loss_fn = SimpleNamespace(loss_agg_mode="token-mean")
        trainer.replay_buffer = NaiveReplayBuffer(sample_batch_size=mbs, cpu_offload=False, dynamic_batch=False)
        for s in samples:
            trainer.replay_buffer.append(s)
        trainer.dataloader_pin_memory = False
        trainer.max_epochs = 1
        trainer._mfu = None
        calls = []

        def _fake_step(exp, kl_ctl, batch_num_tokens, num_mbs, is_opt, _seqs=None, _ptc=None, sync_gradients=None):
            calls.append((exp.sequences.shape[0], exp.media_keys, is_opt, sync_gradients))
            return {}

        trainer.training_step = _fake_step
        trainer._record_status = lambda *a, **kw: None
        monkeypatch.setattr(policy_actor.torch.cuda, "current_device", lambda: torch.device("cpu"))
        monkeypatch.setattr(policy_actor.torch.cuda, "reset_peak_memory_stats", lambda *a: None)
        trainer.policy_train(kl_ctl=0.0)
        return calls

    # 1) Configured 64-sample batch (dp=8, accum_steps=8) trimmed to 7/rank by media remainder:
    # must run 1 shortened optimizer step rather than dropping to 0 steps.
    trimmed_7 = [_exp(i, [None] if i < 3 else _vision_inputs()) for i in range(7)]
    calls_7 = _run(trimmed_7, accum_steps=8, mbs=1, force_on_policy=False)
    assert len(calls_7) == 7
    assert [c[2] for c in calls_7] == [False] * 6 + [True]
    assert [c[3] for c in calls_7] == [False] * 6 + [True]

    # 2) micro_batch_size=2 with 3 samples [T, T, V]: drop_last=False keeps trailing [V].
    calls_mbs2 = _run([_exp(0, [None]), _exp(1, [None]), _exp(2, _vision_inputs())], accum_steps=2, mbs=2)
    assert [(c[0], c[1]) for c in calls_mbs2] == [(2, ()), (1, ("image_grid_thw", "pixel_values"))]
    assert [c[2] for c in calls_mbs2] == [False, True]

    # 3) Multi-encoder window [T, IMG, IMG, AUD, AUD]: deferred grad sync must sync on both
    # the last IMG microbatch (idx 2) and the final AUD microbatch (idx 4).
    multi_enc = [
        _exp(0, [None]),
        _exp(1, _vision_inputs()),
        _exp(2, _vision_inputs()),
        _exp(3, audio),
        _exp(4, audio),
    ]
    calls_multi = _run(multi_enc, accum_steps=5, mbs=1, force_on_policy=True)
    assert [c[2] for c in calls_multi] == [False, False, False, False, True]
    assert [c[3] for c in calls_multi] == [False, False, True, False, True]


def test_dynamic_batch_trimmed_below_local_train_batch_runs_one_step(monkeypatch):
    from molt.trainer.algorithm.replay_buffer import NaiveReplayBuffer

    args = SimpleNamespace(
        fsdp=SimpleNamespace(cp_size=1, tp_size=1),
        train=SimpleNamespace(force_on_policy=False, max_tokens_per_gpu=100, batch_size=64),
    )
    monkeypatch.setattr("molt.trainer.algorithm.replay_buffer.dist.get_world_size", lambda: 8)
    monkeypatch.setattr("molt.trainer.algorithm.replay_buffer.torch.cuda.current_device", lambda: torch.device("cpu"))

    buf = NaiveReplayBuffer(sample_batch_size=1, cpu_offload=False, dynamic_batch=True)
    for i in range(7):
        buf.append(
            Experience(
                sequences=torch.ones(1, 20, dtype=torch.long),
                attention_mask=torch.ones(1, 20, dtype=torch.long),
                action_mask=torch.ones(1, 19, dtype=torch.bool),
                total_length=torch.tensor([20]),
                mm_train_inputs=_vision_inputs() if i >= 3 else [None],
                rollout_ids=[f"r{i}"],
            )
        )
    buf.setup_dynamic_batch(SimpleNamespace(args=args, all_reduce=lambda t, op="max": t, print=lambda *a: None))
    assert sum(buf.dynamic_optimizer_step) == 1
    assert sorted(idx for mb in buf.dynamic_indices for idx in mb) == list(range(7))
