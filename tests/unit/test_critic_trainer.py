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

import torch

from molt.trainer.algorithm.experience import Experience
from molt.trainer.workers import critic_actor


class _Strategy:
    """Records what the critic's training_step hands to backward()."""

    def __init__(self):
        self.args = SimpleNamespace(
            critic=SimpleNamespace(value_clip=0.2, max_epochs=1),
            train=SimpleNamespace(dynamic_batch_enable=False, max_epochs=1, force_on_policy=True),
        )
        self.dp_size = 1
        self.accumulated_gradient = 1  # what backward() falls back to when the window is not passed
        self.backward_kwargs = []

    def backward(self, loss, model, optimizer, **kwargs):
        self.backward_kwargs.append(kwargs)

    def optimizer_step(self, *args, **kwargs):
        pass

    def sync_replicated_grads(self, params):
        pass

    def get_grad_norm(self, model):
        return 0.0

    def is_rank_0(self):
        return True


class _Critic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(()))
        self.model = None  # the MFU probe reads this and gives up on CPU

    def forward(self, sequences, action_mask, attention_mask=None, cp_context_stack=None, **kwargs):
        return {"action_values": self.param + torch.zeros(action_mask.shape, dtype=torch.float32)}

    def value_head_parameters(self):
        return [self.param]


def _make_trainer(monkeypatch):
    monkeypatch.setattr(critic_actor, "torch_dist_barrier_and_cuda_sync", lambda: None)
    strategy, critic = _Strategy(), _Critic()
    optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    return critic_actor.CriticTrainer(strategy, critic, optimizer, scheduler), strategy


def _experience(batch=1, steps=4):
    return Experience(
        sequences=torch.ones(batch, steps + 1, dtype=torch.long),
        attention_mask=torch.ones(batch, steps + 1, dtype=torch.long),
        action_mask=torch.ones(batch, steps, dtype=torch.bool),
        values=torch.zeros(batch, steps),
        returns=torch.ones(batch, steps),
    )


def test_critic_backward_gets_the_window_microbatch_count(monkeypatch):
    # The MoE aux-loss backward scale averages over the optimizer-step window, which under
    # force_on_policy or dynamic batching is not strategy.accumulated_gradient. The critic runs the
    # same window contract as the actor, so it passes the window size the caller actually used.
    trainer, strategy = _make_trainer(monkeypatch)

    trainer.training_step(
        _experience(), batch_num_tokens=torch.tensor(4.0), num_microbatches=3, is_optimizer_step=True
    )

    assert strategy.backward_kwargs[-1]["num_microbatches"] == 3
    assert strategy.accumulated_gradient != 3  # the fallback would have scaled the aux gradient wrong


def test_critic_value_train_disables_shuffle_for_mixed_media(monkeypatch):
    trainer, strategy = _make_trainer(monkeypatch)
    strategy.args.fsdp = SimpleNamespace(cp_size=1, tp_size=1)
    strategy.global_token_count = lambda x: x
    strategy.all_reduce = lambda d, op="sum": d
    strategy.compute_perf_metrics = lambda *a, **kw: {}
    trainer.replay_buffer.sample_batch_size = 1
    trainer.replay_buffer.cpu_offload = False
    trainer.dataloader_pin_memory = False
    vision = ("image_grid_thw", "pixel_values")
    for idx in range(6):
        exp = _experience(batch=1, steps=3)
        exp.sequences = torch.full((1, 4), idx, dtype=torch.long)
        exp.mm_train_inputs = (
            [{"pixel_values": torch.ones(1), "image_grid_thw": torch.ones(1)}] if idx >= 3 else [None]
        )
        trainer.replay_buffer.append(exp)

    seen = []
    trainer.training_step = lambda exp, *a, **kw: (
        seen.append((int(exp.sequences[0, 0]), exp.media_keys)),
        (0.0, 0.0, 0.0),
    )[1]
    monkeypatch.setattr(critic_actor.torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(critic_actor.torch.cuda, "reset_peak_memory_stats", lambda *a: None)

    torch.manual_seed(0)
    trainer.value_train()
    assert seen == [(i, vision if i >= 3 else ()) for i in range(6)]


def test_critic_value_train_trimmed_batch_and_multi_encoder_deferred_sync(monkeypatch):
    trainer, strategy = _make_trainer(monkeypatch)
    strategy.args.fsdp = SimpleNamespace(cp_size=1, tp_size=1)
    strategy.args.train.force_on_policy = False
    strategy.accumulated_gradient = 4  # configured window larger than trimmed 3-sample buffer
    strategy.global_token_count = lambda x: x
    strategy.all_reduce = lambda d, op="sum": d
    strategy.compute_perf_metrics = lambda *a, **kw: {}
    trainer.replay_buffer.sample_batch_size = 1
    trainer.replay_buffer.cpu_offload = False
    trainer.dataloader_pin_memory = False
    trainer._defer_grad_sync = True

    media_seq = [
        [None],
        [{"pixel_values": torch.ones(1), "image_grid_thw": torch.ones(1)}],
        [{"input_features": torch.ones(1)}],
    ]
    for idx, mm in enumerate(media_seq):
        exp = _experience(batch=1, steps=3)
        exp.sequences = torch.full((1, 4), idx, dtype=torch.long)
        exp.mm_train_inputs = mm
        trainer.replay_buffer.append(exp)

    monkeypatch.setattr(critic_actor.torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(critic_actor.torch.cuda, "reset_peak_memory_stats", lambda *a: None)

    trainer.value_train()
    # Runs 1 shortened optimizer step (not 0), syncing on both image (idx 1) and audio (idx 2).
    assert len(strategy.backward_kwargs) == 3
    assert [kw["sync_gradients"] for kw in strategy.backward_kwargs] == [False, True, True]
