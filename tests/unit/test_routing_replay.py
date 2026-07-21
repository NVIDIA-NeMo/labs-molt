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

"""R3 rollout routing replay — capture alignment + training-side resharding.

The CP-sharded path needs a real process group (exercised by the 4-node run); here
we cover the pure logic: how the rollout per-token expert ids are captured onto a
Trajectory, survive the Experience pad/concat machinery, and get reshaped to the
gate's token order for the non-CP regimes.
"""

from types import SimpleNamespace

import numpy as np
import torch

from molt.agents.base import Trajectory
from molt.models.base import BaseModel
from molt.trainer.algorithm.experience import (
    Experience,
    make_experience_batch,
    remove_padding_in_sequences,
)

L, K = 3, 2  # MoE layers, top-k


def _request_output(action_tokens, gen_routed, prompt_routed):
    """Mimic a vLLM RequestOutput carrying R3 routing (None when capture is off)."""
    completion = SimpleNamespace(token_ids=list(action_tokens), routed_experts=gen_routed)
    return SimpleNamespace(outputs=[completion], prompt_routed_experts=prompt_routed)


def _routing(base, n):
    """n distinct [L, K] int16 rows starting at `base` (so we can assert exact ids)."""
    return (base + np.arange(n * L * K)).reshape(n, L, K).astype(np.int16)


def test_absorb_routing_single_turn_aligns_by_absolute_position():
    # The engine returns one array for [prompt + generation], indexed from position 0.
    # absorb_routing (called after append_action) lays it down by absolute position.
    traj = Trajectory(prompt="p", label="l", images=None, observation_text="", observation_tokens=[10, 11])
    prompt_routed = _routing(0, 2)  # 2 prompt tokens
    gen_routed = _routing(1000, 3)  # 3 generated tokens

    traj.append_action([20, 21, 22])
    traj.absorb_routing(_request_output([20, 21, 22], gen_routed, prompt_routed))

    assert len(traj.routed_experts) == len(traj.observation_tokens) == 5
    assert np.array_equal(np.stack(traj.routed_experts[:2]), prompt_routed)  # positions 0-1: prompt
    assert np.array_equal(np.stack(traj.routed_experts[2:]), gen_routed)  # positions 2-4: generation


def test_absorb_routing_disabled_is_noop():
    traj = Trajectory(prompt="p", label="l", images=None, observation_text="", observation_tokens=[10, 11])
    # capture off -> CompletionOutput.routed_experts is None
    assert traj.absorb_routing(_request_output([20], None, None)) is None
    assert traj.routed_experts is None


def test_absorb_routing_multi_turn_first_writer_wins():
    traj = Trajectory(prompt="p", label="l", images=None, observation_text="", observation_tokens=[10])

    # turn 1: prompt [10], action [20, 21]
    traj.append_action([20, 21])
    traj.absorb_routing(_request_output([20, 21], _routing(1000, 2), _routing(0, 1)))
    traj.append_feedback("", "fb", [30])  # feedback token -> placeholder

    # turn 2 re-prefills [10, 20, 21, 30]; only the new feedback pos (idx 3) is still unfilled
    traj.append_action([40])
    traj.absorb_routing(_request_output([40], _routing(2000, 1), _routing(50, 4)))

    assert len(traj.routed_experts) == len(traj.observation_tokens) == 5
    # feedback token (idx 3) got its row from turn-2 prefill (base 50, row 3)
    assert np.array_equal(traj.routed_experts[3], _routing(50, 4)[3])
    # prompt/action positions keep their first-appearance routing (first-writer-wins)
    assert np.array_equal(traj.routed_experts[0], _routing(0, 1)[0])
    assert np.array_equal(traj.routed_experts[1], _routing(1000, 2)[0])


def test_absorb_routing_partial_coverage_leaves_tail_unfilled():
    # Engines that don't split out a prompt array hand back one position-0-indexed array on
    # the completion; if it is shorter than the full sequence, absorb places what it has by
    # absolute position and leaves the tail None (-> natural routing), never offsetting it.
    traj = Trajectory(prompt="p", label="l", images=None, observation_text="", observation_tokens=[10, 11, 12])
    full_seq_routed = _routing(0, 4)  # covers absolute positions 0..3 only
    traj.append_action([20, 21])  # sequence is now 5 tokens
    traj.absorb_routing(_request_output([20, 21], full_seq_routed, None))  # prompt_routed=None

    assert len(traj.routed_experts) == 5
    assert np.array_equal(np.stack(traj.routed_experts[:4]), full_seq_routed)  # positions 0-3 aligned
    assert traj.routed_experts[4] is None  # tail beyond the array -> natural routing


def test_remove_padding_trims_routed_experts_on_seq_dim():
    # post-split single-sample shapes: step tensors are 1D [T], routing is [L, K, T]
    attn = torch.tensor([1, 1, 1, 0, 0])  # 2 trailing pad tokens
    routed = torch.arange(L * K * 5).reshape(L, K, 5)
    exp = Experience(sequences=torch.arange(5), attention_mask=attn, routed_experts=routed)

    remove_padding_in_sequences([exp])

    assert exp.sequences.shape == (3,)
    assert exp.routed_experts.shape == (L, K, 3)  # seq (last) dim trimmed, L/K kept
    assert torch.equal(exp.routed_experts, routed[..., :3])


def test_make_experience_batch_pads_routed_experts_with_sentinel():
    # Batching samples of unequal length pads step tensors on the last (seq) dim.
    # routed_experts must pad with the R3 -1 sentinel (keep live routing), NOT 0 —
    # 0 is a valid expert id and would force the pad tokens to expert 0.
    long = Experience(
        sequences=torch.tensor([1, 2, 3]),
        attention_mask=torch.tensor([1, 1, 1]),
        routed_experts=torch.zeros(L, K, 3, dtype=torch.int16),
    )
    short = Experience(
        sequences=torch.tensor([1, 2]),
        attention_mask=torch.tensor([1, 1]),
        routed_experts=torch.zeros(L, K, 2, dtype=torch.int16),
    )
    batch = make_experience_batch([long, short])

    assert batch.routed_experts.shape == (2, L, K, 3)
    assert (batch.routed_experts[1, :, :, 2] == -1).all()  # short sample's padded tail -> sentinel
    assert batch.sequences[1, 2].item() == 0  # sequences still pad with 0


def test_build_routing_targets_selects_sparse_hybrid_global_layer_ids():
    # routed_experts is (B, vllm_layers, K, S) seq-last; gate sees B*S tokens row-major.
    # vLLM sizes the layer dim to num_hidden_layers and indexes it by the GLOBAL
    # decoder-layer id; a hybrid model's MoE gates sit at sparse ids (e.g. 1, 3, 6)
    # with non-MoE rows in between. _build_routing_targets must pick each gate's own
    # global-id row — NOT the first n_gates rows (which would grab Mamba/attention).
    B, S = 2, 3
    vllm_layers = 8  # hybrid backbone: more layers than MoE gates
    global_ids = [1, 3, 6]  # the MoE gates' global decoder-layer ids
    routed = torch.zeros(B, vllm_layers, K, S, dtype=torch.int16)
    for b in range(B):
        for layer in range(vllm_layers):
            for t in range(S):
                routed[b, layer, 0, t] = 100 * b + 10 * layer + t

    stub = SimpleNamespace(packing_samples=False, _num_routing_gates=len(global_ids), _moe_layer_global_ids=global_ids)
    targets = BaseModel._build_routing_targets(stub, routed, indices=None, cp_forward=False)

    assert len(targets) == len(global_ids)
    for i, gid in enumerate(global_ids):
        assert targets[i].shape == (B * S, K)
        # gate i replays its global-layer gid row (not row i); token order row-major B*S
        expected = [100 * (tok // S) + 10 * gid + (tok % S) for tok in range(B * S)]
        assert targets[i][:, 0].tolist() == expected


def test_build_routing_targets_cp_takes_this_ranks_local_shard():
    # cp=2, rank 0 holds the head chunk [0] and the mirrored tail chunk [3] of a
    # length-4 sequence (PyTorch CP load balancing). The local length is S'/cp = 2;
    # _build_routing_targets must take exactly those tokens (regression: it used to
    # divide by cp twice and hand back S'/cp/cp tokens, mismatching the gate).
    from types import SimpleNamespace as NS

    cp_size, S = 2, 4
    cp_mesh = NS(size=lambda: cp_size, get_local_rank=lambda: 0)
    routed = torch.zeros(1, L, K, S, dtype=torch.int16)
    for layer in range(L):
        for t in range(S):
            routed[0, layer, 0, t] = 10 * layer + t

    stub = SimpleNamespace(
        packing_samples=False,
        _num_routing_gates=L,
        _moe_layer_global_ids=list(range(L)),
        cp_size=cp_size,
        cp_mesh=cp_mesh,
    )
    targets = BaseModel._build_routing_targets(stub, routed, indices=None, cp_forward=True)

    assert len(targets) == L
    for layer in range(L):
        assert targets[layer].shape == (S // cp_size, K)  # local tokens, NOT S//cp//cp
        assert targets[layer][:, 0].tolist() == [10 * layer + 0, 10 * layer + 3]  # head[0] + tail[3]


def test_build_routing_targets_preserves_minus_one_sentinel():
    # vLLM returns routing only for generated tokens, so prompt / feedback positions
    # carry a -1 sentinel; the reshape must preserve it (RouterReplay keeps the live
    # selection there and replays only the captured response rows). Token 0 = prompt.
    B, S = 1, 3
    routed = torch.zeros(B, L, K, S, dtype=torch.int16)
    routed[:, :, :, 0] = -1  # prompt token: no rollout routing
    for layer in range(L):
        for t in range(1, S):
            routed[0, layer, 0, t] = 10 * layer + t

    stub = SimpleNamespace(packing_samples=False, _num_routing_gates=L, _moe_layer_global_ids=list(range(L)))
    targets = BaseModel._build_routing_targets(stub, routed, indices=None, cp_forward=False)

    for layer in range(L):
        assert (targets[layer][0] == -1).all()  # prompt row stays sentinel -> live routing
        assert targets[layer][1:, 0].tolist() == [10 * layer + 1, 10 * layer + 2]


def test_build_routing_targets_cp_pads_with_sentinel():
    # CP pad tokens must be the -1 sentinel (keep live routing), never 0 (which would
    # force-route them to expert 0). cp=2, S=2 pads to 2*cp_size=4; rank 1 holds global
    # position 1 (real) and position 2 (pad).
    from types import SimpleNamespace as NS

    cp_size, S = 2, 2
    cp_mesh = NS(size=lambda: cp_size, get_local_rank=lambda: 1)
    routed = torch.zeros(1, L, K, S, dtype=torch.int16)
    for layer in range(L):
        for t in range(S):
            routed[0, layer, 0, t] = 10 * layer + t + 1  # 1-based: real ids never look like pad

    stub = SimpleNamespace(
        packing_samples=False,
        _num_routing_gates=L,
        _moe_layer_global_ids=list(range(L)),
        cp_size=cp_size,
        cp_mesh=cp_mesh,
    )
    targets = BaseModel._build_routing_targets(stub, routed, indices=None, cp_forward=True)

    for layer in range(L):
        assert targets[layer].shape == (2, K)
        assert targets[layer][0, 0].item() == 10 * layer + 2  # global pos 1 (real)
        assert (targets[layer][1] == -1).all()  # global pos 2 (pad -> sentinel)


def test_make_experience_batch_mixed_none_routed_experts():
    # A batch mixing a captured-routing sample with an unrouted (None) one must neither
    # silently drop the batch's routing (leading None -> first-item dispatch returned None)
    # nor crash on None.size(-1) (trailing None). The None sample becomes an all-(-1)
    # sentinel block: natural routing on every token, i.e. exactly "no captured routing".
    routed = torch.zeros(L, K, 3, dtype=torch.int16)
    have = Experience(sequences=torch.tensor([1, 2, 3]), attention_mask=torch.tensor([1, 1, 1]), routed_experts=routed)
    missing = Experience(
        sequences=torch.tensor([4, 5, 6]), attention_mask=torch.tensor([1, 1, 1]), routed_experts=None
    )

    trailing = make_experience_batch([have, missing])  # trailing None used to crash
    assert trailing.routed_experts.shape == (2, L, K, 3)
    assert torch.equal(trailing.routed_experts[0], routed)
    assert (trailing.routed_experts[1] == -1).all()

    leading = make_experience_batch([missing, have])  # leading None used to drop the batch's routing
    assert leading.routed_experts.shape == (2, L, K, 3)
    assert (leading.routed_experts[0] == -1).all()
    assert torch.equal(leading.routed_experts[1], routed)


def test_make_experience_batch_none_routed_experts_longest_stays_seq_aligned():
    # The unrouted (None) sample is the LONGEST: its synthesized sentinel must carry its own
    # sequence length, or routed_experts would pad to a shorter seq dim than `sequences` and
    # desync from the token axis the training forward replays against.
    have = Experience(
        sequences=torch.tensor([1, 2]),
        attention_mask=torch.tensor([1, 1]),
        routed_experts=torch.zeros(L, K, 2, dtype=torch.int16),
    )
    missing = Experience(sequences=torch.tensor([3, 4, 5, 6]), attention_mask=torch.tensor([1, 1, 1, 1]))
    batch = make_experience_batch([have, missing])

    assert batch.sequences.shape == (2, 4)
    assert batch.routed_experts.shape == (2, L, K, 4)  # seq dim matches sequences, not the routed sample's 2
    assert (batch.routed_experts[1] == -1).all()


def test_make_experience_batch_all_none_routed_experts_stays_none():
    a = Experience(sequences=torch.tensor([1, 2]), attention_mask=torch.tensor([1, 1]), routed_experts=None)
    b = Experience(sequences=torch.tensor([3, 4]), attention_mask=torch.tensor([1, 1]), routed_experts=None)
    assert make_experience_batch([a, b]).routed_experts is None


def test_make_experience_batch_mixed_none_routed_experts_leading_none():
    # A None routed_experts as the FIRST item (the make_experience_batch first-item dispatch)
    # must still merge: the None is filled seq-aligned and right-padded by the -1 sentinel,
    # not dropped or crashed on. Single-sample convention: per-sample tensors are 1D (T,) /
    # (L, K, T), stacked to (B, ...).
    routed = torch.zeros(L, K, 2, dtype=torch.int16)
    have = Experience(sequences=torch.tensor([1, 2]), attention_mask=torch.tensor([1, 1]), routed_experts=routed)
    missing = Experience(  # leading None AND the longest sequence
        sequences=torch.tensor([3, 4, 5]), attention_mask=torch.tensor([1, 1, 1]), routed_experts=None
    )

    merged = make_experience_batch([missing, have])
    assert merged.sequences.shape == (2, 3)
    assert merged.routed_experts.shape == (2, L, K, 3)  # seq dim tracks sequences
    assert (merged.routed_experts[0] == -1).all()  # unrouted sample -> natural routing everywhere
    assert (merged.routed_experts[1, :, :, :2] == 0).all()  # captured rows preserved
    assert (merged.routed_experts[1, :, :, 2] == -1).all()  # its pad tail -> sentinel
