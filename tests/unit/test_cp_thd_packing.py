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

"""Universal CP for THD-packed DSA models (GLM `glm_moe_dsa`, DeepSeek-V4).

molt runs CP on one layout for every model: a padded ``[B, S]`` batch handed to the
model-owned ``ContextParallelSharder``. round_robin models (omni3/qwen3.6) pass
``attention_mask``; THD DSA models (GLM/DSv4) pass ``seq_lens`` + ``qkv_format='thd'``
and the sharder flattens ``[B,S] -> [B*S]`` itself. The gather inverts the layout back
to ``[B,S]`` — no packing/unpacking on the CP path. Covered here without GPU/tilelang:

1. ``_restore_full_sequence``: cp>1 just gathers to ``[B,S]``; cp1 packing scatters.
2. The molt<->AutoModel contract: the ``[B,S]`` cp_batch molt builds is a valid sharder
   input yielding the layout molt's gather inverts (GLM ``input_row_shape``, DSv4 repad).
3. The real cross-rank gather roundtrip + cp_size sum-backward (gloo).
"""

import os
import socket
from types import SimpleNamespace

import pytest
import torch

from molt.models.base import BaseModel


def test_restore_cp_gathers_to_bs_and_drops_cp_pad():
    # Under cp>1 (padded [B,S] CP, for both round_robin and THD DSA) _restore just
    # gathers and slices off the cp-multiple pad the forward added — never unpacks.
    restored = torch.arange(8).view(2, 4).float()  # gathered with one pad column
    calls = {}

    def _gather(t, seq_dim=1, trim=False, fill=None):
        calls.update(seq_dim=seq_dim, trim=trim, fill=fill)
        return restored

    for packing in (True, False):  # THD DSA vs round_robin
        calls.clear()
        stub = SimpleNamespace(packing_samples=packing, _cp_sharder=SimpleNamespace(gather_token_tensor=_gather))
        out = BaseModel._restore_full_sequence(
            stub, torch.zeros(2, 3), cp_forward=True, batch=2, seqlen=3, indices=None
        )
        assert calls == {"seq_dim": 1, "trim": True, "fill": 0.0}
        assert torch.equal(out, restored[:, :3])  # [B,S] without the pad; no unpack under cp>1


def test_restore_packing_cp1_scatters_to_bs():
    # cp1 real-token packing (no CP): scatter the packed [1, total] back to [B, S].
    B, S = 2, 3
    indices = torch.tensor([0, 1, 3, 4, 5])
    packed = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    stub = SimpleNamespace(packing_samples=True, _cp_sharder=None)
    out = BaseModel._restore_full_sequence(stub, packed, cp_forward=False, batch=B, seqlen=S, indices=indices)
    assert out.shape == (B, S)
    assert out.reshape(-1).tolist() == [1.0, 2.0, 0.0, 3.0, 4.0, 5.0]


class _FakeMesh:  # single-process stand-in; make_* reads size() + get_group()
    def size(self):
        return 2

    def get_group(self):
        return None  # cp disabled single-process -> rank 0 slice


def _padded_thd_cp_batch(B=2, S=4, real=(3, 4)):
    # Exactly what molt builds on the CP path for a THD DSA model: padded [B,S] +
    # per-row real length in seq_lens, seq_lens_padded = S, qkv_format='thd'.
    input_ids = torch.arange(B * S).view(B, S)
    return {
        "input_ids": input_ids,
        "labels": input_ids + 100,
        "position_ids": torch.arange(S).unsqueeze(0).expand(B, S).contiguous(),
        "seq_lens": torch.tensor([[r] for r in real], dtype=torch.int32),
        "seq_lens_padded": torch.full((B, 1), S, dtype=torch.int32),
        "qkv_format": "thd",
    }


def test_glm_sharder_accepts_padded_thd_cp_batch():
    # Contract: molt's padded [B,S] THD cp_batch is a valid GLM sharder input; it
    # flattens [B,S]->[B*S], contiguous-shards per rank, and reports input_row_shape
    # == [B,S] (molt's gather reshapes the gathered [B*S] straight back to [B,S]).
    glm_cp = pytest.importorskip("nemo_automodel.components.models.glm_moe_dsa.cp")
    B, S = 2, 4  # B*S=8 divisible by cp=2
    _ctx, sharded, layout = glm_cp.shard_glm_dsa_packed_cp_batch(_FakeMesh(), None, _padded_thd_cp_batch(B, S))

    assert sharded["input_ids"].tolist() == [0, 1, 2, 3]  # rank0: first half of the [B*S] flatten
    assert sharded["qkv_format"] == "thd"
    assert layout.input_row_shape == (B, S)
    assert layout.padded_seq_len == B * S


def _tiny_glm_model():
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    cfg = GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        q_lora_rank=32,
        kv_lora_rank=16,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=4,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        max_position_embeddings=256,
        indexer_types=["full", "shared"],
        mlp_layer_types=["dense", "sparse"],
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-6,
        dtype="float32",
    )
    return GlmMoeDsaForCausalLM.from_config(
        cfg,
        backend=BackendConfig(attn="tilelang", linear="torch", experts="torch", dispatcher="torch", rms_norm="torch"),
    )


def _cp_gather_worker(rank, world, port):
    # Runs molt's exact CP data path in a gloo worker: resolve the model-owned GLM DSA
    # sharder on a padded [B,S] THD cp_batch, shard, then gather_token_tensor back.
    import torch.distributed as dist
    from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
    from torch.distributed.device_mesh import init_device_mesh

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(world))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("cp",))
        torch.manual_seed(0)
        model = _tiny_glm_model()
        B, S = 1, 8 * world  # B*S = 8*world divisible by cp=world; all real (no intra-row pad)
        input_ids = (torch.arange(B * S) % 251 + 3).view(B, S)
        seq_lens = torch.full((B, 1), S, dtype=torch.int32)
        cp_batch = {
            "input_ids": input_ids.clone(),
            "labels": input_ids.clone(),
            "position_ids": torch.arange(S).unsqueeze(0).expand(B, S).contiguous(),
            "seq_lens": seq_lens,
            "seq_lens_padded": seq_lens,
            "qkv_format": "thd",
        }
        sharder = ContextParallelSharder(model, mesh, dict(cp_batch), invoke_pre_embed=True)
        _ctx, local = sharder.shard(dict(cp_batch))
        proxy = local["input_ids"].reshape(1, -1).to(torch.float32).requires_grad_(True)
        gathered = sharder.gather_token_tensor(proxy, seq_dim=1, trim=True, fill=0.0)
        assert torch.equal(gathered.detach(), input_ids.to(torch.float32))  # reconstructs [B,S]
        gathered.sum().backward()
        assert torch.allclose(proxy.grad, torch.full_like(proxy.grad, float(world)))  # all-gather sums ×cp

        # R3: the rollout routing ids arrive as (B, layers, topk, S) seq-last, and a THD
        # layout shards the flattened [B*S] stream, so molt hands the verb the token-major
        # view. Each row must carry the global token id this rank actually forwards.
        layers, topk = 2, 3
        token_ids = torch.arange(B * S).view(B, S)
        routing = token_ids.view(B, 1, 1, S).expand(B, layers, topk, S).contiguous()
        local_routing = sharder.shard_token_tensor(routing.permute(0, 3, 1, 2), fill=-1)
        assert local_routing.shape == (B * S // world, layers, topk)
        expected = token_ids.reshape(-1).chunk(world)[rank]
        assert torch.equal(local_routing[:, 0, 0], expected)
    finally:
        dist.barrier()
        dist.destroy_process_group()


def test_cp_gather_and_r3_shard_across_ranks_glm():
    # The cross-rank half the stub tests can't reach: drive molt's real
    # ContextParallelSharder + gather_token_tensor over a live CP group with the real
    # GLM DSA sharder. Asserts the gather reconstructs molt's [B,S] input coordinate
    # and the differentiable all-gather sums grads ×cp_size (the factor that cancels
    # FSDP's mean over dp_cp), plus the R3 routing shard landing in the same token order.
    # cp2 for CI speed; cp4/cp8 verified out of band.
    pytest.importorskip("nemo_automodel.components.models.glm_moe_dsa.model")
    import torch.multiprocessing as mp

    if (os.cpu_count() or 1) < 2:
        pytest.skip("needs >= 2 CPUs for a 2-rank gloo group")
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    mp.spawn(_cp_gather_worker, args=(2, port), nprocs=2, join=True)
