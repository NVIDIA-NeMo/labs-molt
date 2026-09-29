# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-rank CPU regression tests for DeepSeek V4 context parallelism."""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.cp import (
    build_dsv4_cp_causal_padding_mask,
    dsv4_cp_all_gather,
)
from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4Attention,
    DeepseekV4RotaryEmbedding,
    build_causal_padding_mask,
)

# Run only on the GPU job. Each test mp.spawns two gloo worker processes that
# re-import the full package (~6s/test, ~18s total on the CPU unit-test job).
# These are context-parallel tests (a multi-GPU feature), so skip them on CPU.
# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = [
    pytest.mark.timeout(60),
    pytest.mark.run_only_on("GPU"),
]


def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _tiny_attention_config(compress_ratio: int = 0) -> DeepseekV4Config:
    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        index_head_dim=8,
        index_n_heads=2,
        index_topk=2,
        n_routed_experts=2,
        n_shared_experts=0,
        num_experts_per_tok=1,
        max_position_embeddings=64,
        compress_ratios=[compress_ratio],
        sliding_window=8,
        attention_dropout=0.0,
        num_hash_layers=0,
        hc_mult=1,
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-6,
        torch_dtype="float32",
    )


def _rotary_embeddings(
    cfg: DeepseekV4Config,
    hidden_states: torch.Tensor,
    position_ids: torch.Tensor,
    *,
    compress: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    rotary = DeepseekV4RotaryEmbedding(
        rope_theta=float(cfg.compress_rope_theta if compress else cfg.rope_theta),
        head_dim=int(cfg.head_dim),
        partial_rotary_factor=float(cfg.qk_rope_head_dim) / float(cfg.head_dim),
        rope_scaling=cfg.rope_scaling if compress else None,
    )
    return rotary(hidden_states, position_ids)


def _compress_rotary(cfg: DeepseekV4Config) -> DeepseekV4RotaryEmbedding:
    return DeepseekV4RotaryEmbedding(
        rope_theta=float(cfg.compress_rope_theta),
        head_dim=int(cfg.head_dim),
        partial_rotary_factor=float(cfg.qk_rope_head_dim) / float(cfg.head_dim),
        rope_scaling=cfg.rope_scaling,
    )


def _init_gloo(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def _cp_collective_worker(rank: int, world_size: int, port: int) -> None:
    try:
        _init_gloo(rank, world_size, port)

        local = torch.arange(rank * 2, rank * 2 + 2, dtype=torch.float32).requires_grad_(True)
        gathered = dsv4_cp_all_gather(local, dim=0, cp_group=dist.group.WORLD)
        torch.testing.assert_close(gathered, torch.arange(world_size * 2, dtype=torch.float32))

        weights = torch.arange(1, world_size * 2 + 1, dtype=torch.float32)
        (gathered * weights).sum().backward()
        expected_grad = world_size * weights[rank * 2 : (rank + 1) * 2]
        torch.testing.assert_close(local.grad, expected_grad)

        position_ids = torch.arange(rank * 2, rank * 2 + 2).view(1, -1)
        padding_mask = torch.tensor([[False, rank == 1]])
        mask = build_dsv4_cp_causal_padding_mask(
            position_ids=position_ids,
            key_len=world_size * 2,
            dtype=torch.float32,
            device=torch.device("cpu"),
            cp_group=dist.group.WORLD,
            padding_mask=padding_mask,
        )

        min_value = torch.finfo(torch.float32).min
        expected_mask = torch.full((1, 1, 2, 4), min_value)
        if rank == 0:
            expected_mask[0, 0, 0, 0] = 0
            expected_mask[0, 0, 1, :2] = 0
        else:
            expected_mask[0, 0, 0, :3] = 0
            expected_mask[0, 0, 1, :3] = 0
        torch.testing.assert_close(mask, expected_mask)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _attention_equivalence_worker(rank: int, world_size: int, port: int) -> None:
    try:
        _init_gloo(rank, world_size, port)
        torch.set_num_threads(1)
        torch.manual_seed(123)

        cfg = _tiny_attention_config()
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32")
        full_attn = DeepseekV4Attention(cfg, layer_idx=0, backend=backend)
        full_attn.init_weights(torch.device("cpu"))
        cp_attn = DeepseekV4Attention(cfg, layer_idx=0, backend=backend)
        cp_attn.load_state_dict(full_attn.state_dict())
        full_attn.eval()
        cp_attn.eval()

        batch, global_seq = 1, 8
        local_seq = global_seq // world_size
        start = rank * local_seq
        end = start + local_seq
        full_hidden = torch.randn(batch, global_seq, cfg.hidden_size).requires_grad_(True)
        local_hidden = full_hidden.detach()[:, start:end].clone().requires_grad_(True)

        full_position_ids = torch.arange(global_seq).view(1, -1)
        local_position_ids = torch.arange(start, end).view(1, -1)
        full_mask = build_causal_padding_mask(
            None,
            seq_len=global_seq,
            dtype=full_hidden.dtype,
            device=full_hidden.device,
            batch_size=batch,
            sliding_window=cfg.sliding_window,
        )
        local_mask = build_dsv4_cp_causal_padding_mask(
            position_ids=local_position_ids,
            key_len=global_seq,
            dtype=local_hidden.dtype,
            device=local_hidden.device,
            cp_group=dist.group.WORLD,
            sliding_window=cfg.sliding_window,
        )

        full_out, _ = full_attn(
            full_hidden,
            position_embeddings=_rotary_embeddings(cfg, full_hidden, full_position_ids),
            attention_mask=full_mask,
            position_ids=full_position_ids,
        )
        local_out, _ = cp_attn(
            local_hidden,
            position_embeddings=_rotary_embeddings(cfg, local_hidden, local_position_ids),
            attention_mask=local_mask,
            position_ids=local_position_ids,
            _dsv4_cp_group=dist.group.WORLD,
        )

        torch.testing.assert_close(local_out, full_out.detach()[:, start:end], atol=1e-5, rtol=1e-5)

        full_out.square().sum().backward()
        local_out.square().sum().backward()
        torch.testing.assert_close(local_hidden.grad, full_hidden.grad[:, start:end], atol=1e-5, rtol=1e-5)

        full_grads = dict(full_attn.named_parameters())
        for name, param in cp_attn.named_parameters():
            if param.grad is None:
                assert full_grads[name].grad is None
                continue
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            torch.testing.assert_close(param.grad, full_grads[name].grad, atol=1e-5, rtol=1e-5)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _compressed_attention_equivalence_worker(rank: int, world_size: int, port: int) -> None:
    try:
        _init_gloo(rank, world_size, port)
        torch.set_num_threads(1)
        torch.manual_seed(123)

        cfg = _tiny_attention_config(compress_ratio=4)
        backend = BackendConfig(attn="torch", linear="torch", rms_norm="torch_fp32")
        full_attn = DeepseekV4Attention(cfg, layer_idx=0, backend=backend)
        full_attn.init_weights(torch.device("cpu"))
        cp_attn = DeepseekV4Attention(cfg, layer_idx=0, backend=backend)
        cp_attn.load_state_dict(full_attn.state_dict())
        full_attn.eval()
        cp_attn.eval()

        batch, global_seq = 1, 8
        local_seq = global_seq // world_size
        start = rank * local_seq
        end = start + local_seq
        full_hidden = torch.randn(batch, global_seq, cfg.hidden_size).requires_grad_(True)
        local_hidden = full_hidden.detach()[:, start:end].clone().requires_grad_(True)

        full_position_ids = torch.arange(global_seq).view(1, -1)
        local_position_ids = torch.arange(start, end).view(1, -1)
        full_mask = build_causal_padding_mask(
            None,
            seq_len=global_seq,
            dtype=full_hidden.dtype,
            device=full_hidden.device,
            batch_size=batch,
            sliding_window=cfg.sliding_window,
        )
        local_mask = build_dsv4_cp_causal_padding_mask(
            position_ids=local_position_ids,
            key_len=global_seq,
            dtype=local_hidden.dtype,
            device=local_hidden.device,
            cp_group=dist.group.WORLD,
            sliding_window=cfg.sliding_window,
        )
        rotary_compress = _compress_rotary(cfg)

        full_out, _ = full_attn(
            full_hidden,
            position_embeddings=_rotary_embeddings(cfg, full_hidden, full_position_ids),
            attention_mask=full_mask,
            position_embeddings_compress=_rotary_embeddings(cfg, full_hidden, full_position_ids, compress=True),
            rotary_compress=rotary_compress,
            position_ids=full_position_ids,
        )
        local_out, _ = cp_attn(
            local_hidden,
            position_embeddings=_rotary_embeddings(cfg, local_hidden, local_position_ids),
            attention_mask=local_mask,
            position_embeddings_compress=_rotary_embeddings(cfg, local_hidden, local_position_ids, compress=True),
            rotary_compress=rotary_compress,
            position_ids=local_position_ids,
            _dsv4_cp_group=dist.group.WORLD,
        )

        torch.testing.assert_close(local_out, full_out.detach()[:, start:end], atol=1e-5, rtol=1e-5)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_dsv4_cp_collectives_have_autograd_and_global_padding_mask():
    mp.spawn(_cp_collective_worker, args=(2, _free_port()), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_dsv4_attention_cp_matches_full_sequence_sliding_window():
    mp.spawn(_attention_equivalence_worker, args=(2, _free_port()), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_dsv4_compressed_attention_cp_uses_global_pool_positions():
    mp.spawn(_compressed_attention_equivalence_worker, args=(2, _free_port()), nprocs=2, join=True)
