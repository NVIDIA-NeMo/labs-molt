# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Two-rank REAL-collective (gloo, CPU) regression tests for CP vision-tower sharding.

The mocked tests in ``test_cp_vision_frame_shard.py`` prove the math with simulated ranks; these
tests drive ``maybe_distribute_visual`` forward AND backward through the actual
``all_gather`` / ``reduce_scatter(SUM)`` collectives on a 2-rank gloo group and assert
numerical parity with the single-process replicated reference:

- forward: every rank's gathered ``pooler_output`` (and ``deepstack_features``) equals the
  replicated full forward;
- backward: each rank backprops only its sequence shard of the gathered embeds (the real
  training contract), so the differentiable gather's reduce-scatter(SUM) routes each
  frame's gradient back to its compute rank -- the vision-parameter gradients all-reduced
  (summed) over the group must equal the replicated full backward;
- pad path: with fewer frame units than ranks, the padded rank runs only the dummy frame
  and must receive exactly zero gradient.
"""

from __future__ import annotations

import importlib.util
import os
import socket
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_HAS_QWEN3_5 = importlib.util.find_spec("transformers.models.qwen3_5") is not None

# Run only on the GPU job. Each test mp.spawns two gloo worker processes that re-import
# the full package (several seconds per test on the CPU unit-test job); the sharding they
# exercise is a multi-GPU context-parallel feature, so keep them off the CPU job like the
# other 2-rank gloo CP tests.
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


def _init_gloo(rank: int, world_size: int, port: int, timeout: timedelta | None = None) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    kwargs = {"timeout": timeout} if timeout is not None else {}
    dist.init_process_group("gloo", rank=rank, world_size=world_size, **kwargs)


class _GlooVisual(torch.nn.Module):
    """Entry-independent stub vision tower (mirrors ``test_cp_vision_frame_shard._StubVisual``).

    ``forward`` takes ``pixel_values`` ``[total_patch_rows, in_dim]`` (frame-contiguous
    patch rows; ``in_dim`` = patch feature size), projects each row with a linear layer
    ("vision params"; bias so the bias grad depends only on the upstream gradient), and
    mean-merges every consecutive ``spatial_merge_size**2`` rows into one token, returning
    ``pooler_output`` ``[total_patch_rows / spatial_merge_size**2, hidden]`` plus optional
    ``deepstack_features`` (scaled copies of ``pooler_output``).  ``grid_thw`` ([N, 3]
    (t, h, w) rows) is accepted but unused: the merge is per consecutive row group, so the
    output is entry-independent exactly like the real per-frame ViT attention.
    """

    def __init__(self, in_dim: int = 8, hidden: int = 6, sms: int = 2, n_deepstack: int = 0):
        super().__init__()
        self.spatial_merge_size = sms
        self.dtype = torch.float32
        self.proj = torch.nn.Linear(in_dim, hidden, bias=True)
        self.n_deepstack = n_deepstack

    def forward(self, pixel_values, grid_thw=None, return_dict=True):
        sms_sq = self.spatial_merge_size**2
        x = self.proj(pixel_values)  # [total_patches, hidden] -- per row, entry-independent
        merged = x.reshape(-1, sms_sq, x.shape[-1]).mean(dim=1)  # [total_patches/sms_sq, hidden]
        out = SimpleNamespace(pooler_output=merged, last_hidden_state=x, deepstack_features=None)
        if self.n_deepstack:
            out.deepstack_features = [merged * (k + 1) for k in range(self.n_deepstack)]
        return out


def _replicated_reference(
    visual: _GlooVisual, pixel: torch.Tensor, grid: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Full single-process forward + backward baseline.

    ``pixel`` is ``[total_patch_rows, in_dim]`` frame-contiguous patch rows and ``grid``
    the matching [N, 3] (t, h, w) rows.  Returns ``(pooler, deepstack, targets, grads)``:
    the detached full ``[tokens, hidden]`` ``pooler_output``, detached deepstack tensors of
    the same shape, the deterministic loss targets (one per gathered tensor, same shapes),
    and the reference ``[proj.weight.grad, proj.bias.grad]`` clones.
    """
    rep = visual(pixel, grid_thw=grid, return_dict=True)
    gen = torch.Generator().manual_seed(11)
    targets = [torch.randn(rep.pooler_output.shape, generator=gen)]
    deepstack = rep.deepstack_features or []
    targets += [torch.randn(rep.pooler_output.shape, generator=gen) for _ in deepstack]
    loss = (rep.pooler_output * targets[0]).sum()
    for d, t in zip(deepstack, targets[1:]):
        loss = loss + (d * t).sum()
    visual.zero_grad(set_to_none=True)
    loss.backward()
    grads = [visual.proj.weight.grad.clone(), visual.proj.bias.grad.clone()]
    return (
        rep.pooler_output.detach().clone(),
        [d.detach().clone() for d in deepstack],
        targets,
        grads,
    )


def _sharded_forward_backward(
    visual: _GlooVisual,
    pixel: torch.Tensor,
    grid: torch.Tensor,
    targets: list[torch.Tensor],
    rank: int,
    world_size: int,
    min_local_patch_rows: int = 0,
):
    """Drive ``maybe_distribute_visual`` with real collectives and backprop this rank's shard.

    ``pixel``/``grid`` follow the module contract ([total_patch_rows, in_dim] frame-contiguous
    rows / [N, 3] (t, h, w) rows).  Each rank keeps only its contiguous sequence shard of the
    gathered ``[tokens, hidden]`` embeds for the loss (targets[0] for ``pooler_output``, the
    rest per deepstack tensor), mirroring how CP sequence sharding consumes the full embeds.
    Returns the output object.
    """
    from nemo_automodel.components.distributed import cp_vision_frame_shard as vs

    visual.zero_grad(set_to_none=True)
    config = vs.CpVisionFrameShardingConfig(
        enabled=True,
        min_tokens=0,
        min_local_patch_rows=min_local_patch_rows,
    )
    token = vs.set_cp_vision_group(dist.group.WORLD, config=config)
    try:
        out = vs.maybe_distribute_visual(visual, pixel, grid)
    finally:
        vs.reset_cp_vision_group(token)

    n = out.pooler_output.shape[0]
    lo, hi = n * rank // world_size, n * (rank + 1) // world_size
    loss = (out.pooler_output[lo:hi] * targets[0][lo:hi]).sum()
    for d, t in zip(out.deepstack_features or [], targets[1:]):
        loss = loss + (d[lo:hi] * t[lo:hi]).sum()
    loss.backward()
    return out


def _parity_worker(rank: int, world_size: int, port: int) -> None:
    """Main path: mixed image/video entries (frame units >= world), with deepstack."""
    try:
        _init_gloo(rank, world_size, port)
        torch.set_num_threads(1)
        torch.manual_seed(0)  # identical weights on every rank, like the FSDP all-gather
        visual = _GlooVisual(n_deepstack=2)
        grid = torch.tensor([[1, 2, 2], [2, 2, 4], [1, 4, 4], [2, 2, 2], [1, 2, 4]], dtype=torch.long)
        gen = torch.Generator().manual_seed(7)
        pixel = torch.randn(int(grid.prod(dim=-1).sum()), 8, generator=gen)

        rep_pooler, rep_deepstack, targets, (gw_ref, gb_ref) = _replicated_reference(visual, pixel, grid)

        out = _sharded_forward_backward(visual, pixel, grid, targets, rank, world_size)
        torch.testing.assert_close(out.pooler_output, rep_pooler, atol=1e-6, rtol=1e-6)
        assert len(out.deepstack_features) == len(rep_deepstack)
        for got, exp in zip(out.deepstack_features, rep_deepstack):
            torch.testing.assert_close(got, exp, atol=1e-6, rtol=1e-6)

        # per-rank grads are partial (each frame's grad lands on its compute rank);
        # summed over the group they must reconstruct the replicated full backward.
        gw = visual.proj.weight.grad.clone()
        gb = visual.proj.bias.grad.clone()
        dist.all_reduce(gw, op=dist.ReduceOp.SUM)
        dist.all_reduce(gb, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(gw, gw_ref, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(gb, gb_ref, atol=1e-6, rtol=1e-6)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _pad_path_worker(rank: int, world_size: int, port: int) -> None:
    """Pad path: one frame unit < world, so the last rank runs only a dummy frame."""
    try:
        _init_gloo(rank, world_size, port)
        torch.set_num_threads(1)
        torch.manual_seed(0)
        visual = _GlooVisual()
        grid = torch.tensor([[1, 4, 4]], dtype=torch.long)  # 1 frame unit < 2 ranks -> pad path
        gen = torch.Generator().manual_seed(7)
        pixel = torch.randn(16, 8, generator=gen)

        rep_pooler, _, targets, (gw_ref, gb_ref) = _replicated_reference(visual, pixel, grid)

        out = _sharded_forward_backward(visual, pixel, grid, targets, rank, world_size)
        assert out.pooler_output.shape == rep_pooler.shape  # dummy tail sliced off
        torch.testing.assert_close(out.pooler_output, rep_pooler, atol=1e-6, rtol=1e-6)

        gw = visual.proj.weight.grad.clone()
        gb = visual.proj.bias.grad.clone()
        if rank == world_size - 1:
            # this rank computed ONLY the dummy pad frame; the gathered dummy tail is
            # sliced off before the loss, so reduce-scatter must route back exactly zero.
            # The bias grad depends only on the upstream gradient, so any non-zero value
            # would mean the dummy frame leaked gradient into the shared vision params.
            assert torch.count_nonzero(gb) == 0
            assert torch.count_nonzero(gw) == 0
        dist.all_reduce(gw, op=dist.ReduceOp.SUM)
        dist.all_reduce(gb, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(gw, gw_ref, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(gb, gb_ref, atol=1e-6, rtol=1e-6)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _local_patch_floor_worker(rank: int, world_size: int, port: int) -> None:
    """Local patch-row floor: padded ViT work is removed before real collectives."""
    try:
        _init_gloo(rank, world_size, port)
        torch.set_num_threads(1)
        torch.manual_seed(0)
        visual = _GlooVisual(n_deepstack=2)
        grid = torch.tensor([[8, 2, 2]], dtype=torch.long)
        gen = torch.Generator().manual_seed(7)
        pixel = torch.randn(int(grid.prod(dim=-1).sum()), 8, generator=gen)

        rep_pooler, rep_deepstack, targets, (gw_ref, gb_ref) = _replicated_reference(visual, pixel, grid)

        out = _sharded_forward_backward(
            visual,
            pixel,
            grid,
            targets,
            rank,
            world_size,
            min_local_patch_rows=32,
        )
        torch.testing.assert_close(out.pooler_output, rep_pooler, atol=1e-6, rtol=1e-6)
        for got, exp in zip(out.deepstack_features, rep_deepstack):
            torch.testing.assert_close(got, exp, atol=1e-6, rtol=1e-6)

        gw = visual.proj.weight.grad.clone()
        gb = visual.proj.bias.grad.clone()
        dist.all_reduce(gw, op=dist.ReduceOp.SUM)
        dist.all_reduce(gb, op=dist.ReduceOp.SUM)
        torch.testing.assert_close(gw, gw_ref, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(gb, gb_ref, atol=1e-6, rtol=1e-6)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class _DivergentVisual(_GlooVisual):
    """Vision stub that, on the diverging rank only, drops one output token so its per-rank
    ``pooler_output`` fails the planned-token-count check.

    ``forward`` follows the ``_GlooVisual`` contract (``pixel_values`` [total_patch_rows,
    in_dim] -> ``pooler_output`` [total_patch_rows / sms**2, hidden]); on the diverging rank
    the returned ``pooler_output`` is shortened by one row.  Used to prove that the
    token-count consensus makes EVERY rank raise instead of deadlocking a peer in all_gather.
    """

    def __init__(self, *, divergent: bool, **kwargs) -> None:
        super().__init__(**kwargs)
        self._divergent = divergent

    def forward(self, pixel_values, grid_thw=None, return_dict=True):
        out = super().forward(pixel_values, grid_thw=grid_thw, return_dict=return_dict)
        if self._divergent:
            out.pooler_output = out.pooler_output[:-1]  # one token short -> per-rank count mismatch
        return out


def _divergent_count_worker(rank: int, world_size: int, port: int) -> None:
    """One rank's visual output diverges from its planned token count; ALL ranks must raise.

    A bare per-rank ``raise`` before the all_gather would hang the non-diverging rank; the
    group consensus (all-reduce of a boolean flag) must make every rank raise a ``ValueError``
    instead.  A short init timeout bounds the collective so a regression fails fast rather
    than hanging CI.
    """
    try:
        _init_gloo(rank, world_size, port, timeout=timedelta(seconds=60))
        torch.set_num_threads(1)
        from nemo_automodel.components.distributed import cp_vision_frame_shard as vs

        torch.manual_seed(0)
        # >= world frame units -> balanced path (not the pad path); only the last rank diverges.
        visual = _DivergentVisual(divergent=(rank == world_size - 1))
        grid = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)  # 2 frame units, 4 tokens each
        gen = torch.Generator().manual_seed(7)
        pixel = torch.randn(int(grid.prod(dim=-1).sum()), 8, generator=gen)

        config = vs.CpVisionFrameShardingConfig(enabled=True, min_tokens=0)
        token = vs.set_cp_vision_group(dist.group.WORLD, config=config)
        try:
            # EVERY rank must raise (consensus), not only the diverging one.
            with pytest.raises(ValueError, match="cp_vision_frame_shard"):
                vs.maybe_distribute_visual(visual, pixel, grid)
        finally:
            vs.reset_cp_vision_group(token)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_cp_vision_frame_shard_two_rank_gloo_forward_backward_parity():
    mp.spawn(_parity_worker, args=(2, _free_port()), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_cp_vision_frame_shard_two_rank_gloo_divergent_count_all_ranks_raise():
    mp.spawn(_divergent_count_worker, args=(2, _free_port()), nprocs=2, join=True)


def _real_tower_worker(rank: int, world_size: int, port: int) -> None:
    """Run the real Qwen3.5 vision tower through two-rank gloo sharding.

    Assert pooled-output parity with the replicated full forward. This is the exact tower
    used by the dense Qwen3.5 VLM CP pre-embed.
    """
    try:
        _init_gloo(rank, world_size, port, timeout=timedelta(seconds=120))
        torch.set_num_threads(1)
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

        from nemo_automodel.components.distributed import cp_vision_frame_shard as vs

        cfg = Qwen3_5VisionConfig(
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            out_hidden_size=32,
            depth=2,
            patch_size=4,
            temporal_patch_size=2,
            spatial_merge_size=2,
            in_channels=3,
            num_position_embeddings=64,
        )
        torch.manual_seed(0)  # identical weights on every rank (mirrors the FSDP all-gather)
        tower = Qwen3_5VisionModel(cfg).eval().to(torch.float32)
        feat = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        grid = torch.tensor([[1, 4, 4], [1, 2, 2], [1, 4, 2], [1, 2, 4]], dtype=torch.long)
        gen = torch.Generator().manual_seed(7)
        pixel = torch.randn(int(grid.prod(dim=-1).sum()), feat, generator=gen)

        with torch.no_grad():
            rep = tower(pixel, grid_thw=grid, return_dict=True)
            config = vs.CpVisionFrameShardingConfig(enabled=True, min_tokens=0)
            token = vs.set_cp_vision_group(dist.group.WORLD, config=config)
            try:
                out = vs.maybe_distribute_visual(tower, pixel, grid)
            finally:
                vs.reset_cp_vision_group(token)

        torch.testing.assert_close(out.pooler_output, rep.pooler_output, atol=1e-5, rtol=1e-5)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
@pytest.mark.skipif(not _HAS_QWEN3_5, reason="transformers Qwen3.5 vision tower is not available")
def test_cp_vision_frame_shard_two_rank_gloo_real_qwen3_5_tower_forward_parity():
    mp.spawn(_real_tower_worker, args=(2, _free_port()), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_cp_vision_frame_shard_two_rank_gloo_pad_path_parity():
    mp.spawn(_pad_path_worker, args=(2, _free_port()), nprocs=2, join=True)


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is not available")
def test_cp_vision_frame_shard_two_rank_gloo_local_patch_floor_parity():
    mp.spawn(_local_patch_floor_worker, args=(2, _free_port()), nprocs=2, join=True)
