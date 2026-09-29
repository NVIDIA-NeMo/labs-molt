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

"""Composition tests for the real Qwen3.5 vision tower and CP vision frame sharding.

The dense Qwen3.5 VLM runs Transformers' ``Qwen3_5VisionModel`` in its CP pre-embed and,
when the CP vision group is published, routes it through
``cp_vision_frame_shard.maybe_distribute_visual``. The helper relies on the exact tower's
per-frame independence, so

    visual(all_frames).pooler_output == concat_r visual(frames_of_rank_r).pooler_output

up to numerical precision for ``pooler_output``.

These CPU tests build a small real ``Qwen3_5VisionModel`` and assert that property using the
module's OWN contiguous partitioner (``_contiguous_balanced_bounds``) -- i.e. the tower plus the
real partition/gather-order logic together reproduce the replicated full forward.  The
collective/backward path of ``maybe_distribute_visual`` itself is covered (with real gloo
collectives) in ``test_cp_vision_frame_shard_gloo.py``.
"""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.distributed import cp_vision_frame_shard as vs

# The real tower class only exists in recent transformers; skip cleanly otherwise.
_qwen3_5 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")
Qwen3_5VisionConfig = pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5").Qwen3_5VisionConfig
Qwen3_5VisionModel = _qwen3_5.Qwen3_5VisionModel


def _tiny_tower() -> Qwen3_5VisionModel:
    """Build a small real Qwen3.5 vision tower on CPU in fp32."""
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
    torch.manual_seed(0)
    return Qwen3_5VisionModel(cfg).eval().to(torch.float32)


def _patch_feature_dim(tower: Qwen3_5VisionModel) -> int:
    """Return the flattened input dimension of one temporal patch."""
    c = tower.config
    return c.in_channels * c.temporal_patch_size * c.patch_size * c.patch_size


def _pixels(tower: Qwen3_5VisionModel, grid: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Create deterministic pixel rows for a temporal/spatial grid.

    Args:
        tower: Vision tower whose patch geometry determines the feature dimension.
        grid: Integer tensor with shape ``[num_entries, 3]`` containing ``(t, h, w)``.
        seed: Random seed used to generate the pixel rows.

    Returns:
        Float tensor with shape ``[sum(t*h*w), patch_feature_dim]``.
    """
    total = int(grid.prod(dim=-1).sum())
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(total, _patch_feature_dim(tower), generator=gen)


def _assert_gathered_matches_full(
    tower: Qwen3_5VisionModel,
    pixel: torch.Tensor,
    grid: torch.Tensor,
    cuts: list[int],
) -> None:
    """Forward each contiguous entry-slice ``[cuts[r], cuts[r+1])`` through the real tower and
    assert the concatenation equals the full pooled forward. This verifies frame independence
    and that concatenation preserves original entry order.

    Args:
        tower: Real Qwen3.5 vision tower.
        pixel: Frame-contiguous pixel rows with shape ``[sum(t*h*w), patch_feature_dim]``.
        grid: Integer tensor with shape ``[num_entries, 3]`` containing ``(t, h, w)``.
        cuts: Contiguous entry boundaries with length ``world_size + 1``.
    """
    full = tower(pixel, grid_thw=grid, return_dict=True)
    pix_bounds = [0] + grid.prod(dim=-1).cumsum(0).tolist()

    pooler_blocks = []
    for r in range(len(cuts) - 1):
        lo, hi = cuts[r], cuts[r + 1]
        out = tower(pixel[pix_bounds[lo] : pix_bounds[hi]], grid_thw=grid[lo:hi], return_dict=True)
        pooler_blocks.append(out.pooler_output)

    gathered_pooler = torch.cat(pooler_blocks, dim=0)
    assert gathered_pooler.shape == full.pooler_output.shape
    torch.testing.assert_close(gathered_pooler, full.pooler_output, atol=1e-5, rtol=1e-5)


# entries of varied size; every (t*h*w) is a multiple of spatial_merge_size**2 = 4.
_IMAGE_ENTRIES = [(1, 4, 4), (1, 2, 2), (1, 4, 2), (1, 2, 4), (1, 6, 2)]


@pytest.mark.parametrize("world", [2, 3, 4])
def test_real_tower_image_entries_frame_independent(world):
    """Real tower + module partitioner: contiguous entry partition reassembles to the full
    forward at world 2-4."""
    tower = _tiny_tower()
    grid = torch.tensor(_IMAGE_ENTRIES, dtype=torch.long)
    pixel = _pixels(tower, grid, seed=1)

    cuts = vs._contiguous_balanced_bounds(grid.prod(dim=-1), world, cost_alpha_source=tower)
    assert cuts is not None and len(cuts) == world + 1
    _assert_gathered_matches_full(tower, pixel, grid, cuts)


@pytest.mark.parametrize("world", [2, 4])
def test_real_tower_single_video_frame_split(world):
    """A single video (t>1) split across ranks at FRAME granularity still reassembles to the
    full forward -- the entry-level partitioner falls back (1 entry < world), so build per-rank
    ``(count, h, w)`` grids exactly as ``maybe_distribute_visual`` coalesces same-entry frames."""
    tower = _tiny_tower()
    t, h, w = 8, 4, 4
    grid = torch.tensor([[t, h, w]], dtype=torch.long)
    pixel = _pixels(tower, grid, seed=2)
    rows_per_frame = h * w

    # frame-level partition (module falls back at entry level for 1 entry < world).
    frame_patches = torch.full((t,), rows_per_frame, dtype=torch.long)
    frame_cuts = vs._contiguous_balanced_bounds(frame_patches, world, cost_alpha_source=tower)
    assert frame_cuts is not None

    full = tower(pixel, grid_thw=grid, return_dict=True)
    pooler_blocks = []
    for r in range(world):
        count = frame_cuts[r + 1] - frame_cuts[r]
        lo_row = frame_cuts[r] * rows_per_frame
        hi_row = frame_cuts[r + 1] * rows_per_frame
        out = tower(
            pixel[lo_row:hi_row],
            grid_thw=torch.tensor([[count, h, w]], dtype=torch.long),
            return_dict=True,
        )
        pooler_blocks.append(out.pooler_output)

    torch.testing.assert_close(torch.cat(pooler_blocks, dim=0), full.pooler_output, atol=1e-5, rtol=1e-5)
