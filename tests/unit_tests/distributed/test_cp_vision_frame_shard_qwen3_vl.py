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

"""Composition tests for the real Qwen3-VL vision tower and CP partitioning."""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.distributed import cp_vision_frame_shard as vision_shard

qwen3_vl = pytest.importorskip("transformers.models.qwen3_vl.modeling_qwen3_vl")
Qwen3VLVisionConfig = pytest.importorskip("transformers.models.qwen3_vl.configuration_qwen3_vl").Qwen3VLVisionConfig
Qwen3VLVisionModel = qwen3_vl.Qwen3VLVisionModel


def _tiny_tower() -> Qwen3VLVisionModel:
    """Build a small real Qwen3-VL vision tower on CPU in fp32."""
    config = Qwen3VLVisionConfig(
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
        deepstack_visual_indexes=[0, 1],
    )
    torch.manual_seed(0)
    return Qwen3VLVisionModel(config).eval().to(torch.float32)


def _pixels(tower: Qwen3VLVisionModel, grid: torch.Tensor, *, seed: int) -> torch.Tensor:
    """Create deterministic patch rows for a temporal/spatial grid.

    Args:
        tower: Vision tower defining the patch geometry.
        grid: Integer tensor of shape ``[num_entries, 3]`` containing ``(t, h, w)``.
        seed: Random seed for the patch values.

    Returns:
        Float tensor of shape ``[sum(t*h*w), patch_dim]`` in entry/frame order.
    """
    config = tower.config
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(int(grid.prod(dim=-1).sum()), patch_dim, generator=generator)


def _assert_partition_matches_full(
    tower: Qwen3VLVisionModel,
    pixel_values: torch.Tensor,
    grid: torch.Tensor,
    cuts: list[int],
) -> None:
    """Compare a gathered entry partition with one full-tower forward.

    Args:
        tower: Real Qwen3-VL vision tower.
        pixel_values: Patch rows of shape ``[sum(t*h*w), patch_dim]``.
        grid: Integer tensor of shape ``[num_entries, 3]``.
        cuts: Contiguous entry boundaries of length ``world_size + 1``.
    """
    full = tower(pixel_values, grid_thw=grid, return_dict=True)
    pixel_bounds = [0] + grid.prod(dim=-1).cumsum(0).tolist()
    outputs = []
    for lower, upper in zip(cuts[:-1], cuts[1:]):
        outputs.append(
            tower(
                pixel_values[pixel_bounds[lower] : pixel_bounds[upper]],
                grid_thw=grid[lower:upper],
                return_dict=True,
            )
        )

    torch.testing.assert_close(
        torch.cat([output.pooler_output for output in outputs]),
        full.pooler_output,
        atol=1e-5,
        rtol=1e-5,
    )
    for layer_idx, expected in enumerate(full.deepstack_features):
        actual = torch.cat([output.deepstack_features[layer_idx] for output in outputs])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_real_qwen3_vl_tower_entries_are_independent(world_size: int) -> None:
    """Entry partitioning preserves pooled and DeepStack feature ordering."""
    tower = _tiny_tower()
    grid = torch.tensor([(1, 4, 4), (1, 2, 2), (1, 4, 2), (1, 2, 4), (1, 6, 2)])
    pixel_values = _pixels(tower, grid, seed=1)
    cuts = vision_shard._contiguous_balanced_bounds(grid.prod(dim=-1), world_size, cost_alpha_source=tower)
    assert cuts is not None
    _assert_partition_matches_full(tower, pixel_values, grid, cuts)


@pytest.mark.parametrize("world_size", [2, 4])
def test_real_qwen3_vl_tower_single_video_is_frame_independent(world_size: int) -> None:
    """Frame partitioning of one video preserves pooled and DeepStack outputs."""
    tower = _tiny_tower()
    temporal, height, width = 8, 4, 4
    grid = torch.tensor([[temporal, height, width]])
    pixel_values = _pixels(tower, grid, seed=2)
    frame_cuts = vision_shard._contiguous_balanced_bounds(
        torch.full((temporal,), height * width),
        world_size,
        cost_alpha_source=tower,
    )
    assert frame_cuts is not None
    full = tower(pixel_values, grid_thw=grid, return_dict=True)
    outputs = []
    for lower, upper in zip(frame_cuts[:-1], frame_cuts[1:]):
        outputs.append(
            tower(
                pixel_values[lower * height * width : upper * height * width],
                grid_thw=torch.tensor([[upper - lower, height, width]]),
                return_dict=True,
            )
        )

    torch.testing.assert_close(
        torch.cat([output.pooler_output for output in outputs]),
        full.pooler_output,
        atol=1e-5,
        rtol=1e-5,
    )
    for layer_idx, expected in enumerate(full.deepstack_features):
        actual = torch.cat([output.deepstack_features[layer_idx] for output in outputs])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
