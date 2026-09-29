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

"""CPU unit tests for frame-level context-parallel vision-tower sharding
(nemo_automodel/components/distributed/cp_vision_frame_shard.py).

Headline guarantee: distributing the vision tower across CP ranks does not change the
result.  Because the ViT attends per IMAGE/FRAME ("entry") via ``cu_seqlens``, entries are
independent and

    visual(all_entries) == concat_r visual(entries_owned_by_rank_r)    (entry order)

holds up to numerical precision (``allclose``) for both the forward embeddings AND the
vision-parameter gradients.  These tests prove that on CPU with the CP ranks simulated
in-process and the collectives mocked -- no GPU / no distributed init.
"""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.distributed import cp_vision_frame_shard as vs


# --------------------------------------------------------------------------------------
# Stub vision tower: per-row linear ("vision params") + per-entry 2x2-style merge.
# Entry-independent by construction (the merge groups consecutive sms_sq patch rows, and
# every entry's patch count is a multiple of sms_sq), so it faithfully models the real
# ViT's property that entries do not interact.
# --------------------------------------------------------------------------------------
class _StubVisual(torch.nn.Module):
    def __init__(self, in_dim=8, hidden=6, sms=2, with_deepstack=False, n_deepstack=2, bias=False):
        super().__init__()
        self.spatial_merge_size = sms
        self.dtype = torch.float32
        # bias=True gives a "vision param" whose grad depends ONLY on the upstream grad
        # (d_loss/d_bias = sum of upstream over rows), independent of the input -- used by
        # the end-to-end pad backward test to prove the dummy tail is sliced off (zero
        # upstream) rather than merely zero because the dummy pixels are zero.
        self.proj = torch.nn.Linear(in_dim, hidden, bias=bias)
        self.with_deepstack = with_deepstack
        self.n_deepstack = n_deepstack

    def forward(self, pixel_values, grid_thw=None, return_dict=True):
        sms_sq = self.spatial_merge_size**2
        x = self.proj(pixel_values)  # [total_patches, hidden] — per row, entry-independent
        merged = x.reshape(-1, sms_sq, x.shape[-1]).mean(dim=1)  # [total_patches/sms_sq, hidden]
        out = SimpleNamespace(pooler_output=merged, last_hidden_state=x, deepstack_features=None)
        if self.with_deepstack:
            out.deepstack_features = [merged * (k + 1) for k in range(self.n_deepstack)]
        return out


class _RecordingStubVisual(_StubVisual):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.patch_rows = []

    def forward(self, pixel_values, grid_thw=None, return_dict=True):
        """Record the local patch-row count and delegate to the stub vision tower.

        Args:
            pixel_values: Tensor of shape [patch_rows, patch_dim].
            grid_thw: Optional tensor of shape [entries, 3] containing ``(time, height, width)`` rows.
            return_dict: Whether to return the vision output as a structured object.

        Returns:
            Vision output whose ``pooler_output`` tensor has shape [tokens, hidden],
            ``last_hidden_state`` tensor has shape [patch_rows, hidden], and optional
            ``deepstack_features`` tensors each have shape [tokens, hidden].
        """
        self.patch_rows.append(int(pixel_values.shape[0]))
        return super().forward(pixel_values, grid_thw=grid_thw, return_dict=return_dict)


class _RecorderVisual(torch.nn.Module):
    """Parameter-free stub that records its call args and returns a fixed output.

    ``pixel_values`` / ``grid_thw`` are treated as opaque pass-through objects (may be
    ``None``); the returned ``pooler_output`` is a constant ``[1, 4]`` zero tensor.
    """

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, pixel_values, grid_thw=None, return_dict=True):
        self.calls.append((pixel_values, grid_thw))
        return SimpleNamespace(pooler_output=torch.zeros(1, 4), last_hidden_state=None, deepstack_features=None)


def _grid(entries):
    """entries: list of (t, h, w); returns a [N, 3] long grid_thw."""
    return torch.tensor(entries, dtype=torch.long)


def _pixels(grid, in_dim=8, seed=0):
    total_patches = int(grid.prod(dim=-1).sum())
    g = torch.Generator().manual_seed(seed)
    return torch.randn(total_patches, in_dim, generator=g)


def _policy(*, enabled=True, min_tokens=0, min_frames=32, min_local_patch_rows=0, cost_alpha="auto"):
    return vs.CpVisionFrameShardingConfig(
        enabled=enabled,
        min_tokens=min_tokens,
        min_frames=min_frames,
        min_local_patch_rows=min_local_patch_rows,
        cost_alpha=cost_alpha,
    )


# entries with varied sizes; every patch count (t*h*w) is a multiple of sms_sq=4.
_ENTRIES = [(1, 2, 2), (1, 4, 4), (1, 2, 4), (1, 6, 4), (1, 2, 2), (1, 4, 6)]


# ======================================================================================
# A. partition: contiguous, >=1 entry/rank, covers all, attention-cost balanced, None when N<world
# ======================================================================================
@pytest.mark.parametrize("world", [2, 3, 4])
def test_partition_is_contiguous_complete_and_nonempty(world):
    patches = _grid(_ENTRIES).prod(dim=-1)
    cuts = vs._contiguous_balanced_bounds(patches, world)
    assert cuts[0] == 0 and cuts[-1] == len(_ENTRIES)
    assert len(cuts) == world + 1
    # strictly increasing => contiguous, every rank gets >=1 entry, all entries covered
    for r in range(world):
        assert cuts[r + 1] > cuts[r], f"rank {r} got 0 entries (world={world})"


@pytest.mark.parametrize("world", [2, 4])
def test_partition_is_balanced_by_attention_cost(world):
    patches = _grid(_ENTRIES).prod(dim=-1)
    cuts = vs._contiguous_balanced_bounds(patches, world)
    costs = patches * patches
    loads = [int(costs[cuts[r] : cuts[r + 1]].sum()) for r in range(world)]
    total = int(costs.sum())
    # each rank within one max-entry cost of the ideal share (contiguous balance is approximate)
    biggest_entry = int(costs.max())
    for load in loads:
        assert abs(load - total / world) <= biggest_entry


def test_partition_balances_large_frame_attention_hotspots():
    patches = torch.tensor([100, 1, 1, 1, 100, 1, 1, 1], dtype=torch.long)
    cuts = vs._contiguous_balanced_bounds(patches, 2)
    costs = patches * patches
    loads = [int(costs[cuts[r] : cuts[r + 1]].sum()) for r in range(2)]

    assert max(loads) - min(loads) <= int(costs.max())


def test_partition_none_when_fewer_entries_than_ranks():
    patches = _grid([(1, 2, 2), (1, 2, 2)]).prod(dim=-1)  # 2 entries
    assert vs._contiguous_balanced_bounds(patches, 4) is None


@pytest.mark.parametrize(
    ("n_frames", "world", "expected_cuts"),
    [
        (4, 2, [0, 2, 4]),  # 4 equal-cost frames / world 2 -> even [2, 2] (not [1, 3])
        (6, 3, [0, 2, 4, 6]),  # 6 equal-cost frames / world 3 -> even [2, 2, 2]
        (8, 4, [0, 2, 4, 6, 8]),  # 8 equal-cost frames / world 4 -> even [2, 2, 2, 2]
    ],
)
def test_partition_equal_cost_frames_split_evenly(n_frames, world, expected_cuts):
    """Equal-cost frames must split evenly.  Every cut lands exactly on a cumulative-cost
    boundary, the case where ``bisect_left`` undershot by one (e.g. [1, 3] for 4 frames /
    world 2); ``bisect_right`` places the cut past the exact boundary for an even split."""
    patches = torch.full((n_frames,), 3, dtype=torch.long)  # identical per-frame cost
    cuts = vs._contiguous_balanced_bounds(patches, world)
    assert cuts == expected_cuts
    loads = [cuts[r + 1] - cuts[r] for r in range(world)]
    assert max(loads) - min(loads) == 0  # perfectly even


def test_partition_uneven_cost_exact_boundary_splits_evenly():
    """Packed mixed-frame case: UNEVEN per-frame costs whose ideal split lands exactly on a
    cumulative-cost boundary.  ``bisect_left`` would undershoot (skewed load), ``bisect_right``
    places the whole boundary frame on the lower rank for a balanced split.

    patches=[3, 4, 5] -> costs (alpha=0, p*p)=[9, 16, 25], total=50, r=1 target=25 == cum[1].
    Even split is loads [9+16, 25] = [25, 25]; the off-by-one bug gave [9, 41].
    """
    patches = torch.tensor([3, 4, 5], dtype=torch.long)
    cuts = vs._contiguous_balanced_bounds(patches, 2)
    assert cuts == [0, 2, 3]
    costs = patches * patches
    loads = [int(costs[cuts[r] : cuts[r + 1]].sum()) for r in range(2)]
    assert loads == [25, 25]  # perfectly balanced, not [9, 41]


def test_partition_cut_lands_exactly_on_cumulative_sum():
    """Boundary case: a target that coincides with a cumulative sum must put the whole frame
    that closes that sum on the LOWER rank (``bisect_right``), not split before it."""
    # costs = [1, 1, 2] (alpha=0 -> p*p on patches [1, 1, sqrt2]-ish); use explicit patches so
    # cum = [1, 2, 4], total = 4, world 2 -> target = 2 lands exactly on cum[1].
    patches = torch.tensor([1, 1, 2], dtype=torch.long)  # costs [1, 1, 4], cum [1, 2, 6]
    # target for r=1 = total(6)*1/2 = 3; cum = [1, 2, 6]; bisect_right(cum, 3) = 2 -> cut at 2.
    cuts = vs._contiguous_balanced_bounds(patches, 2)
    assert cuts == [0, 2, 3]
    # exact-boundary variant: patches [1, 1, 1, 1] -> cum [1, 2, 3, 4], target 2 == cum[1].
    even = vs._contiguous_balanced_bounds(torch.tensor([1, 1, 1, 1], dtype=torch.long), 2)
    assert even == [0, 2, 4]  # cut placed PAST the exact boundary -> [2, 2]


@pytest.mark.parametrize(
    ("source", "expected_hidden", "expected_alpha"),
    [
        # Qwen3-VL / Qwen3.5 visual modules expose visual.config.hidden_size.
        (SimpleNamespace(config=SimpleNamespace(hidden_size=1152)), 1152, 3456),
        # Composite VLM configs must prefer vision_config over the text width.
        (
            SimpleNamespace(
                hidden_size=7168,
                vision_config=SimpleNamespace(hidden_size=1024),
            ),
            1024,
            3072,
        ),
        # Nemotron Omni uses vit_hidden_size on the multimodal config.
        (
            SimpleNamespace(
                vit_hidden_size=1280,
                llm_config=SimpleNamespace(hidden_size=7168),
            ),
            1280,
            3840,
        ),
        # RADIO-style patch generators commonly call the same width embed_dim.
        (SimpleNamespace(patch_generator=SimpleNamespace(embed_dim=768)), 768, 2304),
    ],
)
def test_auto_cost_alpha_discovers_supported_vision_widths(source, expected_hidden, expected_alpha):
    assert vs._infer_vision_hidden_size(source) == expected_hidden
    assert vs._vision_cost_alpha(source) == expected_alpha


def test_cost_alpha_override_and_unknown_model_fallback():
    qwen_visual = SimpleNamespace(config=SimpleNamespace(hidden_size=1152))

    assert vs.CpVisionFrameShardingConfig().cost_alpha == "auto"
    assert vs.CpVisionFrameShardingConfig().min_frames == 32

    with pytest.raises(ValueError, match="min_frames must be a non-negative integer"):
        vs.CpVisionFrameShardingConfig(min_frames=-1)
    assert vs._vision_cost_alpha(qwen_visual, _policy(cost_alpha="auto")) == 3456
    assert vs._vision_cost_alpha(qwen_visual, _policy(cost_alpha=None)) == 3456
    assert vs._vision_cost_alpha(qwen_visual, _policy(cost_alpha=777)) == 777
    assert vs._vision_cost_alpha(qwen_visual, _policy(cost_alpha=0)) == 0
    assert vs._vision_cost_alpha(qwen_visual) == 3456
    assert vs._vision_cost_alpha(SimpleNamespace()) == 0

    for invalid_alpha in (-1, True, "AUTO", "not-auto"):
        with pytest.raises(ValueError, match="cost_alpha"):
            vs.CpVisionFrameShardingConfig(cost_alpha=invalid_alpha)


def test_min_local_patch_rows_default_override_and_validation():
    assert vs.CpVisionFrameShardingConfig().min_local_patch_rows == 96
    assert vs.CpVisionFrameShardingConfig(min_local_patch_rows=0).min_local_patch_rows == 0

    for invalid_floor in (-1, True, 1.5, "96"):
        with pytest.raises(ValueError, match="min_local_patch_rows"):
            vs.CpVisionFrameShardingConfig(min_local_patch_rows=invalid_floor)


@pytest.mark.parametrize(
    ("local_rows", "frame_rows", "floor", "expected"),
    [
        (4, 4, 96, 23),
        (8, 4, 96, 22),
        (92, 4, 96, 1),
        (96, 4, 96, 0),
        (100, 4, 96, 0),
        (4, 0, 96, 0),
        (4, 4, 0, 0),
    ],
)
def test_padding_frames_for_min_patch_rows(local_rows, frame_rows, floor, expected):
    assert vs._padding_frames_for_min_patch_rows(local_rows, frame_rows, floor) == expected


def test_partition_cost_alpha_flattens_mixed_frame_sizes():
    """Configured cost_alpha adds the linear per-patch term to the cost.

    A pack mixing a few BIG image frames with many small video frames is the pathological
    case for the pure ``p**2`` model: one big frame "costs" as much as thousands of small
    ones, so the small frames heap onto few ranks. Setting the alpha to the linear
    per-patch proxy (~3x ViT hidden) flattens the frame-count imbalance. This test pins the
    partition-level behavior.
    """
    world = 8
    patches = torch.tensor([1024] * 4 + [112] * 400, dtype=torch.long)  # 4 big + 400 small

    cuts0 = vs._contiguous_balanced_bounds(patches, world)
    per0 = [cuts0[r + 1] - cuts0[r] for r in range(world)]

    cuts_a = vs._contiguous_balanced_bounds(patches, world, config=_policy(cost_alpha=3456))
    per_a = [cuts_a[r + 1] - cuts_a[r] for r in range(world)]

    # default (unset) keeps the original pure-quadratic partition: heavily skewed
    ratio0 = max(per0) / max(min(per0), 1)
    ratio_a = max(per_a) / max(min(per_a), 1)
    assert ratio0 > 10
    # alpha=3456 flattens the frame-count imbalance by an order of magnitude
    assert ratio_a < ratio0 / 10
    # both are valid contiguous complete partitions
    for cuts in (cuts0, cuts_a):
        assert cuts[0] == 0 and cuts[-1] == len(patches) and len(cuts) == world + 1


def test_partition_uses_model_aware_alpha_by_default():
    patches = torch.tensor([1024] * 4 + [112] * 400, dtype=torch.long)
    qwen_visual = SimpleNamespace(config=SimpleNamespace(hidden_size=1152))

    auto = vs._contiguous_balanced_bounds(
        patches,
        8,
        cost_alpha_source=qwen_visual,
    )
    explicit = vs._contiguous_balanced_bounds(patches, 8, config=_policy(cost_alpha=3456))

    assert auto == explicit


# ======================================================================================
# B. THE math: visual(full) == concat_r visual(slice_r), forward AND vision-param grad
# ======================================================================================
@pytest.mark.parametrize("world", [2, 3, 4])
def test_distribute_matches_replicate_forward_and_grad(world):
    torch.manual_seed(0)
    grid = _grid(_ENTRIES)
    pixel = _pixels(grid)

    visual = _StubVisual()
    target = torch.randn_like(visual(pixel, grid).pooler_output)

    # replicate: full forward
    visual.zero_grad(set_to_none=True)
    rep = visual(pixel, grid).pooler_output
    (rep * target).sum().backward()
    g_rep = visual.proj.weight.grad.clone()

    # distributed: per-rank slice forward, concat in rank order (= entry order, contiguous)
    cuts = vs._contiguous_balanced_bounds(grid.prod(dim=-1), world)
    pix_bounds = [0] + grid.prod(dim=-1).cumsum(0).tolist()
    visual.zero_grad(set_to_none=True)
    blocks = []
    for r in range(world):
        lp = pixel[pix_bounds[cuts[r]] : pix_bounds[cuts[r + 1]]]
        lg = grid[cuts[r] : cuts[r + 1]]
        blocks.append(visual(lp, lg).pooler_output)
    dist_out = torch.cat(blocks, dim=0)
    (dist_out * target).sum().backward()
    g_dist = visual.proj.weight.grad.clone()

    assert dist_out.shape == rep.shape
    assert torch.allclose(dist_out, rep, atol=1e-6), "forward embeds differ"
    assert torch.allclose(g_dist, g_rep, atol=1e-6), "vision-param grads differ"


# Video entries carry t>1 (multiple temporal frames -> grid_thw.prod() = t*h*w patches),
# the case where ViT sharding pays off most.  The helper treats each entry as an opaque
# unit (patches = grid.prod, tokens = patches//sms^2), so a t>1 entry is just a
# larger-patch-count entry; this locks that path.  Entry patch counts are multiples of
# sms^2=4 so the stub's per-group merge never crosses an entry boundary.
@pytest.mark.parametrize("world", [2, 3])
def test_distribute_matches_replicate_with_video_entries(world):
    torch.manual_seed(3)
    video_mix = [(1, 2, 2), (2, 2, 4), (1, 4, 4), (2, 2, 2), (1, 2, 4), (2, 4, 2)]
    grid = _grid(video_mix)
    assert (grid[:, 0] > 1).any(), "fixture must include video (t>1) entries"
    pixel = _pixels(grid, seed=7)

    visual = _StubVisual()
    target = torch.randn_like(visual(pixel, grid).pooler_output)

    visual.zero_grad(set_to_none=True)
    rep = visual(pixel, grid).pooler_output
    (rep * target).sum().backward()
    g_rep = visual.proj.weight.grad.clone()

    cuts = vs._contiguous_balanced_bounds(grid.prod(dim=-1), world)
    pix_bounds = [0] + grid.prod(dim=-1).cumsum(0).tolist()
    visual.zero_grad(set_to_none=True)
    blocks = [
        visual(pixel[pix_bounds[cuts[r]] : pix_bounds[cuts[r + 1]]], grid[cuts[r] : cuts[r + 1]]).pooler_output
        for r in range(world)
    ]
    dist_out = torch.cat(blocks, dim=0)
    (dist_out * target).sum().backward()
    g_dist = visual.proj.weight.grad.clone()

    assert torch.allclose(dist_out, rep, atol=1e-6), "video forward embeds differ"
    assert torch.allclose(g_dist, g_rep, atol=1e-6), "video vision-param grads differ"


@pytest.mark.parametrize("world", [4, 8])
def test_pad_dummies_match_replicate_forward_and_grad(world):
    """Pad path math: with fewer frame-units than ranks we append minimal (1,sms,sms) zero
    dummy frames (1 frame/rank), concat all per-rank blocks, then SLICE off the dummy tail.
    The sliced forward AND the vision-param grad must equal the replicated full forward --
    i.e. the dummy frames are dropped before the loss and contribute exactly zero gradient."""
    torch.manual_seed(0)
    sms = 2
    grid = _grid([(1, 2, 2), (1, 4, 4)])  # 2 image units < world (4 or 8)
    pixel = _pixels(grid)
    n_units = int(grid[:, 0].sum())
    assert n_units < world

    visual = _StubVisual(sms=sms)
    target = torch.randn_like(visual(pixel, grid).pooler_output)

    visual.zero_grad(set_to_none=True)
    rep = visual(pixel, grid).pooler_output
    (rep * target).sum().backward()
    g_rep = visual.proj.weight.grad.clone()

    # pad with (world - n_units) minimal dummy frames, 1 frame unit per rank
    n_pad = world - n_units
    d_rows = sms * sms
    pix = torch.cat([pixel, pixel.new_zeros(n_pad * d_rows, pixel.shape[1])], dim=0)
    f_patches = [int(h) * int(w) for t, h, w in grid.tolist() for _ in range(int(t))] + [d_rows] * n_pad
    bounds = [0]
    for p in f_patches:
        bounds.append(bounds[-1] + p)
    n_real_tokens = sum(f_patches[i] // (sms * sms) for i in range(n_units))

    visual.zero_grad(set_to_none=True)
    blocks = [visual(pix[bounds[r] : bounds[r + 1]]).pooler_output for r in range(world)]
    dist_out = torch.cat(blocks, dim=0)[:n_real_tokens]  # drop dummy tail
    (dist_out * target).sum().backward()
    g_dist = visual.proj.weight.grad.clone()

    assert dist_out.shape == rep.shape
    assert torch.allclose(dist_out, rep, atol=1e-6), "pad forward embeds differ"
    assert torch.allclose(g_dist, g_rep, atol=1e-6), "pad vision-param grads differ (dummy leaked grad)"


# ======================================================================================
# C. _all_gather_var_tokens: forward concatenates per-rank blocks; backward sums grad
# ======================================================================================
@pytest.mark.parametrize("world", [2, 3])
def test_all_gather_var_tokens_forward_backward(monkeypatch, world):
    # per-rank token counts differ -> exercises the pad-to-max + slice-back path
    token_counts = [3, 1, 2][:world]
    H = 5
    rank_blocks = [torch.randn(n, H) for n in token_counts]
    max_tok = max(token_counts)

    def fake_all_gather(out_list, x, group=None):
        # simulate every rank's padded [max_tok, H] block being gathered
        for o, blk in zip(out_list, rank_blocks):
            padded = torch.cat([blk, blk.new_zeros(max_tok - blk.shape[0], H)], 0)
            o.copy_(padded)

    def fake_reduce_scatter(local, chunks, op=None, group=None):
        local.copy_(sum(chunks))

    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: world)
    monkeypatch.setattr(torch.distributed, "all_gather", fake_all_gather)
    monkeypatch.setattr(torch.distributed, "reduce_scatter", fake_reduce_scatter)

    local = rank_blocks[0].clone().requires_grad_(True)
    # pad local to max so it matches the all-gather shape contract
    out = vs._all_gather_var_tokens(local, group=None, world=world, token_counts=token_counts)

    assert out.shape == (sum(token_counts), H)
    # first block (unpadded) == rank 0's data
    assert torch.allclose(out[: token_counts[0]], rank_blocks[0])

    out.sum().backward()
    assert local.grad is not None and local.grad.shape == local.shape


# ======================================================================================
# D. maybe_distribute_visual end-to-end (simulated ranks) + fallbacks + deepstack
# ======================================================================================
def _simulate_maybe_distribute(
    visual,
    pixel,
    grid,
    world,
    rank,
    monkeypatch,
    *,
    grad=False,
    spans_only_cp=True,
    policy=None,
):
    """Drive the real maybe_distribute_visual on `rank` with all ranks' data supplied to a
    mocked all-gather, returning its output object.

    ``grad=True`` additionally mocks ``reduce_scatter`` (the backward of ``_AllGatherSeqDiff``)
    so backward can flow through the production code: it hands THIS rank its own grad chunk
    (in the real collective each rank gets sum-over-processes of ``chunk[rank]``; with one
    process per simulated rank that is just ``chunk[rank]``).  Running every rank and summing
    the per-rank vision grads then reconstructs the full backward -- see
    ``test_pad_backward_through_real_code_matches_replicate``."""
    import torch.distributed as dist

    resolved_policy = policy or _policy()
    sms = visual.spatial_merge_size
    sms_sq = sms**2
    # frame-level units (mirror maybe_distribute_visual): expand (t,h,w) -> t x (h*w patches)
    f_patches = []
    for t, h, w in grid.tolist():
        for _ in range(int(t)):
            f_patches.append(int(h) * int(w))
    n_units = len(f_patches)
    # mirror maybe_distribute_visual: when fewer frame-units than ranks, pad with minimal
    # (sms,sms) dummy frames so every rank runs exactly one frame (cuts = 1 unit/rank).
    if 0 < n_units < world:
        n_pad = world - n_units
        d_rows = sms * sms
        pixel = torch.cat([pixel, pixel.new_zeros(n_pad * d_rows, pixel.shape[1])], dim=0)
        f_patches = f_patches + [d_rows] * n_pad
        cuts = list(range(world + 1))
    else:
        cuts = vs._contiguous_balanced_bounds(torch.tensor(f_patches, dtype=torch.long), world)
    token_counts = [sum(f_patches[i] // sms_sq for i in range(cuts[r], cuts[r + 1])) for r in range(world)]
    max_tok = max(token_counts)
    pix_bounds = [0]
    for p in f_patches:
        pix_bounds.append(pix_bounds[-1] + p)

    def _padded_local(field_getter):
        padded = []
        for r in range(world):
            lp = pixel[pix_bounds[cuts[r]] : pix_bounds[cuts[r + 1]]]
            local_real_start = min(cuts[r], n_units)
            local_real_end = min(cuts[r + 1], n_units)
            if resolved_policy.min_local_patch_rows > 0 and local_real_end > local_real_start:
                frame_patch_rows = f_patches[local_real_end - 1]
                pad_frames = vs._padding_frames_for_min_patch_rows(
                    int(lp.shape[0]),
                    frame_patch_rows,
                    resolved_policy.min_local_patch_rows,
                )
                if pad_frames:
                    lp = torch.cat(
                        [lp, lp.new_zeros(pad_frames * frame_patch_rows, lp.shape[1])],
                        dim=0,
                    )
            t = field_getter(visual(lp))[: token_counts[r]]
            if t.shape[0] < max_tok:
                t = torch.cat([t, t.new_zeros(max_tok - t.shape[0], t.shape[-1])], 0)
            padded.append(t.detach())
        return padded

    # all-gather is called once for pooler, then once per deepstack layer (in order),
    # so queue the per-call padded data.
    queue = [_padded_local(lambda o: o.pooler_output)]
    sample = visual(pixel[: pix_bounds[cuts[1]]])  # stub ignores grid; just probe for deepstack
    if getattr(sample, "deepstack_features", None) is not None:
        for k in range(len(sample.deepstack_features)):
            queue.append(_padded_local(lambda o, k=k: o.deepstack_features[k]))

    call = {"i": 0}

    def fake_all_gather(out_list, x, group=None):
        data = queue[call["i"]]
        call["i"] += 1
        for o, p in zip(out_list, data):
            o.copy_(p)

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: world)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(dist, "all_gather", fake_all_gather)
    # The token-count consensus all-reduce is a no-op with a single in-process rank: the
    # stub always produces the planned token count, so the flag stays 0 and no rank raises.
    monkeypatch.setattr(dist, "all_reduce", lambda tensor, op=None, group=None: None)
    if grad:
        # _AllGatherSeqDiff.backward does reduce_scatter(local, chunks, SUM): hand rank
        # `rank` its own chunk (no other process contributes in this in-process sim).
        def fake_reduce_scatter(local, chunks, op=None, group=None):
            local.copy_(chunks[rank])

        monkeypatch.setattr(dist, "reduce_scatter", fake_reduce_scatter)

    tok = vs.set_cp_vision_group(
        object(),
        config=resolved_policy,
        spans_only_cp=spans_only_cp,
    )  # any non-None group activates the path
    try:
        return vs.maybe_distribute_visual(visual, pixel, grid)
    finally:
        vs.reset_cp_vision_group(tok)


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("rank", [0, 1])
def test_maybe_distribute_matches_replicate(monkeypatch, world, rank):
    torch.manual_seed(1)
    grid = _grid(_ENTRIES)
    pixel = _pixels(grid)
    visual = _StubVisual()

    rep = visual(pixel, grid).pooler_output
    out = _simulate_maybe_distribute(visual, pixel, grid, world, rank, monkeypatch)

    assert out.pooler_output.shape == rep.shape
    assert torch.allclose(out.pooler_output, rep, atol=1e-6)


@pytest.mark.parametrize("world", [2, 4])
def test_maybe_distribute_gathers_deepstack(monkeypatch, world):
    torch.manual_seed(2)
    grid = _grid(_ENTRIES)
    pixel = _pixels(grid)
    visual = _StubVisual(with_deepstack=True, n_deepstack=3)

    rep = visual(pixel, grid)
    out = _simulate_maybe_distribute(visual, pixel, grid, world, rank=0, monkeypatch=monkeypatch)

    assert out.deepstack_features is not None
    assert len(out.deepstack_features) == 3
    for got, exp in zip(out.deepstack_features, rep.deepstack_features):
        assert got.shape == exp.shape
        assert torch.allclose(got, exp, atol=1e-6)


def test_min_local_patch_rows_preserves_forward_and_deepstack(monkeypatch):
    grid = _grid([(8, 2, 2)])
    pixel = _pixels(grid, seed=23)
    visual = _RecordingStubVisual(with_deepstack=True)
    replicated = visual(pixel, grid)

    out = _simulate_maybe_distribute(
        visual,
        pixel,
        grid,
        world=4,
        rank=0,
        monkeypatch=monkeypatch,
        policy=_policy(min_local_patch_rows=24),
    )

    assert visual.patch_rows[-1] == 24
    assert torch.equal(out.pooler_output, replicated.pooler_output)
    assert out.deepstack_features is not None
    for got, expected in zip(out.deepstack_features, replicated.deepstack_features):
        assert torch.equal(got, expected)


def test_min_local_patch_rows_does_not_grow_dummy_only_rank(monkeypatch):
    grid = _grid([(1, 2, 2)])
    pixel = _pixels(grid, seed=29)
    visual = _RecordingStubVisual()
    replicated = visual(pixel, grid).pooler_output

    out = _simulate_maybe_distribute(
        visual,
        pixel,
        grid,
        world=4,
        rank=1,
        monkeypatch=monkeypatch,
        policy=_policy(min_local_patch_rows=96),
    )

    assert visual.patch_rows[-1] == 4
    assert torch.equal(out.pooler_output, replicated)


def test_min_local_patch_rows_backward_matches_replicate(monkeypatch):
    torch.manual_seed(31)
    grid = _grid([(8, 2, 2)])
    pixel = _pixels(grid, seed=29)
    visual = _StubVisual(bias=True)
    replicated = visual(pixel, grid).pooler_output
    target = torch.randn_like(replicated)

    visual.zero_grad(set_to_none=True)
    (visual(pixel, grid).pooler_output * target).sum().backward()
    expected_weight_grad = visual.proj.weight.grad.clone()
    expected_bias_grad = visual.proj.bias.grad.clone()

    weight_grads, bias_grads = [], []
    for rank in range(4):
        visual.zero_grad(set_to_none=True)
        out = _simulate_maybe_distribute(
            visual,
            pixel,
            grid,
            world=4,
            rank=rank,
            monkeypatch=monkeypatch,
            grad=True,
            policy=_policy(min_local_patch_rows=24),
        )
        assert torch.equal(out.pooler_output, replicated)
        (out.pooler_output * target).sum().backward()
        weight_grads.append(visual.proj.weight.grad.clone())
        bias_grads.append(visual.proj.bias.grad.clone())

    assert torch.allclose(torch.stack(weight_grads).sum(0), expected_weight_grad, atol=1e-6)
    assert torch.allclose(torch.stack(bias_grads).sum(0), expected_bias_grad, atol=1e-6)


@pytest.mark.parametrize("world", [2, 4])
def test_maybe_distribute_splits_a_single_video(monkeypatch, world):
    """Frame-level sharding: ONE video entry (t>1) with no other entries.  Entry-level would
    fall back (1 entry < world); frame-level splits the video's frames across ranks and the
    gathered embeds still equal the replicated full forward."""
    torch.manual_seed(5)
    grid = _grid([(8, 2, 2)])  # single 8-frame video -> 8 frame-units (4 patches each)
    pixel = _pixels(grid, seed=11)

    # entry-level partition falls back (1 entry < world); frame-level does NOT.
    assert vs._contiguous_balanced_bounds(grid.prod(dim=-1), world) is None
    assert vs._contiguous_balanced_bounds(torch.tensor([4] * 8), world) is not None

    visual = _StubVisual()
    rep = visual(pixel, grid).pooler_output
    out = _simulate_maybe_distribute(visual, pixel, grid, world, rank=0, monkeypatch=monkeypatch)
    assert out.pooler_output.shape == rep.shape
    assert torch.allclose(out.pooler_output, rep, atol=1e-6)


def test_maybe_distribute_falls_back_without_group():
    """No CP group set -> plain replicated visual() call (object identity of pooler)."""
    grid = _grid(_ENTRIES)
    pixel = _pixels(grid)
    visual = _StubVisual()
    vs.reset_cp_vision_group(None)  # ensure no group
    out = vs.maybe_distribute_visual(visual, pixel, grid)
    assert torch.allclose(out.pooler_output, visual(pixel, grid).pooler_output, atol=1e-6)


def test_grid_for_visual_follows_vision_execution_device():
    """Grid metadata follows pixel values even for SDPA vision towers.

    Current Qwen3.5 uses the grid to construct indices for a device-resident position
    embedding, so attention backend alone cannot make a CPU grid safe.
    """
    grid = _grid([(1, 2, 2)])
    pixel = _pixels(grid)

    assert vs._grid_for_visual(grid, pixel) is grid

    meta_pixel = torch.empty(pixel.shape, device="meta")
    assert vs._grid_for_visual(grid, meta_pixel).device.type == "meta"


@pytest.mark.parametrize(
    "pixel_is_none,grid_is_none",
    [(True, True), (False, True), (True, False)],
)
def test_maybe_distribute_falls_back_when_media_inputs_are_none(monkeypatch, pixel_is_none, grid_is_none):
    """No media inputs (grid_thw and/or pixel_values is None) must route straight to the
    plain visual(...) call -- forwarding both arguments UNCHANGED, with no grid device
    handling (which would dereference the missing input) and no collectives -- even while
    a multi-rank group is active."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)

    def fail_collective(*args, **kwargs):
        raise AssertionError("no-media fallback must not enter collectives")

    monkeypatch.setattr(dist, "all_gather", fail_collective)

    pixel = None if pixel_is_none else _pixels(_grid(_ENTRIES))
    grid = None if grid_is_none else _grid(_ENTRIES)
    visual = _RecorderVisual()
    tok = vs.set_cp_vision_group(object(), config=_policy())
    try:
        out = vs.maybe_distribute_visual(visual, pixel, grid)
    finally:
        vs.reset_cp_vision_group(tok)

    assert len(visual.calls) == 1
    called_pixel, called_grid = visual.calls[0]
    assert called_pixel is pixel
    assert called_grid is grid
    assert out.pooler_output.shape == (1, 4)


def test_trainable_tower_rejects_group_not_declared_cp_only(monkeypatch):
    """A TRAINABLE vision tower published with spans_only_cp=False must raise before any
    collective: the ViT is replicated across TP ranks, so gathering frames over a CP x TP
    group would accumulate the vision gradient tp-fold (silent wrong gradients)."""
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)

    def fail_collective(*args, **kwargs):
        raise AssertionError("trainable-tower scope violation must raise before collectives")

    monkeypatch.setattr(dist, "all_gather", fail_collective)

    grid = _grid(_ENTRIES)
    pixel = _pixels(grid)
    visual = _StubVisual()
    assert any(p.requires_grad for p in visual.parameters())
    tok = vs.set_cp_vision_group(object(), config=_policy(), spans_only_cp=False)
    try:
        with pytest.raises(ValueError, match="spans_only_cp"):
            vs.maybe_distribute_visual(visual, pixel, grid)
    finally:
        vs.reset_cp_vision_group(tok)


def test_frozen_tower_shards_over_group_not_declared_cp_only(monkeypatch):
    """A FROZEN tower may shard across a wider (e.g. CP x TP) group: requires_grad=False
    means no backward ever runs through the gather, and forward parity still holds."""
    torch.manual_seed(4)
    grid = _grid(_ENTRIES)
    pixel = _pixels(grid)
    visual = _StubVisual()
    visual.requires_grad_(False)

    rep = visual(pixel, grid).pooler_output
    out = _simulate_maybe_distribute(visual, pixel, grid, world=2, rank=0, monkeypatch=monkeypatch, spans_only_cp=False)

    assert out.pooler_output.shape == rep.shape
    assert torch.allclose(out.pooler_output, rep, atol=1e-6)


def test_maybe_distribute_disabled_by_config(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 2)
    grid = _grid(_ENTRIES)
    pixel = _pixels(grid)
    visual = _StubVisual()
    tok = vs.set_cp_vision_group(object(), config=_policy(enabled=False))
    try:
        out = vs.maybe_distribute_visual(visual, pixel, grid)
    finally:
        vs.reset_cp_vision_group(tok)
    assert torch.allclose(out.pooler_output, visual(pixel, grid).pooler_output, atol=1e-6)


def test_maybe_distribute_falls_back_below_min_tokens(monkeypatch):
    import torch.distributed as dist

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: 4)

    def fail_if_all_gather(*args, **kwargs):
        raise AssertionError("small visual workload should not enter sharded all_gather")

    monkeypatch.setattr(dist, "all_gather", fail_if_all_gather)

    grid = _grid([(1, 2, 2)])
    pixel = _pixels(grid)
    visual = _StubVisual()
    tok = vs.set_cp_vision_group(object(), config=_policy(min_tokens=999, min_frames=999))
    try:
        out = vs.maybe_distribute_visual(visual, pixel, grid)
    finally:
        vs.reset_cp_vision_group(tok)

    assert torch.allclose(out.pooler_output, visual(pixel, grid).pooler_output, atol=1e-6)


def test_frame_threshold_shards_long_video_below_token_cutoff(monkeypatch):
    grid = _grid([(32, 2, 2)])
    pixel = _pixels(grid, seed=17)
    visual = _StubVisual()
    expected = visual(pixel, grid).pooler_output
    out = _simulate_maybe_distribute(
        visual,
        pixel,
        grid,
        world=8,
        rank=0,
        monkeypatch=monkeypatch,
        policy=_policy(min_tokens=999, min_frames=32),
    )

    assert out.pooler_output.shape == expected.shape
    assert torch.allclose(out.pooler_output, expected, atol=1e-6)


@pytest.mark.parametrize(
    "world,entries",
    [
        (8, [(1, 2, 2), (1, 4, 4)]),  # 2 image entries < world 8
        (4, [(1, 2, 2)]),  # single image < world 4
        (4, [(2, 2, 2)]),  # single 2-frame video (2 units) < world 4
    ],
)
def test_maybe_distribute_pads_when_fewer_entries_than_ranks(monkeypatch, world, entries):
    """Fewer frame-units than ranks: instead of replicating the full ViT on every rank, PAD
    with minimal dummy frames so every rank runs exactly ONE frame.  The gathered embeds are
    sliced back to the real token count and still equal the replicated full forward (dummy
    frames are dropped and contribute zero gradient)."""
    torch.manual_seed(7)
    grid = _grid(entries)
    pixel = _pixels(grid)
    visual = _StubVisual()
    # frame-level partition would fall back here (units < world); the pad path does NOT.
    n_units = int(grid[:, 0].sum())
    assert n_units < world
    rep = visual(pixel, grid).pooler_output
    out = _simulate_maybe_distribute(visual, pixel, grid, world, rank=0, monkeypatch=monkeypatch)
    assert out.pooler_output.shape == rep.shape
    assert torch.allclose(out.pooler_output, rep, atol=1e-6)


@pytest.mark.parametrize(
    "world,entries",
    [
        (4, [(1, 2, 2)]),  # single image, 1 unit < world 4 -> 3 dummy pads
        (4, [(1, 2, 2), (1, 4, 4)]),  # 2 image units < world 4 -> 2 dummy pads
        (4, [(2, 2, 2)]),  # single 2-frame video (2 units) < world 4 -> 2 pads
        (8, [(1, 2, 2), (1, 4, 4)]),  # 2 units < world 8 -> 6 dummy pads
    ],
)
def test_pad_backward_through_real_code_matches_replicate(monkeypatch, world, entries):
    """END-TO-END backward through the PRODUCTION pad path -- maybe_distribute_visual ->
    _all_gather_var_tokens -> _AllGatherSeqDiff.backward (reduce_scatter SUM) -> the
    ``[:n_real_tokens]`` slice -- NOT a hand-rolled re-implementation.

    (``test_pad_dummies_match_replicate_forward_and_grad`` checks the same math but with a
    plain ``torch.cat(blocks)[:n_real_tokens]``, so it never exercises the real gather/slice
    or the reduce_scatter backward; this test closes that gap.)

    All ``world`` CP ranks are simulated in-process; ``grad=True`` makes reduce_scatter hand
    each rank its own grad chunk, so the per-rank vision grads SUM to the full backward.
    Asserts:
      (1) ``sum_r grad_r == replicate grad`` -- the sharded backward is exact; and
      (2) every PADDED (dummy) rank's BIAS grad is exactly zero -- the slice drops the dummy
          tail so zero gradient is routed back to the dummy ViT forward.  The bias grad
          depends only on the upstream grad, so a non-zero value would mean a dummy frame
          leaked gradient into the shared vision params (the exact failure the slice
          prevents)."""
    torch.manual_seed(3)
    grid = _grid(entries)
    pixel = _pixels(grid)
    n_units = int(grid[:, 0].sum())
    assert 0 < n_units < world  # exercise the pad path

    visual = _StubVisual(bias=True)
    rep_out = visual(pixel, grid).pooler_output
    target = torch.randn_like(rep_out)

    visual.zero_grad(set_to_none=True)
    (visual(pixel, grid).pooler_output * target).sum().backward()
    gw_rep = visual.proj.weight.grad.clone()
    gb_rep = visual.proj.bias.grad.clone()

    gws, gbs = [], []
    for r in range(world):
        visual.zero_grad(set_to_none=True)
        out = _simulate_maybe_distribute(visual, pixel, grid, world, rank=r, monkeypatch=monkeypatch, grad=True)
        assert out.pooler_output.shape == rep_out.shape
        assert torch.allclose(out.pooler_output, rep_out, atol=1e-6)  # forward still exact
        (out.pooler_output * target).sum().backward()
        gws.append(visual.proj.weight.grad.clone())
        gbs.append(visual.proj.bias.grad.clone())

    # (2) padded ranks (r >= n_units) receive zero upstream grad -> exactly zero bias grad
    for r in range(n_units, world):
        assert torch.count_nonzero(gbs[r]) == 0, f"dummy rank {r} leaked gradient (slice failed)"

    # (1) the sharded backward, summed over ranks, reconstructs the replicate backward
    assert torch.allclose(torch.stack(gws).sum(0), gw_rep, atol=1e-6), "sharded weight grad != replicate"
    assert torch.allclose(torch.stack(gbs).sum(0), gb_rep, atol=1e-6), "sharded bias grad != replicate"
