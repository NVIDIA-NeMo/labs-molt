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

"""Frame-level context-parallel vision-tower sharding.

Under context parallelism (CP), a VLM must encode media and splice the resulting
features before it shards the language sequence. Without vision frame sharding,
``self.visual(...)`` runs the ENTIRE vision tower on the full, un-sharded set of
images on EVERY CP rank. With ``cp_size=N`` that is N x redundant compute and
O(all-images) vision activations per rank.

Qwen3-VL-style vision towers attend per IMAGE/FRAME ("entry"): the forward builds
``cu_seqlens`` from ``grid_thw`` so each entry only attends within itself.  Entries
are therefore INDEPENDENT units, and

    visual(all_entries) == concat_r visual(entries_owned_by_rank_r)    (entry order)

holds up to numerical precision (``allclose``; all CP ranks already hold identical
vision weights after the FSDP2 all-gather -- the replicated design relies on that
too).  So we partition the entries across the CP group, run ``self.visual`` on each
rank's slice, and all-gather the per-entry embeddings back to the full set.  The
downstream scatter into ``inputs_embeds`` and the sequence shard consume numerically
equivalent embeds through the unchanged code path.

Memory/compute: vision forward + activations drop ~cp_size x; the final gathered
embeds (small vs the ViT's internal patch activations) are reconstructed in full
on every rank, then immediately sequence-sharded by the existing CP sharder.

Sharding-group scope (deliberate, machine-checked design constraint): the caller
declares the scope of the process group it publishes via
``set_cp_vision_group(group, config=..., spans_only_cp=...)``. With a FROZEN vision tower,
frames may be sharded across the full CP x TP rank set (``spans_only_cp=False``):
the forward is numerically equivalent (``allclose``; per-frame independence,
replicated weights) and
``requires_grad=False`` means no backward ever runs through the gather.  With a
TRAINABLE vision tower, only a CP-only group (``spans_only_cp=True``, the default)
is valid: the vision tower is replicated (not tensor-parallel) across TP ranks,
so gathering frames across CP x TP would make the all-gather's
reduce-scatter(SUM) backward accumulate the vision gradient tp-fold (each
TP-replicated sequence shard backprops into the single compute rank).
:func:`maybe_distribute_visual` raises when a trainable tower meets a group not
declared CP-only.  Under pure TP (``cp_size == 1``) the published group has size
1, so this sharding is not enabled.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.distributed as dist

__all__ = [
    "CpVisionFrameShardingConfig",
    "cp_vision_frame_sharding_active",
    "maybe_distribute_visual",
    "reset_cp_vision_group",
    "set_cp_vision_group",
]

logger = logging.getLogger(__name__)
_LOGGED_ONCE = False
_LOGGED_SMALL_FALLBACK = False
_LOGGED_COST_ALPHAS: set[tuple[str, int, str]] = set()


@dataclass(frozen=True)
class CpVisionFrameShardingConfig:
    """Declarative policy for sharding VLM vision work across CP ranks.

    Args:
        enabled: Enable vision frame sharding when a multi-rank CP group is published. Disabled
            by default so existing CP recipes retain their replicated vision behavior.
        mesh_dims: Device-mesh dimensions across which frames are distributed. Only
            ``("cp",)`` is currently supported.
        min_tokens: Minimum number of merged visual tokens required to use the sharded
            path unless ``min_frames`` is reached. Smaller workloads stay replicated
            to avoid collective overhead.
        min_frames: Minimum number of independent image/frame units that can select
            sharding even below ``min_tokens``. Long videos reduce per-frame spatial
            resolution, so merged-token count alone underestimates their ViT work.
        min_local_patch_rows: Optional minimum number of pre-merge patch rows processed
            by each rank that owns real frames. Smaller local shards append whole,
            same-shaped zero frames before the vision forward and remove their outputs
            before the gather. This can reduce shape-dependent numerical variation from
            tiny reduced-precision ViT kernels. ``0`` disables local padding.
        cost_alpha: Non-negative linear term in the partition cost
            ``p * (p + cost_alpha)``. ``"auto"`` infers ``3 * vision_hidden_size``
            and falls back to ``0`` when the width is unavailable. ``None`` remains
            accepted as a backward-compatible alias for ``"auto"``.
    """

    enabled: bool = False
    mesh_dims: tuple[Literal["cp"]] = ("cp",)
    min_tokens: int = 2048
    min_frames: int = 32
    min_local_patch_rows: int = 96
    cost_alpha: int | Literal["auto"] | None = "auto"

    def __post_init__(self) -> None:
        """Validate the serialized policy fields."""
        config_path = "distributed.multimodal.vision.frame_sharding"
        if not isinstance(self.enabled, bool):
            raise TypeError(f"{config_path}.enabled must be a boolean")
        if not isinstance(self.mesh_dims, tuple):
            raise TypeError(f"{config_path}.mesh_dims must be a list of mesh-dimension names")
        if self.mesh_dims != ("cp",):
            raise ValueError(f'{config_path}.mesh_dims currently supports only ["cp"]')
        if isinstance(self.min_tokens, bool) or not isinstance(self.min_tokens, int) or self.min_tokens < 0:
            raise ValueError(f"{config_path}.min_tokens must be a non-negative integer")
        if isinstance(self.min_frames, bool) or not isinstance(self.min_frames, int) or self.min_frames < 0:
            raise ValueError(f"{config_path}.min_frames must be a non-negative integer")
        if (
            isinstance(self.min_local_patch_rows, bool)
            or not isinstance(self.min_local_patch_rows, int)
            or self.min_local_patch_rows < 0
        ):
            raise ValueError(f"{config_path}.min_local_patch_rows must be a non-negative integer")
        if self.cost_alpha not in (None, "auto") and (
            isinstance(self.cost_alpha, bool) or not isinstance(self.cost_alpha, int) or self.cost_alpha < 0
        ):
            raise ValueError(f'{config_path}.cost_alpha must be "auto", null, or a non-negative integer')


@dataclass(frozen=True)
class _GroupScope:
    """The published sharding group plus the caller's declaration of its scope.

    ``spans_only_cp=False`` marks a group that also spans replicated non-CP axes
    (e.g. the flattened CP x TP set); that is only safe for a frozen vision tower.
    """

    group: dist.ProcessGroup
    config: CpVisionFrameShardingConfig
    spans_only_cp: bool


# Token restoring the previous published scope; opaque to callers.
CpVisionGroupToken = _GroupScope | None


class _GroupHolder:
    """Plain get/set/reset holder for the sharding-group scope active during the
    VLM forward. Set by the VLM CP recipe around the model forward and read
    synchronously by ``maybe_distribute_visual`` before the vision tower runs.

    A plain attribute (not a ``ContextVar``) is sufficient: activation-checkpoint
    recompute re-enters vision submodules after frame selection, not this helper,
    so there is no cross-thread recompute read. Training steps are sequential,
    and token-based restore handles nested callers.
    """

    def __init__(self):
        self._value: _GroupScope | None = None

    def get(self) -> _GroupScope | None:
        return self._value

    def set(self, value: _GroupScope | None) -> _GroupScope | None:
        token = self._value
        self._value = value
        return token

    def reset(self, token: _GroupScope | None) -> None:
        self._value = token


# Holds the published sharding-group scope (or ``None``) for the current model forward.
_CP_VISION_GROUP = _GroupHolder()


class _AllGatherSeqDiff(torch.autograd.Function):
    """Differentiable all-gather of equal-size per-rank shards along ``seq_dim``.

    Forward concatenates every rank's shard in rank order, producing the full
    sequence.  Backward reduce-scatters the incoming gradient: each position may
    be read on any rank, so its gradient is the sum over ranks of the per-rank
    grad slice; reduce-scatter(SUM) hands each rank the summed gradient for the
    shard it owns.  All shards must have equal length along ``seq_dim``.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, seq_dim: int) -> torch.Tensor:
        ctx.group = group
        ctx.seq_dim = seq_dim
        world = dist.get_world_size(group)
        ctx.world = world
        x = x.contiguous()
        gathered = [torch.empty_like(x) for _ in range(world)]
        dist.all_gather(gathered, x, group=group)
        return torch.cat(gathered, dim=seq_dim)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        chunks = [c.contiguous() for c in grad_out.chunk(ctx.world, dim=ctx.seq_dim)]
        local = torch.empty_like(chunks[0])
        dist.reduce_scatter(local, chunks, op=dist.ReduceOp.SUM, group=ctx.group)
        return local, None, None


def _grid_for_visual(grid_thw: torch.Tensor, pixel_values: torch.Tensor) -> torch.Tensor:
    """Move ``grid_thw`` ([N, 3] long, rows of (t, h, w)) to the device ``visual`` expects.

    Vision forwards may use the grid both as host-readable shape metadata and as indices
    into device-resident position embeddings. Keep it on ``pixel_values``'s device; callers
    that need Python values can synchronize through ``tolist()``, while a CPU grid cannot
    index a CUDA embedding table. Attention backend alone does not determine every grid
    consumer's device requirement.

    Args:
        grid_thw: Tensor of shape [entries, 3] containing ``(time, height, width)`` rows.
        pixel_values: Tensor of shape [patches, patch_dim] whose device owns vision execution.

    Returns:
        Tensor of shape [entries, 3] on ``pixel_values``'s device. The input is returned
        without copying when it is already on that device.
    """
    if grid_thw.device == pixel_values.device:
        return grid_thw
    return grid_thw.to(pixel_values.device, non_blocking=True)


def _grid_list_for_planning(grid_thw: torch.Tensor) -> tuple[list[list[int]], torch.Tensor]:
    """Return host-side grid metadata for Python partitioning.

    The generic VLM recipe moves batch tensors, including grid metadata, to the model
    device. Make the required host copy explicit instead of hiding the synchronization
    behind ``tolist()``. Callers that already keep the grid on CPU avoid the copy.

    Args:
        grid_thw: [N, 3] tensor, one (t, h, w) row per media entry (N = entries).

    Returns:
        Tuple of ``grid_thw.tolist()`` (a list of ``[t, h, w]`` ints) and the CPU long
        tensor it was read from.
    """
    grid_host = grid_thw
    if grid_host.device.type != "cpu":
        # Blocking D2H copy: `.tolist()` below reads host memory immediately.
        grid_host = grid_host.to("cpu")
    if grid_host.dtype != torch.long:
        grid_host = grid_host.to(torch.long)
    return grid_host.tolist(), grid_host


def set_cp_vision_group(
    group: dist.ProcessGroup | None,
    *,
    config: CpVisionFrameShardingConfig,
    spans_only_cp: bool = True,
) -> CpVisionGroupToken:
    """Install the sharding process group for the current model forward.

    Args:
        group: Process group to shard vision frames across, or ``None`` to disable
            sharding for the call.
        config: Declarative sharding policy resolved from the recipe configuration.
        spans_only_cp: Declaration of the group's scope.  ``True`` (default) states
            that ``group`` spans context-parallel ranks only -- always
            gradient-correct.  Pass ``False`` only for a group that also spans
            replicated non-CP axes (e.g. the flattened CP x TP rank set); that is
            valid solely for a FROZEN vision tower, and
            :func:`maybe_distribute_visual` raises when a trainable tower meets a
            group not declared CP-only (the gather's reduce-scatter(SUM) backward
            would otherwise accumulate the vision gradient tp-fold).

    Returns:
        A token to pass to :func:`reset_cp_vision_group`.
    """
    scope = None if group is None else _GroupScope(group=group, config=config, spans_only_cp=spans_only_cp)
    return _CP_VISION_GROUP.set(scope)


def reset_cp_vision_group(token: CpVisionGroupToken) -> None:
    """Restore the previous sharding-group scope (pair with :func:`set_cp_vision_group`)."""
    _CP_VISION_GROUP.reset(token)


def cp_vision_frame_sharding_active() -> bool:
    """Return whether the current model forward has an active CP shard group.

    This intentionally shares :func:`maybe_distribute_visual`'s typed policy.
    Model-specific multimodal implementations can use it to leave their ordinary
    (replicated / CP-off) multimodal forward completely untouched.
    """
    scope = _CP_VISION_GROUP.get()
    if scope is None or not scope.config.enabled:
        return False
    return dist.get_world_size(scope.group) > 1


def _check_group_scope_for_trainable(visual: torch.nn.Module, scope: _GroupScope) -> None:
    """Raise when a trainable vision tower is sharded over a group not declared CP-only.

    The ViT is replicated (not tensor-parallel) across TP ranks, so gathering frames
    across a CP x TP group makes the all-gather's reduce-scatter(SUM) backward
    accumulate the vision gradient tp-fold.  This check is deterministic across ranks
    (``requires_grad`` and the declaration are identical everywhere), so it raises on
    every rank before any collective.
    """
    if scope.spans_only_cp:
        return
    if any(p.requires_grad for p in visual.parameters()):
        raise ValueError(
            "cp_vision_frame_shard: the vision tower has trainable parameters but the published "
            "sharding group was declared spans_only_cp=False (e.g. the flattened CP x TP rank "
            "set). The vision tower is replicated across TP ranks, so gathering frames over "
            "that group would accumulate the vision gradient tp-fold in the all-gather's "
            "reduce-scatter(SUM) backward. Publish the CP-only group "
            "(set_cp_vision_group(cp_group, config=...)) or freeze the vision tower."
        )


def _config_value(source: object, name: str) -> object | None:
    if isinstance(source, Mapping):
        return source.get(name)
    return getattr(source, name, None)


def _infer_vision_hidden_size(source: object | None) -> int | None:
    """Best-effort vision-width discovery without depending on model class names.

    Supported VLMs expose the width in different places: Qwen vision towers use
    ``visual.config.hidden_size``, while Nemotron Omni's multimodal config uses
    ``vit_hidden_size`` and RADIO-style towers may expose ``embed_dim``.  Prefer
    explicitly vision-named fields and nested vision configs over a root
    ``hidden_size`` so a text-model width is not accidentally selected.

    Args:
        source: A model, module, or config object (or a ``Mapping``) that may
            carry the vision width somewhere in its attributes/children.

    Returns:
        The discovered positive width, or ``None`` when no vision width is found.
    """
    visited: set[int] = set()

    def _positive_int(value: object | None) -> int | None:
        if isinstance(value, bool) or value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _walk(current: object | None, depth: int) -> int | None:
        if current is None or depth > 6:
            return None
        direct = _positive_int(current)
        if direct is not None:
            return direct
        identity = id(current)
        if identity in visited:
            return None
        visited.add(identity)

        for name in ("vit_hidden_size", "vision_hidden_size", "embed_dim"):
            found = _positive_int(_config_value(current, name))
            if found is not None:
                return found

        # Search vision-specific children before accepting a generic hidden size.
        for name in (
            "vision_config",
            "visual",
            "vision_model",
            "radio_model",
            "patch_generator",
            "config",
            "module",
            "_orig_mod",
            "_fsdp_wrapped_module",
            "model",
        ):
            child = _config_value(current, name)
            if child is current:
                continue
            found = _walk(child, depth + 1)
            if found is not None:
                return found

        for name in ("hidden_size", "d_model"):
            found = _positive_int(_config_value(current, name))
            if found is not None:
                return found
        return None

    return _walk(source, 0)


def _vision_cost_alpha(
    source: object | None = None,
    config: CpVisionFrameShardingConfig | None = None,
) -> int:
    """Resolve the linear term in the vision partition cost ``p*(p+alpha)``.

    A configured non-negative integer is an exact override (``0`` selects the legacy
    pure-quadratic model). With ``cost_alpha="auto"`` or the backward-compatible
    ``None`` alias, use ``3 * vision_hidden_size``: the three Q/K/V projections are a
    portable proxy for linear per-patch ViT work. Unknown architectures safely retain
    the legacy ``alpha=0`` behavior.

    Args:
        source: Object to infer the vision width from when in ``auto`` mode.
        config: Optional typed sharding policy containing an exact override.

    Returns:
        The resolved non-negative ``alpha``.
    """
    mode = "auto"
    configured_alpha = config.cost_alpha if config is not None else "auto"
    alpha = configured_alpha if isinstance(configured_alpha, int) else None
    if alpha is not None:
        mode = "configured"

    hidden_size = _infer_vision_hidden_size(source)
    if alpha is None:
        alpha = 3 * hidden_size if hidden_size is not None else 0
        mode = "auto" if hidden_size is not None else "legacy-fallback"

    if source is not None:
        model_name = type(source).__name__
        log_key = (model_name, alpha, mode)
        if log_key not in _LOGGED_COST_ALPHAS:
            _LOGGED_COST_ALPHAS.add(log_key)
            if mode == "auto":
                detail = f"auto: 3 x vision hidden_size {hidden_size}"
            elif mode == "configured":
                detail = "configured override"
            else:
                detail = "vision width unavailable; legacy fallback"
            logger.info(
                "vision partition cost alpha=%d (%s) for %s",
                alpha,
                detail,
                model_name,
            )
    return alpha


def _contiguous_balanced_bounds(
    patches: torch.Tensor,
    world: int,
    *,
    cost_alpha_source: object | None = None,
    config: CpVisionFrameShardingConfig | None = None,
) -> list[int] | None:
    """Partition ``len(patches)`` entries into ``world`` CONTIGUOUS groups balanced by
    approximate vision-attention cost, with >=1 entry per group.

    ``patches[i]`` is entry ``i``'s pixel-row count (= ``grid_thw[i].prod()``), a proxy
    for one frame-unit's attention sequence length.  Since the ViT attends within each
    frame/window, the hot path scales closer to ``patches[i] ** 2`` than to patch count.
    Returns cut points ``cuts`` of length ``world+1`` where rank ``r`` owns entries
    ``[cuts[r], cuts[r+1])``; returns ``None`` when ``num_entries < world``
    (caller falls back to the replicate path so every rank still runs the ViT once and
    FSDP collectives stay uniform).

    Contiguous (not round-robin) so the gathered per-rank blocks concatenate back in the
    original entry order with no reshuffle.  Deterministic: ``patches`` is identical on
    every rank, so all ranks compute the same partition (and thus the same per-rank token
    counts they need to unpack the all-gather).

    Cost model: ``p*(p + alpha)``. A configured ``cost_alpha`` wins; otherwise
    ``alpha`` is inferred as ``3 * vision_hidden_size`` from ``cost_alpha_source``.
    The quadratic term is per-frame attention; ``alpha*p`` adds
    the LINEAR per-patch work (qkv/MLP projections) that the pure quadratic ignores --
    without it, packs mixing big image frames with many small video frames heap the small
    frames onto few ranks (attention-cost-"balanced" but frame-count- and wall-clock-
    imbalanced).  Unknown architectures fall back to ``alpha=0``.  Partition-only choice:
    forward/grad are identical for any cuts.
    """
    n = int(patches.shape[0])
    if n < world:
        return None
    if world <= 1:
        return [0, n]
    # Pure-Python on a CPU list: ``patches`` may live on GPU and mixing a GPU cumsum with a
    # CPU search tensor raises a device-mismatch error.  The entry count is tiny, so a single
    # host sync + Python bisect is both cheap and device-agnostic.
    import bisect

    alpha = _vision_cost_alpha(cost_alpha_source, config)
    costs = [int(x) * (int(x) + alpha) for x in patches.tolist()]
    if any(value <= 0 for value in costs):
        raise ValueError("partition costs must be positive")
    cum = []
    acc = 0
    for c in costs:
        acc += c
        cum.append(acc)
    total = acc
    cuts = [0]
    for r in range(1, world):
        target = total * r / world
        # ``bisect_right`` returns the first index whose cumulative sum EXCEEDS ``target``.
        # ``bisect_left`` would stop one short on an exact cumulative-cost boundary (e.g. 4
        # equal-cost frames at world=2 would split [1, 3] instead of [2, 2]), so use
        # ``bisect_right`` to place the cut past the frame that lands exactly on the target.
        j = bisect.bisect_right(cum, target)
        lo = cuts[-1] + 1  # leave >=1 entry for the rank we just closed
        hi = n - (world - r)  # leave >=1 entry for each of ranks r..world-1
        j = max(lo, min(j, hi))
        cuts.append(j)
    cuts.append(n)
    return cuts


def _all_gather_var_tokens(
    local: torch.Tensor, group: dist.ProcessGroup, world: int, token_counts: list[int]
) -> torch.Tensor:
    """Differentiable all-gather of per-rank ``[n_r, H]`` token blocks into the full
    ``[sum(n_r), H]`` tensor in rank order.

    Each rank holds a different token count, so we pad to the global max, all-gather
    equal ``[max_tokens, H]`` shards (via ``_AllGatherSeqDiff``: forward all-gather +
    cat, backward reduce-scatter SUM), then slice each rank's block back to its true
    ``token_counts[r]`` and concat.  Backward = SUM is correct: every rank reassembles
    the full embeds and keeps only its sequence shard, so an entry's gradient is
    produced on the shard-owner rank and reduce-scatter routes it back to the
    compute-rank (summing the straddling case); padding rows are never used downstream
    so they contribute zero.
    """
    max_tokens = max(token_counts)
    n_local, h = local.shape[0], local.shape[-1]
    if n_local < max_tokens:
        pad = local.new_zeros(max_tokens - n_local, h)
        local_padded = torch.cat([local, pad], dim=0)
    else:
        local_padded = local
    gathered = _AllGatherSeqDiff.apply(local_padded, group, 0)  # [world*max_tokens, H]
    blocks = [gathered[r * max_tokens : r * max_tokens + token_counts[r]] for r in range(world)]
    return torch.cat(blocks, dim=0)


def _padding_frames_for_min_patch_rows(
    local_patch_rows: int,
    frame_patch_rows: int,
    min_patch_rows: int,
) -> int:
    """Return the whole same-shaped frames needed to reach a local patch-row floor."""
    if min_patch_rows <= local_patch_rows or frame_patch_rows <= 0:
        return 0
    missing_rows = min_patch_rows - local_patch_rows
    return (missing_rows + frame_patch_rows - 1) // frame_patch_rows


def _raise_if_any_rank_failed(
    local_ok: bool,
    group: dist.ProcessGroup,
    device: torch.device,
    local_detail: str,
) -> None:
    """Reach group consensus before raising so a per-rank validation failure aborts on EVERY
    rank instead of deadlocking peers on the next collective.

    A per-rank ``raise`` placed BEFORE a collective hangs the group: the failing rank raises
    while its peers block forever in ``all_gather``.  Instead, each rank contributes ``0``
    (ok) or ``1`` (failed) and an all-reduce(MAX) over ``group`` makes the outcome identical
    everywhere.  When any rank failed, the offending rank(s) raise their own ``local_detail``
    and the rest raise a group-level message, so all ranks raise and none is left blocking.

    Args:
        local_ok: Whether THIS rank's local validation passed.
        group: The sharding process group shared by every rank.
        device: Device for the 1-element flag tensor (the compute device, so the all-reduce
            matches the active backend, e.g. CUDA under NCCL).
        local_detail: Actionable message raised by a rank that itself failed.
    """
    flag = torch.tensor([0 if local_ok else 1], dtype=torch.int64, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    if int(flag.item()) == 0:
        return
    if not local_ok:
        raise ValueError(local_detail)
    raise ValueError(
        "cp_vision_frame_shard: another rank in the sharding group produced a visual output that did "
        "not match its planned token count, so the collective was aborted on every rank to "
        "avoid a hang. Check the diverging rank's log for the mismatch detail."
    )


def maybe_distribute_visual(
    visual: torch.nn.Module,
    pixel_values: torch.Tensor | None,
    grid_thw: torch.Tensor | None,
) -> Any:
    """Run ``visual(pixel_values, grid_thw=grid_thw, return_dict=True)`` but distribute the
    forward across the CP group when enabled, returning an output object whose
    ``pooler_output`` (and ``deepstack_features`` list, if any) are the FULL gathered
    embeds in original entry order -- a drop-in for the direct ``visual(...)`` call.

    Falls back to the plain replicated call (exact pre-sharding behaviour) when sharding
    is disabled, no CP group is active, ``cp_size <= 1``, there are no media inputs, or
    the visual workload is below the minimum sharding size.

    Args:
        visual: The vision tower; must expose ``spatial_merge_size`` and accept
            ``visual(pixel_values, grid_thw=..., return_dict=True)``.
        pixel_values: ``[total_patch_rows, patch_dim]`` pixel rows for ALL entries,
            frame-contiguous in entry order (entry order, then frame order within each
            entry) -- the exact tensor the replicated call would receive.  ``None``
            (no media in the batch) is forwarded to ``visual`` unchanged.
        grid_thw: ``[N, 3]`` tensor, one (t, h, w) row per media entry (N = entries;
            ``t * h * w`` patch rows per entry).  Expected on CPU.  ``None`` (no media
            in the batch) is forwarded to ``visual`` unchanged.

    Returns:
        The vision tower's output object.  ``pooler_output`` is the full gathered
        ``[total_patch_rows / spatial_merge_size**2, H]`` merged-token tensor in original
        entry order (H = vision output hidden size); each entry of ``deepstack_features``
        (when present) is gathered to the same shape.  ``last_hidden_state`` is left as
        the local shard (unused downstream).  Forward and vision-parameter gradients are
        numerically equivalent (``allclose``) to the replicated call's.

    Raises:
        ValueError: When the vision tower has trainable parameters but the published
            group was declared ``spans_only_cp=False`` (see :func:`set_cp_vision_group`),
            or when any rank's local visual output does not match its planned token count.
            The token-count mismatch is reduced across the sharding group before raising, so
            a single diverging rank makes EVERY rank raise (rather than deadlocking peers on
            the gather).
    """
    if grid_thw is None or pixel_values is None:
        # No media inputs: nothing to shard and no grid to place; keep the exact
        # replicated call (grid/device handling would dereference the missing input).
        return visual(pixel_values, grid_thw=grid_thw, return_dict=True)

    scope = _CP_VISION_GROUP.get()
    if scope is None or not scope.config.enabled:
        return visual(
            pixel_values,
            grid_thw=_grid_for_visual(grid_thw, pixel_values),
            return_dict=True,
        )

    group = scope.group
    world = dist.get_world_size(group)
    if world <= 1:
        return visual(
            pixel_values,
            grid_thw=_grid_for_visual(grid_thw, pixel_values),
            return_dict=True,
        )

    _check_group_scope_for_trainable(visual, scope)

    sms_sq = int(visual.spatial_merge_size) ** 2

    # Expand each (t, h, w) entry into t per-FRAME units.  Qwen3-VL-style vision towers have NO
    # cross-frame coupling -- attention is per-frame (cu_seqlens repeats h*w over t), and both
    # the rotary and the learned position embeddings are spatial-only, repeated identically per
    # frame -- so visual((t,h,w)) == visual(t x (1,h,w)) up to numerical precision.  Sharding at frame granularity
    # lets a single large video (or a pack with fewer videos than ranks) split across ranks;
    # images (t == 1) are simply one unit each, so their behaviour is unchanged.
    grid_list, grid_host = _grid_list_for_planning(grid_thw)
    f_patches: list[int] = []  # h*w pixel rows per frame unit
    f_entry: list[int] = []  # original entry index per frame unit (for same-entry coalescing)
    f_hw: list[tuple[int, int]] = []
    for e, (t, h, w) in enumerate(grid_list):
        for _ in range(int(t)):
            f_patches.append(int(h) * int(w))
            f_entry.append(e)
            f_hw.append((int(h), int(w)))
    n_units = len(f_patches)
    n_real_tokens = sum(p // sms_sq for p in f_patches)
    min_shard_tokens = scope.config.min_tokens
    min_shard_frames = scope.config.min_frames
    if n_real_tokens < min_shard_tokens and n_units < min_shard_frames:
        global _LOGGED_SMALL_FALLBACK
        if not _LOGGED_SMALL_FALLBACK:
            _LOGGED_SMALL_FALLBACK = True
            logger.info(
                "vision tower using replicated path for small visual workload "
                "(tokens=%d < distributed.multimodal.vision.frame_sharding.min_tokens=%d and "
                "frames=%d < distributed.multimodal.vision.frame_sharding.min_frames=%d)",
                n_real_tokens,
                min_shard_tokens,
                n_units,
                min_shard_frames,
            )
        return visual(
            pixel_values,
            grid_thw=_grid_for_visual(grid_thw, pixel_values),
            return_dict=True,
        )

    # Fewer frame units than ranks: instead of replicating the full ViT on every rank
    # (each rank redundantly runs all `n_units` frames), PAD with dummy frames up to
    # `world` so every rank runs exactly ONE frame.  The dummy embeddings are gathered
    # then sliced off below (never reach masked_scatter / the loss), so they contribute
    # zero gradient and the result is numerically equivalent (`allclose`) to the replicate
    # path -- but per-rank ViT work drops from `n_units` frames to 1.  (`n_units == 0` -> no frame to copy ->
    # keep the replicate path.)
    n_pad = 0
    if 0 < n_units < world:
        n_pad = world - n_units
        # Dummy = a MINIMAL (1, sms, sms) zero frame: sms*sms patch rows -> exactly 1 merged
        # token, the smallest valid ViT input.  It is gathered then sliced off (never reaches
        # masked_scatter / the loss), so it carries zero gradient AND minimises the wasted ViT
        # compute on the padded ranks.  Each dummy is its own synthetic entry so it never
        # coalesces with a real frame.
        sms = int(visual.spatial_merge_size)
        d_rows = sms * sms  # == sms_sq; -> 1 merged token
        pixel_values = torch.cat([pixel_values, pixel_values.new_zeros(n_pad * d_rows, pixel_values.shape[1])], dim=0)
        base_entry = len(grid_list)
        for k in range(n_pad):
            f_patches.append(d_rows)
            f_entry.append(base_entry + k)  # unique -> no coalescing with reals or each other
            f_hw.append((sms, sms))
        cuts = list(range(world + 1))  # exactly one frame unit per rank
    else:
        cuts = _contiguous_balanced_bounds(
            torch.tensor(f_patches, dtype=torch.long),
            world,
            cost_alpha_source=visual,
            config=scope.config,
        )
        if cuts is None:  # n_units == 0 (no frames) -> replicate (keeps collectives uniform)
            return visual(
                pixel_values,
                grid_thw=_grid_for_visual(grid_thw, pixel_values),
                return_dict=True,
            )

    # Tokens belonging to REAL frames (frames 0..n_units-1, which land on ranks 0..n_units-1
    # in rank order).  We slice the gathered embeds to this many rows, dropping the dummy
    # tail.  In the non-padded path this equals the full token count, so the slice is a
    # no-op there.
    n_real_tokens = sum(f_patches[i] // sms_sq for i in range(n_units))

    rank = dist.get_rank(group)
    token_counts = [sum(f_patches[i] // sms_sq for i in range(cuts[r], cuts[r + 1])) for r in range(world)]

    global _LOGGED_ONCE
    if not _LOGGED_ONCE:
        _LOGGED_ONCE = True
        frames_per_rank = [cuts[r + 1] - cuts[r] for r in range(world)]
        logger.info(
            "vision tower SHARDED across vision-frame sharding group (world=%d): %d entries / %d "
            "frame-units (+%d dummy pad) -> per-rank frames=%s tokens=%s (was: full ViT replicated "
            "on every rank)",
            world,
            len(grid_list),
            n_units,
            n_pad,
            frames_per_rank,
            token_counts,
        )

    # pixel rows are frame-contiguous (entry order, then frame order within each entry).
    pix_bounds = [0]
    for p in f_patches:
        pix_bounds.append(pix_bounds[-1] + p)
    p0, p1 = pix_bounds[cuts[rank]], pix_bounds[cuts[rank + 1]]
    local_pixel = pixel_values[p0:p1]

    # Build this rank's grid by coalescing consecutive frame units from the SAME original entry
    # into a (count, h, w) row (visual treats it as `count` independent frames).  Frames of one
    # video that land on this rank collapse to one row; distinct entries (incl. every image)
    # stay separate, so the t==1 image path is numerically equivalent to the entry-level version.
    local_rows: list[list[int]] = []  # [count, h, w, entry_idx]
    for i in range(cuts[rank], cuts[rank + 1]):
        h, w = f_hw[i]
        if local_rows and local_rows[-1][3] == f_entry[i] and local_rows[-1][1:3] == [h, w]:
            local_rows[-1][0] += 1
        else:
            local_rows.append([1, h, w, f_entry[i]])

    # Very small reduced-precision ViT shards can select materially different GEMM
    # kernels. Keep ranks that own real frames above a resolution-aware patch-row floor
    # by appending whole, same-shaped zero frames. Ranks that own only a global rank-fill
    # dummy stay minimal. The synthetic outputs are validated and sliced below before
    # they enter the differentiable gather, so they cannot reach the loss or vision grads.
    numeric_pad_frames = 0
    numeric_pad_patch_rows = 0
    local_real_start = min(cuts[rank], n_units)
    local_real_end = min(cuts[rank + 1], n_units)
    if scope.config.min_local_patch_rows > 0 and local_real_end > local_real_start:
        pad_h, pad_w = f_hw[local_real_end - 1]
        frame_patch_rows = pad_h * pad_w
        numeric_pad_frames = _padding_frames_for_min_patch_rows(
            int(local_pixel.shape[0]),
            frame_patch_rows,
            scope.config.min_local_patch_rows,
        )
        if numeric_pad_frames:
            numeric_pad_patch_rows = numeric_pad_frames * frame_patch_rows
            local_pixel = torch.cat(
                [
                    local_pixel,
                    local_pixel.new_zeros(numeric_pad_patch_rows, local_pixel.shape[1]),
                ],
                dim=0,
            )
            local_rows.append([numeric_pad_frames, pad_h, pad_w, -1])
    local_grid = torch.tensor(
        [[c, h, w] for c, h, w, _ in local_rows],
        dtype=grid_host.dtype,
        device=pixel_values.device,
    )

    local_out = visual(local_pixel, grid_thw=local_grid, return_dict=True)

    # Validate the planned per-rank token count before the collective: a mismatch would
    # otherwise silently mis-slice every rank's block out of the gathered tensor.  Route the
    # check through a group consensus (all-reduce of a boolean flag) so a single diverging
    # rank makes EVERY rank raise -- a bare per-rank ``raise`` here would hang the group,
    # since the failing rank would raise while peers block in the all-gather below.
    expected_local_tokens = token_counts[rank]
    numeric_pad_tokens = numeric_pad_patch_rows // sms_sq
    expected_visual_tokens = expected_local_tokens + numeric_pad_tokens
    actual_local_tokens = local_out.pooler_output.shape[0]
    _raise_if_any_rank_failed(
        actual_local_tokens == expected_visual_tokens,
        group,
        local_out.pooler_output.device,
        f"cp_vision_frame_shard: rank {rank} produced {actual_local_tokens} visual tokens for its "
        f"frame slice, expected {expected_visual_tokens} including {numeric_pad_tokens} local "
        "padding tokens.",
    )

    # Local numerical padding is not part of the gather contract. Removing it here gives
    # every synthetic output zero upstream gradient and keeps token_counts unchanged.
    local_out.pooler_output = local_out.pooler_output[:expected_local_tokens]

    # Gather all per-rank blocks (real + any dummy) in rank order, then slice to the real
    # token count: dummy frames live on the last `n_pad` ranks, so their tokens are the
    # tail and ``[:n_real_tokens]`` drops them.  In the non-padded path n_real_tokens is the
    # full count, so the slice is a no-op.  Slicing keeps backward exact: the dropped dummy
    # rows get zero gradient -> the differentiable all-gather's reduce-scatter routes zero
    # back to the padded ranks, so dummy ViT forwards contribute nothing to the vision grads.
    local_out.pooler_output = _all_gather_var_tokens(local_out.pooler_output, group, world, token_counts)[
        :n_real_tokens
    ]
    deepstack = getattr(local_out, "deepstack_features", None)
    if deepstack is not None:
        # Same consensus contract as the pooler check above: a per-rank ``raise`` before the
        # deepstack all-gathers would deadlock, so agree across the group first.
        bad_k = next((k for k, d in enumerate(deepstack) if d.shape[0] != expected_visual_tokens), None)
        _raise_if_any_rank_failed(
            bad_k is None,
            group,
            local_out.pooler_output.device,
            ""
            if bad_k is None
            else (
                f"cp_vision_frame_shard: rank {rank} deepstack feature {bad_k} has "
                f"{deepstack[bad_k].shape[0]} visual tokens, expected {expected_visual_tokens} "
                f"including {numeric_pad_tokens} local padding tokens."
            ),
        )
        deepstack = [d[:expected_local_tokens] for d in deepstack]
        local_out.deepstack_features = [
            _all_gather_var_tokens(d, group, world, token_counts)[:n_real_tokens] for d in deepstack
        ]
    # ``last_hidden_state`` (the local shard, unused downstream) is intentionally left
    # un-gathered; embed_multimodal reads only ``pooler_output`` / ``deepstack_features``.
    return local_out
