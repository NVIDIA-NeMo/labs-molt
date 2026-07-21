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

import itertools
from dataclasses import dataclass, field, fields
from typing import Any, List, Union

import ray
import torch

from molt.utils.logging_utils import init_logger
from molt.utils.seqlen_balancing import get_seqlen_balanced_partitions
from molt.utils.utils import zero_pad_sequences

logger = init_logger(__name__)


def tensor_field(role: str, **kwargs):
    metadata = dict(kwargs.pop("metadata", {}))
    metadata["tensor_role"] = role
    return field(metadata=metadata, **kwargs)


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def get_model_parallel_size(args) -> int:
    """Members of one DP group — the ranks that share a data shard, i.e. ``cp * tp``.

    EP shards experts on a separate MoE mesh but each EP rank still owns its full data
    shard, so EP must NOT enter the divisor that splits a batch across DP groups.
    """
    fsdp = args.fsdp
    return int(fsdp.cp_size) * int(fsdp.tp_size)


def _fill_missing_routed_experts(items: List["Experience"]) -> None:
    """Give un-routed samples (routed_experts=None) an all -1 (natural-routing) block sized to
    their own sequence, so a batch mixing captured routing with None neither drops the routing
    (leading None) nor crashes on None.size(-1) (trailing None) in the first-element merge."""
    routed = next((it.routed_experts for it in items if it.routed_experts is not None), None)
    if routed is None:  # all-None batches merge to None untouched; all-present skip the fill
        return
    layers, topk = routed.shape[-3:-1]  # fixed by the model; routed_experts is (..., L, topk, T)
    for it in items:
        if it.routed_experts is None:
            *lead, seq = it.sequences.shape  # seq-aligned with sequences (..., T)
            it.routed_experts = torch.full((*lead, layers, topk, seq), -1, dtype=routed.dtype)


@dataclass
class Experience:
    """A batch of RL experience for policy optimization.

    Fields are grouped by RL semantics:
    - Trajectory: token-level state-action sequences and masks (B, T)
    - Policy: next-token step tensors under different policies (B, T-1)
    - Optimization: per-step returns and advantages (B, T-1)
    - Outcome: per-episode rewards and generation metadata (B,)
    - Metadata: non-tensor fields for logging and data tracking

    Policy/target tensors keep the dense next-token axis instead of compressing
    to action-only positions. In multi-turn rollouts, observation/tool feedback
    remains present on that axis and is excluded by action_mask=False.
    """

    # Trajectory: state-action sequences
    sequences: torch.Tensor = tensor_field("step", default=None)  # (B, T) token ids [prompt + response]
    attention_mask: torch.LongTensor = tensor_field("step", default=None)  # (B, T)
    action_mask: torch.BoolTensor = tensor_field("step", default=None)  # (B, T-1) generated-token steps

    # Policy: log probs under current, reference, and rollout policies
    action_log_probs: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) log pi_theta(a|s)
    base_action_log_probs: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) log pi_ref(a|s)
    rollout_log_probs: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) log pi_old(a|s)
    # R3 rollout routing replay: the rollout router's top-k expert ids per token, one row
    # per MoE layer. Stored seq-LAST as (B, num_moe_layers, topk, T) so it rides the same
    # right-pad/concat/stack machinery as the (B, T) step tensors; the actor forward
    # permutes it back to token-major and replays it. None when R3 off.
    routed_experts: torch.Tensor = tensor_field("step", default=None)

    # Policy-gradient targets
    returns: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) G_t (PPO: value-regression target)
    advantages: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) A(s,a)
    values: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) critic V(s) at collection (PPO old_values)
    kl: torch.Tensor = tensor_field("step", default=None)  # (B, T-1) D_KL(pi_theta || pi_ref)

    # Episode outcomes (per-sample scalars)
    rewards: torch.Tensor = tensor_field("episode", default=None)  # (B,) R, used for advantage calculation
    scores: torch.Tensor = tensor_field("episode", default=None)  # (B,) binary score for dynamic sampling
    response_length: torch.Tensor = tensor_field("episode", default=None)  # (B,) number of generated action tokens
    truncated: torch.Tensor = tensor_field("episode", default=None)  # (B,) whether generation was truncated
    total_length: torch.Tensor = tensor_field("episode", default=None)  # (B,) prompt + response length

    # Per-sample row id within the rollout batch (set to [i] per sample by
    # make_experience). len(index) = number of samples in this Experience — the
    # advantage/merge logic relies on this count, so it is NOT pure metadata.
    index: list[int] = None

    # Metadata (not part of RL computation)
    prompts: list[str] = field(default_factory=list)
    labels: list[str] = field(default_factory=list)
    images: list = field(default_factory=list)  # per-sample image paths/URLs for VLM (None entries for text-only)
    mm_train_inputs: list = field(default_factory=list)  # per-sample processor outputs (pixel_values dicts) for VLM
    info: dict = field(default_factory=dict)  # per-sample metrics for logging
    # GRPO grouping identity. `group_ids` (= prompt id) is shared by all N rollouts of one
    # prompt — the trainer averages their rewards to form the baseline. `rollout_ids` is
    # unique per trajectory; multi-turn agents emit several step-samples sharing one
    # rollout_id so the trainer dedups to one reward per rollout before grouping by group_id.
    group_ids: list[str] = field(default_factory=list)
    rollout_ids: list[str] = field(default_factory=list)

    # Distributed rollout: when set, the heavy tensors below (HEAVY_FIELDS) live in the object
    # store — produced and kept on the runner that generated the sample — and this holds the ref
    # to them. The lightweight fields (masks, rewards, ids, info) stay in place, so the controller
    # groups, scores and length-balances the sample without ever fetching its images. `reload()`
    # restores the heavy tensors on whichever rank consumes the sample. None once the sample is local.
    heavy_ref: Any = None

    # The fields offloaded to `heavy_ref` — everything the training forward needs (image bytes +
    # pixel_values, token ids, rollout routing) but the controller's advantage/length-balance logic
    # does not, so they never reach the controller. Rule: keep only what the controller reads light;
    # offload the rest (a byte-embedded `images` dataset column can be large).
    HEAVY_FIELDS = ("sequences", "attention_mask", "rollout_log_probs", "routed_experts", "mm_train_inputs", "images")

    def offload(self) -> "Experience":
        """Move this sample's heavy fields into the object store (on the producing runner) and keep
        only a ref, so the controller ships a handle, not images. Returns self. Undo with `reload()`.
        Idempotent (like `reload()`): a no-op if already offloaded, so a second call never re-puts the
        now-nulled fields and overwrites the ref."""
        if self.heavy_ref is not None:
            return self
        heavy = {name: getattr(self, name) for name in self.HEAVY_FIELDS}
        self.heavy_ref = ray.put(heavy)
        for name in self.HEAVY_FIELDS:
            setattr(self, name, None)  # only the ref remains on the controller now
        return self

    def reload(self) -> "Experience":
        """Restore the heavy fields from the object store — fetched peer-to-peer from wherever they
        live (the producing runner), never via the controller. Idempotent: a no-op if already local."""
        if self.heavy_ref is not None:
            heavy = ray.get(self.heavy_ref)
            for name, value in heavy.items():
                setattr(self, name, value)
            self.heavy_ref = None
        return self

    @classmethod
    def is_step_tensor_field(cls, name: str) -> bool:
        field_info = cls.__dataclass_fields__.get(name)
        return field_info is not None and field_info.metadata.get("tensor_role") == "step"

    @classmethod
    def is_episode_tensor_field(cls, name: str) -> bool:
        field_info = cls.__dataclass_fields__.get(name)
        return field_info is not None and field_info.metadata.get("tensor_role") == "episode"

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for name, value in self.__dict__.items():
            if isinstance(value, dict):
                setattr(self, name, {key: to(val, device) for key, val in value.items()})
            else:
                setattr(self, name, to(value, device))

        return self


# Batch manipulation utilities


def split_experience_batch(experience: Experience) -> List[Experience]:
    """Split a batched Experience into individual single-sample Experiences."""
    batch_size = len(experience.sequences)
    experience.index = None

    items = []
    for i in range(batch_size):
        kwargs = {}
        for f in fields(Experience):
            value = getattr(experience, f.name)
            if value is None:
                kwargs[f.name] = None
            elif isinstance(value, torch.Tensor):
                if len(value) != batch_size:
                    raise ValueError(f"Size of {f.name} ({len(value)}) does not match batch_size ({batch_size})")
                kwargs[f.name] = value[i]
            elif isinstance(value, dict):
                d = {}
                for k, v in value.items():
                    if isinstance(v, (torch.Tensor, list)):
                        if len(v) != batch_size:
                            raise ValueError(
                                f"Size of {f.name}[{k}] ({len(v)}) does not match batch_size ({batch_size})"
                            )
                        d[k] = v[i]
                    else:
                        raise TypeError(f"Unsupported type for {f.name}[{k}]: {type(v)}")
                kwargs[f.name] = d
            elif isinstance(value, list):
                kwargs[f.name] = [value[i]] if len(value) == batch_size else value
        items.append(Experience(**kwargs))

    return items


def make_experience_batch(items: List[Experience]) -> Experience:
    """Combine individual single-sample Experiences into a batched Experience."""
    if not items:
        raise ValueError("Empty items list")

    # A rollout with no captured routing has routed_experts=None; fill it (sized to its own
    # sequence) before batching so a mix doesn't drop the batch's routing or crash on
    # None.size(-1).
    _fill_missing_routed_experts(items)

    kwargs = {}
    for f in fields(Experience):
        first = getattr(items[0], f.name)
        if first is None:
            kwargs[f.name] = None
        elif isinstance(first, torch.Tensor):
            tensors = [getattr(item, f.name) for item in items]
            if Experience.is_step_tensor_field(f.name):
                # routed_experts pads with the R3 -1 sentinel (keep live routing); 0 is a
                # valid expert id and would force pad tokens to expert 0. Others pad with 0.
                pad_value = -1 if f.name == "routed_experts" else 0
                kwargs[f.name] = zero_pad_sequences(tensors, "right", stack=True, value=pad_value)
            elif Experience.is_episode_tensor_field(f.name) or first.dim() == 0:
                kwargs[f.name] = torch.stack(tensors)
            else:
                raise ValueError(f"Unsupported tensor field batching rule for {f.name}")
        elif isinstance(first, dict):
            kwargs[f.name] = {}
            for key in first.keys():
                vals = [getattr(item, f.name)[key] for item in items]
                if not vals:
                    continue
                first_type = type(vals[0])
                if not all(isinstance(v, first_type) for v in vals):
                    raise TypeError(f"Inconsistent types in {f.name}[{key}]")
                if all(isinstance(v, (int, float)) for v in vals):
                    kwargs[f.name][key] = torch.tensor(vals)
                else:
                    kwargs[f.name][key] = vals
        elif isinstance(first, list):
            kwargs[f.name] = list(itertools.chain.from_iterable(getattr(item, f.name) for item in items))

    return Experience(**kwargs)


def remove_padding_in_sequences(items: List[Experience]) -> List[Experience]:
    """Remove right padding from per-step fields of single-sample Experiences."""
    for item in items:
        right_pad = item.attention_mask.flip(0).argmax()
        right_pad = None if right_pad == 0 else -right_pad

        for f in fields(Experience):
            value = getattr(item, f.name)
            if isinstance(value, torch.Tensor) and Experience.is_step_tensor_field(f.name):
                # Slice the LAST (sequence) dim: 1D step tensors are [T], but
                # routed_experts is [num_moe_layers, topk, T] (seq last).
                setattr(item, f.name, value[..., :right_pad])

    return items


def balance_experiences(experiences, args):
    """Assign the rollout equally across DP ranks by total sequence length, returning the samples
    reordered into contiguous per-rank blocks. ``async_run_method_batch``'s even slice then hands
    each rank its block, and each rank restores its own heavy tensors via ``Experience.reload()``.

    Every DP rank must receive the SAME number of samples — unequal counts give different
    ``num_steps`` per rank -> mismatched collective shapes at the world all-reduces -> NCCL hang — so
    the trailing remainder is dropped. Within a rank the block is sorted by length descending so the
    k-th micro-batch is size-matched across ranks (else a straggler trips the 600s NCCL watchdog).
    Metadata only — the heavy tensors stay in shared memory, so a full image batch never reaches the
    controller.
    """
    actor_world_size = args.actor.num_nodes * args.actor.num_gpus_per_node
    effective_num = actor_world_size // get_model_parallel_size(args)
    if effective_num <= 0:
        raise ValueError(f"Invalid effective actor count: {effective_num}")
    if len(experiences) < effective_num:
        raise ValueError(
            f"Cannot balance {len(experiences)} samples across {effective_num} DP ranks. "
            "Increase rollout.batch_size/n_samples_per_prompt or drop the final partial batch."
        )

    lengths = []
    for exp in experiences:
        length = exp.total_length
        lengths.append(int(length.item() if isinstance(length, torch.Tensor) else length))

    # Drop the trailing partial batch so the count divides evenly across the DP ranks.
    remainder = len(experiences) % effective_num
    keep = len(experiences) - remainder
    if remainder:
        logger.warning(
            f"[balance_experiences] dropping {remainder} trailing sample(s) so "
            f"{len(experiences)} divides evenly across {effective_num} DP ranks."
        )
    partitions = get_seqlen_balanced_partitions(lengths[:keep], effective_num, equal_size=True)

    # Concatenate the per-rank blocks into one flat list. Within each rank, place the longest
    # samples first so the k-th micro-batch is size-matched across ranks (a straggler otherwise
    # makes short ranks wait long enough to trip the NCCL watchdog). async_run_method_batch's even
    # slice then hands each rank its contiguous block.
    balanced = []
    for partition in partitions:
        longest_first = sorted(partition, key=lambda i: lengths[i], reverse=True)
        balanced.extend(experiences[i] for i in longest_first)
    return balanced
