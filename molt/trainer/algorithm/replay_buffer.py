# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

from typing import List

import torch
from torch import distributed as dist

from molt.trainer.algorithm.experience import (
    Experience,
    get_model_parallel_size,
    make_experience_batch,
    remove_padding_in_sequences,
    split_experience_batch,
)
from molt.utils.seqlen_balancing import get_minimum_num_micro_batch_size, get_seqlen_balanced_partitions


class NaiveReplayBuffer:
    """Stores rollout experiences and yields training microbatches via the Dataset protocol.

    Consumed by a DataLoader (``__len__`` / ``__getitem__`` / ``collate_fn``), not by random sampling.

    Args:
        sample_batch_size (int): DataLoader batch size (microbatch size); forced to 1 in dynamic-batch mode.
        limit (int, optional): Cap on stored samples. A number <= 0 means unlimited. Defaults to 0.
        cpu_offload (bool, optional): Offload stored experience to CPU until fetched for a microbatch. Defaults to True.
    """

    def __init__(
        self,
        sample_batch_size: int,
        limit: int = 0,
        cpu_offload: bool = True,
        dynamic_batch: bool = False,
    ) -> None:
        super().__init__()
        self.sample_batch_size = sample_batch_size
        # limit <= 0 means unlimited
        self.limit = limit
        self.cpu_offload = cpu_offload
        self.items: List[Experience] = []
        self.dynamic_batch = dynamic_batch
        self.dynamic_indices: List[List[int]] = []
        self.dynamic_optimizer_step: List[int] = []

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device("cpu"))
        items = split_experience_batch(experience)
        items = remove_padding_in_sequences(items)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    def __len__(self) -> int:
        if self.dynamic_batch:
            return len(self.dynamic_indices)
        else:
            return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        if self.dynamic_batch:
            indices = self.dynamic_indices[idx]
            return [self.items[i] for i in indices]
        else:
            return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        if self.dynamic_batch:
            batch = batch[0]
        experience = make_experience_batch(batch)
        return experience

    def setup_dynamic_batch(self, strategy):
        args = strategy.args
        sample_lengths = [sample.total_length.item() for sample in self.items]

        world_size = dist.get_world_size()
        dp_size = world_size // get_model_parallel_size(args)
        if args.train.force_on_policy:
            # On-policy: the whole buffer is ONE optimizer step. A fixed
            # train.batch_size would split the rollout into several steps (every
            # step after the first off-policy) and drop the trailing samples that
            # don't fill a batch. balance_experiences already equalized the
            # per-rank sample count, so every rank takes exactly one step in
            # lockstep; the token-budget split below still bounds microbatch size.
            local_train_batch_size = len(sample_lengths)
            num_steps = 1 if sample_lengths else 0
        else:
            local_train_batch_size = args.train.batch_size // dp_size
            # Async generation can deliver a short buffer at episode boundaries.
            # Also, multi-turn agents may flatten to a variable number of samples
            # per prompt — use the actual buffer size, not the formula.
            num_steps = len(sample_lengths) // local_train_batch_size
            # balance_experiences only equalizes the per-rank count to a multiple
            # of dp_size, not of local_train_batch_size, so a remainder here is
            # dropped from this update. Surface it instead of dropping silently.
            dropped = len(sample_lengths) - num_steps * local_train_batch_size
            if dropped:
                strategy.print(
                    f"[ReplayBuffer] dropping {dropped} trailing sample(s) per rank that don't fill a "
                    f"{local_train_batch_size}-sample train batch (buffer={len(sample_lengths)})."
                )

        # split by train_batch_size, sync num_microbatches across dp
        num_microbatches = []
        for i in range(num_steps):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            num_microbatches.append(
                get_minimum_num_micro_batch_size(
                    sample_lengths[start:end],
                    args.train.max_tokens_per_gpu,
                    args.fsdp.cp_size,
                    args.fsdp.tp_size,
                )
            )

        num_microbatches = torch.tensor(num_microbatches, dtype=torch.int, device=torch.cuda.current_device())
        num_microbatches = strategy.all_reduce(num_microbatches, op="max")
        num_microbatches = num_microbatches.tolist()

        # balance the number of microbatches across steps
        micro_batch_indices = []
        data_partitions = []
        for i, num_mbs in enumerate(num_microbatches):
            start, end = i * local_train_batch_size, (i + 1) * local_train_batch_size
            samples = sample_lengths[start:end]
            partitions = get_seqlen_balanced_partitions(samples, num_mbs, equal_size=False)  # List[List[int]], index
            for j in range(num_mbs):
                for k in range(len(partitions[j])):
                    partitions[j][k] += start
            micro_batch_indices.extend(partitions)
            data_partitions.append(partitions)
        self.dynamic_indices = micro_batch_indices
        self.sample_batch_size = 1

        # Mark each step's last microbatch as the optimizer-step boundary. The
        # global token-mean (one denominator per window) handles per-microbatch
        # token-count weighting, so no per-microbatch loss scale is needed.
        optimizer_steps = []
        for partitions in data_partitions:
            optimizer_steps.extend([0] * (len(partitions) - 1) + [1])
        self.dynamic_optimizer_step = optimizer_steps
