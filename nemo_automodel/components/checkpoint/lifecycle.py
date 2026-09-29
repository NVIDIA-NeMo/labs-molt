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

"""Checkpoint directory publication and retention lifecycle."""

from __future__ import annotations

import logging
import math
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import torch

from nemo_automodel.components.checkpoint.utils import (
    clear_checkpoint_incomplete,
    find_pointer_protected_checkpoints,
    format_missing_checkpoint_dir_error,
    is_checkpoint_incomplete,
    is_cloud_path,
    list_automodel_checkpoints,
    mark_checkpoint_incomplete,
    read_checkpoint_metric,
    read_checkpoint_pointer,
)

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

    from nemo_automodel.components.checkpoint.config import CheckpointingConfig

logger = logging.getLogger(__name__)

_RESERVATION_STAGING_SUFFIX = ".reserving-"


@dataclass(frozen=True)
class _PendingBestCheckpoint:
    """Best-checkpoint pointer update deferred until an async save completes."""

    path: str
    value: float | None
    metric_key: str | None


class CheckpointLifecycle:
    """Own checkpoint reservation, publication, pointers, and retention.

    The lifecycle is built with the same process group as its ``Checkpointer``.
    Filesystem work runs on that group's rank 0 and failures are reduced over the
    same participants, so every participating rank either continues or raises.

    Args:
        config: Declarative checkpoint configuration.
        process_group: Optional model-local group participating in checkpointing.
    """

    def __init__(
        self,
        config: CheckpointingConfig,
        process_group: ProcessGroup | None = None,
    ) -> None:
        self.config = config
        self.process_group = process_group
        self._best_val_loss = float("inf")
        self._pending_checkpoint_dir: str | None = None
        self._pending_best_checkpoint: _PendingBestCheckpoint | None = None

    def reserve(self, path: str) -> None:
        """Reserve ``path`` for a save, replacing only an interrupted checkpoint.

        Local checkpoint directories are created under a marked staging name and
        atomically renamed into place. An existing marked directory can be
        replaced, while an unmarked published checkpoint is never overwritten.
        ``msc://`` roots retain their existing local recipe-metadata shadow but do
        not use filesystem lifecycle markers.

        Args:
            path: Checkpoint directory for the current step.

        Raises:
            FileExistsError: If ``path`` already holds a published checkpoint.
            RuntimeError: If the directory could not be prepared.
        """
        if is_cloud_path(path):

            def prepare_remote_metadata_dir() -> None:
                logger.warning(
                    "Checkpoint directory %s uses MSC storage; interrupted-save detection and "
                    "automatic stale-directory replacement are unavailable",
                    path,
                )
                os.makedirs(path, exist_ok=True)

            self.run_coordinator_step(
                prepare_remote_metadata_dir,
                description=f"prepare local recipe metadata directory for remote checkpoint {path}",
            )
            return

        holds_published = self._is_coordinator() and os.path.exists(path) and not is_checkpoint_incomplete(path)
        if self._any_participant_reported(holds_published):
            raise FileExistsError(
                f"Checkpoint directory {path} already holds a published checkpoint. Remove it, or point "
                "checkpoint.checkpoint_dir at a fresh directory, before re-running this step."
            )

        def reserve_local_directory() -> None:
            if os.path.exists(path):
                logger.warning("Replacing checkpoint directory %s left behind by an interrupted save", path)
                shutil.rmtree(path)

            parent = os.path.dirname(path)
            staging_prefix = f"{os.path.basename(path)}{_RESERVATION_STAGING_SUFFIX}"
            if os.path.isdir(parent):
                for entry in os.listdir(parent):
                    if entry.startswith(staging_prefix):
                        shutil.rmtree(os.path.join(parent, entry), ignore_errors=True)

            staging = f"{path}{_RESERVATION_STAGING_SUFFIX}{uuid4().hex}"
            os.makedirs(staging)
            mark_checkpoint_incomplete(staging)
            os.rename(staging, path)

        self.run_coordinator_step(
            reserve_local_directory,
            description=f"prepare checkpoint directory {path}",
        )

    def run_coordinator_step(self, operation: Callable[[], None], *, description: str) -> None:
        """Run one coordinator operation and report any failure to all participants.

        Args:
            operation: Work to run on process-group rank 0.
            description: Step name used for logging and collective error reporting.

        Raises:
            RuntimeError: If the coordinator could not complete the operation.
        """
        failure: Exception | None = None
        if self._is_coordinator():
            try:
                operation()
            except Exception as exc:
                # The coordinator must still reach the reduction below; otherwise
                # its peers block in the next collective until the job is reaped.
                logger.exception("Failed to %s", description)
                failure = exc

        if self._any_participant_reported(failure is not None):
            raise RuntimeError(
                f"Failed to {description}; see the coordinator log for the underlying error."
            ) from failure

    def publish(
        self,
        path: str,
        *,
        best_val_metric: float | None,
        metric_key: str | None,
    ) -> None:
        """Publish a synchronous checkpoint, update pointers, and apply retention.

        Args:
            path: Fully written checkpoint directory.
            best_val_metric: Optional validation value eligible for ``LOWEST_VAL``.
            metric_key: Validation metric key stored in checkpoint loss metadata.
        """

        def publish_checkpoint() -> None:
            self._publish_checkpoint(path)
            if best_val_metric is not None:
                self._update_best_checkpoint(path, best_val_metric, metric_key)
            self._prune_old_checkpoints()

        self.run_coordinator_step(publish_checkpoint, description=f"publish checkpoint {path}")
        self._barrier()

    def defer_publication(
        self,
        path: str,
        *,
        best_val_metric: float | None,
        metric_key: str | None,
    ) -> None:
        """Record an async checkpoint for publication after its writes complete.

        Args:
            path: Checkpoint directory being written asynchronously.
            best_val_metric: Optional validation value eligible for ``LOWEST_VAL``.
            metric_key: Validation metric key stored in checkpoint loss metadata.
        """
        self._pending_checkpoint_dir = path
        self._pending_best_checkpoint = _PendingBestCheckpoint(
            path=path,
            value=best_val_metric,
            metric_key=metric_key,
        )

    def complete_pending(self) -> None:
        """Publish a completed async checkpoint and apply pointer retention."""
        pending_checkpoint_dir = self._pending_checkpoint_dir
        pending_best_checkpoint = self._pending_best_checkpoint
        if pending_checkpoint_dir is None and pending_best_checkpoint is None:
            return

        if pending_checkpoint_dir is not None:
            self.run_coordinator_step(
                lambda: self._publish_checkpoint(pending_checkpoint_dir),
                description=f"publish checkpoint {pending_checkpoint_dir}",
            )
            self._pending_checkpoint_dir = None
            self._barrier()

        if pending_best_checkpoint is not None:

            def update_best_checkpoint() -> None:
                if pending_best_checkpoint.value is None:
                    return
                self._update_best_checkpoint(
                    pending_best_checkpoint.path,
                    pending_best_checkpoint.value,
                    pending_best_checkpoint.metric_key,
                )

            self.run_coordinator_step(
                update_best_checkpoint,
                description="update the LOWEST_VAL pointer",
            )
            self._pending_best_checkpoint = None
            self._barrier()

        self.run_coordinator_step(self._prune_old_checkpoints, description="prune old checkpoints")
        self._barrier()

    def validate_checkpoint_dir_exists(self, ckpt_dir: str, restore_from: str) -> None:
        """Validate a resolved restore directory collectively.

        Args:
            ckpt_dir: Resolved checkpoint directory.
            restore_from: User-provided restore selector.

        Raises:
            FileNotFoundError: If ``ckpt_dir`` does not exist.
        """
        if os.path.exists(ckpt_dir):
            return

        if self._is_coordinator():
            error_message = format_missing_checkpoint_dir_error(
                checkpoint_dir=self.config.checkpoint_dir,
                restore_from=restore_from,
                resolved_ckpt_dir=ckpt_dir,
            )
        else:
            error_message = f"Checkpoint directory does not exist: {ckpt_dir}"

        self._barrier()
        raise FileNotFoundError(error_message)

    def _is_coordinator(self) -> bool:
        """Return whether this rank coordinates lifecycle filesystem work."""
        return not torch.distributed.is_initialized() or torch.distributed.get_rank(group=self.process_group) == 0

    def _any_participant_reported(self, reported: bool) -> bool:
        """Return whether any checkpoint process-group participant reported a condition."""
        flag = torch.tensor([int(reported)], dtype=torch.int32)
        if torch.distributed.is_initialized():
            if torch.cuda.is_available():
                flag = flag.cuda()
            torch.distributed.all_reduce(
                flag,
                op=torch.distributed.ReduceOp.MAX,
                group=self.process_group,
            )
        return bool(flag.item())

    def _barrier(self) -> None:
        """Synchronize checkpoint participants when distributed is initialized."""
        if torch.distributed.is_initialized():
            torch.distributed.barrier(group=self.process_group)

    def _publish_checkpoint(self, path: str) -> None:
        """Advance ``LATEST`` and commit a filesystem checkpoint's marker."""
        self._update_checkpoint_pointer("LATEST", path)
        if not is_cloud_path(path):
            clear_checkpoint_incomplete(path)

    def _update_checkpoint_pointer(self, link_name: str, target_dir: str) -> None:
        """Atomically update a checkpoint-root pointer."""
        checkpoint_root = os.fspath(self.config.checkpoint_dir)
        link_path = os.path.join(checkpoint_root, link_name)
        text_path = f"{link_path}.txt"

        checkpoint_root_abs = os.path.abspath(checkpoint_root)
        target_abs = os.path.abspath(target_dir)
        relative_target = os.path.relpath(target_abs, start=checkpoint_root_abs)
        temporary_path = os.path.join(checkpoint_root, f".{link_name}.{uuid4().hex}.tmp")
        try:
            try:
                os.symlink(relative_target, temporary_path)
            except OSError:
                if os.path.lexists(temporary_path):
                    os.remove(temporary_path)
                with open(temporary_path, "x") as pointer_file:
                    pointer_file.write(relative_target)
                    pointer_file.flush()
                    os.fsync(pointer_file.fileno())
                os.replace(temporary_path, text_path)
                temporary_path = ""
                if os.path.lexists(link_path):
                    os.remove(link_path)
            else:
                os.replace(temporary_path, link_path)
                temporary_path = ""
                if os.path.exists(text_path):
                    os.remove(text_path)
        finally:
            if temporary_path and os.path.lexists(temporary_path):
                os.remove(temporary_path)

    def _remove_checkpoint_pointer(self, link_name: str) -> None:
        """Remove a checkpoint pointer symlink and fallback text file."""
        checkpoint_root = os.fspath(self.config.checkpoint_dir)
        link_path = os.path.join(checkpoint_root, link_name)
        if os.path.lexists(link_path):
            os.remove(link_path)
        text_path = f"{link_path}.txt"
        if os.path.exists(text_path):
            os.remove(text_path)

    def _remove_stale_checkpoint_pointer(self, link_name: str) -> None:
        """Remove a checkpoint pointer whose target no longer exists."""
        target = read_checkpoint_pointer(self.config.checkpoint_dir, link_name)
        if target is not None and not target.is_dir():
            self._remove_checkpoint_pointer(link_name)

    def _prune_old_checkpoints(self) -> None:
        """Prune checkpoint directories according to the configured recent window."""
        max_recent_checkpoints = self.config.max_recent_checkpoints
        if max_recent_checkpoints is None:
            return

        checkpoint_root = Path(self.config.checkpoint_dir)
        checkpoints = list_automodel_checkpoints(checkpoint_root)
        try:
            protected_checkpoints = find_pointer_protected_checkpoints(checkpoint_root, checkpoints)
        except (OSError, UnicodeError):
            logger.warning(
                "Failed to scan checkpoint pointers in %s; skipping pruning",
                checkpoint_root,
                exc_info=True,
            )
            return

        complete_checkpoints = [checkpoint for checkpoint in checkpoints if not is_checkpoint_incomplete(checkpoint)]
        retained_window = set(complete_checkpoints[-max_recent_checkpoints:])
        checkpoints_to_delete = [
            checkpoint for checkpoint in checkpoints if checkpoint not in retained_window | protected_checkpoints
        ]
        for checkpoint in checkpoints_to_delete:
            try:
                shutil.rmtree(checkpoint)
            except OSError:
                logger.warning("Failed to prune old checkpoint directory %s", checkpoint, exc_info=True)
            else:
                logger.info("Pruned old checkpoint directory %s", checkpoint)

        self._remove_stale_checkpoint_pointer("LATEST")
        self._remove_stale_checkpoint_pointer("LOWEST_VAL")

    def _initialize_best_val_loss_from_pointer(self, metric_key: str | None) -> None:
        """Initialize best-metric state from the current complete ``LOWEST_VAL`` target."""
        if self._best_val_loss != float("inf"):
            return
        target = read_checkpoint_pointer(self.config.checkpoint_dir, "LOWEST_VAL")
        if target is None or not target.is_dir() or is_checkpoint_incomplete(target):
            return
        existing_best = read_checkpoint_metric(target, metric_key)
        if existing_best is not None:
            self._best_val_loss = existing_best

    def _update_best_checkpoint(
        self,
        target_dir: str,
        val_loss: float,
        metric_key: str | None,
    ) -> None:
        """Update ``LOWEST_VAL`` when a finite validation metric improves."""
        if not math.isfinite(val_loss):
            logger.warning("Ignoring non-finite validation metric for checkpoint %s: %s", target_dir, val_loss)
            return
        self._initialize_best_val_loss_from_pointer(metric_key)
        if val_loss < self._best_val_loss:
            self._best_val_loss = val_loss
            self._update_checkpoint_pointer("LOWEST_VAL", target_dir)
            logger.info(
                "Updated LOWEST_VAL checkpoint symlink to %s (val_loss=%.4f)",
                os.path.basename(target_dir),
                val_loss,
            )


__all__ = ["CheckpointLifecycle"]
