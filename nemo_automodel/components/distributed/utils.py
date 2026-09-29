# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


import logging
from contextlib import ContextDecorator, nullcontext
from datetime import timedelta

import torch
import torch.distributed
import torch.distributed as dist


logger = logging.getLogger(__name__)


def _create_gloo_group() -> dist.ProcessGroup | None:
    """
    Create a Gloo process group for barrier operations.

    This allows us to use monitored_barrier with Gloo backend while
    keeping NCCL for the main training operations.

    Returns:
        ProcessGroup: Gloo process group for barriers
    """
    if not dist.is_initialized():
        return None

    try:
        gloo_group = dist.new_group(backend="gloo")
        logger.debug("Created Gloo group for barrier operations")
        return gloo_group
    except Exception as e:
        logger.warning(f"Failed to create Gloo group: {e}")
        return None


def _barrier_with_timeout(timeout: timedelta, group: dist.ProcessGroup | None = None) -> bool:
    """
    Run a barrier using the timeout configured on its process group.

    Explicit model-local groups are reused directly so subset ranks do not need
    to create another process group in a globally coordinated order. The default
    group path uses a temporary Gloo group and ``monitored_barrier``.

    Args:
        timeout: Maximum time to wait for the barrier
        group: Process group for the barrier operation

    Returns:
        bool: True if barrier completed successfully, False if timeout occurred
    """
    if not dist.is_initialized() or dist.get_world_size(group=group) == 1:
        return True

    if group is not None:
        dist.barrier(group=group)
        return True

    # Use Gloo group for barrier operations
    gloo_group = _create_gloo_group()
    if gloo_group is None:
        # Fallback to regular barrier if Gloo group creation fails
        try:
            dist.barrier(group=group)
            return True
        except Exception as e:
            logger.warning(f"Barrier failed: {e}")
            return False

    try:
        # Use monitored_barrier with Gloo group
        dist.monitored_barrier(group=gloo_group, timeout=timeout)
        return True
    except Exception as e:
        logger.warning(f"Monitored barrier failed: {e}")
        return False
    finally:
        # Clean up the Gloo group
        try:
            dist.destroy_process_group(gloo_group)
        except Exception:
            pass


class FirstRankPerNode(ContextDecorator):
    """
    Context manager to enforce rank0 to process section over other ranks.

      - Lets RANK==0 run the protected code first on each node.
      - Inserts an extra barrier across *only* the node-local rank-0 processes.
      - Works on a single GPU (no env flags, no distributed initialisation).

    Note: it is assumed the scoped code is not torch.distributed heavy.

    Args:
        group: Optional model-local process group. When provided, only members
            of that group participate in serialization and temporary Gloo
            barrier creation.
    """

    def __init__(self, group: dist.ProcessGroup | None = None) -> None:
        self.group = group

    def __enter__(self, timeout=timedelta(hours=10)):
        """
        Create / bootstrap a (distributed) proc. group that rank0 enters first.

        Returns:
            bool: ``True``  - if the current process is node-rank-0
                  ``False`` - otherwise
        """
        self._created_pg = False
        self._node0_group = None
        self._first = True  # default for single-GPU / no-dist case
        self._timeout = timeout
        if not dist.is_initialized() or dist.get_world_size(group=self.group) == 1:
            # pure single GPU
            return True

        # Figure out rank
        self._first = dist.get_rank(group=self.group) == 0

        # Synchronisation logic
        if not self._first:
            # Non-rank-0 processes wait for their node-rank-0
            # Use Gloo group for monitored_barrier to avoid NCCL timeout issues
            success = _barrier_with_timeout(timeout=self._timeout, group=self.group)
            if not success:
                logger.warning("Barrier timed out, continuing anyway")

        return self._first

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Tear down the context.

        1. If the current process was the first on its node, release the
           waiting peer ranks by issuing a barrier.
        2. If an exception occurred, abort the *entire* distributed job.
        3. If this context manager created the process group, destroy it.

        Args:
            exc_type (Type[BaseException] | None): Exception class if one
                occurred inside the ``with`` block.
            exc_val  (BaseException | None): The raised exception instance.
            exc_tb   (TracebackType | None): Traceback associated with the
                exception.

        Returns:
            bool: ``False`` so that any exception raised inside the ``with``
                  block is propagated to the caller (standard CM semantics).
        """
        try:
            if self._first and dist.is_initialized():
                # Re-sync the whole world so that non-rank-0s can proceed
                # Use Gloo group for monitored_barrier to avoid NCCL timeout issues
                success = _barrier_with_timeout(timeout=self._timeout, group=self.group)
                if not success:
                    logger.warning("Barrier timed out during exit, continuing anyway")
                if exc_type is not None:
                    # Log the exception and make the error visible
                    logger.error(
                        "Exception inside FirstRankPerNode: %s: %s",
                        exc_type.__name__,
                        exc_val,
                        exc_info=(exc_type, exc_val, exc_tb),
                    )
                    raise SystemExit(1) from exc_val
        finally:
            if self._created_pg:
                dist.destroy_process_group()

        # propagate any exception to outer scope
        return False


def get_sync_ctx(model, is_optim_step, defer_fsdp_grad_sync: bool):
    """
    Get the synchronization context for the model.

    Args:
        model: The model to synchronize.
        is_optim_step: Whether the current step is an optimizer step.
        defer_fsdp_grad_sync: Controls FSDP2 gradient synchronization during gradient accumulation.
            - True: disable gradient sync on non-final micro-batches (saves comm, can increase peak memory).
            - False: always sync gradients on every micro-batch (more comm, lower peak memory).

    Returns:
        A context manager that synchronizes the model.
    """
    # Use `no_sync` on DDP models when we are *not* on the final micro-batch for
    # this gradient update (i.e., when `is_grad` is False). This avoids an
    # all-reduce for every micro-batch and greatly improves throughput.
    sync_ctx = nullcontext()
    if isinstance(model, dist.fsdp._fully_shard._fully_shard.FSDPModule):
        if defer_fsdp_grad_sync:
            model.set_requires_gradient_sync(is_optim_step)
        else:
            model.set_requires_gradient_sync(True)
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel) and not is_optim_step:
        sync_ctx = model.no_sync()
    return sync_ctx


