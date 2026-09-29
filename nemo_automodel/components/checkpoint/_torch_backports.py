# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Runtime back-ports for old PyTorch versions. Will be deleted in future stable PyTorch versions."""

from __future__ import annotations

import importlib
import logging
import threading

logger = logging.getLogger(__name__)


def apply_patches() -> None:
    """
    Inject modified modules into an *old* ``torch.distributed.checkpoint``.
    """
    # -----------------------------------------------------------------
    # Ensure SavePlanner provides the _cached_metadata class attribute.
    # This is required by NeMo-Automodel's extended planners but may be
    # missing from older PyTorch versions (< 2.4).  Monkey-patch it here
    # so downstream code can rely on its existence independent of the
    # installed torch release.
    # -----------------------------------------------------------------
    try:
        planner_mod = importlib.import_module("torch.distributed.checkpoint.planner")
        SavePlanner = getattr(planner_mod, "SavePlanner", None)
        if SavePlanner is not None and not hasattr(SavePlanner, "_cached_metadata"):
            # Forward-declare attribute; note we don't import Metadata to
            # avoid circular deps – a forward reference string in the
            # annotation keeps static checkers happy while remaining
            # runtime-safe.
            SavePlanner._cached_metadata = {}

            # Update type annotations dynamically for better type hints
            anns = getattr(SavePlanner, "__annotations__", {})
            anns.setdefault("_cached_metadata", "dict[str, 'Metadata']")
            SavePlanner.__annotations__ = anns  # type: ignore[attr-defined]

            logger.debug("Added missing SavePlanner._cached_metadata back-port")
    except ModuleNotFoundError:
        # planner module unavailable – nothing to patch
        pass


def apply_async_checkpoint_patch() -> None:
    """
    Apply stabilization patch for torch.distributed.checkpoint async process executor.

    This serializes creation of the global background process across concurrent
    async_save calls and makes its first initialization synchronous. PyTorch
    otherwise checks for the process on the caller thread but assigns it later on
    a worker thread, allowing consecutive distributed saves to disagree across
    ranks about whether process-group initialization is required.
    """
    try:
        ape_mod = importlib.import_module("torch.distributed.checkpoint._async_process_executor")

        # Idempotent guard
        if getattr(ape_mod, "_NEMO_PATCHED_CREATE_LOCK", False):
            return

        # Global creation lock
        if not hasattr(ape_mod, "_NEMO_CREATE_LOCK"):
            ape_mod._NEMO_CREATE_LOCK = threading.Lock()

        Exec = getattr(ape_mod, "_ProcessBasedAsyncCheckpointExecutor", None)
        if Exec is not None and not hasattr(Exec, "_nemo_orig_execute_save_impl"):
            Exec._nemo_orig_execute_save_impl = Exec._execute_save_impl

            def _nemo_locked_execute_save_impl(*args, **kwargs):
                with ape_mod._NEMO_CREATE_LOCK:
                    return Exec._nemo_orig_execute_save_impl(*args, **kwargs)

            try:
                Exec._execute_save_impl = staticmethod(_nemo_locked_execute_save_impl)
                logger.debug("Applied creation-lock patch to DCP process executor")
            except Exception:
                # Defensive: if staticmethod replacement fails, leave as-is
                logger.debug("Failed to assign locked _execute_save_impl", exc_info=True)

        ProcessGroupInitInfo = getattr(ape_mod, "_ProcessGroupInitInfo", None)
        AsyncCheckpointProcess = getattr(ape_mod, "_AsyncCheckpointProcess", None)
        if (
            Exec is not None
            and ProcessGroupInitInfo is not None
            and AsyncCheckpointProcess is not None
            and not hasattr(Exec, "_nemo_orig_execute_save")
        ):
            Exec._nemo_orig_execute_save = Exec.execute_save

            def _nemo_initialize_checkpoint_process(self, *args, **kwargs):
                with ape_mod._NEMO_CREATE_LOCK:
                    if ape_mod._CHECKPOINT_PROCESS is None:
                        pg_init_info = ProcessGroupInitInfo(kwargs.get("process_group"))
                        ape_mod._CHECKPOINT_PROCESS = AsyncCheckpointProcess(pg_init_info=pg_init_info)
                return Exec._nemo_orig_execute_save(self, *args, **kwargs)

            try:
                Exec.execute_save = _nemo_initialize_checkpoint_process
                logger.debug("Applied synchronous initialization patch to DCP process executor")
            except Exception:
                # Defensive: if method replacement fails, leave as-is
                logger.debug("Failed to assign patched execute_save", exc_info=True)

        ape_mod._NEMO_PATCHED_CREATE_LOCK = True
    except ModuleNotFoundError:
        # async_process_executor unavailable – nothing to patch
        pass
    except Exception:
        logger.debug("Unexpected error while applying DCP process executor patch", exc_info=True)
