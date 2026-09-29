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

"""Runtime knobs and activation-checkpoint-safe state for block-diagonal CP.

The knob normalization (synonym maps and defaults) lives here, in a
kernel-independent leaf, so every consumer (the kernel-side
:func:`configure_cp_varlen` entry point and any policy-side config parser)
derives the accepted values from the same table and can never disagree on
synonyms or defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor
from torch.distributed import ProcessGroup

# Canonical value -> accepted synonyms (each canonical is also a synonym of itself).
ATTN_BACKEND_SYNONYMS: dict[str, tuple[str, ...]] = {
    "flash": ("flash", "flash_attn", "flash_attention", "flash_attention_2", "fa2"),
    "te": ("te", "transformer_engine", "transformerengine"),
    "dense": ("dense", "sdpa", "torch_sdpa"),
}
KV_EXCHANGE_SYNONYMS: dict[str, tuple[str, ...]] = {
    "allgather": ("allgather", "all_gather"),
    "halo": ("halo", "neighbor", "needed", "needed_only"),
    "a2a": ("a2a", "a2av", "all_to_all", "alltoall", "needed_a2a"),
}

ATTN_BACKEND_DEFAULT = "flash"
KV_EXCHANGE_DEFAULT = "allgather"

# Canonical user-facing values (single source for config schemas + docs).
ATTN_BACKEND_VALUES: tuple[str, ...] = tuple(ATTN_BACKEND_SYNONYMS)
KV_EXCHANGE_VALUES: tuple[str, ...] = tuple(KV_EXCHANGE_SYNONYMS)


@dataclass(frozen=True)
class BlockdiagCpModelState:
    """Typed model-facing view of an active block-diagonal CP step.

    Args:
        group: Process group that shards the sequence across context-parallel ranks.
        packed_cu_seqlens: Global packed-document boundaries of shape
            ``[num_documents + 1]`` on the compute device.
        packed_cu_seqlens_cpu: CPU copy of ``packed_cu_seqlens`` with shape
            ``[num_documents + 1]`` for FLA's host-side CP planning.
    """

    group: ProcessGroup
    packed_cu_seqlens: Tensor
    packed_cu_seqlens_cpu: Tensor


_ATTN_BACKEND_REVERSE = {syn: canon for canon, syns in ATTN_BACKEND_SYNONYMS.items() for syn in syns}
_KV_EXCHANGE_REVERSE = {syn: canon for canon, syns in KV_EXCHANGE_SYNONYMS.items() for syn in syns}


def _clean(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def normalize_attn_backend(value: Any) -> str:
    """Canonicalize an attention-backend knob to one of ``ATTN_BACKEND_VALUES``.

    ``None``/``""``/``auto``/``default`` map to the default (``flash``).

    Args:
        value: The raw user-facing knob value (string-like or ``None``).

    Returns:
        The canonical backend name (``"flash"``, ``"te"``, or ``"dense"``).

    Raises:
        ValueError: If the value is not a recognized backend or synonym.
    """
    if value is None:
        return ATTN_BACKEND_DEFAULT
    cleaned = _clean(value)
    if cleaned in ("", "auto", "default"):
        return ATTN_BACKEND_DEFAULT
    canon = _ATTN_BACKEND_REVERSE.get(cleaned)
    if canon is not None:
        return canon
    raise ValueError(
        f"Unsupported block-diagonal CP attention backend '{value}'. Use one of: {', '.join(ATTN_BACKEND_VALUES)}."
    )


def normalize_kv_exchange(value: Any) -> str:
    """Canonicalize a KV-exchange knob to one of ``KV_EXCHANGE_VALUES``.

    ``None``/``""``/``auto``/``default`` map to the default (``allgather``).

    Args:
        value: The raw user-facing knob value (string-like or ``None``).

    Returns:
        The canonical exchange name (``"allgather"``, ``"halo"``, or ``"a2a"``).

    Raises:
        ValueError: If the value is not a recognized exchange mode or synonym.
    """
    if value is None:
        return KV_EXCHANGE_DEFAULT
    cleaned = _clean(value)
    if cleaned in ("", "auto", "default"):
        return KV_EXCHANGE_DEFAULT
    canon = _KV_EXCHANGE_REVERSE.get(cleaned)
    if canon is not None:
        return canon
    raise ValueError(
        f"Unsupported block-diagonal CP kv_exchange '{value}'. Use one of: {', '.join(KV_EXCHANGE_VALUES)}."
    )


_CP_ATTN_BACKEND = ATTN_BACKEND_DEFAULT
# KV delivery for the block-diagonal path:
#   "allgather" - full O(S) K/V all-gather to every rank (default; always correct)
#   "halo"      - needed-only left-neighbor halo exchange: each rank fetches only its
#                 boundary document's straddle from rank r-1 and attends local+halo.
#                 O(S/cp) per-rank KV + ~doc-sized comm instead of O(S). Auto-falls back
#                 to all-to-all-v when a document spans >2 ranks; the decision is computed
#                 from the replicated doc_ids so all ranks agree without communication.
#   "a2a"       - needed-only all-to-all-v (general case, handles docs spanning >2 ranks).
_CP_KV_EXCHANGE = KV_EXCHANGE_DEFAULT


def configure_cp_varlen(*, attn_backend: str = "flash", kv_exchange: str = "allgather") -> None:
    """Configure the block-diagonal CP attention path from parsed runtime config.

    Args:
        attn_backend: Varlen kernel selection; any synonym accepted by
            :func:`normalize_attn_backend` (default ``"flash"``).
        kv_exchange: K/V delivery mode; any synonym accepted by
            :func:`normalize_kv_exchange` (default ``"allgather"``).
    """
    backend = normalize_attn_backend(attn_backend)
    exchange = normalize_kv_exchange(kv_exchange)

    global _CP_ATTN_BACKEND, _CP_KV_EXCHANGE
    changed = (_CP_ATTN_BACKEND, _CP_KV_EXCHANGE) != (backend, exchange)
    _CP_ATTN_BACKEND = backend
    _CP_KV_EXCHANGE = exchange
    if changed:
        # Keep the one-shot kernel marker in its owning module. Import lazily to
        # avoid a state <-> kernels import cycle during module initialization.
        from nemo_automodel.components.distributed.blockdiag_cp import kernels

        kernels._CP_FLASH_ENGAGED = False


def cp_varlen_runtime_config() -> dict[str, str]:
    """Return the currently configured block-diagonal CP runtime settings.

    Returns:
        A dict with keys ``"attn_backend"`` and ``"kv_exchange"`` holding the
        canonical configured values.
    """
    return {
        "attn_backend": _CP_ATTN_BACKEND,
        "kv_exchange": _CP_KV_EXCHANGE,
    }


class _ThreadSharedVar:
    """A ``contextvars.ContextVar``-style get/set/reset holder visible ACROSS threads.

    The block-diagonal CP state must be readable inside the autograd worker thread
    that runs activation-checkpointing recompute during backward. A real
    ``ContextVar`` is per-thread and reads its default (None) there, which would
    silently drop the CP state mid-recompute -- the softmax SDPA would fall back to
    local-only attention. Training steps are sequential, so a single shared slot
    with token-based restore (supporting nesting) is safe; backward only ever READS
    the slot.
    """

    def __init__(self):
        self._value = None

    def get(self):
        """Return the current value (``None`` when unset)."""
        return self._value

    def set(self, value):
        """Set the value; returns a token (the previous value) for :meth:`reset`."""
        token = self._value
        self._value = value
        return token

    def reset(self, token):
        """Restore the value captured by a previous :meth:`set`."""
        self._value = token


# Per-step state read by the block-diagonal SDPA (set by make_cp_blockdiag_batch_and_ctx).
# Holds internal runtime fields plus a typed ``model_state`` view for model-owned
# recurrent-attention integrations.
_CP_BLOCKDIAG_STATE = _ThreadSharedVar()

# Sentinel: counts block-diagonal CP attention invocations (the CP path actually
# running) so a recipe can fail loud if cp>1 but the hook never fired. A plain
# list cell suffices: the forward runs single-threaded; AC-recompute increments in
# backward are post-check and only matter as ">0".
_CP_ATTN_FIRE_COUNT: list[int] = [0]


def current_blockdiag_cp_state() -> BlockdiagCpModelState | None:
    """Return the active block-diagonal CP step state, or ``None``.

    The state is published by
    :func:`~nemo_automodel.components.distributed.blockdiag_cp.batch.make_cp_blockdiag_batch_and_ctx`
    for the duration of forward and backward. Model-owned recurrent-attention
    implementations use it to share the same contiguous layout and packed
    document boundaries as the softmax-attention transport.

    Returns:
        The immutable model-facing state, whose packed-boundary tensors have
        shape ``[num_documents + 1]``, or ``None`` outside the CP context.
    """
    step_state = _CP_BLOCKDIAG_STATE.get()
    return None if step_state is None else step_state["model_state"]


