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

"""cp_size==1 packed-sequence varlen SDPA integration.

Same varlen kernel as the CP block-diagonal path, but for ``cp_size == 1`` (no
sequence sharding). The whole packed sequence lives on one rank, so it
degenerates to ``_cp_blockdiag_varlen`` with ``row_offset=0`` (q == k == full
sequence; square; ``cu_q == cu_k``). This gives packed-sequence block-diagonal
attention to models whose softmax attention only dispatches to sdpa.

Integration mirrors :func:`nemo_automodel.components.distributed.cp_utils.attach_cp_sdpa_hooks`:

1. :func:`attach_cp1_packed_varlen_hooks` registers forward pre/post hooks on every
   ``self_attn`` module (the checkpoint-wrapped INNER module, so the hooks also fire
   during activation-checkpointing recompute). The pre-hook swaps
   ``F.scaled_dot_product_attention`` for :func:`_packed_varlen_sdpa`; the post-hook
   (``always_call=True``) restores stock SDPA. The patch is therefore live only
   while a hooked attention forward is running -- never process-wide.
2. The model arms the per-forward state with :func:`enable_cp1_packed_varlen`
   (doc_ids + backend) at the start of its forward and clears stale state before
   the NEXT forward with :func:`disable_cp1_packed_varlen`.

IMPORTANT -- state must survive activation-checkpointing RECOMPUTE in backward:
the state is set per-forward and NOT reset at the end of the step, so the AC
worker thread re-running a layer's forward during backward reads the SAME doc_ids
and reproduces the exact varlen output. (A per-step reset caused the forward to
use varlen but the recompute to fall back to dense -> AC "saved vs recomputed
metadata" shape mismatch.) ``_ThreadSharedVar`` makes the state visible in the
autograd worker thread. Because the outer model clears stale state before each
new forward, vision/unpacked attention cannot inherit it -- and the shape check
in :func:`_packed_varlen_sdpa` passes any non-matching call straight through.
"""

from __future__ import annotations

import torch

from nemo_automodel.components.distributed.blockdiag_cp import kernels, runtime, state

_PACKED_STATE = state._ThreadSharedVar()


def _packed_varlen_sdpa(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    **kwargs,
):
    """SDPA replacement for cp1 packed runs: block-diagonal varlen from doc_ids.

    Only intervenes for a full packed-sequence call whose batch and sequence
    dims match the armed ``doc_ids``; any other SDPA call (or state unset)
    passes straight through to stock SDPA.

    Args:
        query: Queries ``[B, Hq, S, D]`` over the full (unsharded) packed
            sequence (``B`` = batch, ``Hq`` = query heads, ``S`` = sequence
            length, ``D`` = head dim).
        key: Keys ``[B, Hkv, S, D]``.
        value: Values ``[B, Hkv, S, D]``.
        attn_mask: Forwarded to stock SDPA on pass-through; ignored on the
            varlen path (masking is rebuilt from ``doc_ids``).
        dropout_p: Dropout probability.
        is_causal: Forwarded on pass-through; the varlen path is always
            per-document causal.
        scale: Softmax scale (``None`` -> ``D**-0.5``).
        enable_gqa: Grouped-query attention flag as passed by HF's sdpa path.
        **kwargs: Ignored; accepted for SDPA signature compatibility.

    Returns:
        Attention output ``[B, Hq, S, D]``.
    """
    packed_state = _PACKED_STATE.get()
    # Only intervene for the full packed sequence whose length matches doc_ids;
    # any other SDPA call (or state unset) passes straight through.
    doc_ids = None if packed_state is None else packed_state.get("doc_ids")
    expected_shape = tuple(doc_ids.shape) if isinstance(doc_ids, torch.Tensor) and doc_ids.dim() == 2 else None
    if (
        packed_state is None
        or expected_shape is None
        or query.dim() != 4
        or key.dim() != 4
        or value.dim() != 4
        or query.shape[0] != expected_shape[0]
        or key.shape[0] != expected_shape[0]
        or value.shape[0] != expected_shape[0]
        or query.shape[2] != expected_shape[1]
        or key.shape[2] != expected_shape[1]
        or value.shape[2] != expected_shape[1]
    ):
        return runtime._ORIGINAL_SDPA(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    # query/key/value: [B, Hq, S, D] / [B, Hkv, S, D]; full (unsharded) sequence.
    out = kernels._cp_blockdiag_varlen(
        query,
        key,
        value,
        packed_state["doc_ids"],
        0,
        scale,
        packed_state["backend"],
        dropout_p=dropout_p,
        meta=packed_state.get("varlen_meta"),
    )
    if out is None:  # backend unavailable / unsupported -> safe fallback
        B = query.shape[0]
        L = query.shape[2]
        S = key.shape[2]
        n_q_heads = query.shape[1]
        n_kv_heads = key.shape[1]
        if enable_gqa and n_kv_heads != n_q_heads:
            n_rep = n_q_heads // n_kv_heads
            key = key.repeat_interleave(n_rep, dim=1)
            value = value.repeat_interleave(n_rep, dim=1)
            enable_gqa = False
        allow = kernels._cp_blockdiag_mask(packed_state["doc_ids"], 0, L, S, B)
        return runtime._ORIGINAL_SDPA(
            query,
            key,
            value,
            attn_mask=allow,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale,
            enable_gqa=enable_gqa,
        )
    return out


def attach_cp1_packed_varlen_hooks(model: torch.nn.Module) -> None:
    """Scope the cp1 packed varlen SDPA patch to the model's attention forwards.

    Registers a forward pre-hook / post-hook pair on every ``self_attn`` module
    that installs :func:`_packed_varlen_sdpa` as ``F.scaled_dot_product_attention``
    for the duration of that module's forward and restores stock SDPA afterwards
    (``always_call=True``, so a raising forward cannot leak the patch). Hooks are
    attached to the checkpoint-wrapped INNER module because CheckpointWrapper's
    recompute bypasses ``__call__`` on the wrapper -- this is what keeps the
    varlen path active during activation-checkpointing recompute in backward.

    Same bounded-patch pattern as
    :func:`nemo_automodel.components.distributed.cp_utils.attach_cp_sdpa_hooks`.
    Outside these hooks, ``F.scaled_dot_product_attention`` is untouched.

    Args:
        model: The model whose ``self_attn`` submodules route softmax attention
            through ``F.scaled_dot_product_attention``.
    """
    import torch.nn.functional as F_module
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    def _pre_hook(module, args, kwargs):
        F_module.scaled_dot_product_attention = _packed_varlen_sdpa
        return args, kwargs

    def _post_hook(module, inputs, output):
        F_module.scaled_dot_product_attention = runtime._ORIGINAL_SDPA

    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            target = module._checkpoint_wrapped_module if isinstance(module, CheckpointWrapper) else module
            target.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            target.register_forward_hook(_post_hook, always_call=True)


def enable_cp1_packed_varlen(doc_ids: torch.Tensor, backend: str) -> None:
    """Arm cp1 packed block-diagonal varlen for the rest of this step.

    Sets the per-forward doc_ids/backend state read by the SDPA patch that
    :func:`attach_cp1_packed_varlen_hooks` scopes to the attention forwards. The
    state remains armed through backward's activation-checkpoint recomputation;
    the next outer model forward clears it via :func:`disable_cp1_packed_varlen`.

    Args:
        doc_ids: Per-position document ids ``[1, S]`` or ``[S]`` (0 == padding)
            over the full packed sequence.
        backend: Varlen kernel backend, ``"flash"`` or ``"te"``.
    """
    if doc_ids.dim() == 1:
        doc_ids = doc_ids.unsqueeze(0)

    # Segmentation depends only on the packed document ids, not on the layer's
    # Q/K/V tensors. Compute it once per outer forward so every attention
    # layer (and activation-checkpoint recompute) reuses the same CUDA
    # cu_seqlens without repeating several GPU->CPU ``.item()`` synchronizations.
    # Packing currently requires one packed row per local batch. Preserve the
    # old inline fallback for an unexpected multi-row direct caller rather than
    # silently applying row 0's metadata to every row.
    varlen_meta = None
    if isinstance(doc_ids, torch.Tensor) and (doc_ids.dim() == 1 or (doc_ids.dim() == 2 and doc_ids.shape[0] == 1)):
        varlen_meta = kernels.precompute_blockdiag_varlen_meta(
            doc_ids,
            row_offset=0,
            local_len=int(doc_ids.shape[-1]),
            device=doc_ids.device,
        )
    _PACKED_STATE.set({"doc_ids": doc_ids, "backend": backend, "varlen_meta": varlen_meta})


def disable_cp1_packed_varlen() -> None:
    """Disarm stale cp1 state before starting a new outer model forward.

    Decoder-layer activation-checkpoint recomputation happens before the next
    outer forward, so the preceding step's state remains available for backward
    and is cleared before vision or any other attention in the next batch runs.
    """
    _PACKED_STATE.set(None)


def cp1_packed_varlen_backend() -> str | None:
    """The configured cp1 packed varlen backend ('te'/'flash'), or None if disabled (dense)."""
    backend = state._CP_ATTN_BACKEND
    return backend if backend in ("te", "flash") else None
