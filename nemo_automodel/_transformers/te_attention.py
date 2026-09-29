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

"""TransformerEngine attention injection for HuggingFace models.

Replaces ``F.scaled_dot_product_attention`` within each HF ``self_attn``
module's forward pass with TE's ``DotProductAttention``, enabling the
FlashAttention-3 kernel and FP8 training without requiring model-specific
rewrites.

The injection works by:
1. Detecting ``self_attn`` modules with a standard HF projection layout
   (separate ``q_proj``, ``k_proj``, ``v_proj``).
2. Creating a ``DotProductAttention`` instance (stored as ``module.attn_module``)
   so that :func:`_uses_te_attention` can detect it.
3. Monkey-patching ``module.forward`` to temporarily swap in a TE-backed
   replacement for ``torch.nn.functional.scaled_dot_product_attention``
   while the original HF forward runs.

Call :func:`inject_te_attention` on the model *before* FSDP wrapping and
*after* any weight loading (so head-count/head-dim values are correct).

Supported patterns
------------------
- Standard Llama-style layout: separate ``q_proj``/``k_proj``/``v_proj``,
  GQA via ``repeat_kv`` (``enable_gqa=False``) or ``enable_gqa=True``.
  Covers Llama, Gemma, Qwen2, Mistral, and most popular HF causal LMs.

Mask handling
-------------
- ``is_causal=True`` with ``attn_mask=None`` → TE ``"causal"``.
- ``is_causal=False`` with ``attn_mask=None`` → TE ``"no_mask"``.
- A 4D ``attn_mask`` matching HF's canonical causal or causal+sliding pattern
  is detected by :func:`_detect_causal_mask` in O(S) and converted to
  ``("causal", window_size)``, so HF's always-present mask doesn't force a
  fallback. Per-sample padded batches or non-canonical mask patterns still
  fall back to native SDPA.

Sliding window
--------------
- The per-layer ``module.sliding_window`` attribute is read at injection time
  and converted to TE's ``(window_size[0], 0)`` convention
  (``sliding_window - 1`` tokens to the left). Both the runtime mask detector
  and the ``attn_mask=None`` path pass this to TE as ``window_size``.

Limitations
-----------
- Models using ``from torch.nn.functional import scaled_dot_product_attention``
  (a local import) will not pick up the runtime patch; affected modules are
  skipped with a warning.
"""

import contextlib
import logging
import os
import types
from typing import Any

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Attribute set on ``self_attn`` modules to hold the TE module,
# and on the top-level model to signal that injection was performed.
_TE_MODULE_ATTR = "attn_module"
_TE_MODEL_FLAG = "_te_attention_injected"

# Dispatch counters for diagnosing how often the TE path actually runs versus
# falling back to native SDPA.  Incremented inside ``te_sdpa``; exposed via
# :func:`get_te_attention_stats` / :func:`reset_te_attention_stats`.
_TE_STATS: dict[str, int] = {
    "te_hits": 0,
    "fallback_mask": 0,
    "fallback_scale_mismatch": 0,
}

# How often to auto-log the dispatch counters (call count).  Override with
# ``AUTOMODEL_TE_STATS_EVERY=<int>``; 0 disables auto-logging entirely.
_STATS_LOG_EVERY = int(os.environ.get("AUTOMODEL_TE_STATS_EVERY", "500"))

# One-shot flag so we warn at most once when the runtime ``scale`` disagrees
# with the ``softmax_scale`` captured at TE-module creation time.
_SCALE_MISMATCH_WARNED = False


def get_te_attention_stats() -> dict[str, int]:
    """Return a snapshot of the TE dispatch counters.

    Keys: ``te_hits`` (ran the real TE kernel), ``fallback_mask`` (fell back
    because ``attn_mask`` was non-None), ``fallback_scale_mismatch`` (fell
    back because the runtime ``scale`` argument disagreed with the TE
    module's fixed ``softmax_scale``).
    """
    return dict(_TE_STATS)


def reset_te_attention_stats() -> None:
    """Zero out the TE dispatch counters (test / benchmarking helper)."""
    for k in _TE_STATS:
        _TE_STATS[k] = 0


# ---------------------------------------------------------------------------
# Parameter inference
# ---------------------------------------------------------------------------


def _proj_out_features(proj: torch.nn.Module | None) -> int | None:
    """Return the output feature count of a projection module.

    Handles three layouts:
    - Standard ``nn.Linear``: reads ``proj.out_features`` directly.
    - Weight-only: reads ``proj.weight.shape[0]`` (works on meta device).
    - Wrapped linear (e.g. ``Gemma4ClippableLinear``): recurses into the
      ``proj.linear`` child module.
    """
    if proj is None:
        return None
    out = getattr(proj, "out_features", None)
    if out is not None:
        return int(out)
    w = getattr(proj, "weight", None)
    if w is not None:
        return int(w.shape[0])
    inner = getattr(proj, "linear", None)
    if inner is not None:
        return _proj_out_features(inner)
    return None


def _infer_attn_params(module: torch.nn.Module) -> dict[str, Any] | None:
    """Infer attention hyper-parameters from a HF ``self_attn`` module.

    Returns ``None`` when the module does not match the expected layout.

    Head counts are read from module attributes when present (standard HF),
    or inferred from projection ``out_features`` when absent (e.g.
    ``Gemma4TextAttention`` which stores head count only in the config).
    """
    if not (hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj")):
        return None

    head_dim = getattr(module, "head_dim", None)
    if head_dim is None:
        return None
    head_dim = int(head_dim)

    num_heads = getattr(module, "num_heads", None) or getattr(module, "num_attention_heads", None)
    if num_heads is None:
        # Infer from q_proj output dimension (works even on meta device).
        # Fall back to weight.shape[0] for custom linears lacking out_features
        # (e.g. Gemma4ClippableLinear).
        q_proj = getattr(module, "q_proj", None)
        q_out = _proj_out_features(q_proj)
        if q_out is None:
            return None
        num_heads = q_out // head_dim
    num_heads = int(num_heads)

    num_kv_heads = getattr(module, "num_key_value_heads", None)
    if num_kv_heads is None:
        k_proj = getattr(module, "k_proj", None)
        k_out = _proj_out_features(k_proj)
        num_kv_heads = (k_out // head_dim) if k_out is not None else num_heads
    num_kv_heads = int(num_kv_heads)

    # Sliding-window attention: convert HF's window token count to TE's
    # (left_tokens, right_tokens) convention.  (-1, 0) = unbounded / global.
    sliding_window = getattr(module, "sliding_window", None)
    if sliding_window is not None and sliding_window > 0:
        # TE window_size[0] = number of tokens to the LEFT of current position.
        # A window of W tokens (inclusive of current) → W-1 to the left.
        te_window_size = (int(sliding_window) - 1, 0)
    else:
        te_window_size = (-1, 0)

    # Some models (e.g. Gemma4) store a pre-computed softmax scale as
    # ``module.scaling`` instead of using the standard head_dim**-0.5.
    softmax_scale = getattr(module, "scaling", None)
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    return {
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "window_size": te_window_size,
        "softmax_scale": float(softmax_scale),
    }


# ---------------------------------------------------------------------------
# TE module creation
# ---------------------------------------------------------------------------


def _create_te_dot_product_attention(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    window_size: tuple[int, int] = (-1, 0),
    softmax_scale: float | None = None,
) -> "transformer_engine.pytorch.attention.DotProductAttention":  # noqa: F821
    """Instantiate a TE ``DotProductAttention`` for the given attention shape."""
    from transformer_engine.pytorch.attention import DotProductAttention

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    return DotProductAttention(
        num_attention_heads=num_heads,
        kv_channels=(head_dim, head_dim),
        attn_mask_type="causal",
        qkv_format="bshd",
        softmax_scale=softmax_scale,
        num_gqa_groups=num_kv_heads,
        window_size=window_size,
    )


# ---------------------------------------------------------------------------
# Mask detection
# ---------------------------------------------------------------------------


def _detect_causal_mask(
    attn_mask: torch.Tensor,
    window_size: tuple[int, int],
) -> tuple[str, tuple[int, int]] | None:
    """Map a 4D HF additive attention mask to a TE (attn_mask_type, window_size) pair.

    HF's ``create_causal_mask`` and ``create_sliding_window_causal_mask`` always
    emit a 4D float mask even when the batch has no padding.  This causes the
    generic ``attn_mask is not None`` guard to fire for every sliding/full-attention
    layer and route every call to native SDPA instead of TE.

    This function detects the two structurally trivial cases:
    - Pure causal (lower-triangular 0 / -inf): returns ``("causal", (-1, 0))``.
    - Sliding-window causal: returns ``("causal", window_size)``.

    Returns ``None`` when the mask cannot be safely converted — e.g. it encodes
    per-sample padding, has an unexpected shape, or is not a float additive mask.
    The caller should fall back to native SDPA in that case.

    Detection is O(S) per call (two row reductions of length S_k) and transfers
    one combined validation result to the host:
    1. Upper-right corner scalar check: must be < -1e4 (causal-like).
    2. First-row visible-key count across all batch items must equal 1
       (each first query token can only attend to itself).
    3. Last-row visible-key count across all batch items must equal ``S``
       (full causal) or ``min(S, window_size[0]+1)`` (sliding window).
    Any deviation indicates padding or a non-standard mask structure.
    """
    logger.debug(
        "_detect_causal_mask: dtype=%s shape=%s",
        attn_mask.dtype,
        tuple(attn_mask.shape),
    )

    is_bool = attn_mask.dtype == torch.bool
    is_float = attn_mask.dtype in (torch.float16, torch.bfloat16, torch.float32)
    if not (is_bool or is_float):
        logger.debug("_detect_causal_mask: unsupported dtype %s → None", attn_mask.dtype)
        return None
    if attn_mask.ndim != 4:
        logger.debug("_detect_causal_mask: ndim=%d (need 4) → None", attn_mask.ndim)
        return None

    sq, sk = attn_mask.shape[2], attn_mask.shape[3]
    if sq != sk:
        logger.debug("_detect_causal_mask: sq=%d != sk=%d → None", sq, sk)
        return None

    S = sq

    # Scalar guard: upper-right corner must indicate "masked" (causal structure).
    # bool mask: False = masked; float mask: < -1e4 = masked.
    # Count visible keys per row for all batch items.
    # bool: True = visible; float: > -1e4 = visible.
    first_row = attn_mask[:, 0, 0, :]
    last_row = attn_mask[:, 0, -1, :]
    if is_bool:
        visible_first = first_row.sum(dim=-1)
        visible_last = last_row.sum(dim=-1)
    else:
        visible_first = (first_row > -1e4).sum(dim=-1)
        visible_last = (last_row > -1e4).sum(dim=-1)

    expected_last = S if window_size[0] < 0 else min(S, window_size[0] + 1)
    corner = attn_mask[0, 0, 0, -1]
    diag = attn_mask[0, 0, 0, 0]
    if is_bool:
        corner_masked = ~corner
        diag_visible = diag
    else:
        corner_masked = corner < -1e4
        diag_visible = diag > -1e4
    first_valid = (visible_first == 1).all()
    last_valid = (visible_last == expected_last).all()
    mask_valid = corner_masked & diag_visible & first_valid & last_valid

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "_detect_causal_mask: S=%d visible_first=%s visible_last=%s window_size=%s",
            S,
            visible_first.tolist(),
            visible_last.tolist(),
            window_size,
        )

    if not mask_valid.item():
        logger.debug("_detect_causal_mask: structural validation failed → None")
        return None

    if window_size[0] < 0:
        return "causal", (-1, 0)
    return "causal", window_size


# ---------------------------------------------------------------------------
# SDPA replacement
# ---------------------------------------------------------------------------


def _maybe_log_stats() -> None:
    """Periodically emit the dispatch counters when auto-logging is enabled."""
    if _STATS_LOG_EVERY <= 0:
        return
    total = _TE_STATS["te_hits"] + _TE_STATS["fallback_mask"] + _TE_STATS["fallback_scale_mismatch"]
    if total > 0 and total % _STATS_LOG_EVERY == 0:
        logger.info(
            "te_sdpa dispatch stats (first %d calls): te_hits=%d fallback_mask=%d fallback_scale_mismatch=%d",
            total,
            _TE_STATS["te_hits"],
            _TE_STATS["fallback_mask"],
            _TE_STATS["fallback_scale_mismatch"],
        )


def _make_te_sdpa(
    te_module: torch.nn.Module,
    num_heads: int,
    num_kv_heads: int,
    original_sdpa,
    window_size: tuple[int, int] = (-1, 0),
    softmax_scale: float | None = None,
) -> Any:
    """Return a callable that replaces ``F.scaled_dot_product_attention``.

    The replacement:
    - Transposes Q/K/V from HF's ``[B, H, S, D]`` to TE's ``[B, S, H, D]``.
    - Undoes ``repeat_kv`` when TE can handle GQA natively.
    - Falls back to ``original_sdpa`` for non-trivial ``attn_mask`` inputs.
    - Falls back when the caller passes an explicit ``scale`` that disagrees
      with the ``softmax_scale`` captured at TE-module creation time (TE
      freezes that value at construction; trying to override it silently
      would change numerics).
    - Transposes the TE output back to ``[B, H, S, D]`` before returning.
    """

    def te_sdpa(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        **_unused: Any,
    ) -> torch.Tensor:
        global _SCALE_MISMATCH_WARNED

        # Runtime ``scale`` must match the value baked into ``te_module``;
        # otherwise TE would silently use the baked-in one and produce wrong
        # numerics.  Tolerate tiny float drift.
        scale_mismatch = (
            scale is not None and softmax_scale is not None and abs(float(scale) - float(softmax_scale)) > 1e-6
        )
        if scale_mismatch:
            if not _SCALE_MISMATCH_WARNED:
                logger.warning(
                    "te_sdpa: caller passed scale=%s but TE module was built with softmax_scale=%s; "
                    "falling back to native SDPA to preserve numerics. (This warning is emitted once.)",
                    scale,
                    softmax_scale,
                )
                _SCALE_MISMATCH_WARNED = True
            _TE_STATS["fallback_scale_mismatch"] += 1
            _maybe_log_stats()
            return original_sdpa(
                query,
                key,
                value,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        # Try to convert HF's explicit causal / sliding-window mask to TE parameters
        # so we can run the TE kernel instead of falling back to native SDPA.
        # _detect_causal_mask returns None when the mask encodes padding or is
        # otherwise non-trivial; in that case we fall back as before.
        if attn_mask is not None:
            converted = _detect_causal_mask(attn_mask, window_size)
            if converted is None:
                logger.debug("TE attention: non-trivial attn_mask — falling back to native SDPA.")
                _TE_STATS["fallback_mask"] += 1
                _maybe_log_stats()
                return original_sdpa(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                )
            mask_type, effective_window = converted
            logger.debug(
                "te_sdpa: converted attn_mask → mask_type=%s window=%s q_shape=%s",
                mask_type,
                effective_window,
                query.shape,
            )
        else:
            mask_type = "causal" if is_causal else "no_mask"
            # TE requires window_size=(-1, -1) for no_mask; sliding window only applies to causal.
            effective_window = window_size if mask_type == "causal" else (-1, -1)

        # HF passes Q/K/V in [B, H, S, D]; TE expects [B, S, H, D].
        q = query.transpose(1, 2).contiguous()
        k = key.transpose(1, 2).contiguous()
        v = value.transpose(1, 2).contiguous()

        # If repeat_kv was applied (enable_gqa=False with a GQA model), undo it:
        # after transpose k/v are [B, S, H, D] but TE needs [B, S, Hkv, D].
        if not enable_gqa and num_kv_heads < num_heads and k.shape[2] > num_kv_heads:
            step = k.shape[2] // num_kv_heads
            k = k[:, :, ::step, :].contiguous()
            v = v[:, :, ::step, :].contiguous()

        logger.debug(
            "te_sdpa: mask_type=%s effective_window=%s q=%s k=%s",
            mask_type,
            effective_window,
            q.shape,
            k.shape,
        )
        # Set the current CUDA device to match the inputs so that any internal
        # scratch allocations inside TE land on the correct GPU (device_map="auto").
        # Skip the context manager on CPU (torch.cuda.device raises for non-CUDA devices).
        ctx = torch.cuda.device(q.device) if q.device.type == "cuda" else contextlib.nullcontext()
        with ctx:
            out = te_module(q, k, v, attn_mask_type=mask_type, window_size=effective_window)

        _TE_STATS["te_hits"] += 1
        _maybe_log_stats()

        # TE returns [B, S, H, D]; transpose back to HF's [B, H, S, D].
        return out.transpose(1, 2).contiguous()

    return te_sdpa


# ---------------------------------------------------------------------------
# Forward patching
# ---------------------------------------------------------------------------


def _patch_module_forward(module: torch.nn.Module, te_sdpa) -> None:
    """Shadow ``module.forward`` with a version that uses TE for SDPA."""
    # Capture the *class-level* (unbound) forward method so that we call the
    # original implementation rather than our own patched instance method.
    original_forward = type(module).forward

    def patched_forward(inner_self, *args, **kwargs):
        orig = torch.nn.functional.scaled_dot_product_attention
        torch.nn.functional.scaled_dot_product_attention = te_sdpa
        try:
            return original_forward(inner_self, *args, **kwargs)
        finally:
            torch.nn.functional.scaled_dot_product_attention = orig

    # Bind to instance to shadow the class method lookup.
    module.forward = types.MethodType(patched_forward, module)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def inject_te_attention_into_module(module: torch.nn.Module) -> bool:
    """Inject TE attention into a single HF ``self_attn`` module.

    Returns ``True`` on success, ``False`` when the module does not match
    the expected layout.
    """
    params = _infer_attn_params(module)
    if params is None:
        return False

    # Capture the original SDPA *before* any patching so the fallback path
    # inside ``te_sdpa`` always reaches the real kernel.
    original_sdpa = F.scaled_dot_product_attention

    te_module = _create_te_dot_product_attention(**params)
    te_sdpa = _make_te_sdpa(
        te_module=te_module,
        num_heads=params["num_heads"],
        num_kv_heads=params["num_kv_heads"],
        original_sdpa=original_sdpa,
        window_size=params["window_size"],
        softmax_scale=params["softmax_scale"],
    )
    _patch_module_forward(module, te_sdpa)

    # Store as a plain instance attribute (bypass nn.Module.__setattr__) so
    # that the TE module is NOT registered as a child submodule.  This keeps
    # attn_module._extra_state out of the model's state_dict, preventing DCP
    # from demanding it when loading a pretrained HF checkpoint.
    # hasattr / getattr still work because the attribute lives in __dict__.
    object.__setattr__(module, _TE_MODULE_ATTR, te_module)
    return True


def inject_te_attention(model: torch.nn.Module) -> None:
    """Walk *model* and inject TE attention into all compatible ``self_attn`` modules.

    Skips modules that already carry ``attn_module`` (i.e. custom models or
    modules that were already patched).  Sets ``model._te_attention_injected``
    on success so that :func:`_uses_te_attention` can short-circuit the walk.
    """
    injected = 0
    for name, module in model.named_modules():
        if not name.endswith("self_attn"):
            continue
        if hasattr(module, _TE_MODULE_ATTR):
            # Custom model or already patched.
            continue
        if inject_te_attention_into_module(module):
            injected += 1
            logger.debug("Injected TE attention into %s", name)

    if injected > 0:
        logger.info("Injected TE DotProductAttention into %d self_attn module(s).", injected)
        setattr(model, _TE_MODEL_FLAG, True)
    else:
        logger.warning(
            "inject_te_attention: no compatible self_attn modules found. "
            "The model may not use the standard HF q_proj/k_proj/v_proj layout."
        )
