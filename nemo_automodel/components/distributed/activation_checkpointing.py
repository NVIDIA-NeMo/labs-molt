# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Selective activation checkpointing core.

TorchTitan-style selective activation checkpointing: the policy decides, per op,
whether to save or recompute an activation, saving the expensive ops (attention,
half of the matmuls, comm collectives) while recomputing the cheap ones.

This module holds the parts of the AC implementation that do not depend on the
rest of ``parallelizer.py`` (notably the heavy, transformers-aware
``_extract_model_layers``). ``parallelizer.py`` imports from here -- never the
other way around -- so the dependency stays one-directional and the central
parallelizer file stays small.
"""

import functools
import logging
import os
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from typing import List

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    CheckpointWrapper,
    checkpoint_wrapper,
)
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from nemo_automodel.shared.import_utils import get_torch_version, safe_import

logger = logging.getLogger(__name__)

_TORCH_PROFILER_SAC_IGNORE_MIN_VERSION = (2, 13)


def unwrap_checkpoint_wrapper(module: nn.Module) -> nn.Module:
    """Return the activation-checkpointed module, or the input module if it is not wrapped.

    Args:
        module: Module that may have been wrapped by ``checkpoint_wrapper``.

    Returns:
        The inner checkpointed module when present, otherwise ``module``.
    """
    return getattr(module, "_checkpoint_wrapped_module", module)


def _resolve_torch_op(dotted_path: str):
    """Resolve a dotted ``torch.ops`` path, defaulting to the ``default`` overload."""
    if dotted_path.count(".") == 1:
        dotted_path = f"{dotted_path}.default"
    return _resolve_op_attr(torch.ops, dotted_path)


def _resolve_op_attr(root: object, dotted_path: str):
    """Resolve a dotted attribute path from ``root``, or ``None`` if any part is absent.

    Used for ops that live outside ``torch.ops`` (higher-order ops, optional
    custom backends such as DeepEP/HybridEP). Missing namespaces/ops raise
    ``AttributeError`` on access, so they are swallowed and reported as ``None``.
    """
    obj = root
    try:
        for part in dotted_path.split("."):
            obj = getattr(obj, part)
    except AttributeError:
        return None
    return obj


def _existing_ops(*ops):
    return frozenset(op for op in ops if op is not None)


# Matmul ops whose activations alternate between save and recompute (every other
# one is saved). Following TorchTitan, plain ``mm``/``linear`` alternate;
# ``addmm``/``bmm`` stay in the always-save set built below. The grouped-GEMM
# variants are the dominant compute in expert-parallel MoE blocks (custom
# ``torch._grouped_mm`` kernels), so they alternate too -- otherwise selective AC
# would recompute every expert GEMM, matching full checkpointing and giving no
# speedup while still paying the policy overhead.
_SELECTIVE_AC_MATMUL_OPS = _existing_ops(
    _resolve_torch_op("aten.mm"),
    _resolve_torch_op("aten.linear"),
    _resolve_torch_op("aten._grouped_mm"),
    _resolve_torch_op("aten._scaled_grouped_mm"),
)


def _default_compute_intensive_ops() -> tuple:
    """Compute-intensive aten ops from PyTorch's partitioner, or ``()`` if unavailable.

    Mirrors TorchTitan: seeding from PyTorch's own ``compute_intensive_ops`` list
    keeps the save-set in sync with upstream rather than relying on a frozen,
    hand-maintained list. ``torch._functorch.partitioners`` is a private API, so
    any failure falls back to the curated supplement in
    :func:`_build_selective_ac_save_ops`.
    """
    try:
        from torch._functorch.partitioners import get_default_op_list

        return tuple(op.default for op in get_default_op_list().compute_intensive_ops)
    except (ImportError, AttributeError):
        return ()


def _ffpa_forward_ops() -> tuple:
    """FFPA forward ops (dense + varlen); import the CuTeDSL kernel so they register first, else ``()``.

    Their ops register only on that import, which otherwise lands after this
    save-set is frozen -- hence the eager import rather than a bare resolve.
    """
    try:
        import ffpa_attn.cute  # noqa: F401
    except Exception:
        return ()
    return (
        _resolve_op_attr(torch.ops, "ffpa_attn._fwd_cute.default"),
        _resolve_op_attr(torch.ops, "ffpa_attn._varlen_fwd_cute.default"),
    )


def _build_selective_ac_save_ops() -> frozenset:
    """Build the set of ops whose activations are always saved under selective AC.

    The set is seeded from PyTorch's compute-intensive op list and supplemented
    with attention variants, low-precision/reduction ops, the compiled HOP, and
    communication collectives whose outputs are expensive to recompute.
    """
    save_ops = set(_default_compute_intensive_ops())

    # Compute ops the partitioner list may not classify as compute-intensive.
    compute_ops = _existing_ops(
        *list(
            map(
                _resolve_torch_op,
                [
                    "aten.mm",
                    "aten.addmm",
                    "aten.bmm",
                    "aten.linear",
                    "aten._scaled_mm",
                    "aten._scaled_dot_product_cudnn_attention",
                    "aten._scaled_dot_product_efficient_attention",
                    "aten._scaled_dot_product_flash_attention",
                    "aten._scaled_dot_product_flash_attention_for_cpu",
                    "aten._scaled_dot_product_fused_attention_overrideable",
                    "aten.scaled_dot_product_attention",
                    "aten._flex_attention",
                    # topk is saved to keep MoE expert assignments stable across recompute;
                    # max is saved for low-precision scaling factors.
                    "aten.topk",
                    "aten.max",
                ],
            )
        ),
        # FlexAttention HOP and the inductor compiled-graph HOP (present only when
        # torch.compile is used); custom torch_attn varlen backend.
        _resolve_op_attr(torch, "_higher_order_ops.flex_attention"),
        _resolve_op_attr(torch, "_higher_order_ops.inductor_compiled_code"),
        _resolve_op_attr(torch.ops, "torch_attn._varlen_attn.default"),
        # FFPA forward ops (head_dim=512 Gemma4 full attention).
        *_ffpa_forward_ops(),
    )

    # Communication ops whose outputs should be saved to avoid re-communication.
    comm_ops = _existing_ops(
        *list(
            map(
                _resolve_torch_op,
                [
                    "aten.all_to_all_single",
                    "aten.reduce_scatter_tensor",
                    "_c10d_functional.all_to_all_single",
                    "_c10d_functional.reduce_scatter_tensor",
                ],
            )
        ),
        # Optional expert-parallel comm backends.
        _resolve_op_attr(torch.ops, "deepep.dispatch.default"),
        _resolve_op_attr(torch.ops, "deepep.combine.default"),
        _resolve_op_attr(torch.ops, "hybridep.dispatch.default"),
        _resolve_op_attr(torch.ops, "hybridep.combine.default"),
    )

    save_ops.update(compute_ops)
    save_ops.update(comm_ops)
    return frozenset(save_ops)


_SELECTIVE_AC_MUST_SAVE_OPS = _build_selective_ac_save_ops()

_SELECTIVE_AC_TO_COPY_OP = _resolve_torch_op("aten._to_copy")


def is_selective_activation_checkpointing(activation_checkpointing: object) -> bool:
    """Return whether the config value selects selective activation checkpointing.

    Args:
        activation_checkpointing: The configured value (bool or string such as
            ``"selective"``/``"full"``).

    Returns:
        bool: ``True`` only for the string ``"selective"`` (case- and
        hyphen/underscore-insensitive).
    """
    return (
        isinstance(activation_checkpointing, str) and activation_checkpointing.lower().replace("-", "_") == "selective"
    )


def _is_cuda_to_cpu_copy(func, args, kwargs) -> bool:
    if func != _SELECTIVE_AC_TO_COPY_OP or not args:
        return False
    tensor = args[0]
    src_device = getattr(tensor, "device", None)
    target_device = kwargs.get("device")
    if target_device is None:
        return False
    try:
        target_device = torch.device(target_device)
    except (TypeError, RuntimeError):
        return False
    return getattr(src_device, "type", None) == "cuda" and target_device.type == "cpu"


# Opt-in diagnostics: set NEMO_SELECTIVE_AC_TRACE=1 to log, once per unique op,
# whether selective AC saves or recomputes it. Useful for confirming that a
# model's expensive ops (e.g. expert grouped-GEMMs, comm collectives) are
# actually saved rather than silently recomputed.
_SELECTIVE_AC_TRACE = os.environ.get("NEMO_SELECTIVE_AC_TRACE", "0").lower() not in ("0", "", "false", "no")
_SELECTIVE_AC_TRACE_SEEN: set[str] = set()


def _maybe_trace_selective_ac_decision(func, decision, is_alternating: bool, *, is_recompute: bool) -> None:
    """Log a selective-AC decision once per op (no-op unless tracing is enabled).

    Args:
        func: The op the policy was queried about.
        decision: The ``CheckpointPolicy`` the policy returned for ``func``.
        is_alternating: Whether ``func`` is an alternating-save matmul op.
        is_recompute: Whether the policy was queried during the recompute pass;
            decisions are only logged on the forward pass to avoid duplicates.
    """
    if not _SELECTIVE_AC_TRACE or is_recompute:
        return
    key = str(func)
    if key in _SELECTIVE_AC_TRACE_SEEN:
        return
    _SELECTIVE_AC_TRACE_SEEN.add(key)
    if is_alternating:
        verdict = "ALTERNATE (save/recompute every other call)"
    elif decision == CheckpointPolicy.MUST_SAVE:
        verdict = "SAVE"
    else:
        verdict = "RECOMPUTE"
    logger.info("[selective-ac] %s -> %s", key, verdict)


def ignore_sac_ops(ops: list[object | None]) -> None:
    """Add available operators to PyTorch's selective-AC ignore set.

    Args:
        ops: Operators that should execute outside SAC replay accounting.
            Entries may be ``None`` when an optional operator is unavailable.
    """
    sac_ignored = getattr(torch.utils.checkpoint, "SAC_IGNORED_OPS", None)
    if sac_ignored is not None:
        sac_ignored.update(op for op in ops if op is not None)


def ensure_profiler_ops_sac_ignored() -> None:
    """Keep ``torch.ops.profiler`` record-function ops out of SAC's op replay.

    torch 2.13's FSDP2 runs its pre/post-forward hooks under
    ``torch.autograd.profiler.record_function``, which emits dispatchable
    ``torch.ops.profiler._record_function_*`` ops. When an FSDP module boundary
    sits inside a selective-activation-checkpointed region (e.g. MoE experts
    sharded separately inside a checkpointed decoder block), those hooks fire a
    different number of times during the backward recompute than during the
    forward. SAC replays the forward op stream by per-op invocation index, so
    the extra profiler op shifts the stream and training fails with
    ``profiler._record_function_enter_new.default invocation index N
    encountered during backward but not found in storage``.

    Range ops carry no tensors SAC could cache or restore; adding them to
    ``SAC_IGNORED_OPS`` only removes them from the replay accounting (they
    still execute). No-op before torch 2.13 and on torch builds without
    ``SAC_IGNORED_OPS`` or the profiler op namespace.
    """
    if get_torch_version().release < _TORCH_PROFILER_SAC_IGNORE_MIN_VERSION:
        return

    profiler_ops = getattr(torch.ops, "profiler", None)
    if profiler_ops is None:
        return
    ops_to_ignore = []
    for packet_name in ("_record_function_enter", "_record_function_enter_new", "_record_function_exit"):
        packet = getattr(profiler_ops, packet_name, None)
        if packet is None:
            continue
        for overload_name in packet.overloads():
            ops_to_ignore.append(getattr(packet, overload_name))
    ignore_sac_ops(ops_to_ignore)


def ensure_fsdp_ops_sac_ignored() -> None:
    """Keep FSDP2 parameter-lifecycle ops out of SAC's saved-op replay.

    FSDP2 can prefetch an inner module's parameters before a checkpointed
    forward, while backward-time recomputation must unshard them from inside
    the checkpointed region. Its copy-in/copy-out and c10d all-gather ops are
    runtime parameter management, not model activations, and must execute
    normally when needed instead of being indexed against the forward's
    selective-AC op stream.
    """
    # FSDP's all-gather copy-out allocates per-parameter outputs and views the
    # flat communication buffer. When forward prefetch has already unsharded
    # the parameters, these setup ops occur only during recomputation. They do
    # not produce model activations: the FSDP copy ops populate the allocations.
    #
    # Parameters sharded on a non-zero dim (``fsdp_placement.dim != 0``, e.g.
    # MoE expert weights under an expert-parallel FSDP shard axis) take a second
    # copy-out path in ``foreach_all_gather_copy_out``: ``torch.chunk`` (which
    # dispatches as ``aten.split.Tensor``) followed by ``torch.cat(..., out=)``
    # into the unsharded parameter. Those two ops are parameter management as
    # well and otherwise fail SAC replay with ``aten.split.Tensor encountered
    # during backward but not found in storage`` whenever the expert unit is
    # resharded between forward and recompute (per-microbatch gradient sync).
    ignore_sac_ops(
        list(
            map(
                _resolve_torch_op,
                [
                    "fsdp.all_gather_copy_in",
                    "fsdp.split_with_sizes_copy",
                    "fsdp.chunk_cat",
                    "fsdp.copy_",
                    "c10d._allgather_base_",
                    "aten.empty.memory_format",
                    "aten.empty_like",
                    "aten.view",
                    "aten.split.Tensor",
                    "aten.cat.out",
                ],
            )
        )
    )


def make_selective_checkpoint_context_fn():
    """Build a TorchTitan-style selective activation checkpointing context."""
    ensure_profiler_ops_sac_ignored()
    ensure_fsdp_ops_sac_ignored()

    def selective_checkpointing_context_fn():
        # Count matmuls separately for the forward and recompute passes. torch
        # calls ``context_fn`` once per checkpointed region, so a single shared
        # counter would continue from the forward count into recompute and flip
        # the save/recompute parity whenever the region has an odd number of
        # matmuls. Keying on ``ctx.is_recompute`` resets each pass to 0 so the
        # same matmul gets the same decision in both passes.
        mm_counts = {False: 0, True: 0}

        def selective_checkpointing_policy(ctx, func, *args, **kwargs):
            is_alternating = func in _SELECTIVE_AC_MATMUL_OPS
            if is_alternating:
                mm_counts[ctx.is_recompute] += 1
                decision = (
                    CheckpointPolicy.PREFER_RECOMPUTE
                    if mm_counts[ctx.is_recompute] % 2 == 0
                    else CheckpointPolicy.MUST_SAVE
                )
            elif func in _SELECTIVE_AC_MUST_SAVE_OPS or _is_cuda_to_cpu_copy(func, args, kwargs):
                decision = CheckpointPolicy.MUST_SAVE
            else:
                decision = CheckpointPolicy.PREFER_RECOMPUTE
            _maybe_trace_selective_ac_decision(func, decision, is_alternating, is_recompute=ctx.is_recompute)
            return decision

        return create_selective_checkpoint_contexts(selective_checkpointing_policy)

    return selective_checkpointing_context_fn


# Marker set on whole-block selective-AC wrappers so the per-layer compile step
# compiles the wrapper itself (compile OUTER, SAC INNER) instead of unwrapping
# to the inner decoder layer. Compiling outer lets AOT autograd's partitioner
# read the SAC recompute tags; compiling inner would hide every aten op behind a
# single compiled HOP and collapse selective recompute into full recompute.
SELECTIVE_AC_WRAPPER_FLAG = "_nemo_selective_ac"


def _disable_dynamo_lru_cache() -> None:
    """Best-effort disable of TorchDynamo's LRU cache for selective AC + compile.

    With multiple pipeline microbatches, dynamo may compile a second graph with
    dynamic shapes and then select it over the static graph whose compiled-HOP
    output SAC cached for microbatch 0, tripping a missing-symint assertion.
    Selecting graphs in insertion order avoids this. Mirrors TorchTitan. The
    underlying API is private, so failures are swallowed.
    """
    try:
        torch._C._dynamo.eval_frame._set_lru_cache(False)
    except (AttributeError, RuntimeError):
        logger.debug("Could not disable dynamo LRU cache for selective AC + compile.", exc_info=True)


@contextmanager
def _restore_sdpa_state(sdpa: Callable, backends: list[SDPBackend]):
    """Temporarily restore the SDPA callable and backend set captured during forward."""
    backward_sdpa = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = sdpa
    try:
        with sdpa_kernel(backends):
            yield
    finally:
        F.scaled_dot_product_attention = backward_sdpa


def sdpa_backend_snapshot_context_fn() -> tuple[AbstractContextManager, AbstractContextManager]:
    """Snapshot the ambient SDPA state and restore it on checkpoint recompute.

    A ``context_fn`` for non-reentrant ``checkpoint_wrapper``: torch's
    non-reentrant checkpoint invokes it at region entry on every checkpointed
    forward, so the state read here is exactly what the forward runs under.
    Both the enabled backend set and ``F.scaled_dot_product_attention`` are
    restored during recompute. The callable matters for context parallelism:
    a VLM vision tower temporarily suspends CP's ring-SDPA monkeypatch, and
    checkpoint recompute occurs after that forward-only suspension has exited.
    Replaying under the captured callable keeps bidirectional vision attention
    local while restoring the backward-time CP dispatcher after recompute.

    Re-pinning the forward-time backend set also prevents checkpoint metadata
    mismatches when ambient backend forcing (an ``sdpa_kernel`` pin or
    module-level backend toggling) is active at forward time but does not span
    recompute. State toggled *inside* the checkpointed region between attention
    calls is not captured because the snapshot is taken once at region entry.

    Returns:
        ``(forward_ctx, recompute_ctx)``: a no-op context for the checkpoint
        forward, and a context restoring the captured SDPA callable and backend
        set for the backward-time recompute.
    """
    captured_sdpa = F.scaled_dot_product_attention
    captured_backends = [
        backend
        for enabled, backend in (
            (torch.backends.cuda.flash_sdp_enabled(), SDPBackend.FLASH_ATTENTION),
            (torch.backends.cuda.mem_efficient_sdp_enabled(), SDPBackend.EFFICIENT_ATTENTION),
            (torch.backends.cuda.cudnn_sdp_enabled(), SDPBackend.CUDNN_ATTENTION),
            (torch.backends.cuda.math_sdp_enabled(), SDPBackend.MATH),
        )
        if enabled
    ]
    return nullcontext(), _restore_sdpa_state(captured_sdpa, captured_backends)


def _get_transformer_engine_attention_backend_cache() -> dict | None:
    """Return Transformer Engine's attention-backend cache when available."""
    available, dot_product_attention = safe_import(
        "transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention"
    )
    if not available:
        return None
    cache = getattr(dot_product_attention, "_attention_backends", None)
    return cache if isinstance(cache, dict) else None


@contextmanager
def _restore_transformer_engine_attention_backend_cache(
    cache: dict,
    captured_cache: dict,
    recompute_context: AbstractContextManager,
):
    """Restore the forward-entry TE attention cache only for recomputation."""
    backward_cache = dict(cache)
    cache.clear()
    cache.update(captured_cache)
    try:
        with recompute_context:
            yield
    finally:
        cache.clear()
        cache.update(backward_cache)


def transformer_engine_attention_backend_snapshot_context_fn(
    context_fn: Callable[[], tuple[AbstractContextManager, AbstractContextManager]] | None = None,
) -> tuple[AbstractContextManager, AbstractContextManager]:
    """Snapshot Transformer Engine's attention-backend cache for checkpoint replay.

    Transformer Engine caches the parameters and result of attention-backend
    selection in module-global state. A checkpointed forward can populate that
    cache, so backward-time recomputation would otherwise enter a different
    parameter-comparison branch and dispatch a different aten op sequence. The
    cache is restored to its forward-entry state for recomputation, then reset
    to the state owned by the surrounding backward pass.

    Args:
        context_fn: Optional checkpoint context factory to compose inside the
            cache restoration, such as a selective activation-checkpoint policy.

    Returns:
        ``(forward_ctx, recompute_ctx)`` with the supplied forward context and a
        recompute context that restores the captured Transformer Engine cache.
    """
    forward_context, recompute_context = context_fn() if context_fn is not None else (nullcontext(), nullcontext())
    cache = _get_transformer_engine_attention_backend_cache()
    if cache is None:
        return forward_context, recompute_context
    return forward_context, _restore_transformer_engine_attention_backend_cache(
        cache,
        dict(cache),
        recompute_context,
    )


def _registered_child_name(module: nn.Module, attr: str, child: nn.Module) -> str | None:
    """Return the registered name for a child reached through an attribute."""
    if module._modules.get(attr) is child:
        return attr
    for child_name, registered_child in module._modules.items():
        if registered_child is child:
            return child_name
    return None


def _wrap_first_existing_attr(
    module: nn.Module,
    attr_names: tuple[str, ...],
    *,
    skip: bool = False,
    context_fn: Callable[[], tuple[AbstractContextManager, AbstractContextManager]] | None = None,
) -> int:
    """Checkpoint-wrap the first matching registered child attr on ``module``."""
    if skip:
        return 0
    checkpoint_kwargs = {} if context_fn is None else {"context_fn": context_fn}
    for attr in attr_names:
        child = getattr(module, attr, None)
        if isinstance(child, nn.Module):
            child_name = _registered_child_name(module, attr, child)
            if child_name is None:
                continue
            if hasattr(child, "_checkpoint_wrapped_module"):
                return 0
            setattr(module, child_name, checkpoint_wrapper(child, **checkpoint_kwargs))
            return 1
    return 0


def apply_submodule_checkpointing(
    layers: List[nn.Module],
    has_kv_sharing: bool,
    *,
    context_fn: Callable[[], tuple[AbstractContextManager, AbstractContextManager]] | None = (
        sdpa_backend_snapshot_context_fn
    ),
) -> None:
    """Wrap a transformer block's sub-modules with ``checkpoint_wrapper``.

    This is the sub-module granularity path used both as the default
    (non-compile) behavior and as the fallback for selective activation
    checkpointing on KV-shared models, which cannot checkpoint the whole block.

    ``self_attn`` is skipped for KV-shared models: recomputing attention during
    backward would double-write to the ``DynamicCache``, corrupting the K/V
    entries that later shared layers depend on.

    Args:
        layers: Transformer decoder layers to wrap (mutated in place).
        has_kv_sharing: Whether the model reuses K/V across layers via the cache.
        context_fn: Factory returning ``(forward_ctx, recompute_ctx)`` for the
            attention and MLP checkpoint wrappers. Defaults to restoring the
            forward-time SDPA state; pass ``None`` to disable it. Norm wrappers
            stay plain because they dispatch no SDPA.
    """
    wrapped_counts: dict[str, int] = {
        "mlp": 0,
        "attention": 0,
        "pre_norm": 0,
        "post_norm": 0,
        "mot": 0,
    }
    for layer in layers:
        wrapped_counts["mlp"] += _wrap_first_existing_attr(layer, ("mlp", "feed_forward", "ffn"), context_fn=context_fn)
        wrapped_counts["attention"] += _wrap_first_existing_attr(
            layer,
            # "linear_attn" covers hybrid linear-attention blocks (e.g. Qwen3-Next /
            # Qwen3.5 Gated DeltaNet layers), which name their mixer "linear_attn"
            # rather than "self_attn". Without it those layers are never wrapped and
            # long-context training OOMs. A hybrid block has either "self_attn" or
            # "linear_attn" (never both), so first-match wrapping stays unambiguous.
            ("self_attn", "attention", "attn", "linear_attn"),
            skip=has_kv_sharing,
            context_fn=context_fn,
        )
        wrapped_counts["pre_norm"] += _wrap_first_existing_attr(
            layer,
            ("input_layernorm", "attention_norm", "layer_norm1", "norm1"),
        )
        wrapped_counts["post_norm"] += _wrap_first_existing_attr(
            layer,
            ("post_attention_layernorm", "ffn_norm", "layer_norm2", "norm2"),
        )

        # MoT (mixture-of-transformers) sibling submodules -- present in BAGEL's
        # Qwen2MoTDecoderLayer for the generation expert. mlp_moe_gen is a full
        # Qwen2MLP duplicate (same size as mlp), so omitting it from AC roughly
        # doubles per-layer activation memory in Stage-2 BAGEL training.
        if hasattr(layer, "mlp_moe_gen"):
            layer.mlp_moe_gen = checkpoint_wrapper(layer.mlp_moe_gen)  # type: ignore
            wrapped_counts["mot"] += 1
        if hasattr(layer, "input_layernorm_moe_gen"):
            layer.input_layernorm_moe_gen = checkpoint_wrapper(layer.input_layernorm_moe_gen)  # type: ignore
            wrapped_counts["mot"] += 1
        if hasattr(layer, "post_attention_layernorm_moe_gen"):
            layer.post_attention_layernorm_moe_gen = checkpoint_wrapper(layer.post_attention_layernorm_moe_gen)  # type: ignore
            wrapped_counts["mot"] += 1
    logger.info("Applied submodule activation checkpointing to %d layers: %s", len(layers), wrapped_counts)


def _replace_child_module(root: nn.Module, target: nn.Module, replacement: nn.Module) -> bool:
    """Replace ``target`` with ``replacement`` in ``root``'s module tree."""
    for name, child in root.named_children():
        if child is target:
            if isinstance(root, nn.ModuleList):
                root[int(name)] = replacement
            elif isinstance(root, nn.ModuleDict):
                root[name] = replacement
            else:
                setattr(root, name, replacement)
            return True
        if _replace_child_module(child, target, replacement):
            return True
    return False


def apply_full_layer_checkpointing_to_layers(model: nn.Module, layers: List[nn.Module]) -> None:
    """Wrap transformer layers with non-reentrant activation checkpointing."""
    context_fn = functools.partial(
        transformer_engine_attention_backend_snapshot_context_fn,
        sdpa_backend_snapshot_context_fn,
    )
    for index, layer in enumerate(layers):
        if isinstance(layer, CheckpointWrapper):
            continue
        wrapped_layer = checkpoint_wrapper(
            layer,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            context_fn=context_fn,
            preserve_rng_state=True,
        )
        if not _replace_child_module(model, layer, wrapped_layer):
            raise RuntimeError(f"Could not replace layer {index} with an activation checkpoint wrapper.")
        layers[index] = wrapped_layer


def detect_kv_sharing_and_maybe_disable_cache(model: nn.Module) -> bool:
    """Detect KV-sharing and disable ``use_cache`` for non-KV-shared models.

    Models with KV-shared layers (e.g. Gemma4 2B/4B) pass K/V from earlier
    layers to later layers through the ``DynamicCache``; disabling the cache
    breaks that dependency, so ``use_cache`` is left untouched for them.

    Returns:
        bool: Whether the model uses KV-sharing.
    """
    config = getattr(model, "config", None)
    text_cfg = getattr(config, "text_config", None) or config
    has_kv_sharing = getattr(text_cfg, "num_kv_shared_layers", 0) > 0
    if not has_kv_sharing and config is not None:
        # Composite (e.g. VLM) configs carry per-modality sub-configs, and the
        # text model reads ``text_config.use_cache`` rather than the composite
        # value. Leaving a sub-config cache enabled keeps a DynamicCache alive
        # under checkpointing, so self-attention appends K/V twice (forward and
        # recompute) and backward fails with a CheckpointError metadata
        # mismatch. Disable the cache on the composite config and on every
        # sub-config that exposes ``use_cache``.
        sub_config_names = getattr(type(config), "sub_configs", None) or {"text_config": None}
        sub_configs = (getattr(config, name, None) for name in sub_config_names)
        for cfg in (config, *sub_configs):
            if cfg is None or (cfg is not config and not hasattr(cfg, "use_cache")):
                continue
            if getattr(cfg, "use_cache", None) is not False:
                try:
                    cfg.use_cache = False
                except Exception:
                    pass
    return has_kv_sharing


def apply_selective_checkpointing_to_layers(
    model: nn.Module,
    layers: List[nn.Module],
    has_kv_sharing: bool,
    *,
    enable_compile: bool = False,
) -> None:
    """Wrap whole transformer blocks with the selective-AC policy.

    KV-shared models cannot checkpoint attention through the ``DynamicCache``,
    so they fall back to sub-module checkpointing. ``layers`` is mutated in
    place so callers that retain the list (e.g. for subsequent FSDP sharding)
    see the wrapped modules. Works without FSDP/distributed, so it is shared by
    the FSDP2 strategy and the single-GPU path.
    """
    if has_kv_sharing:
        logger.warning(
            "Selective activation checkpointing is not supported for KV-shared models; "
            "falling back to sub-module activation checkpointing."
        )
        apply_submodule_checkpointing(layers, has_kv_sharing)
        return

    # With compile, the per-layer compile step compiles these wrappers OUTER so
    # the SAC policy is traced and respected by the partitioner; disable dynamo's
    # LRU cache to keep graph selection stable across pipeline microbatches.
    if enable_compile:
        _disable_dynamo_lru_cache()
    context_fn = functools.partial(
        transformer_engine_attention_backend_snapshot_context_fn,
        make_selective_checkpoint_context_fn(),
    )
    for i, layer in enumerate(layers):
        wrapped_layer = checkpoint_wrapper(
            layer,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            context_fn=context_fn,
            preserve_rng_state=True,
        )
        setattr(wrapped_layer, SELECTIVE_AC_WRAPPER_FLAG, True)
        if not _replace_child_module(model, layer, wrapped_layer):
            logger.warning("Could not replace layer %d with selective activation checkpoint wrapper.", i)
        layers[i] = wrapped_layer
