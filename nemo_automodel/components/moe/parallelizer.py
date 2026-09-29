# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import functools
import logging
import weakref
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard import MixedPrecisionPolicy, OffloadPolicy
from torch.distributed.tensor import Replicate, Shard, distribute_module, distribute_tensor
from torch.distributed.tensor.parallel import ParallelStyle, parallelize_module
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from nemo_automodel.components.distributed import parallelizer_utils
from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP, GroupedExpertsTE
from nemo_automodel.components.moe.layers import (
    Gate,
    MoE,
)
from nemo_automodel.components.moe.tp_plan_validation import _validate_moe_tp_plan
from nemo_automodel.shared.model_utils import iter_transformer_and_mtp_blocks
from nemo_automodel.shared.multimodal_fsdp import (
    MULTIMODAL_TOWER_NAMES,
    FrozenMultimodalSharding,
    ignored_params_for_root,
    iter_multimodal_modules,
    module_is_fully_frozen,
    module_parameters,
    normalize_frozen_multimodal_sharding,
    shard_multimodal_module,
)
from nemo_automodel.shared.tied_weights import ensure_tied_lm_head
from nemo_automodel.shared.torch_patches import (
    patch_fsdp_accumulated_grad_guard as _patch_fsdp_accumulated_grad_guard,
)
from nemo_automodel.shared.torch_patches import (
    patch_fsdp_uniform_reduce_dtype as _patch_fsdp_uniform_reduce_dtype,
)
from nemo_automodel.shared.utils import dtype_from_str

logger = logging.getLogger(__name__)
_CP_STREAM = None
_EMPTY_TENSOR_DETERMINISM_CHECK = "moe_empty_tensor_dtype_tolerant"


def _moe_checkpoint_metadata_fn(tensor: torch.Tensor):
    """Checkpoint metadata that ignores dtype only for zero-element tensors.

    Mixed-precision DeepEP can expose an empty routed-token tensor as BF16 in the
    original forward and FP32 during backward recomputation. Empty tensors have no values
    whose precision could differ, but PyTorch's default checkpoint check still
    rejects the dtype-only metadata change. Preserve shape/device checks for every
    tensor and dtype checks for every non-empty tensor.
    """
    return {
        "shape": tensor.shape,
        "dtype": tensor.dtype if tensor.numel() else None,
        "device": tensor.device,
    }


def _register_moe_checkpoint_determinism_check() -> str:
    """Register the narrow empty-tensor checker with PyTorch checkpointing."""
    import torch.utils.checkpoint as torch_checkpoint

    checks = getattr(torch_checkpoint, "_allowed_determinism_checks_to_fns", None)
    if checks is None:
        raise RuntimeError(
            "This PyTorch version does not expose checkpoint determinism-check registration; "
            "cannot safely tolerate empty MoE dtype changes."
        )
    checks[_EMPTY_TENSOR_DETERMINISM_CHECK] = _moe_checkpoint_metadata_fn
    return _EMPTY_TENSOR_DETERMINISM_CHECK


def _moe_shard_placement(param):
    """FSDP shard placement for grouped-expert params.

    Shard on dim=1 for the (>=2D) expert weights since there may be more shards than
    experts (dim=0). A 1D param (e.g. the per-expert bias of the experts="te"
    GroupedLinear path, shape [out_features]) has no dim 1, so shard it on dim 0
    instead. FSDP all-gathers before use, so the shard dim is a storage detail and does
    not change compute.
    """
    return Shard(0) if param.ndim < 2 else Shard(1)


def _is_selective_ac(activation_checkpointing: object) -> bool:
    """Return True when the AC mode requests selective checkpointing.

    Kept inline (rather than imported from the dense FSDP2 parallelizer) so that
    threading the mode does not pull the heavy ``distributed.parallelizer`` module
    into the lightweight call path.
    """
    return (
        isinstance(activation_checkpointing, str) and activation_checkpointing.lower().replace("-", "_") == "selective"
    )


def _get_cp_stream() -> torch.cuda.Stream:
    global _CP_STREAM
    if _CP_STREAM is None:
        _CP_STREAM = torch.cuda.Stream()
    return _CP_STREAM


def _get_moe_module(block: nn.Module) -> MoE | None:
    for name in ("moe", "mlp"):
        module = getattr(block, name, None)
        if isinstance(module, MoE):
            return module


def _repeated_mtp_moe_block_ids(model: nn.Module) -> set[int]:
    """Return weight-tied MTP blocks whose experts cannot be recomputed safely.

    A repeated MTP depth may contain multiple physical sublayers. Only MoE
    sublayers own the EP-sharded expert parameter group whose second FSDP2
    checkpoint recompute is unsafe; attention, MLP, and Mamba sublayers remain
    eligible for activation checkpointing.
    """
    mtp_module = getattr(model, "mtp", None)
    if mtp_module is None or not hasattr(mtp_module, "layers"):
        return set()
    mtp_repeated = bool(getattr(getattr(mtp_module, "mtp_config", None), "use_repeated_layer", False))
    if not mtp_repeated:
        return set()
    return {id(block) for block in mtp_module.layers.children() if _get_moe_module(block) is not None}


def _preserve_gate_load_during_recompute(
    block: nn.Module,
    context_fn: Callable[[], tuple[AbstractContextManager, AbstractContextManager]] | None = None,
) -> Callable[[], tuple[AbstractContextManager, AbstractContextManager]] | None:
    """Keep checkpoint recomputation from accumulating MoE gate load twice.

    Activation checkpointing replays the block during backward. The gate's
    cumulative expert load is optimizer-step state, so the replay must start
    from the same state as the original forward without committing its update.
    """
    moe = _get_moe_module(block)
    gate = getattr(moe, "gate", None)
    if not isinstance(gate, nn.Module) or not hasattr(gate, "_cumulative_expert_load"):
        return context_fn
    gate_ref = weakref.ref(gate)

    def checkpoint_context_fn() -> tuple[AbstractContextManager, AbstractContextManager]:
        forward_context, recompute_context = context_fn() if context_fn is not None else (nullcontext(), nullcontext())
        gate = gate_ref()
        preserve_load = gate is not None and gate.training and getattr(gate, "bias_update_factor", 0) > 0
        current_load = gate._cumulative_expert_load if preserve_load else None
        load_before_forward = current_load.clone() if current_load is not None else None

        @contextmanager
        def recompute():
            gate = gate_ref()
            if not preserve_load or gate is None:
                with recompute_context:
                    yield
                return

            load_after_forward = gate._cumulative_expert_load
            gate._cumulative_expert_load = load_before_forward
            try:
                with recompute_context:
                    yield
            finally:
                gate._cumulative_expert_load = load_after_forward

        return forward_context, recompute()

    return checkpoint_context_fn


def get_text_module(model: nn.Module) -> nn.Module:
    """Return the nested text/LLM module if present, else the model itself."""
    if model is None:
        return model
    for attr_name in ("language_model", "text_model", "text_decoder"):
        nested = getattr(model, attr_name, None)
        if isinstance(nested, nn.Module):
            return nested
    return model


def _get_model_moe_config(model: nn.Module):
    """Return the model-level MoE config exposed by custom MoE architectures."""
    candidates = []
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(inner)
        text_model = get_text_module(inner)
        if text_model is not inner:
            candidates.append(text_model)
    candidates.append(model)

    for candidate in candidates:
        moe_config = getattr(candidate, "moe_config", None)
        if moe_config is not None:
            return moe_config

    raise AttributeError("MoE models must expose moe_config on the inner, text, or top-level model.")


def _module_weights_are_tied(left: nn.Module | None, right: nn.Module | None) -> bool:
    """Return True when two modules expose the same ``weight`` parameter object."""
    if left is None or right is None:
        return False
    left_weight = getattr(left, "weight", None)
    right_weight = getattr(right, "weight", None)
    return left_weight is not None and left_weight is right_weight


def _resolve_moe_tp_plan(
    model: nn.Module,
    *,
    sequence_parallel: bool,
    tp_shard_plan: dict[str, ParallelStyle] | str | None,
    tp_size: int,
) -> dict[str, ParallelStyle]:
    """Resolve a fail-closed TP plan for a custom MoE model.

    The generic dense parallelizer has broad fallback plans. Those fallbacks
    can accidentally match ``*.experts`` on custom MoE architectures, so the
    MoE path only accepts an explicit plan or an architecture registration and
    always validates routed-expert ownership before applying it.
    """
    if sequence_parallel:
        raise ValueError(
            "sequence_parallel=True is not supported by the custom MoE TP path yet; "
            "use sequence_parallel=False so replicated router/attention activations remain correct."
        )
    if tp_size <= 1:
        return {}

    if tp_shard_plan is not None:
        # Reuse the standard parser for explicit dictionaries, named plans and
        # import paths, then apply the stricter MoE ownership validation below.
        from nemo_automodel.components.distributed.parallelizer import _get_parallel_plan

        plan = _get_parallel_plan(
            model,
            sequence_parallel=False,
            tp_shard_plan=tp_shard_plan,
            tp_size=tp_size,
        )
    else:
        from nemo_automodel.components.distributed.optimized_tp_plans import (
            PARALLELIZE_FUNCTIONS,
            _get_class_qualname,
        )

        model_cls = type(model)
        plan_factory = PARALLELIZE_FUNCTIONS.get(_get_class_qualname(model_cls))
        if plan_factory is None:
            plan_factory = PARALLELIZE_FUNCTIONS.get(model_cls.__name__)
        if plan_factory is None:
            raise ValueError(
                f"No safe tensor-parallel plan is registered for custom MoE model "
                f"'{model_cls.__name__}' at tp_size={tp_size}. Register a non-expert plan in "
                "PARALLELIZE_FUNCTIONS or pass tp_shard_plan explicitly; routed experts must remain EP-owned."
            )
        try:
            plan = plan_factory(model, False)
        except Exception as exc:
            raise ValueError(
                f"The registered tensor-parallel plan for custom MoE model '{model_cls.__name__}' failed: {exc}"
            ) from exc

    return _validate_moe_tp_plan(plan, model=model)


class ExpertParallel(ParallelStyle):
    """
    ExpertParallel class is used to shard the MoE parameters on the EP mesh.
    Dim `0` of each parameter is sharded since that is the expert dimension.
    """

    def _partition_fn(self, name, module, device_mesh):
        # shard on the expert dimension
        assert device_mesh.ndim == 1

        for name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            dist_param.requires_grad = param.requires_grad
            module.register_parameter(name, dist_param)

        if isinstance(module, GroupedExpertsDeepEP):
            module.init_token_dispatcher(ep_mesh=device_mesh)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        return distribute_module(
            module,
            device_mesh,
            self._partition_fn,
        )


def _iter_moe_blocks(model_wrapper: nn.Module, backbone: nn.Module):
    """Yield decoder blocks that may contain MoE sublayers.

    Covers the main backbone (``backbone.layers``) plus an optional MTP
    auxiliary head (``model_wrapper.mtp.layers``) when present. MTP sublayers
    are not registered under ``backbone.layers`` but carry the same MoE
    structure and must receive the same EP / FSDP treatment so their
    state-dict round-trips cleanly.

    Args:
        model_wrapper: Outer model (e.g. ``NemotronHForCausalLM``) — the
            attribute that may carry the MTP head.
        backbone: Inner backbone (``model_wrapper.model``, possibly text-only
            after VLM unwrapping) whose ``.layers`` holds the main decoder
            stack.
    """
    yield from backbone.layers.children()
    mtp_module = getattr(model_wrapper, "mtp", None)
    if mtp_module is not None and hasattr(mtp_module, "layers"):
        yield from mtp_module.layers.children()


def apply_ep(model: nn.Module, ep_mesh: DeviceMesh, moe_mesh: DeviceMesh | None = None):
    """Applies EP to MoE module."""
    assert ep_mesh.size() > 1

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model
    # Prefer nested text modules when present
    _model = get_text_module(_model)

    for block in _iter_moe_blocks(model, _model):
        moe_module = _get_moe_module(block)
        if moe_module is None:
            continue
        # GroupedExpertsTEGroupedLinear uses TE's GroupedLinear which creates
        # local experts directly. It doesn't support DTensor wrapping, so we
        # skip distribute_module entirely and just initialize token dispatcher.
        if isinstance(moe_module.experts, GroupedExpertsTE):
            moe_module.experts.init_token_dispatcher(ep_mesh=ep_mesh, moe_mesh=moe_mesh)
            # TE creates rank-local parameters for the experts owned by this EP
            # rank. A combined MoE mesh may fold physical TP peers into EP, so
            # these plain tensors are different expert shards, not TP replicas.
            from nemo_automodel.components.distributed.tp_replicas import exclude_from_tp_replica_sync

            exclude_from_tp_replica_sync(moe_module.experts)
        else:
            parallelize_module(
                module=moe_module.experts,
                device_mesh=ep_mesh,
                parallelize_plan=ExpertParallel(),
            )


# Alias of the shared tower taxonomy. Previously a private copy that had drifted
# from the other multimodal name lists in the tree.
_MULTIMODAL_TOWER_ATTRS = MULTIMODAL_TOWER_NAMES


def _has_trainable_multimodal_tower(model: nn.Module) -> bool:
    """Return whether the model (or its inner ``.model``) exposes a trainable vision/audio tower.

    Deliberately a cheap duck-typed gate, not a second owner of the tower
    mapping: it only decides whether importing the heavy, transformers-aware
    dense parallelizer is worthwhile, while the dense parallelizer's per-model
    layer-group mapping remains the sole owner of which blocks get wrapped.
    Requiring a trainable, parameter-bearing tower (rather than mere attribute
    existence) keeps the import off text-only, frozen-tower, and duck-typed
    stub-model call paths.
    """
    for owner in (model, getattr(model, "model", None)):
        if owner is None:
            continue
        for attr in _MULTIMODAL_TOWER_ATTRS:
            tower = getattr(owner, attr, None)
            if tower is None or not hasattr(tower, "parameters"):
                continue
            if any(param.requires_grad for param in tower.parameters()):
                return True
    return False


def _apply_multimodal_tower_ac(model: nn.Module, scopes: tuple[str, ...]) -> None:
    """Checkpoint trainable multimodal (vision/audio) tower blocks on the expert-parallel path.

    ``apply_ac`` iterates only the text/MTP decoder stack
    (``iter_transformer_and_mtp_blocks``), and the generic FSDP2 scope
    handling does not run for expert-parallel configs, so a trainable vision
    tower would otherwise keep every activation. Reuses the per-model
    layer-group mapping from the dense parallelizer and applies the same
    per-submodule wrapping (attention/MLP/norms) as the generic FSDP2/DDP
    path, with ``sdpa_backend_snapshot_context_fn`` on the attention/MLP
    wrappers so the backward-time recompute reruns under the forward-time SDPA
    backend set. Fully frozen vision towers are left untouched, consistent
    with the generic path's frozen-tower behavior.

    Args:
        model: Root model owning the tower(s) to checkpoint.
        scopes: Normalized activation-checkpointing scope tuple. ``("all",)``
            selects the vision and audio groups (matching the generic path's
            scope filter); otherwise only the named non-language groups are
            selected, with ``multimodal`` expanding to vision + audio. Groups
            the model does not expose are simply absent.
    """
    if scopes == ("all",):
        group_names: tuple[str, ...] = ("vision", "audio")
    else:
        group_names = tuple(
            dict.fromkeys(
                name
                for scope in scopes
                for name in (("vision", "audio") if scope == "multimodal" else (scope,))
                if name != "language"
            )
        )
    if not group_names or not _has_trainable_multimodal_tower(model):
        return

    # Lazy imports keep the heavy, transformers-aware dense parallelizer module
    # off the text-only MoE call path.
    from nemo_automodel.components.distributed.activation_checkpointing import (
        apply_submodule_checkpointing,
        sdpa_backend_snapshot_context_fn,
    )
    from nemo_automodel.components.distributed.parallelizer import get_model_layer_groups

    layer_groups = get_model_layer_groups(model)
    tower_layers = [
        layer
        for name in group_names
        for layer in layer_groups.get(name, [])
        if any(param.requires_grad for param in layer.parameters())
    ]
    if not tower_layers:
        logger.info(
            "No trainable multimodal tower blocks selected by activation checkpointing scope %s; "
            "skipping vision-tower activation checkpointing.",
            scopes,
        )
        return
    # Vision towers have no KV cache, so KV-sharing (a text-decoder concern)
    # never applies here.
    apply_submodule_checkpointing(tower_layers, has_kv_sharing=False, context_fn=sdpa_backend_snapshot_context_fn)


def _replay_deepep_dispatch_on_recompute(
    context_fn: Callable[[], tuple[AbstractContextManager, AbstractContextManager]],
) -> Callable[[], tuple[AbstractContextManager, AbstractContextManager]]:
    """Let a checkpointed block's recompute reuse the DeepEP layout it already computed.

    Activation checkpointing replays the block during backward, and the replayed
    MoE dispatch recomputes its routing layout from scratch even though the
    routing is identical to the forward's. DeepEP skips that exchange when handed
    the previous dispatch's handle -- the same shortcut its backward already
    takes, which is why the backward dispatch is ~1000x cheaper than a
    layout-computing forward one.

    The recorder is built inside ``checkpoint_context_fn``, so each checkpointed
    call gets its own: a block called once per forward pass keeps its passes'
    dispatches separate, and the recompute consumes them in the order the
    matching forward produced them.
    """
    from nemo_automodel.components.moe.megatron.fused_a2a import (
        DispatchReplayRecorder,
        dispatch_replay_scope,
    )

    def checkpoint_context_fn() -> tuple[AbstractContextManager, AbstractContextManager]:
        forward_context, recompute_context = context_fn()
        recorder = DispatchReplayRecorder()

        @contextmanager
        def scoped(mode: str, inner: AbstractContextManager):
            if mode == "replay":
                recorder.rewind()
            with dispatch_replay_scope(recorder, mode), inner:
                yield

        return scoped("record", forward_context), scoped("replay", recompute_context)

    return checkpoint_context_fn


def _replay_hybridep_dispatch_on_recompute(
    context_fn: Callable[[], tuple[AbstractContextManager, AbstractContextManager]],
) -> Callable[[], tuple[AbstractContextManager, AbstractContextManager]]:
    """Reuse checkpoint-forward HybridEP layouts during backward recomputation.

    HybridEP can produce a different receive-token extent when it rebuilds a
    layout during activation-checkpoint recompute, even when the saved router
    top-k is identical. Reusing the forward handle keeps the layout stable while
    still redispatching the recomputed activations, so activation memory remains
    checkpointed.
    """
    from nemo_automodel.components.moe.megatron.fused_a2a import (
        HybridEPDispatchReplayRecorder,
        hybridep_dispatch_replay_scope,
    )

    def checkpoint_context_fn() -> tuple[AbstractContextManager, AbstractContextManager]:
        forward_context, recompute_context = context_fn()
        recorder = HybridEPDispatchReplayRecorder()

        @contextmanager
        def scoped(mode: str, inner: AbstractContextManager):
            if mode == "replay":
                recorder.rewind()
            with hybridep_dispatch_replay_scope(recorder, mode):
                with inner:
                    yield
                if mode == "record":
                    # Cache the dispatch extents only after the selective-op
                    # context exits. Recompute can then reuse them without an
                    # extra aten.sum that would diverge from the forward trace.
                    recorder.finalize()

        return scoped("record", forward_context), scoped("replay", recompute_context)

    return checkpoint_context_fn


def _uses_hybridep_dispatch(model: nn.Module) -> bool:
    """Return whether any expert module uses the HybridEP dispatcher."""
    modules = getattr(model, "modules", None)
    if not callable(modules):
        return False
    return any(
        isinstance(module, (GroupedExpertsDeepEP, GroupedExpertsTE))
        and getattr(module, "dispatcher_backend", None) == "hybridep"
        for module in modules()
    )


def apply_ac(
    model: nn.Module,
    ignore_router: bool = True,
    hidden_size: int | None = None,
    num_experts: int | None = None,
    selective: bool = False,
    activation_checkpointing_scope: str | list[str] | tuple[str, ...] = "all",
):
    """Apply activation checkpointing to the model.

    Args:
        model: The model to apply activation checkpointing to.
        ignore_router: If True (the default), saves the MoE router projection and top-k
            outputs so recompute preserves expert assignments (avoids a CheckpointError from
            non-deterministic re-routing changing dispatch shapes). If False, a warning is emitted.
        hidden_size: Hidden dimension size. If None, derived from model.config.hidden_size.
        num_experts: Number of routed experts. If None, derived from moe_config.n_routed_experts
            first, then falls back to model.config attributes.
        selective: If True, applies TorchTitan-style per-op selective activation checkpointing
            (shared with the dense FSDP2 path) to each block. Takes precedence over
            ``ignore_router``; the shared policy saves ``topk``, and HybridEP reuses the
            checkpoint-forward dispatch layout while redispatching recomputed activations.
        activation_checkpointing_scope: Which layer groups to checkpoint -- the same field
            and semantics as the generic FSDP2/DDP path. ``"all"`` (the default) checkpoints
            the text/MoE decoder blocks plus the trainable vision tower; ``"language"`` the
            decoder blocks only; ``"vision"`` (or ``"multimodal"``) only the trainable
            tower blocks, skipping decoder checkpointing entirely. The scope decides WHICH
            groups participate; the ``selective``/``ignore_router`` mode decides HOW the
            selected decoder blocks are checkpointed.

    Trainable VLM vision-tower blocks selected by the scope get the same per-submodule
    wrapping (attention/MLP/norms) as the generic FSDP2/DDP path, with the SDPA backend
    set snapshotted at checkpoint-forward time and restored during the backward-time
    recompute; frozen vision towers are left untouched.
    """
    # Lazy import: keeps the scope normalization single-sourced with the strategy
    # configs without importing distributed.config on the (torch-stub-friendly)
    # module import path.
    from nemo_automodel.components.distributed.config import normalize_activation_checkpointing_scope

    scopes = normalize_activation_checkpointing_scope(activation_checkpointing_scope)
    checkpoint_decoder = "all" in scopes or "language" in scopes
    uses_hybridep_dispatch = checkpoint_decoder and _uses_hybridep_dispatch(model)
    repeated_mtp_moe_block_ids = _repeated_mtp_moe_block_ids(model) if checkpoint_decoder else set()
    if repeated_mtp_moe_block_ids:
        logger.info(
            "Skipping activation checkpointing on %d weight-tied MTP MoE block(s)",
            len(repeated_mtp_moe_block_ids),
        )
    if checkpoint_decoder and not selective and not ignore_router:
        logger.warning(
            "Activation checkpointing is enabled with ignore_router_for_ac=False. The MoE "
            "router/dispatch will be recomputed in the backward pass, which can route a "
            "different number of tokens per expert than the forward pass and crash with "
            "torch.utils.checkpoint.CheckpointError ('Recomputed values ... have different "
            "metadata'). Set ignore_router_for_ac=True (the default) to save the router "
            "projection and top-k outputs and keep routing consistent across recompute."
        )

    if selective:
        if checkpoint_decoder:
            # Reuse the dense FSDP2 selective policy so the save-op set (attention,
            # matmuls, comm collectives, topk, D2H copies) stays single-sourced.
            from nemo_automodel.components.distributed.activation_checkpointing import (
                SELECTIVE_AC_WRAPPER_FLAG,
                make_selective_checkpoint_context_fn,
                transformer_engine_attention_backend_snapshot_context_fn,
            )

            selective_context_fn = make_selective_checkpoint_context_fn()
            attention_context_fn = functools.partial(
                transformer_engine_attention_backend_snapshot_context_fn,
                selective_context_fn,
            )
            if uses_hybridep_dispatch:
                attention_context_fn = _replay_hybridep_dispatch_on_recompute(attention_context_fn)
            for parent_layers, layer_id, block in iter_transformer_and_mtp_blocks(model):
                if id(block) in repeated_mtp_moe_block_ids:
                    continue
                if bool(getattr(block, "_nemo_disable_activation_checkpointing", False)):
                    logger.info("Skipping activation checkpointing for model-owned eager block %s", layer_id)
                    continue
                block = ptd_checkpoint_wrapper(
                    block,
                    preserve_rng_state=True,
                    context_fn=_preserve_gate_load_during_recompute(block, attention_context_fn),
                )
                # Tag so _apply_per_layer_compile compiles the wrapper OUTER (keeping the
                # selective policy visible to the partitioner) instead of unwrapping and
                # compiling the block inner, which would collapse selective AC into full
                # recompute. The flag is only read when per-layer torch.compile is
                # enabled, so it is a no-op for every other mode.
                setattr(block, SELECTIVE_AC_WRAPPER_FLAG, True)
                parent_layers.register_module(layer_id, block)
        _apply_multimodal_tower_ac(model, scopes)
        return

    if not checkpoint_decoder:
        # Vision-only (or otherwise non-language) scope: skip decoder checkpointing
        # entirely, including the hidden_size/num_experts derivation it needs.
        _apply_multimodal_tower_ac(model, scopes)
        return

    # Derive hidden_size and num_experts from model.config if not provided
    if hidden_size is None:
        cfg = getattr(model, "config", None)
        # VLM models nest language model config under text_config or llm_config
        hidden_size = (
            getattr(getattr(cfg, "text_config", None), "hidden_size", None)
            or getattr(getattr(cfg, "llm_config", None), "hidden_size", None)
            or getattr(cfg, "hidden_size", None)
        )
        if hidden_size is None:
            raise ValueError("hidden_size must be provided or model must have config.hidden_size attribute")

    if num_experts is None:
        _inner = getattr(model, "model", model)
        if hasattr(_inner, "moe_config") and hasattr(_inner.moe_config, "n_routed_experts"):
            num_experts = _inner.moe_config.n_routed_experts
        else:
            cfg = getattr(model, "config", None)
            text_cfg = getattr(cfg, "text_config", None) or getattr(cfg, "llm_config", None) or cfg
            for attr in ["num_experts", "moe_num_experts", "n_routed_experts", "num_local_experts"]:
                if text_cfg is not None and hasattr(text_cfg, attr):
                    num_experts = getattr(text_cfg, attr)
                    break
            else:
                raise ValueError("num_experts must be provided or model must have config.num_experts attribute")

    def _is_router_projection(func, args) -> bool:
        aten = torch.ops.aten
        mm = getattr(getattr(aten, "mm", None), "default", None)
        addmm = getattr(getattr(aten, "addmm", None), "default", None)
        linear = getattr(getattr(aten, "linear", None), "default", None)
        if func == mm:
            return len(args) == 2 and args[1].shape == (hidden_size, num_experts)
        if func == addmm:
            return len(args) >= 3 and args[2].shape == (hidden_size, num_experts)
        if func == linear:
            return len(args) >= 2 and args[1].shape == (num_experts, hidden_size)
        return False

    router_topk = getattr(getattr(torch.ops.aten, "topk", None), "default", None)

    def _custom_policy(ctx, func, *args, **kwargs):
        if (router_topk is not None and func == router_topk) or _is_router_projection(func, args):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    def selective_checkpointing_context_fn():
        return create_selective_checkpoint_contexts(_custom_policy)

    from nemo_automodel.components.distributed.activation_checkpointing import (
        ensure_fsdp_ops_sac_ignored,
        ensure_profiler_ops_sac_ignored,
        transformer_engine_attention_backend_snapshot_context_fn,
    )

    ensure_profiler_ops_sac_ignored()
    ensure_fsdp_ops_sac_ignored()

    def _with_attention_backend_snapshot(context_fn=None):
        return functools.partial(transformer_engine_attention_backend_snapshot_context_fn, context_fn)

    for parent_layers, layer_id, block in iter_transformer_and_mtp_blocks(model):
        # A weight-tied MoE block is recomputed once per logical MTP depth.
        # FSDP2 cannot re-unshard its shared EP-sharded experts group after the
        # first recompute, so leave only those blocks uncheckpointed. Repeated
        # dense/attention/Mamba blocks have no such expert group and keep AC.
        if id(block) in repeated_mtp_moe_block_ids:
            continue
        if bool(getattr(block, "_nemo_disable_activation_checkpointing", False)):
            logger.info("Skipping activation checkpointing for model-owned eager block %s", layer_id)
            continue
        if ignore_router:
            # Only this branch pins routing across recompute (the policy saves the
            # router projection and top-k), which makes replaying a recorded
            # dispatch layout sound. Do not extend replay to the else-branch:
            # there the router may choose a different layout during recompute.
            block_context_fn = _preserve_gate_load_during_recompute(
                block,
                _with_attention_backend_snapshot(selective_checkpointing_context_fn),
            )
            block_context_fn = _replay_deepep_dispatch_on_recompute(block_context_fn)
            if uses_hybridep_dispatch:
                block_context_fn = _replay_hybridep_dispatch_on_recompute(block_context_fn)
            block = ptd_checkpoint_wrapper(
                block,
                preserve_rng_state=True,
                determinism_check=_register_moe_checkpoint_determinism_check(),
                context_fn=block_context_fn,
            )
        else:
            block = ptd_checkpoint_wrapper(
                block,
                preserve_rng_state=True,
                context_fn=_preserve_gate_load_during_recompute(block, _with_attention_backend_snapshot()),
            )

        parent_layers.register_module(layer_id, block)

    _apply_multimodal_tower_ac(model, scopes)


def apply_fsdp(
    model: torch.nn.Module,
    fsdp_mesh: DeviceMesh,
    ep_enabled: bool,
    ep_shard_enabled: bool,
    ep_shard_mesh: DeviceMesh | None = None,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    reshard_after_forward: bool = False,
    lm_head_precision: str | torch.dtype | None = None,
    wrap_outer_model: bool = True,
    frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
) -> None:
    """Apply FSDP wrapping to MoE transformer blocks and model-level modules."""
    frozen_multimodal_sharding = normalize_frozen_multimodal_sharding(frozen_multimodal_sharding)
    # MoE normally keeps fully frozen skipped towers with an always-run root,
    # but trainable multimodal towers still get standalone FSDP units. Install
    # the same lazy-state guard as dense FSDP for modality-free batches.
    _patch_fsdp_accumulated_grad_guard()
    # ``moe/fsdp_mixin`` drives post-backward by hand under pipeline parallelism,
    # so a group can reach the reduction holding both reduce-dtype accumulations
    # and a param-dtype gradient that arrived after its last no-sync reduction.
    _patch_fsdp_uniform_reduce_dtype()

    if isinstance(lm_head_precision, str):
        lm_head_precision = dtype_from_str(lm_head_precision, default=None)

    if mp_policy is None:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
    experts_mp_policy = parallelizer_utils.get_internal_fsdp_mp_policy(mp_policy)
    fp32_compute_module_names = tuple(getattr(model, "_keep_in_fp32_modules_strict", None) or ())

    fully_shard_impl = getattr(model, "_nemo_fully_shard", fully_shard)

    fully_shard_default = functools.partial(
        fully_shard_impl,
        mesh=fsdp_mesh,
        reshard_after_forward=reshard_after_forward,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
    )

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model
    # Prefer nested text modules when present (VLM models)
    _model = get_text_module(_model)

    # Models may construct a rank-local shell first (so meta initialization is
    # cheap) and turn it into a globally shaped DTensor only after the runtime
    # mesh exists. Run that private capability before collecting ignored
    # parameters so every FSDP unit records the final Parameter identity.
    prepare_model_owned_dtensors = getattr(model, "_nemo_prepare_model_owned_dtensors", None)
    prepared_model_owned_dtensors: set[nn.Parameter] = set()
    if prepare_model_owned_dtensors is not None:
        prepared_model_owned_dtensors = set(prepare_model_owned_dtensors(fsdp_mesh))

    # Some trainable parameters are already physically sharded by model-owned
    # communication. Letting FSDP shard those local owner partitions again
    # would invalidate the model's lookup and autograd routing. The explicit
    # parameter marker keeps this exception narrow and fail-visible.
    # Some unit-test and integration wrappers intentionally expose the nested
    # model without subclassing nn.Module.  They cannot own parameters
    # themselves, so treat a missing ``parameters`` method as an empty outer
    # parameter set while preserving the normal nn.Module path.
    outer_parameters = model.parameters() if hasattr(model, "parameters") else ()
    externally_sharded_params = prepared_model_owned_dtensors | {
        parameter
        for parameter in outer_parameters
        if getattr(parameter, "_nemo_model_owned_grad_divisor", None) is not None
    }
    if externally_sharded_params:
        logger.info(
            "Excluding %d model-owned sharded parameters from FSDP ownership",
            len(externally_sharded_params),
        )

    multimodal_modules: list[tuple[str, nn.Module, set[nn.Parameter], bool]] = []
    for module_name, module in iter_multimodal_modules(model):
        module_params = set(module_parameters(module))
        if module_params:
            multimodal_modules.append((module_name, module, module_params, module_is_fully_frozen(module)))

    frozen_multimodal_modules = [
        (module_name, module_params)
        for module_name, _, module_params, is_fully_frozen in multimodal_modules
        if is_fully_frozen
    ]
    if frozen_multimodal_sharding == "root" and not wrap_outer_model and model is not _model:
        inner_root_params = set(module_parameters(_model))
        unowned_module_names = [
            module_name
            for module_name, module_params in frozen_multimodal_modules
            if not module_params.issubset(inner_root_params)
        ]
        if unowned_module_names:
            raise ValueError(
                "distributed.multimodal.frozen_sharding='root' requires wrap_outer_model=True when fully frozen "
                f"multimodal modules are outside the inner text FSDP root: {', '.join(unowned_module_names)}. "
                "Set wrap_outer_model=True, or choose distributed.multimodal.frozen_sharding='per_layer' with "
                "collective-uniform modality execution, or distributed.multimodal.frozen_sharding='replicate'."
            )
    if frozen_multimodal_sharding == "per_layer" and frozen_multimodal_modules:
        logger.warning(
            "distributed.multimodal.frozen_sharding='per_layer' selected for %s. Every rank in the FSDP group "
            "must execute or skip these modules the same number of times and in the same order on every "
            "microbatch; rank-asymmetric modality execution can hang or desynchronize FSDP collectives.",
            ", ".join(module_name for module_name, _ in frozen_multimodal_modules),
        )

    # MTP auxiliary-head blocks keep their EP-sharded experts un-resharded
    # (reshard_after_forward=False) even when the backbone reshards. The MTP head is
    # tiny (1-2 MoE sublayers) so keeping its experts gathered costs negligible resident
    # memory, while it removes FSDP2 unshard fragility for the weight-tied head whose
    # single physical experts param group is used multiple times per step (see apply_ac,
    # which skips AC on these blocks for the same reason). Bulk backbone experts keep the
    # configured reshard_after_forward.
    mtp_module = getattr(model, "mtp", None)
    mtp_block_ids = set()
    if mtp_module is not None and hasattr(mtp_module, "layers"):
        mtp_block_ids = {id(b) for b in mtp_module.layers.children()}

    for block in _iter_moe_blocks(model, _model):
        moe_module = _get_moe_module(block)
        gate = getattr(moe_module, "gate", None)
        if isinstance(gate, Gate) and gate.e_score_correction_bias is not None:
            # FSDP2 does not convert buffers to DTensors. Replicate the routing
            # bias over the full DP/CP mesh so its existing Partial-to-Replicate
            # update path aggregates every rank's expert load.
            gate.e_score_correction_bias = distribute_tensor(
                gate.e_score_correction_bias,
                device_mesh=fsdp_mesh,
                placements=[Replicate()] * fsdp_mesh.ndim,
            )
        experts_reshard_after_forward = False if id(block) in mtp_block_ids else reshard_after_forward
        if isinstance(moe_module, MoE) and ep_shard_enabled:
            # Apply FSDP on dim=1 for grouped experts since we may have more
            # shards than experts (dim=0).
            # Preserve the enclosing policy's parameter, reduction, and input-cast
            # settings so FP32 master weights still compute in param_dtype (required
            # by BF16 grouped-MM / TE kernels). Experts are an internal FSDP boundary, so
            # their policy does not override the activation dtype returned to the
            # rest of the block.
            fully_shard(
                moe_module.experts,
                mesh=ep_shard_mesh,
                shard_placement_fn=_moe_shard_placement,
                reshard_after_forward=experts_reshard_after_forward,
                mp_policy=experts_mp_policy,
                offload_policy=offload_policy,
            )
        # If FSDP is disabled for grouped experts because the parameters are already
        # fully sharded by PP and EP, then we need to explicitly remove the parameters
        # from FSDP for the transformer block.
        # If FSDP is enabled for grouped experts, the parameters are automatically
        # removed from the FSDP for the transformer block due to the rules of the
        # PyTorch FSDP implementation.
        ignored_params: set[nn.Parameter] = set()
        if isinstance(moe_module, MoE) and ep_enabled:
            ignored_params.update(moe_module.experts.parameters())
        if externally_sharded_params:
            ignored_params.update(externally_sharded_params.intersection(block.parameters()))

        # Reuse the dense dtype-aware path for model-owned fp32 contracts while
        # leaving EP-owned experts out of the block's dtype and FSDP ownership.
        parallelizer_utils.fully_shard_by_dtype(
            block,
            mesh=fsdp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            fp32_compute_module_names=fp32_compute_module_names,
            reshard_after_forward=reshard_after_forward,
            ignored_params=ignored_params or None,
            fully_shard_fn=fully_shard_impl,
        )

    # Re-establish weight tying before detecting it: a device/dtype move during
    # from_pretrained (HF replaces param tensors) can silently break a tie set in
    # __init__ (lm_head.weight = embed_tokens.weight), so the identity check below
    # would miss it and shard embed and lm_head as two INDEPENDENT FSDP params.
    # For tie_word_embeddings models those two then drift apart during full
    # fine-tuning, diverging from the tied architecture and breaking checkpoint
    # resume (the checkpoint drops lm_head and reconstructs it from embed on load,
    # silently discarding the drifted lm_head). Re-tying here keeps them in one
    # FSDP root so they remain a single shared parameter.
    # NB: re-point output->input embedding directly (model.__init__'s own tie);
    # HF's tie_weights() is incompatible with this model's _tied_weights_keys
    # format ('list' object has no attribute 'keys').
    if getattr(getattr(model, "config", None), "tie_word_embeddings", False):
        _out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        _inp = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
        if _out is not None and _inp is not None and hasattr(_out, "weight") and hasattr(_inp, "weight"):
            _out.weight = _inp.weight

    embed_tokens = getattr(_model, "embed_tokens", None)
    embed_norm = getattr(_model, "embed_norm", None)
    inner_lm_head = getattr(_model, "lm_head", None)
    outer_lm_head = getattr(model, "lm_head", None) if model is not _model else None
    lm_head = inner_lm_head or outer_lm_head
    tied_input_output_embeddings = _module_weights_are_tied(embed_tokens, lm_head)
    tied_embeddings_cross_fsdp_roots = tied_input_output_embeddings and lm_head is outer_lm_head

    if embed_tokens is not None and not tied_input_output_embeddings:
        fully_shard_default(embed_tokens)

    # The outer VLM forward calls embed_norm before the text FSDP root.
    if embed_norm is not None:
        fully_shard_default(embed_norm)

    if lm_head is not None and not tied_input_output_embeddings:
        # Use custom mixed precision policy for lm_head if lm_head_precision is specified
        if lm_head_precision == torch.float32:
            lm_head_mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )
            fully_shard(
                lm_head,
                mesh=fsdp_mesh,
                reshard_after_forward=reshard_after_forward,
                mp_policy=lm_head_mp_policy,
                offload_policy=offload_policy,
            )
        else:
            fully_shard_default(lm_head)
    elif tied_input_output_embeddings and lm_head_precision is not None:
        logger.warning(
            "Skipping separate lm_head FSDP wrapping because lm_head.weight is tied to embed_tokens.weight; "
            "lm_head_precision=%s will not be applied independently.",
            lm_head_precision,
        )

    ignored_multimodal_params: set[nn.Parameter] = set()
    for module_name, module, module_params, is_fully_frozen in multimodal_modules:
        if not is_fully_frozen:
            # Trainable multimodal modules keep normal standalone sharding; this
            # policy is intentionally limited to fully frozen modules.
            #
            # NOTE: this is wider than pre-#2763, which sharded only top-level
            # ``audio_tower``/``visual``. Narrowing it back is NOT safe: with
            # ``wrap_outer_model=False`` a trainable multimodal module on the
            # outer model would then land in no FSDP unit at all, losing
            # gradient reduction across DP ranks (covered by
            # test_apply_fsdp_without_outer_root_allows_supported_multimodal_policies).
            # ``moe/fsdp_mixin.py::_iter_fsdp_modules`` reuses the same shared
            # multimodal taxonomy so every unit created here also participates
            # in the gradient-accumulation sync state machine.
            fully_shard_default(module)
        elif frozen_multimodal_sharding == "per_layer":
            shard_multimodal_module(module, fully_shard_default)
        else:
            logger.info(
                "Keeping frozen multimodal module %s at FSDP policy %s",
                module_name,
                frozen_multimodal_sharding,
            )
            if frozen_multimodal_sharding == "replicate":
                ignored_multimodal_params.update(module_params)

    if tied_embeddings_cross_fsdp_roots and wrap_outer_model and model is not _model:
        logger.info(
            "Skipping separate inner-model FSDP root because lm_head.weight is tied to embed_tokens.weight "
            "across the outer model boundary; the outer FSDP root will own the tied parameter."
        )
    else:
        if tied_embeddings_cross_fsdp_roots and not wrap_outer_model:
            raise ValueError(
                "lm_head.weight is tied to embed_tokens.weight across the outer model boundary, but "
                "wrap_outer_model=False cannot preserve that parameter in one FSDP root. "
                "Use wrap_outer_model=True or untie the embeddings explicitly."
            )
        inner_ignored_params = ignored_multimodal_params | externally_sharded_params
        fully_shard_default(_model, ignored_params=ignored_params_for_root(_model, inner_ignored_params))

    # If model has a nested structure (outer model wrapping inner _model), wrap the outer model if requested.
    if wrap_outer_model and model is not _model:
        outer_ignored_params = ignored_multimodal_params | externally_sharded_params
        fully_shard_default(model, ignored_params=ignored_params_for_root(model, outer_ignored_params))


def apply_cp(model: torch.nn.Module, cp_mesh: DeviceMesh, cp_comm_type: str = "p2p"):
    """Configure context parallelism for attention and MoE layers."""

    from transformer_engine.pytorch.attention import DotProductAttention

    if hasattr(model, "model") and model.model is not None:
        _model = model.model
    else:
        _model = model
    # Prefer nested text modules when present (VLM models)
    _model = get_text_module(_model)

    # Set CP flags so wrapper and text-module forward paths can prepare
    # full-sequence embeddings and null out local attention masks.
    # With CP the mask is not sharded along the sequence dim and TE asserts
    # "Padding mask not supported with context parallelism!".
    model._cp_enabled = True
    _model._cp_enabled = True

    # Hand the CP submesh to the model so a forward that embeds and
    # sequence-shards its own primary stream (Megatron-style per-microbatch CP;
    # see shard_sequence_for_cp_round_robin / shard_batch_aux_only) can build this rank's
    # round-robin shard. Set on the top-level model the recipe calls.
    model.cp_mesh = cp_mesh

    # Route each attention block's CP setup by capability:
    #   * TE DotProductAttention -> TE's own context-parallel group;
    #   * a module exposing setup_cp_attention (e.g. Gemma4's p2p ring or MiniMax
    #     M3's block-sparse DSA) -> installs its own CP attention + mask handling
    #     (model-owned, like TE/DSV4).
    # Any other (non-TE, non-model-owned) attention is not supported under CP here.
    for _parent, _layer_id, block in iter_transformer_and_mtp_blocks(model):
        layer_type = getattr(block, "layer_type", getattr(block, "attention_type", "full_attention"))

        if layer_type in ("full_attention", "sliding_attention"):
            self_attn = block.self_attn
            attn_module = getattr(self_attn, "attn_module", None)
            if isinstance(attn_module, DotProductAttention):
                attn_cp_comm_type = "all_gather" if layer_type == "sliding_attention" else cp_comm_type
                attn_module.set_context_parallel_group(
                    cp_mesh.get_group(),
                    torch.distributed.get_process_group_ranks(cp_mesh.get_group()),
                    _get_cp_stream(),
                    cp_comm_type=attn_cp_comm_type,
                )
            elif hasattr(self_attn, "setup_cp_attention"):
                # Model-owned CP attention (e.g. Gemma4's p2p ring): the model
                # installs its own SDPA hook + mask handling.
                self_attn.setup_cp_attention(cp_mesh)
            else:
                logger.warning(
                    "Skipping CP setup for block with unsupported attention module "
                    "(neither TE DotProductAttention nor model-owned setup_cp_attention): %s",
                    type(attn_module).__name__ if attn_module is not None else type(self_attn).__name__,
                )
        elif layer_type == "mamba":
            from nemo_automodel.components.distributed.context_parallel.mamba import MambaContextParallel

            mixer = block.self_attn  # NemotronV3Block.self_attn aliases mixer
            mixer.cp = MambaContextParallel(
                cp_group=cp_mesh.get_group(),
                num_heads=mixer.num_heads,
                head_dim=mixer.head_dim,
                n_groups=mixer.n_groups,
                d_state=mixer.ssm_state_size,
                mixer=mixer,
            )
        elif layer_type == "linear_attention":
            # FLA-based CP: store the CP mesh on the linear attention module so it
            # can recover dense token order and build its CP context during forward.
            if hasattr(block, "linear_attn") and hasattr(block.linear_attn, "_cp_mesh"):
                block.linear_attn._cp_mesh = cp_mesh
            else:
                logger.warning(
                    "Block %s has linear_attention but no CP-aware linear_attn module; "
                    "skipping CP setup for this layer.",
                    getattr(block, "layer_idx", "?"),
                )

        moe_module = block.moe if hasattr(block, "moe") else block.mlp
        if isinstance(moe_module, MoE):
            moe_module.cp_mesh = cp_mesh


def parallelize_model(
    model: torch.nn.Module,
    world_mesh: DeviceMesh,
    moe_mesh: DeviceMesh | None,
    *,
    dp_axis_names: tuple[str, ...],
    cp_axis_name: str | None = None,
    tp_axis_name: str | None = None,
    ep_axis_name: str | None = None,
    ep_shard_axis_names: tuple[str, ...] | None = None,
    activation_checkpointing: bool | str = False,
    ignore_router_for_ac: bool = True,
    activation_checkpointing_scope: str | list[str] | tuple[str, ...] = "all",
    reshard_after_forward: bool = False,
    lm_head_precision: str | torch.dtype | None = None,
    wrap_outer_model: bool = True,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    tp_shard_plan: dict[str, ParallelStyle] | str | None = None,
    sequence_parallel: bool = False,
    enable_async_tensor_parallel: bool = False,
    frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
    reapply_trainability: Callable[[nn.Module], None] | None = None,
) -> None:
    """Apply tensor, context, expert, activation-checkpointing, and FSDP parallelism.

    Args:
        reapply_trainability: Optional callback that re-resolves parameter
            trainability after TP/EP/AC surgery and immediately before FSDP
            construction.
    """

    tp_enabled = tp_axis_name is not None and world_mesh[tp_axis_name].size() > 1
    if tp_enabled:
        if enable_async_tensor_parallel:
            raise ValueError(
                "enable_async_tensor_parallel=True is not supported by the custom MoE TP path; "
                "it requires sequence parallelism, which is intentionally disabled for replicated router/attention paths."
            )
        tp_mesh = world_mesh[tp_axis_name]
        model_parallel_plan = _resolve_moe_tp_plan(
            model,
            sequence_parallel=sequence_parallel,
            tp_shard_plan=tp_shard_plan,
            tp_size=tp_mesh.size(),
        )
        # Every custom-MoE TP plan keeps the token path (attention, router)
        # replicated across TP ranks, so each expert gradient accumulates
        # tp_size identical contributions through the EP all-gather.
        # get_expert_tp_replication_factor reads this marker to remove that
        # factor in scale_grads_and_clip_grad_norm.
        model._nemo_moe_tp_requires_replica_sync = True
        # The custom-MoE TP support contract requires a complete pretrained base
        # model and excludes PEFT because adapter and initialization ownership are
        # undefined across combined TP/EP. This marker enforces that contract for
        # both registered and explicit plans during setup and checkpoint loading.
        model._nemo_moe_tp_requires_pretrained_weights = True
        # PEFT is applied before distributed sharding. Translate each style so
        # LoRA-wrapped shared-expert/lm-head modules keep the same TP semantics.
        from nemo_automodel.components.distributed.parallel_styles import translate_to_lora

        model_parallel_plan = {path: translate_to_lora(style) for path, style in model_parallel_plan.items()}
        logger.info(
            "Applying safe custom-MoE tensor parallelism (tp_size=%d) to %d non-expert module patterns: %s",
            tp_mesh.size(),
            len(model_parallel_plan),
            ", ".join(sorted(model_parallel_plan)),
        )
        parallelize_module(model, tp_mesh, model_parallel_plan)
        ensure_tied_lm_head(model)

    cp_enabled = cp_axis_name is not None and world_mesh[cp_axis_name].size() > 1
    if cp_enabled:
        parallelizer_utils.reject_unsupported_mtp_cp_pp(model)
        parallelizer_utils.reject_unsupported_mtp_cp(model)
        apply_cp(model, world_mesh[cp_axis_name])

    ep_enabled = ep_axis_name is not None and moe_mesh is not None and moe_mesh[ep_axis_name].size() > 1
    if ep_enabled:
        moe_config = _get_model_moe_config(model)
        assert moe_config.n_routed_experts % moe_mesh[ep_axis_name].size() == 0, (
            f"n_routed_experts {moe_config.n_routed_experts} must be divisible by "
            f"expert_parallel_degree {moe_mesh[ep_axis_name].size()}"
        )

        apply_ep(model, moe_mesh[ep_axis_name], moe_mesh=moe_mesh)

    if activation_checkpointing:
        apply_ac(
            model,
            ignore_router=ignore_router_for_ac,
            selective=_is_selective_ac(activation_checkpointing),
            activation_checkpointing_scope=activation_checkpointing_scope,
        )

    if reapply_trainability is not None:
        reapply_trainability(model)

    if ep_shard_axis_names is not None:
        ep_shard_mesh = moe_mesh[ep_shard_axis_names]
    else:
        ep_shard_mesh = None

    from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh, get_submesh

    axis_names = tuple(dp_axis_names or ())
    # HSDP combines a native replica axis with a flattened shard/CP axis.
    # Keep their common root mesh so FSDP retains the replica process group.
    if axis_names == ("dp_replicate", "dp_shard_cp"):
        fsdp_mesh = get_fsdp_dp_mesh(world_mesh, *axis_names)
    else:
        fsdp_mesh = get_submesh(world_mesh, axis_names) if axis_names else None
    fsdp_enabled = fsdp_mesh is not None and fsdp_mesh.size() > 1
    if fsdp_enabled:
        apply_fsdp(
            model,
            fsdp_mesh,
            ep_enabled=ep_enabled,
            ep_shard_enabled=ep_shard_mesh is not None and ep_shard_mesh.size() > 1,
            ep_shard_mesh=ep_shard_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=reshard_after_forward,
            lm_head_precision=lm_head_precision,
            wrap_outer_model=wrap_outer_model,
            frozen_multimodal_sharding=frozen_multimodal_sharding,
        )
        if cp_enabled:
            configured_units = parallelizer_utils.configure_fsdp_unused_param_reduction(model)
            logger.info(
                "Enabled unused-parameter reduce-scatter on %d MoE FSDP units for context parallelism",
                configured_units,
            )
