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

from copy import copy
from typing import Callable, Dict, Iterator, List, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)

from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from nemo_automodel.shared.torch_patches import (
    patch_fsdp_uniform_reduce_dtype as _patch_fsdp_uniform_reduce_dtype,
)
from nemo_automodel.shared.torch_patches import (
    patch_fsdp_unused_param_reduction as _patch_fsdp_unused_param_reduction,
)

UniformSubtreeItem = Union[Tuple[nn.Module, torch.dtype], Tuple[str, nn.Module, torch.dtype]]


def reject_unsupported_mtp_cp(model: nn.Module) -> None:
    """Reject enabled MTP when the model has not declared CP support."""
    if model.supports.mtp_enabled and not model.supports.supports_mtp_cp:
        raise RuntimeError(f"{type(model).__name__} does not support MTP with context parallelism")


def reject_unsupported_mtp_cp_pp(model: nn.Module) -> None:
    """Reject MTP+CP on every trimmed pipeline stage before CP collectives."""
    is_pp_stage_fn = getattr(model, "_is_pipeline_parallel_stage", None)
    if (
        model.supports.mtp_enabled
        and not model.supports.supports_mtp_cp_pp
        and callable(is_pp_stage_fn)
        and is_pp_stage_fn()
    ):
        raise NotImplementedError(
            "MTP with context and pipeline parallelism is not supported; use PP size 1 or CP size 1"
        )


def configure_fsdp_unused_param_reduction(module: nn.Module) -> int:
    """Reduce zero gradients for FSDP parameters unused on a local CP rank.

    Packed or modality-dependent context-parallel batches may execute a module
    on only a subset of ranks. FSDP must still issue the same reduce-scatter
    sequence everywhere; otherwise a rank with ``grad is None`` can omit a
    collective and discard peer contributions. PyTorch's public API fills the
    missing local contribution with zero, analogous to DDP unused-parameter
    handling. AutoModel keeps a compatibility fallback for supported PyTorch
    versions that predate that public API.

    Args:
        module: Root module containing the FSDP units to configure.

    Returns:
        Number of FSDP units configured.
    """
    fsdp_modules = [candidate for candidate in module.modules() if isinstance(candidate, FSDPModule)]
    if not fsdp_modules:
        return 0

    # Install first so the zero fill below wraps it: the filled zero is in param
    # dtype and must be aligned with the peers' reduce-dtype accumulations before
    # the group reaches ``foreach_reduce``.
    _patch_fsdp_uniform_reduce_dtype()
    if hasattr(fsdp_modules[0], "set_reduce_scatter_unused_params"):
        for fsdp_module in fsdp_modules:
            fsdp_module.set_reduce_scatter_unused_params(True, recurse=False)
    else:
        _patch_fsdp_unused_param_reduction()
    return len(fsdp_modules)


def iter_maximal_uniform_dtype_subtrees(
    module: nn.Module,
    *,
    include_buffers: bool = True,
    tensor_pred: Callable[[torch.Tensor], bool] | None = None,
    dtype_of: Callable[[torch.Tensor], torch.dtype] | None = None,
    return_paths: bool = False,
) -> Iterator[UniformSubtreeItem]:
    """
    Traverse `module` and yield maximal submodules whose entire subtree has a unified dtype.

    - include_buffers: include buffers in dtype unification checks.
    - tensor_pred: predicate to choose which tensors to consider (default: all).
                   Example: tensor_pred=torch.is_floating_point  (to consider only FP tensors)
    - dtype_of: maps a tensor to the dtype used for unification (default: its storage
                dtype ``t.dtype``). Pass a custom function to group by *compute* dtype
                rather than storage dtype.
    - return_paths: if True, yields (qualified_name, module, dtype); else (module, dtype).

    Notes:
    - If a module subtree has no tensors passing `tensor_pred`, it is ignored.
    - Maximality ensures no yielded module is a strict child of another yielded module.
    """
    if tensor_pred is None:
        tensor_pred = lambda t: True
    if dtype_of is None:
        dtype_of = lambda t: t.dtype

    def _local_dtype_set(m: nn.Module) -> Set[torch.dtype]:
        ds: Set[torch.dtype] = set()
        for p in m.parameters(recurse=False):
            if tensor_pred(p):
                ds.add(dtype_of(p))
        if include_buffers:
            for b in m.buffers(recurse=False):
                if tensor_pred(b):
                    ds.add(dtype_of(b))
        return ds

    def _visit(m: nn.Module, path: Tuple[str, ...]) -> Tuple[Set[torch.dtype], List[UniformSubtreeItem]]:
        local = _local_dtype_set(m)
        subtree_dtypes: Set[torch.dtype] = set(local)
        collected: List[UniformSubtreeItem] = []

        # Recurse into children
        for name, child in m.named_children():
            child_set, child_yields = _visit(child, path + (name,))
            subtree_dtypes |= child_set
            collected.extend(child_yields)

        # If entire subtree has exactly one dtype (and not empty), this node is maximal: override children yields
        if len(subtree_dtypes) == 1:
            if subtree_dtypes:
                dtype = next(iter(subtree_dtypes))
                if return_paths:
                    qname = ".".join(path)  # empty string at root
                    return subtree_dtypes, [(qname, m, dtype)]
                else:
                    return subtree_dtypes, [(m, dtype)]
            # else: no tensors in subtree -> ignore entirely
        # Not uniform -> keep whatever maximal sets children produced
        return subtree_dtypes, collected

    _, items = _visit(module, ())
    # Stream results
    for it in items:
        yield it


def _group_params_by_dtype(
    layer: nn.Module,
    dtype_of: Callable[[torch.Tensor], torch.dtype] | None = None,
    ignored_params: set[nn.Parameter] | None = None,
) -> Dict[torch.dtype, List[nn.Parameter]]:
    if dtype_of is None:
        dtype_of = lambda t: t.dtype
    ignored_param_ids = {id(param) for param in ignored_params or ()}
    ans: Dict[torch.dtype, List[nn.Parameter]] = {}
    for name, param in layer.named_parameters():
        if id(param) in ignored_param_ids:
            continue
        dtype = dtype_of(param)
        if dtype not in ans:
            ans[dtype] = []
        ans[dtype].append(param)
    return ans


def _get_module_from_path(layer: nn.Module, path: str) -> nn.Module:
    for name in path.split("."):
        layer = getattr(layer, name)
    return layer


def _fully_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None,
    reshard_after_forward: bool | int | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    fully_shard_fn: Callable[..., None] | None = None,
) -> None:
    if isinstance(module, nn.ModuleList):
        for layer in module:
            _fully_shard(
                layer,
                mesh,
                mp_policy,
                offload_policy,
                reshard_after_forward,
                ignored_params,
                fully_shard_fn,
            )
    else:
        _call_fully_shard(
            module,
            mesh,
            mp_policy,
            offload_policy,
            reshard_after_forward,
            ignored_params,
            fully_shard_fn,
        )


def _call_fully_shard(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None,
    reshard_after_forward: bool | int | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    fully_shard_fn: Callable[..., None] | None = None,
) -> None:
    if fully_shard_fn is None:
        fully_shard_fn = fully_shard

    kwargs = {
        "mesh": mesh,
        "mp_policy": mp_policy,
        "offload_policy": offload_policy,
    }
    if reshard_after_forward is not None:
        kwargs["reshard_after_forward"] = reshard_after_forward

    if ignored_params:
        module_param_ids = {id(param) for param in module.parameters()}
        module_ignored_params = {param for param in ignored_params if id(param) in module_param_ids}
        if module_ignored_params:
            kwargs["ignored_params"] = module_ignored_params

    fully_shard_fn(module, **kwargs)


def _mp_policy_with_param_dtype(
    mp_policy: MixedPrecisionPolicy | None,
    param_dtype: torch.dtype,
) -> MixedPrecisionPolicy | None:
    if mp_policy is None:
        return None
    mp_policy_copy = copy(mp_policy)
    object.__setattr__(mp_policy_copy, "param_dtype", param_dtype)
    if param_dtype == torch.float32:
        object.__setattr__(mp_policy_copy, "reduce_dtype", torch.float32)
        object.__setattr__(mp_policy_copy, "output_dtype", torch.float32)
        # FP32 compute modules own any required input cast. Casting at the nested
        # FSDP boundary changes the module-visible input dtype and can make an
        # activation-checkpoint recompute disagree with the original forward.
        object.__setattr__(mp_policy_copy, "cast_forward_inputs", False)
    return mp_policy_copy


def get_internal_fsdp_mp_policy(
    mp_policy: MixedPrecisionPolicy | None,
) -> MixedPrecisionPolicy | None:
    """Clone an FSDP policy without imposing an external output dtype.

    Internal FSDP units are implementation details inside a parent module's
    forward. Their outputs may feed an unwrapped sibling before another FSDP
    input cast, so they preserve the wrapped module's natural output dtype.

    Args:
        mp_policy: Mixed-precision policy inherited from the enclosing FSDP
            boundary, or ``None`` when mixed precision is disabled.

    Returns:
        A cloned policy with ``output_dtype=None``, or ``None`` when no policy
        was provided. Parameter, reduction, and input-cast settings are unchanged.
    """
    if mp_policy is None:
        return None
    mp_policy_copy = copy(mp_policy)
    object.__setattr__(mp_policy_copy, "output_dtype", None)
    return mp_policy_copy


def _make_compute_dtype_fn(
    module: nn.Module,
    mp_policy: MixedPrecisionPolicy | None,
    fp32_compute_module_names: Tuple[str, ...],
    ignored_params: set[nn.Parameter] | None = None,
) -> Callable[[torch.Tensor], torch.dtype]:
    """Build the per-parameter *compute* dtype resolver used to group FSDP units.

    The compute dtype of a floating tensor is resolved by precedence:

      1. Pinned fp32 -- the tensor's name matches ``fp32_compute_module_names``
         (from the model's ``_keep_in_fp32_modules_strict``). Authoritative, works
         even from-scratch / quantized where there is no checkpoint to read.
      2. HF-recorded dtype -- ``tensor._hf_compute_dtype``, the checkpoint's original
         dtype recorded at load time (see ``_restore_loaded_model_dtype``). This makes
         any checkpoint-loaded model keep its intrinsically-fp32 params in fp32 compute
         automatically, even after storage was upcast for fp32 master weights.
      3. Fallback -- when the tensor carries no compute hint, an fp32 storage under a
         lower-precision policy is an fp32 master weight and computes in
         ``mp_policy.param_dtype`` (the requested compute dtype, typically bf16); any
         other storage keeps its own dtype (and so does the fp32 case when no policy is
         given). Resolved per-param -- a single genuinely lower-precision sibling (e.g.
         Qwen3.5-MoE's bf16 ``shared_expert_gate``) no longer forces the layer's fp32
         master weights into fp32 compute. Intrinsic fp32 is already covered by #1/#2;
         the ``(storage, compute)`` grouping still keeps each FSDP unit storage-uniform.
         See NVIDIA-NeMo/Automodel#3327.

    Non-floating tensors always keep their storage dtype.
    """
    policy_dtype = getattr(mp_policy, "param_dtype", None)

    ignored_param_ids = {id(param) for param in ignored_params or ()}

    pinned_ids: Set[int] = set()
    if fp32_compute_module_names:
        for name, tensor in (*module.named_parameters(), *module.named_buffers()):
            name = canonical_parameter_fqn(name)
            if id(tensor) not in ignored_param_ids and any(token in name for token in fp32_compute_module_names):
                pinned_ids.add(id(tensor))

    def compute_dtype_of(t: torch.Tensor) -> torch.dtype:
        if not t.dtype.is_floating_point:
            return t.dtype
        if id(t) in pinned_ids:
            return torch.float32
        recorded = getattr(t, "_hf_compute_dtype", None)
        if recorded is not None and recorded.is_floating_point:
            return recorded
        # Unhinted fp32 storage under a lower-precision policy is an fp32 master
        # weight -> compute in the policy dtype (intrinsic fp32 handled by #1/#2).
        if policy_dtype is not None and t.dtype == torch.float32 and policy_dtype != torch.float32:
            return policy_dtype
        return t.dtype

    return compute_dtype_of


def fully_shard_by_dtype(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None,
    fp32_compute_module_names: Tuple[str, ...] = (),
    reshard_after_forward: bool | int | None = None,
    ignored_params: set[nn.Parameter] | None = None,
    fully_shard_fn: Callable[..., None] | None = None,
) -> None:
    """Fully shard a module so every parameter computes in its required dtype.

    The intent is simple: compute everything in ``mp_policy.param_dtype`` (e.g. bf16)
    except parameters that must stay in fp32 -- their FSDP unit gets ``param_dtype=fp32``
    while the rest of the module computes in the policy dtype. A parameter "must stay
    fp32" if it is pinned via ``fp32_compute_module_names`` or HF stored it in fp32 (see
    ``_make_compute_dtype_fn`` for the full precedence). This decouples *compute* dtype
    from *storage* dtype, so fp32 master weights (uniform fp32 storage) still compute in
    bf16 for the bulk.

    Implementation: group the module's parameters by their resolved compute dtype and
    shard so each FSDP unit is compute-dtype-uniform. The three cases below differ only
    in sharding granularity:

      * 1 compute dtype  -> shard the whole module once.
      * 2 compute dtypes -> shard the minority-dtype subtrees on their own, then shard
        the parent with the majority dtype (keeps the bulk as one FSDP unit).
      * 3+ compute dtypes -> shard every maximal compute-dtype-uniform subtree on its own.

    Dtype-specific child units are internal to the enclosing module's forward, so they
    preserve their module's natural output dtype. Any enclosing FSDP boundary created
    by this function retains the caller's ``output_dtype`` as its external contract.

    Args:
        fp32_compute_module_names: Parameter/buffer name substrings that must compute in
            fp32 (e.g. ``("_fp32_params",)`` for Qwen3.5's GatedDeltaNet fp32 holder).
            Sourced from the model's ``_keep_in_fp32_modules_strict``. Matched callable
            modules must cast their own inputs when required; their nested FP32 FSDP
            units preserve the parent activation dtype at the module boundary.
        reshard_after_forward: Optional FSDP2 reshard override for this module.
            ``None`` leaves the caller's default FSDP2 behavior unchanged.
        ignored_params: Parameters already owned by another FSDP or parallelism
            unit. They are excluded from dtype grouping and forwarded to the
            enclosing FSDP unit.
        fully_shard_fn: Optional model-specific replacement for ``fully_shard``.
            Every FSDP unit created by this function uses this callback.
    """
    ignored_params = set(ignored_params or ())
    ignored_param_ids = {id(param) for param in ignored_params}
    compute_dtype_of = _make_compute_dtype_fn(
        module,
        mp_policy,
        fp32_compute_module_names,
        ignored_params=ignored_params,
    )

    # FSDP2 requires every param group to be uniform in *storage* (original) dtype
    # -- ``_init_mp_dtypes`` asserts ``{p.orig_dtype}`` is a singleton -- while a group's
    # ``param_dtype`` controls *compute* dtype. These are independent axes, so we group by
    # the (storage, compute) pair: this keeps each FSDP unit storage-uniform (satisfying the
    # assertion even when two different storage dtypes share one compute dtype, e.g. bf16 and
    # fp32 weights both computing in bf16) while still splitting params that need a different
    # compute dtype. ``key[1]`` is the compute dtype used as the unit's ``param_dtype``.
    group_key_of = lambda t: (t.dtype, compute_dtype_of(t))

    # calling _group_params_by_dtype is not optimal here, because we may
    # end up with two traversals over the module, but this code is not in the hot path.
    grouped_params = _group_params_by_dtype(
        module,
        dtype_of=group_key_of,
        ignored_params=ignored_params,
    )
    if len(grouped_params) == 0:
        if ignored_params:
            _call_fully_shard(
                module,
                mesh,
                mp_policy,
                offload_policy,
                reshard_after_forward,
                ignored_params,
                fully_shard_fn,
            )
        return
    elif len(grouped_params) == 1:
        key = next(iter(grouped_params))
        _call_fully_shard(
            module,
            mesh,
            _mp_policy_with_param_dtype(mp_policy, key[1]),
            offload_policy,
            reshard_after_forward,
            ignored_params,
            fully_shard_fn,
        )
    else:
        least_items_key = min(grouped_params.items(), key=lambda x: len(x[1]))[0]
        uniform_subtrees = list(
            iter_maximal_uniform_dtype_subtrees(
                module,
                tensor_pred=lambda tensor: torch.is_floating_point(tensor) and id(tensor) not in ignored_param_ids,
                dtype_of=group_key_of,
                return_paths=True,
            )
        )
        selected_subtrees = [
            (path, key, subtree)
            for path, subtree, key in uniform_subtrees
            if (len(grouped_params) == 2 and key == least_items_key) or len(grouped_params) > 2
        ]

        expected_keys = {least_items_key} if len(grouped_params) == 2 else set(grouped_params)
        expected_param_ids = {id(param) for key in expected_keys for param in grouped_params[key]}
        covered_param_ids = {
            id(param)
            for _, _, subtree in selected_subtrees
            for param in subtree.parameters()
            if id(param) not in ignored_param_ids
        }
        unresolved_param_ids = expected_param_ids - covered_param_ids
        if unresolved_param_ids:
            unresolved_names = [name for name, param in module.named_parameters() if id(param) in unresolved_param_ids]
            raise ValueError(
                "FSDP could not isolate parameters with a distinct dtype from siblings in the same module: "
                f"{', '.join(unresolved_names)}. Place them in a dedicated parameter-owning module."
            )

        for path, key, _ in selected_subtrees:
            subtree_kwargs = {
                "mesh": mesh,
                "mp_policy": get_internal_fsdp_mp_policy(_mp_policy_with_param_dtype(mp_policy, key[1])),
                "offload_policy": offload_policy,
                "reshard_after_forward": reshard_after_forward,
            }
            if ignored_params:
                subtree_kwargs["ignored_params"] = ignored_params
            if fully_shard_fn is not None:
                subtree_kwargs["fully_shard_fn"] = fully_shard_fn
            _fully_shard(_get_module_from_path(module, path), **subtree_kwargs)
        if len(grouped_params) == 2:
            parent_key = next(key for key in grouped_params if key != least_items_key)
            _call_fully_shard(
                module,
                mesh,
                _mp_policy_with_param_dtype(mp_policy, parent_key[1]),
                offload_policy,
                reshard_after_forward,
                ignored_params,
                fully_shard_fn,
            )
        elif ignored_params:
            # Preserve the caller's FSDP ownership boundary after every managed
            # parameter has been assigned to a dtype-specific child unit.
            _call_fully_shard(
                module,
                mesh,
                mp_policy,
                offload_policy,
                reshard_after_forward,
                ignored_params,
                fully_shard_fn,
            )
