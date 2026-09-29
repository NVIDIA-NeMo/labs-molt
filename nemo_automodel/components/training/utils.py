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

import math
import re
from typing import Iterable, Literal

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate

from nemo_automodel.components.models.common.utils import set_is_first_microbatch, set_is_optim_step
from nemo_automodel.components.training.triton.grad_norm import (
    HAVE_TRITON as HAVE_FUSED_GRAD_NORM,
)
from nemo_automodel.components.training.triton.grad_norm import multi_tensor_absmax, multi_tensor_sumsq
from nemo_automodel.shared.import_utils import safe_import, safe_import_te

_GradNormBackend = Literal["triton", "te"]

# Regex pattern to match expert parameters in GroupedExpertsTE.
# Matches FQNs like:
# - model.layers.X.mlp.experts.gate_up_linear.weight0
# - model.layers.X.mlp.experts.gate_up_linear.bias0
# - model.layers.X.mlp.experts.down_linear.weight0
# - model.layers.X.mlp.experts.down_linear.bias0
_TE_EXPERT_PARAM_PATTERN = re.compile(r"(^|\.)mlp\.experts\.(gate_up_linear|down_linear)\.(weight|bias)\d+")


def _combine_norms(norms: list[torch.Tensor], norm_type: float, target_device: torch.device) -> torch.Tensor:
    if len(norms) == 0:
        return torch.tensor(0.0, dtype=torch.float64, device=target_device)

    norm_stack = torch.stack([n.to(device=target_device, dtype=torch.float64) for n in norms])
    if math.isinf(norm_type):
        return norm_stack.max()

    max_norm = norm_stack.abs().max()
    scale = torch.where(torch.isfinite(max_norm) & max_norm.ne(0), max_norm, torch.ones_like(max_norm))
    return max_norm * norm_stack.div(scale).pow(norm_type).sum().pow(1.0 / norm_type)


def _all_reduce_scalar(
    scalar: torch.Tensor,
    op: torch.distributed.ReduceOp,
    mesh: DeviceMesh,
    mesh_dim: int | None = None,
) -> torch.Tensor:
    """All-reduce a 0-dim norm accumulator over ``mesh``, communicating on the mesh device.

    The norm math stays on the gradients' own device, which under FSDP2
    ``CPUOffloadPolicy`` is CPU while the mesh's process group is NCCL and has no CPU
    backend. Only the scalar hops to ``mesh.device_type`` for the collective and comes
    straight back, so a genuinely-CPU (gloo) mesh is never forced onto an accelerator.

    Args:
        scalar: 0-dim tensor to reduce, on the gradients' device.
        op: Reduction operation.
        mesh: Device mesh whose process group performs the collective.
        mesh_dim: Mesh dimension to reduce over, or None for the whole mesh.

    Returns:
        The reduced scalar on ``scalar``'s original device. Callers must use the return
        value: when the devices differ the reduction is out-of-place.
    """
    group = mesh.get_group(mesh_dim=mesh_dim)
    if scalar.device.type == mesh.device_type:
        torch.distributed.all_reduce(scalar, op=op, group=group)
        return scalar

    comm_scalar = scalar.to(device=mesh.device_type)
    torch.distributed.all_reduce(comm_scalar, op=op, group=group)
    return comm_scalar.to(device=scalar.device)


@torch.no_grad()
def count_tail_padding(labels, ignore_label=-100):
    """Counts the total number of padding token in the tail of labels

    e.g.
        labels = torch.tensor([
            [-100, 1, 1, -100, -100],   # 2 tail -100s
            [-100, -100, 2, 3, 4],      # 0 tail -100s
            [5, 6, -100, -100, -100],   # 3 tail -100s
        ])
        count_tail_padding will return 5. Please do note there's more than 5 ignore labels.
    Args:
        labels (torch.Tensor): the labels
        ignore_label (int, optional): ignore label index. Defaults to -100.

    Returns:
        int: total number of ignored tokens in the `labels` input.
    """
    # Flip along the last dimension (seq_len)
    flipped = labels.flip(dims=[1])
    tail_mask = flipped == ignore_label

    # Compute cumulative product to "break" on first non ignore_label
    prod_mask = torch.cumprod(tail_mask.int(), dim=1)

    # Count tail -100s by summing cumprod mask along the sequence dimension
    return prod_mask.view(-1).sum().item()


def _use_fused_grad_norm(params, norm_type: float) -> bool:
    """Whether the fused multi-tensor reduction applies to this group.

    Only the 2-norm and inf-norm are implemented by the kernel, and the whole
    group has to be CUDA -- a mixed CPU/CUDA group would silently take two
    different reduction paths.
    """
    if not HAVE_FUSED_GRAD_NORM:
        return False
    if not (math.isinf(norm_type) or norm_type == 2.0):
        return False
    return all(p.grad is not None and p.grad.is_cuda for p in params)


def _local_te_l2_norm(gradients: list[torch.Tensor], target_device: torch.device) -> torch.Tensor:
    """Reduce local gradients with Transformer Engine where supported.

    Args:
        gradients: Plain local tensors of arbitrary shape, without DTensor placements.
            Contiguous CUDA FP16/BF16/FP32 tensors use TE, grouped by device and dtype.
            Other layouts, devices, and dtypes use PyTorch's FP64 vector norm. Inputs
            are read-only and may alias parameter gradients.
        target_device: Device for the returned scalar; only scalars move between devices.

    Returns:
        Independent scalar FP64 L2 norm on ``target_device``. TE squares and accumulates
        in FP32; PyTorch retries tiny or non-finite TE results in FP64. This check
        synchronizes the CUDA scalar.

    Raises:
        RuntimeError: If eligible CUDA gradients require TE but TE is unavailable.
    """
    norms: list[torch.Tensor] = []
    te_groups: dict[tuple[torch.device, torch.dtype], list[torch.Tensor]] = {}
    for gradient in gradients:
        if gradient.numel() == 0:
            continue
        if (
            type(gradient) is torch.Tensor
            and gradient.is_cuda
            and gradient.is_contiguous()
            and gradient.dtype in (torch.float16, torch.bfloat16, torch.float32)
        ):
            te_groups.setdefault((gradient.device, gradient.dtype), []).append(gradient)
        elif gradient.dtype in (torch.float64, torch.complex128):
            maximum = gradient.abs().max()
            scale = torch.where(torch.isfinite(maximum) & maximum.ne(0), maximum, torch.ones_like(maximum))
            norms.append((maximum * torch.linalg.vector_norm(gradient / scale)).to(target_device))
        else:
            dtype = torch.complex128 if gradient.is_complex() else torch.float64
            norms.append(torch.linalg.vector_norm(gradient, dtype=dtype).to(target_device))

    if te_groups:
        have_te, _ = safe_import_te()
        have_te_optimizers, te_optimizers = (
            safe_import("transformer_engine.pytorch.optimizers") if have_te else (False, None)
        )
        if not have_te_optimizers:
            raise RuntimeError("Transformer Engine gradient-norm backend requested but unavailable")
        for (device, _), group in te_groups.items():
            overflow = torch.zeros(1, dtype=torch.int32, device=device)
            norm, _ = te_optimizers.multi_tensor_applier(te_optimizers.multi_tensor_l2norm, overflow, [group], False)
            norm = norm.reshape(()).to(dtype=torch.float64)
            finfo = torch.finfo(torch.float32)
            minimum_squared_norm = sum(g.numel() for g in group) * finfo.tiny / finfo.eps
            if not bool(torch.isfinite(norm) & norm.square().gt(minimum_squared_norm)):
                norm = torch.linalg.vector_norm(
                    torch.stack([torch.linalg.vector_norm(g, dtype=torch.float64) for g in group])
                )
            norms.append(norm.to(target_device))

    return _combine_norms(norms, 2.0, target_device)


@torch.no_grad()
def _clip_grad_norm_impl(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
    *,
    grad_norm_backend: _GradNormBackend = "triton",
) -> torch.Tensor:
    """Compute and clip the norm of local and DTensor gradients.

    Args:
        parameters: One parameter tensor or an iterable of parameter tensors
            with arbitrary shapes. DTensors retain their declared mesh and
            placements.
        max_norm: Maximum allowed global gradient norm.
        norm_type: Norm exponent, including ``inf``.
        error_if_nonfinite: Whether to raise for a non-finite global norm.
        foreach: Optional foreach implementation preference for clipping.
        pp_mesh: Optional pipeline mesh over which the scalar norm is reduced.
        grad_norm_backend: Local L2 reducer. ``"triton"`` uses this PR's FP64
            multi-tensor kernel; ``"te"`` uses Transformer Engine where eligible.

    Returns:
        Scalar tensor containing the pre-clipping global gradient norm.
    """
    if grad_norm_backend not in ("triton", "te"):
        raise ValueError(f"grad_norm_backend must be 'triton' or 'te', got {grad_norm_backend!r}")
    if grad_norm_backend == "te" and norm_type != 2.0:
        raise ValueError("The TE gradient-norm backend supports only norm_type=2.0")

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    else:
        parameters = list(parameters)

    # Group parameters by their sharding pattern
    # Key: (device_mesh_id, tuple of placements)
    sharding_groups = {}

    for p in parameters:
        if p.grad is None:
            continue

        if isinstance(p, DTensor):
            # Create a hashable key from device_mesh and placements
            mesh_id = id(p.device_mesh)
            placements_tuple = tuple(str(placement) for placement in p.placements)
            key = (mesh_id, placements_tuple)
        else:
            # Regular tensor - group separately
            key = ("regular", "regular")

        if key not in sharding_groups:
            sharding_groups[key] = []
        sharding_groups[key].append(p)

    target_device = None
    for group_params in sharding_groups.values():
        for p in group_params:
            g = p.grad
            if g is None:
                continue
            if isinstance(g, DTensor):
                g = g.to_local()
            target_device = g.device
            break
        if target_device is not None:
            break
    if target_device is None:
        target_device = torch.device("cpu")

    # Compute norm for each sharding group using a scalar-first reduction:
    # sum(|g_local|^p) locally → single-scalar allreduce per Shard mesh dim.
    # Going through torch.nn.utils.get_total_norm on DTensor grads would stack
    # per-param scalar DTensors into a 1-D DTensor whose local length equals
    # the number of local param tensors in the group. Under EP, that length
    # can differ across ranks, and the vector_norm redistribute (Partial →
    # Replicate) then allreduces with mismatched numel and hangs.
    is_inf = math.isinf(norm_type)
    group_norms = []
    for group_params in sharding_groups.values():
        first = group_params[0]
        is_dtensor = isinstance(first, DTensor)
        # Partial placements can't be reduced via sum-of-local-norms; materialize
        # those per-grad (each full_tensor() is a same-shape collective, safe).
        has_partial = is_dtensor and any(isinstance(pl, Partial) for pl in first.placements)

        if grad_norm_backend == "te" and not has_partial:
            local_gradients = [
                (p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad).detach() for p in group_params
            ]
            local_norm = _local_te_l2_norm(local_gradients, target_device)
            if is_dtensor:
                maximum = local_norm.clone()
                for dim_idx, placement in enumerate(first.placements):
                    if not isinstance(placement, Replicate):
                        maximum = _all_reduce_scalar(
                            maximum, torch.distributed.ReduceOp.MAX, first.device_mesh, dim_idx
                        )
                scale = torch.where(torch.isfinite(maximum) & maximum.ne(0), maximum, torch.ones_like(maximum))
                sum_squares = local_norm.div(scale).square()
                for dim_idx, placement in enumerate(first.placements):
                    if not isinstance(placement, Replicate):
                        sum_squares = _all_reduce_scalar(
                            sum_squares, torch.distributed.ReduceOp.SUM, first.device_mesh, dim_idx
                        )
                local_norm = maximum * sum_squares.sqrt()
            group_norms.append(local_norm)
            continue

        # Fused path: one kernel launch per dtype instead of ~7 per parameter.
        # Restricted to the 2- and inf-norms, the only orders the kernel
        # implements; anything else falls through to the loops below.
        if grad_norm_backend == "triton" and _use_fused_grad_norm(group_params, norm_type):
            locals_ = []
            for p in group_params:
                g = p.grad
                if isinstance(g, DTensor):
                    g = g.full_tensor() if has_partial else g.to_local()
                if g.numel():
                    locals_.append(g.detach())

            if is_inf:
                group_val = multi_tensor_absmax(locals_)
                reduce_op = torch.distributed.ReduceOp.MAX
            else:
                # fp64 accumulation removes the need to divide by the max first
                # (that pass exists only to keep BF16 squares from overflowing),
                # so this is a single pass over the gradients.
                group_val = multi_tensor_sumsq(locals_)
                reduce_op = torch.distributed.ReduceOp.SUM

            if is_dtensor and not has_partial:
                mesh = first.device_mesh
                for dim_idx, pl in enumerate(first.placements):
                    if isinstance(pl, Replicate):
                        continue
                    group_val = _all_reduce_scalar(group_val, reduce_op, mesh, dim_idx)

            group_norms.append(group_val if is_inf else group_val.pow(1.0 / norm_type))
            continue

        local_max = torch.zeros((), dtype=torch.float64, device=target_device)
        for p in group_params:
            g = p.grad
            if isinstance(g, DTensor):
                g = g.full_tensor() if has_partial else g.to_local()
            if g.numel() == 0:
                continue
            g_abs_max = g.detach().abs().max().to(device=target_device, dtype=torch.float64)
            local_max = torch.maximum(local_max, g_abs_max)

        if is_dtensor and not has_partial:
            mesh = first.device_mesh
            for dim_idx, pl in enumerate(first.placements):
                if isinstance(pl, Replicate):
                    continue
                local_max = _all_reduce_scalar(local_max, torch.distributed.ReduceOp.MAX, mesh, dim_idx)

        if is_inf:
            group_norms.append(local_max)
            continue

        scale = torch.where(torch.isfinite(local_max) & local_max.ne(0), local_max, torch.ones_like(local_max))
        local_val = torch.zeros((), dtype=torch.float64, device=target_device)
        for p in group_params:
            g = p.grad
            if isinstance(g, DTensor):
                g = g.full_tensor() if has_partial else g.to_local()
            if g.numel() == 0:
                continue
            g = g.detach().abs().div(scale)
            if norm_type == 2.0:
                local_val = local_val + g.square().sum(dtype=torch.float64)
            else:
                local_val = local_val + g.pow(norm_type).sum(dtype=torch.float64)

        if is_dtensor and not has_partial:
            mesh = first.device_mesh
            for dim_idx, pl in enumerate(first.placements):
                if isinstance(pl, Replicate):
                    continue
                local_val = _all_reduce_scalar(local_val, torch.distributed.ReduceOp.SUM, mesh, dim_idx)

        group_norms.append(local_max * local_val.pow(1.0 / norm_type))

    # Combine norms across groups (all rank-identical scalars, no comm)
    total_norm = _combine_norms(group_norms, norm_type, target_device)

    # Reduce across pipeline parallel mesh if provided
    if pp_mesh is not None:
        if math.isinf(norm_type):
            total_norm = _all_reduce_scalar(total_norm, torch.distributed.ReduceOp.MAX, pp_mesh)
        else:
            pp_max_norm = total_norm.abs().clone()
            pp_max_norm = _all_reduce_scalar(pp_max_norm, torch.distributed.ReduceOp.MAX, pp_mesh)
            scale = torch.where(
                torch.isfinite(pp_max_norm) & pp_max_norm.ne(0), pp_max_norm, torch.ones_like(pp_max_norm)
            )
            total_norm = total_norm.div(scale).pow(norm_type)
            total_norm = _all_reduce_scalar(total_norm, torch.distributed.ReduceOp.SUM, pp_mesh)
            total_norm = pp_max_norm * total_norm.pow(1.0 / norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from `parameters` is non-finite, "
            "so it cannot be clipped."
        )

    # Clip gradients for each sharding group separately
    # This is necessary because clip_grads_with_norm_ doesn't support mixing tensors from different device meshes
    for group_params in sharding_groups.values():
        torch.nn.utils.clip_grads_with_norm_(group_params, max_norm, total_norm, foreach)

    return total_norm


@torch.no_grad()
def clip_grad_norm(
    max_grad_norm: float | None,
    model_parts: list[torch.nn.Module],
    *,
    norm_type: float = 2.0,
    pp_enabled: bool = False,
    device_mesh: DeviceMesh | None = None,
    pp_axis_name: str | None = None,
    foreach: bool = True,
    use_torch_clip_grad_norm: bool = False,
    grad_norm_backend: _GradNormBackend = "triton",
) -> torch.Tensor | float:
    """Apply sharding-aware gradient clipping.

    Handles all parallelism strategies (TP, PP, EP/MoE) with automatic sharding-aware grouping.
    Returns the gradient norm as a scalar tensor on the gradients' device, or 0.0 if clipping is skipped.
    This function does not synchronize TP-replicated gradients; optimizer loops
    must do that exactly once before calling this function.

    This function automatically:
    - Groups parameters by sharding pattern (device mesh + placements)
    - Computes norms correctly across different sharding strategies
    - Handles MoE with separate DP/EP meshes
    - Reduces norms across pipeline parallel stages when enabled

    Args:
        max_grad_norm: Maximum gradient norm. If None, skips clipping.
        model_parts: List of model modules to clip.
        norm_type: Type of norm to use (default: 2.0 for L2).
        pp_enabled: Whether pipeline parallelism is enabled.
        device_mesh: Device mesh for parallelism.
        pp_axis_name: Pipeline parallel axis name.
        foreach: Whether to use foreach implementation for clipping.
        use_torch_clip_grad_norm: Use PyTorch's optimized regular-tensor clipping path when possible.
        grad_norm_backend: Local L2 reducer, either ``"triton"`` or ``"te"``.

    Returns:
        Scalar tensor containing the total gradient norm without synchronizing it to the host,
        or 0.0 when clipping is disabled.
    """
    if max_grad_norm is None:
        return 0.0

    # Collect all parameters
    parameters = [p for m in model_parts for p in m.parameters() if p.requires_grad]

    # Determine pp_mesh if PP is enabled
    pp_mesh = None
    if pp_enabled:
        assert pp_axis_name is not None, "pp_axis_name must be provided when pp_enabled is True"
        pp_mesh = device_mesh[pp_axis_name] if device_mesh is not None else None

    can_use_torch_clip = use_torch_clip_grad_norm and grad_norm_backend == "triton" and pp_mesh is None
    if can_use_torch_clip:
        for p in parameters:
            if (
                isinstance(p, DTensor)
                or isinstance(p.grad, DTensor)
                or getattr(p, "_nemo_model_owned_grad_divisor", None) is not None
            ):
                can_use_torch_clip = False
                break

    if can_use_torch_clip:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_grad_norm,
            norm_type=norm_type,
            error_if_nonfinite=False,
            foreach=foreach,
        )
    else:
        # Use the sharding-aware implementation for DTensor, PP, EP, and mixed placement cases.
        grad_norm = _clip_grad_norm_impl(
            parameters=parameters,
            max_norm=max_grad_norm,
            norm_type=norm_type,
            error_if_nonfinite=False,
            foreach=foreach,
            pp_mesh=pp_mesh,
            grad_norm_backend=grad_norm_backend,
        )

    return grad_norm


def prepare_for_grad_accumulation(model_parts: list[torch.nn.Module], pp_enabled: bool = False):
    """Prepare model parts before starting gradient accumulation.

    This is typically called once at the start of gradient accumulation to prepare
    FSDP states for the upcoming forward and backward passes.

    Args:
        model_parts: List of model parts (modules) to prepare.
        pp_enabled: Whether pipeline parallelism is enabled.
    """
    set_is_optim_step(False)
    set_is_first_microbatch(True)
    if pp_enabled:
        return

    for mp in model_parts:
        if hasattr(mp, "prepare_for_grad_accumulation"):
            mp.prepare_for_grad_accumulation(pp_enabled=pp_enabled)


def prepare_for_final_backward(model_parts: list[torch.nn.Module], pp_enabled: bool = False):
    """Prepare model parts before the final backward pass.

    This is typically called before the final gradient accumulation step to prepare
    FSDP states for gradient synchronization and resharding.

    Args:
        model_parts: List of model parts (modules) to prepare.
        pp_enabled: Whether pipeline parallelism is enabled.
    """
    set_is_optim_step(True)
    if pp_enabled:
        return

    for mp in model_parts:
        if hasattr(mp, "prepare_for_final_backward"):
            mp.prepare_for_final_backward(pp_enabled=pp_enabled)


def get_expert_tp_replication_factor(
    model_parts: list[torch.nn.Module],
    device_mesh: DeviceMesh | None,
) -> int:
    """Return the TP token-replication factor for custom-MoE expert gradients.

    The custom-MoE tensor-parallel path keeps the token path (attention,
    router) replicated across TP ranks, so every TP rank feeds the same tokens
    into the expert-parallel all-gather and each expert gradient is accumulated
    ``tp_size`` times. ``scale_grads_and_clip_grad_norm`` divides expert
    gradients by this factor to restore the correct scale.
    """
    if device_mesh is None or "tp" not in (device_mesh.mesh_dim_names or ()):
        return 1
    tp_size = device_mesh["tp"].size()
    if tp_size <= 1:
        return 1
    if any(getattr(part, "_nemo_moe_tp_requires_replica_sync", False) for part in model_parts):
        return int(tp_size)
    return 1


@torch.no_grad()
def scale_grads_and_clip_grad_norm(
    max_grad_norm: float | None,
    model_parts: list[torch.nn.Module],
    *,
    norm_type: float = 2.0,
    pp_enabled: bool = False,
    device_mesh: DeviceMesh | None = None,
    moe_mesh: DeviceMesh | None = None,
    ep_axis_name: str | None = None,
    pp_axis_name: str | None = None,
    foreach: bool = True,
    num_label_tokens: int | None = None,
    dp_group_size: int | None = None,
    expert_tp_replication_factor: int = 1,
    use_torch_clip_grad_norm: bool = False,
    grad_norm_backend: _GradNormBackend = "triton",
) -> torch.Tensor | float:
    """Scale gradients for PP/EP and model-owned shards, then clip.

    The caller must synchronize TP-replicated gradients once after accumulation
    and before calling this function. This helper does not synchronize replicas.

    - PP scaling: divide all local grads by (num_label_tokens / dp_group_size).
    - EP scaling: for parameters on the expert axis, divide grads by
      ``(dp_group_size / ep_shard_size) * expert_tp_replication_factor``.
    - Owner-sharded scaling: divide each marked gradient by the explicit factor
      declared by its model-owned sharding contract.
    - Finally, perform grad clipping with PP/EP-aware reductions.

    Args:
        max_grad_norm: Maximum global gradient norm, or None to skip clipping.
        model_parts: Model modules whose parameters have gradients of arbitrary shape.
            Gradients retain their original local or DTensor layout and are scaled in place.
        norm_type: Norm order.
        pp_enabled: Whether pipeline-parallel normalization is required.
        device_mesh: Training mesh used for gradient norm reductions.
        moe_mesh: Expert-parallel mesh used to normalize expert gradients.
        ep_axis_name: Expert axis in the parameter mesh.
        pp_axis_name: Pipeline axis in the training mesh.
        foreach: Whether to use foreach for in-place clipping.
        num_label_tokens: Global supervised-token count for PP normalization.
        dp_group_size: Data-parallel group size, including CP when configured.
        expert_tp_replication_factor: Number of identical TP copies of expert tokens.
        use_torch_clip_grad_norm: Prefer PyTorch's regular-tensor clipping fast path.
        grad_norm_backend: Local L2 reducer, either ``"triton"`` or ``"te"``.

    Returns:
        Scalar tensor containing the total gradient norm without synchronizing it to the host,
        or 0.0 when clipping is disabled.
    """

    # Precompute scale factors
    pp_divisor: float | None = None
    if pp_enabled and num_label_tokens is not None and dp_group_size is not None:
        if dp_group_size != 0:
            candidate = num_label_tokens / dp_group_size
            pp_divisor = float(candidate) if candidate != 0 else None

    if not isinstance(expert_tp_replication_factor, int) or isinstance(expert_tp_replication_factor, bool):
        raise TypeError("expert_tp_replication_factor must be an integer")
    if expert_tp_replication_factor < 1:
        raise ValueError("expert_tp_replication_factor must be >= 1")

    ep_ratio: float | None = None
    if moe_mesh is not None and dp_group_size is not None:
        ep_shard_size = moe_mesh["ep_shard"].size() if "ep_shard" in moe_mesh.mesh_dim_names else 1
        if ep_shard_size > 0:
            ep_ratio = float(dp_group_size) / float(ep_shard_size)
            ep_ratio *= float(expert_tp_replication_factor)

    has_model_owned_sharded_params = any(
        getattr(parameter, "_nemo_model_owned_grad_divisor", None) is not None
        for model_part in model_parts
        for parameter in model_part.parameters()
    )

    # Single pass over parameters to apply both scalings where applicable
    if pp_divisor is not None or ep_ratio is not None or has_model_owned_sharded_params:
        for mp in model_parts:
            for name, p in mp.named_parameters():
                if p.grad is None:
                    continue
                if pp_divisor is not None:
                    p.grad.div_(pp_divisor)
                owner_divisor = getattr(p, "_nemo_model_owned_grad_divisor", None)
                if owner_divisor is not None:
                    p.grad.div_(float(owner_divisor))
                if ep_ratio is not None:
                    # Scale expert gradients by the FSDP/EP ratio and by any
                    # identical TP token replicas that were gathered inside EP.
                    # DTensor experts: check device mesh for EP sharding axis
                    # Non-DTensor experts (e.g., DeepEP): check param name
                    is_ep_sharded_dtensor = (
                        isinstance(p, DTensor)
                        and isinstance(p.grad, DTensor)
                        and ep_axis_name
                        and ep_axis_name in p.device_mesh.mesh_dim_names
                    )
                    is_expert_param = (
                        isinstance(p, torch.Tensor)
                        and isinstance(p.grad, torch.Tensor)
                        and _TE_EXPERT_PARAM_PATTERN.search(name) is not None
                    )
                    if owner_divisor is None and (is_ep_sharded_dtensor or is_expert_param):
                        p.grad.div_(ep_ratio)

    # Clip with the existing PP/EP-aware helper
    return clip_grad_norm(
        max_grad_norm,
        model_parts,
        norm_type=norm_type,
        pp_enabled=pp_enabled,
        device_mesh=device_mesh,
        pp_axis_name=pp_axis_name,
        foreach=foreach,
        use_torch_clip_grad_norm=use_torch_clip_grad_norm,
        grad_norm_backend=grad_norm_backend,
    )


