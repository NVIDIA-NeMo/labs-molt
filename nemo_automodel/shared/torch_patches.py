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
"""
Torch compatibility patches.

These patches are intentionally NOT applied at `import nemo_automodel` time to keep
tokenizer-only imports lightweight. Call `apply_torch_patches()` from code paths
that already depend on torch (training / distributed / dataloading).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

_logger = logging.getLogger(__name__)

_TORCH_PATCHES_APPLIED = False


def _widest_float_dtype(dtypes: Iterable[Any]) -> Any:
    """Return the float dtype among ``dtypes`` that every other one converts into losslessly.

    Args:
        dtypes: Gradient dtypes from a single reduce-scatter group.

    Returns:
        The dtype with the largest element size; ties resolve to float32 over
        the 2-byte float types.
    """
    import torch

    return max(dtypes, key=lambda dtype: (torch.finfo(dtype).bits, dtype is torch.float32))


def patch_fsdp_uniform_reduce_dtype() -> None:
    """Give every FSDP2 reduce-scatter group local gradients of one dtype.

    Gradient accumulation leaves a group holding ``reduce_dtype`` accumulations
    for the parameters used so far, while any parameter whose gradient joins
    later -- a locally unused parameter zero-filled by PyTorch's public API or
    :func:`patch_fsdp_unused_param_reduction`, or one whose gradient lands after
    its group's post-backward already ran -- contributes ``param_dtype``.
    ``foreach_reduce`` then aborts with ``FSDP reduce-scatter expects uniform
    gradient dtype``.

    Normalize and widen gradients at the last possible moment, inside
    ``foreach_reduce`` itself. That placement matters:

    * ``FSDPParam`` normally unwraps gradients through
      ``_get_grad_inner_tensor``. PyTorch versions whose public unused-parameter
      API appends ``zeros_like(unsharded_param)`` directly can still leave a
      ``DTensor`` in this list, so unwrap that residual value before sizing the
      reduce-scatter buffer;
    * FSDP2's own bookkeeping (``unsharded_param.grad`` /
      ``unsharded_accumulated_grad``) is left exactly as upstream leaves it, so
      no later reader of that state sees anything unusual;
    * ``foreach_reduce`` immediately copies these gradients into a
      ``reduce_dtype`` buffer anyway, so widening first changes no value.

    Uniform groups are passed straight through, so the upstream assertion still
    fires for genuinely inconsistent gradients such as fp8 weights that fail to
    produce higher-precision ones. The patch is process-global and idempotent.
    """
    try:
        import torch.distributed.fsdp._fully_shard._fsdp_collectives as collectives
        import torch.distributed.fsdp._fully_shard._fsdp_param_group as param_group
    except ImportError:
        return

    original_foreach_reduce = collectives.foreach_reduce
    if getattr(original_foreach_reduce, "_automodel_uniform_reduce_dtype", False):
        return

    def foreach_reduce_uniform_dtype(fsdp_params, unsharded_grads, *args, **kwargs):
        from torch.distributed.tensor import DTensor

        # PyTorch 2.13a0's unused-parameter branch can append a DTensor zero
        # directly, while gradients from used parameters are already local
        # tensors. Besides making ``fsdp.chunk_cat`` reject the mixed list, the
        # DTensor's global numel makes FSDP size a global-shape staging buffer.
        # Current PyTorch routes the zero through ``_get_grad_inner_tensor``;
        # localizing here is the equivalent compatibility path for that build.
        unsharded_grads[:] = [grad.to_local() if isinstance(grad, DTensor) else grad for grad in unsharded_grads]
        dtypes = {grad.dtype for grad in unsharded_grads}
        if len(dtypes) > 1 and all(dtype.is_floating_point for dtype in dtypes):
            target = _widest_float_dtype(dtypes)
            # Mutate in place: ``foreach_reduce`` frees the gradients by clearing
            # this list, and that must still release the caller's references.
            unsharded_grads[:] = [grad if grad.dtype is target else grad.to(target) for grad in unsharded_grads]
        return original_foreach_reduce(fsdp_params, unsharded_grads, *args, **kwargs)

    foreach_reduce_uniform_dtype._automodel_uniform_reduce_dtype = True
    collectives.foreach_reduce = foreach_reduce_uniform_dtype
    param_group.foreach_reduce = foreach_reduce_uniform_dtype


def patch_fsdp_unused_param_reduction() -> None:
    """Backport FSDP2 unused-parameter reduction when the public API is absent.

    The patch is process-global and idempotent. It only fills a missing local
    gradient with zeros immediately before FSDP2 post-backward reduction, so
    ranks that skipped a parameter still participate in the same collective as
    ranks that used it. Callers must first prefer the public
    ``FSDPModule.set_reduce_scatter_unused_params`` API.

    Raises:
        RuntimeError: If the installed PyTorch exposes neither the public API
            nor the compatible private FSDP2 implementation.
    """
    try:
        import torch
        from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
        from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup
    except ImportError as error:
        raise RuntimeError(
            "Context parallelism requires FSDP unused-parameter reduction, but this PyTorch "
            "version provides neither the public API nor the compatible FSDP2 implementation."
        ) from error

    original_post_backward = FSDPParamGroup.post_backward
    if getattr(original_post_backward, "_automodel_reduce_scatter_unused_params", False):
        return

    def _post_backward_with_unused_param_reduction(self, *args, **kwargs):
        if self.reduce_grads and self._training_state == TrainingState.PRE_BACKWARD:
            for fsdp_param in self.fsdp_params:
                if not hasattr(fsdp_param, "_unsharded_param"):
                    continue
                if fsdp_param.unsharded_accumulated_grad is not None:
                    continue
                param = fsdp_param.unsharded_param
                if param.requires_grad and param.grad is None:
                    # ``zeros_like`` on the *unsharded* parameter is the dtype
                    # autograd would have produced under any precision policy
                    # (bf16 compute over fp32 storage, an fp32-pinned unit, or no
                    # casting at all). ``_align_accumulated_grad_dtype`` then
                    # promotes it to ``reduce_dtype`` if the group is accumulating.
                    param.grad = torch.zeros_like(param, memory_format=torch.preserve_format)
        return original_post_backward(self, *args, **kwargs)

    _post_backward_with_unused_param_reduction._automodel_reduce_scatter_unused_params = True
    FSDPParamGroup.post_backward = _post_backward_with_unused_param_reduction


def patch_fsdp_accumulated_grad_guard() -> None:
    """Guard FSDP2 post-backward against params that were never unsharded.

    PyTorch FSDP2 creates ``_unsharded_param`` lazily from an FSDP unit's
    forward pre-hook. If a separately wrapped unit is skipped by the batch
    (for example a vision tower on text-only data), deferred post-backward can
    dereference that missing field. Missing lazy state means there is no
    unsharded grad to upcast, so the exact missing-field case can return early.
    """
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    except Exception:
        return

    orig = FSDPParam.to_accumulated_grad_if_needed
    if getattr(orig, "_nemo_automodel_guarded", False):
        return

    def guarded(self):
        try:
            return orig(self)
        except AttributeError as exc:
            if "_unsharded_param" not in str(exc) or hasattr(self, "_unsharded_param"):
                raise
            return None

    setattr(guarded, "_nemo_automodel_guarded", True)
    # Preserve the previous Gemma4 marker name for callers/tests that only need
    # idempotency and do not care which entry point installed the patch.
    setattr(guarded, "_gemma4_guarded", True)
    FSDPParam.to_accumulated_grad_if_needed = guarded


def apply_torch_patches() -> None:
    """
    Apply small, version/packaging-specific torch monkey patches.

    This function is idempotent and safe to call multiple times.
    """
    global _TORCH_PATCHES_APPLIED
    if _TORCH_PATCHES_APPLIED:
        return

    try:
        import torch as _torch
    except Exception:
        # torch not installed or failing to import: nothing to patch.
        return

    # -------------------------------------------------------------------------
    # Patch #1: torchdata compatibility
    # Monkey patch pin_memory to optionally accept a device argument.
    # The device argument was removed in some newer torch versions but torchdata
    # still passes it in some versions.
    # -------------------------------------------------------------------------
    try:
        import functools
        import inspect

        from torch.utils.data import _utils as torch_data_utils

        _original_pin_memory_loop = torch_data_utils.pin_memory._pin_memory_loop
        _original_pin_memory = torch_data_utils.pin_memory.pin_memory
        _original_pin_memory_sig = inspect.signature(_original_pin_memory)

        if "device" not in _original_pin_memory_sig.parameters:

            @functools.wraps(_original_pin_memory)
            def _patched_pin_memory(data, device=None):
                return _original_pin_memory(data)

            @functools.wraps(_original_pin_memory_loop)
            def _pin_memory_loop(in_queue, out_queue, device_id, done_event, device):
                return _original_pin_memory_loop(in_queue, out_queue, device_id, done_event)

            torch_data_utils.pin_memory.pin_memory = _patched_pin_memory
            torch_data_utils.pin_memory._pin_memory_loop = _pin_memory_loop

    except Exception as e:
        _logger.debug(f"Could not apply torch pin_memory patch: {e}")

    # -------------------------------------------------------------------------
    # Patch #2: DeviceMesh slicing corner case (specific PyTorch regression)
    # Fixes issue where _dim_group_names is accessed without checking if rank is in mesh.
    # Based on https://github.com/pytorch/pytorch/pull/169454/files
    # -------------------------------------------------------------------------
    try:
        # Only apply the patch for the specific PyTorch version with the regression
        # TODO: Remove this once bump up to a newer PyTorch version with the fix
        if "2.10.0" in _torch.__version__ and "nv25.11" in _torch.__version__:
            from torch.distributed._mesh_layout import _MeshLayout
            from torch.distributed.device_mesh import _MeshEnv

            def _patched_get_slice_mesh_layout(self, device_mesh, mesh_dim_names):
                # 1. Build the layout manually to bypass the legacy 'stride < pre_stride' check
                slice_from_root = device_mesh == self.get_root_mesh(device_mesh)
                flatten_name_to_root_layout = (
                    {
                        key: mesh._layout
                        for key, mesh in self.root_to_flatten_mapping.setdefault(device_mesh, {}).items()
                    }
                    if slice_from_root
                    else {}
                )

                mesh_dim_names_list = getattr(device_mesh, "mesh_dim_names", [])
                valid_mesh_dim_names = [*mesh_dim_names_list, *flatten_name_to_root_layout]
                if not all(name in valid_mesh_dim_names for name in mesh_dim_names):
                    raise KeyError(f"Invalid mesh_dim_names {mesh_dim_names}. Valid: {valid_mesh_dim_names}")

                layout_sliced = []
                for name in mesh_dim_names:
                    if name in mesh_dim_names_list:
                        layout_sliced.append(device_mesh._layout[mesh_dim_names_list.index(name)])
                    elif name in flatten_name_to_root_layout:
                        layout_sliced.append(flatten_name_to_root_layout[name])

                sliced_sizes = tuple(layout.sizes for layout in layout_sliced)
                sliced_strides = tuple(layout.strides for layout in layout_sliced)

                # Bypass the 'stride < pre_stride' check that exists in the original and create MeshLayout directly.
                slice_mesh_layout = _MeshLayout(sliced_sizes, sliced_strides)

                if not slice_mesh_layout.check_non_overlap():
                    raise RuntimeError(f"Slicing overlapping dim_names {mesh_dim_names} is not allowed.")

                # 2. Replicate the _dim_group_names fix (commit f6c8092)
                if hasattr(device_mesh, "_dim_group_names") and len(device_mesh._dim_group_names) > 0:
                    slice_dim_group_name = []
                    submesh_dim_names = mesh_dim_names if isinstance(mesh_dim_names, tuple) else (mesh_dim_names,)
                    for name in submesh_dim_names:
                        if name in mesh_dim_names_list:
                            slice_dim_group_name.append(device_mesh._dim_group_names[mesh_dim_names_list.index(name)])
                        elif hasattr(device_mesh, "_flatten_mapping") and name in device_mesh._flatten_mapping:
                            flatten_mesh = device_mesh._flatten_mapping[name]
                            slice_dim_group_name.append(
                                flatten_mesh._dim_group_names[flatten_mesh.mesh_dim_names.index(name)]
                            )

                    object.__setattr__(slice_mesh_layout, "_dim_group_names", slice_dim_group_name)

                return slice_mesh_layout

            _MeshEnv._get_slice_mesh_layout = _patched_get_slice_mesh_layout
            _logger.debug(f"Applied DeviceMesh fix for PyTorch {_torch.__version__}")

    except (ImportError, AttributeError) as e:
        _logger.debug(f"Could not apply DeviceMesh patch: {e}")

    # -------------------------------------------------------------------------
    # Patch #3: aten.alias.default sharding strategy (PyTorch 2.9 regression)
    # torch.ops.aten.alias.default has no sharding strategy registered in
    # PyTorch 2.9.0, causing NotImplementedError when DTensor dispatches
    # through aten.alias (e.g. via HF Qwen3's logits_to_keep slice).
    # See https://github.com/pytorch/pytorch/pull/166867 for the upstream fix.
    # Remove this patch once we upgrade to a torch version that includes it.
    # -------------------------------------------------------------------------
    try:
        from packaging.version import parse as _vparse

        if _vparse(_torch.__version__).base_version == "2.9.0":
            from torch.distributed.tensor._ops._tensor_ops import propagate_single_input_strategy
            from torch.distributed.tensor._ops.utils import register_op_strategy

            register_op_strategy(_torch.ops.aten.alias.default)(propagate_single_input_strategy)
            _logger.debug("Applied aten.alias.default sharding strategy patch for PyTorch 2.9.0")
    except Exception as e:
        _logger.debug(f"Could not apply aten.alias.default sharding strategy patch: {e}")

    _TORCH_PATCHES_APPLIED = True
