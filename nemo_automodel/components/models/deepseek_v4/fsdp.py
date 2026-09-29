# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import torch
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

_DSV4_CLASS_NAMES = {
    "DeepseekV4ForCausalLM",
    "DeepseekV4Model",
    "DeepseekV4Block",
    "DeepseekV4VisionBlock",
    "DeepseekV4VisionTransformer",
}

_DSV4_FP32_MODULE_SUFFIXES = (
    "attn_hc",
    "ffn_hc",
    "hc_head",
    "lm_head",
    "self_attn.sinks_param",
    "self_attn.compressor.wkv",
    "self_attn.compressor.wgate",
    "self_attn.compressor.ape_param",
    "self_attn.compressor.indexer.wkv",
    "self_attn.compressor.indexer.wgate",
    "self_attn.compressor.indexer.ape_param",
    "norm1",
    "norm2",
    "vision.norm",
)

_DSV4_RETURNED_FP32_PARAMETER_CLASS_NAMES = {
    "DeepseekV4FP32Parameter",
}

# These modules own fp32 parameters but explicitly upcast their arithmetic and
# cast the result back to the incoming activation dtype.  Their nested FSDP
# unit must therefore be transparent at the module boundary: casting the input
# and forcing an fp32 output would leak fp32 activations into the following
# bf16 projection.
_DSV4_SELF_CASTING_FP32_MODULE_CLASS_NAMES = {
    "DeepseekV4VisionRMSNorm",
}


def _hca_param_sync_group_from_1d_mesh(mesh):
    """Return the 1D PyTorch FSDP2 group used for HCA graph alignment.

    HCA graph alignment is an FSDP/FSDP2 parameter-sync invariant: ranks that
    synchronize the same sharded HCA parameters must agree on whether the HCA
    compressor path participates in backward. This DeepSeek-V4 wrapper gets
    that domain from its 1D PyTorch FSDP2 mesh. The mesh may be named or
    unnamed; multi-dimensional meshes need an explicit owner dimension to avoid
    reducing across unrelated parallel groups. Until that is available, disable
    HCA graph alignment instead of using a broader or wrong group.
    """
    if mesh is None:
        return None

    mesh_ndim = getattr(mesh, "ndim", None)
    mesh_shape = getattr(mesh, "shape", None)
    mesh_dim_names = getattr(mesh, "mesh_dim_names", None)
    if mesh_ndim is not None:
        is_1d_mesh = mesh_ndim == 1
    elif mesh_shape is not None:
        is_1d_mesh = len(mesh_shape) == 1
    elif mesh_dim_names is not None:
        is_1d_mesh = len(mesh_dim_names) == 1
    else:
        return None
    if not is_1d_mesh:
        return None

    try:
        if mesh.size() <= 1:
            return None
        return mesh.get_group()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _matches_suffix(name: str, suffix: str) -> bool:
    return name == suffix or name.endswith(f".{suffix}")


def _has_fsdp_state(module: nn.Module) -> bool:
    try:
        from torch.distributed.fsdp._fully_shard._fsdp_state import _get_module_fsdp_state
    except ImportError:
        return False

    return _get_module_fsdp_state(module) is not None


def _module_config_model_type(module: nn.Module) -> str | None:
    return getattr(getattr(module, "config", None), "model_type", None)


def _is_deepseek_v4_module(module: nn.Module) -> bool:
    if module.__class__.__name__ in _DSV4_CLASS_NAMES or _module_config_model_type(module) == "deepseek_v4":
        return True

    wrapped = getattr(module, "_checkpoint_wrapped_module", None)
    if wrapped is not None and _is_deepseek_v4_module(wrapped):
        return True

    return any(
        sub.__class__.__name__ in _DSV4_CLASS_NAMES or _module_config_model_type(sub) == "deepseek_v4"
        for sub in module.modules()
        if sub is not module
    )


def _floating_param_dtypes(module: nn.Module) -> set[torch.dtype]:
    return {param.dtype for param in module.parameters() if torch.is_floating_point(param)}


def _fp32_mp_policy(mp_policy, *, preserve_activation_dtype: bool = False):
    if not isinstance(mp_policy, MixedPrecisionPolicy):
        return mp_policy

    return MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=None if preserve_activation_dtype else torch.float32,
        cast_forward_inputs=False if preserve_activation_dtype else mp_policy.cast_forward_inputs,
    )


def _fsdp_kwargs_for_module(module: nn.Module, fsdp_kwargs: dict) -> dict:
    ignored_params = fsdp_kwargs.get("ignored_params")
    if not ignored_params:
        return fsdp_kwargs

    module_param_ids = {id(param) for param in module.parameters()}
    filtered_ignored_params = {param for param in ignored_params if id(param) in module_param_ids}
    if len(filtered_ignored_params) == len(ignored_params):
        return fsdp_kwargs

    filtered_kwargs = dict(fsdp_kwargs)
    if filtered_ignored_params:
        filtered_kwargs["ignored_params"] = filtered_ignored_params
    else:
        filtered_kwargs.pop("ignored_params", None)
    return filtered_kwargs


def _fully_shard_once(module: nn.Module, *, mesh, mp_policy, offload_policy, fp32_policy: bool = False, **fsdp_kwargs):
    if module is None or _has_fsdp_state(module):
        return module

    return fully_shard(
        module,
        mesh=mesh,
        mp_policy=(
            _fp32_mp_policy(
                mp_policy,
                preserve_activation_dtype=module.__class__.__name__ in _DSV4_SELF_CASTING_FP32_MODULE_CLASS_NAMES,
            )
            if fp32_policy
            else mp_policy
        ),
        offload_policy=offload_policy,
        **_fsdp_kwargs_for_module(module, fsdp_kwargs),
    )


def _iter_dsv4_fp32_modules(module: nn.Module):
    seen: set[int] = set()
    for name, submodule in module.named_modules():
        if not name or id(submodule) in seen:
            continue
        # A standalone vision tower names its final norm simply "norm".
        # Match the shared vision norm type as well as paths relative to a model.
        if submodule.__class__.__name__ not in _DSV4_SELF_CASTING_FP32_MODULE_CLASS_NAMES and not any(
            _matches_suffix(name, suffix) for suffix in _DSV4_FP32_MODULE_SUFFIXES
        ):
            continue
        if _floating_param_dtypes(submodule) != {torch.float32}:
            continue
        seen.add(id(submodule))
        yield submodule


def _fp32_module_fsdp_kwargs(module: nn.Module, fsdp_kwargs: dict) -> dict:
    if module.__class__.__name__ not in _DSV4_RETURNED_FP32_PARAMETER_CLASS_NAMES:
        return fsdp_kwargs

    # These holders return their raw parameter tensor to the parent module. If
    # FSDP2 reshards immediately after the holder's tiny forward, the parent can
    # observe a zero-storage tensor while still inside its own forward/autograd.
    if fsdp_kwargs.get("reshard_after_forward") is False:
        return fsdp_kwargs
    holder_kwargs = dict(fsdp_kwargs)
    holder_kwargs["reshard_after_forward"] = False
    return holder_kwargs


def _attach_hca_param_sync_group(module: nn.Module, mesh) -> None:
    process_group = _hca_param_sync_group_from_1d_mesh(mesh)
    for submodule in module.modules():
        setter = getattr(submodule, "_set_hca_param_sync_group", None)
        if submodule.__class__.__name__ == "DeepseekV4Compressor" and setter is not None:
            # The FSDP2 mesh is only known while wrapping. Bind its parameter
            # sync group narrowly to the DeepSeek-V4 HCA compressor instead of
            # adding public config.
            setter(process_group)


def fully_shard_deepseek_v4(module: nn.Module, mesh, mp_policy, offload_policy=None, **fsdp_kwargs):
    """Apply FSDP2 to DeepSeek-V4 without mixing fp32 and bf16 params in one unit.

    This is intentionally model-specific.  DeepSeek-V4 keeps a small set of
    reference-sensitive tensors in fp32, while the existing DeepEP path expects
    the transformer block itself to remain the main FSDP unit.

    Uniform fp32 storage does not necessarily mean fp32 compute: full-parameter
    training commonly keeps fp32 master weights while the FSDP mixed-precision
    policy requests bf16 compute.  In that case, preserve fp32 compute only for
    the explicitly listed reference-sensitive islands and let the parent block
    use the caller's bf16 policy.
    """
    is_dsv4 = _is_deepseek_v4_module(module)
    if is_dsv4:
        _attach_hca_param_sync_group(module, mesh)

    policy_param_dtype = getattr(mp_policy, "param_dtype", None)
    use_all_fp32_fast_path = _floating_param_dtypes(module) == {torch.float32} and (
        not is_dsv4 or policy_param_dtype in (None, torch.float32)
    )
    if use_all_fp32_fast_path:
        fsdp_kwargs = _fp32_module_fsdp_kwargs(module, fsdp_kwargs)
        return _fully_shard_once(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            fp32_policy=True,
            **fsdp_kwargs,
        )

    if not is_dsv4:
        return _fully_shard_once(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            fp32_policy=False,
            **fsdp_kwargs,
        )

    for fp32_module in _iter_dsv4_fp32_modules(module):
        _fully_shard_once(
            fp32_module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            fp32_policy=True,
            **_fp32_module_fsdp_kwargs(fp32_module, fsdp_kwargs),
        )

    ignored_params = set(fsdp_kwargs.get("ignored_params") or ())
    parent_kwargs = dict(fsdp_kwargs)
    if ignored_params:
        parent_kwargs["ignored_params"] = ignored_params

    return _fully_shard_once(
        module,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        fp32_policy=False,
        **parent_kwargs,
    )
