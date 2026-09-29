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

import logging
import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import (
    BLOCK_SIZE,
    create_scale_inv_for_weight,
    dequantize_from_fp8,
)
from nemo_automodel.components.models.glm4_moe.state_dict_adapter import Glm4MoeStateDictAdapter
from nemo_automodel.components.moe.state_dict_utils import is_dtensor

try:
    import triton
    import triton.language as tl

    _GLM_TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _GLM_TRITON_AVAILABLE = False

logger = logging.getLogger(__name__)

# GLM-5.3 keeps these tensors in BF16 while storing the remaining matrix
# weights as 128x128 blockwise FP8 with ``weight_scale_inv`` siblings.
NON_QUANTIZED_KEY_PATTERNS = [
    "norm.weight",
    "lm_head.weight",
    "embed_tokens.weight",
    "mlp.gate.weight",
    "eh_proj.weight",
    "indexer.weights_proj.weight",
]


def should_quantize_key(key: str) -> bool:
    """Return whether ``key`` has a blockwise-FP8 scale in GLM checkpoints."""
    if not key.endswith(".weight") or ".lora_" in key:
        return False
    return not any(pattern in key for pattern in NON_QUANTIZED_KEY_PATTERNS)


if _GLM_TRITON_AVAILABLE:

    @triton.jit
    def _glm_weight_dequant_offset_kernel(
        x_ptr,
        s_ptr,
        y_ptr,
        M,
        N,
        row_offset,
        col_offset,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Dequantize a local GLM shard whose origin is inside a global FP8 block."""
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=mask).to(tl.float32)
        scale_m = (offs_m + row_offset) // BLOCK_SIZE
        scale_n = (offs_n + col_offset) // BLOCK_SIZE
        s = tl.load(
            s_ptr + scale_m[:, None] * stride_sm + scale_n[None, :] * stride_sn,
            mask=mask,
            other=0.0,
        )
        y = x * s
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, y, mask=mask)


def _glm_dtensor_local_offsets(
    weight_dtensor: torch.Tensor,
    weight_local: torch.Tensor,
    scale_inv: torch.Tensor,
    block_size: int,
) -> tuple[int, int]:
    """Return the exact global row/column origin of a GLM DTensor shard.

    Args:
        weight_dtensor: Global two-dimensional DTensor checkpoint destination.
        weight_local: Rank-local weight shard with shape ``[rows, cols]``.
        scale_inv: Full global blockwise scale grid. Used only by the fallback
            placement calculation when checkpoint chunk metadata is unavailable.
        block_size: Height and width of each square FP8 quantization block.

    Returns:
        Absolute element offset ``(row, col)`` of the local shard in the global weight.
    """
    create_chunks = getattr(weight_dtensor, "__create_chunk_list__", None)
    if callable(create_chunks):
        try:
            chunks = create_chunks()
            if isinstance(chunks, (list, tuple)) and len(chunks) == 1:
                offsets = tuple(int(value) for value in chunks[0].offsets)
                sizes = tuple(int(value) for value in chunks[0].sizes)
                if sizes != tuple(weight_local.shape):
                    raise RuntimeError(
                        f"DTensor checkpoint metadata reports local shape {sizes}, "
                        f"but to_local() returned {tuple(weight_local.shape)}"
                    )
                if len(offsets) != 2:
                    raise ValueError(f"GLM blockwise FP8 dequantization requires a matrix, got offsets {offsets}")
                return offsets[0], offsets[1]
        except (AttributeError, TypeError):
            # Lightweight unit-test doubles may not expose checkpoint metadata.
            pass

    from torch.distributed._tensor import Shard

    offsets = [0, 0]
    global_shape = getattr(weight_dtensor, "shape", None)
    has_global_shape = isinstance(global_shape, (torch.Size, tuple, list)) and len(global_shape) == 2
    for mesh_dim, placement in enumerate(weight_dtensor.placements):
        if not isinstance(placement, Shard):
            continue
        shard_dim = placement.dim
        mesh_dim_size = weight_dtensor.device_mesh.size(mesh_dim)
        mesh_coord = weight_dtensor.device_mesh.get_local_rank(mesh_dim=mesh_dim)
        global_size = int(global_shape[shard_dim]) if has_global_shape else int(scale_inv.shape[shard_dim]) * block_size
        _, shard_offset = placement._local_shard_size_and_offset(global_size, mesh_dim_size, mesh_coord)
        offsets[shard_dim] = int(shard_offset)
    return offsets[0], offsets[1]


def _slice_glm_scale_for_dtensor(
    scale_inv: torch.Tensor,
    weight_dtensor: torch.Tensor,
    weight_local: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Slice the global GLM scale grid for one rank-local weight shard.

    Args:
        scale_inv: Full global scale grid with shape ``[global_block_rows, global_block_cols]``.
        weight_dtensor: Global two-dimensional DTensor checkpoint destination.
        weight_local: Rank-local FP8 weight shard with shape ``[rows, cols]``.
        block_size: Height and width of each square FP8 quantization block.

    Returns:
        Contiguous scale grid covering every global block intersected by the local shard.
    """
    offsets = _glm_dtensor_local_offsets(weight_dtensor, weight_local, scale_inv, block_size)
    scale_slices = []
    for dim, offset in enumerate(offsets):
        start_block = offset // block_size
        end_block = (offset + weight_local.shape[dim] + block_size - 1) // block_size
        scale_slices.append(slice(start_block, min(end_block, scale_inv.shape[dim])))
    return scale_inv[scale_slices[0], scale_slices[1]].contiguous()


def _dequantize_glm_with_torch_offsets(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
    offsets_within_first_block: tuple[int, int],
) -> torch.Tensor:
    """Dequantize a GLM local shard while preserving the global FP8 block grid.

    Args:
        weight: Local FP8 weight shard with shape ``[rows, cols]``.
        scale_inv: Sliced scale grid with shape ``[block_rows, block_cols]``, where
            each dimension covers the local shard plus its offset into the first block.
        dtype: Output floating-point dtype.
        block_size: Height and width of each square FP8 quantization block.
        offsets_within_first_block: Element offset ``(row, col)`` of the shard origin
            within its first global block. Each value is in ``[0, block_size)``.

    Returns:
        Dequantized local tensor with the same shape as ``weight``.
    """
    row_offset, col_offset = offsets_within_first_block
    row_scale_ids = torch.div(
        torch.arange(weight.shape[0], device=weight.device) + row_offset,
        block_size,
        rounding_mode="floor",
    )
    col_scale_ids = torch.div(
        torch.arange(weight.shape[1], device=weight.device) + col_offset,
        block_size,
        rounding_mode="floor",
    )
    element_scales = scale_inv.index_select(0, row_scale_ids).index_select(1, col_scale_ids)
    return (weight.float() * element_scales.float()).to(dtype=dtype)


def _dequantize_glm_with_triton_offsets(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
    offsets_within_first_block: tuple[int, int],
) -> torch.Tensor:
    """Run offset-aware GLM FP8 dequantization with Triton.

    Args:
        weight: Local CUDA FP8 weight shard with shape ``[rows, cols]``.
        scale_inv: Sliced CUDA scale grid covering the shard's intersecting global blocks.
        dtype: Output floating-point dtype.
        block_size: Height and width of each square FP8 quantization block.
        offsets_within_first_block: Element offset ``(row, col)`` of the shard origin
            within its first global block. Each value is in ``[0, block_size)``.

    Returns:
        Dequantized local CUDA tensor with the same shape as ``weight``.
    """
    if not _GLM_TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for GLM FP8 dequantization.")

    m, n = weight.shape
    output = torch.empty((m, n), device=weight.device, dtype=dtype)
    grid = (triton.cdiv(m, block_size), triton.cdiv(n, block_size))
    _glm_weight_dequant_offset_kernel[grid](
        weight,
        scale_inv,
        output,
        m,
        n,
        offsets_within_first_block[0],
        offsets_within_first_block[1],
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        scale_inv.stride(0),
        scale_inv.stride(1),
        BLOCK_SIZE=block_size,
    )
    return output


def _dequantize_glm_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
    block_size: int = BLOCK_SIZE,
    name: str = "",
) -> torch.Tensor:
    """Dequantize GLM FP8 weights without changing other model adapters.

    Args:
        weight: FP8 matrix or DTensor matrix. A DTensor uses its global shape while
            ``to_local()`` supplies the rank-local ``[rows, cols]`` shard.
        scale_inv: Matching blockwise scales. For a DTensor weight this is the full
            global scale grid loaded from the Hugging Face checkpoint.
        dtype: Output floating-point dtype.
        block_size: Height and width of each square FP8 quantization block.
        name: Checkpoint tensor name used in diagnostics.

    Returns:
        Dequantized tensor preserving the input DTensor mesh and placements when present.
    """
    weight_is_dtensor = is_dtensor(weight)
    scale_is_dtensor = is_dtensor(scale_inv)
    if not weight_is_dtensor or scale_is_dtensor:
        return dequantize_from_fp8(weight, scale_inv, dtype=dtype, BLOCK_SIZE=block_size, name=name)

    weight_local = weight.to_local()
    global_offsets = _glm_dtensor_local_offsets(weight, weight_local, scale_inv, block_size)
    offsets_within_first_block = (
        global_offsets[0] % block_size,
        global_offsets[1] % block_size,
    )
    scale_local = _slice_glm_scale_for_dtensor(scale_inv, weight, weight_local, block_size)
    expected_scale_shape = torch.Size(
        (
            (offsets_within_first_block[0] + weight_local.shape[0] + block_size - 1) // block_size,
            (offsets_within_first_block[1] + weight_local.shape[1] + block_size - 1) // block_size,
        )
    )
    if scale_local.shape != expected_scale_shape:
        raise RuntimeError(
            f"{name} scale_inv shape {tuple(scale_local.shape)} does not cover DTensor local weight "
            f"shape {tuple(weight_local.shape)} at global block offset {offsets_within_first_block}; "
            f"expected {tuple(expected_scale_shape)}"
        )

    scale_local = scale_local.to(device=weight_local.device)
    if not weight_local.is_contiguous():
        weight_local = weight_local.contiguous()
    if not scale_local.is_contiguous():
        scale_local = scale_local.contiguous()

    if offsets_within_first_block == (0, 0):
        dequantized_local = dequantize_from_fp8(
            weight_local,
            scale_local,
            dtype=dtype,
            BLOCK_SIZE=block_size,
            name=name,
        )
    else:
        use_triton = (
            _GLM_TRITON_AVAILABLE
            and weight_local.is_cuda
            and scale_local.is_cuda
            and weight_local.dim() == 2
            and scale_local.dim() == 2
        )
        if use_triton:
            try:
                dequantized_local = _dequantize_glm_with_triton_offsets(
                    weight_local,
                    scale_local,
                    dtype,
                    block_size,
                    offsets_within_first_block,
                )
            except Exception as exc:
                logger.warning("GLM Triton dequant failed for %s (%s). Falling back to torch.", name, exc)
                dequantized_local = _dequantize_glm_with_torch_offsets(
                    weight_local,
                    scale_local,
                    dtype,
                    block_size,
                    offsets_within_first_block,
                )
        else:
            dequantized_local = _dequantize_glm_with_torch_offsets(
                weight_local,
                scale_local,
                dtype,
                block_size,
                offsets_within_first_block,
            )

    from torch.distributed._tensor import DTensor

    return DTensor.from_local(dequantized_local, weight.device_mesh, weight.placements)


class GlmMoeDsaStateDictAdapter(Glm4MoeStateDictAdapter):
    """Converts between HF GLM-MoE-DSA checkpoints and native format.

    Extends Glm4MoeStateDictAdapter with handling for the DSA indexer weights
    that should not be quantized (k_norm, weights_proj).
    """

    _supports_low_memory_dcp_load = False

    _indexer_non_quantized_keys = [
        "indexer.k_norm.weight",
        "indexer.k_norm.bias",
        "indexer.weights_proj.weight",
    ]

    def _uses_blockwise_fp8_checkpoint(self) -> bool:
        quantization_config = getattr(self.config, "quantization_config", None)
        if isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method")
            weight_block_size = quantization_config.get("weight_block_size")
        else:
            quant_method = getattr(quantization_config, "quant_method", None)
            weight_block_size = getattr(quantization_config, "weight_block_size", None)

        if quant_method != "fp8" or not isinstance(weight_block_size, (list, tuple)):
            return False
        return tuple(weight_block_size) == (BLOCK_SIZE, BLOCK_SIZE)

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        scale_inv_keys = []
        dequantized_count = 0
        for key, weight in state_dict.items():
            scale_key = key + "_scale_inv"
            if key.endswith(".weight") and scale_key in state_dict:
                state_dict[key] = _dequantize_glm_fp8(
                    weight,
                    state_dict[scale_key],
                    dtype=self.dtype,
                    name=key,
                )
                scale_inv_keys.append(scale_key)
                dequantized_count += 1

        for key in scale_inv_keys:
            state_dict.pop(key)

        logger.debug(
            "[GLM FP8 Dequant] Dequantized %d weights and removed %d scale tensors",
            dequantized_count,
            len(scale_inv_keys),
        )
        return state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional["DeviceMesh"] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Dequantize blockwise FP8 tensors before merging per-expert weights."""
        hf_state_dict = self._dequantize(hf_state_dict)
        return super().from_hf(hf_state_dict, device_mesh=device_mesh, **kwargs)

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        quantization = kwargs.get("quantization", False) and self._uses_blockwise_fp8_checkpoint()
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        expert_kwargs = {**kwargs, "quantization": quantization} if "quantization" in kwargs else kwargs
        expert_result = self._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, **expert_kwargs)
        if expert_result is not None:
            result = expert_result
        else:
            result = [(fqn, tensor)]

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        if quantization:
            quantized_result = []
            for key, value in result:
                if should_quantize_key(key):
                    value = value.to(dtype=torch.float8_e4m3fn)
                    quantized_result.append((key, value))
                    quantized_result.append(
                        (key + "_scale_inv", create_scale_inv_for_weight(value, block_size=BLOCK_SIZE))
                    )
                else:
                    quantized_result.append((key, value))
            return quantized_result

        return result
