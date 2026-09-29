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

import torch

from nemo_automodel.components.moe.state_dict_utils import is_dtensor

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

logger = logging.getLogger(__name__)

# Fixed block size of 128x128 as specified in https://arxiv.org/pdf/2412.19437
BLOCK_SIZE = 128

# Keys that should not be quantized (layernorms, embeddings, gates)
NON_QUANTIZED_KEY_PATTERNS = [
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "norm.weight",
    "lm_head.weight",
    "embed_tokens.weight",
    "mlp.gate.weight",
]


def should_quantize_key(key: str) -> bool:
    """Check if a key should be quantized based on its name."""
    if not key.endswith(".weight"):
        return False
    return not any(pattern in key for pattern in NON_QUANTIZED_KEY_PATTERNS)


def create_scale_inv_for_weight(weight: torch.Tensor, block_size: int = BLOCK_SIZE) -> torch.Tensor:
    """Create a scale_inv tensor for a weight.

    Note: scale_inv is always created as a regular tensor (not DTensor) because
    the scale_inv shape (based on 128x128 blocks) doesn't align with DTensor
    sharding boundaries. During dequantization, _slice_scale_for_dtensor handles
    extracting the correct scale blocks for DTensor weights.

    Args:
        weight: The weight tensor (may be a DTensor)
        block_size: The FP8 quantization block size

    Returns:
        scale_inv tensor with shape based on GLOBAL weight shape
    """
    weight_is_dtensor = is_dtensor(weight)
    weight_local = weight.to_local() if weight_is_dtensor else weight

    # For DTensor weights, use GLOBAL shape (DTensor.shape returns global shape)
    # For regular tensors, use the tensor's shape directly
    if weight_is_dtensor:
        global_shape = weight.shape
        block_rows = (global_shape[0] + block_size - 1) // block_size
        block_cols = (global_shape[1] + block_size - 1) // block_size
        scale_shape = torch.Size((block_rows, block_cols))
    else:
        scale_shape = calculate_scale_shape(weight_local, block_size)

    return torch.ones(scale_shape, dtype=torch.float32, device=weight_local.device)


if _TRITON_AVAILABLE:
    # Adapted from https://github.com/nvidia-cosmos/cosmos-rl/blob/main/cosmos_rl/policy/model/deepseek_v3/weight_mapper.py#L233
    @triton.jit
    def _weight_dequant_kernel(
        x_ptr,
        s_ptr,
        y_ptr,
        M,
        N,
        stride_xm,
        stride_xn,
        stride_ym,
        stride_yn,
        stride_sm,
        stride_sn,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_n = tl.program_id(axis=1)
        offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn, mask=mask).to(tl.float32)
        s = tl.load(s_ptr + pid_m * stride_sm + pid_n * stride_sn)
        y = x * s
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn, y, mask=mask)


def _slice_scale_for_dtensor(
    scale_inv: torch.Tensor,
    weight_dtensor: torch.Tensor,
    weight_local: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Slice scale_inv tensor to match a DTensor weight's local portion.

    When weight is sharded via DTensor but scale_inv is a regular tensor,
    we need to extract only the scale blocks that correspond to the local
    portion of the weight.

    Args:
        scale_inv: The full (global) scale_inv tensor
        weight_dtensor: The DTensor weight (has device_mesh and placements)
        weight_local: The local portion of the weight
        block_size: The FP8 quantization block size (default 128)

    Returns:
        The sliced scale_inv tensor matching the local weight's blocks
    """
    from torch.distributed._tensor import Shard

    # Get the DTensor's placement info
    device_mesh = weight_dtensor.device_mesh
    placements = weight_dtensor.placements

    # Find which dimension is sharded and get the mesh coordinate
    scale_slices = [slice(None), slice(None)]  # Default: take all

    for dim, placement in enumerate(placements):
        if isinstance(placement, Shard):
            shard_dim = placement.dim
            mesh_dim_size = device_mesh.size(dim)
            mesh_coord = device_mesh.get_local_rank(mesh_dim=dim)

            # Calculate the global weight shape for this dimension
            local_size = weight_local.shape[shard_dim]

            # For DTensor sharding, the global size is distributed across ranks
            # We need to compute the exact global row range this rank owns
            # DTensor uses contiguous sharding: rank i owns rows [i*chunk, (i+1)*chunk)
            # where chunk = ceil(global_size / mesh_dim_size)

            # Compute global size from scale_inv shape
            global_num_blocks = scale_inv.shape[shard_dim]
            global_size = global_num_blocks * block_size  # Upper bound

            # Compute chunk size per rank (how DTensor divides)
            chunk_size = (global_size + mesh_dim_size - 1) // mesh_dim_size

            # Global row range this rank owns
            global_start_row = mesh_coord * chunk_size
            global_end_row = global_start_row + local_size

            # Convert row range to block range
            # Start block: floor(start_row / block_size)
            # End block: ceil(end_row / block_size)
            start_block = global_start_row // block_size
            end_block = (global_end_row + block_size - 1) // block_size

            # Clamp to valid range
            end_block = min(end_block, global_num_blocks)

            # Update the slice for the corresponding scale dimension
            scale_slices[shard_dim] = slice(start_block, end_block)

    return scale_inv[scale_slices[0], scale_slices[1]].contiguous()


def calculate_scale_shape(weight: torch.Tensor, BLOCK_SIZE: int = BLOCK_SIZE) -> torch.Size:
    # Calculate the scale tensor shape
    orig_shape = weight.shape

    # Calculate number of blocks needed
    block_rows = (orig_shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
    block_cols = (orig_shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE

    return torch.Size((block_rows, block_cols))


def _dequantize_with_torch(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    float_weight = weight.to(torch.float32)
    orig_shape = weight.shape
    block_rows = (orig_shape[0] + block_size - 1) // block_size
    block_cols = (orig_shape[1] + block_size - 1) // block_size

    # NOTE: When processing large models on-the-fly, misalignment between block boundaries
    # and DTensor local shape partitioning can lead to silent numerical inaccuracies.
    dequantized = float_weight.detach().clone().to(dtype=dtype)

    for i in range(block_rows):
        row_start = i * block_size
        row_end = min(row_start + block_size, orig_shape[0])

        for j in range(block_cols):
            col_start = j * block_size
            col_end = min(col_start + block_size, orig_shape[1])

            block = float_weight[row_start:row_end, col_start:col_end]
            scale = scale_inv[i, j]
            block = block * scale

            block_converted = block.to(dtype=torch.float32)
            dequantized[row_start:row_end, col_start:col_end] = block_converted

    return dequantized


def _dequantize_with_triton(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype: torch.dtype,
    block_size: int,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available for dequantization.")

    m, n = weight.shape
    output = torch.empty((m, n), device=weight.device, dtype=dtype)
    grid = (triton.cdiv(m, block_size), triton.cdiv(n, block_size))
    _weight_dequant_kernel[grid](
        weight,
        scale_inv,
        output,
        m,
        n,
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        scale_inv.stride(0),
        scale_inv.stride(1),
        BLOCK_SIZE=block_size,
    )
    return output


def dequantize_from_fp8(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    dtype=torch.bfloat16,
    BLOCK_SIZE: int = BLOCK_SIZE,
    name: str = "",
) -> torch.Tensor:
    weight_is_dtensor = is_dtensor(weight)
    scale_is_dtensor = is_dtensor(scale_inv)

    weight_local = weight.to_local() if weight_is_dtensor else weight
    scale_local = scale_inv.to_local() if scale_is_dtensor else scale_inv

    expected_scale_shape = calculate_scale_shape(weight_local, BLOCK_SIZE)
    if scale_local.shape != expected_scale_shape:
        logger.debug(
            f"{name} scale_inv shape {scale_local.shape} doesn't match expected shape {expected_scale_shape}, slicing scale_inv"
        )
        # If weight is DTensor but scale_inv is not, we need to slice scale_inv
        # to match the local weight's block boundaries
        if weight_is_dtensor and not scale_is_dtensor:
            scale_local = _slice_scale_for_dtensor(scale_inv, weight, weight_local, BLOCK_SIZE)
            # Verify the slice worked
            if scale_local.shape != expected_scale_shape:
                logger.warning(
                    f"scale_inv shape {scale_local.shape} still doesn't match expected shape {expected_scale_shape} after slicing"
                )

    scale_local = scale_local.to(device=weight_local.device)
    if not weight_local.is_contiguous():
        weight_local = weight_local.contiguous()
    if not scale_local.is_contiguous():
        scale_local = scale_local.contiguous()

    use_triton = (
        _TRITON_AVAILABLE
        and weight_local.is_cuda
        and scale_local.is_cuda
        and weight_local.dim() == 2
        and scale_local.dim() == 2
    )

    if use_triton:
        try:
            dequantized_local = _dequantize_with_triton(weight_local, scale_local, dtype, BLOCK_SIZE)
        except Exception as exc:
            logger.warning(f"Triton dequant failed ({exc}). Falling back to torch.")
            dequantized_local = _dequantize_with_torch(weight_local, scale_local, dtype, BLOCK_SIZE)
    else:
        dequantized_local = _dequantize_with_torch(weight_local, scale_local, dtype, BLOCK_SIZE)

    if weight_is_dtensor:
        from torch.distributed._tensor import DTensor

        return DTensor.from_local(dequantized_local, weight.device_mesh, weight.placements)

    return dequantized_local
