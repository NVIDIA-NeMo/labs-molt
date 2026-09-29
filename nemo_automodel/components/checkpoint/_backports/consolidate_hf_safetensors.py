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
# taken and edited from
# https://github.com/pytorch/pytorch/blob/2fde10d9148d2dae9dc168ccd647de7f3cafea6b/torch/distributed/checkpoint/_consolidate_hf_safetensors.py

import concurrent.futures
import glob
import json
import logging
import math
import mmap
import os
import shutil
import struct
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import distributed as dist

from nemo_automodel.components.checkpoint._backports.hf_utils import (
    DATA_OFFSETS_KEY,
    DEFAULT_EXTRA_METADATA_KEY,
    DTYPE_KEY,
    DTYPE_MAP,
    SAVED_OFFSETS_KEY,
    SHAPE_KEY,
    SUFFIX,
    _gen_file_name,
    _get_dcp_custom_metadata,
    _get_dtype,
    _get_safetensors_file_metadata,
    _metadata_fn,
)

logger: logging.Logger = logging.getLogger(__name__)

_DTYPE_CAST_ALIASES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "fp64": torch.float64,
    "float64": torch.float64,
    "double": torch.float64,
}

_DTYPE_TO_SAFETENSORS_DTYPE: dict[torch.dtype, str] = {dtype: key for key, dtype in DTYPE_MAP.items()}
_FP8_DTYPES = {torch.float8_e4m3fn, torch.float8_e5m2, torch.float8_e8m0fnu}
_QUANTIZED_DTYPE_WARNING_LIMIT = 5
_MISSING_FQN_WARNING_LIMIT = 5


def _get_known_dtype(dtype_str: str) -> torch.dtype:
    """Return the torch dtype for a safetensors dtype string."""
    if dtype_str in DTYPE_MAP:
        return DTYPE_MAP[dtype_str]
    return _get_dtype(dtype_str)


def _get_dtype_size(dtype: torch.dtype) -> int:
    """Return the serialized element size for a torch dtype."""
    try:
        return torch.finfo(dtype).bits // 8
    except TypeError:
        return torch.empty((), dtype=dtype).element_size()


def _get_safetensors_dtype_str(dtype: torch.dtype) -> str:
    """Return the safetensors dtype string for a supported torch dtype."""
    try:
        return _DTYPE_TO_SAFETENSORS_DTYPE[dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported cast dtype for safetensors consolidation: {dtype}") from e


def _should_cast_dtype(source_dtype: torch.dtype, cast_dtype: torch.dtype | None) -> bool:
    """Return True when a tensor should be cast to the requested output dtype."""
    if cast_dtype is None or source_dtype == cast_dtype:
        return False
    return _is_regular_floating_dtype(source_dtype)


def _is_regular_floating_dtype(dtype: torch.dtype) -> bool:
    """Return True for floating dtypes that are safe dtype-preservation targets."""
    return torch.empty((), dtype=dtype).is_floating_point() and dtype not in _FP8_DTYPES


def _warn_quantized_dtype_mismatches(mismatches: list[tuple[str, str, str]]) -> None:
    """Warn once when original quantized/packed tensors are kept as dequantized floats."""
    if not mismatches:
        return

    examples = ", ".join(
        f"{fqn}: original {original_dtype_str}, saved {source_dtype_str}"
        for fqn, original_dtype_str, source_dtype_str in mismatches[:_QUANTIZED_DTYPE_WARNING_LIMIT]
    )
    omitted = len(mismatches) - _QUANTIZED_DTYPE_WARNING_LIMIT
    suffix = f"; {omitted} more tensor(s) omitted" if omitted > 0 else ""
    logger.warning(
        "Original checkpoint tensor(s) were quantized or packed, but saved tensor(s) are floating point. "
        "Leaving them as float and treating output as a dequantized Hugging Face checkpoint. Examples: %s%s.",
        examples,
        suffix,
    )


def _resolve_output_dtype(
    fqn: str,
    source_dtype_str: str,
    cast_dtype: torch.dtype | None,
    fqn_to_dtype_mapping: dict[str, str] | None,
    quantized_dtype_mismatches: list[tuple[str, str, str]] | None = None,
) -> tuple[torch.dtype, str]:
    """Resolve the output dtype for a tensor.

    Per-FQN fp32 metadata is an intrinsic model contract and remains authoritative
    when ``cast_dtype`` is set. The explicit cast wins for other ordinary
    floating-point tensors. Without a cast, per-FQN metadata restores ordinary
    floating-point tensors when available; checkpoints without that metadata keep
    the saved dtype.
    """
    source_dtype = _get_known_dtype(source_dtype_str)
    mapped_dtype_str = fqn_to_dtype_mapping.get(fqn) if fqn_to_dtype_mapping else None
    mapped_dtype = _get_known_dtype(mapped_dtype_str) if mapped_dtype_str is not None else None

    if (
        cast_dtype is not None
        and mapped_dtype_str is not None
        and mapped_dtype is torch.float32
        and _is_regular_floating_dtype(source_dtype)
    ):
        return mapped_dtype, mapped_dtype_str

    if cast_dtype is not None:
        output_dtype = cast_dtype if _should_cast_dtype(source_dtype, cast_dtype) else source_dtype
        output_dtype_str = (
            _get_safetensors_dtype_str(output_dtype) if output_dtype != source_dtype else source_dtype_str
        )
        return output_dtype, output_dtype_str

    if mapped_dtype is None or mapped_dtype_str is None:
        return source_dtype, source_dtype_str

    if mapped_dtype == source_dtype:
        return source_dtype, source_dtype_str

    if _is_regular_floating_dtype(mapped_dtype) and torch.empty((), dtype=source_dtype).is_floating_point():
        return mapped_dtype, mapped_dtype_str

    # If a quantized/packed original tensor was saved as float, keep the float
    # value as a dequantized export instead of pretending we can restore packing.
    if torch.empty((), dtype=source_dtype).is_floating_point() and quantized_dtype_mismatches is not None:
        quantized_dtype_mismatches.append((fqn, mapped_dtype_str, source_dtype_str))
    return source_dtype, source_dtype_str


@dataclass
class _FqnData:
    """
    Dataclass to store information about a tensor (identified by its fully qualified name).

    Attributes:
        offset_in_file: Byte offset where this tensor's data begins in the output file
        shape_in_file: Shape of the tensor in the output file
        dtype_size: Size of the tensor's data type in bytes
        dtype_str: String representation of the tensor's data type
        source_dtype_size: Size of the source tensor's data type in bytes
        source_dtype_str: String representation of the source tensor's data type
    """

    offset_in_file: int = 0
    shape_in_file: list[int] = field(default_factory=list)
    dtype_size: int = 0
    dtype_str: str = ""
    source_dtype_size: int = 0
    source_dtype_str: str = ""


@dataclass
class _OutputFileData:
    """
    Dataclass to store information about an output safetensors file.

    Attributes:
        metadata_size: Size of the metadata section in bytes
        fqn_data: Dictionary mapping tensor names to their metadata
    """

    metadata_size: int = 0
    fqn_data: dict[str, _FqnData] = field(default_factory=dict)


def _drop_missing_output_fqns(
    output_files_data: dict[str, _OutputFileData],
    available_fqns: set[str],
) -> None:
    """Drop mapped output tensors that are not present in the input shard metadata."""
    missing_fqns: list[str] = []
    for output_path in list(output_files_data):
        output_data = output_files_data[output_path]
        for fqn in list(output_data.fqn_data):
            if fqn in available_fqns:
                continue
            missing_fqns.append(fqn)
            del output_data.fqn_data[fqn]
        if not output_data.fqn_data:
            del output_files_data[output_path]

    if not missing_fqns:
        return

    examples = ", ".join(missing_fqns[:_MISSING_FQN_WARNING_LIMIT])
    omitted = len(missing_fqns) - _MISSING_FQN_WARNING_LIMIT
    suffix = f"; {omitted} more tensor(s) omitted" if omitted > 0 else ""
    logger.warning(
        "Ignoring %d tensor(s) from the consolidation shard mapping because they were not present in the input "
        "safetensors shard metadata. This can happen for tied-weight aliases. Examples: %s%s.",
        len(missing_fqns),
        examples,
        suffix,
    )


@dataclass
class _InputFileData:
    """
    Dataclass to store information about an input safetensors file.

    Attributes:
        metadata_size: Size of the metadata section in bytes
        metadata: Json metadata from the safetensors file
    """

    metadata_size: int = 0
    metadata: Any = None


def _parse_input_metadata(
    input_files_data: dict[str, _InputFileData],
    output_files_data: dict[str, _OutputFileData],
    cast_dtype: torch.dtype | None = None,
    fqn_to_dtype_mapping: dict[str, str] | None = None,
) -> None:
    """
    Parse metadata from input safetensors files to determine the full tensor shapes and types.

    This function analyzes the metadata from all input files to determine the complete shape
    of each tensor after consolidation. It updates the output_files_data with this information.

    Args:
        input_files_data: dict of metadata from input safetensors files
        output_files_data: Dictionary mapping output file paths to their metadata

    Raises:
        ValueError: If no DCP custom metadata is found in a safetensors file
    """

    # Dictionary to track the full size of each tensor across all shards
    fqn_to_size_mapping: dict[str, tuple[list[int], str]] = {}

    for file_data in input_files_data.values():
        safetensors_metadata = file_data.metadata
        dcp_sharding_info = _get_dcp_custom_metadata(safetensors_metadata)
        if not dcp_sharding_info:
            raise ValueError(
                "No DCP custom metadata found in safetensors file. The file must be saved with DCP to be consolidated."
            )

        for key, val in safetensors_metadata.items():
            if key == DEFAULT_EXTRA_METADATA_KEY:
                continue

            # Get the shape of this tensor shard and its offset in the full tensor
            sizes = val[SHAPE_KEY]
            offsets = dcp_sharding_info[key][SAVED_OFFSETS_KEY]

            if key not in fqn_to_size_mapping:
                # First time seeing this tensor - calculate its full size by adding offsets to dimensions
                cur_size = [size + offset for size, offset in zip(sizes, offsets)]
                fqn_to_size_mapping[key] = (cur_size, val[DTYPE_KEY])
            else:
                # We've seen this tensor before - update its size if this shard extends beyond current known dimensions
                cur_size = fqn_to_size_mapping[key][0]
                for i in range(len(sizes)):
                    cur_size[i] = max(cur_size[i], sizes[i] + offsets[i])

    # Now that we know the full size of each tensor, populate the output file data
    quantized_dtype_mismatches: list[tuple[str, str, str]] = []
    for fqn, tensor_info in fqn_to_size_mapping.items():
        tensor_size = tensor_info[0]
        dtype_str = tensor_info[1]
        for output_data in output_files_data.values():
            # Add this tensor to the output file if it's already assigned there
            if fqn in output_data.fqn_data:
                source_dtype = _get_known_dtype(dtype_str)
                output_dtype, output_dtype_str = _resolve_output_dtype(
                    fqn,
                    dtype_str,
                    cast_dtype,
                    fqn_to_dtype_mapping,
                    quantized_dtype_mismatches,
                )
                output_data.fqn_data[fqn] = _FqnData(
                    shape_in_file=tensor_size,
                    dtype_size=_get_dtype_size(output_dtype),
                    dtype_str=output_dtype_str,
                    source_dtype_size=_get_dtype_size(source_dtype),
                    source_dtype_str=dtype_str,
                )
    _warn_quantized_dtype_mismatches(quantized_dtype_mismatches)
    _drop_missing_output_fqns(output_files_data, set(fqn_to_size_mapping))


def _compute_safetensors_metadata_and_offsets(output_data: _OutputFileData) -> dict[str, Any]:
    """Compute the safetensors header metadata and output tensor offsets."""
    metadata: dict[str, Any] = {}
    curr_offset = 0

    for fqn, fqn_data in output_data.fqn_data.items():
        end_offset = curr_offset + math.prod(fqn_data.shape_in_file) * fqn_data.dtype_size
        metadata[fqn] = {
            SHAPE_KEY: fqn_data.shape_in_file,
            DTYPE_KEY: fqn_data.dtype_str,
            DATA_OFFSETS_KEY: [curr_offset, end_offset],
        }
        fqn_data.offset_in_file = curr_offset
        curr_offset = end_offset

    return metadata


def _write_safetensors_header(output_stream: Any, output_data: _OutputFileData, metadata: dict[str, Any]) -> None:
    """Write a safetensors header to an already-open output stream."""
    json_metadata = json.dumps(metadata)
    json_bytes = json_metadata.encode("utf-8")
    header_len = struct.pack("<Q", len(json_bytes))

    output_stream.write(header_len)
    output_stream.write(json_bytes)
    output_data.metadata_size = 8 + len(json_bytes)


def _read_tensor_data_mmap(
    file_path: str,
    start_offset: int,
    end_offset: int,
    metadata_size: int,
) -> bytes:
    """
    Read tensor data from a safetensors file using memory mapping for efficiency.

    Args:
        file_path: Path to the safetensors file
        start_offset: Start offset of tensor data within the data section
        end_offset: End offset of tensor data within the data section
        metadata_size: Size of the metadata header

    Returns:
        Raw tensor data as bytes
    """
    # Use mmap for efficient access
    with open(file_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            absolute_start = metadata_size + start_offset
            absolute_end = metadata_size + end_offset
            return bytes(mm[absolute_start:absolute_end])


def _process_output_file(
    output_file: str,
    output_data: _OutputFileData,
    input_files_data: dict[str, _InputFileData],
) -> None:
    """
    Process a single output file by writing tensor data from input files using memory mapping.

    This function is designed to be run in parallel for different output files.

    Args:
        output_file: Path to the output file
        output_data: Metadata for the output file
        input_files_data: Dictionary mapping input file paths to their metadata
    """
    with open(output_file, "wb") as output_stream:
        metadata = _compute_safetensors_metadata_and_offsets(output_data)
        _write_safetensors_header(output_stream, output_data, metadata)
        sorted_tensors = sorted(output_data.fqn_data.items(), key=lambda x: x[1].offset_in_file)

        # Process each tensor in sequential output order
        for tensor_fqn, tensor_fqn_data in sorted_tensors:
            full_tensor_mv = memoryview(
                bytearray(math.prod(tensor_fqn_data.shape_in_file) * tensor_fqn_data.source_dtype_size)
            )

            # Process each input safetensors file
            for safetensors_file in input_files_data.keys():
                file_metadata = input_files_data[safetensors_file].metadata
                input_metadata_size = input_files_data[safetensors_file].metadata_size

                if tensor_fqn not in file_metadata.keys():
                    continue

                metadata = file_metadata[tensor_fqn]

                data_offsets = metadata[DATA_OFFSETS_KEY]

                # Use memory mapping to read tensor data efficiently
                data_to_write = _read_tensor_data_mmap(
                    safetensors_file,
                    data_offsets[0],
                    data_offsets[1],
                    input_metadata_size,
                )

                # Get the offsets of this tensor shard within the full tensor
                fqn_custom_metadata = _get_dcp_custom_metadata(file_metadata)[tensor_fqn]  # type: ignore[index]
                offsets_of_tensor_being_read = fqn_custom_metadata[SAVED_OFFSETS_KEY]  # type: ignore[index]

                # Write this tensor shard to the appropriate position in the output file buffer
                _write_sub_tensor_to_file_optimized(
                    full_tensor_mv,
                    data_to_write,
                    tensor_fqn_data.source_dtype_size,  # Size of each source element in bytes
                    tensor_fqn_data.shape_in_file,  # Full tensor shape
                    offsets_of_tensor_being_read,  # Where this shard belongs in the full tensor
                    metadata[SHAPE_KEY],  # Shape of this shard
                )

            if tensor_fqn_data.source_dtype_str == tensor_fqn_data.dtype_str:
                output_stream.write(full_tensor_mv)
            else:
                output_stream.write(
                    _cast_tensor_bytes(
                        full_tensor_mv,
                        tensor_fqn_data.shape_in_file,
                        tensor_fqn_data.source_dtype_str,
                        tensor_fqn_data.dtype_str,
                    )
                )


def _cast_tensor_bytes(
    tensor_bytes: memoryview,
    tensor_shape: list[int],
    source_dtype_str: str,
    cast_dtype_str: str,
) -> bytes:
    """Cast a contiguous tensor byte buffer from its source dtype to the cast dtype."""
    source_dtype = _get_known_dtype(source_dtype_str)
    cast_dtype = _get_known_dtype(cast_dtype_str)
    tensor = torch.frombuffer(tensor_bytes, dtype=source_dtype)
    if tensor_shape:
        tensor = tensor.reshape(tensor_shape)
    cast_tensor = tensor.to(dtype=cast_dtype).contiguous()
    return cast_tensor.view(torch.uint8).numpy().tobytes()


def _write_data(
    input_files_data: dict[str, _InputFileData],
    output_files_data: dict[str, _OutputFileData],
    num_threads: int = 1,
) -> None:
    """
    Write tensor data from input files to the output files using memory mapping.

    This function reads tensor data from each input file and writes it to the appropriate
    position in the output files based on the tensor's offsets. When num_threads > 1,
    the work is split across threads with each thread handling a different output file.

    Args:
        input_files_data: Dictionary mapping input file paths to their metadata
        output_files_data: Dictionary mapping output file paths to their metadata
        num_threads: Number of threads to use for parallel processing
    """
    if num_threads <= 1 or len(output_files_data) <= 1:
        # Sequential processing
        for output_file, output_data in output_files_data.items():
            _process_output_file(output_file, output_data, input_files_data)
    else:
        # Parallel processing with ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_threads, len(output_files_data))) as executor:
            futures = {}
            for output_file, output_data in output_files_data.items():
                futures[
                    executor.submit(
                        _process_output_file,
                        output_file,
                        output_data,
                        input_files_data,
                    )
                ] = output_file

            # Wait for all futures to complete
            for future in concurrent.futures.as_completed(futures):
                # Handle any exceptions that might have occurred
                output_file = futures[future]
                try:
                    future.result()
                except Exception:
                    logger.exception("Error processing output file %s.", os.path.basename(output_file))
                    raise


def _write_sub_tensor_to_file_optimized(
    full_tensor_mv: memoryview,
    sub_tensor_bytes: bytes,
    element_size: int,
    tensor_shape: list[int],
    sub_tensor_offsets: list[int],
    sub_tensor_shape: list[int],
) -> None:
    """
    Optimized version that writes the maximum number of contiguous bytes possible.

    Uses a unified algorithm that calculates the maximum contiguous bytes that can be
    written in each iteration and continues until the entire subtensor is written.
    Handles all sharding patterns efficiently:
    - Full sub-tensor at once for row-wise sharding
    - Row-by-row for column-wise sharding
    - Optimized chunks for other patterns

    Args:
        full_tensor_mv: Buffer to write the full tensor to
        sub_tensor_bytes: Raw tensor data as bytes
        element_size: Size of each element in bytes
        tensor_shape: Shape of the full tensor
        sub_tensor_offsets: Starting offsets of the sub-tensor within the full tensor
        sub_tensor_shape: Shape of the sub-tensor
    """
    # Handle 0-dim (scalar) tensors explicitly
    if len(tensor_shape) == 0 and len(sub_tensor_shape) == 0:
        bytes_to_write = min(len(sub_tensor_bytes), element_size)
        if bytes_to_write:
            full_tensor_mv[:bytes_to_write] = sub_tensor_bytes[:bytes_to_write]
        return

    # Handle invalid/empty shapes (non-scalar)
    if not tensor_shape or not sub_tensor_shape:
        return

    # Calculate tensor strides for efficient indexing
    tensor_strides = [1]
    for i in range(len(tensor_shape) - 1, 0, -1):
        tensor_strides.insert(0, tensor_strides[0] * tensor_shape[i])

    sub_tensor_strides = [1]
    for i in range(len(sub_tensor_shape) - 1, 0, -1):
        sub_tensor_strides.insert(0, sub_tensor_strides[0] * sub_tensor_shape[i])

    total_elements = math.prod(sub_tensor_shape)

    elements_written = 0
    while elements_written < total_elements:
        # Convert linear index to multi-dimensional indices
        temp_idx = elements_written
        indices = []
        for dim_size in reversed(sub_tensor_shape):
            indices.append(temp_idx % dim_size)
            temp_idx //= dim_size
        indices.reverse()

        # Calculate maximum contiguous elements we can write from this position
        max_contiguous = _calculate_max_contiguous_elements(indices, sub_tensor_shape, tensor_shape)

        # Calculate source position in bytes
        src_pos = sum(idx * stride for idx, stride in zip(indices, sub_tensor_strides))
        src_byte_offset = src_pos * element_size

        # Calculate destination position in bytes
        dest_indices = [idx + offset for idx, offset in zip(indices, sub_tensor_offsets)]
        dest_pos = sum(idx * stride for idx, stride in zip(dest_indices, tensor_strides))
        dest_byte_offset = dest_pos * element_size

        # Write the contiguous chunk
        bytes_to_write = max_contiguous * element_size
        chunk_data = sub_tensor_bytes[src_byte_offset : src_byte_offset + bytes_to_write]
        full_tensor_mv[dest_byte_offset : dest_byte_offset + bytes_to_write] = chunk_data

        elements_written += max_contiguous


def _calculate_max_contiguous_elements(
    indices: list[int],
    sub_tensor_shape: list[int],
    tensor_shape: list[int],
) -> int:
    """
    Calculate the maximum number of contiguous elements that can be written from current position.

    This determines the largest chunk by checking how elements are laid out in memory
    and finding natural boundaries where contiguity breaks.

    Args:
        indices: Current position indices in the sub-tensor
        sub_tensor_shape: Shape of the sub-tensor being written
        tensor_shape: Shape of the full tensor

    Raises:
        ValueError: If input lists are empty, have mismatched lengths, or contain invalid values
    """
    # Validate input lists are not empty
    if not indices or not sub_tensor_shape or not tensor_shape:
        raise ValueError("Input lists cannot be empty")

    # Validate all lists have the same length (same number of dimensions)
    if not (len(indices) == len(sub_tensor_shape) == len(tensor_shape)):
        raise ValueError(
            f"All input lists must have the same length. Got indices: {len(indices)}, "
            f"sub_tensor_shape: {len(sub_tensor_shape)}, tensor_shape: {len(tensor_shape)}"
        )

    # Validate indices are within bounds of sub_tensor_shape
    for i, (idx, sub_dim) in enumerate(zip(indices, sub_tensor_shape)):
        if idx >= sub_dim:
            raise ValueError(f"Index {idx} at dimension {i} is out of bounds for sub-tensor shape {sub_tensor_shape}")

    # Validate sub_tensor dimensions don't exceed tensor dimensions
    for i, (sub_dim, tensor_dim) in enumerate(zip(sub_tensor_shape, tensor_shape)):
        if sub_dim > tensor_dim:
            raise ValueError(f"Sub-tensor dimension {sub_dim} at position {i} exceeds tensor dimension {tensor_dim}")

    # Start with elements remaining in the last dimension
    max_contiguous = sub_tensor_shape[-1] - indices[-1]

    # Check if we can extend across multiple dimensions
    # We can write across dimension boundaries if we're writing complete "rows"
    # and the layout in destination tensor maintains contiguity

    # For 2D case: check if we can write multiple complete rows
    if len(sub_tensor_shape) >= 2:
        # If we're at the start of a row and can write complete rows
        if indices[-1] == 0:  # At start of last dimension (column)
            rows_remaining = sub_tensor_shape[-2] - indices[-2]  # Rows left to write

            # Check if writing complete rows maintains contiguity in destination
            # This is true for row-wise sharding or when sub-tensor spans full width
            if sub_tensor_shape[-1] == tensor_shape[-1]:  # Full width
                max_contiguous = rows_remaining * sub_tensor_shape[-1]

            # For higher dimensions, check if we can extend further
            if len(sub_tensor_shape) >= 3 and indices[-2] == 0:
                # Check if we can write complete 2D slices
                remaining_in_dim = sub_tensor_shape[-3] - indices[-3]
                if sub_tensor_shape[-1] == tensor_shape[-1] and sub_tensor_shape[-2] == tensor_shape[-2]:
                    max_contiguous = remaining_in_dim * sub_tensor_shape[-2] * sub_tensor_shape[-1]

    return max_contiguous


def _write_overall_metadata_file(
    output_dir: str,
    output_files_data: dict[str, _OutputFileData],
) -> None:
    """
    Write the overall metadata file that maps tensor names to their file locations.

    This creates a model.safetensors.index.json file that HuggingFace models use
    to locate tensors across multiple files.

    Args:
        output_dir: Directory where the metadata file will be written
        output_files_data: Dictionary mapping output file paths to their metadata
    """
    total_size = 0
    weight_map = {}
    for output_path, value in output_files_data.items():
        for fqn, fqn_data in value.fqn_data.items():
            total_size += math.prod(fqn_data.shape_in_file) * fqn_data.dtype_size
            weight_map[fqn] = os.path.basename(output_path)

    metadata_to_write: dict[str, Any] = {}
    metadata_to_write["metadata"] = {"total_size": total_size}
    metadata_to_write["weight_map"] = weight_map

    metadata_path = os.path.join(output_dir, f"{_metadata_fn}")
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata_to_write, metadata_file, indent=2)


def _write_overall_metadata_file_from_shards(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    cast_dtype: torch.dtype | None = None,
    fqn_to_dtype_mapping: dict[str, str] | None = None,
) -> None:
    """
    Write the overall metadata file by reading metadata from input shard files.

    This creates a model.safetensors.index.json file that HuggingFace models use
    to locate tensors across multiple files. Unlike _write_overall_metadata_file,
    this function reads the necessary shape/dtype information directly from the
    input shard files, avoiding the need for distributed gather operations.

    Args:
        input_dir: Directory containing the input shard safetensors files
        output_dir: Directory where the metadata file will be written
        fqn_to_index_mapping: Mapping from tensor names to output file indices
    """
    # Find all safetensors files in the input directory
    safetensors_files = glob.glob(os.path.join(input_dir, f"*{SUFFIX}"))

    # Read metadata from all input files
    input_files_data: dict[str, _InputFileData] = {}
    for input_file in safetensors_files:
        with open(input_file, "rb") as f:
            metadata, metadata_size = _get_safetensors_file_metadata(f)
            input_files_data[input_file] = _InputFileData(
                metadata_size=metadata_size,
                metadata=metadata,
            )

    # Compute full tensor shapes from sharded metadata (same logic as _parse_input_metadata)
    fqn_to_size_mapping: dict[str, tuple[list[int], str]] = {}
    for file_data in input_files_data.values():
        safetensors_metadata = file_data.metadata
        dcp_sharding_info = _get_dcp_custom_metadata(safetensors_metadata)
        if not dcp_sharding_info:
            raise ValueError(
                "No DCP custom metadata found in safetensors file. The file must be saved with DCP to be consolidated."
            )

        for key, val in safetensors_metadata.items():
            if key == DEFAULT_EXTRA_METADATA_KEY:
                continue

            sizes = val[SHAPE_KEY]
            offsets = dcp_sharding_info[key][SAVED_OFFSETS_KEY]

            if key not in fqn_to_size_mapping:
                cur_size = [size + offset for size, offset in zip(sizes, offsets)]
                fqn_to_size_mapping[key] = (cur_size, val[DTYPE_KEY])
            else:
                cur_size = fqn_to_size_mapping[key][0]
                for i in range(len(sizes)):
                    cur_size[i] = max(cur_size[i], sizes[i] + offsets[i])

    # Compute total_size and weight_map
    max_index = max(fqn_to_index_mapping.values())
    total_size = 0
    weight_map = {}

    for fqn, (tensor_shape, dtype_str) in fqn_to_size_mapping.items():
        output_dtype, _ = _resolve_output_dtype(
            fqn,
            dtype_str,
            cast_dtype,
            fqn_to_dtype_mapping,
        )
        dtype_size = _get_dtype_size(output_dtype)

        total_size += math.prod(tensor_shape) * dtype_size

        idx = fqn_to_index_mapping[fqn]
        weight_map[fqn] = _gen_file_name(idx, max_index)

    # Write the metadata file
    metadata_to_write: dict[str, Any] = {}
    metadata_to_write["metadata"] = {"total_size": total_size}
    metadata_to_write["weight_map"] = weight_map

    metadata_path = os.path.join(output_dir, _metadata_fn)
    with open(metadata_path, "w") as metadata_file:
        json.dump(metadata_to_write, metadata_file, indent=2)


def _consolidate_safetensors_files(
    input_dir: str,
    output_dir: str,
    fqn_to_file_mapping: dict[str, str],
    num_threads: int,
    use_staging: bool = False,
    staging_dir: str | None = None,
    cast_dtype: torch.dtype | None = None,
    fqn_to_dtype_mapping: dict[str, str] | None = None,
) -> dict[str, _OutputFileData]:
    # Build output paths
    output_files_data: dict[str, _OutputFileData] = {}
    for fqn, filename in fqn_to_file_mapping.items():
        output_path = os.path.join(output_dir, filename)

        if output_path not in output_files_data:
            output_files_data[output_path] = _OutputFileData(fqn_data={fqn: _FqnData()})
        else:
            output_files_data[output_path].fqn_data[fqn] = _FqnData()

    # Find all safetensors files in the input directory
    safetensors_files = glob.glob(os.path.join(input_dir, f"*{SUFFIX}"))

    # Read metadata from all input files
    input_files_data: dict[str, _InputFileData] = {}
    for safetensor_file in safetensors_files:
        with open(safetensor_file, "rb") as f:
            metadata, size = _get_safetensors_file_metadata(f)
            input_files_data[safetensor_file] = _InputFileData(metadata_size=size, metadata=metadata)

    if use_staging:
        # Use staging directory for writing files first, then copy to final destination.
        if staging_dir is not None:
            os.makedirs(staging_dir, exist_ok=True)
            temp_dir = tempfile.mkdtemp(prefix="safetensors_consolidate_", dir=staging_dir)
        else:
            temp_dir = tempfile.mkdtemp(prefix="safetensors_consolidate_")

        try:
            # Create temp output files data with temp paths
            temp_output_files_data: dict[str, _OutputFileData] = {}
            temp_to_final_mapping: dict[str, str] = {}

            for final_path, file_data in output_files_data.items():
                filename = os.path.basename(final_path)
                temp_path = os.path.join(temp_dir, filename)
                temp_output_files_data[temp_path] = file_data
                temp_to_final_mapping[temp_path] = final_path

            # Step 1: Parse metadata to determine tensor shapes and types
            _parse_input_metadata(input_files_data, temp_output_files_data, cast_dtype, fqn_to_dtype_mapping)

            # Step 2: Write metadata and tensor data from input files to temp output files
            _write_data(input_files_data, temp_output_files_data, num_threads)

            # Step 3: Copy completed files from temp to final destination
            for temp_path, final_path in temp_to_final_mapping.items():
                if temp_path not in temp_output_files_data:
                    continue
                shutil.copy2(temp_path, final_path)
                logger.info("Copied %s to %s", temp_path, final_path)

        finally:
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        # Write directly to output directory
        # Step 1: Parse metadata to determine tensor shapes and types
        _parse_input_metadata(input_files_data, output_files_data, cast_dtype, fqn_to_dtype_mapping)

        # Step 2: Write metadata and tensor data from input files to output files
        _write_data(input_files_data, output_files_data, num_threads)

    return output_files_data


def consolidate_safetensors_files(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 1,
    use_staging: bool = False,
    staging_dir: str | None = None,
    cast_dtype: torch.dtype | None = None,
    fqn_to_dtype_mapping: dict[str, str] | None = None,
) -> None:
    """
    Main function to consolidate sharded safetensors files into one or more output files.

    This function orchestrates the entire consolidation process:
    1. Sets up the output file structure based on the fqn_to_index_mapping
    2. Finds all safetensors files in the input directory
    3. Parses metadata from all input files
    4. Writes metadata to the output files
    5. Writes tensor data from input files to output files
    6. Writes overall model.index.safetensors.json file with weight map

    Args:
        input_dir: Directory containing sharded safetensors files
        output_dir: Directory where consolidated files will be written
        fqn_to_index_mapping: Optional mapping of tensor names to output file indices.
                             If None, all tensors will be consolidated into a single file.
        num_threads: Number of threads to use for parallel processing of saving data to output files.
        use_staging: If True, write to a staging directory first then copy to output_dir. Default is False.
        staging_dir: Optional directory for staging files during consolidation. If provided,
                    temporary files will be created in this directory instead of the system temp.
                    Only used when use_staging=True. Useful when system temp has limited space.
        cast_dtype: Optional dtype used to cast floating-point tensors during consolidation.
        fqn_to_dtype_mapping: Optional mapping from tensor FQN to target safetensors dtype string. This can combine
            original HF dtype metadata with model-owned export overrides.
    """
    start_time = time.time()
    logger.info("Consolidating safetensors files from %s to %s.", input_dir, output_dir)
    if cast_dtype is not None:
        logger.info(
            "Requested cast dtype %s for consolidation. Only ordinary floating-point tensors with a different "
            "source dtype will be cast; tensors mapped to FP32, tensors already in this dtype, FP8 tensors, and "
            "non-floating tensors are unchanged.",
            cast_dtype,
        )

    max_index = max(fqn_to_index_mapping.values())
    fqn_to_file_mapping = {fqn: _gen_file_name(idx, max_index) for fqn, idx in fqn_to_index_mapping.items()}

    output_files_data = _consolidate_safetensors_files(
        input_dir=input_dir,
        output_dir=output_dir,
        fqn_to_file_mapping=fqn_to_file_mapping,
        num_threads=num_threads,
        use_staging=use_staging,
        staging_dir=staging_dir,
        cast_dtype=cast_dtype,
        fqn_to_dtype_mapping=fqn_to_dtype_mapping,
    )

    # Step 4: Write overall model.index.safetensors.json file with weight map
    _write_overall_metadata_file(output_dir, output_files_data)

    logger.info("Done consolidating. Took %.2f secs.", time.time() - start_time)


def _raise_if_any_rank_failed(
    local_exception: Exception | None,
    local_failure: str | None,
    distributed: bool,
    rank: int,
    world_size: int,
    process_group: dist.ProcessGroup | None,
) -> None:
    """
    Share per-rank consolidation failures across ranks and raise if any rank failed.

    In a multi-rank run this also acts as a barrier, so callers can rely on every rank having
    finished the preceding step once this returns.

    Args:
        local_exception: Exception raised on this rank, if any.
        local_failure: Human-readable description of this rank's failure, if any.
        distributed: Whether a distributed environment is initialized.
        rank: Rank of the current process within ``process_group``.
        world_size: Number of ranks in ``process_group``.
        process_group: Process group the synchronization collective runs on.

    Raises:
        RuntimeError: If any rank reported a failure, or if the failures could not be
            synchronized across ranks.
    """
    if distributed and world_size > 1:
        failures: list[str | None] = [None] * world_size
        try:
            dist.all_gather_object(failures, local_failure, group=process_group)
        except Exception as exc:
            if local_exception is not None:
                raise RuntimeError(
                    f"Safetensors consolidation failed on {local_failure}, and rank {rank} could not synchronize "
                    f"the failure across ranks: {type(exc).__name__}: {exc}"
                ) from local_exception
            raise RuntimeError(
                "Failed to synchronize safetensors consolidation status across ranks. A peer may have exited before "
                "reporting its original exception; inspect the first failing rank's traceback. "
                f"Synchronization error: {type(exc).__name__}: {exc}"
            ) from exc

        reported_failures = [failure for failure in failures if failure is not None]
        if reported_failures:
            message = "Safetensors consolidation failed before final synchronization: " + " | ".join(reported_failures)
            if local_exception is not None:
                raise RuntimeError(message) from local_exception
            raise RuntimeError(message)
    elif local_exception is not None:
        raise local_exception


def consolidate_safetensors_files_on_every_rank(
    input_dir: str,
    output_dir: str,
    fqn_to_index_mapping: dict[str, int],
    num_threads: int = 1,
    process_group: dist.ProcessGroup | None = None,
    use_staging: bool = False,
    staging_dir: str | None = None,
    cast_dtype: torch.dtype | None = None,
    fqn_to_dtype_mapping: dict[str, str] | None = None,
) -> None:
    """
    Consolidate sharded safetensors files across multiple ranks, with each rank handling a subset of output files.

    This function distributes the consolidation work by assigning output files to different ranks.
    All tensors with the same index in fqn_to_index_mapping are processed by the same rank,
    as they belong to the same output file.

    If process_group is provided, rank and world_size will be derived from it. Otherwise,
    they will be automatically detected from the distributed environment if available.

    Args:
        input_dir: Directory containing sharded safetensors files
        output_dir: Directory where consolidated files will be written
        fqn_to_index_mapping: Mapping of tensor names to output file indices
        num_threads: Number of threads to use for parallel processing on each rank
        process_group: PyTorch distributed process group (default: None, will use default group)
        use_staging: If True, write to a staging directory first then copy to output_dir. Default is False.
        staging_dir: Optional directory for staging files during consolidation. If provided,
                    temporary files will be created in this directory instead of the system temp.
                    Only used when use_staging=True. Useful when system temp has limited space.
        cast_dtype: Optional dtype used to cast floating-point tensors during consolidation.
        fqn_to_dtype_mapping: Optional mapping from tensor FQN to target safetensors dtype string. This can combine
            original HF dtype metadata with model-owned export overrides.

    Raises:
        RuntimeError: If consolidation fails on any rank of a multi-rank
            distributed run. The error identifies the originating rank,
            operation, and exception.
    """

    start_time = time.time()
    # Derive rank and world_size from process_group or default distributed environment
    distributed = dist.is_available() and dist.is_initialized()
    if distributed:
        rank = dist.get_rank(group=process_group)
        world_size = dist.get_world_size(group=process_group)
    else:
        # Default to single process mode if distributed is not initialized
        rank = 0
        world_size = 1
        logger.warning("Distributed environment not initialized. Running in single process mode.")
    if rank == 0:
        logger.info("Consolidating safetensors files from %s to %s.", input_dir, output_dir)
        if cast_dtype is not None:
            logger.info(
                "Requested cast dtype %s for consolidation. Only ordinary floating-point tensors with a different "
                "source dtype will be cast; tensors mapped to FP32, tensors already in this dtype, FP8 tensors, and "
                "non-floating tensors are unchanged.",
                cast_dtype,
            )

    indices_for_this_rank: list[int] = []
    operation = "preparing assigned output shards"
    local_exception: Exception | None = None
    local_failure: str | None = None
    try:
        # Find all unique indices in the mapping
        unique_indices = set(fqn_to_index_mapping.values())

        # Distribute indices across ranks
        for idx in unique_indices:
            # Output shard indices are 1-based. Assign shard 1 to rank 0.
            if (idx - 1) % world_size == rank:
                indices_for_this_rank.append(idx)

        # Filter the fqn_to_index_mapping to only include tensors for this rank
        filtered_mapping = {fqn: idx for fqn, idx in fqn_to_index_mapping.items() if idx in indices_for_this_rank}
        logger.debug(
            "Rank %d/%d: assigned %d of %d output file(s) (%d tensor(s)).",
            rank,
            world_size,
            len(indices_for_this_rank),
            len(unique_indices),
            len(filtered_mapping),
        )

        if filtered_mapping:
            operation = "consolidating assigned output shards"
            # Convert index mapping to filename mapping
            max_index = max(unique_indices)
            filtered_filename_mapping = {}
            for fqn, idx in filtered_mapping.items():
                filename = _gen_file_name(idx, max_index)
                filtered_filename_mapping[fqn] = filename

            # Call the existing consolidation function with the filtered mapping
            _consolidate_safetensors_files(
                input_dir=input_dir,
                output_dir=output_dir,
                fqn_to_file_mapping=filtered_filename_mapping,
                num_threads=num_threads,
                use_staging=use_staging,
                staging_dir=staging_dir,
                cast_dtype=cast_dtype,
                fqn_to_dtype_mapping=fqn_to_dtype_mapping,
            )
    except Exception as exc:
        local_exception = exc
        local_failure = f"rank {rank} while {operation}: {type(exc).__name__}: {exc}"

    # Doubles as a barrier: the index must not be published before every rank has written
    # its assigned shards.
    _raise_if_any_rank_failed(local_exception, local_failure, distributed, rank, world_size, process_group)

    # Write overall model.index.safetensors.json file with weight map (rank 0 only)
    local_exception = None
    local_failure = None
    if rank == 0:
        try:
            _write_overall_metadata_file_from_shards(
                input_dir,
                output_dir,
                fqn_to_index_mapping,
                cast_dtype,
                fqn_to_dtype_mapping,
            )
        except Exception as exc:
            local_exception = exc
            local_failure = f"rank {rank} while writing the consolidated safetensors index: {type(exc).__name__}: {exc}"
    _raise_if_any_rank_failed(local_exception, local_failure, distributed, rank, world_size, process_group)

    logger.debug(
        "Rank %d: Done consolidating. Processed %d unique indices in %.2f secs.",
        rank,
        len(indices_for_this_rank),
        time.time() - start_time,
    )
    if distributed and rank == 0:
        logger.debug("Total time taken: %.2f secs.", time.time() - start_time)
