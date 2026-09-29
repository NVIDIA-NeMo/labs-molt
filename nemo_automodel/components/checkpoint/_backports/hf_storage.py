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
# https://github.com/pytorch/pytorch/pull/155940
# https://github.com/pytorch/pytorch/pull/155707
# pylint: disable=missing-function-docstring,line-too-long

import dataclasses
import json
import logging
import mmap
import os
import queue
import re
from typing import Any

import torch
from torch.distributed._shard._utils import narrow_tensor_by_index
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.planner import (
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
)
from torch.distributed.checkpoint.storage import WriteResult
from torch.futures import Future

from nemo_automodel.components.checkpoint._backports._fsspec_filesystem import FsspecReader, FsspecWriter
from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import consolidate_safetensors_files
from nemo_automodel.components.checkpoint._backports.filesystem import SerializationFormat
from nemo_automodel.components.checkpoint._backports.hf_utils import (
    CUSTOM_METADATA_KEY,
    DATA_OFFSETS_KEY,
    DEFAULT_EXTRA_METADATA_KEY,
    DTYPE_KEY,
    SAVED_OFFSETS_KEY,
    SHAPE_KEY,
    SUFFIX,
    _gen_file_name,
    _get_dtype,
    _get_safetensors_file_metadata,
    _HFStorageInfo,
    _metadata_fn,
)

__all__ = ["_HuggingFaceStorageWriter", "_HuggingFaceStorageReader"]

_DIFFUSERS_INDEX_FN = "diffusion_pytorch_model.safetensors.index.json"
logger = logging.getLogger(__name__)


def _is_integrated_cuda_device(device: torch.device) -> bool:
    """Whether CUDA reports *device* as sharing physical memory with the host."""
    if device.type != "cuda":
        return False
    properties = torch.cuda.get_device_properties(device)
    return bool(getattr(properties, "is_integrated", False))


class _HuggingFaceStorageWriter(FsspecWriter):
    """
    A writer that writes to a huggingface repository in the huggingface format.
    Uses Fsspec back-end to communicate with back-end storage.
    Fsspec registration of the storage solution is required.
    """

    def __init__(
        self,
        path: str,
        fqn_to_index_mapping: dict[str, int] | None = None,
        thread_count: int = 1,
        token: str | None = None,
        save_sharded: bool = False,
        consolidated_output_path: str | None = None,
        num_threads_consolidation: int | None = None,
        staging_dir: str | None = None,
        fqn_to_dtype_mapping: dict[str, str] | None = None,
    ) -> None:
        """
        Initialize the huggingface writer pointing to path.

        Args:
            path: hf directory where the checkpoint will be read from.
                  Needs to have .safetensors files, but can be from any fsspec supported storage,
                  including localFS and hf://.
                  This needs to be a remote path if you want to enable consolidation after saving.
            fqn_to_index_mapping: A mapping from tensor FQN to the index of the file that the tensor should be written to.
                              Indices are from 1 to N, where N is the number of files. If not provided,
                              the tensors will be written to a single file. If none, then all the tensors on the
                              same rank will be written to the same file.
            token: The token to use to authenticate with huggingface hub.
            save_sharded: If True, save the checkpoint as a sharded checkpoint where every rank saves its own shard.
                        Default is False which assumes full tensors are being saved.
            consolidated_output_path: If provided, the output path where the consolidated files will be written in the finish step. This needs to be a local fs path right now.
            num_threads_consolidation: Number of threads to use for parallel processing of saving data to output files. If not provided, the default value is the number of output files.
            staging_dir: Optional directory for staging files during consolidation. If provided,
                        temp files will be created here instead of system temp.
            fqn_to_dtype_mapping: Optional mapping from tensor FQN to original HF safetensors dtype string.
        """
        if token is not None:
            super().__init__(
                path=path,
                token=token,
                serialization_format=SerializationFormat.SAFETENSORS,
            )
        else:
            super().__init__(
                path=path,
                serialization_format=SerializationFormat.SAFETENSORS,
            )
        self._fqn_to_index_mapping: dict[str, int] | None = fqn_to_index_mapping
        self._save_sharded = save_sharded
        self._consolidated_output_path = consolidated_output_path
        self._staging_dir = staging_dir
        self._fqn_to_dtype_mapping = fqn_to_dtype_mapping

        if num_threads_consolidation:
            self._num_threads_consolidation = num_threads_consolidation
        elif self._fqn_to_index_mapping:
            self._num_threads_consolidation = max(self._fqn_to_index_mapping.values())
        else:
            self._num_threads_consolidation = 1

        self.thread_count = thread_count

    def prepare_global_plan(self, plans: list[SavePlan]) -> list[SavePlan]:
        new_plans = []
        for i, plan in enumerate(plans, start=1):
            storage_data: dict[str, Any] = {}
            # save default shard mapping. We only use fqn_to_index_mapping for consolidation.
            # if self._fqn_to_index_mapping is not None:
            #     storage_data["fqn_to_index_mapping"] = self._fqn_to_index_mapping
            if self._save_sharded:
                storage_data["shard_index"] = i

            new_plans.append(dataclasses.replace(plan, storage_data=storage_data))

        return new_plans

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[list[WriteResult]]:
        if len(plan.items) == 0:
            fut: Future = Future()
            fut.set_result([])
            return fut

        # storage_plan is a map from key to file index
        storage_data: dict[str, Any] = plan.storage_data
        storage_plan: dict[str, int] | None = None
        shard_index: int | None = None
        if "fqn_to_index_mapping" in storage_data:
            storage_plan = storage_data["fqn_to_index_mapping"]
        if "shard_index" in storage_data:
            shard_index = storage_data["shard_index"]

        buckets = self._split_by_storage_plan(storage_plan, plan.items)
        highest_index = max(storage_plan.values()) if storage_plan is not None else 1

        file_queue: queue.Queue = queue.Queue()
        for file_index, write_items in buckets.items():
            file_name = _gen_file_name(file_index, highest_index, shard_index)
            file_queue.put((self.fs.concat_path(self.path, file_name), file_name, write_items))

        return super()._write_data(planner, file_queue)

    def finish(self, metadata: Metadata, results: list[list[WriteResult]]) -> None:
        if self._save_sharded and not self._consolidated_output_path:
            return
        if self._save_sharded:
            consolidate_safetensors_files(
                input_dir=self.path,
                output_dir=self._consolidated_output_path,
                num_threads=self._num_threads_consolidation,
                fqn_to_index_mapping=self._fqn_to_index_mapping,
                use_staging=self._staging_dir is not None,
                staging_dir=self._staging_dir,
                fqn_to_dtype_mapping=self._fqn_to_dtype_mapping,
            )
            return

        metadata_to_write = {}
        storage_md = {}
        total_size = 0
        for wr_list in results:
            storage_md.update({wr.index.fqn: wr.storage_data.relative_path for wr in wr_list})
            total_size += sum([wr.storage_data.length for wr in wr_list])
        metadata_to_write["metadata"] = {"total_size": total_size}
        metadata_to_write["weight_map"] = storage_md

        metadata_path = self.fs.concat_path(self.path, f"{_metadata_fn}")
        with self.fs.create_stream(metadata_path, "w") as metadata_file:
            json.dump(metadata_to_write, metadata_file, indent=2)

    def _split_by_storage_plan(
        self, storage_plan: dict[str, int] | None, items: list[WriteItem]
    ) -> dict[int, list[WriteItem]]:
        # storage_plan is a map from key to index
        if storage_plan is None:
            return {1: items}

        buckets = {}
        for item in items:
            key = item.index.fqn

            idx = storage_plan[key]
            if idx not in buckets:
                buckets[idx] = [item]
            else:
                buckets[idx].append(item)

        return buckets

    @property
    def metadata_path(self) -> str:
        return _metadata_fn


class _HuggingFaceStorageReader(FsspecReader):
    """
    A reader that reads from a huggingface repository in the huggingface format.
    Uses in Fsspec back-end to communicate with storage.
    Fsspec registration of the storage solution is required.
    """

    def __init__(self, path: str, token: str | None = None, key_mapping: dict[str, str] | None = None) -> None:
        """
        Initialize the huggingface reader pointing to path.

        Args:
            path: hf directory where the checkpoint will be read from.
            Needs to have .safetensors file, but can be from any fsspec supported storage,
            including localFS and hf://.
            token: The token to use to authenticate with huggingface hub.
            key_mapping: VLMs in HuggingFace can have their FQNs remapped at load time. This means that the state dict keys are not the same as the loaded model's FQNs.
                         This mapping is used to map the state dict keys to the loaded model's FQNs.
        """

        if token is not None:
            super().__init__(path=path, token=token)
        else:
            super().__init__(path=path)

        self.key_mapping = key_mapping
        self._metadata_cache: Metadata | None = None

    def reset(self, checkpoint_id: str | os.PathLike | None = None) -> None:
        """Reset reader state and discard metadata only when the checkpoint changes."""
        previous_path = str(self.path)
        super().reset(checkpoint_id)
        if checkpoint_id is not None and str(self.path) != previous_path:
            self._metadata_cache = None

    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        per_file: dict[str, list[ReadItem]] = {}

        for read_item in plan.items:
            item_md: _HFStorageInfo = self.storage_data[read_item.storage_index]
            file_name = item_md.relative_path
            per_file.setdefault(file_name, []).append(read_item)

        for file_name, reqs in per_file.items():
            # Prefer mmap for local files: each request copies out only its narrowed slice,
            # so only those pages fault in -- and they are file-backed (reclaimable) rather
            # than anonymous host RAM. The previous path read every full tensor into a host
            # bytearray (plus a second copy), which host-OOM-kills very large checkpoints
            # (e.g. a 355B fp8 model with 8 ranks/node). Falls back to the streaming read for
            # remote/non-local files.
            mm = None
            mfile = None
            try:
                if os.path.isfile(file_name):
                    mfile = open(file_name, "rb")  # noqa: SIM115
                    # Copy-on-write mmap keeps pages file-backed unless mutated, while
                    # giving torch.frombuffer a writable view so it does not warn.
                    mm = mmap.mmap(mfile.fileno(), 0, access=mmap.ACCESS_COPY)
            except (OSError, ValueError):
                if mm is not None:
                    mm.close()
                    mm = None
                if mfile is not None:
                    mfile.close()
                    mfile = None

            try:
                if mm is not None:
                    view = memoryview(mm)
                    for req in reqs:
                        item_md = self.storage_data[req.storage_index]
                        # torch.frombuffer is zero-copy but requires the byte offset to be
                        # aligned to the dtype; safetensors headers can misalign (e.g. bf16),
                        # so copy just that one tensor's bytes when unaligned.
                        if item_md.offset % item_md.dtype.itemsize == 0:
                            numel = item_md.length // item_md.dtype.itemsize
                            tensor = torch.frombuffer(
                                view, dtype=item_md.dtype, count=numel, offset=item_md.offset
                            ).reshape(item_md.shape)
                            # This view still faults pages from the file mmap.
                            file_backed = True
                        else:
                            tensor = torch.frombuffer(
                                bytearray(view[item_md.offset : item_md.offset + item_md.length]),
                                dtype=item_md.dtype,
                            ).reshape(item_md.shape)
                            # The alignment fallback already owns resident bytes.
                            file_backed = False
                        tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                        target_tensor = planner.resolve_tensor(req).detach()

                        assert target_tensor.size() == tensor.size(), (
                            f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                        )

                        # On integrated GPUs, copying a file-backed mmap directly to CUDA can
                        # migrate it page-by-page and take minutes per tensor. Stage only this
                        # requested slice in anonymous host memory; unlike the full-state loader,
                        # peak host memory is bounded by one tensor. Discrete GPUs keep the
                        # zero-copy mmap source used to avoid host OOM on very large checkpoints.
                        if file_backed and _is_integrated_cuda_device(target_tensor.device):
                            tensor = tensor.clone()

                        # copy_ from a pageable host buffer is synchronous, so the mmap can be
                        # released right after this loop without racing an in-flight H2D copy.
                        target_tensor.copy_(tensor)
                        planner.commit_tensor(req, target_tensor)
                        del tensor
                    view.release()
                else:
                    with self.fs.create_stream(file_name, "rb") as stream:
                        for req in reqs:
                            item_md = self.storage_data[req.storage_index]

                            stream.seek(item_md.offset)
                            tensor_bytes = bytearray(stream.read(item_md.length))

                            tensor = torch.frombuffer(
                                tensor_bytes,
                                dtype=item_md.dtype,
                            )
                            tensor = tensor.reshape(item_md.shape)
                            tensor = narrow_tensor_by_index(tensor, req.storage_offsets, req.lengths)
                            target_tensor = planner.resolve_tensor(req).detach()

                            assert target_tensor.size() == tensor.size(), (
                                f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                            )

                            target_tensor.copy_(tensor)
                            planner.commit_tensor(req, target_tensor)
            finally:
                if mm is not None:
                    mm.close()
                if mfile is not None:
                    mfile.close()

        fut: Future = Future()
        fut.set_result(None)
        return fut

    def read_metadata(self) -> Metadata:
        if self._metadata_cache is not None:
            if getattr(self._metadata_cache, "storage_meta", None) is not None:
                self._metadata_cache.storage_meta.load_id = self.load_id
            return self._metadata_cache

        state_dict_metadata: dict[str, TensorStorageMetadata] = {}
        storage_data: dict[MetadataIndex, _HFStorageInfo] = {}

        safetensors_files = []
        for file in self.fs.ls(self.path):
            if file.endswith(SUFFIX):
                safetensors_files.append(file)

        for safetensor_file in safetensors_files:
            with self.fs.create_stream(safetensor_file, "rb") as f:
                safetensors_metadata, metadata_size = _get_safetensors_file_metadata(f)
                custom_metadata = safetensors_metadata.get(DEFAULT_EXTRA_METADATA_KEY)

                dcp_sharding_info = None
                if custom_metadata and custom_metadata.get(CUSTOM_METADATA_KEY):
                    dcp_sharding_info = json.loads(custom_metadata.get(CUSTOM_METADATA_KEY))

                for key, val in safetensors_metadata.items():
                    if key == DEFAULT_EXTRA_METADATA_KEY:
                        continue

                    key = _get_key_renaming_mapping(key, self.key_mapping)

                    # construct state_dict_metadata
                    if dcp_sharding_info is not None:
                        offset = dcp_sharding_info[key][SAVED_OFFSETS_KEY]
                    else:
                        offset = [0] * len(val[SHAPE_KEY])

                    if key not in state_dict_metadata:
                        state_dict_metadata[key] = TensorStorageMetadata(
                            properties=TensorProperties(dtype=_get_dtype(val[DTYPE_KEY])),
                            size=torch.Size([saved + offset for saved, offset in zip(val[SHAPE_KEY], offset)]),
                            chunks=[
                                ChunkStorageMetadata(
                                    offsets=torch.Size(offset),
                                    sizes=torch.Size(val[SHAPE_KEY]),
                                )
                            ],
                        )
                    else:
                        state_dict_metadata[key].chunks.append(
                            ChunkStorageMetadata(torch.Size(offset), sizes=torch.Size(val[SHAPE_KEY]))
                        )
                        size = list(state_dict_metadata[key].size)
                        for i in range(len(size)):
                            size[i] = max(size[i], val[SHAPE_KEY][i] + offset[i])
                        state_dict_metadata[key].size = torch.Size(size)

                    # construct storage data
                    if dcp_sharding_info is not None:
                        metadata_index = MetadataIndex(fqn=key, offset=dcp_sharding_info[key][SAVED_OFFSETS_KEY])
                    else:
                        metadata_index = MetadataIndex(fqn=key, offset=[0] * len(val[SHAPE_KEY]))
                    storage_data[metadata_index] = _HFStorageInfo(
                        relative_path=safetensor_file,
                        offset=val[DATA_OFFSETS_KEY][0] + metadata_size,
                        length=val[DATA_OFFSETS_KEY][1] - val[DATA_OFFSETS_KEY][0],
                        shape=torch.Size(val[SHAPE_KEY]),
                        dtype=_get_dtype(val[DTYPE_KEY]),
                    )

        metadata = Metadata(
            state_dict_metadata=state_dict_metadata,  # type: ignore[arg-type]
            storage_data=storage_data,
        )

        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id  # type: ignore[union-attr]

        self._metadata_cache = metadata
        return metadata


def _extract_file_index_with_status(filename: str) -> tuple[int, bool]:
    """Return the 1-based shard index encoded in a safetensors filename.

    Supported patterns::

        model-00001-of-00008.safetensors
        model.safetensors-00001-of-00008.safetensors
        shard-00000-model-00002-of-00008.safetensors
        model.safetensors  (single-file checkpoints)

    Args:
        filename: The (relative) safetensors filename.

    Returns:
        The numeric shard index and whether the filename was recognized.
        Unparseable filenames return ``(1, False)``.
    """
    # Strip any leading directory components so we only deal with the basename.
    basename = filename.split("/")[-1]

    # Single-file checkpoints which usually carry the name ``model.safetensors``.
    if basename == "model.safetensors":
        return 1, True

    match = re.search(r"-(\d+)-of-(\d+)\.safetensors$", basename)
    if match:
        idx = int(match.group(1).lstrip("0") or "0")
        total = int(match.group(2).lstrip("0") or "0")
        if 1 <= idx <= total:
            return idx, True

    # default to the first shard.
    return 1, False


def get_fqn_to_file_index_mapping(
    reference_model_path: str, key_mapping: dict[str, str] | None = None
) -> dict[str, int]:
    """
    Get the FQN to file index mapping from the metadata.

    Args:
        reference_model_path: Path to reference model to copy file structure from.

    Returns:
        A mapping from tensor FQN to the index of the file that the tensor should be written to.
        Indices are from 1 to N, where N is the number of files.
    """
    fqn_to_file_index_mapping: dict[str, int] = {}
    fqn_to_filename_mapping: dict[str, str] = {}
    filename_to_index: dict[str, tuple[int, bool]] = {}

    index_file = os.path.join(reference_model_path, _metadata_fn)
    if os.path.isfile(index_file):
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        for fqn, filename in weight_map.items():
            fqn = _get_key_renaming_mapping(fqn, key_mapping)
            idx, parsed = _extract_file_index_with_status(filename)
            fqn_to_file_index_mapping[str(fqn)] = idx
            fqn_to_filename_mapping[str(fqn)] = filename
            filename_to_index[filename] = (idx, parsed)
    else:
        hf_reader = _HuggingFaceStorageReader(reference_model_path)
        metadata = hf_reader.read_metadata()

        for md_index, storage_info in metadata.storage_data.items():
            fqn = getattr(md_index, "fqn", md_index)
            fqn = _get_key_renaming_mapping(fqn, key_mapping)
            filename = storage_info.relative_path
            idx, parsed = _extract_file_index_with_status(filename)
            fqn_to_file_index_mapping[str(fqn)] = idx
            fqn_to_filename_mapping[str(fqn)] = filename
            filename_to_index[filename] = (idx, parsed)

    distinct_filenames = set(filename_to_index)
    parsed_indices = {idx for idx, parsed in filename_to_index.values() if parsed}
    expected_indices = set(range(1, len(distinct_filenames) + 1))

    # Expected parsed indices are 1..N for N observed source files.
    # If not, fall back to sorted filename order.
    if len(distinct_filenames) > 1 and parsed_indices != expected_indices:
        filename_to_dense_index = {filename: idx for idx, filename in enumerate(sorted(distinct_filenames), start=1)}
        for fqn, filename in fqn_to_filename_mapping.items():
            fqn_to_file_index_mapping[fqn] = filename_to_dense_index[filename]

        logger.warning(
            "Safetensors shard index parsing failed or produced unexpected indices under %s. "
            "Expected indices to be 1..%d; falling back to sorted filename order for output indices. "
            "Example assignments: %s",
            reference_model_path,
            len(distinct_filenames),
            dict(list(filename_to_dense_index.items())[:5]),
        )

    return fqn_to_file_index_mapping


def get_fqn_to_dtype_mapping(reference_model_path: str, key_mapping: dict[str, str] | None = None) -> dict[str, str]:
    """
    Get the FQN to original safetensors dtype mapping from HF shard headers.

    Args:
        reference_model_path: Path to reference model to copy dtype metadata from.
        key_mapping: Optional regex key mapping applied in the same way as load-time HF key conversion.

    Returns:
        A mapping from tensor FQN to the original safetensors dtype string.
    """
    fqn_to_dtype_mapping: dict[str, str] = {}
    filenames: set[str] = set()

    index_file = os.path.join(reference_model_path, _metadata_fn)
    if os.path.isfile(index_file):
        with open(index_file) as f:
            index = json.load(f)
        filenames.update(index.get("weight_map", {}).values())
    else:
        filenames.update(filename for filename in os.listdir(reference_model_path) if filename.endswith(SUFFIX))

    for filename in sorted(filenames):
        shard_path = os.path.join(reference_model_path, filename)
        if not os.path.isfile(shard_path):
            continue
        with open(shard_path, "rb") as f:
            safetensors_metadata, _ = _get_safetensors_file_metadata(f)
        for key, val in safetensors_metadata.items():
            if key == DEFAULT_EXTRA_METADATA_KEY:
                continue
            fqn = _get_key_renaming_mapping(key, key_mapping)
            fqn_to_dtype_mapping[str(fqn)] = val[DTYPE_KEY]

    return fqn_to_dtype_mapping


# the following function is taken from https://github.com/huggingface/transformers/blob/b85ed49e0a5f1bd9fd887f497d055b22b9319a12/src/transformers/modeling_utils.py#L4989-L5047
def _get_key_renaming_mapping(
    key: str,
    key_mapping: dict[str, str] | None = None,
) -> str:
    if key_mapping is None:
        return key

    # Optionally map the key according to `key_mapping`
    for pattern, replacement in key_mapping.items():
        new_key, n_replace = re.subn(pattern, replacement, key)
        # Early exit of the loop
        if n_replace > 0:
            return new_key
    return key
