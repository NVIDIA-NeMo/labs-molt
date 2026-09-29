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

import builtins
import json
import logging
import os
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from safetensors.torch import load_file, save_file

from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import _write_sub_tensor_to_file_optimized, consolidate_safetensors_files, consolidate_safetensors_files_on_every_rank
from nemo_automodel.components.checkpoint._backports.hf_utils import CUSTOM_METADATA_KEY

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


@pytest.mark.run_only_on("CPU")
def test_write_scalar_tensor(tmp_path):
    """Ensure that a 0-dim (scalar) tensor shard is written to the output file.

    Regression test for a bug where `_write_sub_tensor_to_file_optimized` used to
    early-return on ``tensor_shape == []`` and therefore omitted scalar payloads,
    which produced corrupt `.safetensors` files (incomplete metadata).
    """

    # Prepare an empty temporary output file
    output_file = tmp_path / "scalar_tensor.bin"
    output_file.write_bytes(b"")  # create the file

    # Fake scalar tensor payload (2-byte BF16 value)
    sub_tensor_bytes = b"\x34\x12"
    element_size = len(sub_tensor_bytes)  # 2 bytes for BF16

    # Prepare destination buffer for a scalar tensor (element_size bytes)
    full_tensor_mv = memoryview(bytearray(element_size))

    # Call the routine under test: scalar has empty shapes and offsets
    _write_sub_tensor_to_file_optimized(
        full_tensor_mv,
        sub_tensor_bytes,
        element_size,
        tensor_shape=[],  # scalar
        sub_tensor_offsets=[],
        sub_tensor_shape=[],
    )

    # Emulate file write as done by the caller in production code
    output_file.write_bytes(full_tensor_mv.tobytes())

    # The file must now contain exactly the scalar payload
    written = output_file.read_bytes()
    assert written == sub_tensor_bytes
    assert os.path.getsize(output_file) == element_size


@pytest.mark.run_only_on("CPU")
def test_consolidate_writes_output_safetensors_without_append(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {"weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)}
    dcp_metadata = {"weight": {"saved_offsets": [0, 0]}}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )

    output_file = output_dir / "model-00001-of-00001.safetensors"
    real_open = builtins.open
    output_modes: list[str] = []

    def tracking_open(file, mode="r", *args, **kwargs):
        if os.fspath(file) == os.fspath(output_file):
            output_modes.append(mode)
            if "a" in mode:
                raise AssertionError(f"Output safetensors file was opened in append mode: {mode}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={"weight": 1},
    )

    # Stop tracking before verifying output to avoid counting the read-open from load_file().
    monkeypatch.setattr(builtins, "open", real_open)

    output_tensors = load_file(output_file)
    torch.testing.assert_close(output_tensors["weight"], tensors["weight"])
    assert output_modes == ["wb"]


@pytest.mark.run_only_on("CPU")
def test_consolidate_preserves_semantics_for_multi_shard_outputs_without_append(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    expected_tensors = {
        "linear.weight": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "norm.weight": torch.arange(4, dtype=torch.float32).to(torch.float16),
    }
    save_file(
        {
            "linear.weight": expected_tensors["linear.weight"][:2].contiguous(),
            "norm.weight": expected_tensors["norm.weight"],
        },
        input_dir / "model-00001-of-00002.safetensors",
        metadata={
            CUSTOM_METADATA_KEY: json.dumps(
                {
                    "linear.weight": {"saved_offsets": [0, 0]},
                    "norm.weight": {"saved_offsets": [0]},
                }
            )
        },
    )
    save_file(
        {"linear.weight": expected_tensors["linear.weight"][2:].contiguous()},
        input_dir / "model-00002-of-00002.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps({"linear.weight": {"saved_offsets": [2, 0]}})},
    )

    expected_weight_map = {
        "linear.weight": "model-00001-of-00002.safetensors",
        "norm.weight": "model-00002-of-00002.safetensors",
    }
    output_paths = {os.fspath(output_dir / file_name) for file_name in expected_weight_map.values()}
    real_open = builtins.open
    output_modes: dict[str, list[str]] = {}

    def tracking_open(file, mode="r", *args, **kwargs):
        file_path = os.fspath(file)
        if file_path in output_paths:
            output_modes.setdefault(os.path.basename(file_path), []).append(mode)
            if "a" in mode:
                raise AssertionError(f"Output safetensors file was opened in append mode: {mode}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={"linear.weight": 1, "norm.weight": 2},
        num_threads=2,
    )

    # Stop tracking before verifying output to avoid counting read-opens from load_file().
    monkeypatch.setattr(builtins, "open", real_open)

    output_tensors = {}
    for file_name in expected_weight_map.values():
        output_tensors.update(load_file(output_dir / file_name))

    assert set(output_tensors) == set(expected_tensors)
    for name, expected_tensor in expected_tensors.items():
        assert output_tensors[name].shape == expected_tensor.shape
        assert output_tensors[name].dtype is expected_tensor.dtype
        torch.testing.assert_close(output_tensors[name], expected_tensor)

    with open(output_dir / "model.safetensors.index.json", "r") as f:
        index = json.load(f)
    assert index["weight_map"] == expected_weight_map
    expected_total_size = sum(tensor.numel() * tensor.element_size() for tensor in expected_tensors.values())
    assert index["metadata"]["total_size"] == expected_total_size
    assert output_modes == {
        "model-00001-of-00002.safetensors": ["wb"],
        "model-00002-of-00002.safetensors": ["wb"],
    }


@pytest.mark.run_only_on("CPU")
def test_consolidate_casts_float_tensors_only_when_cast_dtype_is_set(tmp_path, caplog):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {
        "float_weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "int_weight": torch.arange(4, dtype=torch.int64),
    }
    dcp_metadata = {name: {"saved_offsets": [0 for _ in tensor.shape]} for name, tensor in tensors.items()}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )
    caplog.set_level(logging.INFO)

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={"float_weight": 1, "int_weight": 1},
        cast_dtype=torch.bfloat16,
    )

    output_tensors = load_file(output_dir / "model-00001-of-00001.safetensors")
    assert output_tensors["float_weight"].dtype is torch.bfloat16
    assert output_tensors["int_weight"].dtype is torch.int64
    torch.testing.assert_close(output_tensors["float_weight"], tensors["float_weight"].to(torch.bfloat16))
    torch.testing.assert_close(output_tensors["int_weight"], tensors["int_weight"])

    with open(output_dir / "model.safetensors.index.json", "r") as f:
        index = json.load(f)
    expected_total_size = tensors["float_weight"].numel() * 2 + tensors["int_weight"].numel() * 8
    assert index["metadata"]["total_size"] == expected_total_size
    assert "Requested cast dtype torch.bfloat16 for consolidation." in caplog.text
    assert "tensors already in this dtype, FP8 tensors, and non-floating tensors are unchanged." in caplog.text


@pytest.mark.run_only_on("CPU")
def test_consolidate_cast_dtype_does_not_cast_fp8_tensors(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {
        "fp8_weight": torch.arange(4, dtype=torch.float32).to(torch.float8_e4m3fn),
        "fp32_weight": torch.arange(4, dtype=torch.float32),
    }
    dcp_metadata = {name: {"saved_offsets": [0 for _ in tensor.shape]} for name, tensor in tensors.items()}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={"fp8_weight": 1, "fp32_weight": 1},
        cast_dtype=torch.bfloat16,
    )

    output_tensors = load_file(output_dir / "model-00001-of-00001.safetensors")
    assert output_tensors["fp8_weight"].dtype is torch.float8_e4m3fn
    assert output_tensors["fp32_weight"].dtype is torch.bfloat16
    torch.testing.assert_close(output_tensors["fp8_weight"].to(torch.float32), tensors["fp8_weight"].to(torch.float32))
    torch.testing.assert_close(output_tensors["fp32_weight"], tensors["fp32_weight"].to(torch.bfloat16))


@pytest.mark.run_only_on("CPU")
def test_consolidate_cast_dtype_preserves_mapped_fp32_tensors(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {
        "protected_fp32_weight": torch.arange(4, dtype=torch.float32),
        "ordinary_weight": torch.arange(4, dtype=torch.float32),
        "ordinary_with_original_dtype": torch.arange(4, dtype=torch.float32),
    }
    dcp_metadata = {name: {"saved_offsets": [0 for _ in tensor.shape]} for name, tensor in tensors.items()}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={name: 1 for name in tensors},
        cast_dtype=torch.float16,
        fqn_to_dtype_mapping={
            "protected_fp32_weight": "F32",
            "ordinary_with_original_dtype": "BF16",
        },
    )

    output_tensors = load_file(output_dir / "model-00001-of-00001.safetensors")
    assert output_tensors["protected_fp32_weight"].dtype is torch.float32
    assert output_tensors["ordinary_weight"].dtype is torch.float16
    assert output_tensors["ordinary_with_original_dtype"].dtype is torch.float16
    torch.testing.assert_close(output_tensors["protected_fp32_weight"], tensors["protected_fp32_weight"])
    torch.testing.assert_close(output_tensors["ordinary_weight"], tensors["ordinary_weight"].to(torch.float16))
    torch.testing.assert_close(
        output_tensors["ordinary_with_original_dtype"], tensors["ordinary_with_original_dtype"].to(torch.float16)
    )

    with open(output_dir / "model.safetensors.index.json", "r") as f:
        index = json.load(f)
    expected_total_size = (
        tensors["protected_fp32_weight"].numel() * 4
        + (tensors["ordinary_weight"].numel() + tensors["ordinary_with_original_dtype"].numel()) * 2
    )
    assert index["metadata"]["total_size"] == expected_total_size


@pytest.mark.run_only_on("CPU")
def test_consolidate_preserves_original_hf_float_dtypes_when_available(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {
        "bf16_weight": torch.arange(4, dtype=torch.float32).reshape(2, 2),
        "fp32_weight": torch.arange(4, dtype=torch.float32),
        "missing_original_dtype": torch.arange(2, dtype=torch.float32),
    }
    dcp_metadata = {name: {"saved_offsets": [0 for _ in tensor.shape]} for name, tensor in tensors.items()}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={name: 1 for name in tensors},
        fqn_to_dtype_mapping={"bf16_weight": "BF16", "fp32_weight": "F32"},
    )

    output_tensors = load_file(output_dir / "model-00001-of-00001.safetensors")
    assert output_tensors["bf16_weight"].dtype is torch.bfloat16
    assert output_tensors["fp32_weight"].dtype is torch.float32
    assert output_tensors["missing_original_dtype"].dtype is torch.float32
    torch.testing.assert_close(output_tensors["bf16_weight"], tensors["bf16_weight"].to(torch.bfloat16))
    torch.testing.assert_close(output_tensors["fp32_weight"], tensors["fp32_weight"])
    torch.testing.assert_close(output_tensors["missing_original_dtype"], tensors["missing_original_dtype"])

    with open(output_dir / "model.safetensors.index.json", "r") as f:
        index = json.load(f)
    expected_total_size = (
        tensors["bf16_weight"].numel() * 2
        + tensors["fp32_weight"].numel() * 4
        + tensors["missing_original_dtype"].numel() * 4
    )
    assert index["metadata"]["total_size"] == expected_total_size


@pytest.mark.run_only_on("CPU")
def test_consolidate_keeps_saved_dtype_when_original_hf_dtype_metadata_is_missing(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {
        "fp32_weight": torch.arange(4, dtype=torch.float32),
        "bf16_weight": torch.arange(4, dtype=torch.bfloat16),
    }
    dcp_metadata = {name: {"saved_offsets": [0 for _ in tensor.shape]} for name, tensor in tensors.items()}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={name: 1 for name in tensors},
    )

    output_tensors = load_file(output_dir / "model-00001-of-00001.safetensors")
    assert output_tensors["fp32_weight"].dtype is torch.float32
    assert output_tensors["bf16_weight"].dtype is torch.bfloat16
    torch.testing.assert_close(output_tensors["fp32_weight"], tensors["fp32_weight"])
    torch.testing.assert_close(output_tensors["bf16_weight"], tensors["bf16_weight"])


@pytest.mark.run_only_on("CPU")
def test_consolidate_ignores_mapping_keys_missing_from_input_shards(tmp_path, caplog):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {"transformer.wte.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)}
    dcp_metadata = {"transformer.wte.weight": {"saved_offsets": [0, 0]}}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )
    caplog.set_level(logging.WARNING)

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={"transformer.wte.weight": 1, "lm_head.weight": 1},
    )

    output_tensors = load_file(output_dir / "model-00001-of-00001.safetensors")
    assert set(output_tensors) == {"transformer.wte.weight"}
    torch.testing.assert_close(output_tensors["transformer.wte.weight"], tensors["transformer.wte.weight"])

    with open(output_dir / "model.safetensors.index.json", "r") as f:
        index = json.load(f)
    assert index["weight_map"] == {"transformer.wte.weight": "model-00001-of-00001.safetensors"}
    assert index["metadata"]["total_size"] == tensors["transformer.wte.weight"].numel() * 4
    assert "Ignoring 1 tensor(s) from the consolidation shard mapping" in caplog.text
    assert "lm_head.weight" in caplog.text


@pytest.mark.run_only_on("CPU")
def test_consolidate_keeps_float_when_original_dtype_is_quantized(tmp_path, caplog):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    tensors = {"weight": torch.arange(4, dtype=torch.float32)}
    dcp_metadata = {"weight": {"saved_offsets": [0]}}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )
    caplog.set_level(logging.WARNING)

    consolidate_safetensors_files(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fqn_to_index_mapping={"weight": 1},
        fqn_to_dtype_mapping={"weight": "F8_E4M3"},
    )

    output_tensors = load_file(output_dir / "model-00001-of-00001.safetensors")
    assert output_tensors["weight"].dtype is torch.float32
    torch.testing.assert_close(output_tensors["weight"], tensors["weight"])
    assert "Original checkpoint tensor(s) were quantized or packed" in caplog.text
    assert "weight: original F8_E4M3, saved F32" in caplog.text


@pytest.mark.run_only_on("CPU")
def test_every_rank_consolidation_uses_supplied_group_for_failure_sync():
    process_group = MagicMock()
    call_order = []

    def _record_sync(*args, **kwargs):
        call_order.append("sync")

    def _record_index_write(*args, **kwargs):
        call_order.append("index")

    with (
        patch(
            "nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.dist.is_available",
            return_value=True,
        ),
        patch(
            "nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.dist.get_rank",
            return_value=0,
        ) as get_rank,
        patch(
            "nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.dist.get_world_size",
            return_value=2,
        ) as get_world_size,
        patch(
            "nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.dist.all_gather_object",
            side_effect=_record_sync,
        ) as all_gather_object,
        patch(
            "nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors._consolidate_safetensors_files"
        ),
        patch(
            "nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors."
            "_write_overall_metadata_file_from_shards",
            side_effect=_record_index_write,
        ),
    ):
        consolidate_safetensors_files_on_every_rank(
            input_dir="input",
            output_dir="output",
            fqn_to_index_mapping={"weight": 1},
            process_group=process_group,
        )

    get_rank.assert_called_once_with(group=process_group)
    get_world_size.assert_called_once_with(group=process_group)
    assert all_gather_object.call_count == 2
    for call in all_gather_object.call_args_list:
        failures, local_failure = call.args
        assert failures == [None, None]
        assert local_failure is None
        assert call.kwargs == {"group": process_group}
    # The index is published only after every rank has finished writing its assigned shards.
    assert call_order == ["sync", "index", "sync"]


@pytest.mark.run_only_on("CPU")
def test_every_rank_consolidation_explains_peer_failure_during_sync():
    process_group = MagicMock()

    with (
        patch(
            "nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.dist.is_available",
            return_value=True,
        ),
        patch(
            "nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.dist.get_rank",
            return_value=1,
        ),
        patch(
            "nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.dist.get_world_size",
            return_value=2,
        ),
        patch(
            "nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors.dist.all_gather_object",
            side_effect=RuntimeError("Connection closed by peer"),
        ),
    ):
        with pytest.raises(RuntimeError, match="A peer may have exited before reporting its original exception") as exc:
            consolidate_safetensors_files_on_every_rank(
                input_dir="input",
                output_dir="output",
                fqn_to_index_mapping={"weight": 1},
                process_group=process_group,
            )

    assert "Synchronization error: RuntimeError: Connection closed by peer" in str(exc.value)


def _run_two_rank_consolidation_failure_worker(
    rank: int,
    world_size: int,
    init_file: str,
    input_dir: str,
    output_dir: str,
    error_dir: str,
) -> None:
    """Verify that both Gloo ranks receive the originating consolidation error."""
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        try:
            consolidate_safetensors_files_on_every_rank(
                input_dir=input_dir,
                output_dir=output_dir,
                fqn_to_index_mapping={"weight": 1},
            )
        except RuntimeError as exc:
            Path(error_dir, f"rank_{rank}.txt").write_text(str(exc))
        else:
            raise AssertionError("Expected distributed consolidation to fail")
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.run_only_on("CPU")
def test_every_rank_consolidation_surfaces_originating_rank_error(tmp_path):
    """A rank-0 index failure is reported identically instead of closing peers."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    tensors = {
        "weight": torch.ones(2),
        "adapter.created.bias": torch.zeros(2),
    }
    dcp_metadata = {fqn: {"saved_offsets": [0]} for fqn in tensors}
    save_file(
        tensors,
        input_dir / "model-00001-of-00001.safetensors",
        metadata={CUSTOM_METADATA_KEY: json.dumps(dcp_metadata)},
    )

    torch.multiprocessing.spawn(
        _run_two_rank_consolidation_failure_worker,
        args=(
            2,
            str(tmp_path / "gloo_init"),
            str(input_dir),
            str(output_dir),
            str(tmp_path),
        ),
        nprocs=2,
        join=True,
    )

    errors = [(tmp_path / f"rank_{rank}.txt").read_text() for rank in range(2)]
    assert errors[0] == errors[1]
    assert "rank 0 while writing the consolidated safetensors index: KeyError: 'adapter.created.bias'" in errors[0]


# =============================================================================
# Tests for _maybe_rename_index_for_diffusers
# =============================================================================


# =============================================================================
# Tests for _HuggingFaceStorageWriter.finish — single-rank consolidation path
# =============================================================================


class TestStorageWriterFinishDiffusersCompatible:
    """Tests that finish() renames the index when diffusers_compatible=True."""

    @pytest.fixture()
    def _mock_consolidate(self, monkeypatch):
        """Patch consolidate_safetensors_files so finish() doesn't need real shards."""
        self._consolidate_calls = []

        def _fake(*, input_dir, output_dir, **kwargs):
            self._consolidate_calls.append(output_dir)
            index = os.path.join(output_dir, "model.safetensors.index.json")
            with open(index, "w") as f:
                json.dump({"weight_map": {}}, f)

        monkeypatch.setattr(
            "nemo_automodel.components.checkpoint._backports.hf_storage.consolidate_safetensors_files",
            _fake,
        )


