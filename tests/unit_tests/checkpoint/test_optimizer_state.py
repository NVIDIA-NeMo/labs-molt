# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import os
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


class _ModelOwnedDTensorOptimizerModel(nn.Module):
    """Tiny globally shaped row DTensor consumed through its local shard."""

    def __init__(self, process_group: torch.distributed.ProcessGroup, *, global_rows: int = 8) -> None:
        super().__init__()
        world_size = torch.distributed.get_world_size(process_group)
        rank = torch.distributed.get_rank(process_group)
        if global_rows % world_size:
            raise ValueError("global_rows must divide evenly over the test process group")
        local_rows = global_rows // world_size
        local_weight = torch.arange(rank * local_rows, (rank + 1) * local_rows, dtype=torch.float32).unsqueeze(1)
        local_weight = local_weight.expand(-1, 2).contiguous()
        mesh = DeviceMesh.from_group(
            process_group,
            device_type="cpu",
            mesh_dim_names=("ple_owner",),
        )
        weight = DTensor.from_local(
            local_weight,
            device_mesh=mesh,
            placements=(Shard(0),),
            run_check=False,
            shape=(global_rows, 2),
            stride=(2, 1),
        )
        self.weight = nn.Parameter(weight)
        self.weight._nemo_model_owned_grad_divisor = float(world_size)


def _make_distributed_checkpointer(rank: int, checkpoint_dir: str) -> Checkpointer:
    """Build the ordinary full-training DCP wrapper for one test rank."""
    return Checkpointer(
        CheckpointingConfig(
            checkpoint_dir=checkpoint_dir,
            model_save_format="safetensors",
            save_consolidated=False,
            is_peft=False,
        ),
        dp_rank=rank,
        tp_rank=0,
        pp_rank=0,
        process_group=torch.distributed.group.WORLD,
    )


def _state_local(tensor: torch.Tensor) -> torch.Tensor:
    """Return a writable optimizer-state shard."""
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _fill_row_coded_adam_state(model: nn.Module, optimizer: torch.optim.AdamW, rank: int) -> None:
    """Materialize Adam state and overwrite it with deterministic owner rows."""
    parameter = model.weight
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    state = optimizer.state[parameter]
    local_rows = _state_local(state["exp_avg"]).shape[0]
    row_start = rank * local_rows
    rows = torch.arange(row_start, row_start + local_rows, dtype=torch.float32).unsqueeze(1)
    columns = torch.arange(2, dtype=torch.float32).unsqueeze(0)
    _state_local(state["exp_avg"]).copy_(rows * 10 + columns + 0.25)
    _state_local(state["exp_avg_sq"]).copy_(rows * 10 + columns + 0.75)
    state["step"].fill_(5)


def _assert_row_coded_adam_state(model: nn.Module, optimizer: torch.optim.AdamW, rank: int) -> None:
    """Check the deterministic state expected for this rank's row interval."""
    state = optimizer.state[model.weight]
    exp_avg = _state_local(state["exp_avg"])
    exp_avg_sq = _state_local(state["exp_avg_sq"])
    local_rows = exp_avg.shape[0]
    rows = torch.arange(rank * local_rows, (rank + 1) * local_rows, dtype=torch.float32).unsqueeze(1)
    columns = torch.arange(2, dtype=torch.float32).unsqueeze(0)
    torch.testing.assert_close(exp_avg, rows * 10 + columns + 0.25, rtol=0, atol=0)
    torch.testing.assert_close(exp_avg_sq, rows * 10 + columns + 0.75, rtol=0, atol=0)
    torch.testing.assert_close(state["step"], torch.tensor(5.0), rtol=0, atol=0)


def _make_stepped_adam_parts(
    count: int = 2,
) -> tuple[list[nn.Module], list[torch.optim.AdamW]]:
    model_parts: list[nn.Module] = []
    optimizers: list[torch.optim.AdamW] = []
    for index in range(count):
        model_part = nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model_part.parameters(), lr=0.01 * (index + 1))
        for parameter in model_part.parameters():
            parameter.grad = torch.full_like(parameter, float(index + 1))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        model_parts.append(model_part)
        optimizers.append(optimizer)
    return model_parts, optimizers


def _run_native_dtensor_optimizer_save(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _ModelOwnedDTensorOptimizerModel(torch.distributed.group.WORLD)
        optimizer = torch.optim.AdamW([model.weight], lr=0.031, foreach=False)
        _fill_row_coded_adam_state(model, optimizer, rank)
        checkpointer = _make_distributed_checkpointer(rank, checkpoint_dir)
        checkpointer.save_optimizer(optimizer, model, checkpoint_dir)
        torch.distributed.barrier()
        metadata = FileSystemReader(os.path.join(checkpoint_dir, "optim")).read_metadata().state_dict_metadata
        exp_avg_metadata = metadata["optim.state.weight.exp_avg"]
        assert tuple(exp_avg_metadata.size) == (8, 2)
        assert len(exp_avg_metadata.chunks) == world_size
    finally:
        torch.distributed.destroy_process_group()


def _run_native_dtensor_optimizer_load(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
    resaved_checkpoint_dir: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        model = _ModelOwnedDTensorOptimizerModel(torch.distributed.group.WORLD)
        optimizer = torch.optim.AdamW([model.weight], lr=0.9, foreach=False)
        checkpointer = _make_distributed_checkpointer(rank, checkpoint_dir)
        checkpointer.load_optimizer(optimizer, model, checkpoint_dir)
        _assert_row_coded_adam_state(model, optimizer, rank)
        assert optimizer.param_groups[0]["lr"] == 0.031

        resave_checkpointer = _make_distributed_checkpointer(rank, resaved_checkpoint_dir)
        resave_checkpointer.save_optimizer(optimizer, model, resaved_checkpoint_dir)
        torch.distributed.barrier()
        metadata = FileSystemReader(os.path.join(resaved_checkpoint_dir, "optim")).read_metadata().state_dict_metadata
        assert tuple(metadata["optim.state.weight.exp_avg"].size) == (8, 2)
        assert len(metadata["optim.state.weight.exp_avg"].chunks) == world_size
    finally:
        torch.distributed.destroy_process_group()


def _assert_adam_states_equal(
    actual: torch.optim.AdamW,
    expected: torch.optim.AdamW,
) -> None:
    actual_parameter = actual.param_groups[0]["params"][0]
    expected_parameter = expected.param_groups[0]["params"][0]
    actual_state = actual.state[actual_parameter]
    expected_state = expected.state[expected_parameter]

    torch.testing.assert_close(actual_state["step"], expected_state["step"])
    torch.testing.assert_close(actual_state["exp_avg"], expected_state["exp_avg"])
    torch.testing.assert_close(actual_state["exp_avg_sq"], expected_state["exp_avg_sq"])
    assert actual.param_groups[0]["lr"] == expected.param_groups[0]["lr"]


def _assert_native_state_dicts_equal(actual: dict, expected: dict) -> None:
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for parameter_id in actual["state"]:
        assert actual["state"][parameter_id].keys() == expected["state"][parameter_id].keys()
        for state_name, actual_value in actual["state"][parameter_id].items():
            expected_value = expected["state"][parameter_id][state_name]
            if isinstance(actual_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def _make_peft_ep_checkpointer(tmp_path, *, pp_rank: int = 0) -> Checkpointer:
    config = CheckpointingConfig(
        checkpoint_dir=tmp_path,
        model_save_format="safetensors",
        save_consolidated=False,
        is_peft=True,
    )
    return Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=pp_rank, moe_mesh=object())


def _run_native_pp_optimizer_dcp_round_trip(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        source_models, source_optimizers = _make_stepped_adam_parts(count=1)
        source_optimizers[0].param_groups[0]["lr"] = 0.01 * (rank + 1)
        source_parameter = source_optimizers[0].param_groups[0]["params"][0]
        source_optimizers[0].state[source_parameter]["exp_avg"].fill_(rank + 1)
        source_optimizers[0].state[source_parameter]["exp_avg_sq"].fill_(rank + 2)
        checkpointer = _make_peft_ep_checkpointer(checkpoint_dir, pp_rank=rank)
        checkpointer.save_optimizer(
            source_optimizers,
            source_models,
            checkpoint_dir,
            optimizer_part_ids=[rank],
        )

        target_models = [nn.Linear(2, 1, bias=False)]
        target_optimizers = [torch.optim.AdamW(target_models[0].parameters(), lr=0.5)]
        checkpointer.load_optimizer(
            target_optimizers,
            target_models,
            checkpoint_dir,
            optimizer_part_ids=[rank],
        )
        _assert_adam_states_equal(target_optimizers[0], source_optimizers[0])
    finally:
        torch.distributed.destroy_process_group()


def test_checkpointer_uses_native_state_for_all_optimizer_parts_with_ep_topology(tmp_path):
    model_parts, optimizers = _make_stepped_adam_parts()
    checkpointer = _make_peft_ep_checkpointer(tmp_path)

    with patch.object(checkpointer, "_do_save") as do_save:
        checkpointer.save_optimizer(
            optimizers,
            model_parts,
            str(tmp_path),
            optimizer_part_ids=[0, 8],
        )

    state_dict = do_save.call_args.args[0]
    saved_parts = state_dict["optim"]["optimizer_parts"]
    assert saved_parts.keys() == {"stage_0", "stage_8"}
    for part_key, optimizer in zip(("stage_0", "stage_8"), optimizers, strict=True):
        _assert_native_state_dicts_equal(saved_parts[part_key], optimizer.state_dict())


def test_native_single_optimizer_state_preserves_existing_checkpoint_schema():
    model_parts, optimizers = _make_stepped_adam_parts(count=1)
    state_dict = OptimizerState(
        model_parts[0],
        optimizers[0],
        is_peft=True,
        has_expert_parallelism=True,
    ).state_dict()

    assert state_dict["optim"].keys() == {"state", "param_groups"}


def test_flattened_adam_state_materializes_never_used_parameters():
    class PartiallyUsedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(2, 1, bias=False)
            self.unused = nn.Linear(2, 1, bias=False)

    model = PartiallyUsedModel()
    optimizer = torch.optim.AdamW(model.parameters())
    model.used(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    assert model.unused.weight not in optimizer.state

    OptimizerState(model, optimizer).state_dict()

    assert optimizer.state[model.unused.weight].keys() == {"step", "exp_avg", "exp_avg_sq"}
    assert optimizer.state[model.unused.weight]["step"].item() == 0
    assert torch.count_nonzero(optimizer.state[model.unused.weight]["exp_avg"]).item() == 0
    assert torch.count_nonzero(optimizer.state[model.unused.weight]["exp_avg_sq"]).item() == 0


def test_flattened_untagged_optimizer_state_preserves_existing_checkpoint_schema():
    model = nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters())
    expected_state = {
        "state.weight.exp_avg": torch.ones_like(model.weight),
        "param_groups.weight.lr": 0.01,
    }

    with patch(
        "nemo_automodel.components.checkpoint.stateful_wrappers.get_optimizer_state_dict",
        return_value=expected_state,
    ):
        actual_state = OptimizerState(model, optimizer).state_dict()["optim"]

    assert actual_state.keys() == expected_state.keys()
    assert actual_state["state.weight.exp_avg"] is expected_state["state.weight.exp_avg"]
    assert actual_state["param_groups.weight.lr"] == expected_state["param_groups.weight.lr"]


def test_native_multi_optimizer_state_dcp_round_trip(tmp_path):
    source_models, source_optimizers = _make_stepped_adam_parts()
    checkpointer = _make_peft_ep_checkpointer(tmp_path)
    checkpointer.save_optimizer(
        source_optimizers,
        source_models,
        str(tmp_path),
        optimizer_part_ids=[0, 2],
    )

    target_models = [nn.Linear(2, 1, bias=False) for _ in source_models]
    target_optimizers = [torch.optim.AdamW(model.parameters(), lr=0.5) for model in target_models]
    checkpointer.load_optimizer(
        target_optimizers,
        target_models,
        str(tmp_path),
        optimizer_part_ids=[0, 2],
    )

    for target, source in zip(target_optimizers, source_optimizers, strict=True):
        _assert_adam_states_equal(target, source)


def test_native_pipeline_optimizer_load_rejects_stage_mismatch():
    model_parts, optimizers = _make_stepped_adam_parts()
    optimizer_state = OptimizerState(
        model_parts,
        optimizers,
        is_peft=True,
        has_expert_parallelism=True,
        optimizer_part_ids=[0, 8],
    )
    state_dict = optimizer_state.state_dict()
    state_dict["optim"]["optimizer_parts"]["stage_7"] = state_dict["optim"]["optimizer_parts"].pop("stage_8")

    with pytest.raises(ValueError, match=r"checkpoint has \['stage_0', 'stage_7'\].*expects \['stage_0', 'stage_8'\]"):
        optimizer_state.load_state_dict(state_dict)


def test_native_multi_optimizer_state_requires_global_part_ids():
    model_parts, optimizers = _make_stepped_adam_parts()
    optimizer_state = OptimizerState(
        model_parts,
        optimizers,
        is_peft=True,
        has_expert_parallelism=True,
    )

    with pytest.raises(ValueError, match="requires global optimizer part IDs"):
        optimizer_state.state_dict()


def test_native_pipeline_optimizer_state_dcp_round_trip_across_ranks(tmp_path):
    world_size = 2
    torch.multiprocessing.spawn(
        _run_native_pp_optimizer_dcp_round_trip,
        args=(world_size, str(tmp_path / "dist_init"), str(tmp_path / "checkpoint")),
        nprocs=world_size,
        join=True,
    )


def test_model_owned_dtensor_optimizer_state_reshards_world_two_to_four(tmp_path):
    checkpoint_dir = str(tmp_path / "native-dtensor-checkpoint")
    torch.multiprocessing.spawn(
        _run_native_dtensor_optimizer_save,
        args=(2, str(tmp_path / "native-save-init"), checkpoint_dir),
        nprocs=2,
        join=True,
    )
    torch.multiprocessing.spawn(
        _run_native_dtensor_optimizer_load,
        args=(
            4,
            str(tmp_path / "native-load-init"),
            checkpoint_dir,
            str(tmp_path / "native-dtensor-resaved"),
        ),
        nprocs=4,
        join=True,
    )
