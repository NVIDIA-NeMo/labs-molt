# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Focused CPU tests for Qwen3.8-Flash-Next state-dict conversion."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors.torch import save_file
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
    Qwen3_8_FlashNextEngramTableConfig,
    Qwen3_8_FlashNextOwnerShardedEmbedding,
)
from nemo_automodel.components.models.qwen3_8_flash_next.state_dict_adapter import Qwen3_8_FlashNextStateDictAdapter
from nemo_automodel.components.moe.layers import MoEConfig

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)

_TABLE_KEY = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight"
_TABLE_PREFIX = _TABLE_KEY.removesuffix(".weight")


def _make_text_config(*, split_ngram_parts: int = 4) -> Qwen3_8_FlashNextTextConfig:
    """Build the tiny architecture contract used by adapter tests."""
    return Qwen3_8_FlashNextTextConfig(
        vocab_size=32,
        hidden_size=3,
        intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=3,
        moe_intermediate_size=2,
        shared_expert_intermediate_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=1,
        ple_layer_ids=[2],
        ple_embed_dim=3,
        split_ngram_parts=split_ngram_parts,
        dtype="float32",
    )


def _make_moe_config() -> MoEConfig:
    """Build the tiny grouped-MoE layout contract used by adapter tests."""
    return MoEConfig(
        dim=3,
        inter_dim=4,
        moe_inter_dim=2,
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=2,
    )


def _make_backend() -> BackendConfig:
    """Build a kernel-free backend descriptor for checkpoint conversion."""
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


class _TinyOwnerShardedCheckpointModel(HFCheckpointingMixin, nn.Module):
    """Minimal model that preserves the production PLE parameter hierarchy."""

    def __init__(self, process_group: torch.distributed.ProcessGroup) -> None:
        super().__init__()
        self.config = _make_text_config(split_ngram_parts=128)
        self.ordinary = nn.Parameter(torch.empty(3, dtype=torch.float32))

        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([nn.Module(), nn.Module()])
        self.model.language_model.layers[1].ple = nn.Module()
        self.model.language_model.layers[1].ple.ple_embedding = nn.Module()
        self.model.language_model.layers[1].ple.ple_embedding.ngram_embedding = Qwen3_8_FlashNextEngramTableConfig(
            num_embeddings=256,
            embedding_dim=3,
        ).build(
            process_group=process_group,
            dtype=torch.float32,
        )
        self.state_dict_adapter = Qwen3_8_FlashNextStateDictAdapter(
            config=self.config,
            moe_config=_make_moe_config(),
            backend=_make_backend(),
            engram_table=self.engram_table,
            dtype=torch.float32,
        )
        # ``apply_model_infrastructure`` records this global adapter key list
        # before FSDP in production.  Keep the exact same checkpoint contract.
        self._pre_shard_hf_state_dict_keys = self.state_dict_adapter.get_hf_state_dict_keys(self.state_dict())
        self.engram_table.parallelize_weight(
            DeviceMesh.from_group(
                process_group,
                device_type="cpu",
                mesh_dim_names=("dp_shard_cp",),
            )
        )

    @property
    def engram_table(self) -> Qwen3_8_FlashNextOwnerShardedEmbedding:
        """Return this rank's physical owner shard."""
        return self.model.language_model.layers[1].ple.ple_embedding.ngram_embedding


def _run_model_checkpoint_round_trip(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    """Exercise the distributed HF-mixin model save and Checkpointer load path."""
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    checkpointer = None
    try:
        source = _TinyOwnerShardedCheckpointModel(torch.distributed.group.WORLD)
        global_rows = torch.arange(
            source.engram_table.global_row_start,
            source.engram_table.global_row_end,
            dtype=torch.float32,
        ).unsqueeze(1)
        expected_local_table = global_rows * 10 + torch.arange(3, dtype=torch.float32)
        with torch.no_grad():
            source.ordinary.copy_(torch.tensor([701.0, 703.0, 709.0]))
            source.engram_table.weight.to_local().copy_(expected_local_table)
        assert isinstance(source.engram_table.weight, DTensor)
        assert tuple(source.engram_table.weight.shape) == (256, 3)
        assert tuple(source.engram_table.weight.placements) == (Shard(0),)

        global_keys = source._pre_shard_hf_state_dict_keys
        global_table_keys = [key for key in global_keys if key.startswith(f"{_TABLE_PREFIX}.shard_")]
        assert len(global_keys) == 129
        assert global_table_keys == [f"{_TABLE_PREFIX}.shard_{index}.weight" for index in range(128)]
        gathered_global_keys: list[list[str] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered_global_keys, global_keys)
        assert all(keys == global_keys for keys in gathered_global_keys)

        # Each owner contributes exactly its 64 physical files to DCP.  The
        # ordinary parameter remains replicated and is deliberately excluded
        # from this disjointness check.
        local_hf_state = source.state_dict_adapter.to_hf(source.state_dict())
        local_table_keys = {key for key in local_hf_state if key.startswith(f"{_TABLE_PREFIX}.shard_")}
        gathered_local_table_keys: list[set[str] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered_local_table_keys, local_table_keys)
        assert gathered_local_table_keys == [
            {f"{_TABLE_PREFIX}.shard_{index}.weight" for index in range(owner_rank * 64, (owner_rank + 1) * 64)}
            for owner_rank in range(world_size)
        ]
        assert gathered_local_table_keys[0].isdisjoint(gathered_local_table_keys[1])
        assert gathered_local_table_keys[0] | gathered_local_table_keys[1] == set(global_table_keys)

        config = CheckpointingConfig(
            checkpoint_dir=checkpoint_dir,
            model_save_format="safetensors",
            save_consolidated=False,
            is_peft=False,
        )
        checkpointer = config.build(
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
            process_group=torch.distributed.group.WORLD,
        )
        source.save_pretrained(checkpoint_dir, checkpointer=checkpointer)

        model_path = os.path.join(checkpoint_dir, "model")
        checkpoint_keys = _HuggingFaceStorageReader(model_path).read_metadata().state_dict_metadata
        saved_table_keys = {key for key in checkpoint_keys if key.startswith(f"{_TABLE_PREFIX}.shard_")}
        assert saved_table_keys == set(global_table_keys)
        assert "ordinary" in checkpoint_keys
        assert len(checkpoint_keys) == 129

        target = _TinyOwnerShardedCheckpointModel(torch.distributed.group.WORLD)
        with torch.no_grad():
            target.ordinary.fill_(-1)
            target.engram_table.weight.to_local().fill_(-1000 - rank)
        checkpointer.load_model(target, model_path=model_path)

        torch.testing.assert_close(target.ordinary, source.ordinary)
        target_local_weight = target.engram_table.weight.to_local()
        torch.testing.assert_close(target_local_weight, expected_local_table)
        gathered_tables = [torch.empty_like(target_local_weight) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tables, target_local_weight)
        restored_global_table = torch.cat(gathered_tables)
        expected_global_rows = torch.arange(256, dtype=torch.float32).unsqueeze(1)
        expected_global_table = expected_global_rows * 10 + torch.arange(3, dtype=torch.float32)
        torch.testing.assert_close(restored_global_table, expected_global_table)
        assert not torch.equal(gathered_tables[0], gathered_tables[1])
    finally:
        if checkpointer is not None:
            checkpointer.close()
        torch.distributed.destroy_process_group()


def _run_model_checkpoint_resharded_load(
    rank: int,
    world_size: int,
    init_file: str,
    checkpoint_dir: str,
) -> None:
    """Load the physical PLE checkpoint with a different owner world size."""
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    checkpointer = None
    try:
        target = _TinyOwnerShardedCheckpointModel(torch.distributed.group.WORLD)
        with torch.no_grad():
            target.ordinary.fill_(-1)
            target.engram_table.weight.to_local().fill_(-1000 - rank)

        config = CheckpointingConfig(
            checkpoint_dir=checkpoint_dir,
            model_save_format="safetensors",
            save_consolidated=False,
            is_peft=False,
        )
        checkpointer = config.build(
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
            process_group=torch.distributed.group.WORLD,
        )
        checkpointer.load_model(target, model_path=os.path.join(checkpoint_dir, "model"))

        torch.testing.assert_close(target.ordinary, torch.tensor([701.0, 703.0, 709.0]))
        local_weight = target.engram_table.weight.to_local()
        rows = torch.arange(
            target.engram_table.global_row_start,
            target.engram_table.global_row_end,
            dtype=torch.float32,
        ).unsqueeze(1)
        torch.testing.assert_close(local_weight, rows * 10 + torch.arange(3, dtype=torch.float32))

        gathered_tables = [torch.empty_like(local_weight) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tables, local_weight)
        restored_global_table = torch.cat(gathered_tables)
        expected_rows = torch.arange(256, dtype=torch.float32).unsqueeze(1)
        torch.testing.assert_close(restored_global_table, expected_rows * 10 + torch.arange(3, dtype=torch.float32))

        local_hf_state = target.state_dict_adapter.to_hf(target.state_dict())
        local_table_keys = {key for key in local_hf_state if key.startswith(f"{_TABLE_PREFIX}.shard_")}
        assert len(local_table_keys) == 128 // world_size
    finally:
        if checkpointer is not None:
            checkpointer.close()
        torch.distributed.destroy_process_group()


@pytest.fixture
def text_config() -> Qwen3_8_FlashNextTextConfig:
    """Build a tiny architecture config with four physical PLE shards."""
    return Qwen3_8_FlashNextTextConfig(
        vocab_size=32,
        hidden_size=3,
        intermediate_size=4,
        num_hidden_layers=2,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=3,
        moe_intermediate_size=2,
        shared_expert_intermediate_size=2,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=1,
        ple_layer_ids=[2],
        ple_embed_dim=3,
        split_ngram_parts=4,
        dtype="float32",
    )


@pytest.fixture
def moe_config() -> MoEConfig:
    """Build the grouped-MoE shape contract used by conversion tests."""
    return MoEConfig(
        dim=3,
        inter_dim=4,
        moe_inter_dim=2,
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=2,
    )


@pytest.fixture
def backend() -> BackendConfig:
    """Build a backend descriptor; the adapter does not instantiate kernels."""
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


@pytest.fixture
def table() -> Qwen3_8_FlashNextOwnerShardedEmbedding:
    """Build a real rank-1-of-2 table without starting a process group."""
    owner_group = Mock()
    with (
        patch("nemo_automodel.components.models.qwen3_8_flash_next.engram.dist.is_initialized", return_value=True),
        patch("nemo_automodel.components.models.qwen3_8_flash_next.engram.dist.get_world_size", return_value=2),
        patch("nemo_automodel.components.models.qwen3_8_flash_next.engram.dist.get_rank", return_value=1),
    ):
        return Qwen3_8_FlashNextEngramTableConfig(num_embeddings=16, embedding_dim=3).build(
            process_group=owner_group,
            dtype=torch.float32,
        )


@pytest.fixture
def adapter(
    text_config: Qwen3_8_FlashNextTextConfig,
    moe_config: MoEConfig,
    backend: BackendConfig,
    table: Qwen3_8_FlashNextOwnerShardedEmbedding,
) -> Qwen3_8_FlashNextStateDictAdapter:
    """Build an adapter whose rank owns global table rows ``[8, 16)``."""
    return Qwen3_8_FlashNextStateDictAdapter(
        config=text_config,
        moe_config=moe_config,
        backend=backend,
        engram_table=table,
        dtype=torch.float32,
    )


def test_to_hf_emits_only_local_aliasing_physical_shards(
    adapter: Qwen3_8_FlashNextStateDictAdapter,
    table: Qwen3_8_FlashNextOwnerShardedEmbedding,
) -> None:
    """The local ``[8, 3]`` table becomes two ``[4, 3]`` write-through views."""
    with torch.no_grad():
        table.weight.copy_(torch.arange(24, dtype=torch.float32).reshape(8, 3))
    native_table = table.state_dict()["weight"]
    buffer = torch.tensor([11, 13, 17], dtype=torch.long)

    converted = adapter.to_hf(
        {
            _TABLE_KEY: native_table,
            "model.language_model.layers.1.ple.ple_embedding.layer_multipliers": buffer,
        }
    )

    shard_2 = f"{_TABLE_PREFIX}.shard_2.weight"
    shard_3 = f"{_TABLE_PREFIX}.shard_3.weight"
    assert set(converted) == {
        shard_2,
        shard_3,
        "model.language_model.layers.1.ple.ple_embedding.layer_multipliers",
    }
    torch.testing.assert_close(converted[shard_2], native_table[:4])
    torch.testing.assert_close(converted[shard_3], native_table[4:])
    assert converted[shard_2].untyped_storage().data_ptr() == native_table.untyped_storage().data_ptr()
    assert converted[shard_3].untyped_storage().data_ptr() == native_table.untyped_storage().data_ptr()
    assert converted[shard_2].storage_offset() == native_table.storage_offset()
    assert converted[shard_3].storage_offset() == native_table.storage_offset() + 12

    converted[shard_2].fill_(29)
    torch.testing.assert_close(table.weight[:4], torch.full((4, 3), 29.0))


def test_get_hf_state_dict_keys_lists_all_global_shards_without_table_views(
    adapter: Qwen3_8_FlashNextStateDictAdapter,
    table: Qwen3_8_FlashNextOwnerShardedEmbedding,
) -> None:
    """Every rank advertises all four PLE keys without touching local table storage."""
    buffer_key = "model.language_model.layers.1.ple.ple_embedding.layer_multipliers"
    buffer = torch.tensor([11, 13, 17], dtype=torch.long)
    native_table = table.state_dict()["weight"]

    with patch.object(
        torch.Tensor,
        "narrow",
        side_effect=AssertionError("key discovery must not construct rank-local table views"),
    ):
        keys = adapter.get_hf_state_dict_keys(
            {
                _TABLE_KEY: native_table,
                buffer_key: buffer,
                "mtp.fc_hidden.weight": torch.ones(2, 2),
                "model.language_model.layers.0._extra_state": torch.empty(0, dtype=torch.uint8),
            }
        )

    assert keys == [
        f"{_TABLE_PREFIX}.shard_0.weight",
        f"{_TABLE_PREFIX}.shard_1.weight",
        f"{_TABLE_PREFIX}.shard_2.weight",
        f"{_TABLE_PREFIX}.shard_3.weight",
        buffer_key,
    ]


def test_from_hf_drops_table_aliases_and_marks_native_weight_loaded(
    adapter: Qwen3_8_FlashNextStateDictAdapter,
    table: Qwen3_8_FlashNextOwnerShardedEmbedding,
) -> None:
    """DCP-style writes remain in the local ``[8, 3]`` weight after aliases are dropped."""
    buffer_key = "model.language_model.layers.1.ple.ple_embedding.ngram_heads_offsets"
    buffer = torch.tensor([0, 4], dtype=torch.long)
    destinations = adapter.to_hf({_TABLE_KEY: table.state_dict()["weight"], buffer_key: buffer})
    destinations[f"{_TABLE_PREFIX}.shard_2.weight"].fill_(2)
    destinations[f"{_TABLE_PREFIX}.shard_3.weight"].fill_(3)
    destinations["mtp.fc_hidden.weight"] = torch.ones(3, 3)

    native = adapter.from_hf(destinations)

    assert native == {buffer_key: buffer}
    assert adapter.view_loaded_native_keys == {_TABLE_KEY}
    torch.testing.assert_close(table.weight[:4], torch.full((4, 3), 2.0))
    torch.testing.assert_close(table.weight[4:], torch.full((4, 3), 3.0))


def test_adapter_rejects_owner_ranges_that_cut_a_checkpoint_shard(
    text_config: Qwen3_8_FlashNextTextConfig,
    moe_config: MoEConfig,
    backend: BackendConfig,
    table: Qwen3_8_FlashNextOwnerShardedEmbedding,
) -> None:
    """A range such as ``[6, 14)`` cannot expose full ``[4, 3]`` DCP targets."""
    table.global_row_start = 6
    table.global_row_end = 14

    with pytest.raises(ValueError, match="align to complete physical checkpoint shards"):
        Qwen3_8_FlashNextStateDictAdapter(
            config=text_config,
            moe_config=moe_config,
            backend=backend,
            engram_table=table,
        )


def test_grouped_experts_and_fp32_gdn_use_inherited_qwen35_conversion(
    adapter: Qwen3_8_FlashNextStateDictAdapter,
) -> None:
    """Expert ``[E, in, out]`` transposes and fp32-holder routing round-trip."""
    gate_up = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    down = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    a_log = torch.tensor([1.0, 2.0], dtype=torch.float32)
    mtp = torch.ones(2, 2)
    native_state = {
        "model.language_model.layers.0.mlp.experts.gate_and_up_projs": gate_up,
        "model.language_model.layers.0.mlp.experts.down_projs": down,
        "model.language_model.layers.0.linear_attn._fp32_params.A_log": a_log,
        "mtp.fc_hidden.weight": mtp,
    }

    hf_state = adapter.to_hf(native_state)

    gate_hf_key = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    down_hf_key = "model.language_model.layers.0.mlp.experts.down_proj"
    a_log_hf_key = "model.language_model.layers.0.linear_attn.A_log"
    assert "mtp.fc_hidden.weight" not in hf_state
    torch.testing.assert_close(hf_state[gate_hf_key], gate_up.transpose(1, 2))
    torch.testing.assert_close(hf_state[down_hf_key], down.transpose(1, 2))
    assert hf_state[gate_hf_key].untyped_storage().data_ptr() == gate_up.untyped_storage().data_ptr()
    assert hf_state[down_hf_key].untyped_storage().data_ptr() == down.untyped_storage().data_ptr()
    assert hf_state[a_log_hf_key].dtype == torch.float32
    assert hf_state[a_log_hf_key] is a_log

    # A lower-precision checkpoint tensor is upcast while it is routed back to
    # the model's intrinsic fp32 holder.
    hf_state[a_log_hf_key] = hf_state[a_log_hf_key].to(torch.bfloat16)
    restored = adapter.from_hf(hf_state)

    torch.testing.assert_close(restored["model.language_model.layers.0.mlp.experts.gate_and_up_projs"], gate_up)
    torch.testing.assert_close(restored["model.language_model.layers.0.mlp.experts.down_projs"], down)
    restored_a_log = restored["model.language_model.layers.0.linear_attn._fp32_params.A_log"]
    assert restored_a_log.dtype == torch.float32
    torch.testing.assert_close(restored_a_log, a_log)


def test_adapter_advertises_nonquantized_low_memory_loading(
    adapter: Qwen3_8_FlashNextStateDictAdapter,
) -> None:
    """The loader may bypass host staging for valid BF16/fp32 model state."""
    assert adapter.supports_low_memory_dcp_load


def test_table_export_honors_exclude_regex(
    adapter: Qwen3_8_FlashNextStateDictAdapter,
    table: Qwen3_8_FlashNextOwnerShardedEmbedding,
) -> None:
    """Filtering one ``[4, 3]`` physical shard does not create a replacement copy."""
    converted = adapter.to_hf(
        {_TABLE_KEY: table.state_dict()["weight"]},
        exclude_key_regex=r".*shard_2\.weight$",
    )

    assert set(converted) == {f"{_TABLE_PREFIX}.shard_3.weight"}


def test_hf_storage_reader_writes_only_owned_shards_through_views(
    tmp_path: Path,
    adapter: Qwen3_8_FlashNextStateDictAdapter,
    table: Qwen3_8_FlashNextOwnerShardedEmbedding,
) -> None:
    """A real DCP read fills rank 1's two ``[4, 3]`` views and skips shards 0/1."""
    checkpoint = {
        f"{_TABLE_PREFIX}.shard_{shard_idx}.weight": torch.full((4, 3), float(shard_idx)) for shard_idx in range(4)
    }
    save_file(checkpoint, tmp_path / "model.safetensors")
    destinations = adapter.to_hf({_TABLE_KEY: table.state_dict()["weight"]})

    dcp.load(destinations, storage_reader=_HuggingFaceStorageReader(str(tmp_path)))
    native = adapter.from_hf(destinations)

    assert native == {}
    assert adapter.view_loaded_native_keys == {_TABLE_KEY}
    torch.testing.assert_close(table.weight[:4], torch.full((4, 3), 2.0))
    torch.testing.assert_close(table.weight[4:], torch.full((4, 3), 3.0))


def test_model_safetensors_checkpoint_round_trip_preserves_owner_shards_across_ranks(tmp_path: Path) -> None:
    """Two owners save 128 PLE shards, then two and four owners restore them."""
    world_size = 2
    checkpoint_dir = str(tmp_path / "checkpoint")
    torch.multiprocessing.spawn(
        _run_model_checkpoint_round_trip,
        args=(world_size, str(tmp_path / "dist_init"), checkpoint_dir),
        nprocs=world_size,
        join=True,
    )
    resharded_world_size = 4
    torch.multiprocessing.spawn(
        _run_model_checkpoint_resharded_load,
        args=(resharded_world_size, str(tmp_path / "resharded_dist_init"), checkpoint_dir),
        nprocs=resharded_world_size,
        join=True,
    )
