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

"""Numerical checkpoint mapping and row-owner tests for DeepSeek-V4.1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from transformers import PreTrainedModel

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config, DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import (
    DeepseekV41StateDictAdapter,
    dequantize_checkpoint_weight,
)
from nemo_automodel.components.moe.config import MoEConfig

# Over the default 5s budget on purpose: distributed checkpoint round trips spawn Gloo workers.
# Shrink process startup and checkpoint fixtures before lowering this further.
pytestmark = pytest.mark.timeout(60)


def _checkpoint_model(tensors: dict[str, torch.Tensor]) -> torch.nn.Module:
    model = torch.nn.Module()
    for fqn, tensor in tensors.items():
        owner = model
        parts = fqn.split(".")
        for part in parts[:-1]:
            if part not in owner._modules:
                owner.add_module(part, torch.nn.Module())
            owner = owner._modules[part]
        owner.register_buffer(parts[-1], tensor)
    return model


class _CheckpointOnlyV41(DeepseekV41ForCausalLM):
    """Keep the real custom-model checkpoint contract with tiny tensor fixtures."""

    def __init__(self, config: DeepseekV41Config, tensors: dict[str, torch.Tensor]) -> None:
        PreTrainedModel.__init__(self, config)
        for name, module in _checkpoint_model(tensors).named_children():
            self.add_module(name, module)
        for module in self.modules():
            for name, value in list(module._buffers.items()):
                del module._buffers[name]
                module.register_parameter(name, torch.nn.Parameter(value))


def _adapter(
    engram_rows: int | None = None,
    *,
    second_engram_rows: int | None = None,
    dtype: torch.dtype = torch.float32,
    experts: str = "torch_mm",
    dim: int = 32,
) -> DeepseekV41StateDictAdapter:
    engram_layers = [] if engram_rows is None else [1]
    table_rows = [] if engram_rows is None else [engram_rows]
    if second_engram_rows is not None:
        engram_layers.append(14)
        table_rows.append(second_engram_rows)
    text = DeepseekV41TextConfig(
        vocab_size=64,
        hidden_size=dim,
        moe_intermediate_size=dim,
        num_hidden_layers=15 if second_engram_rows is not None else 2,
        n_routed_experts=2,
        num_experts_per_tok=1,
        engram_layer_ids=engram_layers,
        engram_num_embeddings=table_rows,
        dtype=str(dtype).removeprefix("torch."),
    )
    moe = MoEConfig(
        dim=dim,
        inter_dim=dim,
        moe_inter_dim=dim,
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0,
        aux_loss_coeff=0,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        norm_topk_prob=True,
        dtype=dtype,
    )
    return DeepseekV41StateDictAdapter(
        DeepseekV41Config(text_config=text),
        moe,
        BackendConfig(linear="torch", attn="sdpa", dispatcher="torch", experts=experts),
        dtype=dtype,
    )


def test_released_projection_names_and_grouped_experts_roundtrip() -> None:
    adapter = _adapter()
    source = {
        "embed.weight": torch.randn(64, 32),
        "norm.weight": torch.randn(32),
        "head.weight": torch.randn(64, 32),
        "layers.0.attn_norm.weight": torch.randn(32),
        "layers.0.ffn_norm.weight": torch.randn(32),
        "layers.0.attn.attn_sink": torch.randn(4),
        "layers.0.attn.compressor.norm.weight": torch.randn(32),
        "layers.0.attn.indexer.wk.weight": torch.randn(8, 32),
        "layers.0.hc_attn_fn": torch.randn(24, 128),
        "layers.0.hc_attn_base": torch.randn(24),
        "layers.0.hc_ffn_scale": torch.randn(3),
        "layers.0.ffn.gate.weight": torch.randn(2, 32),
        "layers.0.ffn.gate.bias": torch.randn(2),
        "layers.0.ffn.gate.bias_vl": torch.randn(2),
        "layers.0.ffn.shared_experts.w1.weight": torch.randn(32, 32),
        "layers.0.ffn.shared_experts.w2.weight": torch.randn(32, 32),
        "layers.0.ffn.shared_experts.w3.weight": torch.randn(32, 32),
        "vision.patch_embed.proj.weight": torch.randn(4, 12),
        "aligner.w1.weight": torch.randn(32, 36),
        "image_start": torch.randn(32),
    }
    for expert in range(2):
        for projection in (1, 2, 3):
            source[f"layers.0.ffn.experts.{expert}.w{projection}.weight"] = (
                torch.arange(1024).reshape(32, 32).float() + 10000 * expert + 2000 * projection
            )
    expected = dict(source)
    native = adapter.from_hf(source)
    assert source == {}
    assert "model.layers.0.attn_hc.fn" in native
    assert "model.layers.0.ffn_hc.scale" in native
    assert "model.layers.0.ffn.gate.e_score_correction_bias" in native
    assert "model.layers.0.attn.sinks_param.weight" in native
    assert "model.layers.0.attn.compressor.norm.weight" in native
    assert "model.layers.0.ffn.shared_experts.gate_proj.weight" in native
    assert "model.hc_head" not in native
    for expert in range(2):
        grouped = native["model.layers.0.ffn.experts.gate_and_up_projs"][expert]
        torch.testing.assert_close(grouped[:, :32], expected[f"layers.0.ffn.experts.{expert}.w1.weight"].T)
        torch.testing.assert_close(grouped[:, 32:], expected[f"layers.0.ffn.experts.{expert}.w3.weight"].T)
        torch.testing.assert_close(
            native["model.layers.0.ffn.experts.down_projs"][expert],
            expected[f"layers.0.ffn.experts.{expert}.w2.weight"].T,
        )
    exported = adapter.to_hf(native)
    assert exported.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(exported[key], expected[key], rtol=0, atol=0)


def test_dense_fp8_partial_blocks_use_32_by_32_scales() -> None:
    weight = torch.ones(33, 35).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1, 2], [4, 8]], dtype=torch.float32).to(torch.float8_e8m0fnu)
    decoded = dequantize_checkpoint_weight(weight, scales, dtype=torch.float32)
    assert decoded.shape == (33, 35)
    assert (decoded[:32, :32] == 1).all()
    assert (decoded[:32, 32:] == 2).all()
    assert (decoded[32:, :32] == 4).all()
    assert (decoded[32:, 32:] == 8).all()


def test_fp4_decoding_preserves_nibble_order_sign_and_row_scales() -> None:
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8).view(torch.int8)
    scale = torch.tensor([[2.0]]).to(torch.float8_e8m0fnu)
    decoded = dequantize_checkpoint_weight(packed, scale, dtype=torch.float32)
    expected = torch.tensor([[0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12] * 2]).float()
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)


def test_engram_fp8_uses_per_row_scales() -> None:
    weight = torch.ones(2, 64).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1, 2], [4, 8]], dtype=torch.float32).to(torch.float8_e8m0fnu)
    decoded = dequantize_checkpoint_weight(weight, scales, dtype=torch.float32, rowwise=True)
    assert (decoded[0, :32] == 1).all()
    assert (decoded[0, 32:] == 2).all()
    assert (decoded[1, :32] == 4).all()
    assert (decoded[1, 32:] == 8).all()
    with pytest.raises(ValueError, match="Scale shape"):
        dequantize_checkpoint_weight(weight, scales)


def test_missing_scales_incomplete_experts_and_orphan_scales_fail() -> None:
    with pytest.raises(ValueError, match="missing its scale"):
        _adapter().from_hf({"layers.0.attn.wq_a.weight": torch.ones(32, 32).to(torch.float8_e4m3fn)})
    with pytest.raises(ValueError, match="no matching weight"):
        _adapter().from_hf({"layers.0.attn.wq_a.scale": torch.ones(1, 1)})
    with pytest.raises(RuntimeError, match="Expert weights missing"):
        _adapter().from_hf({"layers.0.ffn.experts.0.w1.weight": torch.ones(32, 32)})


def test_backbone_prefix_omits_draft_and_unused_layer_weights() -> None:
    source = {
        "mtp.0.main_proj.weight": torch.empty(32, 32, dtype=torch.int8),
        "layers.2.attn.wq_a.weight": torch.empty(32, 32, dtype=torch.int8),
        "norm.weight": torch.ones(32),
    }
    native = _adapter().from_hf(source)
    assert set(native) == {"model.norm.weight"}


def test_global_key_audit_accepts_owner_local_meta_tables() -> None:
    state = {
        "model.layers.1.engram.embed.weight": torch.empty(9, 64, device="meta"),
        "model.layers.0.ffn.experts.gate_and_up_projs": torch.empty(1, 32, 64, device="meta"),
        "model.layers.0.ffn.experts.down_projs": torch.empty(1, 32, 32, device="meta"),
    }
    keys = _adapter(engram_rows=17).get_hf_state_dict_keys(state)
    assert "layers.1.engram.embed.weight" in keys
    assert len(keys) == 7
    for index in range(2):
        for projection in (1, 2, 3):
            assert f"layers.0.ffn.experts.{index}.w{projection}.weight" in keys


def test_quantized_load_destinations_match_dump_and_do_not_alias_model() -> None:
    adapter = _adapter()
    native = {
        "model.layers.0.attn.wq_a.weight": torch.ones(33, 64),
        "model.layers.0.ffn.experts.gate_and_up_projs": torch.ones(2, 32, 64),
        "model.layers.0.ffn.experts.down_projs": torch.ones(2, 32, 32),
        "model.layers.0.attn.compressor.wkv.weight": torch.ones(32, 32),
    }
    destinations = adapter.to_hf(native, quantization=True, for_checkpoint_load=True)
    assert destinations["layers.0.attn.wq_a.weight"].shape == (33, 64)
    assert destinations["layers.0.attn.wq_a.weight"].dtype == torch.float8_e4m3fn
    assert destinations["layers.0.attn.wq_a.scale"].shape == (2, 2)
    assert destinations["layers.0.attn.wq_a.scale"].dtype == torch.float8_e8m0fnu
    assert destinations["layers.0.ffn.experts.1.w1.weight"].shape == (32, 16)
    assert destinations["layers.0.ffn.experts.1.w1.weight"].dtype == torch.int8
    assert destinations["layers.0.ffn.experts.1.w1.scale"].shape == (32, 1)
    assert destinations["layers.0.attn.compressor.wkv.weight"] is native["model.layers.0.attn.compressor.wkv.weight"]
    assert "layers.0.attn.compressor.wkv.scale" not in destinations
    assert not adapter.supports_low_memory_dcp_load
    with pytest.raises(ValueError, match="checkpoint loading only"):
        adapter.to_hf(native, quantization=True)
    for key, target in destinations.items():
        if key.endswith(".scale"):
            target.copy_(torch.ones(target.shape))
        elif target.dtype == torch.int8:
            target.fill_(0x44)
        elif target.dtype == torch.float8_e4m3fn:
            target.copy_(torch.full(target.shape, 2.0))
    loaded = adapter.from_hf(destinations)
    assert (loaded["model.layers.0.attn.wq_a.weight"] == 2).all()
    assert (loaded["model.layers.0.ffn.experts.gate_and_up_projs"] == 2).all()
    assert (native["model.layers.0.attn.wq_a.weight"] == 1).all()


def _owner_checkpoint_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("owner",))
        adapter = _adapter(engram_rows=17)
        local = torch.arange(rank * 9 * 64, (rank + 1) * 9 * 64).reshape(9, 64).float()
        native = DTensor.from_local(local, mesh, (Shard(0),), shape=torch.Size((18, 64)), stride=(64, 1))
        key = "model.layers.1.engram.embed.weight"
        exported = dict(adapter.convert_single_tensor_to_hf(key, native))
        table = exported["layers.1.engram.embed.weight"]
        assert table.shape == (17, 64)
        assert table.to_local().shape == (9 if rank == 0 else 8, 64)
        torch.testing.assert_close(table.to_local(), local[: table.to_local().shape[0]])
        destinations = dict(
            adapter.convert_single_tensor_to_hf(key, native, quantization=True, for_checkpoint_load=True)
        )
        assert destinations["layers.1.engram.embed.scale"].shape == (17, 2)
        for name, target in destinations.items():
            expected_rows = 9 if rank == 0 else 8
            assert target.to_local().shape[0] == expected_rows
            target.to_local().copy_(torch.full(target.to_local().shape, 2.0 if name.endswith(".scale") else 1.0))
        restored = adapter.from_hf(destinations)[key]
        assert restored.shape == (18, 64)
        assert restored.to_local().shape == (9, 64)
        valid = 9 if rank == 0 else 8
        assert (restored.to_local()[:valid] == 2).all()
        if rank == 1:
            assert (restored.to_local()[8] == 0).all()
        # Dense FSDP rows can split a 32-row scale block. Global small scales
        # must be sliced by the true offset instead of restarting on every rank.
        fp8 = DTensor.from_local(
            torch.ones(33 if rank == 0 else 32, 64).to(torch.float8_e4m3fn),
            mesh,
            (Shard(0),),
            shape=torch.Size((65, 64)),
            stride=(64, 1),
        )
        scales = torch.tensor([[1, 2], [4, 8], [16, 32]], dtype=torch.float32).to(torch.float8_e8m0fnu)
        decoded = dequantize_checkpoint_weight(fp8, scales, dtype=torch.float32).to_local()
        if rank == 0:
            assert decoded[0, 0] == 1 and decoded[-1, 0] == 4
        else:
            assert decoded[0, 0] == 4 and decoded[-1, -1] == 32
        checkpoint = Path(rendezvous).parent
        native.to_local()[:valid].fill_(4 if rank == 0 else 8)
        if rank == 1:
            native.to_local()[-1].zero_()
        dense = DTensor.from_local(decoded.clone(), mesh, (Shard(0),), shape=torch.Size((65, 64)), stride=(64, 1))
        model = _checkpoint_model({key: native, "model.layers.0.attn.wq_a.weight": dense})
        expected = {name: value.to_local().clone() for name, value in model.state_dict().items()}
        dcp.save(adapter.to_hf(model.state_dict()), checkpoint_id=checkpoint / "dcp")
        for value in model.state_dict().values():
            value.to_local().zero_()
        destinations = adapter.to_hf(model.state_dict(), for_checkpoint_load=True)
        dcp.load(destinations, checkpoint_id=checkpoint / "dcp")
        restored = adapter.from_hf(destinations)
        for name, value in restored.items():
            torch.testing.assert_close(value.to_local(), expected[name], rtol=0, atol=0)
        model.config = adapter.config
        model.state_dict_adapter = adapter
        checkpointer = Checkpointer(
            CheckpointingConfig(
                checkpoint_dir=str(checkpoint),
                model_save_format="torch_save",
                save_consolidated=False,
                model_cache_dir=str(checkpoint / "cache"),
                model_repo_id="test/deepseek-v41",
                dequantize_base_checkpoint=True,
            ),
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
            process_group=dist.group.WORLD,
        )
        try:
            checkpointer.save_model(model, str(checkpoint / "framework"))
            for value in model.state_dict().values():
                value.to_local().zero_()
            checkpointer.load_model(model, str(checkpoint / "framework" / "model"))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value.to_local(), expected[name], rtol=0, atol=0)
        finally:
            checkpointer.close()
    finally:
        dist.destroy_process_group()


def test_uneven_engram_owner_rows_and_misaligned_dense_shards(tmp_path: Path) -> None:
    mp.spawn(_owner_checkpoint_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def _quantized_checkpointer_worker(rank: int, rendezvous: str, expert_shard_size: int = 1) -> None:
    """Read a real quantized HF dump through DCP, then round-trip SafeTensors."""
    world_size = 2 * expert_shard_size
    dim = 32 * expert_shard_size
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        expert_mesh = (
            init_device_mesh("cpu", (expert_shard_size, 2), mesh_dim_names=("ep_shard", "ep"))
            if expert_shard_size > 1
            else init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        )
        adapter = _adapter(17, second_engram_rows=19, dtype=torch.bfloat16, experts="torch_mm", dim=dim)
        # Match the released checkpoint metadata used to select quantized DCP load targets.
        adapter.config.quantization_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }
        source = {}
        expected = {}
        for layer, rows in ((1, 17), (14, 19)):
            raw = (torch.arange(rows * 64).reshape(rows, 64) % 7 - 3).to(torch.float8_e4m3fn)
            scales = (2.0 ** (torch.arange(rows * 2).reshape(rows, 2) % 5 - 2)).to(torch.float8_e8m0fnu)
            source[f"layers.{layer}.engram.embed.weight"] = raw
            source[f"layers.{layer}.engram.embed.scale"] = scales
            logical = (raw.float() * scales.float().repeat_interleave(32, dim=1)).bfloat16()
            padded_rows = (rows + world_size - 1) // world_size * world_size
            expected[f"model.layers.{layer}.engram.embed.weight"] = torch.nn.functional.pad(
                logical, (0, 0, 0, padded_rows - rows)
            )

        raw = (torch.arange(65 * 64).reshape(65, 64) % 5 - 2).to(torch.float8_e4m3fn)
        scales = torch.tensor([[0.5, 1.0], [2.0, 4.0], [8.0, 16.0]]).to(torch.float8_e8m0fnu)
        source["layers.0.attn.wq_a.weight"] = raw
        source["layers.0.attn.wq_a.scale"] = scales
        expected["model.layers.0.attn.wq_a.weight"] = (
            raw.float() * scales.float().repeat_interleave(32, dim=0)[:65].repeat_interleave(32, dim=1)
        ).bfloat16()

        # The fixture constructs all E2M1 code values explicitly, with distinct
        # low/high nibbles and per-row powers of two. Expected expert matrices
        # are independent of the adapter's decoder and use released layouts.
        fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
        projections = {}
        for expert in range(2):
            for projection in (1, 2, 3):
                row, column = torch.meshgrid(torch.arange(dim), torch.arange(dim // 2), indexing="ij")
                low = (row + column + expert * 3 + projection * 5) % 16
                high = (row * 3 + column * 5 + expert + projection) % 16
                raw = (low | (high << 4)).to(torch.uint8).view(torch.int8)
                scales = (2.0 ** ((torch.arange(dim)[:, None] + 2 * torch.arange(dim // 32)[None, :]) % 3 - 1)).to(
                    torch.float8_e8m0fnu
                )
                key = f"layers.0.ffn.experts.{expert}.w{projection}"
                source[f"{key}.weight"] = raw
                source[f"{key}.scale"] = scales
                decoded = torch.stack((fp4_values[low], fp4_values[high]), dim=-1).flatten(1)
                projections[expert, projection] = (decoded * scales.float().repeat_interleave(32, dim=1)).bfloat16()
        expected["model.layers.0.ffn.experts.gate_and_up_projs"] = torch.stack(
            [torch.cat((projections[expert, 1].T, projections[expert, 3].T), dim=-1) for expert in range(2)]
        )
        expected["model.layers.0.ffn.experts.down_projs"] = torch.stack(
            [projections[expert, 2].T for expert in range(2)]
        )
        for released, native, value in (
            (
                "layers.0.attn.attn_sink",
                "model.layers.0.attn.sinks_param.weight",
                torch.tensor([1.00123, -0.33337, 2.00456, -3.00091]),
            ),
            (
                "layers.0.hc_attn_fn",
                "model.layers.0.attn_hc.fn",
                torch.arange(8 * 128).reshape(8, 128).float() / 10000 + 0.00123,
            ),
            ("layers.0.hc_ffn_scale", "model.layers.0.ffn_hc.scale", torch.tensor([1.00123, 0.33337, 0.00456])),
        ):
            source[released] = value
            expected[native] = value

        root = Path(rendezvous).parent
        checkpoint = root / "quantized_hf"
        if rank == 0:
            checkpoint.mkdir()
            save_file(source, checkpoint / "model.safetensors")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(source, "model.safetensors")})
            )
            adapter.config.save_pretrained(checkpoint)
        dist.barrier()
        local_expected = {}
        tensors = {}
        for key, value in expected.items():
            expert = ".ffn.experts." in key
            if expert:
                local = value[rank % 2 : rank % 2 + 1]
                if expert_shard_size > 1:
                    local = local[:, (rank // 2) * 32 : (rank // 2 + 1) * 32]
                placements = (Shard(1), Shard(0)) if expert_shard_size > 1 else (Shard(0),)
            else:
                local_rows = (value.shape[0] + world_size - 1) // world_size
                local = value[rank * local_rows : (rank + 1) * local_rows]
                placements = (Shard(0),)
            local_expected[key] = local
            tensors[key] = DTensor.from_local(
                torch.full_like(local, -101),
                expert_mesh if expert else mesh,
                placements,
                shape=value.shape,
                stride=value.stride(),
            )
        model = _CheckpointOnlyV41(adapter.config, tensors)
        model.state_dict_adapter = adapter
        checkpointer = Checkpointer(
            CheckpointingConfig(
                checkpoint_dir=str(root),
                model_save_format="safetensors",
                save_consolidated=False,
                model_cache_dir=str(root / "cache"),
                model_repo_id="test/deepseek-v41",
                dequantize_base_checkpoint=True,
            ),
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
            process_group=dist.group.WORLD,
            moe_mesh=expert_mesh,
        )
        try:
            checkpointer.load_model(model, str(checkpoint), is_init_step=True)
            for key, tensor in model.state_dict().items():
                reference = local_expected[key]
                assert tensor.dtype == reference.dtype and tensor.to_local().dtype == reference.dtype
                torch.testing.assert_close(tensor.to_local(), reference, rtol=0, atol=0)
            checkpointer.save_model(model, str(root / "trained_safetensors"))
            for tensor in model.state_dict().values():
                tensor.to_local().fill_(-99)
            checkpointer.load_model(model, str(root / "trained_safetensors" / "model"))
            for key, tensor in model.state_dict().items():
                reference = local_expected[key]
                assert tensor.dtype == reference.dtype and tensor.to_local().dtype == reference.dtype
                torch.testing.assert_close(tensor.to_local(), reference, rtol=0, atol=0)
        finally:
            checkpointer.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("expert_shard_size", [1, 2], ids=["ep2", "ep2_fsdp2"])
def test_real_quantized_hf_initialization_and_safetensors_resume(tmp_path: Path, expert_shard_size: int) -> None:
    mp.spawn(
        _quantized_checkpointer_worker,
        args=(str(tmp_path / "quantized_rendezvous"), expert_shard_size),
        nprocs=2 * expert_shard_size,
        join=True,
    )


@pytest.mark.parametrize("scale_byte", [0, 255], ids=["smallest_exponent", "nan"])
def test_fp8_e8m0_special_values(scale_byte: int) -> None:
    """Released 32x32 scales retain E8M0's subnormal power and NaN encoding."""
    weight = torch.ones((32, 32)).to(torch.float8_e4m3fn)
    scale = torch.full((1, 1), scale_byte, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    actual = dequantize_checkpoint_weight(weight, scale, dtype=torch.float32)
    if scale_byte == 255:
        assert torch.isnan(actual).all()
    else:
        assert torch.equal(actual, torch.full((32, 32), 2.0**-127, dtype=torch.float32))


def test_generic_checkpoint_keys_and_optional_tower_weights_are_preserved() -> None:
    """Generic key conversion retains checkpoint tensors beyond the built towers."""
    adapter = _adapter()
    adapter.config.vision_config.num_hidden_layers = 0
    source = {
        "layers.0.ffn.custom.weight": torch.tensor([1.00123]),
        "layers.1.engram.q_weight": torch.tensor([2.00345]),
        "vision.norm.weight": torch.tensor([3.00456]),
        "aligner.w1.weight": torch.ones(2, 2, dtype=torch.bfloat16),
        "image_start": torch.ones(2, dtype=torch.bfloat16),
        "image_pad": torch.ones(2, dtype=torch.bfloat16),
        "layers.0.ffn.gate.bias_vl": torch.tensor([0.1234567]),
    }
    native = adapter.from_hf(dict(source))
    assert "model.layers.0.ffn.custom.weight" in native
    assert set(adapter.get_hf_state_dict_keys(native)) == source.keys()
    exported = adapter.to_hf(native)
    assert exported.keys() == source.keys()
    for key, value in source.items():
        torch.testing.assert_close(exported[key], value, rtol=0, atol=0)
    assert adapter.forced_hf_dtype_mapping(native) == {
        key: "float32" for key, value in source.items() if value.dtype == torch.float32
    }
