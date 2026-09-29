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

"""Real NeMo initialization derives DCP quantization from checkpoint metadata.

Two Gloo ranks force the distributed SafeTensors reader, avoiding the
single-process custom-model eager fallback. Only the CUDA device query is
adapted to CPU; model construction, metadata inference and checkpoint loading
run unchanged. These tests do not claim EP/FSDP execution or GPU parity.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config, DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM

# Over the default 5s budget on purpose: distributed checkpoint initialization spawns Gloo workers.
# Shrink process startup and checkpoint round trips before lowering this further.
pytestmark = pytest.mark.timeout(60)


def _config(quantized: bool) -> DeepseekV41Config:
    """Build a complete one-layer backbone with a partial FP8 row block."""
    config = DeepseekV41Config(
        vision_config={"num_hidden_layers": 0},
        text_config=DeepseekV41TextConfig(
            vocab_size=64,
            hidden_size=64,
            moe_intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            head_dim=32,
            qk_rope_head_dim=16,
            q_lora_rank=65,
            o_lora_rank=32,
            o_groups=1,
            n_routed_experts=2,
            num_experts_per_tok=1,
            compress_ratios=[0],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            index_head_dim=32,
            candidate_source_layer_id=-1,
            engram_layer_ids=[],
            dspark_noise_token_id=0,
            dtype="bfloat16",
        ),
    )
    if quantized:
        config.quantization_config = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [32, 32],
            "scale_fmt": "ue8m0",
            "expert_dtype": "fp4",
        }
    return config


def _backend() -> BackendConfig:
    """Select CPU-capable construction without disabling the checkpoint adapter."""
    return BackendConfig(attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch_mm", dispatcher="torch")


def _fixture(quantized: bool) -> tuple[DeepseekV41Config, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Create released storage and independent expected native values.

    Args:
        quantized: Emit FP8/FP4 storage with E8M0 scales, or floating weights.

    Returns:
        Configuration, released tensors, and native expected tensors. Dense
        matrices use [output, input]; grouped experts use [experts, hidden,
        2 * intermediate] and [experts, intermediate, hidden]. All values are
        CPU tensors. Expected values never use the adapter's mapping or decoder.
    """
    config = _config(quantized)
    with torch.device("meta"):
        schema = DeepseekV41ForCausalLM(config, backend=_backend()).state_dict()
    names = {
        "model.embed_tokens.weight": "embed.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "head.weight",
        "model.layers.0.attn.sinks_param.weight": "layers.0.attn.attn_sink",
        "model.layers.0.ffn.gate.e_score_correction_bias": "layers.0.ffn.gate.bias",
    }
    for suffix in (
        "attn.wq_a.weight",
        "attn.q_norm.weight",
        "attn.wq_b.weight",
        "attn.wkv.weight",
        "attn.kv_norm.weight",
        "attn.wo_a.weight",
        "attn.wo_b.weight",
        "attn_norm.weight",
        "ffn_norm.weight",
        "ffn.gate.weight",
        "ffn.gate.bias_vl",
    ):
        names[f"model.layers.0.{suffix}"] = f"layers.0.{suffix}"
    for sublayer in ("attn", "ffn"):
        for parameter in ("fn", "base", "scale"):
            names[f"model.layers.0.{sublayer}_hc.{parameter}"] = f"layers.0.hc_{sublayer}_{parameter}"
    for native, released in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
        names[f"model.layers.0.ffn.shared_experts.{native}.weight"] = f"layers.0.ffn.shared_experts.{released}.weight"

    source, expected = {}, {}
    dense_quantized = {f"layers.0.attn.{name}.weight" for name in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b")} | {
        f"layers.0.ffn.shared_experts.w{projection}.weight" for projection in (1, 2, 3)
    }
    for native, released in names.items():
        shape = schema[native].shape
        strict = native == "lm_head.weight" or any(
            part in native for part in ("_hc.", "sinks_param", "e_score_correction_bias", "bias_vl")
        )
        dtype = torch.float32 if strict else torch.bfloat16
        if released in dense_quantized:
            rows, columns = shape
            raw = (torch.arange(rows * columns).reshape(shape) % 7 - 3).to(torch.float8_e4m3fn)
            scale_shape = ((rows + 31) // 32, (columns + 31) // 32)
            scales = (2.0 ** (torch.arange(scale_shape[0] * scale_shape[1]).reshape(scale_shape) % 5 - 2)).to(
                torch.float8_e8m0fnu
            )
            expanded = scales.float().repeat_interleave(32, 0).repeat_interleave(32, 1)[:rows, :columns]
            value = (raw.float() * expanded).to(dtype)
            source[released] = raw if quantized else value
            if quantized:
                source[released.removesuffix(".weight") + ".scale"] = scales
        else:
            value = (torch.arange(schema[native].numel()).reshape(shape).float() / 10000 + 0.00123).to(dtype)
            source[released] = value
        expected[native] = value

    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
    projections = {}
    for expert in range(2):
        for projection, rows, columns in ((1, 32, 64), (3, 32, 64), (2, 64, 32)):
            row, column = torch.meshgrid(torch.arange(rows), torch.arange(columns // 2), indexing="ij")
            low = (row + column + expert * 3 + projection * 5) % 16
            high = (row * 3 + column * 5 + expert + projection) % 16
            raw = (low | (high << 4)).to(torch.uint8).view(torch.int8)
            scales = (2.0 ** ((torch.arange(rows)[:, None] + 2 * torch.arange(columns // 32)[None, :]) % 3 - 1)).to(
                torch.float8_e8m0fnu
            )
            decoded = torch.stack((values[low], values[high]), dim=-1).flatten(1)
            value = (decoded * scales.float().repeat_interleave(32, dim=1)).bfloat16()
            key = f"layers.0.ffn.experts.{expert}.w{projection}.weight"
            source[key] = raw if quantized else value
            if quantized:
                source[key.removesuffix(".weight") + ".scale"] = scales
            projections[expert, projection] = value
    expected["model.layers.0.ffn.experts.gate_and_up_projs"] = torch.stack(
        [torch.cat((projections[expert, 1].T, projections[expert, 3].T), dim=-1) for expert in range(2)]
    )
    expected["model.layers.0.ffn.experts.down_projs"] = torch.stack([projections[expert, 2].T for expert in range(2)])
    assert expected.keys() == schema.keys()
    return config, source, expected


def _load_worker(rank: int, rendezvous: str, quantized: bool) -> None:
    """Run the unchanged NeMo/Checkpointer loading path on two real Gloo ranks."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=120)
    )
    try:
        config, source, expected = _fixture(quantized)
        checkpoint = Path(rendezvous).parent / "checkpoint"
        if rank == 0:
            checkpoint.mkdir()
            config.save_pretrained(checkpoint)
            save_file(source, checkpoint / "model.safetensors")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(source, "model.safetensors")})
            )
        dist.barrier()
        loaded_config = DeepseekV41Config.from_pretrained(checkpoint, local_files_only=True)
        assert hasattr(loaded_config, "quantization_config") is quantized
        # _build_model unconditionally queries CUDA even for a CPU model.
        # This hardware-only adaptation follows the existing MuseGlimmer test;
        # no checkpoint decision, adapter, reader or model constructor is mocked.
        with patch.object(torch.cuda, "current_device", return_value=torch.device("cpu")):
            model = NeMoAutoModelForCausalLM.from_config(
                str(checkpoint),
                load_base_model=True,
                backend=_backend(),
                attn_implementation="eager",
                torch_dtype=torch.bfloat16,
                force_hf=False,
                use_liger_kernel=False,
                use_sdpa_patching=False,
                cache_dir=str(checkpoint.parent / "cache"),
            )
        assert isinstance(model, DeepseekV41ForCausalLM)
        assert model.config.name_or_path == str(checkpoint)
        assert hasattr(model.config, "quantization_config") is quantized
        actual = model.state_dict()
        assert actual.keys() == expected.keys()
        for name, reference in expected.items():
            assert actual[name].device.type == "cpu", name
            assert actual[name].dtype == reference.dtype, name
            torch.testing.assert_close(actual[name], reference, rtol=0, atol=0, msg=name)
        assert model.state_dict_adapter.dtype == torch.bfloat16
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("quantized", [True, False], ids=["fp8_fp4_metadata", "floating_without_metadata"])
def test_nemo_from_config_infers_quantized_dcp_initialization(tmp_path: Path, quantized: bool) -> None:
    """The load-only checkpointer must derive its format from the actual config."""
    mp.spawn(_load_worker, args=(str(tmp_path / "rendezvous"), quantized), nprocs=2, join=True)
