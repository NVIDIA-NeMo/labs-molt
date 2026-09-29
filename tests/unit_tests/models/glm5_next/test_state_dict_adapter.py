# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch
from torch.distributed.tensor import Shard

from nemo_automodel.components.models.glm5_next.state_dict_adapter import (
    _apply_local_block_scales,
    _local_shard_offsets,
    dequantize_block_fp8,
)
from tests.unit_tests.models.glm5_next.conftest import tiny_glm5_next_model


def test_native_hf_round_trip_preserves_every_tensor():
    torch.manual_seed(29)
    model = tiny_glm5_next_model()
    native = model.state_dict()

    hf_state = model.state_dict_adapter.to_hf(native)
    restored = model.state_dict_adapter.from_hf(dict(hf_state))

    assert restored.keys() == native.keys()
    for key, value in native.items():
        torch.testing.assert_close(restored[key], value, rtol=0.0, atol=0.0)


def test_adapter_routes_flat_hyperconnection_and_kda_fp32_parameters():
    adapter = tiny_glm5_next_model().state_dict_adapter
    hf_state = {
        "model.language_model.layers.0.hc_attn_fn": torch.randn(8, 32),
        "model.language_model.layers.0.hc_attn_base": torch.randn(8),
        "model.language_model.layers.0.hc_attn_scale": torch.randn(3),
        "model.language_model.layers.0.self_attn.A_log": torch.randn(2),
        "model.language_model.layers.0.self_attn.dt_bias": torch.randn(8),
    }

    native = adapter.from_hf(dict(hf_state))

    assert native["model.language_model.layers.0.attn_hc.fn"].shape == (8, 32)
    assert native["model.language_model.layers.0.attn_hc._fp32_params.base"].dtype is torch.float32
    assert native["model.language_model.layers.0.attn_hc._fp32_params.scale"].dtype is torch.float32
    assert native["model.language_model.layers.0.self_attn._fp32_params.A_log"].shape == (2,)
    assert native["model.language_model.layers.0.self_attn._fp32_params.A_log"].dtype is torch.float32
    assert native["model.language_model.layers.0.self_attn._fp32_params.dt_bias"].dtype is torch.float32
    assert adapter.to_hf(native).keys() == hf_state.keys()


def test_quantized_load_plan_matches_sparse_but_not_linear_output_projection():
    model = tiny_glm5_next_model()
    planned = model.state_dict_adapter.to_hf(model.state_dict(), quantization=True, for_checkpoint_load=True)

    linear_o = "model.language_model.layers.0.self_attn.o_proj.weight"
    sparse_o = "model.language_model.layers.3.self_attn.o_proj.weight"
    assert linear_o in planned and linear_o + "_scale_inv" not in planned
    assert sparse_o in planned and sparse_o + "_scale_inv" in planned


def test_mtp_layer_is_dropped_on_load():
    model = tiny_glm5_next_model()
    layer_limit = model.config.text_config.num_hidden_layers
    state = {
        "lm_head.weight": torch.randn(64, 16),
        f"model.language_model.layers.{layer_limit}.enorm.weight": torch.randn(16),
    }

    converted = model.state_dict_adapter.from_hf(state)

    assert converted.keys() == {"lm_head.weight"}


def test_block_fp8_dequantization_uses_each_128_square_scale():
    weight = torch.ones((129, 129), dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    output = dequantize_block_fp8(weight, scale, dtype=torch.float32)

    assert output[0, 0] == 1
    assert output[0, 128] == 2
    assert output[128, 0] == 3
    assert output[128, 128] == 4


def test_block_fp8_dequantization_respects_misaligned_dtensor_shard_offset():
    weight = torch.ones((86, 129), dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    output = _apply_local_block_scales(weight, scale, (58, 0), torch.float32)

    assert output[0, 0] == 1
    assert output[0, 128] == 2
    assert output[69, 0] == 1
    assert output[70, 0] == 3
    assert output[85, 128] == 4


def test_dtensor_shard_offset_uses_torch_uneven_chunk_layout():
    class FakeMesh:
        def size(self, mesh_dim):
            assert mesh_dim == 0
            return 144

        def get_local_rank(self, mesh_dim):
            assert mesh_dim == 0
            return 23

    class FakeDTensor:
        ndim = 2
        shape = (12288, 4096)
        placements = (Shard(0),)
        device_mesh = FakeMesh()

    assert _local_shard_offsets(FakeDTensor()) == (1978, 0)
