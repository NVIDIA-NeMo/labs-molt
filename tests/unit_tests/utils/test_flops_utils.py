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

from types import SimpleNamespace

import pytest

from nemo_automodel._transformers import mfu as mfu_module
from nemo_automodel._transformers.mfu import AutoMFU, get_device_flops
from nemo_automodel.components.utils import flops_utils


def _moonlight_16b_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=2048,
        num_hidden_layers=27,
        num_attention_heads=16,
        intermediate_size=11264,
        vocab_size=163840,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        moe_intermediate_size=1408,
        num_experts_per_tok=6,
        moe_layer_freq=[0] * 1 + [1] * 26,
        mtp_num_layers=0,
    )


def _deepseek_v3_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=7168,
        num_hidden_layers=61,
        num_attention_heads=128,
        intermediate_size=18432,
        vocab_size=151936,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        moe_intermediate_size=2048,
        num_experts_per_tok=8,
        moe_layer_freq=[0] * 3 + [1] * 58,
        mtp_num_layers=None,
    )


def _gpt3_cfg() -> SimpleNamespace:
    return SimpleNamespace(hidden_size=2048, num_hidden_layers=16, vocab_size=50257)


def _llama2_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        vocab_size=32000,
    )


def _llama3_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=32000,
    )


def _nemotron_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=24,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=11008,
        vocab_size=32000,
    )


def _mixtral_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
        intermediate_size=14336,
        vocab_size=32000,
        num_experts_per_tok=2,
    )


def _qwen3_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=14336,
        vocab_size=32000,
        moe_topk=1,
        head_dim=256,
    )


def _bert_cfg() -> SimpleNamespace:
    return SimpleNamespace(num_hidden_layers=12, hidden_size=768, vocab_size=30522)


def _transformer_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=24,
        hidden_size=2048,
        num_attention_heads=16,
        intermediate_size=8192,
        vocab_size=50257,
    )


def _gpt_oss_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        num_hidden_layers=16,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_experts_per_tok=1,
        moe_ffn_hidden_size=8192,
        moe_router_topk=1,
        vocab_size=50257,
        kv_channels=128,
        window_size=[128],
        window_attn_skip_freq=2,
    )


def _glm4_moe_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=4096,
        num_hidden_layers=46,
        num_attention_heads=96,
        num_key_value_heads=8,
        intermediate_size=10944,
        vocab_size=151552,
        moe_intermediate_size=1408,
        num_experts_per_tok=8,
        n_shared_experts=1,
        n_routed_experts=128,
        first_k_dense_replace=1,
        max_position_embeddings=131072,
    )


def _nemotronh_nano_v3_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=2688,
        num_hidden_layers=52,
        num_attention_heads=32,
        num_key_value_heads=2,
        intermediate_size=1856,
        vocab_size=131072,
        mamba_num_heads=64,
        mamba_head_dim=64,
        ssm_state_size=128,
        n_groups=8,
        num_experts_per_tok=6,
        moe_intermediate_size=1856,
        moe_shared_expert_intermediate_size=3712,
        n_routed_experts=128,
        hybrid_override_pattern="MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
    )


def _nemotronh_super_v3_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=4096,
        num_hidden_layers=88,
        num_attention_heads=32,
        num_key_value_heads=2,
        intermediate_size=2688,
        vocab_size=131072,
        mamba_num_heads=128,
        mamba_head_dim=64,
        ssm_state_size=128,
        n_groups=8,
        num_experts_per_tok=22,
        moe_intermediate_size=2688,
        moe_shared_expert_intermediate_size=5376,
        n_routed_experts=512,
        moe_latent_size=1024,
        hybrid_override_pattern="MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEM*EMEMEMEME",
    )


def test_gpt_oss_flops_uses_head_dim_from_hf_config():
    config = SimpleNamespace(
        num_hidden_layers=1,
        hidden_size=24,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts_per_tok=2,
        intermediate_size=16,
        vocab_size=32,
        head_dim=8,
    )

    assert not hasattr(config, "kv_channels")
    assert config.head_dim != config.hidden_size // config.num_attention_heads
    assert int(flops_utils.gpt_oss_flops(config, gbs=1, seq_len=8)) == 271872


@pytest.mark.parametrize(
    "tflops, world_size, time_seconds, reference_mfu, expected_mfu",
    [
        # Basic test: 1 total TFLOPs, 1 GPU, 1 second, reference 1 TFLOPs/s -> 100% MFU
        (1.0, 1, 1.0, 1.0, 100.0),
        # Half efficiency: 0.5 total TFLOPs, 1 GPU, 1 second, reference 1 TFLOPs/s -> 50% MFU
        (0.5, 1, 1.0, 1.0, 50.0),
        # Multiple GPUs: 1 total TFLOPs, 8 GPUs, 1 second, reference 1 TFLOPs/s -> 12.5% MFU
        (1.0, 8, 1.0, 1.0, 12.5),
        # Longer time: 10 total TFLOPs, 1 GPU, 10 seconds, reference 1 TFLOPs/s -> 100% MFU
        (10.0, 1, 10.0, 1.0, 100.0),
        # Sparse BF16 or dense FP8 H100 reference: 989 total TFLOPs, 8 GPUs, reference 1979 TFLOPs/s
        (989.0, 8, 1.0, 1979.0, 6.2468418393127845),
        # Real-world scenario: 500 total TFLOPs, 64 GPUs, 2 seconds, H100 reference -> 0.197% MFU
        (500.0, 64, 2.0, 1979.0, 0.19738504295098536),
    ],
)
def test_calculate_mfu(tflops, world_size, time_seconds, reference_mfu, expected_mfu):
    """Test calculate_mfu function with various scenarios."""
    actual_mfu = flops_utils.calculate_mfu(
        tflops,
        world_size,
        time_seconds,
        reference_mfu=reference_mfu,
    )
    assert pytest.approx(actual_mfu, rel=1e-3) == expected_mfu


def test_calculate_mfu_warns_for_legacy_default_reference():
    """Keep the former H100 default temporarily while directing callers to an explicit peak."""
    with pytest.warns(FutureWarning, match="reference_mfu"):
        actual_mfu = flops_utils.calculate_mfu(tflops=1979.0, world_size=1, time_seconds=1.0)

    assert actual_mfu == 100.0


def test_automfu_h100_reference_uses_dense_bf16_peak():
    """Test AutoMFU's H100 dense-BF16 training convention."""
    h100_tflops = get_device_flops(unit="T", device_name="H100")

    assert h100_tflops == 989.0


@pytest.mark.parametrize(
    "device_name, expected_tflops",
    [
        ("NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition", 438.9),
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 503.8),
    ],
)
def test_get_device_flops_distinguishes_rtx_pro_6000_variants(device_name, expected_tflops):
    assert get_device_flops(unit="T", device_name=device_name) == expected_tflops


def test_get_device_flops_detects_current_cuda_device(monkeypatch):
    monkeypatch.setattr(mfu_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mfu_module.torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        mfu_module.torch.cuda,
        "get_device_name",
        lambda device: "NVIDIA H100 80GB HBM3" if device == 3 else "unexpected",
    )

    assert get_device_flops(unit="T") == 989.0


def test_automfu_defaults_to_detected_device_peak(monkeypatch):
    monkeypatch.setattr(mfu_module, "get_flops_formula_for_hf_config", lambda config: None)
    monkeypatch.setattr(mfu_module, "get_device_flops", lambda *, unit, device_name: 989.0)

    calculator = AutoMFU(SimpleNamespace())

    assert calculator.reference_mfu == 989.0


def test_automfu_accepts_precision_peak_override(monkeypatch):
    monkeypatch.setattr(mfu_module, "get_flops_formula_for_hf_config", lambda config: None)
    monkeypatch.setattr(
        mfu_module,
        "get_device_flops",
        lambda **kwargs: pytest.fail("explicit peak must bypass device lookup"),
    )

    calculator = AutoMFU(SimpleNamespace(), peak_tflops=1979.0)

    assert calculator.reference_mfu == 1979.0


def test_automfu_warns_and_reports_zero_for_unknown_device(monkeypatch, caplog):
    monkeypatch.setattr(
        mfu_module,
        "get_flops_formula_for_hf_config",
        lambda config: lambda config, gbs, seq_len: 1e12,
    )

    with caplog.at_level("WARNING", logger=mfu_module.__name__):
        calculator = AutoMFU(SimpleNamespace(), device="NVIDIA Example GPU")

    assert calculator.reference_mfu == float("inf")
    assert "MFU will be reported as 0%" in caplog.text
    assert calculator((1, 1), time_delta=1.0, world_size=1) == 0.0


@pytest.mark.parametrize("peak_tflops", [0.0, -1.0, float("inf"), float("nan")])
def test_automfu_rejects_invalid_precision_peak(monkeypatch, peak_tflops):
    monkeypatch.setattr(mfu_module, "get_flops_formula_for_hf_config", lambda config: None)

    with pytest.raises(ValueError, match="finite value greater than zero"):
        AutoMFU(SimpleNamespace(), peak_tflops=peak_tflops)


