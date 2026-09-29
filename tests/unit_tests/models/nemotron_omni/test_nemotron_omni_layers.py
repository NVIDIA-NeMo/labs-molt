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

"""Pure-Python unit tests for NemotronOmni helper modules and the registry entry.

These tests exercise the small, dependency-free building blocks of the
NemotronOmni wrapper (activation, norm, projectors, config) plus the
architecture registration that lets ``MODEL_ARCH_MAPPING`` resolve the v3 dump.
The heavy ``NemotronOmniForConditionalGeneration`` end-to-end forward path
requires a full HF v3 checkpoint and is covered by the functional/integration
suites — not here.
"""

import pytest
import torch

from nemo_automodel.components.models.nemotron_omni.model import (
    NemotronOmniConfig,
    NemotronOmniForConditionalGeneration,
    RMSNorm,
    SoundProjection,
    SquaredReLU,
    VisionProjector,
)

# ---------------------------------------------------------------------------
# Activations / norms
# ---------------------------------------------------------------------------


def test_squared_relu_matches_reference():
    """SquaredReLU(x) == relu(x)**2 elementwise (zeros out negatives, squares positives)."""
    act = SquaredReLU()
    x = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    expected = torch.tensor([0.0, 0.0, 0.0, 0.25, 4.0])
    out = act(x)
    assert torch.allclose(out, expected)


def test_squared_relu_preserves_shape():
    act = SquaredReLU()
    x = torch.randn(2, 3, 5)
    assert act(x).shape == x.shape


def test_rmsnorm_matches_reference():
    """RMSNorm equals torch.nn.functional reference: x / sqrt(mean(x^2)) * weight."""
    hidden = 8
    norm = RMSNorm(hidden, eps=1e-5)
    # Set weight to a known non-trivial value.
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, hidden))

    x = torch.randn(2, 4, hidden)
    out = norm(x)

    # Reference: float32 normalisation, then cast back to input dtype.
    x32 = x.to(torch.float32)
    rms = x32.pow(2).mean(-1, keepdim=True).add(1e-5).rsqrt()
    expected = (norm.weight.to(torch.float32) * (x32 * rms)).to(x.dtype)
    assert torch.allclose(out, expected, atol=1e-6)


def test_rmsnorm_preserves_input_dtype():
    """RMSNorm computes in fp32 internally but returns the input dtype."""
    norm = RMSNorm(4)
    x = torch.randn(1, 2, 4, dtype=torch.bfloat16)
    out = norm(x)
    assert out.dtype == torch.bfloat16


# ---------------------------------------------------------------------------
# Projectors
# ---------------------------------------------------------------------------


def test_vision_projector_shapes():
    """VisionProjector: (B, T, vit_hidden * pixel_shuffle_channels) -> (B, T, llm_hidden)."""
    vit_hidden = 16
    proj_hidden = 32
    llm_hidden = 24
    downsample = 0.5  # pixel_shuffle factor: input dim becomes vit_hidden * (1/0.5)^2
    proj = VisionProjector(vit_hidden, proj_hidden, llm_hidden, downsample_ratio=downsample)

    in_features = vit_hidden * int(1 / downsample) ** 2
    x = torch.randn(2, 5, in_features)
    out = proj(x)

    assert out.shape == (2, 5, llm_hidden)


def test_vision_projector_layer_dims():
    """Internal Linear layers should size to the pixel-shuffled input width."""
    vit_hidden = 16
    proj = VisionProjector(vit_hidden, projector_hidden_size=32, llm_hidden_size=24)
    expected_in = vit_hidden * 4  # default downsample_ratio=0.5 -> 4x channel multiplier
    assert proj.linear1.in_features == expected_in
    assert proj.linear1.out_features == 32
    assert proj.linear2.in_features == 32
    assert proj.linear2.out_features == 24


def test_sound_projection_shapes():
    """SoundProjection: (B, T, sound_hidden) -> (B, T, llm_hidden)."""
    sound_hidden = 12
    proj_hidden = 20
    llm_hidden = 8
    proj = SoundProjection(sound_hidden, proj_hidden, llm_hidden)

    x = torch.randn(3, 7, sound_hidden)
    out = proj(x)
    assert out.shape == (3, 7, llm_hidden)


def test_sound_projection_no_bias_by_default():
    proj = SoundProjection(8, 16, 4)
    assert proj.linear1.bias is None
    assert proj.linear2.bias is None


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


def test_nemotron_omni_config_defaults():
    """NemotronOmniConfig defaults match the v3 architecture's expected settings."""
    cfg = NemotronOmniConfig()
    assert cfg.model_type == "NemotronH_Nano_Omni_Reasoning_V3"
    assert cfg.is_composition is True
    assert cfg.downsample_ratio == 0.5
    assert cfg.patch_size == 16
    assert cfg.vit_hidden_size == 1280
    # v3 uses dynamic-resolution video; default pruning rate is non-zero (EVS on).
    assert cfg.video_pruning_rate == pytest.approx(0.7)


def test_nemotron_omni_config_overrides_propagate():
    cfg = NemotronOmniConfig(
        downsample_ratio=0.25,
        patch_size=14,
        vit_hidden_size=2048,
        video_pruning_rate=0.0,
    )
    assert cfg.downsample_ratio == 0.25
    assert cfg.patch_size == 14
    assert cfg.vit_hidden_size == 2048
    assert cfg.video_pruning_rate == 0.0


# ---------------------------------------------------------------------------
# Model compatibility properties
# ---------------------------------------------------------------------------


def test_lm_head_property_exposes_nested_head_without_registering_alias():
    """The FSDP-facing property must preserve the nested checkpoint hierarchy."""
    model = object.__new__(NemotronOmniForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.language_model = torch.nn.Module()
    model.language_model.lm_head = torch.nn.Linear(4, 8, bias=False)

    assert model.lm_head is model.language_model.lm_head
    assert "lm_head" not in model._modules
    assert set(model.state_dict()) == {"language_model.lm_head.weight"}


# ---------------------------------------------------------------------------
# Registry — verify the architecture name dispatches to our wrapper
# ---------------------------------------------------------------------------


def test_registry_entry_present():
    """``MODEL_ARCH_MAPPING`` should resolve the v3 architecture name to our wrapper."""
    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

    mapping = dict(MODEL_ARCH_MAPPING)
    assert "NemotronH_Nano_Omni_Reasoning_V3" in mapping
    module_path, class_name, *_ = mapping["NemotronH_Nano_Omni_Reasoning_V3"]
    assert module_path == "nemo_automodel.components.models.nemotron_omni.model"
    assert class_name == "NemotronOmniForConditionalGeneration"


def test_registry_v2_entry_removed():
    """V2 dispatch was deleted along with V2 dump support — keep it gone."""
    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

    assert "NemotronH_Nano_VL_V2" not in dict(MODEL_ARCH_MAPPING)


def test_initialize_weights_preserves_sharded_mamba_fp32(tmp_path):
    """Preserve FP32 storage and values so DCP can initialize fresh optimizer state."""
    import torch.distributed as dist
    from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    from nemo_automodel.components.models.nemotron_v3.layers import NemotronV3MambaFP32Params

    class TinyLanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._fp32_params = NemotronV3MambaFP32Params(num_heads=4)

        @torch.no_grad()
        def initialize_weights(self, buffer_device=None, dtype=torch.bfloat16):
            # Values deliberately not representable in BF16 expose rounding on a broad cast.
            for index, parameter in enumerate(self._fp32_params.parameters()):
                parameter.fill_(0.1234567 + index)

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/init", rank=0, world_size=1)
    try:
        model = object.__new__(NemotronOmniForConditionalGeneration)
        torch.nn.Module.__init__(model)
        model.language_model = TinyLanguageModel()
        model.vision_model = torch.nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        mesh = init_device_mesh("cpu", (1,))
        fully_shard(model.language_model._fp32_params, mesh=mesh)
        fully_shard(model, mesh=mesh)

        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)

        for index, parameter in enumerate(model.language_model._fp32_params.parameters()):
            local = parameter.to_local()
            assert parameter.dtype == torch.float32
            assert local.dtype == torch.float32
            torch.testing.assert_close(local, torch.full_like(local, 0.1234567 + index), rtol=0, atol=0)
            parameter.grad = torch.zeros_like(parameter)
            parameter.grad = None
        assert model.vision_model.weight.dtype == torch.bfloat16
        state = get_optimizer_state_dict(model, torch.optim.AdamW(model.parameters()))
        assert set(state["state"]) == set(dict(model.named_parameters()))
    finally:
        dist.destroy_process_group()
