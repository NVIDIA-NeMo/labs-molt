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

"""Regression test: Qwen3.5 vision-tower rotary ``inv_freq`` stays fp32 across dtype casts.

``initialize_weights`` casts the model to bf16/fp16 via ``cast_model_to_dtype``,
whose bulk ``model.to()`` would round the vision tower's non-persistent fp32
``inv_freq`` buffer down to the low-precision dtype (0.5994842 -> 0.59765625),
corrupting every patch's RoPE phases (the error scales with the position id).
The model lists ``"rotary_pos_emb"`` in ``_keep_in_fp32_modules`` so the buffer
is snapshotted and restored at its exact fp32 values.

The same declaration also protects the frozen-vision recipes: after
initialization, ``apply_model_infrastructure`` calls
``cast_frozen_modules_to_compute_dtype``, which casts frozen-module buffers
unconditionally unless their names match the fp32 declarations. Runs on CPU.
"""

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import (
    cast_frozen_modules_to_compute_dtype,
    cast_model_to_dtype,
)
from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration


def _backend() -> BackendConfig:
    """CPU-friendly backend: plain torch kernels, no fused RoPE."""
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


def _tiny_vlm_config() -> Qwen3_5Config:
    text_config = Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=["full_attention"],
        attn_implementation="eager",
        torch_dtype="float32",
        tie_word_embeddings=False,
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
    )
    return Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
        tie_word_embeddings=False,
    )


def _build_model() -> Qwen3_5ForConditionalGeneration:
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(_tiny_vlm_config(), backend=_backend()).eval()


def _expected_inv_freq(rotary) -> torch.Tensor:
    """The exact fp32 table implied by the module's theta/dim."""
    return 1.0 / (rotary.theta ** (torch.arange(0, rotary.dim, 2, dtype=torch.float32) / rotary.dim))


def _assert_exact_fp32(rotary) -> None:
    # Exact values, not merely the dtype: a bf16-rounded value promoted back to
    # fp32 still passes a dtype check while corrupting RoPE phases.
    assert rotary.inv_freq.dtype == torch.float32
    torch.testing.assert_close(rotary.inv_freq, _expected_inv_freq(rotary), rtol=0.0, atol=0.0)


def test_class_declares_vision_rotary_in_fp32_modules():
    # Buffer-preservation requirement only: the non-strict list keeps matched
    # buffers fp32 through both casts without forcing a separate fp32 FSDP group.
    assert "rotary_pos_emb" in Qwen3_5ForConditionalGeneration._keep_in_fp32_modules
    assert "rotary_pos_emb" not in (getattr(Qwen3_5ForConditionalGeneration, "_keep_in_fp32_modules_strict", None) or [])


def test_initialize_weights_bf16_keeps_vision_inv_freq_exact_fp32():
    model = _build_model()
    model.initialize_weights(dtype=torch.bfloat16, buffer_device=torch.device("cpu"))

    _assert_exact_fp32(model.model.visual.rotary_pos_emb)
    # The bulk cast did happen for regular weights.
    assert model.lm_head.weight.dtype == torch.bfloat16


def test_repeated_bf16_fp16_casts_keep_exact_values():
    model = _build_model()
    rotary = model.model.visual.rotary_pos_emb
    # Round-trip through the shared cast machinery repeatedly: each pass must
    # restore the snapshotted fp32 values, never accumulate rounding.
    for dtype in (torch.bfloat16, torch.float16, torch.bfloat16, torch.float32, torch.bfloat16):
        cast_model_to_dtype(model, dtype)
        _assert_exact_fp32(rotary)
    assert model.lm_head.weight.dtype == torch.bfloat16


def test_frozen_vision_cast_keeps_inv_freq_exact_fp32():
    # The frozen-vision recipes freeze the whole tower, then
    # cast_frozen_modules_to_compute_dtype casts every frozen buffer to the
    # compute dtype unless its name matches an fp32 declaration.
    model = _build_model()
    model.initialize_weights(dtype=torch.bfloat16, buffer_device=torch.device("cpu"))
    for p in model.model.visual.parameters():
        p.requires_grad_(False)
    # Some trainable param elsewhere so the walk has work to do.
    assert any(p.requires_grad for p in model.parameters())

    cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)

    _assert_exact_fp32(model.model.visual.rotary_pos_emb)
    assert model.lm_head.weight.dtype == torch.bfloat16


def test_rope_output_after_frozen_cast_equals_fp32_reference():
    # End-to-end within the rotary: angles produced after a full bf16 init +
    # frozen-module cast must equal the fp32 reference angles bit-for-bit.
    model = _build_model()
    model.initialize_weights(dtype=torch.bfloat16, buffer_device=torch.device("cpu"))
    for p in model.model.visual.parameters():
        p.requires_grad_(False)
    cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)

    rotary = model.model.visual.rotary_pos_emb
    # 3-D grid the way the vision tower feeds it: (t, h, w) per patch.
    pos_ids = torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1]]], dtype=torch.float32)
    with torch.no_grad():
        out = rotary(pos_ids)

    # Reference built the way the module does: outer(pos, inv_freq) flattened.
    ref = (pos_ids.unsqueeze(-1) * _expected_inv_freq(rotary)).flatten(1)
    torch.testing.assert_close(out, ref, rtol=0.0, atol=0.0)
