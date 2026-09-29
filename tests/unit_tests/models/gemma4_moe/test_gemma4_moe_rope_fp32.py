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

"""Regression test: Gemma4 MoE RoPE ``inv_freq`` must stay fp32 after bf16 init.

HF's ``Gemma4TextRotaryEmbedding`` precomputes ``inv_freq`` as an fp32 buffer.
``Gemma4ForConditionalGeneration.initialize_weights(dtype=torch.bfloat16)`` casts
the whole model to bf16 via ``cast_model_to_dtype``; ``nn.Module.to`` would round
that fp32 buffer to bf16 and corrupt RoPE precision. The model declares
``_keep_in_fp32_modules = ["rotary_emb"]`` so ``cast_model_to_dtype`` restores the
rotary buffers to fp32 afterwards. This test builds a tiny MoE model through the
real ``Gemma4MoETextModelBackend`` path and asserts the invariant holds while a
regular weight is bf16.

Runs on CPU (no CUDA / TE / DeepEP required).
"""

import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.model import (
    Gemma4ForConditionalGeneration,
    Gemma4MoETextModelBackend,
)


def _make_text_config(**overrides):
    """Tiny Gemma4TextConfig (2 layers, small hidden, tiny vocab, few experts)."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        enable_moe_block=True,  # routes construction through the NeMo MoE backend
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=64,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=128,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="bfloat16",
    )
    defaults.update(overrides)
    return Gemma4TextConfig(**defaults)


def _make_cpu_backend():
    """CPU-friendly backend: no TE, no DeepEP, plain torch kernels."""
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def test_initialize_weights_bf16_keeps_rope_inv_freq_fp32():
    """bf16 init must leave rotary ``inv_freq`` in fp32 while regular weights are bf16."""
    config = Gemma4Config(text_config=_make_text_config())
    model = Gemma4ForConditionalGeneration(config, backend=_make_cpu_backend())

    # Sanity: construction hit the real NeMo MoE backend, so initialize_weights
    # runs the cast_model_to_dtype path under test (not the early-return guard).
    assert isinstance(model.model.language_model, Gemma4MoETextModelBackend)

    model.initialize_weights(dtype=torch.bfloat16, buffer_device=torch.device("cpu"))

    rope_buffers = [
        (name, buf.dtype) for name, buf in model.named_buffers() if "inv_freq" in name or "freqs_cis" in name
    ]
    # The rotary module exposes at least one inv_freq buffer.
    assert rope_buffers, "no inv_freq/freqs_cis buffers found on the model"
    for name, dtype in rope_buffers:
        assert dtype == torch.float32, f"rotary buffer {name} was cast to {dtype}, expected float32"

    # A regular weight must still be bf16 (the cast actually happened).
    reg_weight = model.model.language_model.layers["0"].self_attn.q_proj.weight
    assert reg_weight.dtype == torch.bfloat16
