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

"""Regression test: DeepseekV4 RoPE ``inv_freq`` must stay fp32 after bf16 init.

``DeepseekV4RotaryEmbedding`` registers ``inv_freq`` as an fp32 buffer.
``initialize_weights(dtype=torch.bfloat16)`` casts the whole model to bf16 via
``cast_model_to_dtype``; ``nn.Module.to`` would round that buffer to bf16 and
corrupt RoPE precision. The model lists ``"rotary_emb"`` in
``_keep_in_fp32_modules_strict`` so the rotary buffers (``rotary_emb`` and
``rotary_emb_compress``) are restored to fp32 afterwards. Runs on CPU.
"""

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM


def _tiny_config(**overrides):
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 64,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        hc_mult=4,
        num_hash_layers=0,
        compress_ratios=[0, 4],  # layer 0 full, layer 1 compressed
        sliding_window=16,
        num_nextn_predict_layers=0,  # no MTP
        rms_norm_eps=1e-6,
        torch_dtype="float32",
    )
    defaults.update(overrides)
    return DeepseekV4Config(**defaults)


def _cpu_backend():
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        dispatcher="torch",
        experts="torch_mm",
    )


def test_initialize_weights_bf16_keeps_rope_inv_freq_fp32():
    model = DeepseekV4ForCausalLM(_tiny_config(), backend=_cpu_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)

    rope_buffers = [
        (name, buf.dtype) for name, buf in model.named_buffers() if "inv_freq" in name or "freqs_cis" in name
    ]
    # Both the main and the compressed rotary embeddings expose inv_freq.
    assert len(rope_buffers) >= 2, f"expected >=2 rotary buffers, found {rope_buffers}"
    for name, dtype in rope_buffers:
        assert dtype == torch.float32, f"rotary buffer {name} was cast to {dtype}, expected float32"

    # A regular weight must be bf16 (the cast actually ran).
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
