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

"""CPU-safe execution coverage for the Qwen3.5-MoE text-only causal LM."""

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForCausalLM
from nemo_automodel.components.moe.layers import MoEConfig


def test_qwen3_5_moe_causal_lm_cpu_forward_backward():
    """A tiny text-only model can initialize and execute without VLM state."""
    config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=8,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        tie_word_embeddings=False,
        layer_types=["full_attention"],
    )
    moe_config = MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.hidden_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.num_experts,
        n_shared_experts=1,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=config.router_aux_loss_coef,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        softmax_before_topk=True,
        shared_expert_gate=True,
        shared_expert_inter_dim=config.shared_expert_intermediate_size,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )

    model = Qwen3_5MoeForCausalLM.from_config(config, moe_config=moe_config, backend=backend)
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.train()

    input_ids = torch.randint(0, config.vocab_size, (2, 6))
    position_ids = torch.arange(6).view(1, 1, 6).expand(3, 2, 6)
    output = model(input_ids=input_ids, position_ids=position_ids, output_hidden_states=True)
    loss = output.logits.square().mean()
    loss.backward()

    embedding_grad = model.get_input_embeddings().weight.grad
    assert output.logits.shape == (2, 6, config.vocab_size)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(loss)
    assert embedding_grad is not None
    assert torch.isfinite(embedding_grad).all()
    assert embedding_grad.abs().sum() > 0
