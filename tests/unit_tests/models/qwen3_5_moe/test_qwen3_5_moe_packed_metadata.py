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

"""CPU regression coverage for checkpointed Qwen3.5-MoE packed metadata."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from nemo_automodel.components.models.common import BackendConfig, packing
from nemo_automodel.components.models.qwen3_5 import packing as qwen3_5_packing
from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeTextModelBackend


def test_reuses_packed_metadata_across_checkpointed_moe_layers():
    torch.manual_seed(123)
    config = Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        layer_types=["linear_attention", "linear_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        use_cache=False,
        torch_dtype="float32",
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    model = Qwen3_5MoeTextModelBackend(config, backend).float().train()
    linear_attn_modules = []
    for name in list(model.layers):
        linear_attn_modules.append(model.layers[name].linear_attn)
        model.layers[name] = checkpoint_wrapper(
            model.layers[name],
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )

    get_unpad_data = MagicMock(wraps=packing.get_unpad_data)
    chunk_gated_delta_rule = MagicMock(side_effect=lambda *args, **_kwargs: (args[2], None))
    for linear_attn in linear_attn_modules:
        linear_attn.causal_conv1d_fn = MagicMock(side_effect=lambda **kwargs: kwargs["x"])
        linear_attn.chunk_gated_delta_rule = chunk_gated_delta_rule
        linear_attn.norm.forward = MagicMock(side_effect=torch.add)

    with patch.object(qwen3_5_packing, "get_unpad_data", get_unpad_data):
        output = model(
            input_ids=torch.tensor([[1, 2, 3, 4]]),
            attention_mask=torch.tensor([[1, 1, 2, 2]]),
        ).last_hidden_state
        output.backward(torch.randn_like(output))

    assert get_unpad_data.call_count == 1
    assert chunk_gated_delta_rule.call_count == 4
    call_kwargs = [call.kwargs for call in chunk_gated_delta_rule.call_args_list]
    device_cu_seqlens = call_kwargs[0]["cu_seqlens"]
    cpu_cu_seqlens = call_kwargs[0]["cu_seqlens_cpu"]
    assert cpu_cu_seqlens.device.type == "cpu"
    assert cpu_cu_seqlens.tolist() == [0, 2, 4]
    assert all(kwargs["cu_seqlens"] is device_cu_seqlens for kwargs in call_kwargs)
    assert all(kwargs["cu_seqlens_cpu"] is cpu_cu_seqlens for kwargs in call_kwargs)
