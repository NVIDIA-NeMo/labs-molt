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

"""CPU unit tests for the Qwen3-MoE compile_attn (fullgraph attention compile) wiring.

These mock the submodule initializers so Qwen3MoeAttention constructs without a GPU or
TransformerEngine. They cover the compile_attn gating in __init__ (default off, and the
warn/no-op path when attn != 'sdpa'); the actual torch.compile call (attn='sdpa' + GPU)
is pragma-excluded as it only runs on the GPU benchmark.
"""

from unittest.mock import Mock, patch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_moe.layers import Qwen3MoeAttention


def _config():
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

    cfg = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        moe_intermediate_size=64,
        num_experts=0,
        num_experts_per_tok=1,
    )
    cfg.head_dim = 16
    return cfg


@patch("nemo_automodel.components.models.qwen3_moe.layers.initialize_attn_module_and_func")
@patch("nemo_automodel.components.models.qwen3_moe.layers.initialize_rms_norm_module")
@patch("nemo_automodel.components.models.qwen3_moe.layers.initialize_linear_module")
class TestQwen3MoeCompileAttn:
    """compile_attn gating in Qwen3MoeAttention.__init__."""

    def _build(self, mock_lin, mock_rms, mock_attn, backend):
        mock_lin.return_value = Mock()
        mock_rms.return_value = Mock()
        mock_attn.return_value = (Mock(), Mock())
        return Qwen3MoeAttention(_config(), backend)

    def test_compile_attn_off_leaves_uncompiled(self, mock_lin, mock_rms, mock_attn):
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False, compile_attn=False)
        attn = self._build(mock_lin, mock_rms, mock_attn, backend)
        assert attn._compiled_forward is None

    def test_compile_attn_on_non_sdpa_warns_and_noops(self, mock_lin, mock_rms, mock_attn):
        # compile_attn=True but attn != 'sdpa' → warn and leave the forward uncompiled.
        backend = BackendConfig(attn="te", linear="torch", rms_norm="torch", rope_fusion=False, compile_attn=True)
        attn = self._build(mock_lin, mock_rms, mock_attn, backend)
        assert attn._compiled_forward is None
