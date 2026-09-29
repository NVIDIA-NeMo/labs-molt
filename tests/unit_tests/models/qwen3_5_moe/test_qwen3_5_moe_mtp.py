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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration


def _tiny_vl_config(**text_kwargs):
    text_config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=["full_attention"],
        **text_kwargs,
    )
    vision_config = dict(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
        num_position_embeddings=8,
    )
    return Qwen3_5MoeConfig(text_config=text_config.to_dict(), vision_config=vision_config)


def _backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


class TestQwen3_5MoeMTP:
    def test_vlm_declares_mtp_cp_and_prepares_multi_axis_inputs(self):
        model = Qwen3_5MoeForConditionalGeneration(_tiny_vl_config(mtp_num_hidden_layers=1), backend=_backend())
        batch = {
            "input_ids": torch.tensor([[10, 11, 20, 21]]),
            "labels": torch.tensor([[11, 12, 21, 22]]),
            "position_ids": torch.tensor(
                [
                    [[0, 1, 0, 1]],
                    [[10, 11, 20, 21]],
                    [[30, 31, 40, 41]],
                ]
            ),
            "_packed_seq_ids": torch.tensor([[0, 0, 1, 1]]),
        }

        prepared = model.prepare_mtp_inputs_for_cp(batch, ignore_index=-100)

        assert model.ModelCapabilities().supports_mtp_cp
        assert prepared.position_ids_seq_dim == 2
        torch.testing.assert_close(prepared.input_ids[0], torch.tensor([[11, 0, 21, 0]]))
        torch.testing.assert_close(
            prepared.position_ids[0],
            torch.tensor(
                [
                    [[1, 0, 1, 0]],
                    [[11, 0, 21, 0]],
                    [[31, 0, 41, 0]],
                ]
            ),
        )
        torch.testing.assert_close(prepared.targets[0], torch.tensor([[12, -100, 22, -100]]))
        torch.testing.assert_close(prepared.valid_masks[0], torch.tensor([[True, False, True, False]]))

    def test_moe_mtp_builds_full_attention_moe_block(self):
        model = Qwen3_5MoeForConditionalGeneration(_tiny_vl_config(mtp_num_hidden_layers=1), backend=_backend())

        assert model.mtp is not None
        mtp_layer = model.mtp.layers[0]
        assert mtp_layer.layer_type == "full_attention"
        assert hasattr(mtp_layer, "self_attn")
        assert hasattr(mtp_layer, "eh_proj")
        assert hasattr(mtp_layer, "final_layernorm")

        mtp_keys = [key for key in model.state_dict() if key.startswith("mtp.")]
        assert "mtp.layers.0.mlp.gate.weight" in mtp_keys
        assert "mtp.layers.0.mlp.experts.gate_and_up_projs" in mtp_keys
        assert "mtp.layers.0.mlp.shared_experts.gate_proj.weight" in mtp_keys

    def test_moe_mtp_emits_with_media_token_ids(self, monkeypatch):
        cfg = _tiny_vl_config(mtp_num_hidden_layers=1)
        cfg.image_token_id = 1
        cfg.vision_start_token_id = 2
        model = Qwen3_5MoeForConditionalGeneration(cfg, backend=_backend())
        model.train()

        hidden_states = torch.randn(1, 4, cfg.text_config.hidden_size, dtype=model.lm_head.weight.dtype)

        def fake_model_forward(**kwargs):
            assert "pixel_values" in kwargs
            return SimpleNamespace(last_hidden_state=hidden_states)

        def fake_mtp_forward(hidden_states, **kwargs):
            assert "pixel_values" not in kwargs
            assert "image_grid_thw" not in kwargs
            return [hidden_states]

        monkeypatch.setattr(model.model, "forward", fake_model_forward)
        monkeypatch.setattr(model.mtp, "forward", fake_mtp_forward)

        out = model(
            input_ids=torch.tensor([[3, 2, 1, 4]], dtype=torch.long),
            pixel_values=torch.empty(0),
            image_grid_thw=torch.empty((0, 3), dtype=torch.long),
        )

        assert out.logits.shape == (1, 4, cfg.text_config.vocab_size)
        assert out.mtp_per_depth_h is not None
        assert len(out.mtp_per_depth_h) == 1
        assert out.mtp_per_depth_h[0].shape == (1, 4, cfg.text_config.hidden_size)
