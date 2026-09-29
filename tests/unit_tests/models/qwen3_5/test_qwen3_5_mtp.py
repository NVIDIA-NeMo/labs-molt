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

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as HFQwen3_5ForCausalLM

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5.model import (
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    build_mtp_config_from_hf,
)


def _tiny_config(**kwargs):
    layer_types = kwargs.pop("layer_types", ["full_attention"])
    cfg = Qwen3_5TextConfig(
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
        layer_types=layer_types,
        attn_implementation="eager",
        # Match HF's default fp32 init so native-vs-HF parity comparisons line up;
        # the native backbone otherwise defaults its params to bf16.
        torch_dtype="float32",
        **kwargs,
    )
    return cfg


def _tiny_vlm_config(**kwargs):
    text_config = _tiny_config(**kwargs)
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
    )


def _backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        # rope_fusion defaults to (HAVE_TE and cuda) -> True on the GPU CI runner.
        # Qwen3.5's gated-query RoPE is incompatible with TE's fused rope kernel,
        # so every recipe sets rope_fusion=false. Pin it for deterministic tests.
        rope_fusion=False,
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


class TestQwen3_5MTPConfig:
    def test_uses_mtp_num_hidden_layers(self):
        cfg = _tiny_config(mtp_num_hidden_layers=2)
        mtp_config = build_mtp_config_from_hf(cfg)

        assert mtp_config.enabled
        assert mtp_config.num_layers == 2
        assert mtp_config.layer_pattern == "*"

    def test_num_nextn_override_takes_precedence(self):
        cfg = _tiny_config(mtp_num_hidden_layers=1)
        mtp_config = build_mtp_config_from_hf(cfg, num_nextn_predict_layers=3)

        assert mtp_config.num_layers == 3


class TestQwen3_5MTPModel:
    def test_dense_adapter_routes_fp32_linear_attn_keys(self):
        # The native backbone builds the fp32 ``_fp32_params`` SSMGate holder at
        # construction, so the adapter must rename ``_fp32_params.A_log`` keys back
        # to the bare HF layout at the save/load boundary.
        cfg = _tiny_config(mtp_num_hidden_layers=1)
        model = Qwen3_5ForCausalLM(cfg, backend=_backend())

        assert model.state_dict_adapter.route_linear_attn_fp32_params

    def test_mtp_disabled_matches_hf_forward(self):
        cfg = _tiny_config(mtp_num_hidden_layers=0, use_cache=False)
        torch.manual_seed(1234)
        hf_model = HFQwen3_5ForCausalLM(cfg).eval()
        model = Qwen3_5ForCausalLM(cfg, backend=_backend()).eval()
        model.load_state_dict(hf_model.state_dict(), strict=True)

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids, use_cache=False).logits
            custom_logits = model(input_ids=input_ids, use_cache=False).logits

        torch.testing.assert_close(custom_logits, hf_logits)

    def test_dense_mtp_builds_full_attention_block(self):
        cfg = _tiny_config(mtp_num_hidden_layers=1, layer_types=["linear_attention"])
        model = Qwen3_5ForCausalLM(cfg, backend=_backend())

        assert model.mtp is not None
        mtp_layer = model.mtp.layers[0]
        assert mtp_layer.layer_type == "full_attention"
        assert hasattr(mtp_layer, "self_attn")
        assert hasattr(mtp_layer, "mlp")
        assert hasattr(mtp_layer, "eh_proj")
        assert hasattr(mtp_layer, "final_layernorm")

    def test_forward_emits_mtp_hidden_states_in_training(self):
        cfg = _tiny_config(mtp_num_hidden_layers=1)
        model = Qwen3_5ForCausalLM(cfg, backend=_backend())
        model.train()

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        out = model(input_ids=input_ids)

        assert out.logits.shape == (1, 4, cfg.vocab_size)
        assert out.mtp_per_depth_h is not None
        assert len(out.mtp_per_depth_h) == 1
        assert out.mtp_per_depth_h[0].shape == (1, 4, cfg.hidden_size)
        assert out.mtp_loss_scaling_factor == 0.1

    def test_eval_does_not_emit_mtp_hidden_states(self):
        cfg = _tiny_config(mtp_num_hidden_layers=1)
        model = Qwen3_5ForCausalLM(cfg, backend=_backend())
        model.eval()

        out = model(input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long))

        assert out.mtp_per_depth_h is None
        assert out.mtp_loss_scaling_factor is None


class TestQwen3_5VLMMTPModel:
    def test_vlm_adapter_routes_fp32_linear_attn_keys(self):
        # Same native fp32 SSMGate routing as the text-only model (see above).
        cfg = _tiny_vlm_config(mtp_num_hidden_layers=1)
        model = Qwen3_5ForConditionalGeneration(cfg, backend=_backend())

        assert model.state_dict_adapter.route_linear_attn_fp32_params

    def test_vlm_forward_emits_mtp_hidden_states_in_training(self):
        cfg = _tiny_vlm_config(mtp_num_hidden_layers=1)
        model = Qwen3_5ForConditionalGeneration(cfg, backend=_backend(), mtp_loss_scaling_factor=0.2)
        model.train()

        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

        assert out.logits.shape == (1, 4, cfg.text_config.vocab_size)
        assert out.mtp_per_depth_h is not None
        assert len(out.mtp_per_depth_h) == 1
        assert out.mtp_per_depth_h[0].shape == (1, 4, cfg.text_config.hidden_size)
        assert out.mtp_loss_scaling_factor == 0.2

    def test_vlm_image_forward_emits_mtp_hidden_states_in_training(self):
        cfg = _tiny_vlm_config(mtp_num_hidden_layers=1)
        model = Qwen3_5ForConditionalGeneration(cfg, backend=_backend(), mtp_loss_scaling_factor=0.2)
        model.train()

        input_ids = torch.tensor([[1, cfg.vision_start_token_id, cfg.image_token_id, cfg.vision_end_token_id, 2]])
        attention_mask = torch.ones_like(input_ids)
        mm_token_type_ids = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.int)
        image_grid_thw = torch.tensor([[1, 1, 1]], dtype=torch.long)
        pixel_values = torch.randn(1, 3 * cfg.vision_config.temporal_patch_size * cfg.vision_config.patch_size**2)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
        )

        assert out.logits.shape == (1, 5, cfg.text_config.vocab_size)
        assert out.mtp_per_depth_h is not None
        assert len(out.mtp_per_depth_h) == 1
        assert out.mtp_per_depth_h[0].shape == (1, 5, cfg.text_config.hidden_size)
        assert out.mtp_loss_scaling_factor == 0.2
