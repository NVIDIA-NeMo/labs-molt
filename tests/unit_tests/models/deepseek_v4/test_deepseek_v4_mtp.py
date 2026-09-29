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

"""Unit tests for DeepSeek V4 Multi-Token Prediction (MTP) support.

All tests run on CPU with a tiny model config.

Run:
    PYTHONPATH=/path/to/Automodel_lao python -m pytest \
        tests/unit_tests/models/deepseek_v4/test_deepseek_v4_mtp.py -v -s
"""

import types

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM
from nemo_automodel.components.models.deepseek_v4.mtp import build_mtp_config_from_hf

# Over the default 5s budget on purpose: CUDA model forward and backward initialize compiled kernels.
# Reduce cold compiler startup before lowering this further.
pytestmark = pytest.mark.timeout(60)

# MoE.forward unconditionally creates a torch.cuda.Stream() for shared experts.
# Gate the tests that actually call model.forward() on CUDA availability.
_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MoE.forward unconditionally allocates torch.cuda.Stream() for shared experts",
)


def _tiny_config(**overrides) -> DeepseekV4Config:
    """Tiny V4 config for MTP tests: small enough to run fast on CPU."""
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
        rope_scaling=None,
        hc_mult=4,
        num_hash_layers=0,
        compress_ratios=[0, 0],
        sliding_window=16,
        num_nextn_predict_layers=0,  # disabled by default
        rms_norm_eps=1e-6,
        torch_dtype="float32",
    )
    defaults.update(overrides)
    return DeepseekV4Config(**defaults)


def _make_model(config: DeepseekV4Config) -> DeepseekV4ForCausalLM:
    """Build a tiny model with no HF state dict adapter."""
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        dispatcher="torch",
        experts="torch_mm",
    )
    model = DeepseekV4ForCausalLM(config, backend=backend)
    model = model.float()
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.zero_()
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMTPConfig:
    def test_mtp_config_disabled(self):
        """num_nextn_predict_layers=0 -> mtp_config.enabled == False."""
        cfg = _tiny_config(num_nextn_predict_layers=0)
        mtp_config = build_mtp_config_from_hf(cfg)
        assert not mtp_config.enabled
        assert mtp_config.num_layers == 0
        assert mtp_config.layer_pattern == ""

    def test_mtp_config_enabled(self):
        """num_nextn_predict_layers=1 -> mtp_config.enabled == True."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        mtp_config = build_mtp_config_from_hf(cfg)
        assert mtp_config.enabled
        assert mtp_config.num_layers == 1
        assert mtp_config.layer_pattern == "*"
        assert mtp_config.pattern_length == 1  # one full DSV4 MTP block per depth


class TestModelMTPConstruction:
    def test_model_has_no_mtp_by_default(self):
        """Default config (num_nextn_predict_layers=0) -> model.mtp is None."""
        cfg = _tiny_config(num_nextn_predict_layers=0)
        model = _make_model(cfg)
        assert model.mtp is None

    def test_model_has_mtp_when_configured(self):
        """With num_nextn_predict_layers=1 -> model.mtp is not None.

        DSV4 MTP has one full HC-enabled MTP block per depth.
        """
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        assert model.mtp is not None
        assert len(model.mtp.layers) == 1


class TestMTPForward:
    def test_mtp_rolls_input_ids_not_position_ids(self):
        """MTP predicts future tokens but uses the current sequence positions."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        block = model.mtp.layers[0]

        captured = {}

        def fake_forward(self, hidden_states, *, embed_input, input_ids=None, position_ids=None, **kwargs):
            del self, embed_input, kwargs
            captured["input_ids"] = input_ids.detach().clone()
            captured["position_ids"] = position_ids.detach().clone()
            return hidden_states, hidden_states.mean(dim=2)

        block.forward = types.MethodType(fake_forward, block)

        input_ids = torch.tensor([[10, 11, 12, 13]])
        position_ids = torch.arange(input_ids.shape[-1]).unsqueeze(0)
        hidden_states = torch.zeros(1, input_ids.shape[-1], cfg.hc_mult, cfg.hidden_size)

        out = model.mtp(
            input_ids=input_ids,
            hidden_states=hidden_states,
            embed_fn=model.model.embed_tokens,
            position_ids=position_ids,
        )

        assert len(out) == 1
        assert captured["input_ids"].tolist() == [[11, 12, 13, 0]]
        assert captured["position_ids"].tolist() == [[0, 1, 2, 3]]

    def test_pp_first_stage_propagates_shifted_mtp_embeddings(self):
        """First PP stage sends shifted token embeddings as auxiliary activations."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        model.train()
        model.lm_head = None
        model.mtp = None

        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(
                torch.arange(cfg.vocab_size * cfg.hidden_size, dtype=torch.float32).view(
                    cfg.vocab_size, cfg.hidden_size
                )
            )

        def fake_backbone(self, input_ids, **kwargs):
            del self, input_ids, kwargs
            return torch.ones(1, 4, cfg.hc_mult, cfg.hidden_size)

        model.model.forward = types.MethodType(fake_backbone, model.model)

        input_ids = torch.tensor([[10, 11, 12, 13]])
        out = model(input_ids)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[0].shape == (1, 4, cfg.hc_mult, cfg.hidden_size)
        expected_ids = torch.tensor([[11, 12, 13, 0]])
        torch.testing.assert_close(out[1], model.model.embed_tokens(expected_ids))

    def test_pp_final_stage_uses_propagated_mtp_embeddings(self):
        """Final PP stage computes MTP from the carried embeddings, not a local embed table."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        model.train()
        model.model.embed_tokens = None
        captured = {}

        def fake_backbone(self, input_ids, return_hc_hidden=False, **kwargs):
            del self, kwargs
            assert return_hc_hidden is True
            batch, seq = input_ids.shape[:2]
            hidden = torch.ones(batch, seq, cfg.hidden_size)
            hc_hidden = torch.ones(batch, seq, cfg.hc_mult, cfg.hidden_size) * 2
            return hidden, hc_hidden

        def fake_mtp(self, **kwargs):
            captured.update(kwargs)
            return [kwargs["hidden_states"].mean(dim=2)]

        model.model.forward = types.MethodType(fake_backbone, model.model)
        model.mtp.forward = types.MethodType(fake_mtp, model.mtp)

        activation = torch.zeros(1, 4, cfg.hc_mult, cfg.hidden_size)
        mtp_embed = torch.randn(1, 4, cfg.hidden_size)
        out = model(activation, mtp_embed)

        assert isinstance(out, tuple)
        assert len(out) == 2
        torch.testing.assert_close(captured["embed_inputs"][0], mtp_embed)
        assert "input_ids" not in captured
        assert "embed_fn" not in captured
        assert out[1].shape == (1, 4, cfg.hidden_size)

    @_REQUIRES_CUDA
    def test_forward_eval_no_mtp_output(self):
        """In eval mode, mtp_per_depth_h should be None."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        model.eval()

        bsz, seq = 1, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
        with torch.no_grad():
            out = model(input_ids)

        assert out.mtp_per_depth_h is None
        assert isinstance(out.logits, torch.Tensor)
        assert out.logits.shape == (bsz, seq, cfg.vocab_size)

    @_REQUIRES_CUDA
    def test_forward_train_mtp_output(self):
        """In train mode, mtp_per_depth_h is list of length 1, each [B, S, hidden]."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        model.train()

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
        out = model(input_ids)

        assert out.mtp_per_depth_h is not None
        assert len(out.mtp_per_depth_h) == 1  # 1 MTP depth
        h = out.mtp_per_depth_h[0]
        assert h.shape == (bsz, seq, cfg.hidden_size), f"unexpected shape: {h.shape}"

    @_REQUIRES_CUDA
    def test_mtp_gradient_backprop(self):
        """MTP hidden states are differentiable; e_proj gradient is non-None."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        model.train()

        bsz, seq = 1, 4
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
        out = model(input_ids)

        # Backward through MTP head only.
        out.mtp_per_depth_h[0].sum().backward()

        e_proj_weight = model.mtp.layers[0].e_proj.weight
        assert e_proj_weight.grad is not None, "e_proj.weight.grad is None after backward"

    @_REQUIRES_CUDA
    def test_logits_is_tensor(self):
        """out.logits must be a raw torch.Tensor, not wrapped."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        model.eval()

        input_ids = torch.randint(0, cfg.vocab_size, (1, 4))
        with torch.no_grad():
            out = model(input_ids)

        assert isinstance(out.logits, torch.Tensor), f"expected Tensor, got {type(out.logits)}"


class TestMTPStateDict:
    def test_state_dict_has_mtp_keys(self):
        """mtp.layers.0.input_layernorm.weight should be in model.state_dict()."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        sd = model.state_dict()
        assert "mtp.layers.0.input_layernorm.weight" in sd, (
            f"MTP key not found; state_dict keys with 'mtp': {[k for k in sd if 'mtp' in k]}"
        )

    def test_rotary_not_in_state_dict(self):
        """_rotary_emb stored on sublayer must NOT appear in state_dict."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        # The sublayer stores rotary refs via object.__setattr__ to avoid registration.
        sublayer = model.mtp.layers[0]
        param_names = dict(sublayer.named_parameters())
        assert "_rotary_emb" not in param_names, "_rotary_emb should not be a registered parameter"
        assert "_rotary_emb_compress" not in param_names, "_rotary_emb_compress should not be a registered parameter"

    def test_rotary_not_in_model_state_dict(self):
        """MTP rotary refs should not pollute the top-level model state dict."""
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        sd = model.state_dict()
        mtp_rotary_keys = [k for k in sd if "mtp" in k and "rotary" in k]
        assert not mtp_rotary_keys, f"Unexpected MTP rotary keys in state_dict: {mtp_rotary_keys}"


class TestPipelineHooks:
    def test_customize_pipeline_stage_modules_keeps_dsv4_dependencies(self):
        cfg = _tiny_config(num_nextn_predict_layers=1)
        model = _make_model(cfg)
        stages = [
            ["model.embed_tokens", "model.layers.0", "model.rotary_emb"],
            ["model.layers.1", "model.norm", "lm_head", "model.rotary_emb"],
        ]

        out = model.customize_pipeline_stage_modules(stages, layers_prefix="model.", text_model=model.model)

        for stage_modules in out:
            assert "model.rotary_emb_compress" in stage_modules
        assert "model.hc_head" not in out[0]
        assert "model.hc_head" in out[-1]
        assert "mtp" not in out[0]
        assert "mtp" in out[-1]

    def test_pipeline_stage_metas_for_mtp_first_middle_and_final_stages(self):
        cfg = _tiny_config(num_nextn_predict_layers=2)

        first = _make_model(cfg)
        first.lm_head = None
        first.model.norm = None
        first.mtp = None
        first_inputs, first_outputs = first.get_pipeline_stage_metas(
            is_first=True, microbatch_size=2, seq_len=16, dtype=torch.float16
        )
        assert first_inputs[0].shape == (2, 16)
        assert first_inputs[0].dtype == torch.long
        assert len(first_outputs) == 3
        assert first_outputs[0].shape == (2, 16, cfg.hc_mult, cfg.hidden_size)
        assert first_outputs[1].shape == (2, 16, cfg.hidden_size)

        middle = _make_model(cfg)
        middle.model.embed_tokens = None
        middle.lm_head = None
        middle.model.norm = None
        middle.mtp = None
        middle_inputs, middle_outputs = middle.get_pipeline_stage_metas(
            is_first=False, microbatch_size=2, seq_len=16, dtype=torch.float16
        )
        assert len(middle_inputs) == 3
        assert middle_inputs[0].shape == (2, 16, cfg.hc_mult, cfg.hidden_size)
        assert middle_inputs[1].shape == (2, 16, cfg.hidden_size)
        assert len(middle_outputs) == 3
        assert middle_outputs[0].shape == (2, 16, cfg.hc_mult, cfg.hidden_size)

        final = _make_model(cfg)
        final.model.embed_tokens = None
        final_inputs, final_outputs = final.get_pipeline_stage_metas(
            is_first=False, microbatch_size=2, seq_len=16, dtype=torch.float16
        )
        assert len(final_inputs) == 3
        assert final_inputs[0].shape == (2, 16, cfg.hc_mult, cfg.hidden_size)
        assert len(final_outputs) == 3
        assert final_outputs[0].shape == (2, 16, cfg.vocab_size)
        assert final_outputs[1].shape == (2, 16, cfg.hidden_size)
        assert final_outputs[2].shape == (2, 16, cfg.hidden_size)


if __name__ == "__main__":
    import sys

    suite = [
        ("MTPConfig disabled", TestMTPConfig().test_mtp_config_disabled),
        ("MTPConfig enabled", TestMTPConfig().test_mtp_config_enabled),
        ("Model no MTP by default", TestModelMTPConstruction().test_model_has_no_mtp_by_default),
        ("Model has MTP when configured", TestModelMTPConstruction().test_model_has_mtp_when_configured),
        ("State dict has MTP keys", TestMTPStateDict().test_state_dict_has_mtp_keys),
        ("Rotary not in state dict", TestMTPStateDict().test_rotary_not_in_state_dict),
        ("Rotary not in model state dict", TestMTPStateDict().test_rotary_not_in_model_state_dict),
    ]

    if torch.cuda.is_available():
        fwd = TestMTPForward()
        suite += [
            ("Forward eval no MTP output", fwd.test_forward_eval_no_mtp_output),
            ("Forward train MTP output", fwd.test_forward_train_mtp_output),
            ("MTP gradient backprop", fwd.test_mtp_gradient_backprop),
            ("Logits is tensor", fwd.test_logits_is_tensor),
        ]

    failed = []
    for name, fn in suite:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed.append(name)

    print()
    if failed:
        print(f"FAILED: {len(failed)}/{len(suite)} tests")
        sys.exit(1)
    else:
        print(f"All {len(suite)} tests passed.")
