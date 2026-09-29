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

"""Tests for the native dense MuseGlimmer vision-language model."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.muse_glimmer.config import MuseGlimmerConfig
from nemo_automodel.components.models.muse_glimmer.model import (
    MuseGlimmerAttention,
    MuseGlimmerFinalRMSNorm,
    MuseGlimmerForConditionalGeneration,
    MuseGlimmerRMSNorm,
    MuseGlimmerRotaryEmbedding,
    MuseGlimmerScalelessRMSNorm,
    apply_rotary_emb,
)
from nemo_automodel.components.models.muse_glimmer.state_dict_adapter import MuseGlimmerStateDictAdapter


def _tiny_config(*, has_vision: bool = False, **overrides) -> MuseGlimmerConfig:
    values = {
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "intermediate_size": 64,
        "vocab_size": 128,
        "max_position_embeddings": 128,
        "qk_scale_factor": 4.0,
        "sliding_window": 64,
        "has_vision": has_vision,
        "vision_latent_dim": 16,
        "vision_output_dim": 64,
        "vision_layers": 2,
        "vision_heads": 2,
        "vision_mlp_ratio": 2.0,
        "vision_patch_size": 2,
        "vision_patch_temporal": 2,
        "vision_downsample_factor": 2,
        "vision_sparse_attention_factor": 2,
        "vision_pos_emb_grid_h": 4,
        "vision_pos_emb_grid_w": 4,
        "vision_adapter_dim": 24,
        "patch_token_id": 120,
        "video_token_id": 121,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }
    values.update(overrides)
    return MuseGlimmerConfig(**values)


def test_registered_config_and_model_resolve_without_remote_code(tmp_path):
    config = _tiny_config()
    payload = config.to_dict()
    payload["auto_map"] = {
        "AutoConfig": "configuration_muse_glimmer.MuseGlimmerConfig",
        "AutoModelForImageTextToText": "modeling_muse_glimmer.MuseGlimmerForConditionalGeneration",
    }
    (tmp_path / "config.json").write_text(json.dumps(payload))

    resolved = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)

    assert type(resolved) is MuseGlimmerConfig
    assert (
        ModelRegistry.resolve_custom_model_cls("MuseGlimmerForConditionalGeneration", resolved)
        is MuseGlimmerForConditionalGeneration
    )


def test_canonical_nested_config_and_model_resolve_without_remote_code(tmp_path):
    payload = {
        "architectures": ["MuseGlimmerForConditionalGeneration"],
        "model_type": "muse_glimmer",
        "image_token_id": 120,
        "video_token_id": 121,
        "out_hidden_size": 64,
        "projector_hidden_size": 24,
        "text_config": {
            "model_type": "muse_glimmer_text",
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 64,
            "vocab_size": 128,
            "max_position_embeddings": 1024,
            "qk_scale_factor": 3.87,
            "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0.0],
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
        },
        "vision_config": {
            "model_type": "muse_glimmer_vision",
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "patch_size": 2,
            "patch_temporal": 2,
            "merge_size": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(payload))

    resolved = AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)

    assert type(resolved) is MuseGlimmerConfig
    assert resolved.max_position_embeddings == 1024
    assert resolved.patch_token_id == 120
    assert resolved.scale_query_by == 3.87
    assert resolved.no_rope_layers == [1, 1, 1, 0]
    assert resolved.vision_latent_dim == 16
    assert resolved.vision_config.layer_types == ["window_attention", "full_attention"]
    assert (
        ModelRegistry.resolve_custom_model_cls("MuseGlimmerForConditionalGeneration", resolved)
        is MuseGlimmerForConditionalGeneration
    )


def test_text_forward_loss_and_external_loss_hidden_states():
    torch.manual_seed(7)
    config = _tiny_config()
    model = MuseGlimmerForConditionalGeneration(config, backend=BackendConfig(attn="sdpa"))
    model.model.norm.weight.data.fill_(1.0)
    input_ids = torch.randint(0, config.vocab_size, (2, 12))
    labels = input_ids.clone()

    output = model(input_ids=input_ids, labels=labels)
    backbone_hidden = model.model(input_ids=input_ids).last_hidden_state
    expected_loss = torch.nn.functional.cross_entropy(
        output.logits[:, :-1].reshape(-1, config.vocab_size),
        labels[:, 1:].reshape(-1),
    )

    assert output.logits.shape == (2, 12, config.vocab_size)
    assert torch.isfinite(output.logits).all()
    assert output.logits.abs().max() <= config.output_soft_cap_temp
    torch.testing.assert_close(output.loss, expected_loss)
    torch.testing.assert_close(output.hidden_states, backbone_hidden * config.output_multiplier)


def test_text_logits_match_canonical_transformers_bfloat16_order_exactly():
    torch.manual_seed(13)
    config = _tiny_config()
    model = MuseGlimmerForConditionalGeneration(config, backend=BackendConfig(attn="sdpa")).to(torch.bfloat16).eval()
    model.model.norm.weight.data.fill_(1.0)
    input_ids = torch.randint(0, config.vocab_size, (1, 8))

    with torch.no_grad():
        hidden_states = model.model(input_ids=input_ids, use_cache=False).last_hidden_state
        reference = model.lm_head(hidden_states)
        reference = reference * config.output_multiplier
        reference = reference / config.output_soft_cap_temp
        reference = torch.tanh(reference)
        reference = reference * config.output_soft_cap_temp
        output = model(input_ids=input_ids, use_cache=False).logits

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_text_norms_match_canonical_transformers_math():
    torch.manual_seed(17)
    x = torch.randn(2, 3, 8, dtype=torch.bfloat16)

    centered = MuseGlimmerRMSNorm(8, eps=1e-6)
    centered.weight.data.copy_(torch.randn(8))
    centered_reference = x.float()
    centered_reference = centered_reference * torch.rsqrt(
        centered_reference.pow(2).mean(-1, keepdim=True) + centered.eps
    )
    centered_reference = centered_reference * (1.0 + centered.weight.float())
    torch.testing.assert_close(centered(x), centered_reference.type_as(x), rtol=0, atol=0)

    final = MuseGlimmerFinalRMSNorm(8, eps=1e-6)
    final.weight.data.copy_(torch.randn(8))
    final_reference = x.float()
    final_reference = final_reference * torch.pow(final_reference.pow(2).mean(-1, keepdim=True) + final.eps, -0.5)
    final_reference = final_reference * final.weight.float()
    torch.testing.assert_close(final(x), final_reference.type_as(x), rtol=0, atol=0)

    scaleless = MuseGlimmerScalelessRMSNorm(8, eps=1e-6)
    scaleless_reference = x.float()
    scaleless_reference = scaleless_reference * torch.pow(
        scaleless_reference.pow(2).mean(-1, keepdim=True) + scaleless.eps,
        -0.5,
    )
    torch.testing.assert_close(scaleless(x), scaleless_reference.type_as(x), rtol=0, atol=0)


def test_text_rotary_matches_canonical_split_half_layout_for_bshd_and_thd():
    torch.manual_seed(19)
    rotary = MuseGlimmerRotaryEmbedding(dim=8, max_position_embeddings=16, theta=10_000.0)
    positions = torch.arange(4).unsqueeze(0)
    bshd = torch.randn(1, 4, 2, 8, dtype=torch.bfloat16)
    cos, sin = rotary(bshd, positions)
    rotated = torch.cat((-bshd[..., 4:], bshd[..., :4]), dim=-1)
    reference = bshd * cos.unsqueeze(2) + rotated * sin.unsqueeze(2)

    torch.testing.assert_close(apply_rotary_emb(bshd, (cos, sin)), reference, rtol=0, atol=0)

    thd = bshd.squeeze(0)
    thd_cos, thd_sin = rotary(thd, positions.squeeze(0))
    thd_rotated = torch.cat((-thd[..., 4:], thd[..., :4]), dim=-1)
    thd_reference = thd * thd_cos.squeeze(0).unsqueeze(1) + thd_rotated * thd_sin.squeeze(0).unsqueeze(1)
    torch.testing.assert_close(apply_rotary_emb(thd, (thd_cos, thd_sin)), thd_reference, rtol=0, atol=0)


def test_cached_text_forward_uses_current_transformers_mask_api():
    model = MuseGlimmerForConditionalGeneration(_tiny_config(), backend=BackendConfig(attn="sdpa"))

    first = model(input_ids=torch.ones(1, 4, dtype=torch.long), use_cache=True)
    second = model(
        input_ids=torch.ones(1, 1, dtype=torch.long),
        past_key_values=first.past_key_values,
        use_cache=True,
    )

    assert first.logits.shape[:2] == (1, 4)
    assert second.logits.shape[:2] == (1, 1)
    assert second.past_key_values.get_seq_length() == 5


def test_complete_vision_path_splices_one_feature_per_placeholder():
    torch.manual_seed(11)
    config = _tiny_config(has_vision=True)
    model = MuseGlimmerForConditionalGeneration(config, backend=BackendConfig(attn="sdpa"))
    model.model.norm.weight.data.fill_(1.0)
    input_ids = torch.tensor([[3, 120, 120, 120, 120, 4, 5, 6]])
    pixel_values = torch.randn(16, 24)
    image_grid_thw = torch.tensor([[1, 4, 4]])

    output = model(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    assert output.logits.shape == (1, 8, config.vocab_size)
    assert torch.isfinite(output.logits).all()


def test_parameter_tree_matches_hf_checkpoint_names():
    model = MuseGlimmerForConditionalGeneration(_tiny_config(has_vision=True), backend=BackendConfig(attn="sdpa"))
    keys = set(model.state_dict())

    assert {
        "model.embed_tokens.weight",
        "model.vision_encoder.conv1_linear.weight",
        "model.vision_encoder.transformer.0.attn.q_proj.weight",
        "model.vision_adapter.c_fc.weight",
        "model.vision_projection.weight",
        "model.rotary_emb.freqs",
        "model.layers.0.self_attn.output_gate_proj.weight",
        "model.layers.0.post_attn_norm.weight",
        "model.layers.0.post_ffn_norm.weight",
        "model.norm.weight",
        "lm_head.weight",
    } <= keys


def test_state_dict_adapter_exports_canonical_keys_and_filters_extra_state():
    adapter = MuseGlimmerStateDictAdapter(_tiny_config())
    state = {
        "model.embed_tokens.weight": torch.randn(3, 4),
        "model.layers.0.self_attn.attn_module._extra_state": torch.tensor(0),
    }

    converted = adapter.from_hf(state)
    exported = adapter.to_hf(converted, exclude_key_regex=r".*_extra_state.*")

    assert isinstance(adapter, StateDictAdapter)
    assert converted == state
    assert set(exported) == {"model.language_model.embed_tokens.weight"}
    assert exported["model.language_model.embed_tokens.weight"] is state["model.embed_tokens.weight"]
    assert adapter.convert_single_tensor_to_hf("model.embed_tokens.weight", state["model.embed_tokens.weight"]) == [
        ("model.language_model.embed_tokens.weight", state["model.embed_tokens.weight"])
    ]
    assert (
        adapter.convert_single_tensor_to_hf(
            "model.layers.0.self_attn.attn_module._extra_state",
            state["model.layers.0.self_attn.attn_module._extra_state"],
            exclude_key_regex=r".*_extra_state.*",
        )
        == []
    )


def test_canonical_state_dict_adapter_round_trip_is_exact():
    config = MuseGlimmerConfig(
        text_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 64,
            "vocab_size": 128,
            "layer_rope_theta": [500000.0],
            "layer_types": ["sliding_attention"],
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "patch_size": 2,
            "patch_temporal": 2,
            "merge_size": 2,
            "pos_emb_height": 4,
            "pos_emb_width": 4,
        },
        out_hidden_size=64,
        projector_hidden_size=24,
    )
    adapter = MuseGlimmerStateDictAdapter(config)
    state = {
        "lm_head.weight": torch.randn(3, 4),
        "model.language_model.embed_tokens.weight": torch.randn(3, 4),
        "model.language_model.layers.0.post_attention_layernorm.weight": torch.randn(4),
        "model.language_model.layers.0.pre_feedforward_layernorm.weight": torch.randn(4),
        "model.language_model.layers.0.post_feedforward_layernorm.weight": torch.randn(4),
        "model.language_model.layers.0.self_attn.gate_proj.weight": torch.randn(4, 4),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(8, 4),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(4, 4),
        "model.vision_tower.patch_embedder.patch_embedding.weight": torch.randn(4, 4),
        "model.vision_tower.patch_embedder.position_embedding_table.weight": torch.randn(4, 4),
        "model.vision_tower.layers.0.attn.proj.weight": torch.randn(4, 4),
        "model.vision_tower.layers.0.mlp.fc1.weight": torch.randn(4, 4),
        "model.vision_tower.layers.0.mlp.fc2.weight": torch.randn(4, 4),
        "model.vision_tower.layers.0.norm1.weight": torch.randn(4),
        "model.vision_tower.layers.0.norm2.weight": torch.randn(4),
        "model.vision_adapter.fc1.weight": torch.randn(4, 4),
        "model.vision_adapter.fc2.weight": torch.randn(4, 4),
        "model.vision_projection.weight": torch.randn(4, 4),
    }

    native = adapter.from_hf(state)
    round_trip = adapter.to_hf(native)

    assert set(round_trip) == set(state)
    assert "model.rotary_emb.freqs" in native
    assert "model.vision_encoder.rotary_emb.inv_freq" in native
    assert adapter.convert_single_tensor_to_hf(
        "model.layers.0.post_attn_norm.weight", native["model.layers.0.post_attn_norm.weight"]
    ) == [
        (
            "model.language_model.layers.0.post_attention_layernorm.weight",
            native["model.layers.0.post_attn_norm.weight"],
        )
    ]
    assert adapter.convert_single_tensor_to_hf("model.rotary_emb.freqs", native["model.rotary_emb.freqs"]) == []
    assert (
        adapter.convert_single_tensor_to_hf(
            "model.vision_encoder.rotary_emb.inv_freq",
            native["model.vision_encoder.rotary_emb.inv_freq"],
        )
        == []
    )
    assert (
        native["model.layers.0.post_attn_norm.weight"]
        is state["model.language_model.layers.0.post_attention_layernorm.weight"]
    )
    for key, value in state.items():
        assert round_trip[key] is value


def test_legacy_and_canonical_configs_use_the_same_effective_query_scale():
    legacy = _tiny_config(qk_scale_factor=3.87 * (8**0.5))
    canonical = MuseGlimmerConfig(
        text_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 64,
            "vocab_size": 128,
            "qk_scale_factor": 3.87,
            "layer_rope_theta": [500000.0],
            "layer_types": ["sliding_attention"],
        }
    )

    assert legacy.scale_query_by == pytest.approx(3.87)
    assert canonical.scale_query_by == pytest.approx(3.87)


def test_cp_preparation_selects_aux_only_bshd_and_framework_thd_sharders():
    model = MuseGlimmerForConditionalGeneration(_tiny_config(has_vision=True), backend=BackendConfig(attn="sdpa"))

    bshd = model.prepare_model_inputs_for_cp({"input_ids": torch.ones(1, 8, dtype=torch.long)})
    thd = model.prepare_model_inputs_for_cp({"qkv_format": "thd", "seq_lens": torch.tensor([[8]])})
    packed_vlm_ids = torch.tensor([[3, 120, 120, 4]])
    packed_vlm = model.prepare_model_inputs_for_cp(
        {
            "qkv_format": "thd",
            "seq_lens": torch.tensor([[4]]),
            "input_ids": packed_vlm_ids,
            "pixel_values": torch.randn(16, 24),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
        }
    )

    assert set(bshd) == {"cp_sharder"}
    assert bshd["cp_sharder"].shard_batch.__name__ == "shard_batch_aux_only"
    assert thd == {"_muse_glimmer_thd_max_real_seqlen": 8}
    assert packed_vlm["_muse_glimmer_thd_max_real_seqlen"] == 4
    torch.testing.assert_close(
        packed_vlm["_muse_glimmer_global_vision_mask"],
        torch.tensor([[False, True, True, False]]),
    )


def test_packed_thd_requires_documents_within_muse_glimmer_sliding_window():
    model = MuseGlimmerForConditionalGeneration(_tiny_config(), backend=BackendConfig(attn="sdpa"))
    model.cp_mesh = SimpleNamespace(size=lambda: 2)
    transports = []
    model._set_te_cp_transport = transports.append

    model.prepare_model_inputs_for_cp({"qkv_format": "thd", "seq_lens": torch.tensor([[64, 32]])})
    with pytest.raises(ValueError, match="sliding window.*64 tokens"):
        model.prepare_model_inputs_for_cp({"qkv_format": "thd", "seq_lens": torch.tensor([[65, 32]])})

    assert transports == ["p2p"]


@pytest.mark.parametrize(
    ("local_indices", "feature_indices"),
    [
        ([6, 0, 4, 2], [3, 2, 1]),
        ([6, 4, 2, 1], [3, 2, 1, 0]),
    ],
)
def test_packed_vlm_splices_local_features_using_te_partition_indices(local_indices, feature_indices):
    config = _tiny_config(has_vision=True)
    model = MuseGlimmerForConditionalGeneration(config, backend=BackendConfig(attn="sdpa"))
    features = torch.arange(4 * config.hidden_size, dtype=torch.float32).view(4, config.hidden_size)

    class _Vision(torch.nn.Module):
        def forward(self, _pixel_values, _grid_thw):
            return features

    model.model.vision_encoder = _Vision()
    model.model.vision_adapter = torch.nn.Identity()
    model.model.vision_projection = torch.nn.Identity()
    model.model.perception_emb_norm = None

    global_ids = torch.tensor([3, 120, 120, 4, 120, 5, 120, 6])
    local_indices = torch.tensor(local_indices)
    local_ids = global_ids.index_select(0, local_indices)
    hidden = torch.zeros(local_ids.shape[0], config.hidden_size)

    actual = model.model._embed_vision(
        local_ids,
        hidden,
        torch.randn(16, 24),
        torch.tensor([[1, 4, 4]]),
        None,
        None,
        None,
        global_vision_mask=global_ids == config.patch_token_id,
        thd_local_indices=local_indices,
    )

    expected = hidden.clone()
    expected[local_ids == config.patch_token_id] = features[torch.tensor(feature_indices)]
    torch.testing.assert_close(actual, expected)


def test_te_attention_is_constructed_natively(monkeypatch):
    calls = {}
    sentinel = torch.nn.Identity()

    def _fake_initialize(**kwargs):
        calls.update(kwargs)
        return sentinel, sentinel.forward

    monkeypatch.setattr(
        "nemo_automodel.components.models.muse_glimmer.model.initialize_attn_module_and_func",
        _fake_initialize,
    )
    config = _tiny_config()

    attention = MuseGlimmerAttention(config, layer_idx=0, backend=BackendConfig(attn="te"))

    assert attention.attn_module is sentinel
    assert calls["attn_impl"] == "te"
    assert calls["num_attention_heads"] == config.num_attention_heads
    assert calls["num_gqa_groups"] == config.num_key_value_heads
    assert calls["num_qk_channels"] == config.head_dim


@pytest.mark.parametrize(
    "tp_size",
    [1, 2],
)
def test_parallel_strategy_accepts_supported_tp_sizes(monkeypatch, tp_size):
    from nemo_automodel.components.distributed.parallelizer import (
        PARALLELIZATION_STRATEGIES,
        DefaultParallelizationStrategy,
    )

    class _SubMesh:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

    class _Mesh:
        mesh_dim_names = ("tp", "cp")

        def __getitem__(self, name):
            return _SubMesh(tp_size if name == "tp" else 1)

    monkeypatch.setattr(
        DefaultParallelizationStrategy,
        "parallelize",
        lambda _self, model, _device_mesh, **_kwargs: model,
    )
    model = MuseGlimmerForConditionalGeneration(
        _tiny_config(num_attention_heads=8),
        backend=BackendConfig(attn="sdpa"),
    )

    PARALLELIZATION_STRATEGIES["MuseGlimmerForConditionalGeneration"].parallelize(model, _Mesh())

    assert all(not hasattr(layer.self_attn, "te_tp_replicated") for layer in model.model.layers)


@pytest.mark.parametrize("tp_size", [4, 8])
def test_parallel_strategy_rejects_tp_above_kv_head_count(monkeypatch, tp_size):
    from nemo_automodel.components.distributed.parallelizer import (
        PARALLELIZATION_STRATEGIES,
        DefaultParallelizationStrategy,
    )

    class _SubMesh:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

    class _Mesh:
        mesh_dim_names = ("tp", "cp")

        def __getitem__(self, name):
            return _SubMesh(tp_size if name == "tp" else 1)

    monkeypatch.setattr(
        DefaultParallelizationStrategy,
        "parallelize",
        lambda _self, model, _device_mesh, **_kwargs: model,
    )
    model = MuseGlimmerForConditionalGeneration(
        _tiny_config(num_attention_heads=8),
        backend=BackendConfig(attn="sdpa"),
    )

    with pytest.raises(ValueError, match=r"supports TP1 or TP2.*tp_size=[48]"):
        PARALLELIZATION_STRATEGIES["MuseGlimmerForConditionalGeneration"].parallelize(model, _Mesh())
