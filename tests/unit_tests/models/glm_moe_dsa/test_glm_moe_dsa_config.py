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

import json

import pytest
from transformers import AutoConfig

from nemo_automodel._transformers.registry import resolve_custom_config_cls
from nemo_automodel.components.models.glm_moe_dsa.config import GlmMoeDsaConfig

# The released GLM-5.2 config.json carries both keys; transformers aliases
# head_dim onto qk_rope_head_dim, so loading it through the built-in config
# overwrites the rope dim with 192.
_RELEASED_FIELDS = {
    "model_type": "glm_moe_dsa",
    "head_dim": 192,
    "qk_rope_head_dim": 64,
    "index_head_dim": 128,
    "num_hidden_layers": 2,
}


def test_registry_resolves_automodel_glm_config():
    assert resolve_custom_config_cls("glm_moe_dsa") is GlmMoeDsaConfig


def test_released_config_keeps_qk_rope_head_dim(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_RELEASED_FIELDS))

    config = AutoConfig.from_pretrained(tmp_path)

    assert type(config) is GlmMoeDsaConfig
    assert config.qk_rope_head_dim == 64
    # The DSA indexer splits Q into [rope, nope]; a clobbered rope dim makes the
    # nope half negative and torch.split raises.
    assert config.index_head_dim - config.qk_rope_head_dim == 64


def test_expert_count_alias_is_preserved():
    config = GlmMoeDsaConfig(n_routed_experts=8)

    assert config.num_local_experts == 8


def test_head_dim_stays_a_plain_field():
    # Without the alias, head_dim is just the checkpoint's own value (the full
    # QK width) and no longer writes through to the rope dim.
    config = GlmMoeDsaConfig(head_dim=192, qk_rope_head_dim=64, qk_nope_head_dim=128)

    assert config.head_dim == 192
    assert config.qk_rope_head_dim == 64
    assert config.qk_head_dim == 192


# Every config field Automodel's GLM DSA modeling code reads. The config is
# standalone, so this list is the field protocol the model depends on; a missing
# entry means the model breaks at init rather than at load.
_FIELDS_READ_BY_MODEL = (
    "hidden_size",
    "index_head_dim",
    "index_n_heads",
    "index_topk",
    "indexer_types",
    "intermediate_size",
    "kv_lora_rank",
    "max_position_embeddings",
    "moe_intermediate_size",
    "mlp_layer_types",
    "n_group",
    "n_routed_experts",
    "n_shared_experts",
    "norm_topk_prob",
    "num_attention_heads",
    "num_experts_per_tok",
    "num_hidden_layers",
    "q_lora_rank",
    "qk_head_dim",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "rms_norm_eps",
    "rope_parameters",
    "routed_scaling_factor",
    "topk_group",
    "v_head_dim",
    "vocab_size",
)


@pytest.mark.parametrize("field", _FIELDS_READ_BY_MODEL)
def test_field_protocol_is_complete(field):
    assert hasattr(GlmMoeDsaConfig(), field), f"GLM DSA modeling reads config.{field}"


def test_per_layer_patterns_are_derived_when_absent():
    config = GlmMoeDsaConfig(num_hidden_layers=6, index_topk_freq=2)

    assert config.mlp_layer_types == ["dense"] * 3 + ["sparse"] * 3
    # Layer 0 and then every 2nd layer are full; the rest share.
    assert config.indexer_types == ["full", "full", "shared", "full", "shared", "full"]
