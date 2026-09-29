# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Dense Nemotron-H configs must route to the custom nemotron_v3 model.

See https://github.com/NVIDIA-NeMo/Automodel/issues/2004. The custom
NemotronHForCausalLM was originally gated to the MoE ("v3") variant only;
dense Nemotron-H (Nano 4B/9B/12B BF16) has no n_routed_experts and used to
fall back to the HF (force_hf) loader.
"""

from types import SimpleNamespace

from nemo_automodel._transformers.model_init import (
    _is_config_compatible_with_custom_model,
    _resolve_custom_model_cls_for_config,
)


def _dense_cfg():
    # Dense Nemotron-H: hybrid layer pattern present, no expert fields.
    return SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        layers_block_type=["mamba", "attention", "mlp"],
    )


def _moe_cfg():
    return SimpleNamespace(
        architectures=["NemotronHForCausalLM"],
        layers_block_type=["mamba", "moe", "attention"],
        n_routed_experts=8,
    )


class TestNemotronHCompatibilityGate:
    def test_dense_is_compatible(self):
        assert _is_config_compatible_with_custom_model("NemotronHForCausalLM", _dense_cfg()) is True

    def test_moe_is_still_compatible(self):
        assert _is_config_compatible_with_custom_model("NemotronHForCausalLM", _moe_cfg()) is True

    def test_hybrid_override_pattern_only_falls_back(self):
        # A config with only a raw hybrid_override_pattern and no layers_block_type is not
        # something the custom model can build (it never normalizes the pattern), so it
        # should fall back to HF rather than route to the custom model and crash.
        cfg = SimpleNamespace(architectures=["NemotronHForCausalLM"], hybrid_override_pattern="M-M*-")
        assert _is_config_compatible_with_custom_model("NemotronHForCausalLM", cfg) is False

    def test_nemotron_without_hybrid_signal_falls_back(self):
        # Neither expert fields nor a hybrid pattern: not a Nemotron-H we recognize,
        # so it stays on the HF fallback path.
        cfg = SimpleNamespace(architectures=["NemotronHForCausalLM"])
        assert _is_config_compatible_with_custom_model("NemotronHForCausalLM", cfg) is False


class TestNemotronHResolve:
    def test_dense_resolves_to_custom_class(self):
        cls = _resolve_custom_model_cls_for_config(_dense_cfg())
        assert cls is not None
        assert cls.__name__ == "NemotronHForCausalLM"
