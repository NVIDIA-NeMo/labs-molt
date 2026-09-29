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

"""NemotronOmni exposes the wrapped NemotronV3 LM's MTP head to the recipe.

The recipe decides whether to compute the MTP loss from ``model.supports.mtp_enabled``
(reads ``model.mtp_config``) and, under context parallelism, requires
``supports_mtp_cp`` plus ``prepare_mtp_inputs_for_cp``. The omni wrapper delegates
all three to its language model.
"""

from types import SimpleNamespace

from nemo_automodel.components.models.nemotron_omni.model import NemotronOmniForConditionalGeneration


def _bare_omni_with_lm(lm):
    model = NemotronOmniForConditionalGeneration.__new__(NemotronOmniForConditionalGeneration)
    # nn.Module.__setattr__ needs the module machinery; bypass it for a stub LM.
    object.__setattr__(model, "language_model", lm)
    return model


def test_mtp_config_and_cp_prep_delegate_to_language_model():
    calls = []

    def prepare(batch, *, ignore_index=-100):
        calls.append((batch, ignore_index))
        return "prepared"

    lm = SimpleNamespace(mtp_config=SimpleNamespace(enabled=True, num_layers=1), prepare_mtp_inputs_for_cp=prepare)
    model = _bare_omni_with_lm(lm)

    assert model.mtp_config.enabled is True
    assert model.prepare_mtp_inputs_for_cp({"input_ids": None}, ignore_index=-7) == "prepared"
    assert calls == [({"input_ids": None}, -7)]


def test_mtp_config_absent_when_language_model_has_none():
    model = _bare_omni_with_lm(SimpleNamespace())
    assert model.mtp_config is None


def test_declares_mtp_cp_support():
    assert NemotronOmniForConditionalGeneration.ModelCapabilities().supports_mtp_cp is True
