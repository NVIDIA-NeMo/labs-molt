# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Signature contract for the sharder-only ``prepare_model_inputs_for_cp`` CP hook.

``ContextParallelSharder`` construction invokes each hook-aware model directly as
``model.prepare_model_inputs_for_cp(batch, num_chunks=n)`` (a plain method: the
sharder-only hook touches no weights, so no ``__call__`` / FSDP2 unshard routing).
Every model that answers the CP protocol must therefore expose that method with a
signature that binds ``(instance, batch, num_chunks=n)``.
"""

import inspect

import pytest

PRE_EMBED_MODELS = [
    ("nemo_automodel.components.models.deepseek_v4.model", "DeepseekV4ForCausalLM"),
    ("nemo_automodel.components.models.glm_moe_dsa.model", "GlmMoeDsaForCausalLM"),
    ("nemo_automodel.components.models.gemma4_moe.model", "Gemma4ForConditionalGeneration"),
    ("nemo_automodel.components.models.step3p7.model", "Step3p7ForConditionalGeneration"),
    ("nemo_automodel.components.models.qwen3_5.model", "Qwen3_5ForConditionalGeneration"),
    ("nemo_automodel.components.models.qwen3_5_moe.model", "Qwen3_5MoeForConditionalGeneration"),
    ("nemo_automodel.components.models.minimax_m3_vl.model", "MiniMaxM3SparseForConditionalGeneration"),
    ("nemo_automodel.components.models.nemotron_omni.model", "NemotronOmniForConditionalGeneration"),
]


@pytest.mark.parametrize("module_path,class_name", PRE_EMBED_MODELS)
def test_prepare_model_inputs_for_cp_binds_dispatch_call(module_path, class_name):
    module = pytest.importorskip(module_path)
    cls = getattr(module, class_name)
    hook = getattr(cls, "prepare_model_inputs_for_cp", None)
    assert hook is not None, f"{class_name} must expose prepare_model_inputs_for_cp for the CP dispatch"
    signature = inspect.signature(hook)
    try:
        signature.bind(object(), {}, num_chunks=1)
    except TypeError as err:
        pytest.fail(
            f"{class_name}.prepare_model_inputs_for_cp cannot be called as "
            f"model.prepare_model_inputs_for_cp(batch, num_chunks=n): {err}. "
            "ContextParallelSharder construction calls it exactly that way."
        )
