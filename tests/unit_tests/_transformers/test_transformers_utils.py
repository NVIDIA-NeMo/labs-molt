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


import torch.nn as nn

from nemo_automodel._transformers.utils import apply_qwen3_omni_config_patch


class TestApplyQwen3OmniConfigPatch:
    """Test cases for apply_qwen3_omni_config_patch function."""

    def test_patch_sets_use_sliding_window_default(self):
        """Verify the patch adds use_sliding_window=False to the config class."""
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerCodePredictorConfig,
        )

        apply_qwen3_omni_config_patch()
        assert hasattr(Qwen3OmniMoeTalkerCodePredictorConfig, "use_sliding_window")

    def test_patch_is_idempotent(self):
        """Calling the patch twice does not raise or change the value."""
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerCodePredictorConfig,
        )

        apply_qwen3_omni_config_patch()
        apply_qwen3_omni_config_patch()
        assert Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window is False

    def test_patch_does_not_overwrite_existing_attribute(self):
        """If the attribute already exists (e.g. fixed upstream), patch is a no-op."""
        from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
            Qwen3OmniMoeTalkerCodePredictorConfig,
        )

        original = getattr(Qwen3OmniMoeTalkerCodePredictorConfig, "use_sliding_window", None)
        Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window = True
        try:
            apply_qwen3_omni_config_patch()
            assert Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window is True
        finally:
            if original is None:
                del Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window
            else:
                Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window = original


class _RopeInnerModel(nn.Module):
    """Stand-in for a transformers base model that owns ``get_rope_index``."""

    def get_rope_index(self, input_ids=None, **kwargs):
        return None, None


class _RopeWrapper(nn.Module):
    """Mimics the ``*ForConditionalGeneration`` layout of Qwen-VL models.

    ``get_rope_index`` lives on the base model, and ``base_model`` resolves
    through ``base_model_prefix`` exactly as ``PreTrainedModel`` does.
    """

    base_model_prefix = "model"

    def __init__(self, inner=None):
        super().__init__()
        if inner is not None:
            self.model = inner

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)


class _OmniLike(nn.Module):
    """Mimics Qwen3-Omni: the builder is on a submodule the two-step lookup misses.

    The top-level module owns no ``get_rope_index``, and ``base_model_prefix``
    of ``model`` resolves back to itself because there is no ``model`` child.
    """

    base_model_prefix = "model"

    def __init__(self, talker=None):
        super().__init__()
        self.thinker = _RopeInnerModel()
        if talker is not None:
            self.talker = talker

    @property
    def base_model(self):
        return getattr(self, self.base_model_prefix, self)


