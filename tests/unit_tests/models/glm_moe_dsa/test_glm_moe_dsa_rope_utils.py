# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import pytest

from nemo_automodel.components.models.deepseek_v3.rope_utils import yarn_get_mscale
from nemo_automodel.components.models.glm_moe_dsa.rope_utils import mla_softmax_scale


class TestMlaSoftmaxScale:
    """Tests for the shared MLA softmax scale (base scale + YaRN mscale correction)."""

    @staticmethod
    def _config(rope_parameters, max_position_embeddings=4096):
        return SimpleNamespace(rope_parameters=rope_parameters, max_position_embeddings=max_position_embeddings)

    def test_plain_scale_without_rope_parameters(self):
        assert mla_softmax_scale(self._config(None), 64) == pytest.approx(64**-0.5)

    def test_plain_scale_for_a_default_rope(self):
        config = self._config({"rope_theta": 10000.0, "rope_type": "default"})
        assert mla_softmax_scale(config, 64) == pytest.approx(64**-0.5)

    def test_yarn_spec_applies_mscale_squared(self):
        config = self._config(
            {"factor": 4.0, "mscale": 1.0, "original_max_position_embeddings": 1024},
            max_position_embeddings=4096,
        )
        mscale = yarn_get_mscale(4.0, 1.0)
        assert mla_softmax_scale(config, 64) == pytest.approx(64**-0.5 * mscale * mscale)

    def test_mscale_is_uncorrected_within_the_original_window(self):
        """Inside the original context the raw mscale is used, not the log-scaled one."""
        config = self._config(
            {"factor": 4.0, "mscale": 2.0, "original_max_position_embeddings": 4096},
            max_position_embeddings=4096,
        )
        assert mla_softmax_scale(config, 64) == pytest.approx(64**-0.5 * 4.0)

    def test_falls_back_to_rope_scaling_on_older_configs(self):
        config = SimpleNamespace(
            rope_scaling={"factor": 4.0, "mscale": 1.0, "original_max_position_embeddings": 1024},
            max_position_embeddings=4096,
        )
        mscale = yarn_get_mscale(4.0, 1.0)
        assert mla_softmax_scale(config, 64) == pytest.approx(64**-0.5 * mscale * mscale)
