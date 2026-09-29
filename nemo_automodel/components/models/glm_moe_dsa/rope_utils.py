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

"""Rotary helpers shared by the GLM MoE DSA model and its DSpark draft."""

from nemo_automodel.components.models.deepseek_v3.rope_utils import yarn_get_mscale

__all__ = ["mla_softmax_scale"]


def mla_softmax_scale(config, qk_head_dim: int) -> float:
    """MLA attention scale ``qk_head_dim ** -0.5``, with the YaRN ``mscale`` correction.

    When the rotary parameters carry a full YaRN spec (``factor`` / ``mscale`` /
    ``original_max_position_embeddings``) and the context was extended past the original
    window, the scale is multiplied by ``mscale ** 2`` -- the DeepSeek-V3 MLA convention
    that GLM-5.2 inherits. Shared so the DSpark draft, trained on this model's hidden
    states, cannot drift from the target's attention temperature.

    Args:
        config: Model config exposing ``rope_parameters`` (or the older ``rope_scaling``)
            and ``max_position_embeddings``.
        qk_head_dim: Query/key head dimension the scale is derived from.

    Returns:
        The softmax scale to pass to the attention kernel.
    """
    scale = float(qk_head_dim**-0.5)
    rope_parameters = config.rope_parameters if hasattr(config, "rope_parameters") else config.rope_scaling
    if not rope_parameters or not all(
        key in rope_parameters for key in ("factor", "mscale", "original_max_position_embeddings")
    ):
        return scale
    mscale = rope_parameters["mscale"]
    if config.max_position_embeddings > rope_parameters["original_max_position_embeddings"]:
        mscale = yarn_get_mscale(rope_parameters["factor"], mscale)
    return scale * mscale * mscale
