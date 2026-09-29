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

"""Runtime backend configuration owned by Qwen3.8-Flash-Next."""

from dataclasses import dataclass
from typing import Literal

from nemo_automodel.components.models.common import BackendConfig


@dataclass(kw_only=True)
class Qwen3_8_FlashNextBackendConfig(BackendConfig):
    """Extend shared backend choices with this model's optional FA4 QSA.

    ``attn="cute"`` selects FA4 SM90 BF16 sparse GQA; ``attn="flex"`` selects
    FlexAttention. CPU execution uses the numerical oracle with either choice.
    Other values and the environment-dependent default are retained for
    compatibility with existing BackendConfig callers; CUDA QSA requires
    ``"flex"`` or ``"cute"``. All other component settings are inherited.
    """

    attn: Literal["te", "sdpa", "flex", "eager", "tilelang", "cudnn", "cute"] = BackendConfig.attn
