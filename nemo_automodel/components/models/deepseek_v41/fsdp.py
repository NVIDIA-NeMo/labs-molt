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

"""FSDP policy for V4.1's shared V4 vision tower."""

from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, OffloadPolicy, fully_shard

from nemo_automodel.components.models.deepseek_v4.fsdp import fully_shard_deepseek_v4
from nemo_automodel.components.models.deepseek_v4.vision import DeepseekV4VisionBlock, DeepseekV4VisionTransformer


def fully_shard_deepseek_v41(
    module: nn.Module,
    *,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: OffloadPolicy | None = None,
    reshard_after_forward: bool | int | None = None,
    ignored_params: set[nn.Parameter] | None = None,
) -> nn.Module:
    """Reuse V4's vision norm policy while preserving ordinary language FSDP.

    Args:
        module: Module to shard in place; vision towers/blocks retain FP32 norms.
        mesh: Runtime FSDP device mesh.
        mp_policy: Mixed-precision policy for the module's non-norm parameters.
        offload_policy: Parameter/gradient offload policy.
        reshard_after_forward: PyTorch FSDP resharding setting.
        ignored_params: Parameter tensors of arbitrary shapes already managed
            outside this FSDP unit, retaining their existing DTensor placements.

    Returns:
        The input module with FSDP applied.
    """
    wrapped = getattr(module, "_checkpoint_wrapped_module", module)
    shard = (
        fully_shard_deepseek_v4
        if isinstance(wrapped, (DeepseekV4VisionTransformer, DeepseekV4VisionBlock))
        else fully_shard
    )
    return shard(
        module,
        mesh=mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=reshard_after_forward,
        ignored_params=ignored_params,
    )
