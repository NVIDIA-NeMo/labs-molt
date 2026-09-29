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

import logging


logger = logging.getLogger(__name__)


def _should_load_before_shard(
    *,
    tp_size: int,
    ep_size: int,
    dp_shard_size: int = 1,
    pretrained_model_name_or_path: str,
    load_base_model: bool,
    peft_config: object | None,
) -> bool:
    """Decide whether to load the checkpoint before FSDP/TP/EP sharding.

    Load-before-shard is only safe when running single-GPU (no TP, EP, or
    DP sharding) and a checkpoint actually needs loading.
    With any model parallelism the post-shard load path must be used to avoid
    NCCL collective mismatches or key/device inconsistencies.

    PEFT models skip this path and use the post-shard load so that base and
    adapter weights load in the same way as multi-GPU.
    """
    no_tp = tp_size <= 1
    no_ep = ep_size <= 1
    no_dp_shard = dp_shard_size <= 1
    no_peft = peft_config is None
    need_checkpoint_load = bool(pretrained_model_name_or_path and load_base_model)
    result = no_tp and no_ep and no_dp_shard and no_peft and need_checkpoint_load
    logger.debug(
        "[_should_load_before_shard] no_tp={} no_ep={} no_dp_shard={} no_peft={} need_load={} -> {}".format(
            no_tp, no_ep, no_dp_shard, no_peft, need_checkpoint_load, result
        )
    )
    return result


def apply_qwen3_omni_config_patch():
    """Fix Qwen3OmniMoeTalkerCodePredictorConfig accessing use_sliding_window."""
    from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorConfig

    if not hasattr(Qwen3OmniMoeTalkerCodePredictorConfig, "use_sliding_window"):
        Qwen3OmniMoeTalkerCodePredictorConfig.use_sliding_window = False


