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

from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from nemo_automodel.shared.parameter_names import canonical_parameter_fqn


def test_canonical_parameter_fqns_match_checkpoint_wrapper_state_dict_keys():
    model = nn.Sequential(checkpoint_wrapper(nn.Linear(4, 2)))

    canonical_names = {canonical_parameter_fqn(name) for name, _ in model.named_parameters()}

    assert canonical_names == set(model.state_dict()) == {"0.weight", "0.bias"}


def test_canonical_parameter_fqn_preserves_unwrapped_name():
    name = "model.layers.0.mlp.up_proj.weight"
    assert canonical_parameter_fqn(name) == name


def test_canonical_parameter_fqn_strips_nested_checkpoint_wrappers():
    assert (
        canonical_parameter_fqn(
            "_checkpoint_wrapped_module.layers.0._checkpoint_wrapped_module.self_attn.q_proj.weight"
        )
        == "layers.0.self_attn.q_proj.weight"
    )
