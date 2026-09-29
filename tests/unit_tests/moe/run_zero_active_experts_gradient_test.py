# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import sys

import torch
import torch.nn.functional as F

from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import GroupedExperts

# Track whether expert_activation was called
activation_called = [False]


def tracking_activation(gate_and_up_out, weights):
    """Tracking version of expert_activation_grouped that sets activation_called[0] = True."""
    global activation_called
    activation_called[0] = True
    gate_out, up_out = torch.chunk(gate_and_up_out, 2, -1)
    inter = F.silu(gate_out) * up_out
    return inter * weights


def main(device_str: str = "cuda:0") -> int:
    """
    Run the zero active experts gradient test.

    Args:
        device_str: Device to run on ("cuda:0" or "cpu")

    Returns:
        0 if test passed, 1 if test failed
    """
    # Use global activation_called to track across function boundaries
    global activation_called
    activation_called[0] = False  # Reset at start

    moe_config = MoEConfig(
        n_routed_experts=8,
        n_shared_experts=2,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.1,
        aux_loss_coeff=0.01,
        score_func="softmax",
        route_scale=1.0,
        dim=128,
        inter_dim=256,
        moe_inter_dim=256,
        norm_topk_prob=False,
        router_bias=False,
        expert_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        dtype=torch.float32,
    )

    device = torch.device(device_str)
    experts = GroupedExperts(moe_config)
    experts.expert_activation_grouped = tracking_activation
    experts = experts.to(device)

    with torch.no_grad():
        experts.gate_and_up_projs.normal_(0, 0.02)
        experts.down_projs.normal_(0, 0.02)

    num_tokens = 8
    x = torch.randn(num_tokens, moe_config.dim, dtype=torch.float32, device=device)
    token_mask = torch.ones(num_tokens, dtype=torch.bool, device=device)
    weights = torch.rand(num_tokens, moe_config.n_activated_experts, dtype=torch.float32, device=device)

    # Set indices to non-existent expert (simulates all tokens routed elsewhere)
    indices = torch.full(
        (num_tokens, moe_config.n_activated_experts),
        fill_value=moe_config.n_routed_experts + 100,
        dtype=torch.long,
        device=device,
    )

    output = experts.forward(x, token_mask, weights, indices)

    if activation_called[0]:
        print("SUCCESS: expert_activation was called even when no tokens select any expert")
        return 0
    else:
        print("FAIL: expert_activation was NOT called - the zero active experts fix is missing or broken")
        return 1


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
    sys.exit(main(device))
