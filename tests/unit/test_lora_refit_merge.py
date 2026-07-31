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

"""LoRA adapters reach vLLM merged into their base weight (`lora_refit_merges`).

vLLM has no adapter parameter and its loader *raises* on an unknown name, so refit pushes
``W + scale * (first @ second)`` under the base FQN. The trap is operand order: a dense
``nn.Linear`` weight is ``[out, in]`` (``lora_B @ lora_A``) while a grouped expert weight is
``[E, in, out]`` (``lora_A @ lora_B``) -- reversing either yields a shape error at best and a
transposed policy at worst. `scale` is per module, so it is read off the module, not recomputed.
"""

import pytest
import torch
import torch.nn as nn

from molt.trainer.fsdp.refit import lora_refit_merges


def _dense_lora(in_features=6, out_features=4, dim=2, alpha=8):
    """A real AutoModel LinearLoRA with a non-zero lora_B (it is zero-initialized)."""
    from nemo_automodel.components._peft.lora import LinearLoRA

    module = LinearLoRA(nn.Linear(in_features, out_features), dim=dim, alpha=alpha)
    with torch.no_grad():
        module.lora_B.weight.normal_()
    return module.eval()  # materialize_effective_weight() refuses training-mode dropout


def _real_experts_lora(dim=8, moe_inter_dim=6, n_experts=4, lora_dim=2, alpha=8):
    """A real GroupedExpertsLoRA plus a factory for the plain GroupedExperts it wraps.

    float64 and the per-expert ``torch`` loop keep the comparison exact. ``lora_B`` ships
    zero-initialized, so the adapters are randomized or the merge delta would be a no-op.
    """
    from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
    from nemo_automodel.components.models.common.utils import BackendConfig
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.experts import GroupedExperts

    config = MoEConfig(
        n_routed_experts=n_experts, n_shared_experts=0, n_activated_experts=2, n_expert_groups=1,
        n_limited_groups=1, train_gate=True, gate_bias_update_factor=0.0, aux_loss_coeff=0.0,
        score_func="softmax", route_scale=1.0, dim=dim, inter_dim=2 * moe_inter_dim,
        moe_inter_dim=moe_inter_dim, norm_topk_prob=True, dtype=torch.float64,
    )  # fmt: skip
    backend = BackendConfig(attn="sdpa", linear="torch", experts="torch", rope_fusion=False)

    experts = GroupedExperts(config, backend=backend)
    with torch.no_grad():
        experts.gate_and_up_projs.normal_(0, 0.1)
        experts.down_projs.normal_(0, 0.1)
    lora = GroupedExpertsLoRA(experts, lora_dim=lora_dim, alpha=alpha)
    with torch.no_grad():
        for adapter in (lora.lora_gate_and_up_A, lora.lora_gate_and_up_B, lora.lora_down_A, lora.lora_down_B):
            adapter.normal_(0, 0.3)
    return lora, lambda: GroupedExperts(config, backend=backend)


def _moe_inputs(config, n_tokens=16):
    """(x, token_mask, weights, indices) for GroupedExperts.forward, every expert routed to."""
    torch.manual_seed(0)
    indices = torch.stack(
        [torch.randperm(config.n_routed_experts)[: config.n_activated_experts] for _ in range(n_tokens)]
    )
    weights = torch.rand(n_tokens, config.n_activated_experts, dtype=torch.float64)
    x = torch.randn(n_tokens, config.dim, dtype=torch.float64)
    return x, torch.ones(n_tokens, dtype=torch.bool), weights / weights.sum(-1, keepdim=True), indices


def _merged(model, key):
    """The weight refit would push for `key`, computed the way broadcast_to_vllm does."""
    state_dict = model.state_dict()
    first, second, scale = lora_refit_merges(model, state_dict)[key]
    return state_dict[key] + scale * (first @ second).to(state_dict[key].dtype)


def test_dense_merge_matches_automodel_effective_weight():
    # AutoModel's own materialize_effective_weight() is the reference; refit cannot call it
    # (it raises under train() + dropout), so the arithmetic here must agree with it exactly.
    module = _dense_lora()
    torch.testing.assert_close(_merged(module, "weight"), module.materialize_effective_weight())


def test_dense_merge_uses_alpha_over_rank_from_the_module():
    # scale is alpha/rank per module; AutoModel's moe_rank_scaling gives experts a different
    # rank than the dense layers, so a globally recomputed scale would be wrong for one of them.
    module = _dense_lora(dim=2, alpha=8)
    _, _, scale = lora_refit_merges(module, module.state_dict())["weight"]
    assert scale == 4.0


def test_expert_merge_matches_the_real_grouped_kernel():
    # The reference is GroupedExpertsLoRA's own forward, not a re-derivation of it: the merged
    # weight in a plain GroupedExperts must produce the same tokens the trainer computes, which
    # pins the batched operand order to lora_A @ lora_B for the [E, in, out] grouped layout.
    lora, plain_experts = _real_experts_lora()
    state_dict = lora.state_dict()

    merged, unmerged = plain_experts(), plain_experts()
    with torch.no_grad():
        for key, (first, second, scale) in lora_refit_merges(lora, state_dict).items():
            getattr(merged, key).copy_(state_dict[key] + scale * (first @ second))
            getattr(unmerged, key).copy_(state_dict[key])

    inputs = _moe_inputs(lora.config)
    torch.testing.assert_close(lora(*inputs), merged(*inputs))
    # Guard against a vacuous pass: without the delta the outputs must differ.
    assert not torch.allclose(lora(*inputs), unmerged(*inputs))


def test_merge_keys_are_base_weights_not_adapters():
    # The map is keyed by what vLLM holds. A `lora_A.weight` key would push an adapter and
    # make the engine's loader raise.
    model = nn.Module()
    model.proj = _dense_lora()
    assert set(lora_refit_merges(model, model.state_dict())) == {"proj.weight"}


def test_adapter_entries_are_exactly_the_lora_prefixed_keys():
    # broadcast_to_vllm skips a state_dict key when any dot-part starts with `lora_`. If an
    # AutoModel rename broke that match, the adapters would be pushed to vLLM as-is.
    keys = set(_dense_lora().state_dict())
    adapters = {k for k in keys if any(part.startswith("lora_") for part in k.split("."))}
    assert adapters == {"lora_A.weight", "lora_B.weight"}
    assert keys - adapters == {"weight", "bias"}  # both are genuine vLLM parameters


def test_absent_base_weight_fails_loud():
    # A module-FQN / state_dict-key divergence is silent otherwise: base weights are frozen
    # under LoRA, so the engine would serve the initial policy for the whole run.
    model = nn.Module()
    model.proj = _dense_lora()
    with pytest.raises(RuntimeError, match="unadapted base policy"):
        lora_refit_merges(model, {"proj.bias": torch.zeros(4)})


def test_full_parameter_model_yields_no_merges():
    # No LoRA -> empty map, so the non-PEFT refit path is untouched.
    assert lora_refit_merges(nn.Linear(4, 4), {}) == {}
