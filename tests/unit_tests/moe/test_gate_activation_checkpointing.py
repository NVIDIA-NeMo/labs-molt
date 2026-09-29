# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU regression coverage for MoE gate accounting under activation checkpointing."""

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import set_checkpoint_early_stop

from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import Gate, MoE
from nemo_automodel.components.moe.parallelizer import apply_ac


class _TinyMoEBlock(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        backend = BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        )
        self.mlp = MoE(config, backend)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the tiny MoE block.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return self.mlp(hidden_states)


class _TinyMoEDecoder(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.layers = nn.ModuleDict({"0": _TinyMoEBlock(config)})
        self.moe_config = config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the decoder block.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return self.layers["0"](hidden_states)


class _TinyMoEModel(nn.Module):
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.model = _TinyMoEDecoder(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the tiny MoE model.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return self.model(hidden_states)


def _tiny_moe_config() -> MoEConfig:
    return MoEConfig(
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=1e-3,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=8,
        inter_dim=16,
        moe_inter_dim=16,
        norm_topk_prob=False,
        dtype=torch.float32,
    )


def _get_gate(model: _TinyMoEModel) -> Gate:
    block = unwrap_checkpoint_wrapper(model.model.layers["0"])
    assert isinstance(block.mlp.gate, Gate)
    return block.mlp.gate


def _forward_backward(
    model: _TinyMoEModel,
    hidden_states: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Run one forward/backward step and capture numerical results.

    Args:
        model: Tiny MoE model under test.
        hidden_states: Tensor of shape [batch, sequence, hidden].

    Returns:
        Output tensor of shape [batch, sequence, hidden], input gradient tensor
        of the same shape, parameter gradients keyed by parameter name, and
        cumulative expert-load tensors of shape [experts] captured after the
        forward and backward passes.
    """
    with set_checkpoint_early_stop(False):
        output = model(hidden_states)
    load_after_forward = _get_gate(model)._cumulative_expert_load
    assert load_after_forward is not None
    load_after_forward = load_after_forward.clone()

    output.float().square().mean().backward()
    assert hidden_states.grad is not None
    parameter_grads = {
        name.replace("._checkpoint_wrapped_module", ""): parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    load_after_backward = _get_gate(model)._cumulative_expert_load
    assert load_after_backward is not None
    return (
        output.detach(),
        hidden_states.grad.detach().clone(),
        parameter_grads,
        load_after_forward,
        load_after_backward.clone(),
    )


def _assert_gate_load_and_gradient_parity(device: torch.device, checkpoint_kwargs: dict[str, bool]) -> None:
    torch.manual_seed(1234)
    reference_model = _TinyMoEModel(_tiny_moe_config()).to(device).train()
    with torch.no_grad():
        for parameter in reference_model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    checkpointed_model = copy.deepcopy(reference_model)
    apply_ac(checkpointed_model, hidden_size=8, num_experts=4, **checkpoint_kwargs)

    inputs = torch.randn(2, 3, 8, device=device)
    reference = _forward_backward(reference_model, inputs.clone().requires_grad_(True))
    checkpointed = _forward_backward(checkpointed_model, inputs.clone().requires_grad_(True))

    (
        reference_output,
        reference_input_grad,
        reference_parameter_grads,
        reference_forward_load,
        reference_backward_load,
    ) = reference
    (
        checkpointed_output,
        checkpointed_input_grad,
        checkpointed_parameter_grads,
        checkpointed_forward_load,
        checkpointed_backward_load,
    ) = checkpointed

    torch.testing.assert_close(checkpointed_output, reference_output)
    torch.testing.assert_close(checkpointed_input_grad, reference_input_grad)
    assert checkpointed_parameter_grads.keys() == reference_parameter_grads.keys()
    for name, reference_grad in reference_parameter_grads.items():
        torch.testing.assert_close(checkpointed_parameter_grads[name], reference_grad)

    torch.testing.assert_close(reference_backward_load, reference_forward_load)
    torch.testing.assert_close(checkpointed_forward_load, reference_forward_load)
    torch.testing.assert_close(checkpointed_backward_load, checkpointed_forward_load)
    assert checkpointed_backward_load.sum().item() == inputs.shape[0] * inputs.shape[1] * 2


@pytest.mark.parametrize(
    "checkpoint_kwargs",
    [
        pytest.param({}, id="router-selective"),
        pytest.param({"selective": True}, id="operator-selective"),
        pytest.param({"ignore_router": False}, id="full"),
    ],
)
def test_gate_load_is_accumulated_once_for_each_activation_checkpointing_mode(
    checkpoint_kwargs: dict[str, bool],
) -> None:
    _assert_gate_load_and_gradient_parity(torch.device("cpu"), checkpoint_kwargs)
