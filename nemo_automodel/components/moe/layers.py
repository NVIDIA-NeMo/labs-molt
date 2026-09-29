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
import warnings
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Partial, Replicate

from nemo_automodel.components.distributed.init_utils import get_world_size_safe
from nemo_automodel.components.models.common import BackendConfig, initialize_linear_module
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    GroupedExpertsTE,
    is_gated_activation,
)
from nemo_automodel.components.moe.experts import (
    _init_weights as _init_expert_weights,
)
from nemo_automodel.components.moe.megatron.moe_utils import (
    MoEAuxLossAutoScaler,
)
from nemo_automodel.components.moe.router_replay import RouterReplay, replay_selection


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Supports both gated activations (SwiGLU) and simple activations (ReLU²).

    Attributes:
        gate_proj (nn.Module): Linear layer for gate in gated activations (or up_proj for simple).
        down_proj (nn.Module): Linear layer for hidden-to-output transformation.
        up_proj (nn.Module): Additional linear layer for gated activations (None for simple).
    """

    def __init__(
        self,
        dim: int,
        inter_dim: int,
        backend: str,
        dtype: torch.dtype = torch.bfloat16,
        activation: str = "swiglu",
        bias: bool = False,
        swiglu_limit: float = 0.0,
    ):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
            backend (str): Backend for linear layers.
            dtype (torch.dtype): Data type for weights.
            activation (str): Activation function - "swiglu" (default) or "relu2".
            bias (bool): Whether to use bias in linear layers.
            swiglu_limit (float): When > 0 and activation is gated, run SwiGLU
                in fp32 with one-sided gate clamp ``max=limit`` and symmetric
                up clamp ``±limit``. Matches DSV4 reference ``Expert.forward``.
        """
        super().__init__()
        if activation not in ("swiglu", "relu2"):
            raise ValueError(f"Unsupported activation: {activation}. Choose 'swiglu' or 'relu2'.")

        self.activation = activation
        self.is_gated = is_gated_activation(activation)
        self.swiglu_limit = float(swiglu_limit)

        self.up_proj = initialize_linear_module(
            linear_impl=backend, in_features=dim, out_features=inter_dim, bias=bias, dtype=dtype
        )
        if self.is_gated:
            self.gate_proj = initialize_linear_module(
                linear_impl=backend, in_features=dim, out_features=inter_dim, bias=bias, dtype=dtype
            )
        else:
            self.gate_proj = None

        self.down_proj = initialize_linear_module(
            linear_impl=backend, in_features=inter_dim, out_features=dim, bias=bias, dtype=dtype
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        if not self.is_gated:
            return self.down_proj(F.relu(self.up_proj(x)).pow(2))
        if self.swiglu_limit > 0.0:
            # Mirror DSV4 reference Expert.forward: fp32 SwiGLU with one-sided
            # gate clamp and symmetric up clamp; cast back before down_proj.
            dtype = x.dtype
            gate = self.gate_proj(x).float().clamp(max=self.swiglu_limit)
            up = self.up_proj(x).float().clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
            return self.down_proj((F.silu(gate) * up).to(dtype))
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        init_weights_fn = partial(_init_weights, buffer_device=buffer_device, init_std=init_std)
        self.apply(init_weights_fn)


class FakeBalancedGate(nn.Module):
    """
    Load balanced gate implementation, spreads tokens uniformly across all experts.
    The rationale for this class is to do performance experiments to understand
    how the load imbalance with real data is impacting end-to-end performance.

    When ``noise > 0``, random perturbation is added to mimic realistic routing
    imbalance.  A noise value of 0.0 gives perfectly balanced assignment, while
    1.0 gives fully random expert selection and non-uniform weights.
    """

    def __init__(self, config: MoEConfig, skip_first_n_experts: int = 0, noise: float = 0.0):
        super().__init__()
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts
        self.skip_first_n_experts = skip_first_n_experts
        self.noise = noise
        self.bias_update_factor = 0.0

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        cp_mesh: DeviceMesh | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
            cp_mesh (Optional[DeviceMesh]): Device mesh for context parallel computation.

        Returns:
            weights (torch.Tensor): Routing weights for the selected experts.
            indices (torch.Tensor): Indices of the selected experts.
            aux_loss (Optional[torch.Tensor]): Auxiliary loss for load balancing.
        """
        del token_mask
        del cp_mesh

        n_tokens = x.size(0)
        n_exp = self.n_routed_experts
        a_exp = self.n_activated_experts
        available_experts = n_exp - self.skip_first_n_experts

        if self.noise > 0:
            # Derive the generator seed from the input content so that:
            #  - Forward and activation-checkpointing recompute get the same x,
            #    hence the same seed, hence identical routing (no shape mismatch).
            #  - Different training steps get different x (different data + updated
            #    weights), hence different seeds, hence dynamic routing that
            #    reproduces the varying tokens_per_expert pattern of real Gate.
            seed = int(x.view(-1)[:4].to(torch.float32).sum().item() * 1e6) & 0x7FFFFFFF
            gen = torch.Generator(device=x.device).manual_seed(seed)

            # Noisy weights: interpolate between uniform (1/a_exp) and random
            uniform_weights = torch.ones(n_tokens, a_exp, device=x.device) / a_exp
            raw_weights = torch.rand(n_tokens, a_exp, device=x.device, generator=gen)
            raw_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
            weights = (1 - self.noise) * uniform_weights + self.noise * raw_weights

            # Noisy indices via biased topk selection to mimic real routing where
            # some experts are systematically "hot".  A per-expert popularity bias
            # creates correlated selections across tokens; the noise parameter
            # scales this bias.  topk guarantees unique experts per token (required
            # by the downstream scatter-back which uses y[token_ids] += ...).
            expert_bias = torch.randn(available_experts, device=x.device, generator=gen) * self.noise * 0.1
            scores = torch.rand(n_tokens, available_experts, device=x.device, generator=gen) + expert_bias
            _, indices = scores.topk(a_exp, dim=-1)
            indices = indices + self.skip_first_n_experts
        else:
            weights = torch.ones(n_tokens, a_exp, device=x.device) / a_exp
            indices = (
                torch.arange(n_tokens * a_exp, device=x.device).view(-1, a_exp) % available_experts
            ) + self.skip_first_n_experts

        return weights.type_as(x), indices, None

    def update_bias(self) -> None:
        pass

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.apply(partial(_init_weights, buffer_device=buffer_device, init_std=init_std))


class _GateRoutingCore(nn.Module):
    """Parameterless, fixed-shape portion of a learned MoE router.

    The gate projection remains eager so FSDP can unshard its DTensor parameters.
    The owning gate is passed as a non-tensor control so this child remains an
    independent, parameterless CUDA graph boundary and survives model copies.
    """

    def forward(self, scores: torch.Tensor, gate: "Gate") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Select experts and compute routing probabilities from projected scores."""
        return gate._route_scores(scores)


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax', 'sigmoid', 'sigmoid_with_bias',
            'softmax_with_bias', or 'sqrtsoftplus').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """

    def __init__(
        self,
        config: MoEConfig,
        gate_precision: torch.dtype | None = None,
    ):
        """
        Initializes the Gate module.

        Args:
            config (MoEConfig): Model configuration containing gating parameters.
            gate_precision (torch.dtype | None): Precision for gate computations (linear, softmax/sigmoid).
        """
        super().__init__()
        self.dim = config.dim
        self.n_experts = config.n_routed_experts
        self.topk = config.n_activated_experts
        self.softmax_before_topk = config.softmax_before_topk
        self.router_weights_fp32 = config.router_weights_fp32
        self.router_weight_uses_score_correction_bias = config.router_weight_uses_score_correction_bias
        self.n_groups = config.n_expert_groups
        self.topk_groups = config.n_limited_groups
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.train_gate = config.train_gate
        self.bias_update_factor = config.gate_bias_update_factor
        self.aux_loss_coeff = config.aux_loss_coeff
        self.norm_topk_prob = config.norm_topk_prob
        self.gate_precision = gate_precision
        # Score arithmetic always has a concrete dtype: an explicit
        # gate_precision wins, otherwise fp32 (the shared default of both
        # score branches). gate_precision itself stays tri-state because
        # None means "project in the input's runtime dtype", which several
        # reference routers require (AM-821: Qwen3-MoE, GPT-OSS, DSV4,
        # Laguna project in bf16 while scoring in fp32).
        self.score_dtype = gate_precision if gate_precision is not None else torch.float32

        if self.bias_update_factor > 0:
            assert self.train_gate, "Require train_gate to be set to True to apply the bias update"

        gate_dtype = config.gate_dtype or config.dtype
        self.weight = nn.Parameter(
            torch.empty(config.n_routed_experts, config.dim, dtype=gate_dtype), requires_grad=self.train_gate
        )
        if config.router_bias:
            self.bias = nn.Parameter(
                torch.empty(config.n_routed_experts, dtype=gate_dtype), requires_grad=self.train_gate
            )
        else:
            self.bias = None

        # e_score_correction_bias is only created when bias_update_factor > 0 (for training)
        # or when force_e_score_correction_bias is True (for loading HF checkpoints that have it).
        # This flag is useful in cases where we want to load the bias but not update it.
        # Must be float32 when created - small quantization errors in bf16 can cause
        # completely different expert routing.
        if self.bias_update_factor > 0 or config.force_e_score_correction_bias:
            self.register_buffer("e_score_correction_bias", torch.zeros((self.n_experts), dtype=torch.float32))
        else:
            self.e_score_correction_bias = None

        self.e_score_correction_bias_master = None

        # Cumulative expert load is a tensor representing the number of tokens
        # routed to each expert on the current rank, accumulated across gradient
        # accumulation steps.
        self._cumulative_expert_load: torch.Tensor | None = None

        # Load balance tracking (enabled externally via enable_load_balance_tracking)
        self._track_load_balance: bool = False
        self._last_expert_load: torch.Tensor | None = None
        self._last_aux_loss: torch.Tensor | None = None

        # Rollout Routing Replay (R3): owns a handle only when enabled so the
        # default routing path stays a no-op.
        self.router_replay: RouterReplay | None = (
            RouterReplay() if getattr(config, "enable_routing_replay", False) else None
        )
        self.routing_core = _GateRoutingCore()
        self.use_routing_core = False

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        """Load gate state while preserving the routing-bias DTensor layout.

        Args:
            state_dict: State mapping whose correction bias is either a full tensor
                of shape ``[experts]`` or an already distributed tensor.
            prefix: Prefix applied to this module's state-dict keys.
            local_metadata: Metadata saved for this module.
            strict: Whether state-dict keys must match exactly.
            missing_keys: Missing state-dict keys collected during loading.
            unexpected_keys: Unexpected state-dict keys collected during loading.
            error_msgs: State-dict loading errors collected during loading.

        Raises:
            RuntimeError: If a full correction-bias tensor targets an incompatible
                shape or a DTensor that is not replicated on every mesh dimension.
        """
        bias_key = f"{prefix}e_score_correction_bias"
        target_bias = self.e_score_correction_bias
        checkpoint_bias = state_dict.get(bias_key)

        if (
            isinstance(target_bias, DTensor)
            and isinstance(checkpoint_bias, torch.Tensor)
            and not isinstance(checkpoint_bias, DTensor)
        ):
            if checkpoint_bias.shape != target_bias.shape:
                raise RuntimeError(
                    f"Cannot load {bias_key}: checkpoint shape {checkpoint_bias.shape} does not match "
                    f"the routing bias global shape {target_bias.shape}."
                )
            if not all(isinstance(placement, Replicate) for placement in target_bias.placements):
                raise RuntimeError(
                    f"Cannot load a full tensor into {bias_key}: the routing bias DTensor must be replicated "
                    f"on every mesh dimension, but has placements {target_bias.placements}."
                )

            local_target = target_bias.to_local()
            local_bias = checkpoint_bias.to(device=local_target.device, dtype=local_target.dtype)
            state_dict[bias_key] = DTensor.from_local(
                local_bias,
                device_mesh=target_bias.device_mesh,
                placements=target_bias.placements,
                run_check=False,
                shape=target_bias.shape,
                stride=target_bias.stride(),
            )

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _local_score_correction_bias(self) -> torch.Tensor | None:
        """Return the local tensor used to adjust this rank's routing scores.

        Returns:
            Tensor of shape [experts] replicated across the gate's DP/CP mesh, or
            ``None`` when score correction is disabled. The returned tensor aliases
            the registered correction-bias buffer.
        """
        if isinstance(self.e_score_correction_bias, DTensor):
            return self.e_score_correction_bias.to_local()
        return self.e_score_correction_bias

    def _route_scores(self, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply fixed-shape expert selection and probability math to router logits.

        Args:
            scores: Tensor of shape [tokens, experts] containing local router logits.

        Returns:
            Tuple containing routing weights of shape [tokens, activated_experts],
            expert indices of shape [tokens, activated_experts], and optional
            original routing scores of shape [tokens, experts].
        """
        num_tokens = scores.size(0)
        correction_bias = self._local_score_correction_bias()

        if self.score_func == "softmax":
            if self.softmax_before_topk:
                scores = scores.softmax(dim=-1, dtype=self.score_dtype)
                original_scores = scores
                indices = torch.topk(scores, k=self.topk, dim=-1)[1]
                indices = replay_selection(self.router_replay, indices)
                weights = scores.gather(1, indices)
            else:
                values, indices = torch.topk(scores, k=self.topk, dim=-1)
                replayed = replay_selection(self.router_replay, indices)
                if replayed is not indices:
                    # Replay swapped the selection: re-gather the values for the
                    # replayed experts. Skipped (zero overhead) on the default path.
                    values = scores.gather(1, replayed)
                    indices = replayed
                weights = values.softmax(dim=1, dtype=self.score_dtype)
                # Use full softmax for aux_loss so P_i represents proper probabilities.
                # Raw logits can be negative, causing aux_loss to diverge negative.
                original_scores = scores.softmax(dim=-1, dtype=self.score_dtype)
        elif self.score_func == "softmax_with_bias":
            # softmax first, then add bias for expert selection,
            # group routing on biased scores, final weights from unbiased softmax scores.
            scores = scores.softmax(dim=-1, dtype=self.score_dtype)
            original_scores = scores

            # Add correction bias for expert SELECTION only
            if correction_bias is not None:
                scores_for_choice = scores + correction_bias
            else:
                scores_for_choice = scores

            if self.n_groups > 1:
                scores_for_choice = scores_for_choice.view(num_tokens, self.n_groups, -1)
                group_scores = scores_for_choice.topk(2, dim=-1)[0].sum(dim=-1)

                group_idx = group_scores.topk(self.topk_groups, dim=-1)[1]
                mask = torch.zeros_like(scores_for_choice[..., 0]).scatter_(1, group_idx, True)
                scores_for_choice = (scores_for_choice * mask.unsqueeze(-1)).flatten(1)

            indices = torch.topk(scores_for_choice, self.topk, dim=-1)[1]
            indices = replay_selection(self.router_replay, indices)
            # Final weights gathered from UNBIASED softmax scores
            weights = original_scores.gather(1, indices)
        elif self.score_func == "sqrtsoftplus":
            # sqrt(softplus(x)) = sqrt(log(1 + exp(x))), used in DeepSeek V4.
            scores = torch.sqrt(F.softplus(scores.float()))
            original_scores = scores

            if correction_bias is not None:
                scores = scores + correction_bias

            indices = torch.topk(scores, self.topk, dim=-1)[1]
            indices = replay_selection(self.router_replay, indices)
            weights = original_scores.gather(1, indices)
        elif self.score_func == "sigmoid_with_bias":
            # Score in fp32 like the softmax path: HF sigmoid-router references
            # compute sigmoid(logits.float()), and bf16 sigmoid quantizes scores
            # at ~2e-3 — enough to flip knife-edge e_score_correction_bias
            # selections (AMINT-286).
            scores = torch.sigmoid(scores.to(dtype=self.score_dtype))
            original_scores = scores
            scores_for_choice = scores

            if correction_bias is not None:
                scores_for_choice = scores_for_choice + correction_bias

            if self.n_groups > 1:
                scores_for_choice = scores_for_choice.view(num_tokens, self.n_groups, -1)
                group_scores = scores_for_choice.topk(2, dim=-1)[0].sum(dim=-1)
                group_idx = torch.topk(group_scores, k=self.topk_groups, dim=-1, sorted=False)[1]
                group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1)
                score_mask = group_mask.unsqueeze(-1).expand_as(scores_for_choice).reshape(num_tokens, -1)
                scores_for_choice = scores_for_choice.reshape(num_tokens, -1).masked_fill(
                    ~score_mask.bool(), float("-inf")
                )

            indices = torch.topk(scores_for_choice, k=self.topk, dim=-1, sorted=False)[1]
            indices = replay_selection(self.router_replay, indices)
            weights = original_scores.gather(1, indices)
        else:
            # Score in fp32 like the softmax path (see sigmoid_with_bias above).
            scores = torch.sigmoid(scores.to(dtype=self.score_dtype))
            original_scores = scores

            # Add correction bias to balance tokens across gates.
            if correction_bias is not None:
                scores = scores + correction_bias
                if self.router_weight_uses_score_correction_bias:
                    original_scores = scores

            if self.n_groups > 1:
                scores = scores.view(num_tokens, self.n_groups, -1)
                if correction_bias is None:
                    group_scores = scores.amax(dim=-1)
                else:
                    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)

                indices = group_scores.topk(self.topk_groups, dim=-1)[1]
                mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
                scores = (scores * mask.unsqueeze(-1)).flatten(1)

            indices = torch.topk(scores, self.topk, dim=-1)[1]
            indices = replay_selection(self.router_replay, indices)
            weights = original_scores.gather(1, indices)

        if self.norm_topk_prob and self.topk > 1:
            denom_w = weights.sum(dim=-1, keepdim=True) + 1e-20
            denom_s = original_scores.sum(dim=-1, keepdim=True) + 1e-20
            weights = weights / denom_w
            original_scores = original_scores / denom_s

        weights = weights * self.route_scale
        return weights, indices, original_scores

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        cp_mesh: DeviceMesh | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
            cp_mesh (Optional[DeviceMesh]): Device mesh for context parallel computation.

        Returns:
            weights (torch.Tensor): Routing weights for the selected experts.
            indices (torch.Tensor): Indices of the selected experts.
            aux_loss (Optional[torch.Tensor]): Auxiliary loss for load balancing.
        """
        original_dtype = x.dtype

        if self.gate_precision is not None:
            x_compute = x.to(dtype=self.gate_precision)
            weight = self.weight.to(dtype=self.gate_precision)
            bias = self.bias.to(dtype=self.gate_precision) if self.bias is not None else None
        else:
            x_compute = x
            weight = self.weight.to(dtype=x.dtype)
            bias = self.bias.to(dtype=x.dtype) if self.bias is not None else None

        scores = F.linear(x_compute, weight, bias=bias)
        if self.use_routing_core:
            weights, indices, original_scores = self.routing_core(scores, self)
        else:
            weights, indices, original_scores = self._route_scores(scores)

        if self.gate_precision is not None and not self.router_weights_fp32:
            weights = weights.to(dtype=original_dtype)
            original_scores = original_scores.to(dtype=original_dtype)

        if self.bias_update_factor > 0 or self.aux_loss_coeff > 0 or self._track_load_balance:
            expert_load = self._compute_expert_load(indices, token_mask)

        if self._track_load_balance:
            self._last_expert_load = expert_load.detach()

        if self.bias_update_factor > 0 and self.training:
            if self._cumulative_expert_load is None:
                self._cumulative_expert_load = expert_load.detach()
            else:
                # Out-of-place add: under activation checkpointing this gate forward is
                # recomputed in backward, and an in-place op on non-checkpointed module
                # state trips torch's guard ("aten.add_.Tensor ... not found in storage").
                # The bias update uses sign(avg - load), so the recompute double-count is a no-op.
                self._cumulative_expert_load = self._cumulative_expert_load + expert_load.detach()

        aux_loss = None
        if self.aux_loss_coeff > 0 and self.training:
            aux_loss = self._compute_aux_loss(original_scores, expert_load, token_mask, cp_mesh)
            # Scale the aux_loss by the number of tokens.
            # Training scales all gradients by 1/(number of tokens).
            # To correct this scaling, we need to scale the aux_loss by number of tokens here.
            weights = MoEAuxLossAutoScaler.apply(weights, self.aux_loss_coeff * aux_loss)

        if self._track_load_balance and aux_loss is not None:
            self._last_aux_loss = aux_loss.detach()

        if self.router_weights_fp32:
            return weights.float(), indices, aux_loss
        return weights.type_as(x), indices, aux_loss

    def update_bias(self) -> None:
        """
        Updates the correction bias used in the gate based on the popularity of experts.
        This function is a NoOp if the gate is not trained.

        To avoid routing collapse, and to promote better load balance of experts,
        DeepSeek-V3 uses a correction mechanism to adjust the scores of experts using
        a learned bias parameter. The bias parameter is updated based on the popularity
        of experts, i.e., the number of tokens routed to each expert. If an expert is
        more popular than the average, its bias term is decreased, and vice versa.
        This encourages the model to route tokens to less popular experts, promoting
        better load balance.
        """
        assert self.train_gate and self.bias_update_factor > 0, "Gate bias update is disabled"

        assert self.training, "Gate bias update is only supported during training"
        assert self._cumulative_expert_load is not None, (
            "Score correction bias cannot be updated without the current expert load"
        )

        # 1) Compute the expert load across all DP ranks.
        # Copy the cumulative load into a local variable, and set the stored load to None.
        expert_load = self._cumulative_expert_load
        self._cumulative_expert_load = None

        # Place the expert load on the same device mesh as the score correction
        # bias parameter, and sum across all ranks.
        if isinstance(self.e_score_correction_bias, DTensor):
            expert_load = DTensor.from_local(
                expert_load,
                device_mesh=self.e_score_correction_bias.device_mesh,
                placements=[Partial()] * self.e_score_correction_bias.device_mesh.ndim,
            )
            expert_load = expert_load.full_tensor()
        # 2) Compute the bias update by comparing the expert load to the average expert load.
        expert_load = expert_load.float()
        average_expert_load = expert_load.mean()
        bias_update = torch.sign(average_expert_load - expert_load)

        if isinstance(self.e_score_correction_bias, DTensor):
            # Convert the bias update back to a replicated DTensor with the same device
            # mesh as the score correction bias parameter.
            bias_update = DTensor.from_local(
                bias_update,
                device_mesh=self.e_score_correction_bias.device_mesh,
                placements=[Replicate()] * self.e_score_correction_bias.device_mesh.ndim,
            )

            # The score correction bias parameter could be sharded across FSDP
            # ranks (dim=-1), and/or optionally replicated across DDP ranks (dim=0).
            # Redistribute the bias update with the same placement.
            bias_update = bias_update.redistribute(placements=self.e_score_correction_bias.placements)

        # 3) Update the correction bias using the bias update.
        with torch.no_grad():
            # Create full precision master weights
            if self.e_score_correction_bias_master is None:
                self.e_score_correction_bias_master = self.e_score_correction_bias.clone().detach().float()
            self.e_score_correction_bias_master += bias_update * self.bias_update_factor
            self.e_score_correction_bias.copy_(self.e_score_correction_bias_master)

    def _compute_expert_load(
        self,
        indices: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes the load of each expert based on the selected indices.
        Args:
            indices (torch.Tensor): Indices of the selected experts.
                Shape is [num_tokens, num_activated_experts].
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
                Shape is [num_tokens].

        Returns:
            torch.Tensor: Load of each expert (number of tokens routed to each expert).
                Shape is [num_local_experts].
        """
        # Create a mask for the experts based on the selected indices.
        expert_mask = indices.new_zeros((indices.shape[0], self.n_experts))
        contribution = token_mask.to(dtype=expert_mask.dtype).unsqueeze(-1).expand(-1, indices.shape[1])
        expert_mask.scatter_(dim=1, index=indices, src=contribution)
        return expert_mask.sum(dim=0)

    def _compute_aux_loss(
        self,
        original_scores: torch.Tensor,
        expert_load: torch.Tensor,
        token_mask: torch.Tensor,
        cp_mesh: DeviceMesh | None,
    ) -> torch.Tensor:
        """
        Computes the auxiliary loss for load balancing.

        **Warning**: Assumes batch size = 1, if batch size > 1, the aux_loss will
        be computed across multiple sequences.

        Args:
            original_scores (torch.Tensor): Original scores from the gating mechanism.
                Shape is [num_tokens, num_experts].
            expert_load (torch.Tensor): Load of each expert (number of tokens routed to each expert).
                Shape is [num_experts].
            token_mask (torch.Tensor): Boolean mask indicating valid tokens.
                Shape is [num_tokens].
            cp_mesh (Optional[DeviceMesh]): Device mesh for context parallel computation.

        Returns:
            torch.Tensor: Auxiliary loss for load balancing.
                Shape is [].
        """
        # Pin aux-loss arithmetic to fp32. The activation-checkpoint recompute pass
        # does not replay FSDP2 MixedPrecisionPolicy cast_forward_inputs, so `x` and
        # anything derived from `x.dtype` (including original_scores in some MoE
        # configs) can differ between forward and recompute. Doing the cast inside
        # the AC region — rather than at the Gate level — guarantees the saved
        # tensors of this region (expert_scores [n_experts] and the aux-loss scalar)
        # are fp32 on both passes regardless of upstream behavior.
        original_scores = original_scores.float()
        expert_load = expert_load.float()

        context_length = token_mask.sum()
        expert_scores = (original_scores * token_mask.unsqueeze(-1)).sum(dim=0)

        if cp_mesh is not None:
            context_length = DTensor.from_local(
                context_length, device_mesh=cp_mesh, placements=[Partial()]
            ).full_tensor()
            expert_load = DTensor.from_local(expert_load, device_mesh=cp_mesh, placements=[Partial()]).full_tensor()
            expert_scores = DTensor.from_local(expert_scores, device_mesh=cp_mesh, placements=[Partial()]).full_tensor()

        # Compute f_i (fraction of tokens dispatched to each expert).
        # If uniform distribution, expert_load will be topk * num_location / n_experts, and f_i will be 1
        # Maximum value f_i entries happens when expert_load = num_location, the value will be n_experts / topk
        # Protect against division by zero when all tokens are masked (e.g. padding-heavy CP rank).
        context_length = torch.clamp(context_length, min=1)
        f_i = expert_load * self.n_experts / (self.topk * context_length)  # Normalized fraction, (n_experts)

        # Compute P_i (average routing probability per expert)
        P_i = expert_scores / context_length  # (n_experts)

        loss = torch.sum(f_i * P_i)
        return loss

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        self.apply(partial(_init_weights, buffer_device=buffer_device, init_std=init_std))


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """

    def __init__(self, config: MoEConfig, backend: BackendConfig):
        """
        Initializes the MoE module.

        Args:
            args (MoEArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.backend = backend
        self.dim = config.dim
        self.n_routed_experts = config.n_routed_experts
        self.n_activated_experts = config.n_activated_experts

        if backend.fake_balanced_gate:
            self.gate = FakeBalancedGate(config, noise=backend.fake_gate_noise)
        else:
            self.gate = Gate(config, gate_precision=backend.gate_precision)
            self.gate.use_routing_core = "moe_router" in backend.cuda_graph.modules
        if backend.dispatcher in ("deepep", "hybridep", "uccl_ep") and get_world_size_safe() == 1:
            warnings.warn(
                f"'{backend.dispatcher}' dispatcher is enabled in config, but world size is 1. "
                "Expert parallelism requires multiple GPUs. Falling back to standard GroupedExperts.",
                category=UserWarning,
                stacklevel=2,
            )
            self.experts = GroupedExperts(config, backend=backend)
        elif backend.dispatcher in ("deepep", "hybridep", "uccl_ep"):
            if backend.experts in ("gmm", "torch_mm", "torch_mm_mxfp8"):
                self.experts = GroupedExpertsDeepEP(
                    config,
                    backend=backend,
                    dispatcher_backend=backend.dispatcher,
                    dispatcher_num_sms=backend.dispatcher_num_sms,
                    dispatcher_share_token_dispatcher=backend.dispatcher_share_token_dispatcher,
                    dispatcher_async_dispatch=backend.dispatcher_async_dispatch,
                )
            else:
                # All other expert backends use the TE grouped implementation.
                self.experts = GroupedExpertsTE(
                    config,
                    backend=backend,
                    dispatcher_backend=backend.dispatcher,
                    dispatcher_num_sms=backend.dispatcher_num_sms,
                    dispatcher_share_token_dispatcher=backend.dispatcher_share_token_dispatcher,
                    dispatcher_async_dispatch=backend.dispatcher_async_dispatch,
                )
        else:
            # Default to torch experts
            self.experts = GroupedExperts(config, backend=backend)

        if config.n_shared_experts > 0:
            self.shared_experts = MLP(
                config.dim,
                config.n_shared_experts * (config.shared_expert_inter_dim or config.moe_inter_dim),
                backend.linear,
                dtype=config.dtype,
                activation=config.shared_expert_activation,
                bias=config.expert_bias,
                swiglu_limit=config.swiglu_limit,
            )
            if config.shared_expert_gate:
                self.shared_expert_gate = initialize_linear_module(
                    backend.linear, config.dim, 1, False, dtype=config.dtype
                )
            else:
                self.shared_expert_gate = None
        else:
            self.shared_experts = None
            self.shared_expert_gate = None

        # When enabled, input is projected to latent space before MoE and back after
        if config.moe_latent_size is not None:
            self.fc1_latent_proj = initialize_linear_module(
                backend.linear, config.dim, config.moe_latent_size, bias=config.expert_bias, dtype=config.dtype
            )
            self.fc2_latent_proj = initialize_linear_module(
                backend.linear, config.moe_latent_size, config.dim, bias=config.expert_bias, dtype=config.dtype
            )
        else:
            self.fc1_latent_proj = None
            self.fc2_latent_proj = None

        # Set during model parallelization (see parallelizer.apply_cp)
        self.cp_mesh: DeviceMesh | None = None

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        cp_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        """Route tokens through shared and routed experts.

        Args:
            x: Input tensor of shape ``[..., hidden]``.
            padding_mask: Boolean tensor matching ``x.shape[:-1]`` where true
                entries are padding.
            cp_mesh: Optional context-parallel mesh used by the router.

        Returns:
            Tensor with the same shape and dtype as ``x``.
        """
        if cp_mesh is None:
            cp_mesh = self.cp_mesh

        # Reshape the inputs to 2-D since we are just distributing tokens.
        shape = x.size()
        x = x.view(-1, self.dim)
        if padding_mask is not None:
            token_mask = (~padding_mask).flatten()
        else:
            token_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        # Apply latent projection before MoE if enabled
        if self.fc1_latent_proj is not None:
            x_latent = self.fc1_latent_proj(x)
        else:
            x_latent = x

        weights, indices, aux_loss = self.gate(x, token_mask, cp_mesh)
        # Shared-expert output (optionally gated), computed inline on the main stream.
        z = None
        if self.shared_experts is not None:
            z = self.shared_experts(x)
            if self.shared_expert_gate is not None:
                z = torch.nn.functional.sigmoid(self.shared_expert_gate(x)) * z

        # Routed experts on the main stream.
        y = self.experts(x_latent, token_mask, weights, indices)

        if self.fc2_latent_proj is not None:
            # ``self.experts`` is its own FSDP unit; an ``output_dtype`` in the FSDP
            # MixedPrecisionPolicy (NeMo-RL uses float32) casts its output above the
            # block's compute dtype, and TE/torch linears reject an input dtype that
            # differs from the (param_dtype-cast) weight. Re-enter the compute dtype
            # of the latent input before the back-projection.
            y = self.fc2_latent_proj(y.to(x_latent.dtype))
        if z is not None:
            y = y + z
        return y.view(shape)

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        init_weights_fn = partial(_init_weights, buffer_device=buffer_device, init_std=init_std)
        self.apply(init_weights_fn)


def _init_weights(module, buffer_device: torch.device, init_std: float = 0.02):
    def to_local(tensor):
        if isinstance(tensor, DTensor):
            return tensor.to_local()
        else:
            return tensor

    with torch.device(buffer_device):
        if isinstance(module, Gate):
            to_local(module.weight).normal_(mean=0.0, std=init_std)
            if module.e_score_correction_bias is not None:
                to_local(module.e_score_correction_bias).zero_()
            if module.bias is not None:
                to_local(module.bias).zero_()
        elif isinstance(module, (GroupedExperts, GroupedExpertsDeepEP, GroupedExpertsTE)):
            # Delegate expert initialization to experts.py
            _init_expert_weights(module, buffer_device, init_std)
        elif isinstance(module, MLP):
            to_local(module.down_proj.weight).normal_(mean=0.0, std=init_std)
            to_local(module.up_proj.weight).normal_(mean=0.0, std=init_std)
            if module.gate_proj is not None:
                to_local(module.gate_proj.weight).normal_(mean=0.0, std=init_std)
        elif isinstance(module, MoE):
            if module.fc1_latent_proj is not None:
                to_local(module.fc1_latent_proj.weight).normal_(mean=0.0, std=init_std)
                if module.fc1_latent_proj.bias is not None:
                    to_local(module.fc1_latent_proj.bias).zero_()
            if module.fc2_latent_proj is not None:
                to_local(module.fc2_latent_proj.weight).normal_(mean=0.0, std=init_std)
                if module.fc2_latent_proj.bias is not None:
                    to_local(module.fc2_latent_proj.bias).zero_()
