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

import math

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn_f
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from nemo_automodel.components.moe.experts import (
    GroupedExperts,
    GroupedExpertsDeepEP,
    _AllGatherConcatVarlenFn,
    _apply_bias,
    _permute_tokens_for_grouped_mm,
)
from nemo_automodel.shared.utils import dtype_from_str


def _to_local(proj):
    """Convert DTensor to local tensor, or return as-is."""
    return proj.to_local() if isinstance(proj, DTensor) else proj


def _to_grouped_mm_operand(proj, dtype: torch.dtype):
    """Convert a projection tensor to the dtype/layout expected by grouped MM."""
    return _to_local(proj).to(dtype).contiguous()


def _pad_lora_rank_for_grouped_mm(lora_A: torch.Tensor, lora_B: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad LoRA rank tensors so grouped-mm strides are 16-byte aligned."""
    rank = lora_A.size(-1)
    alignment = max(1, 16 // lora_A.element_size())
    padding = (-rank) % alignment
    if padding == 0:
        return lora_A, lora_B
    return F.pad(lora_A, (0, padding)), F.pad(lora_B, (0, 0, 0, padding))


class GroupedExpertsLoRA(GroupedExperts):
    """
    GroupedExperts + LoRA.

    This class wraps `GroupedExperts` to apply LoRA to the expert weights.

    Attributes:
        lora_dim (int): Rank of the LoRA adapter.
        scale (float): Scaling factor for the LoRA adapter (alpha / dim).
        lora_gate_and_up_A (nn.Parameter): LoRA A matrix for gate and up projections.
        lora_gate_and_up_B (nn.Parameter): LoRA B matrix for gate and up projections.
        lora_down_A (nn.Parameter): LoRA A matrix for down projection.
        lora_down_B (nn.Parameter): LoRA B matrix for down projection.
    """

    def __init__(self, orig_module: GroupedExperts, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None):
        super().__init__(orig_module.config)

        self.gate_and_up_projs.data.copy_(orig_module.gate_and_up_projs.data)
        self.down_projs.data.copy_(orig_module.down_projs.data)

        if self.expert_bias:
            self.gate_up_proj_bias.data.copy_(orig_module.gate_up_proj_bias.data)
            self.down_proj_bias.data.copy_(orig_module.down_proj_bias.data)

        # Copy backend setting from original (super().__init__ defaults to False without backend)
        self.use_torch_mm = orig_module.use_torch_mm

        GroupedExpertsLoRA._init_adapter(
            self,
            lora_dim=lora_dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )

    @staticmethod
    def _init_adapter(obj, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None):
        obj.lora_dim = lora_dim
        obj.scale = alpha / lora_dim

        # Freeze base weights
        obj.gate_and_up_projs.requires_grad = False
        obj.down_projs.requires_grad = False
        if obj.expert_bias:
            obj.gate_up_proj_bias.requires_grad = False
            obj.down_proj_bias.requires_grad = False

        # Determine dtype
        if isinstance(lora_dtype, str):
            lora_dtype = dtype_from_str(lora_dtype)
        dtype = lora_dtype or obj.gate_and_up_projs.dtype
        device = obj.gate_and_up_projs.device

        up_proj_dim = obj.config.moe_inter_dim * 2 if obj.is_gated else obj.config.moe_inter_dim
        expert_dim = obj.config.expert_dim

        # LoRA weights for gate+up (or just up if non-gated) and down projections
        obj.lora_gate_and_up_A = nn.Parameter(
            torch.empty(obj.n_routed_experts, expert_dim, lora_dim, dtype=dtype, device=device)
        )
        obj.lora_gate_and_up_B = nn.Parameter(
            torch.empty(obj.n_routed_experts, lora_dim, up_proj_dim, dtype=dtype, device=device)
        )

        obj.lora_down_A = nn.Parameter(
            torch.empty(obj.n_routed_experts, obj.config.moe_inter_dim, lora_dim, dtype=dtype, device=device)
        )
        obj.lora_down_B = nn.Parameter(
            torch.empty(obj.n_routed_experts, lora_dim, expert_dim, dtype=dtype, device=device)
        )

        # Initialize LoRA weights
        GroupedExpertsLoRA.init_lora_weights(obj, lora_A_init_method)

    @torch.no_grad
    def init_lora_weights(self, init_method):
        """Initialize LoRA weights.

        IMPORTANT: This method is called by the PEFT framework's `_init_peft_adapters`
        after the model is materialized from meta device to the target device. The method
        name is critical - it serves as a hook for the framework.
        Do not rename or remove this method.

        Args:
            init_method (str): Initialization method ('xavier' or 'kaiming').
        """
        if init_method == "xavier":
            nn.init.xavier_normal_(self.lora_gate_and_up_A)
            nn.init.xavier_normal_(self.lora_down_A)
        else:
            nn.init.kaiming_uniform_(self.lora_gate_and_up_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_down_A, a=math.sqrt(5))

        nn.init.zeros_(self.lora_gate_and_up_B)
        nn.init.zeros_(self.lora_down_B)

    def materialize_effective_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the grouped expert weights with the LoRA updates folded in.

        Returns:
            A pair containing the input-projection tensor of shape
            [experts, expert_dim, fused_intermediate] and the down-projection
            tensor of shape [experts, intermediate, expert_dim].
        """
        gate_and_up_delta = torch.bmm(self.lora_gate_and_up_A, self.lora_gate_and_up_B) * self.scale
        down_delta = torch.bmm(self.lora_down_A, self.lora_down_B) * self.scale
        return self.gate_and_up_projs + gate_and_up_delta, self.down_projs + down_delta

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor):
        """Forward pass for GroupedExpertsLoRA with LoRA injection.

        Mirrors GroupedExperts.forward but injects LoRA computations into
        the expert processing at the projection level.
        """
        assert not isinstance(x, DTensor)
        input_dtype = x.dtype

        if isinstance(self.gate_and_up_projs, DTensor):
            ep_mesh = self.gate_and_up_projs.device_mesh
            assert ep_mesh is not None
            assert ep_mesh.ndim == 1
            ep_size = ep_mesh.size()
            ep_rank = ep_mesh.get_local_rank()
        else:
            ep_mesh = None
            ep_size = 1
            ep_rank = 0

        assert self.n_routed_experts % ep_size == 0

        compute_dtype = x.dtype
        gate_and_up_projs = _to_grouped_mm_operand(self.gate_and_up_projs, compute_dtype)
        down_projs = _to_grouped_mm_operand(self.down_projs, compute_dtype)
        lora_gate_and_up_A = _to_grouped_mm_operand(self.lora_gate_and_up_A, compute_dtype)
        lora_gate_and_up_B = _to_grouped_mm_operand(self.lora_gate_and_up_B, compute_dtype)
        lora_down_A = _to_grouped_mm_operand(self.lora_down_A, compute_dtype)
        lora_down_B = _to_grouped_mm_operand(self.lora_down_B, compute_dtype)

        local_num_tokens = x.size(0)
        if ep_size > 1:
            # Variable-length EP gather, mirroring GroupedExperts.forward.
            # DTensor.from_local(..., Shard(0)).full_tensor() assumes identical local shapes on
            # every EP rank; with ragged (unpacked / unpadded) batches each rank infers a different
            # global shape and the all_gather never completes. Exchange lengths, pad, gather, narrow.
            ep_group = ep_mesh.get_group()
            local_len_t = torch.tensor([local_num_tokens], device=x.device, dtype=torch.int64)
            gathered_len_t = [torch.zeros_like(local_len_t) for _ in range(ep_size)]
            dist.all_gather(gathered_len_t, local_len_t, group=ep_group)
            gathered_lens = [int(t.item()) for t in gathered_len_t]
            max_len = max(gathered_lens)

            def _gather_var(t: torch.Tensor, *, differentiable: bool) -> torch.Tensor:
                if differentiable:
                    return _AllGatherConcatVarlenFn.apply(t, ep_group, gathered_lens, max_len)
                if max_len > t.size(0):
                    pad = torch.zeros((max_len - t.size(0),) + tuple(t.shape[1:]), dtype=t.dtype, device=t.device)
                    t = torch.cat([t, pad], dim=0)
                gathered = [torch.empty_like(t) for _ in range(ep_size)]
                dist.all_gather(gathered, t, group=ep_group)
                return torch.cat([g[:n] for g, n in zip(gathered, gathered_lens)], dim=0)

            x = _gather_var(x, differentiable=True)
            weights = _gather_var(weights.float(), differentiable=True)
            indices = _gather_var(indices, differentiable=False)
            token_mask = _gather_var(token_mask, differentiable=False)

        n_local_experts = self.n_routed_experts // ep_size
        experts_start_idx = ep_rank * n_local_experts
        experts_end_idx = experts_start_idx + n_local_experts

        if self.use_torch_mm:
            lora_gate_and_up_A, lora_gate_and_up_B = _pad_lora_rank_for_grouped_mm(
                lora_gate_and_up_A, lora_gate_and_up_B
            )
            lora_down_A, lora_down_B = _pad_lora_rank_for_grouped_mm(lora_down_A, lora_down_B)
            y = self._forward_grouped_mm(
                x,
                token_mask,
                weights,
                indices,
                gate_and_up_projs,
                down_projs,
                lora_gate_and_up_A,
                lora_gate_and_up_B,
                lora_down_A,
                lora_down_B,
                n_local_experts,
                experts_start_idx,
            )
        else:
            y = self._forward_loop(
                x,
                weights,
                indices,
                token_mask,
                gate_and_up_projs,
                down_projs,
                lora_gate_and_up_A,
                lora_gate_and_up_B,
                lora_down_A,
                lora_down_B,
                n_local_experts,
                experts_start_idx,
                experts_end_idx,
            )

        if ep_size > 1:
            # Ragged-aware combine, mirroring GroupedExperts.forward:
            # redistribute(Shard(0)) would split the gathered output into equal chunks.
            y.add_(x.sum(dtype=torch.float32) * 0.0)  # keep the differentiable gather attached
            y = dist_nn_f.all_reduce(y, op=dist.ReduceOp.SUM, group=ep_group)
            start = sum(gathered_lens[:ep_rank])
            y = y.narrow(0, start, local_num_tokens).contiguous()

        return y.to(input_dtype)

    def _forward_loop(
        self,
        x,
        weights,
        indices,
        token_mask,
        gate_and_up_projs,
        down_projs,
        lora_gate_and_up_A,
        lora_gate_and_up_B,
        lora_down_A,
        lora_down_B,
        n_local_experts,
        experts_start_idx,
        experts_end_idx,
    ):
        """Per-expert loop forward path with LoRA injection."""
        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

        active_local_experts = 0
        for i in range(experts_start_idx, experts_end_idx):
            indices_mask = torch.logical_and(indices == i, token_mask.unsqueeze(-1))
            idx, top = torch.where(indices_mask)

            if idx.numel() == 0:
                continue
            active_local_experts += 1

            local_idx = i - experts_start_idx
            idx_b = idx[:, None].expand(-1, x.size(1))
            x_idx = x.gather(dim=0, index=idx_b)

            # Up projection + LoRA
            gate_and_up_out = x_idx @ gate_and_up_projs[local_idx]
            gate_and_up_out = (
                gate_and_up_out + (x_idx @ lora_gate_and_up_A[local_idx] @ lora_gate_and_up_B[local_idx]) * self.scale
            )

            if self.expert_bias:
                gate_and_up_out = gate_and_up_out + self.gate_up_proj_bias[local_idx]

            # Weighted activation (routing weight applied BETWEEN up and down projections)
            w = weights[idx, top, None]
            activated = self.expert_activation_grouped(gate_and_up_out, w)

            # Down projection + LoRA
            expert_out = activated @ down_projs[local_idx]
            expert_out = expert_out + (activated @ lora_down_A[local_idx] @ lora_down_B[local_idx]) * self.scale

            if self.expert_bias:
                expert_out = expert_out + self.down_proj_bias[local_idx] * w

            y.scatter_add_(dim=0, index=idx_b, src=expert_out.float())

        # Dummy computation for gradient flow when no tokens routed locally
        if active_local_experts == 0:
            dummy_x = torch.zeros_like(x[0]).unsqueeze(0)
            gate_and_up_out = dummy_x @ gate_and_up_projs[0]
            gate_and_up_out = gate_and_up_out + (dummy_x @ lora_gate_and_up_A[0] @ lora_gate_and_up_B[0]) * self.scale
            activated = self.expert_activation_grouped(gate_and_up_out, weights[0, 0, None].unsqueeze(0))
            expert_out = activated @ down_projs[0]
            expert_out = expert_out + (activated @ lora_down_A[0] @ lora_down_B[0]) * self.scale
            y[0] += expert_out[0]

        return y

    def _forward_grouped_mm(
        self,
        x,
        token_mask,
        weights,
        indices,
        gate_and_up_projs,
        down_projs,
        lora_gate_and_up_A,
        lora_gate_and_up_B,
        lora_down_A,
        lora_down_B,
        n_local_experts,
        experts_start_idx,
    ):
        """Grouped GEMM forward path with LoRA injection using torch._grouped_mm."""
        sorted_token_ids, sorted_weights, tokens_per_expert, offs = _permute_tokens_for_grouped_mm(
            indices,
            weights,
            token_mask,
            n_local_experts,
            experts_start_idx,
        )

        y = torch.zeros(x.shape, dtype=torch.float32, device=x.device)

        if sorted_token_ids.numel() > 0:
            permuted_x = x[sorted_token_ids]
            permuted_probs = sorted_weights.unsqueeze(-1)

            if self.expert_bias:
                gate_up_proj_bias = _to_local(self.gate_up_proj_bias)
                down_proj_bias = _to_local(self.down_proj_bias)

            # Gate+Up projection + LoRA
            output1 = torch._grouped_mm(permuted_x, gate_and_up_projs, offs=offs)
            lora_out1_A = torch._grouped_mm(permuted_x, lora_gate_and_up_A, offs=offs)
            lora_out1 = torch._grouped_mm(lora_out1_A, lora_gate_and_up_B, offs=offs)
            output1 = output1 + lora_out1 * self.scale

            if self.expert_bias:
                output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)

            output1 = self.expert_activation_grouped(output1, permuted_probs)

            # Down projection + LoRA
            output2 = torch._grouped_mm(output1, down_projs, offs=offs)
            lora_out2_A = torch._grouped_mm(output1, lora_down_A, offs=offs)
            lora_out2 = torch._grouped_mm(lora_out2_A, lora_down_B, offs=offs)
            output2 = output2 + lora_out2 * self.scale

            if self.expert_bias:
                output2 = _apply_bias(output2, down_proj_bias, tokens_per_expert, permuted_probs)

            scatter_ids = sorted_token_ids.unsqueeze(1).expand_as(output2)
            y.scatter_add_(0, scatter_ids, output2.float())
        else:
            # Dummy computation for gradient flow
            output1 = torch.matmul(x[0] * 0, gate_and_up_projs[0])
            output1 = (
                output1
                + torch.matmul(torch.matmul(x[0] * 0, lora_gate_and_up_A[0]), lora_gate_and_up_B[0]) * self.scale
            )
            output1_ = self.expert_activation_grouped(output1, weights[0, 0, None].unsqueeze(0))
            output2 = torch.matmul(output1_, down_projs[0])
            output2 = output2 + torch.matmul(torch.matmul(output1_ * 0, lora_down_A[0]), lora_down_B[0]) * self.scale
            y[0] += output2[0]

        return y


class GroupedExpertsDeepEPLoRA(GroupedExpertsDeepEP):
    """
    GroupedExpertsDeepEP + LoRA.

    This class wraps `GroupedExpertsDeepEP` to apply LoRA to the expert weights using DeepEP kernels.

    Attributes:
        lora_dim (int): Rank of the LoRA adapter.
        scale (float): Scaling factor for the LoRA adapter (alpha / dim).
        lora_gate_and_up_A (nn.Parameter): LoRA A matrix for gate and up projections.
        lora_gate_and_up_B (nn.Parameter): LoRA B matrix for gate and up projections.
        lora_down_A (nn.Parameter): LoRA A matrix for down projection.
        lora_down_B (nn.Parameter): LoRA B matrix for down projection.
    """

    def __init__(
        self, orig_module: GroupedExpertsDeepEP, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None
    ):
        super().__init__(
            orig_module.config,
            dispatcher_backend=orig_module.dispatcher_backend,
            dispatcher_num_sms=orig_module.dispatcher_num_sms,
            dispatcher_share_token_dispatcher=orig_module.dispatcher_share_token_dispatcher,
            dispatcher_async_dispatch=orig_module.dispatcher_async_dispatch,
        )

        self.gate_and_up_projs.data.copy_(orig_module.gate_and_up_projs.data)
        self.down_projs.data.copy_(orig_module.down_projs.data)

        if self.expert_bias:
            self.gate_up_proj_bias.data.copy_(orig_module.gate_up_proj_bias.data)
            self.down_proj_bias.data.copy_(orig_module.down_proj_bias.data)

        # Copy DeepEP state from orig_module (set by init_token_dispatcher, not __init__)
        self.n_routed_experts = getattr(orig_module, "n_routed_experts", self.config.n_routed_experts)
        self.ep_size = getattr(orig_module, "ep_size", 1)
        self.ep_rank = getattr(orig_module, "ep_rank", 0)
        self.token_dispatcher = getattr(orig_module, "token_dispatcher", None)
        self.use_mxfp8 = getattr(orig_module, "use_mxfp8", False)

        GroupedExpertsDeepEPLoRA._init_adapter(
            self,
            lora_dim=lora_dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )

    @staticmethod
    def _init_adapter(obj, lora_dim=8, alpha=32, lora_A_init_method="xavier", lora_dtype=None):
        obj.lora_dim = lora_dim
        obj.scale = alpha / lora_dim

        obj.gate_and_up_projs.requires_grad = False
        obj.down_projs.requires_grad = False
        if obj.expert_bias:
            obj.gate_up_proj_bias.requires_grad = False
            obj.down_proj_bias.requires_grad = False

        if isinstance(lora_dtype, str):
            lora_dtype = dtype_from_str(lora_dtype)
        dtype = lora_dtype or obj.gate_and_up_projs.dtype
        device = obj.gate_and_up_projs.device

        up_proj_dim = obj.config.moe_inter_dim * 2 if obj.is_gated else obj.config.moe_inter_dim
        expert_dim = obj.config.expert_dim

        # LoRA weights
        obj.lora_gate_and_up_A = nn.Parameter(
            torch.empty(obj.config.n_routed_experts, expert_dim, lora_dim, dtype=dtype, device=device)
        )
        obj.lora_gate_and_up_B = nn.Parameter(
            torch.empty(obj.config.n_routed_experts, lora_dim, up_proj_dim, dtype=dtype, device=device)
        )

        obj.lora_down_A = nn.Parameter(
            torch.empty(obj.config.n_routed_experts, obj.config.moe_inter_dim, lora_dim, dtype=dtype, device=device)
        )
        obj.lora_down_B = nn.Parameter(
            torch.empty(obj.config.n_routed_experts, lora_dim, expert_dim, dtype=dtype, device=device)
        )

        GroupedExpertsDeepEPLoRA.init_lora_weights(obj, lora_A_init_method)

    @torch.no_grad
    def init_lora_weights(self, init_method):
        """Initialize LoRA weights.

        IMPORTANT: This method is called by the PEFT framework's `_init_peft_adapters`
        after the model is materialized from meta device to the target device. The method
        name is critical - it serves as a hook for the framework.
        Do not rename or remove this method.

        Args:
            init_method (str): Initialization method ('xavier' or 'kaiming').
        """
        if init_method == "xavier":
            nn.init.xavier_normal_(self.lora_gate_and_up_A)
            nn.init.xavier_normal_(self.lora_down_A)
        else:
            nn.init.kaiming_uniform_(self.lora_gate_and_up_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_down_A, a=math.sqrt(5))

        nn.init.zeros_(self.lora_gate_and_up_B)
        nn.init.zeros_(self.lora_down_B)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        weights: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for GroupedExpertsDeepEPLoRA with LoRA injection.

        Mirrors GroupedExpertsDeepEP.forward but injects LoRA computations
        into the expert processing at the projection level.
        """
        assert not isinstance(x, DTensor)
        assert self.n_routed_experts % self.ep_size == 0

        indices = indices.masked_fill(~token_mask.unsqueeze(-1), -1)

        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = self.token_dispatcher.token_permutation2(
            hidden_states=x,
            num_local_tokens=x.size(0),
            token_probs=weights,
            token_indices=indices,
        )
        permuted_probs = permuted_probs.unsqueeze(-1)

        compute_dtype = x.dtype
        gate_and_up_projs = _to_grouped_mm_operand(self.gate_and_up_projs, compute_dtype)
        down_projs = _to_grouped_mm_operand(self.down_projs, compute_dtype)
        lora_gate_and_up_A = _to_grouped_mm_operand(self.lora_gate_and_up_A, compute_dtype)
        lora_gate_and_up_B = _to_grouped_mm_operand(self.lora_gate_and_up_B, compute_dtype)
        lora_down_A = _to_grouped_mm_operand(self.lora_down_A, compute_dtype)
        lora_down_B = _to_grouped_mm_operand(self.lora_down_B, compute_dtype)

        if torch.count_nonzero(tokens_per_expert) > 0:
            lora_gate_and_up_A, lora_gate_and_up_B = _pad_lora_rank_for_grouped_mm(
                lora_gate_and_up_A, lora_gate_and_up_B
            )
            lora_down_A, lora_down_B = _pad_lora_rank_for_grouped_mm(lora_down_A, lora_down_B)
            tokens_per_expert_gpu = tokens_per_expert.to(device=permuted_local_hidden_states.device, non_blocking=True)
            offs = tokens_per_expert_gpu.cumsum(dim=0).to(torch.int32)

            # Gate+Up projection + LoRA
            output1 = torch._grouped_mm(permuted_local_hidden_states, gate_and_up_projs, offs=offs)
            lora_out1_A = torch._grouped_mm(permuted_local_hidden_states, lora_gate_and_up_A, offs=offs)
            lora_out1 = torch._grouped_mm(lora_out1_A, lora_gate_and_up_B, offs=offs)
            output1 = output1 + lora_out1 * self.scale

            if self.expert_bias:
                gate_up_proj_bias = _to_local(self.gate_up_proj_bias)
                output1 = _apply_bias(output1, gate_up_proj_bias, tokens_per_expert)

            output1 = self.expert_activation(output1, permuted_probs)

            # Down projection + LoRA
            output2 = torch._grouped_mm(output1, down_projs, offs=offs)
            lora_out2_A = torch._grouped_mm(output1, lora_down_A, offs=offs)
            lora_out2 = torch._grouped_mm(lora_out2_A, lora_down_B, offs=offs)
            output2 = output2 + lora_out2 * self.scale

            if self.expert_bias:
                down_bias = _to_local(self.down_proj_bias)
                output2 = _apply_bias(output2, down_bias, tokens_per_expert, permuted_probs)
        else:
            # Dummy computation for gradient flow
            output1 = torch.matmul(x[0] * 0, gate_and_up_projs[0])
            output1 = (
                output1
                + torch.matmul(torch.matmul(x[0] * 0, lora_gate_and_up_A[0]), lora_gate_and_up_B[0]) * self.scale
            )
            output1_ = self.expert_activation(output1, permuted_probs)
            output2 = torch.matmul(output1_, down_projs[0])
            output2 = output2 + torch.matmul(torch.matmul(output1_ * 0, lora_down_A[0]), lora_down_B[0]) * self.scale

        y = self.token_dispatcher.token_unpermutation(output2)
        return y
