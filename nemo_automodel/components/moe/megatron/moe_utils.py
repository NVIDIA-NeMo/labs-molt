# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import torch
import torch.nn.functional as F

from nemo_automodel.shared.import_utils import safe_import_te

HAS_TE, transformer_engine = safe_import_te()
if HAS_TE:
    from transformer_engine.pytorch import (
        moe_permute,
        moe_permute_with_probs,
        moe_unpermute,
    )


def permute(
    tokens,
    routing_map,
    probs: torch.Tensor | None = None,
    num_out_tokens: int | None = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """Permute the tokens and probs based on the mask.
    Tokens with the same designated expert will be grouped together.
    The shape of mask is [tokens, num_experts], it indicates which experts were selected
    by each token.

    When drop_and_pad=True, in routing_map, the number of non-zeros in each column equals to
    expert capacity. This function exploits this feature to use ops that support cuda graph.

    Args:
        tokens (torch.Tensor): The input token tensor, [num_tokens, hidden].
        routing_map (torch.Tensor): The sparse token to expert mapping, [num_tokens, num_experts].
        probs (torch.Tensor, optional): The probs tensor, [num_tokens, num_experts].
        num_out_tokens (int, optional): The number of output tokens. If None, it's set to
                                        the number of input tokens.
        fused (bool, optional): Whether use the fused permute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.
                                       If set to true, routing_map has a fixed number of non-zeros
                                       in each column.

    Returns:
        permuted_input (torch.Tensor): The permuted token tensor.
        permuted_probs (torch.Tensor, optional): The permuted probs tensor.
        sorted_indices (torch.Tensor): The tensor of a mapping table for sorted indices used to unpermute the tokens.
    """
    if fused and probs is None:
        if not HAS_TE or moe_permute is None:
            raise ValueError("moe_permute is not available. Please install TE >= 2.1.0.")
        permuted_input, sorted_indices = moe_permute(tokens, routing_map, num_out_tokens=num_out_tokens)
        return permuted_input, None, sorted_indices

    if fused and probs is not None:
        if not HAS_TE or moe_permute_with_probs is None:
            raise ValueError("moe_permute_with_probs is not available. Please install TE >= 2.1.0.")
        return moe_permute_with_probs(tokens, probs, routing_map, num_out_tokens=num_out_tokens)

    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None
    if drop_and_pad and num_out_tokens is not None:
        capacity = num_out_tokens // num_experts
        assert not routing_map.requires_grad
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.to(dtype=torch.int8).T.contiguous()
        # use argsort to put indices of all non-zeros in the beginning of list
        # and keep the first `capacity` number of indices
        sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)[:, :capacity].contiguous()
        # flatten from [num_experts, capacity] to 1D
        sorted_indices = sorted_indices.view(-1)

        if probs is not None:
            # [num_tokens, num_experts] -> num_experts * num_tokens
            probs_T_1D = probs.T.contiguous().view(-1)
            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_tokens + indices_dim1).view(-1)
            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
    else:
        # mask [num_tokens, num_experts] -> [num_experts, num_tokens]
        routing_map = routing_map.bool().T.contiguous()

        # Create a dense expert-to-token mapping from the sparse token-to-expert mapping
        token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
        sorted_indices = token_indices.masked_select(routing_map)

        if probs is not None:
            permuted_probs = probs.T.contiguous().masked_select(routing_map)

    # use the mapping to permute the tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """
    Restore the original order of tokens after permutation. If probs are provided, it
    will also apply them to the tokens before restoring the order.

    When drop_and_pad=True, the tensors will have the following properties:
      - In routing_map, the number of non-zeros in each column equals to expert capacity
      - The size of sorted_indices equals to num_experts * capacity, each split of `capacity`
        contains the indices of tokens routed to an expert.
    This function exploits these features to use ops that support cuda graph.

    Args:
        permuted_tokens (torch.Tensor): The permuted token tensor.
        sorted_indices (torch.Tensor): The indices used to sort the tokens.
        restore_shape (torch.Size): The shape of the unpermuted tensor.
        probs (torch.Tensor, optional): The unpermuted probs tensor,
        routing_map (torch.Tensor, optional): Token to expert mapping, shape
            [num_tokens, num_experts].
        fused (bool, optional): Whether use the fused unpermute function.
        drop_and_pad (bool, optional): Whether or not the token dispatcher uses token-drop
                                       and pads the number of tokens to the expert capacity.

    Returns:
        torch.Tensor: The tokens restored to their original order.
    """
    if fused:
        if not HAS_TE or moe_unpermute is None:
            raise ValueError("moe_unpermute is not available. Please install TE >= 2.1.0.")
        return moe_unpermute(
            permuted_tokens,
            sorted_indices,
            merging_probs=probs,
            restore_shape=restore_shape,
        )

    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype

    if probs is not None:
        assert routing_map is not None, "Mask must be provided to permute the probs."
        if drop_and_pad:
            num_experts = routing_map.size(1)
            num_permuted_tokens = sorted_indices.size(0)
            capacity = num_permuted_tokens // num_experts
            num_unpermuted_tokens = probs.size(0)

            # [num_unpermuted_tokens, num_experts] -> num_experts * num_unpermuted_tokens
            probs_T_1D = probs.T.contiguous().view(-1)

            # get 1D indices of the probs selected by routing_map
            indices_dim0 = torch.arange(num_experts, device=routing_map.device).unsqueeze(-1)
            indices_dim1 = sorted_indices.view(num_experts, capacity)
            indices_1D = (indices_dim0 * num_unpermuted_tokens + indices_dim1).view(-1)

            # get probs from indices
            permuted_probs = probs_T_1D.index_select(0, indices_1D)
        else:
            permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        # Here may promote permuted_tokens to higher precision (fp32/fp64) if probs is in
        # higher precision due to moe_router_dtype being enabled. This can lead to
        # additional GPU memory usage. Use --moe-permute-fusion flag to avoid this extra memory
        # allocation.
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create an output tensor filled with zeros
    output_tokens = torch.zeros(restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device)
    # Scatter add the permuted_input back to the original positions
    output_tokens.scatter_add_(0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)
    return output_tokens.to(dtype=input_dtype)


@torch.compile
def swiglu(y):
    """Apply SwiGLU activation to an interleaved gate/up tensor."""

    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.silu(y_1) * y_2


@torch.compile
def weighted_swiglu(y, weights):
    """Apply SwiGLU activation and token-wise routing weights."""

    dtype = y.dtype
    res = swiglu(y) * weights
    return res.to(dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.compile
def swiglu_back(g, y):
    """Compute the input gradient for SwiGLU activation."""

    y_1, y_2 = torch.chunk(y, 2, -1)
    return torch.cat(
        (
            g * torch.sigmoid(y_1) * (1 + y_1 * (1 - torch.sigmoid(y_1))) * y_2,
            g * F.silu(y_1),
        ),
        -1,
    )


@torch.compile
def weighted_swiglu_back(g, y, weights):
    """Compute input and weight gradients for weighted SwiGLU."""

    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = swiglu_back(g * weights, y)
    # precison of w may be higher than y and g, so we need to cast g to w_dtype
    weights_grad = swiglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


@torch.compile
def geglu(y):
    """GEGLU activation function.
    Splits the input in half along the last dimension and applies:
        GEGLU(y) = GELU_tanh(y_gate) * y_up

    Used by Gemma4 MoE expert layers (hidden_activation="gelu_pytorch_tanh").
    """
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.gelu(y_1, approximate="tanh") * y_2


@torch.compile
def weighted_geglu(y, weights):
    """Apply GEGLU activation and token-wise routing weights."""

    dtype = y.dtype
    res = geglu(y) * weights
    return res.to(dtype)


@torch.compile
def geglu_back(g, y):
    """Compute the input gradient for tanh-approximated GEGLU activation."""

    y_1, y_2 = torch.chunk(y, 2, -1)
    # d/dx gelu_tanh(x) ≈ 0.5*(1+tanh(k)) + 0.5*x*(1-tanh(k)^2)*dk/dx
    # where k = sqrt(2/pi)*(x + 0.044715*x^3)
    # Use autograd-friendly form via F.gelu for the gate gradient.
    gelu_y1 = F.gelu(y_1, approximate="tanh")
    # Numerical gradient of tanh-approximated GELU
    SQRT_2_OVER_PI = 0.7978845608028654
    COEFF = 0.044715
    inner = SQRT_2_OVER_PI * (y_1 + COEFF * y_1.pow(3))
    tanh_inner = torch.tanh(inner)
    dtanh = 1.0 - tanh_inner.pow(2)
    dinner = SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * y_1.pow(2))
    gelu_grad = 0.5 * (1.0 + tanh_inner) + 0.5 * y_1 * dtanh * dinner
    return torch.cat(
        (
            g * gelu_grad * y_2,
            g * gelu_y1,
        ),
        -1,
    )


@torch.compile
def weighted_geglu_back(g, y, weights):
    """Compute input and weight gradients for weighted GEGLU."""

    input_dtype = y.dtype
    w_dtype = weights.dtype
    input_grad = geglu_back(g * weights, y)
    weights_grad = geglu(y) * g.to(w_dtype)
    weights_grad = torch.sum(weights_grad, dim=-1, keepdim=True)
    return input_grad.to(input_dtype), weights_grad.to(w_dtype)


class WeightedGEGLUFunction(torch.autograd.Function):
    """Autograd function for token-wise weighted GEGLU."""

    @staticmethod
    def forward(ctx, input, weights, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_geglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = weighted_geglu_back(grad_output, input, weights)
        return tmp, wgrad, None


def weighted_bias_geglu_impl(input, weights, fp8_input_store=False):
    """Token-wise-weighted bias GEGLU fusion (tanh-approximated GELU gating)."""
    ori_shape = input.shape
    if len(ori_shape) > 1:
        input = input.view(-1, ori_shape[-1])

    output = WeightedGEGLUFunction.apply(input, weights, fp8_input_store)

    return output if len(ori_shape) <= 2 else output.view(ori_shape[0], ori_shape[1], -1)


class WeightedSwiGLUFunction(torch.autograd.Function):
    """Autograd function for token-wise weighted SwiGLU."""

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weights, fp8_input_store):
        input_for_backward = input.to(torch.float8_e4m3fn) if fp8_input_store else input
        ctx.save_for_backward(input_for_backward, weights)
        ctx.ori_input_dtype = input.dtype
        ctx.fp8_input_store = fp8_input_store
        return weighted_swiglu(input, weights)

    @staticmethod
    def backward(ctx, grad_output):
        input, weights = ctx.saved_tensors
        input = input.to(ctx.ori_input_dtype) if ctx.fp8_input_store else input
        tmp, wgrad = weighted_swiglu_back(grad_output, input, weights)
        return tmp, wgrad, None


def weighted_bias_swiglu_impl(input, weights, fp8_input_store=False):
    """
    Token-wise-weighted bias swiglu fusion.
    """
    ori_shape = input.shape
    # assert len(ori_shape) in [2, 3]
    if len(ori_shape) > 1:
        input = input.view(-1, ori_shape[-1])

    output = WeightedSwiGLUFunction.apply(input, weights, fp8_input_store)

    return output if len(ori_shape) <= 2 else output.view(ori_shape[0], ori_shape[1], -1)


@torch.compile
def quick_gelu(y: torch.Tensor, alpha: float = 1.702) -> torch.Tensor:
    """Sigmoid approximation of gelu"""
    return y * torch.sigmoid(alpha * y)


@torch.compile
def quick_geglu(y: torch.Tensor, linear_offset: float = 0.0) -> torch.Tensor:
    """Performs Quick-GELU-based GEGLU activation : quick_gelu(y1) * (y2 + offset).

    Args:
        y: Input tensor split into two halves on the last dimension.
        linear_offset: Optional linear offset added to the second half before gating.

    Returns:
        Tensor after applying the GEGLU activation.
    """
    y_1, y_2 = torch.chunk(y, 2, dim=-1)
    return quick_gelu(y_1) * (y_2 + linear_offset)


# gradient of sigmoid approximation of gelu
@torch.compile
def quick_geglu_back(g, y, linear_offset: float = 0.0) -> torch.Tensor:
    """Compute the input gradient for Quick-GEGLU activation."""

    y_1, y_2 = torch.chunk(y, 2, -1)
    sigmoid_out = torch.sigmoid(1.702 * y_1)
    dy_1 = g * sigmoid_out * (1 + 1.702 * y_1 * (1 - sigmoid_out)) * (y_2 + linear_offset)
    dy_2 = g * y_1 * sigmoid_out
    return torch.cat((dy_1, dy_2), -1)


# ---------------- Weighted Bias Quick-GEGLU helpers -----------------


class MoEAuxLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for auxiliary loss."""

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the aux_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The auxiliary loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Pin aux_loss dtype so the AC saved/recomputed metadata cannot diverge
        # regardless of upstream casts. The backward path uses torch.ones_like(aux_loss)
        # only for shape, so dtype here is behaviorally invisible to gradient flow.
        ctx.save_for_backward(aux_loss.float())
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for auxiliary loss..

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled auxiliary loss
                                               gradient.
        """
        (aux_loss,) = ctx.saved_tensors
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=aux_loss.device)
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad
