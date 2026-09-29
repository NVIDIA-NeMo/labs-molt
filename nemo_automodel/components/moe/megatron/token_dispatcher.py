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

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Literal, Tuple

import torch
from torch import nn

from .fused_a2a import (
    fused_combine,
    fused_dispatch,
    hybrid_ep_combine,
    hybrid_ep_dispatch,
    set_deepep_num_sms,
    set_uccl_num_sms,
    uccl_fused_combine,
    uccl_fused_dispatch,
)
from .fused_indices_converter import (
    fused_indices_to_multihot,
)
from .moe_utils import (
    permute,
    unpermute,
)

""" We use the following notation throughout this file:
     H: hidden size
     B: micro batch size
     S: sequence length
     TP: tensor model parallel size
     EP: expert model parallel size
     num_local_tokens: S/TP*B
     num_global_tokens: num_local_tokens*TP*EP
"""


class _DispatchManager(ABC):
    """
    A manager class to handle dispatch and combine processes for MoE models.

    DispatcherManager handles token dispatching according to the routing_map of format
    [num_local_tokens, world_size, num_instances]. The routing_map is a 3D tensor where each
    element indicates whether a token should be sent to a specific rank.

    num_instances is the maximum number of tokens instances dispatched into a target rank, it
    can be the number of local experts, or the size of sub_group.
    """

    @abstractmethod
    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        """Set up metadata of routing_map and probs."""
        pass

    @abstractmethod
    def dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Dispatch the hidden_states according to the routing_map."""
        pass

    @abstractmethod
    def combine(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Combine the hidden_states after expert processing."""
        pass

    @abstractmethod
    def get_dispatched_metadata(self) -> torch.Tensor:
        """Get the metadata of the dispatched hidden_states."""
        pass

    @abstractmethod
    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the permuted hidden states by instances."""
        pass

    @abstractmethod
    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Get the restored hidden states by instances."""
        pass


class _DeepepManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    DeepEP backend. See https://github.com/deepseek-ai/deepep for more details.

    The workflow of the DeepEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Use fused kernel to permute tokens and perform all-to-all communication in single step
    (3) get_permuted_hidden_states_by_instances():
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
    (4) get_restored_hidden_states_by_instances():
        - Reverse permutation using fused kernel
    (5) combine():
        - Reverse process using fused kernel to unpermute and perform all-to-all in single step

    This implementation uses fused communication kernels (fused_dispatch/fused_combine) that
    combine permutation and communication operations for improved efficiency compared to
    separate permute+alltoall steps.
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        router_topk: int,
        permute_fusion: bool = False,
        capacity_factor: float | None = None,
        num_experts: int | None = None,
        num_local_experts: int | None = None,
        router_dtype: str | None = None,
        moe_router_expert_pad_multiple: int | None = None,
        _dispatch_fn=None,
        _combine_fn=None,
    ):
        self.group = group
        self.router_topk = router_topk
        self.capacity_factor = capacity_factor
        self.permute_fusion = permute_fusion
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.router_dtype = router_dtype
        self.moe_router_expert_pad_multiple = moe_router_expert_pad_multiple

        # Metadata
        self.token_indices: torch.Tensor | None = None
        self.token_probs: torch.Tensor | None = None
        # Handle used for combine operation
        self.handle = None

        self._fused_dispatch = _dispatch_fn if _dispatch_fn is not None else fused_dispatch
        self._fused_combine = _combine_fn if _combine_fn is not None else fused_combine

        if self._fused_dispatch is None:
            raise ImportError(
                "Neither DeepEP nor UCCL-EP is installed. "
                "Install DeepEP from https://github.com/deepseek-ai/deepep "
                "or UCCL-EP via: bash scripts/setup_uccl_ep.sh"
            )

    def setup_metadata(self, num_local_tokens: int, probs: torch.Tensor):
        """
        Process routing map and probabilities to prepare dispatch metadata
        """
        probs = probs.reshape(num_local_tokens, self.num_experts)
        # Convert the format of routing map from multihot to indices.
        self.token_probs, self.token_indices = torch.topk(probs, self.router_topk, dim=-1)
        # Mask the indices of dropped tokens with -1
        if self.capacity_factor is not None:
            mask = self.token_probs == 0
            self.token_indices = self.token_indices.masked_fill(mask, -1)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        """
        Dispatch the hidden_states
        """
        # DeepEP only supports float32 probs
        if self.token_probs.dtype != torch.float32:
            if self.token_probs.dtype in [torch.bfloat16, torch.float16]:
                # print("DeepEP only supports float32 probs, please set --moe-router-dtype=fp32")
                # TODO: remove this
                pass
            self.token_probs = self.token_probs.float()  # downcast or upcast
        (
            hidden_states,
            dispatched_indices,
            dispatched_probs,
            num_tokens_per_expert,
            handle,
        ) = self._fused_dispatch(
            hidden_states,
            self.token_indices,
            self.token_probs,
            self.num_experts,
            self.group,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        self.handle = handle
        self.tokens_per_expert = num_tokens_per_expert
        self.dispatched_indices = dispatched_indices
        self.dispatched_probs = dispatched_probs

        return hidden_states

    def _indices_to_multihot(self, indices, probs):
        """
        Converts a tensor of indices to a multihot vector.

        Args:
            indices (torch.Tensor): [num_tokens, topk] token indices, where -1 means masked out.
            probs (torch.Tensor): [num_tokens, topk] token probabilities.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - routing_map: Multihot vector.
                - probs: Multihot probabilities.
        """
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_local_experts),
            dtype=torch.long,
            device=indices.device,
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_local_experts),
            dtype=torch.float,
            device=indices.device,
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(mask.sum(dim=1))
        multihot_routing_map[row_indices, valid_indices] = 1
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map.bool(), multihot_probs

    def get_dispatched_metadata(self) -> torch.Tensor:
        return self.dispatched_indices, self.dispatched_probs

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        """
        Get the number of tokens per expert.
        """
        return self.tokens_per_expert

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = False,
        allocate_on_comm_stream: bool = False,
    ) -> torch.Tensor:
        """
        Reverse process using fused kernel to unpermute and perform all-to-all in single step
        """
        hidden_states, _ = self._fused_combine(
            hidden_states,
            self.group,
            self.handle,
            async_finish=async_finish,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        # Release the handle after combine operation
        self.handle = None
        return hidden_states

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        - Convert routing map and probabilities to multihot format
        - Permute tokens using fused kernel
        """
        if self.permute_fusion:
            self.dispatched_routing_map, self.dispatched_probs = fused_indices_to_multihot(
                self.dispatched_indices,
                self.dispatched_probs,
                self.num_local_experts,
            )
        else:
            self.dispatched_routing_map, self.dispatched_probs = self._indices_to_multihot(
                self.dispatched_indices, self.dispatched_probs
            )
        if self.moe_router_expert_pad_multiple:
            with torch.cuda.nvtx.range("pad_routing_map"):
                from megatron.core.transformer.moe.moe_utils import pad_routing_map

                self.dispatched_routing_map = pad_routing_map(
                    self.dispatched_routing_map, self.moe_router_expert_pad_multiple
                )
            # self.tokens_per_expert = self.dispatched_routing_map.sum(dim=0)
            self.tokens_per_expert = (
                torch.ceil(self.tokens_per_expert / self.moe_router_expert_pad_multiple)
                * self.moe_router_expert_pad_multiple
            )
            self.tokens_per_expert = self.tokens_per_expert.long()

        self.hidden_shape_before_permute = hidden_states.shape
        assert self.dispatched_probs.dtype == torch.float32, "DeepEP only supports float32 probs"
        hidden_states, permuted_probs, self.reversed_mapping_for_combine = permute(
            hidden_states,
            self.dispatched_routing_map,
            probs=self.dispatched_probs,
            num_out_tokens=self.tokens_per_expert.sum().item(),
            fused=self.permute_fusion,
        )
        if self.router_dtype == "fp64":
            permuted_probs = permuted_probs.to(torch.float64)
        return hidden_states, permuted_probs

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Restore the hidden states to their original ordering before expert processing
        """
        hidden_states = unpermute(
            hidden_states,
            self.reversed_mapping_for_combine,
            restore_shape=self.hidden_shape_before_permute,
            routing_map=self.dispatched_routing_map,
            fused=self.permute_fusion,
        )
        return hidden_states


class _HybridEPMetadataProcessor(nn.Module):
    """Fixed-shape conversion from top-k metadata to HybridEP metadata."""

    def __init__(self, *, num_experts: int, permute_fusion: bool):
        super().__init__()
        self.num_experts = num_experts
        self.permute_fusion = permute_fusion

    def forward(self, token_indices: torch.Tensor, token_probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert compact top-k indices and probabilities to multihot tensors."""
        if self.permute_fusion:
            return fused_indices_to_multihot(token_indices, token_probs, self.num_experts)

        batch_size = token_indices.shape[0]
        routing_map = torch.zeros(
            (batch_size, self.num_experts),
            dtype=torch.bool,
            device=token_indices.device,
        )
        multihot_probs = torch.zeros(
            (batch_size, self.num_experts),
            dtype=torch.float,
            device=token_indices.device,
        )
        mask = token_indices != -1
        valid_indices = token_indices[mask]
        row_indices = torch.arange(batch_size, device=token_indices.device).repeat_interleave(mask.sum(dim=1))
        routing_map[row_indices, valid_indices] = True
        multihot_probs[row_indices, valid_indices] = token_probs[mask]
        return routing_map, multihot_probs


# DeepEP's hybrid-ep metadata allgather asserts bytes_per_rank % 16 == 0 on a
# 4-byte-per-token array, so per-rank token counts must be multiples of 4.
_HYBRIDEP_TOKEN_ALIGNMENT = 4
# Benchmark A/B switch for the static-routing pad-size pin (default on); see _HybridEPManager.dispatch.
_STATIC_ROUTING_PAD_PIN = os.environ.get("NEMO_STATIC_ROUTING_PAD_PIN", "1") != "0"


class _HybridEPManager(_DispatchManager):
    """
    A manager class to handle fused all-to-all communication processes for MoE models using
    HybridEP backend. See https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep for more details.

    The workflow of the HybridEP dispatcher is:
    (1) setup_metadata(): Process routing map and probabilities to prepare dispatch metadata
    (2) dispatch():
        - Permute tokens for communication, perform all-to-all communication,
        and permute tokens for experts in single step
    (3) combine():
        - Unpermute tokens for communication, perform all-to-all communication,
        and unpermute tokens for attention in single step
    """

    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        num_local_experts: int,
        num_experts: int,
        router_topk: int,
        permute_fusion: bool = False,
        moe_hybridep_num_sms: int = 24,
        benchmark_static_routing: bool = False,
    ):
        self.group = group
        self.num_local_experts = num_local_experts
        self.num_experts = num_experts
        self.router_topk = router_topk
        self.permute_fusion = permute_fusion
        self.moe_hybridep_num_sms = moe_hybridep_num_sms
        # Benchmark-only (TokenDispatcherConfig.moe_benchmark_static_routing):
        # persist num_permuted_tokens across dispatches, see dispatch()/reset.
        self.benchmark_static_routing = benchmark_static_routing
        self.num_permuted_tokens = None
        # Benchmark-only companion pin: the EP-group padded token count (see dispatch()). Under static
        # routing every dispatch pads to the same size, so the per-dispatch MAX all-reduce + host sync
        # that computes it runs once. NEMO_STATIC_ROUTING_PAD_PIN=0 keeps the per-dispatch path (A/B).
        self._static_target_tokens: int | None = None

        # Metadata
        self.token_probs: torch.Tensor | None = None
        self.routing_map: torch.Tensor | None = None
        # Handle used for combine operation
        self.handle = None
        self.pad_multiple = None
        # Set by dispatch() when this rank padded its tokens up to the EP-group
        # maximum; combine() slices the padding back off.
        self.num_unpadded_tokens: int | None = None

        if hybrid_ep_dispatch is None:
            raise ImportError(
                "HybridEP is not installed. Please install HybridEP package from "
                "https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep."
            )

    def setup_metadata(self, routing_map: torch.Tensor, probs: torch.Tensor):
        """Process routing map and probabilities to prepare dispatch metadata."""
        num_tokens = routing_map.shape[0]
        self.routing_map = routing_map.reshape(num_tokens, self.num_experts)
        self.token_probs = probs.reshape(num_tokens, self.num_experts)

    def _indices_to_multihot(self, indices: torch.Tensor, probs: torch.Tensor):
        """Converts a tensor of indices to a multihot vector."""
        batch_size = indices.shape[0]
        multihot_routing_map = torch.zeros(
            (batch_size, self.num_experts),
            dtype=torch.bool,
            device=indices.device,
        )

        multihot_probs = torch.zeros(
            (batch_size, self.num_experts),
            dtype=torch.float,
            device=indices.device,
        )

        mask = indices != -1
        valid_indices = indices[mask]
        row_indices = torch.arange(batch_size, device=indices.device).repeat_interleave(mask.sum(dim=1))
        multihot_routing_map[row_indices, valid_indices] = True
        multihot_probs[row_indices, valid_indices] = probs[mask]
        return multihot_routing_map, multihot_probs

    def setup_metadata_from_indices(self, token_indices: torch.Tensor, token_probs: torch.Tensor):
        """Convert from topk indices format to multihot routing_map format."""
        if self.permute_fusion:
            self.routing_map, self.token_probs = fused_indices_to_multihot(token_indices, token_probs, self.num_experts)
        else:
            self.routing_map, self.token_probs = self._indices_to_multihot(token_indices, token_probs)

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,  # noqa: ARG002 - not supported by HybridEP backend
        allocate_on_comm_stream: bool = True,  # noqa: ARG002 - not supported by HybridEP backend
    ) -> torch.Tensor:
        # Reset num_permuted_tokens to None to avoid reusing cached state from a prior dispatch.
        # This can happen in non-reentrant activation checkpointing mode.
        # Benchmark-only exception: with static routing every dispatch permutes the same
        # count, so reusing the first dispatch's value keeps hybrid_ep_dispatch on its
        # non-blocking path and removes a per-microbatch host wait on a device-side flag.
        if self.benchmark_static_routing and getattr(self, "_static_num_permuted_tokens", None) is not None:
            self.num_permuted_tokens = self._static_num_permuted_tokens
        else:
            self.num_permuted_tokens = None
        if self.token_probs.dtype != torch.float32:
            self.token_probs = self.token_probs.float()

        # HybridEP's fused all-to-all exchanges fixed-extent buffers, so every rank
        # in the EP group must dispatch the same number of tokens, and the kernel's
        # metadata allgather additionally requires that count to be 16-byte aligned
        # (4 tokens). Unequal or unaligned counts (variable-length or packed dynamic
        # batches) abort or deadlock the collective. Pad this rank's tokens up to
        # the aligned group-wide maximum: padded rows route to no expert (all-False
        # routing map), and combine() slices them back off.
        self.num_unpadded_tokens = None
        if torch.distributed.is_initialized() and torch.distributed.get_world_size(self.group) > 1:
            num_tokens = hidden_states.shape[0]
            pin = self.benchmark_static_routing and _STATIC_ROUTING_PAD_PIN
            if pin and self._static_target_tokens is not None and self._static_target_tokens >= num_tokens:
                target_tokens = self._static_target_tokens
            else:
                group_max = torch.tensor(num_tokens, device=hidden_states.device)
                torch.distributed.all_reduce(group_max, op=torch.distributed.ReduceOp.MAX, group=self.group)
                target_tokens = -(-int(group_max) // _HYBRIDEP_TOKEN_ALIGNMENT) * _HYBRIDEP_TOKEN_ALIGNMENT
                if pin:
                    self._static_target_tokens = target_tokens
                    if torch.distributed.get_rank() == 0:
                        logging.getLogger(__name__).info(
                            "benchmark_static_routing: HybridEP pad size pinned at %d tokens "
                            "(per-dispatch EP-group max all-reduce + host sync skipped from now on)",
                            target_tokens,
                        )
            pad_tokens = target_tokens - num_tokens
            if pad_tokens > 0:
                self.num_unpadded_tokens = num_tokens
                hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_tokens))
                self.routing_map = nn.functional.pad(self.routing_map, (0, 0, 0, pad_tokens))
                self.token_probs = nn.functional.pad(self.token_probs, (0, 0, 0, pad_tokens))

        dispatched_hidden, self.dispatched_probs, _, tokens_per_expert, self.handle = hybrid_ep_dispatch(
            x=hidden_states,
            routing_map=self.routing_map,
            probs=self.token_probs,
            group=self.group,
            num_local_experts=self.num_local_experts,
            num_sms_dispatch_api=self.moe_hybridep_num_sms,
            num_sms_combine_api=self.moe_hybridep_num_sms,
            num_permuted_tokens=self.num_permuted_tokens,
            pad_multiple=self.pad_multiple,
        )

        self.tokens_per_expert = tokens_per_expert
        self.num_permuted_tokens = self.tokens_per_expert.sum()
        if self.benchmark_static_routing and getattr(self, "_static_num_permuted_tokens", None) is None:
            self._static_num_permuted_tokens = self.num_permuted_tokens

        return dispatched_hidden

    def combine(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,  # noqa: ARG002 - not supported by HybridEP backend
        allocate_on_comm_stream: bool = True,  # noqa: ARG002 - not supported by HybridEP backend
    ) -> torch.Tensor:
        hidden_states = hybrid_ep_combine(
            x=hidden_states,
            handle=self.handle,
            num_permuted_tokens=self.num_permuted_tokens,
            pad_multiple=self.pad_multiple,
        )
        self.handle = None
        self.num_permuted_tokens = None
        if self.num_unpadded_tokens is not None:
            hidden_states = hidden_states[: self.num_unpadded_tokens]
            self.num_unpadded_tokens = None
        return hidden_states

    def get_dispatched_metadata(self) -> torch.Tensor:
        return None, self.dispatched_probs

    def get_permuted_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states, self.dispatched_probs

    def get_restored_hidden_states_by_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def get_number_of_tokens_per_expert(self) -> torch.Tensor:
        return self.tokens_per_expert


@dataclass
class TokenDispatcherConfig:
    """Configuration for MoE token dispatch and combine backends."""

    moe_enable_deepep: bool = True
    """Enable DeepEP for efficient token dispatching and combine in MoE models."""

    moe_permute_fusion: bool = False
    """Fuse token rearrangement ops during token dispatching."""

    moe_expert_capacity_factor: float | None = None
    """moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token
    will be dropped. The default is None."""

    moe_router_topk: int = 2
    """Number of experts to route to for each token."""

    moe_router_expert_pad_multiple: int | None = None
    """Number of tokens to pad to a multiple of for each expert."""

    num_moe_experts: int = 64
    """Number of experts to use for MoE layer. When set, it replaces MLP with MoE layer. Set to None
    for no MoE."""

    moe_router_dtype: str = "fp32"
    """Data type for routing and expert output weighted averaging. Using fp32 or fp64 can
    improve stability especially when the number of experts is large (e.g. finegrained-moe).
    None means no changes for dtype."""

    moe_flex_dispatcher_backend: Literal["deepep", "hybridep", "uccl_ep"] = "deepep"
    """Backend for the flex token dispatcher. Options: 'deepep', 'hybridep', or 'uccl_ep'."""

    moe_deepep_num_sms: int = 20
    """Number of SMs to use for DeepEP backend."""

    moe_hybridep_num_sms: int = 24
    """Number of SMs to use for HybridEP dispatch and combine APIs."""

    moe_share_token_dispatcher: bool = True
    """Share one communication manager instance across MoE layers for the configured backend."""

    moe_deepep_async_dispatch: bool = False
    """Use asynchronous DeepEP/UCCL-EP dispatch/combine and communication-stream allocations."""

    moe_benchmark_static_routing: bool = False
    """Benchmark-only (mirrors BackendConfig.benchmark_static_routing, validated there):
    routing is forced-balanced with no noise, so every dispatch permutes the same token
    count. Persist num_permuted_tokens across dispatches to keep HybridEP on its
    non-blocking size path instead of waiting on a device-side count per microbatch."""


class MoEFlexTokenDispatcher:
    """
    Flex token dispatcher supporting DeepEP, HybridEP, and UCCL-EP backends.
    """

    shared_deepep_manager: _DeepepManager = None
    shared_hybridep_manager: _HybridEPManager = None
    shared_uccl_manager: _DeepepManager = None

    def __init__(
        self,
        num_local_experts: int,
        local_expert_indices: List[int],
        config: TokenDispatcherConfig,
        ep_group: torch.distributed.ProcessGroup,
    ):
        """
        Initialize the Flex token dispatcher.

        Args:
            num_local_experts (int): Number of local experts on the current device.
            local_expert_indices (List[int]): Indices of local experts on the current device.
            config (TokenDispatcherConfig): Configuration for the transformer model.
            ep_group (torch.distributed.ProcessGroup): Process group for MoE operations.
        """
        self.config = config
        self.shared_experts = None

        self.group = ep_group
        self.ep_size = ep_group.size()

        self.tp_size = 1  # TP is not used

        self.num_local_experts = num_local_experts
        self.local_expert_indices = local_expert_indices
        self.hybridep_metadata_processor: _HybridEPMetadataProcessor | None = None
        assert self.tp_size * self.ep_size > 1, "Flex token dispatcher requires TPxEP > 1"

        backend = self.config.moe_flex_dispatcher_backend

        if backend == "uccl_ep":
            if set_uccl_num_sms is not None:
                set_uccl_num_sms(self.config.moe_deepep_num_sms)
            dispatch_fn = uccl_fused_dispatch
            combine_fn = uccl_fused_combine
            manager_kwargs = dict(
                group=ep_group,
                router_topk=self.tp_size * self.config.moe_router_topk,
                permute_fusion=self.config.moe_permute_fusion,
                capacity_factor=self.config.moe_expert_capacity_factor,
                num_experts=self.tp_size * self.config.num_moe_experts,
                num_local_experts=self.num_local_experts,
                router_dtype=self.config.moe_router_dtype,
                moe_router_expert_pad_multiple=self.config.moe_router_expert_pad_multiple,
                _dispatch_fn=dispatch_fn,
                _combine_fn=combine_fn,
            )
            if self.config.moe_share_token_dispatcher:
                if MoEFlexTokenDispatcher.shared_uccl_manager is None:
                    MoEFlexTokenDispatcher.shared_uccl_manager = _DeepepManager(**manager_kwargs)
                self._comm_manager = MoEFlexTokenDispatcher.shared_uccl_manager
            else:
                self._comm_manager = _DeepepManager(**manager_kwargs)
        elif backend == "deepep":
            if set_deepep_num_sms is not None:
                set_deepep_num_sms(self.config.moe_deepep_num_sms)
            if self.config.moe_share_token_dispatcher:
                if MoEFlexTokenDispatcher.shared_deepep_manager is None:
                    MoEFlexTokenDispatcher.shared_deepep_manager = _DeepepManager(
                        group=ep_group,
                        router_topk=self.tp_size * self.config.moe_router_topk,
                        permute_fusion=self.config.moe_permute_fusion,
                        capacity_factor=self.config.moe_expert_capacity_factor,
                        num_experts=self.tp_size * self.config.num_moe_experts,
                        num_local_experts=self.num_local_experts,
                        router_dtype=self.config.moe_router_dtype,
                        moe_router_expert_pad_multiple=self.config.moe_router_expert_pad_multiple,
                    )
                self._comm_manager = MoEFlexTokenDispatcher.shared_deepep_manager
            else:
                self._comm_manager = _DeepepManager(
                    group=ep_group,
                    router_topk=self.tp_size * self.config.moe_router_topk,
                    permute_fusion=self.config.moe_permute_fusion,
                    capacity_factor=self.config.moe_expert_capacity_factor,
                    num_experts=self.tp_size * self.config.num_moe_experts,
                    num_local_experts=self.num_local_experts,
                    router_dtype=self.config.moe_router_dtype,
                    moe_router_expert_pad_multiple=self.config.moe_router_expert_pad_multiple,
                )
        elif backend == "hybridep":
            if self.config.moe_share_token_dispatcher:
                if MoEFlexTokenDispatcher.shared_hybridep_manager is None:
                    MoEFlexTokenDispatcher.shared_hybridep_manager = _HybridEPManager(
                        group=ep_group,
                        num_local_experts=self.num_local_experts,
                        num_experts=self.tp_size * self.config.num_moe_experts,
                        router_topk=self.tp_size * self.config.moe_router_topk,
                        permute_fusion=self.config.moe_permute_fusion,
                        moe_hybridep_num_sms=self.config.moe_hybridep_num_sms,
                        benchmark_static_routing=self.config.moe_benchmark_static_routing,
                    )
                self._comm_manager = MoEFlexTokenDispatcher.shared_hybridep_manager
            else:
                self._comm_manager = _HybridEPManager(
                    group=ep_group,
                    num_local_experts=self.num_local_experts,
                    num_experts=self.tp_size * self.config.num_moe_experts,
                    router_topk=self.tp_size * self.config.moe_router_topk,
                    permute_fusion=self.config.moe_permute_fusion,
                    moe_hybridep_num_sms=self.config.moe_hybridep_num_sms,
                    benchmark_static_routing=self.config.moe_benchmark_static_routing,
                )
            self.hybridep_metadata_processor = _HybridEPMetadataProcessor(
                num_experts=self.tp_size * self.config.num_moe_experts,
                permute_fusion=self.config.moe_permute_fusion,
            )
        else:
            raise ValueError(
                f"Invalid backend: {backend}. Please set moe_flex_dispatcher_backend='deepep', 'hybridep', or 'uccl_ep'"
            )

    def _initialize_metadata(self, num_local_tokens: int, probs: torch.Tensor) -> torch.Tensor:
        """
        Initialize the routing map and probs to a unified format covering the TPxEP group.
        This design decouples the communication group from underlying model parallelism groups,
        such that the communication strategy of tokens can be agnostic of TP size and EP size.
        """
        world_size = self.tp_size * self.ep_size
        probs = (
            probs.reshape(num_local_tokens, self.ep_size, 1, self.num_local_experts)
            .expand(-1, -1, self.tp_size, -1)
            .reshape(num_local_tokens, world_size, self.num_local_experts)
        ).contiguous()
        return probs

    def dispatch_preprocess2(
        self,
        hidden_states: torch.Tensor,
        num_local_tokens: int,
        token_probs: torch.Tensor,
        token_indices: torch.Tensor,
    ):
        """
        Preprocesses the hidden states and routing information before dispatching tokens to experts.

        For DeepEP backend: uses token_indices and token_probs directly.
        For HybridEP backend: converts token_indices to routing_map (multihot format).
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        if isinstance(self._comm_manager, _HybridEPManager):
            assert self.hybridep_metadata_processor is not None
            routing_map, multihot_probs = self.hybridep_metadata_processor(token_indices, token_probs)
            self._comm_manager.routing_map = routing_map
            self._comm_manager.token_probs = multihot_probs
        else:
            self._comm_manager.token_probs = token_probs
            self._comm_manager.token_indices = token_indices

        return hidden_states, self._comm_manager.token_probs

    def dispatch_preprocess(self, hidden_states: torch.Tensor, num_local_tokens: int, probs: torch.Tensor):
        """
        Preprocesses the hidden states and routing information before dispatching tokens to experts.
        Args:
            hidden_states (torch.Tensor): Input hidden states to be processed
            num_local_tokens (int): Number of tokens to be processed
            probs (torch.Tensor): Routing probabilities for each token-expert pair

        Returns:
            Tuple containing:
            - torch.Tensor: Reshaped hidden states
            - torch.Tensor: Token probabilities from the communication manager
            - None: Placeholder for compatibility
        """
        self.hidden_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_shape[-1])

        # Initialize metadata
        probs = self._initialize_metadata(num_local_tokens=num_local_tokens, probs=probs)

        self._comm_manager.setup_metadata(num_local_tokens=num_local_tokens, probs=probs)
        return hidden_states, self._comm_manager.token_probs

    def dispatch_all_to_all(
        self,
        hidden_states: torch.Tensor,
        probs: torch.Tensor = None,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Performs all-to-all communication to dispatch tokens across expert parallel ranks.
        """
        return (
            self._comm_manager.dispatch(hidden_states, async_finish, allocate_on_comm_stream),
            self._comm_manager.dispatched_probs,
        )

    def dispatch_postprocess(self, hidden_states: torch.Tensor):
        """
        Post-processes the dispatched hidden states after all-to-all communication.

        This method retrieves the permuted hidden states by experts, calculates the number of tokens
        per expert, and returns the processed data ready for expert processing.
        """
        global_input_tokens, permuted_probs = self._comm_manager.get_permuted_hidden_states_by_experts(hidden_states)
        tokens_per_expert = self._comm_manager.get_number_of_tokens_per_expert()
        return global_input_tokens, tokens_per_expert, permuted_probs

    def token_permutation(
        self, hidden_states: torch.Tensor, num_local_tokens: int, probs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Permutes tokens according to probs and dispatches them to experts.

        This method implements the token permutation process in three steps:
        1. Preprocess the hidden states
        2. Perform all-to-all communication to dispatch tokens
        3. Post-process the dispatched tokens for expert processing
        """
        hidden_states, _ = self.dispatch_preprocess(
            hidden_states=hidden_states, num_local_tokens=num_local_tokens, probs=probs
        )
        async_dispatch = isinstance(self._comm_manager, _DeepepManager) and self.config.moe_deepep_async_dispatch
        hidden_states, _ = self.dispatch_all_to_all(
            hidden_states,
            async_finish=async_dispatch,
            allocate_on_comm_stream=async_dispatch,
        )
        global_input_tokens, tokens_per_expert, permuted_probs = self.dispatch_postprocess(hidden_states)

        return global_input_tokens, tokens_per_expert, permuted_probs

    def token_permutation2(
        self,
        hidden_states: torch.Tensor,
        num_local_tokens: int,
        token_probs: torch.Tensor,
        token_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Permutes tokens according to probs and dispatches them to experts.

        This method implements the token permutation process in three steps:
        1. Preprocess the hidden states
        2. Perform all-to-all communication to dispatch tokens
        3. Post-process the dispatched tokens for expert processing
        """
        hidden_states, _ = self.dispatch_preprocess2(
            hidden_states=hidden_states,
            num_local_tokens=num_local_tokens,
            token_probs=token_probs,
            token_indices=token_indices,
        )
        async_dispatch = isinstance(self._comm_manager, _DeepepManager) and self.config.moe_deepep_async_dispatch
        hidden_states, _ = self.dispatch_all_to_all(
            hidden_states,
            async_finish=async_dispatch,
            allocate_on_comm_stream=async_dispatch,
        )
        global_input_tokens, tokens_per_expert, permuted_probs = self.dispatch_postprocess(hidden_states)

        return global_input_tokens, tokens_per_expert, permuted_probs

    def combine_preprocess(self, hidden_states: torch.Tensor):
        """
        Pre-processes the hidden states before combining them after expert processing.

        This method restores the hidden states to their original ordering before expert processing
        by using the communication manager's restoration function.
        """
        hidden_states = self._comm_manager.get_restored_hidden_states_by_experts(hidden_states)
        return hidden_states

    def combine_all_to_all(
        self,
        hidden_states: torch.Tensor,
        async_finish: bool = True,
        allocate_on_comm_stream: bool = True,
    ):
        """
        Performs all-to-all communication to combine tokens after expert processing.
        """
        return self._comm_manager.combine(hidden_states, async_finish, allocate_on_comm_stream)

    def combine_postprocess(self, hidden_states: torch.Tensor):
        """
        Post-processes the combined hidden states after all-to-all communication.

        This method reshapes the combined hidden states to match the original input shape.
        """
        return hidden_states.view(self.hidden_shape)

    def token_unpermutation(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Reverses the token permutation process to restore the original token order.

        This method implements the token unpermutation process in three steps:
        1. Pre-process the hidden states to restore their original ordering
        2. Perform all-to-all communication to combine tokens
        3. Post-process the combined tokens to match the original input shape
        """
        hidden_states = self.combine_preprocess(hidden_states)
        async_combine = isinstance(self._comm_manager, _DeepepManager) and self.config.moe_deepep_async_dispatch
        hidden_states = self.combine_all_to_all(
            hidden_states,
            async_finish=async_combine,
            allocate_on_comm_stream=async_combine,
        )
        hidden_states = self.combine_postprocess(hidden_states)

        return hidden_states
