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

"""Gemma4-owned tensor-parallel plan and FSDP2 strategy registration."""

from __future__ import annotations

import types
import warnings
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.parallel_styles import ReplicatedWithGradAllReduce


class _ReduceFromTensorParallelRegion(torch.autograd.Function):
    """Sum local TP values in forward and leave replicated gradients local."""

    @staticmethod
    def forward(ctx, local_output: torch.Tensor, process_group) -> torch.Tensor:
        """Reduce a local contribution Tensor into one replicated Tensor."""
        del ctx
        torch.cuda.synchronize(local_output.device)
        output = local_output.clone()
        dist.all_reduce(output, group=process_group)
        # The eager HF block consumes a plain Tensor outside DTensor dispatch.
        # Bracket the collective to keep ProcessGroupNCCL's comm stream from
        # racing the producing matmul or following residual/norm operation.
        torch.cuda.synchronize(output.device)
        return output.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Pass replicated gradients unchanged to each owning TP shard."""
        del ctx
        return grad_output, None


def _gemma4_rowwise_linear_forward(module: nn.Linear, input_: torch.Tensor) -> torch.Tensor:
    """Apply one Gemma4 row-parallel linear and synchronously reduce its output.

    Args:
        module: Linear whose weight is a DTensor sharded on input features.
        input_: Local Tensor shaped ``[..., input_features / tp_size]``.

    Returns:
        Local Tensor shaped ``[..., output_features]``, replicated across the
        Gemma4 TP group after summing the input-feature contributions.
    """
    if isinstance(input_, DTensor):
        input_ = input_.to_local()
    local_output = F.linear(input_, module.weight.to_local(), None)
    output = _ReduceFromTensorParallelRegion.apply(local_output, module._gemma4_tp_group)
    if module.bias is not None:
        output = output + module.bias.to_local()
    # Keep the most recent reduced output alive until the next row-parallel
    # projection. PyTorch 2.10 may otherwise recycle the custom-autograd
    # output's storage while an eager HF consumer is still queued.
    module._gemma4_tp_output_holder[0] = output
    return output


class _Gemma4RowwiseParallel(ParallelStyle):
    """Shard a Gemma4 linear on input features with a synchronous reduction."""

    def __init__(self, output_holder: list[torch.Tensor | None]) -> None:
        super().__init__()
        self.output_holder = output_holder

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Partition ``[output, input]`` weight as ``Shard(1)`` and install the local linear."""
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Gemma4 row parallelism requires nn.Linear, got {type(module).__name__}.")
        module.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(
                    module.weight,
                    device_mesh,
                    (Shard(1),),
                    src_data_rank=self.src_data_rank,
                ),
                requires_grad=module.weight.requires_grad,
            ),
        )
        if module.bias is not None:
            module.register_parameter(
                "bias",
                nn.Parameter(
                    distribute_tensor(
                        module.bias,
                        device_mesh,
                        (Replicate(),),
                        src_data_rank=self.src_data_rank,
                    ),
                    requires_grad=module.bias.requires_grad,
                ),
            )
        module._gemma4_tp_group = device_mesh.get_group()
        module._gemma4_tp_output_holder = self.output_holder
        module.forward = types.MethodType(_gemma4_rowwise_linear_forward, module)
        return module


def _gemma4_vocab_parallel_forward(module: nn.Embedding, input_ids: torch.Tensor) -> torch.Tensor:
    """Look up replicated token ids in one Gemma4 vocabulary shard.

    Args:
        module: Gemma4 embedding whose weight is a DTensor sharded on vocabulary.
        input_ids: Replicated integer Tensor shaped ``[batch, sequence]``.

    Returns:
        Local Tensor shaped ``[batch, sequence, embedding]``, replicated across
        the Gemma4 TP group after summing the disjoint vocabulary contributions.
    """
    local_ids = input_ids - module._gemma4_vocab_start
    outside_local_vocab = (local_ids < 0) | (local_ids >= module._gemma4_vocab_size)
    local_ids = local_ids.masked_fill(outside_local_vocab, 0)
    local_output = F.embedding(
        local_ids,
        module.weight.to_local(),
        module._gemma4_local_padding_idx,
        module.max_norm,
        module.norm_type,
        module.scale_grad_by_freq,
        module.sparse,
    )
    local_output = local_output.masked_fill(outside_local_vocab.unsqueeze(-1), 0)
    output = _ReduceFromTensorParallelRegion.apply(local_output, module._gemma4_tp_group)
    embed_scale = getattr(module, "embed_scale", None)
    return output if embed_scale is None else output * embed_scale.to(output.dtype)


class _Gemma4VocabParallelEmbedding(ParallelStyle):
    """Shard a Gemma4 embedding by vocabulary without DTensor ``MaskPartial``.

    The input ids are replicated ``[batch, sequence]``. The weight changes from
    ``[vocab, embedding]`` to a DTensor with ``Shard(0)``; the forward returns a
    local, replicated ``[batch, sequence, embedding]`` Tensor.
    """

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        """Partition one Gemma4 embedding weight and install its local lookup."""
        if not isinstance(module, nn.Embedding):
            raise TypeError(f"Gemma4 vocab parallelism requires nn.Embedding, got {type(module).__name__}.")

        vocab_size = module.num_embeddings
        tp_size = device_mesh.size()
        tp_rank = device_mesh.get_local_rank()
        chunk_size, remainder = divmod(vocab_size, tp_size)
        local_size = chunk_size + int(tp_rank < remainder)
        local_start = tp_rank * chunk_size + min(tp_rank, remainder)

        module.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(
                    module.weight,
                    device_mesh,
                    (Shard(0),),
                    src_data_rank=self.src_data_rank,
                ),
                requires_grad=module.weight.requires_grad,
            ),
        )
        module._gemma4_vocab_start = local_start
        module._gemma4_vocab_size = local_size
        module._gemma4_tp_group = device_mesh.get_group()
        padding_idx = module.padding_idx
        module._gemma4_local_padding_idx = (
            padding_idx - local_start
            if padding_idx is not None and local_start <= padding_idx < local_start + local_size
            else None
        )
        module.forward = types.MethodType(_gemma4_vocab_parallel_forward, module)
        return module


def _gemma4_tp_plan(model: nn.Module, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Return the TP plan for the concrete Gemma4 checkpoint variant.

    E2B/E4B add a packed per-layer embedding table that is absent from 31B.
    Row-sharding that table by vocabulary is important: at E4B dimensions it is
    roughly 2.8 billion parameters, so leaving it replicated defeats much of
    the memory benefit of TP.

    Args:
        model: Gemma4 conditional-generation model whose text config selects
            the dense variant.
        sequence_parallel: Whether sequence parallelism was requested. Gemma4
            does not currently support it, so the request is ignored.

    Returns:
        Mapping from Gemma4 module paths to PyTorch parallel styles.
    """
    if sequence_parallel:
        warnings.warn(
            "sequence_parallel=True is not yet supported for Gemma4 and will be ignored.",
            stacklevel=2,
        )

    model_prefix = "model.language_model"
    output_holder: list[torch.Tensor | None] = [None]
    plan: dict[str, ParallelStyle] = {
        f"{model_prefix}.embed_tokens": _Gemma4VocabParallelEmbedding(),
        f"{model_prefix}.layers.*.self_attn.q_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.k_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.q_norm": ReplicatedWithGradAllReduce(),
        f"{model_prefix}.layers.*.self_attn.k_norm": ReplicatedWithGradAllReduce(),
        f"{model_prefix}.layers.*.self_attn.v_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.self_attn.o_proj": _Gemma4RowwiseParallel(output_holder),
        f"{model_prefix}.layers.*.mlp.up_proj": ColwiseParallel(),
        f"{model_prefix}.layers.*.mlp.gate_proj": ColwiseParallel(),
        "lm_head": ColwiseParallel(output_layouts=Shard(-1), use_local_output=False),
    }

    plan[f"{model_prefix}.layers.*.mlp.down_proj"] = _Gemma4RowwiseParallel(output_holder)

    text_config = model.config.text_config
    if text_config.hidden_size_per_layer_input:
        # E-series checkpoints tie lm_head to embed_tokens. Parallelizing both
        # modules creates two DTensor Parameters; re-tying them afterward leaves
        # the lm_head Colwise hook bound to the discarded parameterization. Keep
        # one row-sharded embedding Parameter and let the model-owned forward
        # compute its vocab-sharded logits directly from that alias.
        plan.pop("lm_head")
        plan[f"{model_prefix}.embed_tokens_per_layer"] = _Gemma4VocabParallelEmbedding()
    return plan


def _get_attention_head_counts(text_config) -> set[tuple[int, int]]:
    """Return every per-layer attention-head shape used by a Gemma4 config."""
    return {
        (
            int(layer_config.num_attention_heads),
            int(layer_config.num_key_value_heads),
        )
        for layer_config in text_config.per_layer_config
    }


def register_gemma4_parallel_strategy() -> None:
    """Register Gemma4's model-owned FSDP2 strategy once."""
    from nemo_automodel.components.distributed.parallelizer import (
        PARALLELIZATION_STRATEGIES,
        DefaultParallelizationStrategy,
        register_parallel_strategy,
    )

    name = "Gemma4ForConditionalGeneration"
    if name in PARALLELIZATION_STRATEGIES:
        return

    @register_parallel_strategy(name=name)
    class Gemma4ParallelizationStrategy(DefaultParallelizationStrategy):
        """Apply the variant-aware Gemma4 TP plan before standard FSDP2."""

        def parallelize(self, model: nn.Module, device_mesh: DeviceMesh, **kwargs: Any) -> nn.Module:
            """Validate and apply Gemma4 tensor parallelism.

            Args:
                model: Gemma4 model to shard.
                device_mesh: Global mesh containing the tensor-parallel axis.
                **kwargs: Standard ``DefaultParallelizationStrategy`` options.

            Returns:
                The TP/FSDP2-sharded Gemma4 model.
            """
            tp_mesh_name = kwargs.get("tp_mesh_name", "tp")
            tp_mesh = device_mesh[tp_mesh_name]
            tp_size = tp_mesh.size()
            text_config = model.config.text_config

            if tp_size > 1:
                if text_config.enable_moe_block:
                    raise ValueError("Gemma4 MoE does not support tensor parallelism; use expert parallelism instead.")
                attention_head_counts = _get_attention_head_counts(text_config)
                incompatible_head_counts = {
                    head_counts
                    for head_counts in attention_head_counts
                    if head_counts[0] % tp_size != 0 or head_counts[1] % tp_size != 0
                }
                if incompatible_head_counts:
                    raise ValueError(
                        "Gemma4 TP requires every layer attention head count to be divisible by tp_size; "
                        f"got incompatible_head_counts={sorted(incompatible_head_counts)}, tp_size={tp_size}."
                    )
                model._gemma4_tp_enabled = True
                model._gemma4_tp_size = tp_size
                model._gemma4_tp_mesh = tp_mesh

                if kwargs.get("tp_shard_plan") is None:
                    kwargs["tp_shard_plan"] = _gemma4_tp_plan(
                        model,
                        sequence_parallel=bool(kwargs.get("sequence_parallel", False)),
                    )

            return super().parallelize(model, device_mesh, **kwargs)
