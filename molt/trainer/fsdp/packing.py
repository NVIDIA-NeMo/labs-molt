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

"""Padded <-> packed conversion for the FSDP2 model backend.

Molt's datasets emit `(B, S)` padded batches. Packing removes padding and
creates `(1, total_tokens)` streams plus sequence-boundary metadata. The exact
kwargs depend on the selected model path:

- HF flash-attn2 consumes ``FlashAttentionKwargs``.
- AutoModel custom TE consumes THD kwargs
  (``qkv_format=thd`` / ``cu_seqlens`` / ``max_seqlen``).
"""

import itertools
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def unshard_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a DTensor to a plain unsharded tensor on every rank.

    Use after model forward to gather TP-sharded outputs (logits with vocab dim
    sharded by ``ColwiseParallel(output_layouts=Shard(-1))``, or hidden states
    with sequence dim sharded by ``SequenceParallel`` in SP mode) so downstream
    loss / log-prob / entropy computations can run on plain tensors.

    No-op when not under TP/SP (input is a regular ``torch.Tensor``).
    Memory cost under TP=k: each rank holds the full unsharded tensor (∝ k× the
    sharded one); ok for activation-side tensors at training resolution.
    """
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def is_automodel_custom_model(model: Any) -> bool:
    """Best-effort check for AutoModel's native implementations.

    ``NeMoAutoModel*`` may return either a HF model (when ``force_hf`` is used or
    no native implementation exists) or a class under
    ``nemo_automodel.components.models``. Only the latter accepts THD packing
    kwargs directly.

    FSDP2's ``fully_shard`` swaps ``model.__class__`` for a dynamic subclass
    whose ``__module__`` no longer starts with ``nemo_automodel``; walk the MRO
    so the check survives that wrap.
    """
    for cls in type(model).__mro__:
        if getattr(cls, "_molt_automodel_custom", False):
            return True
        if issubclass(cls, torch.nn.Module) and cls.__module__.startswith("nemo_automodel.components.models"):
            return True
    return False


def packed_attn_kwargs(seq_lens: list[int], *, style: str, device, trailing_pad: int = 0) -> dict:
    """Varlen attention kwargs for an already-packed `(1, total_tokens)` batch.

    The collate builds the pack (`make_experience_batch(packed=True)`); this only
    describes its sequence boundaries to the attention kernel.

    ``trailing_pad`` covers the EP-equalized suffix, whose length is a collective
    and so is known only at forward time: it extends the last sequence, is flagged
    in ``padding_mask`` so experts skip it, and is causal-suffixed, leaving real
    tokens' outputs unchanged.
    """
    if style not in {"hf", "automodel"}:
        raise ValueError(f"Unsupported packing style: {style}")

    boundaries = list(itertools.accumulate(seq_lens, initial=0))
    max_length = max(seq_lens)
    if trailing_pad:
        boundaries[-1] += trailing_pad
        max_length = max(max_length, seq_lens[-1] + trailing_pad)
    cu_seq_lens = torch.tensor(boundaries, dtype=torch.int32, device=device)  # varlen kernels need int32

    if style == "hf":
        return {
            "cu_seq_lens_q": cu_seq_lens,
            "cu_seq_lens_k": cu_seq_lens,
            "max_length_q": max_length,
            "max_length_k": max_length,
        }
    kwargs = {
        "qkv_format": "thd",
        "cu_seqlens": cu_seq_lens,
        "cu_seqlens_padded": cu_seq_lens,
        "max_seqlen": max_length,
    }
    if trailing_pad:
        real_tokens = sum(seq_lens)
        kwargs["padding_mask"] = (torch.arange(real_tokens + trailing_pad, device=device) >= real_tokens).unsqueeze(0)
    return kwargs


def _distributed_log_softmax(local_logits: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
    logits_max = local_logits.amax(dim=-1, keepdim=True)
    dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=group)

    shifted_logits = local_logits - logits_max
    exp_sum = shifted_logits.exp().sum(dim=-1, keepdim=True).float()
    dist.all_reduce(exp_sum, op=dist.ReduceOp.SUM, group=group)
    return shifted_logits - exp_sum.log().to(shifted_logits.dtype)


class _DistributedLogProb(torch.autograd.Function):
    """Gather selected logprobs from TP-sharded vocab logits without all-gathering logits."""

    @staticmethod
    def forward(
        ctx,
        local_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start: int,
        vocab_end: int,
        group: dist.ProcessGroup,
        inference_only: bool,
    ) -> torch.Tensor:
        target_mask = (target < vocab_start) | (target >= vocab_end)
        local_target = (target - vocab_start).masked_fill(target_mask, 0)

        log_probs = _distributed_log_softmax(local_logits.float(), group)
        softmax = log_probs.exp()

        selected = torch.gather(log_probs, -1, local_target.unsqueeze(-1)).squeeze(-1)
        selected = selected.masked_fill(target_mask, 0.0)
        dist.all_reduce(selected, op=dist.ReduceOp.SUM, group=group)

        if not inference_only:
            # softmax/grad math runs in fp32 for numerical stability, but the
            # gradient must come back in the input's dtype — autocast feeds bf16
            # logits and PyTorch will warn or auto-cast a dtype-mismatched grad.
            ctx.input_dtype = local_logits.dtype
            ctx.save_for_backward(softmax, target_mask, local_target)
        return selected

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        softmax, target_mask, local_target = ctx.saved_tensors
        grad_input = -softmax * grad_output.unsqueeze(-1)

        if softmax.ndim == 3:
            grad_input = grad_input.contiguous()
            bsz, seq, vocab = softmax.shape
            row = torch.arange(bsz, device=softmax.device).view(-1, 1).expand(-1, seq).reshape(-1)
            col = torch.arange(seq, device=softmax.device).expand(bsz, -1).reshape(-1)
            flat_base = (row * seq + col) * vocab
            valid = ~target_mask.reshape(-1)
            flat_index = flat_base.masked_select(valid) + local_target.reshape(-1).masked_select(valid)
            grad_input.view(-1).scatter_add_(0, flat_index, grad_output.reshape(-1).masked_select(valid))
        else:
            valid = ~target_mask
            grad_input.scatter_add_(
                -1,
                local_target.unsqueeze(-1),
                (grad_output * valid).unsqueeze(-1),
            )

        return grad_input.to(ctx.input_dtype), None, None, None, None, None


def log_probs_from_vocab_parallel_logits(
    vocab_parallel_logits: DTensor,
    target: torch.Tensor,
    *,
    temperature: float = 1.0,
    inference_only: bool | None = None,
) -> torch.Tensor:
    """Compute selected token logprobs for TP-vocab-sharded DTensor logits.

    ``target`` is already aligned with ``vocab_parallel_logits`` positions. This
    mirrors Molt's local ``log_probs_from_logits(logits, rolled_ids)``
    contract and avoids materializing full-vocab logits on each TP rank.
    """
    device_mesh = vocab_parallel_logits.device_mesh
    if device_mesh.mesh_dim_names is None or "tp" not in device_mesh.mesh_dim_names:
        raise ValueError("vocab_parallel_logits must be sharded on a mesh with a 'tp' dimension")

    tp_group = device_mesh.get_group("tp")
    tp_rank = dist.get_rank(group=tp_group)
    tp_size = dist.get_world_size(group=tp_group)

    local_logits = vocab_parallel_logits.to_local()
    global_vocab_size = vocab_parallel_logits.shape[-1]
    if global_vocab_size % tp_size == 0:
        vocab_per_rank = global_vocab_size // tp_size
        vocab_start = vocab_per_rank * tp_rank
        vocab_end = vocab_start + vocab_per_rank
    else:
        local_vocab_size = torch.tensor(local_logits.shape[-1], device=local_logits.device, dtype=torch.long)
        shard_sizes = [torch.zeros_like(local_vocab_size) for _ in range(tp_size)]
        dist.all_gather(shard_sizes, local_vocab_size, group=tp_group)
        shard_sizes = torch.stack(shard_sizes)
        vocab_start = int(shard_sizes[:tp_rank].sum().item())
        vocab_end = vocab_start + int(shard_sizes[tp_rank].item())

    if temperature != 1.0:
        # fp32 before the divide, same contract as log_probs_from_logits: rounding the quotient back
        # to bf16 costs ~1 ULP per logit on top of the logits' own quantization. The log-softmax
        # below upcasts anyway, so this only moves the cast earlier; autograd casts the grad back.
        local_logits = local_logits.float() / temperature
    if inference_only is None:
        inference_only = not torch.is_grad_enabled()
    return _DistributedLogProb.apply(local_logits, target, vocab_start, vocab_end, tp_group, inference_only)
