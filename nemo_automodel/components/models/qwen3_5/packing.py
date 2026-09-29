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

"""Shared packed-sequence metadata for dense and MoE Qwen3.5 models."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from nemo_automodel.components.models.common.packing import get_unpad_data, is_indexed_packed_mask


@dataclass(frozen=True)
class GatedDeltaPackedMetadata:
    """Packed-sequence metadata shared by every GatedDeltaNet layer.

    Args:
        document_ids: Indexed document mask of shape [batch, sequence] on the
            compute device, with zero denoting padding.
        indices: Flattened valid-token indices of shape [tokens] on the compute
            device.
        cu_seqlens: Cumulative document lengths of shape [documents + 1] on the
            compute device.
        cu_seqlens_cpu: CPU mirror of ``cu_seqlens`` with shape [documents + 1]
            for FLA host-side chunk planning.
    """

    document_ids: torch.Tensor
    indices: torch.Tensor
    cu_seqlens: torch.Tensor
    cu_seqlens_cpu: torch.Tensor


def prepare_gated_delta_packed_metadata(
    attention_mask: torch.Tensor | None,
    packed_seq_ids: torch.Tensor | None,
) -> GatedDeltaPackedMetadata | None:
    """Build shared GatedDeltaNet metadata once for a model forward.

    Args:
        attention_mask: Optional indexed document mask of shape [batch,
            sequence] or a backend-specific attention mask.
        packed_seq_ids: Optional indexed document IDs of shape [batch,
            sequence] supplied beside a backend-specific attention mask.

    Returns:
        Device and CPU packed-sequence metadata whose tensor layouts are
        documented by :class:`GatedDeltaPackedMetadata`, or ``None`` for an
        unpacked mask.
    """
    # ``_packed_seq_ids`` is the collater's authoritative indexed mask. Prefer
    # it when present so a backend-specific attention mask does not need a
    # device scalar decision. Otherwise a structurally eligible 2D attention
    # mask is the only candidate.
    document_ids = None
    document_ids_cpu = None
    for candidate in (packed_seq_ids, attention_mask):
        if candidate is None or candidate.dtype == torch.bool or candidate.dim() != 2:
            continue
        candidate_cpu = candidate.detach().to(device="cpu")
        if is_indexed_packed_mask(candidate_cpu):
            document_ids = candidate
            document_ids_cpu = candidate_cpu
            break
    if document_ids is None or document_ids_cpu is None:
        return None

    # FLA needs a CPU cu_seqlens mirror for host-side chunk planning. Derive all
    # dynamic-size metadata from the single host copy above, and coalesce indices
    # + cu_seqlens into one H2D transfer. The normal packed path therefore replaces
    # three CUDA scalar reads, two dynamic ``nonzero`` synchronizations, and a
    # final D2H mirror with one boundary in each direction per model forward.
    indices_cpu, cu_seqlens_cpu, _ = get_unpad_data(document_ids_cpu)
    indices_cpu = indices_cpu.to(torch.long)
    cu_seqlens_cpu = cu_seqlens_cpu.to(torch.long)
    num_indices = indices_cpu.numel()
    device_metadata = torch.cat((indices_cpu, cu_seqlens_cpu)).to(device=document_ids.device)
    return GatedDeltaPackedMetadata(
        document_ids=document_ids,
        indices=device_metadata[:num_indices],
        cu_seqlens=device_metadata[num_indices:],
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
