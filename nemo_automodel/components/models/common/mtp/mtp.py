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

"""Model-agnostic MTP scaffolding: depth iteration, token rolling, and loss."""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MTPContextParallelInputs:
    """Globally shifted MTP tensors prepared before context-parallel sharding.

    Attributes:
        input_ids: Per-depth token IDs of shape ``[batch, sequence]`` in global
            sequence order.
        position_ids: Per-depth position IDs in global sequence order. Standard
            positions have shape ``[batch, sequence]``; multi-axis RoPE uses
            ``[axes, batch, sequence]``.
        targets: Per-depth loss targets of shape ``[batch, sequence]`` in global
            sequence order, with invalid positions set to the loss ignore index.
        valid_masks: Per-depth masks of shape ``[batch, sequence]`` identifying
            positions whose future token remains in the same packed sequence.
        position_ids_seq_dim: Sequence dimension of each position-ID tensor.
    """

    input_ids: tuple[torch.LongTensor, ...]
    position_ids: tuple[torch.LongTensor, ...]
    targets: tuple[torch.LongTensor, ...]
    valid_masks: tuple[torch.BoolTensor, ...]
    position_ids_seq_dim: int


def roll_tensor(t: torch.Tensor, shifts: int = -1, dim: int = -1) -> torch.Tensor:
    """Roll a tensor along ``dim`` by ``shifts`` and zero the wrapped slice.

    Used to shift ``input_ids`` / ``position_ids`` / ``labels`` left by one
    position per MTP depth. Single-GPU path only (no CP / packed-sequence
    handling).

    Args:
        t: Input tensor.
        shifts: Number of positions to shift (negative = left shift).
        dim: Dimension to roll along.

    Returns:
        New tensor with the trailing ``|shifts|`` positions along ``dim``
        zero-filled (i.e. no real wrap-around).
    """
    rolled = torch.roll(t, shifts=shifts, dims=dim)
    if shifts == 0 or t.shape[dim] == 0:
        return rolled
    n = abs(shifts)
    if shifts < 0:
        idx = torch.arange(t.shape[dim] - n, t.shape[dim], device=t.device)
    else:
        idx = torch.arange(0, n, device=t.device)
    rolled = rolled.index_fill(dim, idx, 0)
    return rolled


def shift_packed_tensor(
    tensor: torch.Tensor,
    *,
    depth: int,
    seq_idx: torch.Tensor | None = None,
    fill_value: float | int = 0,
    batch_dim: int = 0,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Shift a token-aligned tensor left without crossing sequence boundaries.

    Args:
        tensor: Token-aligned tensor in global sequence order. Its batch and
            sequence axes are selected by ``batch_dim`` and ``seq_dim``.
        depth: Number of future-token positions to shift; must be positive.
        seq_idx: Optional sequence IDs of shape ``[batch, sequence]``. Tokens
            whose shifted source has a different ID are filled.
        fill_value: Scalar used for trailing and cross-sequence positions.
        batch_dim: Batch dimension in ``tensor``.
        seq_dim: Sequence dimension in ``tensor``.

    Returns:
        Tensor with the same shape, dtype, and device as ``tensor``. The output
        owns independent storage and positions without a valid future source
        contain ``fill_value``.
    """
    if tensor.dim() < 2:
        raise ValueError(f"tensor must have batch and sequence dimensions, got {tuple(tensor.shape)}")
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    batch_dim %= tensor.dim()
    seq_dim %= tensor.dim()
    if batch_dim == seq_dim:
        raise ValueError("batch_dim and seq_dim must be different")
    token_shape = (tensor.shape[batch_dim], tensor.shape[seq_dim])
    if seq_idx is not None and seq_idx.shape != token_shape:
        raise ValueError(f"seq_idx shape {tuple(seq_idx.shape)} must match tensor token shape {token_shape}")

    shifted = torch.roll(tensor, shifts=-depth, dims=seq_dim)
    sequence = tensor.shape[seq_dim]
    positions = torch.arange(sequence, device=tensor.device)
    valid = (positions + depth < sequence).unsqueeze(0).expand(tensor.shape[batch_dim], -1)
    if seq_idx is not None:
        shifted_seq_idx = torch.roll(seq_idx, shifts=-depth, dims=1)
        valid = valid & (shifted_seq_idx == seq_idx)
    valid_shape = [1] * tensor.dim()
    valid_shape[batch_dim] = tensor.shape[batch_dim]
    valid_shape[seq_dim] = sequence
    valid = valid.reshape(valid_shape)
    return torch.where(valid, shifted, torch.as_tensor(fill_value, dtype=tensor.dtype, device=tensor.device))


def _packed_seq_ids_from_padded_lengths(
    seq_lens_padded: torch.Tensor,
    *,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> torch.LongTensor:
    """Expand padded packed-sequence lengths into token-aligned sequence IDs.

    Args:
        seq_lens_padded: Tensor of shape [batch, num_sequences] or
            [num_sequences] for a single batch row. Negative entries are
            unused sentinels; nonnegative entries are materialized sequence
            lengths including padding.
        batch_size: Expected batch dimension of the returned tensor.
        seq_len: Expected materialized token count in each batch row.
        device: Device for the returned token-aligned IDs.

    Returns:
        Sequence-ID tensor of shape [batch, sequence] on device.
    """
    if seq_lens_padded.dim() == 1:
        seq_lens_padded = seq_lens_padded.unsqueeze(0)
    if seq_lens_padded.dim() != 2 or seq_lens_padded.shape[0] != batch_size:
        raise ValueError(
            "seq_lens_padded must have shape [batch, num_sequences], "
            f"got {tuple(seq_lens_padded.shape)} for batch size {batch_size}"
        )

    seq_idx = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
    for batch_idx, row in enumerate(seq_lens_padded):
        lengths = row.to(device=device, dtype=torch.long).clamp_min(0)
        sequence_ids = torch.arange(1, row.numel() + 1, dtype=torch.long, device=device)
        try:
            seq_idx[batch_idx] = torch.repeat_interleave(sequence_ids, lengths, output_size=seq_len)
        except RuntimeError as exc:
            raise ValueError(
                f"seq_lens_padded row {batch_idx} must cover exactly the input sequence length {seq_len}"
            ) from exc
    return seq_idx


def _packed_seq_ids_from_batch(
    batch: MutableMapping[str, object],
    *,
    input_ids: torch.Tensor,
) -> torch.LongTensor | None:
    """Normalize supported packed-boundary metadata to token-aligned IDs.

    Args:
        batch: Unsharded batch. Optional seq_idx or _packed_seq_ids tensors
            have shape [batch, sequence] (or [sequence] when batch is one).
            seq_lens_padded has shape [batch, num_sequences].
            cu_seqlens_padded or cu_seqlens contains flattened cumulative
            boundaries of shape [num_sequences + 1] with optional negative
            sentinels.
        input_ids: Global token-ID tensor of shape [batch, sequence] whose
            materialized token layout defines the expected output shape.

    Returns:
        Sequence-ID tensor of shape [batch, sequence] on the input device,
        or None when the batch has no packed-boundary metadata.
    """
    seq_idx = batch.get("seq_idx")
    if not isinstance(seq_idx, torch.Tensor):
        seq_idx = batch.get("_packed_seq_ids")
    if isinstance(seq_idx, torch.Tensor):
        if seq_idx.dim() == 1 and input_ids.shape[0] == 1:
            seq_idx = seq_idx.unsqueeze(0)
        if seq_idx.shape != input_ids.shape:
            raise ValueError(
                f"packed sequence IDs must match input_ids, got {tuple(seq_idx.shape)} and {tuple(input_ids.shape)}"
            )
        return seq_idx.to(device=input_ids.device)

    seq_lens_padded = batch.get("seq_lens_padded")
    if isinstance(seq_lens_padded, torch.Tensor):
        return _packed_seq_ids_from_padded_lengths(
            seq_lens_padded,
            batch_size=input_ids.shape[0],
            seq_len=input_ids.shape[1],
            device=input_ids.device,
        )

    # Padded boundaries describe the materialized token layout. Real
    # cu_seqlens alone are safe only when the materialized layout is unpadded.
    cu_seqlens = batch.get("cu_seqlens_padded")
    uses_unpadded_boundaries = not isinstance(cu_seqlens, torch.Tensor)
    if uses_unpadded_boundaries:
        cu_seqlens = batch.get("cu_seqlens")
    if not isinstance(cu_seqlens, torch.Tensor):
        return None
    if input_ids.shape[0] != 1:
        raise ValueError("cu_seqlens MTP boundary metadata requires batch size 1")

    cu_seqlens = cu_seqlens.reshape(-1)
    cu_seqlens = cu_seqlens[cu_seqlens >= 0].to(device=input_ids.device)
    if cu_seqlens.numel() < 2:
        return None
    if uses_unpadded_boundaries and bool(cu_seqlens[-1] != input_ids.shape[1]):
        raise ValueError(
            "cu_seqlens cannot describe a padded materialized MTP token layout: "
            f"its final boundary must equal input sequence length {input_ids.shape[1]}; "
            "provide cu_seqlens_padded"
        )
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    return torch.searchsorted(cu_seqlens[1:].contiguous(), positions, right=True).unsqueeze(0)


def prepare_mtp_context_parallel_inputs(
    batch: MutableMapping[str, object],
    *,
    num_depths: int,
    ignore_index: int = -100,
) -> MTPContextParallelInputs:
    """Prepare global future-token tensors before context-parallel sharding.

    Each MTP depth is shifted in global sequence order before CP partitions the
    token axis. Packed boundaries are preserved, so no future token, position,
    or target crosses from one document into another. Missing or shared
    position IDs are materialized in batch so the main model and MTP heads are
    subsequently sharded from the same global source.

    Args:
        batch: Mutable unsharded batch. input_ids and labels are tensors of
            shape [batch, sequence]. Optional position_ids has shape
            [batch, sequence], shared shape [1, sequence], or multi-axis RoPE
            shape [axes, batch, sequence]. Packed boundaries may use the tensor layouts documented by
            _packed_seq_ids_from_batch.
        num_depths: Number of MTP future-token depths; must be positive.
        ignore_index: Fill value for invalid targets at trailing and packed
            boundary positions.

    Returns:
        Per-depth input IDs, position IDs, targets, and validity masks. Token
        tensors have global shape [batch, sequence] and independent storage;
        targets without a valid same-sequence future token contain ignore_index.
    """
    if num_depths <= 0:
        raise ValueError(f"num_depths must be positive, got {num_depths}")

    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise ValueError("MTP with context parallelism requires tensor input_ids and labels")
    if input_ids.dim() != 2 or labels.shape != input_ids.shape:
        raise ValueError(
            "MTP with context parallelism requires input_ids and labels with matching "
            f"[batch, sequence] shapes, got {tuple(input_ids.shape)} and {tuple(labels.shape)}"
        )

    position_ids = batch.get("position_ids")
    if position_ids is None:
        position_ids = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            .unsqueeze(0)
            .expand(input_ids.shape[0], -1)
            .contiguous()
        )
        batch["position_ids"] = position_ids
    elif (
        isinstance(position_ids, torch.Tensor)
        and position_ids.dim() == 2
        and position_ids.shape == (1, input_ids.shape[1])
    ):
        position_ids = position_ids.expand(input_ids.shape[0], -1).contiguous()
        batch["position_ids"] = position_ids
    elif (
        isinstance(position_ids, torch.Tensor) and position_ids.dim() == 3 and position_ids.shape[1:] == input_ids.shape
    ):
        pass
    elif not isinstance(position_ids, torch.Tensor) or position_ids.shape != input_ids.shape:
        position_shape = tuple(position_ids.shape) if isinstance(position_ids, torch.Tensor) else type(position_ids)
        raise ValueError(
            f"MTP position_ids must be a tensor matching input_ids, got {position_shape} and {tuple(input_ids.shape)}"
        )

    position_batch_dim = 1 if position_ids.dim() == 3 else 0
    position_seq_dim = 2 if position_ids.dim() == 3 else 1
    seq_idx = _packed_seq_ids_from_batch(batch, input_ids=input_ids)
    depths = tuple(range(1, num_depths + 1))
    return MTPContextParallelInputs(
        input_ids=tuple(shift_packed_tensor(input_ids, depth=depth, seq_idx=seq_idx) for depth in depths),
        position_ids=tuple(
            shift_packed_tensor(
                position_ids,
                depth=depth,
                seq_idx=seq_idx,
                batch_dim=position_batch_dim,
                seq_dim=position_seq_dim,
            )
            for depth in depths
        ),
        targets=tuple(
            shift_packed_tensor(labels, depth=depth, seq_idx=seq_idx, fill_value=ignore_index) for depth in depths
        ),
        valid_masks=tuple(
            shift_packed_tensor(
                torch.ones_like(input_ids, dtype=torch.bool), depth=depth, seq_idx=seq_idx, fill_value=False
            )
            for depth in depths
        ),
        position_ids_seq_dim=position_seq_dim,
    )


def get_mtp_loss_scaling_factor(model: nn.Module, default: float = 0.1) -> float:
    """Return the model's configured MTP auxiliary-loss scaling factor."""
    mtp_config = getattr(model, "mtp_config", None)
    if mtp_config is not None:
        return float(getattr(mtp_config, "loss_scaling_factor", default))
    return default


@dataclass
class MTPConfig:
    """Runtime configuration for the MTP block.

    Attributes:
        num_layers: Number of MTP forward iterations (D). ``0`` disables MTP.
            Equivalent to Megatron's ``--mtp-num-layers``.
        layer_pattern: Per-depth inner-block pattern, e.g. ``"*E"`` for one
            attention + one MoE sublayer per depth.
        loss_scaling_factor: Coefficient applied to the summed per-depth CE
            loss (default ``0.1``). The effective per-depth weight is
            ``loss_scaling_factor / num_layers``.
        use_repeated_layer: When ``True``, build a single physical depth's
            worth of sublayers and reuse it for all ``num_layers`` forward
            iterations (weight-tied across depths). Equivalent to Megatron's
            ``--mtp-use-repeated-layer``.
    """

    num_layers: int = 0
    layer_pattern: str = ""
    loss_scaling_factor: float = 0.1
    use_repeated_layer: bool = False

    @property
    def pattern_length(self) -> int:
        return len(self.layer_pattern)

    @property
    def num_physical_depths(self) -> int:
        return 1 if self.use_repeated_layer else self.num_layers

    @property
    def total_sublayers(self) -> int:
        return self.num_physical_depths * self.pattern_length

    @property
    def enabled(self) -> bool:
        return self.num_layers > 0 and self.pattern_length > 0


class MTPModule(nn.Module):
    """Multi-Token Prediction block.

    Holds a flat :class:`nn.ModuleList` of sublayers (length
    ``num_physical_depths * pattern_length``) where the first sublayer of
    each physical depth carries the fusion modules (``enorm``, ``hnorm``,
    ``eh_proj``) and the last sublayer of each physical depth carries
    ``final_layernorm``. This flat layout matches the HuggingFace export
    format used by Nemotron-V3 (``mtp.layers.{i}.*``).

    The model-specific sublayer construction (which decoder block to use, how
    to handle MoE / attention / Mamba) is delegated to the caller via
    ``sublayer_factory``.

    Args:
        mtp_config: :class:`MTPConfig` describing depth and pattern.
        block_types_per_sublayer: List of block-type strings (one per inner
            sublayer position), length must equal ``mtp_config.pattern_length``.
            Caller is responsible for parsing the model-specific symbol
            convention; this module does not interpret symbols.
        sublayer_factory: Callable
            ``factory(global_idx, depth, sublayer_idx, block_type, has_fusion, has_final_norm) -> nn.Module``
            constructing one sublayer. The returned module must be callable
            as ``sublayer(hidden_states, **kwargs) -> Tensor`` and, when
            ``has_fusion=True``, expose attributes ``enorm``, ``hnorm``,
            ``eh_proj``. When ``has_final_norm=True`` it must expose
            ``final_layernorm``.
    """

    def __init__(
        self,
        mtp_config: MTPConfig,
        block_types_per_sublayer: list[str],
        sublayer_factory: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        if not mtp_config.enabled:
            raise ValueError("MTPModule constructed with disabled MTPConfig")
        if len(block_types_per_sublayer) != mtp_config.pattern_length:
            raise ValueError(
                f"len(block_types_per_sublayer)={len(block_types_per_sublayer)} "
                f"!= mtp_config.pattern_length={mtp_config.pattern_length}"
            )
        self.mtp_config = mtp_config
        num_sublayers_per_depth = mtp_config.pattern_length
        num_physical_depths = mtp_config.num_physical_depths
        layers: list[nn.Module] = []
        for depth in range(num_physical_depths):
            for sublayer_idx in range(num_sublayers_per_depth):
                global_idx = depth * num_sublayers_per_depth + sublayer_idx
                layers.append(
                    sublayer_factory(
                        global_idx=global_idx,
                        depth=depth,
                        sublayer_idx=sublayer_idx,
                        block_type=block_types_per_sublayer[sublayer_idx],
                        has_fusion=(sublayer_idx == 0),
                        has_final_norm=(sublayer_idx == num_sublayers_per_depth - 1),
                    )
                )
        self.layers = nn.ModuleList(layers)

    @property
    def num_depths(self) -> int:
        return self.mtp_config.num_layers

    @property
    def pattern_length(self) -> int:
        return self.mtp_config.pattern_length

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        input_ids: torch.LongTensor | None = None,
        input_ids_per_depth: tuple[torch.LongTensor, ...] | None = None,
        embed_fn: Callable[[torch.LongTensor], torch.Tensor] | None = None,
        embed_inputs: tuple[torch.Tensor, ...] | None = None,
        position_ids: torch.LongTensor | None = None,
        position_ids_per_depth: tuple[torch.LongTensor, ...] | None = None,
        **block_kwargs,
    ) -> list[torch.Tensor]:
        """Iterate over MTP depths and return per-depth hidden states.

        Three mutually-exclusive input modes:

        * **Single-rank / first-stage PP** (default): pass ``input_ids`` plus
          ``embed_fn``. The module rolls ``input_ids`` cumulatively left by 1
          per depth and applies ``embed_fn`` to produce the future-token
          embedding for that depth.
        * **Context parallel**: pass ``input_ids_per_depth`` plus ``embed_fn``.
          Each tensor is globally shifted and then sharded into the local CP
          token layout, so this module embeds it directly without a rank-local
          roll.
        * **Final-stage PP / multimodal**: pass ``embed_inputs`` (a tuple of
          pre-rolled per-depth embeddings, length ``num_depths``). Used when
          the last PP stage no longer owns ``embed_tokens``, or for multimodal
          models (e.g. SALM) where some positions carry continuous audio
          embeddings that cannot be recovered by re-embedding an integer token
          id — the caller pre-rolls the fused embedding tensor and passes it
          here.

        Args:
            hidden_states: Output of the main model's final norm (``h_0``);
                tensor of shape ``[batch, sequence, hidden]`` or
                ``[tokens, hidden]`` for THD.
            input_ids: Token ids of shape ``[batch, sequence]`` (or
                ``[tokens]`` in THD). Rolled
                cumulatively left by 1 per depth. Mutually exclusive with
                ``input_ids_per_depth`` and ``embed_inputs``.
            input_ids_per_depth: Optional tuple of ``num_depths`` pre-shifted
                token-ID tensors. Each has local CP shape
                ``[batch, sequence]`` or ``[tokens]`` and is embedded directly
                with ``embed_fn``. Requires ``position_ids_per_depth``.
            embed_fn: Callable applied to rolled ``input_ids`` to produce the
                future-token embedding (typically the model's input embedding
                layer). Required when ``input_ids`` is supplied.
            embed_inputs: Optional tuple of ``num_depths`` pre-computed
                future-token embeddings, each of shape
                ``[batch, sequence, hidden]`` or ``[tokens, hidden]``.
                Mutually exclusive with ``input_ids``/
                ``input_ids_per_depth``/``embed_fn``.
            position_ids: Position ids matching ``input_ids``. When supplied,
                rolled cumulatively per depth in lockstep with ``input_ids``
                (so slot ``t`` carries the original position of the rolled
                token) and forwarded to each sublayer via ``block_kwargs``.
                Required for RoPE-using sublayers; ignored by sublayers that
                don't consume it.
            position_ids_per_depth: Optional tuple of ``num_depths``
                pre-computed future-token position tensors. Each tensor has
                shape ``[batch, sequence]``, ``[axes, batch, sequence]`` for
                multi-axis RoPE, or ``[tokens]`` for THD. When supplied, these
                tensors are forwarded directly instead of rolling rank-local
                ``position_ids``. Use this when context parallelism has already sharded the sequence.
                Required with ``input_ids_per_depth`` and incompatible with
                rank-local ``input_ids`` rolling.
            **block_kwargs: Forwarded to each sublayer's ``__call__`` (e.g.
                ``attention_mask``).

        Returns:
            List of length ``num_depths`` containing hidden states of shape
            ``[batch, sequence, hidden]`` or ``[tokens, hidden]``.
        """
        if embed_inputs is not None:
            if input_ids is not None or input_ids_per_depth is not None or embed_fn is not None:
                raise ValueError("embed_inputs is mutually exclusive with input_ids/input_ids_per_depth/embed_fn")
            if len(embed_inputs) != self.num_depths:
                raise ValueError(f"embed_inputs length {len(embed_inputs)} does not match num_depths {self.num_depths}")
        elif input_ids_per_depth is not None:
            if input_ids is not None or embed_fn is None:
                raise ValueError("input_ids_per_depth requires embed_fn and is mutually exclusive with input_ids")
            if len(input_ids_per_depth) != self.num_depths:
                raise ValueError(
                    f"input_ids_per_depth length {len(input_ids_per_depth)} does not match num_depths {self.num_depths}"
                )
        else:
            if input_ids is None or embed_fn is None:
                raise ValueError(
                    "MTPModule.forward requires embed_inputs, (input_ids_per_depth, embed_fn), or (input_ids, embed_fn)"
                )
        if input_ids_per_depth is not None and position_ids_per_depth is None:
            raise ValueError("input_ids_per_depth and position_ids_per_depth must be provided together")
        if input_ids is not None and position_ids_per_depth is not None:
            raise ValueError(
                "position_ids_per_depth cannot be combined with rank-local input_ids rolling; "
                "provide input_ids_per_depth or precomputed embed_inputs"
            )
        if position_ids_per_depth is not None:
            if len(position_ids_per_depth) != self.num_depths:
                raise ValueError(
                    f"position_ids_per_depth length {len(position_ids_per_depth)} "
                    f"does not match num_depths {self.num_depths}"
                )
            if position_ids is not None:
                expected_position_shape = position_ids.shape
                expected_position_source = "base position_ids"
            elif input_ids is not None:
                expected_position_shape = input_ids.shape
                expected_position_source = "input_ids"
            elif input_ids_per_depth is not None:
                expected_position_shape = input_ids_per_depth[0].shape
                expected_position_source = "input_ids_per_depth token shape"
            else:
                expected_position_shape = embed_inputs[0].shape[:-1]
                expected_position_source = "embed_inputs token shape"
            for depth, depth_position_ids in enumerate(position_ids_per_depth, start=1):
                position_shape_matches = depth_position_ids.shape == expected_position_shape
                if len(expected_position_shape) == 2:
                    position_shape_matches = position_shape_matches or (
                        depth_position_ids.dim() == 3 and depth_position_ids.shape[1:] == expected_position_shape
                    )
                if not position_shape_matches:
                    raise ValueError(
                        f"MTP depth {depth} position_ids shape {tuple(depth_position_ids.shape)} "
                        f"does not match {expected_position_source} {tuple(expected_position_shape)}"
                    )

        num_iterations = self.num_depths
        num_sublayers_per_depth = self.pattern_length
        use_repeated = self.mtp_config.use_repeated_layer
        per_depth_h: list[torch.Tensor] = []
        cur_input_ids = input_ids
        cur_position_ids = position_ids
        for depth in range(num_iterations):
            if embed_inputs is not None:
                decoder_input = embed_inputs[depth]
            elif input_ids_per_depth is not None:
                decoder_input = embed_fn(input_ids_per_depth[depth])
            else:
                cur_input_ids = roll_tensor(cur_input_ids, shifts=-1, dim=-1)
                decoder_input = embed_fn(cur_input_ids)
            if position_ids_per_depth is not None:
                depth_position_ids = position_ids_per_depth[depth]
            elif cur_position_ids is not None:
                cur_position_ids = roll_tensor(cur_position_ids, shifts=-1, dim=-1)
                depth_position_ids = cur_position_ids
            else:
                depth_position_ids = None

            physical_depth = 0 if use_repeated else depth
            for sublayer_idx in range(num_sublayers_per_depth):
                sublayer = self.layers[physical_depth * num_sublayers_per_depth + sublayer_idx]
                kwargs = dict(block_kwargs)
                if depth_position_ids is not None:
                    kwargs["position_ids"] = depth_position_ids
                if sublayer_idx == 0:
                    kwargs["embed_input"] = decoder_input
                hidden_states = sublayer(hidden_states, **kwargs)
            per_depth_h.append(hidden_states)
        return per_depth_h
