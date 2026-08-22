# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin adapters from Molt's collated RL ``Experience`` to AutoModel ``Datum``."""

from dataclasses import dataclass
from typing import Mapping

import torch

from nemo_automodel.components.datasets.datum import Datum, LossInputLayout
from nemo_automodel.components.datasets.utils import pack_features_for_thd, packed_sequence_thd_collater

from molt.trainer.fsdp.packing import unpack_to_padded
from molt.utils.vlm_utils import merge_mm_train_inputs


@dataclass(frozen=True)
class PreparedRLEngineDatum:
    """One prebatched Datum plus the information needed to undo THD packing."""

    datum: Datum
    dense_shape: tuple[int, int]
    packed_indices: torch.Tensor | None = None

    def restore_token_output(self, tensor: torch.Tensor) -> torch.Tensor:
        """Restore an Engine token output to the Experience's dense ``[B, T-1]`` axis."""

        if self.packed_indices is None:
            return tensor
        batch, sequence = self.dense_shape
        return unpack_to_padded(tensor, self.packed_indices.to(tensor.device), batch, sequence)


def extract_model_logits(output):
    """Normalize native AutoModel tensors and Hugging Face model outputs."""

    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        return output["logits"]
    return output.logits


def prepare_rl_engine_datum(
    experience,
    model_wrapper,
    *,
    loss_fields: Mapping[str, torch.Tensor | None],
    packing_samples: bool,
    loss_pad_values: Mapping[str, int | float | bool] | None = None,
) -> PreparedRLEngineDatum:
    """Convert one already-collated Experience microbatch into one Engine Datum.

    Molt keeps replay-buffer batching and multimodal processor aggregation. The
    AutoModel Engine owns device movement, CP layout, normalization, backward,
    gradient finalization, clipping, and optimizer mutation. Autoregressive model
    inputs use states ``tokens[:-1]`` while loss fields use next-token positions
    ``tokens[1:]``.
    """

    sequences = experience.sequences
    attention_mask = experience.attention_mask
    action_mask = experience.action_mask
    if not all(
        isinstance(value, torch.Tensor) and value.ndim == 2 for value in (sequences, attention_mask, action_mask)
    ):
        raise ValueError("RL Engine Datums require 2-D sequences, attention_mask, and action_mask tensors")
    batch, full_sequence = sequences.shape
    sequence = full_sequence - 1
    if sequence <= 0 or tuple(attention_mask.shape) != (batch, full_sequence):
        raise ValueError("RL Engine Datums require matching token and attention axes with at least two tokens")
    if tuple(action_mask.shape) != (batch, sequence):
        raise ValueError(
            f"action_mask must have shape {(batch, sequence)} for sequences {tuple(sequences.shape)}, "
            f"got {tuple(action_mask.shape)}"
        )

    # Only positions that can predict a real next token need a model forward.
    # For right-padded rows, attention_mask[:, 1:] has exactly L-1 real states.
    input_ids = sequences[:, :-1]
    prediction_mask = attention_mask[:, 1:].bool()
    losses: dict[str, torch.Tensor] = {
        "weights": action_mask.to(torch.float32),
        "target_tokens": sequences[:, 1:],
    }
    losses.update({name: value for name, value in loss_fields.items() if value is not None})
    for name, value in losses.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or tuple(value.shape[:2]) != (batch, sequence):
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ValueError(f"RL loss field {name!r} must start with {(batch, sequence)}, got {shape}")

    is_vlm = bool(getattr(model_wrapper, "is_vlm", False))
    if packing_samples and is_vlm:
        raise NotImplementedError(
            "RL VLM packing is not supported: rollout media tensors do not yet have an AutoModel RL Datum collater"
        )

    packed_indices = None
    if packing_samples:
        if getattr(model_wrapper, "_packing_style", "automodel") != "automodel":
            raise NotImplementedError(
                "RL Engine THD packing requires an AutoModel-native THD model; Hugging Face packing belongs to "
                "Molt's removed legacy forward adapter"
            )
        packed_indices = prediction_mask.reshape(-1).nonzero(as_tuple=False).flatten()
        features = []
        for row in range(batch):
            row_ids = input_ids[row][prediction_mask[row]]
            if row_ids.numel() == 0:
                raise ValueError("RL Engine packing cannot pack a sequence with no real prediction positions")
            features.append({"input_ids": row_ids.tolist()})
        model_inputs = packed_sequence_thd_collater([pack_features_for_thd(features)])
        model_inputs.pop("labels", None)
        packed_losses = {}
        for name, value in losses.items():
            flat = value.reshape(batch * sequence, *value.shape[2:])
            packed_losses[name] = flat.index_select(0, packed_indices).unsqueeze(0)
        losses = packed_losses
    else:
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": prediction_mask.to(attention_mask.dtype),
        }
        if is_vlm:
            mm_inputs = (
                merge_mm_train_inputs(experience.mm_train_inputs, sequences.device)
                if experience.mm_train_inputs
                else {}
            )
            if mm_inputs:
                image_token_id = getattr(model_wrapper, "_image_token_id", None)
                if image_token_id is None:
                    raise AttributeError("VLM config is missing an image token id")
                token_type_ids = (input_ids == image_token_id).to(torch.int32)
                video_token_id = getattr(model_wrapper, "_video_token_id", None)
                if video_token_id is not None:
                    token_type_ids[input_ids == video_token_id] = 2
                token_type_key = "mm_token_type_ids" if "image_grid_thw" in mm_inputs else "token_type_ids"
                mm_inputs[token_type_key] = token_type_ids
            model_inputs.update(mm_inputs)
        else:
            position_ids = prediction_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(~prediction_mask, 1)
            model_inputs["position_ids"] = position_ids

    layouts = {name: LossInputLayout.PER_TOKEN for name in losses}
    datum = Datum(
        model_inputs=model_inputs,
        loss_fn_inputs=losses,
        loss_fn_input_layouts=layouts,
        loss_fn_input_pad_values=dict(loss_pad_values or {}),
    )
    return PreparedRLEngineDatum(datum=datum, dense_shape=(batch, sequence), packed_indices=packed_indices)
