# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thin adapters from Molt's collated RL ``Experience`` to AutoModel ``Datum``."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import torch
from nemo_automodel._transformers.utils import resolve_get_rope_index
from nemo_automodel.components.datasets.datum import Datum, LossInputLayout, collate_datums, collate_vlm_datums
from nemo_automodel.components.loss import vocab_parallel_log_probs
from nemo_automodel.engine import Engine, LossFnOutputBatch, PerTokenOutput
from torch.distributed.tensor import DTensor

from molt.models.utils import log_probs_from_logits
from molt.utils.vlm_utils import merge_mm_train_inputs


@dataclass(frozen=True)
class PreparedRLEngineDatum:
    """One replay-buffer microbatch represented by one or more Engine Datums."""

    datums: tuple[Datum, ...]
    dense_shape: tuple[int, int]
    prediction_indices: tuple[torch.Tensor, ...]

    @property
    def num_datums(self) -> int:
        return len(self.datums)

    def restore_token_outputs(self, tensors: Sequence[torch.Tensor]) -> torch.Tensor:
        """Restore per-Datum Engine outputs to the Experience's dense token axis."""

        if len(tensors) != len(self.prediction_indices):
            raise ValueError(
                f"RL microbatch expects {len(self.prediction_indices)} Engine outputs, got {len(tensors)}"
            )
        if not tensors:
            raise ValueError("RL microbatches cannot be empty")

        batch, sequence = self.dense_shape
        trailing_shape = tensors[0].shape[1:]
        restored = tensors[0].new_zeros((batch, sequence, *trailing_shape))
        for row, (tensor, indices) in enumerate(zip(tensors, self.prediction_indices)):
            if tensor.shape[1:] != trailing_shape or tensor.shape[0] != indices.numel():
                raise ValueError(
                    "Engine output does not match its original prediction axis: "
                    f"output={tuple(tensor.shape)}, indices={indices.numel()}"
                )
            restored[row].index_copy_(0, indices.to(tensor.device), tensor)
        return restored


def resolve_rl_engine_collation(model_wrapper, tokenizer, strategy, micro_train_batch_size: int):
    """Select the AutoModel collater and outer Datum batch size for RL."""

    if not bool(getattr(model_wrapper, "is_vlm", False)):
        return partial(collate_datums, packed=bool(model_wrapper.packing_samples)), micro_train_batch_size
    if tokenizer is None or not hasattr(tokenizer, "image_processor"):
        raise ValueError("RL VLM training requires the model's AutoProcessor")

    packing_samples = bool(model_wrapper.packing_samples)
    mesh_names = getattr(strategy.device_mesh, "mesh_dim_names", ()) or ()
    cp_size = strategy.device_mesh["cp"].size() if "cp" in mesh_names else 1
    get_rope_index = resolve_get_rope_index(model_wrapper.model) if packing_samples else None
    if packing_samples and cp_size > 1 and get_rope_index is not None:
        raise NotImplementedError(
            "AutoModel does not yet support multi-axis mRoPE with packed THD context parallelism; "
            "use cp_size=1 or disable VLM packing."
        )
    if (
        packing_samples
        and cp_size > 1
        and not bool(getattr(model_wrapper.model, "supports_cp_with_sequence_packing", False))
    ):
        raise NotImplementedError(
            f"{type(model_wrapper.model).__name__} does not support VLM sequence packing with "
            f"context parallelism (cp_size={cp_size}) on its active attention backend."
        )

    collate_fn = partial(
        collate_vlm_datums,
        processor=tokenizer,
        packed=packing_samples,
        get_rope_index=get_rope_index,
        sequence_alignment=2 * cp_size if packing_samples and cp_size > 1 else 1,
    )
    # Callers pass each replay microbatch's actual Datum count to Engine. Keep
    # the dynamic default at one so an omitted explicit boundary stays within
    # the replay buffer's token-budget memory bound.
    engine_microbatch_size = 1 if strategy.args.train.dynamic_batch_enable else micro_train_batch_size
    return collate_fn, engine_microbatch_size


def extract_model_logits(output):
    """Normalize native AutoModel tensors and Hugging Face model outputs."""

    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        return output["logits"]
    return output.logits


def action_log_probs_from_output(output, loss_inputs, temperature: float) -> torch.Tensor:
    """Compute realized next-token log-probabilities from an Engine callback."""

    logits = extract_model_logits(output)
    target_tokens = loss_inputs["target_tokens"]
    if isinstance(logits, DTensor):
        return vocab_parallel_log_probs(logits, target_tokens, temperature=temperature)
    return log_probs_from_logits(logits, target_tokens, temperature=temperature)


def run_rl_engine_forward(
    engine: Engine,
    prepared: PreparedRLEngineDatum,
    output_key: str,
    token_output_fn: Callable[[object, Mapping[str, torch.Tensor]], torch.Tensor],
) -> torch.Tensor:
    """Run collection-time inference and restore one token output to replay coordinates."""

    def loss_fn(output, loss_inputs):
        token_output = token_output_fn(output, loss_inputs)
        weights = loss_inputs["weights"]
        if token_output.shape != weights.shape:
            raise ValueError(
                f"collection output {output_key!r} has shape {tuple(token_output.shape)}, "
                f"expected token shape {tuple(weights.shape)}"
            )
        return token_output.new_zeros(()), LossFnOutputBatch(
            per_token={output_key: PerTokenOutput(token_output * weights, fill_value=0.0)}
        )

    result = engine.forward(prepared.datums, loss_fn, microbatch_sizes=(prepared.num_datums,))
    if len(result.loss_fn_outputs) != prepared.num_datums:
        raise RuntimeError(
            f"Engine returned {len(result.loss_fn_outputs)} collection outputs for {prepared.num_datums} Datums"
        )
    return prepared.restore_token_outputs([record[output_key] for record in result.loss_fn_outputs])


def _prepare_vlm_rl_engine_datums(
    experience,
    model_wrapper,
    *,
    losses: Mapping[str, torch.Tensor],
    replicated_losses: Mapping[str, torch.Tensor],
    loss_pad_values: Mapping[str, int | float | bool],
) -> PreparedRLEngineDatum:
    """Split one padded replay microbatch into processor-ready VLM Datums."""

    sequences = experience.sequences
    attention_mask = experience.attention_mask
    batch, full_sequence = sequences.shape
    mm_items = experience.mm_train_inputs or [None] * batch
    if len(mm_items) != batch:
        raise ValueError(f"VLM mm_train_inputs must contain {batch} per-sample entries, got {len(mm_items)}")
    if "labels" in losses or "labels" in replicated_losses:
        raise ValueError("RL VLM loss fields reserve 'labels' for the AutoModel collater")

    datums = []
    prediction_indices = []
    for row in range(batch):
        valid = attention_mask[row].bool().nonzero(as_tuple=False).flatten()
        if valid.numel() < 2:
            raise ValueError("RL VLM Datums require at least two contiguous real tokens per sample")
        start = int(valid[0])
        stop = int(valid[-1]) + 1
        if valid.numel() != stop - start:
            raise ValueError("RL VLM Datums require a contiguous real-token attention span")

        input_ids = sequences[row, start:stop]
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids, dtype=attention_mask.dtype),
        }
        mm_inputs = merge_mm_train_inputs([mm_items[row]], sequences.device) if mm_items[row] is not None else {}
        if mm_inputs:
            image_token_id = getattr(model_wrapper, "_image_token_id", None)
            if image_token_id is None:
                raise AttributeError("VLM config is missing an image token id")
            token_type_ids = (input_ids == image_token_id).to(torch.int32)
            video_token_id = getattr(model_wrapper, "_video_token_id", None)
            if video_token_id is not None:
                token_type_ids[input_ids == video_token_id] = 2
            token_type_key = (
                "mm_token_type_ids"
                if any(key in mm_inputs for key in ("image_grid_thw", "video_grid_thw"))
                else "token_type_ids"
            )
            mm_inputs[token_type_key] = token_type_ids
            model_inputs.update(mm_inputs)

        sample_losses = {name: value[row, start : stop - 1] for name, value in losses.items()}
        sample_losses.update(replicated_losses)
        labels = sample_losses["target_tokens"].clone()
        labels.masked_fill_(sample_losses["weights"] == 0, -100)
        sample_losses["labels"] = labels
        layouts = {
            **{name: LossInputLayout.PER_TOKEN for name in losses},
            **{name: LossInputLayout.REPLICATED for name in replicated_losses},
            "labels": LossInputLayout.PER_TOKEN,
        }
        datums.append(
            Datum(
                model_inputs=model_inputs,
                loss_fn_inputs=sample_losses,
                loss_fn_input_layouts=layouts,
                loss_fn_input_pad_values={**loss_pad_values, "labels": -100},
            )
        )
        prediction_indices.append(torch.arange(start, stop - 1, device=sequences.device))

    return PreparedRLEngineDatum(
        datums=tuple(datums),
        dense_shape=(batch, full_sequence - 1),
        prediction_indices=tuple(prediction_indices),
    )


def prepare_rl_engine_datum(
    experience,
    model_wrapper,
    *,
    loss_fields: Mapping[str, torch.Tensor | None],
    include_sequence_ids: bool = False,
    routed_experts: torch.Tensor | None = None,
) -> PreparedRLEngineDatum:
    """Convert one already-collated Experience microbatch into Engine Datums.

    Molt keeps replay-buffer batching and dense-coordinate mapping. AutoModel
    owns text/VLM collation, padding/packing, device movement, CP layout,
    normalization, backward, gradient finalization, clipping, and optimizer
    mutation. Autoregressive model inputs use states ``tokens[:-1]`` while loss
    fields use next-token positions ``tokens[1:]``.
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

    # Keep only real-token -> real-token transitions. Using attention_mask[:, 1:]
    # alone would incorrectly turn the final left-padding slot into a live model
    # token, while attention_mask[:, :-1] alone retains the final right-padded
    # state even though it has no real next-token target.
    input_ids = sequences[:, :-1]
    prediction_mask = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    if bool((action_mask.bool() & ~prediction_mask).any()):
        raise ValueError("action_mask may select only real-token -> real-token transitions")
    losses: dict[str, torch.Tensor] = {
        "weights": action_mask.to(torch.float32),
        "target_tokens": sequences[:, 1:],
    }
    losses.update({name: value for name, value in loss_fields.items() if value is not None})
    for name, value in losses.items():
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or tuple(value.shape[:2]) != (batch, sequence):
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ValueError(f"RL loss field {name!r} must start with {(batch, sequence)}, got {shape}")

    resolved_pad_values: dict[str, int | float | bool] = {}
    if include_sequence_ids:
        sequence_ids = torch.arange(batch, device=sequences.device).unsqueeze(1).expand(batch, sequence).clone()
        losses["sequence_ids"] = sequence_ids.masked_fill(~prediction_mask, -1)
        resolved_pad_values["sequence_ids"] = -1
    if routed_experts is not None:
        adapter = getattr(model_wrapper, "_routing_replay_adapter", None)
        if adapter is None:
            raise RuntimeError("routed_experts requires an Actor constructed with routing_replay=True")
        if routed_experts.ndim != 4 or routed_experts.shape[0] != batch or routed_experts.shape[-1] != full_sequence:
            raise ValueError("rollout routed_experts must have shape [batch, global_layers, topk, full_sequence]")
        prepared_routes = adapter.prepare_routed_experts(routed_experts[..., :-1])
        prepared_routes = prepared_routes.masked_fill(~prediction_mask[..., None, None], -1)
        losses["routed_experts"] = prepared_routes
        resolved_pad_values["routed_experts"] = -1

    replicated_losses: dict[str, torch.Tensor] = {}
    if include_sequence_ids:
        replicated_losses["num_sequences"] = torch.tensor(batch, device=sequences.device)

    is_vlm = bool(getattr(model_wrapper, "is_vlm", False))
    if is_vlm:
        return _prepare_vlm_rl_engine_datums(
            experience,
            model_wrapper,
            losses=losses,
            replicated_losses=replicated_losses,
            loss_pad_values=resolved_pad_values,
        )

    datums = []
    prediction_indices = []
    for row in range(batch):
        valid = attention_mask[row].bool().nonzero(as_tuple=False).flatten()
        if valid.numel() < 2:
            raise ValueError("RL text Datums require at least two contiguous real tokens per sample")
        start = int(valid[0])
        stop = int(valid[-1]) + 1
        if valid.numel() != stop - start:
            raise ValueError("RL text Datums require a contiguous real-token attention span")

        sample_losses = {name: value[row, start : stop - 1] for name, value in losses.items()}
        sample_losses.update(replicated_losses)
        layouts = {
            **{name: LossInputLayout.PER_TOKEN for name in losses},
            **{name: LossInputLayout.REPLICATED for name in replicated_losses},
        }
        datums.append(
            Datum(
                model_inputs={"input_ids": input_ids[row, start : stop - 1]},
                loss_fn_inputs=sample_losses,
                loss_fn_input_layouts=layouts,
                loss_fn_input_pad_values=resolved_pad_values,
            )
        )
        prediction_indices.append(torch.arange(start, stop - 1, device=sequences.device))

    return PreparedRLEngineDatum(
        datums=tuple(datums),
        dense_shape=(batch, sequence),
        prediction_indices=tuple(prediction_indices),
    )
