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
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

from collections.abc import Mapping
from contextlib import nullcontext
from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from molt.models.loading import configure_loaded_model, load_automodel
from molt.models.packing import pack_padded_batch, unpack_to_padded


class _AttrDict(dict):
    """Model output with both mapping and attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    __setattr__ = dict.__setitem__


def _normalize_output(output):
    if torch.is_tensor(output):
        return _AttrDict(logits=output)
    if isinstance(output, Mapping) and not isinstance(output, _AttrDict):
        return _AttrDict(output)
    return output


def _first_token_id(config, *names):
    for name in names:
        token_id = getattr(config, name, None)
        if isinstance(token_id, int):
            return token_id
    return None


class BaseModel(nn.Module):
    """Shared base for the RL model wrappers (``Actor`` and ``Critic``).

    Owns the runtime forward contract: turning a logical ``[batch, sequence]``
    batch into the model's physical inputs (padded, packed, VLM, CP, R3),
    running the backbone, and restoring token outputs to dense coordinates.
    Checkpoint loading and backend selection live in ``molt.models.loading``;
    the AutoModel ``Engine`` wraps the eager module for backward and
    optimizer steps.
    """

    def __init__(
        self,
        pretrain_or_model,
        attn_implementation: str = "flash_attention_2",
        param_dtype: str = "bf16",
        device_mesh=None,
        moe_mesh=None,
        distributed_config=None,
        moe_config=None,
        activation_checkpointing: Union[bool, str] = False,
        packing_samples: bool = False,
        temperature: float = 1.0,
        freeze_visual_encoder: bool = False,
        freeze_moe_router: bool = False,
        use_fp32_master_weights: bool = True,
        moe_aux_loss_coef: float = 0.0,
        routing_replay: bool = False,
        pre_fsdp_hook=None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected {type(self).__name__} keyword argument(s): {unexpected}")
        self.temperature = temperature
        self.device_mesh = device_mesh
        self._moe_mesh = moe_mesh
        self._routing_replay_adapter = None
        mesh_dims = getattr(device_mesh, "mesh_dim_names", ()) or ()
        cp_mesh = device_mesh["cp"] if device_mesh is not None and "cp" in mesh_dims else None
        self.cp_size = cp_mesh.size() if cp_mesh is not None else 1
        self._hybridep_equalization_groups = None
        self._model_owns_hybridep_packed_cp_equalization = False

        if isinstance(pretrain_or_model, str):
            from molt.utils.utils import is_vlm_model

            self.is_vlm = is_vlm_model(pretrain_or_model)
            self.model = load_automodel(
                pretrain_or_model,
                is_vlm=self.is_vlm,
                attn_implementation=attn_implementation,
                param_dtype=param_dtype,
                device_mesh=device_mesh,
                moe_mesh=moe_mesh,
                cp_size=self.cp_size,
                distributed_config=distributed_config,
                moe_config=moe_config,
                activation_checkpointing=activation_checkpointing,
                packing_samples=packing_samples,
                freeze_visual_encoder=freeze_visual_encoder,
                freeze_moe_router=freeze_moe_router,
                use_fp32_master_weights=use_fp32_master_weights,
                moe_aux_loss_coef=moe_aux_loss_coef,
                pre_fsdp_hook=pre_fsdp_hook,
            )
        else:
            if pre_fsdp_hook is not None:
                raise ValueError("pre_fsdp_hook requires loading the model through NeMoAutoModel")
            self.model = pretrain_or_model
            self.is_vlm = False
            # Pre-instantiated models (tests, inference utilities) skip the CLI
            # backend-flag check: attn_implementation only applies to models
            # loaded through load_automodel.
            attn_implementation = None

        self.packing_layout = configure_loaded_model(
            self.model,
            packing_samples=packing_samples,
            attn_implementation=attn_implementation,
            moe_aux_loss_coef=moe_aux_loss_coef,
            device_mesh=device_mesh,
            moe_mesh=moe_mesh,
            cp_size=self.cp_size,
        )
        if routing_replay:
            self._enable_routing_replay()

        if self.is_vlm:
            self._vlm_config = self.model.config
            self._image_token_id = _first_token_id(
                self._vlm_config, "image_token_id", "image_token_index", "img_context_token_id"
            )
            self._video_token_id = _first_token_id(
                self._vlm_config, "video_token_id", "video_token_index", "video_context_token_id"
            )

    def _enable_routing_replay(self) -> None:
        """Bind AutoModel's model-scoped rollout routing adapter."""
        from nemo_automodel.components.moe.router_replay import RouterReplayAdapter

        self._routing_replay_adapter = RouterReplayAdapter(self.model)
        print(f"[R3] Routing replay enabled at global layer ids {list(self._routing_replay_adapter.layer_ids)}.")

    @property
    def module(self) -> nn.Module:
        """The trainable module, whether or not ``model`` is an Engine."""
        return getattr(self.model, "module", self.model)

    def _hybridep_target_width(self, token_tensor, *, packed: bool) -> int | None:
        """Return the common physical token width required by live HybridEP dispatchers."""
        if self._hybridep_equalization_groups is None:
            uses_uniform_tokens = False
            owns_packed_cp_equalization = False
            for module in self.module.modules():
                dispatcher = getattr(module, "token_dispatcher", None)
                uses_uniform_tokens |= getattr(dispatcher, "requires_uniform_token_count", False) is True
                owns_packed_cp_equalization |= bool(getattr(module, "owns_hybridep_packed_cp_equalization", False))

            groups = []
            if uses_uniform_tokens:
                mesh_names = getattr(self._moe_mesh, "mesh_dim_names", ()) or ()
                if self._moe_mesh is None or "ep" not in mesh_names or self._moe_mesh["ep"].size() <= 1:
                    raise ValueError("a live HybridEP dispatcher requires a moe_mesh with ep_size > 1")
                if not dist.is_available() or not dist.is_initialized():
                    raise RuntimeError("a live HybridEP dispatcher requires initialized distributed process groups")
                groups.append(self._moe_mesh["ep"].get_group())
                if self.cp_size > 1:
                    if "ep_shard" not in mesh_names:
                        raise ValueError("HybridEP with context parallelism requires an ep_shard mesh axis")
                    if self._moe_mesh["ep_shard"].size() > 1:
                        groups.append(self._moe_mesh["ep_shard"].get_group())
            self._hybridep_equalization_groups = tuple(groups)
            self._model_owns_hybridep_packed_cp_equalization = owns_packed_cp_equalization

        if not self._hybridep_equalization_groups:
            return None
        if token_tensor.ndim < 2 or token_tensor.shape[0] < 1 or token_tensor.shape[1] < 1:
            raise ValueError("HybridEP requires a non-empty [batch, tokens, ...] model input")

        local = torch.tensor(
            (int(packed), int(token_tensor.shape[0]), int(token_tensor.shape[1])),
            dtype=torch.int64,
            device=token_tensor.device,
        )
        extrema = torch.cat((local, -local))
        for group in self._hybridep_equalization_groups:
            dist.all_reduce(extrema, op=dist.ReduceOp.MAX, group=group)
        upper = extrema[: local.numel()].tolist()
        lower = (-extrema[local.numel() :]).tolist()
        if lower[0] != upper[0]:
            raise ValueError("HybridEP ranks must all use packed inputs or all use padded inputs")
        if packed:
            if lower[1] != 1 or upper[1] != 1:
                raise ValueError("HybridEP packed equalization requires one physical token row per rank")
            if self.cp_size > 1 and self._model_owns_hybridep_packed_cp_equalization:
                return None
        elif lower[1] != upper[1]:
            raise NotImplementedError(
                "HybridEP padded equalization requires the same batch size on every participating rank"
            )
        return None if lower[2] == upper[2] else int(upper[2])

    def _restore_full_sequence(self, values, *, cp_forward, batch, seqlen, indices):
        """Restore token values to shape ``[batch, sequence, ...]``."""
        if cp_forward:
            seq_dim = 0 if values.ndim == 1 else 1
            values = self._cp_sharder.gather_token_tensor(values, seq_dim=seq_dim, trim=True, fill=0.0)
        if indices is not None:
            return unpack_to_padded(values, indices, batch, seqlen)
        return values[:, :seqlen]

    def _prepare_routed_experts(self, routed_experts, indices, cp_forward):
        """Match rollout routes to the model input's physical token order.

        ``routed_experts`` has shape ``[batch, global_layers, topk, sequence]``.
        The result keeps ``[global_layers, topk]`` as its final axes.
        """
        if self._routing_replay_adapter is None:
            raise RuntimeError("routed_experts requires constructing the model with routing_replay=True")
        if routed_experts.ndim != 4:
            raise ValueError("routed_experts must have shape [batch, global_layers, topk, sequence]")
        per_token = routed_experts.permute(0, 3, 1, 2).contiguous()
        if indices is not None:
            batch, sequence, num_layers, topk = per_token.shape
            dense = per_token.reshape(batch * sequence, num_layers, topk)
            physical = dense.new_full((indices.numel(), num_layers, topk), -1)
            valid = indices >= 0
            physical[valid] = dense.index_select(0, indices[valid])
            per_token = physical.unsqueeze(0)
        if cp_forward:
            return self._cp_sharder.shard_token_tensor(per_token, seq_dim=1, fill=-1)
        return per_token

    def _pack_vlm_batch(self, sequences, attention_mask, media):
        """Pack a padded VLM batch and retain dense-output restore indices.

        ``sequences`` and ``attention_mask`` have shape ``[batch, sequence]``;
        ``media`` contains one processor mapping per sample. Returned model
        inputs use THD or indexed-mask packing, targets follow the same token
        order, and restore indices map real predictions back to the dense batch.
        """
        from nemo_automodel.components.datasets.vlm import pack_vlm_samples, resolve_get_rope_index

        if not isinstance(media, list) or len(media) != sequences.shape[0]:
            raise ValueError("packed VLM forward requires one mm_train_inputs entry per sequence")
        if self._image_token_id is None:
            raise AttributeError(f"VLM config {type(self._vlm_config).__name__} has no image placeholder token id")
        get_rope_index = resolve_get_rope_index(self.module)
        has_mrope = get_rope_index is not None
        if has_mrope and self.cp_size > 1:
            raise NotImplementedError(
                "Multi-axis mRoPE with packed THD context parallelism is intentionally unsupported; "
                "use cp_size=1 or disable VLM packing."
            )

        samples = []
        restore_indices = []
        batch, seqlen = sequences.shape
        for row in range(batch):
            valid = attention_mask[row].bool().nonzero(as_tuple=False).flatten()
            if valid.numel() < 2:
                raise ValueError("packed VLM samples require at least two real tokens")
            start, stop = int(valid[0]), int(valid[-1]) + 1
            if valid.numel() != stop - start:
                raise ValueError("packed VLM samples require one contiguous attention span")
            ids = sequences[row, start:stop]
            sample = {
                "input_ids": ids,
                "labels": ids,
                "attention_mask": torch.ones_like(ids),
            }
            if media[row] is not None:
                sample.update(
                    {
                        key: value.to(sequences.device, non_blocking=True)
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in media[row].items()
                    }
                )
            token_types = (ids == self._image_token_id).to(torch.long)
            if self._video_token_id is not None:
                token_types[ids == self._video_token_id] = 2
            sample["mm_token_type_ids"] = token_types
            samples.append(sample)
            restore_indices.append(torch.arange(start, stop - 1, device=sequences.device) + row * seqlen)

        padding_token_id = getattr(getattr(self.module, "config", None), "pad_token_id", None) or 0
        sequence_alignment = 2 * self.cp_size if self.packing_layout == "thd" and self.cp_size > 1 else 1
        packed = pack_vlm_samples(
            samples,
            padding_idx=padding_token_id,
            get_rope_index=get_rope_index,
            sequence_alignment=sequence_alignment,
        )
        packed_ids = torch.as_tensor(packed["input_ids"], device=sequences.device).reshape(1, -1)
        target_width = self._hybridep_target_width(packed_ids, packed=True)
        collate_kwargs = {"padding_idx": padding_token_id}
        if target_width is not None:
            collate_kwargs["max_length"] = target_width
        if self.packing_layout == "thd":
            from nemo_automodel.components.datasets.vlm import packed_sequence_thd_vlm_collater

            collated = packed_sequence_thd_vlm_collater([packed], **collate_kwargs)
        else:
            from nemo_automodel.components.datasets.vlm import neat_packed_vlm_collater

            collated = neat_packed_vlm_collater(
                [packed],
                attn_implementation="flash_attention_2",
                **collate_kwargs,
            )
        collated = {
            key: value.to(sequences.device) if isinstance(value, torch.Tensor) else value
            for key, value in collated.items()
        }
        labels = collated.pop("labels")
        valid_predictions = labels.reshape(-1) != -100
        dense_positions = torch.cat(restore_indices)
        if int(valid_predictions.sum()) != dense_positions.numel():
            raise ValueError(
                "packed VLM labels do not match the real next-token positions: "
                f"got {int(valid_predictions.sum())} labels and {dense_positions.numel()} dense positions"
            )
        physical_to_dense = dense_positions.new_full((labels.numel(),), -1)
        physical_to_dense[valid_predictions] = dense_positions
        targets = labels.clamp_min(0)
        if self.packing_layout == "thd":
            collated["padding_mask"] = labels.eq(-100)
        collated.pop("_packed_seq_ids", None)
        return collated, targets, physical_to_dense

    def _forward_backbone(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        cp_context_stack,
        mm_inputs,
        output_hidden_states: bool = False,
        routed_experts: Optional[torch.Tensor] = None,
    ):
        """Prepare physical model inputs and run the backbone.

        ``sequences`` and optional ``attention_mask`` have shape ``[batch,
        sequence]``. ``position_ids`` has shape ``[batch, sequence]`` or the
        model's documented multi-axis VLM layout. ``routed_experts`` has shape
        ``[batch, global_layers, topk, sequence]``. The returned targets follow
        the physical model-token order; the remaining values describe how to
        restore token outputs to ``[batch, sequence]``.
        """
        batch, seqlen = sequences.size()
        indices = None
        cp_forward = False
        cp_ctx_factory = nullcontext
        self._cp_sharder = None
        padding_token_id = getattr(getattr(self.module, "config", None), "pad_token_id", None) or 0

        if self.packing_layout is not None:
            if attention_mask is None:
                raise ValueError("packing requires an attention_mask")
            if self.is_vlm:
                if not mm_inputs:
                    mm_inputs = [None] * batch
                if not isinstance(mm_inputs, list):
                    raise ValueError("packed VLM forward requires one mm_train_inputs mapping per sample")
                model_batch, rolled_sequences, indices = self._pack_vlm_batch(sequences, attention_mask, mm_inputs)
                model_batch["labels"] = rolled_sequences
            else:
                sequence_alignment = 2 * self.cp_size if self.packing_layout == "thd" and self.cp_size > 1 else 1
                packed = pack_padded_batch(
                    sequences,
                    attention_mask,
                    layout=self.packing_layout,
                    sequence_alignment=sequence_alignment,
                    padding_token_id=padding_token_id,
                )
                packed_ids, packed_positions, rolled_sequences, indices, packed_attention = packed
                target_width = self._hybridep_target_width(packed_ids, packed=True)
                if target_width is not None:
                    packed_ids, packed_positions, rolled_sequences, indices, packed_attention = pack_padded_batch(
                        sequences,
                        attention_mask,
                        layout=self.packing_layout,
                        sequence_alignment=sequence_alignment,
                        pad_to_tokens=target_width,
                        padding_token_id=padding_token_id,
                    )
                model_batch = {
                    "input_ids": packed_ids,
                    "labels": rolled_sequences,
                    "position_ids": packed_positions,
                    **{
                        key: value
                        for key, value in packed_attention.items()
                        if key not in {"cu_seqlens", "cu_seqlens_padded", "max_seqlen"}
                    },
                }
        else:
            if attention_mask is None:
                attention_mask = torch.ones_like(sequences)
            if attention_mask.shape != sequences.shape:
                raise ValueError("attention_mask must match the [batch, sequence] input_ids shape")
            if self.cp_size > 1:
                valid = attention_mask.bool()
                if not bool(valid[:, 0].all()) or bool((valid[:, 1:] & ~valid[:, :-1]).any()):
                    raise NotImplementedError(
                        "padded context parallelism requires right-padded contiguous sequences; enable packing "
                        "for arbitrary padding layouts"
                    )

            target_width = self._hybridep_target_width(sequences, packed=False)
            if target_width is not None:
                pad = target_width - sequences.shape[1]
                sequences = F.pad(sequences, (0, pad), value=padding_token_id)
                attention_mask = F.pad(attention_mask, (0, pad))
                if position_ids is not None:
                    position_ids = F.pad(position_ids, (0, pad))
                if routed_experts is not None:
                    routed_experts = F.pad(routed_experts, (0, pad), value=-1)

            if self.is_vlm and isinstance(mm_inputs, list):
                from molt.utils.vlm_utils import merge_mm_train_inputs

                mm_inputs = merge_mm_train_inputs(mm_inputs, sequences.device)
            if self.is_vlm and mm_inputs:
                if self._image_token_id is None:
                    raise AttributeError(f"VLM config {type(self._vlm_config).__name__} has no image token id")
                token_type_ids = (sequences == self._image_token_id).to(torch.int32)
                if self._video_token_id is not None:
                    token_type_ids[sequences == self._video_token_id] = 2
                has_media_grid = "image_grid_thw" in mm_inputs or "video_grid_thw" in mm_inputs
                key = "mm_token_type_ids" if has_media_grid else "token_type_ids"
                mm_inputs[key] = token_type_ids
            elif not self.is_vlm and position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

            rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
            model_batch = {
                "input_ids": sequences,
                "labels": rolled_sequences,
                "attention_mask": attention_mask,
                "padding_mask": ~attention_mask.bool(),
                **mm_inputs,
            }
            if position_ids is not None:
                model_batch["position_ids"] = position_ids

        use_cp_sharder = self.packing_layout == "thd" or self.cp_size > 1
        if use_cp_sharder:
            if self.is_vlm and self.cp_size > 1 and not hasattr(self.module, "prepare_model_inputs_for_cp"):
                raise RuntimeError(
                    "VLM + CP requires an AutoModel model with prepare_model_inputs_for_cp; "
                    "use cp_size=1 for this model."
                )
            from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder

            self._cp_sharder = ContextParallelSharder(
                self.module,
                self.device_mesh,
                model_batch,
                invoke_pre_embed=True,
                padding_token_id=padding_token_id,
            )
            cp_ctx_factory, model_batch = self._cp_sharder.shard(model_batch)
            cp_forward = True

        from nemo_automodel.components.utils.model_utils import filter_forward_kwargs

        sequences = model_batch.pop("input_ids")
        rolled_sequences = model_batch.pop("labels").clamp_min_(0)
        forward_attention_mask = model_batch.pop("attention_mask", None)
        position_ids = model_batch.pop("position_ids", None)
        model_kwargs = filter_forward_kwargs(self.module, model_batch)

        forward_ctx = cp_ctx_factory()
        if cp_context_stack is not None and cp_forward:
            cp_context_stack.enter_context(forward_ctx)
            forward_ctx = nullcontext()

        forward_kwargs = {
            "attention_mask": forward_attention_mask,
            "position_ids": position_ids,
            "input_ids": sequences,
            **model_kwargs,
        }
        if output_hidden_states:
            forward_kwargs["output_hidden_states"] = True

        replay_ctx = nullcontext()
        if routed_experts is not None:
            if self._routing_replay_adapter is None:
                raise RuntimeError("routed_experts requires constructing the model with routing_replay=True")
            prepared_routes = self._prepare_routed_experts(routed_experts, indices, cp_forward)
            replay_ctx = self._routing_replay_adapter.replay(prepared_routes)
            if cp_context_stack is not None:
                cp_context_stack.enter_context(replay_ctx)
                replay_ctx = nullcontext()

        with forward_ctx, replay_ctx:
            output = self.model(**forward_kwargs)
        return _normalize_output(output), rolled_sequences, cp_forward, indices, batch, seqlen
