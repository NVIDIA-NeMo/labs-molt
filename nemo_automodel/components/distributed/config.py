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

"""
Strategy-specific distributed training configuration classes.

Design principle:
- Size params (dp_size, dp_replicate_size, tp_size, pp_size, cp_size, ep_size)
  are grouped in ``ParallelismSizes``.
- dp_replicate_size is FSDP2-only: raises assertion if passed with non-FSDP2 config
- Strategy-specific configs contain only *additional* flags unique to each strategy
- Managers become normal classes that accept (config, device_mesh)

Usage:
    from nemo_automodel.components.distributed.config import FSDP2Config, MegatronFSDPConfig, DDPConfig

    # FSDP2 with custom options
    config = FSDP2Config(sequence_parallel=True, activation_checkpointing=True)

    # MegatronFSDP with custom options
    config = MegatronFSDPConfig(zero_dp_strategy=3, overlap_grad_reduce=True)

    # DDP with activation checkpointing
    config = DDPConfig(activation_checkpointing=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Tuple, Union

import torch
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy

from nemo_automodel.components.distributed.cp_vision_frame_shard import CpVisionFrameShardingConfig
from nemo_automodel.shared.multimodal_fsdp import FrozenMultimodalSharding, normalize_frozen_multimodal_sharding

if TYPE_CHECKING:
    from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes

# Type aliases for API signatures.
ActivationCheckpointingMode = Union[bool, Literal["full", "selective"]]
ActivationCheckpointingScope = Union[str, List[str], Tuple[str, ...]]
# FSDP2 is the only strategy; keep the Union alias so `X | None` works at runtime in importers.
DistributedStrategyConfig = Union["FSDP2Config"]
# Backwards-compatible alias for external / type-checking references.
DistributedConfig = DistributedStrategyConfig

_VALID_ACTIVATION_CHECKPOINTING_SCOPES = {"all", "language", "vision", "audio", "multimodal"}


def normalize_activation_checkpointing_scope(value: Any) -> Tuple[str, ...]:
    """Validate and normalize activation-checkpointing scope values."""
    if value is None:
        return ("all",)
    if isinstance(value, str):
        raw_parts = value.lower().replace("-", "_").replace("+", ",").split(",")
    elif isinstance(value, (list, tuple, set)):
        raw_parts = []
        for item in value:
            if not isinstance(item, str):
                raise ValueError("activation_checkpointing_scope entries must be strings.")
            raw_parts.extend(item.lower().replace("-", "_").replace("+", ",").split(","))
    else:
        raise ValueError("activation_checkpointing_scope must be a string or list of strings.")

    scopes: list[str] = []
    for part in raw_parts:
        scope = part.strip()
        if not scope:
            continue
        if scope in {"default", "auto"}:
            scope = "all"
        if scope not in _VALID_ACTIVATION_CHECKPOINTING_SCOPES:
            valid = ", ".join(sorted(_VALID_ACTIVATION_CHECKPOINTING_SCOPES))
            raise ValueError(f"activation_checkpointing_scope must use only: {valid}. Got {part!r}.")
        if scope not in scopes:
            scopes.append(scope)

    if not scopes:
        return ("all",)
    if "all" in scopes and len(scopes) > 1:
        raise ValueError("activation_checkpointing_scope='all' cannot be combined with other scopes.")
    return tuple(scopes)


@dataclass(frozen=True)
class DistributedSetup:
    """Resolved distributed topology and execution policies."""

    mesh_context: "MeshContext"
    strategy_config: DistributedStrategyConfig | None = None
    moe_parallel_config: "MoEParallelizerConfig | None" = None
    activation_checkpointing: ActivationCheckpointingMode = False

    @classmethod
    def build(
        cls,
        strategy: str | DistributedStrategyConfig = "fsdp2",
        parallelism_sizes: "ParallelismSizes | None" = None,
        moe_parallel_config: "MoEParallelizerConfig | dict | None" = None,
        activation_checkpointing: ActivationCheckpointingMode = False,
        world_size: int | None = None,
        timeout_minutes: int | None = None,
        ranks: list[int] | tuple[int, ...] | None = None,
    ) -> "DistributedSetup":
        """Create a resolved distributed setup from sizes and policy configs.

        Intentionally, this function is forgiving wrt the input types, allowing
        strings for the strategy and a dict for the MoE config.
        """
        from nemo_automodel.components.distributed.init_utils import get_world_size_safe
        from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes

        if world_size is None:
            world_size = get_world_size_safe()

        strategy_config = _resolve_strategy_config(strategy)

        if parallelism_sizes is None:
            parallelism_sizes = ParallelismSizes()

        if parallelism_sizes.pp_size > 1:
            raise ValueError("pipeline parallelism is not supported")
        ep_size = parallelism_sizes.ep_size
        if moe_parallel_config is not None and ep_size <= 1:
            raise ValueError("moe_parallel_config requires ep_size > 1")
        if ep_size > 1 and moe_parallel_config is None:
            moe_parallel_config = MoEParallelizerConfig()
        if isinstance(moe_parallel_config, dict):
            moe_parallel_config = MoEParallelizerConfig(**moe_parallel_config)

        mesh_context = MeshContext.build(
            strategy_config,
            parallelism_sizes=parallelism_sizes,
            world_size=world_size,
            timeout_minutes=timeout_minutes,
            ranks=ranks,
        )

        return cls(
            mesh_context=mesh_context,
            strategy_config=strategy_config,
            moe_parallel_config=moe_parallel_config,
            activation_checkpointing=activation_checkpointing,
        )


@dataclass
class MoEParallelizerConfig:
    """Configuration for MoE model parallelization (EP + FSDP settings)."""

    # Default True: under activation checkpointing the MoE router projection and top-k outputs
    # must be saved rather than recomputed. Recomputing tied top-k selections can route a
    # different number of tokens per expert than the forward pass, which makes
    # torch.utils.checkpoint raise a CheckpointError on the backward recompute.
    ignore_router_for_ac: bool = True
    reshard_after_forward: bool = False
    lm_head_precision: Union[str, torch.dtype] | None = None
    wrap_outer_model: bool = True
    mp_policy: MixedPrecisionPolicy | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class MultimodalVisionConfig:
    """Distributed policies for resolved vision modules.

    Attributes:
        frame_sharding: Controls frame-level encoder compute sharding over the
            selected text-model mesh dimensions.
    """

    frame_sharding: CpVisionFrameShardingConfig = field(default_factory=CpVisionFrameShardingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.frame_sharding, CpVisionFrameShardingConfig):
            raise TypeError("MultimodalVisionConfig.frame_sharding must be a CpVisionFrameShardingConfig instance.")


@dataclass
class MultimodalDistributedConfig:
    """Distributed policies for resolved multimodal modules.

    Attributes:
        frozen_sharding: Controls fully frozen multimodal modules such as
            vision/audio towers and projectors. ``"root"`` (default) keeps
            their parameters in an always-run outer FSDP root, which is safe
            when modality execution differs across ranks. ``"per_layer"``
            uses normal layer/container FSDP units and requires every rank in
            the FSDP group to execute or skip the module identically on every
            microbatch. ``"replicate"`` excludes the frozen parameters from
            FSDP roots so each rank keeps a full copy. Modules with any
            trainable parameters use normal layer/container sharding
            regardless of this setting.
        vision: Policies that apply specifically to resolved vision modules.
    """

    frozen_sharding: FrozenMultimodalSharding = "root"
    vision: MultimodalVisionConfig = field(default_factory=MultimodalVisionConfig)

    def __post_init__(self) -> None:
        self.frozen_sharding = normalize_frozen_multimodal_sharding(self.frozen_sharding)
        if not isinstance(self.vision, MultimodalVisionConfig):
            raise TypeError("MultimodalDistributedConfig.vision must be a MultimodalVisionConfig instance.")


@dataclass
class FSDP2Config:
    """
    Additional configuration for FSDP2 distributed training.

    Note: Size parameters (dp_size, dp_replicate_size, tp_size, pp_size, cp_size, ep_size)
    are grouped separately in ``ParallelismSizes``.

    Attributes:
        sequence_parallel (bool): Enable sequence parallelism in TP plan.
        tp_plan (Optional[dict]): Custom TP plan. If None, auto-selected based on model type.
        patch_is_packed_sequence (bool): Patch ``transformers._is_packed_sequence`` to always
            return Python ``False``. This does two things: (1) removes a CPU-GPU sync per
            attention layer (``aten::is_nonzero`` triggered by HF when batch_size==1), and
            (2) ensures static attention shapes for ``torch.compile``. Safe for standard
            (non-packed) training only. Disable if using packed-sequence training
            (position_ids that reset to 0 mid-sequence). Default ``False``.
        mp_policy (Optional[MixedPrecisionPolicy]): MixedPrecisionPolicy for FSDP2.
            If ``None`` (default), uses bf16 forward/backward compute with fp32
            gradient reduction. Pair this with ``model.torch_dtype: float32`` for
            the Megatron-style fp32 master-weights pattern. Override from YAML
            using the ``_target_`` pattern::

                mp_policy:
                  _target_: torch.distributed.fsdp.MixedPrecisionPolicy
                  param_dtype: bfloat16
                  reduce_dtype: float32
                  output_dtype: bfloat16

            See ``docs/guides/mixed-precision-training.md`` for the full set of recommended
            patterns and the bf16-storage trap.
        offload_policy (Optional[CPUOffloadPolicy]): CPUOffloadPolicy for CPU offloading.
        autocast_dtype (Optional[torch.dtype]): If set, wraps the forward pass in
            ``torch.autocast(device_type="cuda", dtype=autocast_dtype)``.  Use with
            ``output_dtype=float32`` in mp_policy to keep the residual stream in fp32
            while running matmuls in lower precision.  Set to ``None`` to disable.
            Can be set from YAML as a string (e.g. ``autocast_dtype: bfloat16``).
        activation_checkpointing (bool | "full" | "selective"): Enable activation checkpointing. ``True`` or
            ``"full"`` keeps the existing full activation checkpointing behavior. ``"selective"`` wraps transformer
            blocks with PyTorch selective activation checkpointing.
        activation_checkpointing_scope (str | list[str]): Which extracted
            layer groups activation checkpointing should wrap. ``"all"``
            selects every extracted group. Scoped values such as
            ``"language"``, ``"vision"``, and ``"multimodal"`` are filtered
            to trainable layers before generic wrapping.
        defer_fsdp_grad_sync (bool): Defer FSDP gradient sync to final micro-batch.
        reshard_after_forward (Optional[bool]): Override layer-level FSDP2 resharding.
            ``None`` preserves AutoModel's heuristic: pipeline-parallel layers do
            not reshard after forward, while non-pipeline layers reshard all but
            the last layer. Set ``False`` for a ZeRO-2-like benchmark where
            gathered parameters stay resident after forward. Set ``True`` to force
            resharding everywhere, including pipeline-parallel layers, which may
            reduce throughput by adding per-microbatch all-gathers.
        enable_async_tensor_parallel (bool): Enable async tensor parallelism via
            ``torch._inductor.config._micro_pipeline_tp``.  Overlaps ReduceScatter with
            compute in row-parallel layers.  Requires ``sequence_parallel=True`` (forced
            automatically with a warning if not set).  Also enables symmetric memory for
            the TP group.
        enable_compile (bool): Apply per-layer ``torch.compile`` to transformer decoder
            layers (with NO_REENTRANT activation checkpointing inside each compiled layer).
            Skips whole-model compile so that checkpoint loading does not produce
            ``_orig_mod`` key-prefix mismatches.
        enable_fsdp2_prefetch (bool): Enable explicit forward/backward prefetch chains
            between FSDP2 sharded layers.  Default ``True``.
        fsdp2_backward_prefetch_depth (int): Number of FSDP units to prefetch during
            backward pass.  ``2`` hides AllGather behind compute; ``1`` reduces peak
            memory at a small throughput cost.  Default ``2``.
        fsdp2_forward_prefetch_depth (int): Number of FSDP units to prefetch during
            forward pass.  Default ``1``.
        multimodal: Policies for resolved multimodal modules that use the text
            model's distributed axes.
    """

    sequence_parallel: bool = False
    tp_plan: dict | None = None
    patch_is_packed_sequence: bool = False
    mp_policy: MixedPrecisionPolicy | None = field(
        default_factory=lambda: MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
    )
    offload_policy: CPUOffloadPolicy | None = None
    autocast_dtype: torch.dtype | None = None
    activation_checkpointing: ActivationCheckpointingMode = False
    activation_checkpointing_scope: ActivationCheckpointingScope = "all"
    defer_fsdp_grad_sync: bool = True
    reshard_after_forward: bool | None = None
    enable_async_tensor_parallel: bool = False
    enable_compile: bool = False
    enable_fsdp2_prefetch: bool = False
    fsdp2_backward_prefetch_depth: int = 2
    fsdp2_forward_prefetch_depth: int = 1
    multimodal: MultimodalDistributedConfig = field(default_factory=MultimodalDistributedConfig)

    def __post_init__(self) -> None:
        if self.mp_policy is None:
            # FSDP2 default: bf16 compute and fp32 gradient reduction. Pair with
            # ``model.torch_dtype: float32`` for fp32 optimizer state. See
            # ``docs/guides/mixed-precision-training.md``.
            self.mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
                cast_forward_inputs=True,
            )
        self.activation_checkpointing_scope = normalize_activation_checkpointing_scope(
            self.activation_checkpointing_scope
        )
        if not isinstance(self.multimodal, MultimodalDistributedConfig):
            raise TypeError("FSDP2Config.multimodal must be a MultimodalDistributedConfig instance.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (shallow, preserves policy objects)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


_STRATEGY_MAP: Dict[str, type[FSDP2Config]] = {"fsdp2": FSDP2Config}


def _resolve_strategy_config(
    strategy: str | DistributedStrategyConfig,
    **strategy_kwargs: Any,
) -> DistributedStrategyConfig:
    """Resolve a setup-level strategy name or config object."""
    if isinstance(strategy, FSDP2Config):
        if strategy_kwargs:
            raise ValueError("Strategy kwargs cannot be passed with an instantiated strategy config.")
        return strategy

    if not isinstance(strategy, str):
        raise ValueError(f"Unknown distributed strategy type: {type(strategy)}")

    strategy_name = strategy.lower()
    if strategy_name not in _STRATEGY_MAP:
        valid = sorted(_STRATEGY_MAP)
        raise ValueError(f"Unknown strategy: {strategy}. Valid strategies: {valid}")
    strategy_cls = _STRATEGY_MAP[strategy_name]
    valid_fields = {f.name for f in fields(strategy_cls)}
    unknown = set(strategy_kwargs) - valid_fields
    if unknown:
        raise ValueError(f"Unknown options for strategy '{strategy_name}': {sorted(unknown)}")
    return strategy_cls(**strategy_kwargs)


__all__ = [
    "DistributedSetup",
    "DistributedStrategyConfig",
    "FSDP2Config",
    "MoEParallelizerConfig",
    "normalize_activation_checkpointing_scope",
]
