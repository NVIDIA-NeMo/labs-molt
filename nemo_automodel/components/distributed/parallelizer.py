# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import importlib
import logging
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from types import FunctionType
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    checkpoint_wrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import (
    FSDPModule,
    MixedPrecisionPolicy,
    OffloadPolicy,
    fully_shard,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    ParallelStyle,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.placement_types import Replicate, Shard

from nemo_automodel.components.distributed.activation_checkpointing import (
    SELECTIVE_AC_WRAPPER_FLAG,
    apply_full_layer_checkpointing_to_layers,
    apply_selective_checkpointing_to_layers,
    apply_submodule_checkpointing,
    detect_kv_sharing_and_maybe_disable_cache,
    is_selective_activation_checkpointing,
)
from nemo_automodel.components.distributed.config import (
    ActivationCheckpointingScope,
    normalize_activation_checkpointing_scope,
)
from nemo_automodel.components.distributed.mesh_utils import get_fsdp_dp_mesh
from nemo_automodel.shared.multimodal_fsdp import (
    FrozenMultimodalSharding,
    ignored_params_for_root,
    is_multimodal_module_name,
    iter_multimodal_modules,
    module_is_fully_frozen,
    module_parameters,
    normalize_frozen_multimodal_sharding,
)
from nemo_automodel.shared.tied_weights import ensure_tied_lm_head
from nemo_automodel.shared.torch_patches import (
    patch_fsdp_accumulated_grad_guard as _patch_fsdp_accumulated_grad_guard,
)


@lru_cache(maxsize=1)
def _gemma4_for_conditional_generation() -> type:
    """Return Gemma4ForConditionalGeneration, or a placeholder on older transformers.

    Cached so the placeholder branch yields one stable class object, which the
    callers rely on for identity comparisons and as a dict key.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

        return Gemma4ForConditionalGeneration
    except (ImportError, ModuleNotFoundError):

        class Gemma4ForConditionalGeneration:  # type: ignore[no-redef]
            """Placeholder when the installed transformers build has no Gemma4."""

        return Gemma4ForConditionalGeneration


import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils
from nemo_automodel.components.distributed.optimized_tp_plans import (
    PARALLELIZE_FUNCTIONS,
    VocabParallelEmbedding,
    _get_class_qualname,
)
from nemo_automodel.components.distributed.parallel_styles import ReplicatedWithGradAllReduce, translate_to_lora

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def apply_selective_activation_checkpointing(
    model: nn.Module,
    *,
    enable_compile: bool = False,
    activation_checkpointing_scope: ActivationCheckpointingScope | None = "all",
) -> None:
    """Apply selective activation checkpointing to ``model`` end to end.

    Standalone entry point (detects KV-sharing, disables ``use_cache``, and
    wraps transformer blocks) for paths where the FSDP2 parallelize flow is
    skipped -- notably single-GPU training.

    Args:
        model: The model to checkpoint.
        enable_compile: Whether per-layer ``torch.compile`` will be applied.
        activation_checkpointing_scope: Which extracted layer groups to wrap.
    """
    layer_groups = _extract_model_layer_groups(model)
    layers, _ = _filter_layer_groups_for_activation_checkpointing(layer_groups, activation_checkpointing_scope)
    if not layers:
        logger.warning("No transformer layers found; skipping selective activation checkpointing.")
        return
    has_kv_sharing = detect_kv_sharing_and_maybe_disable_cache(model)
    apply_selective_checkpointing_to_layers(model, layers, has_kv_sharing, enable_compile=enable_compile)


_BAGEL_FULL_LAYER_CHECKPOINT_MODULE_LISTS = (
    "model.language_model.model.layers",
    "model.vit_model.vision_model.encoder.layers",
)


def _get_module_by_fqn(module: nn.Module, fqn: str) -> nn.Module | None:
    obj = module
    for part in fqn.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _is_checkpoint_wrapped(module: nn.Module) -> bool:
    return hasattr(module, "_checkpoint_wrapped_module")


def _apply_bagel_full_layer_activation_checkpointing(model: nn.Module) -> bool:
    """Apply native BAGEL-style activation checkpointing to whole logical layers."""
    if type(model).__name__ != "BagelForUnifiedMultimodal":
        return False

    wrapped_count = 0
    for fqn in _BAGEL_FULL_LAYER_CHECKPOINT_MODULE_LISTS:
        container = _get_module_by_fqn(model, fqn)
        if container is None:
            logger.warning("BAGEL activation checkpointing skipped missing module list %s", fqn)
            continue
        if not isinstance(container, (nn.ModuleList, nn.ModuleDict)):
            logger.warning(
                "BAGEL activation checkpointing expected %s to be a module list, got %s",
                fqn,
                type(container),
            )
            continue

        items = container.items() if isinstance(container, nn.ModuleDict) else enumerate(container)
        for key, layer in list(items):
            if _is_checkpoint_wrapped(layer):
                continue
            container[key] = checkpoint_wrapper(layer, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            wrapped_count += 1

    logger.info("Applied BAGEL full-layer activation checkpointing to %d layers", wrapped_count)
    return wrapped_count > 0


class ParallelizationStrategy(ABC):
    """Abstract base class for model parallelization strategies."""

    @abstractmethod
    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        reshard_after_forward: bool | None = None,
        activation_checkpointing_scope: ActivationCheckpointingScope | None = "all",
        frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
        reapply_trainability: Callable[[nn.Module], None] | None = None,
        **kwargs,
    ) -> nn.Module:
        """Apply parallelization strategy to the model."""
        pass


def _get_input_output_embeddings(model: nn.Module) -> tuple[nn.Module | None, nn.Module | None]:
    """Resolve the optional Hugging Face embedding getters."""

    def _resolve(getter_name: str) -> nn.Module | None:
        getter = getattr(model, getter_name, None)
        if not callable(getter):
            return None
        try:
            module = getter()
        except (AttributeError, NotImplementedError):
            return None
        return module if isinstance(module, nn.Module) else None

    return _resolve("get_input_embeddings"), _resolve("get_output_embeddings")


def _fully_shard_untied_input_output_embeddings(
    model: nn.Module,
    *,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy,
    offload_policy: OffloadPolicy | None,
    input_reshard_after_forward: bool,
    fully_shard_fn: Callable[..., nn.Module],
) -> None:
    """Give large trainable untied embedding tables independent FSDP buffers.

    The generic dense path otherwise leaves both tables in the root FSDP unit.
    With fp32 gradient reduction, that unit allocates one contiguous
    reduce-scatter input containing both gradients. Keeping the two trainable
    leaf modules in separate FSDP units bounds that allocation by the larger
    table instead of their sum. This pass skips tied tables to preserve aliasing,
    and frozen tables because they have no gradient communication buffer
    to split. Tables already wrapped as FSDP units retain their existing
    reshard policy. With an explicit ``reshard_after_forward=True``, a
    container-hosted head reshards while a newly split top-level head stays
    gathered. FusedLinearCrossEntropy handles the sharded weight with correct
    gradients, at the cost of an extra all-gather.

    Args:
        model: Model whose input and output embedding modules may be sharded.
        mesh: Device mesh that owns the FSDP shards.
        mp_policy: Mixed-precision policy used by the surrounding FSDP units.
        offload_policy: Optional offload policy used by the surrounding FSDP
            units.
        input_reshard_after_forward: Whether the input embedding unit reshards
            its parameters after forward.
        fully_shard_fn: FSDP sharding callable, injectable for unit tests.
    """
    weights_are_tied = ensure_tied_lm_head(model)
    input_embeddings, output_embeddings = _get_input_output_embeddings(model)
    input_weight = getattr(input_embeddings, "weight", None)
    output_weight = getattr(output_embeddings, "weight", None)
    weights_are_physically_tied = input_embeddings is not None and (
        input_embeddings is output_embeddings or (input_weight is not None and input_weight is output_weight)
    )
    if weights_are_tied or weights_are_physically_tied:
        logger.info("Skipping independent sharding of tied input/output embeddings")
        return

    seen: set[int] = set()
    for role, module, module_reshard_after_forward in (
        ("input embedding", input_embeddings, input_reshard_after_forward),
        # The output projection is the last compute unit. Keep it gathered until
        # backward, matching the old root-owned behavior and allowing
        # FusedLinearCrossEntropy to consume its mixed-precision compute weight
        # outside the module's forward.
        ("output embedding", output_embeddings, False),
    ):
        if module is None or id(module) in seen:
            continue
        seen.add(id(module))
        # Skip tables that are themselves FSDP units (e.g. ModuleDict children).
        # This does not detect ownership by an ancestor FSDP unit.
        if isinstance(module, FSDPModule):
            continue
        if not any(param.requires_grad for param in module.parameters()):
            continue
        fully_shard_fn(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=module_reshard_after_forward,
            offload_policy=offload_policy,
        )
        logger.info("Sharded %s as an independent FSDP unit", role)


class DefaultParallelizationStrategy(ParallelizationStrategy):
    """Default parallelization strategy used by most models."""

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        enable_async_tensor_parallel: bool = False,
        enable_compile: bool = False,
        enable_fsdp2_prefetch: bool = True,
        fsdp2_backward_prefetch_depth: int = 2,
        fsdp2_forward_prefetch_depth: int = 1,
        reshard_after_forward: bool | None = None,
        activation_checkpointing_scope: ActivationCheckpointingScope | None = "all",
        frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
        reapply_trainability: Callable[[nn.Module], None] | None = None,
        fully_shard_fn=None,
    ) -> nn.Module:
        """Apply the default parallelization flow."""
        frozen_multimodal_sharding = normalize_frozen_multimodal_sharding(frozen_multimodal_sharding)
        tp_mesh = device_mesh[tp_mesh_name]
        if fully_shard_fn is None:
            fully_shard_fn = fully_shard

        # Set FSDP sharding mesh to context parallel mesh if CP > 1, else default to the data parallel mesh.
        # if dp_replicate_size > 1, use HSDP, else use FSDP
        dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)
        pp_enabled = "pp" in dp_mesh.mesh_dim_names and dp_mesh["pp"].size() > 1
        if pp_enabled and reshard_after_forward is True:
            logger.warning(
                "reshard_after_forward=True overrides the pipeline-parallel default of keeping layer weights "
                "gathered across microbatches. This may increase per-microbatch all-gathers and reduce throughput."
            )

        # Extract layers from the model for parallelization.
        layer_groups = _extract_model_layer_groups(model)

        # TP sharding with enhanced plan generation
        if tp_mesh.size() > 1:
            # async-TP (_micro_pipeline_tp) overlaps ReduceScatter with compute.
            # Without SP, row-parallel layers emit AllReduce (not ReduceScatter),
            # so there is nothing for the micro-pipeline to overlap — force SP on.
            if enable_async_tensor_parallel and not sequence_parallel:
                raise ValueError("enable_async_tensor_parallel=True requires sequence_parallel=True")

            # Validate that attention heads are divisible by TP size
            validate_tp_mesh(model, tp_mesh)

            # Generate or use tensor parallel plan
            model_parallel_plan = {
                k: translate_to_lora(v)
                for k, v in _get_parallel_plan(
                    model,
                    sequence_parallel,
                    tp_shard_plan,
                    tp_size=tp_mesh.size(),
                ).items()
            }

            # Apply tensor parallelism
            if model_parallel_plan:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*could not be resolved.*",
                        category=UserWarning,
                    )
                    parallelize_module(model, tp_mesh, model_parallel_plan)
                # TP styles replace module weights with DTensors independently.
                # Restore an architectural embedding/head alias before FSDP
                # records ownership; re-tying after FSDP can leave two roots
                # that disagree about the shared parameter.
                ensure_tied_lm_head(model)
                if _attention_is_head_sharded(model_parallel_plan):
                    _update_attention_head_counts_for_tp(model, tp_mesh.size())

            if enable_async_tensor_parallel:
                torch._inductor.config._micro_pipeline_tp = True
                logger.info("Async tensor parallel enabled — ensure torch.compile is also enabled")
                # Enable symmetric memory for the TP group so Inductor's
                # fused_all_gather_matmul and fused_matmul_reduce_scatter kernels
                # can fire (both are gated on is_symm_mem_enabled_for_group).
                if tp_mesh.size() > 1:
                    try:
                        from torch.distributed._symmetric_memory import enable_symm_mem_for_group

                        tp_group_name = tp_mesh._dim_group_names[0]
                        enable_symm_mem_for_group(tp_group_name)
                        logger.info(f"Symmetric memory enabled for TP group '{tp_group_name}'")
                    except Exception as e:
                        logger.warning(f"Could not enable symmetric memory for TP group: {e}")

        # Apply activation checkpointing to transformer blocks if requested
        if activation_checkpointing:
            _has_kv_sharing = detect_kv_sharing_and_maybe_disable_cache(model)
            ac_layers, ac_scopes = _filter_layer_groups_for_activation_checkpointing(
                layer_groups,
                activation_checkpointing_scope,
            )

            if is_selective_activation_checkpointing(activation_checkpointing):
                apply_selective_checkpointing_to_layers(
                    model,
                    ac_layers,
                    _has_kv_sharing,
                    enable_compile=enable_compile,
                )
            elif ac_scopes == ("all",) and _apply_bagel_full_layer_activation_checkpointing(model):
                logger.info("Using BAGEL full-layer activation checkpointing; skipping submodule checkpoint wrappers.")
            elif enable_compile:
                # NO_REENTRANT is required for compile: REENTRANT's first forward runs under
                # no_grad, causing AOT autograd to trace a forward-only graph that drops LoRA
                # (and other trainable) weight gradients.  Wrapping must happen BEFORE FSDP2
                # sharding so the module structure is stable when fully_shard() indexes params.
                for layer in ac_layers:
                    for attr in ("self_attn", "attention", "attn", "linear_attn", "mlp", "feed_forward", "ffn"):
                        m = getattr(layer, attr, None)
                        if m is not None:
                            setattr(layer, attr, checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT))
            else:
                if _should_use_hf_native_gradient_checkpointing(
                    model,
                    layer_groups,
                    ac_scopes,
                    enable_compile=enable_compile,
                ) and (not _has_kv_sharing or _kv_sharing_survives_checkpoint_replay(model)):
                    # Work around a PyTorch FSDP2 bug that skips mixed-precision input casts during
                    # checkpoint recomputation. Remove when the minimum PyTorch version is 2.13.
                    apply_full_layer_checkpointing_to_layers(model, ac_layers)
                else:
                    apply_submodule_checkpointing(ac_layers, _has_kv_sharing)

        if reapply_trainability is not None:
            reapply_trainability(model)

        # Evaluate frozen-module ownership only after TP/AC transformations and
        # trainability rebinding so FSDP sees the final module hierarchy.
        frozen_multimodal_modules = [
            name for name, module in iter_multimodal_modules(model) if module_is_fully_frozen(module)
        ]
        if frozen_multimodal_sharding == "per_layer" and frozen_multimodal_modules:
            logger.warning(
                "distributed.multimodal.frozen_sharding='per_layer' selected for %s. Every rank in the FSDP "
                "group must execute or skip these modules the same number of times and in the same order on every "
                "microbatch; rank-asymmetric modality execution can hang or desynchronize FSDP collectives.",
                ", ".join(frozen_multimodal_modules),
            )

        # Set up mixed precision policy
        if not mp_policy:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.float32,
            )

        # Install this only when NeMo actually enters FSDP2 sharding.
        _patch_fsdp_accumulated_grad_guard()

        ignored_multimodal_params: set[nn.Parameter] = set()

        # Wrapping distinct modules separately replaces their shared Parameter
        # and breaks the tie. Reject this layout before recursive sharding.
        ensure_tied_lm_head(model)
        input_embeddings, output_embeddings = _get_input_output_embeddings(model)
        input_weight = getattr(input_embeddings, "weight", None)
        output_weight = getattr(output_embeddings, "weight", None)
        if input_embeddings is not output_embeddings and input_weight is not None and input_weight is output_weight:
            for module in model.modules():
                if isinstance(module, (nn.ModuleList, nn.ModuleDict)) and any(
                    child is input_embeddings or child is output_embeddings for child in module.modules()
                ):
                    raise ValueError(
                        "Distinct tied input/output embedding modules inside a ModuleList or ModuleDict are not "
                        "supported by recursive FSDP sharding. Keep both tied modules outside these containers, "
                        "or use a single shared embedding module."
                    )

        # Find transformer layers and apply parallelisms
        self._apply_fsdp_sharding(
            model,
            dp_mesh,
            mp_policy,
            offload_policy,
            enable_fsdp2_prefetch,
            fsdp2_backward_prefetch_depth,
            fsdp2_forward_prefetch_depth,
            reshard_after_forward,
            fully_shard_fn=fully_shard_fn,
            frozen_multimodal_sharding=frozen_multimodal_sharding,
            ignored_multimodal_params=ignored_multimodal_params,
        )

        input_embedding_reshard_after_forward = (
            reshard_after_forward if reshard_after_forward is not None else not pp_enabled
        )
        _fully_shard_untied_input_output_embeddings(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            input_reshard_after_forward=input_embedding_reshard_after_forward,
            fully_shard_fn=fully_shard_fn,
        )

        # Apply FSDP to the root model
        # Do not reshard after forward for root model because its parameters
        # will be used in backward immediately
        root_ignored_params = ignored_params_for_root(model, ignored_multimodal_params)
        root_kwargs = {
            "mesh": dp_mesh,
            "mp_policy": mp_policy,
            "reshard_after_forward": False,
            "offload_policy": offload_policy,
        }
        if root_ignored_params is not None:
            root_kwargs["ignored_params"] = root_ignored_params
        model = fully_shard_fn(model, **root_kwargs)

        cp_enabled = "cp" in device_mesh.mesh_dim_names and device_mesh["cp"].size() > 1
        if cp_enabled:
            configured_units = parallelizer_utils.configure_fsdp_unused_param_reduction(model)
            logger.info(
                "Enabled unused-parameter reduce-scatter on %d FSDP units for context parallelism",
                configured_units,
            )

        return model

    def _apply_fsdp_sharding(
        self,
        module: nn.Module,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None,
        offload_policy: OffloadPolicy | None = None,
        enable_fsdp2_prefetch: bool = True,
        fsdp2_backward_prefetch_depth: int = 2,
        fsdp2_forward_prefetch_depth: int = 1,
        reshard_after_forward: bool | None = None,
        fully_shard_fn=None,
        frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
        ignored_multimodal_params: set[nn.Parameter] | None = None,
    ) -> None:
        """Wrap the model's submodules into FSDP2 units.

        Strategies deriving from this class (in-tree or registered through
        :func:`register_parallel_strategy`) override this hook to change how parameters
        are grouped into FSDP units without reimplementing the surrounding
        TP/AC/mixed-precision flow. ``fully_shard_fn`` selects the primitive that wraps
        each unit and is also used for the root and embedding units, so overrides
        should honor it.
        """
        apply_fsdp2_sharding_recursively(
            module,
            mesh,
            mp_policy,
            offload_policy,
            enable_fsdp2_prefetch,
            fsdp2_backward_prefetch_depth,
            fsdp2_forward_prefetch_depth,
            reshard_after_forward,
            fully_shard_fn=fully_shard_fn,
            frozen_multimodal_sharding=frozen_multimodal_sharding,
            ignored_multimodal_params=ignored_multimodal_params,
        )


def _nemotronh_decoder_blocks(model: nn.Module) -> tuple[nn.Module, list[nn.Module]]:
    """Return ``(container, blocks)`` for a NemotronH model's decoder blocks.

    Two distinct classes share the name ``NemotronHForCausalLM``:

    * the HF model keeps its blocks in ``model.backbone.layers`` (an ``nn.ModuleList``), while
    * the native Nemotron-V3 model (``NemotronV3Model``) keeps them in ``model.model.layers``
      (an ``nn.ModuleDict`` keyed ``"0".."N-1"``).

    ``container`` is the underlying ``ModuleList``/``ModuleDict`` (so callers can write rewrapped
    blocks back into the model), and ``blocks`` is the ordered list of block modules.
    """
    inner = model.backbone if hasattr(model, "backbone") else model.model
    container = inner.layers
    blocks = list(container.values()) if isinstance(container, nn.ModuleDict) else list(container)
    return container, blocks


class NemotronHParallelizationStrategy(ParallelizationStrategy):
    """Specialized parallelization strategy for NemotronH models."""

    def parallelize(
        self,
        model: nn.Module,
        device_mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None = None,
        offload_policy: OffloadPolicy | None = None,
        sequence_parallel: bool = False,
        activation_checkpointing: bool = False,
        tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
        dp_replicate_mesh_name: str = "dp_replicate",
        dp_shard_cp_mesh_name: str = "dp_shard_cp",
        tp_mesh_name: str = "tp",
        reshard_after_forward: bool | None = None,
        reapply_trainability: Callable[[nn.Module], None] | None = None,
        **kwargs,
    ) -> nn.Module:
        """Apply NemotronH-specific parallelization."""
        assert not sequence_parallel, "Sequence parallelism is not supported for NemotronHForCausalLM"
        logger.info("Custom parallel plan is not supported for NemotronHForCausalLM. Using NemotronH-specific TP plan.")

        block_container, layers = _nemotronh_decoder_blocks(model)
        tp_mesh = device_mesh[tp_mesh_name]
        if tp_mesh.size() > 1:
            model_tp_plan: dict[str, ParallelStyle] = {
                "lm_head": translate_to_lora(ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)),
            }

            mlp_tp_plan: dict[str, ParallelStyle] = {
                "mixer.up_proj": translate_to_lora(ColwiseParallel()),
                "mixer.down_proj": translate_to_lora(RowwiseParallel()),
            }

            parallelize_module(model, tp_mesh, model_tp_plan)

            for layer in layers:
                if layer.block_type == "mlp":
                    parallelize_module(layer, tp_mesh, mlp_tp_plan)

        # Set up context parallel for Mamba and Attention layers
        cp_mesh = device_mesh["cp"] if "cp" in device_mesh.mesh_dim_names else None
        if cp_mesh is not None and cp_mesh.size() > 1:
            cp_group = cp_mesh.get_group()
            cp_global_ranks = torch.distributed.get_process_group_ranks(cp_group)
            cp_layers = list(layers)
            mtp_module = getattr(model, "mtp", None)
            mtp_layers = getattr(mtp_module, "layers", None)
            mtp_cp_enabled = model.supports.mtp_enabled
            parallelizer_utils.reject_unsupported_mtp_cp_pp(model)
            parallelizer_utils.reject_unsupported_mtp_cp(model)
            if mtp_cp_enabled and mtp_layers is None:
                raise RuntimeError(
                    "MTP is enabled but model.mtp.layers is unavailable; cannot configure context parallelism for MTP"
                )
            if mtp_cp_enabled and mtp_layers is not None:
                # MTP blocks live outside the backbone container but execute
                # the same attention/Mamba CP collectives.
                cp_layers.extend(mtp_layers)

            for layer in cp_layers:
                if hasattr(layer, "block_type") and layer.block_type == "mamba":
                    from nemo_automodel.components.distributed.context_parallel.mamba import MambaContextParallel

                    mixer = layer.mixer
                    mixer.cp = MambaContextParallel(
                        cp_group=cp_group,
                        num_heads=mixer.num_heads,
                        head_dim=mixer.head_dim,
                        n_groups=mixer.n_groups,
                        d_state=mixer.ssm_state_size,
                        mixer=mixer,
                    )
                elif hasattr(layer, "block_type") and layer.block_type == "attention":
                    from transformer_engine.pytorch.attention import DotProductAttention

                    attn_module = layer.mixer.attn_module
                    if isinstance(attn_module, DotProductAttention):
                        attn_module.set_context_parallel_group(
                            cp_group,
                            cp_global_ranks,
                            torch.cuda.Stream(),
                            cp_comm_type="p2p",
                        )

        if activation_checkpointing:
            # Write rewrapped blocks back into the real container (ModuleList -> int key,
            # ModuleDict -> str key) so the model, not just the local handle, is updated.
            block_items = (
                block_container.items() if isinstance(block_container, nn.ModuleDict) else enumerate(block_container)
            )
            for key, layer in list(block_items):
                if getattr(layer, "block_type", None) in ("mlp", "mamba"):
                    block_container[key] = checkpoint_wrapper(layer)
            # Refresh the local handle so the FSDP wrap below sees the wrapped blocks.
            _, layers = _nemotronh_decoder_blocks(model)

        if reapply_trainability is not None:
            reapply_trainability(model)

        dp_mesh = get_fsdp_dp_mesh(device_mesh, dp_replicate_mesh_name, dp_shard_cp_mesh_name)

        fp32_compute_module_names = tuple(getattr(model, "_keep_in_fp32_modules_strict", None) or ())

        for layer in layers:
            parallelizer_utils.fully_shard_by_dtype(
                layer,
                mesh=dp_mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                fp32_compute_module_names=fp32_compute_module_names,
                reshard_after_forward=reshard_after_forward,
            )

        # do not reshard after forward for root model
        # because its parameters will be used in backward immediately
        return fully_shard(
            model,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=False,
        )


class Qwen3_5ParallelizationStrategy(DefaultParallelizationStrategy):
    """Parallelization strategy for Qwen3.5 dense models with mixed-dtype GatedDeltaNet.

    Qwen3.5 has linear_attn layers with float32 params (A_log, norm) alongside
    bfloat16 params. Overrides the FSDP sharding step to use fully_shard_by_dtype
    per layer, and sets the CP mesh on CPAwareGatedDeltaNet modules.
    """

    # The Qwen3.5 model builds CPAwareGatedDeltaNet with a fp32 ``SSMGate``
    # (``_fp32_params``) at construction — no runtime patch needed. Keep those
    # params in their own dtype-uniform fp32 FSDP group (true master weights).
    _fp32_compute_module_names: tuple[str, ...] = ("_fp32_params",)

    def _apply_fsdp_sharding(
        self,
        module: nn.Module,
        mesh: DeviceMesh,
        mp_policy: MixedPrecisionPolicy | None,
        offload_policy: OffloadPolicy | None = None,
        enable_fsdp2_prefetch: bool = True,
        fsdp2_backward_prefetch_depth: int = 2,
        fsdp2_forward_prefetch_depth: int = 1,
        reshard_after_forward: bool | None = None,
        fully_shard_fn=None,
        frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
        ignored_multimodal_params: set[nn.Parameter] | None = None,
    ) -> None:
        """Shard each decoder layer with :func:`fully_shard_by_dtype`.

        Overrides the default recursive walk so fp32 and bfloat16 parameters end up
        in separate, dtype-uniform FSDP groups. ``fully_shard_fn`` is forwarded to
        every unit; the prefetch knobs are not supported by the dtype walk.
        """
        del enable_fsdp2_prefetch, fsdp2_backward_prefetch_depth, fsdp2_forward_prefetch_depth
        frozen_multimodal_sharding = normalize_frozen_multimodal_sharding(frozen_multimodal_sharding)
        pp_enabled = "pp" in mesh.mesh_dim_names and mesh["pp"].size() > 1

        if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
            all_items = list(module.items()) if isinstance(module, nn.ModuleDict) else list(enumerate(module))
            flat_layer_items = [
                (layer_id, child)
                for layer_id, child in all_items
                if not isinstance(child, (nn.ModuleList, nn.ModuleDict))
            ]
            nested_items = [
                (layer_id, child) for layer_id, child in all_items if isinstance(child, (nn.ModuleList, nn.ModuleDict))
            ]

            for _, child in nested_items:
                self._apply_fsdp_sharding(
                    child,
                    mesh,
                    mp_policy,
                    offload_policy,
                    reshard_after_forward=reshard_after_forward,
                    fully_shard_fn=fully_shard_fn,
                    frozen_multimodal_sharding=frozen_multimodal_sharding,
                    ignored_multimodal_params=ignored_multimodal_params,
                )

            for enum_id, (_, child) in enumerate(flat_layer_items):
                if reshard_after_forward is not None:
                    layer_reshard_after_forward = reshard_after_forward
                elif pp_enabled:
                    layer_reshard_after_forward = False
                else:
                    layer_reshard_after_forward = enum_id < len(flat_layer_items) - 1
                parallelizer_utils.fully_shard_by_dtype(
                    child,
                    mesh,
                    mp_policy,
                    offload_policy,
                    fp32_compute_module_names=self._fp32_compute_module_names,
                    reshard_after_forward=layer_reshard_after_forward,
                    fully_shard_fn=fully_shard_fn,
                )
        else:
            for name, sub in module.named_children():
                if is_multimodal_module_name(name) and module_is_fully_frozen(sub):
                    if frozen_multimodal_sharding in ("root", "replicate"):
                        logger.info(
                            "Keeping frozen multimodal module %s at FSDP policy %s",
                            name,
                            frozen_multimodal_sharding,
                        )
                        if frozen_multimodal_sharding == "replicate" and ignored_multimodal_params is not None:
                            ignored_multimodal_params.update(module_parameters(sub))
                        continue
                self._apply_fsdp_sharding(
                    sub,
                    mesh,
                    mp_policy,
                    offload_policy,
                    reshard_after_forward=reshard_after_forward,
                    fully_shard_fn=fully_shard_fn,
                    frozen_multimodal_sharding=frozen_multimodal_sharding,
                    ignored_multimodal_params=ignored_multimodal_params,
                )

    def parallelize(self, model, device_mesh, dp_shard_cp_mesh_name="dp_shard_cp", **kwargs):
        cp_mesh_name = dp_shard_cp_mesh_name.replace("dp_shard_", "")
        cp_enabled = cp_mesh_name in device_mesh.mesh_dim_names and device_mesh[cp_mesh_name].size() > 1

        # TP, AC and mixed precision come from the default strategy; the FSDP wrapping
        # step is customized through the ``_apply_fsdp_sharding`` hook above.
        result = super().parallelize(
            model,
            device_mesh,
            dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
            **kwargs,
        )

        # Set CP mesh on CPAwareGatedDeltaNet modules
        if cp_enabled:
            from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet

            cp_mesh = device_mesh[cp_mesh_name]
            for _, mod in model.named_modules():
                if isinstance(mod, CPAwareGatedDeltaNet):
                    mod._cp_mesh = cp_mesh
            # Hand the CP submesh to the model so a forward that embeds and
            # sequence-shards its own primary stream (Megatron-style per-microbatch
            # CP; see shard_sequence_for_cp_round_robin / shard_batch_aux_only) can build this
            # rank's round-robin shard.
            model.cp_mesh = cp_mesh

        return result


class DeepseekV4ParallelizationStrategy(DefaultParallelizationStrategy):
    """DeepSeek-V4 keeps a small set of reference-sensitive parameters in fp32."""

    def parallelize(self, model, device_mesh, dp_shard_cp_mesh_name="dp_shard_cp", **kwargs):
        from nemo_automodel.components.models.deepseek_v4.fsdp import fully_shard_deepseek_v4

        return super().parallelize(
            model,
            device_mesh,
            dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
            fully_shard_fn=fully_shard_deepseek_v4,
            **kwargs,
        )


# Strategy registry mapping model class names to parallelization strategies
PARALLELIZATION_STRATEGIES: Dict[str, ParallelizationStrategy] = {
    "NemotronHForCausalLM": NemotronHParallelizationStrategy(),
    "DeepseekV4ForCausalLM": DeepseekV4ParallelizationStrategy(),
    "Qwen3_5ForConditionalGeneration": Qwen3_5ParallelizationStrategy(),
    "Qwen3_5ForCausalLM": Qwen3_5ParallelizationStrategy(),
}

# Default strategy instance
_DEFAULT_STRATEGY = DefaultParallelizationStrategy()


def get_parallelization_strategy(model: nn.Module) -> ParallelizationStrategy:
    """Get the appropriate parallelization strategy for the given model."""
    model_name = type(model).__name__
    return PARALLELIZATION_STRATEGIES.get(model_name, _DEFAULT_STRATEGY)


def register_parallel_strategy(arg=None, *, name: str | None = None):
    """Decorator to register out-of-tree parallelism strategies.

    Supports:
    - @register_parallel_strategy(name="CustomModelName")
    """

    def _register(cls):
        # The decorator receives a class, not an instance.
        assert isinstance(cls, type) and issubclass(cls, ParallelizationStrategy), (
            f"cls must be a subclass of ParallelizationStrategy, but got {type(cls)} {cls}"
        )
        assert name is not None, "name is required"
        assert name not in PARALLELIZATION_STRATEGIES, f"name {name} already registered"
        PARALLELIZATION_STRATEGIES[name] = cls()
        return cls

    if name is None:
        raise ValueError("name is required")
    # If used with parentheses (possibly with arguments)
    return _register


def _patch_dtensor_spec_hash_for_symint() -> None:
    """Fix a crash when torch.compile + DTensor are used together.

    Problem: torch.compile traces with symbolic shapes (SymInt). DTensorSpec hashes
    its shape to cache sharding decisions, but SymInt is not hashable -> crash.

    Fix: if hashing the shape fails, fall back to hashing only (mesh, placements).
    Cache hits are slightly reduced but correctness is unaffected.
    """
    from torch.distributed.tensor._dtensor_spec import DTensorSpec

    if getattr(DTensorSpec, "_symint_hash_patched", False):
        return

    _original_hash_impl = DTensorSpec._hash_impl

    def _hash_impl_symint_safe(self) -> int:
        try:
            return _original_hash_impl(self)
        except TypeError:
            return hash((self.mesh, self.placements))

    DTensorSpec._hash_impl = _hash_impl_symint_safe
    DTensorSpec._symint_hash_patched = True


def _apply_per_layer_compile(model: nn.Module) -> None:
    """Compile each decoder layer in-place after FSDP2 sharding.

    Compiles at decoder-layer granularity (not sub-module) so that AOT autograd traces
    the joint fwd+bwd graph under the training loop's enable_grad context.  Sub-module
    compile (e.g. on mlp alone) would be traced during activation checkpointing's first
    forward pass which runs under no_grad, producing a forward-only graph that drops
    LoRA and other trainable-parameter gradients.

    Prerequisite: NO_REENTRANT checkpoint_wrapper must already be applied to self_attn
    and mlp before FSDP2 sharding (done in DefaultParallelizationStrategy).  This
    function only handles the compile step.

    Whole-block selective-AC wrappers (tagged with ``SELECTIVE_AC_WRAPPER_FLAG``)
    are compiled OUTER -- the wrapper itself is compiled so the selective policy
    is traced and the partitioner honors its recompute tags. Other layer-level
    CheckpointWrappers (e.g. the PP path) are unwrapped and the decoder layer is
    compiled directly.

    nn.Module.compile() is used instead of torch.compile() to compile in-place without
    introducing an _orig_mod wrapper, which would add a key prefix and break checkpoint
    loading.

    _patch_dtensor_spec_hash_for_symint() is called to allow torch.compile with dynamic
    shapes to coexist with DTensor's lru_cache-based sharding propagation.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    _patch_dtensor_spec_hash_for_symint()

    compiled_count = 0
    compiled_modules: set[int] = set()

    def _compile_target(layer: nn.Module) -> nn.Module:
        # Whole-block selective-AC wrappers must be compiled OUTER so the SAC
        # policy is traced and the partitioner honors its recompute tags.
        # Other CheckpointWrappers (e.g. PP full-layer wrap with sub-module AC
        # inside) are unwrapped so the decoder layer is compiled directly.
        if isinstance(layer, CheckpointWrapper):
            if getattr(layer, SELECTIVE_AC_WRAPPER_FLAG, False):
                return layer
            return layer._checkpoint_wrapped_module
        return layer

    def _compile_module_list(module_list: nn.ModuleList | nn.ModuleDict) -> None:
        nonlocal compiled_count
        # PP converts model.model.layers from nn.ModuleList to nn.ModuleDict (str keys).
        # enumerate(nn.ModuleDict) yields string keys, not modules -- use .items() instead.
        items = module_list.items() if isinstance(module_list, nn.ModuleDict) else enumerate(module_list)
        for _, layer in items:
            actual_layer = _compile_target(layer)
            module_id = id(actual_layer)
            if module_id in compiled_modules:
                continue
            actual_layer.compile()
            compiled_modules.add(module_id)
            compiled_count += 1

    module_lists = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        module_lists.append(model.model.layers)
    if hasattr(model, "layers"):
        module_lists.append(model.layers)
    for attr_name in ("transformer_blocks", "single_transformer_blocks"):
        module_list = getattr(model, attr_name, None)
        if isinstance(module_list, (nn.ModuleList, nn.ModuleDict)):
            module_lists.append(module_list)

    if module_lists:
        for module_list in module_lists:
            _compile_module_list(module_list)
    else:
        logger.warning("_apply_per_layer_compile: using heuristic layer extraction")
        for layer in _extract_model_layers(model):
            actual_layer = _compile_target(layer)
            module_id = id(actual_layer)
            if module_id in compiled_modules:
                continue
            actual_layer.compile()
            compiled_modules.add(module_id)
            compiled_count += 1

    logger.info("Per-layer torch.compile applied to %d decoder layers", compiled_count)


def apply_fsdp2_sharding_recursively(
    module: nn.Module,
    mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None,
    offload_policy: OffloadPolicy | None = None,
    enable_fsdp2_prefetch: bool = True,
    fsdp2_backward_prefetch_depth: int = 2,
    fsdp2_forward_prefetch_depth: int = 1,
    reshard_after_forward: bool | None = None,
    fully_shard_fn=None,
    frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
    ignored_multimodal_params: set[nn.Parameter] | None = None,
) -> None:
    """
    Recursively apply FSDP2 sharding to modules, with optimizations for ModuleList.

    This utility function traverses a model hierarchy and applies FSDP2 sharding
    to each module. For ModuleList instances (commonly used for transformer layers),
    it applies an optimization where the last layer doesn't reshard after forward
    since FSDP2 will prefetch it immediately.

    Handles both single-level and nested ModuleList/ModuleDict structures. If a
    ModuleList contains other ModuleLists, it will recurse into them instead of trying
    to wrap them (since ModuleList doesn't have a forward method).

    Args:
        module (nn.Module): The module to apply FSDP sharding to.
        mesh (DeviceMesh): The device mesh for FSDP sharding.
        mp_policy (Optional[MixedPrecisionPolicy]): Mixed precision policy for FSDP.
        offload_policy (Optional[OffloadPolicy]): CPU offload policy for FSDP.
            Defaults to None.
        enable_fsdp2_prefetch (bool): Enable explicit forward/backward prefetch chains.
        fsdp2_backward_prefetch_depth (int): Backward prefetch depth.
        fsdp2_forward_prefetch_depth (int): Forward prefetch depth.
        reshard_after_forward (Optional[bool]): Optional override for each layer's
            ``fully_shard`` reshard behavior.
        frozen_multimodal_sharding: Whether fully frozen multimodal modules are
            owned by the root FSDP unit, sharded per layer, or replicated.
        ignored_multimodal_params: Accumulator for replicated frozen multimodal
            parameters that must be ignored by ancestor FSDP roots.
    Note:
        This function modifies the module in-place by replacing modules with their
        FSDP2-subclassed versions.
    """
    frozen_multimodal_sharding = normalize_frozen_multimodal_sharding(frozen_multimodal_sharding)
    if fully_shard_fn is None:
        fully_shard_fn = fully_shard

    pp_enabled = "pp" in mesh.mesh_dim_names and mesh["pp"].size() > 1

    if isinstance(module, (nn.ModuleList, nn.ModuleDict)):
        # After pipeline splitting, functional.py replaces nn.ModuleList with nn.ModuleDict
        # (keyed by string layer indices). Normalise both to a list of (key, child) pairs.
        if isinstance(module, nn.ModuleDict):
            all_items = list(module.items())
            _is_container = lambda c: isinstance(c, (nn.ModuleList, nn.ModuleDict))
        else:
            all_items = [(i, module[i]) for i in range(len(module))]
            _is_container = lambda c: isinstance(c, nn.ModuleList)

        flat_layer_items = [(k, c) for k, c in all_items if not _is_container(c)]
        nested_items = [(k, c) for k, c in all_items if _is_container(c)]
        nested_lists = nested_items  # kept for len() checks below

        # Recurse into any nested ModuleLists first (unchanged behavior).
        for layer_id, child_module in nested_lists:
            apply_fsdp2_sharding_recursively(
                child_module,
                mesh,
                mp_policy,
                offload_policy,
                enable_fsdp2_prefetch,
                fsdp2_backward_prefetch_depth,
                fsdp2_forward_prefetch_depth,
                reshard_after_forward,
                fully_shard_fn=fully_shard_fn,
                frozen_multimodal_sharding=frozen_multimodal_sharding,
                ignored_multimodal_params=ignored_multimodal_params,
            )

        for enum_id, (layer_key, child_module) in enumerate(flat_layer_items):
            # With PP: keep weights gathered across microbatches (no per-microbatch all-gather).
            # Without PP: reshard all but last layer to enable forward+backward weight prefetching.
            if reshard_after_forward is not None:
                layer_reshard_after_forward = reshard_after_forward
            elif pp_enabled:
                layer_reshard_after_forward = False
            else:
                layer_reshard_after_forward = enum_id < len(flat_layer_items) - 1
            fully_shard_fn(
                child_module,
                mesh=mesh,
                mp_policy=mp_policy,
                reshard_after_forward=layer_reshard_after_forward,
                offload_policy=offload_policy,
            )
            module[layer_key] = child_module

        # Set up explicit forward/backward prefetch chains when layers are being resharded.
        # With PP or an explicit no-reshard override, weights are always gathered -- no prefetch needed.
        if reshard_after_forward is False:
            should_prefetch = False
        else:
            should_prefetch = not pp_enabled and enable_fsdp2_prefetch
        if should_prefetch:
            fsdp_units = [c for _, c in flat_layer_items if not _is_container(c)]
            if fsdp2_forward_prefetch_depth > 0:
                for i in range(len(fsdp_units) - 1):
                    targets = [
                        fsdp_units[i + j] for j in range(1, fsdp2_forward_prefetch_depth + 1) if i + j < len(fsdp_units)
                    ]
                    if targets:
                        fsdp_units[i].set_modules_to_forward_prefetch(targets)
            for i in range(1, len(fsdp_units)):
                targets = []
                for d in range(1, fsdp2_backward_prefetch_depth + 1):
                    if i - d >= 0:
                        targets.append(fsdp_units[i - d])
                if targets:
                    fsdp_units[i].set_modules_to_backward_prefetch(targets)
    else:
        for name, sub_module in module.named_children():
            if is_multimodal_module_name(name) and module_is_fully_frozen(sub_module):
                if frozen_multimodal_sharding in ("root", "replicate"):
                    logger.info(
                        "Keeping frozen multimodal module %s at FSDP policy %s",
                        name,
                        frozen_multimodal_sharding,
                    )
                    if frozen_multimodal_sharding == "replicate" and ignored_multimodal_params is not None:
                        ignored_multimodal_params.update(module_parameters(sub_module))
                    continue
            apply_fsdp2_sharding_recursively(
                sub_module,
                mesh,
                mp_policy,
                offload_policy,
                enable_fsdp2_prefetch,
                fsdp2_backward_prefetch_depth,
                fsdp2_forward_prefetch_depth,
                reshard_after_forward,
                fully_shard_fn=fully_shard_fn,
                frozen_multimodal_sharding=frozen_multimodal_sharding,
                ignored_multimodal_params=ignored_multimodal_params,
            )


def get_hf_tp_shard_plan(model):
    """Get the Hugging Face tensor parallel plan from the model.

    This function:
    - Retrieves TP strategies from model class, instance, and inner model levels.
    - Handles special cases for `embed_tokens` and `lm_head` for speed up.
    - Converts string-based parallel styles to DTensor parallelization strategies.

    Taken and modified from: https://github.com/NVIDIA/NeMo/blob/6c6169db01bcca73ae8ad3ac35242fadbb9a78ba/nemo/lightning/pytorch/strategies/utils.py#L532

    Args:
        model: A Hugging Face model instance

    Returns:
        dict: A dictionary mapping model component paths to their parallelization strategies

    Raises:
        AssertionError: If no TP plan is found
    """
    # Imported inside the function, not at module scope: pulling in a single
    # ``transformers.models.*.modeling_*`` module drags the whole model-zoo
    # dependency graph along with it (sklearn -> pandas/scipy, torchvision,
    # opentelemetry). That cost more than ``import torch`` itself and was paid
    # by every process that so much as touched this module -- including each
    # ``mp.spawn`` child in the unit-test suite, which re-imports from scratch.
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
    from transformers.models.llama4.modeling_llama4 import Llama4ForConditionalGeneration
    from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
    from transformers.models.llava_next.modeling_llava_next import LlavaNextForConditionalGeneration
    from transformers.models.llava_next_video.modeling_llava_next_video import (
        LlavaNextVideoForConditionalGeneration,
    )
    from transformers.models.llava_onevision.modeling_llava_onevision import (
        LlavaOnevisionForConditionalGeneration,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration

    model_cls = type(model)

    # Handle VL models structure
    if model_cls in [
        Qwen2VLForConditionalGeneration,
        Qwen2_5_VLForConditionalGeneration,
    ]:
        inner_model = model.model.language_model
        model_prefix = "model.language_model"

    elif model_cls == Gemma3ForConditionalGeneration:
        # Gemma3 releases before the mid-4.x VLM standardization hang the text
        # tower off a top-level `language_model`; later releases nest everything
        # under the shared `model` backbone. Resolve structurally via registered
        # child modules (not `hasattr`) because standardized 4.x releases keep a
        # deprecated `language_model` alias property on the wrapper class.
        if any(name == "language_model" for name, _ in model.named_children()):
            inner_model = model.language_model
            model_prefix = "language_model"
        else:
            inner_model = model.model
            model_prefix = "model"

    elif model_cls == Llama4ForConditionalGeneration:
        inner_model = model.language_model.model
        model_prefix = "language_model.model"

    elif model_cls in [
        LlavaForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        LlavaNextVideoForConditionalGeneration,
        LlavaOnevisionForConditionalGeneration,
    ]:
        inner_model = model.model.language_model
        model_prefix = "model.language_model"

    elif model_cls.__name__ == "Qwen3_5ForConditionalGeneration":
        inner_model = model.model.language_model
        model_prefix = "model.language_model"

    else:
        inner_model = model.model
        model_prefix = "model"

    hf_tp_plan = {}

    # model_cls._tp_plan will override model_cls after xxxForCausalLM.post_init() (transformers==4.51.3)
    if hasattr(model_cls, "_tp_plan") and model_cls._tp_plan is not None:
        assert isinstance(model_cls._tp_plan, dict), f"model_cls._tp_plan is not a dict: {model_cls._tp_plan}"
        hf_tp_plan.update(model_cls._tp_plan)

    if hasattr(model, "_tp_plan") and model._tp_plan is not None:
        hf_tp_plan.update(model._tp_plan)

    if hasattr(inner_model, "_tp_plan") and inner_model._tp_plan is not None:
        hf_tp_plan.update({f"{model_prefix}.{k}": v for k, v in inner_model._tp_plan.items()})

    assert len(hf_tp_plan) > 0, (
        f"Hugging Face tp plan is not supported for {model_cls}, please set dtensor_cfg.tensor_parallel_size to 1 or provide a custom_parallel_plan. "
        "The usage example of custom_parallel_plan can refer to `docs/design-docs/fsdp2-parallel-plan.md`."
    )

    # hf tp plan not contain embed_tokens, we add it and set to rowwise_rep
    if f"{model_prefix}.embed_tokens" not in hf_tp_plan:
        hf_tp_plan[f"{model_prefix}.embed_tokens"] = "rowwise_rep"

    # Build translated plan, skipping HF's MoE-related styles.
    #
    # HuggingFace transformers v5 introduced these styles for MoE models, but they do NOT
    # implement true expert parallelism (where each rank stores only a subset of experts).
    # Instead, HF's approach:
    # - local_colwise/local_rowwise: Store expert weights as local tensors (NOT sharded).
    #   Despite the names, these do NOT perform tensor parallelism on the experts.
    #   Each rank stores ALL expert weights (full shape), which is memory inefficient.
    # - ep_router: Modifies routing so each rank only computes with a subset of experts.
    #   This distributes compute but not memory.
    # - gather: All-reduces expert outputs across ranks.
    #
    # Since these styles result in replicated expert weights (not sharded), and we don't
    # support HF's routing modification approach, we skip them entirely. The experts will
    # be replicated across all ranks and computed redundantly, which is correct but not
    # memory/compute efficient for large MoE models.
    _hf_moe_styles = {"ep_router", "local_colwise", "local_rowwise", "gather"}
    translated_plan = {}
    for k, v in hf_tp_plan.items():
        if isinstance(v, str) and (v.startswith("ep_") or v in _hf_moe_styles):
            continue
        # speed up the tp plan for lm_head
        if (k == "lm_head" or k == "language_model.lm_head") and v == "colwise_rep":
            translated_plan[k] = ColwiseParallel(output_layouts=Shard(-1), use_local_output=False)
        else:
            translated_plan[k] = translate_to_torch_parallel_style(v)

    logger.info(f"Hugging Face tp plan: {translated_plan}")
    return translated_plan


def import_class_from_path(name: str) -> Any:
    """Import a class from a string path (e.g. 'torch.optim.AdamW').

    Args:
        full_path: Full path to class including module path and class name

    Returns:
        The imported class object
    """
    module_name, cls_name = name.rsplit(".", 1)
    cls_instance = getattr(importlib.import_module(module_name), cls_name)
    return cls_instance


@lru_cache
def translate_to_torch_parallel_style(style: str):
    """
    Translates string descriptions to parallelism plans.

    In model configurations, we use a neutral type (string) to specify parallel
    styles, here we translate them into torch.distributed tensor-parallel
    types.
    """
    assert isinstance(style, str), f"parallel style type should be str, but got {type(style)}"

    if style == "colwise":
        return ColwiseParallel()
    elif style == "rowwise":
        return RowwiseParallel()
    elif style == "colwise_rep":
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "colwise_gather_output":
        # HF maps this to ColwiseParallel(gather_output=True); gathering the output
        # is the same as replicating it, so this matches "colwise_rep" above.
        return ColwiseParallel(output_layouts=Replicate())
    elif style == "rowwise_rep":
        return RowwiseParallel(input_layouts=Replicate())
    elif style == "sequence_parallel":
        return SequenceParallel()
    elif style == "replicated_with_grad_allreduce":
        return ReplicatedWithGradAllReduce()
    else:
        raise ValueError(f"Unknown parallel style: {style}")


def _attention_is_head_sharded(model_parallel_plan: dict) -> bool:
    """Return True when the TP plan column-wise shards any QKV attention projection.

    When Q/K/V projections use ``ColwiseParallel`` with sharded output (the
    default), each TP rank holds ``num_heads / tp_size`` heads and the model
    config / layer attributes must be updated accordingly.

    Plans that keep attention replicated (e.g. Phi-3 with ``RowwiseParallel``
    on fused QKV and ``Replicate`` output) should *not* trigger a head-count
    update.
    """
    attn_proj_suffixes = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.qkv_proj")
    for key, style in model_parallel_plan.items():
        if not any(key.endswith(s) for s in attn_proj_suffixes):
            continue
        if isinstance(style, ColwiseParallel):
            out = getattr(style, "output_layouts", None)
            if out is None:
                return True
            if isinstance(out, (list, tuple)):
                if any(isinstance(p, Shard) for p in out):
                    return True
            elif isinstance(out, Shard):
                return True
    return False


def _update_attention_head_counts_for_tp(model: nn.Module, tp_size: int) -> None:
    """
    After TP sharding, the Q/K/V outputs are split across ranks (each rank has
    num_heads/tp_size heads). Update the config and each attention layer's
    num_heads / num_key_value_heads so the forward uses the local head count
    instead of the global one (avoids shape mismatches in .view()).
    """
    if tp_size <= 1:
        return
    config = getattr(model, "config", None)
    if config is None or not hasattr(config, "num_attention_heads"):
        return
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None and hasattr(model, "language_model"):
        inner = model.language_model
        layers = getattr(inner, "layers", None)
    if layers is None:
        return
    # Preserve the true head_dim before dividing num_attention_heads.
    # RoPE utilities derive head_dim via getattr(config, "head_dim",
    # config.hidden_size // config.num_attention_heads).  Without an
    # explicit head_dim, the division would compute a wrong (too large)
    # head_dim after we halve num_attention_heads for TP.
    if not hasattr(config, "head_dim") or config.head_dim is None:
        config.head_dim = config.hidden_size // config.num_attention_heads
    local_num_attention_heads = config.num_attention_heads // tp_size
    local_num_key_value_heads = None
    if hasattr(config, "num_key_value_heads") and config.num_key_value_heads is not None:
        local_num_key_value_heads = config.num_key_value_heads // tp_size

    # PP converts ModuleList → ModuleDict; iterating a ModuleDict yields keys, not modules.
    layer_iter = layers.values() if isinstance(layers, nn.ModuleDict) else layers
    for layer in layer_iter:
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            if hasattr(attn, "num_heads"):
                attn.num_heads = local_num_attention_heads
            if hasattr(attn, "num_key_value_heads"):
                # Use config's value if set, else derive from local num_heads and num_key_value_groups (e.g. DeciLM)
                if local_num_key_value_heads is not None:
                    attn.num_key_value_heads = local_num_key_value_heads
                elif hasattr(attn, "num_key_value_groups"):
                    attn.num_key_value_heads = local_num_attention_heads // attn.num_key_value_groups
                else:
                    attn.num_key_value_heads = local_num_attention_heads


def validate_tp_mesh_for_nemotron_nas(model, tp_size):
    """Validate that a Nemotron-NAS model can be tensor-parallel sharded."""
    num_attention_heads = model.config.num_attention_heads
    assert num_attention_heads % tp_size == 0, "num_attention_heads in config does not match the TP size"

    assert len(model.config.block_configs) >= model.config.num_hidden_layers, (
        "num_hidden_layers in config does not match the number of block configs"
    )

    for i in range(model.config.num_hidden_layers):
        # Valid layer
        if model.config.block_configs[i].attention.replace_with_linear:
            print(f"By pass checking for linear layer in layer {i}")
            # TODO: Check if the linear layer could support TP.
        else:
            if model.config.block_configs[i].attention.n_heads_in_group is not None:
                num_key_value_heads = num_attention_heads // model.config.block_configs[i].attention.n_heads_in_group
                assert num_key_value_heads % tp_size == 0, (
                    f"layer {i}: num_key_value_heads in config does not match the TP size"
                )
            else:
                assert model.config.block_configs[i].attention.no_op == True


def validate_tp_mesh(model, tp_mesh):
    """
    Validate that attention heads and key value heads are divisible by TP size
    """
    # Imported here rather than at module scope; see get_hf_tp_shard_plan.

    if tp_mesh.size() == 1:
        return  # if tp_mesh.size() == 1, we don't need to validate

    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration

    Gemma4ForConditionalGeneration = _gemma4_for_conditional_generation()
    from transformers.models.llama4.modeling_llama4 import Llama4ForConditionalGeneration
    from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
    from transformers.models.llava_next.modeling_llava_next import LlavaNextForConditionalGeneration
    from transformers.models.llava_next_video.modeling_llava_next_video import (
        LlavaNextVideoForConditionalGeneration,
    )
    from transformers.models.llava_onevision.modeling_llava_onevision import (
        LlavaOnevisionForConditionalGeneration,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration

    model_cls = type(model)

    # There are cases like DeciLMForCausalLM is defined in transformers_modules
    # which hardly has predefined path to import. Guard access to config/architectures.
    model_arch = None
    if hasattr(model, "config") and hasattr(model.config, "architectures") and model.config.architectures:
        try:
            model_arch = model.config.architectures[0]
        except Exception:
            model_arch = None

    if model_cls in [
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
    ]:
        # VL models have the language model at model.language_model
        num_attention_heads = model.language_model.config.num_attention_heads
        num_key_value_heads = model.language_model.config.num_key_value_heads

    elif model_cls == SmolVLMForConditionalGeneration:
        num_attention_heads = model.model.text_model.config.num_attention_heads
        num_key_value_heads = model.model.text_model.config.num_key_value_heads

    elif model_cls in [
        LlavaForConditionalGeneration,
        LlavaNextForConditionalGeneration,
        LlavaNextVideoForConditionalGeneration,
        LlavaOnevisionForConditionalGeneration,
    ]:
        num_attention_heads = model.language_model.config.num_attention_heads
        num_key_value_heads = model.language_model.config.num_key_value_heads

    elif model_cls == Llama4ForConditionalGeneration:
        num_attention_heads = model.language_model.model.config.num_attention_heads
        num_key_value_heads = model.language_model.model.config.num_key_value_heads

    elif model_cls in [Gemma3ForConditionalGeneration, Gemma4ForConditionalGeneration]:
        num_attention_heads = model.config.text_config.num_attention_heads
        num_key_value_heads = model.config.text_config.num_key_value_heads
    elif model_arch == "DeciLMForCausalLM" and getattr(model.config, "model_type", None) == "nemotron-nas":
        validate_tp_mesh_for_nemotron_nas(model, tp_mesh.size())

        # SKip following code and return.
        return
    elif hasattr(model, "config"):
        num_attention_heads = getattr(model.config, "num_attention_heads", 0)
        num_key_value_heads = getattr(model.config, "num_key_value_heads", 0)
    else:
        num_attention_heads = 0
        num_key_value_heads = 0

    # TP sharding with enhanced plan generation
    # Validate that attention heads are divisible by TP size
    assert num_key_value_heads % tp_mesh.size() == 0, (
        f"num_key_value_heads ({num_key_value_heads}) must be divisible by TP size ({tp_mesh.size()})"
    )
    assert num_attention_heads % tp_mesh.size() == 0, (
        f"num_attention_heads ({num_attention_heads}) must be divisible by TP size ({tp_mesh.size()})"
    )


def _find_largest_module_list(model: nn.Module) -> Union[nn.ModuleList, nn.ModuleDict] | None:
    """
    Heuristic function to find the largest layer container in a model.

    This function recursively traverses the model to find all nn.ModuleList and
    pipeline-split nn.ModuleDict instances and returns the one with the most
    modules. This is useful as a fallback when the model architecture is unknown,
    since transformer layers are typically organized in ModuleLists. Pipeline
    splitting converts ModuleLists to ModuleDicts keyed by original layer index.

    Args:
        model (nn.Module): The model to search through.

    Returns:
        Optional[Union[nn.ModuleList, nn.ModuleDict]]: The largest layer container found, or None.
    """
    largest_module_list: Union[nn.ModuleList, nn.ModuleDict] | None = None
    largest_size = 0

    def _is_pp_layer_module_dict(module: nn.ModuleDict) -> bool:
        # functional.py converts split ModuleLists to ModuleDicts with stringified
        # numeric indices. Avoid treating arbitrary named ModuleDicts (for example
        # adapter registries) as transformer layer containers in the heuristic path.
        return all(key.isdigit() for key in module.keys())

    def _recursive_search(module: nn.Module, path: str = ""):
        nonlocal largest_module_list, largest_size

        for name, child in module.named_children():
            current_path = f"{path}.{name}" if path else name

            if isinstance(child, nn.ModuleList) or (
                isinstance(child, nn.ModuleDict) and _is_pp_layer_module_dict(child)
            ):
                current_size = len(child)
                if current_size > largest_size:
                    largest_size = current_size
                    largest_module_list = child
                    logger.debug(f"Found {type(child).__name__} at {current_path} with {current_size} modules")

            # Continue recursive search
            _recursive_search(child, current_path)

    _recursive_search(model)

    if largest_module_list is not None:
        logger.info(f"Largest layer container found with {largest_size} modules")
    else:
        logger.warning("No ModuleList or ModuleDict found in the model")

    return largest_module_list


def _reduce_attrs(model: nn.Module, fqns: Sequence[str]) -> List[nn.Module]:
    ans = []
    for fqn in fqns:
        parts = fqn.split(".")
        obj = model
        for part in parts:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            ans.append(obj)
    return ans


def _extend_layers(layers: List[nn.Module], modules: Sequence[nn.Module]) -> None:
    for m in modules:
        if isinstance(m, nn.ModuleList):
            layers.extend(m)
        elif isinstance(m, nn.ModuleDict):
            layers.extend(m.values())
        else:
            layers.append(m)


def _get_model_layer_group_specs() -> Dict[Any, Dict[str, List[str]]]:
    # Imported here rather than at module scope; see get_hf_tp_shard_plan.
    from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
    from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

    Gemma4ForConditionalGeneration = _gemma4_for_conditional_generation()
    from transformers.models.llama4.modeling_llama4 import Llama4ForConditionalGeneration
    from transformers.models.llava.modeling_llava import LlavaForConditionalGeneration
    from transformers.models.llava_next.modeling_llava_next import LlavaNextForConditionalGeneration
    from transformers.models.llava_next_video.modeling_llava_next_video import (
        LlavaNextVideoForConditionalGeneration,
    )
    from transformers.models.llava_onevision.modeling_llava_onevision import (
        LlavaOnevisionForConditionalGeneration,
    )
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLForConditionalGeneration
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration

    # Each group lists every known location of its layer container across
    # transformers releases; ``_extract_model_layer_groups`` takes the first
    # candidate that resolves, so the specs need no version gating. The VLM
    # module-tree standardization landed mid-4.x (not at the v5 boundary), so
    # gating paths on ``transformers.__version__`` picks wrong paths for parts
    # of the 4.x line. The shapes below were verified by meta-instantiating
    # each class on transformers 4.51.3, 4.57.1, 5.8.1 and 5.12.1.
    #
    # Gemma3 tree history:
    #   pre-standardization (verified 4.51.3):
    #     `language_model.model.layers` + `vision_tower.vision_model.encoder.layers`
    #   standardized 4.x (verified 4.57.1):
    #     `model.language_model.layers` + `model.vision_tower.vision_model.encoder.layers`
    #   v5 (verified 5.8.1 / 5.12.1): `model.language_model.layers` +
    #     `model.vision_tower.encoder.layers` (SigLIP tower flattened, no inner
    #     `vision_model`).
    # Canonical paths come first: standardized 4.x releases keep deprecated
    # top-level alias properties (`language_model`, `vision_tower`) that also
    # resolve, and first-match-wins must not pick the alias.
    _gemma3_layers = {
        "language": ["model.language_model.layers", "language_model.model.layers"],
        "vision": [
            "model.vision_tower.vision_model.encoder.layers",
            "model.vision_tower.encoder.layers",
            "vision_tower.vision_model.encoder.layers",
        ],
    }
    # Qwen2-VL / Qwen2.5-VL tree history:
    #   pre-standardization (verified 4.51.3): `model.layers` + `visual.blocks`
    #   standardized 4.x and v5 (verified 4.57.1 / 5.8.1 / 5.12.1):
    #     `model.language_model.layers` + `model.visual.blocks` (4.x also keeps
    #     deprecated top-level `language_model` / `visual` alias properties).
    _qwen2_vl_layers = {
        "language": ["model.language_model.layers", "model.layers"],
        "vision": ["model.visual.blocks", "visual.blocks"],
    }
    # Llava family: same tree history as Gemma3 (CLIP instead of SigLIP tower),
    # verified on the same versions for Llava/LlavaNext/LlavaNextVideo/
    # LlavaOnevision.
    _llava_layers = {
        "language": ["model.language_model.layers", "language_model.model.layers"],
        "vision": [
            "model.vision_tower.vision_model.encoder.layers",
            "model.vision_tower.encoder.layers",
            "vision_tower.vision_model.encoder.layers",
        ],
    }
    return {
        Gemma3ForConditionalGeneration: _gemma3_layers,
        Qwen2_5_VLForConditionalGeneration: _qwen2_vl_layers,
        Qwen2VLForConditionalGeneration: _qwen2_vl_layers,
        # Note: `model.` is not a mistake here, it's the full fqn.
        SmolVLMForConditionalGeneration: {
            "language": ["model.text_model.layers"],
            "vision": ["model.vision_model.encoder.layers"],
        },
        LlavaForConditionalGeneration: _llava_layers,
        LlavaNextForConditionalGeneration: _llava_layers,
        LlavaNextVideoForConditionalGeneration: _llava_layers,
        LlavaOnevisionForConditionalGeneration: _llava_layers,
        Llama4ForConditionalGeneration: {
            "language": ["language_model.model.layers"],
            "vision": ["vision_model.model.layers"],
        },
        # String-keyed to avoid eagerly importing transformers.models.qwen3_5 at
        # module load (which would defeat test monkeypatches that stub the
        # module before first import).
        "Qwen3_5ForConditionalGeneration": {
            "language": ["model.language_model.layers"],
            "vision": ["model.visual.blocks"],
        },
        "Qwen3_5MoeForConditionalGeneration": {
            "language": ["model.language_model.layers"],
            "vision": ["model.visual.blocks"],
        },
        "Qwen3VLMoeForConditionalGeneration": {
            "language": ["model.language_model.layers"],
            "vision": ["model.visual.blocks"],
        },
        Gemma4ForConditionalGeneration: {"language": ["model.language_model.layers"]},
        # String fallback in case of class identity mismatch across imports.
        "Gemma4ForConditionalGeneration": {"language": ["model.language_model.layers"]},
        "KimiVLForConditionalGeneration": {
            "language": ["model.language_model.layers"],
            "vision": ["model.vision_tower.encoder.blocks"],
        },
        "KimiK25VLForConditionalGeneration": {
            "language": ["model.language_model.layers"],
            "vision": ["model.vision_tower.encoder.blocks"],
        },
        "MiniMaxM3SparseForConditionalGeneration": {
            "language": ["model.layers"],
            "vision": ["vision_tower.vision_model.encoder.layers"],
        },
        "Step3p7ForConditionalGeneration": {
            "language": ["model.language_model.layers"],
            "vision": ["model.vision_model.transformer.resblocks"],
        },
        "DeepseekV4ForCausalLM": {
            "language": ["model.layers"],
            "vision": ["model.vision.blocks"],
        },
        # BAGEL (text-to-image + understanding). String-keyed to avoid an
        # import cycle: parallelizer is core distributed code, the BAGEL
        # model lives under components/models/bagel/. Lists both the Qwen2
        # decoder ModuleList and the SigLIP encoder ModuleList so each
        # member becomes its own FSDP unit (matching upstream BAGEL's
        # transformer_auto_wrap_policy class set; without the SigLIP
        # entry, Stage 2 OOMs on 8x80GB because the SigLIP layers sit in
        # the root FSDP unit's all-gather peak).
        "BagelForUnifiedMultimodal": {
            "language": ["model.language_model.model.layers"],
            "vision": ["model.vit_model.vision_model.encoder.layers"],
        },
        "NemotronHForCausalLM": {"language": ["backbone.layers", "model.layers"]},
        GPT2LMHeadModel: {"language": ["transformer.h"]},
    }


def _extract_model_layer_groups(model: nn.Module) -> Dict[str, List[nn.Module]]:
    """Extract transformer layers grouped by model role."""
    model_cls = type(model)
    model_cls_to_layer_groups = _get_model_layer_group_specs()
    layer_group_specs = None
    if model_cls in model_cls_to_layer_groups:
        layer_group_specs = model_cls_to_layer_groups[model_cls]
    elif model_cls.__name__ in model_cls_to_layer_groups:
        layer_group_specs = model_cls_to_layer_groups[model_cls.__name__]

    layer_groups: Dict[str, List[nn.Module]] = {}
    if layer_group_specs is not None:
        for group_name, fqns in layer_group_specs.items():
            layers: List[nn.Module] = []
            # Candidate FQNs are alternative locations of the same container
            # across transformers versions; take the first that resolves so
            # deprecated alias properties (e.g. top-level `visual` on
            # standardized 4.x Qwen2-VL aliasing `model.visual`) cannot
            # double-count layers.
            for fqn in fqns:
                _extend_layers(layers, _reduce_attrs(model, [fqn]))
                if layers:
                    break
            if layers:
                layer_groups[group_name] = layers
        if not layer_groups:
            logger.warning(
                "Layer-group spec for %s resolved no modules: none of the expected FQNs %s exist in the "
                "model tree (likely transformers version drift). Activation checkpointing and layer-based "
                "sharding will skip this model until the spec is updated.",
                model_cls.__name__,
                {group_name: list(fqns) for group_name, fqns in layer_group_specs.items()},
            )
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        # Default case for all other models (assumed to be a causal LM).
        layer_groups["language"] = (
            list(model.model.layers.values())
            if isinstance(model.model.layers, nn.ModuleDict)
            else list(model.model.layers)
        )
    elif hasattr(model, "layers"):
        layer_groups["language"] = (
            list(model.layers.values()) if isinstance(model.layers, nn.ModuleDict) else list(model.layers)
        )
    else:
        # Use heuristic to find the largest layer container in the model.
        logger.warning(f"Unknown model type: {model_cls}. Using heuristic to find transformer layers.")
        largest_module_list = _find_largest_module_list(model)
        if largest_module_list is None:
            # If no layer container is found, still raise an exception.
            print(model)
            raise ValueError(
                f"Unknown model type: {model_cls} and no ModuleList or ModuleDict found in model structure"
            )

        layer_groups["unknown"] = (
            list(largest_module_list.values())
            if isinstance(largest_module_list, nn.ModuleDict)
            else list(largest_module_list)
        )
        logger.info(f"Successfully extracted {len(largest_module_list)} layers using heuristic")

    layers = [layer for group_layers in layer_groups.values() for layer in group_layers]
    assert all(isinstance(m, nn.Module) for m in layers), "layers should be nn.Module instances"
    return layer_groups


def get_model_layer_groups(model: nn.Module) -> Dict[str, List[nn.Module]]:
    """Return transformer layers grouped by model role (e.g. ``language``, ``vision``).

    Public accessor over the per-model layer-group mapping used for FSDP wrapping
    and activation checkpointing, so other components (e.g. the MoE parallelizer)
    can consume the grouping without importing private helpers.

    Args:
        model: Root model to extract grouped layers from.

    Returns:
        Mapping from group name to the list of transformer blocks in that group.
    """
    return _extract_model_layer_groups(model)


def _extract_model_layers(model: nn.Module) -> List[nn.Module]:
    """
    Extract layers from different model architectures for parallelization.

    This compatibility wrapper flattens grouped language/vision/audio layers.
    New activation-checkpointing code should use ``_extract_model_layer_groups``
    so scope decisions can be explicit.
    """
    layer_groups = _extract_model_layer_groups(model)
    return [layer for group_layers in layer_groups.values() for layer in group_layers]


def _dedupe_layers(layers: Sequence[nn.Module]) -> List[nn.Module]:
    deduped: List[nn.Module] = []
    seen: set[int] = set()
    for layer in layers:
        layer_id = id(layer)
        if layer_id in seen:
            continue
        seen.add(layer_id)
        deduped.append(layer)
    return deduped


def _has_trainable_parameters(module: nn.Module) -> bool:
    return any(param.requires_grad for param in module.parameters(recurse=True))


def _filter_layer_groups_for_activation_checkpointing(
    layer_groups: Dict[str, List[nn.Module]],
    activation_checkpointing_scope: ActivationCheckpointingScope | None = "all",
) -> Tuple[List[nn.Module], Tuple[str, ...]]:
    """Select trainable activation-checkpointed layers from grouped model layers."""
    scopes = normalize_activation_checkpointing_scope(activation_checkpointing_scope)
    all_layers = [layer for group_layers in layer_groups.values() for layer in group_layers]

    selected = []
    if scopes == ("all",):
        selected = all_layers
    else:
        for scope in scopes:
            if scope == "multimodal":
                selected.extend(layer_groups.get("vision", []))
                selected.extend(layer_groups.get("audio", []))
            else:
                selected.extend(layer_groups.get(scope, []))

    selected = _dedupe_layers(selected)
    skipped_frozen = [layer for layer in selected if not _has_trainable_parameters(layer)]
    if skipped_frozen:
        selected = [layer for layer in selected if _has_trainable_parameters(layer)]
    group_counts = {name: len(layers) for name, layers in layer_groups.items()}
    selected_counts = {
        name: sum(1 for layer in layers if any(layer is selected_layer for selected_layer in selected))
        for name, layers in layer_groups.items()
    }
    logger.info(
        "Activation checkpointing scope %s selected %d/%d trainable layers; groups=%s selected_groups=%s "
        "skipped_frozen=%d",
        scopes,
        len(selected),
        len(all_layers),
        group_counts,
        selected_counts,
        len(skipped_frozen),
    )
    if all_layers and not selected:
        logger.warning("Activation checkpointing scope %s selected no layers.", scopes)
    return selected, scopes


def _should_use_hf_native_gradient_checkpointing(
    model: nn.Module,
    layer_groups: Dict[str, List[nn.Module]],
    scopes: Tuple[str, ...],
    *,
    enable_compile: bool = False,
) -> bool:
    """Return whether HF-native gradient checkpointing can preserve AutoModel's AC scope."""
    if enable_compile or scopes != ("all",):
        return False
    if set(layer_groups) != {"language"}:
        return False
    language_layers = layer_groups.get("language", [])
    if not language_layers:
        return False
    if any(not _has_trainable_parameters(layer) for layer in language_layers):
        return False
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer as _HFGradLayer
    except ImportError:
        return False
    return (
        language_layers[0].__class__.__module__.startswith("transformers.")
        and isinstance(language_layers[0], _HFGradLayer)
        and getattr(model, "supports_gradient_checkpointing", False)
        and hasattr(model, "gradient_checkpointing_enable")
    )


def _kv_sharing_survives_checkpoint_replay(model: nn.Module) -> bool:
    """Return whether whole-block activation checkpointing is safe for a KV-shared model.

    ``checkpoint_wrapper`` replays a whole decoder block during backward with the
    arguments the forward saw. Unlike HF's ``GradientCheckpointingLayer.__call__``
    it cannot drop ``past_key_values`` from that replay, so every layer that
    writes to a cache writes to it a second time. A KV-shared model then needs
    both halves to hold:

    * the layers that populate the cache -- the *non*-shared ones, which are what
      call ``Cache.update()`` -- must not accumulate on the replay, or the
      recomputed K/V stops matching the forward;
    * the shared layers must still read the K/V their source layer produced.

    Neither holds for a model backed by an accumulating ``Cache``: the second
    ``Cache.update()`` grows the entry and backward dies with a
    ``CheckpointError`` about changed tensor metadata (observed on native HF
    ``Gemma3nForCausalLM`` with ``use_cache=True``). KV-shared models therefore
    stay on ``apply_submodule_checkpointing``, which leaves attention unwrapped,
    by default.

    A model that satisfies both halves opts in by setting the class attribute
    ``kv_sharing_survives_checkpoint_replay = True``. Gemma4 E2B/E4B qualify: a
    pass-through holder stands in for the cache, and the shared layers read a
    separate store that the replay does not disturb (see
    ``gemma4_moe/model.py``).

    Args:
        model: The model about to be checkpointed.

    Returns:
        Whether the model declares its KV sharing safe under whole-block replay.
    """
    return bool(getattr(model, "kv_sharing_survives_checkpoint_replay", False))


def _uses_custom_moe_modules(model: nn.Module) -> bool:
    """Return whether ``model`` contains Automodel's grouped custom-MoE layer."""
    iter_modules = getattr(model, "modules", None)
    if not callable(iter_modules):
        return False
    try:
        from nemo_automodel.components.moe.layers import MoE
    except ImportError:
        return False
    return any(isinstance(module, MoE) for module in iter_modules())


def _get_parallel_plan(
    model: nn.Module,
    sequence_parallel: bool = False,
    tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
    tp_size: int = 1,
) -> Dict[str, ParallelStyle]:
    """
    Select the tensor-parallel plan for the given model.

    Priority order:
    1) If ``tp_shard_plan`` is provided as a dict or import path, use it.
    2) If the model type exists in ``PARALLELIZE_FUNCTIONS``, use its optimised plan; on failure, fall back to HF plan.
    3) Otherwise, prefer the model's HF-native ``_tp_plan`` (via ``get_hf_tp_shard_plan``).
    4) Otherwise, fall back to the default base plan.

    When ``tp_size > 1`` and the model falls through to path 4 *and* the
    model class was loaded from a custom-code source (HF's
    ``trust_remote_code=True`` path, where the dynamic class lives under
    ``transformers_modules.*``), this raises ``ValueError`` instead of
    returning the default base plan. On recent PyTorch the default plan's
    placements do not populate ``shard_order`` and trip an internal assert in
    ``torch.distributed.tensor._redistribute`` on the first weight
    redistribute, which surfaces to the user as an opaque PyTorch internal
    error. Custom-code architectures are the only known-broken case (see
    https://github.com/NVIDIA-NeMo/Automodel/issues/2243); known HF
    architectures that happen to fall through (e.g. Mixtral) are left on the
    default plan with a warning, since they have been working in practice.

    When the model *did* define a ``_tp_plan`` but ``get_hf_tp_shard_plan``
    raised while translating it (e.g. styles nemo does not recognize), the
    translator's error message is folded into the ``ValueError`` as a
    diagnostic so the user can tell whether to add a ``_tp_plan`` from
    scratch or fix the styles in the one they already have.
    """
    model_parallel_plan = None
    model_cls = type(model)

    if isinstance(tp_shard_plan, dict):
        model_parallel_plan = tp_shard_plan
        col_w = max(55, max(map(len, tp_shard_plan.keys()), default=0))
        plan_lines = "\n".join(f"  {k:<{col_w}} {v}" for k, v in tp_shard_plan.items())
        logger.info(f"Using parallel plan (dictionary):\n{plan_lines}")
    elif tp_shard_plan is not None:
        try:
            plan_obj = import_class_from_path(tp_shard_plan)
            if isinstance(plan_obj, FunctionType):
                model_parallel_plan = plan_obj()
            else:
                model_parallel_plan = plan_obj
            assert isinstance(model_parallel_plan, dict), (
                f"Parallel plan must be a dictionary, got {type(model_parallel_plan)}"
            )
            logger.info(f"Using provided parallel plan (from path). {tp_shard_plan}")
        except Exception as e:
            raise ValueError(
                f"Custom parallel plan '{tp_shard_plan}' is not valid. "
                f"Please ensure it is one of the following:\n"
                "1. A dictionary mapping module names to parallel styles\n"
                "2. A path to a dictionary\n"
                "3. A path to a function that returns a dictionary\n"
                f"Error: {e}"
            )

    elif (func := PARALLELIZE_FUNCTIONS.get(_get_class_qualname(model_cls))) is not None:
        try:
            model_parallel_plan = func(model, sequence_parallel)
            logger.info(f"Using optimized parallel plan for {model_cls.__name__}.")
        except Exception as e:
            logger.info(f"Optimized parallel plan not available: {e}. Falling back to the HF tp plan.")
            model_parallel_plan = get_hf_tp_shard_plan(model)

    # Fallback: match by bare class __name__ for trust_remote_code models whose
    # qualified module path contains a snapshot hash and so cannot be stably
    # registered via _get_class_qualname().
    elif (func := PARALLELIZE_FUNCTIONS.get(model_cls.__name__)) is not None:
        try:
            model_parallel_plan = func(model, sequence_parallel)
            logger.info(f"Using optimized parallel plan for {model_cls.__name__} (matched by class name).")
        except Exception as e:
            logger.info(f"Optimized parallel plan not available: {e}. Falling back to the HF tp plan.")
            model_parallel_plan = get_hf_tp_shard_plan(model)

    else:
        # Try HF's per-model _tp_plan first — it correctly handles multimodal
        # architectures whose text layers live under model.language_model.layers.*
        # and would be missed by the
        # hardcoded llama-style wildcards below.
        hf_plan = None
        hf_plan_error: Exception | None = None
        try:
            hf_plan = get_hf_tp_shard_plan(model)
        except Exception as e:
            hf_plan_error = e
            logger.info(f"HF tp plan not available ({e}). Falling back to default base plan.")

        if hf_plan:
            model_parallel_plan = hf_plan
            logger.info(f"Using HF-native tp plan for {model_cls.__name__}.")
        else:
            # HF places dynamic classes loaded via ``trust_remote_code=True`` under the
            # ``transformers_modules.*`` namespace. Those are the only archs known to
            # actually crash inside ``_redistribute`` with the default base plan, so we
            # only fail-fast for them. See https://github.com/NVIDIA-NeMo/Automodel/issues/2243.
            is_remote_code = (model_cls.__module__ or "").startswith("transformers_modules.")
            if tp_size > 1 and is_remote_code:
                # If the model author *did* define `_tp_plan` but it was unusable
                # (e.g. styles nemo does not recognize), surface that diagnostic so the
                # user knows whether to (a) add a missing `_tp_plan` from scratch or
                # (b) fix the styles in the one they already have.
                diag = f" Note: {hf_plan_error}." if hf_plan_error is not None else ""
                raise ValueError(
                    f"No tensor-parallel plan is registered for the custom-code architecture "
                    f"'{model_cls.__name__}' (loaded via trust_remote_code=True), and no usable "
                    f"HuggingFace `_tp_plan` was found.{diag} The default base plan cannot be used "
                    f"at tp_size={tp_size}: it produces DTensor placements without `shard_order` "
                    "metadata, which trips an internal assert in "
                    "`torch.distributed.tensor._redistribute` on the first weight redistribute. "
                    "Register a working plan in one of the following ways:\n"
                    f"  1. Add an entry for '{model_cls.__name__}' to "
                    "`nemo_automodel.components.distributed.optimized_tp_plans.PARALLELIZE_FUNCTIONS`.\n"
                    "  2. Define a `_tp_plan` on the model class with styles nemo recognizes "
                    "(e.g. `colwise`, `rowwise`, `colwise_rep`, `rowwise_rep`).\n"
                    "  3. Pass `tp_shard_plan` (dict or import path) when constructing the parallelizer.\n"
                    "Alternatively, run with tp_size=1."
                )
            if tp_size > 1:
                logger.warning(
                    "No usable tensor-parallel plan is registered for '%s'. Falling back to the "
                    "default base plan at tp_size=%d. If you hit an internal assert in "
                    "`torch.distributed.tensor._redistribute` on `shard_order is not None`, "
                    "register a plan via `PARALLELIZE_FUNCTIONS`, `_tp_plan`, or `tp_shard_plan`.",
                    model_cls.__name__,
                    tp_size,
                )
            base_model_tp_plan = {
                "model.embed_tokens": VocabParallelEmbedding(input_layouts=Replicate()),
                "model.layers.*.self_attn.q_proj": ColwiseParallel(),
                "model.layers.*.self_attn.k_proj": ColwiseParallel(),
                "model.layers.*.self_attn.v_proj": ColwiseParallel(),
                "model.layers.*.self_attn.qkv_proj": ColwiseParallel(),  # Combined QKV projection
                "model.layers.*.self_attn.o_proj": RowwiseParallel(),
                "model.layers.*.mlp.gate_up_proj": ColwiseParallel(),  # Fused gate and up projection
                "model.layers.*.mlp.up_proj": ColwiseParallel(),
                "model.layers.*.mlp.gate_proj": ColwiseParallel(),
                "model.layers.*.mlp.down_proj": RowwiseParallel(),
                "lm_head": ColwiseParallel(output_layouts=Replicate()),
            }
            if sequence_parallel:
                base_model_sp_plan = {
                    "model.embed_tokens": VocabParallelEmbedding(
                        input_layouts=Replicate(),
                        output_layouts=Shard(1),
                        use_local_output=False,
                    ),
                    "model.norm": SequenceParallel(),
                    "model.layers.*.input_layernorm": SequenceParallel(),
                    "model.layers.*.self_attn.o_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
                    "model.layers.*.post_attention_layernorm": SequenceParallel(),
                    "model.layers.*.mlp.down_proj": RowwiseParallel(output_layouts=Shard(1), use_local_output=False),
                    "lm_head": ColwiseParallel(input_layouts=Shard(1), output_layouts=Replicate()),
                }
                base_model_tp_plan.update(base_model_sp_plan)
            model_parallel_plan = base_model_tp_plan
            logger.info("Using default base TP plan. Compatible with huggingface llama3-style models.")

    # EP=1 uses this generic FSDP2 path rather than the dedicated MoE
    # parallelizer. Apply the same routed-expert ownership validation here so
    # an explicit/wildcard TP plan cannot silently shard expert or router
    # modules merely because expert parallelism is disabled.
    if tp_size > 1 and _uses_custom_moe_modules(model):
        from nemo_automodel.components.moe.tp_plan_validation import _validate_moe_tp_plan

        model_parallel_plan = _validate_moe_tp_plan(model_parallel_plan, model=model)

    return model_parallel_plan


# Taken and modified from torchtitan
# https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
def fsdp2_strategy_parallelize(
    model,
    device_mesh: DeviceMesh,
    mp_policy: MixedPrecisionPolicy | None = None,
    offload_policy: OffloadPolicy | None = None,
    sequence_parallel: bool = False,
    activation_checkpointing: bool = False,
    tp_shard_plan: Union[Dict[str, ParallelStyle], str] | None = None,
    dp_replicate_mesh_name: str = "dp_replicate",
    dp_shard_cp_mesh_name: str = "dp_shard_cp",
    tp_mesh_name: str = "tp",
    enable_async_tensor_parallel: bool = False,
    enable_compile: bool = False,
    enable_fsdp2_prefetch: bool = True,
    fsdp2_backward_prefetch_depth: int = 2,
    fsdp2_forward_prefetch_depth: int = 1,
    reshard_after_forward: bool | None = None,
    activation_checkpointing_scope: ActivationCheckpointingScope | None = "all",
    frozen_multimodal_sharding: FrozenMultimodalSharding = "root",
    reapply_trainability: Callable[[nn.Module], None] | None = None,
) -> nn.Module:
    """
    Apply parallelisms and activation checkpointing to the model.

    Enhanced version that uses a strategy pattern for different model parallelization approaches:
    - Automatic strategy selection based on model type
    - Polymorphic parallelization strategies for different model families
    - Custom parallel plan support (dict or string path)
    - Sequence parallel support
    - Activation checkpointing for linear layers
    - Model validation (attention heads divisible by TP size)
    - Better fallback logic

    Args:
        model: The model to be parallelized.
        device_mesh (DeviceMesh): The device mesh for distributed training.
        mp_policy (Optional[MixedPrecisionPolicy]): Mixed precision policy for model parallelism.
        offload_policy (Optional[OffloadPolicy]): The offload policy for FSDP.
        sequence_parallel (bool): Whether to use sequence parallelism. Defaults to False.
        activation_checkpointing (bool): Whether to use activation checkpointing. Defaults to False.
        tp_shard_plan (Optional[Union[Dict[str, ParallelStyle], str]]):
            Custom tensor parallel plan for the model. Can be:
            - A dictionary mapping module names to parallel styles
            - A string path to a dictionary or function that returns a dictionary
            If provided, this takes precedence over automatic plan generation.
        dp_replicate_mesh_name (str): Key name for the data parallel replicate mesh in device_mesh.
            Used when data parallel replicate is enabled. Defaults to "dp_replicate".
        dp_shard_cp_mesh_name (str): Key name for the data parallel shard + context parallel mesh in device_mesh.
            Used when data parallel shard is enabled. Defaults to "dp_shard_cp".
        tp_mesh_name (str): Key name for the tensor parallel mesh in device_mesh.
            Defaults to "tp".
        frozen_multimodal_sharding: Whether fully frozen multimodal modules are
            owned by the root FSDP unit (``"root"``), sharded normally
            (``"per_layer"``), or excluded from FSDP and copied on every rank
            (``"replicate"``).
        reapply_trainability: Optional callback that re-resolves parameter
            trainability after strategy-specific model surgery and immediately
            before FSDP construction.

    Returns:
        The parallelized model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # Get the appropriate parallelization strategy for this model
    strategy = get_parallelization_strategy(model)

    # Delegate to the strategy
    return strategy.parallelize(
        model=model,
        device_mesh=device_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        sequence_parallel=sequence_parallel,
        activation_checkpointing=activation_checkpointing,
        tp_shard_plan=tp_shard_plan,
        dp_replicate_mesh_name=dp_replicate_mesh_name,
        dp_shard_cp_mesh_name=dp_shard_cp_mesh_name,
        tp_mesh_name=tp_mesh_name,
        enable_async_tensor_parallel=enable_async_tensor_parallel,
        enable_compile=enable_compile,
        enable_fsdp2_prefetch=enable_fsdp2_prefetch,
        fsdp2_backward_prefetch_depth=fsdp2_backward_prefetch_depth,
        fsdp2_forward_prefetch_depth=fsdp2_forward_prefetch_depth,
        reshard_after_forward=reshard_after_forward,
        activation_checkpointing_scope=activation_checkpointing_scope,
        frozen_multimodal_sharding=frozen_multimodal_sharding,
        reapply_trainability=reapply_trainability,
    )
