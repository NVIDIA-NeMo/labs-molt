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

"""Infrastructure instantiation and application.

Distributed manager instantiation, sharding, PEFT/quantization application,
and checkpoint loading utilities.  These free functions operate on an
already-instantiated ``nn.Module`` and have no coupling to the
``_BaseNeMoAutoModelClass`` hierarchy.

``MeshContext`` (from ``mesh``) is the single source of truth
for device meshes, parallelism sizes, and axis names.
"""

import logging
from contextlib import nullcontext
from dataclasses import is_dataclass, replace
from functools import partial

import torch

from nemo_automodel._transformers.utils import _should_load_before_shard
from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
    _maybe_adapt_state_dict_to_hf,
)
from nemo_automodel.components.distributed.config import (
    DistributedStrategyConfig,
    FSDP2Config,
    MoEParallelizerConfig,
)
from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager
from nemo_automodel.components.distributed.init_utils import get_world_size_safe
from nemo_automodel.components.distributed.mesh import MeshContext
from nemo_automodel.components.distributed.tp_replicas import broadcast_tp_replicas
from nemo_automodel.components.models.common.utils import cast_frozen_modules_to_compute_dtype
from nemo_automodel.components.quantization.fp8 import apply_fp8_to_model
from nemo_automodel.components.utils.model_utils import (
    FreezeConfig,
    apply_parameter_freezing,
    enable_radio_vit_fused_attn,
    freeze_deepseek_v4_indexer_params,
    init_empty_weights,
    parse_freeze_config,
    print_trainable_parameters,
)
from nemo_automodel.shared.tied_weights import ensure_tied_lm_head

logger = logging.getLogger(__name__)


def _ensure_tied_lm_heads(model) -> None:
    """Re-apply local tied LM-head aliases on model parts that own both tensors."""
    model_parts = model.parts if hasattr(model, "parts") else [model]
    for model_part in model_parts:
        ensure_tied_lm_head(model_part)


def _safe_moe_tp_parts(model) -> list[torch.nn.Module]:
    """Return model parts using the conservative custom-MoE TP plan."""
    model_parts = model.parts if hasattr(model, "parts") else [model]
    return [
        model_part
        for model_part in model_parts
        if getattr(model_part, "_nemo_moe_tp_requires_pretrained_weights", False)
    ]


def _validate_safe_moe_tp_weight_source(
    model,
    *,
    checkpoint_source_available: bool,
    peft_config,
) -> None:
    """Fail closed when replicated MoE-TP paths cannot start from identical weights.

    The supported custom-MoE TP contract is pretrained, full-parameter training
    without PEFT. Replica synchronization keeps the dense token path consistent,
    but does not define adapter or initialization ownership across combined TP/EP,
    so those unsupported sources must fail before model surgery.
    """
    parts = _safe_moe_tp_parts(model)
    if not parts:
        return
    if peft_config is not None:
        raise ValueError(
            "Safe custom-MoE tensor parallelism does not support PEFT: "
            "adapter ownership is undefined across the combined TP/EP mesh."
        )
    if not checkpoint_source_available:
        raise ValueError(
            "Safe custom-MoE tensor parallelism requires pretrained weights on every TP rank. "
            "from_config/random initialization and load_base_model=False are unsupported; "
            "use from_pretrained (or an explicitly preloaded shared checkpoint)."
        )


def _verify_safe_moe_tp_weights_loaded(model, *, checkpoint_loaded: bool) -> None:
    """Fail closed when the safe custom-MoE TP path skipped the checkpoint load."""
    if not _safe_moe_tp_parts(model):
        return
    if not checkpoint_loaded:
        raise RuntimeError(
            "Safe custom-MoE tensor parallelism reached post-load setup without a completed checkpoint load."
        )


#  PEFT / quantization helpers
def _apply_peft_and_lower_precision(model, peft_config, quantization_config, fp8_config):
    if peft_config is not None:
        # Skip freeze here - will do global freeze after checkpoint loading
        apply_lora_to_linear_modules(model, peft_config, quantization_config=quantization_config, skip_freeze=True)

    # FP8
    if fp8_config is not None:
        model = apply_fp8_to_model(model, config=fp8_config)

    return model


def _shard_ep_fsdp(model, model_wrapper, parallelize_fn, mesh: MeshContext, reapply_trainability):
    """Apply EP + FSDP sharding (non-PP path)."""
    if parallelize_fn is not None and get_world_size_safe() > 1:
        parallelize_fn(
            model,
            world_mesh=mesh.device_mesh,
            moe_mesh=mesh.moe_mesh,
            reapply_trainability=reapply_trainability,
            **mesh.parallelize_axis_kwargs(),
        )
    elif callable(getattr(model_wrapper, "parallelize", None)):
        model = model_wrapper.parallelize(model, reapply_trainability=reapply_trainability)
    return model


#  Infrastructure instantiation (config -> runtime objects)
def _instantiate_distributed(
    config: DistributedStrategyConfig | None,
    mesh: MeshContext,
) -> FSDP2Manager | None:
    """Instantiate the FSDP2 manager from config.

    Args:
        config: Distributed config (FSDP2Config).
        mesh: MeshContext holding device_mesh and moe_mesh references.

    Returns:
        The instantiated manager, or None if config is None.

    Raises:
        ValueError: If device_mesh is required but not provided.
    """
    if config is None:
        return None

    if isinstance(config, FSDP2Config):
        if mesh.device_mesh is None:
            raise ValueError("device_mesh is required for FSDP2Config")
        return FSDP2Manager(config, device_mesh=mesh.device_mesh, moe_mesh=mesh.moe_mesh)
    else:
        raise ValueError(f"Unknown distributed config type: {type(config)}")


def _with_activation_checkpointing(
    config: DistributedStrategyConfig | None,
    activation_checkpointing: bool,
) -> DistributedStrategyConfig | None:
    """Return a strategy config whose AC flag matches the resolved setup."""
    if config is None or not hasattr(config, "activation_checkpointing"):
        return config
    if getattr(config, "activation_checkpointing") is activation_checkpointing:
        return config
    if not is_dataclass(config):
        return config
    return replace(config, activation_checkpointing=activation_checkpointing)


def instantiate_infrastructure(
    *,
    distributed_config: DistributedStrategyConfig | None = None,
    moe_parallel_config: MoEParallelizerConfig | None = None,
    activation_checkpointing: bool | str | None = None,
    mesh: MeshContext | None = None,
) -> tuple:
    """Instantiate infrastructure objects from config classes.

    This function converts config objects into the runtime objects needed by
    apply_model_infrastructure. It provides a cleaner, more HuggingFace-like API
    where users pass config objects instead of constructing runtime objects directly.

    Args:
        distributed_config: Distributed training config (FSDP2Config).
        moe_parallel_config: MoE parallelizer config (for expert parallel models).
        activation_checkpointing: Enable activation checkpointing for transformer blocks.
            If ``None``, inferred from ``distributed_config.activation_checkpointing``.
        mesh: MeshContext holding device meshes, sizes, and axis names.

    Returns:
        tuple: (model_wrapper, parallelize_fn)
            - model_wrapper: FSDP2 manager instance (or None)
            - parallelize_fn: Parallelization function (or None) - the MoE-specific
                parallelizer when ep_size > 1
    """
    if mesh is None:
        mesh = MeshContext()

    if activation_checkpointing is None:
        activation_checkpointing = bool(getattr(distributed_config, "activation_checkpointing", False))
    distributed_config = _with_activation_checkpointing(distributed_config, activation_checkpointing)

    model_wrapper = _instantiate_distributed(distributed_config, mesh)

    parallelize_fn = None
    if mesh.ep_size > 1:
        from nemo_automodel.components.moe.parallelizer import parallelize_model

        if moe_parallel_config is None:
            moe_parallel_config = MoEParallelizerConfig()
        # Forward the model wrapper's mp_policy (from FSDP2Config) to expert
        # sharding when the MoE config doesn't set its own, so a custom precision
        # policy isn't silently dropped for EP models.
        moe_kwargs = moe_parallel_config.to_dict()
        if moe_kwargs.get("mp_policy") is None and model_wrapper is not None:
            moe_kwargs["mp_policy"] = getattr(model_wrapper, "mp_policy", None)
        if isinstance(model_wrapper, FSDP2Manager):
            # The dedicated MoE parallelizer replaces FSDP2Manager.parallelize
            # whenever EP is enabled, so forward every FSDP2 setting it owns
            # rather than silently dropping TP/SP/offload configuration.
            moe_kwargs.setdefault("tp_shard_plan", model_wrapper.tp_plan)
            moe_kwargs.setdefault("sequence_parallel", bool(model_wrapper.sequence_parallel))
            moe_kwargs.setdefault("offload_policy", model_wrapper.offload_policy)
            if model_wrapper.reshard_after_forward is not None:
                # FSDP2Config is the canonical distributed policy. Preserve
                # the MoE-specific default only when the manager leaves this
                # setting unspecified.
                moe_kwargs["reshard_after_forward"] = model_wrapper.reshard_after_forward
            moe_kwargs.setdefault(
                "enable_async_tensor_parallel",
                bool(model_wrapper.enable_async_tensor_parallel),
            )
            moe_kwargs.setdefault(
                "frozen_multimodal_sharding",
                model_wrapper.frozen_multimodal_sharding,
            )
        parallelize_fn = partial(
            parallelize_model,
            activation_checkpointing=activation_checkpointing,
            # The AC scope lives on the strategy config (normalized in its
            # __post_init__); thread it through so expert-parallel configs keep
            # scope parity with the generic FSDP2 path.
            activation_checkpointing_scope=getattr(distributed_config, "activation_checkpointing_scope", "all"),
            **moe_kwargs,
        )

    return model_wrapper, parallelize_fn


def _uses_te_attention(model) -> bool:
    """Return True if any self_attn module uses TE's DotProductAttention."""
    try:
        from transformer_engine.pytorch.attention import DotProductAttention
    except ImportError:
        return False

    model_parts = model.parts if hasattr(model, "parts") else [model]
    for part in model_parts:
        for name, module in part.named_modules():
            if name.endswith("self_attn"):
                attn_module = getattr(module, "attn_module", None)
                if isinstance(attn_module, DotProductAttention):
                    return True
    return False


def _uses_thd_only_te_attention(model) -> bool:
    """Return whether TE is restricted to packed THD attention on this model."""
    model_parts = model.parts if hasattr(model, "parts") else [model]
    return any(
        getattr(module, "_te_thd_only", False)
        for part in model_parts
        for name, module in part.named_modules()
        if name.endswith("self_attn")
    )


def _apply_trainability_policy(
    model: torch.nn.Module,
    *,
    peft_enabled: bool,
    freeze_config: FreezeConfig | None,
    strict: bool,
) -> None:
    """Resolve the complete trainability policy on the current module hierarchy.

    Parallelization and checkpoint loading can replace modules and parameters.
    Re-running this policy after each such surgery selects the current objects by
    module path instead of transferring stale parameter-name state.

    Args:
        model: Model whose trainability is being resolved.
        peft_enabled: Whether the PEFT baseline should freeze non-LoRA parameters.
        freeze_config: Optional user freeze/unfreeze policy.
        strict: Whether every generic selector must match this model. Full-model
            validation is strict; the post-sharding rebinding is non-strict.
    """
    if peft_enabled:
        for name, param in model.named_parameters(remove_duplicate=False):
            param.requires_grad_("lora_" in name)
    if freeze_config is not None:
        apply_parameter_freezing(model, freeze_config, strict=strict)

    # Framework invariant, always applied last so a user unfreeze selector cannot override it.
    freeze_deepseek_v4_indexer_params(model)


#  apply_model_infrastructure  --  the main post-init orchestration function
def apply_model_infrastructure(
    model,
    *,
    is_meta_device,
    device,
    model_wrapper=None,
    mesh=None,
    peft_config=None,
    quantization_config=None,
    fp8_config=None,
    parallelize_fn=None,
    load_base_model=False,
    cache_dir=None,
    pretrained_model_name_or_path="",
    weights_already_loaded=False,
    inject_te_attention: bool = False,
    **_kwargs,
):
    """Apply sharding, PEFT, quantization, and checkpoint loading to a model.

    This function contains the common post-init logic shared between from_pretrained
    and from_config methods. It can also be called directly for models built via
    custom builder functions (e.g., build_gpt2_model). It handles:
    - PEFT and lower precision application (LoRA, FP8)
    - EP/FSDP sharding
    - Device placement and compilation
    - Checkpoint loading for meta device models

    Args:
        model: The model to apply infrastructure to
        is_meta_device: Whether model was initialized on meta device
        device: Target device for model
        model_wrapper: FSDP2Manager. Default: None
        mesh: MeshContext with parallelism sizes (tp_size, cp_size, etc.) and mesh
            references. Default: None (treated as single-GPU defaults).
        peft_config: PEFT/LoRA configuration dict. Default: None
        quantization_config: Quantization configuration. Default: None
        fp8_config: FP8 configuration. Default: None
        parallelize_fn: Function to apply parallelization (EP + FSDP2). Default: None
        pretrained_model_name_or_path: Model name or path for checkpoint loading. Default: ""
        load_base_model: Whether to load base model weights (True for from_pretrained). Default: False
        cache_dir: Cache directory for model weights. Default: None
        weights_already_loaded: Whether pretrained weights were already loaded during
            model init (e.g., by HF's from_pretrained on a real device, which also
            handles BnB quantization atomically). When True, checkpoint loading in
            this function is skipped. Default: False.
        inject_te_attention: When True, inject TransformerEngine DotProductAttention
            into all ``self_attn`` modules of HF models (has no effect on custom
            models that already use TE via BackendConfig). Default: False.
        **_kwargs: Additional keyword arguments (ignored, allows passing extra kwargs)

    Returns:
        The model with all infrastructure applied
    """
    if mesh is None:
        mesh = MeshContext()

    # Create a checkpointer for loading base weights only. Keep consolidation disabled
    # so load-only infrastructure does not emit save/export warnings.
    ckpt_config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir="",
        model_save_format="safetensors",
        model_cache_dir=cache_dir,
        model_repo_id=pretrained_model_name_or_path,
        save_consolidated=False,
        is_peft=peft_config is not None,
    )
    checkpointer = Checkpointer(
        ckpt_config,
        0,
        0,
        0,
        getattr(model_wrapper, "moe_mesh", None),
        process_group=getattr(mesh, "process_group", None),
    )

    # Apply PEFT and lower precision if configured
    # When on meta device, wrap in init_empty_weights() so new LoRA modules are also on meta device
    # This allows copy operations between meta tensors to succeed (they're no-ops)
    peft_ctx = init_empty_weights() if is_meta_device else nullcontext()
    with peft_ctx:
        model = _apply_peft_and_lower_precision(model, peft_config, quantization_config, fp8_config)

    # Inject TE attention into HF models when requested.
    # Done after PEFT (so projection shapes are final) and before sharding
    # (so TE modules are included in the FSDP unit).
    if inject_te_attention and not _uses_te_attention(model):
        from nemo_automodel._transformers.te_attention import inject_te_attention as _inject_te

        _inject_te(model)

    # When no TP, load checkpoint first (unwrapped model) so weights and dtypes come from
    # the checkpoint; then apply FSDP. With TP>1 we must shard first and load after so all ranks
    # stay in sync (load-before-shard can cause NCCL collective mismatch).
    # Skip load-before-shard for PEFT: base load into unwrapped PEFT then later adapter load
    # after shard can leave base/adapter out of sync (e.g. key/device mismatch). Use the
    # post-shard load path so base and adapter load in the same way as multi-GPU.
    need_checkpoint_load = bool(pretrained_model_name_or_path and load_base_model)
    load_before_shard = _should_load_before_shard(
        tp_size=mesh.tp_size,
        ep_size=mesh.ep_size,
        # FSDP shards across the combined DP-shard and CP mesh.
        dp_shard_size=mesh.dp_shard_size * mesh.cp_size,
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        load_base_model=load_base_model,
        peft_config=peft_config,
    )

    checkpoint_already_loaded = False
    if load_before_shard:
        if weights_already_loaded:
            # HF's from_pretrained already populated the weights during model init.
            # Still call load_base_model with load_base_model=False to
            # handle weight tying
            checkpointer.load_base_model(model, device, cache_dir, pretrained_model_name_or_path, load_base_model=False)
        else:
            # Only meta-device models need their parameter shells materialized first.
            if is_meta_device:
                lora_a_init = getattr(peft_config, "lora_A_init", None)
                checkpointer.initialize_model_weights(model, device, peft_init_method=lora_a_init)
            checkpointer.load_base_model(
                model,
                device,
                cache_dir,
                pretrained_model_name_or_path,
                load_base_model=load_base_model,
            )
        checkpoint_already_loaded = True

    # hold a list copy of the model state dict keys before any parallelization. To be used during checkpoint saving in safetensors format.
    state_dict_adapter = getattr(model, "state_dict_adapter", None)
    get_hf_state_dict_keys = getattr(state_dict_adapter, "get_hf_state_dict_keys", None)
    if get_hf_state_dict_keys is not None:
        pre_shard_hf_state_dict_keys = get_hf_state_dict_keys(model.state_dict())
    else:
        pre_shard_hf_state_dict_keys = [
            key
            for key in _maybe_adapt_state_dict_to_hf(model, model.state_dict(), quantization=False)
            if not key.endswith("_extra_state")
        ]

    # Validate selectors on the complete pre-parallelization hierarchy. The
    # same policy is rebound after model surgery and before FSDP capture.
    freeze_config = parse_freeze_config(_kwargs.get("freeze_config"))
    _apply_trainability_policy(
        model,
        peft_enabled=peft_config is not None,
        freeze_config=freeze_config,
        strict=True,
    )
    reapply_trainability = partial(
        _apply_trainability_policy,
        peft_enabled=peft_config is not None,
        freeze_config=freeze_config,
        strict=False,
    )

    # NemotronOmni RADIO: opt into the fused SDPA path on ViT attention blocks.
    enable_radio_vit_fused_attn(model)

    model = _shard_ep_fsdp(model, model_wrapper, parallelize_fn, mesh, reapply_trainability)
    _ensure_tied_lm_heads(model)
    if isinstance(model_wrapper, FSDP2Manager):
        model_parts = model.parts if hasattr(model, "parts") else [model]
        for mp in model_parts:
            model_wrapper.maybe_compile(mp)
    setattr(model, "_pre_shard_hf_state_dict_keys", pre_shard_hf_state_dict_keys)

    _validate_safe_moe_tp_weight_source(
        model,
        checkpoint_source_available=bool(need_checkpoint_load or checkpoint_already_loaded or weights_already_loaded),
        peft_config=peft_config,
    )

    # Materialize meta-device parameters and initialize weights after sharding.
    # This is needed for both from_pretrained (before checkpoint loading overwrites)
    # and from_config (where this is the only weight initialization).
    # Skipped when load_before_shard already handled materialization + init.
    need_materialize = (
        is_meta_device
        and not load_before_shard
        and any(
            [
                get_world_size_safe() == 1,
                parallelize_fn is not None and get_world_size_safe() > 1,
                callable(getattr(model_wrapper, "parallelize", None)),
            ]
        )
    )
    # When FSDP2 CPU offload is enabled, params must be materialized on CPU —
    # FSDP2 manages GPU placement itself during forward/backward.
    _has_cpu_offload = model_wrapper is not None and getattr(model_wrapper, "offload_policy", None) is not None
    if need_materialize:
        init_device = torch.device("cpu") if _has_cpu_offload else device
        model_parts = model.parts if hasattr(model, "parts") else [model]
        lora_a_init = getattr(peft_config, "lora_A_init", None)
        for mp in model_parts:
            checkpointer.initialize_model_weights(mp, init_device, peft_init_method=lora_a_init)

    # Load the checkpoint if pretrained weights are needed and weren't already loaded
    # (e.g., by HF's from_pretrained on a real device, which also handles BnB
    # quantization atomically).  Decoupled from the meta-device materialization
    # decision so that changes to the meta-device policy cannot silently skip loading.
    should_load_checkpoint = need_checkpoint_load and not checkpoint_already_loaded and not weights_already_loaded
    if should_load_checkpoint:
        model_parts = model.parts if hasattr(model, "parts") else [model]
        for mp in model_parts:
            checkpointer.load_base_model(
                mp,
                device,
                cache_dir,
                pretrained_model_name_or_path,
                load_base_model=load_base_model,
            )

    _verify_safe_moe_tp_weights_loaded(
        model,
        checkpoint_loaded=bool(checkpoint_already_loaded or weights_already_loaded or should_load_checkpoint),
    )

    # Checkpoint loading can perform another round of parameter replacement, so
    # re-resolve once more on the final model parts before optimizer construction.
    trainability_models: list[torch.nn.Module] = list(model.parts) if hasattr(model, "parts") else [model]
    for mp in trainability_models:
        reapply_trainability(mp)
    if peft_config is not None or freeze_config is not None:
        if not any(param.requires_grad for mp in trainability_models for param in mp.parameters()):
            logger.warning(
                "The configured trainability policy left no trainable parameters; "
                "check freeze_config and the PEFT configuration."
            )

    print_trainable_parameters(model)  # Once model's been sharded
    # Ensure model is on the correct device.
    # Skip only when params are actually sharded (any DTensor in the model)
    # AND the checkpoint was loaded post-shard. Calling model.to(device) on
    # sharded params triggers FSDP's reset_sharded_param, which fails on
    # tied parameters (e.g. lm_head/embed_tokens with TP>1).
    # See: https://github.com/pytorch/pytorch/issues/151085
    # In unsharded cases (single-GPU, or any combination of TP/DP/CP/EP
    # that left params as plain tensors), model.to(device) must still run so
    # that persistent buffers not present in the checkpoint (e.g. Gemma4's
    # Gemma4ClippableLinear input_min/max, Gemma4TextDecoderLayer
    # layer_scalar) reach the GPU.
    from torch.distributed.tensor import DTensor

    has_sharded_params = any(isinstance(p, DTensor) for p in model.parameters())
    # Skip model.to(device) when CPU offload is on — FSDP2 expects params on
    # CPU and moves them to GPU during forward/backward itself.
    if not _has_cpu_offload and not (should_load_checkpoint and has_sharded_params):
        try:
            model.to(device, non_blocking=True)
        except NotImplementedError as e:
            if "Cannot copy out of meta tensor" in str(e):
                logger.warning("model.to(device) failed (meta tensors); using model.to_empty(device=device) instead.")
                model.to_empty(device=device)
            else:
                raise

    # Configure dense attention parallelism. Transformer Engine must know its
    # TP head partition even without CP; for CP, TE owns THD communication while
    # SDPA uses DTensor context-parallel hooks.
    # These hooks strip attention_mask and set is_causal=True on self_attn modules
    # so that SDPA handles causal masking internally (compatible with DTensor sharding).
    #
    # MoE models (ep_size > 1) get their full CP setup from the MoE parallelizer's
    # apply_cp (via _shard_ep_fsdp): TE attention -> its own CP group; model-owned
    # attention (e.g. Gemma4's ring) -> setup_cp_attention. Re-running this dense
    # pass for them is not just redundant -- it would mask-strip their vision tower
    # and clobber the model-owned ring (the original double-apply bug). Non-TE MoE
    # is not excluded by the _uses_te_attention check, so gate on ep_size: only
    # dense (non-MoE) models need this pass.
    uses_te_attention = _uses_te_attention(model) if mesh.cp_size > 1 or mesh.tp_size > 1 else False
    uses_thd_only_te_attention = _uses_thd_only_te_attention(model) if uses_te_attention else False
    if mesh.ep_size <= 1 and (mesh.cp_size > 1 or (mesh.tp_size > 1 and uses_thd_only_te_attention)):
        from nemo_automodel.components.distributed.context_parallel.utils import (
            attach_context_parallel_hooks,
            attach_cp_sdpa_hooks,
            attach_te_context_parallel,
        )

        model_parts = model.parts if hasattr(model, "parts") else [model]
        if uses_te_attention:
            cp_mesh = mesh.device_mesh["cp"] if mesh.cp_size > 1 else None
            tp_mesh = mesh.device_mesh["tp"] if mesh.tp_size > 1 else None
            configured = sum(attach_te_context_parallel(mp, cp_mesh, tp_mesh) for mp in model_parts)
            if configured == 0:
                raise ValueError(
                    "Tensor or context parallelism selected Transformer Engine attention, but no "
                    "DotProductAttention modules were found on the model."
                )
        if mesh.cp_size > 1 and (not uses_te_attention or uses_thd_only_te_attention):
            is_compile_enabled = isinstance(model_wrapper, FSDP2Manager) and model_wrapper.enable_compile
            cp_mesh = mesh.device_mesh["cp"] if is_compile_enabled else None
            for mp in model_parts:
                attach_context_parallel_hooks(mp)
                if is_compile_enabled:
                    attach_cp_sdpa_hooks(mp, cp_mesh)

    # Frozen submodules (e.g. a frozen vision tower) either land in the root FSDP unit
    # (sharded) or are excluded from wrapping, depending on the model/parallelizer. In
    # both cases FSDP mixed precision never casts their buffers, and an excluded frozen
    # module also keeps its storage-dtype params. Under fp32 master weights + bf16 compute
    # that leaves frozen fp32 tensors feeding bf16 trainable modules -> dtype-mismatch
    # matmul at the seam. Cast frozen params/buffers to the compute dtype so the whole
    # forward runs uniformly. Freeze configuration only controls requires_grad; trainable
    # parameters keep their storage dtype (compute dtype is owned by autocast or the
    # distributed mixed-precision policy). No-op for pure-fp32 / pure-bf16 runs.
    compute_dtype = getattr(getattr(model_wrapper, "mp_policy", None), "param_dtype", None)
    if compute_dtype is not None:
        for mp in model.parts if hasattr(model, "parts") else [model]:
            cast_frozen_modules_to_compute_dtype(mp, compute_dtype)

    synchronized_tp_replicas = broadcast_tp_replicas(
        model.parts if hasattr(model, "parts") else [model],
        mesh.device_mesh,
    )
    if synchronized_tp_replicas:
        logger.info("Synchronized %d replicated tensors across TP ranks", synchronized_tp_replicas)
    return model
