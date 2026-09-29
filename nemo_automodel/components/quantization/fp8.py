# Copyright (c) NVIDIA CORPORATION and affiliates.
# All rights reserved.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Literal

import torch
import torch.nn as nn

from nemo_automodel.shared.import_utils import MISSING_TORCHAO_MSG

logger = logging.getLogger(__name__)

try:
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training

    HAVE_TORCHAO = True
except ImportError:
    HAVE_TORCHAO = False


@dataclass
class FP8Config:
    """Configuration for FP8 quantization settings."""

    enabled: bool = False
    """Whether FP8 quantization is enabled."""

    recipe_name: Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"] | None = None
    """FP8 recipe to use. If None, uses tensorwise scaling with manual configuration."""

    enable_fsdp_float8_all_gather: bool = False
    """Whether to enable float8 all-gather in FSDP, recommended for tensorwise scaling."""

    precompute_float8_dynamic_scale_for_fsdp: bool = False
    """Whether to precompute float8 scales dynamically for FSDP, recommended for tensorwise scaling."""

    force_recompute_fp8_weight_in_bwd: bool = False
    """Whether to force the recomputation of FP8 weights during backward pass."""

    filter_fqns: List[str] = field(default_factory=list)
    """
    List of fully qualified names of modules to skip applying float8 training to.
    nn.Linear modules with any dim size not divisible by 16 are always skipped due to hardware requirements.
    Example: ["attention.wq", "attention.wk", "attention.wv", "lm_head"]
    """

    emulate: bool = False
    """If True, emulation is used instead of hardware accelerated gemm. This is for test purpose only"""

    def __init__(
        self,
        enabled: bool = False,
        recipe_name: Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"] | None = None,
        enable_fsdp_float8_all_gather: bool = False,
        precompute_float8_dynamic_scale_for_fsdp: bool = False,
        force_recompute_fp8_weight_in_bwd: bool = False,
        filter_fqns: List[str] = None,
        emulate: bool = False,
    ):
        self.enabled = enabled
        self.recipe_name = recipe_name
        self.enable_fsdp_float8_all_gather = enable_fsdp_float8_all_gather
        self.precompute_float8_dynamic_scale_for_fsdp = precompute_float8_dynamic_scale_for_fsdp
        self.force_recompute_fp8_weight_in_bwd = force_recompute_fp8_weight_in_bwd
        self.filter_fqns = filter_fqns or []
        self.emulate = emulate

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "fp8_recipe_name": self.recipe_name,
            "enable_fsdp_float8_all_gather": self.enable_fsdp_float8_all_gather,
            "precompute_float8_dynamic_scale_for_fsdp": self.precompute_float8_dynamic_scale_for_fsdp,
            "force_recompute_fp8_weight_in_bwd": self.force_recompute_fp8_weight_in_bwd,
            "fp8_filter_fqns": self.filter_fqns,
            "fp8_emulate": self.emulate,
        }


def _has_cuda_capability(major: int, minor: int) -> bool:
    """Check if CUDA device has required compute capability."""
    if not torch.cuda.is_available():
        return False

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)
    return capability >= (major, minor)


def _module_filter_fn(module, name, filter_fqns: List[str] = None):
    """
    Filter function to exclude certain modules from FP8 conversion.

    Args:
        module: The module to check
        name: Fully qualified name of the module
        filter_fqns: List of FQNs to filter out

    Returns:
        True if module should be converted to FP8, False otherwise
    """
    if filter_fqns is None:
        filter_fqns = []

    # Skip modules in filter list
    for fqn in filter_fqns:
        if fqn in name:
            return False

    # Always skip non-linear layers
    if not isinstance(module, nn.Linear):
        return False

    # Skip layers with dimensions not divisible by 16
    if hasattr(module, "weight"):
        weight = module.weight
        if weight.shape[0] % 16 != 0 or weight.shape[1] % 16 != 0:
            logger.info(f"Skipping fp8 for layer {name} with weight shape {weight.shape}")
            return False

    return True


def apply_fp8_to_model(
    model: nn.Module,
    config: FP8Config | None = None,
    # Individual parameter options for backward compatibility
    filter_fqns: List[str] | None = None,
    recipe_name: str | None = None,
    force_recompute_fp8_weight_in_bwd: bool = False,
    enable_fsdp_float8_all_gather: bool = False,
    emulate: bool = False,
    enabled: bool = True,
    precompute_float8_dynamic_scale_for_fsdp: bool = False,
) -> nn.Module:
    """
    Apply FP8 quantization to a PyTorch model using torchao.

    This function can be called in two ways:
    1. With an FP8Config object: apply_fp8_to_model(model, config=fp8_config)
    2. With individual parameters: apply_fp8_to_model(model, filter_fqns=..., recipe_name=..., etc.)

    Args:
        model: The model to convert
        config: FP8Config object containing all configuration. If provided, individual
               parameters are ignored.
        filter_fqns: List of module names to exclude from FP8 conversion
        recipe_name: Recipe name for FP8 configuration ("tensorwise", "rowwise", etc.)
        force_recompute_fp8_weight_in_bwd: Whether to force recompute FP8 weight in backward pass
        enable_fsdp_float8_all_gather: Whether to enable FSDP FP8 all-gather
        emulate: Use emulation instead of hardware acceleration (for testing on older GPUs)
        enabled: Whether FP8 quantization is enabled (only used when config is None)
        precompute_float8_dynamic_scale_for_fsdp: Whether to precompute float8 scales dynamically

    Returns:
        The model with FP8 linear layers (modified in-place)

    Raises:
        ImportError: If torchao is not installed
        ValueError: If hardware doesn't support FP8 and emulation is disabled
    """
    # If config is provided, use it; otherwise create config from individual parameters
    if config is not None:
        # Use provided FP8Config
        fp8_config = config
    else:
        # Create FP8Config from individual parameters
        fp8_config = FP8Config(
            enabled=enabled,
            recipe_name=recipe_name,
            enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
            precompute_float8_dynamic_scale_for_fsdp=precompute_float8_dynamic_scale_for_fsdp,
            force_recompute_fp8_weight_in_bwd=force_recompute_fp8_weight_in_bwd,
            filter_fqns=filter_fqns or [],
            emulate=emulate,
        )

    # Check if FP8 is disabled
    if not fp8_config.enabled:
        logger.info("FP8 quantization is disabled")
        return model

    # Check if torchao is available
    if not HAVE_TORCHAO:
        raise ImportError(MISSING_TORCHAO_MSG)

    # Set precompute attribute on model
    model.precompute_float8_dynamic_scale_for_fsdp = (
        fp8_config.precompute_float8_dynamic_scale_for_fsdp
        and fp8_config.recipe_name == "tensorwise"
        and fp8_config.enable_fsdp_float8_all_gather
    )

    # Handle config creation or recipe-based configuration
    if fp8_config.recipe_name is not None and fp8_config.recipe_name != "tensorwise":
        torchao_config = Float8LinearConfig.from_recipe_name(fp8_config.recipe_name)
        logger.info(f"Using FP8 recipe: {fp8_config.recipe_name}")

        # Enable inductor precision cast emulation for rowwise recipe
        if fp8_config.recipe_name == "rowwise":
            torch._inductor.config.emulate_precision_casts = True
            logger.debug("Enabled torch._inductor.config.emulate_precision_casts for rowwise recipe")
    else:
        # Manual configuration for tensorwise scaling
        torchao_config = Float8LinearConfig(
            enable_fsdp_float8_all_gather=fp8_config.enable_fsdp_float8_all_gather,
            force_recompute_fp8_weight_in_bwd=fp8_config.force_recompute_fp8_weight_in_bwd,
            emulate=fp8_config.emulate,
        )
        logger.info("Using FP8 tensorwise scaling")

    # Check hardware capability if not using emulation
    config_emulate = getattr(torchao_config, "emulate", fp8_config.emulate)
    if not _has_cuda_capability(8, 9) and not config_emulate:
        raise ValueError(
            "FP8 is only supported on SM89 or later GPUs (H100+). "
            "To enable testing on older hardware, set emulate=True in Float8LinearConfig or pass emulate=True."
        )

    try:
        filter_fqns_list = fp8_config.filter_fqns or []
        filter_fn = partial(_module_filter_fn, filter_fqns=filter_fqns_list)

        # Convert model to use FP8 linear layers
        convert_to_float8_training(
            model,
            config=torchao_config,
            module_filter_fn=filter_fn,
        )

        logger.info(
            f"Successfully converted model to FP8 with torchAO, recipe: {fp8_config.recipe_name or 'tensorwise'}, "
            f"fp8 all-gather enabled: {torchao_config.enable_fsdp_float8_all_gather}, "
            f"force recompute FP8 weight in backward pass: {torchao_config.force_recompute_fp8_weight_in_bwd}"
        )
        verify_fp8_conversion(model)
        logger.info("FP8 quantization applied successfully")

        return model

    except Exception as e:
        logger.warning(f"FP8 quantization failed: {e}. Returning original model")
        return model


def verify_fp8_conversion(model: nn.Module) -> dict:
    """
    Verify that FP8 conversion was successful by counting converted modules.

    Args:
        model: The model to verify

    Returns:
        Dict with conversion statistics
    """
    from torchao.float8.float8_linear import Float8Linear

    total_linear = 0
    fp8_modules = []

    for name, module in model.named_modules():
        module_type = type(module).__name__

        # Count both nn.Linear and Float8Linear as linear layers
        if isinstance(module, nn.Linear):
            total_linear += 1
            logger.debug(f"Found nn.Linear: {name} ({module_type})")
            # Check if it's a Float8Linear by comparing class names or checking attributes
            if isinstance(module, Float8Linear):
                fp8_modules.append(
                    {
                        "name": name,
                        "type": module_type,
                        "weight_shape": list(module.weight.shape) if hasattr(module, "weight") else None,
                    }
                )
                logger.debug(f"Found Float8Linear: {name} ({module_type})")
            elif module_type == "Float8Linear":
                # Fallback: check by class name in case isinstance fails
                fp8_modules.append(
                    {
                        "name": name,
                        "type": module_type,
                        "weight_shape": list(module.weight.shape) if hasattr(module, "weight") else None,
                    }
                )
                logger.debug(f"Found Float8Linear by name: {name} ({module_type})")

    logger.info(f"FP8 conversion: {len(fp8_modules)} Float8Linear modules, {total_linear} total linear modules")
    return {
        "linear_count": total_linear,
        "fp8_count": len(fp8_modules),
        "conversion_rate": (len(fp8_modules) / total_linear * 100) if total_linear > 0 else 0,
        "fp8_modules": fp8_modules,
        "success": len(fp8_modules) > 0,
    }


def create_fp8_config_from_dict(config_dict: Dict[str, Any]) -> FP8Config:
    """Create a FP8Config from a dictionary.

    Args:
        config_dict: Dictionary containing FP8 configuration.

    Returns:
        FP8Config instance.
    """

    return FP8Config(
        enabled=config_dict.get("enabled", False),
        recipe_name=config_dict.get("recipe_name", None),
        enable_fsdp_float8_all_gather=config_dict.get("enable_fsdp_float8_all_gather", False),
        precompute_float8_dynamic_scale_for_fsdp=config_dict.get("precompute_float8_dynamic_scale_for_fsdp", False),
        force_recompute_fp8_weight_in_bwd=config_dict.get("force_recompute_fp8_weight_in_bwd", False),
        filter_fqns=config_dict.get("filter_fqns", []),
        emulate=config_dict.get("emulate", False),
    )


def build_fp8_config(cfg: Dict[str, Any] | None) -> FP8Config:
    """Build a FP8 config from configuration.

    Args:
        cfg: Configuration dictionary for FP8 quantization.

    Returns:
        FP8Config instance.
    """

    if cfg is None:
        return FP8Config(enabled=False)
    else:
        return create_fp8_config_from_dict(cfg)
