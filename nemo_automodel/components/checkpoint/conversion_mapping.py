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

"""
Checkpoint conversion mappings for loading HuggingFace checkpoints.

This module provides conversion mappings for transforming checkpoint keys and tensors
when loading models. It primarily uses the transformers library's conversion_mapping
module which handles both key renaming and tensor operations (merging/splitting).

For MoE models, the conversion handles:
- Key renaming from checkpoint format (e.g., block_sparse_moe.experts.X.w1) to
  model format (e.g., mlp.experts.gate_up_proj)
- Tensor merging for grouped expert formats (individual experts -> single 3D tensor)

The primary entry points are:
- `get_checkpoint_conversion_mapping(model_type)`: Get conversion rules for a model type
- `get_model_conversion_mapping(model, ...)`: Get all conversion rules for a model instance
- `requires_tensor_merging(model_type)`: Check if model needs tensor operations
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


# Try to import from transformers - this is the preferred source
_TRANSFORMERS_AVAILABLE = False
try:
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping as _transformers_get_checkpoint_conversion_mapping,
    )
    from transformers.conversion_mapping import (
        get_model_conversion_mapping as _transformers_get_model_conversion_mapping,
    )
    from transformers.core_model_loading import WeightConverter, WeightRenaming

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    # Transformers not available or doesn't have conversion_mapping
    WeightConverter = None
    WeightRenaming = None


# Model types that require tensor merging (individual experts -> grouped experts)
# For these models, simple key renaming is not sufficient - they need WeightConverter
# operations to merge individual expert weights into grouped format
MODELS_REQUIRING_TENSOR_MERGING = {
    "mixtral",
    "minimax",
    "phimoe",
    "qwen2_moe",
    "qwen3_moe",
    "deepseek_v2",
    "deepseek_v3",
    "jamba",
    "olmoe",
    "lfm2_moe",
    "dots1",
    "ernie4_5_moe",
    "glm4_moe",
    "glm4v_moe",
    "longcat_flash",
    "qwen3_omni_moe",
    "qwen3_next",
    "qwen3_vl_moe",
    "hunyuan_v1_moe",
    "flex_olmo",
}


def requires_tensor_merging(model_type: str) -> bool:
    """
    Check if a model type requires tensor merging during checkpoint loading.

    Some MoE models store expert weights in grouped format (single 3D tensor for all experts)
    but checkpoints store individual expert weights. These models require tensor merging
    that cannot be done via simple key renaming.

    Args:
        model_type: The model type string from config.model_type

    Returns:
        True if the model type requires tensor merging during loading.
    """
    return model_type in MODELS_REQUIRING_TENSOR_MERGING


def get_checkpoint_conversion_mapping(model_type: str) -> list | None:
    """
    Get the checkpoint conversion mapping for a given model type.

    This returns a list of WeightConverter and/or WeightRenaming objects from
    transformers that define how to convert checkpoint keys and tensors to
    model state dict format.

    Args:
        model_type: The model type string (e.g., "mixtral", "qwen2_moe", "phimoe")

    Returns:
        A list of WeightConverter/WeightRenaming objects defining the conversion,
        or None if no conversion mapping is defined for this model type.

    Example:
        >>> mapping = get_checkpoint_conversion_mapping("mixtral")
        >>> # Returns list with WeightRenaming for gate and WeightConverter
        >>> # for merging individual expert weights into grouped format
    """
    if not _TRANSFORMERS_AVAILABLE:
        return None
    return _transformers_get_checkpoint_conversion_mapping(model_type)


def get_model_conversion_mapping(
    model: "nn.Module",
    key_mapping: dict[str, str] | None = None,
    hf_quantizer: object | None = None,
    add_legacy: bool = True,
) -> list:
    """
    Get all weight conversion mappings for a model instance.

    This is the main entry point for getting conversion rules. It combines:
    1. Custom key_mapping if provided
    2. Model's _checkpoint_conversion_mapping attribute (for VLMs)
    3. Model-type specific conversions (MoE merging, etc.)
    4. Legacy conversions (LayerNorm.gamma -> LayerNorm.weight, etc.)
    5. Quantizer-specific conversions if provided

    Args:
        model: The model instance to get conversions for
        key_mapping: Optional custom key mapping (source -> target patterns)
        hf_quantizer: Optional HuggingFace quantizer with additional conversions
        add_legacy: Whether to include legacy LayerNorm conversions (default True)

    Returns:
        List of WeightConverter/WeightRenaming objects defining all conversions.
        Returns empty list if transformers is not available.

    Example:
        >>> from transformers import AutoModelForCausalLM
        >>> model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B")
        >>> conversions = get_model_conversion_mapping(model)
        >>> # Use conversions to transform checkpoint state dict
    """
    if not _TRANSFORMERS_AVAILABLE:
        return []
    return _transformers_get_model_conversion_mapping(
        model,
        key_mapping=key_mapping,
        hf_quantizer=hf_quantizer,
        add_legacy=add_legacy,
    )


_VLM_KEY_MAPPINGS: dict[str, dict[str, str]] = {
    "gemma3": {
        r"^language_model\.model\.": "model.language_model.",
        # transformers 5.8 flattened SiglipVisionModel: the v5.5 `vision_model.`
        # wrapper that Gemma3's vision_tower used to have is gone, so HF gemma3
        # checkpoints saved before this flip (keys like `vision_tower.vision_model.X`)
        # no longer match the in-memory FQNs (`model.vision_tower.X`). Strip the
        # legacy `vision_model.` segment before applying the generic vision_tower
        # rule. Order matters: dict iteration is insertion order and
        # `_get_key_renaming_mapping` returns on the first regex match.
        r"^vision_tower\.vision_model\.": "model.vision_tower.",
        r"^vision_tower\.": "model.vision_tower.",
        r"^multi_modal_projector\.": "model.multi_modal_projector.",
        r"^language_model\.lm_head\.": "lm_head.",
    },
}


def get_combined_key_mapping(
    model_type: str,
    model_key_mapping: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """
    Get combined key mapping for simple regex-based key renaming.

    This is a simpler alternative to get_model_conversion_mapping that only
    handles key renaming (not tensor operations). Useful when you just need
    to rename keys without merging tensors.

    Note: For MoE models that require tensor merging, use get_model_conversion_mapping
    instead, which returns WeightConverter objects that handle both renaming and merging.

    Args:
        model_type: The model type string from config.model_type
        model_key_mapping: Optional key mapping from the model's
                          `_checkpoint_conversion_mapping` attribute

    Returns:
        Combined key mapping dictionary (regex pattern -> replacement),
        or None if no mappings are defined.
    """
    # VLM models with known restructured hierarchies get explicit mappings
    # that override the generic transformers conversion (e.g. transformers 5.5.0
    # aliases gemma3→llava, but the llava mapping produces wrong FQNs for
    # Gemma3's model.language_model.* hierarchy).
    if model_type in _VLM_KEY_MAPPINGS:
        return dict(_VLM_KEY_MAPPINGS[model_type])

    result = {}

    # First add model-specific key mapping (takes precedence)
    if model_key_mapping:
        result.update(model_key_mapping)

    # Try to get conversion mapping from transformers and extract simple renamings
    if _TRANSFORMERS_AVAILABLE:
        conversions = get_checkpoint_conversion_mapping(model_type)
        if conversions:
            for conv in conversions:
                # Only extract simple WeightRenaming, not WeightConverter
                if WeightRenaming is not None and isinstance(conv, WeightRenaming):
                    # WeightRenaming stores patterns as source_patterns and target_patterns (as lists)
                    sources = getattr(conv, "source_patterns", None)
                    targets = getattr(conv, "target_patterns", None)
                    if sources and targets:
                        # Handle both list and string formats
                        if isinstance(sources, str):
                            sources = [sources]
                        if isinstance(targets, str):
                            targets = [targets]
                        # Add each source->target pair
                        for source, target in zip(sources, targets):
                            if source not in result:
                                result[source] = target

    return result if result else None


