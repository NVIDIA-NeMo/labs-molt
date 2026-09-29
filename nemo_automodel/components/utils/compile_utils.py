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

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class CompileConfig:
    """Configuration for torch.compile."""

    enabled: bool = False
    mode: str = "default"
    fullgraph: bool = False
    dynamic: bool = False
    backend: str | None = None
    options: Dict[str, Any] | None = None
    dynamo_cache_size_limit: int = 256

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "default",
        fullgraph: bool = False,
        dynamic: bool = False,
        backend: str | None = None,
        options: Dict[str, Any] | None = None,
        dynamo_cache_size_limit: int = 256,
    ):
        self.enabled = enabled
        self.mode = mode
        self.fullgraph = fullgraph
        self.dynamic = dynamic
        self.backend = backend
        self.options = options or {}
        self.dynamo_cache_size_limit = dynamo_cache_size_limit

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic,
            "backend": self.backend,
            "options": self.options,
            "dynamo_cache_size_limit": self.dynamo_cache_size_limit,
        }


def configure_torch_dynamo(cache_size_limit: int = 256, capture_scalar_outputs: bool = True):
    """Configure torch._dynamo settings for compilation.

    Args:
        cache_size_limit: Cache size limit for dynamo compilation
        capture_scalar_outputs: Whether to capture scalar outputs for Flash Attention compatibility
    """
    try:
        import torch._dynamo as dynamo

        # Set cache size limit (never lower an already-higher value)
        dynamo.config.cache_size_limit = max(dynamo.config.cache_size_limit, cache_size_limit)
        logger.debug(f"Set torch._dynamo cache_size_limit to {dynamo.config.cache_size_limit}")

        # Configure scalar output capture if requested
        if capture_scalar_outputs:
            dynamo.config.capture_scalar_outputs = True
            logger.debug("Enabled torch._dynamo scalar output capture")

    except ImportError:
        logger.warning("torch._dynamo not available, skipping dynamo configuration")


def enable_torch_dynamo_scalar_outputs():
    """Enable torch.dynamo to capture scalar outputs for better Flash Attention + torch.compile compatibility."""
    try:
        import torch._dynamo.config

        torch._dynamo.config.capture_scalar_outputs = True
    except ImportError:
        logger.warning("torch._dynamo not available, skipping scalar output capture configuration")


def patch_prepare_fa2_from_position_ids():
    """
    Apply a simple targeted patch to fix the prepare_fa2_from_position_ids function
    for torch.compile compatibility.

    This is the key function that needs the fix for the max_length computation.
    """
    try:
        import transformers.modeling_flash_attention_utils as fa_utils

        def prepare_fa2_from_position_ids(query, key, value, position_ids):
            """
            This function returns necessary arguments to call `flash_attn_varlen_func`.
            All three query, key, value states will be flattened.
            Cumulative lengths of each examples in the batch will be extracted from position_ids.

            NOTE: ideally cumulative lengths should be prepared at the data collator stage

            This version includes the torch.compile fix for max_length computation.
            """
            query = query.view(-1, query.size(-2), query.size(-1))
            key = key.contiguous().view(-1, key.size(-2), key.size(-1))
            value = value.contiguous().view(-1, value.size(-2), value.size(-1))
            position_ids = position_ids.flatten()
            indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)

            cu_seq_lens = torch.cat(
                (
                    indices_q[position_ids == 0],
                    torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32),
                )
            )

            # The .item() call ensures we get an integer instead of a FakeTensor during torch.compile
            max_length = position_ids.max().item() + 1

            return (query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))

        # Apply the patch
        fa_utils.prepare_fa2_from_position_ids = prepare_fa2_from_position_ids

        return True

    except Exception as e:
        logger.warning(f"Failed to patch prepare_fa2_from_position_ids: {e}")
        return False


def apply_flash_attention_compile_fix():
    """
    Apply the Flash Attention + torch.compile compatibility fix.

    This enables scalar output capture and patches the key function that causes issues.
    Note: This function is focused solely on Flash Attention compatibility.
    For dynamo configuration (cache size, etc.), use configure_torch_dynamo() separately.
    """
    # Enable scalar output capture for Flash Attention compatibility
    enable_torch_dynamo_scalar_outputs()

    # Apply the targeted patch
    success = patch_prepare_fa2_from_position_ids()

    if not success:
        logger.warning("Flash Attention + torch.compile compatibility fix failed")

    return success


def _resolve_compile_kwargs(config: CompileConfig) -> Dict[str, Any]:
    """Translate a CompileConfig into torch.compile keyword arguments.

    ``torch.compile`` is keyword-only, rejects unknown kwargs, and treats
    ``mode`` and ``options`` as mutually exclusive -- inductor options must go
    through the ``options`` dict, not be flattened into the call. A non-empty
    ``CompileConfig.options`` therefore replaces ``mode``, and combining
    options with a non-default mode is an explicit error.
    """
    options_dict = config.options.to_dict() if hasattr(config.options, "to_dict") else dict(config.options)
    compile_kwargs: Dict[str, Any] = {
        "fullgraph": config.fullgraph,
        "dynamic": config.dynamic,
    }
    if options_dict:
        if config.mode and config.mode != "default":
            raise ValueError(
                "CompileConfig.mode and CompileConfig.options are mutually exclusive in torch.compile; set only one."
            )
        compile_kwargs["options"] = options_dict
    else:
        compile_kwargs["mode"] = config.mode
    if config.backend is not None:
        compile_kwargs["backend"] = config.backend
    return compile_kwargs


def compile_model(model: nn.Module, config: CompileConfig) -> nn.Module:
    """Compile the model with Flash Attention compatibility.

    Args:
        model: The model to compile.
        config: Compile configuration.

    Returns:
        The compiled model.
    """
    if not config.enabled:
        logger.info("torch.compile is disabled")
        return model

    # Configure torch._dynamo settings
    configure_torch_dynamo(cache_size_limit=config.dynamo_cache_size_limit)

    # Apply Flash Attention compatibility fix
    apply_flash_attention_compile_fix()

    compile_kwargs = _resolve_compile_kwargs(config)

    logger.info(f"Compiling model with backend={config.backend}, mode={config.mode}, dynamic={config.dynamic}")

    try:
        compiled_model = torch.compile(model, **compile_kwargs)
        logger.info("Model compilation successful")
        return compiled_model
    except Exception as e:
        logger.error(f"Model compilation failed: {type(e).__name__}: {e}")
        logger.info("Returning original model")
        return model


# Apply Flash Attention fix when module is imported (dynamo config happens per-compilation)
_FLASH_ATTENTION_FIX_APPLIED = apply_flash_attention_compile_fix()

# Set a reasonable dynamo cache limit at import time so that per-method @torch.compile
# decorators (e.g. Float32RMSNorm) don't hit FailOnRecompileLimitHit from varying guard
# states (ndim, autocast, grad mode combinations). The PyTorch default of 8 is too low.
# When full model compile is enabled, configure_torch_dynamo is called again with the
# user-configured value (default 256) via compile_model(), and the max() in
# configure_torch_dynamo ensures the limit is never lowered.
configure_torch_dynamo(cache_size_limit=256)
