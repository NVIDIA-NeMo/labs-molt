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

"""AutoMFU: Automatic Model FLOPs Utilization calculator.

Similar interface to HuggingFace AutoModel, this module provides automatic
MFU calculation for various model architectures.
"""

from __future__ import annotations

import logging
import math
from os import PathLike
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PretrainedConfig

from nemo_automodel.components.utils.flops_utils import (
    calculate_mfu,
    get_flops_formula_for_hf_config,
)

logger = logging.getLogger(__name__)

# Device theoretical FLOPS (FLOPs/s) adapted from https://github.com/verl-project/verl/blob/main/verl/utils/flops_counter.py#L22-L85
_DEVICE_FLOPS: dict[str, float] = {
    "CPU": 448e9,
    "GB200": 2.5e15,
    "B200": 2.25e15,
    "MI300X": 1336e12,
    "H100": 989e12,
    "H800": 989e12,
    "H200": 989e12,
    "A100": 312e12,
    "A800": 312e12,
    "L40S": 362.05e12,
    "L40": 181.05e12,
    "A40": 149.7e12,
    "L20": 119.5e12,
    "H20": 148e12,
    "910B": 354e12,
    "Ascend910": 354e12,
    "RTX 3070 Ti": 21.75e12,
    "RTX PRO 6000 Blackwell Max-Q Workstation Edition": 438.9e12,
    "RTX PRO 6000 Blackwell Workstation Edition": 503.8e12,
}

_UNIT_TO_SCALE = {
    "B": 1e9,
    "K": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
    "P": 1e15,
}

_UNWRAP_ATTRS = ("module", "_orig_mod", "_fsdp_wrapped_module", "model")
_CONFIG_ALIAS_ATTRS = (
    ("n_embd", "hidden_size"),
    ("n_layer", "num_hidden_layers"),
    ("n_head", "num_attention_heads"),
    ("n_positions", "max_position_embeddings"),
    ("n_inner", "intermediate_size"),
)


def get_device_flops(unit: str = "T", device_name: str | None = None) -> float:
    """Get theoretical device FLOPS in a requested unit.

    Args:
        unit: One of ``B/K/M/G/T/P``. Default ``T`` (TFLOPs/s).
        device_name: Optional explicit device name for lookup. If ``None``,
            the current torch device name is inferred.

    Returns:
        Theoretical FLOPS in requested unit. Returns ``float("inf")`` for
        unknown devices.
    """
    unit = unit.upper()
    if unit not in _UNIT_TO_SCALE:
        supported = ", ".join(_UNIT_TO_SCALE.keys())
        raise ValueError(f"Unsupported unit '{unit}'. Supported units: {supported}")

    if device_name is None:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        else:
            device_name = "CPU"

    flops = float("inf")
    normalized_device = str(device_name).lower()
    for key, value in sorted(_DEVICE_FLOPS.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key.lower() in normalized_device:
            flops = value
            break

    return flops / _UNIT_TO_SCALE[unit]


class AutoMFU:
    """Auto MFU calculator - provides MFU calculation for various model architectures.

    This class provides a HuggingFace AutoModel-like interface for calculating
    Model FLOPs Utilization (MFU) during training.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        device: str | None = None,
        peak_tflops: float | None = None,
    ) -> None:
        """Initialize AutoMFU with a model config.

        Args:
            config: HuggingFace PretrainedConfig object
            device: Optional device name override. By default, infer the current device.
            peak_tflops: Optional peak TFLOPs/s override for the training precision.
        """
        self.config = config
        self.flops_formula = get_flops_formula_for_hf_config(config)
        self.reference_mfu = (
            _validate_peak_tflops(peak_tflops)
            if peak_tflops is not None
            else get_device_flops(unit="T", device_name=device)
        )
        if not math.isfinite(self.reference_mfu):
            device_description = repr(device) if device is not None else "the current device"
            logger.warning(
                "Unable to determine the theoretical peak for %s; MFU will be reported as 0%%. "
                "Pass a supported device or peak_tflops directly, or set mfu.device or mfu.peak_tflops in recipe YAML.",
                device_description,
            )

    @classmethod
    def register_device(cls, device: str, peak_tflops: float) -> None:
        """Register or override a device peak TFLOPs entry used for MFU calculation."""
        _DEVICE_FLOPS[str(device)] = float(peak_tflops) * 1e12

    @classmethod
    def from_config(
        cls,
        config_or_path_or_model: PretrainedConfig | str | PathLike[str] | object,
        device: str | None = None,
        peak_tflops: float | None = None,
        **kwargs: Any,
    ) -> AutoMFU:
        """Create AutoMFU from a config object, model object, or model path/ID.

        Args:
            config_or_path_or_model: Either a PretrainedConfig object, a model object
                (the .config attribute will be extracted), or a model ID/local path.
            device: Optional device name override. By default, infer the current device.
            peak_tflops: Optional peak TFLOPs/s override for the training precision.
            **kwargs: Additional arguments passed to AutoConfig.from_pretrained
                when loading from model ID/path.

        Returns:
            AutoMFU instance
        """
        config = config_or_path_or_model
        if isinstance(config, (str, PathLike)):
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(str(config), **kwargs)
        else:
            config = cls._unwrap_config(config)
            cls._ensure_common_config_aliases(config)
        return cls(config, device=device, peak_tflops=peak_tflops)

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_local_path_or_model: str | PathLike[str] | object,
        device: str | None = None,
        peak_tflops: float | None = None,
        **kwargs: Any,
    ) -> AutoMFU:
        """Create AutoMFU from model ID, local path, or a model object.

        Args:
            model_id_or_local_path_or_model: Model ID (e.g., "meta-llama/llama-3-70b"),
                local path, or model object (the .config attribute will be extracted)
            device: Optional device name override. By default, infer the current device.
            peak_tflops: Optional peak TFLOPs/s override for the training precision.
            **kwargs: Additional arguments passed to AutoConfig.from_pretrained

        Returns:
            AutoMFU instance
        """
        return cls.from_config(
            model_id_or_local_path_or_model,
            device=device,
            peak_tflops=peak_tflops,
            **kwargs,
        )

    def __call__(
        self,
        input_ids_or_tensor: torch.Tensor | tuple[int, int],
        time_delta: float,
        world_size: int,
    ) -> float | None:
        """Calculate MFU percentage.

        Args:
            input_ids_or_tensor: Either a tensor (batch_size, seq_len) or
                a tuple of (batch_size, seq_len)
            time_delta: Time taken for forward/backward pass in seconds
            world_size: Number of GPUs used for training

        Returns:
            MFU as a percentage, or None if model not supported
        """
        flops = self.get_flops(input_ids_or_tensor)
        if flops is None:
            return None
        tflops = flops / 1e12
        return calculate_mfu(tflops, world_size, time_delta, reference_mfu=self.reference_mfu)

    def get_flops(
        self,
        input_ids_or_tensor: torch.Tensor | tuple[int, int],
    ) -> float | None:
        """Calculate FLOPs for given input shape.

        Args:
            input_ids_or_tensor: Either a tensor (batch_size, seq_len) or
                a tuple of (batch_size, seq_len)

        Returns:
            FLOPs as a float, or None if model not supported
        """
        if self.flops_formula is None:
            return None

        if isinstance(input_ids_or_tensor, torch.Tensor):
            batch_size, seq_len = input_ids_or_tensor.shape[:2]
        else:
            batch_size, seq_len = input_ids_or_tensor

        try:
            # Explicitly gate transformer fallback on required attributes.
            if getattr(self.flops_formula, "__name__", None) == "transformer_flops":
                required = (
                    "hidden_size",
                    "num_hidden_layers",
                    "num_attention_heads",
                    "intermediate_size",
                    "vocab_size",
                )
                if not all(hasattr(self.config, attr) for attr in required):
                    return None
            return self.flops_formula(self.config, gbs=batch_size, seq_len=seq_len)
        except Exception as e:
            logger.debug("Unable to compute FLOPs for config %s: %s", type(self.config).__name__, e)
            return None

    @staticmethod
    def _unwrap_config(config_or_model: object):
        cur = config_or_model
        visited = set()
        while cur is not None and id(cur) not in visited:
            visited.add(id(cur))

            if hasattr(cur, "config"):
                next_obj = getattr(cur, "config")
                if next_obj is not None and next_obj is not cur:
                    cur = next_obj
                    continue

            moved = False
            for attr in _UNWRAP_ATTRS:
                if hasattr(cur, attr):
                    next_obj = getattr(cur, attr)
                    if next_obj is not None and next_obj is not cur:
                        cur = next_obj
                        moved = True
                        break
            if not moved:
                break
        return cur

    @staticmethod
    def _ensure_common_config_aliases(config: object) -> None:
        for src, dst in _CONFIG_ALIAS_ATTRS:
            if not hasattr(config, dst) and hasattr(config, src):
                try:
                    setattr(config, dst, getattr(config, src))
                except Exception:
                    # Some config objects may block dynamic attrs; fallback
                    # behavior will return None if attrs remain unavailable.
                    pass


def _validate_peak_tflops(peak_tflops: float) -> float:
    peak = float(peak_tflops)
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError("peak_tflops must be a finite value greater than zero")
    return peak
