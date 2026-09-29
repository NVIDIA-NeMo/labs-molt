# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Native log-mel feature extraction for Inkling audio inputs."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.audio_utils import mel_filter_bank
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor
from transformers.feature_extraction_utils import BatchFeature
from transformers.utils import PaddingStrategy, TensorType

LOGGER = logging.getLogger(__name__)


def _to_exact_int(value: float, name: str, tolerance: float = 1e-6) -> int:
    """Convert a floating sample count to an exact integer."""
    rounded = round(value)
    if abs(value - rounded) > tolerance:
        raise ValueError(f"{name} must resolve to an integer sample count, got {value}")
    return int(rounded)


def _to_mono_audio(clip: np.ndarray | torch.Tensor | list[float]) -> torch.Tensor:
    """Convert one audio clip to a mono fp32 waveform.

    Args:
        clip: Tensor or array of shape ``[samples]`` or ``[samples, channels]``.

    Returns:
        Tensor of shape ``[samples]`` in fp32.
    """
    waveform = clip if isinstance(clip, torch.Tensor) else torch.as_tensor(np.asarray(clip))
    waveform = waveform.to(torch.float32)
    if waveform.ndim == 2:
        LOGGER.warning("Inkling supports mono audio; averaging the channel axis")
        waveform = waveform.mean(dim=-1)
    elif waveform.ndim != 1:
        raise ValueError(f"Each audio clip must have shape [samples] or [samples, channels], got {waveform.shape}")
    return waveform


class InklingFeatureExtractor(SequenceFeatureExtractor):
    """Extract log-mel spectrograms for Inkling dMel quantization."""

    model_input_names = ["input_features", "input_features_mask"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16_000,
        padding_value: float = 0.0,
        audio_token_duration_s: float = 0.05,
        window_size_multiplier: float = 2.0,
        n_fft: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            **kwargs,
        )
        self.audio_token_duration_s = audio_token_duration_s
        self.window_size_multiplier = window_size_multiplier
        self.hop_length = _to_exact_int(
            audio_token_duration_s * sampling_rate,
            "audio_token_duration_s * sampling_rate",
        )
        self.window_size = _to_exact_int(
            audio_token_duration_s * window_size_multiplier * sampling_rate,
            "audio_token_duration_s * window_size_multiplier * sampling_rate",
        )
        self.n_fft = n_fft or self.window_size
        if self.hop_length <= 0 or self.window_size <= 0 or self.n_fft <= 0:
            raise ValueError("hop_length, window_size, and n_fft must all be positive")

        self.window = torch.hann_window(self.window_size, periodic=True, dtype=torch.float32)
        mel_filters = mel_filter_bank(
            num_frequency_bins=self.n_fft // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=0.0,
            max_frequency=sampling_rate / 2.0,
            sampling_rate=sampling_rate,
            norm="slaney",
            mel_scale="slaney",
        )
        self.mel_filters = torch.from_numpy(np.ascontiguousarray(mel_filters.T, dtype=np.float32))

    def _extract_log_mel(self, waveform: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Compute batched log-mel features.

        Args:
            waveform: Tensor of shape ``[batch, samples]``.
            device: Device used for STFT and filter-bank computation.

        Returns:
            Tensor of shape ``[batch, frames, mel_bins]``.
        """
        right_pad = math.ceil(waveform.shape[-1] / self.hop_length) * self.hop_length - waveform.shape[-1]
        left_pad = max(self.n_fft - self.hop_length, 0)
        waveform = F.pad(waveform, (left_pad, right_pad))
        stft = torch.stft(
            waveform,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window=self.window.to(device),
            center=False,
            return_complex=True,
        )
        magnitudes = torch.view_as_real(stft).pow(2).sum(-1).clamp_min(1e-10).sqrt()
        mel_spectrogram = self.mel_filters.to(device) @ magnitudes
        return mel_spectrogram.clamp_min(1e-10).log10().transpose(1, 2)

    def __call__(
        self,
        raw_speech: np.ndarray | torch.Tensor | list[float] | list[np.ndarray] | list[list[float]],
        sampling_rate: int | None = None,
        padding: bool | str | PaddingStrategy = True,
        max_length: int | None = None,
        truncation: bool = False,
        pad_to_multiple_of: int | None = None,
        return_attention_mask: bool | None = True,
        return_tensors: str | TensorType | None = None,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> BatchFeature:
        """Extract log-mel features from one clip or a batch.

        Args:
            raw_speech: One waveform of shape ``[samples]`` or ``[samples, channels]``,
                or a list of such waveforms.
            sampling_rate: Sampling rate used by the supplied waveform.
            padding: Transformers padding strategy for the waveform batch.
            max_length: Optional maximum waveform length in samples.
            truncation: Whether to truncate waveforms to ``max_length``.
            pad_to_multiple_of: Optional waveform padding multiple.
            return_attention_mask: Whether to return a valid-frame mask.
            return_tensors: Requested output tensor framework.
            device: Device used for feature extraction.
            **kwargs: Additional padding arguments.

        Returns:
            A batch containing ``input_features`` with shape ``[batch, frames, mel_bins]``
            and optionally ``input_features_mask`` with shape ``[batch, frames]``.
        """
        del kwargs
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            raise ValueError(f"Inkling expects audio sampled at {self.sampling_rate} Hz, got {sampling_rate} Hz")
        if sampling_rate is None:
            LOGGER.warning("Pass sampling_rate=%s to avoid silent audio errors", self.sampling_rate)

        if isinstance(raw_speech, (np.ndarray, torch.Tensor)):
            if raw_speech.ndim > 2:
                raise ValueError(f"A single audio array must have one or two dimensions, got {raw_speech.ndim}")
            clips: list[Any] = [raw_speech]
        elif isinstance(raw_speech, (list, tuple)):
            if not raw_speech:
                raise ValueError("Received an empty audio input")
            clips = (
                [raw_speech] if isinstance(raw_speech[0], (int, float, np.integer, np.floating)) else list(raw_speech)
            )
        else:
            raise TypeError(f"Unsupported audio input type: {type(raw_speech).__name__}")

        waveforms = [_to_mono_audio(clip)[:, None] for clip in clips]
        audio_lengths = [len(waveform) for waveform in waveforms]
        padded_inputs = self.pad(
            BatchFeature({"input_features": waveforms, "audio_lengths": audio_lengths}),
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )
        input_waveforms = padded_inputs.input_features.squeeze(-1)
        resolved_device = torch.device(device)
        input_features = self._extract_log_mel(input_waveforms.to(resolved_device), resolved_device)
        num_frames = torch.div(
            padded_inputs.audio_lengths.to(resolved_device) + self.hop_length - 1,
            self.hop_length,
            rounding_mode="floor",
        )
        input_features_mask = (
            torch.arange(input_features.shape[1], device=resolved_device)[None, :] < num_frames[:, None]
        )
        input_features = input_features * input_features_mask.unsqueeze(-1)
        data = {"input_features": input_features}
        if return_attention_mask:
            data["input_features_mask"] = input_features_mask
        return BatchFeature(data=data, tensor_type=return_tensors)


__all__ = ["InklingFeatureExtractor"]
