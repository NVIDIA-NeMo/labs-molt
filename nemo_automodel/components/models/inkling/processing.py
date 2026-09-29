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

"""Native processor construction and multimodal token replacement for Inkling."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from .feature_extraction import InklingFeatureExtractor
from .image_processing import InklingImageProcessor

_INKLING_END_OF_SAMPLING_TOKEN = "<|content_model_end_sampling|>"


class InklingProcessorKwargs(ProcessingKwargs, total=False):
    """Processor kwargs with Inkling's reference audio-loading default."""

    _defaults = {"audio_kwargs": {"load_audio_backend": "torchaudio"}}


class InklingProcessor(ProcessorMixin):
    """Combine the native Inkling image/audio processors with a tokenizer."""

    valid_processor_kwargs = InklingProcessorKwargs

    def __init__(
        self,
        feature_extractor: InklingFeatureExtractor,
        image_processor: InklingImageProcessor,
        tokenizer: PreTrainedTokenizerBase,
        chat_template: str | None = None,
        image_token: str = "<|unused_200054|>",
        audio_token: str = "<|unused_200053|>",
        image_bos_token: str = "<|content_image|>",
        audio_bos_token: str = "<|content_audio_input|>",
        num_dmel_bins: int = 16,
        dmel_min_value: float = -7.0,
        dmel_max_value: float = 2.0,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.image_token = getattr(tokenizer, "image_token", image_token)
        self.image_token_id = tokenizer.encode(self.image_token, add_special_tokens=False)[0]
        self.audio_token = getattr(tokenizer, "audio_token", audio_token)
        self.audio_token_id = tokenizer.encode(self.audio_token, add_special_tokens=False)[0]
        self.image_bos_token = image_bos_token
        self.image_bos_token_id = tokenizer.encode(image_bos_token, add_special_tokens=False)[0]
        self.audio_bos_token = audio_bos_token
        self.audio_bos_token_id = tokenizer.encode(audio_bos_token, add_special_tokens=False)[0]
        self.num_dmel_bins = num_dmel_bins
        self.dmel_min_value = dmel_min_value
        self.dmel_max_value = dmel_max_value
        self.bin_centers = torch.linspace(dmel_min_value, dmel_max_value, num_dmel_bins, dtype=torch.float64)
        super().__init__(feature_extractor, image_processor, tokenizer, chat_template=chat_template)

    def __call__(
        self,
        images: Any | None = None,
        text: str | list[str] | None = None,
        videos: Any | None = None,
        audio: Any | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """Prepare text, image patches, and dMel tokens without version-specific HF hooks.

        Transformers releases before Inkling was upstreamed dispatch audio directly
        to the feature extractor and do not expand multimodal placeholders. Owning
        this small dispatcher keeps the checkpoint usable with AutoModel's pinned
        Transformers version.
        """
        if images is None and text is None and videos is None and audio is None:
            raise ValueError(f"You need to provide at least one input to call {type(self).__name__}")
        if videos is not None:
            raise ValueError("Inkling accepts images and audio, but not video inputs")

        merged_kwargs = self._merge_kwargs(
            self.valid_processor_kwargs,
            tokenizer_init_kwargs=getattr(self.tokenizer, "init_kwargs", {}),
            **kwargs,
        )
        processed_images: dict[str, Any] = {}
        image_replacements: list[str] = []
        if images is not None:
            processed_images = self.image_processor(images, **merged_kwargs["images_kwargs"])
            image_replacements = [
                self.replace_image_token(processed_images, image_idx=idx)
                for idx in range(len(processed_images["num_patches"]))
            ]

        processed_audio: dict[str, Any] = {}
        audio_replacements: list[str] = []
        if audio is not None:
            audio_batch = self._normalize_audio_batch(audio)
            processed_audio, audio_replacements = self._process_audio(
                audio_batch,
                **merged_kwargs["audio_kwargs"],
            )

        text_inputs: dict[str, Any] = {}
        text_kwargs = merged_kwargs["text_kwargs"]
        return_tensors = text_kwargs.get("return_tensors")
        if text is not None:
            text_batch = [text] if isinstance(text, str) else list(text)
            text_batch = self._replace_multimodal_tokens(
                text_batch,
                image_replacements=image_replacements,
                audio_replacements=audio_replacements,
            )
            text_inputs = self.tokenizer(text_batch, **text_kwargs)

        data = {**text_inputs, **processed_images, **processed_audio}
        data = {key: value for key, value in data.items() if key not in self.unused_input_names}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @staticmethod
    def _normalize_audio_batch(audio: Any) -> list[Any]:
        """Normalize one waveform or a batch into a list of clips."""
        if isinstance(audio, (np.ndarray, torch.Tensor)):
            return [audio]
        if isinstance(audio, (list, tuple)):
            if not audio:
                raise ValueError("Received an empty audio input")
            if isinstance(audio[0], (int, float, np.integer, np.floating)):
                return [audio]
            return list(audio)
        return [audio]

    def _replace_multimodal_tokens(
        self,
        text: list[str],
        *,
        image_replacements: list[str],
        audio_replacements: list[str],
    ) -> list[str]:
        """Expand media placeholders once per corresponding input, in batch order."""
        replacements_by_token = (
            (self.image_token, image_replacements),
            (self.audio_token, audio_replacements),
        )
        for token, replacements in replacements_by_token:
            expected = sum(sample.count(token) for sample in text)
            if replacements and expected != len(replacements):
                raise ValueError(f"Received {len(replacements)} media inputs for {expected} {token!r} placeholders")
            if not replacements:
                continue
            replacement_iter = iter(replacements)
            for idx, sample in enumerate(text):
                parts = sample.split(token)
                text[idx] = parts[0] + "".join(next(replacement_iter) + part for part in parts[1:])
        return text

    def _extract_dmel_bins(self, input_features: torch.Tensor) -> torch.Tensor:
        """Quantize continuous log-mel values into dMel token IDs.

        Args:
            input_features: Tensor of shape ``[batch, frames, mel_bins]``.

        Returns:
            Int tensor of shape ``[batch, frames, mel_bins]``.
        """
        bin_centers = self.bin_centers.to(input_features.device)
        mel = input_features.double().clamp(min=self.dmel_min_value, max=self.dmel_max_value)
        return (mel.unsqueeze(-1) - bin_centers).abs().argmin(dim=-1).to(torch.int32)

    def _process_audio(self, audio: Any, **kwargs: Any) -> tuple[dict[str, torch.Tensor], list[str]]:
        """Extract, quantize, and count a batch of audio clips."""
        audio_inputs = self.feature_extractor(audio, **kwargs)
        processed_audio = {
            "audio_input_ids": self._extract_dmel_bins(audio_inputs["input_features"]),
            "audio_input_ids_mask": audio_inputs.get("input_features_mask"),
        }
        replacements = [self.replace_audio_token(processed_audio, audio_idx) for audio_idx in range(len(audio))]
        return processed_audio, replacements

    def replace_image_token(self, image_inputs: dict[str, torch.Tensor], image_idx: int) -> str:
        """Return one soft placeholder per encoded image patch."""
        return self.image_token * int(image_inputs["num_patches"][image_idx])

    def replace_audio_token(self, audio_inputs: dict[str, torch.Tensor], audio_idx: int) -> str:
        """Return one soft placeholder per valid audio frame."""
        audio_mask = audio_inputs.get("audio_input_ids_mask")
        token_count = (
            int(audio_mask[audio_idx].sum())
            if audio_mask is not None
            else int(audio_inputs["audio_input_ids"][audio_idx].shape[-2])
        )
        return self.audio_token * token_count

    @property
    def unused_input_names(self) -> list[str]:
        """Return processor-only fields omitted from model inputs."""
        return ["num_patches"]

    @property
    def model_input_names(self) -> list[str]:
        """Return the deduplicated model input field names."""
        names = [
            "audio_input_ids",
            "audio_input_ids_mask",
            *self.image_processor.model_input_names,
            *self.tokenizer.model_input_names,
        ]
        return [name for name in dict.fromkeys(names) if name not in self.unused_input_names]


def build_inkling_processor(pretrained_model_name_or_path: str, **kwargs: Any) -> InklingProcessor:
    """Load Inkling's native processor without Transformers model registration.

    Args:
        pretrained_model_name_or_path: Hugging Face model ID or local snapshot.
        **kwargs: Download/cache arguments accepted by Transformers ``from_pretrained`` methods.

    Returns:
        A configured native Inkling processor.
    """
    load_keys = {
        "cache_dir",
        "force_download",
        "local_files_only",
        "revision",
        "subfolder",
        "token",
        "trust_remote_code",
    }
    load_kwargs = {key: value for key, value in kwargs.items() if key in load_keys}
    processor_dict, _ = InklingProcessor.get_processor_dict(pretrained_model_name_or_path, **load_kwargs)

    image_config = dict(processor_dict.pop("image_processor", {}))
    image_config.pop("image_processor_type", None)
    feature_config = dict(processor_dict.pop("feature_extractor", {}))
    feature_config.pop("feature_extractor_type", None)
    processor_dict.pop("processor_class", None)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **load_kwargs)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = _INKLING_END_OF_SAMPLING_TOKEN
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor_keys = {
        "image_token",
        "audio_token",
        "image_bos_token",
        "audio_bos_token",
        "num_dmel_bins",
        "dmel_min_value",
        "dmel_max_value",
        "chat_template",
    }
    processor_kwargs = {key: value for key, value in processor_dict.items() if key in processor_keys}
    processor_kwargs.update({key: value for key, value in kwargs.items() if key in processor_keys})
    return InklingProcessor(
        feature_extractor=InklingFeatureExtractor(**feature_config),
        image_processor=InklingImageProcessor(**image_config),
        tokenizer=tokenizer,
        **processor_kwargs,
    )


__all__ = ["InklingProcessor", "build_inkling_processor"]
