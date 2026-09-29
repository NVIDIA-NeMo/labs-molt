# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build the GLM-5.3 image processor on Transformers versions before 5.16."""

from __future__ import annotations

import json
import os
from typing import Any

from huggingface_hub.utils import EntryNotFoundError
from transformers import AutoTokenizer
from transformers.models.glm46v.processing_glm46v import Glm46VProcessor
from transformers.processing_utils import ProcessorMixin

from nemo_automodel.components.models.glm5_next.image_processing import Glm5NextImageProcessor
from nemo_automodel.shared.import_utils import safe_import_from

_, Glm46VVideoProcessor = safe_import_from(
    "transformers.models.glm46v.video_processing_glm46v",
    "Glm46VVideoProcessor",
    msg="GLM-5.3 processor construction requires torchvision. Install the `vlm` extra.",
)

_MEDIA_REMINDER = (
    '{{- "<reminder>You are unable to process this " ~ media_type ~ '
    '" because you don\'t have multi-modal input ability. Try different methods.</reminder>" }}'
)
_IMAGE_PLACEHOLDER = """{%- if media_type == 'image' -%}
                {{- "<|begin_of_image|><|image|><|end_of_image|>" }}
                {%- else -%}
                {{- "<reminder>You are unable to process this " ~ media_type ~ " because you don't have multi-modal input ability. Try different methods.</reminder>" }}
                {%- endif -%}"""


def _load_processor_config(path_or_id: str, **kwargs: Any) -> dict[str, Any]:
    local = os.path.join(path_or_id, "processor_config.json")
    if not os.path.isfile(local):
        from huggingface_hub import hf_hub_download

        hub_kwargs = {key: kwargs[key] for key in ("cache_dir", "revision", "token") if key in kwargs}
        local = hf_hub_download(path_or_id, "processor_config.json", **hub_kwargs)
    with open(local, encoding="utf-8") as stream:
        return json.load(stream)


def _load_chat_template(path_or_id: str, **kwargs: Any) -> str | None:
    local = os.path.join(path_or_id, "chat_template.jinja")
    if not os.path.isfile(local):
        if os.path.isdir(path_or_id):
            return None
        try:
            from huggingface_hub import hf_hub_download

            hub_kwargs = {key: kwargs[key] for key in ("cache_dir", "revision", "token") if key in kwargs}
            local = hf_hub_download(path_or_id, "chat_template.jinja", **hub_kwargs)
        except EntryNotFoundError:
            return None
    with open(local, encoding="utf-8") as stream:
        return stream.read()


def _enable_image_placeholders(template: str | None) -> str | None:
    """Render image content as GLM image tokens while preserving the shipped template.

    The initial GLM-5.3-Flash checkpoint template renders every media block as a
    text-only capability reminder.  That leaves ``Glm46VProcessor.__call__`` no
    ``<|image|>`` token to expand even though it receives and patchifies the image.
    MedPix fine-tuning requires the native begin/image/end token triplet; video
    retains the checkpoint's reminder because this onboarding is image-only.
    """
    if template is None:
        return None
    if "<|image|>" in template:
        return template
    if _MEDIA_REMINDER not in template:
        raise ValueError("GLM-5.3 chat template has no recognized media rendering branch")
    return template.replace(_MEDIA_REMINDER, _IMAGE_PLACEHOLDER, 1)


def build_glm5_next_processor(pretrained_model_name_or_path: str, **kwargs: Any) -> ProcessorMixin:
    """Create the image-only GLM-5.3 processor used by MedPix recipes."""
    processor_config = _load_processor_config(pretrained_model_name_or_path, **kwargs)
    image_kwargs = dict(processor_config.get("image_processor", {}))
    image_kwargs.pop("image_processor_type", None)
    image_processor = Glm5NextImageProcessor(**image_kwargs)
    video_kwargs = dict(processor_config.get("video_processor", {}))
    video_kwargs.pop("video_processor_type", None)
    video_processor = Glm46VVideoProcessor(**video_kwargs)
    tokenizer_kwargs = dict(kwargs)
    tokenizer_kwargs.pop("trust_remote_code", None)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)
    template = _enable_image_placeholders(_load_chat_template(pretrained_model_name_or_path, **kwargs))
    return Glm46VProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=template,
    )


__all__ = ["build_glm5_next_processor"]
