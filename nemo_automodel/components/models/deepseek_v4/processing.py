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

"""Processor for DeepSeek-V4-Flash-Vision-Exp."""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoTokenizer
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config

IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
COMPRESS_PAD_TO = 4
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
BOS_TOKEN = "<｜begin▁of▁sentence｜>"
EOS_TOKEN = "<｜end▁of▁sentence｜>"
USER_TOKEN = "<｜User｜>"
ASSISTANT_TOKEN = "<｜Assistant｜>"
THINKING_END_TOKEN = "</think>"

# ``default_collate_fn`` derives assistant-only label boundaries through the
# tokenizer's template. The processor itself renders the richer multimodal
# form below; this text-only template intentionally leaves the assistant marker
# out of a user-only prompt when ``add_generation_prompt=False`` so the generic
# marker derivation can isolate it.
DEEPSEEK_V4_LABEL_CHAT_TEMPLATE = """{%- if messages %}{{- bos_token }}{%- endif -%}
{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
{{- message['content'] -}}
{%- elif message['role'] in ['user', 'developer'] -%}
{{- '<｜User｜>' + message['content'] -}}
{%- elif message['role'] == 'assistant' -%}
{{- '<｜Assistant｜></think>' + message['content'] + eos_token -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<｜Assistant｜></think>' -}}{%- endif -%}"""


def grid_tokens(
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
) -> tuple[int, int, int]:
    """Return the aligned LLM grid and N-layout token count for one image."""
    n_llm_h = math.ceil((best_height // patch_size) / downsample_ratio)
    n_llm_w = math.ceil((best_width // patch_size) / downsample_ratio)
    num_tokens = n_llm_h * (n_llm_w + 1) + 2
    if n_llm_h % 2 == 1:
        num_tokens += n_llm_w + 1
    num_tokens += (n_llm_h + 1) // 2 * (n_llm_w + 1) % 2 * 2
    return n_llm_h, n_llm_w, num_tokens


def solve_resize_ratio(
    height: int | float,
    width: int | float,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int, int]:
    """Solve the reference resize approximation for a visual-token budget."""
    ratio = height / width
    max_w_float = math.sqrt((max_n_token - 2) / ratio + 0.25) - 0.5
    max_h_float = max_w_float * ratio
    if max_w_float < 1.0:
        max_w = 1
        max_h = (max_n_token - 2) // (max_w + 1)
        if max_h % 2 == 1:
            max_h -= 1
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    elif max_h_float < 2.0:
        max_h = 2
        max_w = ((max_n_token - 2) // max_h) - 1
        if max_w <= 1:
            raise ValueError("vision_max_n_token is too small for the requested image ratio")
        best_width = max_w * patch_size * downsample_ratio
        best_height = max_h * patch_size * downsample_ratio
    else:
        max_w = math.floor(max_w_float)
        max_h = math.floor(max_h_float)
        if max_h % 2 == 1:
            max_h -= 1
        beta = min(max_w * patch_size * downsample_ratio / width, max_h * patch_size * downsample_ratio / height)
        best_width = math.floor(width * beta / patch_size) * patch_size
        best_height = math.floor(height * beta / patch_size) * patch_size
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    return n_llm_h, n_llm_w, best_height, best_width, num_tokens


def safe_resize(
    height: int | float,
    width: int | float,
    best_height: int,
    best_width: int,
    patch_size: int,
    downsample_ratio: int,
    max_n_token: int,
) -> tuple[int, int, int, int]:
    """Fit an image to the reference N-layout visual-token budget."""
    max_n_token -= COMPRESS_PAD_TO - 1
    n_llm_h, n_llm_w, num_tokens = grid_tokens(best_height, best_width, patch_size, downsample_ratio)
    budget = max_n_token
    while num_tokens > max_n_token:
        n_llm_h, n_llm_w, best_height, best_width, num_tokens = solve_resize_ratio(
            height, width, patch_size, downsample_ratio, budget
        )
        budget -= 1
        if budget <= 2:
            raise ValueError("Unable to fit image into vision_max_n_token")
    return n_llm_h, n_llm_w, best_height, best_width


def build_image_block(n_llm_h: int, n_llm_w: int, start_pos: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build reference N-layout token types and aligned-image permutation.

    Returns:
        ``types`` with layout ``[image_block_tokens]`` and ``perm`` with layout
        ``[n_llm_h * n_llm_w]``. ``perm`` maps row-major aligner output into
        the N-layout IMAGE slots.
    """
    compress_pad = COMPRESS_PAD_TO - 1 - start_pos % COMPRESS_PAD_TO
    pad_h = n_llm_h % 2
    rows = n_llm_h + pad_h
    row_len = n_llm_w + 1
    pad_last = rows // 2 * row_len % 2 * 2
    types = torch.tensor(
        ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h + [IMAGE_PAD] * (row_len * pad_h),
        dtype=torch.int64,
    )
    order = torch.arange(rows * row_len).view(rows // 2, 2, row_len).transpose(1, 2).reshape(-1)
    image_idx = torch.full((rows * row_len,), -1, dtype=torch.int64)
    image_idx.view(rows, row_len)[:n_llm_h, :n_llm_w] = torch.arange(n_llm_h * n_llm_w).view(n_llm_h, n_llm_w)
    perm = image_idx[order]
    perm = perm[perm >= 0]
    types = torch.cat(
        [
            torch.full((compress_pad,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_START]),
            types[order],
            torch.full((pad_last,), IMAGE_PAD, dtype=torch.int64),
            torch.tensor([IMAGE_END]),
        ]
    )
    return types, perm


def _load_pil_image(value: Any) -> Image.Image:
    """Load a supported local, byte, dictionary, or PIL image as RGB."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB")
    if isinstance(value, bytes):
        with Image.open(io.BytesIO(value)) as image:
            return image.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return _load_pil_image(value["bytes"])
        if value.get("path"):
            return _load_pil_image(value["path"])
        if value.get("image") is not None:
            return _load_pil_image(value["image"])
    raise TypeError(f"Unsupported DeepSeek-V4 image input: {type(value).__name__}")


def preprocess_image(
    value: Any,
    config: DeepseekV4Config,
) -> tuple[torch.Tensor, int, int, int, int]:
    """Transform one image into normalized ViT patches.

    Returns:
        Patches with layout ``[n_vit_h * n_vit_w, 3, patch_size, patch_size]``
        followed by ``n_vit_h``, ``n_vit_w``, ``n_llm_h``, and ``n_llm_w``.
    """
    image = _load_pil_image(value)
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image dimensions must be positive, got {width}x{height}")

    patch_size = int(config.vision_patch_size)
    max_wh_ratio = getattr(config, "vision_max_wh_ratio", None)
    logical_width: int | float = width
    logical_height: int | float = height
    if max_wh_ratio is not None and logical_width > logical_height * float(max_wh_ratio):
        logical_width = logical_height * float(max_wh_ratio)
    min_pixels = int(config.vision_min_pixels)
    if 0 < logical_width * logical_height < min_pixels:
        ratio = (min_pixels / (logical_width * logical_height)) ** 0.5
        logical_width = int(logical_width * ratio)
        logical_height = int(logical_height * ratio)

    best_width = math.ceil(logical_width / patch_size) * patch_size
    best_height = math.ceil(logical_height / patch_size) * patch_size
    n_llm_h, n_llm_w, best_height, best_width = safe_resize(
        logical_height,
        logical_width,
        best_height,
        best_width,
        patch_size,
        int(config.vision_downsample_ratio),
        int(config.vision_max_n_token),
    )
    n_vit_h, n_vit_w = best_height // patch_size, best_width // patch_size
    if max_wh_ratio is not None and image.width >= float(max_wh_ratio) * image.height:
        image = image.resize((best_width, best_height))
    else:
        image = ImageOps.pad(image, (best_width, best_height), color=(127, 127, 127))
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
    pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        pixels.reshape(3, n_vit_h, patch_size, n_vit_w, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(n_vit_h * n_vit_w, 3, patch_size, patch_size)
    )
    return patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w


class DeepseekV4VisionProcessor(ProcessorMixin):
    """Exact prompt and dynamic-resolution image processor for DSV4 Vision."""

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Any,
        config: DeepseekV4Config,
        chat_template: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.config = config
        if getattr(tokenizer, "chat_template", None) is None:
            tokenizer.chat_template = DEEPSEEK_V4_LABEL_CHAT_TEMPLATE
        super().__init__(tokenizer=tokenizer, chat_template=chat_template, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs: Any) -> "DeepseekV4VisionProcessor":
        """Load the tokenizer and vision settings from one checkpoint."""
        tokenizer_kwargs = {
            key: kwargs[key]
            for key in ("cache_dir", "revision", "token", "trust_remote_code", "local_files_only")
            if key in kwargs
        }
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)
        config = DeepseekV4Config.from_pretrained(pretrained_model_name_or_path, **tokenizer_kwargs)
        return cls(tokenizer=tokenizer, config=config)

    @property
    def image_token_id(self) -> int:
        """Return the textual placeholder ID that is expanded before padding."""
        token_id = self.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if token_id is None or token_id == getattr(self.tokenizer, "unk_token_id", None):
            raise ValueError(f"Token not found in tokenizer: {IMAGE_PLACEHOLDER}")
        return int(token_id)

    @staticmethod
    def _content_text_and_images(content: Any) -> tuple[str, list[Any]]:
        """Render ordered content blocks and collect their image values."""
        if isinstance(content, str):
            if IMAGE_PLACEHOLDER in content:
                raise ValueError("Image placeholders must come from image content blocks")
            return content, []
        if content is None:
            return "", []
        if not isinstance(content, list):
            raise TypeError(f"Message content must be a string or list, got {type(content).__name__}")
        parts: list[str] = []
        images: list[Any] = []
        for block in content:
            if not isinstance(block, dict):
                raise TypeError("DeepSeek-V4 content blocks must be dictionaries")
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if IMAGE_PLACEHOLDER in text:
                    raise ValueError("Image placeholders must come from image content blocks")
                parts.append(text)
            elif block_type in ("image", "image_url"):
                value = block.get("image")
                if value is None:
                    value = block.get("image_url")
                if isinstance(value, dict):
                    value = value.get("url") or value.get("image") or value
                if value is None:
                    raise ValueError("Image content block has no image value")
                parts.append(IMAGE_PLACEHOLDER)
                images.append(value)
            else:
                raise NotImplementedError(f"Unsupported DeepSeek-V4 content block type: {block_type}")
        # The released encoder renders structured content blocks with a blank
        # line between blocks, including between an image placeholder and text.
        return "\n\n".join(parts), images

    def _render_conversation(self, conversation: Sequence[dict[str, Any]]) -> tuple[str, list[Any]]:
        """Render one standard conversation into the released chat format."""
        prompt = BOS_TOKEN
        images: list[Any] = []
        for index, message in enumerate(conversation):
            role = message.get("role")
            content, message_images = self._content_text_and_images(message.get("content"))
            images.extend(message_images)
            if role == "system":
                prompt += content
            elif role in ("user", "developer"):
                prompt += USER_TOKEN + content
                next_role = conversation[index + 1].get("role") if index + 1 < len(conversation) else None
                if next_role in ("assistant", None):
                    prompt += ASSISTANT_TOKEN + THINKING_END_TOKEN
            elif role == "assistant":
                prompt += content
                if not message.get("wo_eos", False):
                    prompt += EOS_TOKEN
            else:
                raise NotImplementedError(f"Unsupported DeepSeek-V4 message role: {role}")
        return prompt, images

    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, Any]] | Sequence[Sequence[dict[str, Any]]],
        *,
        tokenize: bool = False,
        return_dict: bool = False,
        return_tensors: str | None = None,
        processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Render or tokenize one conversation or a batch of conversations."""
        del kwargs
        is_batched = bool(conversation) and isinstance(conversation[0], (list, tuple))
        conversations = list(conversation) if is_batched else [conversation]
        rendered = [self._render_conversation(sample) for sample in conversations]
        texts = [item[0] for item in rendered]
        images = [item[1] for item in rendered]
        if not tokenize:
            return texts if is_batched else texts[0]
        result = self(
            text=texts,
            images=images,
            return_tensors=return_tensors,
            **(processor_kwargs or {}),
        )
        if return_dict:
            return result
        return result["input_ids"]

    def __call__(
        self,
        text: str | list[str],
        images: Any = None,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """Tokenize text and expand each image placeholder into pseudo IDs.

        Returns a batch whose ``input_ids``, ``attention_mask``, and
        ``vision_token_types`` use layout ``[batch, sequence]``;
        ``pixel_values`` uses ``[all_patches, 3, patch_size, patch_size]`` and
        ``image_grid_hws`` uses ``[all_images, 2]``.
        """
        texts = [text] if isinstance(text, str) else list(text)
        if images is None:
            image_groups = [[] for _ in texts]
        elif len(texts) == 1 and not isinstance(images, list):
            image_groups = [[images]]
        elif isinstance(images, list) and (not images or isinstance(images[0], list)):
            image_groups = images
        elif isinstance(images, list) and len(images) == len(texts):
            image_groups = [[image] for image in images]
        else:
            raise ValueError("Images must be grouped per text sample")
        if len(image_groups) != len(texts):
            raise ValueError(f"Expected {len(texts)} image groups, got {len(image_groups)}")

        padding = kwargs.pop("padding", False)
        truncation = bool(kwargs.pop("truncation", False))
        max_length = kwargs.pop("max_length", None)
        kwargs.pop("return_dict", None)
        tokenized = self.tokenizer(texts, add_special_tokens=False, padding=False, **kwargs)

        expanded_ids: list[list[int]] = []
        expanded_types: list[list[int]] = []
        all_patches: list[torch.Tensor] = []
        all_grids: list[tuple[int, int]] = []
        for sample_ids, sample_images in zip(tokenized["input_ids"], image_groups):
            placeholder_count = sum(token == self.image_token_id for token in sample_ids)
            if placeholder_count != len(sample_images):
                raise ValueError(f"Found {placeholder_count} image placeholders but got {len(sample_images)} images")
            output_ids: list[int] = []
            output_types: list[int] = []
            image_iter = iter(sample_images)
            for token in sample_ids:
                if token != self.image_token_id:
                    output_ids.append(int(token))
                    output_types.append(-1)
                    continue
                patches, n_vit_h, n_vit_w, n_llm_h, n_llm_w = preprocess_image(next(image_iter), self.config)
                types, _ = build_image_block(n_llm_h, n_llm_w, len(output_ids))
                output_ids.extend((int(self.config.vocab_size) + types).tolist())
                output_types.extend(types.tolist())
                all_patches.append(patches)
                all_grids.append((n_vit_h, n_vit_w))
            if truncation and max_length is not None:
                output_ids = output_ids[:max_length]
                output_types = output_types[:max_length]
                if output_types.count(IMAGE_START) != output_types.count(IMAGE_END):
                    raise ValueError("max_length truncates a DeepSeek-V4 image block")
            expanded_ids.append(output_ids)
            expanded_types.append(output_types)

        if padding in (True, "longest"):
            padded_length = max(len(ids) for ids in expanded_ids)
        elif padding == "max_length":
            if max_length is None:
                raise ValueError("padding='max_length' requires max_length")
            padded_length = int(max_length)
        else:
            lengths = {len(ids) for ids in expanded_ids}
            if len(lengths) != 1 and return_tensors is not None:
                raise ValueError("Variable-length batches require padding")
            padded_length = max(lengths)

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("DeepSeek-V4 tokenizer must define pad_token_id or eos_token_id")
        input_rows: list[list[int]] = []
        type_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for ids, types in zip(expanded_ids, expanded_types):
            if len(ids) > padded_length:
                if not truncation:
                    raise ValueError(f"Sequence length {len(ids)} exceeds padding length {padded_length}")
                ids = ids[:padded_length]
                types = types[:padded_length]
            pad_count = padded_length - len(ids)
            input_rows.append(ids + [int(pad_token_id)] * pad_count)
            type_rows.append(types + [-1] * pad_count)
            mask_rows.append([1] * len(ids) + [0] * pad_count)

        data: dict[str, Any] = {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
            "vision_token_types": torch.tensor(type_rows, dtype=torch.long),
        }
        if all_patches:
            data["pixel_values"] = torch.cat(all_patches, dim=0)
            data["image_grid_hws"] = torch.tensor(all_grids, dtype=torch.long)
        return BatchFeature(data=data, tensor_type=return_tensors)

    def batch_decode(self, *args: Any, **kwargs: Any) -> list[str]:
        """Forward batched decoding to the underlying tokenizer."""
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> str:
        """Forward decoding to the underlying tokenizer."""
        return self.tokenizer.decode(*args, **kwargs)


__all__ = [
    "ASSISTANT_TOKEN",
    "BOS_TOKEN",
    "COMPRESS_PAD_TO",
    "DeepseekV4VisionProcessor",
    "EOS_TOKEN",
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_PAD",
    "IMAGE_PLACEHOLDER",
    "IMAGE_START",
    "USER_TOKEN",
    "build_image_block",
    "grid_tokens",
    "preprocess_image",
    "safe_resize",
    "solve_resize_ratio",
]
