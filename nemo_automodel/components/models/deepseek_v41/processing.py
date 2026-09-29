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

"""DeepSeek-V4.1 image sizing, row-major token spans, and standard chat processor."""

from __future__ import annotations

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessorMixin

from nemo_automodel.shared.import_utils import safe_import

from .config import DeepseekV41Config, DeepseekV41VisionConfig

_, Image = safe_import("PIL.Image")
_, ImageOps = safe_import("PIL.ImageOps")

TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)
IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
_BOS = "<｜begin▁of▁sentence｜>"
_EOS = "<｜end▁of▁sentence｜>"
_USER = "<｜User｜>"
_SYSTEM = "<｜System｜>"
_ASSISTANT = "<｜Assistant｜></think>"

_LABEL_CHAT_TEMPLATE = """{%- if messages %}{{- bos_token }}{%- endif -%}
{%- for message in messages -%}
{%- if message['role'] == 'system' -%}
{{- '<｜System｜>' + message['content'] -}}
{%- elif message['role'] == 'user' -%}
{{- '<｜User｜>' + message['content'] -}}
{%- elif message['role'] == 'assistant' -%}
{{- '<｜Assistant｜></think>' + message['content'] + eos_token -}}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<｜Assistant｜></think>' -}}{%- endif -%}"""


@dataclass(frozen=True)
class DeepseekV41ImageInput:
    """One image's patches and placement in a padded text batch.

    Attributes:
        batch_index: Batch row containing the image.
        start: Sequence position of IMAGE_START.
        patches: Tensor of shape [n_vit_h * n_vit_w, 3, patch_size, patch_size].
        n_vit_h: Number of ViT patch rows.
        n_vit_w: Number of ViT patch columns.
        types: Integer tensor of shape [image_span], ordered START, row-major
            IMAGE/NEW_LINE rows, and END. All corresponding input IDs are the
            same in-vocabulary image_token_id.
    """

    batch_index: int
    start: int
    patches: torch.Tensor
    n_vit_h: int
    n_vit_w: int
    types: torch.Tensor


def image_inputs_from_batch(
    pixel_values: torch.Tensor,
    image_grid_hws: torch.Tensor,
    vision_token_types: torch.Tensor,
    *,
    downsample_ratio: int,
) -> tuple[DeepseekV41ImageInput, ...]:
    """Validate and partition processor tensors into ordered image inputs.

    Args:
        pixel_values: Tensor of shape [all_patches, 3, patch_size, patch_size],
            concatenated in batch-row then image-span order.
        image_grid_hws: Integer tensor of shape [images, 2] storing ViT height
            and width for each image in the same order.
        vision_token_types: Integer tensor of shape [batch, sequence], with
            TEXT=-1 outside complete image spans and types 0 through 3 inside.
        downsample_ratio: Configured spatial ratio mapping the ViT grid to
            the language model's image rows and columns.

    Returns:
        Ordered image records. Each patches/types tensor is a read-only view
        into the corresponding input, with layouts documented on DeepseekV41ImageInput.
    """
    if pixel_values.ndim != 4 or pixel_values.shape[1] != 3 or pixel_values.shape[2] != pixel_values.shape[3]:
        raise ValueError("pixel_values must have shape [all_patches, 3, patch_size, patch_size]")
    if (
        image_grid_hws.ndim != 2
        or image_grid_hws.shape[1] != 2
        or image_grid_hws.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("image_grid_hws must be an integer tensor of shape [images, 2]")
    if vision_token_types.ndim != 2 or vision_token_types.dtype not in (torch.int32, torch.int64):
        raise ValueError("vision_token_types must be an integer tensor of shape [batch, sequence]")
    if downsample_ratio <= 0:
        raise ValueError("downsample_ratio must be positive")
    grids = image_grid_hws.cpu().tolist()
    records: list[DeepseekV41ImageInput] = []
    patch_start = 0
    for batch_index, row in enumerate(vision_token_types.cpu().tolist()):
        position = 0
        while position < len(row):
            if row[position] == TEXT:
                position += 1
                continue
            if row[position] != IMAGE_START or len(records) >= len(grids):
                raise ValueError("Image types must form complete spans with one image_grid_hws entry per span")
            start = position
            position += 1
            row_widths: list[int] = []
            while position < len(row) and row[position] == IMAGE:
                first_image = position
                while position < len(row) and row[position] == IMAGE:
                    position += 1
                row_widths.append(position - first_image)
                if position >= len(row) or row[position] != IMAGE_NEW_LINE:
                    raise ValueError("Every DeepSeek-V4.1 image row must end with IMAGE_NEW_LINE")
                position += 1
            if not row_widths or len(set(row_widths)) != 1 or position >= len(row) or row[position] != IMAGE_END:
                raise ValueError("Image span must contain equal-width row-major IMAGE rows followed by IMAGE_END")
            position += 1
            n_vit_h, n_vit_w = grids[len(records)]
            if n_vit_h <= 0 or n_vit_w <= 0:
                raise ValueError("Image grid height and width must be positive")
            if len(row_widths) != math.ceil(n_vit_h / downsample_ratio) or row_widths[0] != math.ceil(
                n_vit_w / downsample_ratio
            ):
                raise ValueError("Image token rows and columns disagree with image_grid_hws and downsample_ratio")
            patch_end = patch_start + n_vit_h * n_vit_w
            if patch_end > pixel_values.shape[0]:
                raise ValueError("image_grid_hws requests more patches than pixel_values contains")
            records.append(
                DeepseekV41ImageInput(
                    batch_index,
                    start,
                    pixel_values[patch_start:patch_end],
                    n_vit_h,
                    n_vit_w,
                    vision_token_types[batch_index, start:position],
                )
            )
            patch_start = patch_end
    if len(records) != len(grids) or patch_start != pixel_values.shape[0]:
        raise ValueError("Image spans, image_grid_hws, and pixel_values must describe exactly the same images")
    return tuple(records)


@dataclass(frozen=True)
class _ImageGrid:
    height: int
    width: int
    vit_height: int
    vit_width: int
    llm_height: int
    llm_width: int


def _plan_image_grid(width: int, height: int, config: DeepseekV41VisionConfig) -> _ImageGrid:
    """Apply the released aspect-preserving resize and exact row-major token budget."""
    if width <= 0 or height <= 0 or config.max_image_tokens < 4:
        raise ValueError("Images require positive dimensions and max_image_tokens >= 4")
    logical_width: int | float = width
    logical_height: int | float = height
    if config.max_wh_ratio is not None and width > height * config.max_wh_ratio:
        logical_width = height * config.max_wh_ratio
    if logical_width * logical_height < config.min_pixels:
        ratio = (config.min_pixels / (logical_width * logical_height)) ** 0.5
        logical_width, logical_height = int(logical_width * ratio), int(logical_height * ratio)
    patch_size, downsample = config.patch_size, config.downsample_ratio
    best_width = math.ceil(logical_width / patch_size) * patch_size
    best_height = math.ceil(logical_height / patch_size) * patch_size
    llm_height = math.ceil(best_height // patch_size / downsample)
    llm_width = math.ceil(best_width // patch_size / downsample)
    if llm_height * (llm_width + 1) + 2 > config.max_image_tokens:
        aspect = logical_height / logical_width
        max_width = math.sqrt((config.max_image_tokens - 2) / aspect + 0.25) - 0.5
        max_height = max_width * aspect
        cell = patch_size * downsample
        if max_width < 1:
            best_height, best_width = (config.max_image_tokens - 2) // 2 * cell, cell
        elif max_height < 1:
            best_height, best_width = cell, (config.max_image_tokens - 3) * cell
        else:
            beta = min(math.floor(max_width) * cell / logical_width, math.floor(max_height) * cell / logical_height)
            best_height = math.floor(logical_height * beta / patch_size) * patch_size
            best_width = math.floor(logical_width * beta / patch_size) * patch_size
        llm_height = math.ceil(best_height // patch_size / downsample)
        llm_width = math.ceil(best_width // patch_size / downsample)
    if min(best_height, best_width) <= 0 or llm_height * (llm_width + 1) + 2 > config.max_image_tokens:
        raise ValueError("The image cannot fit in the configured visual-token budget")
    return _ImageGrid(
        best_height, best_width, best_height // patch_size, best_width // patch_size, llm_height, llm_width
    )


class _ImageRecord(TypedDict, total=False):
    path: str
    bytes: bytes
    image: Image.Image


def _load_image(value: Image.Image | str | Path | bytes | _ImageRecord) -> Image.Image:
    """Load the local/PIL/byte image forms emitted by the VLM datasets."""
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return _load_image(value["bytes"])
        if value.get("path") is not None:
            return _load_image(value["path"])
        if value.get("image") is not None:
            return _load_image(value["image"])
        raise ValueError("Image records must contain bytes, path, or image")
    if isinstance(value, bytes):
        with Image.open(io.BytesIO(value)) as image:
            return image.convert("RGB")
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            return image.convert("RGB")
    raise TypeError(f"Unsupported DeepSeek-V4.1 image source: {type(value).__name__}")


def _preprocess_image(
    value: Image.Image | str | Path | bytes | _ImageRecord,
    config: DeepseekV41VisionConfig,
) -> tuple[torch.Tensor, _ImageGrid]:
    """Load, resize, normalize, and flatten one image into RGB patches.

    Returns:
        BF16 patches of shape [vit_height * vit_width, 3, patch_size, patch_size]
        in row-major order, and the complete resize/grid metadata.
    """
    image = _load_image(value)
    grid = _plan_image_grid(image.width, image.height, config)
    if config.max_wh_ratio is not None and image.width >= config.max_wh_ratio * image.height:
        image = image.resize((grid.width, grid.height))
    else:
        image = ImageOps.pad(image, (grid.width, grid.height), color=(127, 127, 127))
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
    pixels = ((pixels - 0.5) / 0.5).to(torch.bfloat16)
    patch_size = config.patch_size
    patches = (
        pixels.reshape(3, grid.vit_height, patch_size, grid.vit_width, patch_size)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid.vit_height * grid.vit_width, 3, patch_size, patch_size)
    )
    return patches, grid


class DeepseekV41Processor(ProcessorMixin):
    """Processor for V4.1 text and local image SFT, using the released chat mode.

    Tool schemas/calls, reasoning traces, and internal task formatting must be
    encoded with DeepSeek's full encoder before calling this processor on text.
    ``apply_chat_template`` rejects those fields instead of silently losing them.

    Args:
        tokenizer: Fast tokenizer containing the configured image placeholder.
        config: Typed checkpoint configuration with image and vision settings.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer: PreTrainedTokenizerFast, config: DeepseekV41Config) -> None:
        self.config = config
        known_id = tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if known_id is not None and known_id != tokenizer.unk_token_id and known_id != config.image_token_id:
            raise ValueError("The tokenizer's DeepSeek image placeholder disagrees with config.image_token_id")
        if tokenizer.chat_template is None:
            tokenizer.chat_template = _LABEL_CHAT_TEMPLATE
        super().__init__(tokenizer=tokenizer, chat_template=_LABEL_CHAT_TEMPLATE)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs: Any) -> DeepseekV41Processor:
        """Load the checkpoint's fast tokenizer and nested vision configuration."""
        loading = {
            key: kwargs[key]
            for key in ("cache_dir", "revision", "token", "trust_remote_code", "local_files_only")
            if key in kwargs
        }
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **loading)
        config = DeepseekV41Config.from_pretrained(pretrained_model_name_or_path, **loading)
        return cls(tokenizer=tokenizer, config=config)

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> list[str]:
        """Persist the model's image settings alongside the tokenizer and processor.

        Args:
            save_directory: Output directory for the Hugging Face artifacts.
            **kwargs: Hugging Face ProcessorMixin save options.

        Returns:
            Processor artifact paths returned by ProcessorMixin.save_pretrained.
        """
        self.config.save_pretrained(save_directory)
        return super().save_pretrained(save_directory, **kwargs)

    def __call__(
        self,
        text: str | list[str],
        images: Image.Image | list[Image.Image] | list[list[Image.Image]] | None = None,
        *,
        return_tensors: str | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs: Any,
    ) -> BatchFeature:
        """Expand image placeholders and concatenate patches without pseudo token IDs.

        Args:
            text: Rendered prompt or prompts containing one image placeholder per image.
            images: PIL images grouped per prompt, or one prompt's flat image list.
            return_tensors: ``"pt"`` for PyTorch token tensors or None for token lists.
            padding: False, True/``"longest"``, or ``"max_length"``; padding is on the right.
            truncation: Whether max_length removes trailing text and complete trailing images.
            max_length: Token limit after image expansion. Cutting through an image raises.
            **kwargs: Hugging Face tokenizer keyword arguments.

        Returns:
            BatchFeature with input_ids/attention_mask/vision_token_types of shape
            [batch, sequence] (or ragged lists when return_tensors=None), plus
            pixel_values [all_patches, 3, patch_size, patch_size] and integer
            image_grid_hws [images, 2] when retained image spans are present.
        """
        if return_tensors not in (None, "pt"):
            raise ValueError("DeepSeek-V4.1 processor supports return_tensors=None or 'pt'")
        texts = [text] if isinstance(text, str) else list(text)
        if not texts:
            raise ValueError("The processor requires at least one text prompt")
        if images is None:
            image_groups = [[] for _ in texts]
        elif isinstance(images, list) and images and isinstance(images[0], list):
            image_groups = images
        elif len(texts) == 1:
            image_groups = [images if isinstance(images, list) else [images]]
        elif isinstance(images, list) and len(images) == len(texts):
            image_groups = [[image] for image in images]
        else:
            raise ValueError("Images must be grouped per text prompt")
        if len(image_groups) != len(texts):
            raise ValueError("The number of image groups must equal the number of text prompts")
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive")
        if padding == "max_length" and max_length is None:
            raise ValueError("padding='max_length' requires max_length")
        if padding not in (False, True, "longest", "max_length"):
            raise ValueError("Unsupported padding mode")
        kwargs.setdefault("add_special_tokens", False)
        tokenized = self.tokenizer(texts, padding=False, **kwargs)
        input_rows, type_rows = [], []
        all_patches, all_grids = [], []
        for raw_ids, sample_images in zip(tokenized["input_ids"], image_groups):
            if raw_ids.count(self.config.image_token_id) != len(sample_images):
                raise ValueError("The number of image placeholder tokens must equal the number of supplied images")
            if sample_images and self.config.vision_config.num_hidden_layers == 0:
                raise ValueError("Image inputs require an enabled vision tower")
            ids, types = [], []
            pending_images = []
            image_iter = iter(sample_images)
            for token_id in raw_ids:
                if token_id != self.config.image_token_id:
                    ids.append(token_id)
                    types.append(TEXT)
                    continue
                patches, grid = _preprocess_image(next(image_iter), self.config.vision_config)
                image_types = (
                    [IMAGE_START] + ([IMAGE] * grid.llm_width + [IMAGE_NEW_LINE]) * grid.llm_height + [IMAGE_END]
                )
                pending_images.append((len(ids), len(ids) + len(image_types), patches, grid))
                ids.extend([self.config.image_token_id] * len(image_types))
                types.extend(image_types)
            if truncation and max_length is not None:
                for start, end, _, _ in pending_images:
                    if start < max_length < end:
                        raise ValueError("max_length truncates a DeepSeek-V4.1 image span")
                ids, types = ids[:max_length], types[:max_length]
            for start, _, patches, grid in pending_images:
                if start < len(ids):
                    all_patches.append(patches)
                    all_grids.append((grid.vit_height, grid.vit_width))
            input_rows.append(ids)
            type_rows.append(types)
        longest = max(map(len, input_rows))
        target_length = max_length if padding == "max_length" else longest
        if target_length < longest:
            raise ValueError("A sequence exceeds the padding length; enable truncation or increase max_length")
        if not padding and return_tensors == "pt" and len({len(row) for row in input_rows}) != 1:
            raise ValueError("Variable-length tensor batches require padding")
        mask_rows = [[1] * len(row) for row in input_rows]
        if padding:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = self.config.pad_token_id
            if pad_id is None:
                raise ValueError("Padding requires a tokenizer or model pad_token_id")
            for ids, types, mask in zip(input_rows, type_rows, mask_rows):
                pad_count = target_length - len(ids)
                ids.extend([pad_id] * pad_count)
                types.extend([TEXT] * pad_count)
                mask.extend([0] * pad_count)
        data = {"input_ids": input_rows, "attention_mask": mask_rows, "vision_token_types": type_rows}
        if all_patches:
            data["pixel_values"] = torch.cat(all_patches)
            data["image_grid_hws"] = torch.tensor(all_grids, dtype=torch.long)
        return BatchFeature(data=data, tensor_type=return_tensors)

    def apply_chat_template(
        self,
        conversation: Sequence[dict[str, Any]] | Sequence[Sequence[dict[str, Any]]],
        *,
        tokenize: bool = False,
        return_dict: bool = False,
        return_tensors: str | None = None,
        add_generation_prompt: bool = True,
        processor_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str | list[str] | list[list[int]] | torch.Tensor | BatchFeature:
        """Render standard chat-mode system/user/assistant messages and local images.

        Args:
            conversation: Hugging Face chat messages or batches thereof, using
                string content or ordered text/image content blocks.
            tokenize: Whether to expand images and tokenize the rendered prompts.
            return_dict: Whether tokenization returns the complete processor batch.
            return_tensors: None or ``"pt"`` for the tokenization result.
            add_generation_prompt: Append the assistant header after a final user
                or mid-conversation system message, matching official chat mode.
            processor_kwargs: Arguments forwarded to the image/token processor.
            **kwargs: Hugging Face chat-template options. Non-chat modes and tools
                raise explicitly; pre-render them with the full official encoder.

        Returns:
            Prompt string(s), token IDs [batch, sequence], or BatchFeature with
            the tensor layouts documented in __call__.
        """
        unsupported = set(kwargs) - {"thinking_mode", "enable_thinking", "tools", "tokenizer_kwargs"}
        if (
            unsupported
            or kwargs.get("thinking_mode", "chat") != "chat"
            or kwargs.get("enable_thinking")
            or kwargs.get("tools")
        ):
            raise ValueError(
                "This chat template supports standard chat mode; pre-render tools/thinking with the official encoder"
            )
        batched = bool(conversation) and isinstance(conversation[0], (list, tuple))
        conversations = conversation if batched else [conversation]
        texts, images = [], []
        for sample in conversations:
            rendered, sample_images = self._render_chat(sample, add_generation_prompt=add_generation_prompt)
            texts.append(rendered)
            images.append(sample_images)
        if not tokenize:
            return texts if batched else texts[0]
        options = dict(processor_kwargs or {})
        options.update(kwargs.get("tokenizer_kwargs") or {})
        result = self(text=texts, images=images, return_tensors=return_tensors, **options)
        return result if return_dict else result["input_ids"]

    def _render_chat(
        self, messages: Sequence[dict[str, Any]], *, add_generation_prompt: bool
    ) -> tuple[str, list[Image.Image]]:
        """Render the standard-chat subset of DeepSeek's released encoding.py."""
        normalized: list[tuple[str, str, bool]] = []
        images = []
        for message in messages:
            if set(message) - {"role", "content", "content_blocks", "wo_eos"}:
                raise ValueError(
                    "Tool, reasoning, and task metadata require pre-rendering with the official V4.1 encoder"
                )
            role = message["role"]
            if role not in ("system", "user", "assistant"):
                raise ValueError(f"Unsupported DeepSeek-V4.1 standard-chat role: {role}")
            content = message.get("content_blocks", message.get("content", ""))
            if isinstance(content, str):
                if IMAGE_PLACEHOLDER in content:
                    raise ValueError("Chat images must be separate content blocks")
                text = content
            else:
                parts = []
                for block in content or []:
                    if block["type"] == "text":
                        if IMAGE_PLACEHOLDER in block["text"]:
                            raise ValueError("Chat images must be separate content blocks")
                        parts.append(block["text"])
                    elif block["type"] in ("image", "image_url"):
                        value = block.get("image", block.get("path", block.get("image_url")))
                        if isinstance(value, dict) and "url" in value:
                            value = value["url"]
                        if value is None:
                            raise ValueError("Image content blocks require an image, path, or image_url")
                        images.append(_load_image(value))
                        parts.append(IMAGE_PLACEHOLDER)
                    else:
                        raise ValueError("Only text and image content blocks are supported by standard chat")
                text = "\n\n".join(parts)
            if role == "user" and normalized and normalized[-1][0] == "user":
                previous = normalized.pop()
                text = previous[1] + "\n\n" + text
            normalized.append((role, text, bool(message.get("wo_eos", False))))
        prompt = _BOS
        for index, (role, content, wo_eos) in enumerate(normalized):
            if role == "system":
                prompt += _SYSTEM + content
            elif role == "user":
                prompt += _USER + content
            else:
                prompt += content + ("" if wo_eos else _EOS)
            next_assistant = index + 1 < len(normalized) and normalized[index + 1][0] == "assistant"
            final_generation = index == len(normalized) - 1 and add_generation_prompt
            if (role == "user" or (role == "system" and index > 0)) and (next_assistant or final_generation):
                prompt += _ASSISTANT
        return prompt, images
