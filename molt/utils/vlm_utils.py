# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

"""VLM utilities: image loading, processor-based tokenization, multimodal tensor merging."""

import base64
import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def split_image_placeholder(message: dict) -> dict:
    """User-content string `"<image>\\nproblem"` → structured
    `[{"type":"image"}, {"type":"text", "text":"problem"}]` so chat templates
    that iterate structured content (Qwen2-VL / Qwen3-VL) render correctly.
    Strings with no ``<image>`` token pass through unchanged.
    """
    content = message.get("content")
    if not isinstance(content, str) or "<image>" not in content:
        return message
    parts = []
    remaining = content
    while "<image>" in remaining:
        before, _, remaining = remaining.partition("<image>")
        if before:
            parts.append({"type": "text", "text": before})
        parts.append({"type": "image"})
    if remaining:
        parts.append({"type": "text", "text": remaining})
    return {**message, "content": parts}


def should_expand_image_placeholder(tokenizer) -> bool:
    """Whether stored `"<image>"` strings must be split into structured content.

    A model's chat template emits its image token in one of two ways:
      * **Structured-content branch** (Qwen2-VL / Qwen3-VL): the template
        iterates `[{"type": "image"}, {"type": "text", ...}]` and emits a
        model-specific token (e.g. `<|image_pad|>`). The literal `<image>` the
        dataset stores would never be expanded — so it must be split first.
      * **Literal branch** (Nemotron-Omni): the template renders the stored
        `<image>` verbatim, so splitting it would break the placeholder.

    We detect this from the tokenizer/processor's ``image_token``: anything
    other than the stored `<image>` means the structured branch is required.
    Shared by SFT (`SFTDataset`) and RL (`PromptDataset`) so the two never
    disagree for the same model.
    """
    return getattr(tokenizer, "image_token", "<image>") != "<image>"


def _pad_to_common_hw(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """Right/bottom-pad a list of tensors so they share a common (H, W).

    Each tensor's last two dims are treated as spatial. Zero-padded. Used to
    stack/concat variable-resolution image patches into a single batch tensor.
    """
    max_h = max(int(t.shape[-2]) for t in tensors)
    max_w = max(int(t.shape[-1]) for t in tensors)
    return [F.pad(t, (0, max_w - int(t.shape[-1]), 0, max_h - int(t.shape[-2]))) for t in tensors]


def load_images(image_refs: Union[str, List[str], Image.Image, List[Any]]) -> List[Image.Image]:
    """Load PIL images from paths, URLs, base64 strings, raw bytes, or PIL objects.

    Invalid entries are skipped with a warning. Output is always RGB —
    Qwen3-style processors reject RGBA ("Unable to infer channel dimension
    format") or grayscale.
    """
    if image_refs is None:
        return []
    if not isinstance(image_refs, list):
        image_refs = [image_refs]

    pil_images = []
    for img in image_refs:
        try:
            if isinstance(img, Image.Image):
                loaded = img
            elif isinstance(img, bytes):
                loaded = Image.open(io.BytesIO(img))
            elif isinstance(img, dict):
                # HuggingFace datasets Image() feature serializes to
                # {"bytes": <png-bytes>, "path": <str|None>} on save_to_disk.
                if img.get("bytes") is not None:
                    loaded = Image.open(io.BytesIO(img["bytes"]))
                elif img.get("path"):
                    loaded = Image.open(img["path"])
                else:
                    logger.warning(f"Skipping image dict with no bytes/path: {img!r}")
                    continue
            elif isinstance(img, str):
                if img.startswith(("http://", "https://")):
                    import requests

                    loaded = Image.open(io.BytesIO(requests.get(img, timeout=30).content))
                elif img.startswith("file://"):
                    loaded = Image.open(img[len("file://") :])
                elif img.startswith("data:image") or (
                    len(img) > 256 and re.fullmatch(r"[A-Za-z0-9+/\n\r]+=*", img[:512])
                ):
                    raw = img.split(",", 1)[-1] if img.startswith("data:") else img
                    loaded = Image.open(io.BytesIO(base64.b64decode(raw)))
                else:
                    loaded = Image.open(img)
            else:
                logger.warning(f"Skipping unsupported image type: {type(img)}")
                continue
            if loaded.mode != "RGB":
                loaded = loaded.convert("RGB")
            pil_images.append(loaded)
        except Exception as e:
            logger.warning(f"Failed to load image {img!r}: {e}")
    return pil_images


def estimate_vllm_input_expansion_delta(
    processor,
    token_ids: List[int],
    mm_train_inputs: Optional[Dict],
    pil_images: Optional[List[Image.Image]],
) -> int:
    """Estimate extra vLLM multimodal tokens beyond what's already in ``token_ids``.

    Two processor patterns:
      * **Per-grid expansion** (Qwen2.5-VL / Qwen3-VL): mm_train_inputs carries
        ``image_grid_thw``. Each grid row ``[t, h, w]`` is in *patch* units, and
        the spatial-merge kernel folds ``merge_size×merge_size`` patches into one
        token, so the per-image token count is ``prod(grid) // merge_size**2``
        (matches Automodel's ``image_grid_thw.prod(-1) // spatial_merge_size**2``
        in qwen3_vl_moe and ``_estimate_media_tokens``). The delta is that token
        count minus the media tokens already present in ``token_ids``.
      * **Pre-expanded**: ``token_ids`` already contains the full per-vit-token
        sequence. vLLM's dedup→re-expand round-trip is supposed to leave length
        unchanged, but processor variance can add a handful of tokens — reserve
        1024/image as a margin.
    """
    if not pil_images or not mm_train_inputs:
        return 0

    # Grid rows are in patch units; convert to token units with the processor's
    # spatial-merge factor. Default to 1 (no division) when the processor does
    # not expose ``merge_size`` — a best-effort estimate for unknown processors.
    image_processor = getattr(processor, "image_processor", None)
    merge_size = int(getattr(image_processor, "merge_size", 1) or 1)
    merge_area = max(1, merge_size * merge_size)

    grid_total = 0
    for key in ("image_grid_thw", "video_grid_thw"):
        raw = mm_train_inputs.get(key)
        if raw is None:
            continue
        if not torch.is_tensor(raw):
            try:
                raw = torch.as_tensor(raw)
            except Exception:
                continue
        if raw.numel() == 0 or raw.shape[-1] < 3:
            continue
        per_image_patches = raw.reshape(-1, raw.shape[-1])[..., -3:].long().prod(dim=-1)
        grid_total += int((per_image_patches // merge_area).sum().item())

    if grid_total == 0:
        # Pre-expanded processor (no grid_thw): tokens already include vLLM's
        # multimodal expansion. The vllm_engine still dedups + re-expands at
        # generate-time (vllm_engine.py:145), and the round-trip variance can
        # be hundreds of tokens for high-resolution ViT encoders. Use a
        # 1024/image upper bound — safe for current generation VLMs without
        # needing per-model tuning.
        return 1024 * len(pil_images)

    media_ids = set()
    for obj in (processor, getattr(processor, "tokenizer", None)):
        if obj is None:
            continue
        for attr in ("image_token_id", "video_token_id", "img_context_token_id"):
            tid = getattr(obj, attr, None)
            if tid is not None:
                media_ids.add(int(tid))
    existing = sum(1 for tid in token_ids if int(tid) in media_ids) if media_ids else 0
    return max(0, grid_total - existing)


def process_prompt_with_images(
    processor, prompt: str, images: Any
) -> Tuple[List[int], Optional[Dict], List[Image.Image]]:
    """Tokenize a prompt with images using a VLM processor (AutoProcessor).

    Returns:
        (token_ids, mm_train_inputs, pil_images)
        - mm_train_inputs: dict of multimodal tensors (pixel_values, image_grid_thw, ...)
          or None when no images are present.
        - pil_images: loaded PIL images (reused for vLLM multi_modal_data).
    """
    pil_images = load_images(images)
    refs = images if isinstance(images, list) else ([images] if images is not None else [])
    non_none_refs = [r for r in refs if r is not None]

    # No images: text-only tokenization (valid for text-only samples in mixed datasets).
    if not pil_images:
        if non_none_refs:
            # Caller provided real refs but none loaded — falling through to
            # text-only would leave placeholder tokens with no pixel_values.
            raise ValueError(
                f"All images failed to load ({images!r}). The prompt likely "
                "contains image placeholder tokens that require pixel_values."
            )
        token_ids = processor(text=prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        return token_ids, None, []

    # Warn on partial load failures: prompt may expect more images than were
    # loaded, which would misalign placeholder tokens.
    if len(pil_images) < len(non_none_refs):
        logger.warning(
            f"Only {len(pil_images)}/{len(non_none_refs)} images loaded successfully. "
            "Image placeholder tokens in the prompt may not match pixel_values."
        )

    proc_out = processor(text=[prompt], images=pil_images, add_special_tokens=False, return_tensors="pt")
    token_ids = proc_out["input_ids"][0].tolist()

    # Drop sequence-length-dependent fields (input_ids/attention_mask/token_type_ids):
    # they get reconstructed from input_ids during training, where the sequence
    # also includes the response (the processor only saw the prompt).
    _skip_keys = {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}
    mm_train_inputs = {k: v for k, v in proc_out.items() if k not in _skip_keys}
    return token_ids, (mm_train_inputs or None), pil_images


def accumulate_mm_inputs(existing: Optional[Dict], new: Optional[Dict]) -> Optional[Dict]:
    """Merge a new step's multimodal tensors into the running accumulator.

    Keys present in only one dict are preserved; keys in both are concatenated
    along dim=0.
    """
    if new is None:
        return existing
    if existing is None:
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in new.items()}
    merged = {}
    for k in set(existing) | set(new):
        if k in existing and k in new:
            a, b = existing[k], new[k]
            # Variable-resolution pixel_values across agent steps (some VLM
            # processors return 4D (N, C, H, W) with per-call H/W) — pad before cat.
            if k == "pixel_values" and torch.is_tensor(a) and torch.is_tensor(b) and a.ndim == 4 and b.ndim == 4:
                a, b = _pad_to_common_hw([a, b])
            merged[k] = torch.cat([a, b], dim=0)
        elif k in existing:
            merged[k] = existing[k]
        else:
            merged[k] = new[k]
    return merged


def merge_mm_train_inputs(mm_train_inputs_list: list, device) -> Dict[str, torch.Tensor]:
    """Merge per-sample multimodal tensor dicts into one batched dict on *device*.

    Each ``mm_train_inputs_list`` element is a per-sample dict (or list of dicts,
    or None). Tensors are concatenated along dim=0; pixel_values is padded to a
    common HxW first when entries have ndim==4.
    """
    merged: Dict[str, list] = {}
    for item in mm_train_inputs_list:
        for mm_dict in (item if isinstance(item, list) else [item]):
            if mm_dict is None:
                continue
            for key, val in mm_dict.items():
                merged.setdefault(key, []).append(val if isinstance(val, torch.Tensor) else torch.tensor(val))

    output = {}
    for key, values in merged.items():
        if key == "pixel_values" and all(torch.is_tensor(v) and v.ndim == 4 for v in values):
            values = _pad_to_common_hw(values)
        output[key] = torch.cat(values, dim=0).to(device)
    return output
