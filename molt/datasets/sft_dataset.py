# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

import json
import os
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from molt.utils.logging_utils import init_logger
from molt.utils.utils import zero_pad_sequences
from molt.utils.vlm_utils import should_expand_image_placeholder, split_image_placeholder

logger = init_logger(__name__)


def _find_all(ids: List[int], pattern: List[int]) -> List[int]:
    """Every start index where `pattern` occurs contiguously inside `ids`."""
    width = len(pattern)
    return [i for i in range(len(ids) - width + 1) if ids[i : i + width] == pattern]


def _keep_last_run(is_reply: List[bool]) -> List[bool]:
    """Keep only the last contiguous run of reply tokens (the final assistant turn)."""
    if not any(is_reply):
        return is_reply
    end = max(i for i, r in enumerate(is_reply) if r)
    start = end
    while start > 0 and is_reply[start - 1]:
        start -= 1
    return [start <= i <= end for i in range(len(is_reply))]


def _common_len(a: List[int], b: List[int], from_end: bool = False) -> int:
    """Length of the common prefix (or suffix, if from_end) of two token-id lists."""
    n = 0
    while n < len(a) and n < len(b) and (a[-1 - n] == b[-1 - n] if from_end else a[n] == b[n]):
        n += 1
    return n


def discover_reply_markers(tokenizer):
    """Find where assistant replies start and end in this chat template, so SFT
    supervises only the reply tokens. Nothing is hard-coded per model — we just render
    a few throwaway probe chats and diff them. Returns three values:

    * ``reply_open``      — the tokens right before a reply; supervise what comes after.
    * ``reply_close``     — the token that ends a reply.
    * ``supervise_close`` — True if ``reply_close`` is the model's own stop token (so we
      train it to stop), False if it is only the next turn's opener (e.g. GLM, which has
      no reply terminator).

    Works for Qwen3.x / Nemotron (ChatML), Kimi, GLM, Gemma and DeepSeek.
    """

    def encode(messages, generation_prompt=False):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation_prompt)
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    user = [{"role": "user", "content": "x"}]
    reply_a = user + [{"role": "assistant", "content": "aaa"}]
    reply_b = user + [{"role": "assistant", "content": "bbb"}]

    # A reply begins right after the generation prompt. A reasoning template may add a
    # <think> scaffold to the LAST turn only (Qwen3), so keep just the part the generation
    # prompt shares with a real mid-chat reply — that prefix opens every reply.
    after_user = encode(user)
    gen_prompt = encode(user, generation_prompt=True)[len(after_user) :]
    mid_reply = encode(reply_a + [{"role": "user", "content": "z"}])[len(after_user) :]
    reply_open = gen_prompt[: _common_len(gen_prompt, mid_reply)] or gen_prompt

    # A reply ends at whatever the template appends after it. Render the reply as the LAST
    # turn: a real stop token then sits at the very end (train it); GLM appends nothing, so
    # we fall back to the next turn's opener, which only bounds the reply.
    ids_a, ids_b = encode(reply_a), encode(reply_b)
    terminator = ids_a[len(ids_a) - _common_len(ids_a, ids_b, from_end=True) :]
    if terminator:
        reply_close, supervise_close = terminator[0], True
    else:
        full = encode(reply_a + [{"role": "user", "content": "z"}])
        reply_close, supervise_close = full[len(ids_a)], False

    if not reply_open:
        raise ValueError("SFTDataset: could not find the assistant reply opener in the chat template.")
    return reply_open, reply_close, supervise_close


class SFTDataset(Dataset):
    """SFT dataset for chat models (text or VLM).

    Each sample is a chat conversation. We render it with the model's chat
    template, tokenize it, and supervise only the assistant's generated tokens
    (the reply, any reasoning block, and the turn terminator). System, user, and
    image tokens are masked out.

    Assistant turns are found by scanning the token ids for the reply opener and
    terminator, both *discovered from the chat template* (discover_reply_markers) and
    matched by *token id* — robust to BPE merges and not tied to any one template, so
    Qwen3.x / Nemotron (ChatML), Kimi, GLM, Gemma and DeepSeek all work unchanged.
    Because the scan runs on the final (already image-expanded) ids, VLM needs no
    special handling.

    Set ``train_on_last_turn_only`` to supervise only the final assistant turn
    (per-turn-flattened data, where earlier assistant turns are context).
    """

    def __init__(
        self,
        dataset,
        tokenizer: Callable,
        max_length: int,
        strategy,
        num_processors: int = 8,
        image_key: Optional[str] = None,
        max_images_per_prompt: int = 4,
        train_on_last_turn_only: bool = False,
    ) -> None:
        super().__init__()
        # An AutoProcessor (VLM) keeps the plain text tokenizer under `.tokenizer`.
        self.tokenizer = tokenizer
        self.text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        self.processor = tokenizer if hasattr(tokenizer, "image_processor") else None
        self.max_length = max_length
        self.image_key = image_key
        self.max_images_per_prompt = max_images_per_prompt
        self.train_on_last_turn_only = train_on_last_turn_only
        # Qwen-VL needs a "<image>" string rewritten to structured content; Nemotron-Omni
        # renders the literal "<image>". Auto-detected from the tokenizer (same rule as RL).
        self.expand_image_placeholder = should_expand_image_placeholder(tokenizer)
        # Image/video placeholder token ids. Truncation must never cut into an
        # image's placeholder run: pixel_values / image_grid_thw count every image
        # in full, so dropping placeholder tokens desyncs them and the model
        # forward crashes with a vit-embed shape mismatch. Collected from the
        # processor (same attrs as vlm_utils.estimate_vllm_input_expansion_delta).
        self.media_token_ids = set()
        unk_id = getattr(self.text_tokenizer, "unk_token_id", None)
        if self.processor is not None:
            for obj in (self.processor, self.text_tokenizer):
                for attr in ("image_token_id", "video_token_id", "img_context_token_id"):
                    tid = getattr(obj, attr, None)
                    if tid is not None:
                        self.media_token_ids.add(int(tid))
                # Fallback: some models (e.g. Nemotron-Omni) keep the placeholder
                # id on the model config and expose only the token *string* on the
                # processor. Resolve the string to an id so truncation still
                # protects the run — mirrors the processor's own convert_tokens_to_ids.
                for attr in ("image_token", "video_token"):
                    tokstr = getattr(obj, attr, None)
                    if tokstr:
                        tid = self.text_tokenizer.convert_tokens_to_ids(tokstr)
                        if tid is not None and tid != unk_id:
                            self.media_token_ids.add(int(tid))
        # A VLM run with no resolvable placeholder id would let truncation cut into
        # the image's placeholder tokens and silently desync pixel_values from the
        # vit embeds (a shape mismatch the model forward tolerates by slicing).
        if self.image_key and self.processor is not None and not self.media_token_ids:
            raise ValueError(
                "VLM SFT: no image/video placeholder token id resolvable from the processor "
                "or tokenizer (checked *_token_id attrs and image_token/video_token strings); "
                "truncation cannot protect image placeholders."
            )
        self.pad_token_id = self.text_tokenizer.pad_token_id
        if self.pad_token_id is None:  # not `or`: a real pad id can be 0
            self.pad_token_id = self.text_tokenizer.eos_token_id
        if image_key and self.processor is None:
            raise ValueError("--data.image_key needs an AutoProcessor (must expose .image_processor).")

        data_cfg = strategy.args.data
        self.input_key = data_cfg.input_key
        self.output_key = data_cfg.output_key
        self._maybe_override_chat_template(getattr(data_cfg, "tokenizer_chat_template", None))

        # Assistant-reply span markers, discovered from the (possibly overridden) chat
        # template so the loss mask works for any model — see discover_reply_markers.
        self.reply_open, self.reply_close, self.supervise_close = discover_reply_markers(self.text_tokenizer)

        # Build the chat conversation for every row once via dataset.map (Arrow-backed +
        # multiprocessed). Stored as a JSON string since HF datasets can't hold a raw
        # list-of-dicts column. KEEP the mapped/filtered Arrow dataset (memory-mapped) and index
        # it lazily in __getitem__ — pulling rows["conversation"]/rows["images"] into Python lists
        # materialized the whole corpus in RAM and OOMs on large SFT datasets.
        rows = dataset.map(self._build_row, remove_columns=dataset.column_names, num_proc=num_processors)
        self.rows = rows.filter(lambda r: r["conversation"] is not None)
        self._has_images = image_key is not None

    def _maybe_override_chat_template(self, override: Optional[str]) -> None:
        """--data.tokenizer_chat_template: a .jinja file path or a literal template."""
        if not override:
            return
        if os.path.exists(override):
            with open(override, encoding="utf-8") as f:
                override = f.read()
        self.text_tokenizer.chat_template = override
        if self.tokenizer is not self.text_tokenizer:
            self.tokenizer.chat_template = override

    # ------------------------------------------------------------------
    # Per-row preprocessing (runs inside dataset.map): build the conversation.
    # ------------------------------------------------------------------
    def _build_row(self, row) -> dict:
        images = row.get(self.image_key) if self.image_key else None
        if images is not None and not isinstance(images, list):
            images = [images]
        if images is not None and len(images) > self.max_images_per_prompt:
            return {"conversation": None, "images": None}  # dropped by .filter below

        messages = self._to_messages(row)
        if messages is None:
            return {"conversation": None, "images": None}
        return {"conversation": json.dumps(messages), "images": images if self.image_key else None}

    def _to_messages(self, row) -> Optional[List[dict]]:
        """Assemble a [{role, content}, ...] conversation, or None to drop the row.

        ``input_key`` holds the prompt: a list of messages or a bare user string.
        ``output_key`` (when set) holds the reply: a list of messages, one message
        dict, or a bare assistant string.
        """
        prompt = row[self.input_key]
        reply = row.get(self.output_key) if self.output_key else None
        messages = list(prompt) if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        if isinstance(reply, list):
            messages += reply
        elif isinstance(reply, dict):
            messages.append(reply)
        elif reply is not None:
            messages.append({"role": "assistant", "content": reply})
        if not any(m.get("role") == "assistant" for m in messages):
            return None  # nothing to train on
        if self.expand_image_placeholder:
            messages = [split_image_placeholder(m) for m in messages]
        return messages

    # ------------------------------------------------------------------
    # Sample access.
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx):
        # Lazy: read this row from the mmap'd Arrow dataset and render on demand — no
        # RAM-resident conversations/image_refs lists.
        row = self.rows[idx]
        messages = json.loads(row["conversation"])
        images = row["images"] if self._has_images else None
        text = self.text_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        token_ids, mm_inputs = self._tokenize(text, images)
        loss_mask = self._loss_mask(token_ids)
        if not any(loss_mask):
            # No supervised tokens — the sample contributes zero loss. Most often
            # this is VLM over-length truncation in _tokenize keeping only the
            # tokens up to the last image placeholder (dropping the assistant
            # reply). Surface it instead of silently training on nothing.
            logger.warning(
                f"SFT sample {idx} has an all-zero loss mask "
                f"(len={len(token_ids)}, max_length={self.max_length}, images={bool(images)}); "
                "it will contribute no loss. Likely over-length truncation dropped the assistant reply — "
                "raise --data.max_len or shorten the sample."
            )
        return (
            torch.tensor([token_ids], dtype=torch.long),
            torch.ones(1, len(token_ids), dtype=torch.long),
            torch.tensor([loss_mask], dtype=torch.float32),
            mm_inputs,
        )

    def _tokenize(self, text: str, images):
        """Rendered text -> token ids (capped at max_length). VLM expands the
        ``<image>`` markers and also returns pixel tensors."""
        if images:
            from molt.utils.vlm_utils import process_prompt_with_images

            token_ids, mm_inputs, _ = process_prompt_with_images(self.processor, text, images)
            if len(token_ids) > self.max_length:
                # Right-truncate to max_length, but never before the last image
                # placeholder token (cutting placeholders would desync mm_inputs;
                # see media_token_ids). Samples whose images alone exceed
                # max_length stay whole — rare, and bounded by max_images_per_prompt.
                # keep_len normally == max_length, but may be extended past it to
                # cover a trailing image placeholder run (never split mid-image).
                keep_len = self.max_length
                if self.media_token_ids:
                    last_media = max((i for i, t in enumerate(token_ids) if t in self.media_token_ids), default=-1)
                    keep_len = max(keep_len, last_media + 1)
                token_ids = token_ids[:keep_len]
            return token_ids, mm_inputs
        token_ids = self.text_tokenizer(text, add_special_tokens=False, truncation=True, max_length=self.max_length)[
            "input_ids"
        ]
        return token_ids, None

    def _loss_mask(self, ids: List[int]) -> List[float]:
        """1.0 on the tokens the model is trained to predict (assistant replies), else 0.0."""
        n = len(ids)
        is_reply = [False] * n  # is_reply[i]: token i belongs to an assistant reply
        for start in _find_all(ids, self.reply_open):
            i = start + len(self.reply_open)
            while i < n and ids[i] != self.reply_close:
                is_reply[i] = True
                i += 1
            if i < n and self.supervise_close:
                is_reply[i] = True  # include the turn terminator so the model learns to stop

        if self.train_on_last_turn_only:
            is_reply = _keep_last_run(is_reply)

        # Next-token shift: the trainer scores its prediction at position t against
        # token t+1, so token t is supervised when token t+1 is a reply token.
        return [1.0 if (t + 1 < n and is_reply[t + 1]) else 0.0 for t in range(n)]

    # ------------------------------------------------------------------
    # Batching.
    # ------------------------------------------------------------------
    def collate_fn(self, items):
        input_ids, attention_mask, loss_mask, mm_inputs = zip(*items)
        return (
            zero_pad_sequences(list(input_ids), "right", self.pad_token_id),
            zero_pad_sequences(list(attention_mask), "right"),
            zero_pad_sequences(list(loss_mask), "right"),
            self._stack_mm_inputs(mm_inputs),
        )

    @staticmethod
    def _stack_mm_inputs(mm_inputs) -> Dict[str, torch.Tensor]:
        """Concatenate per-sample VLM tensors (pixel_values, image_grid_thw, ...)
        along dim 0. Returns {} for a text-only batch."""
        dicts = [m for m in mm_inputs if m]
        if not dicts:
            return {}
        out: Dict[str, torch.Tensor] = {}
        for key in set().union(*(m.keys() for m in dicts)):
            tensors = [m[key] for m in dicts if torch.is_tensor(m.get(key))]
            if tensors:
                out[key] = torch.cat(tensors, dim=0)
        return out
