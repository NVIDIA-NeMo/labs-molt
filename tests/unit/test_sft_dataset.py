# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SFT dataset: assistant-reply detection for the loss mask, and VLM truncation.

`discover_reply_markers` must locate the assistant reply span from the chat template
alone, with nothing hard-coded per model. These fakes mirror the structure of the real
templates molt serves (verified against the actual tokenizers): ChatML with a last-turn-
only <think> scaffold (Qwen3.6), ChatML with <think> on every turn (Nemotron omni3),
role-specific openers (Kimi), no reply terminator (GLM), alternation-enforced turns +
<end_of_turn> (Gemma), and fullwidth sentinels + eos (DeepSeek).

The second half covers the VLM path: which token ids the truncation guard treats as
image placeholders, and what that means for `--data.max_len`.
"""

from types import SimpleNamespace

import pytest

from molt.datasets.sft_dataset import SFTDataset, discover_reply_markers
from molt.utils import vlm_utils
from molt.utils.vlm_utils import media_token_ids


class _FakeTok:
    """Renders chats via `template` and tokenizes text into atomic special tokens plus
    one id per remaining char — consistently both ways, and reversibly — so the probes
    in discover_reply_markers and the scan in SFTDataset._loss_mask see the same ids."""

    SPECIALS = (
        "<|im_start|>",
        "<|im_end|>",
        "<|im_user|>",
        "<|im_assistant|>",
        "<|im_middle|>",
        "<start_of_turn>",
        "<end_of_turn>",
        "<bos>",
        "[gMASK]",
        "<sop>",
        "<|user|>",
        "<|assistant|>",
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "</think>",
    )

    def __init__(self, template):
        self.template = template
        self._tok2id, self._id2tok = {}, {}

    def _id(self, tok):
        if tok not in self._tok2id:
            i = len(self._tok2id) + 1
            self._tok2id[tok], self._id2tok[i] = i, tok
        return self._tok2id[tok]

    def _split(self, text):
        out, i = [], 0
        while i < len(text):
            sp = next((s for s in self.SPECIALS if text.startswith(s, i)), None)
            out.append(sp or text[i])
            i += len(sp) if sp else 1
        return out

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize  # discover_reply_markers only renders to text
        return self.template(messages, add_generation_prompt)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self._id(t) for t in self._split(text)]}

    def decode(self, ids):
        return "".join(self._id2tok[i] for i in ids)


def _chatml(think):
    """think: 'last' (Qwen3.6), 'always' (omni3/Kimi-style), or 'none'."""

    def render(messages, gen):
        parts = []
        for idx, m in enumerate(messages):
            if m["role"] == "assistant":
                scaffold = think == "always" or (think == "last" and idx == len(messages) - 1)
                body = f"<think></think>{m['content']}" if scaffold else m["content"]
                parts.append(f"<|im_start|>assistant\n{body}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
        s = "".join(parts)
        return (
            s + "<|im_start|>assistant\n<think>"
            if gen and think != "none"
            else s + ("<|im_start|>assistant\n" if gen else "")
        )

    return render


def _kimi(messages, gen):  # role-specific openers + <think> on every assistant turn (like the real Kimi)
    parts = []
    for m in messages:
        body = f"<think></think>{m['content']}" if m["role"] == "assistant" else m["content"]
        parts.append(f"<|im_{m['role']}|>{m['role']}<|im_middle|>{body}<|im_end|>")
    return "".join(parts) + ("<|im_assistant|>assistant<|im_middle|><think>" if gen else "")


def _gemma(messages, gen):
    parts = []
    for idx, m in enumerate(messages):
        if (m["role"] == "user") != (idx % 2 == 0):
            raise ValueError("Conversation roles must alternate user/assistant/user/assistant/...")
        role = "model" if m["role"] == "assistant" else m["role"]
        parts.append(f"<start_of_turn>{role}\n{m['content']}<end_of_turn>\n")
    s = "<bos>" + "".join(parts)
    return s + "<start_of_turn>model\n" if gen else s


def _glm(messages, gen):  # no explicit reply terminator
    s = "[gMASK]<sop>" + "".join(f"<|{m['role']}|>\n{m['content']}" for m in messages)
    return s + "<|assistant|>\n" if gen else s


def _deepseek(messages, gen):
    s = "<｜begin▁of▁sentence｜>"
    for m in messages:
        s += (
            f"<｜User｜>{m['content']}"
            if m["role"] == "user"
            else f"<｜Assistant｜>{m['content']}<｜end▁of▁sentence｜>"
        )
    return s + "<｜Assistant｜>" if gen else s


# (name, template, reply_open, reply_close, supervise_close)
CASES = [
    ("qwen3.6", _chatml("last"), "<|im_start|>assistant\n", "<|im_end|>", True),
    ("omni3", _chatml("always"), "<|im_start|>assistant\n<think>", "<|im_end|>", True),
    ("plain-chatml", _chatml("none"), "<|im_start|>assistant\n", "<|im_end|>", True),
    ("kimi2.6", _kimi, "<|im_assistant|>assistant<|im_middle|><think>", "<|im_end|>", True),
    ("gemma4", _gemma, "<start_of_turn>model\n", "<end_of_turn>", True),
    ("glm4.1", _glm, "<|assistant|>\n", "<|user|>", False),
    ("deepseek4", _deepseek, "<｜Assistant｜>", "<｜end▁of▁sentence｜>", True),
]


@pytest.mark.parametrize("name, template, open_str, close_str, sup", CASES, ids=[c[0] for c in CASES])
def test_discover_reply_markers(name, template, open_str, close_str, sup):
    tok = _FakeTok(template)
    reply_open, reply_close, supervise_close = discover_reply_markers(tok)
    assert tok.decode(reply_open) == open_str
    assert tok.decode([reply_close]) == close_str
    assert supervise_close is sup


@pytest.mark.parametrize("name, template, open_str, close_str, sup", CASES, ids=[c[0] for c in CASES])
def test_loss_mask_supervises_only_replies(name, template, open_str, close_str, sup):
    tok = _FakeTok(template)
    ds = object.__new__(SFTDataset)  # exercise _loss_mask without the heavy __init__
    ds.reply_open, ds.reply_close, ds.supervise_close = discover_reply_markers(tok)
    ds.train_on_last_turn_only = False

    conv = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "four"},
        {"role": "user", "content": "3+3?"},
        {"role": "assistant", "content": "six"},
    ]
    ids = tok(tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False))["input_ids"]
    shifted = ds._loss_mask(ids)  # shifted[t] == 1 iff token t+1 is a reply token
    is_reply = [False] + [shifted[t] == 1.0 for t in range(len(ids) - 1)]
    supervised = tok.decode([ids[t] for t in range(len(ids)) if is_reply[t]])

    # Both replies are supervised; the terminator is included only when it is a real stop token.
    assert "four" in supervised and "six" in supervised
    assert (close_str in supervised) is sup
    # No prompt tokens leak in: the user questions are never supervised.
    assert "2+2?" not in supervised and "3+3?" not in supervised


def test_train_on_last_turn_only_keeps_final_reply():
    tok = _FakeTok(_chatml("none"))
    ds = object.__new__(SFTDataset)
    ds.reply_open, ds.reply_close, ds.supervise_close = discover_reply_markers(tok)
    ds.train_on_last_turn_only = True

    conv = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "last"},
    ]
    ids = tok(tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=False))["input_ids"]
    shifted = ds._loss_mask(ids)
    is_reply = [False] + [shifted[t] == 1.0 for t in range(len(ids) - 1)]
    supervised = tok.decode([ids[t] for t in range(len(ids)) if is_reply[t]])
    assert "last" in supervised and "first" not in supervised


# The truncation guard in _tokenize must never cut into an image's placeholder run,
# so the media-id set must exclude unk and pad ids that can also occur in ordinary text.


class _MediaTok(_FakeTok):
    """Vocabulary with `<image>` but no `<video>`, so `<video>` resolves to unk."""

    SPECIALS = ("<image>",) + _FakeTok.SPECIALS
    UNK_ID, IMAGE_ID, PAD_ID = 0, 18, 1
    TEXT_ID_BASE = 1000  # ordinary text sits well clear of the special ids above

    def __init__(self):
        super().__init__(_chatml("none"))
        self.unk_token_id, self.pad_token_id, self.eos_token_id = self.UNK_ID, self.PAD_ID, 2
        self.chat_template = None
        self._tok2id = {"<unk>": self.UNK_ID, "<image>": self.IMAGE_ID}
        self._id2tok = {i: t for t, i in self._tok2id.items()}
        self._next = self.TEXT_ID_BASE

    def _id(self, tok):
        if tok not in self._tok2id:
            self._tok2id[tok], self._id2tok[self._next] = self._next, tok
            self._next += 1
        return self._tok2id[tok]

    def convert_tokens_to_ids(self, tok):  # never mints a new id, unlike _id
        return self._tok2id.get(tok, self.unk_token_id)


class _OmniProcessor:
    """An image-only AutoProcessor: both ids are set, but from the tokenizer's vocab."""

    def __init__(self):
        self.tokenizer = _MediaTok()
        self.image_processor = object()  # how SFTDataset detects a VLM processor
        self.image_token, self.video_token = "<image>", "<video>"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = self.tokenizer.convert_tokens_to_ids(self.video_token)
        self.unk_token_id, self.pad_token_id = _MediaTok.UNK_ID, _MediaTok.PAD_ID
        self.chat_template = None


class _Rows:
    """Stand-in for the Arrow dataset SFTDataset maps then filters."""

    def __init__(self, rows):
        self._rows = rows
        self.column_names = ["input", "output", "images"]

    def map(self, fn, remove_columns=None, num_proc=None):
        return _Rows([fn(r) for r in self._rows])

    def filter(self, pred):
        return _Rows([r for r in self._rows if pred(r)])

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        return self._rows[idx]


MAX_LEN = 8


def _vlm_dataset():
    processor = _OmniProcessor()
    strategy = SimpleNamespace(
        args=SimpleNamespace(
            data=SimpleNamespace(input_key="input", output_key="output", tokenizer_chat_template=None)
        )
    )
    rows = _Rows([{"input": "<image>\nq", "output": "a", "images": ["img.png"]}])
    return processor, SFTDataset(rows, processor, MAX_LEN, strategy, image_key="images")


def test_media_token_ids_come_from_the_shared_resolver():
    processor, ds = _vlm_dataset()
    # The premise: this processor reports its video token as the unk id.
    assert processor.video_token_id == processor.unk_token_id
    assert ds.media_token_ids == media_token_ids(processor) == {_MediaTok.IMAGE_ID}


def test_truncation_honors_max_len_when_the_video_token_is_unk(monkeypatch):
    # One image placeholder up front, ordinary <unk> text past max_length.
    token_ids = [_MediaTok.IMAGE_ID] + list(range(1000, 1034)) + [_MediaTok.UNK_ID] + list(range(1034, 1038))
    _, ds = _vlm_dataset()
    monkeypatch.setattr(
        vlm_utils,
        "process_prompt_with_images",
        lambda processor, text, images: (token_ids, {"pixel_values": None}, ["img"]),
    )
    kept, _ = ds._tokenize("rendered", ["img.png"])
    assert len(kept) == MAX_LEN


def test_truncation_still_never_splits_a_trailing_image_run(monkeypatch):
    # An image run straddling max_length is kept whole: cutting it would desync
    # pixel_values from the vit embeds.
    run_start = MAX_LEN - 2
    token_ids = list(range(1000, 1000 + run_start)) + [_MediaTok.IMAGE_ID] * 5 + list(range(2000, 2004))
    _, ds = _vlm_dataset()
    monkeypatch.setattr(
        vlm_utils,
        "process_prompt_with_images",
        lambda processor, text, images: (token_ids, {"pixel_values": None}, ["img"]),
    )
    kept, _ = ds._tokenize("rendered", ["img.png"])
    assert len(kept) == run_start + 5
