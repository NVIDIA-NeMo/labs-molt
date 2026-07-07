"""Template-agnostic assistant-reply detection for the SFT loss mask.

`discover_reply_markers` must locate the assistant reply span from the chat template
alone, with nothing hard-coded per model. These fakes mirror the structure of the real
templates molt serves (verified against the actual tokenizers): ChatML with a last-turn-
only <think> scaffold (Qwen3.6), ChatML with <think> on every turn (Nemotron omni3),
role-specific openers (Kimi), no reply terminator (GLM), alternation-enforced turns +
<end_of_turn> (Gemma), and fullwidth sentinels + eos (DeepSeek).
"""

import pytest

from molt.datasets.sft_dataset import SFTDataset, discover_reply_markers


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
