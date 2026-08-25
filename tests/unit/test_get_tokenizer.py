# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from transformers import AutoProcessor

from molt.utils.utils import get_tokenizer


class _BareTokenizer:
    """What AutoProcessor returns for a vision_config model without a full
    processor bundle: the tokenizer itself, with no .tokenizer attribute."""

    def __init__(self):
        self.padding_side = "right"
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = "</s>"
        self.eos_token_id = 2


class _Processor:
    def __init__(self):
        self.tokenizer = _BareTokenizer()


def _get(monkeypatch, returned):
    # Patch the classmethod rather than the module attribute: transformers'
    # lazy module re-resolves attributes on import, which bypasses module-level
    # monkeypatching.
    monkeypatch.setattr(AutoProcessor, "from_pretrained", classmethod(lambda cls, *args, **kwargs: returned))
    model = SimpleNamespace(is_vlm=True, config=SimpleNamespace(pad_token_id=None))
    return get_tokenizer("some/vlm", model, padding_side="left")


def test_vlm_processor_without_tokenizer_attribute_is_used_directly(monkeypatch):
    bare = _BareTokenizer()
    tokenizer = _get(monkeypatch, bare)

    assert tokenizer is bare
    assert tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "</s>"
    assert tokenizer.pad_token_id == 2


def test_vlm_processor_with_inner_tokenizer_mirrors_essentials(monkeypatch):
    processor = _Processor()
    tokenizer = _get(monkeypatch, processor)

    assert tokenizer is processor
    assert processor.tokenizer.padding_side == "left"
    assert tokenizer.pad_token == "</s>"
    assert tokenizer.pad_token_id == 2
