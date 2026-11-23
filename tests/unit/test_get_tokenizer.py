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

"""`get_tokenizer` on a model with a vision config.

`AutoProcessor` normally returns a processor that wraps the tokenizer. For a processor class
transformers does not know (Inkling 2) it returns the bare tokenizer, which gets the text setup.
"""

from types import SimpleNamespace

import transformers

import molt.utils.utils as utils


def _tokenizer():
    return SimpleNamespace(pad_token=None, pad_token_id=None, eos_token="</s>", eos_token_id=2, padding_side="right")


def test_bare_tokenizer_from_auto_processor_gets_the_text_setup(monkeypatch):
    monkeypatch.setattr(utils, "is_vlm_model", lambda pretrain: True)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: _tokenizer())
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: _tokenizer())
    tokenizer = utils.get_tokenizer("some/model", model=None)
    assert (tokenizer.padding_side, tokenizer.pad_token, tokenizer.pad_token_id) == ("left", "</s>", 2)


def test_processor_wrapping_a_tokenizer_is_returned_as_the_processor(monkeypatch):
    processor = SimpleNamespace(tokenizer=_tokenizer())
    monkeypatch.setattr(utils, "is_vlm_model", lambda pretrain: True)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", lambda *args, **kwargs: processor)
    tokenizer = utils.get_tokenizer("some/model", model=None)
    assert tokenizer is processor
    assert (processor.tokenizer.padding_side, processor.pad_token, processor.pad_token_id) == ("left", "</s>", 2)
