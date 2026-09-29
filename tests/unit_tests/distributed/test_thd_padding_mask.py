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

"""The packed THD padding mask must come from the pack layout, not token values."""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.distributed.thd_utils import (
    process_input_for_thd,
    thd_padding_mask_from_token_ids,
)

# GLM-5.2 sets pad_token_id to <|endoftext|>, which is also its first
# eos_token_id, so the pad id is a legitimate content token.
PAD_AND_EOS = 154820


def _batch(input_ids, seq_lens, seq_lens_padded):
    ids = torch.tensor([input_ids])
    return {
        "input_ids": ids,
        "labels": ids.clone(),
        "position_ids": torch.arange(ids.shape[1]).unsqueeze(0),
        "seq_lens": torch.tensor([seq_lens]),
        "seq_lens_padded": torch.tensor([seq_lens_padded]),
    }


def test_pad_id_that_is_also_eos_is_not_masked():
    # Two 3-token documents each ending in eos, padded to 4 slots.
    stream = [7, 8, PAD_AND_EOS, PAD_AND_EOS, 9, 10, PAD_AND_EOS, PAD_AND_EOS]
    out = process_input_for_thd(
        _batch(stream, seq_lens=[3, 3], seq_lens_padded=[4, 4]),
        padding_token_id=PAD_AND_EOS,
    )

    # Only the 4th slot of each pack is padding; the document-final eos at index
    # 2 and 6 is a real token and must stay visible to the MoE experts.
    assert out["padding_mask"].tolist() == [False, False, False, True, False, False, False, True]


def test_content_token_equal_to_default_pad_id_is_not_masked():
    # Token id 0 is '!' in the GLM tokenizer; the sharder's default
    # padding_token_id is 0, so a value comparison would drop every '!'.
    stream = [5, 0, 6, 99, 7, 8, 9, 99]
    out = process_input_for_thd(
        _batch(stream, seq_lens=[3, 3], seq_lens_padded=[4, 4]),
        padding_token_id=0,
    )

    assert out["padding_mask"].tolist() == [False, False, False, True, False, False, False, True]


def test_trailing_pack_pad_is_masked():
    # Single 3-token document in a stream padded out to 6 slots.
    out = process_input_for_thd(
        _batch([1, 2, 3, 0, 0, 0], seq_lens=[3], seq_lens_padded=[3]),
        padding_token_id=0,
    )

    assert out["padding_mask"].tolist() == [False, False, False, True, True, True]


def test_seq_lens_none_uses_token_value_fallback():
    ids = torch.tensor([[5, 6, 0, 0]])
    out = process_input_for_thd(
        {
            "input_ids": ids,
            "labels": ids.clone(),
            "position_ids": torch.arange(ids.shape[1]).unsqueeze(0),
            "seq_lens": None,
            "seq_lens_padded": None,
        },
        padding_token_id=0,
    )

    assert out["padding_mask"].tolist() == [False, False, True, True]


def test_token_value_fallback_rejects_a_pad_id_used_as_content():
    # The metadata-free fallback: a pad id that also appears as content cannot
    # yield a right-padded run, so it raises instead of masking real tokens.
    with pytest.raises(ValueError, match="also occurs as content"):
        thd_padding_mask_from_token_ids(torch.tensor([5, 0, 6, 0]), padding_token_id=0)


def test_token_value_fallback_accepts_a_right_padded_run():
    mask = thd_padding_mask_from_token_ids(torch.tensor([5, 6, 0, 0]), padding_token_id=0)

    assert mask.tolist() == [False, False, True, True]
