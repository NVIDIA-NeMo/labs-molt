# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for nemo_automodel.components.models.common.packing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.models.common.packing import (
    _passthrough_create_causal_mask,
    _patch_preprocess_mask_arguments_for_packing,
    get_seqlens_in_batch,
    get_unpad_data,
)

# ---------------------------------------------------------------------------
# get_seqlens_in_batch
# ---------------------------------------------------------------------------


class TestGetSeqlensInBatch:
    def test_single_sequence(self):
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        result = get_seqlens_in_batch(mask)
        assert result.tolist() == [3]

    def test_packed_sequences(self):
        mask = torch.tensor([[1, 1, 2, 2, 2, 0]])
        result = get_seqlens_in_batch(mask)
        assert sorted(result.tolist()) == [2, 3]

    def test_no_padding(self):
        mask = torch.tensor([[1, 1, 1]])
        result = get_seqlens_in_batch(mask)
        assert result.tolist() == [3]


# ---------------------------------------------------------------------------
# get_unpad_data
# ---------------------------------------------------------------------------


class TestGetUnpadData:
    def test_basic(self):
        mask = torch.tensor([[1, 1, 0]])
        indices, cu_seqlens, max_seqlen = get_unpad_data(mask)
        assert max_seqlen == 2
        assert cu_seqlens.tolist() == [0, 2]

    def test_packed(self):
        mask = torch.tensor([[1, 1, 2, 2, 0]])
        indices, cu_seqlens, max_seqlen = get_unpad_data(mask)
        assert max_seqlen == 2
        assert indices.tolist() == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# _passthrough_create_causal_mask
# ---------------------------------------------------------------------------


class TestPassthroughCreateCausalMask:
    def test_passthrough_4d_mask(self):
        """4D masks (already block-causal from sdpa collater) are returned as-is."""
        mask = torch.ones(2, 1, 8, 8)
        result = _passthrough_create_causal_mask(attention_mask=mask)
        assert result is mask

    def test_passthrough_indexed_packed_mask(self):
        """Indexed masks with values > 1 (packed sequences) are returned as-is."""
        mask = torch.tensor([[1, 1, 2, 2, 0]])
        result = _passthrough_create_causal_mask(attention_mask=mask)
        assert result is mask

    def test_fa2_passthrough_for_normal_mask(self):
        """FA2 config with normal 2D mask still passes through (FA2 handles masking)."""
        config = SimpleNamespace(_attn_implementation="flash_attention_2")
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        result = _passthrough_create_causal_mask(config=config, attention_mask=mask)
        assert result is mask

    def test_delegates_to_original_for_non_fa2(self):
        """Non-FA2 config with normal 2D mask delegates to HF create_causal_mask."""
        from unittest.mock import patch

        config = SimpleNamespace(_attn_implementation="sdpa")
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        with patch("transformers.masking_utils.create_causal_mask", return_value="delegated") as mock_cm:
            result = _passthrough_create_causal_mask(
                attention_mask=mask,
                config=config,
                inputs_embeds=torch.zeros(1, 5, 64),
                cache_position=torch.arange(5),
            )
        assert result == "delegated"
        mock_cm.assert_called_once()
        assert "inputs_embeds" in mock_cm.call_args.kwargs
        assert "input_embeds" not in mock_cm.call_args.kwargs

    def test_handles_extra_kwargs(self):
        """Extra kwargs don't break — indexed mask still passes through."""
        mask = torch.tensor([[1, 1, 2, 2, 0]])
        result = _passthrough_create_causal_mask(attention_mask=mask, or_mask_function=None, and_mask_function=None)
        assert result is mask


# ---------------------------------------------------------------------------
# get_attn_implementation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# configure_packing
# ---------------------------------------------------------------------------


class TestConfigurePacking:


    def test_fails_when_preprocess_shim_cannot_install(self, monkeypatch):
        """A missing private hook must fail before training can mix packed documents."""
        import transformers.masking_utils as masking_utils

        monkeypatch.setattr(masking_utils, "_nemo_automodel_packing_preprocess_patched", False, raising=False)
        monkeypatch.delattr(masking_utils, "_preprocess_mask_arguments", raising=False)

        with pytest.raises(RuntimeError, match="Cannot enable FA2 neat packing.*_preprocess_mask_arguments"):
            _patch_preprocess_mask_arguments_for_packing()

    def test_fails_on_incompatible_preprocess_result(self, monkeypatch):
        """An incompatible private return contract must fail instead of guessing tuple fields."""
        import transformers.masking_utils as masking_utils

        def incompatible_preprocess(**kwargs):
            """Return a non-early-exit result for the 4D contract probe."""
            return False, kwargs["attention_mask"]

        monkeypatch.setattr(masking_utils, "_nemo_automodel_packing_preprocess_patched", False, raising=False)
        monkeypatch.setattr(masking_utils, "_preprocess_mask_arguments", incompatible_preprocess)

        with pytest.raises(RuntimeError, match="incompatible _preprocess_mask_arguments early-exit result"):
            _patch_preprocess_mask_arguments_for_packing()


class TestConfigurePackingFA3FA4:
    """FA3/FA4 use the same transformers varlen wrapper as FA2 and must be patched alike."""


    @pytest.mark.parametrize("impl", ["flash_attention_3", "flash_attention_4"])
    def test_passthrough_mask_for_fa3_fa4_config(self, impl):
        """_passthrough_create_causal_mask must pass the 2D mask through for any FA version."""
        config = SimpleNamespace(_attn_implementation=impl)
        mask = torch.tensor([[1, 1, 0]])
        out = _passthrough_create_causal_mask(config=config, attention_mask=mask)
        assert out is mask

