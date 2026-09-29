# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for Qwen3_5MoeBlock.forward packing branches (PR #2147).

The sister file ``test_qwen3_5_moe_model.py`` skips on CPU because it builds
full Qwen3_5MoeBlocks. Here we instantiate via ``__new__`` and stub the heavy
submodules so the packing-kwarg threading path is exercised on any host.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers.models.qwen3_5_moe")

from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeBlock


def _build_block(layer_type: str) -> tuple[Qwen3_5MoeBlock, dict]:
    """Build a barebones Qwen3_5MoeBlock with recorder submodules."""
    block = Qwen3_5MoeBlock.__new__(Qwen3_5MoeBlock)
    nn.Module.__init__(block)
    block.layer_type = layer_type
    block.input_layernorm = nn.Identity()
    block.post_attention_layernorm = nn.Identity()

    recorded: dict = {}

    class _RecorderLinearAttn(nn.Module):
        def forward(self, **kwargs):
            recorded["linear_attn_kwargs"] = kwargs
            return kwargs["hidden_states"]

    class _RecorderSelfAttn(nn.Module):
        def forward(self, **kwargs):
            recorded["self_attn_kwargs"] = kwargs
            return kwargs["hidden_states"]

    block.linear_attn = _RecorderLinearAttn()
    block.self_attn = _RecorderSelfAttn()

    def _mlp(*, x, padding_mask):
        recorded["mlp_padding_mask"] = padding_mask
        return torch.zeros_like(x)

    block._mlp = _mlp
    return block, recorded


class TestQwen3_5MoeBlockForward:
    """Cover Qwen3_5MoeBlock.forward branches without requiring CUDA."""

    def test_full_attention_branch_delegates_to_super(self, monkeypatch):
        """layer_type=full_attention skips packing logic and calls Block.forward."""
        from nemo_automodel.components.models.qwen3_next.model import Block

        block, _ = _build_block("full_attention")
        called: dict = {}

        def _fake_super_forward(self, x, **kwargs):
            called["x_shape"] = tuple(x.shape)
            called["kwargs"] = kwargs
            return x

        monkeypatch.setattr(Block, "forward", _fake_super_forward, raising=True)

        x = torch.zeros(1, 5, 4)
        out = block(
            x,
            freqs_cis=torch.zeros(3, 1, 5, 2),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            padding_mask=None,
            position_ids=torch.arange(5).unsqueeze(0),
        )
        assert called["x_shape"] == tuple(x.shape)
        assert torch.equal(out, x)

    def test_linear_attention_without_packing(self):
        """No indexed mask, no _packed_seq_ids → cu_seqlens/indices stay None."""
        block, recorded = _build_block("linear_attention")
        x = torch.zeros(1, 5, 4)
        # 0/1 mask (not indexed-packed)
        mask = torch.ones(1, 5, dtype=torch.long)
        block(
            x,
            freqs_cis=torch.zeros(3, 1, 5, 2),
            attention_mask=mask,
            padding_mask=None,
            position_ids=torch.arange(5).unsqueeze(0),
        )
        la = recorded["linear_attn_kwargs"]
        assert la["cu_seqlens"] is None
        assert la["indices"] is None
        assert la["attention_mask"] is mask
        # padding_mask was synthesized from the mask (no real padding here).
        assert recorded["mlp_padding_mask"].shape == (1, 5)

    def test_linear_attention_with_indexed_mask(self):
        """attention_mask carrying doc ids -> derived cu_seqlens/indices reach linear_attn."""
        block, recorded = _build_block("linear_attention")
        x = torch.zeros(1, 5, 4)
        indexed = torch.tensor([[1, 1, 2, 2, 2]], dtype=torch.long)
        block(
            x,
            freqs_cis=torch.zeros(3, 1, 5, 2),
            attention_mask=indexed,
            padding_mask=None,
            position_ids=torch.arange(5).unsqueeze(0),
        )
        la = recorded["linear_attn_kwargs"]
        assert la["cu_seqlens"].tolist() == [0, 2, 5]
        assert la["indices"].tolist() == [0, 1, 2, 3, 4]
        # linear_attn_mask reused as the indexed mask, not the 0/1 form.
        assert torch.equal(la["attention_mask"], indexed)

    def test_linear_attention_with_packed_seq_ids_kwarg(self):
        """SDPA case: attention_mask is 4D; indexed mask arrives via _packed_seq_ids."""
        block, recorded = _build_block("linear_attention")
        x = torch.zeros(1, 5, 4)
        sdpa_mask = torch.ones(1, 1, 5, 5, dtype=torch.bool).tril()
        packed_seq_ids = torch.tensor([[1, 1, 2, 2, 2]], dtype=torch.long)
        block(
            x,
            freqs_cis=torch.zeros(3, 1, 5, 2),
            attention_mask=sdpa_mask,
            padding_mask=None,
            position_ids=torch.arange(5).unsqueeze(0),
            _packed_seq_ids=packed_seq_ids,
        )
        la = recorded["linear_attn_kwargs"]
        assert la["cu_seqlens"].tolist() == [0, 2, 5]
        assert la["indices"].tolist() == [0, 1, 2, 3, 4]
        # linear_attn sees the indexed mask, not the 4D SDPA mask.
        assert torch.equal(la["attention_mask"], packed_seq_ids)

    def test_linear_attention_forwards_seq_index(self):
        """seq_index in attn_kwargs is threaded through to linear_attn (CP dense index)."""
        block, recorded = _build_block("linear_attention")
        x = torch.zeros(1, 5, 4)
        seq_index = torch.arange(5).unsqueeze(0)
        block(
            x,
            freqs_cis=torch.zeros(3, 1, 5, 2),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            padding_mask=None,
            position_ids=torch.arange(5).unsqueeze(0),
            seq_index=seq_index,
        )
        assert torch.equal(recorded["linear_attn_kwargs"]["seq_index"], seq_index)

    def test_linear_attention_seq_index_defaults_to_none(self):
        """When seq_index is absent, linear_attn still receives the kwarg as None."""
        block, recorded = _build_block("linear_attention")
        x = torch.zeros(1, 5, 4)
        block(
            x,
            freqs_cis=torch.zeros(3, 1, 5, 2),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            padding_mask=None,
            position_ids=torch.arange(5).unsqueeze(0),
        )
        assert recorded["linear_attn_kwargs"]["seq_index"] is None

    def test_full_attention_strips_seq_index_before_super(self, monkeypatch):
        """Full-attention path must drop seq_index so Block.forward gets no unexpected kwarg."""
        from nemo_automodel.components.models.qwen3_next.model import Block

        block, _ = _build_block("full_attention")
        called: dict = {}

        def _fake_super_forward(self, x, **kwargs):
            called["kwargs"] = kwargs
            return x

        monkeypatch.setattr(Block, "forward", _fake_super_forward, raising=True)

        block(
            torch.zeros(1, 5, 4),
            freqs_cis=torch.zeros(3, 1, 5, 2),
            attention_mask=torch.ones(1, 5, dtype=torch.long),
            padding_mask=None,
            position_ids=torch.arange(5).unsqueeze(0),
            seq_index=torch.arange(5).unsqueeze(0),
        )
        assert "seq_index" not in called["kwargs"], "seq_index must be stripped on the full-attention path"
