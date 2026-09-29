# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Tests for freeze_deepseek_v4_indexer_params.

The DSV4 compressor's DeepseekV4Indexer feeds its outputs through
``topk(...).indices``, which is non-differentiable. No gradient ever reaches
the indexer's wkv / wgate / ape_param / wq_b / weights_proj params, so AdamW
never allocates lazy state slots for them, and DCP checkpoint resume fails
with ``RuntimeError: Missing key in checkpoint state_dict:
optim.state.model.layers.{i}.self_attn.compressor.indexer.wkv.weight.step.``

Uses a lightweight mock model so these tests run on CPU without the full
DeepSeek V4 transformers dependency.
"""

import os
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn

from nemo_automodel.components.utils.model_utils import freeze_deepseek_v4_indexer_params


# ---------------------------------------------------------------------------
# Helpers: minimal model that mimics DSV4 compressor.indexer layer naming
# ---------------------------------------------------------------------------
class _FakeIndexer(nn.Module):
    """Mimics DeepseekV4Indexer's five learnable submodules."""

    def __init__(self, dim: int):
        super().__init__()
        self.wkv = nn.Linear(dim, dim, bias=False)
        self.wgate = nn.Linear(dim, dim, bias=False)
        self.ape_param = nn.Linear(dim, dim, bias=False)
        self.wq_b = nn.Linear(dim, dim, bias=False)
        self.weights_proj = nn.Linear(dim, dim, bias=False)


class _FakeCompressor(nn.Module):
    """Mimics DeepseekV4Compressor; only layers with compress_ratio==4 have an indexer."""

    def __init__(self, dim: int, has_indexer: bool):
        super().__init__()
        self.wkv = nn.Linear(dim, dim, bias=False)
        self.indexer = _FakeIndexer(dim) if has_indexer else None


class _FakeAttention(nn.Module):
    """Mimics DSV4Attention with q_proj/o_proj + optional compressor."""

    def __init__(self, dim: int, has_compressor: bool):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.compressor = _FakeCompressor(dim, has_indexer=has_compressor) if has_compressor else None


class _FakeDecoderLayer(nn.Module):
    def __init__(self, dim: int, has_compressor: bool):
        super().__init__()
        self.self_attn = _FakeAttention(dim, has_compressor=has_compressor)
        self.mlp = nn.Linear(dim, dim)


class _FakeModel(nn.Module):
    """Minimal model with ``config.model_type`` and a few layers."""

    def __init__(self, compress_ratios=(0, 0, 4, 128), dim: int = 16, model_type: str = "deepseek_v4"):
        super().__init__()
        self.config = SimpleNamespace(model_type=model_type)
        self.layers = nn.ModuleList([_FakeDecoderLayer(dim, has_compressor=(cr == 4)) for cr in compress_ratios])

    def forward(self, x):
        for layer in self.layers:
            x = layer.mlp(layer.self_attn.q_proj(x))
        return x


def _indexer_param_names(model):
    """Return the set of parameter names that should be frozen."""
    return {n for n, _ in model.named_parameters() if ".self_attn.compressor.indexer." in n}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFreezeDeepseekV4IndexerParams:
    """Unit tests for freeze_deepseek_v4_indexer_params."""

    def test_noop_when_not_deepseek_v4(self):
        """Models whose model_type is not 'deepseek_v4' are untouched."""
        model = _FakeModel(compress_ratios=(0, 0, 4, 128), model_type="llama")
        grad_before = {n: p.requires_grad for n, p in model.named_parameters()}

        freeze_deepseek_v4_indexer_params(model)

        grad_after = {n: p.requires_grad for n, p in model.named_parameters()}
        assert grad_before == grad_after

    def test_noop_when_config_missing(self):
        """Plain nn.Module without a config is silently skipped."""
        model = nn.Linear(4, 4)
        freeze_deepseek_v4_indexer_params(model)
        assert model.weight.requires_grad is True

    def test_indexer_params_frozen(self):
        """All compressor.indexer params have requires_grad=False after freeze."""
        model = _FakeModel(compress_ratios=(0, 0, 4, 128))
        indexer_names = _indexer_param_names(model)
        assert len(indexer_names) > 0, "Sanity: there should be indexer params"

        freeze_deepseek_v4_indexer_params(model)

        for name, param in model.named_parameters():
            if name in indexer_names:
                assert not param.requires_grad, f"{name} should be frozen"
            else:
                assert param.requires_grad, f"{name} should remain trainable"

    def test_non_indexer_compressor_params_untouched(self):
        """compressor.wkv (not under indexer) stays trainable."""
        model = _FakeModel(compress_ratios=(0, 0, 4, 128))

        freeze_deepseek_v4_indexer_params(model)

        for name, param in model.named_parameters():
            if "compressor.wkv" in name and ".indexer." not in name:
                assert param.requires_grad, f"Non-indexer compressor param {name} must stay trainable"

    def test_q_proj_and_o_proj_stay_trainable(self):
        """Attention q_proj and o_proj remain trainable on indexer-bearing layers."""
        model = _FakeModel(compress_ratios=(0, 0, 4, 128))

        freeze_deepseek_v4_indexer_params(model)

        for name, param in model.named_parameters():
            if "q_proj" in name or "o_proj" in name:
                assert param.requires_grad, f"{name} should remain trainable"

    def test_no_indexer_when_no_compress_ratio_4(self):
        """Models with no compress_ratio==4 layers have no indexer params to freeze."""
        model = _FakeModel(compress_ratios=(0, 0, 0, 0))
        assert len(_indexer_param_names(model)) == 0

        freeze_deepseek_v4_indexer_params(model)

        for _, param in model.named_parameters():
            assert param.requires_grad


class TestCheckpointResumeConsistency:
    """Reproduces the DCP missing-key bug and verifies the fix."""

    def test_bug_reproduction_optimizer_tracks_indexer_params(self):
        """WITHOUT fix: indexer params land in the optimizer but get no state.

        The optimizer's param_groups include the indexer params, but AdamW's
        per-param state dict (step / exp_avg / exp_avg_sq) is only populated
        for params that received a gradient. Indexer params never get one,
        so on checkpoint resume the DCP loader complains about a missing
        ``optim.state.*compressor.indexer.*.step`` key.
        """
        model = _FakeModel(compress_ratios=(0, 0, 4, 128))
        indexer_names = _indexer_param_names(model)

        # Do NOT freeze -> all params land in the optimizer
        all_trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(all_trainable, lr=1e-3)

        # Forward/backward only touches q_proj + mlp (mock forward), so indexer
        # params receive no gradient.
        x = torch.randn(2, 16, device="cpu")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        indexer_param_ids = {id(p) for n, p in model.named_parameters() if n in indexer_names}
        params_with_state = {id(p) for p in optimizer.state if len(optimizer.state[p]) > 0}

        # Indexer params are in the optimizer's param_groups
        optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert indexer_param_ids.issubset(optimizer_param_ids), (
            "Without the fix, indexer params ARE in the optimizer param groups"
        )

        # But they have no state (Adam state is lazily initialized on first grad)
        indexer_with_state = indexer_param_ids & params_with_state
        assert len(indexer_with_state) == 0, (
            "Indexer params should have no optimizer state (they got no gradients), "
            "but on checkpoint resume the loader would expect state for ALL optimizer params."
        )

    def test_fix_indexer_params_excluded_from_optimizer(self):
        """WITH fix: frozen indexer params excluded from optimizer -> consistent save/load."""
        model = _FakeModel(compress_ratios=(0, 0, 4, 128))
        indexer_names = _indexer_param_names(model)

        # Apply the fix
        freeze_deepseek_v4_indexer_params(model)

        # Build optimizer the same way as build_optimizer() in train_ft.py
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

        # Verify indexer params are NOT in the optimizer
        optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        for name, param in model.named_parameters():
            if name in indexer_names:
                assert id(param) not in optimizer_param_ids, f"Frozen param {name} should not be in optimizer"

        # Forward/backward/step works normally
        x = torch.randn(2, 16, device="cpu")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        total_model_params = sum(1 for _ in model.parameters())
        assert len(trainable_params) < total_model_params, (
            "Optimizer should track fewer params than the model (indexer ones excluded)"
        )

    def test_checkpoint_save_load_roundtrip(self):
        """Full save/load roundtrip succeeds with the fix applied."""
        model = _FakeModel(compress_ratios=(0, 0, 4, 128))

        freeze_deepseek_v4_indexer_params(model)

        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

        x = torch.randn(2, 16, device="cpu")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint.pt")
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                ckpt_path,
            )

            model2 = _FakeModel(compress_ratios=(0, 0, 4, 128))
            freeze_deepseek_v4_indexer_params(model2)
            trainable_params2 = list(filter(lambda p: p.requires_grad, model2.parameters()))
            optimizer2 = torch.optim.Adam(trainable_params2, lr=1e-3)

            x2 = torch.randn(2, 16, device="cpu")
            loss2 = model2(x2).sum()
            loss2.backward()
            optimizer2.step()

            ckpt = torch.load(ckpt_path, weights_only=False)
            model2.load_state_dict(ckpt["model"])
            optimizer2.load_state_dict(ckpt["optimizer"])

        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1.cpu(), p2.cpu()), f"Mismatch in {n1}"
