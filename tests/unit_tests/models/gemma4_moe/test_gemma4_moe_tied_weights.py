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

"""Weight-tying tests for the Gemma4 MoE conditional-generation model.

HF's ``super().__init__()`` ties ``lm_head.weight`` to the original text
``embed_tokens``. The MoE path then replaces ``language_model`` with
``Gemma4MoETextModelBackend`` (a fresh ``embed_tokens``), which orphans that
alias. The model re-ties ``lm_head`` to the now-active embedding when
``tie_word_embeddings`` is set (Gemma defaults to ``True``); these tests pin
that behavior for both the tied and untied configs.

Runs on CPU (no CUDA / TE / DeepEP required).
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gemma4_moe.model import (
    Gemma4ForConditionalGeneration,
    Gemma4MoETextModelBackend,
)


def _make_text_config(**overrides):
    """Tiny Gemma4TextConfig (2 layers, small hidden, tiny vocab, few experts)."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=2,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        enable_moe_block=True,  # routes construction through the NeMo MoE backend
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=64,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=128,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="bfloat16",
    )
    defaults.update(overrides)
    return Gemma4TextConfig(**defaults)


def _make_cpu_backend():
    """CPU-friendly backend: no TE, no DeepEP, plain torch kernels."""
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _build(tie_word_embeddings: bool, text_tie: bool | None = None) -> Gemma4ForConditionalGeneration:
    # The controlling flag is the top-level Gemma4Config.tie_word_embeddings (matches HF);
    # text_tie lets a test set a conflicting nested value to prove top-level wins.
    config = Gemma4Config(
        text_config=_make_text_config(tie_word_embeddings=tie_word_embeddings if text_tie is None else text_tie),
        tie_word_embeddings=tie_word_embeddings,
    )
    model = Gemma4ForConditionalGeneration(config, backend=_make_cpu_backend())
    # Sanity: construction routed through the real NeMo MoE backend (the path
    # that replaces language_model and breaks HF's tie).
    assert isinstance(model.model.language_model, Gemma4MoETextModelBackend)
    return model


def test_tied_lm_head_shares_active_embedding_after_construction():
    """tie_word_embeddings=True: lm_head must alias the *active* MoE embed_tokens."""
    model = _build(tie_word_embeddings=True)
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight


def test_tied_lm_head_survives_initialize_weights():
    """The tie set in __init__ must survive the bf16 cast in initialize_weights()."""
    model = _build(tie_word_embeddings=True)
    model.initialize_weights(dtype=torch.bfloat16, buffer_device=torch.device("cpu"))

    embed = model.model.language_model.embed_tokens.weight
    lm_head = model.lm_head.weight
    assert lm_head is embed
    assert lm_head.dtype == torch.bfloat16


def test_tied_lm_head_casts_fp32_hidden_after_logits_slicing(monkeypatch):
    model = _build(tie_word_embeddings=True).to(torch.bfloat16)
    hidden = torch.randn(1, 4, model.config.text_config.hidden_size, dtype=torch.float32)

    def fake_language_model_forward(*_args, **_kwargs):
        return SimpleNamespace(last_hidden_state=hidden)

    monkeypatch.setattr(model.model.language_model, "forward", fake_language_model_forward)
    outputs = model(
        inputs_embeds=torch.randn(1, 4, model.config.text_config.hidden_size, dtype=torch.bfloat16),
        logits_to_keep=2,
    )

    assert outputs.logits.shape == (1, 2, model.config.text_config.vocab_size)
    assert outputs.logits.dtype == torch.bfloat16


def test_untied_construction_is_rejected():
    """Gemma4 is tied by default; tie_word_embeddings=False is rejected at construction."""
    with pytest.raises(NotImplementedError, match="does not support tie_word_embeddings=False"):
        _build(tie_word_embeddings=False)


def test_tie_weights_hook_reties_to_active_embedding():
    """The public tie_weights() hook must re-point lm_head at the active MoE embedding.

    Checkpoint/AutoModel paths call ``model.tie_weights()`` after construction
    (e.g. via ``ensure_tied_lm_head``); this guards that the override re-ties to
    the live embedding rather than HF's generic behavior.
    """
    model = _build(tie_word_embeddings=True)
    # Break the tie with a fresh, independent parameter.
    model.lm_head.weight = nn.Parameter(model.lm_head.weight.detach().clone())
    assert model.lm_head.weight is not model.model.language_model.embed_tokens.weight

    model.tie_weights()
    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight


def test_top_level_flag_controls_tie_when_flags_disagree():
    """The controlling flag is top-level Gemma4Config.tie_word_embeddings, not text_config (matches HF)."""
    # top-level True wins over text_config False -> tied (constructs, shares storage)
    tied = _build(tie_word_embeddings=True, text_tie=False)
    assert tied.lm_head.weight is tied.model.language_model.embed_tokens.weight
    # top-level False is the controlling flag -> untie is rejected even when text_config is True
    with pytest.raises(NotImplementedError, match="does not support tie_word_embeddings=False"):
        _build(tie_word_embeddings=False, text_tie=True)
