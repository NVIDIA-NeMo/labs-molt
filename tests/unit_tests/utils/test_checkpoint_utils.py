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

from types import SimpleNamespace

import pytest
import torch.nn as nn

import nemo_automodel.components.checkpoint.utils as checkpoint_utils
import nemo_automodel.components.models.common.tie_word_embeddings as tie_utils
from nemo_automodel.shared import tied_weights


def test_checkpoint_utils_reexports_shared_tied_weight_helpers():
    """Checkpoint callers retain their existing imports while shared code owns the helpers."""
    assert checkpoint_utils.ensure_tied_lm_head is tied_weights.ensure_tied_lm_head
    assert checkpoint_utils.has_local_tied_lm_head is tied_weights.has_local_tied_lm_head


def test_model_tie_policy_reexports_shared_contract():
    """Model policy and cross-component callers use one tie contract."""
    assert tie_utils.TieSupport is tied_weights.TieSupport
    assert tie_utils.get_controlling_tie_word_embeddings is tied_weights.get_controlling_tie_word_embeddings


def test_is_tied_word_embeddings_prefers_top_level_value():
    """Top-level flag controls tying, not nested text_config (matches HF construction)."""

    class DummyTextConfig:
        def __init__(self, tied: bool) -> None:
            self.tie_word_embeddings = tied

    class DummyConfig:
        def __init__(self, top: bool, text: bool) -> None:
            self.tie_word_embeddings = top
            self._text = DummyTextConfig(text)

        def get_text_config(self):
            return self._text

    class DummyModel(nn.Module):
        def __init__(self, top: bool, text: bool) -> None:
            super().__init__()
            self.config = DummyConfig(top, text)

    # top-level True wins over nested text_config False
    assert checkpoint_utils.is_tied_word_embeddings(DummyModel(top=True, text=False)) is True
    # top-level False wins over nested text_config True
    assert checkpoint_utils.is_tied_word_embeddings(DummyModel(top=False, text=True)) is False


def test_is_tied_word_embeddings_qwen3_vl_moe_follows_top_level():
    """Qwen3VLMoe follows its top-level flag, ignoring a conflicting nested text_config."""

    class DummyConfig:
        def __init__(self, top: bool) -> None:
            self.tie_word_embeddings = top

        def get_text_config(self):
            # Conflicting nested flag that must be ignored.
            return SimpleNamespace(tie_word_embeddings=True)

    class Qwen3VLMoeForConditionalGeneration(nn.Module):
        def __init__(self, top: bool) -> None:
            super().__init__()
            self.config = DummyConfig(top)

    # top-level True is honored even though nested text_config is also True
    assert checkpoint_utils.is_tied_word_embeddings(Qwen3VLMoeForConditionalGeneration(top=True)) is True
    # top-level False wins over nested text_config True
    assert checkpoint_utils.is_tied_word_embeddings(Qwen3VLMoeForConditionalGeneration(top=False)) is False


def test_is_tied_word_embeddings_falls_back_to_top_level_when_no_text_config():
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True)

    model = DummyModel()
    assert checkpoint_utils.is_tied_word_embeddings(model) is True


def test_is_tied_word_embeddings_handles_missing_config():
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    model = DummyModel()
    assert checkpoint_utils.is_tied_word_embeddings(model) is False


def test_is_tied_word_embeddings_uses_one_direction_policy_over_outer_config():
    """A fixed model policy wins over a misleading composite outer flag."""

    class UntiedOnlyModel(nn.Module):
        tie_word_embeddings_support = tie_utils.TieSupport.UNTIED_ONLY

        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                tie_word_embeddings=True,
                text_config=SimpleNamespace(tie_word_embeddings=False),
            )

    class TiedOnlyModel(nn.Module):
        tie_word_embeddings_support = tie_utils.TieSupport.TIED_ONLY

        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=False)

    assert checkpoint_utils.is_tied_word_embeddings(UntiedOnlyModel()) is False
    assert checkpoint_utils.is_tied_word_embeddings(TiedOnlyModel()) is True


def test_is_tied_word_embeddings_both_follows_outer_config():
    """A BOTH model still resolves its per-checkpoint top-level flag."""

    class BothModel(nn.Module):
        tie_word_embeddings_support = tie_utils.TieSupport.BOTH

        def __init__(self, tied: bool) -> None:
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=tied)

    assert checkpoint_utils.is_tied_word_embeddings(BothModel(tied=True)) is True
    assert checkpoint_utils.is_tied_word_embeddings(BothModel(tied=False)) is False


def test_is_tied_word_embeddings_qwen3_omni_moe_follows_top_level():
    """Qwen3OmniMoeThinker reports its top-level config intent.

    The resolver returns the actual controlling flag so a constructor guard can
    detect/reject an unsupported top-level=True. Checkpoint save safety comes from
    the storage-based has_local_tied_lm_head(), not from forcing this False.
    """

    class Qwen3OmniMoeThinkerForConditionalGeneration(nn.Module):
        def __init__(self, top: bool) -> None:
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=top)

    assert checkpoint_utils.is_tied_word_embeddings(Qwen3OmniMoeThinkerForConditionalGeneration(top=True)) is True
    assert checkpoint_utils.is_tied_word_embeddings(Qwen3OmniMoeThinkerForConditionalGeneration(top=False)) is False


def test_get_controlling_tie_word_embeddings_top_level_first():
    """Resolver prefers the top-level flag over nested text_config."""
    cfg_top_true = SimpleNamespace(
        tie_word_embeddings=True, get_text_config=lambda: SimpleNamespace(tie_word_embeddings=False)
    )
    cfg_top_false = SimpleNamespace(
        tie_word_embeddings=False, get_text_config=lambda: SimpleNamespace(tie_word_embeddings=True)
    )
    assert tie_utils.get_controlling_tie_word_embeddings(cfg_top_true) is True
    assert tie_utils.get_controlling_tie_word_embeddings(cfg_top_false) is False


def test_get_controlling_tie_word_embeddings_falls_back_to_text_config():
    """When the top-level config has no tie flag, fall back to text_config."""

    class _NoTopFlag:
        def get_text_config(self):
            return SimpleNamespace(tie_word_embeddings=True)

    assert tie_utils.get_controlling_tie_word_embeddings(_NoTopFlag()) is True


def _model_cls(name: str, support: tie_utils.TieSupport) -> type:
    """Build a throwaway model class with the given name and TieSupport policy.

    The class name selects the resolver's composite/omni special-casing and the
    ``tie_word_embeddings_support`` attribute drives the guard, matching how real
    registered model classes declare their policy.
    """
    return type(name, (), {"tie_word_embeddings_support": support})


def test_reject_unsupported_tie_word_embeddings_untied_only_raises_when_tied():
    """An UNTIED_ONLY (separate-head) class with tie_word_embeddings=True is rejected."""
    cls = _model_cls("Qwen3MoeForCausalLM", tie_utils.TieSupport.UNTIED_ONLY)
    config = SimpleNamespace(tie_word_embeddings=True)
    with pytest.raises(NotImplementedError, match="does not support tie_word_embeddings=True"):
        tie_utils.reject_unsupported_tie_word_embeddings(cls, config)


def test_reject_unsupported_tie_word_embeddings_untied_only_noop_when_untied():
    """The default (untied) config passes an UNTIED_ONLY guard without raising."""
    cls = _model_cls("Qwen3MoeForCausalLM", tie_utils.TieSupport.UNTIED_ONLY)
    tie_utils.reject_unsupported_tie_word_embeddings(cls, SimpleNamespace(tie_word_embeddings=False))  # no raise


def test_reject_unsupported_tie_word_embeddings_uses_top_level_for_composite():
    """Composite VLM configs read the controlling top-level flag, not nested text_config."""
    cls = _model_cls("Qwen3VLMoeForConditionalGeneration", tie_utils.TieSupport.UNTIED_ONLY)
    # top-level False (even with nested text True) -> not tied -> no raise
    untied = SimpleNamespace(
        tie_word_embeddings=False, get_text_config=lambda: SimpleNamespace(tie_word_embeddings=True)
    )
    tie_utils.reject_unsupported_tie_word_embeddings(cls, untied)
    # top-level True -> tied -> raise
    tied = SimpleNamespace(tie_word_embeddings=True, get_text_config=lambda: SimpleNamespace(tie_word_embeddings=False))
    with pytest.raises(NotImplementedError):
        tie_utils.reject_unsupported_tie_word_embeddings(cls, tied)


def test_get_controlling_tie_word_embeddings_omni_wrapper_reads_thinker_config():
    """Full Omni wrapper config nests the controlling flag under thinker_config.

    Qwen2_5OmniConfig / Qwen3OmniMoeConfig do not expose tie_word_embeddings at the
    top level; the controlling flag lives on config.thinker_config.
    """
    wrapper_tied = SimpleNamespace(thinker_config=SimpleNamespace(tie_word_embeddings=True))
    wrapper_untied = SimpleNamespace(thinker_config=SimpleNamespace(tie_word_embeddings=False))
    assert tie_utils.get_controlling_tie_word_embeddings(wrapper_tied) is True
    assert tie_utils.get_controlling_tie_word_embeddings(wrapper_untied) is False
    # When the thinker config itself is passed (no nested thinker_config), read its own flag.
    direct = SimpleNamespace(tie_word_embeddings=True)
    assert tie_utils.get_controlling_tie_word_embeddings(direct) is True


def test_reject_unsupported_tie_word_embeddings_omni_wrapper_path():
    """The guard raises for a full Omni wrapper whose thinker_config requests tying."""
    tied_cls = _model_cls("Qwen2_5OmniThinkerForConditionalGeneration", tie_utils.TieSupport.UNTIED_ONLY)
    wrapper = SimpleNamespace(thinker_config=SimpleNamespace(tie_word_embeddings=True))
    with pytest.raises(NotImplementedError):
        tie_utils.reject_unsupported_tie_word_embeddings(tied_cls, wrapper)
    untied_cls = _model_cls("Qwen3OmniMoeThinkerForConditionalGeneration", tie_utils.TieSupport.UNTIED_ONLY)
    wrapper_untied = SimpleNamespace(thinker_config=SimpleNamespace(tie_word_embeddings=False))
    tie_utils.reject_unsupported_tie_word_embeddings(untied_cls, wrapper_untied)  # no raise


def test_reject_unsupported_tie_word_embeddings_tied_only_raises_when_untied():
    """A TIED_ONLY model with tie_word_embeddings=False is rejected."""
    cls = _model_cls("Gemma4ForConditionalGeneration", tie_utils.TieSupport.TIED_ONLY)
    config = SimpleNamespace(tie_word_embeddings=False)
    with pytest.raises(NotImplementedError, match="does not support tie_word_embeddings=False"):
        tie_utils.reject_unsupported_tie_word_embeddings(cls, config)


def test_reject_unsupported_tie_word_embeddings_tied_only_noop_when_tied():
    """The default (tied) config passes a TIED_ONLY guard without raising."""
    cls = _model_cls("Gemma4ForConditionalGeneration", tie_utils.TieSupport.TIED_ONLY)
    tie_utils.reject_unsupported_tie_word_embeddings(cls, SimpleNamespace(tie_word_embeddings=True))  # no raise


def test_reject_unsupported_tie_word_embeddings_defaults_to_both():
    """A class that does not declare a policy defaults to BOTH (guard is a no-op)."""

    class _Undeclared:
        pass

    tie_utils.reject_unsupported_tie_word_embeddings(_Undeclared, SimpleNamespace(tie_word_embeddings=True))
    tie_utils.reject_unsupported_tie_word_embeddings(_Undeclared, SimpleNamespace(tie_word_embeddings=False))


class _DraftLikeModel(nn.Module):
    """Minimal stand-in for an EAGLE-3 draft model.

    Owns ``model.embed_tokens`` (full target vocab) and a separate
    ``lm_head`` (potentially shrunk vocab). Mirrors the FQNs that
    ``get_lm_head_weight_and_name`` and ``get_input_embeddings_weight_and_name``
    look for so the tests exercise the same code paths as the real model.
    """

    def __init__(
        self,
        embed_vocab: int,
        lm_head_vocab: int,
        hidden: int,
        tie_word_embeddings: bool,
        tie_storage: bool = False,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tie_word_embeddings)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(embed_vocab, hidden)
        self.lm_head = nn.Linear(hidden, lm_head_vocab, bias=False)
        if tie_storage:
            self.lm_head.weight = self.model.embed_tokens.weight


def test_has_local_tied_lm_head_false_when_shapes_disagree():
    """Vocab-shrunk EAGLE-3 draft: embed_tokens [V_t,H] != lm_head [V_d,H]."""
    model = _DraftLikeModel(embed_vocab=128256, lm_head_vocab=8192, hidden=2048, tie_word_embeddings=True)
    assert checkpoint_utils.has_local_tied_lm_head(model) is False


def test_has_local_tied_lm_head_true_when_shapes_match_and_tied():
    """Standard tied-embeddings case: shapes match, flag set, storage aliases."""
    model = _DraftLikeModel(
        embed_vocab=32000, lm_head_vocab=32000, hidden=128, tie_word_embeddings=True, tie_storage=True
    )
    assert checkpoint_utils.has_local_tied_lm_head(model) is True


def test_has_local_tied_lm_head_false_when_shapes_match_but_storage_is_untied():
    """Matching shapes and config are not enough; the tensors must alias."""
    model = _DraftLikeModel(embed_vocab=32000, lm_head_vocab=32000, hidden=128, tie_word_embeddings=True)
    assert checkpoint_utils.has_local_tied_lm_head(model) is False


def test_ensure_tied_lm_head_aliases_matching_local_weights():
    """Configured tied embeddings should become an actual local alias."""
    model = _DraftLikeModel(embed_vocab=32000, lm_head_vocab=32000, hidden=128, tie_word_embeddings=True)

    assert checkpoint_utils.ensure_tied_lm_head(model) is True

    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert checkpoint_utils.has_local_tied_lm_head(model) is True


def test_ensure_tied_lm_head_false_when_shapes_disagree():
    """Do not alias intentionally asymmetric heads, even when the config flag is set."""
    model = _DraftLikeModel(embed_vocab=128256, lm_head_vocab=8192, hidden=2048, tie_word_embeddings=True)

    assert checkpoint_utils.ensure_tied_lm_head(model) is False
    assert checkpoint_utils.has_local_tied_lm_head(model) is False


def test_has_local_tied_lm_head_false_when_flag_unset():
    """Even with matching shapes, untied config means not locally tied."""
    model = _DraftLikeModel(embed_vocab=32000, lm_head_vocab=32000, hidden=128, tie_word_embeddings=False)
    assert checkpoint_utils.has_local_tied_lm_head(model) is False
