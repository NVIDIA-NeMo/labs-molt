# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Selection tests for the shared multimodal module taxonomy.

These use real ``nn.Module`` trees (not doubles) so they exercise the same
``named_modules`` path production takes.
"""

import torch.nn as nn

from nemo_automodel.shared.multimodal_fsdp import (
    MULTIMODAL_MODULE_NAMES,
    MULTIMODAL_PROJECTOR_NAMES,
    MULTIMODAL_TOWER_NAMES,
    is_multimodal_module_name,
    iter_multimodal_modules,
    module_is_fully_frozen,
)


class _Leaf(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)


def _selected(model):
    return {name for name, _ in iter_multimodal_modules(model)}


class TestTaxonomy:
    def test_tower_and_projector_groups_are_disjoint(self):
        assert not set(MULTIMODAL_TOWER_NAMES) & set(MULTIMODAL_PROJECTOR_NAMES)
        assert set(MULTIMODAL_MODULE_NAMES) == set(MULTIMODAL_TOWER_NAMES) | set(MULTIMODAL_PROJECTOR_NAMES)

    def test_embed_vision_and_embed_audio_are_both_projectors(self):
        """Gemma4 builds these as two instances of one Gemma4MultimodalEmbedder class.

        Listing only one gave the siblings different FSDP treatment under every
        policy: under ``replicate`` one was replicated and the other stayed
        sharded in the root.
        """
        assert "embed_vision" in MULTIMODAL_PROJECTOR_NAMES
        assert "embed_audio" in MULTIMODAL_PROJECTOR_NAMES

    def test_vit_model_is_a_tower(self):
        """Was present only in moe/parallelizer's private copy of the list (BAGEL)."""
        assert "vit_model" in MULTIMODAL_TOWER_NAMES
        assert is_multimodal_module_name("vit_model")


class TestIterMultimodalModules:
    def test_selects_all_four_gemma4_submodules(self):
        class Gemma4Like(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.vision_tower = _Leaf()
                self.model.audio_tower = _Leaf()
                self.model.embed_vision = _Leaf()
                self.model.embed_audio = _Leaf()
                self.model.language_model = _Leaf()

        assert _selected(Gemma4Like()) == {
            "model.vision_tower",
            "model.audio_tower",
            "model.embed_vision",
            "model.embed_audio",
        }

    def test_selects_vit_model(self):
        class BagelLike(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.vit_model = _Leaf()
                self.model.language_model = _Leaf()

        assert _selected(BagelLike()) == {"model.vit_model"}

    def test_selection_is_maximal(self):
        """A nested tower inside a selected tower must not be yielded separately."""

        class Nested(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_tower = nn.Module()
                self.vision_tower.vision_model = _Leaf()

        assert _selected(Nested()) == {"vision_tower"}

    def test_ignores_non_multimodal_modules(self):
        class TextOnly(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([_Leaf()])
                self.lm_head = nn.Linear(2, 2)

        assert _selected(TextOnly()) == set()


class TestModuleIsFullyFrozen:
    def test_requires_parameters(self):
        assert not module_is_fully_frozen(nn.Module())

    def test_true_when_no_parameter_requires_grad(self):
        leaf = _Leaf()
        leaf.requires_grad_(False)
        assert module_is_fully_frozen(leaf)

    def test_false_when_any_parameter_requires_grad(self):
        leaf = _Leaf()
        leaf.requires_grad_(False)
        leaf.proj.bias.requires_grad_(True)
        assert not module_is_fully_frozen(leaf)
