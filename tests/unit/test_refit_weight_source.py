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

"""What a weight sync hands to vLLM's weight-transfer engine.

`ActorWeights` is the trainer's `WeightSource`: vLLM sizes its receive buffers from `metadata` and
reads the bytes from `iter_full`, so the two must list the same tensors in the same order, and each
tensor must carry the checkpoint name vLLM loads it under.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
from nemo_automodel.components._peft.lora import LinearLoRA

from molt.trainer.fsdp.refit import ActorWeights


def _model(tied: bool) -> nn.Module:
    model = nn.Module()
    model.config = SimpleNamespace(tie_word_embeddings=tied)
    model.embed_tokens = nn.Embedding(8, 4)
    model.router = nn.Linear(4, 2, bias=False).double()  # a weight kept in another dtype
    model.lm_head = nn.Linear(4, 8, bias=False)
    model.register_buffer("position_ids", torch.arange(8))  # not floating: never a weight
    return model


def test_metadata_lists_what_iter_full_yields():
    source = ActorWeights(_model(tied=False), ep_size=1)
    sent = list(source.iter_full())
    assert [name for name, _ in sent] == ["embed_tokens.weight", "router.weight", "lm_head.weight"]
    assert [(m.name, m.dtype, m.shape) for m in source.metadata] == [
        (name, weight.dtype, tuple(weight.shape)) for name, weight in sent
    ]
    assert dict(sent)["router.weight"].dtype == torch.float64  # each tensor keeps its own dtype


def test_tied_lm_head_travels_as_embed_tokens():
    source = ActorWeights(_model(tied=True), ep_size=1)
    assert [m.name for m in source.metadata] == ["embed_tokens.weight", "router.weight"]


def test_lora_adapter_travels_merged_into_its_base_weight():
    model = nn.Module()
    model.q_proj = LinearLoRA(nn.Linear(4, 4, bias=False), dim=2, alpha=4)
    nn.init.normal_(model.q_proj.lora_B.weight)  # B starts at zero; make the delta visible
    sd = model.state_dict()
    sent = dict(ActorWeights(model, ep_size=1).iter_full())
    assert list(sent) == ["q_proj.weight"]
    delta = 2.0 * sd["q_proj.lora_B.weight"] @ sd["q_proj.lora_A.weight"]
    torch.testing.assert_close(sent["q_proj.weight"], sd["q_proj.weight"] + delta)


def test_state_dict_adapter_names_the_tensors_and_takes_no_options():
    class Adapter:
        def convert_single_tensor_to_hf(self, fqn, tensor):  # an option would be a TypeError, as in Inkling 2
            return [(f"model.{fqn}.a", tensor[:1]), (f"model.{fqn}.b", tensor[1:])]

    model = nn.Module()
    model.experts = nn.Linear(4, 2, bias=False)
    model.state_dict_adapter = Adapter()
    source = ActorWeights(model, ep_size=1)
    assert [(m.name, m.shape) for m in source.metadata] == [
        ("model.experts.weight.a", (1, 4)),
        ("model.experts.weight.b", (1, 4)),
    ]


def test_adapter_with_the_whole_dict_api_only_names_the_tensors_through_to_hf():
    class Adapter:  # KimiVL's adapter has to_hf / from_hf and nothing per tensor
        def to_hf(self, state_dict):
            return {f"language_model.{name}": tensor for name, tensor in state_dict.items()}

    model = nn.Module()
    model.proj = nn.Linear(4, 2, bias=False)
    model.state_dict_adapter = Adapter()
    assert [m.name for m in ActorWeights(model, ep_size=1).metadata] == ["language_model.proj.weight"]
