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

"""Inkling 2 refit: the experts return to the checkpoint's packed layout and follow the trained weights."""

import json
import sys
import types

import pytest
import torch
from safetensors.torch import save_file

from molt.trainer.vllm import inkling2
from molt.trainer.vllm.vllm_worker_wrap import WorkerWrap

EXPERTS, OUTPUT, INPUT = 3, 4, 64
NAME = "model.llm.layers.0.mlp.experts.w13_weight"
MAGNITUDES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def decode_nvfp4(codes, scale, outer):
    nibbles = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    values = MAGNITUDES[nibbles & 7] * torch.where(nibbles < 8, 1, -1)
    return (values.unflatten(-1, (-1, 16)) * scale.float().unsqueeze(-1)).flatten(-2) * outer[:, None, None]


def decode_mxfp8(codes, scale):
    return (codes.float().unflatten(-1, (-1, 32)) * torch.exp2(scale.float() - 127).unsqueeze(-1)).flatten(-2)


def released(nvfp4):
    """A random packed projection in the checkpoint layout, with the weights it decodes to."""
    generator = torch.Generator().manual_seed(0)
    if nvfp4:
        codes = torch.randint(0, 256, (EXPERTS, OUTPUT, INPUT // 2), generator=generator, dtype=torch.uint8)
        scale = (torch.rand(EXPERTS, OUTPUT, INPUT // 16, generator=generator) + 0.5).to(torch.float8_e4m3fn)
        scale[0, 0, 0] = 0
        outer = torch.rand(EXPERTS, generator=generator) * 0.01 + 0.001
        return codes, scale, outer, decode_nvfp4(codes, scale, outer)
    codes = (torch.randn(EXPERTS, OUTPUT, INPUT, generator=generator) * 30).to(torch.float8_e4m3fn)
    scale = torch.randint(115, 125, (EXPERTS, OUTPUT, INPUT // 32), generator=generator, dtype=torch.uint8)
    return codes, scale, None, decode_mxfp8(codes, scale)


def decode(encoded, scale, outer):
    return decode_mxfp8(encoded, scale) if outer is None else decode_nvfp4(encoded, scale, outer)


@pytest.mark.parametrize("nvfp4", [True, False])
def test_untrained_experts_encode_to_the_released_weights(nvfp4):
    _, scale, outer, weight = released(nvfp4)
    encoded = inkling2.encode_experts(NAME, weight, scale, outer)
    assert encoded.dtype == (torch.uint8 if nvfp4 else torch.float8_e4m3fn)
    assert torch.equal(decode(encoded, scale, outer), weight)


@pytest.mark.parametrize("nvfp4", [True, False])
def test_encoded_experts_follow_a_small_update(nvfp4):
    _, scale, outer, weight = released(nvfp4)
    scale, weight = scale.repeat(1, 256, 1), weight.repeat(1, 256, 1)
    unit = weight.abs().max() * (1e-3 if nvfp4 else 1e-5)
    moved = decode(inkling2.encode_experts(NAME, weight + unit, scale, outer), scale, outer)
    changed = moved != weight
    # Far below half a grid step: rounding to nearest would change nothing.
    assert 0 < changed.float().mean() < 0.2
    assert ((moved - weight).sum() / (unit * weight.numel())).item() == pytest.approx(1, abs=0.2)
    # Same thresholds at every refit: an unchanged weight keeps its code.
    again = decode(inkling2.encode_experts(NAME, weight + unit, scale, outer), scale, outer)
    assert torch.equal(again, moved)


def test_refit_assigns_checkpoint_names_and_encodes_experts(tmp_path, monkeypatch):
    codes, scale, outer, weight = released(nvfp4=True)
    amax, denominator = outer * 8, torch.tensor([8.0])
    save_file(
        {NAME: codes, NAME + "_scale": scale, NAME + "_amax": amax, NAME + "_outer_scale_denominator": denominator},
        tmp_path / "model.safetensors",
    )
    index = {key: "model.safetensors" for key in (NAME, NAME + "_scale", NAME + "_amax")}
    index[NAME + "_outer_scale_denominator"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))

    def load_shard(param, tensor, shard):
        param.data[shard].copy_(tensor)

    params = {
        "model.layers.0.mlp.experts.w13_weight": torch.nn.Parameter(torch.zeros_like(codes), requires_grad=False),
        "model.layers.0.mlp.experts.w13_weight_scale": torch.nn.Parameter(torch.zeros_like(scale), requires_grad=False),
        "model.layers.0.mlp.experts.w13_weight_amax": torch.nn.Parameter(torch.zeros(EXPERTS)),
        "model.layers.0.mlp.experts.w13_weight_outer_scale_denominator": torch.nn.Parameter(torch.zeros(1)),
        "model.layers.0.attn.qkvr.weight": torch.nn.Parameter(torch.zeros(6, 2, dtype=torch.bfloat16)),
        "lm_head.weight": torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16)),
    }
    params["model.layers.0.attn.qkvr.weight"].weight_loader = load_shard
    model = types.SimpleNamespace(
        named_parameters=lambda: iter(params.items()),
        _renames=(("model.llm.unembed.", "lm_head."), ("model.llm.", "model.")),
        _shards={".attn.wk_dv.": (".attn.qkvr.", 4)},
        visual=None,
    )
    weight_utils = "vllm.model_executor.model_loader.weight_utils"
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.model_loader", weight_utils):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules[weight_utils].default_weight_loader = lambda param, tensor: param.data.copy_(tensor)

    loaded = inkling2.load_weights(
        model,
        [
            (NAME, weight),
            ("model.llm.unembed.weight", torch.ones(2, 2)),
            ("model.llm.layers.0.attn.wk_dv.weight", torch.full((2,), 2.0)),
            ("model.visual.proj_in.weight", torch.ones(2)),
        ],
        str(tmp_path),
    )

    assert loaded == set(params)
    expert = "model.layers.0.mlp.experts.w13_weight"
    assert torch.equal(decode_nvfp4(params[expert].data, params[expert + "_scale"].data, outer), weight)
    assert torch.equal(params[expert + "_amax"].data, amax)
    assert torch.equal(params["lm_head.weight"].data, torch.ones(2, 2, dtype=torch.bfloat16))
    assert params["model.layers.0.attn.qkvr.weight"].data[4].tolist() == [2, 2]


def test_inkling2_skips_the_layerwise_reload():
    worker = WorkerWrap.__new__(WorkerWrap)
    worker.model_config = types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type="inkling2_loop"))
    # Neither call imports vLLM, which this test does not provide.
    worker.begin_weight_update()
    worker.end_weight_update()
