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

"""vLLM weight refit for Inkling 2.

vLLM serves the Inkling 2 experts from the checkpoint's packed NVFP4 / MXFP8 weights, and its
``load_weights`` takes a whole checkpoint. The trainer broadcasts floating-point weights a few
tensors at a time, so this module assigns them one by one and encodes each expert projection
back onto the block scales the checkpoint was released with.
"""

import json
import re
import zlib
from pathlib import Path

import torch

_EXPERT_WEIGHT = re.compile(r"\.mlp\.experts\.w(13|2)_weight$")
# Experts encoded per step: bounds the FP32 temporaries to a few gigabytes.
_EXPERTS_PER_STEP = 16


def _round_to_grid(x: torch.Tensor, grid: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Round each element to one of its two neighbours in the sorted ``grid``.

    The upper neighbour wins when the element's position between the two exceeds a random
    threshold. Rounding to nearest would hold a released weight, which sits on the grid, in its
    cell until training had moved it half a step; here the encoded weights follow the trained
    ones on average from the first update.
    """
    upper = torch.bucketize(x, grid, right=True).clamp_(1, len(grid) - 1)
    low, high = grid[upper - 1], grid[upper]
    # An unchanged weight decodes to its grid value only up to FP32 rounding: keep it there.
    position = (x - low) / (high - low) * (1 + 2e-5) - 1e-5
    threshold = torch.rand(x.shape, generator=generator, device=x.device)
    return torch.where(threshold < position, high, low)


@torch.no_grad()
def encode_experts(name: str, weight: torch.Tensor, scale: torch.Tensor, outer: torch.Tensor | None) -> torch.Tensor:
    """Encode [experts, output, input] floating-point expert weights in the checkpoint's packed layout.

    ``scale`` is the checkpoint's block scale [experts, output, input / block]. NVFP4 (``outer`` is
    the per-expert outer scale) returns uint8 [experts, output, input / 2], two E2M1 codes per byte;
    MXFP8 (``outer`` is None) returns float8_e4m3fn [experts, output, input].

    The random thresholds are seeded by the tensor name: every engine draws the same ones at every
    refit, so replicas agree and a weight changes its code only when training moves it.
    """
    generator = torch.Generator(weight.device).manual_seed(zlib.crc32(name.encode()))
    if outer is None:
        grid = torch.arange(256, dtype=torch.uint8, device=weight.device).view(torch.float8_e4m3fn).float()
        grid = grid[grid.isfinite()].unique()
    else:
        # Magnitudes of the eight E2M1 codes; bit 3 of a code is its sign.
        magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=weight.device)
        grid = torch.cat((-magnitudes.flip(0), magnitudes[1:]))
    encoded = []
    for start in range(0, weight.shape[0], _EXPERTS_PER_STEP):
        step = slice(start, start + _EXPERTS_PER_STEP)
        if outer is None:
            unit = torch.exp2(scale[step].float() - 127).unsqueeze(-1)
        else:
            unit = scale[step].float().unsqueeze(-1) * outer[step, None, None, None]
        blocks = weight[step].float().unflatten(-1, (unit.shape[2], -1))
        # A block with a zero scale holds only zeros.
        values = _round_to_grid(torch.where(unit > 0, blocks / unit, 0), grid, generator).flatten(-2)
        if outer is None:
            encoded.append(values.to(torch.float8_e4m3fn))
        else:
            codes = torch.bucketize(values.abs(), magnitudes) | (values < 0) * 8
            # One byte holds two codes, low nibble first.
            encoded.append((codes[..., 0::2] | codes[..., 1::2] << 4).to(torch.uint8))
    return torch.cat(encoded)


@torch.no_grad()
def load_weights(model: torch.nn.Module, weights: list[tuple[str, torch.Tensor]], checkpoint: str) -> set[str]:
    """Assign refit tensors named as in the checkpoint; returns the vLLM parameter names assigned.

    ``checkpoint`` is the served checkpoint directory: the block scales of the experts are read from
    it, because vLLM keeps them in its kernels' layout.
    """
    from safetensors import safe_open
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    params = dict(model.named_parameters())
    files = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
    loaded: set[str] = set()

    def assign(name: str, tensor: torch.Tensor, *shard: int) -> None:
        param = params[name]
        getattr(param, "weight_loader", default_weight_loader)(param, tensor.to(param.dtype), *shard)
        loaded.add(name)

    for source, weight in weights:
        if source.startswith("model.visual."):
            if model.visual is not None:  # an engine started without image inputs has no vision tower
                pair = (source.removeprefix("model.visual."), weight)
                loaded |= {"visual." + name for name in model.visual.load_weights([pair])}
            continue
        name = source
        for old, new in model._renames:
            name = name.replace(old, new)
        if _EXPERT_WEIGHT.search(source):
            nvfp4 = params[name].dtype == torch.uint8
            suffixes = ("_scale", "_amax", "_outer_scale_denominator") if nvfp4 else ("_scale",)
            with safe_open(Path(checkpoint) / files[source], "pt", device=str(weight.device)) as file:
                released = [file.get_tensor(source + suffix) for suffix in suffixes]
            outer = released[1] / released[2] if nvfp4 else None
            assign(name, encode_experts(source, weight, released[0], outer))
            for suffix, tensor in zip(suffixes, released):
                assign(name + suffix, tensor)
            continue
        shard = ()
        for old, (new, index) in model._shards.items():
            if old in name:
                name, shard = name.replace(old, new), (index,)
                break
        assign(name, weight, *shard)
    return loaded
