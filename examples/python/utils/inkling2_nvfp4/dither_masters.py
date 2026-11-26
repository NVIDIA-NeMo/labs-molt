"""Spread expert master weights over their rounding cells without changing what they quantize to.

The released experts decode to values that sit exactly on the NVFP4 / MXFP8 grid, i.e. in the middle of their
rounding cells, so small optimizer updates need hundreds of steps before any code changes. A training run that
continues from pretraining has masters at arbitrary positions inside the cells. `dither_nvfp4` / `dither_mxfp8`
move every master to a uniformly random position inside its own cell: the weights still quantize to the released
values (only the sign bit of zero codes may differ), and from then on an update of size d flips a code with
probability d / (cell width), in the direction of the update.

Kept fixed: one element per block that carries the block's largest code (it sets the block scale, and one of them
the expert's amax). The other elements with that code only move below it.
"""

import pack_experts as pe
import torch

_MARGIN = 0.9999  # stay strictly inside the cell


def _spread(weight, index, negative, table, block, unit, generator):
    """Masters for grid positions `index` of magnitudes `table`, in blocks of `block` values, times `unit`."""
    below = torch.cat((table[:1], (table[:-1] + table[1:]) / 2))  # lower edge of every cell
    above = torch.cat(((table[:-1] + table[1:]) / 2, table[-1:]))
    grouped = index.reshape(*index.shape[:-1], -1, block)
    block_max = grouped.amax(-1, keepdim=True)
    is_max = grouped == block_max
    anchor = (is_max & (is_max.cumsum(-1) == 1)).reshape(index.shape)  # first element with the block's largest code
    is_max, block_max = is_max.reshape(index.shape), block_max.expand_as(grouped).reshape(index.shape)
    low = below[index]
    high = torch.where(is_max, table[index], above[index])  # nothing may exceed the block's largest value
    position = torch.rand(weight.shape, generator=generator, device=weight.device)
    magnitude = (low + high) / 2 + (2 * position - 1) * (high - low) / 2 * _MARGIN
    signed = torch.where(negative, -magnitude, magnitude)
    # A zero code covers both signs.
    signed = torch.where(index == 0, (2 * position - 1) * above[0] * _MARGIN, signed)
    keep = anchor | (block_max == 0)
    return torch.where(keep, weight.float(), signed * unit)


def dither_nvfp4(weight: torch.Tensor, quantizer, generator: torch.Generator) -> torch.Tensor:
    """One expert weight [out, in] on the NVFP4 grid -> FP32 masters spread over their cells."""
    codes, scales, amax = pe.pack_nvfp4(weight, quantizer)
    table = torch.tensor(pe._MAGNITUDES, dtype=torch.float32, device=weight.device)
    nibbles = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    unit = (scales.float() * (amax / pe.OUTER_DENOMINATOR)).repeat_interleave(16, dim=-1)
    return _spread(weight, nibbles & 7, nibbles >= 8, table, 16, unit, generator)


def dither_mxfp8(weight: torch.Tensor, quantizer, generator: torch.Generator) -> torch.Tensor:
    """One expert weight [out, in] on the MXFP8 grid -> FP32 masters spread over their cells."""
    codes, exponents = pe.pack_mxfp8(weight, quantizer)
    raw = codes.view(torch.uint8).long()
    table = torch.arange(127, dtype=torch.uint8, device=weight.device).view(torch.float8_e4m3fn).float()
    unit = torch.exp2(exponents.float() - 127).repeat_interleave(32, dim=-1)
    return _spread(weight, raw & 127, raw >= 128, table, 32, unit, generator)
