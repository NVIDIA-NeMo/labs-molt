"""Pack expert weights into the checkpoint layout of Inkling 2 Small with Transformer Engine's own weight quantizers.

NVFP4 (layers 0-39): codes U8 [out, in / 2] (two E2M1 codes per byte, low nibble first), block scales F8_E4M3
[out, in / 16], one FP32 amax per expert; value = code x block scale x amax / 2688.
MXFP8 (layers 40-47): codes F8_E4M3 [out, in], block exponents U8 [out, in / 32]; value = code x 2^(byte - 127).
The quantizers are the weight quantizers Transformer Engine builds for the forward GEMM of the recipe, so the packed
tensors hold exactly the values a Transformer Engine expert uses for the same weight.
"""

import torch
from transformer_engine.common.recipe import MXFP8BlockScaling, NVFP4BlockScaling
from transformer_engine.pytorch.quantization import RecipeState

OUTER_DENOMINATOR = 2688.0
_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _weight_quantizer(recipe):
    quantizer = RecipeState.create(recipe, mode="forward", num_quantizers=2).make_quantizers()[1]
    quantizer.set_usage(rowwise=True, columnwise=False)
    quantizer.optimize_for_gemm = False
    return quantizer


def nvfp4_quantizer():
    """Weight quantizer of NVFP4BlockScaling with one block scale per 16 values of a row."""
    return _weight_quantizer(NVFP4BlockScaling(disable_2d_quantization=True))


def mxfp8_quantizer():
    return _weight_quantizer(MXFP8BlockScaling())


def pack_nvfp4(weight: torch.Tensor, quantizer) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One expert weight [out, in] -> (codes U8 [out, in / 2], scales F8_E4M3 [out, in / 16], amax FP32 [])."""
    rows, columns = weight.shape
    quantized = quantizer(weight)
    codes = quantized._rowwise_data
    scales = quantized._rowwise_scale_inv[:rows, : columns // 16].contiguous().view(torch.float8_e4m3fn)
    assert codes.shape == (rows, columns // 2) and codes.dtype == torch.uint8, (codes.shape, codes.dtype)
    return codes, scales, quantized._amax_rowwise.reshape(()).float()


def pack_mxfp8(weight: torch.Tensor, quantizer) -> tuple[torch.Tensor, torch.Tensor]:
    """One expert weight [out, in] -> (codes F8_E4M3 [out, in], exponents U8 [out, in / 32])."""
    rows, columns = weight.shape
    quantized = quantizer(weight)
    codes = quantized._rowwise_data.view(torch.float8_e4m3fn)
    exponents = quantized._rowwise_scale_inv[:rows, : columns // 32].contiguous()
    assert codes.shape == (rows, columns) and exponents.dtype == torch.uint8, (codes.shape, exponents.dtype)
    return codes, exponents


def decode_nvfp4(codes: torch.Tensor, scales: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    """Checkpoint decode of NVFP4 experts [..., out, in / 2] with amax [...] -> FP32 [..., out, in]."""
    magnitudes = torch.tensor(_MAGNITUDES, dtype=torch.float32, device=codes.device)
    nibbles = torch.stack((codes & 15, codes >> 4), -1).flatten(-2).long()
    values = magnitudes[nibbles & 7] * torch.where(nibbles < 8, 1.0, -1.0)
    values = (values.reshape(*values.shape[:-1], -1, 16) * scales.float().unsqueeze(-1)).flatten(-2)
    return values * (amax.float() / OUTER_DENOMINATOR).reshape(*amax.shape, 1, 1)


def decode_mxfp8(codes: torch.Tensor, exponents: torch.Tensor) -> torch.Tensor:
    blocks = codes.float().reshape(*codes.shape[:-1], -1, 32)
    return (blocks * torch.exp2(exponents.float() - 127).unsqueeze(-1)).flatten(-2)
