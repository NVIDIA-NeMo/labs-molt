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

"""Attention lifecycle and independent RoPE, compressor and indexer arithmetic."""

import copy
import math
from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.attention import (
    DeepseekV41Attention,
    DeepseekV41AttentionState,
    _apply_rope,
    _Compressor,
    _Indexer,
    _RotaryEmbedding,
    _select_candidate_blocks,
)
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache
from tests.unit_tests.models.deepseek_v41.test_quantization import _independent_mx_scale


def _config(dtype: str = "float32") -> DeepseekV41TextConfig:
    return DeepseekV41TextConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=6,
        num_attention_heads=4,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_groups=2,
        o_lora_rank=4,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        sliding_window=2,
        compress_ratios=[0, 2, 2, 1, 1, 1],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 4],
        candidate_source_layer_id=3,
        candidate_block_size=2,
        candidate_topk_blocks=2,
        engram_layer_ids=[],
        dtype=dtype,
        rope_scaling={},
    )


def _backend(attn: str = "eager") -> BackendConfig:
    return BackendConfig(attn=attn, linear="torch", rms_norm="torch_fp32", experts="torch", dispatcher="torch")


def _reference_swa(layer: DeepseekV41Attention, hidden: torch.Tensor) -> torch.Tensor:
    """Evaluate each causal window separately using the official complex RoPE definition.

    Args:
        layer: Attention parameters; linear weights have standard [out, in]
            layout and normalization weights have shape [channels].
        hidden: Tensor of shape [batch, sequence, hidden].

    Returns:
        Tensor of shape [batch, sequence, hidden].
    """
    batch, sequence, _ = hidden.shape
    rotary_dim = layer.rotary_emb.dim
    frequency = 1 / layer.rotary_emb.theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim)
    phase = torch.outer(torch.arange(sequence).float(), frequency)
    rotations = torch.polar(torch.ones_like(phase), phase)

    def normalize(values: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """Normalize final channels.

        Args:
            values: Tensor of shape [..., channels], with arbitrary leading dimensions.
            weight: Tensor of shape [channels].
            eps: Variance stabilizer.

        Returns:
            Tensor of shape [..., channels].
        """
        return F.rms_norm(values.float(), (values.shape[-1],), weight.float(), eps).to(values.dtype)

    query_latent = normalize(F.linear(hidden, layer.wq_a.weight), layer.q_norm.weight, layer.q_norm.eps)
    query = F.linear(query_latent, layer.wq_b.weight).reshape(batch, sequence, layer.num_heads, layer.head_dim)
    kv = normalize(F.linear(hidden, layer.wkv.weight), layer.kv_norm.weight, layer.kv_norm.eps)
    query_complex = torch.view_as_complex(
        query[..., -rotary_dim:].float().reshape(batch, sequence, layer.num_heads, -1, 2)
    )
    kv_complex = torch.view_as_complex(kv[..., -rotary_dim:].float().reshape(batch, sequence, -1, 2))
    query = torch.cat(
        (query[..., :-rotary_dim], torch.view_as_real(query_complex * rotations[None, :, None]).flatten(-2)), dim=-1
    )
    kv = torch.cat((kv[..., :-rotary_dim], torch.view_as_real(kv_complex * rotations[None]).flatten(-2)), dim=-1)
    # Cache rounding is validated independently against the official kernels;
    # both attention formulations consume that identical representation.
    kv = quantize_cache(kv, format="fp8", block_size=32)
    outputs = []
    for position in range(sequence):
        visible = kv[:, max(0, position + 1 - layer.window_size) : position + 1]
        logits = torch.einsum("bhd,btd->bht", query[:, position], visible) / layer.head_dim**0.5
        normalizer = torch.logaddexp(logits.logsumexp(-1), layer.attn_sink[None])
        output = torch.einsum("bht,btd->bhd", (logits - normalizer[..., None]).exp(), visible)
        complex_output = torch.view_as_complex(output[..., -rotary_dim:].reshape(batch, layer.num_heads, -1, 2))
        output = torch.cat(
            (
                output[..., :-rotary_dim],
                torch.view_as_real(complex_output * rotations[position].conj()).flatten(-2),
            ),
            dim=-1,
        )
        grouped = output.reshape(batch, layer.num_groups, -1)
        weights = layer.wo_a.weight.unflatten(0, (layer.num_groups, -1))
        projected = torch.stack(
            [F.linear(grouped[:, group], weights[group]) for group in range(layer.num_groups)], dim=1
        )
        outputs.append(F.linear(projected.flatten(1), layer.wo_b.weight))
    return torch.stack(outputs, dim=1)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_swa_matches_independent_window_reference_forward_and_backward(backend: str) -> None:
    torch.manual_seed(17)
    layer = DeepseekV41Attention(_config(), 0, _backend(backend))
    with torch.no_grad():
        layer.attn_sink.copy_(torch.tensor([-3.0, -0.5, 1.0, 4.0]))
    reference = copy.deepcopy(layer)
    hidden = torch.randn(2, 6, 16, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    expected = _reference_swa(reference, reference_hidden)
    actual = layer(hidden, position_ids=torch.arange(6)[None], state=DeepseekV41AttentionState()).hidden_states
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=8e-7, rtol=8e-6)
    for (name, parameter), (reference_name, reference_parameter) in zip(
        layer.named_parameters(), reference.named_parameters()
    ):
        assert name == reference_name
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=2e-6, rtol=2e-5)


def test_compressor_pools_each_channel_and_excludes_incomplete_groups() -> None:
    config = _config()
    compressor = _Compressor(config, ratio=2, dtype=torch.float32)
    with torch.no_grad():
        compressor.wkv.weight.zero_()
        compressor.wkv.weight[:, :8].copy_(torch.eye(8))
        compressor.wgate.weight.zero_()
        compressor.wgate.weight[:, :8].copy_(torch.eye(8))
    hidden = torch.zeros(1, 3, 16, requires_grad=True)
    with torch.no_grad():
        hidden[0, 0, :8] = torch.arange(1, 9).float()
        hidden[0, 1, :8] = -torch.arange(1, 9).float()
        hidden[0, 2] = 1000
    actual = compressor(hidden)
    channels = torch.arange(1, 9).float()
    # Softmax over {+a, -a} gives E[value] = a * tanh(a).
    pooled = channels * channels.tanh()
    expected = pooled / pooled.square().mean().sqrt()
    torch.testing.assert_close(actual[0, 0], expected, atol=2e-7, rtol=2e-6)
    actual.square().mul(torch.arange(1, 9)).sum().backward()
    assert hidden.grad[0, :2].abs().sum() > 0
    assert torch.count_nonzero(hidden.grad[0, 2]) == 0
    assert compressor.wgate.weight.grad.abs().sum() > 0


def test_ratio_one_has_no_gate_and_normalizes_plain_projection() -> None:
    compressor = _Compressor(_config(), ratio=1, dtype=torch.float32)
    assert not hasattr(compressor, "wgate")
    hidden = torch.randn(2, 5, 16)
    projected = F.linear(hidden, compressor.wkv.weight)
    expected = F.rms_norm(projected, (8,), compressor.norm.weight, compressor.norm.eps)
    torch.testing.assert_close(compressor(hidden), expected)


def test_candidate_blocks_pin_latest_partial_block_and_drop_unreachable_blocks() -> None:
    scores = torch.tensor([[[10.0, 9.0, 8.0, 7.0, -2.0, -torch.inf], [-torch.inf] * 6]])
    lengths = torch.tensor([[[5], [0]]])
    actual = _select_candidate_blocks(scores, lengths, topk_blocks=2, block_size=2)
    expected = torch.tensor([[[True, True, False, False, True, True], [False] * 6]])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("sequence", [1, 5, 6])
def test_full_reindex_reuse_and_ratio_transition_preserve_state(sequence: int) -> None:
    torch.manual_seed(21)
    layers = [DeepseekV41Attention(_config(), index, _backend()) for index in range(1, 6)]
    hidden = torch.randn(2, sequence, 16, requires_grad=True)
    positions = torch.arange(sequence)[None]
    state = DeepseekV41AttentionState()
    full = layers[0](hidden, position_ids=positions, state=state)
    assert state.compressed_kv is None
    original_kv = full.state.compressed_kv.detach().clone()
    reuse = layers[1](hidden, position_ids=positions, state=full.state)
    assert reuse.state is full.state
    assert reuse.state.compressed_kv.shape == (2, sequence // 2, 8)
    torch.testing.assert_close(full.state.compressed_kv, original_kv)
    decoder_full = layers[2](hidden, position_ids=positions, state=reuse.state)
    assert decoder_full.state.compression_ratio == 1
    assert decoder_full.state.compressed_kv.shape == (2, sequence, 8)
    reindex = layers[3](hidden, position_ids=positions, state=decoder_full.state)
    assert reindex.state.compressed_kv is decoder_full.state.compressed_kv
    assert reindex.state.index_keys is decoder_full.state.index_keys
    assert reindex.state.candidates is decoder_full.state.candidates
    assert reindex.state.topk_indices is not decoder_full.state.topk_indices
    final = layers[4](hidden, position_ids=positions, state=reindex.state)
    assert final.state is reindex.state
    for layer in layers:
        if layer.indexer is not None:
            assert not any(parameter.requires_grad for parameter in layer.indexer.parameters())
    (reuse.hidden_states.square().sum() + final.hidden_states.square().sum()).backward()
    assert torch.isfinite(hidden.grad).all()
    assert layers[2].compressor.wkv.weight.grad.abs().sum() > 0
    if sequence >= 2:
        assert layers[0].compressor.wkv.weight.grad.abs().sum() > 0
        assert layers[0].compressor.wgate.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("dtype,atol,rtol", [("float32", 1e-6, 1e-5), ("bfloat16", 0.012, 0.08)])
def test_csa2_eager_sdpa_match_including_cross_layer_gradients(dtype: str, atol: float, rtol: float) -> None:
    torch.manual_seed(11)
    config = _config(dtype)
    eager = torch.nn.ModuleList([DeepseekV41Attention(config, index, _backend()) for index in range(6)])
    sdpa = torch.nn.ModuleList([DeepseekV41Attention(config, index, _backend("sdpa")) for index in range(6)])
    sdpa.load_state_dict(eager.state_dict())
    source = torch.randn(1, 7, 16).to(torch.bfloat16 if dtype == "bfloat16" else torch.float32)
    reference_hidden = source.clone().requires_grad_()
    hidden = source.clone().requires_grad_()
    reference_state = DeepseekV41AttentionState()
    state = DeepseekV41AttentionState()
    for reference_layer, layer in zip(eager, sdpa):
        expected = reference_layer(reference_hidden, position_ids=torch.arange(7)[None], state=reference_state)
        actual = layer(hidden, position_ids=torch.arange(7)[None], state=state)
        torch.testing.assert_close(actual.hidden_states, expected.hidden_states, atol=atol, rtol=rtol)
        if expected.state.topk_indices is not None:
            torch.testing.assert_close(actual.state.topk_indices, expected.state.topk_indices)
        # Isolate each kernel's numerical error while retaining shared-KV gradients.
        reference_state, state = expected.state, actual.state
    upstream = torch.randn_like(actual.hidden_states)
    actual.hidden_states.backward(upstream)
    expected.hidden_states.backward(upstream)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        sdpa[3].compressor.wkv.weight.grad, eager[3].compressor.wkv.weight.grad, atol=atol, rtol=rtol
    )


def test_checkpoint_recomputation_preserves_source_gradients_and_state() -> None:
    torch.manual_seed(13)
    full = DeepseekV41Attention(_config(), 1, _backend())
    reuse = DeepseekV41Attention(_config(), 2, _backend())
    expected_full, expected_reuse = copy.deepcopy(full), copy.deepcopy(reuse)
    hidden = torch.randn(2, 6, 16, requires_grad=True)
    expected_hidden = hidden.detach().clone().requires_grad_()
    source = checkpoint(
        full, hidden, position_ids=torch.arange(6)[None], state=DeepseekV41AttentionState(), use_reentrant=False
    )
    actual = checkpoint(reuse, hidden, position_ids=torch.arange(6)[None], state=source.state, use_reentrant=False)
    expected_source = expected_full(
        expected_hidden, position_ids=torch.arange(6)[None], state=DeepseekV41AttentionState()
    )
    expected = expected_reuse(expected_hidden, position_ids=torch.arange(6)[None], state=expected_source.state)
    source_copy = source.state.compressed_kv.detach().clone()
    upstream = torch.randn_like(actual.hidden_states)
    actual.hidden_states.backward(upstream)
    expected.hidden_states.backward(upstream)
    torch.testing.assert_close(hidden.grad, expected_hidden.grad)
    torch.testing.assert_close(full.compressor.wkv.weight.grad, expected_full.compressor.wkv.weight.grad)
    torch.testing.assert_close(source.state.compressed_kv, source_copy)


def test_padding_and_incomplete_groups_do_not_change_real_token_outputs() -> None:
    torch.manual_seed(31)
    layer = DeepseekV41Attention(_config(), 1, _backend())
    hidden = torch.randn(2, 6, 16, requires_grad=True)
    mask = torch.tensor([[1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
    actual = layer(hidden, position_ids=torch.arange(6)[None], state=DeepseekV41AttentionState(), attention_mask=mask)
    expected = layer(hidden[:1, :3], position_ids=torch.arange(3)[None], state=DeepseekV41AttentionState())
    torch.testing.assert_close(actual.hidden_states[:1, :3], expected.hidden_states)
    assert torch.count_nonzero(actual.hidden_states[~mask.bool()]) == 0
    actual.hidden_states.square().sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.count_nonzero(hidden.grad[~mask.bool()]) == 0


def test_rotary_recomputes_exact_fp32_frequencies_after_bfloat16_cast() -> None:
    config = _config()
    rotary = _RotaryEmbedding(config, compressed=True)
    positions = torch.tensor([[0, 65536, 1000000]])
    expected = rotary(positions)
    rotary.to(dtype=torch.bfloat16)
    torch.testing.assert_close(rotary(positions), expected, rtol=0, atol=0)
    values = torch.randn(1, 3, 2, 8)
    original = values.clone()
    rotated = _apply_rope(values, expected)
    torch.testing.assert_close(_apply_rope(rotated, expected, inverse=True), values, atol=3e-7, rtol=3e-7)
    torch.testing.assert_close(values, original, rtol=0, atol=0)


def test_meta_materialization_initializer_and_first_backward_are_finite() -> None:
    with torch.device("meta"):
        layer = DeepseekV41Attention(_config(), 1, _backend())
    layer.to_empty(device="cpu")
    layer.reset_parameters()
    assert all(torch.isfinite(parameter).all() for parameter in layer.parameters())
    hidden = torch.randn(1, 5, 16, requires_grad=True)
    output = layer(hidden, position_ids=torch.arange(5)[None], state=DeepseekV41AttentionState())
    output.hidden_states.square().sum().backward()
    assert torch.isfinite(hidden.grad).all()
    assert torch.isfinite(layer.compressor.wgate.weight.grad).all()


def test_missing_source_and_unsupported_positions_masks_and_backends_fail() -> None:
    hidden = torch.randn(1, 4, 16)
    reuse = DeepseekV41Attention(_config(), 2, _backend())
    with pytest.raises(ValueError, match="preceding Full"):
        reuse(hidden, position_ids=torch.arange(4)[None], state=DeepseekV41AttentionState())
    source = DeepseekV41Attention(_config(), 1, _backend())
    with pytest.raises(ValueError, match="zero-based"):
        source(hidden, position_ids=torch.tensor([[0, 1, 0, 1]]), state=DeepseekV41AttentionState())
    with pytest.raises(ValueError, match="right padding"):
        source(
            hidden,
            position_ids=torch.arange(4)[None],
            state=DeepseekV41AttentionState(),
            attention_mask=torch.tensor([[0, 1, 1, 1]]),
        )
    with pytest.raises(ValueError, match="zero and one"):
        source(
            hidden,
            position_ids=torch.arange(4)[None],
            state=DeepseekV41AttentionState(),
            attention_mask=torch.full((1, 4), 2),
        )
    with pytest.raises(ValueError, match="eager.*sdpa"):
        DeepseekV41Attention(_config(), 1, _backend("te"))


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_attention_dropout_is_training_only_and_keeps_gradients_finite(backend):
    torch.manual_seed(107)
    config = _config()
    config.attention_dropout = 0.5
    layer = DeepseekV41Attention(config, 0, _backend(backend))
    inputs = torch.randn(2, 8, config.hidden_size, requires_grad=True)
    kwargs = dict(position_ids=torch.arange(8)[None], state=DeepseekV41AttentionState())
    layer.eval()
    expected = layer(inputs, **kwargs).hidden_states
    torch.testing.assert_close(layer(inputs, **kwargs).hidden_states, expected, atol=0, rtol=0)
    layer.train()
    torch.manual_seed(108)
    actual = layer(inputs, **kwargs).hidden_states
    assert not torch.equal(actual, expected)
    torch.manual_seed(108)
    torch.testing.assert_close(layer(inputs, **kwargs).hidden_states, actual, atol=0, rtol=0)
    actual.square().sum().backward()
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert layer.sinks_param.weight.grad is not None
    assert torch.isfinite(layer.sinks_param.weight.grad).all()
    layer.eval()
    torch.testing.assert_close(layer(inputs, **kwargs).hidden_states, expected, atol=0, rtol=0)


def test_tilelang_requires_zero_dropout_and_attention_requires_supported_projections():
    config = _config()
    config.attention_dropout = 0.1
    with pytest.raises(ValueError, match="attention_dropout=0"):
        DeepseekV41Attention(config, 0, _backend("tilelang"))
    config.attention_dropout = 0
    with pytest.raises(ValueError, match="torch linear"):
        DeepseekV41Attention(config, 0, replace(_backend(), linear="te"))
    with pytest.raises(ValueError, match="torch_fp32"):
        DeepseekV41Attention(config, 0, replace(_backend(), rms_norm="torch"))


@pytest.mark.parametrize("batch,sequence", [(2, 6), (1, 5)])
def test_reuse_rejects_state_from_another_batch_or_sequence(batch, sequence):
    config = _config()
    full = DeepseekV41Attention(config, 1, _backend())
    reuse = DeepseekV41Attention(config, 2, _backend())
    source = full(
        torch.randn(1, 6, config.hidden_size),
        position_ids=torch.arange(6)[None],
        state=DeepseekV41AttentionState(),
    )
    with pytest.raises(ValueError, match="different batch or sequence"):
        reuse(
            torch.randn(batch, sequence, config.hidden_size),
            position_ids=torch.arange(sequence)[None],
            state=source.state,
        )
    with pytest.raises(ValueError, match="same compression ratio"):
        reuse(
            torch.randn(1, 6, config.hidden_size),
            position_ids=torch.arange(6)[None],
            state=replace(source.state, compression_ratio=1),
        )
    with pytest.raises(FrozenInstanceError):
        source.state.compression_ratio = 1


def test_indexer_rejects_missing_latent_keys_and_candidates():
    config = _config()
    x = torch.randn(1, 4, config.hidden_size)
    kwargs = dict(
        query_latent=torch.randn(1, 4, config.q_lora_rank),
        latent=None,
        angles=torch.zeros(1, 4, config.qk_rope_head_dim // 2),
        compressed_angles=torch.zeros(1, 3, config.qk_rope_head_dim // 2),
    )
    full = _Indexer(config, layer_idx=1, dtype=torch.float32)
    reindex = _Indexer(config, layer_idx=4, dtype=torch.float32)
    with pytest.raises(ValueError, match="unrotated compressed latent"):
        full(x, **kwargs, state=DeepseekV41AttentionState())
    with pytest.raises(ValueError, match="index keys"):
        reindex(x, **kwargs, state=DeepseekV41AttentionState())
    state = DeepseekV41AttentionState(compression_ratio=1, index_keys=torch.randn(1, 3, config.index_head_dim))
    with pytest.raises(ValueError, match="requires candidates"):
        reindex(x, **kwargs, state=state)
    with pytest.raises(ValueError, match="requires candidates"):
        reindex(x, **kwargs, state=replace(state, candidates=torch.ones(1, 4, 2, dtype=torch.bool)))


def test_candidate_blocks_with_no_visible_keys():
    scores = torch.full((2, 3, 7), -torch.inf)
    lengths = torch.zeros(2, 3, 1, dtype=torch.long)
    assert not _select_candidate_blocks(scores, lengths, topk_blocks=2, block_size=4).any()


def _arithmetic_config(**overrides):
    values = _config().to_dict()
    # HF serializes a canonical rope_parameters field alongside the legacy
    # rope_scaling alias; do not let the old canonical value mask an override.
    if "rope_scaling" in overrides:
        values.pop("rope_parameters", None)
    values.update(head_dim=64, index_head_dim=32, qk_rope_head_dim=16, rms_norm_eps=1e-6, hc_sinkhorn_iters=4)
    values.update(overrides)
    return DeepseekV41TextConfig(**values)


def _reference_frequencies(positions, dim, theta, scaling):
    frequencies = 1 / theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    if scaling:
        original = scaling["original_max_position_embeddings"]
        low = max(
            math.floor(dim * math.log(original / (scaling["beta_fast"] * 2 * math.pi)) / (2 * math.log(theta))), 0
        )
        high = min(
            math.ceil(dim * math.log(original / (scaling["beta_slow"] * 2 * math.pi)) / (2 * math.log(theta))), dim - 1
        )
        ramp = ((torch.arange(dim // 2).float() - low) / max(high - low, 1e-3)).clamp(0, 1)
        frequencies = frequencies / scaling["factor"] * ramp + frequencies * (1 - ramp)
    angles = positions.float().unsqueeze(-1) * frequencies
    return torch.polar(torch.ones_like(angles), angles)


@pytest.mark.parametrize("use_yarn", [False, True])
@pytest.mark.parametrize("heads", [None, 3])
def test_rotary_matches_complex_reference_after_bf16_cast_and_backward(use_yarn, heads):
    torch.manual_seed(94)
    config = _arithmetic_config(
        rope_scaling={
            "factor": 4,
            "original_max_position_embeddings": 64,
            "beta_fast": 32,
            "beta_slow": 1,
        }
    )
    scaling = config.rope_scaling if use_yarn else None
    theta = config.compress_rope_theta if use_yarn else config.rope_theta
    module = _RotaryEmbedding(config, compressed=use_yarn).bfloat16()
    positions = torch.tensor([[0, 1, 63, 64, 257, 4096], [0, 3, 0, 1, 9, 65536]])
    shape = (2, 6, 64) if heads is None else (2, 6, heads, 64)
    x = torch.randn(shape, dtype=torch.bfloat16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    phases = _reference_frequencies(positions, 16, theta, scaling)
    angles = module(positions)
    assert angles.dtype == torch.float32
    actual_phases = torch.polar(torch.ones_like(angles), angles)
    torch.testing.assert_close(actual_phases, phases, rtol=0, atol=0)
    if heads is not None:
        phases = phases.unsqueeze(2)
    for inverse in (False, True):
        actual = _apply_rope(x, angles, inverse=inverse)
        pairs = torch.view_as_complex(reference_x[..., -16:].float().reshape(*shape[:-1], 8, 2))
        rotated = torch.view_as_real(pairs * (phases.conj() if inverse else phases)).flatten(-2).bfloat16()
        expected = torch.cat([reference_x[..., :-16], rotated], -1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        upstream = torch.randn_like(actual)
        actual.backward(upstream, retain_graph=True)
        expected.backward(upstream, retain_graph=True)
        torch.testing.assert_close(x.grad, reference_x.grad, rtol=0, atol=0)
        x.grad = reference_x.grad = None


@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_cast_boundary_and_all_gradients_match_literal_projection(ratio):
    torch.manual_seed(73)
    config = _arithmetic_config(dtype="bfloat16")
    module = _Compressor(config, ratio=ratio, dtype=torch.bfloat16)
    # Strict FSDP storage can promote the ratio-1 projection while its compute
    # must follow the BF16 activation. Ordinary norm weight stays BF16.
    module.wkv.float()
    reference = {name: p.detach().clone().requires_grad_() for name, p in module.named_parameters()}
    x = torch.randn(2, 5, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    if ratio == 1:
        latent = F.linear(reference_x, reference["wkv.weight"].bfloat16())
    else:
        chunks = reference_x[:, :4].float().reshape(2, 2, 2, config.hidden_size)
        kv = F.linear(chunks, reference["wkv.weight"])
        logits = F.linear(chunks, reference["wgate.weight"])
        latent = (kv * logits.softmax(2)).sum(2).bfloat16()
    expected = F.rms_norm(
        latent.float(), (config.head_dim,), reference["norm.weight"].float(), config.rms_norm_eps
    ).bfloat16()
    actual = module(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(x.grad, reference_x.grad, atol=0.015625, rtol=0.015625)
    if ratio == 2:
        assert not x.grad[:, -1].any()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference[name].grad, atol=0.0625, rtol=0.02)


def test_indexer_uses_bf16_scores_and_returns_sorted_positions_with_empty_queries():
    torch.manual_seed(83)
    config = _arithmetic_config(dtype="bfloat16", index_topk=3)
    module = _Indexer(config, layer_idx=4, dtype=torch.bfloat16)
    x = torch.randn(2, 8, config.hidden_size, dtype=torch.bfloat16)
    qr = torch.randn(2, 8, config.q_lora_rank, dtype=torch.bfloat16)
    # Identity RoPE isolates projection, mandatory query QAT, BF16 score
    # boundaries, causal/candidate visibility, and frozen ownership.
    angles = torch.zeros(2, 8, config.qk_rope_head_dim // 2)
    keys = torch.randn(2, 7, config.index_head_dim, dtype=torch.bfloat16)
    compressed_valid = torch.ones(2, 7, dtype=torch.bool)
    compressed_valid[:, 0] = False
    candidates = torch.zeros(2, 8, 7, dtype=torch.bool)
    candidates[:, :, [0, 2, 4, 6]] = True
    state = DeepseekV41AttentionState(
        compression_ratio=1,
        index_keys=keys,
        compressed_valid=compressed_valid,
        candidates=candidates,
    )
    actual = module(x, query_latent=qr, latent=None, angles=angles, compressed_angles=angles[:, :7], state=state)
    q = F.linear(qr, module.wq_b.weight).reshape(2, 8, config.index_n_heads, config.index_head_dim)
    # Independent nearest-grid rounding, with exact frexp scale selection.
    quantized = torch.empty_like(q)
    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    for output, row in zip(quantized.reshape(-1, 32), q.reshape(-1, 32)):
        scale = _independent_mx_scale(max(float(row.abs().max()), 6 * 2.0**-126), 6)
        values = []
        for value in (row.float() / scale).tolist():
            index = min(range(len(grid)), key=lambda i: (abs(abs(value) - grid[i]), i % 2))
            values.append(math.copysign(grid[index] * scale, value))
        output.copy_(torch.tensor(values, dtype=torch.bfloat16))
    weights = F.linear(x, module.weights_proj.weight) * (config.index_head_dim**-0.5 * config.index_n_heads**-0.5)
    scores = torch.einsum("bshd,btd->bsht", quantized, keys).relu()
    scores = (scores * weights[..., None]).sum(2)
    allowed = torch.arange(7)[None, None, :] < torch.arange(1, 9)[None, :, None]
    scores = scores.masked_fill(~(allowed & compressed_valid[:, None, :] & candidates), -torch.inf)
    selected = scores.topk(3, sorted=False).indices.sort(-1).values
    expected = torch.where(torch.isfinite(scores.gather(-1, selected)), selected, -1)
    torch.testing.assert_close(actual.topk_indices, expected, rtol=0, atol=0)
    assert actual.candidates is candidates
    assert actual.index_keys is keys
    assert not actual.topk_indices[:, 0].ge(0).any()
    assert state.topk_indices is None
    assert all(not parameter.requires_grad for parameter in module.parameters())
