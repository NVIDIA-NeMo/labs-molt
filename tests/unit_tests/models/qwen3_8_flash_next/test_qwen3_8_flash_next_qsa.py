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

"""Focused CPU tests for Qwen3.8-Flash-Next compressed QSA and sparse GQA."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next import layers as qwen3_8_flash_next_layers
from nemo_automodel.components.models.qwen3_8_flash_next import qsa as qwen3_8_flash_next_qsa
from nemo_automodel.components.models.qwen3_8_flash_next.backend import Qwen3_8_FlashNextBackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from nemo_automodel.components.models.qwen3_8_flash_next.flex_qsa import (
    _membership_flat_offset,
    _routes_to_membership,
    flex_sparse_gqa_attention,
)
from nemo_automodel.components.models.qwen3_8_flash_next.layers import Qwen3_8_FlashNextQSAAttention
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import (
    Qwen3_8_FlashNextQSAIndexer,
    gathered_qsa_gqa_attention,
    select_qsa_token_ids,
)


def _config(*, token_budget: int = 8, compress_ratio: int = 2) -> Qwen3_8_FlashNextTextConfig:
    return Qwen3_8_FlashNextTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["full_attention"],
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=2,
        ple_layer_ids=[],
        indexer_budget=token_budget,
        indexer_compress_ratio=compress_ratio,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_theta": 10000.0,
            "rope_type": "default",
            "partial_rotary_factor": 1.0,
        },
        partial_rotary_factor=1.0,
        dtype="float32",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
    )


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


def _freqs(batch_size: int, sequence_length: int, rotary_width: int = 4) -> torch.Tensor:
    positions = torch.arange(sequence_length, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_width, 2).float() / rotary_width))
    angles = torch.outer(positions, inv_freq)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).unsqueeze(0).expand(batch_size, -1, -1)


def _dense_causal_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """Small loop reference whose accumulation matches the sparse helper's fp32 contract."""
    output = torch.zeros_like(query)
    groups = query.shape[2] // key.shape[2]
    scale = query.shape[-1] ** -0.5
    for batch_idx, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        for query_idx in range(length):
            for query_head in range(query.shape[2]):
                kv_head = query_head // groups
                scores = torch.mv(
                    key[batch_idx, : query_idx + 1, kv_head].float(),
                    query[batch_idx, query_idx, query_head].float(),
                )
                probabilities = torch.softmax(scores * scale, dim=0)
                output[batch_idx, query_idx, query_head] = torch.mv(
                    value[batch_idx, : query_idx + 1, kv_head].float().transpose(0, 1),
                    probabilities,
                ).to(output.dtype)
    return output


def test_qsa_at_or_below_budget_is_dense_equivalent() -> None:
    generator = torch.Generator().manual_seed(1234)
    batch_size, sequence_length = 2, 7
    lengths = torch.tensor([7, 5])
    index_query = torch.randn(batch_size, sequence_length, 2, 3, generator=generator)
    compressed_key = torch.randn(batch_size, sequence_length // 2, 1, 3, generator=generator)
    selected = select_qsa_token_ids(
        index_query,
        compressed_key,
        lengths,
        token_budget=8,
        compress_ratio=2,
        query_chunk_size=3,
    )

    for batch_idx, length_tensor in enumerate(lengths):
        length = int(length_tensor)
        for query_idx in range(length):
            actual = selected[batch_idx, query_idx]
            assert set(actual[actual >= 0].tolist()) == set(range(query_idx + 1))
        assert bool((selected[batch_idx, length:] == -1).all())

    query = torch.randn(batch_size, sequence_length, 4, 3, generator=generator)
    key = torch.randn(batch_size, sequence_length, 2, 3, generator=generator)
    value = torch.randn(batch_size, sequence_length, 2, 3, generator=generator)
    sparse = gathered_qsa_gqa_attention(query, key, value, selected)
    dense = _dense_causal_gqa(query, key, value, lengths)
    torch.testing.assert_close(sparse, dense, rtol=1e-5, atol=1e-6)


def test_qsa_gold_hand_vector_relu_topk_and_tail() -> None:
    index_query = torch.ones(1, 6, 1, 1)
    index_query[:, 5] = -1.0
    compressed_key = torch.tensor([[[[-1.0]], [[2.0]], [[1.0]]]])
    selected = select_qsa_token_ids(
        index_query,
        compressed_key,
        torch.tensor([6]),
        token_budget=2,
        compress_ratio=2,
        query_chunk_size=2,
    )

    expected = torch.tensor(
        [
            [0, -1, -1],
            [0, 1, -1],
            [0, 1, 2],
            [2, 3, -1],
            [2, 3, 4],
            [0, 1, -1],
        ],
        dtype=torch.int32,
    )
    torch.testing.assert_close(selected[0], expected)


def test_qsa_first_sparse_query_is_position_2051() -> None:
    # c4/budget=2048 has 512 block slots. Position 2050 still sees exactly
    # 512 complete blocks plus a three-token tail; position 2051 sees 513
    # complete blocks and must drop the lowest-scoring block.
    sequence_length = 2052
    index_query = torch.ones(1, sequence_length, 1, 1)
    compressed_key = torch.arange(sequence_length // 4, dtype=torch.float32).view(1, -1, 1, 1)
    selected = select_qsa_token_ids(
        index_query,
        compressed_key,
        torch.tensor([sequence_length]),
        token_budget=2048,
        compress_ratio=4,
        query_chunk_size=256,
    )

    before = selected[0, 2050]
    assert int((before >= 0).sum()) == 2051
    torch.testing.assert_close(before[:2051], torch.arange(2051, dtype=torch.int32))

    first_sparse = selected[0, 2051]
    assert int((first_sparse >= 0).sum()) == 2048
    assert set(first_sparse[first_sparse >= 0].tolist()) == set(range(4, 2052))


def test_qsa_indexer_supports_only_right_tail_padding() -> None:
    config = _config(token_budget=4, compress_ratio=2)
    indexer = Qwen3_8_FlashNextQSAIndexer(config, _backend())
    indexer.init_weights()
    hidden_states = torch.randn(2, 5, config.hidden_size)
    attention_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    selected = indexer(hidden_states, freqs_cis=_freqs(2, 5), attention_mask=attention_mask)

    assert selected.shape == (2, 5, 5)
    assert set(selected[1, 2][selected[1, 2] >= 0].tolist()) == {0, 1, 2}
    assert bool((selected[1, 3:] == -1).all())

    interior_padding = torch.tensor([[1, 0, 1, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="right-tail padding"):
        indexer(hidden_states, freqs_cis=_freqs(2, 5), attention_mask=interior_padding)


def test_qsa_main_qkv_backward_and_frozen_hookable_indexer() -> None:
    torch.manual_seed(11)
    config = _config(token_budget=4, compress_ratio=2)
    attention = Qwen3_8_FlashNextQSAAttention(config, layer_idx=0, backend=_backend())
    attention.init_weights(torch.device("cpu"))
    attention.train()
    captured: list[torch.Tensor] = []
    handle = attention.indexer.register_forward_hook(lambda _module, _args, output: captured.append(output))

    hidden_states = torch.randn(2, 6, config.hidden_size, requires_grad=True)
    output = attention(
        hidden_states,
        freqs_cis=_freqs(2, 6),
        attention_mask=torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]]),
    )
    output.square().mean().backward()
    handle.remove()

    assert captured[0].shape == (2, 6, config.indexer_budget + config.indexer_compress_ratio - 1)
    assert captured[0].dtype == torch.int32
    for projection in (attention.q_proj, attention.k_proj, attention.v_proj, attention.o_proj):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert torch.count_nonzero(projection.weight.grad) > 0
    assert hidden_states.grad is not None
    assert all(not parameter.requires_grad for parameter in attention.indexer.parameters())
    assert all(parameter.grad is None for parameter in attention.indexer.parameters())


def test_qsa_attention_trims_fixed_routing_width_for_short_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(token_budget=8, compress_ratio=2)
    attention = Qwen3_8_FlashNextQSAAttention(config, layer_idx=0, backend=_backend())
    attention.init_weights(torch.device("cpu"))
    full_routing_shapes: list[tuple[int, ...]] = []
    compute_routing_shapes: list[tuple[int, ...]] = []
    handle = attention.indexer.register_forward_hook(
        lambda _module, _args, output: full_routing_shapes.append(tuple(output.shape))
    )
    original_attention = qwen3_8_flash_next_layers.qsa_gqa_attention

    def capture_compute_width(*args, **kwargs):
        compute_routing_shapes.append(tuple(args[3].shape))
        return original_attention(*args, **kwargs)

    monkeypatch.setattr(qwen3_8_flash_next_layers, "qsa_gqa_attention", capture_compute_width)
    hidden_states = torch.randn(1, 3, config.hidden_size)
    output = attention(hidden_states, freqs_cis=_freqs(1, 3), attention_mask=torch.ones(1, 3))
    handle.remove()

    assert output.shape == hidden_states.shape
    assert full_routing_shapes == [(1, 3, 9)]
    assert compute_routing_shapes == [(1, 3, 3)]


def test_qsa_attention_rejects_empty_sequences_clearly() -> None:
    config = _config()
    attention = Qwen3_8_FlashNextQSAAttention(config, layer_idx=0, backend=_backend())

    with pytest.raises(ValueError, match="non-empty sequence"):
        attention(
            torch.empty(1, 0, config.hidden_size),
            freqs_cis=torch.empty(1, 0, config.head_dim),
        )


def test_frozen_indexer_survives_meta_materialization_and_checkpoint_load() -> None:
    from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only

    with torch.device("meta"):
        indexer = Qwen3_8_FlashNextQSAIndexer(_config(), _backend())
    assert all(parameter.is_meta for parameter in indexer.parameters())
    assert all(not parameter.requires_grad for parameter in indexer.parameters())

    to_empty_parameters_only(indexer, device=torch.device("cpu"))
    indexer.init_weights()
    checkpoint_state = {
        name: torch.full_like(value, fill_value=(tensor_idx + 1) / 16)
        for tensor_idx, (name, value) in enumerate(indexer.state_dict().items())
    }
    indexer.load_state_dict(checkpoint_state, strict=True)

    assert all(not parameter.is_meta for parameter in indexer.parameters())
    assert all(not parameter.requires_grad for parameter in indexer.parameters())
    for name, value in indexer.state_dict().items():
        torch.testing.assert_close(value, checkpoint_state[name])


def test_gathered_qsa_oracle_handles_duplicate_and_padding_ids() -> None:
    """The CPU oracle keeps duplicate-ID gradients and erases empty rows."""
    torch.manual_seed(41)
    selected = torch.tensor(
        [
            [
                [-1, -1, -1, -1],
                [0, 0, -1, -1],
                [2, 1, 2, -1],
                [0, 3, 3, 1],
            ]
        ],
        dtype=torch.int32,
    )
    inputs = [
        torch.randn(1, 4, 4, 3, requires_grad=True),
        torch.randn(1, 4, 2, 3, requires_grad=True),
        torch.randn(1, 4, 2, 3, requires_grad=True),
    ]
    output = gathered_qsa_gqa_attention(*inputs, selected)
    assert torch.count_nonzero(output[:, 0]) == 0

    output.backward(torch.randn_like(output))
    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()

    padding_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    padding_output = gathered_qsa_gqa_attention(*padding_inputs, selected)
    padding_output[:, 0].sum().backward()
    for padding_input in padding_inputs:
        assert padding_input.grad is not None
        assert torch.count_nonzero(padding_input.grad) == 0


def test_flex_membership_gives_empty_rows_a_kernel_safe_dummy_route() -> None:
    selected = torch.tensor(
        [[[-1, -1, -1], [0, 0, -1], [2, 1, 2], [4, -2, -1]]],
        dtype=torch.int32,
    )

    membership, has_routes = _routes_to_membership(selected, kv_length=4)

    torch.testing.assert_close(has_routes, torch.tensor([[False, True, True, False]]))
    torch.testing.assert_close(
        membership,
        torch.tensor(
            [
                [
                    [True, False, False, False],
                    [True, False, False, False],
                    [False, True, True, False],
                    [True, False, False, False],
                ]
            ]
        ),
    )
    assert bool(membership.any(dim=-1).all())


def test_flex_membership_offset_is_exact_past_int32_range() -> None:
    # A 65536-token square membership table holds 2**32 entries, so the last
    # rows sit beyond INT32_MAX. Python ints are the arbitrary-precision
    # reference: an int32 evaluation would wrap negative here instead.
    query_length = kv_length = 65536
    batch_idx, query_idx, kv_idx = 0, 65535, 65535

    offset = _membership_flat_offset(
        torch.tensor(batch_idx),
        torch.tensor(query_idx),
        torch.tensor(kv_idx),
        query_length,
        kv_length,
    )

    assert offset.dtype == torch.int64
    assert int(offset) == (batch_idx * query_length + query_idx) * kv_length + kv_idx
    assert int(offset) > torch.iinfo(torch.int32).max


def test_flex_membership_offset_matches_direct_indexing() -> None:
    membership = torch.rand(2, 5, 7) > 0.5
    flat = membership.reshape(-1)

    for batch_idx in range(membership.shape[0]):
        for query_idx in range(membership.shape[1]):
            for kv_idx in range(membership.shape[2]):
                offset = _membership_flat_offset(
                    torch.tensor(batch_idx),
                    torch.tensor(query_idx),
                    torch.tensor(kv_idx),
                    membership.shape[1],
                    membership.shape[2],
                )
                assert bool(flat[offset]) == bool(membership[batch_idx, query_idx, kv_idx])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention requires CUDA")
def test_flex_qsa_empty_route_rows_have_zero_output_and_gradients() -> None:
    torch.manual_seed(43)
    inputs = [
        torch.randn(1, 3, 4, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(1, 3, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True),
        torch.randn(1, 3, 2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True),
    ]
    selected = torch.tensor(
        [[[-1, -1, -1], [0, 1, -1], [0, 1, 2]]],
        device="cuda",
        dtype=torch.int32,
    )

    output = flex_sparse_gqa_attention(*inputs, selected)

    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[:, 0]) == 0
    output[:, 0].float().sum().backward()
    for tensor in inputs:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) == 0


@pytest.mark.parametrize("attn_backend", ["flex", "cute"])
def test_qsa_sparse_backend_bypasses_generic_parent_initializer(attn_backend: str) -> None:
    backend = Qwen3_8_FlashNextBackendConfig(
        attn=attn_backend,
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )

    attention = Qwen3_8_FlashNextQSAAttention(_config(), layer_idx=0, backend=backend)

    assert attention.backend is backend
    assert attention.backend.attn == attn_backend
    assert attention.attn_module is None
    assert attention.attn_func is None


def test_qsa_flex_backend_uses_cpu_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    query = torch.randn(1, 3, 4, 3)
    key = torch.randn(1, 3, 2, 3)
    value = torch.randn(1, 3, 2, 3)
    selected = torch.tensor([[[0, -1], [0, 1], [0, 2]]], dtype=torch.int32)
    expected = gathered_qsa_gqa_attention(query, key, value, selected)

    def fail_if_called(*_args, **_kwargs):
        pytest.fail("CPU QSA must not call the FlexAttention kernel")

    monkeypatch.setattr(qwen3_8_flash_next_qsa, "flex_sparse_gqa_attention", fail_if_called)
    actual = qwen3_8_flash_next_qsa.qsa_gqa_attention(query, key, value, selected, backend="flex")

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
