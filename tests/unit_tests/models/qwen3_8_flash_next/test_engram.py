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

"""Exact CPU and two-rank gradient tests for Qwen3.8-Flash-Next PLE."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
    QWEN3_8_FLASH_NEXT_LAYER_MULTIPLIERS,
    QWEN3_8_FLASH_NEXT_NGRAM_HEAD_OFFSETS,
    QWEN3_8_FLASH_NEXT_NGRAM_HEAD_VOCAB_SIZES,
    Qwen3_8_FlashNextEngramTableConfig,
    Qwen3_8_FlashNextNGramEmbedding,
    Qwen3_8_FlashNextPLELayer,
)

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


class _RowIdLookup(nn.Module):
    """Expose requested global rows as one-dimensional integer values."""

    def forward(self, global_ids: torch.Tensor) -> torch.Tensor:
        """Append a value dimension without changing row IDs.

        Args:
            global_ids: Integer tensor of shape
                ``[batch, sequence, ngram_heads]``.

        Returns:
            Integer tensor of shape ``[batch, sequence, ngram_heads, 1]``.
        """
        return global_ids.unsqueeze(-1)


def _tiny_ngram_embedding(table: nn.Module) -> Qwen3_8_FlashNextNGramEmbedding:
    return Qwen3_8_FlashNextNGramEmbedding(
        table,
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=99,
        layer_multipliers=(3, 5, 7),
        ngram_heads_vocab_sizes=(5, 7, 11, 13),
        ngram_heads_offsets=(0, 5, 12, 23),
    )


def _parallelize_owner_table(table: nn.Module) -> None:
    """Wrap one already-local owner shard in the current CPU world mesh."""
    mesh = DeviceMesh.from_group(
        dist.group.WORLD,
        device_type="cpu",
        mesh_dim_names=("dp_shard_cp",),
    )
    table.parallelize_weight(mesh)


def _tiny_ple() -> Qwen3_8_FlashNextPLELayer:
    table = nn.Embedding(36, 2)
    ple = Qwen3_8_FlashNextPLELayer(
        _tiny_ngram_embedding(table),
        hidden_size=2,
        hc_count=2,
        ple_embed_dim=8,
        backend=BackendConfig(linear="torch"),
        dtype=torch.float32,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
    )
    with torch.no_grad():
        table.weight.copy_(torch.arange(72, dtype=torch.float32).view(36, 2) / 25 - 1)
        ple.key_proj.weight.copy_(torch.arange(32, dtype=torch.float32).view(4, 8) / 50 - 0.25)
        ple.value_proj.weight.copy_(torch.arange(16, dtype=torch.float32).view(2, 8) / 40 - 0.2)
        ple.norm_key.weight.copy_(torch.linspace(-0.15, 0.15, 4))
        ple.norm_query.weight.copy_(torch.linspace(0.1, -0.1, 4))
        ple.norm_conv.weight.copy_(torch.linspace(-0.2, 0.2, 4))
        ple.conv1d.weight.copy_(
            torch.tensor(
                [
                    [[0.2, -0.1, 0.05, 0.3]],
                    [[-0.2, 0.15, 0.1, -0.05]],
                    [[0.05, 0.1, -0.15, 0.2]],
                    [[0.3, -0.2, 0.1, 0.05]],
                ]
            )
        )
    return ple


def test_checkpoint_hash_fixture_uses_raw_ids_and_resets_after_eos() -> None:
    embedding = Qwen3_8_FlashNextNGramEmbedding(_RowIdLookup())
    input_ids = torch.tensor([[10, 11, 248044, 12, 13]])

    actual = embedding(input_ids).squeeze(-1)

    # Frozen from the reference SGLang implementation with the three buffers
    # loaded from brightdelta-180b-bf16_vv1. In particular, token 12 cannot see
    # token 11 across the preceding EOS boundary.
    expected = torch.tensor(
        [
            [
                [
                    6826666,
                    27775725,
                    51991156,
                    74082527,
                    82622748,
                    119600976,
                    135816374,
                    152166807,
                    174244281,
                    190221032,
                    211723794,
                    232787707,
                    243645790,
                    275729718,
                    280030017,
                    303574322,
                ],
                [
                    2940504,
                    22060242,
                    51620224,
                    61004244,
                    80476372,
                    100300474,
                    130036651,
                    149684858,
                    176155633,
                    188444023,
                    204598098,
                    221538224,
                    251457628,
                    261736411,
                    285242727,
                    303372722,
                ],
                [
                    4679147,
                    20352667,
                    52599150,
                    74683044,
                    98198172,
                    106977280,
                    121027876,
                    141408284,
                    169789359,
                    198612961,
                    210924170,
                    220705730,
                    246230035,
                    272812674,
                    295242179,
                    309152779,
                ],
                [
                    10272144,
                    38375960,
                    56169038,
                    77269436,
                    82103390,
                    104512879,
                    128875317,
                    156088582,
                    166704986,
                    181067027,
                    214019790,
                    236371812,
                    246423428,
                    277700973,
                    295065937,
                    303933063,
                ],
                [
                    2372911,
                    26244889,
                    58181291,
                    70892635,
                    93217067,
                    113992001,
                    125154417,
                    146704472,
                    177001882,
                    191203173,
                    206256009,
                    222909216,
                    243785810,
                    265022014,
                    285514102,
                    308573248,
                ],
            ]
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert tuple(embedding.layer_multipliers.tolist()) == QWEN3_8_FLASH_NEXT_LAYER_MULTIPLIERS
    assert tuple(embedding.ngram_heads_vocab_sizes.tolist()) == QWEN3_8_FLASH_NEXT_NGRAM_HEAD_VOCAB_SIZES
    assert tuple(embedding.ngram_heads_offsets.tolist()) == QWEN3_8_FLASH_NEXT_NGRAM_HEAD_OFFSETS


def test_tiny_hash_fixture_proves_packed_head_offsets() -> None:
    embedding = _tiny_ngram_embedding(_RowIdLookup())

    actual = embedding(torch.tensor([[10, 11, 99, 12, 13]])).squeeze(-1)

    expected = torch.tensor(
        [
            [
                [2, 5, 12, 27],
                [4, 10, 19, 25],
                [1, 11, 15, 29],
                [4, 9, 15, 33],
                [2, 11, 16, 33],
            ]
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_ple_matches_frozen_reference_and_is_causal() -> None:
    ple = _tiny_ple()
    input_ids = torch.tensor([[10, 11, 99, 12, 13]])
    hidden_states = torch.tensor(
        [
            [
                [0.2, -0.3, 0.5, 0.7],
                [-0.1, 0.4, 0.8, -0.6],
                [0.9, 0.1, -0.2, 0.3],
                [0.4, -0.8, 0.6, 0.2],
                [-0.5, 0.3, 0.7, -0.4],
            ]
        ],
        requires_grad=True,
    )

    actual = ple(hidden_states, input_ids)

    # Frozen from the reference equations: Gemma (1 + weight) branch norms,
    # signed-sqrt gate, shared V, and causal depthwise k=4/dilation=3 conv.
    expected = torch.tensor(
        [
            [
                [0.311479509, 0.098643340, 0.427124828, 0.245068356],
                [0.150163054, 0.287197471, 0.099930868, 0.216401696],
                [0.236315131, 0.276813090, 0.226355195, 0.344233662],
                [0.112979680, 0.138172656, 0.071468778, 0.502847791],
                [0.172592908, 0.449751854, 0.097367570, 0.461553633],
            ]
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    changed_future_ids = input_ids.clone()
    changed_future_ids[0, -1] = 17
    changed_future = ple(hidden_states, changed_future_ids)
    torch.testing.assert_close(changed_future[:, :-1], actual[:, :-1], rtol=0, atol=0)

    actual.square().sum().backward()
    assert hidden_states.grad is not None
    assert torch.isfinite(hidden_states.grad).all()
    assert ple.ple_embedding.ngram_embedding.weight.grad is not None
    assert torch.isfinite(ple.ple_embedding.ngram_embedding.weight.grad).all()


def test_ple_rejects_collapsed_hidden_state() -> None:
    ple = _tiny_ple()
    hidden_states = torch.randn(1, 3, 2)

    with pytest.raises(ValueError, match="full HyperConnection state"):
        ple(hidden_states, torch.tensor([[1, 2, 3]]))


def test_ple_uses_explicit_model_dtype_and_zero_start_conv() -> None:
    table = nn.Embedding(36, 2, dtype=torch.bfloat16)
    ple = Qwen3_8_FlashNextPLELayer(
        _tiny_ngram_embedding(table),
        hidden_size=2,
        hc_count=2,
        ple_embed_dim=8,
        backend=BackendConfig(linear="torch"),
        dtype=torch.bfloat16,
        conv_kernel_size=4,
    )

    assert ple.key_proj.weight.dtype == torch.bfloat16
    assert ple.value_proj.weight.dtype == torch.bfloat16
    assert ple.conv1d.weight.dtype == torch.bfloat16
    torch.testing.assert_close(ple.conv1d.weight, torch.zeros_like(ple.conv1d.weight))


def test_local_engram_table_matches_torch_embedding_forward_and_backward() -> None:
    table = Qwen3_8_FlashNextEngramTableConfig(
        num_embeddings=12,
        embedding_dim=3,
        initializer_range=0.0,
    ).build(process_group=None, dtype=torch.float32)
    assert table.global_row_start == table.local_row_start == 0
    assert table.global_row_end == table.local_row_end == 12
    assert not hasattr(table.weight, "_nemo_model_owned_grad_divisor")
    reference_weight = torch.arange(36, dtype=torch.float32).view(12, 3) / 10
    table.weight.detach().copy_(reference_weight)
    global_ids = torch.tensor([[0, 7, 7], [11, 1, 6]])
    upstream = torch.linspace(-0.4, 0.7, 18).view(2, 3, 3)

    actual = table(global_ids)
    expected_weight = reference_weight.clone().requires_grad_(True)
    expected = F.embedding(global_ids, expected_weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(table.weight.grad, expected_weight.grad, rtol=0, atol=0)


def _owner_sharded_gradient_worker(rank: int, world_size: int, store_path: str) -> None:
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
        )
        config = Qwen3_8_FlashNextEngramTableConfig(
            num_embeddings=12,
            embedding_dim=3,
            initializer_range=0.0,
        )
        table = config.build(process_group=dist.group.WORLD, dtype=torch.float32)
        _parallelize_owner_table(table)
        assert isinstance(table.weight, DTensor)
        assert tuple(table.weight.shape) == (12, 3)
        assert tuple(table.weight.placements) == (Shard(0),)
        full_weight = torch.arange(36, dtype=torch.float32).view(12, 3) / 10
        table.weight.to_local().detach().copy_(full_weight[table.vocab_start_index : table.vocab_end_index])
        optimizer = torch.optim.SGD([table.weight], lr=0.1)

        ids_by_rank = (
            # Rows 5/6 pin the exact contiguous-owner boundary, while rows 7
            # and 10 exercise repeated-row gradient accumulation.
            torch.tensor([0, 5, 6, 7, 7, 10, 10, 11]),
            # A completely empty requester exercises zero-length compact
            # input and output segments while rank 1 still owns remote rows.
            torch.tensor([], dtype=torch.long),
        )
        global_ids = ids_by_rank[rank]
        upstream = torch.arange(global_ids.numel() * 3, dtype=torch.float32).view(-1, 3) / 7 + rank * 0.25

        actual = table(global_ids)
        expected_weight = full_weight.clone().requires_grad_(True)
        expected_optimizer = torch.optim.SGD([expected_weight], lr=0.1)
        expected = F.embedding(global_ids, expected_weight)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        actual.backward(upstream)
        expected.backward(upstream)
        dist.all_reduce(expected_weight.grad, group=dist.group.WORLD)
        expected_local_grad = expected_weight.grad[table.vocab_start_index : table.vocab_end_index]
        assert isinstance(table.weight.grad, DTensor)
        torch.testing.assert_close(table.weight.grad.to_local(), expected_local_grad, rtol=0, atol=0)

        optimizer.step()
        expected_optimizer.step()
        torch.testing.assert_close(
            table.weight.to_local(),
            expected_weight[table.vocab_start_index : table.vocab_end_index],
            rtol=0,
            atol=0,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_owner_sharded_table_matches_full_table_gradients(tmp_path: Path) -> None:
    mp.spawn(
        _owner_sharded_gradient_worker,
        args=(2, str(tmp_path / "owner-sharded-pg")),
        nprocs=2,
        join=True,
    )


def _engram_meta_dtensor_lifecycle_worker(rank: int, world_size: int, store_path: str) -> None:
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
        )
        with torch.device("meta"):
            table = Qwen3_8_FlashNextEngramTableConfig(
                num_embeddings=12,
                embedding_dim=3,
                initializer_range=0.02,
            ).build(process_group=dist.group.WORLD, dtype=torch.float32)
        _parallelize_owner_table(table)
        parameter = table.weight
        assert isinstance(parameter, DTensor)
        assert parameter.to_local().device.type == "meta"

        from nemo_automodel.components.checkpoint.checkpointing import to_empty_parameters_only

        to_empty_parameters_only(table, device=torch.device("cpu"))
        assert table.weight is parameter
        assert table.weight.to_local().device.type == "cpu"
        assert tuple(table.weight.shape) == (12, 3)
        assert tuple(table.weight.placements) == (Shard(0),)
        assert not hasattr(table.weight, "_nemo_model_owned_grad_divisor")
        table.reset_parameters()
        table.mark_sharding_contract()
        assert torch.isfinite(table.weight.to_local()).all()
        assert hasattr(table.weight, "_nemo_model_owned_grad_divisor")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_engram_meta_dtensor_materialization_preserves_parameter_identity(tmp_path: Path) -> None:
    mp.spawn(
        _engram_meta_dtensor_lifecycle_worker,
        args=(2, str(tmp_path / "engram-meta-dtensor-pg")),
        nprocs=2,
        join=True,
    )


def _owner_sharded_misroute_worker(rank: int, world_size: int, store_path: str) -> None:
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
        )
        table = Qwen3_8_FlashNextEngramTableConfig(
            num_embeddings=12,
            embedding_dim=3,
            initializer_range=0.0,
        ).build(process_group=dist.group.WORLD, dtype=torch.float32)
        _parallelize_owner_table(table)
        original_exchange_ids = table._exchange_ids

        def _exchange_ids_with_rank_zero_misroute(
            sorted_global_ids: torch.Tensor,
            send_counts: torch.Tensor,
        ) -> tuple[torch.Tensor, tuple[int, ...], tuple[int, ...], int]:
            """Corrupt rank zero's first received ID after the real route.

            Args:
                sorted_global_ids: Tensor of shape ``[request_rows]`` grouped
                    by destination owner.
                send_counts: Tensor of shape ``[owner_world_size]`` with one
                    request count per owner.

            Returns:
                The routed ID tensor of shape ``[owned_requests]`` and its
                send counts, receive counts, and fixed route capacity.
            """
            received_ids, input_splits, output_splits, capacity = original_exchange_ids(sorted_global_ids, send_counts)
            if rank == 0:
                received_ids = received_ids.clone()
                received_ids[0] = table.vocab_end_index
            return received_ids, input_splits, output_splits, capacity

        # Corrupt rank 0's post-All-to-All payload only.  Rank 1 must still
        # enter the same diagnostic collectives and raise the same error.
        table._exchange_ids = _exchange_ids_with_rank_zero_misroute
        ids_by_rank = (
            torch.tensor([0, 6]),
            torch.tensor([5, 11]),
        )
        with pytest.raises(RuntimeError) as exc_info:
            table(ids_by_rank[rank])

        message = str(exc_info.value)
        assert "owner_group_rank=0 (global_rank=0) expected [0, 6)" in message
        assert "received_count=2, received_min=5, received_max=6" in message
        assert "bad_count=1, bad_min=6, bad_max=6, bad_sample=[6]" in message
        assert "source_segments=[source_group_rank=0(size=1,bad=1,untouched=0)]" in message
        assert "owner_group_rank=1" not in message

        # Model an exchange that returns successfully without touching its
        # output buffer.  The -1 fill must turn every slot into a deterministic
        # owner-validation failure, attributed to the correct source segment.
        table._exchange_ids = original_exchange_ids
        original_all_to_all_single = dist.all_to_all_single

        def _leave_output_untouched(*args: object, **kwargs: object) -> None:
            return None

        setattr(dist, "all_to_all_single", _leave_output_untouched)
        try:
            with pytest.raises(RuntimeError) as exc_info:
                table(ids_by_rank[rank])
        finally:
            setattr(dist, "all_to_all_single", original_all_to_all_single)

        message = str(exc_info.value)
        assert "owner_group_rank=0 (global_rank=0) expected [0, 6)" in message
        assert "owner_group_rank=1 (global_rank=1) expected [6, 12)" in message
        assert "received_count=2, received_min=-1, received_max=-1" in message
        assert "bad_count=2, bad_min=-1, bad_max=-1, bad_sample=[-1, -1]" in message
        assert "source_group_rank=0(size=1,bad=1,untouched=1)" in message
        assert "source_group_rank=1(size=1,bad=1,untouched=1)" in message
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_owner_sharded_table_rejects_misrouted_ids_symmetrically(tmp_path: Path) -> None:
    mp.spawn(
        _owner_sharded_misroute_worker,
        args=(2, str(tmp_path / "owner-sharded-misroute-pg")),
        nprocs=2,
        join=True,
    )


def _owner_sharded_route_metadata_worker(rank: int, world_size: int, store_path: str) -> None:
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
        )
        table = Qwen3_8_FlashNextEngramTableConfig(
            num_embeddings=12,
            embedding_dim=3,
            initializer_range=0.0,
        ).build(process_group=dist.group.WORLD, dtype=torch.float32)
        _parallelize_owner_table(table)
        ids_by_rank = (
            torch.tensor([0, 6]),
            torch.tensor([5, 11]),
        )
        global_ids = ids_by_rank[rank]
        owners = torch.div(global_ids, table.num_embeddings_per_rank, rounding_mode="floor")
        send_counts = torch.bincount(owners, minlength=world_size).to(torch.int64)
        sorted_ids = global_ids[torch.argsort(owners, stable=True)]

        # Only rank 0 corrupts one compact destination segment.  The metadata
        # preflight must make both ranks fail before entering payload A2A.
        invalid_sorted_ids = sorted_ids.clone()
        if rank == 0:
            invalid_sorted_ids[0] = table.vocab_end_index
        with pytest.raises(RuntimeError, match="sorted ID segments"):
            table._validate_sorted_send_ids(invalid_sorted_ids, send_counts)

        # Make one rank observe a different but still plausible count matrix.
        # MIN/MAX validation must detect it symmetrically before the matrix is
        # used to size or unpack a payload.
        original_all_gather_into_tensor = dist.all_gather_into_tensor

        def _corrupt_one_count_matrix(*args: object, **kwargs: object) -> object:
            result = original_all_gather_into_tensor(*args, **kwargs)
            if rank == 0:
                output_tensor = args[0]
                assert isinstance(output_tensor, torch.Tensor)
                output_tensor[0] += 1
            return result

        setattr(dist, "all_gather_into_tensor", _corrupt_one_count_matrix)
        try:
            with pytest.raises(RuntimeError, match="count AllGather produced inconsistent route metadata"):
                table._exchange_ids(sorted_ids, send_counts)
        finally:
            setattr(dist, "all_gather_into_tensor", original_all_gather_into_tensor)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_owner_sharded_table_validates_send_segments_and_count_matrix(tmp_path: Path) -> None:
    mp.spawn(
        _owner_sharded_route_metadata_worker,
        args=(2, str(tmp_path / "owner-sharded-route-metadata-pg")),
        nprocs=2,
        join=True,
    )


def test_ple_casts_fp32_owner_table_to_bfloat16_compute() -> None:
    fp32_table = nn.Embedding(36, 2, dtype=torch.float32)
    ple = Qwen3_8_FlashNextPLELayer(
        _tiny_ngram_embedding(fp32_table),
        hidden_size=2,
        hc_count=2,
        ple_embed_dim=8,
        backend=BackendConfig(linear="torch"),
        dtype=torch.bfloat16,
        conv_kernel_size=4,
    )
    reference_table = nn.Embedding(36, 2, dtype=torch.bfloat16)
    reference = Qwen3_8_FlashNextPLELayer(
        _tiny_ngram_embedding(reference_table),
        hidden_size=2,
        hc_count=2,
        ple_embed_dim=8,
        backend=BackendConfig(linear="torch"),
        dtype=torch.bfloat16,
        conv_kernel_size=4,
    )
    with torch.no_grad():
        fp32_table.weight.copy_(fp32_table.weight.to(torch.bfloat16))
        reference.load_state_dict(ple.state_dict())

    input_ids = torch.tensor([[10, 11, 12]])
    hidden_states = torch.randn(1, 3, 4, dtype=torch.bfloat16, requires_grad=True)
    reference_hidden_states = hidden_states.detach().clone().requires_grad_(True)

    actual = ple(hidden_states, input_ids)
    expected = reference(reference_hidden_states, input_ids)

    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(hidden_states.grad, reference_hidden_states.grad, rtol=0, atol=0)
    assert fp32_table.weight.grad is not None
    assert fp32_table.weight.grad.dtype == torch.float32
    assert torch.isfinite(fp32_table.weight.grad).all()
    assert ple.key_proj.weight.grad is not None
    assert ple.key_proj.weight.grad.dtype == torch.bfloat16
    assert torch.isfinite(ple.key_proj.weight.grad).all()
