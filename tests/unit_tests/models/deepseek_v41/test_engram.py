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

"""Engram hashing, residual arithmetic, owner gradients and distributed checkpoints."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn.functional as F
from tokenizers import Tokenizer, models
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import (
    DeepseekV41Config,
    DeepseekV41TextConfig,
    DeepseekV41VisionConfig,
)
from nemo_automodel.components.models.deepseek_v41.engram import DeepseekV41Engram, DeepseekV41NgramHash
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import DeepseekV41StateDictAdapter
from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextOwnerShardedEmbedding
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.shared.multimodal_fsdp import ignored_params_for_root

# Over the default 5s budget on purpose: distributed owner and checkpoint tests spawn Gloo workers.
# Shrink process startup and checkpoint round trips before lowering this further.
pytestmark = pytest.mark.timeout(60)


def _tiny_config() -> DeepseekV41TextConfig:
    return DeepseekV41TextConfig(
        hidden_size=8,
        hc_mult=2,
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[113, 253],
        engram_max_ngram_size=4,
        engram_vocab_size=11,
        engram_n_heads=2,
        engram_head_dim=4,
        engram_pad_token_id=2,
        engram_compressed_vocab_size=6,
        dtype="float32",
        initializer_range=0.1,
    )


def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {"[UNK]": 0, " The": 1, "the": 2, "THE": 3, "é": 4, "E": 5, " ": 6, "x": 7, "\ufffd": 8}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")


def _reference_residual(
    hidden: torch.Tensor,
    hash_ids: torch.Tensor,
    parameters: dict[str, torch.Tensor],
    token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute branch-local normalized memory injection without a fused norm product.

    Args:
        hidden: Tensor of shape [batch, sequence, hc_mult, hidden].
        hash_ids: Integer tensor of shape [batch, sequence, hash_heads].
        parameters: Reference tensors with keys embed.weight [rows, head_dim],
            wkv.weight [(hc_mult + 1) * hidden, hash_heads * head_dim], and
            q_weight/k_weight [hc_mult, hidden].
        token_mask: Boolean tensor of shape [batch, sequence].
        eps: RMS normalization epsilon.

    Returns:
        Tensor of shape [batch, sequence, hc_mult, hidden].
    """
    hidden_size = hidden.shape[-1]
    projected = F.linear(F.embedding(hash_ids, parameters["embed.weight"]).flatten(-2), parameters["wkv.weight"])
    branches = []
    for branch in range(hidden.shape[-2]):
        key = projected[..., branch * hidden_size : (branch + 1) * hidden_size]
        query = F.rms_norm(hidden[:, :, branch], (hidden_size,), parameters["q_weight"][branch], eps)
        key = F.rms_norm(key, (hidden_size,), parameters["k_weight"][branch], eps)
        score = (query * key).sum(-1) / hidden_size**0.5
        magnitude = score.abs().clamp_min(1e-6).sqrt()
        gate = torch.sigmoid(torch.where(score >= 0, magnitude, -magnitude)) * token_mask
        branches.append(hidden[:, :, branch] + gate.unsqueeze(-1) * projected[..., -hidden_size:])
    return torch.stack(branches, dim=2)


def test_hash_matches_official_fixture_with_compressed_tokens_and_image_barriers() -> None:
    hasher = DeepseekV41NgramHash(_tiny_config(), _tokenizer())
    input_ids = torch.tensor([[7, 1, 4, 3, 8, 7], [2, 3, 1, 7, 5, 6]])
    mask = torch.tensor([[True, True, False, False, True, True], [True] * 6])
    # Frozen from the downloaded DeepSeek-V4.1-Flash inference/engram.py.
    # Different-cased/accented strings merge; the exact one-space token survives.
    torch.testing.assert_close(hasher.token_map, torch.tensor([0, 1, 1, 1, 2, 2, 3, 4, 5]))
    expected = torch.tensor(
        [
            [
                [[10, 13, 39, 43, 71, 100], [3, 34, 100, 141, 191, 245]],
                [[9, 11, 32, 43, 72, 88], [11, 65, 95, 111, 164, 244]],
                [[6, 13, 35, 57, 80, 103], [22, 42, 98, 150, 171, 200]],
                [[6, 13, 35, 57, 80, 103], [22, 42, 98, 150, 171, 200]],
                [[1, 19, 37, 51, 80, 83], [25, 60, 102, 116, 153, 214]],
                [[6, 12, 32, 56, 61, 93], [25, 58, 85, 135, 163, 214]],
            ],
            [
                [[6, 13, 35, 57, 80, 103], [22, 42, 98, 150, 171, 200]],
                [[6, 13, 35, 57, 80, 103], [22, 42, 98, 150, 171, 200]],
                [[6, 13, 35, 57, 80, 103], [22, 42, 98, 150, 171, 200]],
                [[10, 13, 39, 43, 71, 100], [3, 34, 100, 141, 191, 245]],
                [[3, 22, 34, 48, 78, 97], [28, 64, 84, 121, 184, 200]],
                [[4, 19, 31, 55, 76, 94], [5, 42, 106, 115, 181, 220]],
            ],
        ]
    )
    torch.testing.assert_close(hasher(input_ids, token_mask=mask), expected, rtol=0, atol=0)


def test_hash_metadata_restored_after_meta_materialization() -> None:
    tokenizer = _tokenizer()
    reference = DeepseekV41NgramHash(_tiny_config(), tokenizer)
    with torch.device("meta"):
        actual = DeepseekV41NgramHash(_tiny_config(), tokenizer)
    actual.to_empty(device="cpu")
    actual.init_weights()
    for name, value in reference.named_buffers():
        torch.testing.assert_close(actual.get_buffer(name), value, rtol=0, atol=0)
    ids = torch.tensor([[7, 5, 3]])
    torch.testing.assert_close(actual(ids), reference(ids), rtol=0, atol=0)


def test_hash_rejects_mismatched_compressed_vocabulary() -> None:
    config = _tiny_config()
    config.engram_compressed_vocab_size = 7
    with pytest.raises(ValueError, match="compressed tokenizer vocabulary mismatch"):
        DeepseekV41NgramHash(config, _tokenizer())


@pytest.mark.parametrize("all_masked", [False, True])
def test_residual_and_parameter_gradients_match_reference(all_masked: bool) -> None:
    torch.manual_seed(17)
    config = _tiny_config()
    model = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"))
    with torch.no_grad():
        model.q_weight.normal_()
        model.k_weight.normal_()
    parameters = {name: value.detach().clone().requires_grad_() for name, value in model.named_parameters()}
    hidden = torch.randn(2, 4, config.hc_mult, config.hidden_size, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    ids = torch.randint(0, 3, (2, 4, 6))  # Repeated rows exercise embedding gradient accumulation.
    mask = torch.zeros(2, 4, dtype=torch.bool) if all_masked else torch.tensor([[True, False, False, True]] * 2)
    actual = model(hidden, ids, token_mask=mask)
    expected = _reference_residual(reference_hidden, ids, parameters, mask, config.rms_norm_eps)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad, rtol=1e-5, atol=1e-6)
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, parameters[name].grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual[~mask], hidden[~mask], rtol=0, atol=0)


def test_engram_meta_initialization_is_finite_and_trainable() -> None:
    config = _tiny_config()
    with torch.device("meta"):
        model = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"))
    model.to_empty(device="cpu")
    model.init_weights()
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
    hidden = torch.randn(1, 3, config.hc_mult, config.hidden_size, requires_grad=True)
    ids = torch.zeros(1, 3, 6, dtype=torch.long)
    output = model(hidden, ids)
    output.backward(torch.randn_like(output))
    assert torch.isfinite(hidden.grad).all()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def _owner_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(22)
        config = _tiny_config()
        reference = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"))
        model = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"), process_group=dist.group.WORLD)
        assert model.num_embeddings == 113 and model.embed.num_embeddings == 114
        valid_rows = 57 if rank == 0 else 56
        assert torch.count_nonzero(model.embed.weight[valid_rows:]) == 0
        assert torch.count_nonzero(model.embed.weight[:valid_rows]) > 0
        with torch.device("meta"):
            materialized = DeepseekV41Engram(config, 1, BackendConfig(linear="torch"), process_group=dist.group.WORLD)
        materialized.to_empty(device="cpu")
        materialized.init_weights()
        assert torch.count_nonzero(materialized.embed.weight[valid_rows:]) == 0
        assert torch.count_nonzero(materialized.embed.weight[:valid_rows]) > 0
        mesh = DeviceMesh.from_group(dist.group.WORLD, "cpu", mesh_dim_names=("dp_shard_cp",))
        owner_parameter = model.embed.parallelize_weight(mesh)
        assert isinstance(owner_parameter, DTensor) and owner_parameter.placements == (Shard(0),)
        model.init_weights()
        assert torch.count_nonzero(owner_parameter.to_local()[valid_rows:]) == 0
        assert torch.count_nonzero(owner_parameter.to_local()[:valid_rows]) > 0
        with torch.no_grad():
            padded_weight = F.pad(reference.embed.weight, (0, 0, 0, 1))
            owner_parameter.to_local().copy_(padded_weight[rank * 57 : (rank + 1) * 57])
            for name in ("wkv.weight", "q_weight", "k_weight"):
                model.get_parameter(name).copy_(reference.get_parameter(name))
        torch.manual_seed(71)
        hidden = torch.randn(2, 3, config.hc_mult, config.hidden_size)
        upstream = torch.randn_like(hidden)
        ids = torch.tensor([[[0, 0, 1, 2, 56, 57]] * 3, [[57, 57, 90, 112, 0, 1]] * 3])
        mask = torch.tensor([[True, False, True], [True, True, False]])
        expected = reference(hidden, ids, token_mask=mask)
        actual = model(hidden[rank : rank + 1], ids[rank : rank + 1], token_mask=mask[rank : rank + 1])
        torch.testing.assert_close(actual, expected[rank : rank + 1], rtol=1e-5, atol=1e-6)
        expected.backward(upstream)
        actual.backward(upstream[rank : rank + 1])
        expected_table_grad = F.pad(reference.embed.weight.grad, (0, 0, 0, 1))[rank * 57 : (rank + 1) * 57]
        torch.testing.assert_close(owner_parameter.grad.to_local(), expected_table_grad, rtol=1e-5, atol=1e-6)
        for name in ("wkv.weight", "q_weight", "k_weight"):
            dist.all_reduce(model.get_parameter(name).grad)
            torch.testing.assert_close(model.get_parameter(name).grad, reference.get_parameter(name).grad)
        torch.optim.SGD(model.parameters(), lr=0.03).step()
        torch.optim.SGD(reference.parameters(), lr=0.03).step()
        expected_weight = F.pad(reference.embed.weight, (0, 0, 0, 1))[rank * 57 : (rank + 1) * 57]
        torch.testing.assert_close(owner_parameter.to_local(), expected_weight, rtol=1e-5, atol=1e-6)
        # One rank requests the physical padding row: both ranks must reject
        # before either enters the table's ID/value collectives.
        invalid = ids[rank : rank + 1].clone()
        if rank == 1:
            invalid[0, 0, 0] = 113
        with pytest.raises(ValueError, match="logical row IDs"):
            model(hidden[rank : rank + 1], invalid)
    finally:
        dist.destroy_process_group()


def test_two_rank_owner_lookup_gradients_update_and_logical_bounds(tmp_path: Path) -> None:
    mp.spawn(_owner_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=True)


def _owner_model(rows: tuple[int, int]) -> DeepseekV41ForCausalLM:
    """Build a nested tiny model with real tokenizer-derived hash metadata."""
    config = DeepseekV41Config(
        vision_config=DeepseekV41VisionConfig(num_hidden_layers=0),
        text_config=DeepseekV41TextConfig(
            vocab_size=64,
            hidden_size=16,
            moe_intermediate_size=16,
            num_hidden_layers=6,
            num_attention_heads=2,
            head_dim=8,
            qk_rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=8,
            o_groups=1,
            n_routed_experts=4,
            num_experts_per_tok=2,
            compress_ratios=[0, 0, 2, 2, 1, 1],
            kv_source_layer_ids=[2, 4],
            index_source_layer_ids=[2, 4, 5],
            index_n_heads=2,
            index_head_dim=8,
            index_topk=2,
            candidate_source_layer_id=4,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            engram_layer_ids=[1, 4],
            engram_num_embeddings=list(rows),
            engram_vocab_size=2,
            engram_max_ngram_size=3,
            engram_n_heads=1,
            engram_head_dim=32,
            engram_compressed_vocab_size=6,
            num_nextn_predict_layers=0,
            dspark_block_size=0,
            dspark_noise_token_id=0,
            dtype="float32",
        ),
    )
    backend = BackendConfig(attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch_mm", dispatcher="torch")
    # Exercise automatic WORLD ownership with legal hash bucket capacities.
    return DeepseekV41ForCausalLM(config, backend=backend, tokenizer=_tokenizer())


def _owner_gradient_case(model: DeepseekV41ForCausalLM, layer_id: int, *, empty_requester: bool) -> None:
    """Compare local [rows,channels] gradients with an independent embedding oracle.

    Requests have shape [1,requests,1] and outputs [1,requests,1,channels]. Each requester
    independently differentiates F.embedding against a full table; one Gloo
    reduction supplies the global-sum oracle before owner normalization.
    """
    rank = dist.get_rank()
    module = model.model.layers[str(layer_id)].engram
    table = module.embed
    width = table.embedding_dim
    full_weight = torch.arange(table.num_embeddings * width, dtype=torch.float32).reshape(-1, width) / 8
    full_weight[module.num_embeddings :].zero_()
    start, end = table.global_row_start, table.global_row_end
    with torch.no_grad():
        table.weight.to_local().copy_(full_weight[start:end])
    boundary = min(table.num_embeddings_per_rank, module.num_embeddings - 1)
    requested = [0, max(boundary - 1, 0), boundary, module.num_embeddings - 1, boundary, 0]
    ids = torch.tensor([] if empty_requester and rank == 1 else requested[rank:], dtype=torch.long).reshape(1, -1, 1)
    upstream = (torch.arange(ids.numel() * width).reshape(*ids.shape, width).float() + rank + 1) / 8
    actual = table(ids)
    oracle_weight = full_weight.clone().requires_grad_()
    expected = F.embedding(ids, oracle_weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    table.weight.grad = None
    actual.backward(upstream)
    expected.backward(upstream)
    dist.all_reduce(oracle_weight.grad)
    assert isinstance(table.weight.grad, DTensor)
    torch.testing.assert_close(table.weight.grad.to_local(), oracle_weight.grad[start:end], rtol=0, atol=0)

    # Owner gradients sum requests across ranks; production normalization must
    # produce the same gradient as the mean of the independent requester losses.
    actual_norm = scale_grads_and_clip_grad_norm(max_grad_norm=5.0 if empty_requester else None, model_parts=[module])
    oracle_weight.grad.div_(dist.get_world_size())
    tolerance = {"rtol": 0, "atol": 0}
    if empty_requester:
        expected_norm = torch.linalg.vector_norm(oracle_weight.grad.double())
        torch.testing.assert_close(actual_norm, expected_norm, rtol=1e-6, atol=1e-8)
        oracle_weight.grad.mul_((5.0 / (expected_norm + 1e-6)).clamp(max=1).float())
        tolerance = {"rtol": 1e-6, "atol": 1e-7}
    torch.testing.assert_close(table.weight.grad.to_local(), oracle_weight.grad[start:end], **tolerance)
    actual_optimizer = torch.optim.SGD([table.weight], lr=0.125)
    oracle_optimizer = torch.optim.SGD([oracle_weight], lr=0.125)
    actual_optimizer.step()
    oracle_optimizer.step()
    torch.testing.assert_close(table.weight.to_local(), oracle_weight.detach()[start:end], **tolerance)
    valid_rows = max(0, min(end, module.num_embeddings) - start)
    assert torch.count_nonzero(table.weight.grad.to_local()[valid_rows:]) == 0
    assert torch.count_nonzero(table.weight.to_local()[valid_rows:]) == 0


def _owner_checkpoint_roundtrip(model: DeepseekV41ForCausalLM, directory: Path) -> None:
    """Round-trip native and unpadded HF DTensors, then decode FP8 owner placeholders."""
    state = {
        f"model.layers.{index}.engram.embed.weight": layer.engram.embed.weight
        for index, layer in model.model.layers.items()
        if layer.engram is not None
    }
    expected = {name: value.to_local().detach().clone() for name, value in state.items()}
    dcp.save({"owner_tables": state}, checkpoint_id=directory)
    with torch.no_grad():
        for parameter in state.values():
            parameter.to_local().fill_(-123)
    dcp.load({"owner_tables": state}, checkpoint_id=directory)
    for name, parameter in state.items():
        torch.testing.assert_close(parameter.to_local(), expected[name], rtol=0, atol=0)
        assert parameter.placements == (Shard(0),)
        assert parameter._nemo_model_owned_grad_divisor == 2.0

    adapter = DeepseekV41StateDictAdapter(model.config, model.moe_config, model.backend, dtype=torch.float32)
    hf_state = dict(pair for name, value in state.items() for pair in adapter.convert_single_tensor_to_hf(name, value))
    for name, value in hf_state.items():
        module = model.model.layers[name.split(".")[1]].engram
        assert value.shape == (module.num_embeddings, module.embed.embedding_dim)
        expected_rows = max(
            0, min(module.embed.num_embeddings_per_rank, module.num_embeddings - module.embed.global_row_start)
        )
        assert value.to_local().shape[0] == expected_rows
    dcp.save(hf_state, checkpoint_id=directory / "released")
    with torch.no_grad():
        for parameter in state.values():
            parameter.to_local().fill_(-321)
    dcp.load(hf_state, checkpoint_id=directory / "released")
    restored = adapter.from_hf(hf_state)
    assert restored.keys() == state.keys()
    for name, value in restored.items():
        torch.testing.assert_close(value.to_local(), expected[name], rtol=0, atol=0)
        assert value.shape == state[name].shape and value.placements == (Shard(0),)
        with torch.no_grad():
            state[name].copy_(value)

    quantized = dict(
        pair
        for name, value in state.items()
        for pair in adapter.convert_single_tensor_to_hf(name, value, quantization=True, for_checkpoint_load=True)
    )
    for name, value in quantized.items():
        module = model.model.layers[name.split(".")[1]].engram
        is_scale = name.endswith(".scale")
        width = module.embed.embedding_dim // 32 if is_scale else module.embed.embedding_dim
        assert value.shape == (module.num_embeddings, width)
        assert value.dtype == (torch.float8_e8m0fnu if is_scale else torch.float8_e4m3fn)
        assert value.placements == (Shard(0),)
        values = torch.full(value.to_local().shape, 2.0 if is_scale else 1.5, dtype=torch.float32)
        value.to_local().copy_(values.to(value.dtype))
    decoded = adapter.from_hf(quantized)
    for name, value in decoded.items():
        module = model.model.layers[name.split(".")[2]].engram
        wanted = torch.full_like(state[name].to_local(), 3.0)
        valid_rows = max(0, module.num_embeddings - module.embed.global_row_start)
        wanted[valid_rows:].zero_()
        assert value.dtype == torch.float32 and value.shape == state[name].shape
        torch.testing.assert_close(value.to_local(), wanted, rtol=0, atol=0)


def _owner_checkpoint_worker(rank: int, store: str, directory: str, rows: tuple[int, int]) -> None:
    """Exercise two true owner ranks without initializing CUDA or model attention."""
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo", init_method=f"file://{store}", rank=rank, world_size=2, timeout=timedelta(seconds=90)
        )
        mesh = DeviceMesh.from_group(dist.group.WORLD, device_type="cpu", mesh_dim_names=("dp_shard_cp",))
        model = _owner_model(rows)
        modules = [model.model.layers[str(index)].engram for index in (1, 4)]
        pointers = []
        for module, logical_rows in zip(modules, rows, strict=True):
            table = module.embed
            assert isinstance(table, Qwen3_8_FlashNextOwnerShardedEmbedding)
            assert module.num_embeddings == logical_rows
            assert table.num_embeddings == (logical_rows + 1) // 2 * 2
            assert tuple(table.weight.shape) == ((logical_rows + 1) // 2, table.embedding_dim)
            assert table.weight.requires_grad
            pointers.append(table.weight.data_ptr())

        ignored = model._nemo_prepare_model_owned_dtensors(mesh)
        expected_ids = {id(module.embed.weight) for module in modules}
        assert {id(parameter) for parameter in ignored} == expected_ids
        assert {id(parameter) for parameter in model._nemo_prepare_model_owned_dtensors(mesh)} == expected_ids
        # Both FSDP roots and each containing block must retain precisely these
        # registered identities, preventing another shard of an already-local table.
        assert ignored_params_for_root(model, ignored) == ignored
        assert ignored_params_for_root(model.model, ignored) == ignored
        for index, module, pointer in zip((1, 4), modules, pointers, strict=True):
            parameter = module.embed.weight
            assert isinstance(parameter, DTensor) and parameter.placements == (Shard(0),)
            assert parameter.to_local().data_ptr() == pointer
            assert parameter.shape == (module.embed.num_embeddings, module.embed.embedding_dim)
            assert parameter.requires_grad
            assert parameter._nemo_model_owned_grad_divisor == 2.0
            assert ignored_params_for_root(model.model.layers[str(index)], ignored) == {parameter}
            module.init_weights()
            valid_rows = max(0, module.num_embeddings - module.embed.global_row_start)
            assert torch.count_nonzero(parameter.to_local()[valid_rows:]) == 0

        # Same-device casts can clear custom DTensor attributes even when Python
        # Parameter identity survives. The reused owner must re-stamp its contract.
        model.to(device="cpu")
        for module in modules:
            assert module.embed.weight._nemo_model_owned_grad_divisor == 2.0
        for empty_requester in (False, True):
            for layer_id in (1, 4):
                _owner_gradient_case(model, layer_id, empty_requester=empty_requester)
        _owner_checkpoint_roundtrip(model, Path(directory))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_engram_owner_lookup_gradients_and_checkpoint(tmp_path: Path) -> None:
    """Cover odd logical rows, padding, trainable tables and empty requesters."""
    mp.spawn(
        _owner_checkpoint_worker,
        args=(str(tmp_path / "gloo"), str(tmp_path / "checkpoint"), (17, 19)),
        nprocs=2,
        join=True,
    )
