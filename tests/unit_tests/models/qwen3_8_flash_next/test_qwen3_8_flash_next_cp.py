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

"""CPU and two-rank Gloo parity tests for Qwen3.8-Flash-Next context parallelism."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from nemo_automodel.components.distributed.blockdiag_cp import BlockdiagCpModelState
from nemo_automodel.components.distributed.context_parallel.sharder import (
    ContextParallelSharder,
    contiguous_local_indices,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet
from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from nemo_automodel.components.models.qwen3_8_flash_next.cp import (
    Qwen3_8_FlashNextCPContext,
    shard_batch_for_qwen3_8_flash_next_cp,
)
from nemo_automodel.components.models.qwen3_8_flash_next.engram import (
    Qwen3_8_FlashNextNGramEmbedding,
    Qwen3_8_FlashNextPLELayer,
)
from nemo_automodel.components.models.qwen3_8_flash_next.layers import (
    Qwen3_8_FlashNextGatedDeltaNet,
    Qwen3_8_FlashNextQSAAttention,
)
from nemo_automodel.components.models.qwen3_8_flash_next.model import (
    Qwen3_8_FlashNextForConditionalGeneration,
    Qwen3_8_FlashNextTextModelBackend,
)
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import select_qsa_token_ids

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


class _FakeMesh:
    """Minimal mesh used only for fail-before-collective validation tests."""

    def __init__(self, size: int, group: object | None = None) -> None:
        self._size = size
        self._group = group

    def size(self) -> int:
        """Return the configured fake mesh size."""
        return self._size

    def get_group(self) -> object | None:
        """Return the configured fake process-group sentinel."""
        return self._group


def _backend() -> BackendConfig:
    """Return the deterministic all-Torch backend used by tiny CP tests."""
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


def _qsa_config() -> Qwen3_8_FlashNextTextConfig:
    """Return a tiny QSA config whose sparse rows cross the CP rank boundary."""
    return Qwen3_8_FlashNextTextConfig(
        vocab_size=128,
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
        indexer_budget=4,
        indexer_compress_ratio=2,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        max_position_embeddings=64,
        rope_parameters={
            "rope_theta": 10000.0,
            "rope_type": "default",
            "partial_rotary_factor": 1.0,
        },
        partial_rotary_factor=1.0,
        dtype="float32",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=99,
    )


def _gdn_config() -> Qwen3_8_FlashNextTextConfig:
    """Return a tiny GDN config used to inspect contiguous-state wiring."""
    return Qwen3_8_FlashNextTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["linear_attention"],
        linear_conv_kernel_dim=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=2,
        ple_layer_ids=[],
        dtype="float32",
        pad_token_id=0,
    )


def _freqs(sequence_length: int) -> torch.Tensor:
    """Build concatenated cosine/sine values of shape ``[1, sequence, 4]``."""
    positions = torch.arange(sequence_length, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, 4, 2).float() / 4))
    angles = torch.outer(positions, inv_freq)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).unsqueeze(0)


def test_model_advertises_and_returns_its_contiguous_cp_sharder() -> None:
    model = object.__new__(Qwen3_8_FlashNextForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(text_config=SimpleNamespace(indexer_compress_ratio=4))

    prepared = model.prepare_model_inputs_for_cp({}, num_chunks=1)

    assert Qwen3_8_FlashNextForConditionalGeneration._owns_cp_attention is True
    assert Qwen3_8_FlashNextForConditionalGeneration.ModelCapabilities.supports_cp is True
    assert set(prepared) == {"cp_sharder"}
    sharder = prepared["cp_sharder"]
    assert isinstance(sharder, ContextParallelSharder)
    assert sharder.local_token_global_indices is contiguous_local_indices
    assert sharder.shard_batch.func is shard_batch_for_qwen3_8_flash_next_cp
    assert sharder.shard_batch.keywords["pad_multiple"] == 4


def _tiny_ple() -> Qwen3_8_FlashNextPLELayer:
    """Construct a deterministic tiny PLE with a real trainable table."""
    table = nn.Embedding(36, 2)
    ngram = Qwen3_8_FlashNextNGramEmbedding(
        table,
        ngram_size=3,
        heads_per_ngram=2,
        eos_token_id=99,
        layer_multipliers=(3, 5, 7),
        ngram_heads_vocab_sizes=(5, 7, 11, 13),
        ngram_heads_offsets=(0, 5, 12, 23),
    )
    ple = Qwen3_8_FlashNextPLELayer(
        ngram,
        hidden_size=2,
        hc_count=2,
        ple_embed_dim=8,
        backend=BackendConfig(linear="torch"),
        dtype=torch.float32,
        conv_kernel_size=4,
        rms_norm_eps=1e-6,
    )
    generator = torch.Generator().manual_seed(314)
    with torch.no_grad():
        for parameter in ple.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.15)
    return ple


def _cp_context(
    *,
    rank: int,
    world_size: int,
    group: dist.ProcessGroup,
    global_input_ids: torch.Tensor,
) -> Qwen3_8_FlashNextCPContext:
    """Build contiguous CP metadata for a replicated raw sequence.

    Args:
        rank: Rank in ``group``.
        world_size: Number of contiguous sequence shards.
        group: Gloo CP process group.
        global_input_ids: Replicated IDs of shape ``[batch, global_sequence]``.

    Returns:
        Context whose local interval has length ``global_sequence / world_size``.
    """
    local_length = global_input_ids.shape[1] // world_size
    return Qwen3_8_FlashNextCPContext(
        group=group,
        rank=rank,
        size=world_size,
        global_input_ids=global_input_ids,
        global_padding_mask=torch.zeros_like(global_input_ids, dtype=torch.bool),
        local_sequence_start=rank * local_length,
        local_sequence_length=local_length,
    )


def test_selector_uses_absolute_query_positions_for_a_local_cp_shard() -> None:
    generator = torch.Generator().manual_seed(12)
    full_queries = torch.randn(1, 8, 2, 3, generator=generator)
    compressed_keys = torch.randn(1, 4, 1, 3, generator=generator)
    full = select_qsa_token_ids(
        full_queries,
        compressed_keys,
        torch.tensor([8]),
        token_budget=4,
        compress_ratio=2,
    )
    second_rank = select_qsa_token_ids(
        full_queries[:, 4:],
        compressed_keys,
        torch.tensor([8]),
        token_budget=4,
        compress_ratio=2,
        query_position_offset=4,
        global_sequence_length=8,
    )
    torch.testing.assert_close(second_rank, full[:, 4:], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"qkv_format": "thd"}, "supports only cu_seqlens document boundaries"),
        ({"cu_seqlens_padded": torch.tensor([0, 4])}, "supports only cu_seqlens document boundaries"),
        ({"_packed_seq_ids": torch.ones(1, 4)}, "supports only cu_seqlens document boundaries"),
        ({"attention_mask": torch.ones(1, 1, 4, 4)}, "non-packed"),
        ({"attention_mask": torch.tensor([[1, 0, 1, 0]])}, "right-tail padding"),
        ({"padding_mask": torch.tensor([[0, 0, 2, 1]])}, "binary"),
    ],
)
def test_sharder_rejects_unsupported_layouts_before_collectives(extra: dict[str, object], match: str) -> None:
    batch: dict[str, object] = {
        "input_ids": torch.arange(4).view(1, 4),
        "labels": torch.arange(4).view(1, 4),
        **extra,
    }
    with pytest.raises((NotImplementedError, ValueError), match=match):
        shard_batch_for_qwen3_8_flash_next_cp(_FakeMesh(2), None, batch)


def test_sharder_rejects_tp_composition() -> None:
    batch = {"input_ids": torch.arange(4).view(1, 4), "labels": torch.arange(4).view(1, 4)}
    with pytest.raises(NotImplementedError, match="tensor parallelism"):
        shard_batch_for_qwen3_8_flash_next_cp(_FakeMesh(2), _FakeMesh(2), batch)


def test_sharder_normalizes_integer_padding_mask_to_bool() -> None:
    input_ids = torch.arange(4).view(1, 4)
    _, batch, _ = shard_batch_for_qwen3_8_flash_next_cp(
        _FakeMesh(1),
        None,
        {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "padding_mask": torch.tensor([[0, 0, 0, 1]], dtype=torch.int32),
        },
        pad_multiple=2,
    )
    assert batch["padding_mask"].dtype == torch.bool
    torch.testing.assert_close(batch["padding_mask"], torch.tensor([[False, False, False, True]]))


def test_cp_enabled_text_model_requires_model_owned_batch_context() -> None:
    model = Qwen3_8_FlashNextTextModelBackend(_qsa_config(), _backend())
    model._cp_enabled = True
    with pytest.raises(RuntimeError, match="batch context is missing"):
        model(
            input_ids=torch.ones(1, 4, dtype=torch.long),
            position_ids=torch.arange(4).view(1, -1),
        )


def test_gdn_override_synthesizes_one_contiguous_global_segment(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[BlockdiagCpModelState] = []

    def _capture_base_cp(
        self: CPAwareGatedDeltaNet,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor | None,
        seq_index: torch.Tensor | None,
        blockdiag_state: BlockdiagCpModelState | None = None,
    ) -> torch.Tensor:
        """Capture CP metadata while preserving the local hidden-state layout.

        Args:
            self: GDN module under test.
            hidden_states: Tensor of shape ``[batch, local_sequence, hidden]``.
            position_ids: Tensor of shape ``[batch, local_sequence]`` or
                ``None``.
            seq_index: Optional global indices of shape ``[local_sequence]`` or
                ``[batch, local_sequence]``.
            blockdiag_state: Synthesized state whose device and CPU cumulative
                lengths both have shape ``[num_segments + 1]``.

        Returns:
            Tensor of shape ``[batch, local_sequence, hidden]``.
        """
        del self, position_ids, seq_index
        assert blockdiag_state is not None
        captured.append(blockdiag_state)
        return hidden_states + 1

    monkeypatch.setattr(CPAwareGatedDeltaNet, "_forward_with_cp", _capture_base_cp)
    layer = Qwen3_8_FlashNextGatedDeltaNet(_gdn_config(), layer_idx=0)
    group_sentinel = object()
    layer._cp_mesh = _FakeMesh(2, group_sentinel)
    hidden_states = torch.randn(1, 4, 8)
    position_ids = torch.arange(4, 8).view(1, -1)

    output = layer._forward_with_cp(
        hidden_states,
        position_ids=position_ids,
        seq_index=None,
    )

    torch.testing.assert_close(output, hidden_states + 1)
    assert captured[0].group is group_sentinel
    torch.testing.assert_close(captured[0].packed_cu_seqlens, torch.tensor([0, 8]))
    assert captured[0].packed_cu_seqlens_cpu.device.type == "cpu"


def _compare_parameter_gradients(
    distributed_module: nn.Module,
    reference_module: nn.Module,
    group: dist.ProcessGroup,
) -> None:
    """Sum replicated parameter gradients and compare with a dense reference.

    Args:
        distributed_module: CP module whose parameter tensors have arbitrary
            model-owned shapes and rank-local gradient contributions.
        reference_module: CP1 module with matching parameter tensor shapes.
        group: CP group over which each parameter gradient is summed.
    """
    reference_parameters = dict(reference_module.named_parameters())
    for name, parameter in distributed_module.named_parameters():
        reference_gradient = reference_parameters[name].grad
        if parameter.grad is None:
            assert reference_gradient is None
            continue
        dist.all_reduce(parameter.grad, group=group)
        assert reference_gradient is not None
        torch.testing.assert_close(parameter.grad, reference_gradient, rtol=2e-5, atol=2e-6)


def _distributed_cp_parity_worker(rank: int, world_size: int, store_path: str) -> None:
    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
        )
        from torch.distributed.device_mesh import init_device_mesh

        cp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))["cp"]

        # Batch contract: no user mask and a non-divisible length must create
        # synthetic padding that is ignored by labels, QSA, and MoE.
        unpadded_ids = torch.arange(10, dtype=torch.long).view(1, 10)
        _, local_batch, layout = shard_batch_for_qwen3_8_flash_next_cp(
            cp_mesh,
            None,
            {"input_ids": unpadded_ids.clone(), "labels": unpadded_ids.clone()},
            padding_token_id=0,
            pad_multiple=4,
        )
        assert layout.original_seq_len == 10
        assert layout.padded_seq_len == 16
        start = rank * 8
        expected_ids = torch.cat((unpadded_ids, torch.zeros(1, 6, dtype=torch.long)), dim=1)
        torch.testing.assert_close(local_batch["input_ids"], expected_ids[:, start : start + 8])
        torch.testing.assert_close(local_batch["position_ids"], torch.arange(start, start + 8).view(1, -1))
        expected_padding = torch.arange(start, start + 8).view(1, -1) >= 10
        torch.testing.assert_close(local_batch["padding_mask"], expected_padding)
        expected_labels = expected_ids[:, start : start + 8].clone()
        expected_labels[expected_padding] = -100
        torch.testing.assert_close(local_batch["labels"], expected_labels)
        batch_context = local_batch["_qwen3_8_flash_next_cp_context"]
        assert isinstance(batch_context, Qwen3_8_FlashNextCPContext)
        torch.testing.assert_close(batch_context.global_input_ids, expected_ids)
        torch.testing.assert_close(
            batch_context.global_padding_mask,
            torch.arange(16).view(1, -1) >= 10,
        )

        # QSA CP2 versus CP1: exact global routing IDs, local outputs/hidden
        # gradients, and globally summed parameter gradients.
        torch.manual_seed(123)
        qsa_config = _qsa_config()
        reference_qsa = Qwen3_8_FlashNextQSAAttention(qsa_config, layer_idx=0, backend=_backend())
        reference_qsa.init_weights(torch.device("cpu"))
        cp_qsa = Qwen3_8_FlashNextQSAAttention(qsa_config, layer_idx=0, backend=_backend())
        cp_qsa.load_state_dict(reference_qsa.state_dict())
        cp_qsa.setup_cp_attention(cp_mesh)
        qsa_ids = torch.tensor([[5, 7, 11, 13, 17, 19, 23, 29]], dtype=torch.long)
        qsa_context = _cp_context(
            rank=rank,
            world_size=world_size,
            group=dist.group.WORLD,
            global_input_ids=qsa_ids,
        )
        qsa_start = qsa_context.local_sequence_start
        qsa_end = qsa_context.local_sequence_end
        full_hidden = torch.randn(1, 8, qsa_config.hidden_size).requires_grad_(True)
        local_hidden = full_hidden.detach()[:, qsa_start:qsa_end].clone().requires_grad_(True)
        full_freqs = _freqs(8)
        reference_routes: list[torch.Tensor] = []
        cp_routes: list[torch.Tensor] = []
        reference_handle = reference_qsa.indexer.register_forward_hook(
            lambda _module, _args, output: reference_routes.append(output)
        )
        cp_handle = cp_qsa.indexer.register_forward_hook(lambda _module, _args, output: cp_routes.append(output))
        reference_output = reference_qsa(
            full_hidden,
            freqs_cis=full_freqs,
            attention_mask=torch.ones(1, 8, dtype=torch.bool),
        )
        cp_output = cp_qsa(
            local_hidden,
            freqs_cis=full_freqs[:, qsa_start:qsa_end],
            attention_mask=torch.ones(1, 4, dtype=torch.bool),
            cp_context=qsa_context,
        )
        reference_handle.remove()
        cp_handle.remove()
        torch.testing.assert_close(cp_routes[0], reference_routes[0][:, qsa_start:qsa_end], rtol=0, atol=0)
        torch.testing.assert_close(cp_output, reference_output[:, qsa_start:qsa_end], rtol=1e-5, atol=1e-6)
        reference_output.square().sum().backward()
        cp_output.square().sum().backward()
        torch.testing.assert_close(local_hidden.grad, full_hidden.grad[:, qsa_start:qsa_end], rtol=2e-5, atol=2e-6)
        _compare_parameter_gradients(cp_qsa, reference_qsa, dist.group.WORLD)

        # PLE CP2 versus CP1: row zero crosses a normal trigram boundary; row
        # one puts EOS at the last token of rank zero. The k=4/dilation=3
        # convolution consumes a nine-token left halo and sends its gradient
        # back to rank zero.
        reference_ple = _tiny_ple()
        cp_ple = _tiny_ple()
        cp_ple.load_state_dict(reference_ple.state_dict())
        ple_ids = torch.tensor(
            [
                [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 1],
                [4, 6, 8, 10, 12, 99, 14, 16, 18, 20, 22, 24],
            ],
            dtype=torch.long,
        )
        ple_context = _cp_context(
            rank=rank,
            world_size=world_size,
            group=dist.group.WORLD,
            global_input_ids=ple_ids,
        )
        ple_start = ple_context.local_sequence_start
        ple_end = ple_context.local_sequence_end
        full_ple_hidden = torch.randn(2, 12, 4).requires_grad_(True)
        local_ple_hidden = full_ple_hidden.detach()[:, ple_start:ple_end].clone().requires_grad_(True)
        reference_ple_output = reference_ple(full_ple_hidden, ple_ids)
        cp_ple_output = cp_ple(
            local_ple_hidden,
            ple_ids[:, ple_start:ple_end],
            cp_context=ple_context,
        )
        torch.testing.assert_close(
            cp_ple_output,
            reference_ple_output[:, ple_start:ple_end],
            rtol=2e-5,
            atol=2e-6,
        )
        upstream = torch.zeros_like(reference_ple_output)
        upstream[:, 6:] = torch.linspace(-0.5, 0.75, reference_ple_output[:, 6:].numel()).view_as(
            reference_ple_output[:, 6:]
        )
        reference_ple_output.backward(upstream)
        cp_ple_output.backward(upstream[:, ple_start:ple_end])
        torch.testing.assert_close(
            local_ple_hidden.grad,
            full_ple_hidden.grad[:, ple_start:ple_end],
            rtol=2e-5,
            atol=2e-6,
        )
        _compare_parameter_gradients(cp_ple, reference_ple, dist.group.WORLD)
        if rank == 0:
            # Rank zero's local output has zero upstream. Any nonzero hidden
            # gradient therefore came exclusively from rank one's nine-token
            # convolution halo through the autograd-aware collective.
            assert torch.count_nonzero(local_ple_hidden.grad) > 0
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_two_rank_qsa_ple_and_sharder_match_cp1(tmp_path: Path) -> None:
    mp.spawn(
        _distributed_cp_parity_worker,
        args=(2, str(tmp_path / "qwen3-8-flash-next-cp-gloo")),
        nprocs=2,
        join=True,
    )
