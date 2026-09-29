# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch
from torch.distributed.utils import _apply_to_tensors

from nemo_automodel.components.distributed.context_parallel.sharder import contiguous_local_indices
from nemo_automodel.components.models.glm5_next.cp import (
    Glm5NextPackedContext,
    doc_ids_from_cu_seqlens,
    doc_ids_from_seq_lens,
    segment_cu_seqlens,
    shard_batch_for_glm5_next_cp,
)
from tests.unit_tests.models.glm5_next.conftest import tiny_glm5_next_model


class _FakeCPMesh:
    def __init__(self, size: int, rank: int) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


def test_document_metadata_conversions_preserve_padding_boundaries():
    expected = [[1, 1, 1, 2, 2, 0, 0]]
    assert doc_ids_from_seq_lens(torch.tensor([[3, 2, -1000]]), 7).tolist() == expected
    assert doc_ids_from_cu_seqlens(torch.tensor([0, 3, 5, -1000]), 7).tolist() == expected
    assert segment_cu_seqlens(torch.tensor(expected[0], dtype=torch.int32)).tolist() == [0, 3, 5, 7]


def test_packed_context_can_be_rebuilt_by_fsdp_input_transform():
    context = Glm5NextPackedContext(torch.tensor([[1, 1, 2, 2]], dtype=torch.int32))
    context.row_cu_seqlens(0)

    rebuilt = _apply_to_tensors(lambda tensor: tensor.clone(), context)

    assert isinstance(rebuilt, Glm5NextPackedContext)
    assert rebuilt.doc_ids.tolist() == [[1, 1, 2, 2]]
    assert rebuilt.row_cu_seqlens(0)[0].tolist() == [0, 2, 4]


def test_vlm_cp_sharder_keeps_ids_and_media_global_but_shards_labels():
    batch = {
        "input_ids": torch.arange(8).unsqueeze(0),
        "labels": torch.arange(8).unsqueeze(0),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=torch.int32),
        "pixel_values": torch.randn(8, 24),
        "image_grid_thw": torch.tensor([[1, 2, 4]]),
    }

    _, local, layout = shard_batch_for_glm5_next_cp(_FakeCPMesh(2, 1), None, batch)

    assert local["input_ids"].tolist() == [list(range(8))]
    assert local["labels"].tolist() == [[4, 5, 6, 7]]
    assert local["pixel_values"].shape == (8, 24)
    assert local["image_grid_thw"].tolist() == [[1, 2, 4]]
    context = local["glm5_next_packed_context"]
    assert isinstance(context, Glm5NextPackedContext)
    assert context.seq_start == 4
    assert context.local_doc_ids.tolist() == [[2, 2, 2, 2]]
    assert (layout.original_seq_len, layout.padded_seq_len) == (8, 8)


def test_cp_sharder_pads_labels_without_padding_global_primary_ids():
    batch = {
        "input_ids": torch.arange(6).unsqueeze(0),
        "labels": torch.arange(6).unsqueeze(0),
        "attention_mask": torch.ones(1, 6, dtype=torch.int32),
    }

    _, local, layout = shard_batch_for_glm5_next_cp(_FakeCPMesh(4, 3), None, batch)

    assert local["input_ids"].shape == (1, 6)
    assert local["labels"].tolist() == [[-100, -100]]
    assert local["padding_mask"].all()
    assert local["glm5_next_packed_context"].doc_ids.shape == (1, 8)
    assert (layout.original_seq_len, layout.padded_seq_len) == (6, 8)


def test_model_registers_contiguous_model_owned_cp_sharder():
    model = tiny_glm5_next_model()
    sharder = model.prepare_model_inputs_for_cp({"input_ids": torch.arange(8).unsqueeze(0)})["cp_sharder"]

    assert sharder.local_token_global_indices is contiguous_local_indices
    assert model._owns_cp_attention
    assert model._owns_packed_attention
