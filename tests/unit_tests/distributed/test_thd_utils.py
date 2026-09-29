# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import pytest
import torch

from nemo_automodel.components.distributed.thd_utils import process_input_for_thd, split_batch_into_thd_chunks


class TestProcessInputForTHD:
    """Tests for process_input_for_thd function."""

    def test_basic_conversion(self):
        """Test basic conversion from BSHD to THD format with 2D token IDs."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
            "labels": torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]),
            "seq_lens": torch.tensor([[6], [6]]),
            "seq_lens_padded": torch.tensor([[6], [6]]),
        }

        result = process_input_for_thd(batch)

        # Check shapes - for 2D input [batch, seq], output is [batch*seq] (squeezed)
        assert result["input_ids"].shape == (12,)
        assert result["position_ids"].shape == (12,)
        assert result["labels"].shape == (12,)

        # Check values are preserved
        assert torch.equal(result["input_ids"], torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

        # Check cu_seqlens
        assert "cu_seqlens" in result
        assert torch.equal(result["cu_seqlens"], torch.tensor([0, 6, 12], dtype=torch.int32))

    def test_with_multiple_packed_sequences(self):
        """Test with multiple packed sequences per example."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 99, 4, 5], [6, 7, 99, 8, 9, 10]]),
            "labels": torch.tensor([[2, 3, 99, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
            "position_ids": torch.tensor([[0, 1, 2, 0, 0, 1], [0, 1, 0, 0, 1, 2]]),
            "seq_lens": torch.tensor([[3, 2], [2, 3]]),
            "seq_lens_padded": torch.tensor([[4, 2], [3, 3]]),
        }

        result = process_input_for_thd(batch)

        # Check shapes - 2D input [batch, seq] becomes [batch*seq] (squeezed)
        assert result["input_ids"].shape == (12,)
        assert result["position_ids"].shape == (12,)

        # cu_seqlens uses real (unpadded) seq_lens; cu_seqlens_padded uses
        # padded values. Both are emitted because they differ in multiple
        # entries (inter-sub-seq padding) — TE will use pad_between_seqs=True.
        # First batch: real [3, 2] -> cumsum [0, 3, 5], padded [4, 2] -> [0, 4, 6]
        # Second batch: real [2, 3] -> [5, 7, 10], padded [3, 3] -> [6, 9, 12]
        expected_cu_seqlens = torch.tensor([0, 3, 5, 7, 10], dtype=torch.int32)
        expected_cu_seqlens_padded = torch.tensor([0, 4, 6, 9, 12], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"], expected_cu_seqlens)
        assert torch.equal(result["cu_seqlens_padded"], expected_cu_seqlens_padded)

    def test_with_variable_num_sequences_and_padding(self):
        """Test with variable number of sequences per example (seq_lens padding with -1000)."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 99, 4, 5], [6, 7, 8, 9, 10, 11]]),
            "labels": torch.tensor([[2, 3, 99, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
            "position_ids": torch.tensor([[0, 1, 2, 0, 0, 1], [0, 1, 2, 3, 4, 5]]),
            "seq_lens": torch.tensor([[3, 2], [6, -1000]]),  # -1000 is padding
            "seq_lens_padded": torch.tensor([[4, 2], [6, -1000]]),
        }

        result = process_input_for_thd(batch)

        # Check shapes - 2D input [batch, seq] becomes [batch*seq] (squeezed)
        assert result["input_ids"].shape == (12,)
        assert result["position_ids"].shape == (12,)

        # cu_seqlens uses real seq_lens (filtered for -1000); cu_seqlens_padded
        # uses padded. Both are emitted because they differ in multiple entries.
        # First batch: real [3, 2] -> [0, 3, 5], padded [4, 2] -> [0, 4, 6]
        # Second batch: real [6] -> [5, 11], padded [6] -> [6, 12]
        expected_cu_seqlens = torch.tensor([0, 3, 5, 11], dtype=torch.int32)
        expected_cu_seqlens_padded = torch.tensor([0, 4, 6, 12], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"], expected_cu_seqlens)
        assert torch.equal(result["cu_seqlens_padded"], expected_cu_seqlens_padded)

    def test_with_qkv_format_preservation(self):
        """Test that non-tensor keys like qkv_format are preserved."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3]]),
            "qkv_format": "thd",  # Non-tensor key
        }

        result = process_input_for_thd(batch)

        # Check that qkv_format is preserved
        assert "qkv_format" in result
        assert result["qkv_format"] == "thd"

    def test_dtype_preservation(self):
        """Test that dtypes are preserved correctly."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long),
            "seq_lens": torch.tensor([[3], [3]], dtype=torch.long),
            "seq_lens_padded": torch.tensor([[3], [3]], dtype=torch.long),
        }

        result = process_input_for_thd(batch)

        assert result["input_ids"].dtype == torch.long
        assert result["position_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long
        assert result["cu_seqlens"].dtype == torch.int32

    def test_with_embeddings_3d_input(self):
        """Test with 3D embeddings input (pipeline parallelism scenario)."""
        batch_size, seq_len, hidden_dim = 2, 6, 128
        batch = {
            "input_ids": torch.randn(batch_size, seq_len, hidden_dim),  # 3D embeddings
            "labels": torch.tensor([[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13]]),
            "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]),
            "seq_lens": torch.tensor([[6], [6]]),
            "seq_lens_padded": torch.tensor([[6], [6]]),
        }

        result = process_input_for_thd(batch)

        # Check shapes - 3D input [batch, seq, hidden] becomes [batch*seq, hidden]
        assert result["input_ids"].shape == (12, hidden_dim)
        assert result["position_ids"].shape == (12,)
        assert result["cu_seqlens"].shape == (3,)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_different_batch_sizes(self, batch_size):
        """Test with different batch sizes."""
        seq_len = 16
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = process_input_for_thd(batch)

        # Check output shapes
        expected_total_tokens = batch_size * seq_len
        assert result["input_ids"].shape == (expected_total_tokens,)
        assert result["position_ids"].shape == (expected_total_tokens,)
        assert result["labels"].shape == (expected_total_tokens,)
        assert result["cu_seqlens"].shape == (batch_size + 1,)

    @pytest.mark.parametrize("seq_len", [32, 64, 128, 256, 512, 1024])
    def test_different_sequence_lengths(self, seq_len):
        """Test with different sequence lengths."""
        batch_size = 4
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = process_input_for_thd(batch)

        # Check output shapes
        expected_total_tokens = batch_size * seq_len
        assert result["input_ids"].shape == (expected_total_tokens,)
        assert result["cu_seqlens"][-1].item() == expected_total_tokens

    def test_single_batch_single_sequence(self):
        """Test edge case: single batch with single sequence."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "labels": torch.tensor([[2, 3, 4, 5]]),
            "position_ids": torch.tensor([[0, 1, 2, 3]]),
            "seq_lens": torch.tensor([[4]]),
            "seq_lens_padded": torch.tensor([[4]]),
        }

        result = process_input_for_thd(batch)

        assert result["input_ids"].shape == (4,)
        assert torch.equal(result["cu_seqlens"], torch.tensor([0, 4], dtype=torch.int32))

    def test_large_batch_with_packing(self):
        """Test with large batch size and multiple packed sequences."""
        batch_size = 16
        num_packs = 3
        seq_len_per_pack = 128
        total_seq_len = num_packs * seq_len_per_pack

        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, total_seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, total_seq_len)),
            "position_ids": torch.arange(total_seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, num_packs), seq_len_per_pack),
            "seq_lens_padded": torch.full((batch_size, num_packs), seq_len_per_pack),
        }

        result = process_input_for_thd(batch)

        expected_total_tokens = batch_size * total_seq_len
        expected_num_sequences = batch_size * num_packs

        assert result["input_ids"].shape == (expected_total_tokens,)
        assert result["cu_seqlens"].shape == (expected_num_sequences + 1,)
        assert result["cu_seqlens"][-1].item() == expected_total_tokens

    def test_chunking_basic(self):
        """Test basic chunking functionality."""
        batch = {
            "input_ids": torch.tensor(
                [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18], [19, 20, 21, 22, 23, 24]]
            ),
            "labels": torch.tensor(
                [[2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13], [14, 15, 16, 17, 18, 19], [20, 21, 22, 23, 24, 25]]
            ),
            "position_ids": torch.tensor(
                [[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]
            ),
            "seq_lens": torch.tensor([[6], [6], [6], [6]]),
            "seq_lens_padded": torch.tensor([[6], [6], [6], [6]]),
        }

        # Process with 2 chunks (2 batch items per chunk)
        result = split_batch_into_thd_chunks(batch, num_chunks=2, padding_token_id=0)

        # Check shapes - should be [num_chunks, tokens_per_chunk]
        assert result["input_ids"].shape == (2, 12), f"Expected shape (2, 12), got {result['input_ids'].shape}"
        assert result["labels"].shape == (2, 12)
        assert result["position_ids"].shape == (2, 12)

        # Check cu_seqlens has correct shape [num_chunks, seqs_per_chunk+1]
        assert result["cu_seqlens"].shape[0] == 2

        # Each chunk should have cumulative lengths [0, 6, 12]
        assert torch.equal(result["cu_seqlens"][0], torch.tensor([0, 6, 12], dtype=torch.int32))
        assert torch.equal(result["cu_seqlens"][1], torch.tensor([0, 6, 12], dtype=torch.int32))

    def test_chunking_with_packed_sequences(self):
        """Test chunking with multiple packed sequences per example."""
        batch = {
            "input_ids": torch.tensor(
                [[1, 2, 3, 99, 4, 5], [6, 7, 99, 8, 9, 10], [11, 12, 13, 99, 14, 15], [16, 17, 99, 18, 19, 20]]
            ),
            "labels": torch.tensor(
                [[2, 3, 99, 4, 5, 6], [7, 8, 9, 10, 11, 12], [12, 13, 99, 14, 15, 16], [17, 18, 19, 20, 21, 22]]
            ),
            "position_ids": torch.tensor(
                [[0, 1, 2, 0, 0, 1], [0, 1, 0, 0, 1, 2], [0, 1, 2, 0, 0, 1], [0, 1, 0, 0, 1, 2]]
            ),
            "seq_lens": torch.tensor([[3, 2], [2, 3], [3, 2], [2, 3]]),
            "seq_lens_padded": torch.tensor([[4, 2], [3, 3], [4, 2], [3, 3]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2, padding_token_id=0)

        # Check shapes
        assert result["input_ids"].shape == (2, 12)
        assert result["labels"].shape == (2, 12)

        # cu_seqlens uses real (unpadded) seq_lens; cu_seqlens_padded uses
        # padded values. Both arrays present because they differ in multiple
        # entries (inter-sub-seq padding). For each chunk:
        # Real: [3, 2] -> [0, 3, 5]; [2, 3] -> [5, 7, 10]
        # Padded: [4, 2] -> [0, 4, 6]; [3, 3] -> [6, 9, 12]
        expected_cu_seqlens_chunk = torch.tensor([0, 3, 5, 7, 10], dtype=torch.int32)
        expected_cu_padded_chunk = torch.tensor([0, 4, 6, 9, 12], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"][0], expected_cu_seqlens_chunk)
        assert torch.equal(result["cu_seqlens"][1], expected_cu_seqlens_chunk)
        assert "cu_seqlens_padded" in result
        assert torch.equal(result["cu_seqlens_padded"][0], expected_cu_padded_chunk)
        assert torch.equal(result["cu_seqlens_padded"][1], expected_cu_padded_chunk)

    def test_chunking_with_embeddings(self):
        """Test chunking with 3D embeddings input."""
        batch_size, seq_len, hidden_dim = 4, 6, 128
        batch = {
            "input_ids": torch.randn(batch_size, seq_len, hidden_dim),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2, padding_token_id=0)

        # Check shapes - should be [num_chunks, tokens_per_chunk, hidden_dim]
        assert result["input_ids"].shape == (2, 12, hidden_dim)
        assert result["position_ids"].shape == (2, 12)
        assert result["cu_seqlens"].shape[0] == 2


class TestProcessInputForTHDWithChunks:
    """Comprehensive tests for process_input_for_thd_with_chunks function."""

    def test_variable_length_cu_seqlens_padding(self):
        """Test that cu_seqlens with different lengths are padded correctly."""
        # Create a batch where different chunks will have different cu_seqlens lengths
        batch = {
            "input_ids": torch.tensor(
                [
                    [1, 2, 3, 99, 4, 5],  # 2 sequences (lengths 3, 2)
                    [6, 7, 8, 9, 10, 11],  # 1 sequence (length 6)
                    [12, 13, 14, 15, 99, 99],  # 1 sequence (length 4)
                    [16, 17, 99, 18, 19, 20],  # 2 sequences (lengths 2, 3)
                ]
            ),
            "labels": torch.randint(0, 100, (4, 6)),
            "position_ids": torch.tensor(
                [[0, 1, 2, 0, 0, 1], [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 0, 0], [0, 1, 0, 0, 1, 2]]
            ),
            "seq_lens": torch.tensor([[3, 2, -1000], [6, -1000, -1000], [4, -1000, -1000], [2, 3, -1000]]),
            "seq_lens_padded": torch.tensor([[4, 2, -1000], [6, -1000, -1000], [4, -1000, -1000], [3, 3, -1000]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2, padding_token_id=0)

        # First chunk has 2+1=3 sequences, second chunk has 1+2=3 sequences
        # cu_seqlens should be [num_chunks, max_seqs_across_chunks+1]
        assert result["cu_seqlens"].shape[0] == 2

        # cu_seqlens uses REAL (unpadded) seq_lens; cu_seqlens_padded uses
        # padded values. Both arrays are stacked because at least one chunk
        # emits cu_seqlens_padded (multiple per-chunk entries differ).
        # Chunk 0: real [3, 2, 6] -> [0, 3, 5, 11]; padded [4, 2, 6] -> [0, 4, 6, 12]
        # Chunk 1: real [4, 2, 3] -> [0, 4, 6, 9]; padded [4, 3, 3] -> [0, 4, 7, 10]
        expected_cu_0 = torch.tensor([0, 3, 5, 11], dtype=torch.int32)
        expected_cu_1 = torch.tensor([0, 4, 6, 9], dtype=torch.int32)
        expected_cu_padded_0 = torch.tensor([0, 4, 6, 12], dtype=torch.int32)
        expected_cu_padded_1 = torch.tensor([0, 4, 7, 10], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"][0], expected_cu_0)
        assert torch.equal(result["cu_seqlens"][1], expected_cu_1)
        assert "cu_seqlens_padded" in result
        assert torch.equal(result["cu_seqlens_padded"][0], expected_cu_padded_0)
        assert torch.equal(result["cu_seqlens_padded"][1], expected_cu_padded_1)

    def test_single_chunk(self):
        """Test with num_chunks=1 (no actual chunking)."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=1)

        # With num_chunks=1, returns same format as process_input_for_thd (no chunk dimension)
        assert result["input_ids"].shape == (6,)
        assert result["labels"].shape == (6,)
        assert result["position_ids"].shape == (6,)
        assert result["cu_seqlens"].shape == (3,)
        assert torch.equal(result["cu_seqlens"], torch.tensor([0, 3, 6], dtype=torch.int32))

    def test_many_chunks(self):
        """Test with many chunks (num_chunks=8)."""
        batch_size = 16
        seq_len = 32
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=8)

        # With 8 chunks, each chunk processes 2 batch items (16 / 8)
        tokens_per_chunk = 2 * seq_len
        assert result["input_ids"].shape == (8, tokens_per_chunk)
        assert result["labels"].shape == (8, tokens_per_chunk)
        assert result["cu_seqlens"].shape[0] == 8

        # Each chunk should have 3 cu_seqlens values: [0, 32, 64]
        for i in range(8):
            assert torch.equal(result["cu_seqlens"][i], torch.tensor([0, 32, 64], dtype=torch.int32))

    def test_non_tensor_key_preservation(self):
        """Test that non-tensor keys are preserved from the batch."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.full((4, 1), 3),
            "seq_lens_padded": torch.full((4, 1), 3),
            "qkv_format": "thd",
            "metadata": {"batch_id": 123},
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2, padding_token_id=0)

        # Non-tensor keys should be preserved
        assert "qkv_format" in result
        assert result["qkv_format"] == "thd"
        assert "metadata" in result
        assert result["metadata"] == {"batch_id": 123}

    def test_custom_seq_lens_padding_value(self):
        """Test with custom seq_lens_padding_value."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3, -999], [3, -999], [3, -999], [3, -999]]),
            "seq_lens_padded": torch.tensor([[3, -999], [3, -999], [3, -999], [3, -999]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2, seq_lens_padding_value=-999)

        # Check that -999 is used for padding in cu_seqlens
        # cu_seqlens for each chunk should be [0, 3, 6] initially
        # After padding they remain [0, 3, 6]
        assert result["cu_seqlens"].shape == (2, 3)

    def test_cu_seqlens_handling(self):
        """Test that cu_seqlens is properly handled with chunking."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 99, 4, 5], [6, 7, 8, 9, 10, 11]] * 2),
            "labels": torch.tensor([[2, 3, 99, 4, 5, 6], [7, 8, 9, 10, 11, 12]] * 2),
            "position_ids": torch.tensor([[0, 1, 2, 0, 0, 1], [0, 1, 2, 3, 4, 5]] * 2),
            "seq_lens": torch.tensor([[3, 2], [6, -1000]] * 2),
            "seq_lens_padded": torch.tensor([[4, 2], [6, -1000]] * 2),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2, padding_token_id=0)

        # cu_seqlens should be present
        assert "cu_seqlens" in result

        # cu_seqlens should have shape [num_chunks, ...]
        assert result["cu_seqlens"].shape[0] == 2

    def test_chunks_equivalence_to_no_chunks(self):
        """Test that chunking with 1 chunk is equivalent to process_input_for_thd."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3]]),
        }

        result_no_chunk = process_input_for_thd(batch)
        result_with_chunk = split_batch_into_thd_chunks(batch, num_chunks=1)

        # Results should match exactly (no batch dimension when num_chunks=1)
        assert torch.equal(result_with_chunk["input_ids"], result_no_chunk["input_ids"])
        assert torch.equal(result_with_chunk["labels"], result_no_chunk["labels"])
        assert torch.equal(result_with_chunk["position_ids"], result_no_chunk["position_ids"])
        assert torch.equal(result_with_chunk["cu_seqlens"], result_no_chunk["cu_seqlens"])

    def test_dtype_preservation_in_chunks(self):
        """Test that dtypes are preserved correctly through chunking."""
        batch = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=torch.long),
            "labels": torch.tensor([[2, 3, 4], [5, 6, 7], [8, 9, 10], [11, 12, 13]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=torch.long),
            "seq_lens": torch.tensor([[3], [3], [3], [3]], dtype=torch.long),
            "seq_lens_padded": torch.tensor([[3], [3], [3], [3]], dtype=torch.long),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2, padding_token_id=0)

        assert result["input_ids"].dtype == torch.long
        assert result["labels"].dtype == torch.long
        assert result["position_ids"].dtype == torch.long
        assert result["cu_seqlens"].dtype == torch.int32

    def test_padding_mask_correctness(self):
        """Test that padding_mask is correctly generated."""
        batch = {
            "input_ids": torch.tensor([[0, 1, 2], [3, 0, 5], [0, 0, 8], [9, 10, 11]]),
            "labels": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]),
            "position_ids": torch.tensor([[0, 1, 2], [0, 1, 2], [0, 1, 2], [0, 1, 2]]),
            "seq_lens": torch.tensor([[3], [3], [3], [3]]),
            "seq_lens_padded": torch.tensor([[3], [3], [3], [3]]),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=2, padding_token_id=0)

        # Check padding_mask shape
        assert result["padding_mask"].shape == (2, 6)

        # seq_lens == seq_lens_padded, so every slot holds a real token. The 0s
        # in input_ids are content (token id 0 is a real token in most
        # vocabularies), and the pack layout -- not the token value -- decides.
        assert not result["padding_mask"].any()

    @pytest.mark.parametrize("num_chunks", [2, 4, 8])
    def test_different_chunk_sizes(self, num_chunks):
        """Test with different numbers of chunks."""
        batch_size = 16
        seq_len = 8
        batch = {
            "input_ids": torch.randint(0, 1000, (batch_size, seq_len)),
            "labels": torch.randint(0, 1000, (batch_size, seq_len)),
            "position_ids": torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1),
            "seq_lens": torch.full((batch_size, 1), seq_len),
            "seq_lens_padded": torch.full((batch_size, 1), seq_len),
        }

        result = split_batch_into_thd_chunks(batch, num_chunks=num_chunks)

        items_per_chunk = batch_size // num_chunks
        tokens_per_chunk = items_per_chunk * seq_len

        assert result["input_ids"].shape == (num_chunks, tokens_per_chunk)
        assert result["cu_seqlens"].shape[0] == num_chunks


class TestTrailingPadAbsorption:
    """Tests for the trailing-pack-pad absorption in process_input_for_thd.

    The original captured bug: a "short" microbatch (5 sub-seqs of 112 in a
    1024-pack) had its trailing 464-token pad absorbed into the last
    cu_seqlens slot (576 wide), while the collator told TE ``max_seqlen=112``
    — a documented TE-contract violation (``max_seqlen_q`` MUST be >= the
    actual max slot width per ``fused_attn.h:548-551``). cuDNN-fused-attn-bwd
    then wrote OOB.

    The fix: compute ``max_seqlen`` from the FINAL cu_seqlens (after
    absorption), so the value handed to TE always reflects the true max slot
    width. With this in place, absorption is contract-clean for any trailing
    pad size, and the previous dummy-slot extension workaround is no longer
    needed. Verified safe via
    ``/opt/Automodel/te_bug_report/te_thd_repro_MINIMAL.py``: TE handles
    a 576-wide slot cleanly when given truthful ``max_seqlen=576``.
    """

    def test_short_microbatch_absorbs_with_truthful_max_seqlen(self):
        """Captured failing case (5×112 + 464 trailing pad): the absorbed
        cu_seqlens last slot is 576 wide. ``max_seqlen`` must reflect that
        post-absorption width, not the pre-absorption 112.
        """
        packed = 1024
        sub = 112
        seq_lens = torch.tensor([[sub, sub, sub, sub, sub, -1000, -1000, -1000, -1000]])
        seq_lens_padded = torch.tensor([[sub, sub, sub, sub, sub + (packed - 5 * sub), -1000, -1000, -1000, -1000]])
        batch = {
            "input_ids": torch.zeros((1, packed), dtype=torch.long),
            "labels": torch.zeros((1, packed), dtype=torch.long),
            "position_ids": torch.arange(packed).unsqueeze(0),
            "seq_lens": seq_lens,
            "seq_lens_padded": seq_lens_padded,
        }
        result = process_input_for_thd(batch)

        # Absorption fires → cu_seqlens = [0,112,224,336,448,1024];
        # last slot is 576 (real 112 + trailing pad 464).
        expected_cu = torch.tensor([0, 112, 224, 336, 448, 1024], dtype=torch.int32)
        assert torch.equal(result["cu_seqlens"], expected_cu), (
            f"cu_seqlens should be absorbed: expected {expected_cu.tolist()}, got {result['cu_seqlens'].tolist()}"
        )
        # cu_seqlens_padded is dropped (equal to cu_seqlens, gated out).
        assert "cu_seqlens_padded" not in result
        # CRITICAL: max_seqlen reflects the absorbed slot width (576), not
        # the pre-absorption max real sub-seq length (112). This is what
        # makes the layout TE-contract-clean.
        assert int(result["max_seqlen"].item()) == 576, (
            f"max_seqlen should reflect post-absorption slot width 576; got {int(result['max_seqlen'].item())}"
        )

    def test_full_microbatch_absorbs_with_bumped_max_seqlen(self):
        """Common-case (9×112 + 16 trailing pad): absorption fires and
        ``max_seqlen`` reflects the absorbed last slot (128), not the
        pre-absorption max (112). The full-pack perf path is preserved.
        """
        packed = 1024
        sub = 112
        seq_lens = torch.tensor([[sub] * 9])
        seq_lens_padded = torch.tensor([[sub] * 8 + [sub + 16]])
        batch = {
            "input_ids": torch.zeros((1, packed), dtype=torch.long),
            "labels": torch.zeros((1, packed), dtype=torch.long),
            "position_ids": torch.arange(packed).unsqueeze(0),
            "seq_lens": seq_lens,
            "seq_lens_padded": seq_lens_padded,
        }
        result = process_input_for_thd(batch)

        # Absorption fires → cu_seqlens[-1] == packed_size.
        assert int(result["cu_seqlens"][-1].item()) == packed
        # cu_seqlens_padded dropped.
        assert "cu_seqlens_padded" not in result, "cu_seqlens_padded should be omitted when absorption fired"
        # max_seqlen reflects the absorbed last slot width = 112 + 16 = 128.
        assert int(result["max_seqlen"].item()) == 128, (
            f"max_seqlen should reflect post-absorption slot 128; got {int(result['max_seqlen'].item())}"
        )

    def test_split_into_chunks_mixed_short_and_full(self):
        """Two-chunk batch where chunk 0 is a near-full pack (16 trailing
        pad) and chunk 1 is short (464 trailing pad). Both absorb; their
        max_seqlen values differ because the absorbed last-slot widths differ.
        """
        packed = 1024
        sub = 112
        seq_lens = torch.tensor(
            [
                [sub] * 9,
                [sub, sub, sub, sub, sub, -1000, -1000, -1000, -1000],
            ]
        )
        seq_lens_padded = torch.tensor(
            [
                [sub] * 8 + [sub + 16],
                [sub, sub, sub, sub, sub + (packed - 5 * sub), -1000, -1000, -1000, -1000],
            ]
        )
        batch = {
            "input_ids": torch.zeros((2, packed), dtype=torch.long),
            "labels": torch.zeros((2, packed), dtype=torch.long),
            "position_ids": torch.arange(packed).unsqueeze(0).expand(2, -1),
            "seq_lens": seq_lens,
            "seq_lens_padded": seq_lens_padded,
        }
        result = split_batch_into_thd_chunks(batch, num_chunks=2, padding_token_id=0)

        assert "cu_seqlens" in result
        # Both chunks absorbed (cu_seqlens_padded == cu_seqlens for both),
        # so split_batch_into_thd_chunks omits the padded key.
        assert "cu_seqlens_padded" not in result

        # Chunk 0 (full pack, absorbed) — last non-sentinel value == packed_size.
        c0_cu = result["cu_seqlens"][0]
        c0_real = c0_cu[c0_cu != -1000]
        assert int(c0_real[-1].item()) == packed
        c0_widths = c0_real[1:] - c0_real[:-1]
        assert int(c0_widths.max().item()) == 128  # absorbed last slot

        # Chunk 1 (short, absorbed) — last non-sentinel value == packed_size,
        # absorbed last slot is wider (576).
        c1_cu = result["cu_seqlens"][1]
        c1_real = c1_cu[c1_cu != -1000]
        assert int(c1_real[-1].item()) == packed
        c1_widths = c1_real[1:] - c1_real[:-1]
        assert int(c1_widths.max().item()) == 576  # absorbed last slot

        # Per-chunk max_seqlen reflects each chunk's max slot width.
        # split_batch_into_thd_chunks stacks them, so result["max_seqlen"]
        # is a tensor of shape (2,).
        assert result["max_seqlen"].shape == (2,)
        assert int(result["max_seqlen"][0].item()) == 128
        assert int(result["max_seqlen"][1].item()) == 576


def test_process_input_for_thd_mrope_3d_shape():
    """3D mRoPE position_ids [n_rope, batch, seq] flatten to [n_rope, 1, batch*seq]."""
    import torch

    from nemo_automodel.components.distributed.thd_utils import process_input_for_thd

    B, S = 2, 4
    pos = torch.arange(S).view(1, 1, S).expand(3, B, S).contiguous()
    batch = {
        "input_ids": torch.arange(B * S).view(B, S),
        "labels": torch.arange(B * S).view(B, S),
        "position_ids": pos,
        "seq_lens": torch.tensor([[4], [4]]),
        "seq_lens_padded": torch.tensor([[4], [4]]),
    }
    out = process_input_for_thd(batch)
    assert tuple(out["position_ids"].shape) == (3, 1, B * S)
    assert tuple(out["input_ids"].shape) == (B * S,)
    assert out["position_ids"][0, 0].tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_process_input_for_thd_2d_position_ids_unchanged():
    """2D position_ids keep the original [total_tokens] flatten."""
    import torch

    from nemo_automodel.components.distributed.thd_utils import process_input_for_thd

    B, S = 2, 4
    batch = {
        "input_ids": torch.arange(B * S).view(B, S),
        "labels": torch.arange(B * S).view(B, S),
        "position_ids": torch.arange(S).view(1, S).expand(B, S).contiguous(),
        "seq_lens": torch.tensor([[4], [4]]),
        "seq_lens_padded": torch.tensor([[4], [4]]),
    }
    out = process_input_for_thd(batch)
    assert tuple(out["position_ids"].shape) == (B * S,)
