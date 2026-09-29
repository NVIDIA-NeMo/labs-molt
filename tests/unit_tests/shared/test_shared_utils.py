# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_automodel.shared.utils import dtype_from_str


class TestDtypeFromStr:
    """Test cases for the dtype_from_str utility function."""

    def test_dtype_from_str_valid_inputs(self):
        """Test dtype_from_str with various valid inputs including torch.dtype objects,
        full paths, prefixless strings, shorthands, case variations, and aliases."""
        test_cases = [
            # torch.dtype objects (bypass)
            (torch.bfloat16, torch.bfloat16),
            (torch.float32, torch.float32),
            # Full torch paths
            ("torch.bfloat16", torch.bfloat16),
            ("torch.float32", torch.float32),
            ("torch.int64", torch.int64),
            ("torch.bool", torch.bool),
            # Without torch prefix
            ("bfloat16", torch.bfloat16),
            ("float32", torch.float32),
            ("int64", torch.int64),
            ("bool", torch.bool),
            # Shorthand
            ("bf16", torch.bfloat16),
            # Case insensitive
            ("TORCH.BFLOAT16", torch.bfloat16),
            ("BF16", torch.bfloat16),
            ("FLOAT32", torch.float32),
            # Aliases
            ("torch.float", torch.float),
            ("torch.double", torch.float64),
            ("torch.half", torch.float16),
            ("torch.long", torch.int64),
            ("torch.short", torch.int16),
        ]

        for input_val, expected_dtype in test_cases:
            result = dtype_from_str(input_val)
            assert result == expected_dtype

    def test_dtype_from_str_invalid_inputs(self):
        """Test dtype_from_str raises KeyError on invalid input."""
        invalid_inputs = ["abc", "torch.invalid", "invalid_dtype", "torch.xyz"]

        for invalid_input in invalid_inputs:
            with pytest.raises(KeyError, match=f"Unknown dtype string: {invalid_input}"):
                dtype_from_str(invalid_input)
