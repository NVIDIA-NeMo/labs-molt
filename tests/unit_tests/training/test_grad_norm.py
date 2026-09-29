# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from nemo_automodel.components.training.triton.grad_norm import (
    _CHUNK,
    HAVE_TRITON,
    _build_chunk_ends,
    multi_tensor_sumsq,
    sumsq_reference,
)
from nemo_automodel.components.training.utils import _local_te_l2_norm
from nemo_automodel.shared.import_utils import safe_import_te


def test_build_chunk_ends_handles_empty_and_boundary_sized_tensors():
    tensors = [torch.empty(0), torch.empty(1), torch.empty(_CHUNK), torch.empty(_CHUNK + 1)]

    chunk_ends, total = _build_chunk_ends(tensors, torch.device("cpu"))

    torch.testing.assert_close(chunk_ends, torch.tensor([0, 1, 2, 4], dtype=torch.int64))
    assert total == 4


@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="requires Triton and CUDA")
def test_multi_tensor_sumsq_matches_fp64_reference_and_is_repeatable():
    gradients = [
        torch.linspace(-3, 3, _CHUNK + 17, device="cuda").to(torch.bfloat16),
        torch.linspace(-0.25, 0.25, 2 * _CHUNK + 1, device="cuda"),
    ]

    expected = sumsq_reference(gradients)
    actual = [multi_tensor_sumsq(gradients).cpu() for _ in range(3)]

    torch.testing.assert_close(actual[0], expected, rtol=1e-12, atol=1e-12)
    assert all(torch.equal(actual[0], repeated) for repeated in actual[1:])


@pytest.mark.skipif(not HAVE_TRITON or not torch.cuda.is_available(), reason="requires Triton and CUDA")
def test_te_and_triton_l2_backends_match_fp64_reference():
    if not safe_import_te()[0]:
        pytest.skip("requires Transformer Engine")

    gradients = [
        torch.linspace(-3, 3, _CHUNK + 17, device="cuda").to(torch.bfloat16),
        torch.linspace(-0.25, 0.25, 2 * _CHUNK + 1, device="cuda"),
    ]
    expected = sumsq_reference(gradients).sqrt()

    triton_norm = multi_tensor_sumsq(gradients).sqrt().cpu()
    te_norm = _local_te_l2_norm(gradients, torch.device("cuda")).cpu()

    torch.testing.assert_close(triton_norm, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(te_norm, expected, rtol=2e-6, atol=0)
