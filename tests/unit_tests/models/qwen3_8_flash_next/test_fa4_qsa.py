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

"""CPU coverage of optional FA4 dispatch and tensor validation."""

import pytest
import torch

from nemo_automodel.components.models.qwen3_8_flash_next import fa4_qsa
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import (
    gathered_qsa_gqa_attention,
    qsa_gqa_attention,
)


def test_cpu_cute_dispatch_keeps_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(4)
    inputs = [torch.randn(1, 7, heads, 4, requires_grad=True) for heads in (4, 2, 2)]
    routes = torch.arange(7).view(1, 1, 7).expand(1, 7, 7)
    monkeypatch.setattr(fa4_qsa, "safe_import", lambda *args: pytest.fail("CPU must not load FA4"))
    actual = qsa_gqa_attention(*inputs, routes, backend="cute")
    expected = gathered_qsa_gqa_attention(*inputs, routes)
    dy = torch.randn_like(actual)
    grads = torch.autograd.grad(actual, inputs, dy)
    refs = torch.autograd.grad(expected, inputs, dy)
    for result, reference in zip((actual, *grads), (expected, *refs)):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)


def test_missing_optional_dependency_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fa4_qsa._load_fa4.cache_clear()
    monkeypatch.setattr(fa4_qsa, "safe_import", lambda name: (False, None))
    with pytest.raises(ImportError, match="FlashAttention.*SM90"):
        fa4_qsa._load_fa4()
    fa4_qsa._load_fa4.cache_clear()


@pytest.mark.parametrize(
    "case", ["rank", "batch", "kv_shape", "head_dim", "gqa", "empty", "routes", "route_dtype", "dtype", "cpu"]
)
def test_cute_contract_rejects_invalid_inputs(case: str) -> None:
    q = torch.empty(1, 3, 4, 256, dtype=torch.bfloat16)
    k = torch.empty(1, 5, 2, 256, dtype=torch.bfloat16)
    v = torch.empty_like(k)
    routes = torch.zeros(1, 3, 4, dtype=torch.int64)
    error, match = ValueError, "same CUDA device"
    if case == "rank":
        q = q.squeeze(0)
        match = "layout"
    elif case == "batch":
        k = k.expand(2, -1, -1, -1)
        v = v.expand_as(k)
        match = "batch"
    elif case == "kv_shape":
        v = v[:, :3]
        match = "matching K/V"
    elif case == "head_dim":
        q = q[..., :128]
        match = "head_dim"
    elif case == "gqa":
        q = q[:, :, :3]
        match = "divisible"
    elif case == "empty":
        q = q[:, :0]
        match = "nonempty"
    elif case == "routes":
        routes = routes[:, :2]
        match = "routes must have shape"
    elif case == "route_dtype":
        routes = routes.float()
        error, match = TypeError, "signed"
    elif case == "dtype":
        q = q.float()
        error, match = TypeError, "BF16"
    with pytest.raises(error, match=match):
        fa4_qsa.fa4_sparse_gqa_attention(q, k, v, routes)


def test_byte_membership_preserves_route_set() -> None:
    routes = torch.tensor(
        [[[0, 0, -1, 1 << 40, 31, 32, 64], [-1] * 7], [[1, 2, 3, 65, -3, 2, 0], [64] * 7]], dtype=torch.int64
    )
    # Keep CPU unit tests focused on semantics, without compiler cold-start overhead.
    raw = fa4_qsa._preprocess.__wrapped__(routes, 65)
    expected = torch.zeros(2, 2, 65, dtype=torch.uint8)
    expected[0, 0, [0, 31, 32, 64]] = 1
    expected[1, 0, [0, 1, 2, 3]] = 1
    expected[1, 1, 64] = 1
    torch.testing.assert_close(raw[0], expected, rtol=0, atol=0)
    assert raw[0].stride(1) % 16 == 0
    assert all(t.is_contiguous() for t in raw[1:])


def test_block_lists_distinguish_full_partial_and_empty() -> None:
    routes = torch.arange(81).expand(1, 129, 81).clone()
    routes[:, 128] = -1
    # Keep CPU unit tests focused on semantics, without compiler cold-start overhead.
    raw = fa4_qsa._preprocess.__wrapped__(routes, 81)
    mask, pc, pi, fc, fi, rpc, rpi, rfc, rfi = raw
    assert mask[:, 128].count_nonzero() == 0
    assert pc.tolist() == [[[1, 0]]]
    assert fc.tolist() == [[[1, 0]]]
    assert pi[0, 0, 0, 0] == 1
    assert fi[0, 0, 0, 0] == 0
    assert rpc.tolist() == [[[0, 1]]]
    assert rfc.tolist() == [[[1, 0]]]
    assert rpi[0, 0, 1, 0] == 0
    assert rfi[0, 0, 0, 0] == 0
