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
"""Tests for the optional GLM-5.2 DSA TileLang kernels."""

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components._peft.lora import LinearLoRA, PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm_moe_dsa import cp as cp_mod
from nemo_automodel.components.models.glm_moe_dsa import layers as layer_mod
from nemo_automodel.components.models.glm_moe_dsa import optimized_kernels as ok
from nemo_automodel.components.models.glm_moe_dsa.layers import GlmMoeDsaIndexer, GlmMoeDsaMLA
from nemo_automodel.components.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM
from nemo_automodel.shared.import_utils import UnavailableError

# Over the default 5s budget on purpose: TileLang compilation takes up to about 46s on a cold worker.
# Reduce cold compiler startup before lowering this further.
pytestmark = pytest.mark.timeout(120)

# GLM-5.2 DSA kernel dims (kv_lora_rank + qk_rope_head_dim == 576 is hard-coded in the kernel).
KV_LORA = 512
ROPE = 64
QK_NOPE = 128
V_HEAD = 128
QK_HEAD = QK_NOPE + ROPE
N_HEADS = 16
IDX_HEADS = 16
IDX_DIM = 128
TOPK = 64
T = 256

_run_gpu = (
    ok.is_dsa_kernel_available("sparse_attn") and ok.is_dsa_kernel_available("indexer") and torch.cuda.is_available()
)
requires_kernels = pytest.mark.skipif(not _run_gpu, reason="requires tilelang kernels (CUDA + tilelang installed)")


def test_is_dsa_kernel_available_returns_bool():
    assert isinstance(ok.is_dsa_kernel_available("indexer"), bool)
    assert isinstance(ok.is_dsa_kernel_available("sparse_attn"), bool)
    with pytest.raises(ValueError):
        ok.is_dsa_kernel_available("nope")


def test_should_use_tilelang_torch_is_false():
    tensors = (torch.zeros(2),)
    assert ok.should_use_tilelang("torch", available=True, kernel_name="x", tensors=tensors) is False


def test_should_use_tilelang_auto_falls_back_when_unavailable_or_cpu():
    tensors = (torch.zeros(2),)
    assert ok.should_use_tilelang("auto", available=False, kernel_name="indexer", tensors=tensors) is False
    assert ok.should_use_tilelang("auto", available=True, kernel_name="indexer", tensors=tensors) is False


def test_should_use_tilelang_forced_but_unavailable_raises():
    tensors = (torch.zeros(2),)
    with pytest.raises(RuntimeError, match="TileLang backend was requested"):
        ok.should_use_tilelang("tilelang", available=False, kernel_name="indexer", tensors=tensors)
    with pytest.raises(RuntimeError, match="TileLang backend was requested"):
        ok.should_use_tilelang("tilelang", available=True, kernel_name="indexer", tensors=tensors, require_bf16=True)


class _FakeTileLangValue:
    dtype = "float32"

    def __getitem__(self, _key):
        return self

    def __setitem__(self, _key, _value):
        return None

    def __bool__(self):
        return True

    def __add__(self, _other):
        return self

    __radd__ = __add__

    def __sub__(self, _other):
        return self

    __rsub__ = __sub__

    def __mul__(self, _other):
        return self

    __rmul__ = __mul__

    def __truediv__(self, _other):
        return self

    __rtruediv__ = __truediv__

    def __itruediv__(self, _other):
        return self

    def __neg__(self):
        return self

    def __lt__(self, _other):
        return self

    def __le__(self, _other):
        return self

    def __gt__(self, _other):
        return self

    def __ge__(self, _other):
        return self

    def __eq__(self, _other):
        return self

    def __ne__(self, _other):
        return self


class _FakeKernelContext:
    def __init__(self, axes):
        self.axes = axes

    def __enter__(self):
        if len(self.axes) == 1:
            return 0
        return tuple(0 for _ in self.axes)

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTileLangLanguage:
    bfloat16 = "bfloat16"
    float = "float"
    float32 = "float32"
    int32 = "int32"
    GemmWarpPolicy = SimpleNamespace(FullCol="FullCol", FullRow="FullRow")

    @staticmethod
    def Tensor(_shape, _dtype):
        return _FakeTileLangValue()

    @staticmethod
    def prim_func(fn):
        return fn

    @staticmethod
    def Kernel(*axes, **_kwargs):
        return _FakeKernelContext(axes)

    @staticmethod
    def Parallel(*axes):
        if len(axes) == 1:
            return [0]
        return [tuple(0 for _ in axes)]

    @staticmethod
    def Pipelined(*_args, **_kwargs):
        return [0]

    @staticmethod
    def serial(_count):
        return [0]

    @staticmethod
    def ceildiv(a, b):
        if isinstance(a, int) and isinstance(b, int):
            return (a + b - 1) // b
        return 1

    @staticmethod
    def symbolic(_name):
        return 1

    dynamic = symbolic

    @staticmethod
    def alloc_shared(*_args, **_kwargs):
        return _FakeTileLangValue()

    alloc_fragment = alloc_shared
    alloc_var = alloc_shared

    @staticmethod
    def if_then_else(cond, true_value, false_value):
        return true_value if bool(cond) else false_value

    @staticmethod
    def infinity(_dtype):
        return _FakeTileLangValue()

    @staticmethod
    def max(a, _b):
        return a

    min = max
    exp2 = staticmethod(lambda _x: _FakeTileLangValue())
    log2 = staticmethod(lambda _x: _FakeTileLangValue())

    @staticmethod
    def copy(*_args, **_kwargs):
        return None

    fill = copy
    clear = copy
    reduce_sum = copy
    reduce_max = copy
    gemm = copy
    sync_threads = copy
    atomic_add = copy
    atomic_addx4 = copy
    thread_binding = copy
    reshape = staticmethod(lambda value, _shape: value)


class _FakeTileLangModule:
    PassConfigKey = SimpleNamespace(
        TL_DISABLE_TMA_LOWER="TL_DISABLE_TMA_LOWER",
        TL_DISABLE_WARP_SPECIALIZED="TL_DISABLE_WARP_SPECIALIZED",
        TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE="TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE",
        TL_ENABLE_FAST_MATH="TL_ENABLE_FAST_MATH",
    )

    class math:
        @staticmethod
        def next_power_of_2(value):
            return 1 << (value - 1).bit_length()

    @staticmethod
    def cdiv(a, b):
        return (a + b - 1) // b


def _patch_fake_tilelang(monkeypatch, module):
    fake_t = _FakeTileLangLanguage()
    fake_tilelang = _FakeTileLangModule()
    monkeypatch.setattr(module, "T", fake_t)
    if hasattr(module, "tilelang"):
        monkeypatch.setattr(module, "tilelang", fake_tilelang)
    if hasattr(module, "tl"):
        monkeypatch.setattr(module, "tl", fake_tilelang)


def test_tilelang_shim_handles_missing_and_delegates(monkeypatch):
    from nemo_automodel.components.models.glm_moe_dsa.kernels import _tilelang

    assert _tilelang.tilelang.math.next_power_of_2(7) == 8
    assert _tilelang.tilelang.cdiv(5, 2) == 3

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "tilelang" or name.startswith("tilelang."):
            raise ImportError("missing tilelang")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(UnavailableError, match="GLM-5.2 DSA TileLang kernels"):
        _tilelang._load_tilelang()

    assert _tilelang._resolve_pass_configs(object(), None) is None

    real_tilelang = SimpleNamespace(math=SimpleNamespace(custom_math="math-value"), custom_attr="tilelang-value")
    real_language = SimpleNamespace(custom_lang="language-value", prim_func=lambda fn: ("prim", fn))
    monkeypatch.setattr(_tilelang, "_load_tilelang", lambda: (real_tilelang, real_language))

    assert _tilelang.tilelang.math.custom_math == "math-value"
    assert _tilelang.tilelang.custom_attr == "tilelang-value"
    assert _tilelang.T.custom_lang == "language-value"
    prim_result, _ = _tilelang.T.prim_func(lambda: None)
    assert prim_result == "prim"


def test_tilelang_shim_loads_real_modules(monkeypatch):
    from nemo_automodel.components.models.glm_moe_dsa.kernels import _tilelang

    tilelang_module = ModuleType("tilelang")
    language_module = ModuleType("tilelang.language")
    tilelang_module.language = language_module
    monkeypatch.setitem(sys.modules, "tilelang", tilelang_module)
    monkeypatch.setitem(sys.modules, "tilelang.language", language_module)

    assert _tilelang._load_tilelang() == (tilelang_module, language_module)


def test_tilelang_shim_lazy_jit_resolves_pass_configs_and_caches(monkeypatch):
    from nemo_automodel.components.models.glm_moe_dsa.kernels import _tilelang

    compiled_calls = []
    load_calls = []
    fast_math_key = object()

    def fake_jit(*jit_args, **jit_kwargs):
        compiled_calls.append((jit_args, jit_kwargs))

        def decorate(fn):
            def compiled(*args, **kwargs):
                return fn(*args, **kwargs)

            return compiled

        return decorate

    real_tilelang = SimpleNamespace(
        PassConfigKey=SimpleNamespace(TL_ENABLE_FAST_MATH=fast_math_key),
        jit=fake_jit,
    )

    def fake_load_tilelang():
        load_calls.append(True)
        return real_tilelang, SimpleNamespace()

    monkeypatch.setattr(_tilelang, "_load_tilelang", fake_load_tilelang)

    @_tilelang.tilelang.jit("kernel-arg", pass_configs={"TL_ENABLE_FAST_MATH": True})
    def add_one(value):
        return value + 1

    assert add_one(1) == 2
    assert add_one(2) == 3
    assert len(load_calls) == 1
    assert compiled_calls == [(("kernel-arg",), {"pass_configs": {fast_math_key: True}})]

    @_tilelang.tilelang.jit
    def identity(value):
        return value

    assert identity("value") == "value"
    assert len(load_calls) == 2


def test_tilelang_indexer_topk_dispatches_varlen_helpers(monkeypatch):
    index_q = torch.randn(4, 2, 3)
    index_k = torch.randn(4, 3)
    head_weights = torch.randn(4, 2)
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int64)
    starts = torch.tensor([0, 2], dtype=torch.int64)
    ends = torch.tensor([2, 4], dtype=torch.int64)
    captured = {}

    def fake_generate_varlen_mask_params(received_cu_seqlens):
        captured["cu_seqlens"] = received_cu_seqlens
        return starts, ends

    def fake_lighting_indexer(received_q, received_k, received_weights, received_starts, received_ends, topk):
        captured.update(
            q=received_q,
            k=received_k,
            weights=received_weights,
            starts=received_starts,
            ends=received_ends,
            topk=topk,
        )
        scores = torch.zeros(4, 4)
        topk_indices = torch.tensor([[0, -1], [1, 0], [2, 1], [3, 0]], dtype=torch.int64)
        return scores, topk_indices

    monkeypatch.setattr(ok, "_slime_generate_varlen_mask_params", fake_generate_varlen_mask_params)
    monkeypatch.setattr(ok, "_slime_lighting_indexer", fake_lighting_indexer)

    topk = ok.tilelang_indexer_topk(index_q, index_k, head_weights, cu_seqlens, index_topk=2)

    assert captured["cu_seqlens"] is cu_seqlens
    assert captured["q"] is index_q
    assert captured["k"] is index_k
    assert captured["weights"] is head_weights
    assert captured["starts"].dtype == torch.int32
    assert captured["ends"].dtype == torch.int32
    assert captured["topk"] == 2
    assert topk.shape == (4, 1, 2)
    assert topk.dtype == torch.int32
    assert topk.is_contiguous()


def test_tilelang_indexer_topk_uses_padded_windows_and_query_indices(monkeypatch):
    index_q = torch.randn(3, 2, 3)
    index_k = torch.randn(8, 3)
    head_weights = torch.randn(3, 2)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int64)
    cu_seqlens_padded = torch.tensor([0, 4, 8], dtype=torch.int64)
    query_indices = torch.tensor([1, 4, 6], dtype=torch.int64)
    captured = {}

    def fake_lighting_indexer(received_q, received_k, received_weights, received_starts, received_ends, topk):
        captured.update(
            q=received_q,
            k=received_k,
            weights=received_weights,
            starts=received_starts,
            ends=received_ends,
            topk=topk,
        )
        return torch.zeros(3, 8), torch.tensor([[1, 0], [4, -1], [6, 5]], dtype=torch.int64)

    monkeypatch.setattr(ok, "_slime_lighting_indexer", fake_lighting_indexer)

    topk = ok.tilelang_indexer_topk(
        index_q,
        index_k,
        head_weights,
        cu_seqlens,
        index_topk=2,
        query_indices=query_indices,
        cu_seqlens_padded=cu_seqlens_padded,
    )

    assert captured["q"] is index_q
    assert captured["k"] is index_k
    assert captured["weights"] is head_weights
    assert captured["starts"].tolist() == [0, 4, 4]
    assert captured["ends"].tolist() == [2, 5, 7]
    assert captured["starts"].dtype == torch.int32
    assert captured["ends"].dtype == torch.int32
    assert captured["topk"] == 2
    assert topk.tolist() == [[[1, 0]], [[4, -1]], [[6, 5]]]


def test_generate_padded_varlen_mask_params_excludes_cp_padding():
    starts, ends = ok._generate_padded_varlen_mask_params(
        torch.tensor([0, 2, 5], dtype=torch.int64),
        torch.tensor([0, 4, 8], dtype=torch.int64),
    )

    assert starts.tolist() == [0, 0, 0, 0, 4, 4, 4, 4]
    assert ends.tolist() == [1, 2, 2, 2, 5, 6, 7, 7]
    assert starts.dtype == torch.int32
    assert ends.dtype == torch.int32


def test_tilelang_topk_chunks_rows_without_changing_results(monkeypatch):
    from nemo_automodel.components.models.glm_moe_dsa.kernels import indexer as indexer_mod

    logits = torch.randn(7, 11)
    expected_scores, expected_indices = torch.topk(logits, 4, dim=-1)
    original_topk = torch.topk
    call_shapes = []

    def tracked_topk(values, *args, **kwargs):
        call_shapes.append(tuple(values.shape))
        return original_topk(values, *args, **kwargs)

    monkeypatch.setattr(indexer_mod, "_TOPK_MAX_ELEMENTS_PER_CALL", 22)
    monkeypatch.setattr(indexer_mod.torch, "topk", tracked_topk)

    scores, indices = indexer_mod._topk_in_row_chunks(logits, 4)

    assert call_shapes == [(2, 11), (2, 11), (2, 11), (1, 11)]
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(indices, expected_indices)


def test_tilelang_topk_keeps_small_inputs_in_one_call(monkeypatch):
    from nemo_automodel.components.models.glm_moe_dsa.kernels import indexer as indexer_mod

    logits = torch.randn(3, 5)
    original_topk = torch.topk
    call_shapes = []

    def tracked_topk(values, *args, **kwargs):
        call_shapes.append(tuple(values.shape))
        return original_topk(values, *args, **kwargs)

    monkeypatch.setattr(indexer_mod, "_TOPK_MAX_ELEMENTS_PER_CALL", logits.numel())
    monkeypatch.setattr(indexer_mod.torch, "topk", tracked_topk)

    scores, indices = indexer_mod._topk_in_row_chunks(logits, 2)
    expected_scores, expected_indices = original_topk(logits, 2, dim=-1)

    assert call_shapes == [tuple(logits.shape)]
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(indices, expected_indices)


def test_indexer_generates_topk_and_varlen_mask_params(monkeypatch):
    from nemo_automodel.components.models.glm_moe_dsa.kernels import indexer as indexer_mod

    captured = {}

    def fake_indexer_fwd(index_q, _index_k, _weights, _cu_seqlen_ks, _cu_seqlen_ke, clean_logits=True):
        captured["clean_logits"] = clean_logits
        return index_q.new_tensor([[3.0, float("-inf"), 1.0], [float("-inf"), float("-inf"), float("-inf")]])

    def fake_indexer_bwd(index_q, weights, index_k, topk_indices, grad_scores):
        captured.update(backward_topk=topk_indices, backward_grad_scores=grad_scores)
        return torch.ones_like(index_q), torch.ones_like(weights), torch.ones_like(index_k)

    monkeypatch.setattr(indexer_mod, "indexer_fwd_interface", fake_indexer_fwd)
    monkeypatch.setattr(indexer_mod, "indexer_bwd_interface", fake_indexer_bwd)
    monkeypatch.setattr(indexer_mod, "_TOPK_MAX_ELEMENTS_PER_CALL", 3)

    cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32)
    starts, ends = indexer_mod.generate_varlen_mask_params(cu_seqlens)
    torch.testing.assert_close(starts, torch.tensor([0, 0, 0, 3, 3], dtype=torch.int32))
    torch.testing.assert_close(ends, torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64))

    index_q = torch.randn(2, 2, 3, requires_grad=True)
    index_k = torch.randn(3, 3, requires_grad=True)
    weights = torch.randn(2, 2, requires_grad=True)
    scores, topk_indices = indexer_mod.lighting_indexer(index_q, index_k, weights, starts[:2], ends[:2], 2)
    scores.sum().backward()

    torch.testing.assert_close(topk_indices, torch.tensor([[0, 2], [-1, -1]], dtype=torch.int32))
    assert captured["clean_logits"] is True
    torch.testing.assert_close(captured["backward_topk"], topk_indices)
    torch.testing.assert_close(index_q.grad, torch.ones_like(index_q))
    torch.testing.assert_close(index_k.grad, torch.ones_like(index_k))
    torch.testing.assert_close(weights.grad, torch.ones_like(weights))


def test_tilelang_sparse_attention_dispatches_and_projects_value(monkeypatch):
    q = torch.randn(3, 2, 4)
    kv_latent = torch.randn(3, 1, 4)
    topk_indices = torch.tensor([[[0, -1]], [[1, 0]], [[2, 1]]], dtype=torch.int32)
    w_vc = torch.randn(2, 5, 4)
    latent_out = torch.randn(3, 2, 4)
    captured = {}

    class FakeSparseMLA:
        @staticmethod
        def apply(received_q, received_kv, received_topk, received_scale):
            captured.update(q=received_q, kv=received_kv, topk=received_topk, scale=received_scale)
            lse = torch.zeros(received_q.shape[:2])
            return latent_out, lse

    monkeypatch.setattr(ok, "_slime_sparse_mla", FakeSparseMLA)

    out = ok.tilelang_sparse_attention(q, kv_latent, topk_indices, w_vc, softmax_scale=0.125)

    assert captured["q"] is q
    assert captured["kv"] is kv_latent
    assert captured["topk"] is topk_indices
    assert captured["scale"] == 0.125
    expected = torch.einsum("thc,hdc->thd", latent_out, w_vc)
    torch.testing.assert_close(out, expected)


def test_indexer_autograd_backward_dispatches_kernel(monkeypatch):
    from nemo_automodel.components.models.glm_moe_dsa.kernels import indexer as indexer_mod

    captured = {}

    def fake_indexer_fwd(index_q, _index_k, _weights, _cu_seqlen_ks, _cu_seqlen_ke, clean_logits=True):
        captured["clean_logits"] = clean_logits
        return index_q[..., 0].sum(dim=-1, keepdim=True).expand(index_q.shape[0], 2)

    def fake_indexer_bwd(index_q, weights, index_k, topk_indices, grad_scores):
        captured.update(
            backward_q=index_q,
            backward_weights=weights,
            backward_k=index_k,
            backward_topk=topk_indices,
            backward_grad_scores=grad_scores,
        )
        return torch.ones_like(index_q), torch.ones_like(weights), torch.ones_like(index_k)

    monkeypatch.setattr(indexer_mod, "indexer_fwd_interface", fake_indexer_fwd)
    monkeypatch.setattr(indexer_mod, "indexer_bwd_interface", fake_indexer_bwd)

    index_q = torch.randn(2, 2, 3, requires_grad=True)
    index_k = torch.randn(2, 3, requires_grad=True)
    weights = torch.randn(2, 2, requires_grad=True)
    starts = torch.tensor([0, 0], dtype=torch.int32)
    ends = torch.tensor([1, 2], dtype=torch.int32)
    topk_indices = torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)

    scores, returned_topk = indexer_mod.lighting_indexer(index_q, index_k, weights, starts, ends, 2, topk_indices)
    scores.sum().backward()

    torch.testing.assert_close(returned_topk, topk_indices)
    assert captured["clean_logits"] is True
    assert captured["backward_q"] is index_q
    assert captured["backward_k"] is index_k
    assert captured["backward_topk"] is topk_indices
    torch.testing.assert_close(index_q.grad, torch.ones_like(index_q))
    torch.testing.assert_close(index_k.grad, torch.ones_like(index_k))
    torch.testing.assert_close(weights.grad, torch.ones_like(weights))


def test_sparse_mla_autograd_backward_dispatches_kernel(monkeypatch):
    from nemo_automodel.components.models.glm_moe_dsa.kernels import sparse_mla as sparse_mla_mod

    captured = {}

    def fake_sparse_mla_fwd(q, kv, indices, sm_scale=None):
        captured.update(forward_q=q, forward_kv=kv, forward_indices=indices, forward_scale=sm_scale)
        out = torch.ones(q.shape[0], q.shape[1], 4, dtype=q.dtype)
        lse = torch.zeros(q.shape[0], q.shape[1], dtype=torch.float32)
        return out, lse

    def fake_sparse_mla_bwd(q, kv, out, grad_output, indices, lse, sm_scale=None):
        captured.update(
            backward_q=q,
            backward_kv=kv,
            backward_out=out,
            backward_grad_output=grad_output,
            backward_indices=indices,
            backward_lse=lse,
            backward_scale=sm_scale,
        )
        return torch.ones_like(q), torch.ones_like(kv)

    monkeypatch.setattr(sparse_mla_mod, "sparse_mla_fwd_interface", fake_sparse_mla_fwd)
    monkeypatch.setattr(sparse_mla_mod, "sparse_mla_bwd", fake_sparse_mla_bwd)

    q = torch.randn(3, 2, 6, requires_grad=True)
    kv = torch.randn(3, 1, 6, requires_grad=True)
    indices = torch.tensor([[[0, -1]], [[1, 0]], [[2, 1]]], dtype=torch.int32)

    out, lse = sparse_mla_mod.SparseMLA.apply(q, kv, indices, 0.25)
    (out.sum() + lse.sum()).backward()

    assert captured["forward_q"].is_contiguous()
    assert captured["forward_kv"].is_contiguous()
    assert captured["forward_indices"].is_contiguous()
    assert captured["forward_scale"] == 0.25
    assert captured["backward_q"].shape == q.shape
    assert captured["backward_kv"].shape == kv.shape
    assert captured["backward_grad_output"].is_contiguous()
    assert captured["backward_scale"] == 0.25
    torch.testing.assert_close(q.grad, torch.ones_like(q))
    torch.testing.assert_close(kv.grad, torch.ones_like(kv))


def test_tilelang_indexer_backward_interface_launches_kernel(monkeypatch):
    from nemo_automodel.components.models.glm_moe_dsa.kernels import tilelang_indexer_bwd as indexer_bwd_mod

    captured = {}

    def fake_kernel_builder(head_num, head_dim, topk):
        captured.update(head_num=head_num, head_dim=head_dim, topk=topk)

        def fake_kernel(index_q, index_k, weights, topk_indices, grad_scores, grad_q, grad_w, grad_k):
            captured.update(
                index_q=index_q,
                index_k=index_k,
                weights=weights,
                topk_indices=topk_indices,
                grad_scores=grad_scores,
            )
            grad_q.copy_(torch.ones_like(grad_q))
            grad_w.copy_(torch.full_like(grad_w, 2.0))
            grad_k.copy_(torch.full_like(grad_k, 3.0))

        return fake_kernel

    monkeypatch.setattr(indexer_bwd_mod, "tl_indexer_bwd_impl", fake_kernel_builder)

    index_q = torch.randn(2, 4, 3)
    weights = torch.randn(2, 4)
    index_k = torch.randn(5, 3)
    topk_indices = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32)
    grad_scores = torch.randn(2, 2)

    grad_q, grad_w, grad_k = indexer_bwd_mod.indexer_bwd_interface(
        index_q,
        weights,
        index_k,
        topk_indices,
        grad_scores,
    )

    assert captured["head_num"] == 4
    assert captured["head_dim"] == 3
    assert captured["topk"] == 2
    assert captured["index_q"].is_contiguous()
    assert captured["index_k"].is_contiguous()
    assert captured["weights"].is_contiguous()
    assert captured["topk_indices"].is_contiguous()
    assert captured["grad_scores"].is_contiguous()
    torch.testing.assert_close(grad_q, torch.ones_like(grad_q))
    torch.testing.assert_close(grad_w, torch.full_like(grad_w, 2.0))
    torch.testing.assert_close(grad_k, torch.full_like(grad_k, 3.0))


def test_tilelang_sparse_mla_forward_interface_launches_kernel(monkeypatch):
    from nemo_automodel.components.models.glm_moe_dsa.kernels import tilelang_sparse_mla_fwd as sparse_fwd_mod

    captured = {}

    def fake_sparse_mla_fwd(heads, dim, tail_dim, topk, kv_group, sm_scale, is_casual, block_I, num_stages, threads):
        captured.update(
            heads=heads,
            dim=dim,
            tail_dim=tail_dim,
            topk=topk,
            kv_group=kv_group,
            sm_scale=sm_scale,
            is_casual=is_casual,
            block_I=block_I,
            num_stages=num_stages,
            threads=threads,
        )

        def kernel(q, kv, indices):
            captured.update(q=q, kv=kv, indices=indices)
            out = torch.ones(q.shape[0], q.shape[1], q.shape[2], dim, dtype=q.dtype)
            lse = torch.zeros(q.shape[0], q.shape[1], q.shape[2], dtype=torch.float32)
            return out, lse

        return kernel

    monkeypatch.setattr(sparse_fwd_mod, "sparse_mla_fwd", fake_sparse_mla_fwd)

    q = torch.randn(2, 4, 576, dtype=torch.bfloat16).contiguous()
    kv = torch.randn(3, 1, 576, dtype=torch.bfloat16).contiguous()
    indices = torch.zeros(2, 1, 64, dtype=torch.int32).contiguous()

    out, lse = sparse_fwd_mod.sparse_mla_fwd_interface(
        q,
        kv,
        indices,
        sm_scale=0.75,
        d_v=512,
        block_I=32,
        num_stages=3,
        threads=128,
    )

    assert captured["heads"] == 4
    assert captured["dim"] == 512
    assert captured["tail_dim"] == 64
    assert captured["topk"] == 64
    assert captured["kv_group"] == 1
    assert captured["sm_scale"] == 0.75
    assert captured["is_casual"] is True
    assert captured["block_I"] == 32
    assert captured["num_stages"] == 3
    assert captured["threads"] == 128
    assert captured["q"].shape == (1, 2, 4, 576)
    assert captured["kv"].shape == (1, 3, 1, 576)
    assert captured["indices"].shape == (1, 2, 1, 64)
    torch.testing.assert_close(out, torch.ones(2, 4, 512, dtype=torch.bfloat16))
    torch.testing.assert_close(lse, torch.zeros(2, 4, dtype=torch.float32))


def test_tilelang_sparse_mla_backward_wrapper_launches_kernels(monkeypatch):
    from nemo_automodel.components.models.glm_moe_dsa.kernels import tilelang_sparse_mla_bwd as sparse_bwd_mod

    captured = {}

    def fake_preprocess(batch, seq_len, heads, dim):
        captured.update(preprocess=(batch, seq_len, heads, dim))

        def kernel(o, do):
            captured.update(preprocess_o=o, preprocess_do=do)
            return torch.ones(o.shape[:-1], dtype=torch.float32)

        return kernel

    def fake_bwd(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_group, sm_scale, is_causal):
        captured.update(bwd=(batch, seq_len, seq_len_kv, heads, dim, tail_dim, topk, kv_group, sm_scale, is_causal))

        def kernel(q, kv, do, indices, lse, delta, dkv):
            captured.update(bwd_q=q, bwd_kv=kv, bwd_do=do, bwd_indices=indices, bwd_lse=lse, bwd_delta=delta)
            dkv.copy_(torch.full_like(dkv, 2.0))
            return torch.ones_like(q)

        return kernel

    def fake_postprocess(batch, seq_len_kv, dim, tail_dim, kv_group):
        captured.update(postprocess=(batch, seq_len_kv, dim, tail_dim, kv_group))

        def kernel(dkv):
            captured["postprocess_dkv"] = dkv
            return dkv.to(torch.bfloat16)

        return kernel

    monkeypatch.setattr(sparse_bwd_mod, "preprocess", fake_preprocess)
    monkeypatch.setattr(sparse_bwd_mod, "bwd", fake_bwd)
    monkeypatch.setattr(sparse_bwd_mod, "postprocess", fake_postprocess)

    q = torch.randn(2, 4, 576, dtype=torch.bfloat16).contiguous()
    kv = torch.randn(2, 1, 576, dtype=torch.bfloat16).contiguous()
    out = torch.randn(2, 4, 512, dtype=torch.bfloat16).contiguous()
    grad_out = torch.randn(2, 4, 512, dtype=torch.bfloat16).contiguous()
    indices = torch.zeros(2, 1, 64, dtype=torch.int32).contiguous()
    lse = torch.zeros(2, 4, dtype=torch.float32).contiguous()

    dq, dkv = sparse_bwd_mod.sparse_mla_bwd(q, kv, out, grad_out, indices, lse, sm_scale=0.5)

    assert captured["preprocess"] == (1, 2, 4, 512)
    assert captured["bwd"] == (1, 2, 2, 4, 512, 64, 64, 1, 0.5, True)
    assert captured["postprocess"] == (1, 2, 512, 64, 1)
    assert captured["bwd_q"].shape == (1, 2, 4, 576)
    assert captured["bwd_kv"].shape == (1, 2, 1, 576)
    assert captured["bwd_indices"].shape == (1, 2, 1, 64)
    torch.testing.assert_close(dq, torch.ones_like(q))
    torch.testing.assert_close(dkv, torch.full_like(kv, 2.0))


def test_raw_tilelang_kernel_builders_with_fake_language(monkeypatch):
    from nemo_automodel.components.models.glm_moe_dsa.kernels import tilelang_indexer_bwd as indexer_bwd_mod
    from nemo_automodel.components.models.glm_moe_dsa.kernels import tilelang_sparse_mla_bwd as sparse_bwd_mod
    from nemo_automodel.components.models.glm_moe_dsa.kernels import tilelang_sparse_mla_fwd as sparse_fwd_mod

    for module in (indexer_bwd_mod, sparse_bwd_mod, sparse_fwd_mod):
        _patch_fake_tilelang(monkeypatch, module)

    fake = _FakeTileLangValue()

    indexer_kernel = indexer_bwd_mod.tl_indexer_bwd_impl.__wrapped__(heads=8, dim=4, topk=32, block_I=32)
    indexer_kernel(fake, fake, fake, fake, fake, fake, fake, fake)

    preprocess_kernel = sparse_bwd_mod.preprocess.__wrapped__(1, 1, 1, 512)
    preprocess_kernel(fake, fake, fake)
    postprocess_kernel = sparse_bwd_mod.postprocess.__wrapped__(1, 1, 512, 64)
    postprocess_kernel(fake, fake)
    sparse_bwd_kernel = sparse_bwd_mod.bwd.__wrapped__(1, 1, 1, 16, 512, 64, 32, sm_scale=None)
    sparse_bwd_kernel(fake, fake, fake, fake, fake, fake, fake, fake)

    sparse_fwd_kernel = sparse_fwd_mod.sparse_mla_fwd.__wrapped__(
        heads=8,
        dim=512,
        tail_dim=64,
        topk=64,
        kv_group=1,
        sm_scale=None,
    )
    sparse_fwd_kernel(fake, fake, fake, fake, fake)
    sparse_fwd_kernel = sparse_fwd_mod.sparse_mla_fwd.__wrapped__(
        heads=128,
        dim=512,
        tail_dim=64,
        topk=64,
        kv_group=1,
        sm_scale=0.5,
    )
    sparse_fwd_kernel(fake, fake, fake, fake, fake)


def _dense_indexer_logits(index_q, index_k, weights_raw, scale):
    """Dense reference for the lighting indexer: relu(q dot k * scale), then head-weighted sum."""
    scores = torch.relu(torch.einsum("thd,sd->ths", index_q.float(), index_k.float()) * scale)
    logits = torch.einsum("th,ths->ts", weights_raw.float(), scores)
    causal = torch.ones(logits.shape[-2], logits.shape[-1], dtype=torch.bool, device=logits.device).triu(1)
    return logits.masked_fill(causal, float("-inf"))


@requires_kernels
def test_indexer_topk_matches_dense():
    torch.manual_seed(0)
    dev = "cuda"
    scale = IDX_DIM**-0.5
    index_q = torch.randn(T, IDX_HEADS, IDX_DIM, device=dev, dtype=torch.bfloat16)
    index_k = torch.randn(T, IDX_DIM, device=dev, dtype=torch.bfloat16)
    weights_proj = torch.randn(T, IDX_HEADS, device=dev, dtype=torch.float32)

    head_weights = (weights_proj * (IDX_HEADS**-0.5) * scale).contiguous()
    cu_seqlens = torch.tensor([0, T], device=dev, dtype=torch.int32)
    topk_tl = ok.tilelang_indexer_topk(index_q.contiguous(), index_k.contiguous(), head_weights, cu_seqlens, TOPK)
    assert topk_tl.shape == (T, 1, TOPK)
    assert topk_tl.dtype == torch.int32

    logits = _dense_indexer_logits(index_q, index_k, weights_proj * (IDX_HEADS**-0.5), scale)
    topk_ref = logits.topk(TOPK, dim=-1).indices

    tilelang_topk = topk_tl.squeeze(1)
    overlaps = []
    for t in range(TOPK, T):
        actual = set(tilelang_topk[t].tolist()) - {-1}
        expected = set(topk_ref[t].tolist())
        overlaps.append(len(actual & expected) / len(expected))
    mean_overlap = sum(overlaps) / len(overlaps)
    assert mean_overlap > 0.97, f"indexer top-k set overlap too low: {mean_overlap:.4f}"


@requires_kernels
def test_indexer_topk_short_sequence_below_index_topk():
    """When the packed sequence has fewer keys than index_topk, the indexer must not

    crash on torch.topk ("selected index k out of range"); it should select the
    available keys and pad back to the fixed index_topk width with the -1 sentinel so
    the sparse-MLA kernel still gets a width that is a multiple of block_I (64).
    """
    torch.manual_seed(0)
    dev = "cuda"
    short_t = TOPK // 2  # e.g. 32 < TOPK (64): triggers the short-sequence path
    assert short_t < TOPK
    scale = IDX_DIM**-0.5
    index_q = torch.randn(short_t, IDX_HEADS, IDX_DIM, device=dev, dtype=torch.bfloat16)
    index_k = torch.randn(short_t, IDX_DIM, device=dev, dtype=torch.bfloat16)
    weights_proj = torch.randn(short_t, IDX_HEADS, device=dev, dtype=torch.float32)

    head_weights = (weights_proj * (IDX_HEADS**-0.5) * scale).contiguous()
    cu_seqlens = torch.tensor([0, short_t], device=dev, dtype=torch.int32)
    topk_tl = ok.tilelang_indexer_topk(index_q.contiguous(), index_k.contiguous(), head_weights, cu_seqlens, TOPK)

    # Width stays at the fixed index_topk (multiple of block_I) even though only short_t keys exist.
    assert topk_tl.shape == (short_t, 1, TOPK)
    assert TOPK % 64 == 0
    tilelang_topk = topk_tl.squeeze(1)
    # Every selected (non-sentinel) index must be a valid causal key position (< short_t).
    valid = tilelang_topk[tilelang_topk != -1]
    assert int(valid.min()) >= 0 and int(valid.max()) < short_t
    # Each query keeps at most (its causal window) real keys; the rest are -1 padding.
    for t in range(short_t):
        n_real = int((tilelang_topk[t] != -1).sum())
        assert n_real <= t + 1


def _causal_topk_indices(num_tokens, topk, device):
    idx = torch.full((num_tokens, topk), -1, device=device, dtype=torch.int32)
    for t in range(num_tokens):
        n = min(t + 1, topk)
        idx[t, :n] = torch.randperm(t + 1, device=device)[:n].to(torch.int32)
    return idx


def _dense_sparse_mla(q_nope, q_pe, kv_c, k_pe, w_kc, w_vc, topk_idx, scale):
    """Dense MLA attention restricted to the per-query top-k key list."""
    k_nope = torch.einsum("tc,hjc->thj", kv_c.float(), w_kc.float())
    v = torch.einsum("tc,hjc->thj", kv_c.float(), w_vc.float())
    k = torch.cat([k_nope, k_pe.float().unsqueeze(1).expand(-1, q_nope.shape[1], -1)], dim=-1)
    q = torch.cat([q_nope.float(), q_pe.float()], dim=-1)
    scores = torch.einsum("thd,shd->ths", q, k) * scale

    keep = torch.zeros(scores.shape[0], scores.shape[-1], dtype=torch.bool, device=scores.device)
    valid_mask = topk_idx >= 0
    rows = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1).expand_as(topk_idx)[valid_mask]
    cols = topk_idx.long()[valid_mask]
    keep[rows, cols] = True
    scores = scores.masked_fill(~keep.unsqueeze(1), float("-inf"))
    probs = scores.softmax(dim=-1)
    return torch.einsum("ths,shj->thj", probs, v)


@requires_kernels
def test_sparse_mla_absorbed_matches_dense():
    torch.manual_seed(0)
    dev = "cuda"
    scale = QK_HEAD**-0.5
    q_nope = torch.randn(T, N_HEADS, QK_NOPE, device=dev, dtype=torch.bfloat16)
    q_pe = torch.randn(T, N_HEADS, ROPE, device=dev, dtype=torch.bfloat16)
    kv_c = torch.randn(T, KV_LORA, device=dev, dtype=torch.bfloat16)
    k_pe = torch.randn(T, ROPE, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N_HEADS, QK_NOPE + V_HEAD, KV_LORA, device=dev, dtype=torch.bfloat16) * (KV_LORA**-0.5)
    w_kc = w[:, :QK_NOPE, :]
    w_vc = w[:, QK_NOPE:, :]
    topk_idx = _causal_topk_indices(T, TOPK, dev)

    q_absorbed = torch.einsum("thd,hdc->thc", q_nope, w_kc)
    q_tl = torch.cat([q_absorbed, q_pe], dim=-1).to(torch.bfloat16)
    kv_latent = torch.cat([kv_c, k_pe], dim=-1).unsqueeze(1).to(torch.bfloat16)
    out_tl = ok.tilelang_sparse_attention(q_tl, kv_latent, topk_idx.view(T, 1, TOPK).contiguous(), w_vc, scale)

    out_ref = _dense_sparse_mla(q_nope, q_pe, kv_c, k_pe, w_kc, w_vc, topk_idx, scale)
    cosine = torch.nn.functional.cosine_similarity(out_tl.float().flatten(), out_ref.float().flatten(), dim=0).item()
    assert cosine > 0.99, f"sparse MLA cosine vs dense oracle too low: {cosine:.5f}"


@requires_kernels
def test_sparse_mla_lora_effective_weight_matches_dense_forward_and_backward():
    """BF16 TileLang must match the dense oracle for materialized LoRA K/V weights."""
    torch.manual_seed(0)
    dev = "cuda"
    base = torch.nn.Linear(
        KV_LORA,
        N_HEADS * (QK_NOPE + V_HEAD),
        bias=False,
        device=dev,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        base.weight.normal_(std=KV_LORA**-0.5)
    tilelang_lora = LinearLoRA(base, dim=8, alpha=16).train()
    reference_lora = LinearLoRA(base, dim=8, alpha=16).train()
    with torch.no_grad():
        tilelang_lora.lora_A.weight.normal_(std=0.02)
        tilelang_lora.lora_B.weight.normal_(std=0.02)
        reference_lora.lora_A.weight.copy_(tilelang_lora.lora_A.weight)
        reference_lora.lora_B.weight.copy_(tilelang_lora.lora_B.weight)

    q_nope = torch.randn(T, N_HEADS, QK_NOPE, device=dev, dtype=torch.bfloat16, requires_grad=True)
    q_pe = torch.randn(T, N_HEADS, ROPE, device=dev, dtype=torch.bfloat16, requires_grad=True)
    kv_c = torch.randn(T, KV_LORA, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k_pe = torch.randn(T, ROPE, device=dev, dtype=torch.bfloat16, requires_grad=True)
    q_nope_ref = q_nope.detach().clone().requires_grad_(True)
    q_pe_ref = q_pe.detach().clone().requires_grad_(True)
    kv_c_ref = kv_c.detach().clone().requires_grad_(True)
    k_pe_ref = k_pe.detach().clone().requires_grad_(True)
    topk_idx = _causal_topk_indices(T, TOPK, dev)
    scale = QK_HEAD**-0.5

    weight = tilelang_lora.materialize_effective_weight().view(N_HEADS, QK_NOPE + V_HEAD, KV_LORA)
    w_kc = weight[:, :QK_NOPE, :]
    w_vc = weight[:, QK_NOPE:, :]
    q_absorbed = torch.einsum("thd,hdc->thc", q_nope, w_kc)
    q_tl = torch.cat([q_absorbed, q_pe], dim=-1)
    kv_latent = torch.cat([kv_c, k_pe], dim=-1).unsqueeze(1)
    out_tl = ok.tilelang_sparse_attention(q_tl, kv_latent, topk_idx.view(T, 1, TOPK).contiguous(), w_vc, scale)

    reference_weight = reference_lora.materialize_effective_weight().view(
        N_HEADS,
        QK_NOPE + V_HEAD,
        KV_LORA,
    )
    out_ref = _dense_sparse_mla(
        q_nope_ref,
        q_pe_ref,
        kv_c_ref,
        k_pe_ref,
        reference_weight[:, :QK_NOPE, :],
        reference_weight[:, QK_NOPE:, :],
        topk_idx,
        scale,
    )
    output_cosine = torch.nn.functional.cosine_similarity(
        out_tl.float().flatten(),
        out_ref.float().flatten(),
        dim=0,
    ).item()
    assert output_cosine > 0.99, f"LoRA sparse MLA output cosine too low: {output_cosine:.5f}"

    output_grad = torch.randn_like(out_tl)
    out_tl.backward(output_grad)
    out_ref.backward(output_grad.float())
    gradient_pairs = {
        "q_nope": (q_nope.grad, q_nope_ref.grad),
        "q_pe": (q_pe.grad, q_pe_ref.grad),
        "kv_c": (kv_c.grad, kv_c_ref.grad),
        "k_pe": (k_pe.grad, k_pe_ref.grad),
        "lora_A": (tilelang_lora.lora_A.weight.grad, reference_lora.lora_A.weight.grad),
        "lora_B": (tilelang_lora.lora_B.weight.grad, reference_lora.lora_B.weight.grad),
    }
    for name, (actual_grad, reference_grad) in gradient_pairs.items():
        assert actual_grad is not None and reference_grad is not None
        assert torch.isfinite(actual_grad).all() and torch.isfinite(reference_grad).all()
        gradient_cosine = torch.nn.functional.cosine_similarity(
            actual_grad.float().flatten(),
            reference_grad.float().flatten(),
            dim=0,
        ).item()
        assert gradient_cosine > 0.99, f"LoRA sparse MLA {name} gradient cosine too low: {gradient_cosine:.5f}"


def _small_dsa_config():
    return GlmMoeDsaConfig(
        vocab_size=256,
        hidden_size=512,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        max_position_embeddings=512,
        rms_norm_eps=1e-5,
        attention_bias=False,
        kv_lora_rank=KV_LORA,
        q_lora_rank=256,
        qk_head_dim=QK_HEAD,
        qk_nope_head_dim=QK_NOPE,
        qk_rope_head_dim=ROPE,
        v_head_dim=V_HEAD,
        index_n_heads=IDX_HEADS,
        index_head_dim=IDX_DIM,
        index_topk=TOPK,
        mlp_layer_types=["dense"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        torch_dtype="bfloat16",
    )


def _freqs(num_tokens, rope_dim):
    angles = torch.zeros(num_tokens, rope_dim // 2, dtype=torch.float32)
    return torch.polar(torch.ones_like(angles), angles)


def test_indexer_tilelang_requires_thd_and_cu_seqlens(monkeypatch):
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    indexer = GlmMoeDsaIndexer(config, backend)
    monkeypatch.setattr(layer_mod, "should_use_tilelang", lambda *args, **kwargs: True)

    with pytest.raises(ValueError, match="requires THD"):
        indexer(
            torch.randn(1, 4, config.hidden_size, dtype=torch.bfloat16),
            torch.randn(1, 4, config.q_lora_rank, dtype=torch.bfloat16),
            _freqs(4, config.qk_rope_head_dim).unsqueeze(0),
        )

    with pytest.raises(ValueError, match="requires 'cu_seqlens'"):
        indexer(
            torch.randn(4, config.hidden_size, dtype=torch.bfloat16),
            torch.randn(4, config.q_lora_rank, dtype=torch.bfloat16),
            _freqs(4, config.qk_rope_head_dim),
        )


def test_indexer_tilelang_dispatches_topk(monkeypatch):
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    indexer = GlmMoeDsaIndexer(config, backend)
    monkeypatch.setattr(layer_mod, "should_use_tilelang", lambda *args, **kwargs: True)
    expected_topk = torch.zeros(4, 1, config.index_topk, dtype=torch.int32)
    captured = {}

    def fake_tilelang_indexer_topk(
        q,
        k,
        head_weights,
        cu_seqlens,
        index_topk,
        query_indices=None,
        cu_seqlens_padded=None,
    ):
        captured.update(
            q=q,
            k=k,
            head_weights=head_weights,
            cu_seqlens=cu_seqlens,
            index_topk=index_topk,
            query_indices=query_indices,
            cu_seqlens_padded=cu_seqlens_padded,
        )
        return expected_topk

    monkeypatch.setattr(layer_mod, "tilelang_indexer_topk", fake_tilelang_indexer_topk)
    cu_seqlens = torch.tensor([[0, 4]], dtype=torch.int64)
    query_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    cu_seqlens_padded = torch.tensor([[0, 4]], dtype=torch.int64)

    topk = indexer(
        torch.randn(4, config.hidden_size, dtype=torch.bfloat16),
        torch.randn(4, config.q_lora_rank, dtype=torch.bfloat16),
        _freqs(4, config.qk_rope_head_dim),
        cu_seqlens=cu_seqlens,
        glm_dsa_cp_query_indices=query_indices,
        cu_seqlens_padded=cu_seqlens_padded,
    )

    assert topk is expected_topk
    assert captured["q"].shape == (4, config.index_n_heads, config.index_head_dim)
    assert captured["k"].shape == (4, config.index_head_dim)
    assert captured["head_weights"].shape == (4, config.index_n_heads)
    assert captured["head_weights"].dtype == torch.float32
    assert captured["cu_seqlens"].dtype == torch.int32
    assert captured["cu_seqlens"].tolist() == [0, 4]
    assert captured["index_topk"] == config.index_topk
    assert captured["query_indices"] is query_indices
    assert captured["cu_seqlens_padded"].tolist() == [0, 4]


def test_indexer_cp_gathers_keys_before_tilelang_topk(monkeypatch):
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    indexer = GlmMoeDsaIndexer(config, backend)
    monkeypatch.setattr(layer_mod, "should_use_tilelang", lambda *args, **kwargs: True)
    monkeypatch.setattr(layer_mod, "glm_dsa_cp_enabled", lambda group: group == "cp-group")
    captured = {}

    def fake_cp_all_gather(tensor, *, dim, cp_group):
        captured.setdefault("gathered_shapes", []).append(tuple(tensor.shape))
        assert cp_group == "cp-group"
        return torch.cat([tensor, tensor], dim=dim)

    def fake_tilelang_indexer_topk(q, k, head_weights, cu_seqlens, index_topk, **kwargs):
        captured.update(q=q, k=k, cu_seqlens=cu_seqlens, kwargs=kwargs, index_topk=index_topk)
        return torch.zeros(q.shape[0], 1, index_topk, dtype=torch.int32)

    monkeypatch.setattr(layer_mod, "glm_dsa_cp_all_gather", fake_cp_all_gather)
    monkeypatch.setattr(layer_mod, "tilelang_indexer_topk", fake_tilelang_indexer_topk)

    topk = indexer(
        torch.randn(4, config.hidden_size, dtype=torch.bfloat16),
        torch.randn(4, config.q_lora_rank, dtype=torch.bfloat16),
        _freqs(4, config.qk_rope_head_dim),
        cu_seqlens=torch.tensor([0, 8], dtype=torch.int64),
        glm_dsa_cp_query_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        cu_seqlens_padded=torch.tensor([0, 8], dtype=torch.int64),
        _glm_dsa_cp_group="cp-group",
    )

    assert topk.shape == (4, 1, config.index_topk)
    assert captured["gathered_shapes"] == [(4, config.index_head_dim)]
    assert captured["k"].shape == (8, config.index_head_dim)
    assert captured["kwargs"]["query_indices"].tolist() == [0, 1, 2, 3]
    assert captured["kwargs"]["cu_seqlens_padded"].tolist() == [0, 8]


def test_indexer_cp_rejects_unpacked_bshd(monkeypatch):
    config = _small_dsa_config()
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False)
    indexer = GlmMoeDsaIndexer(config, backend)
    monkeypatch.setattr(layer_mod, "glm_dsa_cp_enabled", lambda group: True)

    with pytest.raises(ValueError, match="requires THD"):
        indexer(
            torch.randn(1, 4, config.hidden_size, dtype=torch.bfloat16),
            torch.randn(1, 4, config.q_lora_rank, dtype=torch.bfloat16),
            _freqs(4, config.qk_rope_head_dim).unsqueeze(0),
            _glm_dsa_cp_group="cp-group",
        )


def test_glm_dsa_tilelang_declares_validation_packing():
    config = _small_dsa_config()
    tilelang_backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    sdpa_backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False)

    assert GlmMoeDsaForCausalLM(config, backend=tilelang_backend).should_pack_validation_with_training() is True
    assert GlmMoeDsaForCausalLM(config, backend=sdpa_backend).should_pack_validation_with_training() is False


def test_glm_dsa_tilelang_pipeline_metas_use_thd_shapes():
    config = _small_dsa_config()
    config.indexer_types = ["shared"]
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    model = GlmMoeDsaForCausalLM(config, backend=backend)
    model.lm_head = None
    seq_len = 32
    topk = config.index_topk

    inputs_meta, outputs_meta = model.get_pipeline_stage_metas(
        is_first=False,
        microbatch_size=4,
        seq_len=seq_len,
        dtype=torch.bfloat16,
    )

    assert len(inputs_meta) == 2
    assert inputs_meta[0].shape == (seq_len, config.hidden_size)
    assert inputs_meta[0].dtype == torch.bfloat16
    assert inputs_meta[1].shape == (seq_len, 1, topk)
    assert inputs_meta[1].dtype == torch.float32
    assert len(outputs_meta) == 2
    assert outputs_meta[0].shape == (seq_len, config.hidden_size)
    assert outputs_meta[1].shape == (seq_len, 1, topk)


def test_glm_dsa_sdpa_pipeline_metas_cap_topk_by_seq_len():
    config = _small_dsa_config()
    config.indexer_types = ["shared"]
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False)
    model = GlmMoeDsaForCausalLM(config, backend=backend)
    model.lm_head = None
    seq_len = 32

    inputs_meta, outputs_meta = model.get_pipeline_stage_metas(
        is_first=False,
        microbatch_size=4,
        seq_len=seq_len,
        dtype=torch.bfloat16,
    )

    assert inputs_meta[1].shape == (4, seq_len, seq_len)
    assert outputs_meta[1].shape == (4, seq_len, seq_len)


def test_glm_dsa_pipeline_metas_omit_topk_without_indexshare():
    config = _small_dsa_config()
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False)
    model = GlmMoeDsaForCausalLM(config, backend=backend)
    model.lm_head = None

    inputs_meta, outputs_meta = model.get_pipeline_stage_metas(
        is_first=False,
        microbatch_size=4,
        seq_len=32,
        dtype=torch.bfloat16,
    )

    assert len(inputs_meta) == 1
    assert inputs_meta[0].shape == (4, 32, config.hidden_size)
    assert len(outputs_meta) == 1
    assert outputs_meta[0].shape == (4, 32, config.hidden_size)


def test_glm_dsa_pipeline_stage_returns_only_hidden_without_indexshare(monkeypatch):
    config = _small_dsa_config()
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False)
    model = GlmMoeDsaForCausalLM(config, backend=backend)
    model.model.embed_tokens = None
    model.lm_head = None
    hidden = torch.zeros(1, 4, config.hidden_size)
    monkeypatch.setattr(model.model, "forward", lambda *args, **kwargs: (hidden, None))

    output = model(hidden)

    assert output is hidden


@pytest.mark.parametrize("attn_backend", ["tilelang", "cudnn"])
def test_glm_dsa_prepare_model_inputs_for_cp_binds_batch_sharder(attn_backend):
    config = _small_dsa_config()
    backend = BackendConfig(attn=attn_backend, linear="torch", rms_norm="torch", rope_fusion=False)
    model = GlmMoeDsaForCausalLM(config, backend=backend)

    prepared = model.prepare_model_inputs_for_cp({"input_ids": torch.arange(8).view(1, 8)}, num_chunks=3)

    sharder = prepared["cp_sharder"]
    from nemo_automodel.components.distributed.context_parallel.sharder import contiguous_local_indices

    assert sharder.local_token_global_indices is contiguous_local_indices
    fn = sharder.shard_batch
    assert fn.func is cp_mod.shard_glm_dsa_packed_cp_batch
    assert fn.keywords["num_chunks"] == 3


def test_glm_dsa_prepare_model_inputs_for_cp_requires_packed_backend():
    config = _small_dsa_config()
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False)
    model = GlmMoeDsaForCausalLM(config, backend=backend)

    with pytest.raises(NotImplementedError, match="backend.attn in"):
        model.prepare_model_inputs_for_cp({"input_ids": torch.arange(8).view(1, 8)})


def test_mla_tilelang_sparse_attention_rejects_bshd_without_kernels():
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend, skip_topk=True)
    x = torch.randn(1, 4, config.hidden_size, dtype=torch.bfloat16)
    topk_indices = torch.zeros(1, 4, config.index_topk, dtype=torch.int32)

    with pytest.raises(ValueError, match="requires THD"):
        mla(
            x,
            _freqs(4, config.qk_rope_head_dim).unsqueeze(0),
            prev_topk_indices=topk_indices,
        )


def test_mla_tilelang_sparse_attention_rejects_failed_validation(monkeypatch):
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend, skip_topk=True)
    monkeypatch.setattr(layer_mod, "should_use_tilelang", lambda *args, **kwargs: False)

    x = torch.randn(4, config.hidden_size, dtype=torch.bfloat16)
    topk_indices = torch.zeros(4, 1, config.index_topk, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="did not pass validation"):
        mla(
            x,
            _freqs(4, config.qk_rope_head_dim),
            prev_topk_indices=topk_indices,
        )


def test_mla_tilelang_sparse_attention_dispatches_absorbed_thd(monkeypatch):
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend, skip_topk=True)
    mla.o_proj = torch.nn.Identity()
    monkeypatch.setattr(layer_mod, "should_use_tilelang", lambda *args, **kwargs: True)
    captured = {}

    def fake_tilelang_sparse_attention(q, kv_latent, topk_indices, w_vc, softmax_scale):
        captured.update(q=q, kv_latent=kv_latent, topk_indices=topk_indices, w_vc=w_vc, softmax_scale=softmax_scale)
        return torch.ones(q.shape[0], config.num_attention_heads, config.v_head_dim, dtype=torch.bfloat16)

    monkeypatch.setattr(layer_mod, "tilelang_sparse_attention", fake_tilelang_sparse_attention)
    x = torch.randn(4, config.hidden_size, dtype=torch.bfloat16)
    topk_indices = torch.zeros(4, 1, config.index_topk, dtype=torch.int32)

    out, returned_topk = mla(
        x,
        _freqs(4, config.qk_rope_head_dim),
        prev_topk_indices=topk_indices,
        return_topk_indices=True,
    )
    out_no_carry = mla(x, _freqs(4, config.qk_rope_head_dim), prev_topk_indices=topk_indices)

    assert returned_topk is topk_indices
    assert out.shape == (4, config.num_attention_heads * config.v_head_dim)
    assert out_no_carry.shape == out.shape
    assert captured["q"].shape == (4, config.num_attention_heads, config.kv_lora_rank + config.qk_rope_head_dim)
    assert captured["kv_latent"].shape == (4, 1, config.kv_lora_rank + config.qk_rope_head_dim)
    assert captured["topk_indices"] is topk_indices
    assert captured["w_vc"].shape == (config.num_attention_heads, config.v_head_dim, config.kv_lora_rank)
    assert captured["w_vc"].dtype == torch.bfloat16
    expected_w_vc = mla.kv_b_proj.weight.view(
        config.num_attention_heads,
        config.qk_nope_head_dim + config.v_head_dim,
        config.kv_lora_rank,
    )[:, config.qk_nope_head_dim :, :]
    torch.testing.assert_close(captured["w_vc"], expected_w_vc)
    assert captured["softmax_scale"] == mla.softmax_scale


def test_mla_tilelang_uses_lora_effective_kv_b_weight_and_gradients(monkeypatch):
    """Production PEFT injection must affect both absorbed K and V TileLang weights."""
    torch.manual_seed(0)
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend, skip_topk=True)
    mla.o_proj = torch.nn.Identity()
    matched = apply_lora_to_linear_modules(
        mla,
        PeftConfig(
            target_modules=["kv_b_proj"],
            dim=4,
            alpha=8,
            dropout=0.0,
            use_memory_efficient_lora=False,
        ),
    )
    assert matched == 1
    assert isinstance(mla.kv_b_proj, LinearLoRA)
    with torch.no_grad():
        mla.kv_b_proj.lora_A.weight.normal_(std=0.2)
        mla.kv_b_proj.lora_B.weight.normal_(std=0.2)

    monkeypatch.setattr(layer_mod, "should_use_tilelang", lambda *args, **kwargs: True)
    captured = {}

    def fake_tilelang_sparse_attention(q, kv_latent, topk_indices, w_vc, softmax_scale):
        """Apply a differentiable sparse-kernel stand-in.

        Args:
            q: Tensor of shape [tokens, heads, kv_lora_rank + rope_dim], with the
                absorbed query followed by rotary query channels.
            kv_latent: Tensor of shape [tokens, 1, kv_lora_rank + rope_dim].
            topk_indices: Tensor of shape [tokens, 1, topk].
            w_vc: Tensor of shape [heads, value_dim, kv_lora_rank].
            softmax_scale: Attention score scale, unused by this stand-in.

        Returns:
            Tensor of shape [tokens, heads, value_dim].
        """
        captured.update(q=q, kv_latent=kv_latent, topk_indices=topk_indices, w_vc=w_vc)
        return torch.einsum("thc,hvc->thv", q[..., : config.kv_lora_rank], w_vc)

    monkeypatch.setattr(layer_mod, "tilelang_sparse_attention", fake_tilelang_sparse_attention)
    x = torch.randn(4, config.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    topk_indices = torch.zeros(4, 1, config.index_topk, dtype=torch.int32)

    with torch.no_grad():
        q_resid = mla.q_a_layernorm(mla.q_a_proj(x))
        q = mla.q_b_proj(q_resid).view(4, config.num_attention_heads, config.qk_head_dim)
        q_nope = q[..., : config.qk_nope_head_dim]
        effective_weight = mla.kv_b_proj.materialize_effective_weight().view(
            config.num_attention_heads,
            config.qk_nope_head_dim + config.v_head_dim,
            config.kv_lora_rank,
        )
        expected_q_absorbed = torch.einsum(
            "thd,hdc->thc",
            q_nope,
            effective_weight[:, : config.qk_nope_head_dim, :].to(q_nope.dtype),
        )
        expected_w_vc = effective_weight[:, config.qk_nope_head_dim :, :].to(torch.bfloat16)

    out = mla(
        x,
        _freqs(4, config.qk_rope_head_dim),
        prev_topk_indices=topk_indices,
    )

    torch.testing.assert_close(captured["q"][..., : config.kv_lora_rank], expected_q_absorbed)
    torch.testing.assert_close(captured["w_vc"], expected_w_vc)
    out.float().square().mean().backward()
    for grad in (x.grad, mla.kv_b_proj.lora_A.weight.grad, mla.kv_b_proj.lora_B.weight.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0


def test_mla_tilelang_zero_initialized_lora_b_updates_then_enables_lora_a(monkeypatch):
    """A cold-start TileLang KV adapter must begin learning at the first optimizer step."""
    torch.manual_seed(0)
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend, skip_topk=True)
    mla.o_proj = torch.nn.Identity()
    matched = apply_lora_to_linear_modules(
        mla,
        PeftConfig(
            target_modules=["kv_b_proj"],
            dim=4,
            alpha=8,
            dropout=0.0,
            use_memory_efficient_lora=False,
        ),
    )
    assert matched == 1
    assert isinstance(mla.kv_b_proj, LinearLoRA)
    assert torch.count_nonzero(mla.kv_b_proj.lora_B.weight) == 0

    monkeypatch.setattr(layer_mod, "should_use_tilelang", lambda *args, **kwargs: True)

    def fake_tilelang_sparse_attention(q, kv_latent, topk_indices, w_vc, softmax_scale):
        del kv_latent, topk_indices, softmax_scale
        return torch.einsum("thc,hvc->thv", q[..., : config.kv_lora_rank], w_vc)

    monkeypatch.setattr(layer_mod, "tilelang_sparse_attention", fake_tilelang_sparse_attention)
    optimizer = torch.optim.AdamW(
        [mla.kv_b_proj.lora_A.weight, mla.kv_b_proj.lora_B.weight],
        lr=1.0e-4,
    )
    x = torch.randn(4, config.hidden_size, dtype=torch.bfloat16)
    topk_indices = torch.zeros(4, 1, config.index_topk, dtype=torch.int32)
    freqs = _freqs(4, config.qk_rope_head_dim)
    initial_a = mla.kv_b_proj.lora_A.weight.detach().clone()
    initial_b = mla.kv_b_proj.lora_B.weight.detach().clone()

    first_out = mla(x, freqs, prev_topk_indices=topk_indices)
    first_out.float().square().mean().backward()
    first_a_grad = mla.kv_b_proj.lora_A.weight.grad
    first_b_grad = mla.kv_b_proj.lora_B.weight.grad
    assert first_a_grad is not None and torch.count_nonzero(first_a_grad) == 0
    assert first_b_grad is not None and torch.isfinite(first_b_grad).all()
    assert torch.count_nonzero(first_b_grad) > 0
    optimizer.step()

    torch.testing.assert_close(mla.kv_b_proj.lora_A.weight, initial_a)
    assert not torch.equal(mla.kv_b_proj.lora_B.weight, initial_b)
    assert torch.count_nonzero(mla.kv_b_proj.lora_B.weight) > 0

    optimizer.zero_grad(set_to_none=True)
    second_out = mla(x, freqs, prev_topk_indices=topk_indices)
    second_out.float().square().mean().backward()
    second_a_grad = mla.kv_b_proj.lora_A.weight.grad
    second_b_grad = mla.kv_b_proj.lora_B.weight.grad
    assert second_a_grad is not None and torch.isfinite(second_a_grad).all()
    assert torch.count_nonzero(second_a_grad) > 0
    assert second_b_grad is not None and torch.isfinite(second_b_grad).all()
    assert torch.count_nonzero(second_b_grad) > 0


def test_mla_tilelang_cp_gathers_kv_before_sparse_attention(monkeypatch):
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend, skip_topk=True)
    mla.o_proj = torch.nn.Identity()
    monkeypatch.setattr(layer_mod, "should_use_tilelang", lambda *args, **kwargs: True)
    monkeypatch.setattr(layer_mod, "glm_dsa_cp_enabled", lambda group: group == "cp-group")
    gathered = []
    captured = {}

    def fake_cp_all_gather(tensor, *, dim, cp_group):
        assert cp_group == "cp-group"
        gathered.append(tuple(tensor.shape))
        return torch.cat([tensor, tensor], dim=dim)

    def fake_tilelang_sparse_attention(q, kv_latent, topk_indices, w_vc, softmax_scale):
        captured.update(q=q, kv_latent=kv_latent, topk_indices=topk_indices, w_vc=w_vc, softmax_scale=softmax_scale)
        return torch.ones(q.shape[0], config.num_attention_heads, config.v_head_dim, dtype=torch.bfloat16)

    monkeypatch.setattr(layer_mod, "glm_dsa_cp_all_gather", fake_cp_all_gather)
    monkeypatch.setattr(layer_mod, "tilelang_sparse_attention", fake_tilelang_sparse_attention)
    x = torch.randn(4, config.hidden_size, dtype=torch.bfloat16)
    topk_indices = torch.zeros(4, 1, config.index_topk, dtype=torch.int32)

    out = mla(
        x,
        _freqs(4, config.qk_rope_head_dim),
        prev_topk_indices=topk_indices,
        _glm_dsa_cp_group="cp-group",
    )

    assert out.shape == (4, config.num_attention_heads * config.v_head_dim)
    assert gathered == [(4, config.kv_lora_rank), (4, config.qk_rope_head_dim)]
    assert captured["q"].shape[0] == 4
    assert captured["kv_latent"].shape == (8, 1, config.kv_lora_rank + config.qk_rope_head_dim)
    assert captured["topk_indices"] is topk_indices


@requires_kernels
def test_mla_forward_backward_tilelang_thd():
    torch.manual_seed(0)
    dev = "cuda"
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend).to(dev).to(torch.bfloat16)
    assert mla.attn_func is None and mla.attn_module is None

    angles = torch.randn(T, config.qk_rope_head_dim // 2, device=dev, dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    x = torch.randn(T, config.hidden_size, device=dev, dtype=torch.bfloat16, requires_grad=True)
    cu_seqlens = torch.tensor([0, T], device=dev, dtype=torch.int32)

    out = mla(x, freqs_cis, attention_mask=None, cu_seqlens=cu_seqlens, qkv_format="thd")
    assert out.shape == (T, config.hidden_size)
    assert torch.isfinite(out).all()

    out.float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert mla.kv_b_proj.weight.grad is not None and torch.isfinite(mla.kv_b_proj.weight.grad).all()
    assert mla.q_b_proj.weight.grad is not None and torch.isfinite(mla.q_b_proj.weight.grad).all()


@requires_kernels
def test_mla_tilelang_rejects_bshd():
    config = _small_dsa_config()
    backend = BackendConfig(attn="tilelang", linear="torch", rms_norm="torch", rope_fusion=False)
    mla = GlmMoeDsaMLA(config, backend).to("cuda").to(torch.bfloat16)

    angles = torch.randn(2, T, config.qk_rope_head_dim // 2, device="cuda", dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    x = torch.randn(2, T, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="THD"):
        mla(x, freqs_cis, attention_mask=None, qkv_format="bshd")
