# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from types import SimpleNamespace
from typing import List, Tuple
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.distributed.parallelizer_utils import (
    _fully_shard,
    _get_module_from_path,
    _group_params_by_dtype,
    _make_compute_dtype_fn,
    _mp_policy_with_param_dtype,
    configure_fsdp_unused_param_reduction,
    fully_shard_by_dtype,
    get_internal_fsdp_mp_policy,
    iter_maximal_uniform_dtype_subtrees,
    reject_unsupported_mtp_cp,
    reject_unsupported_mtp_cp_pp,
)
from nemo_automodel.shared.torch_patches import (
    patch_fsdp_uniform_reduce_dtype,
    patch_fsdp_unused_param_reduction,
)


def test_reject_unsupported_mtp_cp_pp_allows_disabled_model():
    model = nn.Linear(2, 2)
    model.supports = SimpleNamespace(mtp_enabled=False, supports_mtp_cp_pp=False)
    reject_unsupported_mtp_cp_pp(model)


def test_reject_unsupported_mtp_cp_rejects_enabled_unsupported_model():
    model = nn.Module()
    model.mtp_config = SimpleNamespace(enabled=True)
    model.supports = SimpleNamespace(mtp_enabled=True, supports_mtp_cp=False)

    with pytest.raises(RuntimeError, match="does not support MTP with context parallelism"):
        reject_unsupported_mtp_cp(model)


def test_reject_unsupported_mtp_cp_allows_supported_or_disabled_model():
    model = nn.Module()
    model.mtp_config = SimpleNamespace(enabled=True)
    model.supports = SimpleNamespace(mtp_enabled=True, supports_mtp_cp=True)
    reject_unsupported_mtp_cp(model)

    model.mtp_config.enabled = False
    model.supports.mtp_enabled = False
    model.supports.supports_mtp_cp = False
    reject_unsupported_mtp_cp(model)


def test_configure_fsdp_unused_param_reduction_uses_public_fsdp_api(monkeypatch):
    from nemo_automodel.components.distributed import parallelizer_utils

    class FakeFSDPModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def set_reduce_scatter_unused_params(self, enabled, *, recurse):
            self.calls.append((enabled, recurse))

    install_fallback = Mock()
    monkeypatch.setattr(parallelizer_utils, "FSDPModule", FakeFSDPModule)
    monkeypatch.setattr(parallelizer_utils, "_patch_fsdp_unused_param_reduction", install_fallback)
    model = nn.Sequential(FakeFSDPModule(), nn.Sequential(FakeFSDPModule()))

    assert configure_fsdp_unused_param_reduction(model) == 2
    install_fallback.assert_not_called()
    assert model[0].calls == [(True, False)]
    assert model[1][0].calls == [(True, False)]


def test_configure_fsdp_unused_param_reduction_uses_legacy_fallback(monkeypatch):
    from nemo_automodel.components.distributed import parallelizer_utils

    class LegacyFSDPModule(nn.Module):
        pass

    install_fallback = Mock()
    monkeypatch.setattr(parallelizer_utils, "FSDPModule", LegacyFSDPModule)
    monkeypatch.setattr(parallelizer_utils, "_patch_fsdp_unused_param_reduction", install_fallback)
    model = nn.Sequential(LegacyFSDPModule(), nn.Sequential(LegacyFSDPModule()))

    assert configure_fsdp_unused_param_reduction(model) == 2
    install_fallback.assert_called_once_with()


def test_legacy_fsdp_unused_param_reduction_fills_missing_local_grad(monkeypatch):
    from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
    from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

    calls = []

    def original_post_backward(self, *args, **kwargs):
        calls.append((self, args, kwargs))
        return "post-backward-result"

    monkeypatch.setattr(FSDPParamGroup, "post_backward", original_post_backward)
    patch_fsdp_unused_param_reduction()
    patched_post_backward = FSDPParamGroup.post_backward

    param = torch.nn.Parameter(torch.ones(2))
    fsdp_param = SimpleNamespace(
        _unsharded_param=param,
        unsharded_accumulated_grad=None,
        unsharded_param=param,
    )
    param_group = SimpleNamespace(
        reduce_grads=True,
        _training_state=TrainingState.PRE_BACKWARD,
        fsdp_params=[fsdp_param, SimpleNamespace()],
    )

    result = patched_post_backward(param_group, "arg", flag=True)
    patch_fsdp_unused_param_reduction()

    assert result == "post-backward-result"
    assert torch.equal(param.grad, torch.zeros_like(param))
    assert calls == [(param_group, ("arg",), {"flag": True})]
    assert FSDPParamGroup.post_backward is patched_post_backward


def _install_uniform_reduce_dtype(monkeypatch, recorder):
    """Install the patch over a stub foreach_reduce that records what it receives."""
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as collectives
    import torch.distributed.fsdp._fully_shard._fsdp_param_group as param_group

    def stub(fsdp_params, unsharded_grads, *args, **kwargs):
        recorder.append([g.dtype for g in unsharded_grads])
        return "reduced"

    monkeypatch.setattr(collectives, "foreach_reduce", stub)
    monkeypatch.setattr(param_group, "foreach_reduce", stub)
    patch_fsdp_uniform_reduce_dtype()
    return collectives


def test_uniform_reduce_dtype_widens_mixed_group(monkeypatch):
    """A bf16 straggler is widened to match its fp32 peers before the reduce."""
    seen = []
    collectives = _install_uniform_reduce_dtype(monkeypatch, seen)

    grads = [torch.ones(2, dtype=torch.float32), torch.full((2,), 5.0, dtype=torch.bfloat16)]
    result = collectives.foreach_reduce(["p0", "p1"], grads)

    assert result == "reduced"
    assert seen == [[torch.float32, torch.float32]]
    # Mutated in place so foreach_reduce's list.clear() still frees the caller's refs.
    assert [g.dtype for g in grads] == [torch.float32, torch.float32]
    assert torch.equal(grads[1], torch.full((2,), 5.0))


def test_uniform_reduce_dtype_localizes_residual_dtensor(monkeypatch):
    """The old public unused-param zero is localized before ``chunk_cat``."""
    import torch.distributed.fsdp._fully_shard._fsdp_collectives as collectives
    import torch.distributed.fsdp._fully_shard._fsdp_param_group as param_group
    import torch.distributed.tensor as tensor_module

    class FakeDTensor(torch.Tensor):
        @staticmethod
        def __new__(cls, tensor):
            return torch.Tensor._make_subclass(cls, tensor, False)

        def to_local(self):
            # Model an EP-local tensor with half of the global expert storage.
            return self.as_subclass(torch.Tensor)[:2]

    seen = []

    def stub(fsdp_params, unsharded_grads, *args, **kwargs):
        seen.append([(type(grad), grad.numel()) for grad in unsharded_grads])
        return "reduced"

    monkeypatch.setattr(tensor_module, "DTensor", FakeDTensor)
    monkeypatch.setattr(collectives, "foreach_reduce", stub)
    monkeypatch.setattr(param_group, "foreach_reduce", stub)
    patch_fsdp_uniform_reduce_dtype()

    grads = [torch.ones(2), FakeDTensor(torch.ones(4))]
    result = collectives.foreach_reduce(["used", "unused"], grads)

    assert result == "reduced"
    assert seen == [[(torch.Tensor, 2), (torch.Tensor, 2)]]
    assert all(type(grad) is torch.Tensor for grad in grads)


def test_uniform_reduce_dtype_leaves_uniform_group_untouched(monkeypatch):
    """Uniform groups pass straight through, preserving upstream's own checks."""
    seen = []
    collectives = _install_uniform_reduce_dtype(monkeypatch, seen)

    grads = [torch.ones(2, dtype=torch.bfloat16), torch.ones(2, dtype=torch.bfloat16)]
    original = [g for g in grads]
    collectives.foreach_reduce(["p0", "p1"], grads)

    assert seen == [[torch.bfloat16, torch.bfloat16]]
    assert all(a is b for a, b in zip(grads, original))


def test_uniform_reduce_dtype_ignores_non_float_mixtures(monkeypatch):
    """Non-float gradients are left alone so the upstream assertion still fires."""
    seen = []
    collectives = _install_uniform_reduce_dtype(monkeypatch, seen)

    grads = [torch.ones(2, dtype=torch.float32), torch.ones(2, dtype=torch.int32)]
    collectives.foreach_reduce(["p0", "p1"], grads)

    assert seen == [[torch.float32, torch.int32]]


def test_uniform_reduce_dtype_patch_is_idempotent(monkeypatch):
    """Re-installing must not stack a second wrapper."""
    seen = []
    collectives = _install_uniform_reduce_dtype(monkeypatch, seen)
    wrapped = collectives.foreach_reduce

    patch_fsdp_uniform_reduce_dtype()

    assert collectives.foreach_reduce is wrapped


def test_configure_fsdp_unused_param_reduction_installs_dtype_alignment_first(monkeypatch):
    """The zero fill must wrap the alignment so filled zeros are aligned too."""
    from nemo_automodel.components.distributed import parallelizer_utils

    class LegacyFSDPModule(nn.Module):
        pass

    order = []
    monkeypatch.setattr(parallelizer_utils, "FSDPModule", LegacyFSDPModule)
    monkeypatch.setattr(parallelizer_utils, "_patch_fsdp_uniform_reduce_dtype", lambda: order.append("uniform_dtype"))
    monkeypatch.setattr(parallelizer_utils, "_patch_fsdp_unused_param_reduction", lambda: order.append("zero_fill"))

    assert configure_fsdp_unused_param_reduction(nn.Sequential(LegacyFSDPModule())) == 1
    assert order == ["uniform_dtype", "zero_fill"]


def _tag_hf_compute_dtype(model: nn.Module) -> None:
    """Simulate an HF checkpoint load by recording each float tensor's dtype.

    ``_restore_loaded_model_dtype`` does this in production; tagging here lets the
    compute-dtype grouping mirror storage dtype (as it would for a loaded model).
    """
    for tensor in (*model.parameters(), *model.buffers()):
        if tensor.dtype.is_floating_point:
            tensor._hf_compute_dtype = tensor.dtype


class Block(nn.Module):
    def __init__(
        self,
        dtype_l1: torch.dtype = torch.float16,
        dtype_l2: torch.dtype = torch.float16,
        add_misleading_buffer: bool = False,
        buffer_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.l1 = nn.Linear(4, 4, bias=False).to(dtype_l1)
        self.l2 = nn.Linear(4, 4, bias=False).to(dtype_l2)
        if add_misleading_buffer:
            # Add a floating-point buffer that can break subtree uniformity when included
            self.register_buffer("buf", torch.zeros(1, dtype=buffer_dtype))


class ToyModel(nn.Module):
    def __init__(
        self,
        a_dtype: torch.dtype = torch.float32,
        b_dtype_l1: torch.dtype = torch.float16,
        b_dtype_l2: torch.dtype = torch.float16,
        c_dtype: torch.dtype | None = None,
        block_has_misleading_buffer: bool = False,
        block_buffer_dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.a = nn.Linear(4, 4, bias=False).to(a_dtype)
        self.b = Block(
            dtype_l1=b_dtype_l1,
            dtype_l2=b_dtype_l2,
            add_misleading_buffer=block_has_misleading_buffer,
            buffer_dtype=block_buffer_dtype,
        )
        if c_dtype is not None:
            # Optional third distinct subtree for >2 dtype scenarios
            self.c = nn.Linear(4, 4, bias=False).to(c_dtype)


def _collect_return_paths_items(items: List[Tuple[str, nn.Module, torch.dtype]]) -> dict[str, torch.dtype]:
    return {path: dtype for path, _mod, dtype in items}


def _collect_return_modules_items(items: List[Tuple[nn.Module, torch.dtype]]) -> dict[int, torch.dtype]:
    return {id(mod): dtype for mod, dtype in items}


def _make_mp_policy() -> MixedPrecisionPolicy:
    return MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )


def test_iter_maximal_uniform_dtype_subtrees_basic_paths():
    model = ToyModel(
        a_dtype=torch.float32,
        b_dtype_l1=torch.float16,
        b_dtype_l2=torch.float16,
    )
    # return_paths=True
    items_with_paths = list(
        iter_maximal_uniform_dtype_subtrees(
            model, include_buffers=True, tensor_pred=torch.is_floating_point, return_paths=True
        )
    )
    paths_to_dtype = _collect_return_paths_items(items_with_paths)
    assert paths_to_dtype == {
        "a": torch.float32,
        "b": torch.float16,
    }

    # return_paths=False
    items_no_paths = list(
        iter_maximal_uniform_dtype_subtrees(
            model, include_buffers=True, tensor_pred=torch.is_floating_point, return_paths=False
        )
    )
    mods_to_dtype = _collect_return_modules_items(items_no_paths)
    expected = {id(model.a): torch.float32, id(model.b): torch.float16}
    assert mods_to_dtype == expected


def test_iter_maximal_uniform_dtype_subtrees_include_buffers_effect():
    # Block has a float32 buffer but float16 parameters; including buffers should break uniformity of 'b'
    model = ToyModel(
        a_dtype=torch.float32,
        b_dtype_l1=torch.float16,
        b_dtype_l2=torch.float16,
        block_has_misleading_buffer=True,
        block_buffer_dtype=torch.float32,
    )
    # include_buffers=True: expect 'a', 'b.l1', 'b.l2'
    items_with_buffers = list(
        iter_maximal_uniform_dtype_subtrees(
            model, include_buffers=True, tensor_pred=torch.is_floating_point, return_paths=True
        )
    )
    paths_to_dtype_with_buffers = _collect_return_paths_items(items_with_buffers)
    assert paths_to_dtype_with_buffers == {
        "a": torch.float32,
        "b.l1": torch.float16,
        "b.l2": torch.float16,
    }

    # include_buffers=False: buffer ignored, expect maximal 'b' again
    items_no_buffers = list(
        iter_maximal_uniform_dtype_subtrees(
            model, include_buffers=False, tensor_pred=torch.is_floating_point, return_paths=True
        )
    )
    paths_to_dtype_no_buffers = _collect_return_paths_items(items_no_buffers)
    assert paths_to_dtype_no_buffers == {
        "a": torch.float32,
        "b": torch.float16,
    }


def test_group_params_by_dtype_counts():
    model = ToyModel(
        a_dtype=torch.float32,
        b_dtype_l1=torch.float16,
        b_dtype_l2=torch.float16,
    )
    grouped = _group_params_by_dtype(model)
    # Expect 1 param tensor in float32 ('a.weight'), 2 param tensors in float16 ('b.l1.weight', 'b.l2.weight')
    assert set(grouped.keys()) == {torch.float32, torch.float16}
    assert len(grouped[torch.float32]) == 1
    assert len(grouped[torch.float16]) == 2


def test_get_module_from_path():
    model = ToyModel()
    mod = _get_module_from_path(model, "b.l1")
    assert mod is model.b.l1
    mod2 = _get_module_from_path(model, "b.l2")
    assert mod2 is model.b.l2


def test__fully_shard_calls_for_single_module(monkeypatch):
    calls: list[tuple[nn.Module, object, object, object]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        calls.append((mod, mesh, mp_policy, offload_policy))

    # Monkeypatch the symbol inside the utils module
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    mod = nn.Linear(2, 2, bias=False)
    mesh, mp_policy, offload_policy = object(), object(), object()
    _fully_shard(mod, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)

    assert len(calls) == 1
    called_mod, called_mesh, called_mp, called_offload = calls[0]
    assert called_mod is mod
    assert called_mesh is mesh and called_mp is mp_policy and called_offload is offload_policy


def test__fully_shard_calls_for_modulelist(monkeypatch):
    calls: list[nn.Module] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        calls.append(mod)

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )

    ml = nn.ModuleList([nn.Linear(2, 2, bias=False), nn.Linear(2, 2, bias=False)])
    mesh, mp_policy, offload_policy = object(), object(), object()
    _fully_shard(ml, mesh=mesh, mp_policy=mp_policy, offload_policy=offload_policy)

    # Should call for each child, not the ModuleList itself
    assert len(calls) == 2
    assert calls[0] is ml[0]
    assert calls[1] is ml[1]


def test_mp_policy_with_param_dtype_copies_policy():
    mp_policy = _make_mp_policy()

    copied_policy = _mp_policy_with_param_dtype(mp_policy, torch.float32)

    assert copied_policy is not mp_policy
    assert copied_policy.param_dtype == torch.float32
    assert copied_policy.reduce_dtype == torch.float32
    assert copied_policy.output_dtype == torch.float32
    assert copied_policy.cast_forward_inputs is False
    assert mp_policy.param_dtype == torch.bfloat16


def test_mp_policy_with_bf16_param_dtype_preserves_policy():
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )

    copied_policy = _mp_policy_with_param_dtype(mp_policy, torch.bfloat16)

    assert copied_policy is not mp_policy
    assert copied_policy.param_dtype == torch.bfloat16
    assert copied_policy.reduce_dtype == torch.float32
    assert copied_policy.output_dtype == torch.bfloat16
    assert copied_policy.cast_forward_inputs is True


def test_internal_fsdp_mp_policy_drops_only_output_dtype():
    mp_policy = _make_mp_policy()

    internal_policy = get_internal_fsdp_mp_policy(mp_policy)

    assert internal_policy is not mp_policy
    assert internal_policy.param_dtype == mp_policy.param_dtype
    assert internal_policy.reduce_dtype == mp_policy.reduce_dtype
    assert internal_policy.output_dtype is None
    assert internal_policy.cast_forward_inputs == mp_policy.cast_forward_inputs
    assert mp_policy.output_dtype == torch.float32
    assert get_internal_fsdp_mp_policy(None) is None


def test_fully_shard_by_dtype_no_params(monkeypatch):
    fully_calls: list[nn.Module] = []
    sub_calls: list[nn.Module] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append(mod)

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        sub_calls.append(mod)

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    model = nn.Identity()
    fully_shard_by_dtype(model, mesh=object(), mp_policy=object(), offload_policy=object())
    assert fully_calls == []
    assert sub_calls == []


def test_fully_shard_by_dtype_single_dtype(monkeypatch):
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy, bool | None]] = []
    sub_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy, reshard_after_forward))

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        sub_calls.append((mod, mp_policy))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    # All parameters are float32 storage, but the policy requests bf16 compute
    # (fp32 master weights). Compute dtype is decoupled from storage: the bulk
    # computes in mp_policy.param_dtype (bf16), NOT the fp32 storage dtype.
    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.float32, b_dtype_l2=torch.float32)
    mp_policy = _make_mp_policy()
    fully_shard_by_dtype(
        model,
        mesh=object(),
        mp_policy=mp_policy,
        offload_policy=object(),
        reshard_after_forward=False,
    )

    assert [mod for mod, _policy, _reshard in fully_calls] == [model]  # whole module sharded once
    assert fully_calls[0][1] is not mp_policy
    assert fully_calls[0][1].param_dtype == torch.bfloat16
    assert fully_calls[0][1].reduce_dtype == mp_policy.reduce_dtype
    assert fully_calls[0][1].output_dtype == mp_policy.output_dtype
    assert fully_calls[0][2] is False
    assert mp_policy.param_dtype == torch.bfloat16
    assert sub_calls == []  # no fp32-compute carve-outs declared


def test_fully_shard_by_dtype_omits_none_reshard_kwarg(monkeypatch):
    calls: list[tuple[nn.Module, dict]] = []

    def fake_fully_shard(mod, **kwargs):
        calls.append((mod, kwargs))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard",
        fake_fully_shard,
        raising=True,
    )

    # Single-dtype branch.
    uniform = ToyModel(
        a_dtype=torch.bfloat16,
        b_dtype_l1=torch.bfloat16,
        b_dtype_l2=torch.bfloat16,
    )
    fully_shard_by_dtype(
        uniform,
        mesh=object(),
        mp_policy=_make_mp_policy(),
        offload_policy=object(),
        reshard_after_forward=None,
    )
    assert len(calls) == 1
    assert calls[0][0] is uniform
    assert "reshard_after_forward" not in calls[0][1]

    # Two-dtype branch: both the minority subtree and parent call must omit
    # the kwarg. Older supported PyTorch releases and some vendor builds
    # reject an explicit ``None``.
    calls.clear()
    mixed = ToyModel(
        a_dtype=torch.float32,
        b_dtype_l1=torch.float16,
        b_dtype_l2=torch.float16,
    )
    _tag_hf_compute_dtype(mixed)
    fully_shard_by_dtype(
        mixed,
        mesh=object(),
        mp_policy=_make_mp_policy(),
        offload_policy=object(),
        reshard_after_forward=None,
    )
    assert {mod for mod, _ in calls} == {mixed.a, mixed}
    assert all("reshard_after_forward" not in kwargs for _, kwargs in calls)


def test_fully_shard_by_dtype_storage_equals_compute_keeps_storage_dtype(monkeypatch):
    """Uniform storage that matches the requested compute dtype shards as before."""
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )

    # bf16 storage and bf16 compute -> param_dtype stays bf16 (no decoupling needed).
    model = ToyModel(a_dtype=torch.bfloat16, b_dtype_l1=torch.bfloat16, b_dtype_l2=torch.bfloat16)
    mp_policy = _make_mp_policy()
    fully_shard_by_dtype(model, mesh=object(), mp_policy=mp_policy, offload_policy=object())

    assert [mod for mod, _ in fully_calls] == [model]
    assert fully_calls[0][1].param_dtype == torch.bfloat16


def test_fully_shard_by_dtype_genuine_fp32_compute_unchanged(monkeypatch):
    """Uniform fp32 storage with an fp32-compute policy keeps fp32 compute."""
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )

    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.float32, b_dtype_l2=torch.float32)
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, output_dtype=torch.float32)
    fully_shard_by_dtype(model, mesh=object(), mp_policy=mp_policy, offload_policy=object())

    assert [mod for mod, _ in fully_calls] == [model]
    assert fully_calls[0][1].param_dtype == torch.float32


def test_make_compute_dtype_fn_precedence():
    """Resolver precedence: pinned fp32 > HF-recorded > mp_policy.param_dtype."""

    class Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    class Mixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj = nn.Linear(4, 4, bias=False).to(torch.float32)  # master-weight fp32 storage
            self.recorded_fp32 = nn.Linear(4, 4, bias=False).to(torch.float32)
            self.recorded_bf16 = nn.Linear(4, 4, bias=False).to(torch.float32)
            self._fp32_params = Holder()

    mixer = Mixer()
    # Simulate HF load: in_proj had bf16 in the checkpoint, the others fp32/bf16.
    mixer.in_proj.weight._hf_compute_dtype = torch.bfloat16
    mixer.recorded_fp32.weight._hf_compute_dtype = torch.float32
    mixer.recorded_bf16.weight._hf_compute_dtype = torch.bfloat16

    fn = _make_compute_dtype_fn(mixer, _make_mp_policy(), ("_fp32_params",))

    # Pinned wins even though storage is fp32 and nothing was recorded.
    assert fn(mixer._fp32_params.weight) == torch.float32
    # HF-recorded bf16 beats the fp32 master storage.
    assert fn(mixer.in_proj.weight) == torch.bfloat16
    assert fn(mixer.recorded_bf16.weight) == torch.bfloat16
    # HF-recorded fp32 is honored.
    assert fn(mixer.recorded_fp32.weight) == torch.float32


def test_make_compute_dtype_fn_fallback_to_policy_then_storage():
    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.float32, b_dtype_l2=torch.float32)

    # No record, no pin, bf16 policy -> fall back to policy (bf16) despite fp32 storage.
    fn = _make_compute_dtype_fn(model, _make_mp_policy(), ())
    assert fn(model.a.weight) == torch.bfloat16

    # No policy -> fall back to storage dtype.
    fn_no_policy = _make_compute_dtype_fn(model, None, ())
    assert fn_no_policy(model.a.weight) == torch.float32


def test_fully_shard_by_dtype_fp32_master_pins_compute(monkeypatch):
    """fp32 master weights (uniform fp32 storage): pinned param keeps fp32, bulk gets bf16."""
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy, bool | None]] = []
    sub_calls: list[tuple[nn.Module, MixedPrecisionPolicy, bool | None]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy, reshard_after_forward))

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        sub_calls.append((mod, mp_policy, reshard_after_forward))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    class Fp32Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    class Mixer(nn.Module):
        def __init__(self):
            super().__init__()
            # Two bulk linears so the bf16-compute group is the strict majority.
            self.in_proj = nn.Linear(4, 4, bias=False).to(torch.float32)
            self.out_proj = nn.Linear(4, 4, bias=False).to(torch.float32)
            self._fp32_params = Fp32Holder()

    mixer = Mixer()
    fully_shard_by_dtype(
        mixer,
        mesh=object(),
        mp_policy=_make_mp_policy(),
        offload_policy=object(),
        fp32_compute_module_names=("_fp32_params",),
        reshard_after_forward=True,
    )

    # Minority fp32 holder sharded on its own; the bf16 bulk is the parent unit.
    assert [mod for mod, _policy, _reshard in sub_calls] == [mixer._fp32_params]
    assert sub_calls[0][1].param_dtype == torch.float32
    assert sub_calls[0][2] is True
    assert [mod for mod, _policy, _reshard in fully_calls] == [mixer]
    assert fully_calls[0][1].param_dtype == torch.bfloat16
    assert fully_calls[0][2] is True


def test_fully_shard_by_dtype_fp32_master_hf_recorded_compute(monkeypatch):
    """fp32 master weights with HF-recorded dtypes: recorded-fp32 param stays fp32, no pin needed."""
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy, bool | None]] = []
    sub_calls: list[tuple[nn.Module, MixedPrecisionPolicy, bool | None]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy, reshard_after_forward))

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        sub_calls.append((mod, mp_policy, reshard_after_forward))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    # Uniform fp32 storage (master weights), but the checkpoint recorded 'a' as fp32
    # and the rest as bf16 -> 'a' computes fp32 automatically, bulk computes bf16.
    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.float32, b_dtype_l2=torch.float32)
    model.a.weight._hf_compute_dtype = torch.float32
    model.b.l1.weight._hf_compute_dtype = torch.bfloat16
    model.b.l2.weight._hf_compute_dtype = torch.bfloat16

    fully_shard_by_dtype(
        model,
        mesh=object(),
        mp_policy=_make_mp_policy(),
        offload_policy=object(),
        reshard_after_forward=True,
    )

    assert [mod for mod, _policy, _reshard in sub_calls] == [model.a]
    assert sub_calls[0][1].param_dtype == torch.float32
    assert [mod for mod, _policy, _reshard in fully_calls] == [model]
    assert fully_calls[0][1].param_dtype == torch.bfloat16
    assert sub_calls[0][2] is True
    assert fully_calls[0][2] is True


def test_fully_shard_by_dtype_two_dtypes(monkeypatch):
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []
    sub_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy))

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        sub_calls.append((mod, mp_policy))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    # Make float32 the least common (1 param) vs float16 (2 params)
    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.float16, b_dtype_l2=torch.float16)
    _tag_hf_compute_dtype(model)  # compute dtype mirrors storage (as for a loaded checkpoint)
    fully_shard_by_dtype(model, mesh=object(), mp_policy=_make_mp_policy(), offload_policy=object())

    # Expect subtree sharding for the least common dtype subtree(s) and full sharding once
    assert [mod for mod, _ in fully_calls] == [model]
    assert fully_calls[0][1].param_dtype == torch.float16
    # The least common dtype is float32 ('a'), so only 'a' subtree should be sharded individually
    assert [mod for mod, _ in sub_calls] == [model.a]
    assert sub_calls[0][1].param_dtype == torch.float32


def test_fully_shard_by_dtype_internal_child_preserves_natural_output_dtype(monkeypatch):
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []
    sub_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy))

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        sub_calls.append((mod, mp_policy))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    # The minority FP32-compute module becomes an internal child unit while the
    # BF16 majority remains owned by the enclosing FSDP boundary.
    model = ToyModel(a_dtype=torch.float32, b_dtype_l1=torch.bfloat16, b_dtype_l2=torch.bfloat16)
    _tag_hf_compute_dtype(model)
    mp_policy = _make_mp_policy()
    fully_shard_by_dtype(model, mesh=object(), mp_policy=mp_policy, offload_policy=object())

    assert [mod for mod, _ in sub_calls] == [model.a]
    assert sub_calls[0][1].param_dtype == torch.float32
    assert sub_calls[0][1].reduce_dtype == torch.float32
    assert sub_calls[0][1].output_dtype is None
    assert sub_calls[0][1].cast_forward_inputs is False
    assert [mod for mod, _ in fully_calls] == [model]
    assert fully_calls[0][1].param_dtype == torch.bfloat16
    assert fully_calls[0][1].output_dtype == torch.float32
    assert mp_policy.output_dtype == torch.float32


def test_fully_shard_by_dtype_excludes_ep_params_and_uses_custom_sharder():
    """Ignored EP experts do not affect grouping and remain excluded from the block unit."""

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, dtype=torch.float32))
            self.register_buffer("e_score_correction_bias", torch.zeros(4, dtype=torch.float32))

    class MoEBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.bulk_1 = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
            self.bulk_2 = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
            self.gate = Router()
            self.experts = nn.Linear(4, 4, bias=False).to(torch.bfloat16)

    block = MoEBlock()
    expert_params = set(block.experts.parameters())
    calls: list[tuple[nn.Module, dict]] = []

    def custom_fully_shard(module, **kwargs):
        calls.append((module, kwargs))

    fully_shard_by_dtype(
        block,
        mesh=object(),
        mp_policy=_make_mp_policy(),
        offload_policy=object(),
        fp32_compute_module_names=("gate.weight", "gate.e_score_correction_bias"),
        reshard_after_forward=False,
        ignored_params=expert_params,
        fully_shard_fn=custom_fully_shard,
    )

    assert [module for module, _ in calls] == [block.gate, block]
    assert calls[0][1]["mp_policy"].param_dtype == torch.float32
    assert "ignored_params" not in calls[0][1]
    assert calls[1][1]["mp_policy"].param_dtype == torch.bfloat16
    assert calls[1][1]["ignored_params"] == expert_params
    assert all(module is not block.experts for module, _ in calls)


def test_fully_shard_by_dtype_fp32_holder_preserves_natural_output_dtype():
    """Internal fp32 holders keep fp32 compute without forcing their output dtype."""

    class Fp32Holder(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, dtype=torch.float32))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.float() * self.weight

    class HybridBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_projection = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
            self.output_projection = nn.Linear(4, 4, bias=False).to(torch.bfloat16)
            self._fp32_params = Fp32Holder()

    block = HybridBlock()
    calls: list[tuple[nn.Module, dict]] = []

    fully_shard_by_dtype(
        block,
        mesh=object(),
        mp_policy=MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
        offload_policy=object(),
        fp32_compute_module_names=("_fp32_params",),
        fully_shard_fn=lambda module, **kwargs: calls.append((module, kwargs)),
    )

    holder_policy = next(kwargs["mp_policy"] for module, kwargs in calls if module is block._fp32_params)
    assert holder_policy.param_dtype == torch.float32
    assert holder_policy.reduce_dtype == torch.float32
    assert holder_policy.output_dtype is None
    assert holder_policy.cast_forward_inputs is False


def test_fully_shard_by_dtype_ignored_params_do_not_change_uniform_storage_fallback():
    """An ignored expert dtype must not make uniform managed master weights look mixed."""

    class BlockWithIgnoredExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.managed = nn.Linear(4, 4, bias=False).to(torch.float32)
            self.experts = nn.Linear(4, 4, bias=False).to(torch.bfloat16)

    block = BlockWithIgnoredExperts()
    expert_params = set(block.experts.parameters())
    calls: list[tuple[nn.Module, dict]] = []

    fully_shard_by_dtype(
        block,
        mesh=object(),
        mp_policy=_make_mp_policy(),
        offload_policy=object(),
        ignored_params=expert_params,
        fully_shard_fn=lambda module, **kwargs: calls.append((module, kwargs)),
    )

    assert [module for module, _ in calls] == [block]
    assert calls[0][1]["mp_policy"].param_dtype == torch.bfloat16
    assert calls[0][1]["ignored_params"] == expert_params


def test_fully_shard_by_dtype_rejects_unisolatable_mixed_parameter_owner():
    """A direct fp32 parameter sharing an owner with bf16 siblings fails early."""

    class MixedRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(4, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))

    router = MixedRouter()

    with pytest.raises(ValueError, match="could not isolate parameters with a distinct dtype"):
        fully_shard_by_dtype(
            router,
            mesh=object(),
            mp_policy=_make_mp_policy(),
            offload_policy=object(),
            fp32_compute_module_names=("weight",),
            fully_shard_fn=lambda *args, **kwargs: None,
        )


def test_fully_shard_by_dtype_three_dtypes(monkeypatch):
    fully_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []
    sub_calls: list[tuple[nn.Module, MixedPrecisionPolicy]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        fully_calls.append((mod, mp_policy))

    def fake__fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        sub_calls.append((mod, mp_policy))

    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils.fully_shard", fake_fully_shard, raising=True
    )
    monkeypatch.setattr(
        "nemo_automodel.components.distributed.parallelizer_utils._fully_shard", fake__fully_shard, raising=True
    )

    # Distinct dtypes across three subtrees: a=float32, b=float16, c=bfloat16
    model = ToyModel(
        a_dtype=torch.float32,
        b_dtype_l1=torch.float16,
        b_dtype_l2=torch.float16,
        c_dtype=torch.bfloat16,
    )
    _tag_hf_compute_dtype(model)  # compute dtype mirrors storage (as for a loaded checkpoint)
    fully_shard_by_dtype(model, mesh=object(), mp_policy=_make_mp_policy(), offload_policy=object())

    # For >2 dtypes: only subtree sharding, no whole-module sharding
    assert fully_calls == []
    # Expect all three subtrees to be individually sharded
    # Note: the 'b' subtree should be sharded as a whole since it is uniform float16
    assert {mod for mod, _ in sub_calls} == {model.a, model.b, model.c}
    assert {mod: policy.param_dtype for mod, policy in sub_calls} == {
        model.a: torch.float32,
        model.b: torch.float16,
        model.c: torch.bfloat16,
    }


def test_compute_dtype_pins_logical_names_through_activation_checkpointing():
    """Checkpoint wrappers preserve strict FP32 parameter and buffer matching."""
    attention = nn.Module()
    attention.sinks_param = nn.Linear(4, 4, bias=False)
    attention.sinks_param.register_buffer("scale", torch.ones(4))
    attention.proj = nn.Linear(4, 4, bias=False)
    model = nn.Module()
    model.attn = checkpoint_wrapper(attention)

    compute_dtype_of = _make_compute_dtype_fn(
        model, _make_mp_policy(), ("attn.sinks_param",)
    )

    assert compute_dtype_of(attention.sinks_param.weight) == torch.float32
    assert compute_dtype_of(attention.sinks_param.scale) == torch.float32
    assert compute_dtype_of(attention.proj.weight) == torch.bfloat16
