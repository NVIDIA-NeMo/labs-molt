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

"""End-to-end contract: every trainable parameter ends up in an FSDP unit whose
*compute* dtype matches what precision correctness requires.

These are CPU unit tests (no real FSDP): we run the production
``fully_shard_by_dtype`` with ``fully_shard`` mocked to record each unit's module
and ``param_dtype``, then map every parameter to the compute dtype of the most
specific FSDP unit that owns it. This exercises the full resolution path
(pinned -> HF-recorded -> mp_policy.param_dtype) across the model archetypes that
actually use ``fully_shard_by_dtype`` (NemotronH dense layers, Qwen3.5 hybrid
mixer), under fp32 master weights and ordinary loads.
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy

import nemo_automodel.components.distributed.parallelizer_utils as parallelizer_utils
from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype


def _mp_policy(param_dtype=torch.bfloat16):
    return MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=torch.float32, output_dtype=torch.float32)


def _resident_compute_dtypes(model, mp_policy, fp32_compute_module_names, monkeypatch):
    """Shard ``model`` (mocked) and return {param_name: resident compute dtype}.

    The resident compute dtype of a parameter is the ``param_dtype`` of the most
    specific (deepest / fewest-params) FSDP unit that owns it -- which is the unit
    FSDP would actually use for that parameter.
    """
    calls: list[tuple[nn.Module, torch.dtype]] = []

    def fake_fully_shard(mod, *, mesh, mp_policy, offload_policy, reshard_after_forward=None):
        del reshard_after_forward
        calls.append((mod, mp_policy.param_dtype if mp_policy is not None else None))

    # Patch only fully_shard; the real _fully_shard (incl. ModuleList expansion)
    # still routes every leaf shard through the mock.
    monkeypatch.setattr(parallelizer_utils, "fully_shard", fake_fully_shard, raising=True)

    fully_shard_by_dtype(
        model,
        mesh=object(),
        mp_policy=mp_policy,
        offload_policy=object(),
        fp32_compute_module_names=fp32_compute_module_names,
    )

    owned = [({id(p) for p in mod.parameters()}, dtype) for mod, dtype in calls]
    resolved: dict[str, torch.dtype] = {}
    for name, param in model.named_parameters():
        candidates = [(len(ids), dtype) for ids, dtype in owned if id(param) in ids]
        assert candidates, f"parameter {name!r} was not covered by any FSDP unit"
        chosen = min(candidates, key=lambda c: c[0])[1]
        # No mixed-precision policy -> FSDP computes in the storage dtype.
        resolved[name] = chosen if chosen is not None else param.dtype
    return resolved


def _tag_hf(tensor, dtype):
    tensor._hf_compute_dtype = dtype


# --------------------------------------------------------------------------- #
# Model archetypes (tiny) that use fully_shard_by_dtype in production.
# --------------------------------------------------------------------------- #


class DenseLayer(nn.Module):
    """NemotronH-style dense layer: only ordinary projection weights."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.attn = nn.Linear(8, 8, bias=False).to(dtype)
        self.mlp = nn.Linear(8, 8, bias=False).to(dtype)


class Fp32Holder(nn.Module):
    """Qwen3.5-style holder isolating an intrinsically-fp32 param (e.g. A_log)."""

    def __init__(self, n=4):
        super().__init__()
        self.A_log = nn.Parameter(torch.zeros(n, dtype=torch.float32))


class HybridLayer(nn.Module):
    """Qwen3.5-style GatedDeltaNet layer: bulk projections + fp32 holder."""

    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.in_proj = nn.Linear(8, 8, bias=False).to(dtype)
        self.out_proj = nn.Linear(8, 8, bias=False).to(dtype)
        self._fp32_params = Fp32Holder()


class MoEMixedStorageLayer(nn.Module):
    """Qwen3.5-MoE layer reproducing #3327: fp32 master projections and the
    fp32 SSM-gating holder (A_log/dt_bias) coexisting with a genuinely-bf16
    ``shared_expert_gate`` sibling, and no HF dtype records (custom-model path)."""

    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(8, 8, bias=False).to(torch.float32)
        self.out_proj = nn.Linear(8, 8, bias=False).to(torch.float32)
        self._fp32_params = nn.Module()
        self._fp32_params.A_log = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self._fp32_params.dt_bias = nn.Parameter(torch.zeros(4, dtype=torch.float32))
        self.shared_expert_gate = nn.Linear(8, 1, bias=False).to(torch.bfloat16)


# --------------------------------------------------------------------------- #
# Dense archetype
# --------------------------------------------------------------------------- #


def test_dense_master_weights_compute_bf16(monkeypatch):
    """fp32 master weights, no fp32 params -> the whole dense layer computes bf16."""
    layer = DenseLayer(dtype=torch.float32)  # uniform fp32 storage (master weights)
    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), (), monkeypatch)

    assert resolved == {
        "attn.weight": torch.bfloat16,
        "mlp.weight": torch.bfloat16,
    }


def test_dense_genuine_fp32_policy_computes_fp32(monkeypatch):
    """An explicit fp32 compute policy keeps the dense layer in fp32."""
    layer = DenseLayer(dtype=torch.float32)
    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.float32), (), monkeypatch)

    assert resolved == {
        "attn.weight": torch.float32,
        "mlp.weight": torch.float32,
    }


# --------------------------------------------------------------------------- #
# Hybrid archetype -- the three dtype-source scenarios
# --------------------------------------------------------------------------- #


def test_hybrid_master_weights_pinned_keeps_holder_fp32(monkeypatch):
    """From-scratch master weights: the pin keeps A_log fp32, bulk computes bf16."""
    layer = HybridLayer(dtype=torch.float32)  # uniform fp32 storage, no HF records
    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), ("_fp32_params",), monkeypatch)

    assert resolved == {
        "in_proj.weight": torch.bfloat16,
        "out_proj.weight": torch.bfloat16,
        "_fp32_params.A_log": torch.float32,
    }


def test_hybrid_master_weights_hf_recorded_keeps_holder_fp32(monkeypatch):
    """Loaded-from-checkpoint master weights: HF records keep A_log fp32 with no pin."""
    layer = HybridLayer(dtype=torch.float32)  # storage upcast to fp32 (master weights)
    # Simulate _restore_loaded_model_dtype recording the checkpoint's original dtypes.
    _tag_hf(layer.in_proj.weight, torch.bfloat16)
    _tag_hf(layer.out_proj.weight, torch.bfloat16)
    _tag_hf(layer._fp32_params.A_log, torch.float32)

    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), (), monkeypatch)

    assert resolved == {
        "in_proj.weight": torch.bfloat16,
        "out_proj.weight": torch.bfloat16,
        "_fp32_params.A_log": torch.float32,
    }


def test_hybrid_non_master_load_mirrors_storage(monkeypatch):
    """Ordinary bf16 load: bulk stored bf16, holder fp32; records mirror storage."""
    layer = HybridLayer(dtype=torch.bfloat16)  # bulk bf16 storage, holder fp32 storage
    _tag_hf(layer.in_proj.weight, torch.bfloat16)
    _tag_hf(layer.out_proj.weight, torch.bfloat16)
    _tag_hf(layer._fp32_params.A_log, torch.float32)

    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), (), monkeypatch)

    assert resolved == {
        "in_proj.weight": torch.bfloat16,
        "out_proj.weight": torch.bfloat16,
        "_fp32_params.A_log": torch.float32,
    }


def test_hybrid_pin_overrides_hf_recorded_dtype(monkeypatch):
    """The pin wins even when an HF record would say bf16."""
    layer = HybridLayer(dtype=torch.float32)
    # Record A_log as bf16 (wrong on purpose); the pin must still force fp32.
    _tag_hf(layer.in_proj.weight, torch.bfloat16)
    _tag_hf(layer.out_proj.weight, torch.bfloat16)
    _tag_hf(layer._fp32_params.A_log, torch.bfloat16)

    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), ("_fp32_params",), monkeypatch)

    assert resolved["_fp32_params.A_log"] == torch.float32
    assert resolved["in_proj.weight"] == torch.bfloat16


def test_moe_mixed_storage_no_hf_hint_bulk_computes_bf16(monkeypatch):
    """#3327: a lone genuinely-bf16 ``shared_expert_gate`` sibling must not drag
    the fp32 master weights into fp32 compute. With no HF records, ordinary
    weights and the gate compute bf16; only the pinned holder stays fp32."""
    layer = MoEMixedStorageLayer()  # mixed storage {fp32, bf16}, no HF records
    resolved = _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), ("_fp32_params",), monkeypatch)

    assert resolved == {
        "in_proj.weight": torch.bfloat16,
        "out_proj.weight": torch.bfloat16,
        "shared_expert_gate.weight": torch.bfloat16,
        "_fp32_params.A_log": torch.float32,
        "_fp32_params.dt_bias": torch.float32,
    }


def test_hybrid_stack_of_layers_master_weights(monkeypatch):
    """A ModuleList of hybrid layers: every A_log fp32, every projection bf16."""
    stack = nn.ModuleList([HybridLayer(dtype=torch.float32) for _ in range(3)])
    # fully_shard_by_dtype is called per-layer by the strategies; mirror that.
    resolved: dict[str, torch.dtype] = {}
    for i, layer in enumerate(stack):
        layer_resolved = _resident_compute_dtypes(layer, _mp_policy(torch.bfloat16), ("_fp32_params",), monkeypatch)
        for name, dtype in layer_resolved.items():
            resolved[f"{i}.{name}"] = dtype

    for i in range(3):
        assert resolved[f"{i}._fp32_params.A_log"] == torch.float32
        assert resolved[f"{i}.in_proj.weight"] == torch.bfloat16
        assert resolved[f"{i}.out_proj.weight"] == torch.bfloat16


def test_no_mixed_precision_policy_falls_back_to_storage(monkeypatch):
    """Without a policy, compute dtype falls back to storage dtype (no downcast)."""
    layer = HybridLayer(dtype=torch.float32)
    resolved = _resident_compute_dtypes(layer, None, ("_fp32_params",), monkeypatch)

    # No policy -> non-pinned params keep their storage dtype (fp32 here).
    assert resolved["in_proj.weight"] == torch.float32
    assert resolved["out_proj.weight"] == torch.float32
    assert resolved["_fp32_params.A_log"] == torch.float32


def test_intrinsic_fp32_holder_models_declare_strict_pin():
    """Models with the _fp32_params holder must list it in
    _keep_in_fp32_modules_strict; the MoE parallelizer keys the fp32 compute
    pin on that list, so a missing entry silently drops A_log/dt_bias to bf16."""
    from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration
    from nemo_automodel.components.models.qwen3_8_flash_next.model import Qwen3_8_FlashNextForConditionalGeneration
    from nemo_automodel.components.models.qwen3_next.model import Qwen3NextForCausalLM

    for cls in (Qwen3_5MoeForConditionalGeneration, Qwen3NextForCausalLM, Qwen3_8_FlashNextForConditionalGeneration):
        assert "_fp32_params" in getattr(cls, "_keep_in_fp32_modules_strict", [])
