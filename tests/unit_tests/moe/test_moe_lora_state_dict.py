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

"""Tests for MoE expert LoRA state dict conversion (to_hf / from_hf round-trip).

Verifies that GroupedExpertsLoRA adapter weights are correctly converted to
per-expert HF PEFT format and back, enabling merge via AutoPeftModelForCausalLM.
"""

import re

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.addons import (
    _extract_target_modules,
    _extract_target_parameters,
)
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import GroupedExperts
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

skip_if_no_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

N_EXPERTS = 4
DIM = 64
MOE_INTER_DIM = 128
LORA_DIM = 8
LORA_ALPHA = 32
N_LAYERS = 2


class _MockMoEConfig:
    def __init__(self):
        self.n_routed_experts = N_EXPERTS
        self.moe_inter_dim = MOE_INTER_DIM
        self.expert_activation = "swiglu"


class _Adapter(MoESplitExpertsStateDictMixin):
    """Minimal adapter to test mixin methods."""

    def __init__(self, uses_model_prefix=True):
        self.moe_config = _MockMoEConfig()
        self.dtype = torch.float32
        self._uses_model_prefix = uses_model_prefix
        self._last_expert_ids = []

    @property
    def _v5_peft_target_parameters(self):
        return ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")


class _LegacyAdapter(_Adapter):
    """Representative unvalidated adapter that must retain v4-style export."""

    @property
    def _v5_peft_target_parameters(self):
        return ()


def _make_moe_config():
    return MoEConfig(
        dim=DIM,
        inter_dim=MOE_INTER_DIM * 2,
        moe_inter_dim=MOE_INTER_DIM,
        n_routed_experts=N_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=False,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
        activation_alpha=1.702,
        activation_limit=7.0,
        softmax_before_topk=True,
    )


def _make_tiny_moe_model(device="cpu", target_modules=None):
    """Build a 2-layer toy model with GroupedExperts + LoRA on experts."""
    moe_cfg = _make_moe_config()

    class TinyMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(DIM, DIM, bias=False)
            self.layers = nn.ModuleList()
            for _ in range(N_LAYERS):
                layer = nn.Module()
                mlp = nn.Module()
                mlp.experts = GroupedExperts(moe_cfg)
                layer.mlp = mlp
                self.layers.append(layer)

        @property
        def config(self):
            return None

    model = TinyMoE().to(device)
    model.state_dict_adapter = _Adapter()
    peft_config = PeftConfig(
        target_modules=["*experts*"] if target_modules is None else target_modules,
        dim=LORA_DIM,
        alpha=LORA_ALPHA,
    )
    apply_lora_to_linear_modules(model, peft_config)
    return model


def _make_grouped_lora_state_dict(prefix="base_model.model."):
    """Build a synthetic PEFT state dict with grouped MoE LoRA keys."""
    sd = {}
    for layer_idx in range(N_LAYERS):
        base = f"{prefix}layers.{layer_idx}.mlp.experts"
        sd[f"{base}.lora_gate_and_up_A"] = torch.randn(N_EXPERTS, DIM, LORA_DIM)
        sd[f"{base}.lora_gate_and_up_B"] = torch.randn(N_EXPERTS, LORA_DIM, 2 * MOE_INTER_DIM)
        sd[f"{base}.lora_down_A"] = torch.randn(N_EXPERTS, MOE_INTER_DIM, LORA_DIM)
        sd[f"{base}.lora_down_B"] = torch.randn(N_EXPERTS, LORA_DIM, DIM)
    return sd


# ---------------------------------------------------------------------------
# Tests: to_hf (grouped -> per-expert HF format)
# ---------------------------------------------------------------------------


class TestMoELoRAToHF:
    """Verify _convert_single_merged_expert_to_hf_split_experts for LoRA keys."""

    def test_lora_gate_and_up_A_splits_into_per_expert_keys(self):
        adapter = _Adapter()
        tensor = torch.randn(N_EXPERTS, DIM, LORA_DIM)
        fqn = "model.layers.0.mlp.experts.lora_gate_and_up_A"

        result = adapter._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, v4_compatible=True)

        assert result is not None
        keys = [k for k, _ in result]
        assert len(keys) == N_EXPERTS * 2  # gate_proj + up_proj per expert
        for eid in range(N_EXPERTS):
            assert f"model.layers.0.mlp.experts.{eid}.gate_proj.lora_A.weight" in keys
            assert f"model.layers.0.mlp.experts.{eid}.up_proj.lora_A.weight" in keys

    def test_lora_gate_and_up_A_shape_and_values(self):
        adapter = _Adapter()
        tensor = torch.randn(N_EXPERTS, DIM, LORA_DIM)
        fqn = "model.layers.0.mlp.experts.lora_gate_and_up_A"

        result = dict(adapter._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, v4_compatible=True))

        for eid in range(N_EXPERTS):
            gate_A = result[f"model.layers.0.mlp.experts.{eid}.gate_proj.lora_A.weight"]
            up_A = result[f"model.layers.0.mlp.experts.{eid}.up_proj.lora_A.weight"]
            # nn.Linear convention: [out_features, in_features] = [lora_dim, dim]
            assert gate_A.shape == (LORA_DIM, DIM)
            assert up_A.shape == (LORA_DIM, DIM)
            # gate and up A are identical (duplicated)
            torch.testing.assert_close(gate_A, up_A)
            # Value check: should be transpose of original
            torch.testing.assert_close(gate_A, tensor[eid].transpose(0, 1))

    def test_lora_gate_and_up_B_splits_gate_and_up(self):
        adapter = _Adapter()
        tensor = torch.randn(N_EXPERTS, LORA_DIM, 2 * MOE_INTER_DIM)
        fqn = "model.layers.0.mlp.experts.lora_gate_and_up_B"

        result = dict(adapter._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, v4_compatible=True))

        for eid in range(N_EXPERTS):
            gate_B = result[f"model.layers.0.mlp.experts.{eid}.gate_proj.lora_B.weight"]
            up_B = result[f"model.layers.0.mlp.experts.{eid}.up_proj.lora_B.weight"]
            assert gate_B.shape == (MOE_INTER_DIM, LORA_DIM)
            assert up_B.shape == (MOE_INTER_DIM, LORA_DIM)
            # gate = first half, up = second half (transposed)
            torch.testing.assert_close(gate_B, tensor[eid, :, :MOE_INTER_DIM].transpose(0, 1))
            torch.testing.assert_close(up_B, tensor[eid, :, MOE_INTER_DIM:].transpose(0, 1))

    def test_lora_down_A(self):
        adapter = _Adapter()
        tensor = torch.randn(N_EXPERTS, MOE_INTER_DIM, LORA_DIM)
        fqn = "model.layers.0.mlp.experts.lora_down_A"

        result = dict(adapter._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, v4_compatible=True))

        for eid in range(N_EXPERTS):
            down_A = result[f"model.layers.0.mlp.experts.{eid}.down_proj.lora_A.weight"]
            assert down_A.shape == (LORA_DIM, MOE_INTER_DIM)
            torch.testing.assert_close(down_A, tensor[eid].transpose(0, 1))

    def test_lora_down_B(self):
        adapter = _Adapter()
        tensor = torch.randn(N_EXPERTS, LORA_DIM, DIM)
        fqn = "model.layers.0.mlp.experts.lora_down_B"

        result = dict(adapter._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, v4_compatible=True))

        for eid in range(N_EXPERTS):
            down_B = result[f"model.layers.0.mlp.experts.{eid}.down_proj.lora_B.weight"]
            assert down_B.shape == (DIM, LORA_DIM)
            torch.testing.assert_close(down_B, tensor[eid].transpose(0, 1))

    def test_preserves_arbitrary_prefix(self):
        """Prefix like 'base_model.model.' from PEFT saving must be preserved."""
        adapter = _Adapter()
        tensor = torch.randn(N_EXPERTS, DIM, LORA_DIM)
        fqn = "base_model.model.model.layers.0.mlp.experts.lora_gate_and_up_A"

        result = dict(adapter._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, v4_compatible=True))

        assert "base_model.model.model.layers.0.mlp.experts.0.gate_proj.lora_A.weight" in result

    def test_non_lora_keys_return_none(self):
        adapter = _Adapter()
        tensor = torch.randn(N_EXPERTS, DIM, 2 * MOE_INTER_DIM)
        fqn = "model.layers.0.mlp.experts.gate_and_up_projs"

        result = adapter._convert_lora_expert_to_hf(fqn, tensor, N_EXPERTS, MOE_INTER_DIM, "mlp.experts")

        # This method is only called for LoRA keys; for base keys
        # the existing code path handles conversion.
        assert result is not None  # _convert_lora_expert_to_hf always returns list


# ---------------------------------------------------------------------------
# Tests: from_hf (per-expert HF format -> grouped)
# ---------------------------------------------------------------------------


class TestMoELoRAFromHF:
    """Verify _recombine_lora_expert_keys correctly recombines per-expert LoRA keys."""

    def test_round_trip_lora_gate_and_up_A_v4(self):
        adapter = _Adapter()
        original = torch.randn(N_EXPERTS, DIM, LORA_DIM)

        # to_hf (v4 mode)
        hf = dict(
            adapter._convert_single_merged_expert_to_hf_split_experts(
                "model.layers.0.mlp.experts.lora_gate_and_up_A",
                original,
                v4_compatible=True,
            )
        )

        # from_hf
        result = adapter._recombine_lora_expert_keys(hf)

        key = "model.layers.0.mlp.experts.lora_gate_and_up_A"
        assert key in result
        torch.testing.assert_close(result[key], original)

    def test_round_trip_lora_gate_and_up_B_v4(self):
        adapter = _Adapter()
        original = torch.randn(N_EXPERTS, LORA_DIM, 2 * MOE_INTER_DIM)

        hf = dict(
            adapter._convert_single_merged_expert_to_hf_split_experts(
                "model.layers.0.mlp.experts.lora_gate_and_up_B",
                original,
                v4_compatible=True,
            )
        )

        result = adapter._recombine_lora_expert_keys(hf)

        key = "model.layers.0.mlp.experts.lora_gate_and_up_B"
        assert key in result
        torch.testing.assert_close(result[key], original)

    def test_round_trip_lora_down_A_v4(self):
        adapter = _Adapter()
        original = torch.randn(N_EXPERTS, MOE_INTER_DIM, LORA_DIM)

        hf = dict(
            adapter._convert_single_merged_expert_to_hf_split_experts(
                "model.layers.0.mlp.experts.lora_down_A",
                original,
                v4_compatible=True,
            )
        )

        result = adapter._recombine_lora_expert_keys(hf)

        key = "model.layers.0.mlp.experts.lora_down_A"
        assert key in result
        torch.testing.assert_close(result[key], original)

    def test_round_trip_lora_down_B_v4(self):
        adapter = _Adapter()
        original = torch.randn(N_EXPERTS, LORA_DIM, DIM)

        hf = dict(
            adapter._convert_single_merged_expert_to_hf_split_experts(
                "model.layers.0.mlp.experts.lora_down_B",
                original,
                v4_compatible=True,
            )
        )

        result = adapter._recombine_lora_expert_keys(hf)

        key = "model.layers.0.mlp.experts.lora_down_B"
        assert key in result
        torch.testing.assert_close(result[key], original)

    def test_round_trip_all_lora_keys_with_prefix_v4(self):
        """Full round-trip for all 4 LoRA parameter types, with PEFT prefix (v4 mode)."""
        adapter = _Adapter()
        original = _make_grouped_lora_state_dict(prefix="base_model.model.")

        # to_hf: convert all keys (v4 mode)
        hf_sd = {}
        for fqn, tensor in original.items():
            converted = adapter._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, v4_compatible=True)
            if converted:
                for k, v in converted:
                    hf_sd[k] = v
            else:
                hf_sd[fqn] = tensor

        # Verify HF format: no grouped LoRA keys remain
        for k in hf_sd:
            assert "lora_gate_and_up_A" not in k
            assert "lora_gate_and_up_B" not in k
            assert "lora_down_A" not in k
            assert "lora_down_B" not in k

        # from_hf: recombine
        result = adapter._recombine_lora_expert_keys(hf_sd)

        # Verify all original keys are restored
        for k, v in original.items():
            assert k in result, f"Missing key after round-trip: {k}"
            torch.testing.assert_close(result[k], v, msg=f"Value mismatch for {k}")

    def test_round_trip_v5_paramwrapper(self):
        """Test v5 ParamWrapper round-trip for all 4 LoRA types."""
        adapter = _Adapter()
        original = _make_grouped_lora_state_dict(prefix="base_model.model.")

        # to_hf: convert to ParamWrapper (v5) format
        pw_sd = {}
        for fqn, tensor in original.items():
            converted = adapter._convert_single_merged_expert_to_hf_split_experts(fqn, tensor, v4_compatible=False)
            if converted:
                for k, v in converted:
                    pw_sd[k] = v
            else:
                pw_sd[fqn] = tensor

        # Verify ParamWrapper format: keys use lora_[AB].weight pattern
        for fqn in pw_sd:
            assert "lora_A.weight" in fqn or "lora_B.weight" in fqn, f"Unexpected key: {fqn}"

        # from_hf: convert back (via _convert_paramwrapper_to_native)
        result = adapter._convert_paramwrapper_to_native(pw_sd)

        # Verify all original keys are restored
        for k, v in original.items():
            assert k in result, f"Missing key after v5 round-trip: {k}"
            torch.testing.assert_close(result[k], v, msg=f"Value mismatch for {k}")

    def test_passthrough_non_lora_keys(self):
        adapter = _Adapter()
        sd = {
            "model.layers.0.self_attn.q_proj.lora_A.weight": torch.randn(8, 64),
            "model.layers.0.mlp.experts.gate_and_up_projs": torch.randn(4, 64, 256),
        }

        result = adapter._recombine_lora_expert_keys(sd)

        assert set(result.keys()) == set(sd.keys())

    def test_incomplete_experts_passthrough(self):
        """If not all experts present, keys are passed through unchanged."""
        adapter = _Adapter()
        sd = {
            "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.randn(LORA_DIM, DIM),
            "model.layers.0.mlp.experts.1.gate_proj.lora_A.weight": torch.randn(LORA_DIM, DIM),
            # Missing experts 2 and 3
        }

        result = adapter._recombine_lora_expert_keys(sd)

        assert "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight" in result
        assert "model.layers.0.mlp.experts.1.gate_proj.lora_A.weight" in result


# ---------------------------------------------------------------------------
# Tests: _extract_target_modules for MoE expert LoRA
# ---------------------------------------------------------------------------


@skip_if_no_cuda
class TestExtractTargetModulesWithMoELoRA:
    """Verify that _extract_target_modules includes per-expert HF module names."""

    def test_includes_per_expert_projections_v4(self):
        model = _make_tiny_moe_model()

        target_modules = _extract_target_modules(model, v4_compatible=True)

        for layer_idx in range(N_LAYERS):
            for eid in range(N_EXPERTS):
                base = f"layers.{layer_idx}.mlp.experts.{eid}"
                assert f"{base}.gate_proj" in target_modules, f"Missing {base}.gate_proj"
                assert f"{base}.up_proj" in target_modules, f"Missing {base}.up_proj"
                assert f"{base}.down_proj" in target_modules, f"Missing {base}.down_proj"

    def test_no_grouped_lora_names_in_targets_v4(self):
        model = _make_tiny_moe_model()

        target_modules = _extract_target_modules(model, v4_compatible=True)

        for name in target_modules:
            assert "lora_gate_and_up" not in name
            assert "lora_down" not in name


class TestExtractTargetsV5Defaults:
    """v5 (``v4_compatible=False``) is the production default.

    The v4 tests above all pass ``v4_compatible=True``, so the defaults that
    actually ship were unasserted: ``target_parameters`` carries the fused
    ParamWrapper paths and ``target_modules`` skips the per-expert expansion.
    CPU-only — none of this touches a device.
    """

    def test_target_parameters_are_the_fused_expert_paths(self):
        model = _make_tiny_moe_model()

        assert _extract_target_parameters(model) == [
            "mlp.experts.gate_up_proj",
            "mlp.experts.down_proj",
        ]

    def test_target_parameters_empty_in_v4_mode(self):
        """v4 expresses experts through target_modules instead."""
        model = _make_tiny_moe_model()

        assert _extract_target_parameters(model, v4_compatible=True) == []

    def test_no_per_expert_expansion_in_v5(self):
        """The v4 path emits layers.N.mlp.experts.<eid>.*; v5 must not."""
        model = _make_tiny_moe_model()

        target_modules = _extract_target_modules(model)

        assert not [n for n in target_modules if re.search(r"experts\.\d+\.", n)]

    def test_v4_and_v5_are_mutually_exclusive(self):
        """Exactly one of the two mechanisms describes the experts."""
        model = _make_tiny_moe_model()

        v5_modules = _extract_target_modules(model)
        v5_params = _extract_target_parameters(model)
        v4_modules = _extract_target_modules(model, v4_compatible=True)
        v4_params = _extract_target_parameters(model, v4_compatible=True)

        assert v5_params and not v5_modules
        assert v4_modules and not v4_params

    def test_target_parameters_empty_without_a_moe_adapter(self):
        """A non-MoE model must not gain fused expert paths."""
        assert _extract_target_parameters(nn.Module()) == []

    def test_target_parameters_empty_when_only_dense_lora_is_injected(self):
        """A v5-capable model must not advertise expert paths for dense-only LoRA."""
        model = _make_tiny_moe_model(target_modules=["dense"])

        assert _extract_target_parameters(model) == []
        assert _extract_target_modules(model) == ["dense"]

    def test_accepts_a_list_of_pp_parts(self):
        """Under PP the caller passes this rank's stages; the first is representative."""
        model = _make_tiny_moe_model()

        assert _extract_target_parameters([model]) == _extract_target_parameters(model)

    def test_unvalidated_adapter_keeps_legacy_targets_by_default(self):
        model = _make_tiny_moe_model()
        model.state_dict_adapter = _LegacyAdapter()

        assert _extract_target_parameters(model) == []
        targets = _extract_target_modules(model)
        assert "layers.0.mlp.experts.0.gate_proj" in targets
        assert "layers.0.mlp.experts.0.up_proj" in targets
        assert "layers.0.mlp.experts.0.down_proj" in targets

    def test_unvalidated_adapter_keeps_legacy_state_dict_format_by_default(self):
        adapter = _LegacyAdapter()
        tensor = torch.randn(N_EXPERTS, DIM, LORA_DIM)

        converted = adapter._convert_single_merged_expert_to_hf_split_experts(
            "model.layers.0.mlp.experts.lora_gate_and_up_A", tensor
        )

        keys = {key for key, _ in converted}
        assert "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight" in keys
        assert not any("base_layer.lora_A.weight" in key for key in keys)


# ---------------------------------------------------------------------------
# Functional: tiny 2-layer MoE LoRA save -> HF load -> merge
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full HF PEFT round-trip: save adapter -> PeftModel.from_pretrained -> merge
# ---------------------------------------------------------------------------


def _convert_grouped_to_hf(grouped_sd, v4_compatible=True):
    """Helper: convert a grouped LoRA state dict to HF format.

    Args:
        grouped_sd: Grouped native LoRA state dict.
        v4_compatible: If True, produce per-expert split format (v4).
            If False, produce ParamWrapper fused format (v5).
    """
    adapter_obj = _Adapter()
    hf_sd = {}
    for fqn, tensor in grouped_sd.items():
        converted = adapter_obj._convert_single_merged_expert_to_hf_split_experts(
            fqn, tensor, v4_compatible=v4_compatible
        )
        if converted:
            for k, v in converted:
                hf_sd[k] = v
        else:
            hf_sd[fqn] = tensor
    return hf_sd


class _Expert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(DIM, MOE_INTER_DIM, bias=False)
        self.up_proj = nn.Linear(DIM, MOE_INTER_DIM, bias=False)
        self.down_proj = nn.Linear(MOE_INTER_DIM, DIM, bias=False)


class _HFStyleMoE(nn.Module):
    """Mimics an HF transformers MoE model with per-expert Linear layers."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(N_LAYERS):
            layer = nn.Module()
            mlp = nn.Module()
            mlp.experts = nn.ModuleList([_Expert() for _ in range(N_EXPERTS)])
            layer.mlp = mlp
            self.layers.append(layer)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        raise NotImplementedError("stub for HF PEFT compatibility")


def _make_hf_expert_model():
    return _HFStyleMoE()


def _write_adapter_dir(tmp_path, hf_lora_sd, lora_r, lora_alpha, target_modules):
    """Persist adapter_model.safetensors + adapter_config.json to *tmp_path*."""
    import json

    from safetensors.torch import save_file

    save_file(hf_lora_sd, str(tmp_path / "adapter_model.safetensors"))

    config = {
        "peft_type": "LORA",
        "auto_mapping": {"base_model_class": None, "parent_library": None},
        "base_model_name_or_path": "",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "modules_to_save": None,
        "r": lora_r,
        "rank_pattern": {},
        "revision": None,
        "target_modules": sorted(target_modules),
        "task_type": "CAUSAL_LM",
    }
    with open(str(tmp_path / "adapter_config.json"), "w") as f:
        json.dump(config, f)


class TestMoELoRASaveRestoreMergeHF:
    """Full pipeline: MoE + PEFT -> save adapter -> restore with HF PEFT -> merge.

    This class tests the *exact* workflow that was broken in issue #1226:
    1. Automodel trains MoE LoRA (simulated with synthetic grouped tensors).
    2. The grouped 3-D adapter weights are converted to per-expert HF keys.
    3. ``adapter_model.safetensors`` + ``adapter_config.json`` are written.
    4. ``peft.PeftModel.from_pretrained`` loads them onto an HF-style model.
    5. ``merge_and_unload()`` folds LoRA into the base weights.
    6. Assertions prove the merge is numerically correct.
    """

    @pytest.fixture()
    def adapter_artifacts(self, tmp_path):
        """Build grouped LoRA state dict, convert to HF format, write to disk."""
        torch.manual_seed(42)
        grouped_sd = _make_grouped_lora_state_dict(prefix="base_model.model.")
        hf_sd = _convert_grouped_to_hf(grouped_sd)

        target_modules = []
        for li in range(N_LAYERS):
            for eid in range(N_EXPERTS):
                for proj in ("gate_proj", "up_proj", "down_proj"):
                    target_modules.append(f"layers.{li}.mlp.experts.{eid}.{proj}")

        _write_adapter_dir(tmp_path, hf_sd, LORA_DIM, LORA_ALPHA, target_modules)
        return tmp_path, grouped_sd, hf_sd

    # ---- test: HF PEFT loads the adapter without errors ----

    def test_peft_loads_adapter(self, adapter_artifacts):
        from peft import PeftModel

        adapter_dir, _, _ = adapter_artifacts
        hf_model = _make_hf_expert_model()

        peft_model = PeftModel.from_pretrained(hf_model, str(adapter_dir))

        assert peft_model is not None
        assert hasattr(peft_model, "base_model")

    # ---- test: merge_and_unload modifies every expert weight ----

    def test_merge_changes_all_expert_weights(self, adapter_artifacts):
        from peft import PeftModel

        adapter_dir, _, _ = adapter_artifacts
        hf_model = _make_hf_expert_model()
        base_weights = {n: p.data.clone() for n, p in hf_model.named_parameters()}

        peft_model = PeftModel.from_pretrained(hf_model, str(adapter_dir))
        merged = peft_model.merge_and_unload()

        changed = 0
        for name, param in merged.named_parameters():
            if name in base_weights and "experts" in name:
                if not torch.equal(param.data, base_weights[name]):
                    changed += 1

        expected = N_LAYERS * N_EXPERTS * 3  # gate, up, down per expert
        assert changed == expected, (
            f"Expected all {expected} expert weight tensors to change after merge, got {changed}"
        )

    # ---- test: merged weights equal base + B @ A * scale exactly ----

    def test_merge_values_are_numerically_correct(self, adapter_artifacts):
        from peft import PeftModel

        adapter_dir, _, hf_sd = adapter_artifacts
        hf_model = _make_hf_expert_model()
        base_weights = {n: p.data.clone() for n, p in hf_model.named_parameters()}

        peft_model = PeftModel.from_pretrained(hf_model, str(adapter_dir))
        merged = peft_model.merge_and_unload()

        scale = LORA_ALPHA / LORA_DIM

        for name, param in merged.named_parameters():
            if "experts" not in name or name not in base_weights:
                continue

            lora_A_key = f"base_model.model.{name}".replace(".weight", ".lora_A.weight")
            lora_B_key = f"base_model.model.{name}".replace(".weight", ".lora_B.weight")

            if lora_A_key not in hf_sd or lora_B_key not in hf_sd:
                continue

            lora_A = hf_sd[lora_A_key]
            lora_B = hf_sd[lora_B_key]
            expected = base_weights[name] + (lora_B @ lora_A) * scale

            torch.testing.assert_close(
                param.data,
                expected,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Merge mismatch for {name}",
            )

    # ---- test: merged model no longer contains LoRA parameters ----

    def test_merge_removes_lora_params(self, adapter_artifacts):
        from peft import PeftModel

        adapter_dir, _, _ = adapter_artifacts
        hf_model = _make_hf_expert_model()

        peft_model = PeftModel.from_pretrained(hf_model, str(adapter_dir))
        merged = peft_model.merge_and_unload()

        for name, _ in merged.named_parameters():
            assert "lora_" not in name, f"LoRA param {name} should be absent after merge"
        for name, _ in merged.named_modules():
            assert "lora_" not in name, f"LoRA module {name} should be absent after merge"


class TestParamWrapperLayoutVersions:
    """peft flipped the ParamWrapper layout for 3-D expert params in 0.19.1
    (huggingface/peft#3165). Export emits the corrected layout by default with
    an explicit legacy option; import resolves the layout from the checkpoint's
    metadata stamp first, then from tensor shapes, and fails clearly when
    neither settles it.
    """

    def _native(self):
        base = "model.layers.0.mlp.experts"
        return {
            f"{base}.lora_gate_and_up_A": torch.randn(N_EXPERTS, DIM, LORA_DIM),
            f"{base}.lora_gate_and_up_B": torch.randn(N_EXPERTS, LORA_DIM, 2 * MOE_INTER_DIM),
            f"{base}.lora_down_A": torch.randn(N_EXPERTS, MOE_INTER_DIM, LORA_DIM),
            f"{base}.lora_down_B": torch.randn(N_EXPERTS, LORA_DIM, DIM),
        }

    def _export(self, adapter, native, legacy=False):
        out = {}
        for fqn, tensor in native.items():
            for key, value in adapter._convert_lora_to_paramwrapper(fqn, tensor.clone(), legacy_layout=legacy):
                out[key] = value
        return out

    def test_default_export_uses_the_corrected_layout(self):
        out = self._export(_Adapter(), self._native())

        base = "model.layers.0.mlp.experts"
        assert out[f"{base}.base_layer.lora_A.weight"].shape == (LORA_DIM * N_EXPERTS, DIM)
        assert out[f"{base}.base_layer.lora_B.weight"].shape == (2 * MOE_INTER_DIM, LORA_DIM * N_EXPERTS)
        assert out[f"{base}.lora_A.weight"].shape == (LORA_DIM * N_EXPERTS, MOE_INTER_DIM)
        assert out[f"{base}.lora_B.weight"].shape == (DIM, LORA_DIM * N_EXPERTS)

    def test_legacy_option_keeps_the_pre_flip_layout(self):
        out = self._export(_Adapter(), self._native(), legacy=True)

        base = "model.layers.0.mlp.experts"
        assert out[f"{base}.base_layer.lora_A.weight"].shape == (LORA_DIM * N_EXPERTS, 2 * MOE_INTER_DIM)
        assert out[f"{base}.base_layer.lora_B.weight"].shape == (DIM, LORA_DIM * N_EXPERTS)
        assert out[f"{base}.lora_A.weight"].shape == (LORA_DIM * N_EXPERTS, DIM)
        assert out[f"{base}.lora_B.weight"].shape == (MOE_INTER_DIM, LORA_DIM * N_EXPERTS)

    @pytest.mark.parametrize("legacy", [False, True])
    def test_round_trip_is_exact_for_both_layouts(self, legacy):
        adapter = _Adapter()
        native = self._native()
        back = adapter._convert_paramwrapper_to_native(self._export(adapter, native, legacy=legacy))

        assert set(back) == set(native)
        for key in native:
            torch.testing.assert_close(back[key], native[key])

    def test_import_detects_the_layout_from_shapes(self):
        """A legacy-layout file loads correctly with no metadata stamp."""
        adapter = _Adapter()
        native = self._native()
        back = adapter._convert_paramwrapper_to_native(self._export(adapter, native, legacy=True))

        assert set(back) == set(native)
        for key in native:
            torch.testing.assert_close(back[key], native[key])

    def test_unambiguous_tensors_decide_the_layout_for_the_whole_file(self):
        """A legacy MiniMax-M2-style file loads fully correctly without metadata.

        With dim == 2 * moe_inter the gate_up pair is shape-ambiguous, but the
        down pair is not; its vote decides the layout for the whole file.
        """
        adapter = _Adapter()
        adapter.moe_config.dim = 2 * MOE_INTER_DIM
        base = "model.layers.0.mlp.experts"
        native = {
            f"{base}.lora_gate_and_up_A": torch.randn(N_EXPERTS, 2 * MOE_INTER_DIM, LORA_DIM),
            f"{base}.lora_gate_and_up_B": torch.randn(N_EXPERTS, LORA_DIM, 2 * MOE_INTER_DIM),
            f"{base}.lora_down_A": torch.randn(N_EXPERTS, MOE_INTER_DIM, LORA_DIM),
            f"{base}.lora_down_B": torch.randn(N_EXPERTS, LORA_DIM, 2 * MOE_INTER_DIM),
        }
        back = adapter._convert_paramwrapper_to_native(self._export(adapter, native, legacy=True))

        assert set(back) == set(native)
        for key in native:
            torch.testing.assert_close(back[key], native[key])

    @pytest.mark.parametrize("layout, expect_key", [("peft-0.18", "lora_down_B"), ("peft-0.19.1", "lora_down_A")])
    def test_metadata_hint_resolves_ambiguous_shapes(self, layout, expect_key):
        """When shapes cannot distinguish the layouts, the metadata stamp decides."""
        adapter = _Adapter()
        adapter.moe_config.dim = MOE_INTER_DIM  # the down pair becomes shape-ambiguous
        adapter._paramwrapper_layout_hint = layout
        base = "model.layers.0.mlp.experts"

        back = adapter._convert_paramwrapper_to_native(
            {f"{base}.lora_A.weight": torch.randn(LORA_DIM * N_EXPERTS, MOE_INTER_DIM)}
        )

        assert list(back) == [f"{base}.{expect_key}"]

    def test_ambiguous_shapes_without_metadata_fail_clearly(self):
        adapter = _Adapter()
        adapter.moe_config.dim = MOE_INTER_DIM
        base = "model.layers.0.mlp.experts"

        with pytest.raises(ValueError, match="huggingface/peft#3165"):
            adapter._convert_paramwrapper_to_native(
                {f"{base}.lora_A.weight": torch.randn(LORA_DIM * N_EXPERTS, MOE_INTER_DIM)}
            )

    @pytest.mark.parametrize("legacy, stamp", [(False, "peft-0.19.1"), (True, "peft-0.18")])
    def test_metadata_stamp_agreeing_with_shapes_loads(self, legacy, stamp):
        adapter = _Adapter()
        native = self._native()
        adapter._paramwrapper_layout_hint = stamp

        back = adapter._convert_paramwrapper_to_native(self._export(adapter, native, legacy=legacy))

        assert set(back) == set(native)
        for key in native:
            torch.testing.assert_close(back[key], native[key])

    def test_unknown_metadata_stamp_fails_clearly(self):
        """Only the two known layout ids are accepted; anything else is an error, not 'modern'."""
        adapter = _Adapter()
        adapter._paramwrapper_layout_hint = "peft-0.21"

        with pytest.raises(ValueError, match="Unknown peft ParamWrapper layout 'peft-0.21'"):
            adapter._convert_paramwrapper_to_native(self._export(adapter, self._native()))

    @pytest.mark.parametrize("file_is_legacy, stamp", [(True, "peft-0.19.1"), (False, "peft-0.18")])
    def test_metadata_stamp_contradicting_shapes_fails_clearly(self, file_is_legacy, stamp):
        """A stamp the tensor shapes rule out must not win silently, in either direction."""
        adapter = _Adapter()
        exported = self._export(adapter, self._native(), legacy=file_is_legacy)
        adapter._paramwrapper_layout_hint = stamp

        with pytest.raises(ValueError, match="do not belong together"):
            adapter._convert_paramwrapper_to_native(exported)

    @pytest.mark.parametrize("stamp", [None, "peft-0.19.1", "peft-0.18"])
    def test_tensors_disagreeing_about_the_layout_fail_even_with_a_valid_stamp(self, stamp):
        """A file mixing both layouts is corrupt; a valid stamp must not mask that."""
        adapter = _Adapter()
        native = self._native()
        mixed = self._export(adapter, native, legacy=True)
        modern_down_b = {
            key: value
            for key, value in self._export(adapter, native).items()
            if key.endswith(".lora_B.weight") and "base_layer" not in key
        }
        mixed.update(modern_down_b)
        adapter._paramwrapper_layout_hint = stamp

        with pytest.raises(ValueError, match="disagree about their layout"):
            adapter._convert_paramwrapper_to_native(mixed)

    @pytest.mark.parametrize("legacy", [False, True])
    def test_non_gated_experts_round_trip_for_both_layouts(self, legacy):
        """Nemotron-V3-style non-gated experts (gate_up width == moe_inter)."""
        adapter = _Adapter()
        adapter.moe_config.expert_activation = "relu2"
        base = "model.layers.0.mlp.experts"
        native = {
            f"{base}.lora_gate_and_up_A": torch.randn(N_EXPERTS, DIM, LORA_DIM),
            f"{base}.lora_gate_and_up_B": torch.randn(N_EXPERTS, LORA_DIM, MOE_INTER_DIM),
            f"{base}.lora_down_A": torch.randn(N_EXPERTS, MOE_INTER_DIM, LORA_DIM),
            f"{base}.lora_down_B": torch.randn(N_EXPERTS, LORA_DIM, DIM),
        }
        exported = self._export(adapter, native, legacy=legacy)
        if not legacy:
            assert exported[f"{base}.base_layer.lora_A.weight"].shape == (LORA_DIM * N_EXPERTS, DIM)
        back = adapter._convert_paramwrapper_to_native(exported)

        assert set(back) == set(native)
        for key in native:
            torch.testing.assert_close(back[key], native[key])
