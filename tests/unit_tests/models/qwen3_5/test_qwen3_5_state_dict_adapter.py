# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Qwen3.5 dense state-dict adapter (fp32-wrapper key rename)."""

from __future__ import annotations

import torch

from nemo_automodel.components.models.qwen3_5.state_dict_adapter import (
    Qwen3_5DenseStateDictAdapter,
    _route_to_fp32_holder,
    _strip_fp32_prefix,
    map_qwen3_5_mtp_from_hf_key,
    map_qwen3_5_mtp_to_hf_key,
)


class TestStripFp32Prefix:
    def test_strips_linear_attn_fp32_prefix(self):
        assert (
            _strip_fp32_prefix("model.language_model.layers.0.linear_attn._fp32_params.A_log")
            == "model.language_model.layers.0.linear_attn.A_log"
        )

    def test_passes_through_keys_without_fp32(self):
        for key in (
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.embed_tokens.weight",
            "model.visual.merger.linear_fc1.weight",
        ):
            assert _strip_fp32_prefix(key) == key

    def test_only_strips_under_linear_attn(self):
        # _fp32_params under any non-linear_attn parent must NOT be stripped.
        key = "model.language_model.layers.0.self_attn._fp32_params.A_log"
        assert _strip_fp32_prefix(key) == key


class TestRouteToFp32Holder:
    def test_routes_bare_a_log_to_holder(self):
        assert (
            _route_to_fp32_holder("model.language_model.layers.0.linear_attn.A_log")
            == "model.language_model.layers.0.linear_attn._fp32_params.A_log"
        )
        assert (
            _route_to_fp32_holder("model.language_model.layers.0.linear_attn.dt_bias")
            == "model.language_model.layers.0.linear_attn._fp32_params.dt_bias"
        )

    def test_passes_through_already_routed_keys(self):
        key = "model.language_model.layers.0.linear_attn._fp32_params.A_log"
        assert _route_to_fp32_holder(key) == key

    def test_routes_bare_dt_bias_to_holder(self):
        # Both SSM-gating master weights (A_log and dt_bias) live in the fp32
        # ``_fp32_params`` holder, so dt_bias is routed the same as A_log.
        assert (
            _route_to_fp32_holder("model.language_model.layers.0.linear_attn.dt_bias")
            == "model.language_model.layers.0.linear_attn._fp32_params.dt_bias"
        )

    def test_does_not_route_other_linear_attn_keys(self):
        # Non SSM-gating params (e.g. the GatedDeltaNet norm) stay in place.
        key = "model.language_model.layers.0.linear_attn.norm.weight"
        assert _route_to_fp32_holder(key) == key

    def test_does_not_route_a_log_outside_linear_attn(self):
        # Defensive: only linear_attn.A_log should be routed.
        key = "model.some.other.path.A_log"
        assert _route_to_fp32_holder(key) == key


class TestAdapter:
    def setup_method(self):
        self.adapter = Qwen3_5DenseStateDictAdapter()

    def _sample_state_dict(self):
        return {
            "model.language_model.layers.0.linear_attn._fp32_params.A_log": torch.zeros(4),
            "model.language_model.layers.0.linear_attn._fp32_params.dt_bias": torch.zeros(4),
            "model.language_model.layers.1.linear_attn._fp32_params.A_log": torch.ones(4),
            "model.language_model.layers.1.linear_attn._fp32_params.dt_bias": torch.ones(4),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
            "model.language_model.embed_tokens.weight": torch.zeros(8, 2),
        }

    def test_to_hf_renames_fp32_params(self):
        sd = self._sample_state_dict()
        out = self.adapter.to_hf(sd)
        assert (
            "model.language_model.layers.0.linear_attn.A_log" in out
            and "model.language_model.layers.0.linear_attn._fp32_params.A_log" not in out
        )
        # Tensors are passed through by reference (no copy).
        assert (
            out["model.language_model.layers.0.linear_attn.A_log"]
            is sd["model.language_model.layers.0.linear_attn._fp32_params.A_log"]
        )
        # Non-fp32 keys are unchanged.
        assert out["model.language_model.embed_tokens.weight"] is sd["model.language_model.embed_tokens.weight"]
        # Number of keys preserved.
        assert len(out) == len(sd)

    def test_to_hf_upcasts_gdn_fp32_params_saved_as_bf16(self):
        sd = {
            "model.language_model.layers.0.linear_attn._fp32_params.A_log": torch.zeros(4, dtype=torch.bfloat16),
            "model.language_model.layers.0.linear_attn._fp32_params.dt_bias": torch.ones(4, dtype=torch.bfloat16),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        }

        out = self.adapter.to_hf(sd)

        assert out["model.language_model.layers.0.linear_attn.A_log"].dtype == torch.float32
        assert out["model.language_model.layers.0.linear_attn.dt_bias"].dtype == torch.float32
        q_proj_key = "model.language_model.layers.0.self_attn.q_proj.weight"
        assert out[q_proj_key] is sd[q_proj_key]
        assert out[q_proj_key].dtype == torch.bfloat16

    def test_forced_hf_dtype_mapping_marks_gdn_fp32_params(self):
        state_dict = {
            "model.language_model.layers.0.linear_attn.A_log": torch.zeros(4, dtype=torch.float32),
            "model.language_model.layers.0.linear_attn.dt_bias": torch.ones(4, dtype=torch.float32),
            "model.language_model.layers.0.linear_attn.conv1d.weight": torch.zeros(4, dtype=torch.float32),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2, dtype=torch.float32),
        }

        assert self.adapter.forced_hf_dtype_mapping(state_dict) == {
            "model.language_model.layers.0.linear_attn.A_log": "F32",
            "model.language_model.layers.0.linear_attn.dt_bias": "F32",
        }

    def test_to_hf_accepts_kwargs(self):
        # Save callsites pass exclude_key_regex, quantization, device_mesh, etc.
        out = self.adapter.to_hf(
            {"x.linear_attn._fp32_params.A_log": torch.zeros(2)},
            exclude_key_regex=r".*_extra_state.*",
            quantization=False,
            v4_compatible=False,
        )
        assert list(out.keys()) == ["x.linear_attn.A_log"]

    def test_from_hf_routes_a_log_to_holder(self):
        hf_sd = {
            "model.language_model.layers.0.linear_attn.A_log": torch.zeros(4),
            "model.language_model.layers.0.linear_attn.dt_bias": torch.ones(4),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        }
        out = self.adapter.from_hf(hf_sd)
        assert (
            "model.language_model.layers.0.linear_attn._fp32_params.A_log" in out
            and "model.language_model.layers.0.linear_attn.A_log" not in out
        )
        assert (
            "model.language_model.layers.0.linear_attn._fp32_params.dt_bias" in out
            and "model.language_model.layers.0.linear_attn.dt_bias" not in out
        )
        assert "model.language_model.layers.0.self_attn.q_proj.weight" in out

    def test_from_hf_upcasts_gdn_fp32_params_loaded_as_bf16(self):
        hf_sd = {
            "model.language_model.layers.0.linear_attn.A_log": torch.zeros(4, dtype=torch.bfloat16),
            "model.language_model.layers.0.linear_attn.dt_bias": torch.ones(4, dtype=torch.bfloat16),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2, dtype=torch.bfloat16),
        }

        out = self.adapter.from_hf(hf_sd)

        a_log_key = "model.language_model.layers.0.linear_attn._fp32_params.A_log"
        dt_bias_key = "model.language_model.layers.0.linear_attn._fp32_params.dt_bias"
        q_proj_key = "model.language_model.layers.0.self_attn.q_proj.weight"
        assert out[a_log_key].dtype == torch.float32
        assert out[dt_bias_key].dtype == torch.float32
        assert out[q_proj_key] is hf_sd[q_proj_key]
        assert out[q_proj_key].dtype == torch.bfloat16

    def test_from_hf_upcasts_bare_gdn_fp32_params_for_unpatched_model(self):
        hf_sd = {
            "model.language_model.layers.0.linear_attn.A_log": torch.zeros(4, dtype=torch.bfloat16),
            "model.language_model.layers.0.linear_attn.dt_bias": torch.ones(4, dtype=torch.bfloat16),
        }
        adapter = Qwen3_5DenseStateDictAdapter(route_linear_attn_fp32_params=False)

        out = adapter.from_hf(hf_sd)

        assert out["model.language_model.layers.0.linear_attn.A_log"].dtype == torch.float32
        assert out["model.language_model.layers.0.linear_attn.dt_bias"].dtype == torch.float32

    def test_from_hf_keeps_bare_a_log_when_configured_for_unpatched_model(self):
        hf_sd = {
            "model.language_model.layers.0.linear_attn.A_log": torch.zeros(4),
            "model.language_model.layers.0.linear_attn.dt_bias": torch.ones(4),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        }
        adapter = Qwen3_5DenseStateDictAdapter(route_linear_attn_fp32_params=False)

        out = adapter.from_hf(hf_sd)

        assert "model.language_model.layers.0.linear_attn.A_log" in out
        assert "model.language_model.layers.0.linear_attn._fp32_params.A_log" not in out
        assert "model.language_model.layers.0.linear_attn.dt_bias" in out
        assert "model.language_model.layers.0.linear_attn._fp32_params.dt_bias" not in out

    def test_from_hf_routes_a_log_when_configured_for_patched_model(self):
        hf_sd = {
            "model.language_model.layers.0.linear_attn.A_log": torch.zeros(4),
            "model.language_model.layers.0.linear_attn.dt_bias": torch.ones(4),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(2, 2),
        }
        adapter = Qwen3_5DenseStateDictAdapter(route_linear_attn_fp32_params=True)

        out = adapter.from_hf(hf_sd)

        assert "model.language_model.layers.0.linear_attn._fp32_params.A_log" in out
        assert "model.language_model.layers.0.linear_attn.A_log" not in out
        assert "model.language_model.layers.0.linear_attn._fp32_params.dt_bias" in out
        assert "model.language_model.layers.0.linear_attn.dt_bias" not in out

    def test_round_trip_is_identity(self):
        sd = self._sample_state_dict()
        round_tripped = self.adapter.from_hf(self.adapter.to_hf(sd))
        assert set(round_tripped.keys()) == set(sd.keys())
        for k, v in sd.items():
            assert round_tripped[k] is v

    def test_convert_single_tensor_to_hf(self):
        t = torch.ones(4)
        out = self.adapter.convert_single_tensor_to_hf(
            "model.language_model.layers.0.linear_attn._fp32_params.A_log", t
        )
        assert out == [("model.language_model.layers.0.linear_attn.A_log", t)]
        out = self.adapter.convert_single_tensor_to_hf(
            "model.language_model.layers.0.linear_attn._fp32_params.dt_bias", t
        )
        assert out == [("model.language_model.layers.0.linear_attn.dt_bias", t)]

    def test_convert_single_tensor_to_hf_upcasts_gdn_fp32_params(self):
        t = torch.zeros(4, dtype=torch.bfloat16)
        out = self.adapter.convert_single_tensor_to_hf(
            "model.language_model.layers.0.linear_attn._fp32_params.A_log", t
        )
        assert out[0][0] == "model.language_model.layers.0.linear_attn.A_log"
        assert out[0][1].dtype == torch.float32

    def test_convert_single_tensor_passthrough(self):
        t = torch.zeros(2, 2)
        out = self.adapter.convert_single_tensor_to_hf("model.language_model.embed_tokens.weight", t)
        assert out == [("model.language_model.embed_tokens.weight", t)]


class TestMTPKeyMapping:
    def test_maps_hf_mtp_fusion_keys_to_native(self):
        assert map_qwen3_5_mtp_from_hf_key("mtp.fc.weight") == "mtp.layers.0.eh_proj.weight"
        assert map_qwen3_5_mtp_from_hf_key("mtp.pre_fc_norm_embedding.weight") == "mtp.layers.0.enorm.weight"
        assert map_qwen3_5_mtp_from_hf_key("mtp.pre_fc_norm_hidden.weight") == "mtp.layers.0.hnorm.weight"
        assert map_qwen3_5_mtp_from_hf_key("mtp.norm.weight") == "mtp.layers.0.final_layernorm.weight"

    def test_maps_native_mtp_fusion_keys_to_hf(self):
        assert map_qwen3_5_mtp_to_hf_key("mtp.layers.0.eh_proj.weight") == "mtp.fc.weight"
        assert map_qwen3_5_mtp_to_hf_key("mtp.layers.0.enorm.weight") == "mtp.pre_fc_norm_embedding.weight"
        assert map_qwen3_5_mtp_to_hf_key("mtp.layers.0.hnorm.weight") == "mtp.pre_fc_norm_hidden.weight"
        assert map_qwen3_5_mtp_to_hf_key("mtp.layers.0.final_layernorm.weight") == "mtp.norm.weight"

    def test_adapter_round_trips_mtp_keys(self):
        adapter = Qwen3_5DenseStateDictAdapter()
        native = {
            "mtp.layers.0.eh_proj.weight": torch.randn(8, 16),
            "mtp.layers.0.self_attn.q_proj.weight": torch.randn(16, 8),
        }

        hf = adapter.to_hf(native)
        assert "mtp.fc.weight" in hf
        assert "mtp.layers.0.self_attn.q_proj.weight" in hf

        roundtrip = adapter.from_hf(hf)
        assert set(roundtrip) == set(native)
        for key, tensor in native.items():
            assert roundtrip[key] is tensor
