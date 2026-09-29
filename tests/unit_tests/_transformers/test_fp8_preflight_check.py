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

"""Tests for the FP8 dequantize pre-flight check in model_init.

See https://github.com/NVIDIA-NeMo/Automodel/issues/2114 for background on why
this check exists.
"""

import json
from types import SimpleNamespace

import pytest
from transformers import PretrainedConfig

from nemo_automodel._transformers import model_init
from nemo_automodel._transformers.model_init import (
    _FP8_PREFLIGHT_DISABLE_ENV,
    _FP8_PREFLIGHT_FOOTPRINT_THRESHOLD,
    _check_fp8_dequantize_will_fit,
    _get_hf_param_count_estimate,
    _has_streaming_fp8_checkpoint_load,
    _param_count_from_local_safetensors,
    _quant_config_is_fp8_full_materialize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hf_config(*, layers=80, hidden=12288, vocab=131072, inter=29568, n_experts=1):
    """Build a SimpleNamespace mimicking a transformer-style HF config."""
    return SimpleNamespace(
        num_hidden_layers=layers,
        hidden_size=hidden,
        vocab_size=vocab,
        intermediate_size=inter,
        num_local_experts=n_experts,
    )


def _fp8_dequant_cfg():
    return {"quant_method": "fp8", "dequantize": True}


def _fp8_no_dequant_cfg():
    return {"quant_method": "fp8", "dequantize": False}


# Mocked free-memory values used to drive the budget logic deterministically.
_MEM_200GB = 200 * 1024**3
_MEM_8GB = 8 * 1024**3


# ---------------------------------------------------------------------------
# _quant_config_is_fp8_full_materialize
# ---------------------------------------------------------------------------


class TestQuantConfigIsFp8FullMaterialize:
    def test_dict_fp8_dequantize_true(self):
        assert _quant_config_is_fp8_full_materialize({"quant_method": "fp8", "dequantize": True}) is True

    def test_dict_fp8_dequantize_false(self):
        assert _quant_config_is_fp8_full_materialize({"quant_method": "fp8", "dequantize": False}) is False

    def test_dict_fp8_no_dequantize_key(self):
        assert _quant_config_is_fp8_full_materialize({"quant_method": "fp8"}) is False

    def test_object_fp8_dequantize_true(self):
        cfg = SimpleNamespace(quant_method="fp8", dequantize=True)
        assert _quant_config_is_fp8_full_materialize(cfg) is True

    def test_non_fp8(self):
        assert _quant_config_is_fp8_full_materialize({"quant_method": "bnb_8bit", "dequantize": True}) is False

    def test_none(self):
        assert _quant_config_is_fp8_full_materialize(None) is False

    def test_rejects_non_canonical_dequant_aliases(self):
        # Only the canonical ``dequantize`` flag triggers; speculative aliases
        # that no transformers/Automodel code path sets must not be treated as a
        # full-materialize request.
        assert _quant_config_is_fp8_full_materialize({"quant_method": "fp8", "is_dequantize": True}) is False
        assert _quant_config_is_fp8_full_materialize({"quant_method": "fp8", "dequant": True}) is False


# ---------------------------------------------------------------------------
# _get_hf_param_count_estimate
# ---------------------------------------------------------------------------


class TestGetHfParamCountEstimate:
    def test_uses_num_parameters_when_present(self):
        cfg = SimpleNamespace(num_parameters=12_345_000_000)
        assert _get_hf_param_count_estimate(cfg) == 12_345_000_000

    def test_formula_dense_model(self):
        # 7B-ish LLaMA shape should land in the single-digit billions.
        cfg = _make_hf_config(layers=32, hidden=4096, vocab=32000, inter=11008, n_experts=1)
        n = _get_hf_param_count_estimate(cfg)
        assert n is not None
        assert 5e9 < n < 2e10

    def test_formula_moe_inflates_param_count(self):
        # MoE inflates params via n_experts: a Mixtral-8x7B-shaped config
        # should produce noticeably more parameters than a dense 7B.
        moe = _get_hf_param_count_estimate(
            _make_hf_config(layers=32, hidden=4096, vocab=32000, inter=14336, n_experts=8)
        )
        dense = _get_hf_param_count_estimate(
            _make_hf_config(layers=32, hidden=4096, vocab=32000, inter=14336, n_experts=1)
        )
        assert moe > dense * 3

    def test_walks_text_subconfig(self):
        # Multimodal wrappers store layer counts on a text_config; the walker
        # must recurse rather than return zero. Use a real PretrainedConfig so
        # the isinstance check passes.
        from transformers import PretrainedConfig

        class _Sub(PretrainedConfig):
            model_type = "tiny"

        text = _Sub(num_hidden_layers=8, hidden_size=2048, vocab_size=32000, intermediate_size=8192)
        wrapper = SimpleNamespace(text_config=text)
        n = _get_hf_param_count_estimate(wrapper)
        assert n is not None and n > 0

    def test_returns_none_when_unestimable(self):
        assert _get_hf_param_count_estimate(SimpleNamespace()) is None

    def test_gqa_estimates_fewer_params_than_mha(self):
        # Same dims; grouped-query attention (8 KV heads) must estimate fewer
        # parameters than full multi-head attention (64 KV heads), proving the
        # attention term is GQA-aware rather than a flat 4*H*H.
        base = dict(num_hidden_layers=32, hidden_size=8192, vocab_size=128000, intermediate_size=28672, head_dim=128)
        mha = _get_hf_param_count_estimate(SimpleNamespace(num_attention_heads=64, num_key_value_heads=64, **base))
        gqa = _get_hf_param_count_estimate(SimpleNamespace(num_attention_heads=64, num_key_value_heads=8, **base))
        assert mha is not None and gqa is not None
        assert gqa < mha

    def test_mha_matches_legacy_4hh_formula(self):
        # When head_dim == H / n_heads and n_kv == n_heads, the GQA-aware term
        # collapses to the original 4*H*H attention estimate.
        hidden_size, layers, vocab_size, intermediate_size, num_heads = 4096, 32, 32000, 11008, 32
        cfg = SimpleNamespace(
            num_hidden_layers=layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            head_dim=hidden_size // num_heads,
        )
        legacy = (
            layers * (4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size) + vocab_size * hidden_size
        )
        assert _get_hf_param_count_estimate(cfg) == legacy


# ---------------------------------------------------------------------------
# _check_fp8_dequantize_will_fit
# ---------------------------------------------------------------------------


class TestCheckFp8DequantizeWillFit:
    @pytest.fixture(autouse=True)
    def _force_formula_estimate(self, monkeypatch):
        # Keep these budget tests hermetic and off the HF cache: route the
        # param-count estimate through the config formula / num_parameters by
        # disabling the local-safetensors lookup.
        monkeypatch.setattr(model_init, "_param_count_from_local_safetensors", lambda *a, **k: None)

    def test_raises_when_footprint_exceeds_budget(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))

        # ~Devstral / Mistral Medium ballpark: ~246 GB BF16 vs an 8 GB budget.
        cfg = _make_hf_config(layers=88, hidden=12288, vocab=131072, inter=28672)

        with pytest.raises(RuntimeError) as excinfo:
            _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "mistralai/Mistral-Medium", force_hf=True)
        msg = str(excinfo.value)
        assert "Mistral-Medium" in msg
        assert "BF16 footprint" in msg
        assert "Total CUDA memory" in msg
        assert "issues/2114" in msg
        assert "force_hf=True selects Transformers' full-materialize loader" in msg
        assert "no streaming FP8 checkpoint loader registered" in msg
        assert _FP8_PREFLIGHT_DISABLE_ENV in msg

    def test_supported_streaming_route_gives_exact_recovery(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))

        class _StreamingModel:
            _supports_streaming_fp8_checkpoint_load = True

        monkeypatch.setattr(model_init, "_resolve_custom_model_cls_for_config", lambda config: _StreamingModel)
        cfg = _make_hf_config(layers=88, hidden=12288, vocab=131072, inter=28672)
        cfg.quantization_config = {"quant_method": "fp8", "weight_block_size": None}

        assert _has_streaming_fp8_checkpoint_load(cfg) is True
        with pytest.raises(RuntimeError) as excinfo:
            _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "mistralai/Devstral-2", force_hf=True)

        msg = str(excinfo.value)
        assert "Remove force_hf=True" in msg
        assert "do not pass FineGrainedFP8Config(dequantize=True)" in msg
        assert "leave the checkpoint's native FP8 quantization_config unchanged" in msg

    def test_per_block_checkpoint_is_not_claimed_by_per_tensor_streaming_loader(self, monkeypatch):
        class _StreamingModel:
            _supports_streaming_fp8_checkpoint_load = True

        monkeypatch.setattr(model_init, "_resolve_custom_model_cls_for_config", lambda config: _StreamingModel)
        cfg = SimpleNamespace(quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]})

        assert _has_streaming_fp8_checkpoint_load(cfg) is False

    def test_fallback_route_does_not_claim_force_hf(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))
        monkeypatch.setattr(model_init, "_resolve_custom_model_cls_for_config", lambda config: None)
        cfg = _make_hf_config(layers=88, hidden=12288, vocab=131072, inter=28672)

        with pytest.raises(RuntimeError) as excinfo:
            _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "unsupported/model", force_hf=False)

        msg = str(excinfo.value)
        assert "Automodel fell back to Transformers' full-materialize loader" in msg
        assert "force_hf=True selects" not in msg
        assert "Remove force_hf=True" not in msg

    def test_returns_none_when_footprint_fits(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_200GB, _MEM_200GB))

        cfg = _make_hf_config(layers=32, hidden=4096, vocab=32000, inter=11008)  # ~7B
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "tinyllama/TinyLlama-FP8") is None

    def test_budgets_against_total_not_free_memory(self, monkeypatch):
        """A transiently low FREE value must not false-trip a model that fits.

        The verdict budgets against TOTAL memory, so another process holding
        most of the card's free memory does not change the decision.
        """
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        cfg = SimpleNamespace(num_parameters=1_000_000_000)  # 2 GB BF16
        total = 80 * 1024**3
        almost_nothing_free = 1 * 1024**3
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (almost_nothing_free, total))
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "x") is None

    def test_noop_when_quant_config_none(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))
        cfg = _make_hf_config()  # would-be huge model
        assert _check_fp8_dequantize_will_fit(cfg, None, "some/model") is None

    def test_noop_when_quant_method_not_fp8(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))
        cfg = _make_hf_config()
        assert _check_fp8_dequantize_will_fit(cfg, {"quant_method": "bnb_8bit", "dequantize": True}, "x") is None

    def test_noop_when_dequantize_false(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))
        cfg = _make_hf_config()
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_no_dequant_cfg(), "x") is None

    def test_noop_when_cuda_unavailable(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)

        def _boom(*a, **k):
            raise AssertionError("mem_get_info should not be called when CUDA is unavailable")

        monkeypatch.setattr("torch.cuda.mem_get_info", _boom)
        cfg = _make_hf_config()
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "x") is None

    def test_noop_when_disable_env_set(self, monkeypatch):
        monkeypatch.setenv(_FP8_PREFLIGHT_DISABLE_ENV, "1")

        def _boom(*a, **k):
            raise AssertionError("no CUDA calls should happen when the disable env is set")

        monkeypatch.setattr("torch.cuda.is_available", _boom)
        cfg = _make_hf_config()
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "x") is None

    def test_noop_when_param_count_unestimable(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        def _boom(*a, **k):
            raise AssertionError("mem_get_info should not run when the estimate is None")

        monkeypatch.setattr("torch.cuda.mem_get_info", _boom)
        assert _check_fp8_dequantize_will_fit(SimpleNamespace(), _fp8_dequant_cfg(), "x") is None

    def test_noop_when_mem_get_info_fails(self, monkeypatch):
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)

        def _raise(*a, **k):
            raise RuntimeError("no CUDA driver")

        monkeypatch.setattr("torch.cuda.mem_get_info", _raise)
        cfg = _make_hf_config(layers=88, hidden=12288)
        # Fail open: we'd rather let the load proceed than block on a flaky driver call.
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "x") is None

    def test_threshold_boundary(self, monkeypatch):
        """At exactly the budget the check passes; shrink TOTAL and it trips."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        cfg = SimpleNamespace(num_parameters=1_000_000_000)  # 2 GB BF16

        # Budget == THRESHOLD * total, so this total makes the budget exactly 2 GB.
        total = int((2 * 1024**3) / _FP8_PREFLIGHT_FOOTPRINT_THRESHOLD)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (total, total))
        assert _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "x") is None

        # Drop TOTAL by 1 GB: the budget falls below the 2 GB footprint -> trips.
        smaller = total - 1024**3
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (smaller, smaller))
        with pytest.raises(RuntimeError):
            _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "x")

    def test_object_quant_config(self, monkeypatch):
        """A FineGrainedFP8Config-like object (not a dict) must still be detected."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))
        cfg = _make_hf_config(layers=88, hidden=12288)
        quant = SimpleNamespace(quant_method="fp8", dequantize=True)
        with pytest.raises(RuntimeError):
            _check_fp8_dequantize_will_fit(cfg, quant, "any/model")

    def test_fallback_wording_does_not_blame_force_hf(self, monkeypatch):
        """With force_hf=False the diagnostic must not suggest dropping force_hf.

        The fallback path never had force_hf on, so that advice would be a no-op
        and point at the wrong cause.
        """
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))
        cfg = _make_hf_config(layers=88, hidden=12288)

        with pytest.raises(RuntimeError) as excinfo:
            _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "any/model", force_hf=False)
        msg = str(excinfo.value)
        assert "force_hf" not in msg
        assert "no streaming FP8 checkpoint loader registered" in msg

    def test_force_hf_without_streaming_loader_does_not_offer_unavailable_route(self, monkeypatch):
        """Dropping force_hf is only actionable when a streaming loader exists."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))
        cfg = _make_hf_config(layers=88, hidden=12288)

        with pytest.raises(RuntimeError) as excinfo:
            _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "any/model")
        msg = str(excinfo.value)
        assert "force_hf=True selects Transformers' full-materialize loader" in msg
        assert "no streaming FP8 checkpoint loader registered" in msg
        assert "Remove force_hf=True" not in msg

    def test_diagnostic_reports_gib(self, monkeypatch):
        """Sizes are computed with 1024**3, so the label has to be GiB, not GB."""
        monkeypatch.setattr("torch.cuda.is_available", lambda: True)
        monkeypatch.setattr("torch.cuda.mem_get_info", lambda *a, **k: (_MEM_8GB, _MEM_8GB))
        cfg = _make_hf_config(layers=88, hidden=12288)

        with pytest.raises(RuntimeError) as excinfo:
            _check_fp8_dequantize_will_fit(cfg, _fp8_dequant_cfg(), "any/model")
        msg = str(excinfo.value)
        assert "GiB" in msg
        assert " GB " not in msg


# ---------------------------------------------------------------------------
# Wiring: the force_hf branch must invoke the pre-flight before HF's loader.
# ---------------------------------------------------------------------------


class TestForceHfBranchWiring:
    def test_force_hf_invokes_preflight_before_loader(self, monkeypatch):
        """Drive _init_model with force_hf=True and a sentinel-raising preflight.

        If the sentinel propagates, the preflight is wired in. If
        _from_pretrained_parent_class is reached, the wiring is broken.
        """
        from nemo_automodel._transformers import model_init

        class _Sentinel(Exception):
            pass

        def _raise_sentinel(*a, **k):
            raise _Sentinel()

        monkeypatch.setattr(model_init, "_check_fp8_dequantize_will_fit", _raise_sentinel)
        monkeypatch.setattr(
            model_init,
            "get_hf_config",
            lambda path, attn, **_: PretrainedConfig(architectures=["Dummy"], quantization_config=None),
        )
        monkeypatch.setattr(model_init, "_propagate_torch_dtype_to_subconfigs", lambda *a, **k: None)
        monkeypatch.setattr(model_init, "_streaming_bnb_supported", lambda *a, **k: False)

        class _DummyCls:
            _model_mapping = {}

            @classmethod
            def _from_pretrained_parent_class(cls, *a, **k):
                raise AssertionError("HF loader must not run when preflight raises")

        with pytest.raises(_Sentinel):
            model_init._init_model(
                _DummyCls,
                "fake/repo",
                "sdpa",
                "bfloat16",
                None,  # quantization_config
                True,  # force_hf
            )

    def test_non_force_hf_fallback_invokes_preflight(self, monkeypatch):
        """The non-force_hf HF fallback (no custom class) must also pre-flight.

        An fp8+dequantize checkpoint with no custom streaming class reaches the
        same full-materialize HF loader, so the guard runs there too (#2114).
        """

        class _Sentinel(Exception):
            pass

        def _raise_sentinel(*a, **k):
            raise _Sentinel()

        monkeypatch.setattr(model_init, "_check_fp8_dequantize_will_fit", _raise_sentinel)
        monkeypatch.setattr(
            model_init,
            "get_hf_config",
            lambda path, attn, **_: PretrainedConfig(
                architectures=["Dummy"], quantization_config={"quant_method": "fp8", "dequantize": True}
            ),
        )
        monkeypatch.setattr(model_init, "_propagate_torch_dtype_to_subconfigs", lambda *a, **k: None)
        monkeypatch.setattr(model_init, "_streaming_bnb_supported", lambda *a, **k: False)
        monkeypatch.setattr(model_init, "_resolve_custom_model_cls_for_config", lambda *a, **k: None)

        class _DummyCls:
            __name__ = "NeMoDummy"
            _model_mapping = {}

            @classmethod
            def _from_pretrained_parent_class(cls, *a, **k):
                raise AssertionError("HF loader must not run when preflight raises")

        with pytest.raises(_Sentinel):
            model_init._init_model(
                _DummyCls,
                "fake/repo",
                "sdpa",
                "bfloat16",
                None,  # quantization_config
                False,  # force_hf -> exercise the fallback branch
            )


class TestConfigOnlyInitSkipsPreflight:
    """from_config builds never touch the checkpoint, so the guard must not run.

    Before this gate, a config that merely carried fp8 dequantize metadata made
    from_config raise even though nothing was going to be materialized.
    """

    class _Sentinel(Exception):
        pass

    @pytest.fixture(autouse=True)
    def _sentinel_preflight(self, monkeypatch):
        def _raise_sentinel(*a, **k):
            raise self._Sentinel()

        monkeypatch.setattr(model_init, "_check_fp8_dequantize_will_fit", _raise_sentinel)
        monkeypatch.setattr(model_init, "_propagate_torch_dtype_to_subconfigs", lambda *a, **k: None)
        monkeypatch.setattr(model_init, "_streaming_bnb_supported", lambda *a, **k: False)

    @staticmethod
    def _fp8_config_object():
        # a config object (not a repo path) carrying fp8 dequantize metadata
        return SimpleNamespace(
            architectures=["Dummy"],
            quantization_config={"quant_method": "fp8", "dequantize": True},
            name_or_path="fake/config-only",
        )

    class _DummyModel:
        pass

    def test_force_hf_config_only_does_not_preflight(self, monkeypatch):
        outer = self

        class _DummyCls:
            _model_mapping = {}

            @classmethod
            def _from_pretrained_parent_class(cls, *a, **k):
                raise AssertionError("from_config path must not hit the pretrained loader")

            @classmethod
            def _from_config_parent_class(cls, *a, **k):
                return outer._DummyModel()

        _, model = model_init._init_model(
            _DummyCls,
            self._fp8_config_object(),
            "sdpa",
            "bfloat16",
            None,  # quantization_config
            True,  # force_hf
        )
        assert model is not None

    def test_fallback_config_only_does_not_preflight(self, monkeypatch):
        monkeypatch.setattr(model_init, "_resolve_custom_model_cls_for_config", lambda *a, **k: None)
        monkeypatch.setattr(model_init, "_prepopulate_remote_code_cache", lambda *a, **k: None)
        monkeypatch.setattr(model_init, "_try_get_remote_code_model_cls", lambda *a, **k: None)
        outer = self

        class _DummyCls:
            __name__ = "NeMoDummy"
            _model_mapping = {}

            @classmethod
            def _from_pretrained_parent_class(cls, *a, **k):
                raise AssertionError("from_config path must not hit the pretrained loader")

            @classmethod
            def _from_config_parent_class(cls, *a, **k):
                return outer._DummyModel()

        _, model = model_init._init_model(
            _DummyCls,
            self._fp8_config_object(),
            "sdpa",
            "bfloat16",
            None,  # quantization_config
            False,  # force_hf -> exercise the fallback branch
        )
        assert model is not None


class TestParamCountFromLocalSafetensors:
    """Exact element-count source used as preference #1 by the estimate."""

    def test_counts_elements_from_single_file(self, tmp_path):
        import torch
        from safetensors.torch import save_file

        save_file(
            {"a.weight": torch.zeros(10, 20), "b.weight": torch.zeros(5)},
            str(tmp_path / "model.safetensors"),
        )
        assert _param_count_from_local_safetensors(str(tmp_path)) == 205

    def test_counts_elements_from_sharded_index(self, tmp_path):
        import torch
        from safetensors.torch import save_file

        save_file({"a.weight": torch.zeros(10, 20)}, str(tmp_path / "model-00001-of-00002.safetensors"))
        save_file({"b.weight": torch.zeros(30)}, str(tmp_path / "model-00002-of-00002.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": 0},
                    "weight_map": {
                        "a.weight": "model-00001-of-00002.safetensors",
                        "b.weight": "model-00002-of-00002.safetensors",
                    },
                }
            )
        )
        assert _param_count_from_local_safetensors(str(tmp_path)) == 230

    def test_none_when_no_safetensors(self, tmp_path):
        assert _param_count_from_local_safetensors(str(tmp_path)) is None

    def test_none_when_path_falsy(self):
        assert _param_count_from_local_safetensors(None) is None
        assert _param_count_from_local_safetensors("") is None
