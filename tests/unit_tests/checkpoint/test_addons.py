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

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.components.checkpoint.addons import (
    ConsolidatedHFAddon,
    _extract_target_modules,
    _group_barrier,
    _is_group_rank_0,
    _maybe_save_custom_model_code,
    _maybe_strip_quantization_config,
)
from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState


def test_group_barrier_uses_model_process_group():
    group = object()
    with (
        patch("nemo_automodel.components.checkpoint.addons.torch.distributed.is_initialized", return_value=True),
        patch("nemo_automodel.components.checkpoint.addons.torch.distributed.barrier") as barrier,
    ):
        _group_barrier(group)
    barrier.assert_called_once_with(group=group)


def test_group_rank_zero_is_relative_to_model_process_group():
    group = object()
    with (
        patch("nemo_automodel.components.checkpoint.addons.torch.distributed.is_initialized", return_value=True),
        patch("nemo_automodel.components.checkpoint.addons.torch.distributed.get_rank", return_value=0) as get_rank,
    ):
        assert _is_group_rank_0(group)
    get_rank.assert_called_once_with(group=group)


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def test_maybe_save_custom_model_code_copies_py_files_and_structure(tmp_path):
    # Arrange: create a nested source tree with .py and non-.py files
    src_root = tmp_path / "src_model_code"
    dst_root = tmp_path / "hf_meta"
    src_root.mkdir(parents=True)
    dst_root.mkdir(parents=True)

    files = {
        "main.py": "print('main')\n",
        "pkg/__init__.py": "# pkg init\n",
        "pkg/subpkg/module.py": "def foo():\n    return 1\n",
        "pkg/readme.txt": "do not copy\n",
    }
    for rel, content in files.items():
        _write(os.path.join(src_root, rel), content)

    # Act
    _maybe_save_custom_model_code(str(src_root), str(dst_root))

    # Assert: .py files copied with preserved structure; non-.py and __init__.py ignored
    assert (dst_root / "main.py").exists()
    assert not (dst_root / "pkg" / "__init__.py").exists()
    assert (dst_root / "pkg" / "subpkg" / "module.py").exists()
    assert not (dst_root / "pkg" / "readme.txt").exists()

    # Verify contents match
    with open(dst_root / "pkg" / "subpkg" / "module.py", "r") as f:
        assert "def foo()" in f.read()


def test_maybe_save_custom_model_code_noop_for_none_or_non_dir(tmp_path):
    dst_root = tmp_path / "hf_meta"
    dst_root.mkdir(parents=True)

    # None input should be a no-op
    _maybe_save_custom_model_code(None, str(dst_root))
    assert list(dst_root.rglob("*.py")) == []

    # Non-directory input should be a no-op
    some_file = tmp_path / "not_a_dir.txt"
    some_file.write_text("hello")
    _maybe_save_custom_model_code(str(some_file), str(dst_root))
    assert list(dst_root.rglob("*.py")) == []


@pytest.mark.parametrize("use_ddp", [False, True])
def test_consolidated_hf_addon_delegates_to_model_metadata_exporter(tmp_path, use_ddp):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    exporter = SimpleNamespace(validate=MagicMock(), save=MagicMock())
    model = nn.Module()
    model._get_consolidated_hf_metadata_exporter = lambda *, tokenizer, original_model_path: exporter
    wrapped_model = object.__new__(DistributedDataParallel)
    nn.Module.__init__(wrapped_model)
    wrapped_model.module = model
    tokenizer = MagicMock()

    ConsolidatedHFAddon().pre_save(
        model_state=SimpleNamespace(model=[wrapped_model if use_ddp else model]),
        hf_metadata_dir=str(metadata_dir),
        tokenizer=tokenizer,
        fqn_to_file_index_mapping={"w": 1},
        fqn_to_dtype_mapping=None,
        original_model_path="/source",
        v4_compatible=True,
    )

    exporter.validate.assert_called_once_with(tokenizer=tokenizer, original_model_path="/source")
    exporter.save.assert_called_once_with(
        hf_metadata_dir=str(metadata_dir),
        tokenizer=tokenizer,
        original_model_path="/source",
    )


def test_consolidated_hf_addon_validates_model_exporter_on_nonzero_rank(tmp_path):
    exporter = SimpleNamespace(
        validate=MagicMock(side_effect=ValueError("invalid export")),
        save=MagicMock(),
    )
    model = nn.Module()
    model._get_consolidated_hf_metadata_exporter = lambda *, tokenizer, original_model_path: exporter

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=1),
        patch("torch.distributed.barrier") as barrier,
        pytest.raises(ValueError, match="invalid export"),
    ):
        ConsolidatedHFAddon().pre_save(
            model_state=SimpleNamespace(model=[model]),
            hf_metadata_dir=str(tmp_path),
            tokenizer=None,
            fqn_to_file_index_mapping={"w": 1},
            fqn_to_dtype_mapping=None,
            original_model_path=None,
            v4_compatible=False,
        )

    exporter.save.assert_not_called()
    barrier.assert_not_called()


def test_consolidated_hf_addon_uses_standard_metadata_when_exporter_declines_without_tokenizer(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    model = nn.Module()
    model.config = {"model_type": "dummy"}
    get_metadata_exporter = MagicMock(return_value=None)
    model._get_consolidated_hf_metadata_exporter = get_metadata_exporter

    ConsolidatedHFAddon().pre_save(
        model_state=SimpleNamespace(model=[model]),
        hf_metadata_dir=str(metadata_dir),
        tokenizer=None,
        fqn_to_file_index_mapping={"w": 1},
        fqn_to_dtype_mapping=None,
        original_model_path=None,
        v4_compatible=False,
    )

    get_metadata_exporter.assert_called_once_with(tokenizer=None, original_model_path=None)
    assert (metadata_dir / "config.json").is_file()


def test_model_state_keeps_lm_head_when_storage_not_shared():
    """Config-tied model with a separate lm_head keeps lm_head.weight on save.

    The resolver reports the top-level config intent (so ``uses_tied_lm_head`` is
    True here), but ModelState gates lm_head dropping on the storage-based
    ``has_local_tied_lm_head``. With a separate lm_head and no shared embedding,
    the save path must keep lm_head.weight. This is the safety that previously came
    from a force-untied exclusion list, now provided by the storage check.
    """

    class _DummyConfig:
        tie_word_embeddings = True

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _DummyConfig()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)

    _DummyModel.__name__ = "Qwen3OmniMoeThinkerForConditionalGeneration"

    model = _DummyModel()
    state = ModelState([model])

    assert state.uses_tied_lm_head is True  # follows the top-level config flag
    assert state.has_local_tied_lm_head is False  # but the tensors do not actually share storage

    state_dict = state.state_dict()
    assert "lm_head.weight" in state_dict  # so the head is kept (storage-gated safety)


def test_model_state_drops_lm_head_when_storage_shared():
    """Config-tied model whose lm_head shares storage with the embedding drops lm_head.weight on save."""

    class _DummyConfig:
        tie_word_embeddings = True

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _DummyConfig()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(4, 2)
            self.lm_head = torch.nn.Linear(2, 4, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight  # genuine tie

    model = _DummyModel()
    state = ModelState([model])

    assert state.uses_tied_lm_head is True
    assert state.has_local_tied_lm_head is True

    state_dict = state.state_dict()
    assert "lm_head.weight" not in state_dict  # dropped because storage is shared
    assert "model.embed_tokens.weight" in state_dict


def test_peft_model_state_can_skip_default_group_broadcast():
    """Subset-mesh ranks already load PEFT state and must not broadcast globally."""

    class _DummyConfig:
        tie_word_embeddings = False

    model = nn.Linear(2, 2)
    model.config = _DummyConfig()
    state = ModelState(model, is_peft=True)

    with patch("nemo_automodel.components.checkpoint.stateful_wrappers.set_model_state_dict") as set_state:
        state.load_state_dict({}, strict=False, broadcast_from_rank0=False)

    options = set_state.call_args.kwargs["options"]
    assert options.full_state_dict is True
    assert options.broadcast_from_rank0 is False


# _extract_target_modules tests
def _make_model_with_named_modules(module_names):
    """Build a dummy model whose ``named_modules`` yields the given names.

    We simulate LoRA sub-modules by adding ``nn.Identity`` leaves under
    the requested paths.  ``_extract_target_modules`` looks for any
    module whose name contains "lora", so we add leaves like
    ``<target>.lora_A``.
    """
    root = nn.Module()
    for name in module_names:
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, nn.Module())
            parent = getattr(parent, part)
        setattr(parent, parts[-1], nn.Identity())
    return root


class _StubStateDictAdapter(StateDictAdapter):
    """Minimal concrete StateDictAdapter, no renames of its own."""

    def to_hf(self, state_dict, **kwargs):
        """Identity pass-through; tensor values keep their native layouts."""
        return state_dict

    def from_hf(self, hf_state_dict, device_mesh=None, **kwargs):
        """Identity pass-through; tensor values keep their HF layouts."""
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn, tensor, **kwargs):
        """Identity pass-through; ``tensor`` keeps its layout, whatever it is."""
        return [(fqn, tensor)]


class TestExtractTargetModules:
    """Tests for _extract_target_modules with combined-projection expansion."""

    def test_simple_non_combined_modules(self):
        """Non-combined module names pass through unchanged."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.o_proj.lora_A",
                "model.layers.0.self_attn.o_proj.lora_B",
                "model.layers.0.mlp.down_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.self_attn.o_proj" in result
        assert "model.layers.0.mlp.down_proj" in result

    def test_qkv_proj_expanded(self):
        """qkv_proj is expanded to q_proj, k_proj, v_proj."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.self_attn.qkv_proj.lora_B",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.self_attn.q_proj" in result
        assert "model.layers.0.self_attn.k_proj" in result
        assert "model.layers.0.self_attn.v_proj" in result
        # Combined name should NOT appear
        assert all("qkv_proj" not in m for m in result)

    def test_gate_up_proj_expanded(self):
        """gate_up_proj is expanded to gate_proj, up_proj."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.mlp.gate_up_proj.lora_A",
                "model.layers.0.mlp.gate_up_proj.lora_B",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.mlp.gate_proj" in result
        assert "model.layers.0.mlp.up_proj" in result
        assert all("gate_up_proj" not in m for m in result)

    def test_mixed_combined_and_regular(self):
        """Mixed combined and regular module names."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.self_attn.o_proj.lora_A",
                "model.layers.0.mlp.gate_up_proj.lora_A",
                "model.layers.0.mlp.down_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        expected = {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        }
        assert set(result) == expected

    def test_torch_compile_prefix_stripped(self):
        """_orig_mod. prefix from torch.compile is stripped before expansion."""
        model = _make_model_with_named_modules(
            [
                "_orig_mod.model.layers.0.self_attn.qkv_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.self_attn.q_proj" in result
        assert "model.layers.0.self_attn.k_proj" in result
        assert "model.layers.0.self_attn.v_proj" in result
        assert all("_orig_mod" not in m for m in result)

    def test_result_is_sorted(self):
        """Return value is sorted."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.mlp.gate_up_proj.lora_A",
                "model.layers.0.self_attn.qkv_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        assert result == sorted(result)


    def test_adapter_target_module_renames_applied(self):
        """An adapter overriding map_peft_target_module_to_hf rewrites the entries.

        Adapters that rename modules between the native and checkpoint layouts
        (e.g. Kimi K3's expert projections) convert the saved keys, so the
        target_modules metadata has to follow or PEFT can't resolve it.
        """

        class _RenamingAdapter(_StubStateDictAdapter):
            def map_peft_target_module_to_hf(self, name: str, *, v4_compatible: bool = False) -> str:
                return name.replace(".mlp.experts.", ".block_sparse_moe.experts.")

        model = _make_model_with_named_modules(
            [
                "model.layers.0.mlp.experts.0.gate_proj.lora_A",
                "model.layers.0.self_attn.q_proj.lora_A",
            ]
        )
        model.state_dict_adapter = _RenamingAdapter()
        result = _extract_target_modules(model)
        assert "model.layers.0.block_sparse_moe.experts.0.gate_proj" in result
        assert "model.layers.0.self_attn.q_proj" in result
        assert all(".mlp.experts." not in m for m in result)

    def test_adapter_without_override_keeps_target_modules_unchanged(self):
        """Adapters that don't rename modules inherit the identity default."""
        adapter = _StubStateDictAdapter()
        assert adapter.map_peft_target_module_to_hf("model.layers.0.self_attn.q_proj") == (
            "model.layers.0.self_attn.q_proj"
        )

        model = _make_model_with_named_modules(
            [
                "model.layers.0.mlp.experts.0.gate_proj.lora_A",
                "model.layers.0.self_attn.q_proj.lora_A",
            ]
        )
        model.state_dict_adapter = adapter
        result = _extract_target_modules(model)
        assert "model.layers.0.mlp.experts.0.gate_proj" in result
        assert "model.layers.0.self_attn.q_proj" in result

    def test_duck_typed_adapter_without_the_method_is_tolerated(self):
        """Adapters that don't subclass StateDictAdapter must not crash the save.

        llama, qwen2/3, and kimivl attach plain classes as state_dict_adapter,
        and some tests attach mixin-only adapters; they have no
        map_peft_target_module_to_hf and just keep their names.
        """

        class _BareAdapter:
            pass

        model = _make_model_with_named_modules(["model.layers.0.self_attn.q_proj.lora_A"])
        model.state_dict_adapter = _BareAdapter()
        result = _extract_target_modules(model)
        assert result == ["model.layers.0.self_attn.q_proj"]


class TestMaybeStripQuantizationConfig:
    """Tests for _maybe_strip_quantization_config."""

    @staticmethod
    def _make_config_with_quant():
        cfg = type("Config", (), {})()
        cfg.quantization_config = {"quant_method": "mxfp4"}
        return cfg

    def test_strips_quantization_config_when_all_params_bf16(self):
        """quantization_config is removed when all params are standard floating-point."""
        model = nn.Linear(4, 4, dtype=torch.bfloat16)
        model.config = self._make_config_with_quant()

        _maybe_strip_quantization_config(model)
        assert not hasattr(model.config, "quantization_config")

    def test_keeps_quantization_config_when_uint8_params_exist(self):
        """quantization_config is preserved when quantized (uint8) parameters exist."""
        model = nn.Module()
        model.register_parameter("weight", nn.Parameter(torch.ones(4, 4, dtype=torch.uint8), requires_grad=False))
        model.config = self._make_config_with_quant()

        _maybe_strip_quantization_config(model)
        assert hasattr(model.config, "quantization_config")

    def test_noop_when_no_quantization_config(self):
        """No error when config has no quantization_config attribute."""
        model = nn.Linear(4, 4)
        model.config = type("Config", (), {})()

        _maybe_strip_quantization_config(model)
        assert not hasattr(model.config, "quantization_config")

    def test_noop_when_no_config(self):
        """No error when model has no config attribute."""
        model = nn.Linear(4, 4)
        _maybe_strip_quantization_config(model)


class TestParamWrapperLayoutStamp:
    """The PEFT save stamps the fused expert LoRA layout into
    automodel_peft_config.json (peft flipped ParamWrapper in 0.19.1,
    huggingface/peft#3165) so loads resolve it from metadata, not shapes."""

    def test_v5_adapter_gets_the_modern_stamp_by_default(self):
        from nemo_automodel.components.checkpoint.addons import _get_paramwrapper_layout_stamp

        adapter = SimpleNamespace(_v5_peft_target_parameters=("mlp.experts.gate_up_proj",))
        assert _get_paramwrapper_layout_stamp(adapter, False, False) == "peft-0.19.1"
        assert _get_paramwrapper_layout_stamp(adapter, False, True) == "peft-0.18"

    def test_non_v5_or_v4_saves_get_no_stamp(self):
        from nemo_automodel.components.checkpoint.addons import _get_paramwrapper_layout_stamp

        v5_adapter = SimpleNamespace(_v5_peft_target_parameters=("mlp.experts.gate_up_proj",))
        plain_adapter = SimpleNamespace(_v5_peft_target_parameters=())
        assert _get_paramwrapper_layout_stamp(plain_adapter, False, False) is None
        assert _get_paramwrapper_layout_stamp(None, False, False) is None
        assert _get_paramwrapper_layout_stamp(v5_adapter, True, False) is None


class TestParamWrapperLayoutHintPlumbing:
    def test_metadata_reader_round_trips(self, tmp_path):
        import json as json_module

        from nemo_automodel.components.checkpoint.checkpointing import _read_paramwrapper_layout_metadata

        assert _read_paramwrapper_layout_metadata(tmp_path) is None
        (tmp_path / "automodel_peft_config.json").write_text(json_module.dumps({"paramwrapper_layout": "peft-0.18"}))
        assert _read_paramwrapper_layout_metadata(tmp_path) == "peft-0.18"

