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

"""Tests for CheckpointingConfig in config.py."""

import pytest
import torch

from nemo_automodel.components.checkpoint.config import (
    CheckpointingConfig,
    _is_geq_torch_2_9,
    _is_leq_torch_2_7_1,
)


class TestCheckpointingConfig:
    def test_construct_safetensors(self):
        cfg = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="/tmp/ckpt",
            model_save_format="safetensors",
            model_cache_dir="/tmp/cache",
            model_repo_id="meta-llama/Llama-3-8B",
            save_consolidated=True,
            is_peft=False,
        )
        assert cfg.enabled is True
        assert str(cfg.checkpoint_dir) == "/tmp/ckpt"
        # __post_init__ converts string to SerializationFormat enum
        assert cfg.model_save_format.value == "safetensors"

    def test_construct_torch_save(self):
        cfg = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="/tmp/ckpt",
            model_save_format="torch_save",
            model_cache_dir="/tmp/cache",
            model_repo_id="test",
            save_consolidated=False,
            is_peft=False,
        )
        assert cfg.model_save_format.value == "torch_save"

    def test_peft_torch_save_coerces_format_but_preserves_other_fields(self):
        # PEFT + torch_save flips only model_save_format -> safetensors and
        # save_consolidated -> FINAL; all other user-set fields survive.
        cfg = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="/tmp/ckpt",
            model_save_format="torch_save",
            model_cache_dir="/tmp/cache",
            model_repo_id="test",
            save_consolidated=False,
            is_peft=True,
            staging_dir="/tmp/staging",
            single_rank_consolidation=True,
            v4_compatible=True,
        )
        # Coerced (the two incompatible fields):
        assert cfg.model_save_format.value == "safetensors"
        assert cfg.save_consolidated.value == "final"
        # Preserved (everything else the user set) — the old builder hard-reset these to defaults.
        assert cfg.staging_dir == "/tmp/staging"
        assert cfg.single_rank_consolidation is True
        assert cfg.v4_compatible is True
        assert str(cfg.checkpoint_dir) == "/tmp/ckpt"

    def test_invalid_format_raises(self):
        with pytest.raises(AssertionError, match="Unsupported model save format"):
            CheckpointingConfig(
                enabled=True,
                checkpoint_dir="/tmp/ckpt",
                model_save_format="invalid_format",
                model_cache_dir="/tmp/cache",
                model_repo_id="test",
                save_consolidated=False,
                is_peft=False,
            )

    def test_optional_fields_defaults(self):
        cfg = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="/tmp/ckpt",
            model_save_format="safetensors",
            model_cache_dir="/tmp/cache",
            model_repo_id="test",
            save_consolidated=False,
            is_peft=False,
        )
        assert cfg.is_async is False
        assert cfg.allow_legacy_pickle_restore is False
        assert cfg.dequantize_base_checkpoint is None
        assert cfg.single_rank_consolidation is False
        assert cfg.staging_dir is None
        assert cfg.v4_compatible is False
        assert cfg.diffusers_compatible is False
        assert cfg.best_metric_key == "default"
        assert cfg.consolidation_timeout_minutes == 30
        assert cfg.max_recent_checkpoints is None

    def test_allow_legacy_pickle_restore_override(self):
        cfg = CheckpointingConfig(allow_legacy_pickle_restore=True)

        assert cfg.allow_legacy_pickle_restore is True

    @pytest.mark.parametrize("invalid_value", [None, 0, 1, "false", "true"])
    def test_allow_legacy_pickle_restore_must_be_boolean(self, invalid_value):
        with pytest.raises(ValueError, match="allow_legacy_pickle_restore must be a boolean"):
            CheckpointingConfig(allow_legacy_pickle_restore=invalid_value)

    def test_consolidation_timeout_override(self):
        cfg = CheckpointingConfig(consolidation_timeout_minutes=45)

        assert cfg.consolidation_timeout_minutes == 45

    @pytest.mark.parametrize("timeout_minutes", [0, -1])
    def test_consolidation_timeout_must_be_positive(self, timeout_minutes):
        with pytest.raises(ValueError, match="consolidation_timeout_minutes must be greater than 0"):
            CheckpointingConfig(consolidation_timeout_minutes=timeout_minutes)

    @pytest.mark.parametrize("invalid_value", [0, -1, True, False, 1.5, "2"])
    def test_max_recent_checkpoints_rejects_invalid_values(self, invalid_value):
        with pytest.raises(ValueError, match="checkpoint.max_recent_checkpoints must be unset or a positive integer"):
            CheckpointingConfig(max_recent_checkpoints=invalid_value)

    def test_max_recent_checkpoints_rejects_msc_checkpoint_dir(self):
        with pytest.raises(ValueError, match="max_recent_checkpoints is only supported for local checkpoint"):
            CheckpointingConfig(
                checkpoint_dir="msc://bucket/checkpoints",
                save_consolidated=False,
                max_recent_checkpoints=1,
            )

    @pytest.mark.parametrize("save_consolidated", [True, "final", "every"])
    def test_consolidated_export_rejects_msc_checkpoint_dir(self, save_consolidated):
        with pytest.raises(ValueError, match="not compatible with remote cloud storage"):
            CheckpointingConfig(
                checkpoint_dir="msc://bucket/checkpoints",
                save_consolidated=save_consolidated,
            )

    def test_accepts_msc_dcp_checkpoint_dir(self):
        cfg = CheckpointingConfig(
            checkpoint_dir="msc://bucket/checkpoints",
            save_consolidated=False,
        )

        assert cfg.checkpoint_dir == "msc://bucket/checkpoints"
        assert cfg.save_consolidated.value == "false"
        assert cfg.max_recent_checkpoints is None

    def test_accepts_local_checkpoint_dir_with_retention(self):
        """The rejection is scoped to remote roots; local paths keep working."""
        cfg = CheckpointingConfig(
            checkpoint_dir="/tmp/checkpoints",
            save_consolidated=False,
            max_recent_checkpoints=1,
        )
        assert cfg.max_recent_checkpoints == 1

    def test_importable_from_checkpointing(self):
        """Verify backward compat: import from checkpointing.py still works."""
        from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig as CkptCfg

        assert CkptCfg is CheckpointingConfig

    def test_defaults_construct_without_args(self):
        """Every field has a default, so the recipe layer can construct directly."""
        from huggingface_hub import constants as hf_constants

        cfg = CheckpointingConfig()
        assert cfg.enabled is True
        assert str(cfg.checkpoint_dir) == "checkpoints/"
        assert cfg.model_save_format.value == "safetensors"
        # save_consolidated defaults to "final" and is normalized to SaveConsolidatedMode.FINAL.
        assert cfg.save_consolidated.value == "final"
        assert cfg.is_peft is False
        assert cfg.model_repo_id is None
        # model_cache_dir falls back to the HF hub cache when None.
        assert str(cfg.model_cache_dir) == str(hf_constants.HF_HUB_CACHE)
        assert cfg.max_recent_checkpoints is None

    def test_explicit_cache_dir_is_kept(self):
        cfg = CheckpointingConfig(model_cache_dir="/tmp/cache")
        assert str(cfg.model_cache_dir) == "/tmp/cache"

    def test_peft_torch_save_fallback(self):
        """PEFT + torch_save should fall back to safetensors, preserving checkpoint_dir."""
        cfg = CheckpointingConfig(
            checkpoint_dir="/keep/this",
            model_save_format="torch_save",
            is_peft=True,
        )
        assert cfg.model_save_format.value == "safetensors"
        # PEFT + torch_save resets save_consolidated to the FINAL default.
        assert cfg.save_consolidated.value == "final"
        assert str(cfg.checkpoint_dir) == "/keep/this"

    def test_non_peft_torch_save_preserved(self):
        """Without PEFT, torch_save must be honored."""
        cfg = CheckpointingConfig(model_save_format="torch_save", is_peft=False)
        assert cfg.model_save_format.value == "torch_save"


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.7.1", True),
        ("2.8.0", False),
        ("2.9.0", False),
        ("2.10.0", False),
        ("2.12.1+cu130", False),
    ],
)
def test_torch_leq_2_7_gate_uses_semantic_versioning(monkeypatch, version, expected):
    monkeypatch.setattr(torch, "__version__", version)

    assert _is_leq_torch_2_7_1() is expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.8.0", False),
        ("2.9.0", True),
        ("2.10.0", True),
        ("2.12.1+cu130", True),
    ],
)
def test_torch_geq_2_9_gate_uses_semantic_versioning(monkeypatch, version, expected):
    monkeypatch.setattr(torch, "__version__", version)

    assert _is_geq_torch_2_9() is expected


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("2.8.0", False),
        ("2.9.0", True),
        ("2.10.0", True),
        ("2.12.1+cu130", True),
    ],
)
def test_async_checkpointing_gate_uses_semantic_versioning(monkeypatch, version, expected):
    monkeypatch.setattr(torch, "__version__", version)

    cfg = CheckpointingConfig(is_async=True, save_consolidated=False)

    assert cfg.is_async is expected


class TestBuild:
    """CheckpointingConfig.build constructs a Checkpointer and forwards runtime arguments."""

    def _build(self, **build_kwargs):
        cfg = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="/tmp/ckpt",
            model_save_format="safetensors",
            model_cache_dir="/tmp/cache",
            model_repo_id="org/model",
            save_consolidated=False,
        )
        return cfg.build(dp_rank=0, tp_rank=0, pp_rank=0, **build_kwargs)

    def test_build_constructs_checkpointer_with_config_and_ranks(self):
        checkpointer = self._build()

        assert checkpointer.config.model_repo_id == "org/model"
        assert (checkpointer.dp_rank, checkpointer.tp_rank, checkpointer.pp_rank) == (0, 0, 0)

    def test_build_forwards_pp_group(self):
        pp_group = object()

        checkpointer = self._build(pp_group=pp_group)

        assert checkpointer.pp_group is pp_group
