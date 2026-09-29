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

import json
import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from nemo_automodel.components.checkpoint._backports.hf_storage import _DIFFUSERS_INDEX_FN
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
)


class _FakeAsyncSaveResponse:
    """Mimics AsyncSaveResponse with an upload_completion future gated on an event."""

    def __init__(self):
        self.upload_finished = threading.Event()
        self.upload_completion = MagicMock()
        self.upload_completion.result.side_effect = self.upload_finished.wait
        self.staging_completion = MagicMock()


def _make_async_checkpointer(
    tmp_path,
    save_consolidated="every",
    single_rank_consolidation=False,
    diffusers_compatible=False,
    future=None,
):
    """Build a real Checkpointer in async mode with mocked internals (no dist)."""
    config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(tmp_path),
        model_save_format="safetensors",
        model_cache_dir=str(tmp_path / "cache"),
        model_repo_id="test/model",
        save_consolidated=save_consolidated,
        is_peft=False,
        is_async=True,
        single_rank_consolidation=single_rank_consolidation,
        diffusers_compatible=diffusers_compatible,
    )
    # CheckpointingConfig.__post_init__ downgrades is_async on torch < 2.9; force it
    # back on so the test exercises the async path regardless of local torch.
    config.is_async = True
    with (
        patch("torch.distributed.is_initialized", return_value=False),
        patch("nemo_automodel.components.checkpoint.checkpointing._new_gloo_process_group", return_value=None),
        patch("nemo_automodel.components.checkpoint.checkpointing.DefaultStager", MagicMock()),
    ):
        checkpointer = Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)

    checkpointer._maybe_build_consolidated_index = MagicMock(return_value={"w": 1})
    checkpointer._maybe_build_original_dtype_mapping = MagicMock(return_value=None)
    checkpointer._get_storage_writer = MagicMock(return_value=MagicMock())
    checkpointer._do_save = MagicMock(return_value=future)
    checkpointer._addons = []
    return checkpointer


def _save_model(checkpointer, tmp_path):
    """Run save_model with a mock model and the HF adaptation patched to identity."""
    model = MagicMock()
    model.state_dict.return_value = {"w": MagicMock()}
    with (
        patch(
            "nemo_automodel.components.checkpoint.checkpointing._maybe_adapt_state_dict_to_hf",
            side_effect=lambda *a, **kw: a[1],
        ),
        patch("torch.distributed.is_initialized", return_value=False),
    ):
        checkpointer.save_model(model, str(tmp_path / "step_1"))


class TestAsyncDeferredConsolidation:
    """save_model in async mode defers distributed consolidation to a background thread."""

    @patch("nemo_automodel.components.checkpoint.checkpointing.consolidate_safetensors_files_on_every_rank")
    def test_consolidation_waits_for_upload_then_runs(self, mock_consolidate, tmp_path):
        """The background thread consolidates only after upload_completion resolves."""
        future = _FakeAsyncSaveResponse()
        checkpointer = _make_async_checkpointer(tmp_path, future=future)

        _save_model(checkpointer, tmp_path)

        assert checkpointer._consolidation_thread is not None
        assert not mock_consolidate.called

        future.upload_finished.set()
        checkpointer.async_wait()

        mock_consolidate.assert_called_once()
        kwargs = mock_consolidate.call_args.kwargs
        assert kwargs["input_dir"] == str(tmp_path / "step_1" / "model")
        assert kwargs["output_dir"] == str(tmp_path / "step_1" / "model" / "consolidated")
        assert kwargs["fqn_to_index_mapping"] == {"w": 1}
        assert kwargs["process_group"] is checkpointer._consolidation_process_group
        assert checkpointer._consolidation_thread is None

    @patch("nemo_automodel.components.checkpoint.checkpointing.consolidate_safetensors_files_on_every_rank")
    def test_writer_consolidation_disabled_when_deferred(self, mock_consolidate, tmp_path):
        """The storage writer must not also consolidate when the background thread will."""
        future = _FakeAsyncSaveResponse()
        future.upload_finished.set()
        checkpointer = _make_async_checkpointer(tmp_path, future=future)

        _save_model(checkpointer, tmp_path)
        checkpointer.async_wait()

        writer_args = checkpointer._get_storage_writer.call_args.args
        assert writer_args[-1] is True  # consolidation_handled_externally

    @patch("nemo_automodel.components.checkpoint.checkpointing.consolidate_safetensors_files_on_every_rank")
    def test_consolidation_error_surfaces_in_async_wait(self, mock_consolidate, tmp_path):
        """A consolidation failure is re-raised on the main thread, then cleared."""
        mock_consolidate.side_effect = RuntimeError("consolidation exploded")
        future = _FakeAsyncSaveResponse()
        future.upload_finished.set()
        checkpointer = _make_async_checkpointer(tmp_path, future=future)

        _save_model(checkpointer, tmp_path)

        with pytest.raises(RuntimeError, match="consolidation exploded"):
            checkpointer.async_wait()
        # error is cleared after being raised once
        checkpointer.async_wait()

    @patch("nemo_automodel.components.checkpoint.checkpointing.consolidate_safetensors_files_on_every_rank")
    def test_diffusers_rename_runs_after_deferred_consolidation(self, mock_consolidate, tmp_path):
        """diffusers_compatible renames the consolidated index in the background thread."""

        def _fake_consolidate(**kwargs):
            os.makedirs(kwargs["output_dir"], exist_ok=True)
            with open(os.path.join(kwargs["output_dir"], "model.safetensors.index.json"), "w") as f:
                json.dump({"weight_map": {}}, f)

        mock_consolidate.side_effect = _fake_consolidate
        future = _FakeAsyncSaveResponse()
        future.upload_finished.set()
        checkpointer = _make_async_checkpointer(tmp_path, diffusers_compatible=True, future=future)

        _save_model(checkpointer, tmp_path)
        checkpointer.async_wait()

        consolidated_dir = tmp_path / "step_1" / "model" / "consolidated"
        assert not (consolidated_dir / "model.safetensors.index.json").exists()
        assert (consolidated_dir / _DIFFUSERS_INDEX_FN).exists()

    @patch("nemo_automodel.components.checkpoint.checkpointing.consolidate_safetensors_files_on_every_rank")
    def test_single_rank_consolidation_keeps_writer_path(self, mock_consolidate, tmp_path):
        """single_rank_consolidation stays on the storage writer's finish() path."""
        checkpointer = _make_async_checkpointer(tmp_path, single_rank_consolidation=True)

        _save_model(checkpointer, tmp_path)
        checkpointer.async_wait()

        assert checkpointer._consolidation_thread is None
        assert not mock_consolidate.called
        writer_args = checkpointer._get_storage_writer.call_args.args
        assert writer_args[-1] is False  # writer keeps its finish() consolidation

    @patch("nemo_automodel.components.checkpoint.checkpointing.consolidate_safetensors_files_on_every_rank")
    def test_save_consolidated_false_schedules_nothing(self, mock_consolidate, tmp_path):
        """save_consolidated=false never schedules a background consolidation."""
        checkpointer = _make_async_checkpointer(tmp_path, save_consolidated="false")

        _save_model(checkpointer, tmp_path)
        checkpointer.async_wait()

        assert checkpointer._consolidation_thread is None
        assert not mock_consolidate.called

    @patch("nemo_automodel.components.checkpoint.checkpointing.consolidate_safetensors_files_on_every_rank")
    def test_consolidation_runs_without_future(self, mock_consolidate, tmp_path):
        """A missing async response does not block the deferred consolidation."""
        checkpointer = _make_async_checkpointer(tmp_path, future=None)

        _save_model(checkpointer, tmp_path)
        checkpointer.async_wait()

        mock_consolidate.assert_called_once()

    def test_join_without_thread_is_noop(self, tmp_path):
        """Joining with no pending consolidation does nothing."""
        checkpointer = _make_async_checkpointer(tmp_path)
        checkpointer._join_deferred_consolidation()
        assert checkpointer._consolidation_thread is None

    def test_async_wait_releases_completed_stagers(self, tmp_path):
        """Completed async saves release stagers that PyTorch cannot safely reuse."""
        checkpointer = _make_async_checkpointer(tmp_path)
        model_stager = MagicMock()
        optimizer_stager = MagicMock()
        checkpointer._model_ctx.stager = model_stager
        checkpointer._optim_ctx.stager = optimizer_stager
        checkpointer._model_ctx.future = MagicMock()
        checkpointer._optim_ctx.future = MagicMock()

        checkpointer.async_wait()

        model_stager.close.assert_called_once_with()
        optimizer_stager.close.assert_called_once_with()
        assert checkpointer._model_ctx.stager is None
        assert checkpointer._optim_ctx.stager is None

    def test_async_save_recreates_released_stager(self, tmp_path):
        """The next async save gets a new stager after async_wait releases the old one."""
        checkpointer = _make_async_checkpointer(tmp_path)
        checkpointer._model_ctx.stager = None
        stager = MagicMock()

        with (
            patch("nemo_automodel.components.checkpoint.checkpointing.DefaultStager", return_value=stager),
            patch("nemo_automodel.components.checkpoint.checkpointing.dcp") as mock_dcp,
        ):
            Checkpointer._do_save(checkpointer, {"weight": MagicMock()}, "/model")

        assert checkpointer._model_ctx.stager is stager
        assert mock_dcp.async_save.call_args.kwargs["async_stager"] is stager


class TestAsyncConsolidationProcessGroupCreation:
    """Async checkpointing creates the timeout-protected consolidation group when needed."""

    def _make_config(self, **overrides):
        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir="ckpt/",
            model_save_format="safetensors",
            model_cache_dir="cache/",
            model_repo_id="test/model",
            save_consolidated=overrides.pop("save_consolidated", "final"),
            is_peft=False,
            is_async=True,
            **overrides,
        )
        config.is_async = True
        return config

    def _build(self, config):
        sentinel = MagicMock(name="gloo_pg")
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "nemo_automodel.components.checkpoint.checkpointing._new_gloo_process_group",
                return_value=sentinel,
            ),
            patch("nemo_automodel.components.checkpoint.checkpointing.DefaultStager", MagicMock()),
        ):
            checkpointer = Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)
        return checkpointer, sentinel

    def test_created_for_async_final_consolidation(self):
        checkpointer, sentinel = self._build(self._make_config(save_consolidated="final"))
        assert checkpointer._consolidation_process_group is sentinel

    def test_not_created_when_consolidation_disabled(self):
        checkpointer, _ = self._build(self._make_config(save_consolidated="false"))
        assert checkpointer._consolidation_process_group is None

    def test_not_created_for_single_rank_consolidation(self):
        checkpointer, _ = self._build(self._make_config(single_rank_consolidation=True))
        assert checkpointer._consolidation_process_group is None

    def test_not_created_without_distributed(self):
        config = self._make_config()
        with (
            patch("torch.distributed.is_initialized", return_value=False),
            patch(
                "nemo_automodel.components.checkpoint.checkpointing._new_gloo_process_group",
                return_value=MagicMock(),
            ),
            patch("nemo_automodel.components.checkpoint.checkpointing.DefaultStager", MagicMock()),
        ):
            checkpointer = Checkpointer(config, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)
        assert checkpointer._consolidation_process_group is None

    def test_close_destroys_consolidation_group(self):
        checkpointer, consolidation_pg = self._build(self._make_config())
        checkpointer._model_ctx.process_group = None
        checkpointer._optim_ctx.process_group = None

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.destroy_process_group") as destroy_process_group,
        ):
            checkpointer.close()

        destroy_process_group.assert_called_once_with(consolidation_pg)
        assert checkpointer._consolidation_process_group is None
