# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resumable + final checkpointing for the FSDP2/AutoModel backend.

Split out of ``FsdpStrategy`` (which keeps a thin ``save_model`` / ``save_ckpt``
/ ``load_ckpt`` delegating surface): this is the self-contained checkpoint
subsystem — AutoModel ``Checkpointer`` construction, HF-export promotion, the
DCP save/load dance, ``torch.save`` of client state, best-metric sidecars, and
retention/pruning. It holds a back-reference to the owning strategy for model
unwrapping, rank/mesh info, and rank-0 logging.
"""

import json
import math
import os

import torch
import torch.distributed as dist
import torch.nn as nn


class CheckpointManager:
    """Owns all on-disk checkpoint I/O for ``FsdpStrategy``."""

    CKPT_METRIC_FILENAME = "metric.json"

    def __init__(self, strategy) -> None:
        self.strategy = strategy

    @staticmethod
    def _checkpoint_source(model: nn.Module) -> tuple[str | None, str | None]:
        config = getattr(model, "config", None)
        model_repo_id = (
            getattr(model, "name_or_path", None)
            or getattr(config, "name_or_path", None)
            or getattr(config, "_name_or_path", None)
        )
        model_cache_dir = os.environ.get("HF_HUB_CACHE")
        if not model_cache_dir and os.environ.get("HF_HOME"):
            model_cache_dir = os.path.join(os.environ["HF_HOME"], "hub")
        return model_cache_dir, model_repo_id

    def save_model(self, model: nn.Module, tokenizer, output_dir: str, **kwargs) -> None:
        # Use AutoModel's Checkpointer: its custom-model save_pretrained mixin
        # requires it (raises "No checkpointer provided" otherwise). Outputs
        # consolidated HF safetensors that vLLM can hot-load.
        model = self.strategy._unwrap_model(model)
        # Declare the export precision on the config and every nested sub-config: loaders (vLLM/HF)
        # build each submodule at its config dtype, so a stale fp32 `dtype` (an fp32 master) left on
        # the training config makes that submodule fp32 and mismatch the param_dtype weights on load.
        from molt.utils.utils import convert_to_torch_dtype

        self._set_config_dtype(model.config, convert_to_torch_dtype(self.strategy.param_dtype))
        ckpt = self._build_checkpointer(output_dir, save_consolidated=True, model=model)
        ckpt.save_model(model=model, weights_path=output_dir, tokenizer=tokenizer)
        if dist.is_initialized():
            dist.barrier()
        self._promote_hf_export(output_dir)
        if dist.is_initialized():
            dist.barrier()

    @staticmethod
    def _set_config_dtype(config, dtype: torch.dtype) -> None:
        """Set ``torch_dtype`` on an HF config and every nested sub-config (vision/audio/text)."""
        from transformers import PretrainedConfig

        config.torch_dtype = dtype
        for value in vars(config).values():
            if isinstance(value, PretrainedConfig):
                CheckpointManager._set_config_dtype(value, dtype)

    @staticmethod
    def _promote_hf_export(output_dir: str) -> None:
        """Move AutoModel's HF export to ``output_dir`` for Molt callers."""
        if dist.is_initialized() and dist.get_rank() != 0:
            return

        import shutil

        model_dir = os.path.join(output_dir, "model")
        export_dir = os.path.join(model_dir, "consolidated")
        if not os.path.isdir(export_dir):
            export_dir = model_dir
        if not os.path.isdir(export_dir):
            return

        for name in os.listdir(export_dir):
            src = os.path.join(export_dir, name)
            dst = os.path.join(output_dir, name)
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            elif os.path.exists(dst):
                os.remove(dst)
            shutil.move(src, dst)
        shutil.rmtree(model_dir, ignore_errors=True)

    def _build_checkpointer(self, output_dir: str, save_consolidated: bool, model: nn.Module | None = None):
        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

        model_cache_dir, model_repo_id = self._checkpoint_source(model) if model is not None else (None, None)
        os.makedirs(output_dir, exist_ok=True)
        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=output_dir,
            model_save_format="safetensors",
            model_cache_dir=model_cache_dir,
            model_repo_id=model_repo_id,
            save_consolidated=save_consolidated,
            original_model_root_dir=model_cache_dir,
            is_peft=False,
        )
        return Checkpointer(
            config=config,
            dp_rank=self.strategy._get_dp_rank(include_cp=True),
            tp_rank=self.strategy._get_automodel_rank("tp"),
            pp_rank=0,
            moe_mesh=self.strategy.moe_mesh,
        )

    def _get_ckpt_metric_path(self, ckpt_dir: str) -> str:
        return os.path.join(ckpt_dir, self.CKPT_METRIC_FILENAME)

    @staticmethod
    def _atomic_write_text(path: str, text: str) -> None:
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            f.write(text)
        os.replace(tmp_path, path)

    @staticmethod
    def _atomic_write_torch(path: str, payload) -> None:
        tmp_path = f"{path}.tmp.{os.getpid()}"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)

    @staticmethod
    def _atomic_write_json(path: str, payload) -> None:
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_path, path)

    def _write_ckpt_metric(self, ckpt_dir: str, metric_value, metric_key=None) -> None:
        if hasattr(metric_value, "item"):  # torch / numpy scalar -> python float
            try:
                metric_value = metric_value.item()
            except Exception:
                pass
        path = self._get_ckpt_metric_path(ckpt_dir)
        self._atomic_write_json(path, {"metric_key": metric_key, "metric_value": metric_value})

    def _read_ckpt_metric(self, ckpt_dir: str) -> float | None:
        metric_path = self._get_ckpt_metric_path(ckpt_dir)
        if not os.path.exists(metric_path):
            return None
        try:
            with open(metric_path) as f:
                payload = json.load(f)
            value = payload.get("metric_value") if isinstance(payload, dict) else None
            return None if value is None else float(value)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            self.strategy.print(f"Warning: failed to read checkpoint metric from {metric_path}: {exc}")
            return None

    @staticmethod
    def _dir_size(path: str) -> int:
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                try:
                    total += os.path.getsize(os.path.join(dirpath, filename))
                except OSError:
                    pass
        return total

    @staticmethod
    def _is_loadable_ckpt_dir(path: str) -> bool:
        # Require extra_state.pt, not just model/: it is written last (rank0, after the
        # all-rank barrier that follows model+optim save), so its presence means the whole
        # checkpoint finalized. Without it a preempted first checkpoint (model/ only, no
        # `latest` marker yet) is picked by the mtime fallback and loaded as complete —
        # partial weights resumed at step 0 with a fresh optimizer/scheduler.
        return (
            os.path.isdir(path)
            and os.path.isdir(os.path.join(path, "model"))
            and os.path.isfile(os.path.join(path, "extra_state.pt"))
        )

    def _checkpoint_candidates(self, ckpt_path: str, *, include_best: bool) -> list[tuple[str, float]]:
        if not os.path.isdir(ckpt_path):
            return []
        candidates = []
        for name in os.listdir(ckpt_path):
            path = os.path.join(ckpt_path, name)
            if not os.path.isdir(path):
                continue
            if not include_best and name.startswith("best"):
                continue
            if self._is_loadable_ckpt_dir(path):
                candidates.append((path, os.path.getmtime(path)))
        return candidates

    def _resolve_ckpt_load_dir(self, ckpt_path: str) -> str | None:
        latest = os.path.join(ckpt_path, "latest")
        if os.path.isfile(latest):
            with open(latest) as f:
                tag = f.read().strip()
            load_dir = os.path.join(ckpt_path, tag)
            if self._is_loadable_ckpt_dir(load_dir):
                if not tag.startswith("best"):
                    return load_dir
                regular = self._checkpoint_candidates(ckpt_path, include_best=False)
                if regular:
                    fallback = max(regular, key=lambda item: item[1])[0]
                    self.strategy.print(
                        f"Warning: latest points to best checkpoint {tag}; "
                        f"resuming from newest regular checkpoint {os.path.basename(fallback)}."
                    )
                    return fallback
                return load_dir
            self.strategy.print(
                f"Warning: latest checkpoint {load_dir} is missing or incomplete; scanning checkpoints."
            )

        regular = self._checkpoint_candidates(ckpt_path, include_best=False)
        if regular:
            fallback = max(regular, key=lambda item: item[1])[0]
            self.strategy.print(f"Warning: latest file missing/stale; resuming from {os.path.basename(fallback)}.")
            return fallback

        best = self._checkpoint_candidates(ckpt_path, include_best=True)
        best = [(path, mtime) for path, mtime in best if os.path.basename(path).startswith("best")]
        if best:
            fallback = max(best, key=lambda item: item[1])[0]
            self.strategy.print(f"Warning: only best checkpoints found; resuming from {os.path.basename(fallback)}.")
            return fallback
        return None

    def _prune_checkpoints(self, ckpt_path: str, current_tag: str, max_num: int, max_mem: int, is_best: bool) -> None:
        import shutil

        if is_best:
            for name in os.listdir(ckpt_path):
                path = os.path.join(ckpt_path, name)
                if name.startswith("best") and name != current_tag and os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
            return

        max_size_bytes = None
        if max_mem is not None and max_mem > 0 and not math.isinf(max_mem):
            max_size_bytes = max_mem * 1024**3

        while True:
            subdirs = [
                (os.path.join(ckpt_path, name), os.path.getmtime(os.path.join(ckpt_path, name)))
                for name in os.listdir(ckpt_path)
                if os.path.isdir(os.path.join(ckpt_path, name))
            ]
            regular_subdirs = [
                (path, mtime)
                for path, mtime in subdirs
                if not os.path.basename(path).startswith("best") and os.path.basename(path) != current_tag
            ]
            current_regular_count = sum(1 for path, _ in subdirs if not os.path.basename(path).startswith("best"))
            overflow_num = max(0, current_regular_count - max_num) if max_num and max_num > 0 else 0
            overflow_mem = (
                max_size_bytes is not None and sum(self._dir_size(path) for path, _ in subdirs) > max_size_bytes
            )
            if overflow_num == 0 and not overflow_mem:
                break
            candidates = sorted(
                [(path, self._read_ckpt_metric(path), mtime) for path, mtime in regular_subdirs],
                key=lambda item: (
                    item[1] is not None,
                    item[1] if item[1] is not None else float("-inf"),
                    item[2],
                ),
            )
            if not candidates:
                break
            shutil.rmtree(candidates[0][0], ignore_errors=True)

    def save_ckpt(
        self,
        model: nn.Module,
        ckpt_path: str,
        tag: str,
        max_num: int = 3,
        max_mem: int = 0,
        client_states=None,
        **kwargs,
    ) -> None:
        """DCP-format checkpoint for resumable training (model + optimizer +
        scheduler + RL stats). HF-safetensors export goes through ``save_model``.
        """
        model = self.strategy._unwrap_model(model)
        is_rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
        is_best = tag.startswith("best")

        if is_rank0:
            os.makedirs(ckpt_path, exist_ok=True)

        if dist.is_initialized():
            dist.barrier()

        save_dir = os.path.join(ckpt_path, tag)
        os.makedirs(save_dir, exist_ok=True)

        ckpt = self._build_checkpointer(save_dir, save_consolidated=False, model=model)
        ckpt.save_model(model=model, weights_path=save_dir, tokenizer=None)
        optimizer = kwargs.get("optimizer")
        scheduler = kwargs.get("scheduler")
        if optimizer is not None:
            ckpt.save_optimizer(optimizer=optimizer, model=model, weights_path=save_dir, scheduler=scheduler)

        if dist.is_initialized():
            dist.barrier()

        if is_rank0:
            extra = {"client_state": dict(client_states or {})}
            self._atomic_write_torch(os.path.join(save_dir, "extra_state.pt"), extra)
            self._write_ckpt_metric(save_dir, kwargs.get("metric_value"), kwargs.get("metric_key"))
            if not is_best:
                self._atomic_write_text(os.path.join(ckpt_path, "latest"), tag)
            self._prune_checkpoints(ckpt_path, tag, max_num, max_mem, is_best)
        if dist.is_initialized():
            dist.barrier()

    def load_ckpt(self, model: nn.Module, ckpt_path: str, optimizer=None, scheduler=None, **kwargs):
        """Load the most recent DCP checkpoint under ``ckpt_path``. Returns
        ``(load_path, states)`` where ``states`` carries ``client_state`` keys
        (e.g. ``consumed_samples``); ``(None, {})`` if no checkpoint is found.
        """
        if not os.path.isdir(ckpt_path):
            return None, {}
        load_dir = self._resolve_ckpt_load_dir(ckpt_path)
        if load_dir is None:
            return None, {}

        model = self.strategy._unwrap_model(model)

        # Load the model through the SAME AutoModel Checkpointer that save_ckpt uses,
        # so load is symmetric with save. save_model writes HF-format shards via the
        # model's state_dict_adapter (to_hf): for custom MoE/VLM models (omni3,
        # qwen3.x-moe, glm, kimi) the on-disk keys are RENAMED (e.g.
        # language_model.model.* -> language_model.backbone.*, vision_projector.* ->
        # mlp1.*, grouped experts split per-expert). load_model applies to_hf before /
        # from_hf after the load so those keys match, and handles tie_word_embeddings
        # dedup (e.g. Qwen2.5-0.5B) + MoE strided-view loads. A prior hand-rolled
        # dcp.load matched the model's NATIVE keys against the HF-keyed shards, so for
        # adapter models the renamed params were silently skipped (allow_partial_load)
        # and the resume kept base weights; dense models were unaffected (native == HF).
        ckpt = self._build_checkpointer(load_dir, save_consolidated=False, model=model)
        ckpt.load_model(model=model, model_path=os.path.join(load_dir, "model"))

        optim_dir = os.path.join(load_dir, "optim")
        if optimizer is not None and os.path.isdir(optim_dir):
            import torch.distributed.checkpoint as dcp
            from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState
            from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

            optimizer_state = OptimizerState(model, optimizer, scheduler)
            optim_state_dict = optimizer_state.state_dict()
            dcp.load(
                optim_state_dict,
                checkpoint_id=optim_dir,
                planner=DefaultLoadPlanner(allow_partial_load=True),
            )
            optimizer_state.load_state_dict(optim_state_dict)
            # With --fsdp.offload optimizer the Adam moments must live on CPU; DCP
            # restores them onto the model param's GPU device, so page them back.
            self.strategy.offload_moments_to_cpu(optimizer)

        # Resumable client state (consumed_samples, dataloader/RNG state, RL
        # bookkeeping), torch.save'd in save_ckpt.
        states = {}
        extra_pt = os.path.join(load_dir, "extra_state.pt")
        if os.path.isfile(extra_pt):
            extra = torch.load(extra_pt, weights_only=False)
            states = extra.get("client_state", {}) or {}
        if dist.is_initialized():
            dist.barrier()
        return load_dir, states
