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

import math
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.optimization import get_scheduler

from molt.models.utils import resolve_ac_mode
from molt.trainer.fsdp.checkpoint import CheckpointManager
from molt.utils.distributed_sampler import DistributedSampler


def _get_model_wrapper_cls():
    """Lazy import to avoid the models -> strategy import cycle."""
    from molt.models.base import BaseModel

    return BaseModel


class FsdpStrategy:
    """FSDP2 + TP/CP/SP/EP backend using NeMo AutoModel.

    Mirrors DeepspeedStrategy's public surface so trainers stay backend-agnostic.
    The model is built/parallelized via ``NeMoAutoModelForCausalLM.from_pretrained``
    inside ``Actor``; this strategy handles distributed setup, optimizer/scheduler
    construction, collectives, and checkpointing. Trainers own their objectives;
    AutoModel Engine owns backward and optimizer-update mechanics.
    """

    def __init__(
        self,
        seed: int = 42,
        full_determinism: bool = False,
        max_norm: float = 1.0,
        micro_train_batch_size: int = 1,
        train_batch_size: int = 1,
        args=None,
    ) -> None:
        self.args = args
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.seed = seed
        self.full_determinism = full_determinism
        self.max_norm = max_norm

        fsdp = args.fsdp
        self.tp_size = getattr(fsdp, "tp_size", 1)
        self.cp_size = getattr(fsdp, "cp_size", 1)
        self.ep_size = getattr(fsdp, "ep_size", 1)
        self.pp_size = getattr(fsdp, "pp_size", 1)
        self.param_dtype = getattr(fsdp, "param_dtype", "bf16")
        offload = getattr(fsdp, "offload", "none")
        if offload not in {"none", "optimizer"}:
            raise ValueError(f"Unsupported --fsdp.offload mode: {offload!r}; choose none or optimizer")
        self.offload_optimizer = offload == "optimizer"
        # SP off by default (opt in via --fsdp.sequence_parallel); avoids the
        # _NormPartial 2D TP+FSDP weight-load hang on the HF-fallback path.
        self.sequence_parallel = bool(getattr(fsdp, "sequence_parallel", False))

        self.world_size: int = 1
        self.device_mesh = None
        self.moe_mesh = None
        self.dp_size = 1
        self.dp_cp_size = 1
        self.accumulated_gradient: int = 1
        # On-disk checkpoint I/O lives in CheckpointManager; the save_*/load_*
        # methods below delegate to it.
        self.checkpoint = CheckpointManager(self)

    # ProcessGroup / DeviceMesh aren't picklable, but `datasets.map(num_proc>1)`
    # pickles the strategy via the bound method. Drop the distributed handles for
    # the CPU-only preprocessing workers, which don't need them.
    _UNPICKLABLE_ATTRS = ("device_mesh", "moe_mesh", "distributed_config")

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._UNPICKLABLE_ATTRS}

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k in self._UNPICKLABLE_ATTRS:
            self.__dict__.setdefault(k, None)

    def _get_automodel_mesh(self, name: str):
        if self.device_mesh is None:
            return None
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

        return get_flat_mesh(self.device_mesh, name)

    def _get_dp_group(self, include_cp: bool = False):
        mesh = self._get_automodel_mesh("dp_cp" if include_cp and self.cp_size > 1 else "dp")
        return mesh.get_group() if mesh is not None else None

    def _get_dp_group_size(self, include_cp: bool = False) -> int:
        group = self._get_dp_group(include_cp=include_cp)
        if group is None:
            return dist.get_world_size() if dist.is_initialized() else 1
        return dist.get_world_size(group=group)

    def _get_automodel_rank(self, name: str) -> int:
        mesh = self._get_automodel_mesh(name)
        return mesh.get_local_rank() if mesh is not None else 0

    def _get_dp_rank(self, include_cp: bool = False) -> int:
        if include_cp and self.cp_size > 1:
            return self._get_automodel_rank("dp_cp")
        return self._get_automodel_rank("dp")

    # ---------------------------------------------------------------- bring-up

    def setup_distributed(self, timeout: timedelta = timedelta(minutes=30)) -> None:
        if self.full_determinism:
            transformers.enable_full_determinism(self.seed)
        else:
            transformers.set_seed(self.seed)

        if self.args.local_rank != -1:
            os.environ["LOCAL_RANK"] = str(self.args.local_rank)

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", timeout=timeout)

        self.world_size = dist.get_world_size()
        if self.pp_size > 1:
            raise NotImplementedError("Molt trainers are not pipeline-parallel aware yet; set --fsdp.pp_size 1")

        from nemo_automodel.components.distributed.config import FSDP2Config, MoEParallelizerConfig
        from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes

        # Allow actor/ref TP embedding calls to reuse equivalent vocab masks.
        if self.tp_size > 1:
            try:
                from torch.distributed.tensor._ops import _mask_buffer
            except ImportError:
                _mask_buffer = None
            if _mask_buffer is not None and not getattr(_mask_buffer.MaskBuffer, "_molt_patched", False):

                def _safe_materialize(self, mask):
                    self.data = mask
                    self.refcount += 1

                _mask_buffer.MaskBuffer.materialize_mask = _safe_materialize
                _mask_buffer.MaskBuffer._molt_patched = True

        from molt.utils.utils import convert_to_torch_dtype

        torch_dtype = convert_to_torch_dtype(self.param_dtype)
        # Params/forward in the requested dtype, reduce-scatter in fp32. Don't force
        # module outputs to fp32: policy/value losses must see the same dtype
        # behavior as the DeepSpeed and PR #1176 paths.
        mp_policy = (
            None
            if torch_dtype == torch.float32
            else MixedPrecisionPolicy(
                param_dtype=torch_dtype,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
        )
        # AC must be set on FSDP2Config: the from_pretrained kwarg only reaches
        # the ep_size>1 MoE parallelizer, so dense/HF models would otherwise
        # train with no AC. Sourced from --actor/--model.gradient_checkpoint.
        _actor_cfg = getattr(self.args, "actor", None)
        _model_cfg = getattr(self.args, "model", None)
        activation_checkpointing = resolve_ac_mode(
            getattr(_actor_cfg, "gradient_checkpoint", False) or getattr(_model_cfg, "gradient_checkpoint", False)
        )
        self.distributed_config = FSDP2Config(
            sequence_parallel=self.sequence_parallel,
            mp_policy=mp_policy,
            activation_checkpointing=activation_checkpointing,
            # Engine selects the runtime accumulation-sync policy for each trainer.
            # Keep the construction default false so non-Engine forwards do not inherit
            # a pending synchronization state.
            defer_fsdp_grad_sync=False,
        )
        # ignore_router_for_ac saves the router projection during AC so topk
        # routing is not recomputed in backward — a near-tie token re-routing on
        # recompute shifts grouped-GEMM shapes and raises CheckpointError.
        # MOLT_MOE_RESHARD_AFTER_FWD=1 trades one extra expert all-gather in
        # backward for a lower activation peak on memory-bound runs.
        self.moe_config = (
            MoEParallelizerConfig(
                mp_policy=mp_policy,
                ignore_router_for_ac=True,
                reshard_after_forward=os.environ.get("MOLT_MOE_RESHARD_AFTER_FWD", "0") == "1",
            )
            if self.ep_size > 1
            else None
        )

        mesh_context = MeshContext.build(
            self.distributed_config,
            ParallelismSizes(
                tp_size=self.tp_size,
                pp_size=self.pp_size,
                cp_size=self.cp_size,
                ep_size=self.ep_size,
            ),
            world_size=self.world_size,
        )
        self.device_mesh, self.moe_mesh = mesh_context.device_mesh, mesh_context.moe_mesh

        # Mesh sub-groups (CP all-to-all, EP reduce-scatter) inherit NCCL's 600s
        # default watchdog, not the world-group `timeout`; a compile-bound cold
        # first step can exceed 600s, so raise every sub-group to the world value.
        from torch.distributed.distributed_c10d import _set_pg_timeout

        _seen_pg = set()
        for _mesh in (self.device_mesh, self.moe_mesh):
            if _mesh is None:
                continue
            for _pg in _mesh.get_all_groups():
                if _pg is not None and id(_pg) not in _seen_pg:
                    _seen_pg.add(id(_pg))
                    _set_pg_timeout(timeout, _pg)

        # Mesh exposes flat "dp" for data loading and "dp_cp" for FSDP reduce-scatter.
        # Grad accumulation is DP-only; CP ranks share samples and split sequence work.
        dp_size = self._get_dp_group_size(include_cp=False)
        self.dp_cp_size = self._get_dp_group_size(include_cp=True)
        if getattr(getattr(self.args, "train", None), "dynamic_batch_enable", False):
            self.accumulated_gradient = 1
        else:
            batch_per_step = self.micro_train_batch_size * dp_size
            accum_steps, remainder = divmod(self.train_batch_size, batch_per_step)
            if accum_steps < 1 or remainder != 0:
                raise ValueError(
                    "Invalid batch config for AutoModel/FSDP2: require "
                    "`train.batch_size = train.micro_batch_size * dp_size * grad_accum_steps` "
                    f"(got train.batch_size={self.train_batch_size}, "
                    f"train.micro_batch_size={self.micro_train_batch_size}, dp_size={dp_size})."
                )
            self.accumulated_gradient = accum_steps
        self.dp_size = dp_size
        self.print(
            f"[FSDP] world={self.world_size} dp={self.dp_size} cp={self.cp_size} tp={self.tp_size} "
            f"ep={self.ep_size} dp_cp={self.dp_cp_size} grad_accum={self.accumulated_gradient}"
        )

    # ---------------------------------------------------------------- prepare

    def prepare(self, *args):
        ret = []
        for arg in args:
            if isinstance(arg, tuple):
                if len(arg) != 2:
                    raise ValueError(f"prepare() tuple must be (model, cfg); got len={len(arg)}")
                model, cfg = arg
                ret.append(self._init_train_model(model, cfg))
            else:
                # Eval/reference models need no optimizer or scheduler — pass through.
                ret.append(arg)
        return ret[0] if len(ret) == 1 else ret

    def _init_train_model(self, model, cfg: dict):
        train_model = self._unwrap_model(model)
        params = [p for p in train_model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("Cannot build optimizer: model has no trainable parameters")

        kind = cfg["optim"]
        adam = cfg["adam"]
        if kind == "muon":
            from molt.trainer.fsdp.muon import build_automodel_muon_optimizer

            optimizer = build_automodel_muon_optimizer(train_model, cfg["muon"], adam, self.device_mesh)
        elif kind == "adam":
            optimizer = torch.optim.AdamW(
                params,
                lr=adam["lr"],
                betas=tuple(adam["betas"]),
                eps=adam["eps"],
                weight_decay=adam["weight_decay"],
                foreach=False,
                fused=False,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {kind}")
        scheduler_steps = cfg["scheduler_steps"]
        scheduler = get_scheduler(
            cfg.get("lr_scheduler", "constant"),
            optimizer,
            num_warmup_steps=math.ceil(scheduler_steps * cfg.get("lr_warmup_ratio", 0.03)),
            num_training_steps=scheduler_steps,
            scheduler_specific_kwargs={"min_lr_rate": cfg.get("min_lr_ratio", 0.1)},
        )
        from nemo_automodel.components.distributed.mesh import MeshContext
        from nemo_automodel.engine import Engine

        # Molt owns CPU optimizer offload: the offloader wraps the AdamW with the
        # step()/zero_grad() surface Engine drives, while checkpointing and the LR
        # scheduler keep using the wrapped optimizer itself.
        engine_optimizer = optimizer
        if self.offload_optimizer:
            if kind != "adam":
                raise ValueError(
                    "--fsdp.offload optimizer supports AdamW only; Muon's Newton-Schulz "
                    "iterations are impractical on CPU. Use --optim adam, or --fsdp.offload none."
                )
            from molt.trainer.fsdp.optimizer_offload import CpuOptimizerOffloader

            engine_optimizer = CpuOptimizerOffloader(optimizer)

        max_grad_norm = cfg.get("max_norm", self.max_norm)
        engine = Engine(
            train_model,
            optimizer=engine_optimizer,
            lr_scheduler=scheduler,
            mesh_context=MeshContext.from_meshes(self.device_mesh, self.moe_mesh),
            gradient_accumulation_steps=self.accumulated_gradient,
            defer_fsdp_grad_sync=os.environ.get("MOLT_DEFER_GRAD_SYNC", "1") == "1",
            max_grad_norm=max_grad_norm if max_grad_norm and max_grad_norm > 0 else None,
        )
        if isinstance(model, _get_model_wrapper_cls()):
            model.model = engine
        else:
            model = engine
        return model, optimizer, scheduler

    def sync_replicated_grads(self, params) -> None:
        """Mean-all-reduce gradients of replicated (non-FSDP-wrapped) params over the
        data-parallel(+CP) group.

        FSDP2 only reduces grads of params inside its wrapped modules; a module added
        after wrapping (e.g. the critic's scalar value head) is replicated with a local
        grad per rank, so it must be averaged over the same ``dp_cp`` group FSDP uses.
        Call right before the Engine's optimizer step. Assumes a flat DP mesh (no
        HSDP/``dp_replicate``).
        """
        group = self._get_dp_group(include_cp=True)
        if group is None:
            return
        world = dist.get_world_size(group=group)
        if world == 1:
            return
        from molt.trainer.fsdp.optimizer_offload import local_shard

        for p in params:
            if p.grad is None:
                continue
            grad = local_shard(p.grad)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
            grad.div_(world)

    def offload_moments_to_cpu(self, optimizer) -> None:
        """Page the Adam moments back to CPU after a checkpoint resume (DCP restores
        them onto the model param's GPU device). No-op unless --fsdp.offload optimizer."""
        if self.offload_optimizer:
            from molt.trainer.fsdp.optimizer_offload import offload_moments_to_cpu

            offload_moments_to_cpu(optimizer)

    def debug_grad_stats(self, model: nn.Module, optim_name: str) -> None:
        debug = os.environ.get("MOLT_FSDP_DEBUG_GRADS", "")
        if not debug or debug == "0":
            return
        enabled = {part.strip() for part in debug.split(",") if part.strip()}
        if "1" not in enabled and "all" not in enabled and optim_name not in enabled:
            return

        rank = dist.get_rank() if dist.is_initialized() else 0
        top_k = int(os.environ.get("MOLT_FSDP_DEBUG_GRADS_TOPK", "8"))
        pattern_env = os.environ.get("MOLT_FSDP_DEBUG_GRADS_FILTER")
        patterns = (
            [part.strip() for part in pattern_env.split(",") if part.strip()]
            if pattern_env
            else ["score", "lm_head", "embed_tokens", "layers.0.", "layers.31.", ".norm"]
        )
        rows = []
        nonfinite_tensors = 0
        total_tensors = 0
        total_elems = 0
        nonfinite_elems = 0
        total_sum_sq = 0.0

        for param_name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            if patterns and not any(pattern in param_name for pattern in patterns):
                continue
            total_tensors += 1
            local_grad = (grad.to_local() if isinstance(grad, DTensor) else grad).detach()
            total_elems += local_grad.numel()
            finite = torch.isfinite(local_grad)
            bad = local_grad.numel() - int(finite.sum().item())
            nonfinite_elems += bad
            if bad:
                nonfinite_tensors += 1
            finite_grad = torch.where(finite, local_grad, torch.zeros_like(local_grad)).double()
            local_sum_sq = finite_grad.pow(2).sum().item()
            total_sum_sq += local_sum_sq
            local_norm = math.sqrt(local_sum_sq)
            local_max = finite_grad.abs().max().item() if finite_grad.numel() else 0.0
            placement = tuple(str(p) for p in grad.placements) if isinstance(grad, DTensor) else ("local",)
            rows.append((local_norm, local_max, bad, param_name, placement, tuple(local_grad.shape)))

        rows.sort(key=lambda item: item[0], reverse=True)
        header = (
            f"[FSDPGradDebug][rank={rank}][{optim_name}] tensors={total_tensors} "
            f"nonfinite_tensors={nonfinite_tensors} elems={total_elems} nonfinite_elems={nonfinite_elems} "
            f"local_norm64={math.sqrt(total_sum_sq):.6e}"
        )
        print(header, flush=True)
        for local_norm, local_max, bad, param_name, placement, shape in rows[:top_k]:
            print(
                f"[FSDPGradDebug][rank={rank}][{optim_name}] "
                f"norm={local_norm:.6e} max={local_max:.6e} bad={bad} "
                f"shape={shape} placement={placement} name={param_name}",
                flush=True,
            )

    def global_token_count(self, mask: torch.Tensor) -> torch.Tensor:
        """All-reduce ``mask.sum()`` across the data-parallel data mesh.

        Call this before the model forward while every CP rank still sees the
        full logical sequence. Token denominators are reduced over DP only
        because CP ranks share samples.
        """
        local = mask if mask.ndim == 0 else mask.sum()
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else local.device
        local = local.to(dtype=torch.float32, device=device)
        dp_group = self._get_dp_group(include_cp=False)
        if dist.is_initialized() and dp_group is not None:
            dist.all_reduce(local, op=dist.ReduceOp.SUM, group=dp_group)
        return local

    def compute_perf_metrics(
        self, mfu, local_seq_count: float, local_token_sum: float, seconds: float, prefix: str = "perf/"
    ) -> dict:
        """Peak memory + training MFU/throughput for one optimization phase.

        MFU is delegated to AutoModel's ``AutoMFU`` over the GLOBAL batch and ALL GPUs
        (parallelism-invariant); ``mfu`` is the caller's ``AutoMFU`` (None -> memory
        only). ``prefix`` namespaces keys so colocated actor/critic report separately
        instead of overwriting on the last-wins status merge.
        """
        metrics = {}
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            metrics[f"{prefix}gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated(dev) / 1024**3
            metrics[f"{prefix}gpu_mem_reserved_gb"] = torch.cuda.max_memory_reserved(dev) / 1024**3
        if mfu is None or seconds <= 0.0:
            return metrics

        # Step time = slowest rank's; reduce to max so MFU is identical on every rank.
        seconds = self.all_reduce(seconds, op="max")
        global_tokens = float(self.global_token_count(torch.tensor(local_token_sum)))
        global_seqs = float(self.global_token_count(torch.tensor(local_seq_count)))
        if global_seqs >= 1:
            mfu_pct = mfu((round(global_seqs), round(global_tokens / global_seqs)), seconds, self.world_size)
            if mfu_pct is not None:
                metrics[f"{prefix}mfu"] = mfu_pct
                metrics[f"{prefix}train_tokens_per_sec"] = global_tokens / seconds
        return metrics

    # ---------------------------------------------------------------- data

    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle: bool = True,
        collate_fn=None,
        drop_last: bool = True,
        sampler=None,
        consumed_samples: int = 0,
        num_workers: int = 0,
    ):
        dp_group = self._get_dp_group(include_cp=False)
        if sampler is None and dist.is_initialized() and dp_group is not None:
            num_replicas = dist.get_world_size(group=dp_group)
            rank = dist.get_rank(group=dp_group)
            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
                consumed_samples=consumed_samples,
            )

        return StatefulDataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    # ---------------------------------------------------------------- comm

    def all_reduce(self, data, op: str = "mean"):
        if isinstance(data, dict):
            return {k: self.all_reduce(v, op) for k, v in data.items()}
        if not torch.is_tensor(data):
            data = torch.tensor(data, device=torch.cuda.current_device(), dtype=torch.float32)
        else:
            data = data.detach().clone().to(torch.cuda.current_device())
        # "mean" reduces with SUM then divides by world_size below.
        reduce_op = {"mean": dist.ReduceOp.SUM, "sum": dist.ReduceOp.SUM, "max": dist.ReduceOp.MAX}[op]
        dist.all_reduce(data, op=reduce_op)
        if op == "mean":
            data = data / dist.get_world_size()
        return data.item() if data.ndim == 0 else data

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        return (not dist.is_initialized()) or dist.get_rank() == 0

    def _unwrap_model(self, model) -> nn.Module:
        if isinstance(model, _get_model_wrapper_cls()):
            return self._unwrap_model(model.model)
        if hasattr(model, "get_base_model_for_fsdp"):
            return model.get_base_model_for_fsdp()
        if hasattr(model, "module"):
            return model.module
        return model

    # ---------------------------------------------------------------- I/O
    # On-disk checkpoint I/O lives in CheckpointManager; these thin wrappers preserve
    # the strategy.save_*/load_* surface trainers and CLIs call.

    def save_model(self, model: nn.Module, tokenizer, output_dir: str, **kwargs) -> None:
        self.checkpoint.save_model(model, tokenizer, output_dir, **kwargs)

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
        self.checkpoint.save_ckpt(
            model, ckpt_path, tag, max_num=max_num, max_mem=max_mem, client_states=client_states, **kwargs
        )

    def prune_checkpoints(self, ckpt_path: str, keep_tag: str, max_num: int, max_mem: float = 0) -> None:
        """Retain at most ``max_num`` versioned subdirs under ``ckpt_path`` (rank-0 only).
        Gives the HF-export dir the same recency-based retention that ``save_ckpt`` applies
        to DCP checkpoints — ``save_model`` alone never prunes. ``keep_tag`` is always kept.
        """
        if self.is_rank_0():
            self.checkpoint._prune_checkpoints(ckpt_path, keep_tag, max_num, max_mem, is_best=False)

    def load_ckpt(self, model: nn.Module, ckpt_path: str, optimizer=None, scheduler=None, **kwargs):
        return self.checkpoint.load_ckpt(model, ckpt_path, optimizer=optimizer, scheduler=scheduler, **kwargs)
