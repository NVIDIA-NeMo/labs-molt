import math
import os
from collections import defaultdict
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import transformers
from torch.distributed.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers.optimization import get_scheduler

from molt.models.utils import resolve_ac_mode
from molt.trainer.fsdp.checkpoint import CheckpointManager
from molt.trainer.fsdp.optimizer_offload import CpuOptimizerOffloader, local_shard
from molt.utils.distributed_sampler import DistributedSampler

try:
    from torch.distributed.fsdp._fully_shard import FSDPModule
except ImportError:  # pragma: no cover - torch version guard
    FSDPModule = None


def _get_actor_cls():
    """Lazy import to avoid circular dep: molt.models.actor imports from this package."""
    from molt.models import Actor

    return Actor


class FsdpStrategy:
    """FSDP2 + TP/CP/SP/EP backend using NeMo AutoModel.

    Mirrors DeepspeedStrategy's public surface so trainers are agnostic to the
    backend. The model is built and parallelized via AutoModel's official
    entry point ``NeMoAutoModelForCausalLM.from_pretrained`` inside ``Actor``;
    this strategy only handles distributed setup, optimizer/scheduler
    construction, the train-step (loss backward, grad clip, optimizer step),
    collectives, and checkpointing. Grad norm / clip are imported directly
    from ``nemo_automodel.components.distributed.grad_utils``.
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
        # CPU-offload level (--fsdp.offload): none / optimizer / full. A nested
        # progression, not orthogonal — 'full' (FSDP2 CPUOffloadPolicy) streams params to
        # CPU and the optimizer follows; 'optimizer' keeps params on GPU and runs only the
        # AdamW step on CPU (MoE-safe; see optimizer_offload.py). So a single level picks
        # at most one, and there is nothing to exclude.
        offload = getattr(fsdp, "offload", "none")
        self.cpu_offload = offload == "full"
        self.offload_optimizer = offload == "optimizer"
        # The 'optimizer' level runs the AdamW step on CPU (params stay on GPU); see
        # molt/trainer/fsdp/optimizer_offload.py. 'full' is FSDP2 CPUOffloadPolicy below.
        self._optimizer_offloader = CpuOptimizerOffloader() if self.offload_optimizer else None
        # SP is OFF by default (opt in via --fsdp.sequence_parallel) — matches the
        # AutoModel omni / Qwen3.5-MoE recipes, and avoids the _NormPartial 2D
        # TP+FSDP weight-load hang on the HF-fallback path.
        self.sequence_parallel = bool(getattr(fsdp, "sequence_parallel", False))

        self.world_size: int = 1
        self.device_mesh = None
        self.moe_mesh = None
        self.dp_size = 1
        self.dp_cp_size = 1
        self.accumulated_gradient: int = 1
        self._last_grad_norm: float = 0.0
        self.time_steps = defaultdict(int)
        self._max_norm_by_optimizer = {}
        # On-disk checkpoint I/O (save/load/HF-export) lives in its own subsystem;
        # the public save_model/save_ckpt/load_ckpt methods below delegate to it.
        self.checkpoint = CheckpointManager(self)

    # ProcessGroup / DeviceMesh aren't picklable. `datasets.map(self.process_data,
    # num_proc>1)` indirectly pickles the strategy via the dataset's bound method;
    # drop the distributed handles so the workers spawn cleanly. Workers don't need
    # them; they only run pure-CPU data preprocessing.
    _UNPICKLABLE_ATTRS = ("device_mesh", "moe_mesh", "distributed_config")

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k not in self._UNPICKLABLE_ATTRS}

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k in self._UNPICKLABLE_ATTRS:
            self.__dict__.setdefault(k, None)

    def _get_automodel_mesh(self, name: str, required: bool = False):
        if self.device_mesh is None:
            return None

        try:
            from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
        except ImportError:
            get_flat_mesh = None

        try:
            if get_flat_mesh is not None:
                return get_flat_mesh(self.device_mesh, name)
            return self.device_mesh[name]
        except (KeyError, RuntimeError, AttributeError):
            if required:
                raise
            return None

    def _get_automodel_group(self, name: str):
        mesh = self._get_automodel_mesh(name, required=self.device_mesh is not None)
        return mesh.get_group() if mesh is not None else None

    def _get_dp_group(self, include_cp: bool = False):
        name = "dp_cp" if include_cp and self.cp_size > 1 else "dp"
        return self._get_automodel_group(name)

    def _get_dp_group_size(self, include_cp: bool = False) -> int:
        group = self._get_dp_group(include_cp=include_cp)
        if group is None:
            return dist.get_world_size() if dist.is_initialized() else 1
        return dist.get_world_size(group=group)

    def _get_automodel_rank(self, name: str) -> int:
        mesh = self._get_automodel_mesh(name, required=self.device_mesh is not None)
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
            backend = "cuda:nccl,cpu:gloo" if self.cpu_offload else "nccl"
            dist.init_process_group(backend=backend, timeout=timeout)

        self.world_size = dist.get_world_size()
        if self.world_size == 1 and self.cpu_offload:
            raise NotImplementedError(
                "CPU offload is not supported by AutoModel/FSDP2 on a single rank; "
                "set --fsdp.offload to none/optimizer or launch with more than one rank."
            )
        if self.pp_size > 1:
            raise NotImplementedError("Molt trainers are not pipeline-parallel aware yet; set --fsdp.pp_size 1")

        # 015b0f621 moved MoEParallelizerConfig moe.config -> distributed.config
        # (same fields: mp_policy, ignore_router_for_ac).
        from nemo_automodel.components.distributed.config import FSDP2Config, MoEParallelizerConfig
        from nemo_automodel.components.distributed.mesh import ParallelismSizes
        from nemo_automodel.components.distributed.mesh_utils import _create_device_meshes

        # Allow actor/ref TP embedding calls to reuse equivalent vocab masks.
        if self.tp_size > 1:
            try:
                from torch.distributed.tensor._ops import _mask_buffer
            except ImportError:
                _mask_buffer = None
            if _mask_buffer is not None and not getattr(_mask_buffer.MaskBuffer, "_orlhf_patched", False):

                def _safe_materialize(self, mask):
                    self.data = mask
                    self.refcount += 1

                _mask_buffer.MaskBuffer.materialize_mask = _safe_materialize
                _mask_buffer.MaskBuffer._orlhf_patched = True

        from molt.utils.utils import convert_to_torch_dtype

        torch_dtype = convert_to_torch_dtype(self.param_dtype)
        # Match the FSDP2 baseline: params/forward in the requested dtype and
        # reduce-scatter in fp32. Do not force module outputs to fp32 here,
        # because policy/value losses should see the same dtype behavior as
        # the DeepSpeed and PR #1176 paths.
        mp_policy = (
            None
            if torch_dtype == torch.float32
            else MixedPrecisionPolicy(
                param_dtype=torch_dtype,
                reduce_dtype=torch.float32,
                cast_forward_inputs=True,
            )
        )
        # Public attribute: `Actor` reads it as `strategy.distributed_config`
        # and forwards to `NeMoAutoModelForCausalLM.from_pretrained`.
        #
        # Activation checkpointing is driven HERE, through FSDP2Config — this is
        # the switch FSDP2Manager reads (fsdp2.py: self.activation_checkpointing
        # = config.activation_checkpointing) for dense / EP=1 / HF-fallback
        # models, where parallelize() applies AC via fsdp2_strategy_parallelize.
        # The from_pretrained(activation_checkpointing=) kwarg the Actor also
        # passes is DEAD on that path (it only reaches the EP MoE parallelizer,
        # built only for ep_size>1); leaving this field at its default False
        # meant any dense model — and every HF-fallback model — trained with NO
        # activation checkpointing and OOM'd on long sequences. For ep_size>1
        # the EP parallelizer takes its own AC from that kwarg, so setting this
        # field is a no-op there (MoE runs unchanged). Single source of truth:
        # --actor.gradient_checkpoint (RL) / --model.gradient_checkpoint (SFT).
        _actor_cfg = getattr(self.args, "actor", None)
        _model_cfg = getattr(self.args, "model", None)
        activation_checkpointing = resolve_ac_mode(
            getattr(_actor_cfg, "gradient_checkpoint", False) or getattr(_model_cfg, "gradient_checkpoint", False)
        )
        self.distributed_config = FSDP2Config(
            sequence_parallel=self.sequence_parallel,
            mp_policy=mp_policy,
            offload_policy=CPUOffloadPolicy(pin_memory=False) if self.cpu_offload else None,
            activation_checkpointing=activation_checkpointing,
            # defer_fsdp_grad_sync=False: every microbatch reduce-scatters and
            # accumulates its gradient into .grad (sync is always on; see
            # backward()). The accumulation window is realized by deferring
            # optimizer_step, not by skipping sync — so grads stay materialized
            # for clipping and logging.
            defer_fsdp_grad_sync=False,
        )
        # MoE-specific parallelization config: required by AutoModel when
        # ep_size > 1 (raises 'NoneType has no to_dict' otherwise).
        # ignore_router_for_ac=True → selective activation checkpointing that
        # MUST_SAVEs the router projection so the topk routing decision is NOT
        # recomputed in backward. Without it, plain AC recomputes the router and
        # FSDP2's mixed-precision input cast isn't replayed bit-identically, so a
        # near-tie token re-routes on recompute → per-expert token counts shift
        # → grouped-GEMM tensor shapes drift ±1 → CheckpointError. Saving the
        # routing decision freezes those shapes (the Automodel analog of
        # Megatron's higher-precision/stabilized router).
        # reshard_after_forward: free the all-gathered params/experts after the
        # forward and re-gather them in backward — trades one extra backward
        # all-gather for a lower activation-phase peak. AutoModel's MoE EP path
        # (moe/parallelizer.py apply_fsdp) defaults this to False (gather-and-hold,
        # throughput-favoring) for EVERY module incl. experts, UNLIKE its dense
        # path which reshards all-but-last. NeMo-RL exposes it the same way: an
        # opt-in `dtensor_cfg.moe_parallelizer.reshard_after_forward` (NotRequired,
        # falls back to AutoModel's False; models/policy/__init__.py). We mirror
        # that — default OFF (bit-identical to today), opt in via env for
        # memory-bound runs (omni3 32K/CP8). MUST smoke-test under deepep+AC before
        # trusting it: re-gathering experts in backward must not perturb the AC
        # recompute (CheckpointError) and logprobs_diff must stay 0.
        _reshard_after_fwd = os.environ.get("MOLT_MOE_RESHARD_AFTER_FWD", "0") == "1"
        self.moe_config = (
            MoEParallelizerConfig(
                mp_policy=mp_policy,
                ignore_router_for_ac=True,
                reshard_after_forward=_reshard_after_fwd,
            )
            if self.ep_size > 1
            else None
        )

        # AutoModel 015b0f621 renamed create_device_mesh -> _create_device_meshes
        # and bundles the per-dim sizes into ParallelismSizes (dp inferred). Return
        # is still (device_mesh, moe_mesh). Mirrors MeshContext.build (mesh.py).
        self.device_mesh, self.moe_mesh = _create_device_meshes(
            self.distributed_config,
            ParallelismSizes(
                tp_size=self.tp_size,
                pp_size=self.pp_size,
                cp_size=self.cp_size,
                ep_size=self.ep_size,
            ),
            world_size=self.world_size,
        )

        # init_device_mesh's sub-process-groups (the CP all-to-all and the
        # EP/ep_shard reduce-scatter — the latter spans nodes under PACK) inherit
        # NCCL's 600s default watchdog, NOT the longer `timeout` we pass to
        # init_process_group for the world group. On the compile-bound cold first
        # step a legitimate cross-node collective wait can approach 600s and trip
        # the watchdog -> SIGABRT (the multi-node DP2 hang, job 4788092). Raise
        # every mesh sub-group's timeout to the world value so a slow-but-
        # progressing step waits instead of aborting. The real fix is balancing
        # the per-microbatch load (balance_experiences sorts by length so DP ranks
        # reach each cross-node collective together); this is the safety net for a
        # residual single-outlier trajectory.
        from torch.distributed.distributed_c10d import _set_pg_timeout

        _seen_pg = set()
        for _mesh in (self.device_mesh, self.moe_mesh):
            if _mesh is None:
                continue
            for _pg in _mesh.get_all_groups():
                if _pg is not None and id(_pg) not in _seen_pg:
                    _seen_pg.add(id(_pg))
                    _set_pg_timeout(timeout, _pg)

        # AutoModel's FSDP2 mesh exposes flattened "dp" for data loading and
        # "dp_cp" for FSDP reduce-scatter. Gradient accumulation is based on
        # DP only; CP ranks share samples and split sequence work.
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
                assert len(arg) == 2, f"prepare() tuple must be (model, cfg); got len={len(arg)}"
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
            if self.offload_optimizer:
                raise NotImplementedError(
                    "--fsdp.offload optimizer supports AdamW only; Muon's Newton-Schulz "
                    "iterations are impractical on CPU. Use --optim adam, or --fsdp.offload none."
                )
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
        self._max_norm_by_optimizer[id(optimizer)] = cfg.get("max_norm", self.max_norm)

        scheduler_steps = cfg["scheduler_steps"]
        scheduler = get_scheduler(
            cfg.get("lr_scheduler", "constant"),
            optimizer,
            num_warmup_steps=math.ceil(scheduler_steps * cfg.get("lr_warmup_ratio", 0.03)),
            num_training_steps=scheduler_steps,
            scheduler_specific_kwargs={"min_lr_rate": cfg.get("min_lr_ratio", 0.1)},
        )
        return model, optimizer, scheduler

    # ---------------------------------------------------------------- step loop

    @staticmethod
    def _set_fsdp_backward_sync(model: nn.Module, sync: bool) -> None:
        if FSDPModule is None:
            return
        fsdp_modules = [module for module in model.modules() if isinstance(module, FSDPModule)]
        if not fsdp_modules:
            return
        # Set the sync/reshard flags on EVERY FSDP root, not just the first.
        # With the DeepEP MoE dispatcher (EP>1) the experts form a SEPARATE FSDP
        # root from the dense backbone, so setting flags only on fsdp_modules[0]
        # left the experts root at its default. In the RL path (gradient
        # accumulation with sync=False on non-last microbatches and no
        # per-microbatch loss scaling) the experts root then kept syncing every
        # microbatch out of step with the deferred dense root, leaving the
        # expert grads effectively unreduced across the window and inflating the
        # grad-norm by ~grad_acc x. Looping covers both roots; in the SFT path
        # (sync always True) it is a no-op, which is why SFT validated clean on
        # the single-root code.
        for fsdp_module in fsdp_modules:
            fsdp_module.set_is_last_backward(sync)
            fsdp_module.set_reshard_after_backward(sync)
            fsdp_module.set_requires_gradient_sync(sync)

    def backward(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer: optim.Optimizer,
        name: str = "model",
        accumulate: bool = True,
        **kwargs,
    ) -> None:
        unwrapped = self._unwrap_model(model)
        if accumulate and self.accumulated_gradient > 1:
            if kwargs.get("scale_loss_by_accumulation", True):
                loss = loss / self.accumulated_gradient
        # Context-parallel gradient compensation.
        # The trainers normalize the loss with the `token-mean` convention
        # scaled by *dp_size* (global tokens reduced over DP only). That keeps
        # the *reported* loss correct — it is detached before this point and
        # logged via all_reduce(mean) over the world — and was the right grad
        # scaling pre-CP, when FSDP's grad-averaging group equaled dp_size.
        # With CP, AutoModel shards/reduces parameter grads over `dp_shard_cp`,
        # so FSDP averages grads over dp_cp = dp_size * cp_size. The CP loss
        # gather (`cp_dtensor_full_sequence` -> DTensor.full_tensor()) makes
        # every CP rank compute the *full* sequence loss, and its autograd
        # *slices* the gradient back to each rank's sequence shard (verified:
        # full_tensor() backward does not sum across CP). Net effect: without
        # this factor the gradient is cp_size x too small (effective LR / cp).
        # Multiply only the backward loss by cp_size so the post-FSDP gradient
        # matches AutoModel's own `(loss * dp_cp_size).backward()` recipe; the
        # detached reported loss is untouched. No-op when cp_size == 1.
        if self.cp_size > 1:
            loss = loss * self.cp_size
        sync_gradients = kwargs.get("sync_gradients", True)
        self._set_fsdp_backward_sync(unwrapped, sync_gradients)
        if self.moe_mesh is not None:
            try:
                from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
            except ImportError:
                if self.is_rank_0():
                    print("[MoE] MoEAuxLossAutoScaler import failed; aux-loss scaling skipped.")
            else:
                MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
                    float(getattr(self, "dp_cp_size", self.dp_size)),
                    device=loss.device,
                )
        loss.backward()

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name: str = "model",
        accumulate: bool = True,
        **kwargs,
    ) -> None:
        # Skip the optimizer step until the last micro-batch in the accum window.
        key = f"step_{name}"
        if accumulate:
            self.time_steps[key] += 1
            if self.time_steps[key] % self.accumulated_gradient != 0:
                return

        model = self._unwrap_model(model)
        params = [p for p in model.parameters() if p.grad is not None]
        self._last_grad_norm = 0.0
        # Clip/scale only when there are grads to act on; the optimizer tail
        # below runs either way (an empty step is a no-op).
        if params:
            self._maybe_debug_grad_stats(model, name)
            max_norm = self._max_norm_by_optimizer.get(id(optimizer), self.max_norm)
            clip_norm = max_norm if max_norm and max_norm > 0 else None
            if clip_norm is not None or self.moe_mesh is not None:
                from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

                self._last_grad_norm = float(
                    scale_grads_and_clip_grad_norm(
                        clip_norm,
                        [model],
                        pp_enabled=False,
                        device_mesh=self.device_mesh,
                        moe_mesh=self.moe_mesh,
                        ep_axis_name=(
                            "ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None
                        ),
                        foreach=False,
                        num_label_tokens=None,
                        dp_group_size=getattr(self, "dp_cp_size", self.dp_size),
                    )
                )
        if self._optimizer_offloader is not None:
            self._optimizer_offloader.step(optimizer, params)  # CPU AdamW (params stay on GPU)
        else:
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    def offload_moments_to_cpu(self, optimizer: optim.Optimizer) -> None:
        """Page the Adam moments back to CPU after a checkpoint resume (DCP restores them
        onto the model param's GPU device). Called from load_ckpt before the first
        forward; a no-op unless the optimizer is CPU-offloaded."""
        if self._optimizer_offloader is not None:
            self._optimizer_offloader.moments_to_cpu(optimizer)

    def sync_replicated_grads(self, params) -> None:
        """Mean-all-reduce gradients of replicated (non-FSDP-wrapped) params over the
        data-parallel(+CP) group.

        FSDP2 only reduces gradients of params inside its wrapped modules. A module
        added after wrapping (e.g. the critic's scalar value head) is replicated and
        gets a *local* gradient per rank, so it must be averaged over the same
        ``dp_cp`` group FSDP uses to match the wrapped params' effective gradient.
        Call it right before ``optimizer_step`` on the relevant optimizer-step
        microbatch.

        Invariant: molt builds a *flat* FSDP2 data-parallel mesh (``_create_device_meshes``
        infers a single ``dp`` shard dim — no ``dp_replicate``/HSDP path), so ``dp_cp``
        is exactly the ``dp_shard_cp`` group FSDP reduces the wrapped params over, and
        this mean matches the backbone's effective gradient. If HSDP (``dp_replicate>1``)
        is ever added, revisit: a fully-replicated head must still reduce over the full
        ``dp_replicate × dp_shard × cp`` set, which ``dp_cp`` must then continue to span.
        """
        group = self._get_dp_group(include_cp=True)
        if group is None:
            return
        world = dist.get_world_size(group=group)
        if world == 1:
            return
        for p in params:
            if p.grad is None:
                continue
            grad = local_shard(p.grad)
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=group)
            grad.div_(world)

    def get_grad_norm(self, model: nn.Module) -> float:
        return self._last_grad_norm

    def _maybe_debug_grad_stats(self, model: nn.Module, optim_name: str) -> None:
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
            local_grad = local_shard(grad).detach()
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

        For CP, call this before ``make_cp_batch_and_ctx`` while each CP rank
        still sees the full local sequence. Token denominators are reduced over
        DP only because CP ranks share samples.
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

        MFU is delegated to AutoModel's ``AutoMFU`` over the GLOBAL batch and ALL
        GPUs (so it is parallelism-invariant); global sequence + token counts reuse
        ``global_token_count`` (DP-mesh all-reduce), and the mean length feeds
        AutoMFU's rectangular ``(batch, seq_len)`` formula. ``mfu`` is the caller's
        ``AutoMFU`` instance (None -> memory only). ``prefix`` namespaces the keys so
        colocated models (actor vs critic) report under distinct names instead of
        overwriting each other on the last-wins status merge — and, since they are
        separate processes sharing the GPU, their peak-memory numbers ADD up to the
        true device pressure.
        """
        metrics = {}
        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            metrics[f"{prefix}gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated(dev) / 1024**3
            metrics[f"{prefix}gpu_mem_reserved_gb"] = torch.cuda.max_memory_reserved(dev) / 1024**3
        if mfu is None or seconds <= 0.0:
            return metrics

        # True step time is the slowest rank's; reduce to max so MFU is the same
        # aggregate number on every rank (the per-worker status merge is last-wins).
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
        if isinstance(model, _get_actor_cls()):
            return self._unwrap_model(model.model)
        if hasattr(model, "get_base_model_for_fsdp"):
            return model.get_base_model_for_fsdp()
        if hasattr(model, "module"):
            return model.module
        return model

    # ---------------------------------------------------------------- I/O
    # All on-disk checkpoint I/O lives in CheckpointManager (self.checkpoint);
    # these thin wrappers preserve the historical strategy.save_*/load_* surface
    # that trainers and CLIs already call.

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

    def load_ckpt(self, model: nn.Module, ckpt_path: str, optimizer=None, scheduler=None, **kwargs):
        return self.checkpoint.load_ckpt(model, ckpt_path, optimizer=optimizer, scheduler=scheduler, **kwargs)
