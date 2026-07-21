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
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

import asyncio
import dataclasses
import inspect
import os
from typing import Any, List, Optional

import ray
import vllm
from packaging import version
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from molt.trainer.placement import get_bundle_indices, ray_noset_visible_devices

_MIN_VLLM_VERSION = version.parse("0.21.0")


def _assert_supported_vllm():
    if version.parse(vllm.__version__) < _MIN_VLLM_VERSION:
        raise RuntimeError(f"vLLM >= 0.21.0 is required, got {vllm.__version__}.")


def _format_ray_gpu_ids(gpu_ids) -> str:
    formatted = []
    for gpu_id in gpu_ids:
        try:
            value = float(gpu_id)
        except (TypeError, ValueError):
            formatted.append(str(gpu_id))
            continue
        formatted.append(str(int(value)) if value.is_integer() else str(gpu_id))
    return ",".join(formatted)


def _vllm_worker_num_gpus(backend: Optional[str], actor_num_gpus: int) -> int:
    # With vLLM's ray executor the top-level Ray actor is CPU-only for TP>1,
    # while each child worker still needs one GPU from its placement-group
    # bundle. Keep those resource concepts separate.
    return 1 if backend == "ray" else actor_num_gpus


def _async_engine_arg_names():
    async_engine_args = vllm.AsyncEngineArgs
    if dataclasses.is_dataclass(async_engine_args):
        return {field.name for field in dataclasses.fields(async_engine_args) if field.init}

    try:
        signature = inspect.signature(async_engine_args)
    except (TypeError, ValueError):
        return None

    parameters = signature.parameters.values()
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters):
        return None
    return {
        name
        for name, param in signature.parameters.items()
        if name != "self"
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }


def _filter_vllm_engine_kwargs(kwargs: dict) -> dict:
    supported_args = _async_engine_arg_names()
    if supported_args is None:
        return dict(kwargs)

    filtered = {key: value for key, value in kwargs.items() if key in supported_args}
    dropped = sorted(set(kwargs) - set(filtered))
    if dropped:
        print(
            "[vLLM] Skipping unsupported AsyncEngineArgs option(s): " + ", ".join(dropped),
            flush=True,
        )
    return filtered


@ray.remote
class RolloutRayActor:
    """Async vLLM-backed actor that exposes generation utilities."""

    async def __init__(self, *args, bundle_indices: list = None, **kwargs):
        backend = kwargs.get("distributed_executor_backend")
        num_gpus = kwargs.pop("num_gpus")
        self._configure_device_env(
            backend=backend,
            bundle_indices=bundle_indices,
            worker_num_gpus=kwargs.pop("worker_num_gpus", _vllm_worker_num_gpus(backend, num_gpus)),
        )
        self._configure_vllm_env(kwargs.pop("full_determinism", False))

        self.kwargs = kwargs

        engine_args = vllm.AsyncEngineArgs(*args, **_filter_vllm_engine_kwargs(self.kwargs))
        self.llm = vllm.AsyncLLMEngine.from_engine_args(engine_args)
        print("vLLM AsyncLLMEngine constructed", flush=True)

    async def ready(self) -> bool:
        """Confirm Ray actor construction has completed."""
        return True

    async def serve_openai(self, host: str = "0.0.0.0", port: int = 0) -> str:
        """Mount vLLM's OpenAI API server on THIS engine and return its URL.

        The router (vllm-router) fronts these per-engine servers for generation; we keep
        ``self.llm`` in the actor so weight sync (``collective_rpc`` over the NCCL group)
        is untouched and bypasses the router. Uvicorn runs on the actor's OWN event loop
        (shared with the AsyncLLM engine — no extra process), same pattern as the chat
        server. Client sets the token-in/out flags per request (``prompt=[ids]``,
        ``return_token_ids``, ``logprobs``); the server just serves ``self.llm``.
        """
        import uvicorn
        from vllm.entrypoints.openai import api_server as _api
        from vllm.entrypoints.openai.cli_args import make_arg_parser

        try:  # location moved across vLLM versions (utils.argparse_utils -> entrypoints.utils)
            from vllm.utils.argparse_utils import FlexibleArgumentParser
        except ImportError:
            from vllm.entrypoints.utils import FlexibleArgumentParser

        args = make_arg_parser(FlexibleArgumentParser()).parse_args([])
        args.model = self.kwargs.get("model")
        args.served_model_name = ["policy"]  # clients request model="policy"
        args.host, args.port = host, port

        supported_tasks = await self.llm.get_supported_tasks()
        model_config = self.llm.model_config
        # build_app / init_app_state signatures vary across vLLM versions; try the
        # newer (args, supported_tasks, model_config) form, fall back to (args).
        try:
            app = _api.build_app(args, supported_tasks, model_config)
        except TypeError:
            app = _api.build_app(args)
        try:  # supported_tasks arg was added in a later vLLM; fall back to the 3-arg form
            await _api.init_app_state(self.llm, app.state, args, supported_tasks)
        except TypeError:
            await _api.init_app_state(self.llm, app.state, args)

        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        # vLLM's error handlers (e.g. the disagg /inference/v1/generate path) reach for
        # ``app.state.server`` to terminate on a fatal engine error; the standard run_server
        # launcher sets it, but we mount uvicorn ourselves, so set it here too (else a real
        # GenerationError gets masked by "'State' object has no attribute 'server'").
        app.state.server = server
        self._server_task = asyncio.create_task(server.serve())
        while not server.started:  # wait for the OS-assigned port to bind
            await asyncio.sleep(0.1)
        actual_port = server.servers[0].sockets[0].getsockname()[1]
        self._server_url = f"http://{ray.util.get_node_ip_address()}:{actual_port}"
        print(f"vLLM OpenAI server up at {self._server_url}", flush=True)
        return self._server_url

    def _configure_device_env(self, backend, bundle_indices, worker_num_gpus):
        if backend == "ray":
            # vLLM's Ray executor launches child workers from the placement
            # group bundles, so the parent rollout actor must not inherit a
            # narrowed device list from Ray.
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        elif ray_noset_visible_devices():
            # We need to set CUDA_VISIBLE_DEVICES to the ray assigned GPU
            # when the distributed_executor_backend is not ray and Ray's
            # CUDA visible-device management is disabled.
            visible_devices = _format_ray_gpu_ids(ray.get_gpu_ids())
            if visible_devices:
                os.environ["CUDA_VISIBLE_DEVICES"] = visible_devices

        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(worker_num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

    def _configure_vllm_env(self, full_determinism: bool):
        _assert_supported_vllm()
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        if full_determinism:
            # https://github.com/vllm-project/vllm/blob/effc5d24fae10b29996256eb7a88668ff7941aed/examples/offline_inference/reproduciblity.py#L11
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

        if not os.environ.get("RAY_ADDRESS"):
            from ray._private.worker import global_worker

            os.environ["RAY_ADDRESS"] = global_worker.gcs_client.address

    async def init_process_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return await self.llm.collective_rpc(
            "init_process_group",
            args=(master_address, master_port, rank_offset, world_size, group_name, backend),
        )

    async def update_weights_packed(self, metas):
        """Receive a single packed broadcast carrying many weights.

        ``metas`` is a list of ``(name, dtype, shape)``. Producer (rank 0 in the
        trainer) cats matching tensors into one buffer in the same order.
        """
        result = await self.llm.collective_rpc(
            "update_weights_packed",
            args=(metas,),
        )
        return result

    async def reset_weight_update_check(self):
        await self.llm.collective_rpc("reset_weight_update_check")

    async def weight_update_missing(self):
        # Union the per-worker missing-param lists (weights are TP/EP-sharded across workers).
        results = await self.llm.collective_rpc("weight_update_missing")
        missing = set()
        for r in results or []:
            missing.update(r or [])
        return sorted(missing)

    async def pause_generation(self):
        await self.llm.pause_generation(mode="keep")

    async def resume_generation(self):
        await self.llm.resume_generation()

    async def reset_prefix_cache(self):
        """Invalidate the prefix KV cache.

        Called after a weight broadcast: stale prefix-cache entries would otherwise
        replay logits generated by the previous policy and silently corrupt rollout
        logprobs. The trainer skips this when prefix caching is off (caller-gated).
        """
        await self.llm.reset_prefix_cache()


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    full_determinism: bool,
    enforce_eager: bool,
    max_model_len: int,
    gpu_memory_utilization=0.95,
    # vLLM V1 (default engine) computes logprobs from the RAW logits, BEFORE
    # temperature/penalty scaling (LogprobsMode default "raw_logprobs"; see
    # v1/sample/sampler.py). Our actor recomputes log-probs at the rollout
    # temperature, so raw rollout log-probs would mismatch by the temperature
    # factor and bias the TIS / vLLM-IS correction whenever rollout temperature
    # != 1.0. "processed_logprobs" makes vLLM return post-temperature log-probs
    # that align with the actor.
    # Pass None to fall back to vLLM's raw behavior.
    logprobs_mode="processed_logprobs",
    max_images_per_prompt: int = 0,
    mm_encoder_attn_backend: Optional[str] = None,
    gdn_prefill_backend: Optional[str] = None,
    attention_backend: Optional[str] = None,
    mamba_ssm_cache_dtype: Optional[str] = None,
    distributed_executor_backend: Optional[str] = None,
    enable_expert_parallel: bool = False,
    # vLLM 0.21 EngineArgs accept None and resolve internally to its own default
    # (prefix_caching True, chunked_prefill True, async_scheduling True for
    # mp/uniproc executors). We override only `enable_prefix_caching` to False:
    # RL weight broadcasts invalidate prefix-cached prefixes, and the user
    # opts in via `--vllm.enable_prefix_caching` once they've validated logprob
    # alignment on their recipe (broadcast_to_vllm calls reset_prefix_cache).
    enable_prefix_caching: bool = False,
    enable_chunked_prefill: Optional[bool] = None,
    max_num_batched_tokens: Optional[int] = None,
    async_scheduling: Optional[bool] = None,
    decode_context_parallel_size: int = 1,
    dtype: str = "bfloat16",
    block_size: Optional[int] = None,
    mtp_num_speculative_tokens: int = 0,
    enable_return_routed_experts: bool = False,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
):
    """Spin up a set of vLLM Ray actors on a dedicated placement group.

    Async-split topology: vLLM engines run on different GPUs from the FSDP2
    actor (no GPU sharing), so we always allocate a fresh placement group.

    A single engine can span nodes via TP+EP or pipeline parallelism: with the
    ``ray`` executor, a ``tensor_parallel_size`` (or TP*``pipeline_parallel_size``)
    larger than one node's GPU count lays out one worker per single-GPU bundle
    across nodes. Best practice for a giant MoE that overflows a node is
    TP=node-size (NVLink) + PP across nodes; cross-node TP works too but pays an
    all-reduce every layer. vLLM reads
    ``VLLM_RAY_BUNDLE_INDICES`` and sorts the bundles by node itself (see
    vllm/v1/executor/ray_executor.py); ``get_bundle_indices`` hands it a
    node-grouped slice. E.g. Kimi-class MoE on 2x8 GPUs:
    ``tensor_parallel_size=16 --vllm.enable_expert_parallel`` -> EP16
    (EP = TP * DP, DP=1) across both nodes. ``mp``/``uni`` are single-node only
    (a Ray bundle cannot span nodes), so cross-node engines need the ray backend.

    vLLM has no standalone expert-parallel size: ``EP = TP * DP``. To decouple EP
    from TP (DeepSeek-V3-style TP8 attention + EP32 experts) raise
    ``data_parallel_size``; the engine's one actor then owns ``TP * DP`` GPUs and
    vLLM spawns the DP replicas as local subprocesses (mp, single node).
    """
    _assert_supported_vllm()

    vllm_engines = []
    # Backend default: a lone-GPU engine uses uni. Data parallelism uses mp so
    # vLLM spawns the DP replicas as local subprocesses (single node) — the ray
    # DP backend instead allocates its own placement groups and collides with the
    # one we pre-create below. Everything else (cross-node TP+EP, pipeline) uses ray.
    distributed_executor_backend = distributed_executor_backend or (
        "uni"
        if tensor_parallel_size * pipeline_parallel_size * data_parallel_size == 1
        else "mp"
        if data_parallel_size > 1
        else "ray"
    )
    if distributed_executor_backend not in {"uni", "ray", "mp"}:
        raise ValueError(
            "vllm.distributed_executor_backend must be one of {'uni', 'ray', 'mp'}, "
            f"got {distributed_executor_backend!r}."
        )
    if distributed_executor_backend == "uni" and tensor_parallel_size != 1:
        raise ValueError("vLLM backend 'uni' only supports tensor_parallel_size=1; use 'ray' or 'mp' for TP > 1.")
    # Pipeline parallelism (best practice for multi-node giant-MoE serving: TP within
    # a node over NVLink, PP across nodes with cheap point-to-point stage handoff,
    # instead of cross-node TP all-reduce every layer). Each engine spans TP*PP GPUs.
    # PP places layer stages on different bundles, so it needs per-worker single-GPU
    # bundles — the ray executor, same as cross-node TP.
    if pipeline_parallel_size > 1 and distributed_executor_backend != "ray":
        raise ValueError("vLLM pipeline_parallel_size > 1 requires the 'ray' executor backend.")
    # Data parallelism is how a rollout engine gets EP > TP (vLLM defines
    # EP = TP * DP). One actor owns the engine's TP*DP GPUs and vLLM spawns the DP
    # replicas as local subprocesses (mp), so DP is single-node: the ray DP backend
    # builds its own placement groups and collides with the one we pre-create.
    if data_parallel_size > 1 and distributed_executor_backend != "mp":
        raise ValueError(
            "vLLM data_parallel_size > 1 requires the 'mp' executor (single-node); the ray "
            "DP backend allocates its own placement groups and collides with molt's."
        )
    if data_parallel_size > 1 and pipeline_parallel_size > 1:
        raise ValueError(
            "vLLM data_parallel_size > 1 (single-node mp) cannot combine with pipeline_parallel_size > 1."
        )
    engine_world = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
    num_gpus = (
        tensor_parallel_size * data_parallel_size if distributed_executor_backend == "mp" else int(engine_world == 1)
    )
    worker_num_gpus = _vllm_worker_num_gpus(distributed_executor_backend, num_gpus)

    # mp executor: one Ray actor owns `tensor_parallel_size * data_parallel_size`
    # GPUs (vLLM spawns the worker/DP-replica processes itself, single node only).
    # ray executor: one single-GPU bundle per worker, so an engine's TP*PP group
    # can span nodes (cross-node TP+EP, e.g. Kimi-class 2x8, or TP-per-node + PP).
    # GPU is the only exclusive resource to reserve; CPU is shared, so pinning CPU in the engine
    # bundles (and on the actor below) only over-constrains PACK — with many small TP-per-node
    # bundles it can leave the placement group infeasible even when GPUs are free. Reserve GPU only.
    if distributed_executor_backend == "mp":
        gpus_per_engine = tensor_parallel_size * data_parallel_size
        bundles = [{"GPU": gpus_per_engine} for _ in range(num_engines)]
    else:
        bundles = [{"GPU": 1} for _ in range(num_engines * engine_world)]
    shared_pg = placement_group(bundles, strategy="PACK")
    ray.get(shared_pg.ready())

    for i in range(num_engines):
        bundle_indices = None
        if engine_world > 1 and distributed_executor_backend == "ray":
            # Node-grouped slice for engine i (TP*PP GPUs). PACK fills node-by-node,
            # so vLLM keeps each PP stage's TP group on one node (intra-node NVLink
            # all-reduce) and pipelines stages across nodes.
            bundle_indices = get_bundle_indices(shared_pg, i, engine_world)

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=bundle_indices[0] if bundle_indices else i,
        )

        actor_kwargs = {
            "model": pretrain,
            "enforce_eager": enforce_eager,
            "worker_extension_cls": "molt.trainer.vllm.vllm_worker_wrap.WorkerWrap",
            "tensor_parallel_size": tensor_parallel_size,
            "pipeline_parallel_size": pipeline_parallel_size,
            "seed": seed + i,
            "distributed_executor_backend": distributed_executor_backend,
            "max_model_len": max_model_len,
            "enable_prefix_caching": enable_prefix_caching,
            "decode_context_parallel_size": decode_context_parallel_size,
            "dtype": dtype,
            "trust_remote_code": True,
            "full_determinism": full_determinism,
            "gpu_memory_utilization": gpu_memory_utilization,
            "bundle_indices": bundle_indices,
            "num_gpus": num_gpus,
            "worker_num_gpus": worker_num_gpus,
        }

        if max_images_per_prompt > 0:
            actor_kwargs["limit_mm_per_prompt"] = {"image": max_images_per_prompt}
            # Disable vLLM's mm preprocessor cache for VLM rollouts. The cache
            # keys image features by content hash; after each weight broadcast
            # cached entries become stale and the engine asserts on cache-miss
            # (`mm_receiver_cache.get_and_update_item -> Expected a cached item`).
            # Cost is recomputing image features per request — fine since the
            # vision encoder is frozen and cheap relative to LM forward.
            actor_kwargs["mm_processor_cache_gb"] = 0

        # Pass-through only when the caller (or CLI) chose an explicit value, so
        # vLLM's own None→auto resolution (see config/vllm.py:844 for async,
        # config/scheduler.py:84 for chunked_prefill) stays the source of truth
        # for the unset case.
        if enable_chunked_prefill is not None:
            actor_kwargs["enable_chunked_prefill"] = enable_chunked_prefill
        if max_num_batched_tokens is not None:
            # >= max_model_len ⇒ every prefill fits in one scheduler chunk, so a
            # recurrent-state model (Mamba2/GDN) never hands its state across chunk
            # boundaries mid-prefill (a rollout-vs-training logprob drift source).
            actor_kwargs["max_num_batched_tokens"] = max_num_batched_tokens
        if async_scheduling is not None:
            actor_kwargs["async_scheduling"] = async_scheduling

        if mm_encoder_attn_backend:
            actor_kwargs["mm_encoder_attn_backend"] = mm_encoder_attn_backend

        if gdn_prefill_backend:
            actor_kwargs["gdn_prefill_backend"] = gdn_prefill_backend

        if attention_backend:
            # vLLM >=0.20 ignores the legacy VLLM_ATTENTION_BACKEND env var; the
            # backend must be passed via EngineArgs (e.g. "TRITON_ATTN" to
            # avoid AOT-compiled FlashAttention 2 PTX kernels on older drivers).
            actor_kwargs["attention_backend"] = attention_backend

        if block_size:
            # KV cache block size (tokens). MiniMax-M3's MSA sparse attention
            # mandates a 128-token block; vLLM's default (16) leaves no common
            # block size across M3's dense (layers 0-2) + sparse (3-59) attention
            # and raises ValueError("No common block size for 16.") at KV setup.
            actor_kwargs["block_size"] = block_size

        if mamba_ssm_cache_dtype:
            # Recurrent-state models (Mamba2 / GDN hybrids): vLLM defaults the
            # SSM state cache to the model dtype, and a low-precision recurrent
            # cache accumulates rounding error across a long rollout, drifting
            # rollout log-probs from the fp32 training recompute (inflates
            # vllm_kl and over-triggers the seq-mask-TIS filter). Force fp32 to
            # match training — NeMo-RL's hybrid-model recipes set the same.
            # No-op for non-recurrent models (vLLM ignores it).
            actor_kwargs["mamba_ssm_cache_dtype"] = mamba_ssm_cache_dtype

        if enable_expert_parallel:
            # vLLM TP+EP hybrid: non-MoE layers stay TP-sharded across `tensor_parallel_size`
            # ranks; MoE experts are EP-distributed across the same ranks (one group per rank).
            actor_kwargs["enable_expert_parallel"] = True

        if data_parallel_size > 1:
            # vLLM EP = TP * DP: raising DP is the only way a rollout engine gets
            # more expert-parallel groups than its TP degree (DeepSeek-V3-style
            # TP8+DP4 -> EP32). The mp DP backend spawns the replicas as local
            # subprocesses across this actor's TP*DP GPUs (single node).
            actor_kwargs["data_parallel_size"] = data_parallel_size
            actor_kwargs["data_parallel_backend"] = "mp"

        if enable_return_routed_experts:
            # R3: make vLLM return the router's per-token top-k expert ids so the
            # training forward can replay the exact rollout routing (RouterReplay).
            actor_kwargs["enable_return_routed_experts"] = True

        if logprobs_mode:
            # Don't cap max_logprobs: the RL path only requests logprobs=1, but
            # the OAI server may serve external top_logprobs>1; vLLM's default
            # (20) covers both. Capping at 1 would reject those requests.
            actor_kwargs["logprobs_mode"] = logprobs_mode

        # MTP speculative decoding (rollout-side). method="mtp" is required: vLLM
        # then resolves the per-architecture MTP draft from the served target's
        # config (qwen3_5_moe -> qwen3_5_mtp / Qwen3_5MoeMTP, reading
        # mtp_num_hidden_layers; see vllm config/speculative.py) and builds it from
        # the target's own MTP head. A bare num_speculative_tokens would instead
        # default to method="draft_model". Lossless: the target verifies every
        # drafted token via rejection sampling, so the returned logprobs remain the
        # target policy's (unbiased for the RL objective). The draft block is not
        # hot-refreshed by weight-sync (vLLM exposes no such API), but it shares
        # embed_tokens/lm_head with the target — which our broadcast DOES update — so
        # it tracks the policy and acceptance degrades only gradually. We do not
        # train the MTP head (matches verl/NeMo-RL: main-head-only RL loss). vLLM
        # errors at init if the served checkpoint has no MTP head (e.g.
        # Nemotron-Nano-Omni — see README).
        if mtp_num_speculative_tokens and mtp_num_speculative_tokens > 0:
            actor_kwargs["speculative_config"] = {
                "method": "mtp",
                "num_speculative_tokens": mtp_num_speculative_tokens,
            }

        vllm_engines.append(
            RolloutRayActor.options(
                num_cpus=0,  # GPU-only actor: CPU is shared, so reserving it only over-constrains the PG
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(**actor_kwargs)
        )

    # Ray returns actor handles before async actor initialization completes.
    # Block here so FSDP rank 0 does not open the trainer<->vLLM weight-sync
    # TCPStore while vLLM worker collectives are still queued behind startup.
    ray.get([engine.ready.remote() for engine in vllm_engines])

    return vllm_engines


def batch_vllm_engine_call(engines: List[Any], method_name: str, *args, rank_0_only: bool = True, **kwargs):
    """Call the same method on vLLM engines and gather results."""
    import torch

    if torch.distributed.is_initialized() and rank_0_only and torch.distributed.get_rank() != 0:
        return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))
    return ray.get(refs)
