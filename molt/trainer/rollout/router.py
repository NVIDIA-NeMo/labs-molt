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

"""Real vllm-router in front of the engine servers, + the rollout transport over it.

The router load-balances the engines' OpenAI API (default consistent_hash: ``x-session-id`` affinity
so a rollout's render+generate co-locate on one engine); weight sync bypasses it (NCCL to the engines). Everything the runners generate goes over HTTP through the router via
one client, ``RouterGenerateClient.generate(token_ids, sp, mm) -> (RequestOutput, off_policy_len)``:
token-in / token-out over vLLM's ``/inference/v1/generate`` (VLM images are rendered server-side
first, over ``/v1/chat/completions/render``, for their mm features). Both the StepEnvRunner and the
chat server (``_chat_server``) share it. These custom routes survive the router verbatim, unlike the
OpenAI ``/v1/*`` routes whose token_ids the router schema-strips.

``AgentRunnerActor`` is the rollout driver process running those runners against the router; the
trainer round-robins prompts across a list of them (``--rollout.num_runners``).
"""

import asyncio
import base64
import io
import socket
import time
from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import aiohttp
import numpy as np
import ray


@ray.remote(num_cpus=1)
class VllmRouterActor:
    """Runs the real vllm-router (Rust) as a subprocess over the given engine URLs.

    The Rust router's ``start()`` blocks while holding the GIL, so it CANNOT run on a thread inside
    this actor: the daemon thread never gives the GIL back, the actor's MainThread can't finish
    ``Thread.start()``, ``__init__`` never returns, and the actor hangs forever in PENDING_CREATION.
    So it gets its own process (own GIL, blocks freely); the actor just supervises it and reports the
    URL. Fixed port (single-job runs don't collide). Weight sync bypasses the router (NCCL straight
    to the engines)."""

    def __init__(self, worker_urls, *, policy="consistent_hash", port=30000, max_payload_mb=512):
        import subprocess
        import sys

        self._host, self._port = ray.util.get_node_ip_address(), port
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm_router.launch_router",
                "--host",
                self._host,
                "--port",
                str(port),
                "--policy",
                policy,
                "--max-payload-size",
                str(max_payload_mb * 1024 * 1024),  # token-id / embed payloads are large
                "--worker-urls",
                *[str(u) for u in worker_urls],
            ]
        )

    def url(self):
        return f"http://{self._host}:{self._port}"

    def ready(self, timeout_s=180.0):
        """Block until the router port accepts connections (it health-checks its workers first).
        Fail fast if the router subprocess died instead of waiting out the full timeout."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            rc = self._proc.poll()
            if rc is not None:
                raise RuntimeError(f"vLLM router subprocess exited ({rc}) before binding {self._host}:{self._port}")
            with socket.socket() as s:
                s.settimeout(1.0)
                if s.connect_ex((self._host, self._port)) == 0:
                    return self.url()
            time.sleep(2.0)
        raise RuntimeError(f"vLLM router did not come up on {self._host}:{self._port}")


def _decode_routed_experts(blob):
    """R3 routed experts -> ndarray [tokens, moe_layers, topk]. The disagg engine ships a base64 .npy
    (faithful); also accept nested JSON lists for forward-compat."""
    if isinstance(blob, str):
        return np.load(io.BytesIO(base64.b64decode(blob)))
    return np.asarray(blob)


def _inference_sampling_params(sp) -> dict:
    """The ``sampling_params`` object for vLLM's ``/inference/v1/generate`` (vime's shape). Forward
    every knob set on the rollout SamplingParams so nothing is silently dropped over HTTP (only
    non-defaults are sent); ``skip_special_tokens`` is always sent (server defaults True, rollouts
    want raw action text)."""
    fields = {
        "max_tokens": sp.max_tokens,
        "logprobs": 1,
        "temperature": sp.temperature,
        "top_p": sp.top_p,
        "skip_special_tokens": getattr(sp, "skip_special_tokens", True),
    }
    # ALWAYS send top_k: when omitted the server silently falls back to the model's
    # generation_config (Qwen ships top_k=20), diverging sampling from the training
    # config AND renormalizing returned logprobs over the truncated support.
    tk = getattr(sp, "top_k", -1)
    fields["top_k"] = tk if tk not in (0, None) else -1
    if getattr(sp, "min_tokens", 0):
        fields["min_tokens"] = sp.min_tokens
    if getattr(sp, "seed", None) is not None:
        fields["seed"] = sp.seed
    # Penalties and min_p are ALSO sent unconditionally: any field omitted from the
    # request is silently filled from the checkpoint's generation_config, decoupling
    # rollout sampling from the training configuration.
    for name, default in (
        ("repetition_penalty", 1.0),
        ("frequency_penalty", 0.0),
        ("presence_penalty", 0.0),
        ("min_p", 0.0),
    ):
        v = getattr(sp, name, None)
        fields[name] = default if v is None else v
    for name in ("stop", "stop_token_ids"):
        v = getattr(sp, name, None)
        if v:
            fields[name] = list(v)
    return fields


def _image_data_uri(image) -> str:
    """PIL image -> a ``data:`` URI for the render endpoint's ``image_url`` content."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _align_features_to_canonical(features: dict, canonical: list, image_token_id) -> None:
    """Point render's mm features at the image-placeholder run(s) in our canonical HF prompt IN PLACE.

    vLLM splices each image's N vision embeds (from ``kwargs_data``) into a contiguous run of
    ``image_token_id`` tokens. We DISCARD render's own ``mm_placeholders`` and take offset+length from
    the canonical runs: for omni3 render over-counts (274, including the IMG_START/END markers) vs the
    272 embeds, and its offset points AT a marker — trusting it splices the wrong count into the wrong
    span (e.g. length 1 -> 1 embed spliced while the actor forwards all 272 -> train/rollout token
    mismatch). The canonical id-``image_token_id`` runs match the embeds by construction (the same HF
    processor built both the prompt and the training pixel_values). One run per image, left to right."""
    ph = features.get("mm_placeholders")
    if not isinstance(ph, dict) or not ph.get("image"):
        raise ValueError("render features missing image mm_placeholders")
    if image_token_id is None:
        raise ValueError("image_token_id is required to align mm placeholders to the canonical prompt")
    runs, search, n = [], 0, len(canonical)
    for i in range(len(ph["image"])):
        start = search
        while start < n and canonical[start] != image_token_id:
            start += 1
        if start >= n:
            raise ValueError(f"image placeholder token {image_token_id} not found in canonical prompt (image {i})")
        end = start + 1
        while end < n and canonical[end] == image_token_id:
            end += 1
        runs.append({"offset": start, "length": end - start})
        search = end
    ph["image"] = runs


class RouterGenerateClient:
    """The unified rollout transport over the router.

    ``generate(token_ids, sp, mm) -> (RequestOutput, off_policy_len)`` is the token-in / token-out
    path for BOTH the StepEnvRunner and the chat server, over vLLM's ``/inference/v1/generate`` (vime's
    transport). It carries token_ids + vLLM-shaped logprobs + R3 routed_experts (base64 npy, unified
    prompt+gen array). VLM: it first ``/v1/chat/completions/render``s the image(s) SERVER-SIDE to get
    vLLM's mm ``features`` (pixel tensors), realigns those placeholders onto our canonical HF prompt
    ids, and sends them with the generate — the image is embedded by STOCK server-side mm processing
    (no vLLM source patch). Both routes are custom, so the vllm-router forwards them VERBATIM (unlike
    the OpenAI ``/v1/*`` routes, whose token_ids the router schema-strips). It also carries the shared
    aiohttp session (``.http``)."""

    def __init__(self, http_client, *, model_name="policy", image_token_id=None):
        self.http = http_client
        self.model_name = model_name
        self.image_token_id = image_token_id  # HF processor's image placeholder id (VLM realign)

    async def generate(self, prompt_token_ids, sampling_params, multi_modal_data=None, session_id=None):
        prompt = list(prompt_token_ids)
        # ``session_id`` pins every request it tags to ONE engine (``x-session-id`` + the router's
        # consistent_hash policy). The runner passes ONE id per rollout, so all of a rollout's turns —
        # and each turn's render + generate — co-locate: render's mm features resolve on the engine that
        # generates, and (with prefix caching on) the multi-turn KV prefix stays warm. vime/slime pin
        # the same way, per sample. Falls back to a per-call id when the caller passes none.
        sid = session_id or uuid4().hex
        features = None
        if multi_modal_data and multi_modal_data.get("image"):
            rendered = await self._render(multi_modal_data["image"], sid)
            features = rendered.get("features")
            if features is None:
                raise RuntimeError("/v1/chat/completions/render returned no features for an image prompt.")
            _align_features_to_canonical(features, prompt, self.image_token_id)
        return await self._generate_inference(prompt, sampling_params, features, sid)

    async def _post(self, path, payload, session_id, *, retries=3):
        """POST to a router route (``model`` + the routing-affinity header injected), return parsed JSON.

        Retries transient failures with linear backoff — a 5xx or a refused/dropped connection while
        the router or an engine is still warming up (its port binds before its workers pass health
        checks) or is briefly overloaded — so a hiccup costs a retry, not a dropped rollout (vime/slime
        retry the same way). A 4xx is a real client bug and a read timeout is a wedged engine (caught by
        the session's ``sock_read``): both fail fast rather than burn the retry budget."""
        headers = {"x-session-id": session_id} if session_id else None
        body = {"model": self.model_name, **payload}
        for attempt in range(retries):
            try:
                async with self.http.post(path, json=body, headers=headers) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError) as e:
                fatal = isinstance(e, aiohttp.ClientResponseError) and e.status < 500  # 4xx = real bug
                if fatal or attempt == retries - 1:
                    raise
            await asyncio.sleep(attempt + 1)

    async def _render(self, images, session_id):
        """POST the image(s) to the stock ``/v1/chat/completions/render`` route (server-side chat
        template + mm processing) and return its ``{token_ids, features}``. We use ONLY ``features``
        (vLLM's mm pixel tensors -> N vision embeds); the placeholder ranges are realigned to our
        canonical prompt by the caller. The router forwards this custom route verbatim."""
        content = [{"type": "image_url", "image_url": {"url": _image_data_uri(im)}} for im in images]
        body = {"messages": [{"role": "user", "content": content}]}
        return await self._post("/v1/chat/completions/render", body, session_id)

    async def _generate_inference(self, prompt_token_ids, sampling_params, features, session_id):
        # R3 routed_experts are enabled ENGINE-side (create_vllm_engines(enable_return_routed_experts)
        # <- --train.routing_replay); the engine then returns them on the choice, which we decode below.
        body = {"token_ids": list(prompt_token_ids), "sampling_params": _inference_sampling_params(sampling_params)}
        if features is not None:
            body["features"] = features
        out = await self._post("/inference/v1/generate", body, session_id)
        c = out["choices"][0]
        ids = list(c.get("token_ids") or [])
        # logprobs=1 -> choice.logprobs.content[i] = {"token": "token_id:<id>", "logprob": ...}; a
        # non-empty completion MUST carry them (silent zero-fill corrupts the IS correction).
        content = (c.get("logprobs") or {}).get("content") or []
        if ids and not content:
            raise RuntimeError("/inference/v1/generate returned no logprobs.content despite logprobs=1.")
        if content and len(content) != len(ids):
            raise RuntimeError(f"logprobs.content count {len(content)} != completion tokens {len(ids)}.")
        if any("logprob" not in item for item in content):
            raise RuntimeError("/inference/v1/generate returned a logprobs.content entry without a logprob field.")
        logprobs = [{t: SimpleNamespace(logprob=float(item["logprob"]))} for t, item in zip(ids, content)]
        fr = c.get("finish_reason")
        finish_reason = fr.get("type") if isinstance(fr, dict) else (fr or "stop")
        # routed_experts is the UNIFIED full-sequence [tokens,layer,topk] npy (prompt+gen); absorb_routing
        # lays it down by absolute position from 0 (base64 npy is faithful, unlike the old JSON lists).
        re_blob = c.get("routed_experts")
        gen = SimpleNamespace(
            token_ids=ids,
            text="",  # /inference/v1/generate is token-only; the runner decodes text from token_ids
            finish_reason=finish_reason,
            logprobs=logprobs,
            routed_experts=_decode_routed_experts(re_blob) if (re_blob is not None and ids) else None,
        )
        # off_policy_len=0: the HTTP transport can't observe a mid-request weight-swap boundary and
        # doesn't need to — each token keeps its generation-time logprob, so per-token IS
        # (models/loss.py) corrects a mixed-weights request and the tis band drops the diverged tokens.
        return SimpleNamespace(outputs=[gen], prompt_routed_experts=None), 0


@ray.remote(num_cpus=1)  # MUST reserve a CPU: with num_cpus=0 the SPREAD strategy can't balance
# placement (every node looks empty), so runners pile onto a few nodes and their desktop-env VMs
# starve/OOM (18204 connection refused). 1 CPU/runner lets Ray spread them evenly across nodes.
class AgentRunnerActor:
    """One rollout driver process: runs the user's agent runner against the router.

    Loads the agent runner + tokenizer, holds one aiohttp session to the router + a
    ``RouterGenerateClient`` over it, and runs N rollouts of a prompt concurrently. Grading
    and VLM image processing run in-process (GIL-bound), so a fleet of these actors
    (``--rollout.num_runners``) parallelizes that work; the trainer round-robins prompts
    across them (no pool wrapper — just a list + an index)."""

    async def __init__(self, agent_path, router_url, *, model_path=None, model_name="policy"):
        import aiohttp

        from molt.agents.base import load_agent_runner  # lazy: router.py is imported by _chat_server
        from molt.trainer.rollout.samples_generator import SamplesGenerator
        from molt.utils import get_tokenizer

        self._runner = load_agent_runner(agent_path)
        self._tokenizer = get_tokenizer(model_path, None) if model_path else None
        # Image/video placeholder ids, cached once — _process_response_into_experience uses them to
        # detect an image dropped by max_len truncation (vit-embed misalignment) and count tokens.
        self._media_ids = SamplesGenerator._media_token_ids(self._tokenizer) if self._tokenizer else set()
        # VLM image placeholder id (from the HF processor) — the transport uses it to align render's
        # mm features onto our canonical prompt's image-token run (see _align_features_to_canonical).
        image_token_id = getattr(self._tokenizer, "image_token_id", None)
        # total=None: a single long-context / multi-turn generation can run many minutes (aiohttp's
        # 300s default would silently drop 32K rollouts). sock_read=900 still catches a HUNG engine
        # (stream stalls, no bytes for 15 min) so one wedged request can't hang the whole step; a
        # crashed engine fails fast via a connection error. limit=0: don't cap concurrency to the
        # engine fleet at aiohttp's default of 100 connections.
        self._http = aiohttp.ClientSession(
            base_url=router_url,
            timeout=aiohttp.ClientTimeout(total=None, sock_read=900),
            connector=aiohttp.TCPConnector(limit=0),
        )
        self._client = RouterGenerateClient(self._http, model_name=model_name, image_token_id=image_token_id)

    async def ready(self):
        return True

    async def run_group(self, prompt, label, images, sampling_params, max_length, n_samples, tools=None):
        """N rollouts of one prompt (unchanged runner) -> per step-sample ``(experience, drop_reason)``.

        Each usable trajectory is built into a train-ready Experience HERE, on the producing runner,
        then `offload`ed: its heavy tensors (images / token ids / rollout routing) stay in THIS
        node's object store and only the lazy handle travels to the controller — so a 256-session
        batch of screenshots never concentrates on one node. Tagged group_id (per prompt; GRPO
        baseline) + rollout_id (per rollout; multi-turn step-samples share it). A failed/unusable
        rollout is dropped, never sinks the group."""
        from molt.trainer.rollout.samples_generator import SamplesGenerator

        group_id = uuid4().hex
        tasks = [
            self._runner.execute(
                prompt=prompt,
                label=label,
                sampling_params=deepcopy(sampling_params),
                max_length=max_length,
                hf_tokenizer=self._tokenizer,
                llm_engine=self._client,
                images=images,
                tools=tools,
            )
            for _ in range(n_samples)
        ]
        results = []
        for r in await asyncio.gather(*tasks, return_exceptions=True):  # a failed rollout must not sink the group
            if isinstance(r, BaseException):
                print(f"[runner] dropping failed rollout in group {group_id}: {r!r}", flush=True)
                continue
            rollout_id = uuid4().hex
            for traj in r if isinstance(r, list) else [r]:
                traj.group_id, traj.rollout_id = group_id, rollout_id
                exp, drop_reason = SamplesGenerator._process_response_into_experience(
                    traj, self._media_ids, max_length
                )
                if exp is None:
                    results.append((None, drop_reason))
                    continue
                # Offload the heavy tensors into THIS runner's object store; the controller receives
                # a lazy Experience (a handle) and only its consuming rank calls reload() to fetch them.
                results.append((exp.offload(), None))
        return results


def create_vllm_router(engines, *, policy="consistent_hash", port=30000):
    """Serve each engine's OpenAI API + launch the vllm-router in front; return (router, url).

    Default policy ``consistent_hash`` routes by the ``x-session-id`` header so a rollout's render +
    generate land on ONE engine (mm-feature cache affinity — see ``RouterGenerateClient.generate``);
    prefix caching is off, so ``cache_aware`` would add no KV reuse to trade for that affinity."""
    engine_urls = ray.get([e.serve_openai.remote() for e in engines])
    router = VllmRouterActor.remote(engine_urls, policy=policy, port=port)
    return router, ray.get(router.ready.remote())
