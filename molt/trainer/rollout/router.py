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

The router load-balances the engines' OpenAI API (default cache_aware: each request follows its token
prefix to the engine already serving that rollout while load is even, else the shortest queue); weight
sync bypasses it (NCCL to the engines). Everything the runners generate goes over HTTP through the router via
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
    URL. Weight sync bypasses the router (NCCL straight to the engines).

    Both listen ports are taken from the kernel rather than left at the router's defaults (30000 /
    29000): a router leaked by a killed job keeps those bound, and every later job landing on that
    node then dies at startup — the metrics listener panics first, which kills the whole subprocess.
    Callers discover the address through ``url()``, so nothing depends on a fixed number."""

    @staticmethod
    def _free_port(host: str) -> int:
        with socket.socket() as s:
            s.bind((host, 0))
            return s.getsockname()[1]

    def __init__(self, worker_urls, *, policy="cache_aware", balance=(10, 1.2), port=None, max_payload_mb=512):
        import subprocess
        import sys

        self._host = ray.util.get_node_ip_address()
        self._port = port or self._free_port(self._host)
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm_router.launch_router",
                "--host",
                self._host,
                "--port",
                str(self._port),
                "--prometheus-host",
                self._host,
                "--prometheus-port",
                str(self._free_port(self._host)),
                "--policy",
                policy,
                # cache_aware keeps its prefix placement until (max - min) > abs AND max > rel * min.
                # The router's 64 / 1.5 are sized for a serving fleet: one rollout batch puts only
                # `rollout.generate_batch_size` requests in flight across the engines, so a gap above 64
                # is unreachable and one slow engine kept filling until its KV cache thrashed. vime
                # runs the same workload at 10 / 1.2 (vime/backends/vllm_utils/arguments.py).
                "--balance-abs-threshold",
                str(balance[0]),
                "--balance-rel-threshold",
                str(balance[1]),
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
        "logprobs": getattr(sp, "logprobs", None) or 1,
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
    # The runner decodes the action text from the ids itself, so skip vLLM's frontend detokenizer
    # (with top-k logprobs it re-decodes every candidate of every token and pins that process at
    # 100% CPU; stop STRINGS are the one thing that needs it). Top-k logprobs go to vLLM's flat
    # per-request storage instead of one dict of Logprob objects per token.
    if "stop" not in fields:
        fields["detokenize"] = False
    if fields["logprobs"] > 1:
        fields["flat_logprobs"] = True
        if "stop" in fields:
            # Under the detokenizer vLLM 0.31.0 names a top_logprobs entry by its text only, so the
            # candidate id is lost; stop_token_ids keep the ids.
            raise ValueError("score centering (top-k logprobs) does not support stop strings; use stop_token_ids")
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
        # One ``session_id`` per rollout goes out as ``x-session-id``. The default cache_aware policy
        # ignores headers and places each generate by its token prefix, so a rollout's turns (each
        # extending the last) stay on one engine until load skews; the header is the routing key only
        # under ``--vllm.router_policy consistent_hash``. Falls back to a per-call id when none is passed.
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
        retry the same way). A 4xx is a real client bug, so it fails fast rather than burn the retry
        budget. The session sets no read timeout, so a slow generation never lands here at all."""
        headers = {"x-session-id": session_id} if session_id else None
        body = {"model": self.model_name, **payload}
        for attempt in range(retries):
            try:
                async with self.http.post(path, json=body, headers=headers) as resp:
                    resp.raise_for_status()
                    return await resp.json()
            except (aiohttp.ClientResponseError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as e:
                fatal = isinstance(e, aiohttp.ClientResponseError) and e.status < 500  # 4xx = real bug
                if fatal or attempt == retries - 1:
                    raise
            await asyncio.sleep(attempt + 1)

    async def _render(self, images, session_id):
        """POST the image(s) to the stock ``/v1/chat/completions/render`` route (server-side chat
        template + mm processing) and return its ``{token_ids, features}``. We use ONLY ``features``
        (vLLM's mm pixel tensors -> N vision embeds); the placeholder ranges are realigned to our
        canonical prompt by the caller. The router forwards this custom route verbatim."""
        urls = [await asyncio.to_thread(_image_data_uri, image) for image in images]
        content = [{"type": "image_url", "image_url": {"url": url}} for url in urls]
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
        # Score centering (sampling_params.logprobs = k > 1): the sampler's top-k per token as two
        # [n, k] arrays, the sampled token leading each row. vLLM >= 0.31.1 names the candidate with
        # a `token_id` field, 0.31.0 only as the "token_id:<id>" string.
        top_logprobs = None
        if (getattr(sampling_params, "logprobs", None) or 1) > 1:
            k = sampling_params.logprobs
            heads = [item.get("top_logprobs") or [] for item in content]
            if any(len(h) != k for h in heads):
                raise RuntimeError(f"/inference/v1/generate returned a top_logprobs row without exactly {k} entries.")
            top_logprobs = (
                np.array([e.get("token_id", e["token"].rsplit(":", 1)[-1]) for h in heads for e in h], dtype=np.int64),
                np.array([e["logprob"] for h in heads for e in h], dtype=np.float32),
            )
            top_logprobs = tuple(a.reshape(len(heads), k) for a in top_logprobs)
        fr = c.get("finish_reason")
        finish_reason = fr.get("type") if isinstance(fr, dict) else (fr or "stop")
        # routed_experts is the UNIFIED full-sequence [tokens,layer,topk] npy (prompt+gen); absorb_routing
        # lays it down by absolute position from 0 (base64 npy is faithful, unlike the old JSON lists).
        re_blob = c.get("routed_experts")
        routed_experts = (
            await asyncio.to_thread(_decode_routed_experts, re_blob) if (re_blob is not None and ids) else None
        )
        gen = SimpleNamespace(
            token_ids=ids,
            text="",  # /inference/v1/generate is token-only; the runner decodes text from token_ids
            finish_reason=finish_reason,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            routed_experts=routed_experts,
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
    (``--rollout.num_runners``) parallelizes that work; the trainer round-robins rollouts
    across them (no pool wrapper — just a list + an index)."""

    def __init__(self, agent_path, router_url, *, model_path=None, model_name="policy"):
        import aiohttp

        from molt.agents.base import load_agent_runner  # lazy: router.py is imported by _chat_server
        from molt.utils import get_tokenizer
        from molt.utils.vlm_utils import media_token_ids

        self._runner = load_agent_runner(agent_path)
        self._tokenizer = get_tokenizer(model_path, None) if model_path else None
        # Image/video placeholder ids, cached once — _process_response_into_experience uses them to
        # detect an image dropped by max_len truncation (vit-embed misalignment) and count tokens.
        self._media_ids = media_token_ids(self._tokenizer) if self._tokenizer else set()
        # VLM image placeholder id (from the HF processor) — the transport uses it to align render's
        # mm features onto our canonical prompt's image-token run (see _align_features_to_canonical).
        image_token_id = getattr(self._tokenizer, "image_token_id", None)
        # No timeout at all. /inference/v1/generate is non-streaming, so the engine sends nothing
        # until the turn is fully generated: to the client "still generating" and "wedged" are the
        # same silence, and any finite read timeout eventually kills the former. A slow turn then
        # gets resent and re-runs from token 0, hitting the same wall — the retry can never
        # succeed. slime takes the same position (httpx.Timeout(None) on its rollout client).
        # A crashed engine still surfaces immediately as a connection error, which _post retries.
        # limit=0: don't cap concurrency to the engine fleet at aiohttp's default of 100.
        self._http = aiohttp.ClientSession(
            base_url=router_url,
            timeout=aiohttp.ClientTimeout(total=None),
            connector=aiohttp.TCPConnector(limit=0),
        )
        self._client = RouterGenerateClient(self._http, model_name=model_name, image_token_id=image_token_id)

    async def ready(self):
        return True

    async def run_group(
        self, prompt, label, images, sampling_params, max_length, n_samples, tools=None, group_id=None
    ):
        """N rollouts of one prompt (unchanged runner) -> per step-sample ``(experience, drop_reason)``.

        The trainer dispatches one rollout per call and passes the prompt's ``group_id`` so the
        rollouts it spreads over several runners still form one group (see _gather_group).

        Each usable trajectory is built into a train-ready Experience HERE, on the producing runner,
        then `offload`ed: its heavy tensors (images / token ids / rollout routing) stay in THIS
        node's object store and only the lazy handle travels to the controller — so a 256-session
        batch of screenshots never concentrates on one node. Tagged group_id (per prompt; GRPO
        baseline) + rollout_id (per rollout; multi-turn step-samples share it). A failed/unusable
        rollout is dropped, never sinks the group."""
        from molt.trainer.rollout.samples_generator import SamplesGenerator

        group_id = group_id or uuid4().hex
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
                results.append((None, "runner_error"))
                continue
            rollout_id = uuid4().hex
            for traj in r if isinstance(r, list) else [r]:
                traj.group_id, traj.rollout_id = group_id, rollout_id
                try:
                    exp, drop_reason = SamplesGenerator._process_response_into_experience(
                        traj, self._media_ids, max_length
                    )
                except Exception as e:  # a malformed trajectory must not sink the group either
                    print(f"[runner] dropping unconvertible rollout in group {group_id}: {e!r}", flush=True)
                    exp, drop_reason = None, "convert_error"
                if exp is None:
                    results.append((None, drop_reason))
                    continue
                # Offload the heavy tensors into THIS runner's object store; the controller receives
                # a lazy Experience (a handle) and only its consuming rank calls reload() to fetch them.
                results.append((exp.offload(), None))
        return results


def create_vllm_router(engines, *, policy="cache_aware", balance=(10, 1.2), port=None):
    """Serve each engine's OpenAI API + launch the vllm-router in front; return (router, url).

    Default policy ``cache_aware``: a generate goes to the engine whose radix tree holds its token
    prefix, so a rollout's turns stay together, and to the shortest queue once engines fall out of
    balance. ``consistent_hash`` hashed the per-rollout ``x-session-id`` blind to load, so engines
    filled unevenly and the slow one thrashed its KV cache."""
    engine_urls = ray.get([e.serve_openai.remote() for e in engines])
    router = VllmRouterActor.remote(engine_urls, policy=policy, balance=balance, port=port)
    return router, ray.get(router.ready.remote())
