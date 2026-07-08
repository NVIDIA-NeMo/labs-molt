# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal chat forwarder over the vllm-router (imported by ``chat_agent.py``).

The agent talks plain **chat messages** to a loopback endpoint with a stock OpenAI
or Anthropic client. This server renders each turn's chat template CLIENT-SIDE (the same
``apply_chat_template`` -> ``process_prompt_with_images`` path the RL dataset + step runner
use, so the image-placeholder count is self-consistent with the vision encoder), generates
it token-in / token-out via the shared router transport (``RouterGenerateClient`` — see
``router.py`` for the wire: ``/inference/v1/generate`` with server-side image render), and
records the turn's EXACT tokens — the prompt ids we tokenized + the completion ``token_ids``
+ ``logprobs`` + R3 ``routed_experts`` — as ONE ``Trajectory`` step-sample. A multi-turn
episode becomes N step-samples sharing one ``rollout_id`` + reward (``stitch_session``); a
context rewrite (compaction) is simply the next turn's messages, so it yields a fresh sample
with NO special handling.

Three concerns, each a section below:
  1. WIRE CODECS — OpenAI is canonical (identity decode); Anthropic is translated
     to the same shape (``_decode_anthropic``) and the reply re-encoded per wire.
  2. FORWARD + SPLIT — ``_run_turn`` renders + generates one turn and builds one
     token-exact ``Trajectory`` from the response.
  3. HTTP SERVER — uvicorn on the actor's OWN event loop + the session routes.

Runs on the rollout runner actor's own event loop. A stock client reaches a session
through the URL path prefix ``/s/<session_id>/v1`` that ``ChatAgentRunner`` builds.
"""

from __future__ import annotations

import asyncio
import logging
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from molt.agents.base import Trajectory, _extract_generation_logprobs, _tokenize_observation
from molt.utils.vlm_utils import (
    estimate_vllm_input_expansion_delta,
    load_images,
    should_expand_image_placeholder,
    split_image_placeholder,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# 1. Wire codecs — Anthropic request -> canonical (OpenAI) messages; replies back.
#    OpenAI is the canonical shape, so its decode is identity (``lambda b: b``).
# ===========================================================================
def _content_to_text_and_images(content) -> tuple[str, list]:
    """One OpenAI message's ``content`` -> (flat text with a literal ``<image>`` per image,
    loaded PIL images). Mirrors the dataset convention (user content is ``"<image>...problem"``);
    the client-side chat template + processor then expand each ``<image>`` exactly as the step
    runner does. A bare-string content passes through with no images."""
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return ("" if content is None else str(content)), []
    parts, pil = [], []
    for item in content:
        if not isinstance(item, dict):
            parts.append(str(item))
        elif item.get("type") == "text":
            parts.append(item.get("text") or "")
        elif item.get("type") == "image_url":
            ref = item.get("image_url") or {}
            url = ref.get("url") if isinstance(ref, dict) else ref
            imgs = load_images(url) if url else []
            pil.extend(imgs)
            parts.append("<image>" * len(imgs))
    return "".join(parts), pil


def _decode_anthropic(body: dict) -> dict:
    """Anthropic Messages request -> canonical (OpenAI-shaped) body the router
    understands. Only content blocks + (already same-named) sampling fields are
    touched; top-level ``system`` is left inert."""
    messages = []
    for m in body.get("messages", []):
        content = m.get("content")
        if isinstance(content, list):
            content = [out for b in content for out in _anthropic_blocks(b)]
        messages.append({"role": m.get("role"), "content": content})
    return {**body, "messages": messages}


def _anthropic_blocks(block) -> list:
    """One Anthropic content block -> list of canonical (OpenAI) blocks. image ->
    ``image_url`` (base64 source -> data URI); ``tool_result`` -> its inner
    text/image blocks (an Anthropic harness returns tool output as a ``tool_result``
    block, so without unwrapping the observation would be dropped)."""
    if not isinstance(block, dict):
        return [block]
    btype = block.get("type")
    if btype == "image":
        src = block.get("source") or {}
        url = src.get("url") if src.get("type") == "url" else f"data:{src.get('media_type')};base64,{src.get('data')}"
        return [{"type": "image_url", "image_url": {"url": url}}]
    if btype == "tool_result":
        inner = block.get("content")  # bare string, or a list of text/image blocks
        if isinstance(inner, str):
            return [{"type": "text", "text": inner}]
        if isinstance(inner, list):
            return [out for b in inner for out in _anthropic_blocks(b)]
        return []
    return [block]


# OpenAI's ChatCompletion schema accepts only these finish_reason values and the SDK
# validates them client-side; vLLM also emits abort/error/repetition (and None
# mid-stream), so any non-OpenAI reason maps to "stop" purely to keep the reply
# SDK-parseable (the agent never reads it; truncation uses the RAW reason elsewhere).
_OPENAI_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter", "function_call"})


def _chat_completion_body(model_name: str, content: str, finish_reason: str) -> dict:
    return {
        "id": f"chatcmpl-{uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason if finish_reason in _OPENAI_FINISH_REASONS else "stop",
            }
        ],
    }


# Anthropic's Message schema validates stop_reason client-side too; map the raw vLLM
# reason into its vocab (anything else -> "end_turn").
_ANTHROPIC_STOP_REASONS = {"stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use"}


def _anthropic_message_body(model_name: str, content: str, finish_reason: str) -> dict:
    return {
        "id": f"msg_{uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": [{"type": "text", "text": content}],
        "stop_reason": _ANTHROPIC_STOP_REASONS.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


def _build_sampling_params(base_sampling, body: dict):
    """One vLLM SamplingParams per chat call: inherit THIS rollout's defaults (its train-vs-eval
    temperature / max_tokens — see ``_Session.sampling_params``), honor the caller's temperature /
    max_tokens, and force logprobs on (importance-sampling correction needs token logprobs)."""
    sp = deepcopy(base_sampling)
    requested = body.get("max_completion_tokens", body.get("max_tokens"))
    if requested is not None:
        sp.max_tokens = int(requested)
    if body.get("temperature") is not None:
        sp.temperature = float(body["temperature"])
    if body.get("top_p") is not None:
        sp.top_p = float(body["top_p"])
    if sp.logprobs is None:
        sp.logprobs = 1
    return sp


# ===========================================================================
# 2. Per-session state — each turn is one Trajectory step-sample (no carry-forward).
# ===========================================================================
@dataclass
class _Session:
    prompt: str
    label: Any
    images: list | None
    # This rollout's id, reused as the router ``x-session-id`` so all turns (and each turn's
    # render + generate) pin to one engine (consistent_hash) — see ``RouterGenerateClient.generate``.
    session_id: str | None = None
    # THIS rollout's SamplingParams (carries the train-vs-eval temperature / max_tokens for this
    # call). Used as the per-turn default before the agent's own body overrides it; the fallback is
    # ChatServerState.default_sampling, which is frozen at the FIRST execute — so an agent that
    # doesn't re-send temperature would otherwise sample every later train/eval call at the
    # first-caller's temperature (silent train/eval sampling-temperature mismatch).
    sampling_params: Any = None
    steps: list = field(default_factory=list)  # one Trajectory per turn (merged/split in stitch_session)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # serialize turns within a session
    # Idempotent replay: a stock client retries a turn whose response was lost (e.g. a weight
    # broadcast stalled the loop). The retry re-sends the SAME messages; we return the cached
    # reply WITHOUT recording a second step, so retries can't double-count a turn.
    last_messages: list | None = None
    last_response: tuple | None = None


class ChatServerState:
    """Per-actor state: the unified HTTP transport (over the router) + processor +
    live session accumulators. ``transport`` is the shared ``RouterGenerateClient``
    (``.generate(token_ids, sp, mm)`` + the shared ``.http`` session, base-URL'd to the router)."""

    def __init__(self, transport, processor, model_name: str, max_length: int, default_sampling):
        self.transport = transport  # RouterGenerateClient: .generate(token_ids, sp, mm) over /inference/v1/generate
        self.processor = processor
        self.model_name = model_name
        self.max_length = max_length
        self.default_sampling = default_sampling
        # omni3 uses a literal ``<image>`` (pass content through); qwen3.6-VL uses ``<|image_pad|>``
        # and needs the structured-content split. Mirrors the RL prompt dataset's rendering.
        self.expand_image_placeholder = should_expand_image_placeholder(processor)
        self.sessions: dict[str, _Session] = {}

    def open(self, session_id: str, prompt, label, images, sampling_params=None):
        self.sessions[session_id] = _Session(
            prompt=prompt, label=label, images=images, session_id=session_id, sampling_params=sampling_params
        )

    def discard(self, session_id: str):
        self.sessions.pop(session_id, None)


# ===========================================================================
# 3. Forward one turn -> one token-exact Trajectory step-sample.
# ===========================================================================
async def _run_turn(state: ChatServerState, session: _Session, body: dict) -> tuple[str, str]:
    """Render this turn's prompt CLIENT-SIDE the same way the RL dataset + step runner do
    (``apply_chat_template`` -> ``process_prompt_with_images`` -> image-expanded token ids +
    pixel_values, a placeholder count self-consistent with the vision encoder), generate it token-in
    via ``state.transport`` (the shared router transport — see ``router.py``), and record the EXACT
    tokens as one ``Trajectory`` step-sample. Returns ``(action_text, finish_reason)``; multi-turn /
    compaction just appends more."""
    messages = body.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    if messages == session.last_messages:  # retry of an already-recorded turn -> replay, don't double-record
        return session.last_response
    sp = _build_sampling_params(session.sampling_params or state.default_sampling, body)

    # OpenAI messages -> a ChatML chat with a literal <image> per image + the loaded PIL images, then
    # the same template + tokenize the RL dataset uses (structured-content split for models whose
    # image_token isn't the literal <image>). One tokenize; the image count matches pixel_values.
    chat, pil_images = [], []
    for m in messages:
        text, imgs = _content_to_text_and_images(m.get("content"))
        chat.append({"role": m.get("role"), "content": text})
        pil_images.extend(imgs)
    if state.expand_image_placeholder:
        chat = [split_image_placeholder(m) for m in chat]
    kwargs = {"tools": body["tools"]} if body.get("tools") else {}
    prompt_text = state.processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True, **kwargs)
    # process_prompt_with_images (inside _tokenize_observation) runs the VLM image processing
    # (resize/normalize/patchify) — a multi-second CPU op per image. This server drives ALL concurrent
    # rollout sessions on ONE event loop, so tokenizing synchronously would block the loop and stall
    # every other session's in-flight generation (serializing the batch). Run it in a thread so the
    # loop stays free and sessions overlap.
    prompt_ids, mm_train_inputs, pil_images = await asyncio.get_running_loop().run_in_executor(
        None, _tokenize_observation, state.processor, prompt_text, pil_images
    )

    image_budget = (
        estimate_vllm_input_expansion_delta(state.processor, prompt_ids, mm_train_inputs, pil_images)
        if pil_images
        else 0
    )
    # A grown multi-turn prompt can fill the context. Clamp generation to the remaining budget; if
    # the prompt already fills it, end the rollout truncated instead of erroring / dropping it
    # (dropping long rollouts biases the batch toward short trajectories — the step runner clamps
    # the same way).
    remaining = state.max_length - len(prompt_ids) - image_budget
    if remaining <= 0:
        session.last_messages, session.last_response = messages, ("", "length")
        return "", "length"
    if sp.max_tokens is None or sp.max_tokens > remaining:
        sp.max_tokens = remaining
    if getattr(sp, "min_tokens", 0) and sp.min_tokens > sp.max_tokens:
        sp.min_tokens = sp.max_tokens

    mm_data = {"image": pil_images} if pil_images else None
    request_output, off_policy_len = await state.transport.generate(
        prompt_ids, sp, multi_modal_data=mm_data, session_id=session.session_id
    )
    generation = request_output.outputs[0]
    action_ids = list(generation.token_ids)
    # /inference/v1/generate is token-only (generation.text == ""); decode the action text from the
    # ids so the agent/grader sees it. skip_special_tokens=False keeps answer markers (<answer>,
    # \boxed, tool tags) — matches observation_text decoding in base.py.
    action_text = generation.text or (
        state.processor.decode(action_ids, skip_special_tokens=False) if action_ids else ""
    )
    finish_reason = generation.finish_reason or "stop"
    # a non-empty completion MUST carry aligned per-token logprobs — fails fast otherwise (IS correction).
    action_logprobs = _extract_generation_logprobs(action_ids, generation.logprobs) if action_ids else []

    traj = Trajectory(
        prompt=session.prompt,
        label=session.label,
        images=session.images,
        observation_text="",
        observation_tokens=list(prompt_ids),
        mm_train_inputs=mm_train_inputs,
        pil_images=pil_images,
        image_budget=image_budget,
        rollout_log_probs=[0.0] * len(prompt_ids),
    )
    traj.truncated = finish_reason == "length"
    # off_policy_len: leading tokens generated under stale weights when a broadcast landed
    # mid-request (per-token IS still corrects them); the transport reports it, same as the step runner.
    traj.append_action(action_ids, action_logprobs, off_policy_len=off_policy_len)
    traj.absorb_routing(request_output)  # R3: unified prompt+gen routing by absolute position (same as step)
    session.steps.append(traj)
    session.last_messages, session.last_response = messages, (action_text, finish_reason)
    return action_text, finish_reason


def _merge_prefix_steps(steps: list) -> list:
    """Collapse consecutive turns whose prompt prefix-extends the running trajectory back into
    ONE growing trajectory, so each shared token is trained ONCE — O(final_len), not O(turns^2)
    re-forwards of overlapping prefixes (each stateless chat turn re-sends the full history). A
    prefix break (context compaction) or a turn that adds a new image seals the segment and starts
    a fresh one. The forward+split tokens are already drift-free, so this changes NOTHING about the
    trained tokens / logprobs / routing — it only removes redundant re-forwarding."""
    segments = [steps[0]]
    for step in steps[1:]:
        cur = segments[-1]
        n = len(cur.observation_tokens)
        astart = step.action_ranges[-1][0]  # this turn's context length (tokens before its action)
        extends = astart >= n and step.observation_tokens[:n] == cur.observation_tokens
        if not (extends and len(step.pil_images) == len(cur.pil_images)):
            # New segment: expected on context compaction / a new mid-conversation image, but ALSO
            # fires if the tokenizer didn't round-trip (re-template drift) — logged so the resulting
            # O(turns^2) re-forward + lost KV reuse on long episodes isn't silent.
            logger.info("chat rollout: segment split at turn boundary (compaction or re-template drift)")
            segments.append(step)
            continue
        cur.append_feedback("", "", step.observation_tokens[n:astart])  # new context delta (masked)
        action_lp = step.rollout_log_probs[astart:] if step.rollout_log_probs is not None else None
        cur.append_action(step.observation_tokens[astart:], action_lp, off_policy_len=0)
        if step.routed_experts is not None:  # copy the turn's routing onto the appended [n:] positions
            if cur.routed_experts is None:
                cur.routed_experts = [None] * len(cur.observation_tokens)
            for i in range(n, min(len(cur.observation_tokens), len(step.routed_experts))):
                if cur.routed_experts[i] is None:
                    cur.routed_experts[i] = step.routed_experts[i]
        cur.truncated = cur.truncated or step.truncated
    return segments


def stitch_session(state: ChatServerState, session_id: str, result):
    """Return the rollout's step-sample trajectories. Consecutive prefix-extending turns merge
    into one growing trajectory (``_merge_prefix_steps``); a compaction boundary yields another
    segment. Every segment carries the SAME terminal reward/score/info and shares the one
    ``rollout_id`` the trainer assigns, so group baselines dedup them to one reward per rollout
    while each turn still contributes its generated tokens to the policy-gradient axis (multi-turn
    step-sample contract — see ``experience_maker._merge_rollout_rewards``)."""
    session = state.sessions.get(session_id)
    if session is None:
        raise RuntimeError(f"Unknown session {session_id} at stitch")
    if not session.steps:
        return []  # the first prompt alone exceeded max_length -> nothing was generated; drop this rollout
    segments = _merge_prefix_steps(session.steps)
    for traj in segments:
        traj.reward = result.reward
        traj.scores = result.score if result.score is not None else result.reward
        traj.extra_logs = result.info or {}
        if result.images is not None:
            traj.images = result.images
    return segments


# ===========================================================================
# 4. HTTP server — uvicorn on the runner actor's own event loop + session routes.
# ===========================================================================
class _AutoPortServer(uvicorn.Server):
    """uvicorn Server that reports the OS-assigned port when configured ``port=0``."""

    def __init__(self, config: uvicorn.Config):
        super().__init__(config)
        self.actual_port: int | None = None
        self._ready = asyncio.Event()

    async def startup(self, sockets=None) -> None:
        try:
            await super().startup(sockets=sockets)
            if self.servers and self.config.port == 0:
                self.actual_port = self.servers[0].sockets[0].getsockname()[1]
            else:
                self.actual_port = self.config.port
        finally:
            self._ready.set()

    async def get_port(self) -> int | None:
        await self._ready.wait()
        return self.actual_port


async def serve_on_current_loop(app, host: str = "127.0.0.1") -> tuple[int, asyncio.Task]:
    """Start uvicorn as a task on the CURRENT event loop. Returns ``(port, task)``."""
    config = uvicorn.Config(app, host=host, port=0, log_level="warning")
    server = _AutoPortServer(config)
    task = asyncio.create_task(server.serve())
    port = await server.get_port()
    if port is None:
        await task  # surfaces the startup exception
        raise RuntimeError("Chat server failed to start (no port reported)")
    return port, task


def mount_session_capture(app, state: ChatServerState) -> None:
    """Mount the session-scoped chat routes. The OpenAI and Anthropic wires forward
    the SAME turn (``_run_turn``); they differ only in how the body is decoded to the
    canonical shape and how the reply is encoded back."""
    router = APIRouter()

    async def _serve(session_id, request, decode, encode):
        body = decode(await request.json())
        session = state.sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=400, detail=f"Unknown or unopened session {session_id}")
        # Serialize turns within a session: a stock client may retry, which would
        # otherwise run a second turn concurrently and double-record the sample.
        async with session.lock:
            action_text, finish_reason = await _run_turn(state, session, body)
        return JSONResponse(encode(state.model_name, action_text, finish_reason))

    @router.post("/s/{session_id}/v1/chat/completions")
    async def chat_sess(session_id: str, request: Request):
        return await _serve(session_id, request, lambda b: b, _chat_completion_body)

    @router.post("/s/{session_id}/v1/messages")
    async def messages_sess(session_id: str, request: Request):
        return await _serve(session_id, request, _decode_anthropic, _anthropic_message_body)

    app.include_router(router)
