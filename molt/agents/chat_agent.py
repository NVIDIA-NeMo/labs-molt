# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-completion agent — public face for the chat-agent path.

Subclass `ChatAgent` and implement `run(ctx)`. The rollout is a true black
box: any stock OpenAI *or* Anthropic client (opencode, claude code,
AgentScope, ...) just points at the session URL with `api_key=EMPTY`. The
server transparently captures the token trace because the URL carries the
session id as a path prefix (`/s/<id>`). No `extra_body`, no `logprobs=True`,
no auth/session plumbing leaks into agent code.

    from openai import AsyncOpenAI
    from molt.agents import ChatAgent, ChatAgentRunner, ChatContext, Result

    class MyAgent(ChatAgent):
        async def run(self, ctx: ChatContext) -> Result:
            client = AsyncOpenAI(base_url=ctx.base_url, api_key=ctx.api_key)
            resp = await client.chat.completions.create(
                model=ctx.model_name,
                messages=list(ctx.messages),  # the dataset row, ready to send
                max_tokens=ctx.sampling_params.max_tokens,
            )
            return Result(reward=grade(resp.choices[0].message.content, ctx.label))

    class AgentRunner(ChatAgentRunner):
        def __init__(self):
            super().__init__(MyAgent)

Same server, Anthropic wire — point `AsyncAnthropic` at `ctx.session_url`
(no `/v1` suffix; the SDK appends `/v1/messages` itself):

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(base_url=ctx.session_url, api_key=ctx.api_key)
    msg = await client.messages.create(
        model=ctx.model_name,
        messages=list(ctx.messages),
        max_tokens=ctx.sampling_params.max_tokens,
    )
    text = msg.content[0].text

A FastAPI vLLM server runs on loopback under `ctx.session_url`. The same
endpoint also serves external HTTP callers (third-party agent frameworks that
speak either the OpenAI or the Anthropic HTTP wire protocol).
"""

from __future__ import annotations

import asyncio
import base64
import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from molt.agents._chat_server import (
    ChatServerState,
    mount_session_capture,
    serve_on_current_loop,
    stitch_session,
)
from molt.agents.base import Result, Runner
from molt.utils.logging_utils import init_logger

# The chat server registers this as the policy model id; agents request it via
# ``ctx.model_name`` (vLLM validates ``request.model`` against served names).
_SERVED_MODEL_NAME = "policy"

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# ChatContext — per-rollout bundle handed to ChatAgent.run().
# ---------------------------------------------------------------------------
@dataclass
class ChatContext:
    # The dataset row as OpenAI wire-format messages: system/multi-turn structure preserved,
    # `<image>` placeholders inlined as image_url data URIs. Send them verbatim
    # (`messages=list(ctx.messages)`) — the chat server renders them once with the model's own
    # template, so step and chat runners consume the SAME chat-format dataset.
    messages: list
    # The dataset row's `tools` column (--data.tools_key): OpenAI function-call schemas, the
    # same ones the step runner renders into its prompt. None when the dataset has none.
    tools: list | None
    prompt: str  # the last user turn's text — a scalar view of the task (grading/logging)
    label: Any
    images: list | None
    # Session-scoped base_url WITH the "/v1" suffix, e.g.
    # "http://127.0.0.1:port/s/<sid>/v1". Point a stock OpenAI client here
    # (the SDK appends "/chat/completions"); every call is traced transparently.
    base_url: str
    # The same session root WITHOUT "/v1", e.g. ".../s/<sid>". Point a stock
    # Anthropic client here — AsyncAnthropic appends "/v1/messages" itself.
    session_url: str
    model_name: str
    api_key: str  # always "EMPTY" — session lives in base_url, not auth
    session_id: str  # raw id; rarely needed once base_url carries it
    sampling_params: Any
    max_length: int


# ---------------------------------------------------------------------------
# ChatAgent — behavior class for the chat-agent path. User owns the loop and uses
# `openai.AsyncOpenAI` directly.
# ---------------------------------------------------------------------------
class ChatAgent(ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    async def run(self, ctx: ChatContext) -> Result:
        """Run one episode against an OpenAI-compatible endpoint and score it.

        Point ``openai.AsyncOpenAI(base_url=ctx.base_url, api_key=ctx.api_key)``
        at the session URL; every call through it is traced and stitched into a
        single training Trajectory automatically. Start from ``ctx.messages`` /
        ``ctx.tools`` (the dataset row, ready to send verbatim), grade against
        ``ctx.label``, and respect ``ctx.sampling_params`` / ``ctx.max_length``
        for the generation budget.

        Returns:
            Result with a scalar ``reward`` (required); ``score`` and ``info``
            are optional (same contract as ``Env.step``).
        """
        raise NotImplementedError


def _pil_data_uri(pil) -> str:
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _wire_messages(prompt, images) -> list:
    """Dataset row -> OpenAI wire-format messages, ready to send verbatim.

    Each ``<image>`` in a user turn's text consumes the next image of the row's images
    column (in order) as an ``image_url`` data URI — the wire analogue of the step
    path's placeholder expansion, so both runners read the dataset identically.
    Structured (list) content and images the text never references pass through on
    ``ctx.images`` untouched."""
    messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
    from molt.utils.vlm_utils import load_images  # lazy: PIL/vision deps only on VLM rows

    pil_images = list(load_images(images)) if images else []
    out = []
    for msg in messages:
        content = msg.get("content")
        if pil_images and msg.get("role") == "user" and isinstance(content, str) and "<image>" in content:
            parts = []
            for i, chunk in enumerate(content.split("<image>")):
                if i:
                    # More markers than images: keep the literal marker so the mismatch
                    # surfaces at the processor's count check instead of vanishing here.
                    parts.append(
                        {"type": "image_url", "image_url": {"url": _pil_data_uri(pil_images.pop(0))}}
                        if pil_images
                        else {"type": "text", "text": "<image>"}
                    )
                if chunk:
                    parts.append({"type": "text", "text": chunk})
            msg = {**msg, "content": parts}
        out.append(msg)
    return out


# ---------------------------------------------------------------------------
# ChatAgentRunner — drives a ChatAgent subclass; stitches its chat traces into
# a list of Trajectory step-samples the trainer consumes.
# ---------------------------------------------------------------------------
class ChatAgentRunner(Runner):
    # The dataset hands the chat messages through raw (--data.apply_chat_template is required —
    # asserted at startup); the chat server renders them exactly once via the model's own chat
    # template. A dataset pre-render would double-template and drop the image on
    # structured-content VLMs.
    PRERENDER_PROMPT = False

    def __init__(self, agent_cls: type[ChatAgent]):
        assert issubclass(agent_cls, ChatAgent), "agent_cls must inherit from ChatAgent"
        self.agent_cls = agent_cls
        self._state: ChatServerState | None = None
        self._server_root: str | None = None
        self._server_task: asyncio.Task | None = None
        self._boot_lock = asyncio.Lock()  # serialize first-call server bring-up

    async def _ensure_server(self, llm_engine, hf_tokenizer, max_length, sampling_params):
        """Bring the loopback chat forwarder up once, on the actor's OWN event loop.
        It forwards each turn over HTTP to the router (via the transport's shared aiohttp
        session). Concurrent first calls are serialized; later return at once."""
        if self._state is not None:
            return
        async with self._boot_lock:
            if self._state is not None:
                return
            from fastapi import FastAPI

            app = FastAPI()

            @app.get("/health")
            async def _health():
                return {"status": "ok", "model": _SERVED_MODEL_NAME}

            # transport = the shared RouterGenerateClient; the forwarder renders each turn client-side
            # and generates token-in via the transport (/inference/v1/generate) — see _chat_server.
            state = ChatServerState(llm_engine, hf_tokenizer, _SERVED_MODEL_NAME, max_length, sampling_params)
            mount_session_capture(app, state)
            port, self._server_task = await serve_on_current_loop(app)
            self._server_root = f"http://127.0.0.1:{port}"
            self._state = state
            logger.info(f"Chat server ready at {self._server_root} (model={_SERVED_MODEL_NAME})")

    async def execute(
        self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images=None, tools=None
    ):
        await self._ensure_server(llm_engine, hf_tokenizer, max_length, sampling_params)
        session_id = uuid4().hex
        messages = _wire_messages(prompt, images)
        # Scalar view of the task for grading/logging (and Trajectory.prompt): the last user
        # turn's text — taken from the RAW row (wire messages may have inlined its images).
        raw = prompt if isinstance(prompt, list) else [{"role": "user", "content": str(prompt)}]
        user_texts = [m["content"] for m in raw if m.get("role") == "user" and isinstance(m.get("content"), str)]
        prompt_text = user_texts[-1] if user_texts else prompt
        # Pass THIS call's sampling_params so the server defaults each turn to the right train-vs-eval
        # temperature/max_tokens even if the agent doesn't re-send them (see _Session.sampling_params).
        self._state.open(session_id, prompt_text, label, images, sampling_params)
        session_root = f"{self._server_root}/s/{session_id}"
        ctx = ChatContext(
            messages=messages,
            tools=tools,
            prompt=prompt_text,
            label=label,
            images=images,
            base_url=f"{session_root}/v1",
            session_url=session_root,
            model_name=self._state.model_name,
            api_key="EMPTY",
            session_id=session_id,
            sampling_params=sampling_params,
            max_length=max_length,
        )
        try:
            result = await self.agent_cls().run(ctx)
            if not isinstance(result, Result):
                raise TypeError(f"ChatAgent.run must return a Result, got {type(result).__name__}")
            return stitch_session(self._state, session_id, result)
        finally:
            self._state.discard(session_id)
