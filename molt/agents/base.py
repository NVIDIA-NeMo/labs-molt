# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Molt agent primitives — Gymnasium-aligned.

User writes one of two behavior classes:

    Env       — async step(state) -> Result ; async reset(state) -> dict
    ChatAgent  — async run(ctx: ChatContext) -> Result   (in chat_agent.py)

The trainer drives one of two Runners (both subclass Runner):

    StepEnvRunner   — binds an Env subclass; framework owns the LLM loop.
    ChatAgentRunner — binds a ChatAgent subclass; user owns the loop via the
                      OpenAI or Anthropic SDK against the session URL.
                      (in chat_agent.py)

`Result` mirrors Gymnasium's (observation, reward, terminated, truncated, info)
step-return tuple — used by both the per-step `Env.step()` and the one-shot
`ChatAgent.run()`. `Trajectory` is the accumulated rollout the trainer consumes.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import torch

from molt.utils.logging_utils import init_logger

logger = init_logger(__name__)


def _first_scalar(value):
    if value is None:
        return None
    if hasattr(value, "detach") and hasattr(value, "flatten"):
        value = value.detach().flatten()
        return value[0].item() if value.numel() else None
    if isinstance(value, (list, tuple)):
        return _first_scalar(value[0]) if value else None
    return value


def _tokenize_observation(hf_tokenizer, text, images):
    """VLM-aware first-tokenize for the rollout entry point.

    Returns (tokens, mm_train_inputs, pil_images). When `images` is empty or
    `hf_tokenizer` has no image_processor, the last two are (None, []).
    """
    if images and hasattr(hf_tokenizer, "image_processor"):
        from molt.utils.vlm_utils import process_prompt_with_images

        return process_prompt_with_images(hf_tokenizer, text, images)
    tokens = hf_tokenizer(text=text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
    return tokens, None, []


def _extract_generation_logprobs(action_token_ids, generation_logprobs):
    if generation_logprobs is None:
        raise RuntimeError("vLLM did not return token logprobs while rollout importance correction is enabled.")
    if len(generation_logprobs) != len(action_token_ids):
        raise RuntimeError(
            "vLLM token logprobs must align with generated token ids: "
            f"got {len(generation_logprobs)} logprob entries for {len(action_token_ids)} tokens."
        )
    out = []
    for token_id, entries in zip(action_token_ids, generation_logprobs):
        entry = entries.get(token_id) if entries else None
        out.append(entry.logprob if entry is not None else 0.0)
    return out


# ---------------------------------------------------------------------------
# Result — Gymnasium-style return type from Env.step() and ChatAgent.run().
# Fields mirror gymnasium.Env.step return:
#   observation, reward, terminated, truncated, info
# Molt extras: score (scoreboard), images (multimodal), sampling_params.
# ---------------------------------------------------------------------------
@dataclass
class Result:
    reward: float | torch.Tensor = 0.0
    observation: str = ""  # next-turn observation text
    terminated: bool = True  # episode finished naturally
    truncated: bool = False  # cut off externally (max turns etc.)
    info: dict = field(default_factory=dict)
    score: float | torch.Tensor | None = None  # defaults to reward at trainer boundary
    images: list | None = None
    sampling_params: object = None


# ---------------------------------------------------------------------------
# Trajectory — accumulated rollout the trainer consumes directly (no dict
# coercion). Field names below are the stable contract with samples_generator;
# group_id / rollout_id are stamped by the trainer's vllm_engine after rollout.
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    prompt: str
    label: str
    images: object  # mirrors the dict key "images" (was source_images)
    observation_text: str
    observation_tokens: list
    mm_train_inputs: object = None
    pil_images: list = field(default_factory=list)
    # Extra tokens vLLM injects when re-expanding multimodal placeholder ids.
    # Computed per turn from the processor; tracks the actual model.
    image_budget: int = 0
    action_ranges: list = field(default_factory=list)
    # Per action range: number of LEADING action tokens generated under stale
    # (pre-broadcast) weights during a partial rollout. Those tokens are dropped
    # from the action mask so they get zero gradient and leave the token-mean
    # denominator (slime's mask_offpolicy_in_partial_rollout). 0 when on-policy.
    off_policy_action_lens: list = field(default_factory=list)
    rollout_log_probs: list | None = None
    # R3 rollout routing replay: per-token MoE expert selection captured from the rollout
    # engine, aligned 1:1 with `observation_tokens` by absolute position (filled by
    # absorb_routing). Each entry is a ``[num_moe_layers, topk]`` int16 row, or None where
    # the engine returned no routing (the trailing sampled token, or — on engines that
    # don't route multimodal prompts — those positions); None becomes a sentinel
    # downstream so the training forward routes them naturally. None on the whole field
    # when capture is off. The training forward replays the captured rows so its router
    # picks the rollout's experts (AutoModel RouterReplay).
    routed_experts: list | None = None
    reward: float = 0.0
    scores: float = 0.0  # mirrors the dict key "scores" (plural for legacy reasons)
    truncated: bool = False
    extra_logs: dict = field(default_factory=dict)
    group_id: str | None = None  # one per prompt group (N rollouts); GRPO baseline averaging
    rollout_id: str | None = None  # one per rollout; multi-turn step-samples dedup

    def append_action(self, action_tokens, action_logprobs=None, off_policy_len=0):
        start = len(self.observation_tokens)
        self.observation_tokens.extend(action_tokens)
        self.action_ranges.append((start, len(self.observation_tokens)))
        self.off_policy_action_lens.append(int(off_policy_len))
        if self.rollout_log_probs is not None:
            self.rollout_log_probs.extend(action_logprobs or [0.0] * len(action_tokens))
        if self.routed_experts is not None:
            # placeholders; absorb_routing() fills them by absolute position (R3)
            self.routed_experts.extend([None] * len(action_tokens))

    def append_feedback(self, action_text: str, feedback_text: str, feedback_tokens):
        self.observation_text = self.observation_text + action_text + feedback_text
        self.observation_tokens.extend(feedback_tokens)
        if self.rollout_log_probs is not None:
            self.rollout_log_probs.extend([0.0] * len(feedback_tokens))
        if self.routed_experts is not None:
            # placeholders; the next turn's absorb_routing() fills them from its prefill
            # routing (by absolute position), else they stay None -> natural routing.
            self.routed_experts.extend([None] * len(feedback_tokens))

    def absorb_routing(self, request_output):
        """Place this turn's rollout MoE routing onto ``routed_experts`` (R3).

        The rollout engine returns the per-token expert ids for the tokens it processed,
        indexed from absolute position 0. The engine ships the generated-token ids on the
        completion and, when it splits them out, the prompt ids on the request;
        concatenating reconstructs the sequence (a unified-array engine returns it all on
        the completion, with no prompt split).
        We lay it down first-writer-wins so each token keeps the routing from the turn
        that first produced it, leaving positions the engine gave no routing for as None
        (a sentinel downstream, so the training forward routes them naturally). No-op when
        capture is off. Call AFTER :meth:`append_action` so the generated positions exist.
        """
        gen = getattr(request_output.outputs[0], "routed_experts", None)
        if gen is None:
            return
        prompt = getattr(request_output, "prompt_routed_experts", None)
        routed = (list(prompt) + list(gen)) if prompt is not None else list(gen)
        if self.routed_experts is None:
            self.routed_experts = [None] * len(self.observation_tokens)
        for i in range(min(len(routed), len(self.routed_experts))):
            if self.routed_experts[i] is None:
                # copy: the engine hands back a read-only view under async D2H
                self.routed_experts[i] = routed[i].copy()


# ---------------------------------------------------------------------------
# Env — gym-style per-episode behavior. Subclass and implement step(); reset()
# is optional (default passes observation through).
# ---------------------------------------------------------------------------
class Env(ABC):
    def __init__(self, *args, **kwargs):
        pass

    async def reset(self, state: dict, **kwargs) -> dict:
        """Optional per-episode setup; override to rewrite the initial observation.

        Args:
            state: ``{"observation": <prompt str>, "label": <ground truth>}``.

        Returns:
            The (possibly modified) state dict; ``state["observation"]`` becomes
            the first model input. Default is an identity passthrough.
        """
        return state

    @abstractmethod
    async def step(self, state: dict, **kwargs) -> Result:
        """Score one model turn and optionally produce next-turn feedback.

        Args:
            state: keys ``observation_text`` (context so far), ``action_text``
                (the model's latest generation), ``label`` (ground truth), and
                ``sampling_params`` (this turn's params).

        Returns:
            Result with a scalar ``reward`` (required). Optional: ``observation``
            (next-turn feedback text for multi-turn), ``score`` (defaults to
            reward), ``info`` (logged metrics), ``images``, ``sampling_params``,
            and ``terminated`` / ``truncated`` to end the episode.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Runner — abstract base for "produce one Trajectory per execute() call".
# Two concrete subclasses live in this file (StepEnvRunner) and chat_agent
# (ChatAgentRunner). The trainer consumes the Trajectory dataclass directly.
# ---------------------------------------------------------------------------
class Runner(ABC):
    # WHERE the chat template is applied. Both runners consume the SAME chat-format dataset
    # (--data.apply_chat_template): StepEnvRunner wants the dataset to pre-render (it appends raw
    # env-feedback tokens after a rendered first turn); ChatAgentRunner does NOT — the dataset hands
    # the messages through raw and the chat server renders them exactly once via the model's own
    # template (a pre-render would double-template + drop the image on structured-content VLMs).
    # Model-agnostic: keyed on the runner, never on the model.
    PRERENDER_PROMPT = True

    @abstractmethod
    async def execute(
        self,
        prompt: str,
        label: Any,
        sampling_params,
        max_length: int,
        hf_tokenizer,
        llm_engine,
        images=None,
        tools=None,
    ) -> Trajectory:
        raise NotImplementedError


def load_agent_runner(agent_path: str) -> Runner:
    """Import ``agent_path`` and instantiate its ``AgentRunner`` (a ``Runner``).

    Loaded by the rollout driver/pool; the runner drives generation against the router,
    so it is engine-independent.
    """
    assert agent_path and agent_path.endswith(".py"), "Agent path must be a Python file"
    import importlib.util

    spec = importlib.util.spec_from_file_location("agent_module", agent_path)
    agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(agent_module)

    assert hasattr(agent_module, "AgentRunner"), "Agent module must contain AgentRunner class"
    agent_runner_cls = agent_module.AgentRunner
    assert issubclass(agent_runner_cls, Runner), "AgentRunner must inherit from Runner"
    return agent_runner_cls()


# ---------------------------------------------------------------------------
# StepEnvRunner — drives an Env via step/reset. Owns the LLM generation loop,
# tokenization, multimodal accounting, and per-turn budget enforcement.
# ---------------------------------------------------------------------------
class StepEnvRunner(Runner):
    def __init__(self, env_cls):
        assert issubclass(env_cls, Env), "env_cls must inherit from Env"
        self.env_cls = env_cls

    async def execute(
        self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images=None, tools=None
    ):
        # tools are already rendered into the pre-rendered prompt (dataset chat template);
        # the kwarg exists only for Runner signature parity with the chat path.
        sampling_params = deepcopy(sampling_params)
        env = self.env_cls()  # fresh env instance per episode
        # One session id for the whole rollout: consistent_hash routes every turn (and each turn's
        # render + generate) to ONE engine, so the multi-turn KV prefix stays warm and a VLM turn's
        # render features resolve where it generates. Per-rollout, like vime/slime.
        rollout_sid = uuid4().hex

        reset = await env.reset({"observation": prompt, "label": label})
        observation_text = reset["observation"]

        # process_prompt_with_images (inside _tokenize_observation) is a multi-second CPU op per image.
        # run_group gathers n_samples rollouts on ONE actor event loop, so a synchronous tokenize
        # blocks the loop and serializes the group (+ stalls the others' awaited generations). Run it
        # in a thread so the loop stays free and rollouts overlap (mirrors the chat server).
        obs_tokens, mm_train_inputs, pil_images = await asyncio.get_running_loop().run_in_executor(
            None, _tokenize_observation, hf_tokenizer, observation_text, images
        )
        image_budget = 0
        if pil_images:
            from molt.utils.vlm_utils import estimate_vllm_input_expansion_delta

            image_budget = estimate_vllm_input_expansion_delta(hf_tokenizer, obs_tokens, mm_train_inputs, pil_images)

        generation_reserve = sampling_params.max_tokens or 32
        max_initial_length = max(1, max_length - generation_reserve - image_budget)
        if len(obs_tokens) > max_initial_length:
            if pil_images:
                raise ValueError(
                    f"VLM prompt length ({len(obs_tokens)}) exceeds max_initial_length ({max_initial_length}). "
                    "Truncating VLM prompts would break image token alignment with pixel_values. "
                    "Please increase --max_len or decrease --max_new_tokens."
                )
            logger.warning(
                f"Initial observation length ({len(obs_tokens)}) exceeds max_initial_length "
                f"({max_initial_length}). Truncating to fit within max_length ({max_length})."
            )
            obs_tokens = obs_tokens[-max_initial_length:]
            observation_text = hf_tokenizer.decode(obs_tokens, skip_special_tokens=False)

        trajectory = Trajectory(
            prompt=prompt,
            label=label,
            images=images,
            observation_text=observation_text,
            observation_tokens=obs_tokens,
            mm_train_inputs=mm_train_inputs,
            pil_images=pil_images,
            image_budget=image_budget,
            rollout_log_probs=[0.0] * len(obs_tokens) if sampling_params.logprobs is not None else None,
        )

        # Per-turn cap; remaining context dominates if smaller.
        per_turn_cap = sampling_params.max_tokens

        while True:
            remaining = max_length - len(trajectory.observation_tokens) - trajectory.image_budget
            turn_sp = deepcopy(sampling_params)
            turn_sp.max_tokens = min(per_turn_cap, remaining) if per_turn_cap is not None else remaining
            if turn_sp.max_tokens <= 0:
                trajectory.truncated = True
                break
            # A late turn's remaining budget can drop below a configured min_tokens;
            # vLLM only validates min<=max at construction, not on reassignment.
            min_tokens = getattr(turn_sp, "min_tokens", None)
            if min_tokens is not None and min_tokens > turn_sp.max_tokens:
                turn_sp.min_tokens = turn_sp.max_tokens

            mm_data = {"image": trajectory.pil_images} if trajectory.pil_images else None
            request_output, off_policy_len = await llm_engine.generate(
                trajectory.observation_tokens, turn_sp, multi_modal_data=mm_data, session_id=rollout_sid
            )
            generation = request_output.outputs[0]
            action_tokens = generation.token_ids
            # /inference/v1/generate is token-only (generation.text == ""); decode from the ids so
            # Env.step sees the action text. skip_special_tokens=False keeps answer markers (<answer>,
            # \boxed, tool tags) — matches observation_text decoding above.
            action_text = generation.text or (
                hf_tokenizer.decode(action_tokens, skip_special_tokens=False) if action_tokens else ""
            )
            trajectory.truncated = trajectory.truncated or generation.finish_reason == "length"

            result: Result = await env.step(
                {
                    "observation_text": trajectory.observation_text,
                    "action_text": action_text,
                    "label": label,
                    "sampling_params": deepcopy(turn_sp),
                }
            )
            if not isinstance(result, Result):
                raise TypeError(f"Env.step must return a Result, got {type(result).__name__}")

            reward_val = _first_scalar(result.reward)
            if reward_val is None:
                raise ValueError("Env.step must return a Result with a scalar reward.")
            score_val = _first_scalar(result.score) if result.score is not None else reward_val

            trajectory.reward += reward_val
            trajectory.scores = score_val
            trajectory.extra_logs = result.info or {}

            action_logprobs = None
            if trajectory.rollout_log_probs is not None:
                action_logprobs = _extract_generation_logprobs(action_tokens, generation.logprobs)
            trajectory.append_action(action_tokens, action_logprobs, off_policy_len=off_policy_len)
            trajectory.absorb_routing(request_output)  # fills routing by absolute position (R3)

            feedback_tokens = _tokenize_feedback(
                hf_tokenizer, result.observation, result.images, trajectory, max_length
            )
            trajectory.append_feedback(action_text, result.observation, feedback_tokens)

            if result.sampling_params is not None:
                sampling_params = deepcopy(result.sampling_params)
                per_turn_cap = sampling_params.max_tokens

            if result.terminated or result.truncated:
                trajectory.truncated = trajectory.truncated or result.truncated
                break

        return trajectory


def _tokenize_feedback(hf_tokenizer, feedback_text: str, new_images, trajectory: Trajectory, max_length: int):
    """Tokenize next-turn feedback; absorb new images when budget allows."""
    if new_images and hasattr(hf_tokenizer, "image_processor"):
        from molt.utils.vlm_utils import (
            accumulate_mm_inputs,
            estimate_vllm_input_expansion_delta,
            process_prompt_with_images,
        )

        tokens, new_mm, new_pil = process_prompt_with_images(hf_tokenizer, feedback_text, new_images)
        new_budget = estimate_vllm_input_expansion_delta(hf_tokenizer, tokens, new_mm, new_pil)
        if (
            len(trajectory.observation_tokens) + len(tokens) + trajectory.image_budget + new_budget <= max_length
            or new_mm is None
        ):
            trajectory.pil_images.extend(new_pil)
            trajectory.mm_train_inputs = accumulate_mm_inputs(trajectory.mm_train_inputs, new_mm)
            trajectory.image_budget += new_budget
            return tokens
        # Overflow — drop images and strip their placeholder ids so vLLM
        # does not try to expand them.
        pad_ids = {getattr(hf_tokenizer, a, None) for a in ("image_token_id", "video_token_id")} - {None}
        return [t for t in tokens if t not in pad_ids]
    return hf_tokenizer(text=feedback_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
