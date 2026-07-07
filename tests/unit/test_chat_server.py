import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import molt.agents._chat_server as cs
from molt.agents._chat_server import (
    ChatServerState,
    _anthropic_message_body,
    _chat_completion_body,
    _content_to_text_and_images,
    _decode_anthropic,
    _run_turn,
    stitch_session,
)
from molt.agents.base import Result


# ---------------------------------------------------------------------------
# Fakes: the transport exposes generate(token_ids, sp, multi_modal_data) over the
# router's /v1/completions (same call the step runner uses). Client-side templating
# (apply_chat_template + _tokenize_observation) is monkeypatched so tests drive the
# exact prompt token ids per turn.
# ---------------------------------------------------------------------------
class _FakeTransport:
    def __init__(self, actions):  # actions: list of (action_ids, logprobs, finish, routed)
        self.actions = list(actions)
        self.calls = []

    async def generate(self, prompt_token_ids, sampling_params, multi_modal_data=None, session_id=None):
        self.calls.append((list(prompt_token_ids), multi_modal_data, sampling_params.max_tokens))
        aids, lps, finish, routed = self.actions.pop(0)
        logprobs = [{t: SimpleNamespace(logprob=lp)} for t, lp in zip(aids, lps)]
        gen = SimpleNamespace(
            token_ids=list(aids),
            text="ACT",
            finish_reason=finish,
            logprobs=logprobs,
            routed_experts=np.asarray(routed) if routed is not None else None,
        )
        return SimpleNamespace(outputs=[gen], prompt_routed_experts=None), 0


def _proc():
    # apply_chat_template output is fed to the monkeypatched _tokenize_observation, so its value is
    # irrelevant; image_token == "<image>" -> should_expand is False (no structured-content split).
    return SimpleNamespace(apply_chat_template=lambda chat, **k: "TEXT", image_token="<image>")


def _sampling(max_tokens=8):
    return SimpleNamespace(max_tokens=max_tokens, temperature=1.0, top_p=1.0, logprobs=1, min_tokens=0)


def _state(actions, *, max_length=1000):
    st = ChatServerState(_FakeTransport(actions), _proc(), "policy", max_length, _sampling())
    return st, st.transport


def _act(action_ids, logprobs, *, finish="stop", routed=None):
    return (action_ids, logprobs, finish, routed)


def _patch_prompts(monkeypatch, prompts, *, mm=None, pil=None):
    """Drive the per-turn prompt token ids (what apply_chat_template + processor would produce)."""
    q = list(prompts)
    monkeypatch.setattr(cs, "_tokenize_observation", lambda proc, text, imgs: (q.pop(0), mm, pil or []))


_MSG = {"messages": [{"role": "user", "content": "P"}]}


# ---------------------------------------------------------------------------
# Forward: tokenize (client-side) -> generate (token-in over /v1/completions) ->
# one token-exact Trajectory step-sample.
# ---------------------------------------------------------------------------
def test_run_turn_records_exact_tokens(monkeypatch):
    _patch_prompts(monkeypatch, [[1, 2, 3]])
    state, tp = _state([_act([90, 91], [-0.3, -0.4])])
    state.open("sid", "P", "lab", None)
    session = state.sessions["sid"]

    action, finish = asyncio.run(_run_turn(state, session, _MSG))

    assert (action, finish) == ("ACT", "stop")
    assert tp.calls[0][0] == [1, 2, 3]  # generate fed the tokenized prompt
    traj = session.steps[0]
    assert traj.observation_tokens == [1, 2, 3, 90, 91]
    assert traj.action_ranges == [(3, 5)]
    assert traj.rollout_log_probs == [0.0, 0.0, 0.0, -0.3, -0.4]
    assert traj.prompt == "P" and traj.label == "lab"


def test_run_turn_each_turn_is_its_own_sample(monkeypatch):
    _patch_prompts(monkeypatch, [[1, 2], [1, 2, 90, 5, 6]])
    state, _ = _state([_act([90], [-0.1]), _act([91, 92], [-0.2, -0.3])])
    state.open("sid", "P", "l", None)
    session = state.sessions["sid"]
    asyncio.run(_run_turn(state, session, _MSG))
    asyncio.run(_run_turn(state, session, {"messages": [{"role": "user", "content": "m"}]}))
    assert len(session.steps) == 2
    assert session.steps[0].observation_tokens == [1, 2, 90]
    assert session.steps[1].observation_tokens == [1, 2, 90, 5, 6, 91, 92]
    assert session.steps[1].action_ranges == [(5, 7)]


def test_run_turn_replays_retried_turn_idempotently(monkeypatch):
    _patch_prompts(monkeypatch, [[1, 2]])
    state, tp = _state([_act([90], [-0.1])])
    state.open("sid", "P", "l", None)
    session = state.sessions["sid"]
    r1 = asyncio.run(_run_turn(state, session, _MSG))
    r2 = asyncio.run(_run_turn(state, session, {"messages": [{"role": "user", "content": "P"}]}))
    assert r1 == r2 and len(session.steps) == 1 and len(tp.calls) == 1  # cached, no 2nd generate


def test_run_turn_ends_truncated_on_context_overflow(monkeypatch):
    _patch_prompts(monkeypatch, [list(range(5000))])
    state, tp = _state([_act([90], [-0.1])], max_length=100)
    state.open("sid", "P", "l", None)
    action, finish = asyncio.run(_run_turn(state, state.sessions["sid"], _MSG))
    assert (action, finish) == ("", "length")
    assert state.sessions["sid"].steps == [] and tp.calls == []


def test_run_turn_marks_truncated_on_length_finish(monkeypatch):
    _patch_prompts(monkeypatch, [[1]])
    state, _ = _state([_act([90, 91], [-0.1, -0.2], finish="length")])
    state.open("sid", "P", "l", None)
    _, finish = asyncio.run(_run_turn(state, state.sessions["sid"], _MSG))
    assert finish == "length" and state.sessions["sid"].steps[0].truncated is True


def test_run_turn_clamps_max_tokens_to_remaining(monkeypatch):
    _patch_prompts(monkeypatch, [[1, 2, 3, 4]])
    state, tp = _state([_act([90], [-0.1])], max_length=10)
    state.default_sampling = SimpleNamespace(max_tokens=None, temperature=1.0, top_p=1.0, logprobs=1, min_tokens=0)
    state.open("sid", "P", "l", None)
    asyncio.run(_run_turn(state, state.sessions["sid"], _MSG))
    assert tp.calls[0][2] == 6  # 10 - 4 prompt tokens


def test_run_turn_raises_on_logprobs_count_mismatch(monkeypatch):
    _patch_prompts(monkeypatch, [[1, 2]])
    state, _ = _state([_act([90, 91], [-0.1])])  # 2 action tokens, 1 logprob
    state.open("sid", "P", "l", None)
    with pytest.raises(RuntimeError, match="logprob"):
        asyncio.run(_run_turn(state, state.sessions["sid"], _MSG))


def test_run_turn_vlm_carries_pixel_values(monkeypatch):
    _patch_prompts(monkeypatch, [[999, 999, 7]], mm={"pixel_values": np.ones((1, 3, 4, 4))}, pil=["PIL"])
    monkeypatch.setattr(cs, "estimate_vllm_input_expansion_delta", lambda *a, **k: 5)
    state, tp = _state([_act([90], [-0.1])])
    state.open("sid", "P", "l", None)
    msg = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "u"}}]}]}
    asyncio.run(_run_turn(state, state.sessions["sid"], msg))
    traj = state.sessions["sid"].steps[0]
    assert traj.observation_tokens[:3] == [999, 999, 7]
    assert traj.mm_train_inputs["pixel_values"].shape[0] == 1 and traj.image_budget == 5
    assert tp.calls[0][1] == {"image": ["PIL"]}  # images forwarded to generate as multi_modal_data


def test_run_turn_absorbs_unified_routing_by_position(monkeypatch):
    _patch_prompts(monkeypatch, [[1, 2, 3]])
    routed = [[10], [11], [12], [20], [21]]  # unified [tokens,layer,topk]: 3 prompt + 2 action rows
    state, _ = _state([_act([90, 91], [-0.1, -0.2], routed=routed)])
    state.open("sid", "P", "l", None)
    asyncio.run(_run_turn(state, state.sessions["sid"], _MSG))
    assert [r[0] for r in state.sessions["sid"].steps[0].routed_experts] == [10, 11, 12, 20, 21]


# ---------------------------------------------------------------------------
# _content_to_text_and_images — OpenAI content -> (flat text w/ <image>, PIL list).
# ---------------------------------------------------------------------------
def test_content_flatten_string_passthrough():
    assert _content_to_text_and_images("hello") == ("hello", [])


def test_content_flatten_collects_images(monkeypatch):
    monkeypatch.setattr(cs, "load_images", lambda url: ["PIL"])
    text, pil = _content_to_text_and_images(
        [{"type": "text", "text": "look:"}, {"type": "image_url", "image_url": {"url": "u"}}]
    )
    assert text == "look:<image>" and pil == ["PIL"]


# ---------------------------------------------------------------------------
# stitch_session: per-turn step-samples share the terminal reward; prefix merge.
# ---------------------------------------------------------------------------
def test_stitch_drops_stepless_session():
    # Opened but never generated (e.g. the first prompt alone exceeded max_length) -> no trajectories
    # rather than an error; run_group then drops the rollout.
    state, _ = _state([])
    state.open("sid", "p", "l", None)
    assert stitch_session(state, "sid", Result(reward=0.0)) == []


def test_stitch_raises_on_unknown_session():
    state, _ = _state([])
    with pytest.raises(RuntimeError, match="Unknown session"):
        stitch_session(state, "missing", Result(reward=0.0))


def test_stitch_stamps_reward_across_all_segments(monkeypatch):
    _patch_prompts(monkeypatch, [[1, 2], [7, 8]])
    state, _ = _state([_act([90], [-0.1]), _act([91], [-0.2])])
    state.open("sid", "P", "l", None)
    session = state.sessions["sid"]
    asyncio.run(_run_turn(state, session, _MSG))
    asyncio.run(_run_turn(state, session, {"messages": [{"role": "user", "content": "s"}]}))
    out = stitch_session(state, "sid", Result(reward=1.0, score=0.5))
    assert len(out) == 2 and all(t.reward == 1.0 and t.scores == 0.5 for t in out)


def test_stitch_merges_prefix_extending_turns(monkeypatch):
    _patch_prompts(monkeypatch, [[1, 2], [1, 2, 90, 5, 6]])
    state, _ = _state([_act([90], [-0.1]), _act([91, 92], [-0.2, -0.3])])
    state.open("sid", "P", "l", None)
    session = state.sessions["sid"]
    asyncio.run(_run_turn(state, session, _MSG))
    asyncio.run(_run_turn(state, session, {"messages": [{"role": "user", "content": "P2"}]}))
    out = stitch_session(state, "sid", Result(reward=1.0))
    assert len(out) == 1
    assert out[0].observation_tokens == [1, 2, 90, 5, 6, 91, 92]  # each token once
    assert out[0].action_ranges == [(2, 3), (5, 7)]
    assert out[0].rollout_log_probs == [0.0, 0.0, -0.1, 0.0, 0.0, -0.2, -0.3]


# ---------------------------------------------------------------------------
# Wire codecs — Anthropic decode + reply encoders (transport-independent).
# ---------------------------------------------------------------------------
def test_chat_completion_body_normalizes_finish_reason():
    for raw in ("abort", "error", "repetition", None):
        assert _chat_completion_body("policy", "x", raw)["choices"][0]["finish_reason"] == "stop"
    for raw in ("stop", "length", "tool_calls"):
        assert _chat_completion_body("policy", "x", raw)["choices"][0]["finish_reason"] == raw


def test_decode_anthropic_normalizes_to_openai_shape():
    body = {
        "system": "sys",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "look:"},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                ],
            },
        ],
    }
    out = _decode_anthropic(body)
    assert all(m["role"] != "system" for m in out["messages"]) and len(out["messages"]) == 2
    blocks = out["messages"][1]["content"]
    assert blocks[1] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def test_decode_anthropic_unwraps_tool_result():
    body = {
        "max_tokens": 64,
        "messages": [{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "4"}]}],
    }
    out = _decode_anthropic(body)
    assert out["messages"][0]["content"] == [{"type": "text", "text": "4"}]


def test_anthropic_message_body_maps_stop_reason():
    for raw, want in (("stop", "end_turn"), ("length", "max_tokens"), ("tool_calls", "tool_use")):
        assert _anthropic_message_body("policy", "x", raw)["stop_reason"] == want


def test_response_bodies_parse_with_real_sdk_models():
    chat_types = pytest.importorskip("openai.types.chat")
    anthropic_types = pytest.importorskip("anthropic.types")
    completion = chat_types.ChatCompletion.model_validate(_chat_completion_body("policy", "hi", "abort"))
    assert completion.choices[0].message.content == "hi"
    message = anthropic_types.Message.model_validate(_anthropic_message_body("policy", "hi", "length"))
    assert message.content[0].text == "hi" and message.stop_reason == "max_tokens"
