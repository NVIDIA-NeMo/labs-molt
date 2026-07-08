# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import io
from types import SimpleNamespace

import aiohttp
import numpy as np
import pytest

from molt.trainer.rollout.router import (
    RouterGenerateClient,
    _align_features_to_canonical,
    _decode_routed_experts,
)

GEN = "/inference/v1/generate"
RENDER = "/v1/chat/completions/render"


class _FakeResp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    async def json(self):
        return self._p


class _FakeCtx:
    def __init__(self, payload):
        self._p = payload

    async def __aenter__(self):
        return _FakeResp(self._p)

    async def __aexit__(self, *a):
        return False


class _FakeHttp:
    """Routes POSTs by URL (render vs generate) and records calls."""

    def __init__(self, by_url):
        self.by_url = by_url
        self.calls = []

    def post(self, url, json=None, headers=None):
        self.calls.append((url, json, headers))
        return _FakeCtx(self.by_url[url])


class _RaisingResp:
    def __init__(self, exc):
        self._exc = exc

    def raise_for_status(self):
        raise self._exc

    async def json(self):
        return {}


class _RaisingCtx:
    def __init__(self, exc):
        self._exc = exc

    async def __aenter__(self):
        return _RaisingResp(self._exc)

    async def __aexit__(self, *a):
        return False


class _FlakyHttp:
    """Fails the first ``fail_times`` POSTs with ``exc``, then serves ``payload``."""

    def __init__(self, payload, exc, fail_times):
        self.payload, self.exc, self.fail_times = payload, exc, fail_times
        self.attempts = 0

    def post(self, url, json=None, headers=None):
        self.attempts += 1
        return _RaisingCtx(self.exc) if self.attempts <= self.fail_times else _FakeCtx(self.payload)


def _sp():
    return SimpleNamespace(max_tokens=8, temperature=1.0, top_p=1.0)


def _npy_b64(arr):
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr))
    return base64.b64encode(buf.getvalue()).decode()


def _content(ids):
    # /inference/v1/generate logprobs=1 -> choice.logprobs.content[i] = {"token": "token_id:<id>", ...}
    return [{"token": f"token_id:{t}", "logprob": -0.1 * (i + 1)} for i, t in enumerate(ids)]


def _gen_resp(ids, routed=None):
    c = {"token_ids": ids, "finish_reason": {"type": "stop"}, "logprobs": {"content": _content(ids)}}
    if routed is not None:
        c["routed_experts"] = routed
    return {"choices": [c]}


def test_generate_text_parses_choice_and_unified_routing():
    # disagg carries a SINGLE unified [tokens,layer,topk] npy on the choice (prompt+gen by absolute
    # position); absorb_routing lays it down from 0. No separate prompt_routed_experts (that split +
    # JSON serialization was the /v1/completions R3 corruption). R3 return is enabled ENGINE-side, so
    # the client just decodes whatever routed_experts the response carries — no per-request flag.
    resp = _gen_resp([90, 91], routed=_npy_b64([[10], [11], [12], [20], [21]]))
    ro, off = asyncio.run(RouterGenerateClient(_FakeHttp({GEN: resp})).generate([1, 2, 3], _sp()))
    assert off == 0
    assert ro.outputs[0].token_ids == [90, 91]
    assert ro.prompt_routed_experts is None
    assert ro.outputs[0].routed_experts.reshape(-1).tolist() == [10, 11, 12, 20, 21]
    lp = ro.outputs[0].logprobs  # vLLM-shaped so _extract_generation_logprobs reads it unchanged
    assert lp[0][90].logprob == pytest.approx(-0.1) and lp[1][91].logprob == pytest.approx(-0.2)


def test_generate_posts_inference_endpoint_with_nested_sampling_params():
    # Nothing set on SamplingParams should be silently dropped; they nest inside sampling_params with
    # max_tokens + logprobs. skip_special_tokens is always sent; R3 off by default.
    http = _FakeHttp({GEN: _gen_resp([9])})
    sp = SimpleNamespace(max_tokens=8, temperature=0.7, top_p=0.9, top_k=20, min_tokens=5, skip_special_tokens=False)
    asyncio.run(RouterGenerateClient(http).generate([1, 2], sp))
    url, payload, _ = http.calls[0]
    assert url == GEN and payload["token_ids"] == [1, 2]
    spp = payload["sampling_params"]
    assert spp["max_tokens"] == 8 and spp["logprobs"] == 1
    assert (spp["temperature"], spp["top_p"], spp["top_k"], spp["min_tokens"]) == (0.7, 0.9, 20, 5)
    assert spp["skip_special_tokens"] is False
    assert "enable_return_routed_experts" not in spp


def test_generate_raises_on_missing_logprobs():
    resp = {"choices": [{"token_ids": [90], "finish_reason": "stop"}]}
    with pytest.raises(RuntimeError, match="logprobs.content"):
        asyncio.run(RouterGenerateClient(_FakeHttp({GEN: resp})).generate([1, 2], _sp()))


def test_generate_raises_on_logprobs_count_mismatch():
    resp = {"choices": [{"token_ids": [90, 91], "finish_reason": "stop", "logprobs": {"content": _content([90])}}]}
    with pytest.raises(RuntimeError, match="!= completion tokens"):
        asyncio.run(RouterGenerateClient(_FakeHttp({GEN: resp})).generate([1, 2], _sp()))


def test_post_retries_transient_then_succeeds(monkeypatch):
    # A refused/dropped connection (router/engine still warming up) is retried, not dropped.
    async def _no_sleep(*_a, **_k):
        pass

    monkeypatch.setattr("molt.trainer.rollout.router.asyncio.sleep", _no_sleep)
    http = _FlakyHttp(_gen_resp([90]), aiohttp.ClientConnectionError("engine warming up"), fail_times=2)
    ro, off = asyncio.run(RouterGenerateClient(http).generate([1, 2], _sp()))
    assert http.attempts == 3 and ro.outputs[0].token_ids == [90]  # 2 transient failures, 3rd succeeds


def test_post_fails_fast_on_4xx(monkeypatch):
    # A 4xx is a real client bug, not a transient — fail immediately without burning the retry budget.
    async def _no_sleep(*_a, **_k):
        pass

    monkeypatch.setattr("molt.trainer.rollout.router.asyncio.sleep", _no_sleep)
    http = _FlakyHttp(_gen_resp([90]), aiohttp.ClientResponseError(None, (), status=400), fail_times=10)
    with pytest.raises(aiohttp.ClientResponseError):
        asyncio.run(RouterGenerateClient(http).generate([1, 2], _sp()))
    assert http.attempts == 1


def test_align_features_to_canonical_uses_image_token_run():
    # render's mm_placeholders are DISCARDED (omni3 over-counts + its offset points at IMG_START, not
    # the placeholder). We take offset+length from the run of image_token_id in the canonical prompt.
    features = {"mm_placeholders": {"image": [{"offset": 8, "length": 274}]}}  # render's bogus range
    _align_features_to_canonical(features, canonical=[1, 18, 18, 2], image_token_id=18)
    assert features["mm_placeholders"]["image"][0] == {"offset": 1, "length": 2}


def test_generate_vlm_renders_realigns_then_generates():
    from PIL import Image

    render_payload = {
        "token_ids": [5, 18, 18, 18, 6],  # render's over-counted expansion (3 placeholders)
        "features": {
            "mm_placeholders": {"image": [{"offset": 1, "length": 3}]},
            "kwargs_data": {"image": ["<b64-pixels>"]},
            "mm_hashes": {"image": ["h0"]},
        },
    }
    http = _FakeHttp({RENDER: render_payload, GEN: _gen_resp([90])})
    canonical = [1, 18, 18, 2]  # our HF prompt: 2 image placeholders
    ro, off = asyncio.run(
        RouterGenerateClient(http, image_token_id=18).generate(
            canonical, _sp(), multi_modal_data={"image": [Image.new("RGB", (4, 4))]}
        )
    )
    assert http.calls[0][0] == RENDER  # render first (server-side mm)
    gen_url, gen_payload, _ = http.calls[1]
    assert gen_url == GEN
    assert gen_payload["token_ids"] == canonical  # canonical HF ids, NOT render's 5-token prompt
    ph = gen_payload["features"]["mm_placeholders"]["image"][0]
    assert (ph["offset"], ph["length"]) == (1, 2)  # realigned to the canonical run
    assert gen_payload["features"]["kwargs_data"]["image"] == ["<b64-pixels>"]  # embeds relayed verbatim
    assert ro.outputs[0].token_ids == [90]


def test_generate_pins_render_and_generate_to_one_session():
    # render + generate for a VLM call MUST carry the SAME x-session-id so consistent_hash routes them
    # to one engine (a render mm-cache hit returns kwargs_data=None, resolvable only on that engine).
    from PIL import Image

    render_payload = {
        "token_ids": [5, 18, 6],
        "features": {
            "mm_placeholders": {"image": [{"offset": 1, "length": 1}]},
            "kwargs_data": {"image": ["b"]},
            "mm_hashes": {"image": ["h"]},
        },
    }
    http = _FakeHttp({RENDER: render_payload, GEN: _gen_resp([90])})
    asyncio.run(
        RouterGenerateClient(http, image_token_id=18).generate(
            [1, 18, 2], _sp(), multi_modal_data={"image": [Image.new("RGB", (4, 4))]}
        )
    )
    sid = (http.calls[0][2] or {}).get("x-session-id")
    assert sid and (http.calls[1][2] or {}).get("x-session-id") == sid


def test_align_features_multi_image_finds_separated_runs():
    # Two images: each maps to its own image_token_id run in the canonical (markers separate them).
    features = {"mm_placeholders": {"image": [{}, {}]}}  # 2 images; render offsets discarded
    _align_features_to_canonical(features, canonical=[0, 18, 18, 9, 18, 18, 0], image_token_id=18)
    ranges = features["mm_placeholders"]["image"]
    assert (ranges[0]["offset"], ranges[0]["length"]) == (1, 2)
    assert (ranges[1]["offset"], ranges[1]["length"]) == (4, 2)


def test_decode_routed_experts_handles_both_encodings():
    assert _decode_routed_experts(_npy_b64([[1], [2]])).tolist() == [[1], [2]]  # base64 .npy
    assert _decode_routed_experts([[3], [4]]).tolist() == [[3], [4]]  # nested JSON lists
