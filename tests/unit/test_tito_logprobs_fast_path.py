# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

tito_serving = pytest.importorskip("vllm.entrypoints.scale_out.token_in_token_out.serving")

from vllm.logprobs import Logprob  # noqa: E402

from molt.trainer.vllm.vllm_engine import _fast_tokens_logprobs  # noqa: E402


def test_fast_tokens_logprobs_dumps_the_same_payload_as_vllm():
    token_ids = [5, 6, 7, 8]
    top = [
        {5: Logprob(-0.1, 1, None), 9: Logprob(-2.0, 2, None), 3: Logprob(-3.0, 3, None)},  # k=2 cuts the 3rd
        {8: Logprob(-0.3, 1, None), 6: Logprob(-1.0, 2, None)},  # sampled token is not the top-1
        {8: Logprob(-0.3, 1, None), 1: Logprob(-1.0, 2, None)},  # sampled token missing -> token only
        None,  # no logprobs at this position
    ]
    ref = tito_serving.ServingTokens._create_tokens_logprobs(SimpleNamespace(), token_ids, top, 2).model_dump()
    fast = _fast_tokens_logprobs(SimpleNamespace(), token_ids, top, 2).model_dump()
    assert fast == ref
