import asyncio
from types import SimpleNamespace

import pytest
import torch

from molt.agents.base import Env, Result, StepEnvRunner, _extract_generation_logprobs


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors="pt"):
        return {"input_ids": torch.tensor([[ord(ch) for ch in text]], dtype=torch.long)}

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


class _OneStepEnv(Env):
    async def step(self, state, **kwargs):
        state["sampling_params"].max_tokens = 1
        return Result(reward=[2.0], score=[3.0], observation="!", terminated=True)


class _Engine:
    def __init__(self):
        self.seen_sampling_params = []

    async def generate(self, prompt_token_ids, sampling_params, multi_modal_data=None, session_id=None):
        self.seen_sampling_params.append(sampling_params)
        # generate() returns (RequestOutput, off_policy_len); off_policy_len=0 = no
        # mid-generation weight broadcast (on-policy), the partial_rollout-off case.
        return (
            SimpleNamespace(outputs=[SimpleNamespace(token_ids=[65], text="A", finish_reason="stop", logprobs=None)]),
            0,
        )


def test_step_env_runner_isolates_sampling_params_per_trajectory():
    """Validates list-shaped reward/score unwrap to scalars without a dedicated
    normaliser, and that two concurrent rollouts do not share sampling-param state."""
    params = SimpleNamespace(max_tokens=8, logprobs=None)
    engine = _Engine()
    runner = StepEnvRunner(_OneStepEnv)

    async def _run():
        return await asyncio.gather(
            runner.execute("p", "l", params, 64, _Tokenizer(), engine),
            runner.execute("p", "l", params, 64, _Tokenizer(), engine),
        )

    outputs = asyncio.run(_run())

    assert params.max_tokens == 8
    assert len({id(item) for item in engine.seen_sampling_params}) == 2
    assert [output.reward for output in outputs] == [2.0, 2.0]
    assert [output.scores for output in outputs] == [3.0, 3.0]


def test_generation_logprobs_fail_fast_when_vllm_omits_them():
    with pytest.raises(RuntimeError, match="did not return token logprobs"):
        _extract_generation_logprobs([1], None)
