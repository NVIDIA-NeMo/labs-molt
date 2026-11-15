# Agent contract

An agent is the program that turns a prompt into a reward. Every RL run points at one Python
module with `--train.agent_path`, and that module exports a class named `AgentRunner`. There are
two ways to write one.

> Source: [`molt/agents/base.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/agents/base.py) · [`molt/agents/chat_agent.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/agents/chat_agent.py) · [`molt/agents/_chat_server.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/agents/_chat_server.py)

| | `Env` | `ChatAgent` |
|---|---|---|
| Who runs the model loop | Molt | Your code, through the OpenAI or Anthropic SDK |
| You implement | `step(state)` | `run(ctx)` |
| Bind it with | `StepEnvRunner` | `ChatAgentRunner` |
| Reach for it when | The task is "score this turn, maybe reply" | You already have an agent or harness that calls a chat API |

## Env: Molt runs the loop

```python
from molt.agents import Env, Result, StepEnvRunner


class MathEnv(Env):
    async def step(self, state) -> Result:
        reward = grade(state["action_text"], state["label"])
        return Result(reward=reward, terminated=True)


class AgentRunner(StepEnvRunner):
    def __init__(self):
        super().__init__(MathEnv)
```

Molt drives vLLM, tokenization, multimodal accounting, and per-turn budgets. After each model
turn it calls `step`, and it chains turns until the result is `terminated` or `truncated`.

| `state` key | Holds |
|---|---|
| `observation_text` | The context so far |
| `action_text` | The model's latest generation |
| `label` | The ground truth from `--data.label_key` |
| `sampling_params` | This turn's sampling parameters |

For a multi-turn task, return the next observation and keep the episode open:

```python
return Result(reward=0.0, observation=tool_output, terminated=False)
```

Two optional hooks complete the interface. `reset(state)` can rewrite the first observation
before the episode starts. `close()` runs on every exit path, including an aborted episode, so use
it to release subprocesses or VMs.

## ChatAgent: you run the loop

```python
from openai import AsyncOpenAI
from molt.agents import ChatAgent, ChatAgentRunner, ChatContext, Result


class MyAgent(ChatAgent):
    async def run(self, ctx: ChatContext) -> Result:
        client = AsyncOpenAI(base_url=ctx.base_url, api_key=ctx.api_key)
        resp = await client.chat.completions.create(
            model=ctx.model_name,
            messages=[{"role": "user", "content": ctx.prompt}],
            max_tokens=ctx.sampling_params.max_tokens,
            temperature=ctx.sampling_params.temperature,
        )
        return Result(reward=grade(resp.choices[0].message.content, ctx.label))


class AgentRunner(ChatAgentRunner):
    def __init__(self):
        super().__init__(MyAgent)
```

`ctx.base_url` carries the session, and the server behind it records every token with its
log-probability. The agent needs no `extra_body`, no `logprobs=True`, and no session plumbing.
Nothing in it refers to training.

The same server speaks the Anthropic wire. Point the client at `ctx.session_url`, the session root
without `/v1`:

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic(base_url=ctx.session_url, api_key=ctx.api_key)
msg = await client.messages.create(
    model=ctx.model_name,
    messages=[{"role": "user", "content": ctx.prompt}],
    max_tokens=ctx.sampling_params.max_tokens,
)
```

The server is launched on loopback under the session URL. External callers, such as browser
automation or an eval harness, reach the same engine through `/v1/chat/completions` or
`/v1/messages`.

Chat agents need `--data.apply_chat_template`: the dataset rows are handed to the server as chat
messages and rendered once with the model's own template. An agent that stops on its own turn cap
should return `Result(truncated=True)`. The fields of `ChatContext` are listed in the
[molt.agents API](api-agents.md).

## Result

| Field | Default | Meaning |
|---|---|---|
| `reward` | `0.0` | Scalar reward consumed by the trainer |
| `observation` | `""` | Next-turn observation text, for multi-turn `Env` tasks |
| `terminated` | `True` | The episode finished naturally |
| `truncated` | `False` | The episode was cut off: turn cap, length, and so on |
| `info` | `{}` | Numeric values are logged as metrics; strings stay on the sample as feedback |
| `score` | `None` | Score for dynamic filtering and dashboards; defaults to `reward` |
| `images` | `None` | Next-turn images |
| `sampling_params` | `None` | Per-turn override |

## Retries and failures

The server samples every request it receives, in training and in eval. What to do about a bad
reply is the agent's decision.

| Situation | What happens |
|---|---|
| The agent re-sends a turn | The turn is sampled again. The reply it replaces stays in the rollout as its own segment and shares the rollout's reward. |
| A reply hits `max_tokens` | It returns with `finish_reason="length"`. The agent ends the episode, re-sends, or continues. |
| The client hangs up before the reply arrives | Nothing is recorded. |
| A system error: 5xx or a dropped connection | Molt retries 20 times, a second apart, then fails the rollout. |
| A training group loses a rollout | The whole group is dropped and its slot refilled with a fresh prompt. Eval keeps the rollouts it has. |

## Context compaction

A long-horizon agent often rewrites its own history: it summarizes old turns or drops old
screenshots to stay under the context window. A chat template can do the same, for example by
re-rendering an earlier assistant turn without its thinking block.

The server detects this. When a request rewrites the prefix instead of extending it, the server
seals the current segment and starts a new token-exact one. One rollout then yields several
segments that share its reward and `rollout_id`. Group baselines count one reward per rollout,
and every segment still contributes its generated tokens to the gradient.

No agent-side change is needed. This also covers external harnesses whose compaction logic Molt
cannot see.

## Chat templates

Assistant spans, for the SFT loss mask and for stitching multi-turn rollouts, are derived from the
model's own chat template. No markers are hard-coded. This is verified on ChatML (Qwen3.x,
Nemotron-Omni), Kimi-K2.6, GLM, Gemma, and DeepSeek.

## Reference agents

| File | Path | Task |
|---|---|---|
| `examples/python/agents/math.py` | `Env` | Single-turn boxed-answer grader |
| `examples/python/agents/geo3k.py` | `Env` | VLM, multi-turn, Python tool |
| `examples/python/agents/chat_minimal.py` | `ChatAgent` | Minimal chat loop |
| `examples/python/agents/chat_geo3k.py` | `ChatAgent` | VLM, multi-turn, Python tool |
