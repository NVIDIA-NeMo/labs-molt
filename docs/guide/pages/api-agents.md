# `molt.agents`

> Generated from `molt/agents/` by `docs/guide/build_reference.py`. Edit the source, then rerun the script.

```text
Molt agents — Gymnasium-aligned public surface.

User writes ONE of:
  - `class MyEnv(Env)`         with async step() / reset() returning `Result`
  - `class MyAgent(ChatAgent)`  with async run(ctx) returning `Result`

User binds it via a Runner subclass and exports as `AgentRunner`:
  - `class AgentRunner(StepEnvRunner): super().__init__(MyEnv)`
  - `class AgentRunner(ChatAgentRunner): super().__init__(MyAgent)`

For the chat-agent path, build your own client from the session URL — either
`openai.AsyncOpenAI(base_url=ctx.base_url)` or
`anthropic.AsyncAnthropic(base_url=ctx.session_url)`. We don't wrap the SDKs —
use whichever directly; the server speaks both wires against the same engine.
```

## `Result`

`class Result` in `molt/agents/base.py`

| Field | Type | Default | Note |
|---|---|---|---|
| `reward` | `float \| torch.Tensor` | `0.0` |  |
| `observation` | `str` | `''` | next-turn observation text |
| `terminated` | `bool` | `True` | episode finished naturally |
| `truncated` | `bool` | `False` | cut off externally (max turns etc.) |
| `info` | `dict` | `field(default_factory=dict)` |  |
| `score` | `float \| torch.Tensor \| None` | `None` | defaults to reward at trainer boundary |
| `images` | `list \| None` | `None` |  |
| `sampling_params` | `object` | `None` |  |

## `Trajectory`

`class Trajectory` in `molt/agents/base.py`

| Field | Type | Default | Note |
|---|---|---|---|
| `prompt` | `str` |  |  |
| `label` | `str` |  |  |
| `images` | `object` |  | mirrors the dict key "images" (was source_images) |
| `observation_text` | `str` |  |  |
| `observation_tokens` | `list` |  |  |
| `mm_train_inputs` | `object` | `None` |  |
| `pil_images` | `list` | `field(default_factory=list)` |  |
| `image_budget` | `int` | `0` |  |
| `action_ranges` | `list` | `field(default_factory=list)` |  |
| `off_policy_action_lens` | `list` | `field(default_factory=list)` |  |
| `rollout_log_probs` | `list \| None` | `None` |  |
| `rollout_top_logprobs` | `list \| None` | `None` |  |
| `routed_experts` | `list \| None` | `None` |  |
| `reward` | `float` | `0.0` |  |
| `scores` | `float` | `0.0` | mirrors the dict key "scores" (plural for legacy reasons) |
| `truncated` | `bool` | `False` |  |
| `extra_logs` | `dict` | `field(default_factory=dict)` |  |
| `group_id` | `str \| None` | `None` | one per prompt group (N rollouts); GRPO baseline averaging |
| `rollout_id` | `str \| None` | `None` | one per rollout; multi-turn step-samples dedup |

### `Trajectory.append_action`

```python
def append_action(self, action_tokens, action_logprobs=None, off_policy_len=0, action_top_logprobs=None)
```

### `Trajectory.append_feedback`

```python
def append_feedback(self, action_text: str, feedback_text: str, feedback_tokens)
```

### `Trajectory.absorb_routing`

```python
def absorb_routing(self, request_output)
```

```text
Place this turn's rollout MoE routing onto ``routed_experts`` (R3).

The rollout engine returns the per-token expert ids for the tokens it processed,
indexed from absolute position 0. The engine ships the generated-token ids on the
completion and, when it splits them out, the prompt ids on the request;
concatenating reconstructs the sequence (a unified-array engine returns it all on
the completion, with no prompt split).
We lay it down first-writer-wins so each token keeps the routing from the turn
that first produced it, leaving positions the engine gave no routing for as None
(a sentinel downstream, so the training forward routes them naturally). No-op when
capture is off. Call AFTER :meth:`append_action` so the generated positions exist.
```

## `Env`

`class Env(ABC)` in `molt/agents/base.py`

### `Env.reset`

```python
async def reset(self, state: dict, **kwargs) -> dict
```

```text
Optional per-episode setup; override to rewrite the initial observation.

Args:
    state: ``{"observation": <prompt str>, "label": <ground truth>}``.

Returns:
    The (possibly modified) state dict; ``state["observation"]`` becomes
    the first model input. Default is an identity passthrough.
```

### `Env.step`

```python
async def step(self, state: dict, **kwargs) -> Result
```

```text
Score one model turn and optionally produce next-turn feedback.

Args:
    state: keys ``observation_text`` (context so far), ``action_text``
        (the model's latest generation), ``label`` (ground truth), and
        ``sampling_params`` (this turn's params).

Returns:
    Result with a scalar ``reward`` (required). Optional: ``observation``
    (next-turn feedback text for multi-turn), ``score`` (defaults to
    reward), ``info`` (numeric values are logged as metrics; string values
    stay on the sample as textual feedback and are None on samples that omit
    them), ``images``, ``sampling_params``, and ``terminated`` / ``truncated``
    to end the episode.
```

### `Env.close`

```python
async def close(self)
```

```text
Optional per-episode teardown. StepEnvRunner calls this on every exit
path (episode end, context exhaustion, exception) — override to release
OS resources such as subprocesses or VMs. Default: no-op.

Contract: close() may run before reset() has succeeded (an aborted episode),
so it must tolerate partially-initialized state.
```

## `Runner`

`class Runner(ABC)` in `molt/agents/base.py`

### `Runner.execute`

```python
async def execute(self, prompt: str, label: Any, sampling_params, max_length: int, hf_tokenizer, llm_engine, images=None, tools=None) -> Trajectory
```

## `StepEnvRunner`

`class StepEnvRunner(Runner)` in `molt/agents/base.py`

### `StepEnvRunner.execute`

```python
async def execute(self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images=None, tools=None)
```

## `ChatContext`

`class ChatContext` in `molt/agents/chat_agent.py`

| Field | Type | Default | Note |
|---|---|---|---|
| `messages` | `list` |  |  |
| `tools` | `list \| None` |  |  |
| `prompt` | `str` |  | the last user turn's text — a scalar view of the task (grading/logging) |
| `label` | `Any` |  |  |
| `images` | `list \| None` |  |  |
| `base_url` | `str` |  |  |
| `session_url` | `str` |  |  |
| `model_name` | `str` |  |  |
| `api_key` | `str` |  | always "EMPTY" — session lives in base_url, not auth |
| `session_id` | `str` |  | raw id; rarely needed once base_url carries it |
| `sampling_params` | `Any` |  |  |
| `max_length` | `int` |  |  |

## `ChatAgent`

`class ChatAgent(ABC)` in `molt/agents/chat_agent.py`

### `ChatAgent.run`

```python
async def run(self, ctx: ChatContext) -> Result
```

```text
Run one episode against an OpenAI-compatible endpoint and score it.

Point ``openai.AsyncOpenAI(base_url=ctx.base_url, api_key=ctx.api_key)``
at the session URL; every call through it is traced and stitched into a
single training Trajectory automatically. Start from ``ctx.messages`` /
``ctx.tools`` (the dataset row, ready to send verbatim), grade against
``ctx.label``, and respect ``ctx.sampling_params`` / ``ctx.max_length``
for the generation budget.

Returns:
    Result with a scalar ``reward`` (required); ``score`` and ``info``
    are optional (same contract as ``Env.step``). Set ``truncated=True``
    when your own loop cuts the episode off (a turn cap, a wall-clock
    budget, ...): the server only detects generation-length / context
    truncation by itself, and the trainer's ``truncated_rate`` metrics
    read the OR of the two.
```

## `ChatAgentRunner`

`class ChatAgentRunner(Runner)` in `molt/agents/chat_agent.py`

### `ChatAgentRunner.execute`

```python
async def execute(self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, images=None, tools=None)
```
