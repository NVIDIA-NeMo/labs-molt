# MoE routing

In a mixture-of-experts model the rollout engine and the trainer pick experts independently. Even
at identical weights, numerical differences flip part of the top-k selection in each layer. The
flips compound until most tokens route to different experts than they did during rollout, which
breaks the assumption behind importance sampling. Molt closes the gap at three levels.

> Source: [`molt/models/base.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/models/base.py) · [`molt/agents/base.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/agents/base.py)

| Level | Default | What it fixes |
|---|---|---|
| fp32 router precision | on | The two sides agree on the gate weights |
| Routing replay (R3) | on in the Qwen3.5-MoE recipes | The trainer replays the engine's expert selection |
| Router freeze | off | The router cannot drift at all |

## Router precision

The gate linear and the expert-output combine run in fp32, matching vLLM's router. A bf16 router
drifts from vLLM and makes `vllm_kl` climb during training. Override with
`MOLT_GATE_PRECISION=bfloat16`.

## Routing replay

```bash
--train.routing_replay
```

vLLM returns the expert ids it chose for every token, and the training forward replays that exact
selection ([arXiv:2510.11370](https://arxiv.org/abs/2510.11370)).

- **It fixes the routing, not the router.** Only the discrete top-k selection is replayed. The
  router logits are recomputed from the live weights, so the router keeps learning.
- **It is aligned by absolute token position.** A position the engine returns no routing for keeps
  its natural selection.
- **It cannot be combined with `--train.partial_rollout_enable`**. Every weight sync preempts the
  requests still in flight, and vLLM frees the routing it captured for a preempted request.
- **It needs AutoModel's `RouterReplay`**, in `nemo_automodel.components.moe.router_replay`.

## Router freeze

```bash
--actor.freeze_moe_router
```

The router leaves the optimizer, so the engine and the trainer route to the same experts by
construction. No engine support is needed. The cost is that the router stops learning. It is
redundant with routing replay, so reach for it only when routing drift still dominates and a
fixed router is acceptable.
