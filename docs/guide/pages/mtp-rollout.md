# MTP rollout

Checkpoints that ship a multi-token-prediction head, such as Qwen3.6-MoE, can use it to speed up
generation through vLLM speculative decoding.

> Source: [`molt/cli/train_rl_ray.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/cli/train_rl_ray.py)

## Turn it on

```bash
--vllm.mtp_num_speculative_tokens 1
```

`0` is off and the default. `1` is a good starting point.

## How it works

vLLM detects the draft model from the served checkpoint, drafts N tokens, and the target model
verifies each one by rejection sampling. Accepted tokens follow the target policy's distribution.
The rollout log-probs therefore stay unbiased: the setting changes throughput, never the learned
policy.

## Limits

- **Rollout only.** The RL and SFT losses use the main head, and Molt does not train the MTP head.
  The draft shares the embeddings and the output head with the target, which are refreshed at
  every weight sync, so it tracks the updating policy. Only the small MTP block stays at its
  checkpoint weights.
- **Model support is on the vLLM side.** Qwen3.6-MoE works out of the box. vLLM raises an error at
  engine start if the setting is enabled on a checkpoint without a supported MTP head. The
  official Nemotron-Nano-Omni checkpoint ships none, so MTP rollout is unavailable for it until
  upstream adds support.
