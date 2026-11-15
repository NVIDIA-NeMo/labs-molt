# Scaling

The same entry point runs a 4B model on one node and a 753B MoE on 256 GPUs. What changes is the
checkpoint, the cluster size, and the parallelism flags on this page.

> Source: [`molt/cli/common_args.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/cli/common_args.py) · [`examples/scripts/slurm/rl_glm5_2_753b.sh`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/examples/scripts/slurm/rl_glm5_2_753b.sh)

## Trainer

FSDP2 shards the storage of every parameter. Three forms of parallelism divide the computation.

| Parallelism | Flag | What it splits | Reach for it when |
|---|---|---|---|
| Tensor | `--fsdp.tp_size 2` | Each weight matrix | A dense layer is too large for one GPU |
| Expert | `--fsdp.ep_size 8` | MoE experts across GPUs, with each token routed to its expert | The model is MoE. The largest ones use `256`. |
| Context | `--fsdp.cp_size 8` | One long sequence | Sequences reach 32K tokens and beyond |
| Optimizer offload | `--fsdp.offload optimizer` | Moves the Adam state to CPU memory | GPU memory is the limit |

Context parallelism is delegated to AutoModel, so each model gets the sharding its attention needs:
round-robin for hybrid SSM and linear-attention models such as Nemotron-Omni and Qwen3.5-MoE, and
flat THD streams for sparse-attention models such as GLM-5.2. Vision towers and routing replay
shard with the sequence, so `--fsdp.cp_size` composes with `--data.image_key` and
`--train.routing_replay`. Sample packing, `--fsdp.packing_samples`, is text-only and off by
default. Under context parallelism it takes the THD path.

## Model backend and optimizer

NeMo AutoModel is the primary path: native TP, EP, and CP, a custom MoE and EP parallelizer, and
TransformerEngine fused attention. The Hugging Face `transformers` path is a fallback that
AutoModel drops to only when a model has no native class. It supports text, `flash_attention_2`,
and packing, without TP, EP, or CP.

`--actor.optim adam` is the default and the recommendation. `muon` is experimental: it runs
distributed, but has shown no consistent win over `adam`.

## Rollout engines

| Setting | Flag | Notes |
|---|---|---|
| Engines | `--vllm.num_engines 2` | Required |
| Tensor parallel | `--vllm.tensor_parallel_size 2` | GPUs per engine |
| Expert parallel | `--vllm.enable_expert_parallel` | EP = TP × DP |
| Data parallel | `--vllm.data_parallel_size 4` | Raises EP past TP on a single node |
| Scheduler token budget | `--vllm.max_num_batched_tokens 32768` | |
| Speculative decoding | `--vllm.mtp_num_speculative_tokens 1` | See [MTP rollout](mtp-rollout.md) |
| MoE kernel backend | `--vllm.moe_backend triton` | The default lets vLLM choose |

## Two recipes, one command

| | Qwen3-4B | GLM-5.2 (753B) |
|---|---|---|
| Script | `examples/scripts/quick_start/rl_qwen3_4b.sh` | `examples/scripts/slurm/rl_glm5_2_753b.sh` |
| Trainer GPUs | 1 node | 32 nodes × 8 |
| `--fsdp.ep_size` | `1` | `256` |
| `--fsdp.cp_size` | `1` | `16` |
| `--train.routing_replay` | off | on |

The entry point, the agent, and the estimator are the same in both.

## MoE stability

Rollout and trainer must route each token to the same experts. See
[MoE routing](moe-routing.md) for routing replay and router freeze.
