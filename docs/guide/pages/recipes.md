# Recipes

The repository ships launch scripts for the workloads it is validated on. Each one is a complete
command, so the fastest way to start a new experiment is to copy the closest script and change
the paths.

> Source: [`examples/scripts/`](https://github.com/NVIDIA-NeMo/labs-molt/tree/main/examples/scripts) · [`examples/python/agents/`](https://github.com/NVIDIA-NeMo/labs-molt/tree/main/examples/python/agents)

## Scripts

| Workflow | Single node | Slurm |
|---|---|---|
| Qwen3-4B RL on text math | `quick_start/rl_qwen3_4b.sh` | `slurm/rl_qwen3_4b.sh` |
| Qwen3-4B SFT on text math | `quick_start/sft_qwen3_4b.sh` | `slurm/sft_qwen3_4b.sh` |
| Qwen3.6-35B-A3B VLM RL on geo3k, multi-turn Python tool | `quick_start/rl_qwen3_6_35b.sh` | `slurm/rl_qwen3_6_35b.sh` |
| Qwen3.6-35B-A3B VLM SFT on geo3k | `quick_start/sft_qwen3_6_35b.sh` | `slurm/sft_qwen3_6_35b.sh` |
| FlashREINFORCE on DeepSeek-R1-Distill-Qwen-1.5B | `quick_start/rl_flash_reinforce_r1d_1p5b.sh` | |
| Nemotron-Omni-30B-A3B VLM RL on geo3k, CP8 + EP8 | | `slurm/rl_omni3_30b.sh` |
| Nemotron-Omni-30B-A3B VLM SFT on geo3k | | `slurm/sft_omni3_30b.sh` |
| Nemotron-Omni-30B-A3B on-policy distillation | | `slurm/rl_distill_omni3_30b.sh` |
| Muse-Glimmer-30B dense RL, TP2 | | `slurm/rl_muse_glimmer_30b.sh` |
| GLM-5.2 RL on text math, EP256 | | `slurm/rl_glm5_2_753b.sh` |

All paths are under `examples/scripts/`.

## Run one

On a single node:

```bash
MODEL_PATH=/path/to/Qwen3-4B bash examples/scripts/quick_start/rl_qwen3_4b.sh
```

On Slurm:

```bash
sbatch examples/scripts/slurm/rl_qwen3_6_35b.sh
sbatch --nodes=4 examples/scripts/slurm/rl_qwen3_6_35b.sh
```

The first Slurm command is a smoke run on two nodes. The second scales the same script to four
nodes for convergence.

## Data

The geo3k scripts prepare their dataset on first run. To stage it yourself:

```bash
python3 examples/python/utils/prepare_geo3k.py --num-proc 8 --out-dir .tmp/geo3k
```

To use your own data, point `PROMPT_DATASET` and `EVAL_DATASET` at it.

## The tool-use recipe

`examples/python/agents/geo3k.py` is the multi-turn agent behind the Qwen3.6 RL scripts.

1. The model emits a `<tool_call>` that invokes `python_executor(code=...)`.
2. The agent runs the snippet in a sandboxed subprocess.
3. The captured output goes back as a `<tool_response>` turn.
4. The loop ends at `MAX_AGENT_TURNS` or at a final `<answer>`, which is graded against the ground
   truth and becomes the reward.

`chat_geo3k.py` is the same task written as a `ChatAgent`. See [Agent contract](agents.md).
