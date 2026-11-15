# LoRA

SFT and RL take the same three LoRA flags. A rank of `0`, the default, is plain full fine-tuning.

> Source: [`molt/cli/train_rl_ray.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/cli/train_rl_ray.py) · [`molt/cli/train_sft.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/molt/cli/train_sft.py)

| | SFT | RL |
|---|---|---|
| Rank | `--model.lora_dim` | `--actor.lora_dim` |
| Alpha | `--model.lora_alpha` | `--actor.lora_alpha` |
| Targets | `--model.lora_target_modules` | `--actor.lora_target_modules` |

## Turn it on

```bash
--model.lora_dim 8 --model.lora_alpha 32 \
--model.lora_target_modules '*.q_proj' '*.k_proj' '*.v_proj' '*.o_proj'
```

## Get these right

- **Targets match full module names.** A bare leaf name such as `q_proj` matches nothing. Omit the
  flag to patch every linear layer.
- **Raise the learning rate.** LoRA runs on bf16 master weights. The full fine-tuning defaults,
  `5e-6` for SFT and `1e-6` for RL, are rounded away and the adapters barely move. Use `1e-4` and
  up. Molt warns at startup.
- **RL trains the adapters only.** The reference model keeps the plain checkpoint, which equals
  the step-0 policy.

## Limits in RL

- **MoE experts and a tied output head are refused.** The weight sync merges the adapter into the
  base weight before broadcasting, and neither has such a mapping.
- **The merge rounds once in bf16.** The engine's weights differ from the trainer's by that bit.
  This is harmless under IS correction. A strictly on-policy run should know about it.

## Checkpoints

`--ckpt.save_hf` writes a PEFT adapter directory for LoRA runs: `adapter_model.safetensors` and
`adapter_config.json`. Load it with `PeftModel.from_pretrained(base, adapter_dir)` or vLLM's
`--enable-lora`. Resume checkpoints stay full-weight, so resuming is unchanged.
