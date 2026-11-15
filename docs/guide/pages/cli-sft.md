# `python -m molt.cli.train_sft`

> Generated from `molt/cli/train_sft.py` by `docs/guide/build_reference.py`. Edit the source, then rerun the script.

## `--fsdp.*`

| Flag | Default | Description |
|---|---|---|
| `--fsdp.tp_size` | `1` | Tensor parallel size |
| `--fsdp.cp_size` | `1` | Context parallel size (replaces ring-attn) |
| `--fsdp.ep_size` | `1` | Expert parallel size (MoE) |
| `--fsdp.pp_size` | `1` | Pipeline parallel size |
| `--fsdp.sequence_parallel` | `False` | Opt into sequence parallelism within the TP region. Off by default (matches AutoModel's omni / Qwen3.5-MoE recipes). |
| `--fsdp.offload` | `"none"` | CPU-offload level — a nested progression, not orthogonal toggles (FSDP param offload inherently offloads the optimizer too). 'none': everything on GPU. 'optimizer': run the AdamW step on CPU so the fp32 master + Adam moments never occupy GPU during the step (shrinks the optimizer-step peak, the binding one at long context); params stay on GPU for the forward, so it's safe on Qwen3.6 MoE; AdamW only; numerically equivalent to a GPU step. 'full': FSDP2 CPUOffloadPolicy also streams the \*params\* to CPU (maximal saving, but breaks Qwen3.6 MoE and slows the forward). |
| `--fsdp.param_dtype` | `"bf16"` | Model data type Choices: `bf16`, `fp16`. |
| `--fsdp.attn_implementation` | `"flash_attention_2"` | Attention implementation (e.g., sdpa, eager, flex, te, flash_attention_2) |
| `--fsdp.packing_samples` | `False` |  |

## Other flags

| Flag | Default | Description |
|---|---|---|
| `--optim` | `"adam"` |  Choices: `adam`, `muon`. |
| `--lr_scheduler` | `"constant"` |  |
| `--lr_warmup_ratio` | `0.03` |  |
| `--min_lr_ratio` | `0.1` |  |
| `--max_norm` | `1.0` | Gradient clipping |
| `--local_rank` | `-1` | local_rank from torchrun |
| `--use_ms` | `False` | Resolve models from ModelScope hub. |

## `--muon.*`

| Flag | Default | Description |
|---|---|---|
| `--muon.lr` | `0.02` | LR for the Muon 2D-weight group |
| `--muon.momentum` | `0.95` |  |
| `--muon.weight_decay` |  | Weight decay for the Muon 2D-weight group |
| `--muon.ns_steps` | `5` | Newton-Schulz steps for Muon updates |
| `--muon.nesterov` | `True` |  |
| `--muon.no_nesterov` | `True` |  |

## `--adam.*`

| Flag | Default | Description |
|---|---|---|
| `--adam.lr` | `5e-06` |  |
| `--adam.betas` | `(0.9, 0.95)` |  |
| `--adam.eps` | `1e-08` |  |
| `--adam.weight_decay` | `0.0` |  |

## `--ckpt.*`

| Flag | Default | Description |
|---|---|---|
| `--ckpt.output_dir` | `"./ckpt"` | Directory for the final consolidated HF safetensors export. |
| `--ckpt.save_steps` | `-1` | Save a resumable checkpoint every N steps; -1 disables it. |
| `--ckpt.save_hf` | `False` | Also export an HF-format snapshot alongside each resumable checkpoint. |
| `--ckpt.disable_final_save` | `False` | Skip the final consolidated HF export after training. Useful for smoke tests. |
| `--ckpt.path` | `"./ckpt/checkpoints_sft"` | Directory for resumable FSDP/DCP training checkpoints (optimizer + scheduler state). |
| `--ckpt.max_num` | `3` | How many checkpoints to keep; older ones pruned (\<=0 keeps all). Applies to the HF exports (--ckpt.save_hf, versioned under ckpt.path/_hf/\<step\>) and, unless --ckpt.dcp_max_num is set, to the DCP resumable checkpoints too. |
| `--ckpt.dcp_max_num` |  | Override --ckpt.max_num for DCP resumable checkpoints only; unset -\> --ckpt.max_num. DCP checkpoints carry optimizer state and are large, so keep this small (e.g. 2) while --ckpt.max_num keeps many cheap HF exports for eval. |
| `--ckpt.max_mem` | `inf` | Disk budget in GB for retained checkpoints; inf = no limit. |
| `--ckpt.load_enable` | `False` | Resume from the latest checkpoint in --ckpt.path if present. |

## `--logger.*`

| Flag | Default | Description |
|---|---|---|
| `--logger.logging_steps` | `1` | Log metrics every N steps. |
| `--logger.wandb.key` |  |  |
| `--logger.wandb.org` |  |  |
| `--logger.wandb.group` |  |  |
| `--logger.wandb.project` | `"molt_train_sft"` |  |
| `--logger.wandb.run_name` | `"sft_<MMDD>T<HH:MM>"` |  |
| `--logger.tensorboard_dir` |  | TensorBoard logging path |

## `--model.*`

| Flag | Default | Description |
|---|---|---|
| `--model.model_name_or_path` |  |  |
| `--model.gradient_checkpoint` | `"full"` | Activation-checkpointing mode (string): 'full' = full-block AC (AutoModel recipe default), 'selective' = TorchTitan per-op AC, 'none'/'off'/'' = disable. |
| `--model.aux_loss_coef` | `0` | MoE balancing loss |
| `--model.freeze_visual_encoder` | `False` | VLM only: freeze the vision encoder + projector and train the language backbone only (AutoModel's finetune recipe defaults to freezing the vision tower). cp_size\>1 forces this on. |
| `--model.lora_dim` | `0` | LoRA rank; 0 disables LoRA. |
| `--model.lora_alpha` | `32` | LoRA alpha; scale = alpha / dim. |
| `--model.lora_target_modules` |  | Wildcard patterns matched against FULL module names (AutoModel ModuleMatcher), e.g. '\*.q_proj' '\*.v_proj' — bare leaf names like 'q_proj' match nothing. Default: all linear layers (match_all_linear). On MoE models, name targets explicitly to keep the router gate out of the patch set. |

## `--data.*`

| Flag | Default | Description |
|---|---|---|
| `--data.dataset` |  | Path to the training dataset |
| `--data.dataset_probs` |  | Sampling probabilities for training datasets |
| `--data.dataset_split` | `"train"` |  |
| `--data.max_samples` | `1000000` | Maximum number of samples to use |
| `--data.max_len` | `2048` | Max total sequence length (prompt + response) |
| `--data.input_key` | `"input"` | JSON dataset key |
| `--data.output_key` |  | Dataset column holding the assistant reply (SFT target). |
| `--data.tokenizer_chat_template` |  |  |
| `--data.train_on_last_turn_only` | `False` | Supervise only the last assistant turn (per-turn-flattened SFT; history assistants are context). |
| `--data.image_key` |  | Dataset key for per-sample image list. Set to enable VLM SFT; leave None for text-only. |
| `--data.max_images_per_prompt` | `4` | Cap on images per sample; over-budget samples are filtered out. |
| `--data.disable_fast_tokenizer` | `False` |  |
| `--data.dataloader_num_workers` | `0` | Number of dataloader workers for IO |

## `--train.*`

| Flag | Default | Description |
|---|---|---|
| `--train.micro_batch_size` | `8` | batch size per GPU |
| `--train.batch_size` | `128` | Global training batch size |
| `--train.max_epochs` | `2` |  |
| `--train.seed` | `42` |  |
| `--train.full_determinism_enable` | `False` | Enable reproducible behavior during distributed training |

## `--eval.*`

| Flag | Default | Description |
|---|---|---|
| `--eval.dataset` |  | Path to the evaluation dataset |
| `--eval.split` | `"train"` |  |
| `--eval.steps` | `-1` | Evaluate every N steps; -1 evaluates once per epoch. |
