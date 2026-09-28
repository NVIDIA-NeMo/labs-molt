#!/bin/bash
# The daily (GPU-only) variant of Launch_RL_2GPU.sh: Qwen3-30B-A3B-Instruct-2507 (MoE, non-thinking), 8k
# context, AdamW states on the CPU (--fsdp.offload optimizer) so the bf16 parameters and gradients of 30B fit
# the one actor GPU; activation checkpointing is already full in the base recipe. Same data, batch and checks.
MODEL_ID=Qwen/Qwen3-30B-A3B-Instruct-2507 MAX_LEN=8192 MAX_NEW_TOKENS=6144 FSDP_OFFLOAD=optimizer \
  exec bash "$(dirname "$0")/Launch_RL_2GPU.sh" "$@"
