# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def convert_to_torch_dtype(param_dtype: str) -> torch.dtype:
    """Map a param_dtype string ("bf16" / "fp16" / "fp32") to its torch.dtype."""
    mapping = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if param_dtype not in mapping:
        raise ValueError(f"Invalid param_dtype: {param_dtype}")
    return mapping[param_dtype]


def get_strategy(args):
    from molt.trainer.fsdp import FsdpStrategy

    return FsdpStrategy(
        seed=getattr(args.train, "seed", 42),
        full_determinism=getattr(args.train, "full_determinism_enable", False),
        max_norm=getattr(args, "max_norm", 1.0),
        micro_train_batch_size=getattr(args.train, "micro_batch_size", 1),
        train_batch_size=getattr(args.train, "batch_size", 128),
        args=args,
    )


def is_vlm_model(pretrain: str) -> bool:
    """Check if a pretrained model is a VLM by looking for vision_config in its HF config."""
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(pretrain, trust_remote_code=True)
        return hasattr(cfg, "vision_config")
    except Exception:
        return False


def get_tokenizer(pretrain, model, padding_side="left", use_fast=True):
    is_vlm = getattr(model, "is_vlm", False) if model is not None else False
    if not is_vlm:
        is_vlm = is_vlm_model(pretrain)

    if is_vlm:
        from transformers import AutoProcessor

        # AutoProcessor wraps tokenizer + image_processor; downstream code
        # detects VLM via hasattr(tokenizer, "image_processor").
        tokenizer = AutoProcessor.from_pretrained(pretrain, trust_remote_code=True)
        # AutoProcessor doesn't delegate tokenizer attributes, so set them on
        # the inner tokenizer and mirror the essentials back.
        inner = tokenizer.tokenizer
        inner.padding_side = padding_side
        if inner.pad_token is None:
            inner.pad_token = inner.eos_token
            inner.pad_token_id = inner.eos_token_id
        for attr in ("pad_token", "pad_token_id", "eos_token", "eos_token_id"):
            setattr(tokenizer, attr, getattr(inner, attr))
    else:
        tokenizer = AutoTokenizer.from_pretrained(pretrain, trust_remote_code=True, use_fast=use_fast)
        tokenizer.padding_side = padding_side
        # NOTE: When enable vLLM, do not resize_token_embeddings, or the vocab size will mismatch with vLLM.
        # https://github.com/facebookresearch/llama-recipes/pull/196
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    if model is not None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    return tokenizer


def zero_pad_sequences(
    sequences: List[torch.Tensor], side: str = "left", value: int = 0, stack: bool = False
) -> torch.Tensor:
    assert side in ("left", "right")
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - seq.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_sequences.append(F.pad(seq, padding, value=value))
    if stack:
        return torch.stack(padded_sequences, dim=0)
    else:
        return torch.cat(padded_sequences, dim=0)
