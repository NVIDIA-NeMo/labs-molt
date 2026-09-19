# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Narrow vLLM rollout overrides used by the Qwen alignment configuration."""

import os


class _NoBatchInvariantEnv:
    """Delegate vLLM environment settings except the local BI selector."""

    def __init__(self, source):
        self._source = source

    def __getattr__(self, name):
        if name == "VLLM_BATCH_INVARIANT":
            return False
        return getattr(self._source, name)


def _disable_local_bi_selector(module) -> None:
    if not getattr(module, "_molt_local_bi_selector_disabled", False):
        module.envs = _NoBatchInvariantEnv(module.envs)
        module._molt_local_bi_selector_disabled = True


def install_rollout_perf_overrides() -> None:
    """Keep FA4 and native GEMM for the opt-in Qwen rollout path."""
    if os.environ.get("MOLT_ALIGNMENT_ROLLOUT_PERF") != "1":
        return

    from vllm.model_executor.layers import linear, vocab_parallel_embedding
    from vllm.v1.attention.backends import fa_utils

    for module in (linear, vocab_parallel_embedding, fa_utils):
        _disable_local_bi_selector(module)
    print("[Alignment] enabled native-GEMM and FA4 rollout performance overrides.")
