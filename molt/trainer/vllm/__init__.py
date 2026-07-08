# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy imports for vLLM engine helpers."""

__all__ = [
    "create_vllm_engines",
    "batch_vllm_engine_call",
]


def __getattr__(name):
    if name in __all__:
        from .vllm_engine import batch_vllm_engine_call, create_vllm_engines

        exports = {
            "batch_vllm_engine_call": batch_vllm_engine_call,
            "create_vllm_engines": create_vllm_engines,
        }
        return exports[name]
    raise AttributeError(name)
