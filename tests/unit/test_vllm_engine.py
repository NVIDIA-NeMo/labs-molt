# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import dataclass
from types import ModuleType


def _install_vllm_test_stub():
    vllm = ModuleType("vllm")
    vllm.__version__ = "0.21.0"

    @dataclass
    class AsyncEngineArgs:
        model: str
        dtype: str = "auto"
        enforce_eager: bool = False

    vllm.AsyncEngineArgs = AsyncEngineArgs
    vllm.AsyncLLMEngine = object

    inputs = ModuleType("vllm.inputs")

    class TokensPrompt(dict):
        pass

    inputs.TokensPrompt = TokensPrompt

    utils = ModuleType("vllm.utils")
    utils.random_uuid = lambda: "test-request-id"

    sys.modules["vllm"] = vllm
    sys.modules["vllm.inputs"] = inputs
    sys.modules["vllm.utils"] = utils


try:
    import vllm.inputs  # noqa: F401
except Exception:
    _install_vllm_test_stub()

import molt.trainer.vllm.vllm_engine as vllm_engine


def test_vllm_ray_executor_uses_worker_gpu_even_when_actor_is_cpu_only():
    assert vllm_engine._vllm_worker_num_gpus("ray", 0) == 1
    assert vllm_engine._vllm_worker_num_gpus("mp", 8) == 8
    assert vllm_engine._vllm_worker_num_gpus("uni", 1) == 1


def test_format_ray_gpu_ids_keeps_all_visible_devices():
    assert vllm_engine._format_ray_gpu_ids([0.0, 2.0]) == "0,2"
    assert vllm_engine._format_ray_gpu_ids(["GPU-abc"]) == "GPU-abc"


def test_filter_vllm_engine_kwargs_drops_unsupported_optional_args(monkeypatch):
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        dtype: str = "auto"
        enforce_eager: bool = False

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs(
        {
            "model": "model-path",
            "dtype": "bfloat16",
            "gdn_prefill_backend": "triton",
        }
    )

    assert filtered == {"model": "model-path", "dtype": "bfloat16"}


def test_filter_vllm_engine_kwargs_keeps_speculative_config(monkeypatch):
    # MTP rollout passes speculative_config; it is a real AsyncEngineArgs field,
    # so it must survive the whitelist filter (not be dropped like unknown kwargs).
    @dataclass
    class FakeAsyncEngineArgs:
        model: str
        dtype: str = "auto"
        speculative_config: object = None

    monkeypatch.setattr(vllm_engine.vllm, "AsyncEngineArgs", FakeAsyncEngineArgs)

    filtered = vllm_engine._filter_vllm_engine_kwargs(
        {"model": "m", "speculative_config": {"num_speculative_tokens": 1}}
    )

    assert filtered == {"model": "m", "speculative_config": {"num_speculative_tokens": 1}}


def test_ray_visible_device_flag_is_cuda_only():
    assert vllm_engine.ray_noset_visible_devices({"RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"})
    assert not vllm_engine.ray_noset_visible_devices({"RAY_EXPERIMENTAL_NOSET_OTHER_VISIBLE_DEVICES": "1"})
