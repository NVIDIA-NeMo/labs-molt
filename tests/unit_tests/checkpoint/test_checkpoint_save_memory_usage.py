# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import gc
import os
import sys
import threading
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Ensure we test the in-repo sources even if an older wheel is installed.
# `pytest` (entrypoint script) can sometimes import site-packages before CWD,
# while `python -m pytest` typically prefers the local working directory.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pytest
import torch
import torch.distributed.checkpoint as dcp

from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageWriter


def _rss_bytes() -> int:
    """Best-effort current process RSS (bytes).

    Uses Linux `/proc` when available for an instantaneous RSS sample.
    Falls back to `resource.getrusage` when `/proc` is unavailable.
    """
    # Linux fast-path.
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            parts = f.read().strip().split()
        # statm: size resident shared text lib data dt
        if len(parts) >= 2:
            resident_pages = int(parts[1])
            return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except Exception:
        pass

    # Portable fallback (max RSS so far).
    try:
        import resource

        rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB, macOS reports bytes.
        if sys.platform == "darwin":
            return rss
        return rss * 1024
    except Exception:
        return 0


@dataclass(frozen=True)
class _MemoryPeaks:
    rss_peak_bytes: int
    cuda_peak_allocated_bytes: Optional[int] = None
    cuda_peak_reserved_bytes: Optional[int] = None
    python_peak_bytes: Optional[int] = None


class _RssSampler:
    """Background sampler to catch short-lived RSS spikes."""

    def __init__(self, interval_s: float = 0.001) -> None:
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss_bytes: int = 0

    def start(self) -> None:
        self.peak_rss_bytes = _rss_bytes()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=10)
        self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_bytes = max(self.peak_rss_bytes, _rss_bytes())
            # Using Event.wait provides a responsive stop without busy-looping.
            self._stop.wait(self._interval_s)


def _payload_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    return int(sum(t.numel() * t.element_size() for t in state_dict.values()))

def _max_tensor_bytes(state_dict: dict[str, torch.Tensor]) -> int:
    return int(max(t.numel() * t.element_size() for t in state_dict.values()))


def _make_large_state_dict(*, device: torch.device, payload_mb: int, include_bf16: bool) -> dict[str, torch.Tensor]:
    """Create a state_dict of roughly `payload_mb` MiB total payload."""
    payload_bytes = payload_mb * 1024**2
    half = payload_bytes // 2

    t0 = torch.empty((half // 4,), device=device, dtype=torch.float32)
    # Touch pages to ensure RSS reflects the allocation (avoids false deltas during save).
    t0.fill_(0.25)

    if include_bf16:
        t1 = torch.empty((half // 2,), device=device, dtype=torch.bfloat16)
        t1.fill_(1)
    else:
        t1 = torch.empty((half // 2,), device=device, dtype=torch.float16)
        t1.fill_(1)

    sd = {"t0": t0, "t1": t1}
    assert _payload_bytes(sd) >= int(payload_bytes * 0.98)  # sanity: close to requested payload
    return sd


def _make_many_small_tensors_state_dict(
    *,
    device: torch.device,
    total_payload_mb: int,
    per_tensor_mb: int,
    include_bf16: bool,
) -> dict[str, torch.Tensor]:
    """Create a state_dict with many small tensors (better for detecting full-bytes materialization).

    If a buggy implementation materializes the *entire* safetensors file as one `bytes`,
    RSS tends to spike by ~total payload.

    In contrast, a streaming implementation should only need memory proportional to
    the largest in-flight tensor (plus small overhead), even if there are many tensors.
    """
    total_bytes = int(total_payload_mb * 1024**2)
    per_tensor_bytes = int(per_tensor_mb * 1024**2)
    assert per_tensor_bytes > 0
    assert total_bytes >= per_tensor_bytes

    # Keep sizes divisible by 4 so float32 tensors fit exactly.
    if per_tensor_bytes % 4 != 0:
        raise ValueError("per_tensor_mb must produce a byte size divisible by 4")

    num_tensors = total_bytes // per_tensor_bytes
    # Alternate dtypes to cover bf16 path on CPU and fp16 path on GPU.
    dtypes = [torch.float32, torch.bfloat16 if include_bf16 else torch.float16]

    sd: dict[str, torch.Tensor] = {}
    for i in range(num_tensors):
        dtype = dtypes[i % len(dtypes)]
        elem_size = torch.empty((), dtype=dtype).element_size()
        n = per_tensor_bytes // elem_size
        t = torch.empty((n,), device=device, dtype=dtype)
        # Touch pages so baseline RSS reflects the allocation.
        t.zero_()
        sd[f"t{i:04d}"] = t

    # Ensure we're close to the requested payload.
    assert _payload_bytes(sd) >= int(total_bytes * 0.98)
    return sd


def _save_with_hf_safetensors_writer(state_dict: dict[str, torch.Tensor], ckpt_dir: Path) -> None:
    writer = _HuggingFaceStorageWriter(path=str(ckpt_dir), save_sharded=True)
    # Plan caching is irrelevant here (single process), but keep it off for determinism.
    planner = dcp.DefaultSavePlanner(enable_plan_caching=False)
    dcp.save(state_dict, checkpoint_id=str(ckpt_dir), storage_writer=writer, planner=planner)


def _measure_peaks_during_save(*, state_dict: dict[str, torch.Tensor], ckpt_dir: Path) -> _MemoryPeaks:
    # Encourage stable baseline.
    gc.collect()

    cuda_peak_alloc = None
    cuda_peak_reserved = None
    if torch.cuda.is_available():
        # Warm up CUDA context *before* taking the RSS baseline. Otherwise the first CUDA
        # interaction (e.g., stream creation inside DCP's overlapping loader) can allocate
        # substantial host memory and show up as an apparent RSS "spike" unrelated to
        # checkpointing behavior.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # Keep track of peaks during the save; peak stats are maintained by PyTorch.
        cuda_peak_alloc = torch.cuda.max_memory_allocated()
        cuda_peak_reserved = torch.cuda.max_memory_reserved()

    # Track Python-level peak allocations to catch regressions where we materialize a full
    # safetensors bytes blob (large `bytes` / `bytearray` allocations).
    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start(25)
    tracemalloc.clear_traces()
    tracemalloc.reset_peak()
    baseline_py_current, _baseline_py_peak = tracemalloc.get_traced_memory()

    baseline_rss = _rss_bytes()

    sampler = _RssSampler(interval_s=0.001)
    sampler.start()
    try:
        _save_with_hf_safetensors_writer(state_dict, ckpt_dir)
    finally:
        sampler.stop()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        cuda_peak_alloc = torch.cuda.max_memory_allocated()
        cuda_peak_reserved = torch.cuda.max_memory_reserved()

    _py_current, py_peak = tracemalloc.get_traced_memory()
    python_peak_delta = max(0, int(py_peak - baseline_py_current))
    if not was_tracing:
        tracemalloc.stop()

    # Convert to deltas to make assertions robust to unrelated baseline usage.
    return _MemoryPeaks(
        rss_peak_bytes=max(0, sampler.peak_rss_bytes - baseline_rss),
        cuda_peak_allocated_bytes=cuda_peak_alloc,
        cuda_peak_reserved_bytes=cuda_peak_reserved,
        python_peak_bytes=python_peak_delta,
    )


def test_hf_safetensors_dcp_save_cpu_rss_peak_does_not_spike(tmp_path: Path):
    """Regression: avoid materializing full safetensors bytes blob in memory during save.

    This test samples process RSS while saving via DCP + HF safetensors storage writer.
    The old behavior (calling `safetensors.torch.save(...)` and writing its returned bytes)
    can create a transient allocation roughly equal to the checkpoint payload size.
    """
    ckpt_dir = tmp_path / "ckpt_cpu"
    # Use many small tensors so that streaming implementations have a low peak,
    # while full-file bytes materialization spikes by ~total payload.
    state_dict = _make_many_small_tensors_state_dict(
        device=torch.device("cpu"),
        total_payload_mb=128,
        per_tensor_mb=2,
        include_bf16=True,
    )
    payload = _payload_bytes(state_dict)
    max_tensor = _max_tensor_bytes(state_dict)

    peaks = _measure_peaks_during_save(state_dict=state_dict, ckpt_dir=ckpt_dir)

    # Python allocations: streaming safetensors writer should avoid full-file bytes materialization.
    allowed_python_peak_delta = 64 * 1024**2
    assert peaks.python_peak_bytes is not None
    assert peaks.python_peak_bytes <= allowed_python_peak_delta, (
        "Unexpected Python allocation peak during safetensors checkpoint save.\n"
        f"payload_bytes={payload}\n"
        f"max_tensor_bytes={max_tensor}\n"
        f"num_tensors={len(state_dict)}\n"
        f"python_peak_delta_bytes={peaks.python_peak_bytes}\n"
        f"allowed_python_peak_delta_bytes={allowed_python_peak_delta}\n"
    )

    # Allow some allocator overhead and in-flight buffering, but keep it far below the total payload.
    # If the implementation regresses to materializing full safetensors bytes, this tends to approach ~payload.
    allowed_rss_delta = max(16 * 1024**2, int(max_tensor * 16))
    assert peaks.rss_peak_bytes <= allowed_rss_delta, (
        "Unexpected CPU RSS spike during safetensors checkpoint save.\n"
        f"payload_bytes={payload}\n"
        f"max_tensor_bytes={max_tensor}\n"
        f"num_tensors={len(state_dict)}\n"
        f"rss_peak_delta_bytes={peaks.rss_peak_bytes}\n"
        f"allowed_rss_delta_bytes={allowed_rss_delta}\n"
        f"cuda_peak_allocated_bytes={peaks.cuda_peak_allocated_bytes}\n"
        f"cuda_peak_reserved_bytes={peaks.cuda_peak_reserved_bytes}\n"
        f"python_peak_delta_bytes={peaks.python_peak_bytes}\n"
    )

    # Sanity: something was written.
    assert any(p.name.endswith(".safetensors") for p in ckpt_dir.glob("*.safetensors"))


@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_hf_safetensors_dcp_save_tracks_cpu_and_gpu_peaks(tmp_path: Path):
    """Track both CPU RSS peak and CUDA peak memory during checkpoint save (GPU path)."""
    # Ensure we have enough headroom for payload + test overhead.
    payload_mb = 128
    required_bytes = int(payload_mb * 1024**2 * 1.5)
    free, _total = torch.cuda.mem_get_info()
    if free < required_bytes * 2:
        pytest.skip(
            f"Not enough free CUDA memory for this test. "
            f"Need ~{(required_bytes * 2) / (1024**3):.2f} GiB, have {free / (1024**3):.2f} GiB."
        )

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    ckpt_dir = tmp_path / "ckpt_gpu"

    # Avoid bfloat16 CUDA compatibility issues on older GPUs; float16 still exercises the same writer path.
    state_dict = _make_large_state_dict(device=device, payload_mb=payload_mb, include_bf16=False)
    payload = _payload_bytes(state_dict)

    # Establish baseline GPU usage (tensors are already allocated).
    torch.cuda.synchronize()
    baseline_alloc = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()

    peaks = _measure_peaks_during_save(state_dict=state_dict, ckpt_dir=ckpt_dir)

    allowed_python_peak_delta = 64 * 1024**2
    assert peaks.python_peak_bytes is not None
    assert peaks.python_peak_bytes <= allowed_python_peak_delta, (
        "Unexpected Python allocation peak during GPU safetensors checkpoint save.\n"
        f"payload_bytes={payload}\n"
        f"python_peak_delta_bytes={peaks.python_peak_bytes}\n"
        f"allowed_python_peak_delta_bytes={allowed_python_peak_delta}\n"
    )

    # CPU RSS: GPU checkpointing must stage tensors on CPU, so RSS can approach the payload size.
    # The regression we want to catch is an *additional* full-payload allocation from materializing a
    # safetensors bytes blob (≈ +payload), which would push this well beyond ~1x payload.
    allowed_rss_delta = int(payload * 1.25)
    assert peaks.rss_peak_bytes <= allowed_rss_delta, (
        "Unexpected CPU RSS spike during GPU safetensors checkpoint save.\n"
        f"payload_bytes={payload}\n"
        f"rss_peak_delta_bytes={peaks.rss_peak_bytes}\n"
        f"allowed_rss_delta_bytes={allowed_rss_delta}\n"
    )

    # GPU peak stats are absolute; convert to a delta vs baseline.
    assert peaks.cuda_peak_allocated_bytes is not None
    assert peaks.cuda_peak_reserved_bytes is not None
    peak_alloc_delta = int(peaks.cuda_peak_allocated_bytes - baseline_alloc)
    peak_reserved_delta = int(peaks.cuda_peak_reserved_bytes - baseline_reserved)

    # Saving should primarily offload to CPU; allow some CUDA allocator noise.
    assert peak_alloc_delta <= 16 * 1024**2, (
        "Unexpected CUDA allocated memory spike during checkpoint save.\n"
        f"baseline_alloc_bytes={baseline_alloc}\n"
        f"peak_alloc_bytes={peaks.cuda_peak_allocated_bytes}\n"
        f"delta_bytes={peak_alloc_delta}\n"
    )
    assert peak_reserved_delta <= 16 * 1024**2, (
        "Unexpected CUDA reserved memory spike during checkpoint save.\n"
        f"baseline_reserved_bytes={baseline_reserved}\n"
        f"peak_reserved_bytes={peaks.cuda_peak_reserved_bytes}\n"
        f"delta_bytes={peak_reserved_delta}\n"
    )

    assert any(p.name.endswith(".safetensors") for p in ckpt_dir.glob("*.safetensors"))
