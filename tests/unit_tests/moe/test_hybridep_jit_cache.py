# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import os

import pytest
import torch

import nemo_automodel.components.moe.megatron.fused_a2a as fused_a2a


class _RecordingBuffer:
    kwargs: dict = {}

    def __init__(self, **kwargs):
        type(self).kwargs = kwargs


@pytest.fixture
def jit_env(monkeypatch, tmp_path):
    """Shared cache under tmp_path/shared, HybridEP JIT base under tmp_path/jit, fake toolchain."""
    # deep_ep is absent on the CPU / generic GPU CI runners: install the recording double even when the
    # real HybridEPBuffer never imported (raising=False) and tell the module HybridEP is available.
    monkeypatch.setattr(fused_a2a, "HybridEPBuffer", _RecordingBuffer, raising=False)
    monkeypatch.setattr(fused_a2a, "HAVE_HYBRIDEP", True, raising=False)
    _RecordingBuffer.kwargs = {}
    monkeypatch.setattr(fused_a2a, "_jit_proc_dir", None)
    monkeypatch.setattr(fused_a2a, "_jit_shared_dir", None)
    monkeypatch.setattr(fused_a2a, "_jit_stored", False)
    monkeypatch.setattr(fused_a2a, "_deep_ep_version", lambda: "1.2.1")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setenv("HYBRID_EP_CACHE_DIR", str(tmp_path / "jit"))
    monkeypatch.setenv("LOCAL_RANK", "0")
    shared_root = tmp_path / "shared"
    monkeypatch.setattr(fused_a2a, "_JIT_CACHE_ROOT", str(shared_root))
    yield shared_root, tmp_path / "jit" / ".deepep" / "hybrid_ep" / "jit" / f"proc-{os.getpid()}"
    fused_a2a.reset_hybrid_ep_buffer()


def test_dirs_mirror_deepep_layout(jit_env):
    shared_root, proc_dir = jit_env
    shared, proc = fused_a2a._hybrid_ep_jit_dirs()
    assert proc == str(proc_dir)
    assert shared == str(shared_root / "deep_ep-1.2.1_cuda-13.0_sm100")


def test_warm_start_copies_shared_kernels_and_enables_loading(jit_env):
    shared_root, proc_dir = jit_env
    shared = shared_root / "deep_ep-1.2.1_cuda-13.0_sm100"
    shared.mkdir(parents=True)
    (shared / "3584-2048-56-1-0.so").write_bytes(b"pre")
    (shared / "3584-2048-56-1-1.so").write_bytes(b"disp")
    (shared / "README").write_text("not a kernel")

    fused_a2a.init_hybrid_ep_buffer(None, 8, 4, 2, 8, 8, False)

    assert _RecordingBuffer.kwargs["load_cached_kernels"] is True
    assert sorted(os.listdir(proc_dir)) == ["3584-2048-56-1-0.so", "3584-2048-56-1-1.so"]
    assert (proc_dir / "3584-2048-56-1-1.so").read_bytes() == b"disp"


def test_store_adds_new_kernels_and_keeps_existing_ones(jit_env):
    shared_root, proc_dir = jit_env
    shared = shared_root / "deep_ep-1.2.1_cuda-13.0_sm100"
    shared.mkdir(parents=True)
    (shared / "old.so").write_bytes(b"old")
    fused_a2a.init_hybrid_ep_buffer(None, 8, 4, 2, 8, 8, False)
    # HybridEP compiled a new kernel into the per-process dir and left a source file behind.
    (proc_dir / "new.so").write_bytes(b"new")
    (proc_dir / "new.cu").write_text("code")

    assert fused_a2a.store_hybrid_ep_jit_cache() == 1
    assert sorted(os.listdir(shared)) == ["new.so", "old.so"]
    assert (shared / "old.so").read_bytes() == b"old"
    assert (shared / "new.so").read_bytes() == b"new"
    # Second store finds nothing new.
    assert fused_a2a.store_hybrid_ep_jit_cache() == 0


def test_store_only_from_local_rank_zero(jit_env, monkeypatch):
    shared_root, proc_dir = jit_env
    fused_a2a.init_hybrid_ep_buffer(None, 8, 4, 2, 8, 8, False)
    (proc_dir / "k.so").write_bytes(b"k")
    monkeypatch.setenv("LOCAL_RANK", "3")
    assert fused_a2a.store_hybrid_ep_jit_cache() == 0
    assert not (shared_root / "deep_ep-1.2.1_cuda-13.0_sm100" / "k.so").exists()


def test_unset_hybrid_ep_cache_dir_gets_a_private_base(jit_env, monkeypatch):
    monkeypatch.delenv("HYBRID_EP_CACHE_DIR")
    fused_a2a.init_hybrid_ep_buffer(None, 8, 4, 2, 8, 8, False)
    base = os.environ["HYBRID_EP_CACHE_DIR"]
    assert os.path.isdir(os.path.join(base, ".deepep", "hybrid_ep", "jit", f"proc-{os.getpid()}"))
