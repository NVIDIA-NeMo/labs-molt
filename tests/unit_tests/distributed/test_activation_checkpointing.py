# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""CPU coverage for the selective-AC save-set build (FFPA fold-in wiring) and vision checkpointing."""

import contextlib
import copy

import pytest
import torch
import torch.nn.functional as F
from packaging.version import Version
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper, checkpoint_wrapper
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import CheckpointError, CheckpointPolicy, checkpoint, create_selective_checkpoint_contexts

from nemo_automodel.components.distributed import activation_checkpointing as ac


def test_unwrap_checkpoint_wrapper_returns_input_module_when_unwrapped():
    module = nn.Linear(2, 2)

    assert ac.unwrap_checkpoint_wrapper(module) is module


def test_unwrap_checkpoint_wrapper_returns_inner_module_when_wrapped():
    module = nn.Linear(2, 2)
    wrapped = checkpoint_wrapper(module)

    assert ac.unwrap_checkpoint_wrapper(wrapped) is module


def test_ffpa_forward_ops_folded_into_save_set(monkeypatch):
    """Ops returned by _ffpa_forward_ops() land in the save set (kernel-free wiring)."""
    dense, varlen = object(), object()
    monkeypatch.setattr(ac, "_ffpa_forward_ops", lambda: (dense, varlen))

    save_ops = ac._build_selective_ac_save_ops()
    assert dense in save_ops
    assert varlen in save_ops


def test_build_save_set_ok_when_ffpa_absent(monkeypatch):
    """CPU degrade path: _ffpa_forward_ops() -> () must not break the build."""
    monkeypatch.setattr(ac, "_ffpa_forward_ops", lambda: ())

    save_ops = ac._build_selective_ac_save_ops()
    assert isinstance(save_ops, frozenset) and len(save_ops) > 0


_D = 8

# (flash, mem_efficient, cudnn, math) -- the flag order of _sdp_backend_state.
_MATH_ONLY = (False, False, False, True)


def _sdp_backend_state() -> tuple[bool, bool, bool, bool]:
    """Return the global (flash, mem_efficient, cudnn, math) SDPA backend enablement flags."""
    return (
        torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.cudnn_sdp_enabled(),
        torch.backends.cuda.math_sdp_enabled(),
    )


class _RecordingSdpaAttention(nn.Module):
    """Minimal HF-style vision attention running real SDPA.

    Records the enabled SDPA backend flags on every forward call, so tests can
    assert which backend set the checkpoint forward and its backward-time
    recompute ran under.
    """

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(_D, 3 * _D)
        self.proj = nn.Linear(_D, _D)
        self.backend_states: list[tuple[bool, bool, bool, bool]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run SDPA over a flattened patch sequence.

        Args:
            x: Patch embeddings of shape ``[S, H]`` (``S`` = patches, ``H`` = hidden).

        Returns:
            Attention output of shape ``[S, H]``.
        """
        self.backend_states.append(_sdp_backend_state())
        q, k, v = (t.unsqueeze(0).unsqueeze(0) for t in self.qkv(x).chunk(3, dim=-1))
        return self.proj(F.scaled_dot_product_attention(q, k, v).squeeze(0).squeeze(0))


class _SdpaVisionBlock(nn.Module):
    """Minimal vision block with ``attn``/``mlp`` submodules for ``apply_submodule_checkpointing``."""

    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.attn = _RecordingSdpaAttention()
        self.mlp = nn.Sequential(nn.Linear(_D, _D), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention + MLP to patch embeddings of shape ``[S, H]``, returning ``[S, H]``."""
        return x + self.mlp(self.attn(x))


def test_snapshot_context_fn_recompute_ctx_restores_captured_backend_set():
    """The recompute context must re-pin exactly the backend set captured at snapshot time."""
    with sdpa_kernel([SDPBackend.MATH]):
        forward_ctx, recompute_ctx = ac.sdpa_backend_snapshot_context_fn()
        with forward_ctx:
            # The forward context is a no-op: the ambient forcing stays in effect.
            assert _sdp_backend_state() == _MATH_ONLY

    # Simulate a divergent backward-time ambient state (the forward pin has exited).
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        with recompute_ctx:
            assert _sdp_backend_state() == _MATH_ONLY
        # The backward-time ambient state is restored once the recompute exits.
        assert _sdp_backend_state() == (True, True, False, False)


def test_selective_checkpointing_to_layers_preserves_transformer_engine_attention_cache(monkeypatch):
    """The shared selective-AC wrapper must preserve TE's checkpoint op sequence."""
    attention_cache = {"attention_params": None, "backend": None}
    monkeypatch.setattr(ac, "_get_transformer_engine_attention_backend_cache", lambda: attention_cache)
    monkeypatch.setattr(
        ac,
        "_SELECTIVE_AC_MUST_SAVE_OPS",
        ac._SELECTIVE_AC_MUST_SAVE_OPS | {torch.ops.aten.ne.Tensor},
    )

    class Attention(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Run a cache-dependent attention stand-in.

            Args:
                x: Tensor of shape [features].

            Returns:
                Tensor of shape [features].
            """
            params = x.detach()
            if attention_cache["attention_params"] is None or params != attention_cache["attention_params"]:
                attention_cache["attention_params"] = params
                attention_cache["backend"] = "fused"
            return x.sin()

    model = nn.Module()
    model.attention = Attention()
    layers = [model.attention]
    ac.apply_selective_checkpointing_to_layers(model, layers, has_kv_sharing=False)
    assert isinstance(model.attention, CheckpointWrapper)
    assert layers[0] is model.attention

    x = torch.randn(8, requires_grad=True)
    out = model.attention(x)
    cache_after_forward = dict(attention_cache)

    out.sum().backward()

    assert torch.allclose(x.grad, x.detach().cos())
    assert attention_cache["attention_params"] is cache_after_forward["attention_params"]
    assert attention_cache["backend"] == cache_after_forward["backend"]


def test_recompute_only_fsdp_unshard_op_bypasses_selective_ac_replay(monkeypatch):
    """FSDP prefetch may make unshard ops recompute-only; they must bypass SAC indexing."""
    import torch.distributed.fsdp._fully_shard._fsdp_collectives  # noqa: F401

    fsdp_copy_in = torch.ops.fsdp.all_gather_copy_in.default
    fsdp_empty = torch.ops.aten.empty.memory_format
    fsdp_empty_like = torch.ops.aten.empty_like.default
    fsdp_view = torch.ops.aten.view.default
    sac_ignored = set(torch.utils.checkpoint.SAC_IGNORED_OPS)
    monkeypatch.setattr(torch.utils.checkpoint, "SAC_IGNORED_OPS", sac_ignored)
    ac.ensure_fsdp_ops_sac_ignored()

    def policy(ctx, func, *args, **kwargs):
        if func in (fsdp_copy_in, fsdp_empty, fsdp_empty_like, fsdp_view):
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE

    def selective_context_fn():
        return create_selective_checkpoint_contexts(policy)

    fsdp_state = {"unsharded": False}

    def checkpointed_module(x: torch.Tensor) -> torch.Tensor:
        """Run a checkpointed FSDP-unshard stand-in.

        Args:
            x: Tensor of shape [features].

        Returns:
            Tensor of shape [features].
        """
        if fsdp_state["unsharded"]:
            all_gather_output = torch.empty_like(x)
            torch.empty(x.shape, dtype=x.dtype, device=x.device)
            fsdp_copy_in([x.detach()], all_gather_output, [x.numel()], x.numel(), 0)
            x = x.view(-1)
        fsdp_state["unsharded"] = True
        return x.sin()

    x = torch.randn(8, requires_grad=True)
    out = checkpoint(checkpointed_module, x, use_reentrant=False, context_fn=selective_context_fn)
    out.sum().backward()

    assert torch.allclose(x.grad, x.detach().cos())


def test_fsdp_runtime_ops_are_all_ignored_by_selective_ac(monkeypatch):
    """FSDP2 copy and all-gather ops must bypass SAC replay accounting."""
    import torch.distributed.fsdp._fully_shard._fsdp_collectives  # noqa: F401
    import torch.distributed.fsdp._fully_shard._fsdp_param  # noqa: F401

    sac_ignored = set()
    monkeypatch.setattr(torch.utils.checkpoint, "SAC_IGNORED_OPS", sac_ignored)

    ac.ensure_fsdp_ops_sac_ignored()

    expected = {
        torch.ops.fsdp.all_gather_copy_in.default,
        torch.ops.fsdp.split_with_sizes_copy.default,
        torch.ops.fsdp.chunk_cat.default,
        torch.ops.fsdp.copy_.default,
        torch.ops.c10d._allgather_base_.default,
        torch.ops.aten.empty.memory_format,
        torch.ops.aten.empty_like.default,
        torch.ops.aten.view.default,
        # non-zero-dim shard copy-out path: torch.chunk -> aten.split.Tensor, torch.cat(out=)
        torch.ops.aten.split.Tensor,
        torch.ops.aten.cat.out,
    }
    assert expected <= sac_ignored


def test_submodule_checkpointing_with_snapshot_context_reruns_recompute_under_forward_backends():
    """Recompute must rerun under the forward-time SDPA backend set even after the pin exits."""
    block = _SdpaVisionBlock()
    ac.apply_submodule_checkpointing([block], has_kv_sharing=False, context_fn=ac.sdpa_backend_snapshot_context_fn)
    assert isinstance(block.attn, CheckpointWrapper)
    assert isinstance(block.mlp, CheckpointWrapper)

    attn = ac.unwrap_checkpoint_wrapper(block.attn)
    with sdpa_kernel([SDPBackend.MATH]):  # ambient forcing active only during the forward
        out = block(torch.randn(4, _D))
    out.sum().backward()  # recompute runs here, after the pin has exited

    # Non-reentrant checkpointing runs the attention forward twice (forward +
    # backward-time recompute). The snapshot is taken at checkpoint-region entry
    # (while the ambient pin is still active) and re-pinned for the recompute.
    assert attn.backend_states == [_MATH_ONLY, _MATH_ONLY]
    assert block.attn._checkpoint_wrapped_module.qkv.weight.grad is not None


def test_submodule_checkpointing_without_snapshot_context_recompute_sees_divergent_backends():
    """Without the snapshot context_fn, the recompute runs under whatever is ambient at backward time."""
    block = _SdpaVisionBlock()
    ac.apply_submodule_checkpointing([block], has_kv_sharing=False, context_fn=None)

    attn = ac.unwrap_checkpoint_wrapper(block.attn)
    with sdpa_kernel([SDPBackend.MATH]):
        out = block(torch.randn(4, _D))
    # On fused-CUDA stacks the divergence trips the checkpoint determinism check;
    # on CPU both passes may dispatch the same kernel and only the recorded flags
    # diverge. Either way, the forward-time backend set is not preserved.
    with contextlib.suppress(CheckpointError):
        out.sum().backward()
    assert attn.backend_states[0] == _MATH_ONLY
    assert attn.backend_states[1] != _MATH_ONLY


class _LinearAttnBlock(nn.Module):
    """Minimal hybrid decoder block whose mixer is named ``linear_attn``.

    Mirrors Qwen3-Next / Qwen3.5 Gated DeltaNet layers, which name their token
    mixer ``linear_attn`` (not ``self_attn``). Used to check that
    ``apply_submodule_checkpointing`` wraps that submodule.
    """

    def __init__(self):
        super().__init__()
        self.linear_attn = nn.Linear(_D, _D)
        self.mlp = nn.Sequential(nn.Linear(_D, _D), nn.Linear(_D, _D))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the linear-attention mixer + MLP.

        Args:
            x: Input activations of shape ``[S, H]`` (``S`` = tokens, ``H`` = hidden).

        Returns:
            Output activations of shape ``[S, H]``.
        """
        return x + self.mlp(self.linear_attn(x))


def test_submodule_checkpointing_wraps_linear_attn_mixer():
    """Gated DeltaNet blocks name their mixer ``linear_attn``; it must be checkpoint-wrapped."""
    block = _LinearAttnBlock()
    inner = block.linear_attn
    ac.apply_submodule_checkpointing([block], has_kv_sharing=False)

    assert isinstance(block.linear_attn, CheckpointWrapper)
    assert isinstance(block.mlp, CheckpointWrapper)
    # The wrapper preserves the original mixer as its inner module.
    assert ac.unwrap_checkpoint_wrapper(block.linear_attn) is inner


def test_submodule_checkpointing_preserves_canonical_state_dict_keys_for_property_aliases():
    class Mixer(nn.Module):
        def __init__(self):
            super().__init__()
            self._fp32_params = nn.Module()
            self._fp32_params.A_log = nn.Parameter(torch.ones(2))

    class AliasLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.mixer = Mixer()

        @property
        def mlp(self):
            return self.mixer

        @property
        def self_attn(self):
            return self.mixer

    layer = AliasLayer()

    ac.apply_submodule_checkpointing([layer], has_kv_sharing=False)

    assert isinstance(layer.mixer, CheckpointWrapper)
    assert isinstance(ac.unwrap_checkpoint_wrapper(layer.mixer), Mixer)
    assert [name for name, _ in layer.named_children()] == ["mixer"]
    assert list(layer.state_dict()) == ["mixer._fp32_params.A_log"]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Backend-set divergence across checkpoint recompute requires fused CUDA SDPA backends",
)
def test_checkpointed_sdpa_replay_faults_without_snapshot_context_and_passes_with_it():
    """One-sided ambient forcing faults plain checkpoint replay; the snapshot context_fn restores parity."""
    device = torch.device("cuda")
    x = torch.randn(4, _D, device=device)

    # WITHOUT the snapshot context_fn: forward under an ambient MATH pin that has
    # exited by backward time -> the recompute dispatches a fused backend that
    # saves different tensors -> deterministic CheckpointError.
    plain = _SdpaVisionBlock().to(device)
    ac.apply_submodule_checkpointing([plain], has_kv_sharing=False, context_fn=None)
    with sdpa_kernel([SDPBackend.MATH]):
        out = plain(x)
    with pytest.raises(CheckpointError):
        out.sum().backward()

    # WITH the snapshot context_fn: the same one-sided pin replays cleanly, with
    # gradient parity against an unwrapped reference run entirely under the pin.
    block = _SdpaVisionBlock().to(device)
    reference = copy.deepcopy(block)
    ac.apply_submodule_checkpointing([block], has_kv_sharing=False, context_fn=ac.sdpa_backend_snapshot_context_fn)
    with sdpa_kernel([SDPBackend.MATH]):
        out = block(x)
    out.sum().backward()

    with sdpa_kernel([SDPBackend.MATH]):
        ref_out = reference(x)
        ref_out.sum().backward()

    assert torch.allclose(out, ref_out)
    for got, want in zip(block.parameters(), reference.parameters()):
        assert torch.allclose(got.grad, want.grad)


def test_submodule_checkpointing_with_snapshot_context_preserves_dropout_rng_on_recompute():
    """Backward recompute must redraw the forward's dropout mask, or gradients silently corrupt."""
    model = _SdpaVisionBlock(dropout=0.5)
    reference = copy.deepcopy(model)
    model.train()
    reference.train()

    ac.apply_submodule_checkpointing([model], has_kv_sharing=False, context_fn=ac.sdpa_backend_snapshot_context_fn)

    x = torch.randn(64, _D)
    torch.manual_seed(1234)
    out = model(x)
    out.sum().backward()

    torch.manual_seed(1234)
    ref_out = reference(x)
    ref_out.sum().backward()

    # Same seed -> same forward mask; grads only match if the recompute (which runs
    # after the global RNG has advanced) restores the checkpoint-time RNG state.
    assert torch.allclose(out, ref_out)
    for got, want in zip(model.parameters(), reference.parameters()):
        assert torch.allclose(got.grad, want.grad)


class _TextConfig:
    """Stand-in for a HF text sub-config that owns ``use_cache``."""

    def __init__(self, num_kv_shared_layers: int = 0):
        self.num_kv_shared_layers = num_kv_shared_layers
        self.use_cache = True


class _VisionConfig:
    """Stand-in for a HF vision sub-config without ``use_cache``."""

    def __init__(self):
        self.depth = 2


class _CompositeConfig:
    """Stand-in for a HF composite (VLM) config with declared sub-configs."""

    sub_configs = {"text_config": _TextConfig, "vision_config": _VisionConfig}

    def __init__(self, num_kv_shared_layers: int = 0):
        self.text_config = _TextConfig(num_kv_shared_layers)
        self.vision_config = _VisionConfig()
        self.use_cache = True


def _model_with_config(config) -> nn.Module:
    model = nn.Module()
    model.config = config
    return model


def test_detect_kv_sharing_disables_cache_on_composite_and_text_sub_config():
    """The text model reads ``text_config.use_cache``, so leaving it True keeps a
    DynamicCache alive under checkpointing and recompute double-appends K/V."""
    model = _model_with_config(_CompositeConfig())

    has_kv_sharing = ac.detect_kv_sharing_and_maybe_disable_cache(model)

    assert has_kv_sharing is False
    assert model.config.use_cache is False
    assert model.config.text_config.use_cache is False
    # Sub-configs without ``use_cache`` must not grow a phantom field.
    assert not hasattr(model.config.vision_config, "use_cache")


def test_detect_kv_sharing_disables_text_config_cache_without_sub_configs_attr():
    """Configs lacking the ``sub_configs`` class attribute fall back to ``text_config``."""

    class _LegacyComposite:
        def __init__(self):
            self.text_config = _TextConfig()
            self.use_cache = True

    model = _model_with_config(_LegacyComposite())

    assert ac.detect_kv_sharing_and_maybe_disable_cache(model) is False
    assert model.config.use_cache is False
    assert model.config.text_config.use_cache is False


def test_detect_kv_sharing_leaves_cache_enabled_for_kv_shared_models():
    """KV-shared models pass K/V between layers through the cache; it must stay on."""
    model = _model_with_config(_CompositeConfig(num_kv_shared_layers=2))

    has_kv_sharing = ac.detect_kv_sharing_and_maybe_disable_cache(model)

    assert has_kv_sharing is True
    assert model.config.use_cache is True
    assert model.config.text_config.use_cache is True


def _sac_context_factory():
    from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

    def policy(ctx, func, *args, **kwargs):
        return CheckpointPolicy.PREFER_RECOMPUTE

    return lambda: create_selective_checkpoint_contexts(policy)


def test_ignore_sac_ops_adds_available_ops(monkeypatch):
    sac_ignored = set()
    monkeypatch.setattr(torch.utils.checkpoint, "SAC_IGNORED_OPS", sac_ignored, raising=False)

    ac.ignore_sac_ops([torch.ops.aten.sin.default, None, torch.ops.aten.cos.default])

    assert sac_ignored == {torch.ops.aten.sin.default, torch.ops.aten.cos.default}


def test_profiler_ops_sac_ignore_skips_before_torch_2_13(monkeypatch):
    sac_ignored = set()
    monkeypatch.setattr(ac, "get_torch_version", lambda: Version("2.12.0"))
    monkeypatch.setattr(torch.utils.checkpoint, "SAC_IGNORED_OPS", sac_ignored, raising=False)

    ac.ensure_profiler_ops_sac_ignored()

    assert sac_ignored == set()


def test_profiler_ops_sac_ignore_includes_torch_2_13_alpha(monkeypatch):
    op = object()
    packet = type("FakePacket", (), {"default": op, "overloads": lambda self: ("default",)})()
    profiler_ops = type(
        "FakeProfilerOps",
        (),
        {
            "_record_function_enter": packet,
            "_record_function_enter_new": packet,
            "_record_function_exit": packet,
        },
    )()
    sac_ignored = set()
    monkeypatch.setattr(ac, "get_torch_version", lambda: Version("2.13.0a0+8145d630e8"))
    monkeypatch.setattr(torch.utils.checkpoint, "SAC_IGNORED_OPS", sac_ignored, raising=False)
    monkeypatch.setattr(torch.ops, "profiler", profiler_ops, raising=False)

    ac.ensure_profiler_ops_sac_ignored()

    assert sac_ignored == {op}


def test_sac_replay_tolerates_recompute_only_record_function(monkeypatch):
    """A profiler range entered only during backward recompute must not desync SAC replay.

    torch 2.13's FSDP2 runs its hooks under ``record_function``; with an FSDP
    boundary inside a SAC region the range ops fire a different number of times
    in the recompute than in the forward, which shifts SAC's per-op replay
    index and raises ``... encountered during backward but not found in
    storage``. ``ensure_profiler_ops_sac_ignored`` keeps profiler ops out of
    the replay accounting.
    """
    if not hasattr(torch.utils.checkpoint, "SAC_IGNORED_OPS"):
        pytest.skip("torch build without SAC_IGNORED_OPS")

    monkeypatch.setattr(ac, "get_torch_version", lambda: Version("2.13.0a0+8145d630e8"))
    ac.ensure_profiler_ops_sac_ignored()
    assert torch.ops.profiler._record_function_enter_new.default in torch.utils.checkpoint.SAC_IGNORED_OPS

    linear = nn.Linear(4, 4)
    calls = {"n": 0}

    def fn(x):
        calls["n"] += 1
        if calls["n"] > 1:  # backward-time recompute takes a different hook path
            with torch.autograd.profiler.record_function("recompute-only-range"):
                return linear(x)
        return linear(x)

    x = torch.randn(2, 4, requires_grad=True)
    out = torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False, context_fn=_sac_context_factory())
    out.sum().backward()

    assert calls["n"] == 2
    assert x.grad is not None
