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

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.graph import saved_tensors_hooks
from torch.distributed.tensor import DeviceMesh, DTensor, Replicate, Shard, distribute_tensor

from nemo_automodel.components._peft.lora import (
    LinearLoRA,
    LoRATritonFunction,
    PeftConfig,
    apply_lora_to_linear_modules,
    apply_memory_efficient_lora,
    patch_linear_module,
)
from nemo_automodel.components.distributed.parallel_styles import TPLinear
from nemo_automodel.shared.import_utils import safe_import_te
from nemo_automodel.shared.tp_linear import _async_tp_linear

HAS_TE, transformer_engine = safe_import_te()


class DummyModel(nn.Module):
    """A dummy neural network model with two linear layers used for testing LoRA injection."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 16)
        self.linear2 = nn.Linear(16, 16)
        self.config = {}

    def forward(self, x):
        """Forward pass through two linear layers with ReLU activation in between."""
        x = self.linear1(x).relu()
        x = self.linear2(x)
        return x


class DummyModelNoConfig(nn.Module):
    """Same as DummyModel but without a `config` attribute."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 16)
        self.linear2 = nn.Linear(16, 16)

    def forward(self, x):
        x = self.linear1(x).relu()
        x = self.linear2(x)
        return x


@pytest.fixture
def dummy_input():
    """Provides a dummy input tensor for model testing."""
    return torch.randn(2, 16, requires_grad=True)


@pytest.fixture
def model():
    """Instantiates and returns a DummyModel instance."""
    return DummyModel()


@pytest.fixture
def model_no_config():
    """Instantiates a model that has no `config` attr."""
    return DummyModelNoConfig()


def test_lora_patch_on_model_without_config(model_no_config):
    """LoRA should still patch correctly even if the model lacks `config`."""
    apply_lora_to_linear_modules(model_no_config, PeftConfig(target_modules=["linear1"], dim=4, alpha=8))
    assert isinstance(model_no_config.linear1, LinearLoRA)
    assert not isinstance(model_no_config.linear2, LinearLoRA)


def test_backward_pass_without_config(dummy_input, model_no_config):
    """Backward pass must succeed on a model without `config`."""
    apply_lora_to_linear_modules(model_no_config, PeftConfig(target_modules=["linear1"], dim=4, alpha=8))
    out = model_no_config(dummy_input)
    loss = out.sum()
    loss.backward()

    grads = [p.grad for p in model_no_config.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_lora_patch_applies_to_selected_module(model):
    """Tests that LoRA is only applied to specified target modules."""
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8))
    assert isinstance(model.linear1, LinearLoRA)
    assert not isinstance(model.linear2, LinearLoRA)


def test_lora_patch_applies_to_selected_module_with_str_dtype(model):
    """Tests that LoRA is only applied to specified target modules."""
    apply_lora_to_linear_modules(
        model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8, lora_dtype="torch.bfloat16")
    )
    assert isinstance(model.linear1, LinearLoRA)
    assert model.linear1.lora_A.weight.dtype == torch.bfloat16
    assert model.linear1.lora_B.weight.dtype == torch.bfloat16
    assert not isinstance(model.linear2, LinearLoRA)


def test_peft_config_memory_efficient_lora_round_trip():
    """PeftConfig should default memory-efficient LoRA on and preserve explicit overrides."""
    assert PeftConfig().use_memory_efficient_lora is True

    cfg = PeftConfig.from_dict({"use_memory_efficient_lora": False})
    assert cfg.use_memory_efficient_lora is False
    assert cfg.to_dict()["use_memory_efficient_lora"] is False


def test_forward_output_consistency(dummy_input):
    """Verifies that model output shape remains the same after LoRA patching,
    but values change due to the added LoRA components.
    """
    base = DummyModel()
    model = DummyModel()
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8))

    base.eval()
    model.eval()

    with torch.no_grad():
        out1 = base(dummy_input)
        out2 = model(dummy_input)

    assert out1.shape == out2.shape
    assert not torch.allclose(out1, out2), "Output should differ due to LoRA injection"


def test_backward_pass(dummy_input):
    """Checks that backpropagation works and gradients are correctly computed
    when LoRA is applied.
    """
    model = DummyModel()
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8))
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "Some parameters should receive gradients"
    assert all(torch.isfinite(g).all() for g in grads if g is not None), "Gradients should be finite"


@pytest.mark.parametrize("input_shape", [(5, 16), (2, 3, 16)])
def test_memory_efficient_lora_matches_legacy_forward_and_backward(input_shape):
    """Custom autograd LoRA should match the legacy two-linear implementation."""
    torch.manual_seed(1234)
    scale = 2.0
    lora_dim = 4
    out_features = 12

    x = torch.randn(*input_shape, requires_grad=True)
    lora_A = torch.randn(lora_dim, input_shape[-1], requires_grad=True)
    lora_B = torch.randn(out_features, lora_dim, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    lora_A_ref = lora_A.detach().clone().requires_grad_(True)
    lora_B_ref = lora_B.detach().clone().requires_grad_(True)

    efficient = apply_memory_efficient_lora(x, lora_A, lora_B, scale, False)
    legacy = F.linear(F.linear(x_ref, lora_A_ref) * scale, lora_B_ref)

    grad = torch.randn_like(legacy)
    efficient.backward(grad)
    legacy.backward(grad)

    assert torch.allclose(efficient, legacy)
    assert torch.allclose(x.grad, x_ref.grad)
    assert torch.allclose(lora_A.grad, lora_A_ref.grad)
    assert torch.allclose(lora_B.grad, lora_B_ref.grad)


@pytest.mark.parametrize("input_shape", [(5, 16), (2, 3, 16)])
def test_memory_efficient_lora_mixed_dtype_matches_legacy_forward_and_backward(input_shape):
    """Custom autograd LoRA should preserve mixed-precision output and gradient contracts."""
    torch.manual_seed(1234)
    scale = 1.3
    lora_dim = 4
    out_features = 12

    x = torch.randn(*input_shape, dtype=torch.float32, requires_grad=True)
    lora_A = torch.randn(lora_dim, input_shape[-1], dtype=torch.bfloat16, requires_grad=True)
    lora_B = torch.randn(out_features, lora_dim, dtype=torch.bfloat16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    lora_A_ref = lora_A.detach().clone().requires_grad_(True)
    lora_B_ref = lora_B.detach().clone().requires_grad_(True)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        efficient = apply_memory_efficient_lora(x, lora_A, lora_B, scale, False).float()
        legacy = F.linear(F.linear(x_ref, lora_A_ref) * scale, lora_B_ref).float()

    grad = torch.randn_like(legacy)
    efficient.backward(grad)
    legacy.backward(grad)

    assert efficient.dtype == legacy.dtype == torch.float32
    assert x.grad.dtype == x_ref.grad.dtype == torch.float32
    assert lora_A.grad.dtype == lora_A_ref.grad.dtype == torch.bfloat16
    assert lora_B.grad.dtype == lora_B_ref.grad.dtype == torch.bfloat16
    torch.testing.assert_close(efficient, legacy)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(lora_A.grad, lora_A_ref.grad)
    torch.testing.assert_close(lora_B.grad, lora_B_ref.grad)


@pytest.mark.parametrize("input_shape", [(5, 16), (2, 3, 16)])
def test_memory_efficient_lora_with_residual_matches_legacy_forward_and_backward(input_shape):
    """Custom autograd LoRA should fold residual addition without changing gradients."""
    torch.manual_seed(1234)
    scale = 2.0
    lora_dim = 4
    out_features = 12
    output_shape = (*input_shape[:-1], out_features)

    x = torch.randn(*input_shape, requires_grad=True)
    lora_A = torch.randn(lora_dim, input_shape[-1], requires_grad=True)
    lora_B = torch.randn(out_features, lora_dim, requires_grad=True)
    res = torch.randn(*output_shape, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    lora_A_ref = lora_A.detach().clone().requires_grad_(True)
    lora_B_ref = lora_B.detach().clone().requires_grad_(True)
    res_ref = res.detach().clone().requires_grad_(True)

    efficient = apply_memory_efficient_lora(x, lora_A, lora_B, scale, False, res)
    legacy = res_ref + F.linear(F.linear(x_ref, lora_A_ref) * scale, lora_B_ref)

    grad = torch.randn_like(legacy)
    efficient.backward(grad)
    legacy.backward(grad)

    assert torch.allclose(efficient, legacy)
    assert torch.allclose(x.grad, x_ref.grad)
    assert torch.allclose(lora_A.grad, lora_A_ref.grad)
    assert torch.allclose(lora_B.grad, lora_B_ref.grad)
    assert torch.allclose(res.grad, res_ref.grad)


def test_memory_efficient_lora_saves_less_forward_state():
    """The custom autograd path should not save the intermediate x @ lora_A.T activation."""
    torch.manual_seed(1234)
    x = torch.randn(8, 16, requires_grad=True)
    lora_A = torch.randn(4, 16, requires_grad=True)
    lora_B = torch.randn(12, 4, requires_grad=True)
    scale = 2.0

    def collect_saved_tensors(fn):
        saved = []

        def pack_hook(tensor):
            saved.append(tuple(tensor.shape))
            return tensor

        with saved_tensors_hooks(pack_hook, lambda tensor: tensor):
            fn()
        return saved

    legacy_saved = collect_saved_tensors(lambda: F.linear(F.linear(x, lora_A) * scale, lora_B))
    efficient_saved = collect_saved_tensors(lambda: LoRATritonFunction.apply(x, lora_A, lora_B, scale, x.dtype, False))

    assert (8, 4) in legacy_saved
    assert (8, 4) not in efficient_saved
    assert sum(torch.tensor(shape).prod().item() for shape in efficient_saved) < sum(
        torch.tensor(shape).prod().item() for shape in legacy_saved
    )


def test_linear_lora_memory_efficient_flag_controls_saved_state():
    """LinearLoRA should use the memory-efficient autograd path when the flag is enabled."""
    torch.manual_seed(1234)
    base = nn.Linear(16, 12, bias=False)
    x = torch.randn(8, 16, requires_grad=True)
    legacy = LinearLoRA(base, dim=4, alpha=8, use_memory_efficient_lora=False)
    efficient = LinearLoRA(base, dim=4, alpha=8, use_memory_efficient_lora=True)

    def collect_saved_tensors(fn):
        saved = []

        def pack_hook(tensor):
            saved.append(tuple(tensor.shape))
            return tensor

        with saved_tensors_hooks(pack_hook, lambda tensor: tensor):
            fn()
        return saved

    legacy_saved = collect_saved_tensors(lambda: legacy(x))
    efficient_saved = collect_saved_tensors(lambda: efficient(x))

    assert (8, 4) in legacy_saved
    assert (8, 4) not in efficient_saved


def test_linear_lora_memory_efficient_matches_legacy_module_forward_and_backward():
    """LinearLoRA should preserve legacy module behavior when folding the residual add."""
    torch.manual_seed(1234)
    base = nn.Linear(16, 12, bias=False)
    x = torch.randn(8, 16, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    legacy = LinearLoRA(base, dim=4, alpha=8, use_memory_efficient_lora=False)
    efficient = LinearLoRA(base, dim=4, alpha=8, use_memory_efficient_lora=True)

    with torch.no_grad():
        legacy.lora_A.weight.normal_()
        legacy.lora_B.weight.normal_()
        efficient.lora_A.weight.copy_(legacy.lora_A.weight)
        efficient.lora_B.weight.copy_(legacy.lora_B.weight)

    efficient_out = efficient(x)
    legacy_out = legacy(x_ref)
    grad = torch.randn_like(legacy_out)
    efficient_out.backward(grad)
    legacy_out.backward(grad)

    assert torch.allclose(efficient_out, legacy_out)
    assert torch.allclose(x.grad, x_ref.grad)
    assert torch.allclose(efficient.lora_A.weight.grad, legacy.lora_A.weight.grad)
    assert torch.allclose(efficient.lora_B.weight.grad, legacy.lora_B.weight.grad)


def test_materialized_effective_weight_matches_linear_lora_forward_and_backward():
    """The effective dense weight must preserve ordinary LoRA outputs and gradients."""
    torch.manual_seed(1234)
    base = nn.Linear(7, 5, bias=True, dtype=torch.float32)
    direct = LinearLoRA(base, dim=3, alpha=6, use_memory_efficient_lora=False)
    materialized = LinearLoRA(base, dim=3, alpha=6, use_memory_efficient_lora=False)
    with torch.no_grad():
        direct.lora_A.weight.normal_()
        direct.lora_B.weight.normal_()
        materialized.lora_A.weight.copy_(direct.lora_A.weight)
        materialized.lora_B.weight.copy_(direct.lora_B.weight)

    x_direct = torch.randn(2, 4, 7, requires_grad=True)
    x_materialized = x_direct.detach().clone().requires_grad_(True)
    direct_out = direct(x_direct)
    materialized_out = F.linear(
        x_materialized,
        materialized.materialize_effective_weight(),
        materialized.bias,
    )
    output_grad = torch.randn_like(direct_out)
    direct_out.backward(output_grad)
    materialized_out.backward(output_grad)

    torch.testing.assert_close(materialized_out, direct_out)
    torch.testing.assert_close(x_materialized.grad, x_direct.grad)
    torch.testing.assert_close(materialized.lora_A.weight.grad, direct.lora_A.weight.grad)
    torch.testing.assert_close(materialized.lora_B.weight.grad, direct.lora_B.weight.grad)


def test_materialized_effective_weight_allows_inactive_dropout():
    """Evaluation-time dropout must reduce to the same deterministic effective weight."""
    torch.manual_seed(1234)
    lora = LinearLoRA(nn.Linear(7, 5, bias=False), dim=3, alpha=6, dropout=0.5)
    with torch.no_grad():
        lora.lora_A.weight.normal_()
        lora.lora_B.weight.normal_()
    lora.eval()
    x = torch.randn(2, 7)

    torch.testing.assert_close(F.linear(x, lora.materialize_effective_weight()), lora(x))


def test_materialized_effective_weight_rejects_active_dropout():
    """Training dropout cannot be represented by one deterministic dense weight."""
    lora = LinearLoRA(nn.Linear(7, 5, bias=False), dim=3, alpha=6, dropout=0.5)
    lora.train()

    with pytest.raises(RuntimeError, match="active LoRA training dropout"):
        lora.materialize_effective_weight()


def test_materialized_effective_weight_rejects_dora():
    """DoRA has magnitude normalization that the ordinary LoRA formula omits."""
    dora = LinearLoRA(nn.Linear(7, 5, bias=False), dim=3, alpha=6, use_dora=True)

    with pytest.raises(NotImplementedError, match="does not support DoRA"):
        dora.materialize_effective_weight()


def test_materialized_effective_weight_rejects_quantized_layout():
    """Quantized base weights must not silently use the ordinary dense formula."""
    lora = LinearLoRA(nn.Linear(7, 5, bias=False), dim=3, alpha=6)
    lora.quant_state = object()

    with pytest.raises(NotImplementedError, match="quantized linear implementations"):
        lora.materialize_effective_weight()


def test_lora_layers_are_trainable():
    """Ensures that LoRA layers are trainable while base weights remain frozen."""
    base = nn.Linear(16, 16)
    lora = LinearLoRA(base, dim=4, alpha=8)

    assert lora.weight.requires_grad is False
    assert lora.lora_A.weight.requires_grad
    assert lora.lora_B.weight.requires_grad
    if lora.bias is not None:
        assert lora.bias.requires_grad is False


def test_dora_layers_are_trainable_and_forward_works(dummy_input):
    """Ensures DoRA adds a learnable magnitude vector and forward/backward succeed."""
    base = nn.Linear(16, 16)
    dora = LinearLoRA(base, dim=4, alpha=8, use_dora=True, dropout=0.0)

    assert dora.weight.requires_grad is False
    assert dora.lora_A.weight.requires_grad
    assert dora.lora_B.weight.requires_grad
    assert hasattr(dora, "lora_magnitude")
    assert dora.lora_magnitude.requires_grad

    out = dora(dummy_input)
    loss = out.sum()
    loss.backward()

    assert dora.lora_A.weight.grad is not None
    assert dora.lora_B.weight.grad is not None
    assert dora.lora_magnitude.grad is not None
    assert torch.isfinite(dora.lora_magnitude.grad).all()


def test_apply_lora_with_dora_patches_selected_module(model):
    """apply_lora_to_linear_modules should be able to patch a module with DoRA enabled."""
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["linear1"], dim=4, alpha=8, use_dora=True))
    assert isinstance(model.linear1, LinearLoRA)
    assert getattr(model.linear1, "use_dora", False) is True
    assert hasattr(model.linear1, "lora_magnitude")


def test_dropout_pre_post_effects(dummy_input):
    """Tests that different dropout positions ('pre' vs 'post') lead to different outputs."""
    base = nn.Linear(16, 16)
    lora_pre = LinearLoRA(base, dim=4, alpha=8, dropout=0.5, dropout_position="pre")
    lora_post = LinearLoRA(base, dim=4, alpha=8, dropout=0.5, dropout_position="post")

    with torch.no_grad():
        lora_pre.lora_A.weight.uniform_()
        lora_pre.lora_B.weight.uniform_()

        lora_post.lora_A.weight.copy_(lora_pre.lora_A.weight)
        lora_post.lora_B.weight.copy_(lora_pre.lora_B.weight)

    lora_pre.train()
    lora_post.train()

    out_pre = lora_pre(dummy_input)
    out_post = lora_post(dummy_input)

    assert out_pre.shape == out_post.shape
    assert not torch.allclose(out_pre, out_post), "Dropout positions should affect output differently"


def test_apply_lora_respects_wildcard(model):
    """Validates that wildcard matching correctly applies LoRA to all matching modules."""
    assert isinstance(model.linear1, nn.Linear)
    assert isinstance(model.linear2, nn.Linear)
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=[".*"], dim=4, alpha=8))
    assert isinstance(model.linear1, LinearLoRA), type(model.linear1)
    assert isinstance(model.linear2, LinearLoRA)


def test_no_patch_on_non_matching_module(model):
    """Confirms that no modules are patched if target pattern doesn't match any names."""
    assert isinstance(model.linear1, nn.Linear)
    assert isinstance(model.linear2, nn.Linear)
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["nonexistent_module"], dim=4, alpha=8))
    assert not isinstance(model.linear1, LinearLoRA)
    assert not isinstance(model.linear2, LinearLoRA)


@pytest.mark.skipif(not HAS_TE or not torch.cuda.is_available(), reason="Transformer Engine or CUDA not available")
class TestTELinearLoRA:
    """Tests for LoRA patching of Transformer Engine Linear modules."""

    def test_patch_sets_super_fwd(self):
        """patch_linear_module should set super_fwd for TE Linear."""
        from transformer_engine.pytorch.module.linear import Linear as TELinear

        te_linear = TELinear(in_features=16, out_features=32, bias=False, params_dtype=torch.bfloat16).cuda()
        patched = patch_linear_module(te_linear, dim=4, alpha=8, use_triton=False)
        assert hasattr(patched, "super_fwd"), "super_fwd should be set for TE Linear"
        assert patched.super_fwd is not None
        assert patched.super_fwd != patched.forward

    def test_lora_adapters_are_te_linear(self):
        """lora_A and lora_B should be TE Linear when base module is TE Linear."""
        from transformer_engine.pytorch.module.linear import Linear as TELinear

        te_linear = TELinear(in_features=16, out_features=32, bias=False, params_dtype=torch.bfloat16).cuda()
        patched = patch_linear_module(te_linear, dim=4, alpha=8, use_triton=False)
        assert isinstance(patched.lora_A, TELinear), f"lora_A should be TE Linear, got {type(patched.lora_A)}"
        assert isinstance(patched.lora_B, TELinear), f"lora_B should be TE Linear, got {type(patched.lora_B)}"

    def test_forward_pass(self):
        """Patched TE Linear should produce valid output."""
        from transformer_engine.pytorch.module.linear import Linear as TELinear

        te_linear = TELinear(in_features=16, out_features=32, bias=False, params_dtype=torch.bfloat16).cuda()
        patched = patch_linear_module(te_linear, dim=4, alpha=8, use_triton=False)
        x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16)
        out = patched(x)
        assert out.shape == (2, 32), f"Expected shape (2, 32), got {out.shape}"
        assert torch.isfinite(out).all(), "Output contains non-finite values"

    def test_backward_pass(self):
        """Backward pass through patched TE Linear should produce gradients on LoRA params."""
        from transformer_engine.pytorch.module.linear import Linear as TELinear

        te_linear = TELinear(in_features=16, out_features=32, bias=False, params_dtype=torch.bfloat16).cuda()
        patched = patch_linear_module(te_linear, dim=4, alpha=8, use_triton=False)
        x = torch.randn(2, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        out = patched(x)
        out.sum().backward()
        assert patched.lora_A.weight.grad is not None, "lora_A should have gradients"
        assert patched.lora_B.weight.grad is not None, "lora_B should have gradients"
        assert torch.isfinite(patched.lora_A.weight.grad).all(), "lora_A gradients should be finite"
        assert torch.isfinite(patched.lora_B.weight.grad).all(), "lora_B gradients should be finite"


class _Gemma3nTextModelMini(nn.Module):
    """Minimal stand-in for transformers ``Gemma3nTextModel`` exercising the bug.

    Owns a ``per_layer_model_projection`` linear and a ``project_per_layer_inputs`` method whose
    first op mutates the projection output in place (``*=``), exactly like the real HF method.
    """

    class _Cfg:
        num_hidden_layers = 3

    class _Norm(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(d))

        def forward(self, x):
            var = x.float().pow(2).mean(-1, keepdim=True)
            return (x * torch.rsqrt(var + 1e-6).to(x.dtype)) * (1.0 + self.weight)

    def __init__(self, p=8, d=5, nlayers=3):
        super().__init__()
        self.config = self._Cfg()
        self.config.num_hidden_layers = nlayers
        self.hidden_size_per_layer_input = d
        self.per_layer_model_projection = nn.Linear(p, d * nlayers, bias=False)
        self.per_layer_projection_norm = self._Norm(d)
        self.register_buffer("per_layer_projection_scale", torch.tensor(d**-0.5))
        self.register_buffer("per_layer_input_scale", torch.tensor(2.0**0.5))

    def project_per_layer_inputs(self, inputs_embeds, per_layer_inputs=None, inplace=True):
        per_layer_projection = self.per_layer_model_projection(inputs_embeds)
        scale = self.per_layer_projection_scale.to(dtype=inputs_embeds.dtype, device=per_layer_projection.device)
        if inplace:
            per_layer_projection *= scale  # the offending in-place op (HF parity)
        else:
            per_layer_projection = per_layer_projection * scale
        per_layer_projection = per_layer_projection.reshape(
            *inputs_embeds.shape[:-1], self.config.num_hidden_layers, self.hidden_size_per_layer_input
        )
        per_layer_projection = self.per_layer_projection_norm(per_layer_projection)
        if per_layer_inputs is None:
            return per_layer_projection
        if per_layer_projection.shape != per_layer_inputs.shape:
            per_layer_inputs = per_layer_inputs[..., : self.config.num_hidden_layers, :]
        return (per_layer_projection + per_layer_inputs) * self.per_layer_input_scale.to(
            dtype=inputs_embeds.dtype, device=per_layer_projection.device
        )


def test_memory_efficient_lora_output_is_inplace_safe():
    """A consumer may mutate the memory-efficient LoRA output in place, with no model-specific patch.

    Regression for AM-453: transformers gemma3n ``project_per_layer_inputs`` does
    ``per_layer_projection *= scale`` on the projection output. Previously the memory-efficient LoRA
    returned a *view of a custom-autograd-Function output*, so the in-place op raised
    "Output 0 of LoRATritonFunctionBackward is a view and is being modified inplace". The
    ``(N, out) -> (bs, seq, out)`` reshape now happens outside the Function, so its output is an
    ordinary (in-place-safe) view -- fixing the bug class generically (no gemma3n-specific code).
    """
    p, d, nlayers = 8, 5, 3
    torch.manual_seed(0)
    model = _Gemma3nTextModelMini(p, d, nlayers)
    patch_linear_module(
        model.per_layer_model_projection, dim=4, alpha=8, use_triton=False, use_memory_efficient_lora=True
    )

    # Reference: identical weights, out-of-place consumer.
    torch.manual_seed(0)
    ref_model = _Gemma3nTextModelMini(p, d, nlayers)
    patch_linear_module(
        ref_model.per_layer_model_projection, dim=4, alpha=8, use_triton=False, use_memory_efficient_lora=True
    )

    inputs_embeds = torch.randn(2, 3, p)
    per_layer_inputs = torch.randn(2, 3, nlayers, d)

    for with_pli in (False, True):
        pli = per_layer_inputs if with_pli else None

        for m in (model, ref_model):
            m.per_layer_model_projection.lora_A.weight.grad = None
            m.per_layer_model_projection.lora_B.weight.grad = None

        x_in = inputs_embeds.detach().clone().requires_grad_(True)
        # The in-place consumer must NOT raise -- this is the regression.
        out = model.project_per_layer_inputs(x_in, pli.detach().clone() if with_pli else None, inplace=True)
        out.sum().backward()

        x_ref = inputs_embeds.detach().clone().requires_grad_(True)
        out_ref = ref_model.project_per_layer_inputs(x_ref, pli.detach().clone() if with_pli else None, inplace=False)
        out_ref.sum().backward()

        assert torch.allclose(out, out_ref, atol=1e-6)
        assert torch.allclose(x_in.grad, x_ref.grad, atol=1e-6)
        assert torch.allclose(
            model.per_layer_model_projection.lora_A.weight.grad,
            ref_model.per_layer_model_projection.lora_A.weight.grad,
            atol=1e-6,
        )
        assert torch.allclose(
            model.per_layer_model_projection.lora_B.weight.grad,
            ref_model.per_layer_model_projection.lora_B.weight.grad,
            atol=1e-6,
        )


def _make_lora_with_random_adapters(in_features: int = 16, out_features: int = 12) -> LinearLoRA:
    """Build a LinearLoRA with non-zero adapters so the LoRA delta is exercised."""
    base = nn.Linear(in_features, out_features)
    lora = LinearLoRA(base, dim=4, alpha=8, use_memory_efficient_lora=False)
    with torch.no_grad():
        lora.lora_A.weight.normal_()
        lora.lora_B.weight.normal_()
    return lora


def test_linear_lora_async_tp_mode_matches_eager():
    """The async-TP F.linear shaping must stay numerically identical to eager."""
    torch.manual_seed(0)
    lora = _make_lora_with_random_adapters()
    x = torch.randn(2, 5, 16)
    ref = lora(x)

    original = torch._inductor.config._micro_pipeline_tp
    torch._inductor.config._micro_pipeline_tp = True
    try:
        with patch("torch.compiler.is_compiling", return_value=True):
            out = lora(x)
    finally:
        torch._inductor.config._micro_pipeline_tp = original

    assert torch.allclose(out, ref, atol=1e-6)


def test_linear_lora_2d_input_under_compile_uses_f_linear():
    """2-D input while compile-tracing must skip the 3-D bmm path and match eager."""
    torch.manual_seed(0)
    lora = _make_lora_with_random_adapters()
    x = torch.randn(8, 16)
    ref = lora(x)

    with (
        patch.object(torch._inductor.config, "_micro_pipeline_tp", False),
        patch("torch.compiler.is_compiling", return_value=True),
    ):
        out = lora(x)

    assert torch.allclose(out, ref, atol=1e-6)


@pytest.fixture
def single_rank_pg():
    """Provide a single-rank gloo process group for DTensor placement tests."""
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")
    already = dist.is_initialized()
    if not already:
        dist.init_process_group(backend="gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield
    finally:
        if not already:
            dist.destroy_process_group()


def test_linear_lora_dim1_sharded_dtensor_takes_bmm_not_async_shaping(single_rank_pg):
    """A sequence-sharded DTensor input must bypass async-TP shaping and not crash.

    Builds a LinearLoRA whose base and adapter weights are replicated DTensors
    on a world-size-1 mesh (adapters converted to TPLinear, matching
    parallel_styles) and feeds a ``[B, S, in_features]`` DTensor input sharded
    on dim 1 (the sequence-parallel layout async-TP mandates), with the
    async-TP gate mocked open.  The base projection must route through
    ``torch.bmm``, never through ``_async_tp_linear`` (whose ``F.linear`` view
    cannot flatten the sharded dim), and match the eager local reference.
    """
    torch.manual_seed(0)
    lora = _make_lora_with_random_adapters()
    x_local = torch.randn(2, 5, 16)
    ref = lora(x_local)

    mesh = DeviceMesh("cpu", torch.arange(1))
    for mod in (lora, lora.lora_A, lora.lora_B):
        mod.weight = nn.Parameter(distribute_tensor(mod.weight.detach().clone(), mesh, [Replicate()]))
    lora.bias = nn.Parameter(distribute_tensor(lora.bias.detach().clone(), mesh, [Replicate()]))
    lora.lora_A.__class__ = TPLinear
    lora.lora_B.__class__ = TPLinear
    x = DTensor.from_local(x_local, mesh, [Shard(1)], run_check=False)

    with (
        patch("nemo_automodel.shared.tp_linear._is_async_tp_linear_enabled", return_value=True),
        patch("nemo_automodel.shared.tp_linear._async_tp_linear") as async_spy,
        patch("torch.bmm", wraps=torch.bmm) as bmm_spy,
    ):
        out = lora(x)

    async_spy.assert_not_called()
    assert bmm_spy.call_count == 3  # base projection + lora_A + lora_B
    assert isinstance(out, DTensor)
    assert torch.allclose(out.full_tensor(), ref, atol=1e-6)


def test_linear_lora_negative_last_dim_shard_takes_async_shaping(single_rank_pg):
    """A ``Shard(-1)`` LoRA input must take the fusable async-TP linear graph.

    Builds a LinearLoRA with replicated base and adapter weights on a
    world-size-1 mesh and feeds a ``[B, S, in_features]`` DTensor input sharded
    on its last/feature dimension. The base, LoRA-A, and LoRA-B projections must
    all avoid the batch/sequence-sharded ``bmm`` fallback.
    """
    torch.manual_seed(0)
    lora = _make_lora_with_random_adapters()
    x_local = torch.randn(2, 5, 16)
    ref = lora(x_local)

    mesh = DeviceMesh("cpu", torch.arange(1))
    for mod in (lora, lora.lora_A, lora.lora_B):
        mod.weight = nn.Parameter(distribute_tensor(mod.weight.detach().clone(), mesh, [Replicate()]))
    lora.bias = nn.Parameter(distribute_tensor(lora.bias.detach().clone(), mesh, [Replicate()]))
    lora.lora_A.__class__ = TPLinear
    lora.lora_B.__class__ = TPLinear
    x = DTensor.from_local(x_local, mesh, [Shard(-1)], run_check=False)

    with (
        patch("nemo_automodel.shared.tp_linear._is_async_tp_linear_enabled", return_value=True),
        patch("nemo_automodel.shared.tp_linear._async_tp_linear", wraps=_async_tp_linear) as async_spy,
        patch("torch.bmm", wraps=torch.bmm) as bmm_spy,
    ):
        out = lora(x)

    assert async_spy.call_count == 3  # base projection + lora_A + lora_B
    bmm_spy.assert_not_called()
    assert isinstance(out, DTensor)
    assert torch.allclose(out.full_tensor(), ref, atol=1e-6)


