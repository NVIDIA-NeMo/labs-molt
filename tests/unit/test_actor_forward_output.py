# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Actor.forward returns a single named output dict.

Guards the concern that writing ``output["log_probs"]`` (and ``action_log_probs`` /
``entropy``) could clobber a tensor the underlying model already returned. It
cannot: HF causal-LM outputs and NeMo custom forwards never carry those keys, so
the writes are purely additive, and ``logits`` is the only field Actor.forward
intentionally replaces (with its gathered / chunk-sliced version).
"""

import copy
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from molt.models.base import _AttrDict, _normalize_output


def test_hf_causal_lm_output_has_no_logprob_fields():
    # If any of these pre-existed on the model output, Actor.forward's
    # output[...] = ... assignments would overwrite a model-produced tensor.
    fields = set(CausalLMOutputWithPast.__dataclass_fields__)
    assert {"log_probs", "action_log_probs", "entropy"} & fields == set()


def test_adding_log_probs_to_hf_output_is_additive():
    logits = torch.randn(1, 5, 7)
    pkv = ("past",)
    hf_out = CausalLMOutputWithPast(logits=logits, past_key_values=pkv)

    out = _normalize_output(hf_out)
    assert isinstance(out, _AttrDict)
    assert "log_probs" not in out  # nothing to overwrite

    out["log_probs"] = logits[:, :-1, 0]
    out["action_log_probs"] = logits[:, -2:, 0]

    # Model-produced fields survive untouched; the new keys are additions.
    assert out["logits"] is logits
    assert out["past_key_values"] is pkv
    torch.testing.assert_close(out["log_probs"], logits[:, :-1, 0])
    assert out.log_probs is out["log_probs"]  # attribute access == key access

    # The original HF output object is not mutated (normalize copies into _AttrDict).
    assert not hasattr(hf_out, "log_probs")


def test_nemo_custom_raw_tensor_output_wraps_then_adds():
    # NeMo custom MoE/LLM return a raw logits Tensor (no dict, no log_probs).
    logits = torch.randn(1, 4, 9)
    out = _normalize_output(logits)
    assert isinstance(out, _AttrDict)
    assert out["logits"] is logits
    assert "log_probs" not in out

    out["log_probs"] = logits[:, :-1, 0]
    assert out["logits"] is logits  # logits unchanged by the new key


@pytest.mark.parametrize("names", [("q_proj", "k_proj", "v_proj"), ("gate_proj", "up_proj")])
def test_fused_projections_preserve_parameters_and_gradients(monkeypatch, names):
    """The packing adapter must consume each result once and keep original parameters."""
    from molt.models.base import _enable_qwen_fused_projections

    monkeypatch.setitem(sys.modules, "vllm.model_executor.determinism.batch_invariant",
                        SimpleNamespace(linear_batch_invariant=torch.nn.functional.linear))
    model = torch.nn.Module()
    for index, name in enumerate(names):
        model.add_module(name, torch.nn.Linear(4, index + 2))
    reference = copy.deepcopy(model)
    parameters = dict(model.named_parameters())
    _enable_qwen_fused_projections(model, names)
    assert all(p is dict(model.named_parameters())[name] for name, p in parameters.items())
    for _ in range(2):
        x = torch.randn(2, 4, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_()
        actual = torch.cat([getattr(model, name)(x) for name in names], dim=-1)
        expected = torch.cat([getattr(reference, name)(x_ref) for name in names], dim=-1)
        torch.testing.assert_close(actual, expected)
        actual.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, x_ref.grad)
        for p, q in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad)


@pytest.mark.parametrize("with_residual", [False, True])
def test_aligned_rmsnorm_preserves_inputs_and_residual_gradient(monkeypatch, with_residual):
    """The fused branch mutates its copies and propagates gradients through both outputs."""
    from molt.models.base import _VllmBatchInvariantRMSNorm

    def kernel(x, weight, eps, residual=None):
        if residual is not None:
            residual.add_(x)
            x.copy_(torch.nn.functional.rms_norm(residual, (4,), weight, eps))
            return x, residual
        return torch.nn.functional.rms_norm(x, (4,), weight, eps)

    monkeypatch.setitem(sys.modules, "vllm.model_executor.determinism.batch_invariant",
                        SimpleNamespace(rms_norm_batch_invariant=kernel))
    inputs = [torch.randn(2, 4), torch.randn(4)]
    if with_residual:
        inputs.append(torch.randn(2, 4))
    actual = [x.clone().requires_grad_() for x in inputs]
    reference = [x.clone().requires_grad_() for x in inputs]
    output = _VllmBatchInvariantRMSNorm.apply(actual[0], actual[1], 1e-6, *actual[2:])
    total = reference[0] + reference[2] if with_residual else reference[0]
    expected = torch.nn.functional.rms_norm(total, (4,), reference[1], 1e-6)
    if with_residual:
        torch.testing.assert_close(output[1], total)
        loss = output[0].square().sum() + output[1].square().sum()
        ref_loss = expected.square().sum() + total.square().sum()
        output = output[0]
    else:
        loss, ref_loss = output.square().sum(), expected.square().sum()
    torch.testing.assert_close(output, expected)
    loss.backward()
    ref_loss.backward()
    for value, original, ref in zip(actual, inputs, reference, strict=True):
        assert torch.equal(value, original)
        torch.testing.assert_close(value.grad, ref.grad, atol=1e-5, rtol=1e-4)
