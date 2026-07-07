"""Actor.forward returns a single named output dict.

Guards the concern that writing ``output["log_probs"]`` (and ``action_log_probs`` /
``entropy``) could clobber a tensor the underlying model already returned. It
cannot: HF causal-LM outputs and NeMo custom forwards never carry those keys, so
the writes are purely additive, and ``logits`` is the only field Actor.forward
intentionally replaces (with its gathered / chunk-sliced version).
"""

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
