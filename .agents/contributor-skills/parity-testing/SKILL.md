---
name: parity-testing
description: Verify numerical parity between NeMo AutoModel implementations and reference HuggingFace models, including state dict and forward-pass checks.
when_to_use: Verifying numerical correctness of a new or modified model against its HuggingFace reference, debugging loss divergence or output mismatches, or validating state dict mappings.
---

# Parity Testing Skill for NeMo AutoModel

NeMo AutoModel adds custom model implementations (combined projections, backend switching, kernel patches) on top of HuggingFace transformers. Parity testing verifies that NeMo AutoModel's implementation produces numerically equivalent results to the reference HF implementation.

Key differences that can cause divergence:
- Combined QKV projections (interleaved layout) vs separate Q/K/V
- Combined GateUp MLP vs separate gate/up projections
- TE attention vs SDPA vs flex attention backends
- TE linear vs torch linear
- FP8/BF16 precision differences
- RoPE implementation differences
- State dict adapter conversion (from_hf/to_hf round-trip)
- Kernel patches (Liger kernels, etc.)

## Setup

### Identify the two implementations

```python
from transformers import AutoModelForCausalLM
from nemo_automodel import NeMoAutoModelForCausalLM
```

The HF model is the reference. The NeMo AutoModel is the implementation under test.

```python
# NeMo way
nemo_model = NeMoAutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
# HF way
hf_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
```

### Create identical inputs

Use seeded random tensors to guarantee reproducibility across runs.

```python
import torch

torch.manual_seed(42)
input_ids = torch.randint(0, 32000, (1, 128))
attention_mask = torch.ones_like(input_ids)
```

### Use CPU + float32 for strictest comparison

GPU introduces non-determinism from parallel reductions and kernel launch order. Always start parity testing on CPU with float32 to isolate numerical differences caused by model implementation from those caused by hardware.

```python
device = torch.device("cpu")
dtype = torch.float32
hf_model = hf_model.to(device=device, dtype=dtype).eval()
nemo_model = nemo_model.to(device=device, dtype=dtype).eval()
```

## Test Strategy (3 Levels)

### Level 1: State Dict Round-Trip (CPU/float32)

Verify loaded and exported weights match exactly. Most custom models need an
adapter round-trip; models with matching HF layouts use their state dict directly.

```python
hf_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", torch_dtype=torch.float32, device_map="cpu"
)
nemo_model = NeMoAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", torch_dtype=torch.float32, device_map="cpu"
)

adapter = getattr(nemo_model, "state_dict_adapter", None)
hf_sd = hf_model.state_dict()

# Inspect weights actually loaded into the model, including when no adapter exists.
native_sd = nemo_model.state_dict()
roundtrip_sd = adapter.to_hf(native_sd) if adapter is not None else native_sd
# Module metadata is not part of the HF weight dictionary.
roundtrip_sd = {key: value for key, value in roundtrip_sd.items() if not key.endswith("_extra_state")}

# Check no missing or extra keys
assert set(roundtrip_sd.keys()) == set(hf_sd.keys()), (
    f"Key mismatch.\n"
    f"  Missing from roundtrip: {set(hf_sd.keys()) - set(roundtrip_sd.keys())}\n"
    f"  Extra in roundtrip: {set(roundtrip_sd.keys()) - set(hf_sd.keys())}"
)

# Check all values are exactly equal (max_diff must be 0.0)
for key in hf_sd:
    assert roundtrip_sd[key].shape == hf_sd[key].shape
    assert roundtrip_sd[key].dtype == hf_sd[key].dtype
    max_diff = (hf_sd[key] - roundtrip_sd[key]).abs().max().item()
    assert max_diff == 0.0, f"Round-trip mismatch for {key}: max_diff={max_diff}"

print("Level 1 PASSED: state dict round-trip is exact.")
```

**What to check:**
- All keys present in both dicts (no missing, no extra).
- Every tensor value matches exactly (max_diff == 0.0). Combined projection adapters must perfectly split and recombine weights.
- Tied weight keys (e.g., `lm_head.weight` aliasing `model.embed_tokens.weight`) are handled correctly.

Also check weights and ties after real HF export and native save/reload;
see `tests/unit_tests/checkpoint/test_native_hf_state_dict.py`.

### Level 2: Component Parity (CPU/float32)

Test individual components in isolation to localize any divergence.

**Components to test:** attention, MLP, layer norm / RMSNorm, RoPE, full decoder layer.

```python
# Create a tiny config to keep tests fast
from transformers import AutoConfig

config = AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")
config.num_hidden_layers = 2
config.hidden_size = 256
config.intermediate_size = 512
config.num_attention_heads = 4
config.num_key_value_heads = 2

hf_model = AutoModelForCausalLM.from_config(config).to(dtype=torch.float32, device="cpu").eval()
nemo_model = NeMoAutoModelForCausalLM.from_config(config).to(dtype=torch.float32, device="cpu").eval()

# Load the same weights into both
hf_sd = hf_model.state_dict()
adapter = getattr(nemo_model, "state_dict_adapter", None)
native_sd = adapter.from_hf(hf_sd) if adapter is not None else hf_sd
nemo_model.load_state_dict(native_sd)
```

**Forward pass with identical seeded inputs:**

```python
torch.manual_seed(42)
input_ids = torch.randint(0, config.vocab_size, (1, 64))
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    hf_out = hf_model(input_ids, attention_mask=attention_mask)
    nemo_out = nemo_model(input_ids, attention_mask=attention_mask)

max_diff, mean_diff, cos_sim = compare_tensors(
    hf_out.logits, nemo_out.logits, name="component_parity_logits"
)
assert max_diff < 1e-5, f"Component parity FAILED: max_diff={max_diff}"
print("Level 2 PASSED: component parity within tolerance.")
```

**Strict tolerance:** max_diff < 1e-5 for float32 on CPU. This is tight enough to catch weight loading bugs while allowing for minor floating-point operation reordering.

**Testing individual components (attention example):**

```python
# Extract matching layers
hf_attn = hf_model.model.layers[0].self_attn
nemo_attn = nemo_model.model.layers[0].self_attn

torch.manual_seed(42)
hidden_states = torch.randn(1, 64, config.hidden_size)
position_ids = torch.arange(64).unsqueeze(0)

with torch.no_grad():
    hf_attn_out = hf_attn(hidden_states, position_ids=position_ids)
    nemo_attn_out = nemo_attn(hidden_states, position_ids=position_ids)

compare_tensors(hf_attn_out[0], nemo_attn_out[0], name="attention_output")
```

Repeat for MLP, norm, and full decoder layer.

### Level 3: E2E Forward Pass (GPU/bfloat16)

Full model forward pass on GPU with bfloat16, reflecting realistic deployment conditions.

```python
device = torch.device("cuda")
dtype = torch.bfloat16

hf_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", torch_dtype=dtype, device_map=device
).eval()
nemo_model = NeMoAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B", torch_dtype=dtype, device_map=device
).eval()

torch.manual_seed(42)
input_ids = torch.randint(0, 32000, (1, 128), device=device)
attention_mask = torch.ones_like(input_ids)

with torch.no_grad():
    hf_out = hf_model(input_ids, attention_mask=attention_mask)
    nemo_out = nemo_model(input_ids, attention_mask=attention_mask)

# Verify shapes match
assert hf_out.logits.shape == nemo_out.logits.shape, (
    f"Shape mismatch: {hf_out.logits.shape} vs {nemo_out.logits.shape}"
)

max_diff, mean_diff, cos_sim = compare_tensors(
    hf_out.logits, nemo_out.logits, name="e2e_logits"
)

# Looser tolerance for bfloat16
assert max_diff < 1e-2, f"E2E parity FAILED: max_diff={max_diff}"
assert cos_sim > 0.9999, f"E2E parity FAILED: cosine_sim={cos_sim}"
print("Level 3 PASSED: E2E forward pass parity within tolerance.")
```

**Tolerances for bfloat16:** max_diff < 1e-2, cosine_similarity > 0.9999. bfloat16 has limited mantissa bits, so per-element differences accumulate across layers.

## Comparison Utilities

```python
def compare_tensors(a, b, name=""):
    """Compare two tensors and report multiple similarity metrics.

    Args:
        a: Reference tensor (from HF model).
        b: Test tensor (from NeMo AutoModel).
        name: Label for the comparison (printed in output).

    Returns:
        Tuple of (max_diff, mean_diff, cosine_similarity).
    """
    max_diff = (a - b).abs().max().item()
    mean_diff = (a - b).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        a.flatten().float(), b.flatten().float(), dim=0
    ).item()
    print(
        f"{name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
        f"cosine_sim={cos_sim:.8f}"
    )
    return max_diff, mean_diff, cos_sim


def compare_state_dicts(sd_a, sd_b, prefix=""):
    """Compare two state dicts key-by-key, reporting per-parameter differences."""
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    missing = keys_a - keys_b
    extra = keys_b - keys_a
    if missing:
        print(f"{prefix}Missing keys: {missing}")
    if extra:
        print(f"{prefix}Extra keys: {extra}")

    shared = keys_a & keys_b
    max_diffs = {}
    for key in sorted(shared):
        diff = (sd_a[key].float() - sd_b[key].float()).abs().max().item()
        if diff > 0:
            max_diffs[key] = diff
            print(f"{prefix}{key}: max_diff={diff:.6e}")

    if not max_diffs and not missing and not extra:
        print(f"{prefix}All {len(shared)} parameters match exactly.")
    return missing, extra, max_diffs
```

## Debugging Workflow

Follow this procedure when a parity test fails.

### Step 1: If E2E fails, isolate to component level

Run Level 2 component tests. Determine which component (attention, MLP, norm, RoPE, decoder layer) introduces the divergence.

### Step 2: If component fails, check weight loading

Verify the loaded and exported weights (Level 1). If they differ, inspect the
checkpoint load/save path and, when present, the adapter's `from_hf()` and
`to_hf()` methods.

```python
# Quick check: load NeMo model, export its weights back to HF format, compare
nemo_sd = nemo_model.state_dict()
adapter = getattr(nemo_model, "state_dict_adapter", None)
exported_hf_sd = adapter.to_hf(nemo_sd) if adapter is not None else nemo_sd
exported_hf_sd = {key: value for key, value in exported_hf_sd.items() if not key.endswith("_extra_state")}
compare_state_dicts(hf_sd, exported_hf_sd, prefix="weight_check: ")
```

### Step 3: If weights match but output differs, check backend

Different backends (TE vs SDPA vs flex attention) can produce different results even with identical weights. Force the baseline backend for parity testing:

```python
from nemo.collections.llm import BackendConfig

# Force SDPA attention and torch linear to match HF behavior
nemo_model = NeMoAutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    backend_config=BackendConfig(attn="sdpa", linear="torch"),
)
```

### Step 4: Injection technique

Replace one NeMo component's output with the HF component's output and check if downstream computation matches. This isolates exactly which component introduces divergence.

```python
# Example: inject HF attention output into NeMo decoder layer
with torch.no_grad():
    # Get HF attention output
    hf_attn_out = hf_model.model.layers[0].self_attn(hidden_states, position_ids=position_ids)

    # Manually run NeMo decoder layer but substitute HF attention output
    nemo_layer = nemo_model.model.layers[0]
    residual = hidden_states
    normed = nemo_layer.input_layernorm(hidden_states)
    # Use HF attention output instead of NeMo attention output
    attn_out = hf_attn_out[0]
    hidden_states_after_attn = residual + attn_out

    # Continue with NeMo MLP
    residual = hidden_states_after_attn
    normed = nemo_layer.post_attention_layernorm(hidden_states_after_attn)
    mlp_out = nemo_layer.mlp(normed)
    final = residual + mlp_out

# If final matches HF decoder layer output, the bug is in NeMo attention.
# If final does NOT match, the bug is in NeMo MLP or norm.
```

### Step 5: Gradient parity

After forward pass parity is confirmed, verify gradients:

```python
hf_model.train()
nemo_model.train()

hf_out = hf_model(input_ids, attention_mask=attention_mask, labels=input_ids)
nemo_out = nemo_model(input_ids, attention_mask=attention_mask, labels=input_ids)

hf_out.loss.backward()
nemo_out.loss.backward()

# Compare gradients for corresponding parameters
for (hf_name, hf_param), (nemo_name, nemo_param) in zip(
    hf_model.named_parameters(), nemo_model.named_parameters()
):
    if hf_param.grad is not None and nemo_param.grad is not None:
        compare_tensors(hf_param.grad, nemo_param.grad, name=f"grad_{hf_name}")
```

## Testing Rules

1. **Always test on CPU/float32 first.** GPU and lower precision introduce noise that masks real bugs.
2. **Test both fresh load and save/reload cycle.** Loading successfully does not prove export and reload preserve weights, metadata, or ties. Test these paths with and without a state dict adapter, as applicable to the model.
3. **Never modify reference HF code.** The HF model is the ground truth. Only modify the NeMo AutoModel implementation.
4. **Use deterministic inputs (torch.manual_seed).** Every test must be reproducible.
5. **Compare all outputs, not just loss.** Loss can match by coincidence even when logits diverge. Always compare logits, hidden states, and attention weights where possible.
6. **Check both forward pass and gradient computation.** Forward parity does not guarantee backward parity, especially with custom kernels.
7. **Verify tied weights are handled correctly.** If `tie_word_embeddings=True`, confirm that `lm_head.weight` and `embed_tokens.weight` share the same tensor after loading.
8. **Test with and without kernel patches.** Liger kernels, SDPA patching, and other optimizations may change numerics. Run parity tests with all patches disabled first, then enable them one at a time.

## Code Anchors

These are the key source files relevant to parity testing:

| Component | Path |
|---|---|
| State dict adapter base | `components/checkpoint/state_dict_adapter.py` |
| Model registry | `_transformers/registry.py` |
| AutoModel entry point | `_transformers/auto_model.py` |
| Kernel patches | `_transformers/kernel_patches.py` |
| Model capabilities | `_transformers/capabilities.py` |
