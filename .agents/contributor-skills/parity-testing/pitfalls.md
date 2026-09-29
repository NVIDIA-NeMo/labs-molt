# Parity Testing Pitfalls for NeMo AutoModel

This document catalogs specific pitfalls that cause parity failures between NeMo AutoModel and HuggingFace reference implementations. Each entry describes the issue, how it manifests, and how to detect or work around it.

## 1. Combined projection interleaving (QKV)

NeMo AutoModel fuses separate Q, K, V projections into a single combined QKV projection with an interleaved weight layout. The `from_hf()` adapter reorders weights from `[Q_all, K_all, V_all]` to an interleaved format grouped by head: `[Q_head0, K_head0, V_head0, Q_head1, K_head1, V_head1, ...]`.

**Symptom:** If the interleaving order is wrong, attention output diverges catastrophically. Loss will be very high from the first step.

**Detection:** Compare attention output per-head. Extract individual head outputs from both implementations and compare them. A correct interleaving produces exact per-head matches; a wrong interleaving scrambles heads.

```python
# Per-head comparison
hf_q = hf_attn_output.view(batch, seq, num_heads, head_dim)
nemo_q = nemo_attn_output.view(batch, seq, num_heads, head_dim)
for h in range(num_heads):
    compare_tensors(hf_q[:, :, h], nemo_q[:, :, h], name=f"head_{h}")
```

## 2. GateUp interleaving (MLP)

The combined GateUp projection uses row-interleaved layout: `[gate_row_0, up_row_0, gate_row_1, up_row_1, ...]`. This differs from the concatenated layout `[gate_all, up_all]` that a naive implementation might assume.

**Symptom:** Wrong interleaving order produces nonsensical MLP output. The model may still produce finite loss initially, making this harder to catch than QKV issues. Over training, loss will plateau at a high value or diverge.

**Detection:** Compare MLP gate and up activations separately. Extract the gate and up components from the fused output and compare each to the reference.

```python
# Split the combined output and compare
nemo_fused = nemo_mlp.gate_up_proj(hidden_states)
gate, up = nemo_fused.chunk(2, dim=-1)  # Only correct if layout matches
hf_gate = hf_mlp.gate_proj(hidden_states)
hf_up = hf_mlp.up_proj(hidden_states)
compare_tensors(gate, hf_gate, name="mlp_gate")
compare_tensors(up, hf_up, name="mlp_up")
```

## 3. TE attention vs SDPA

Transformer Engine (TE) attention uses different internal precision handling compared to PyTorch's Scaled Dot-Product Attention (SDPA). TE may accumulate in higher precision or apply scaling differently, producing slightly different results even with identical inputs and weights.

**Symptom:** Small but consistent numerical differences (max_diff ~1e-3 to 1e-2 in bfloat16) that grow across layers.

**Workaround:** Force the SDPA backend for parity testing to eliminate this variable.

```python
from nemo.collections.llm import BackendConfig
nemo_model = NeMoAutoModelForCausalLM.from_pretrained(
    model_name,
    backend_config=BackendConfig(attn="sdpa"),
)
```

## 4. TE linear vs torch linear

TE linear layers use a different internal weight layout and may apply transpositions or casting that differ from `torch.nn.Linear`. This can cause mismatches even when the logical weight values are identical.

**Symptom:** Output differences that appear immediately from the first linear layer. Differences are consistent, not random.

**Workaround:** Force the torch linear backend for parity testing.

```python
nemo_model = NeMoAutoModelForCausalLM.from_pretrained(
    model_name,
    backend_config=BackendConfig(linear="torch"),
)
```

## 5. RoPE precision

Some HF model implementations compute Rotary Position Embeddings (RoPE) in float32 regardless of the model dtype, then cast the result back. NeMo AutoModel custom implementations may compute RoPE in the native model dtype (e.g., bfloat16).

**Symptom:** max_diff up to 1e-3 in bfloat16 that is concentrated in attention layers and grows with sequence position. Early positions match well; later positions diverge more.

**Detection:** Compare RoPE outputs directly at various sequence positions.

```python
# Compare RoPE cos/sin values at different positions
positions = torch.tensor([[0, 1, 10, 100, 500, 1000]])
hf_cos, hf_sin = hf_model.model.rotary_emb(hidden_states, positions)
nemo_cos, nemo_sin = nemo_model.model.rotary_emb(hidden_states, positions)
compare_tensors(hf_cos, nemo_cos, name="rope_cos")
compare_tensors(hf_sin, nemo_sin, name="rope_sin")
```

## 6. Tied weights (tie_word_embeddings)

When `tie_word_embeddings=True`, `lm_head.weight` and `model.embed_tokens.weight` must be the same parameter. The model establishes this tie, and checkpoint loading must preserve it, with or without a state dict adapter. Independent copies add redundant parameters and can diverge during training.

**Symptom:** Model loads without error but produces slightly different logits. Parameter count is higher than expected. After training, `lm_head.weight` and `embed_tokens.weight` diverge because they receive different gradient updates.

**Detection:**

```python
# Check that weights are aliased (same underlying storage)
assert nemo_model.lm_head.weight.data_ptr() == nemo_model.model.embed_tokens.weight.data_ptr(), (
    "Tied weights are not aliased -- lm_head and embed_tokens are separate copies."
)
```

## 7. DTensor bias handling

Combined projections with biases require special handling under FSDP2 / DTensor sharding. The bias must be gathered and restored correctly during state dict conversion. If the bias gather/restore logic is wrong, the round-trip test (Level 1) will fail for bias parameters.

**Symptom:** State dict round-trip fails specifically for bias keys (e.g., `self_attn.qkv_proj.bias`). Weight keys may pass while bias keys fail.

**Detection:** Run the Level 1 round-trip test and filter failures to bias-only keys.

```python
for key in hf_sd:
    if "bias" in key:
        max_diff = (hf_sd[key] - roundtrip_sd[key]).abs().max().item()
        if max_diff > 0:
            print(f"Bias round-trip failure: {key}, max_diff={max_diff}")
```

## 8. Kernel patch interference

Liger kernels, SDPA patching, and other kernel-level optimizations modify the computation graph and may introduce small numerical differences. These patches are applied globally and can affect models even when not explicitly requested.

**Symptom:** Parity tests pass in a clean environment but fail when kernel patches are active. Differences are small but non-zero.

**Workaround:** Disable all patches for strict parity testing.

```python
nemo_model = NeMoAutoModelForCausalLM.from_pretrained(
    model_name,
    use_liger_kernel=False,
    use_sdpa_patching=False,
)
```

**Validation:** Run the same test with and without patches. The difference between "patches on" and "patches off" should be small and well-characterized.

## 9. Config attribute mismatches

HF model configs may have different default values than NeMo AutoModel expects. For example, `rope_scaling`, `attention_bias`, `mlp_bias`, or `rms_norm_eps` may differ between the two configs.

**Symptom:** Numerical differences that are consistent and reproducible but not explained by weight loading or backend differences.

**Detection:** Always diff the configs before debugging numerical issues.

```python
hf_config = hf_model.config
nemo_config = nemo_model.config

for attr in dir(hf_config):
    if attr.startswith("_"):
        continue
    hf_val = getattr(hf_config, attr, None)
    nemo_val = getattr(nemo_config, attr, None)
    if hf_val != nemo_val and not callable(hf_val):
        print(f"Config mismatch: {attr} = {hf_val} (HF) vs {nemo_val} (NeMo)")
```

## 10. Norm implementation differences

RMSNorm epsilon handling can differ between implementations. TE RMSNorm and torch RMSNorm may produce different results at low precision due to where the epsilon is added relative to precision casting.

**Symptom:** Small differences (max_diff ~1e-4 in bfloat16) that are consistent across all layers. Differences are largest for inputs near zero.

**Detection:** Compare norm outputs directly.

```python
torch.manual_seed(42)
x = torch.randn(1, 64, config.hidden_size, dtype=torch.bfloat16)
hf_norm_out = hf_model.model.layers[0].input_layernorm(x)
nemo_norm_out = nemo_model.model.layers[0].input_layernorm(x)
compare_tensors(hf_norm_out, nemo_norm_out, name="rmsnorm_output")
```

**Workaround:** Force the torch backend (`BackendConfig(linear="torch")`) which uses the standard PyTorch RMSNorm implementation.

## 11. State dict key naming

Custom models may use different layer numbering, module prefixes, or parameter names. The state dict adapter must handle all key transformations bidirectionally. A common issue is that the adapter handles forward mapping (`from_hf`) correctly but misses edge cases in the reverse mapping (`to_hf`).

**Symptom:** `from_hf()` succeeds but `to_hf()` produces keys that do not match the original HF state dict. Or vice versa. Certain layers or modules may be silently skipped.

**Detection:** Run the full round-trip and check for key set equality, not just absence of exceptions.

```python
hf_keys = set(hf_sd.keys())
roundtrip_keys = set(roundtrip_sd.keys())
missing = hf_keys - roundtrip_keys
extra = roundtrip_keys - hf_keys
if missing:
    print(f"Keys lost in round-trip: {sorted(missing)}")
if extra:
    print(f"Keys created in round-trip: {sorted(extra)}")
```

## 12. MoE routing non-determinism

Mixture-of-Experts (MoE) models use routing functions that can be non-deterministic on GPU due to top-k selection with ties and parallel reduction order. Two runs with identical inputs may route tokens to different experts, producing different outputs.

**Symptom:** Parity tests fail intermittently on GPU. Results differ between runs even with the same seed. CPU tests pass consistently.

**Workaround:** Test MoE parity on CPU where operations are deterministic. Alternatively, use fixed routing by manually setting the router output to a known assignment.

```python
# Option 1: CPU-only testing for MoE
device = torch.device("cpu")

# Option 2: Fixed routing (bypass the router)
# Manually assign all tokens to expert 0 for deterministic comparison
with torch.no_grad():
    router_logits = torch.zeros(batch * seq_len, num_experts)
    router_logits[:, 0] = 1.0  # All tokens to expert 0
```

## Quick Reference: Tolerance Table

| Test Level | Dtype | Device | Max Diff Tolerance | Cosine Sim Tolerance |
|---|---|---|---|---|
| Level 1 (round-trip) | float32 | CPU | 0.0 (exact) | 1.0 (exact) |
| Level 2 (component) | float32 | CPU | < 1e-5 | > 0.99999999 |
| Level 3 (E2E) | bfloat16 | GPU | < 1e-2 | > 0.9999 |
| Level 3 (E2E) | float16 | GPU | < 1e-3 | > 0.99999 |

## Quick Reference: Backend Flags for Parity Testing

```python
# Most strict: matches HF behavior exactly
backend_config = BackendConfig(attn="sdpa", linear="torch")

# Disable all kernel patches
model = NeMoAutoModelForCausalLM.from_pretrained(
    model_name,
    backend_config=backend_config,
    use_liger_kernel=False,
    use_sdpa_patching=False,
)
```
