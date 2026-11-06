# Liger Qwen3 validation

`--fsdp.use_liger_kernel` is an opt-in for dense text Qwen3. It patches only
Qwen3 MLP and RMSNorm modules; it does not change attention, RoPE, packing
semantics, or the policy/language-model-head loss.

Run the correctness suite through pytest under `torchrun`; executing the test
file directly only defines tests and does not run assertions.

```bash
torchrun --standalone --nproc-per-node=1 -m pytest -q tests/gpu/test_liger_qwen3.py
torchrun --standalone --nproc-per-node=2 -m pytest -q tests/gpu/test_liger_qwen3.py
```

The suite compares native AutoModel baseline and Liger runs for padded BSHD and
TE-packed THD Qwen3. It checks logits, action log-probabilities, entropy,
policy loss, finite gradients, and one AdamW update; the two-rank case also
checks the FSDP-reduced loss.

Performance results are only meaningful after the correctness suite passes.
For each layout, collect five independent baseline and Liger runs with the
same model, GPU count, sequence-length distribution, optimizer, warmups, and
measured steps. Report median step time, tokens per second, and peak allocated
and reserved memory for each arm, plus the raw measurements. Claim an
improvement only when Liger is at least 5% better and a bootstrap confidence
interval excludes no improvement; otherwise report the result without a
performance claim.

Use the tracked harness to produce one raw JSON file per arm and repetition:

```bash
torchrun --standalone --nproc-per-node=2 tests/gpu/benchmark_liger_qwen3.py \
  --model /path/to/Qwen3 --output results/padded-baseline-1.json
torchrun --standalone --nproc-per-node=2 tests/gpu/benchmark_liger_qwen3.py \
  --model /path/to/Qwen3 --use-liger-kernel --output results/padded-liger-1.json
torchrun --standalone --nproc-per-node=2 tests/gpu/benchmark_liger_qwen3.py \
  --model /path/to/Qwen3 --packing-samples --output results/packed-baseline-1.json
torchrun --standalone --nproc-per-node=2 tests/gpu/benchmark_liger_qwen3.py \
  --model /path/to/Qwen3 --packing-samples --use-liger-kernel --output results/packed-liger-1.json
```
