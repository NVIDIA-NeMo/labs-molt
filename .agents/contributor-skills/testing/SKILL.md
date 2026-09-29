---
name: testing
description: Testing reference for NeMo AutoModel — unit and functional test layout, tier semantics (L0/L1/L2), running tests locally, adding or disabling tests, and pytest conventions.
when_to_use: Adding, running, or disabling tests; debugging a test failure; choosing the right test tier; understanding L0 vs L1 vs L2; handling flaky tests; 'add a test', 'which tier', 'functional test layout'.
---

# Testing

## Directory Layout

```
tests/
  unit_tests/          # fast, isolated, no GPU required
  functional_tests/    # GPU-required integration tests
  ci_tests/            # executed in CI pipeline only
```

Unit tests must be CPU-compatible. Functional tests require at least one GPU.
CI test scripts live in `tests/ci_tests/` and should not be run locally
unless reproducing a CI failure.

## Test Tiers

| Tier | Trigger | Blocking |
|---|---|---|
| L0 | Every PR | Yes — PR cannot merge if L0 fails |
| L1 | PRs with `needs-more-tests` label, scheduled | Yes |
| L2 | Scheduled only | Yes (when triggered) |

**Prefer unit tests over functional tests.** CI GPU resources are limited;
every functional test slot has a real cost.

## Running Tests Locally

### Unit Tests (CPU)

```bash
pytest tests/unit_tests/ -v
```

### Functional Tests (GPU Required)

```bash
pytest tests/functional_tests/ -v
```

These require at least one GPU. Mark GPU tests with:

```python
@pytest.mark.gpu
```

or:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")
```

## Adding a Unit Test

1. Place the file under `tests/unit_tests/<domain>/test_<name>.py`.
2. Keep configs tiny: small hidden dims, 1-2 layers, short sequences.
3. Run locally: `pytest tests/unit_tests/<your_test>.py -v`

**No foreign `setattr` on config dataclasses in tests.** When applying
overrides via `setattr(config_obj, key, value)`, always guard first:

```python
if not hasattr(config_obj, key):
    raise ValueError(f"Config has no field '{key}'")
setattr(config_obj, key, value)
```

Setting a non-existent attribute silently creates a phantom field — the test
passes but the recipe fails for a real user.

## Adding a Functional Test

1. Place the test under `tests/functional_tests/`.
2. Functional tests are capped at **2 GPUs** in CI.
3. Set `CUDA_VISIBLE_DEVICES` explicitly for multi-GPU tests.

## Tips

- Keep unit test configs tiny: small hidden dims, 1-2 layers, short sequences.
- Set `CUDA_VISIBLE_DEVICES` explicitly when running multi-GPU tests locally.
- Watch for port conflicts when running multiple `torchrun` processes.
  Use `--master_port` to avoid collisions.
