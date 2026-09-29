---
name: linting-and-formatting
description: Code style and quality rules for NeMo AutoModel — ruff configuration, naming conventions, type hints, docstrings, copyright headers, and the code review checklist.
when_to_use: Writing or reviewing code for style compliance, fixing ruff errors, understanding type hint or docstring conventions, copyright header questions, or pre-commit failures.
---

# Linting and Formatting

Single source of truth for code style in NeMo AutoModel. Read this before
writing new code or reviewing PRs.

## Formatting and Linting

Run before every commit:

```bash
# Auto-format all source files (line length, quotes, trailing commas, etc.)
ruff format .

# Lint and auto-fix what can be fixed (unused imports, isort, etc.)
ruff check --fix .
```

To check without modifying files:

```bash
ruff format --check .   # exits non-zero if any file would change
ruff check .            # exits non-zero on lint violations
bandit -r app.py nemo_automodel examples scripts tools tutorials -t B614
python tools/lint_no_globals.py   # exits non-zero on globals() / module-namespace mutation
```

To lint a single file or directory:

```bash
ruff format nemo_automodel/components/models/llama/model.py
ruff check --fix nemo_automodel/components/models/llama/
```

### What linting enforces (from `pyproject.toml`)

| Rule | ID | Description |
|---|---|---|
| Line length | — | 120 characters (formatter) |
| Quote style | — | Double quotes |
| Unused imports | F401 | Auto-removed by `--fix` (ignored in `__init__.py`) |
| Unused variables | F841 | Auto-removed by `--fix` |
| Undefined names | F821 | Error |
| f-string without placeholders | F541 | Error |
| Import sorting | I | isort-compatible ordering, auto-fixed |
| Docstring convention | D101/D103 | Google style (currently ignored — selected then suppressed) |
| No pickle | S301/S403 | Security: forbids `pickle.load` |
| Restricted PyTorch load | B614 | Requires explicit `weights_only=True` for `torch.load` |
| No module-namespace mutation | `tools/lint_no_globals.py` | Bans `globals()`, module-scope `vars()`, `sys.modules[__name__].__dict__` and `setattr(sys.modules[__name__], ...)`; only the PEP 562 cache `globals()[name] = ...` inside a module-level `__getattr__(name)` is exempt |
| Ambiguous variable names | E741 | Error (e.g., `l`, `O`, `I`) |

Tests (`tests/`) are excluded from Ruff, Bandit and the module-namespace
check. Docstring rules (`D`) are also relaxed in test files.

## Type Hints

Required on all public API functions and methods.

- Use `T | None` instead of `Optional[T]`
- Use `X | Y` instead of `Union[X, Y]`
- Use built-in generics (`list`, `dict`, `tuple`) instead of `typing` equivalents

## Docstrings

Google-style where docstrings are added:

```python
def build_model(config: dict) -> torch.nn.Module:
    """Instantiate and shard the model from config.

    Args:
        config: Mapping with _target_ and model hyperparameters.

    Returns:
        Fully initialized model ready for distributed training.
    """
```

## NVIDIA Copyright Header

Every Python file must start with the NVIDIA copyright block. Do not remove or
modify it. Use the current year (2026).

```python
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
```

## Additional Style Conventions

- **Explicit over implicit.** Inline logic where possible; avoid hiding behavior
  behind unnecessary layers of indirection.
- **No speculative abstractions.** Do not add features, parameters, or
  generalization beyond what is explicitly asked for.
- **Optional dependencies** must be guarded with `safe_import()` from
  `nemo_automodel.shared.import_utils`. Never let an optional import crash
  module loading.
- **Components must not import each other** — enforced by `import-linter`
  (see `pyproject.toml`).

## Code Review Checklist

1. **Copyright header** present on all new Python files
2. **Type hints** on all public functions and methods
3. **Docstrings** on public classes and functions (Google style)
4. **Double quotes** for strings
5. **No bare `print()`** — use `logging.getLogger(__name__)`
6. **No commented-out code** without explanation
7. **Optional imports** guarded with `safe_import()`
8. **No cross-component imports** between `components/` subdirectories

## Automated Review

The review-only maintainability heuristics and thresholds live in
`.github/workflows/claude-review.yml`. Keep repository-wide coding rules here
and automated-review prompt policy there so the detailed checklist has one
source of truth.
