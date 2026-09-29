# Slim AutoModel fork — guide for AI agents

This is a trimmed fork of NVIDIA-NeMo/Automodel that serves as molt's training-side model
backend. See README.md for the kept model families and the removed feature list.

Rules (details: `.claude/skills/simplicity-first`, invoke before any code change):

- Do not add recipes, datasets, launchers or training loops here; molt owns the training loop.
- Keep upstream file layout and names so upstream model directories can be dropped in
  (`components/models/<family>/` + a `registry.py` entry) without edits elsewhere.
- Every deletion must keep `ruff check --select F821,F401` clean and
  `python -m compileall -q nemo_automodel` passing; every kept file imports only kept modules.
- Parallelism (FSDP2, TP, EP with DeepEP/HybridEP, CP incl. the model-owned and blockdiag
  CP paths) is load-bearing for molt and must not be reduced.
- Verify behaviour changes with molt's e2e recipes (dense Qwen3 + Qwen3.6-35B-A3B EP/CP),
  not only with unit tests.
