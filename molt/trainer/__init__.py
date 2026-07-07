"""Training orchestration, algorithms, rollout generation, and worker utilities."""

# Keep this package lightweight so vLLM/Ray workers do not import training backends implicitly.
