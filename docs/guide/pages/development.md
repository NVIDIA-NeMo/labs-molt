# Development

How to check a change, keep these docs correct, and contribute.

> Source: [`CONTRIBUTING.md`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/CONTRIBUTING.md) · [`docs/guide/build_reference.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/docs/guide/build_reference.py)

## Check a change

```bash
python -m compileall -q molt examples/python tests
pytest -q
```

Inside the container:

```bash
SKIP_BUILD=1 DOCKER_GPUS=all DOCKER_SHM_SIZE=32g \
  bash examples/scripts/docker_run.sh "pytest -q"
```

## Keep the docs correct

The [RL flags](cli-rl.md), [SFT flags](cli-sft.md), and [molt.agents API](api-agents.md) pages are
generated from the source. After changing a flag or the agents API:

```bash
python docs/guide/build_reference.py
```

```bash
python docs/guide/build_reference.py --check
```

The check fails if a generated page is stale, or if any page names a flag that neither CLI
defines.

## Contribute

External contributions are welcome. See
[CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/CONTRIBUTING.md). Every commit must be signed off with
`git commit -s`, per the [Developer Certificate of Origin](https://developercertificate.org/).

Model and parallel-stack changes to AutoModel-Slim go to the `automodel-slim` branch as PRs, with
the same `cicd` label check as this repository.

## Cite

```bibtex
@techreport{hu2026molt,
  title       = {Molt: A Hugging Face-Native, Minimal, and Scalable Framework for Agentic Reinforcement Learning},
  author      = {Jian Hu and Molt Contributors},
  institution = {NVIDIA},
  year        = {2026},
  url         = {https://www.researchgate.net/publication/409325071_Molt_A_Scalable_PyTorch-Native_Training_Framework_for_Agentic_Reinforcement_Learning}
}
```

## License

[Apache License 2.0](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/LICENSE). Copyright and third-party attributions are in
[NOTICE](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/NOTICE) and [THIRD_PARTY_NOTICES.md](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/THIRD_PARTY_NOTICES.md).
