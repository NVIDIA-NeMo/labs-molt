# Installation

Molt runs on CUDA 13. The prebuilt container carries the whole stack and is the supported way to
run it. A local install is for development outside the container.

> Source: [`dockerfile/Dockerfile`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/dockerfile/Dockerfile) · [`setup.py`](https://github.com/NVIDIA-NeMo/labs-molt/blob/main/setup.py)

## Get the code

Clone the repository first. The launch scripts, agents, and recipes live in it, and the container
mounts this checkout.

```bash
git clone https://github.com/NVIDIA-NeMo/labs-molt.git
cd labs-molt
```

## Container

```bash
docker pull hijkzzz/molt:latest
```

The image bakes torch 2.14, vLLM, TransformerEngine, flash-attn, mamba, DeepEP, and NeMo AutoModel
for A100, H100, H200, B200, and GB200. SFT and RL run in it as-is.

| You want | Command |
|---|---|
| A pinned release | `docker pull hijkzzz/molt:0.1.10` |
| Different CUDA, vLLM, or AutoModel pins | `docker build -f dockerfile/Dockerfile -t hijkzzz/molt:latest .` |
| To run a command inside the image | `bash examples/scripts/docker_run.sh "pytest -q"` |

Tags from 0.1.9 are multi-arch, so the same pull works on x86 and on Grace-Blackwell hosts.

## Local install

```bash
pip install -e ".[vllm]"
```

- The command installs the git-pinned AutoModel this repository is validated against, so routing
  replay and Muon work out of the box.
- The vLLM pin is the 0.34.0 release from PyPI.
- CUDA 13 is required (`torch==2.14.0+cu130`). A CUDA 12 environment will not work.
- With a host driver older than 580, use the container. It ships the CUDA forward-compatibility
  layer and enables it automatically.
- The `molt-rl` package on PyPI is not a supported install path, because it cannot carry the
  git-pinned AutoModel.

## Optional backend: AutoModel-Slim

The default trainer is upstream NeMo AutoModel, pinned in `setup.py`. Everything above installs it.

[`automodel-slim`](https://github.com/NVIDIA-NeMo/labs-molt/tree/automodel-slim) is Molt's own copy of that pinned commit, trimmed to
the model families and training stack Molt uses. The package name (`nemo_automodel`), import paths,
and API are the same, so Molt's code does not change when you switch.

| Where | How to opt in |
|---|---|
| Local install | `MOLT_AUTOMODEL=slim pip install -e ".[vllm]" --extra-index-url ...` |
| Image | `docker build --build-arg MOLT_AUTOMODEL=slim -f dockerfile/Dockerfile -t molt:slim .` |
| Slurm recipes | `EXTRA_PYTHONPATH=/path/to/automodel-slim`, where a checkout wins over the baked-in package |

`setup.py` holds both pins. The slim install follows the branch head, so a merge there is live on
the next image build or reinstall. The branch README lists the kept and removed families and the
parity results against upstream.

## Next

[Quick start](quick-start.md) runs a first RL job.
