---
name: build-and-dependency
description: Dev environment setup for NeMo AutoModel — container-based development, uv package management, installation options, environment variables, and common build pitfalls.
when_to_use: Setting up a dev environment, adding or removing dependencies, switching container images, configuring environment variables, 'uv sync fails', 'ModuleNotFoundError', 'TransformerEngine version mismatch', stale .venv issues.
---

# Build and Dependency

## Quick Start

Clone and install:

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git && cd Automodel
uv sync --locked --all-groups --extra all
```

Or use the NeMo-AutoModel container from NVIDIA NGC (pick a published tag from
[the NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo-automodel) —
e.g. `26.04`):

```bash
docker pull nvcr.io/nvidia/nemo-automodel:26.04
docker run --gpus all -it nvcr.io/nvidia/nemo-automodel:26.04
```

## Installation Options

### Option 1: NeMo-AutoModel Container (NGC)

The container ships with all dependencies pre-installed at `/opt/Automodel`
(WORKDIR) with the venv at `/opt/venv`. Run as-is:

```bash
docker run --gpus all --network=host -it --rm --shm-size=32g \
    nvcr.io/nvidia/nemo-automodel:26.04 /bin/bash
```

#### Mounting your local checkout into the container

To develop against your host checkout, bind-mount it over `/opt/Automodel` to
override the installed source:

```bash
docker run --gpus all --network=host -it --rm --shm-size=32g \
    -v <local-Automodel-path>:/opt/Automodel \
    nvcr.io/nvidia/nemo-automodel:26.04 /bin/bash
```

Inside the container, patch `pyproject.toml` / `uv.lock` for the PyTorch base
image, then re-sync:

```bash
cd /opt/Automodel
bash docker/common/update_pyproject_pytorch.sh /opt/Automodel
uv sync --locked --all-groups --extra all
```

> **Warning:** the `update_pyproject_pytorch.sh` step is required. Without it,
> `uv sync` will try to reinstall `torch`, which leads to CUDA version
> mismatches and TE import failures — uv cannot recognize the torch baked into
> the PyTorch base container.

### Option 2: uv (Recommended for Local Development)

`--all-groups` pulls the `build`, `docs`, and `test` dev groups (defined in
`pyproject.toml`); drop it for a runtime-only install.

```bash
uv sync --locked --all-groups                          # base + dev groups
uv sync --locked --all-groups --extra cuda             # CUDA support
uv sync --locked --all-groups --extra fa               # flash-attention
uv sync --locked --all-groups --extra moe              # mixture-of-experts
uv sync --locked --all-groups --extra vlm              # vision-language models (core)
uv sync --locked --all-groups --extra vlm-media        # + video/Qwen/Mistral decode (opencv, decord, qwen-utils; FFmpeg-bearing)
uv sync --locked --all-groups --extra diffusion        # diffusion models
uv sync --locked --all-groups --extra diffusion-media  # + diffusion preprocessing/export (imageio-ffmpeg, opencv)
uv sync --locked --all-groups --extra media            # vlm-media + diffusion-media (union)
uv sync --locked --all-groups --extra delta-databricks # Delta Lake / Databricks
uv sync --locked --all-groups --extra msa              # MiniMax M3 MSA sparse prefill (official MiniMax-AI/MSA with the model-private compatibility patch; Linux + SM100; run via `uv run` so torch finds `ninja`)
uv sync --locked --all-groups --extra all              # all standard extras (EXCLUDES media — FFmpeg kept opt-in)
```

The media extras (`vlm-media`, `diffusion-media`, `media`) bundle FFmpeg and are
deliberately **excluded from `all`** and from the container image — add them
explicitly for video/image decode.

### Option 3: uv pip

Full install (matches `uv sync --extra all`):

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"
```

To add NeMo Run submission support to the base package:

```bash
uv venv
source .venv/bin/activate
uv pip install "nemo-automodel[cli]"
```

The `cli` extra is additive: it adds `nemo-run` but does not remove the base
package's core training dependencies, including PyTorch.

## Package Management

Always use `uv`. Do not introduce `pip install` commands in scripts or docs.

| Task | Command |
|---|---|
| Install from lockfile | `uv sync --locked` |
| Add a new dependency | `uv add <package>` |
| Add an optional dependency | `uv add --optional --extra <group> <package>` |
| Regenerate the lockfile | `uv lock` |

## Environment Variables

```bash
export HF_TOKEN="hf_..."           # Hugging Face token for gated models
export WANDB_API_KEY="..."         # Weights & Biases logging
export HF_HOME="/path/to/hf_cache" # Hugging Face cache directory
```

## CLI Usage

The entry point is `automodel` (defined at `nemo_automodel.cli.app:main`).

Pattern: `uv run automodel <config.yaml> [--nproc-per-node N] [--key.subkey value ...]`

```bash
# The YAML's recipe field selects LLM, VLM, diffusion, or retrieval behavior.
uv run automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml --nproc-per-node 8
```

Override any config value from the CLI:

```bash
uv run automodel examples/llm_finetune/llama3_2/llama3_2_1b_squad.yaml \
    --model.pretrained_model_name_or_path meta-llama/Llama-3.2-1B
```

## Common Pitfalls

| Problem | Cause | Fix |
|---|---|---|
| Stale `.venv` after switching branches | Cached environment out of sync | Delete `.venv` and re-run `uv sync --locked` |
| Import errors for optional features (TE, flash-attn, MoE) | Missing extras | Install the matching `uv` extra (`--extra fa`, `--extra moe`, etc.) |
| Import errors for media (`cv2`, `decord`, `qwen_vl_utils`, `imageio_ffmpeg`) | Media extras are opt-in (not in `all`) | Install `--extra vlm-media` (VLM/Qwen/Mistral) or `--extra diffusion-media` (diffusion) |
| TransformerEngine version mismatch | The TE installed by `uv sync` takes precedence over the version baked into the container | Set the desired TE version in `pyproject.toml` / `uv.lock` and re-run `uv sync` — the venv's TE wins, not the container's |
