<div align="center">

# 🪶 AutoModel-Slim

**NeMo AutoModel, trimmed to what [molt](https://github.com/NVIDIA-NeMo/labs-molt) needs. This is where molt's model backend is developed now.**

[![CICD AutoModel-Slim](https://github.com/NVIDIA-NeMo/labs-molt/actions/workflows/cicd-automodel-slim.yml/badge.svg?branch=automodel-slim)](https://github.com/NVIDIA-NeMo/labs-molt/actions/workflows/cicd-automodel-slim.yml)
[![Base: Automodel 8f73178c](https://img.shields.io/badge/base-Automodel%208f73178c-76b900)](https://github.com/NVIDIA-NeMo/Automodel/tree/8f73178ca51d4c1e55ccf05df5da6540a9e24f7e)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue)](#-provenance-and-license)

🧩 [Model families](#-model-families) · ✅ [Parity with upstream](#-parity-with-upstream) · 🧱 [What is kept](#-what-is-kept) ·
🧹 [What was removed](#-what-was-removed) · 📦 [Installation](#-installation) · 🤝 [Contributing](#-contributing) · 🔁 [CI](#-ci) ·
🔧 [Maintenance](#-maintenance) · 📄 [Provenance](#-provenance-and-license)

| | upstream `8f73178c` | AutoModel-Slim |
|---|---|---|
| Python files / lines | 758 / 282k | 252 / ~97k |
| model families | 58 | 10 (below) + shared utilities |
| tests | 1,005 files | 194 files (kept areas only) |
| forward logits vs upstream | | bit-exact on every kept family ([table](#-parity-with-upstream)) |

</div>

AutoModel-Slim is molt's model backend: [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel)
at commit `8f73178c` with everything molt does not use removed, same package name (`nemo_automodel`),
same import paths, same API. It is **PyTorch-native and Hugging Face-native**: a Hugging Face checkpoint
(config, safetensors, tokenizer) is loaded into a native PyTorch implementation, sharded with FSDP2 /
tensor / expert / context parallelism, packed for TransformerEngine attention, and written back as DCP
checkpoints plus consolidated Hugging Face safetensors that `transformers` and vLLM load directly. molt's
RL and SFT actor delegates all of that to it.

## 🧩 Model families

| Family | Directories | Architectures |
|---|---|---|
| Qwen2 / Qwen2.5 | `qwen2` | `Qwen2ForCausalLM` |
| Qwen3 | `qwen3`, `qwen3_moe`, `qwen3_next`, `qwen3_vl`, `qwen3_vl_moe` | `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`, `Qwen3VLForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration` |
| Qwen3.5 / 3.6 | `qwen3_5`, `qwen3_5_moe` | `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` |
| Qwen3.8 Flash Next | `qwen3_8_flash_next` | `Qwen3_8_FlashNextForConditionalGeneration`, `Qwen4ExpForConditionalGeneration` |
| DeepSeek V4.1 Flash | `deepseek_v41`, `deepseek_v4`, `deepseek_v3` (shared RoPE / FP8 helpers) | `DeepseekV41ForCausalLM` (subclasses the V4 implementation) |
| GLM 5.x, 5.3-Flash | `glm_moe_dsa`, `glm5_next`, `glm4_moe` (shared adapter helpers) | `GlmMoeDsaForCausalLM`, `Glm5NextForConditionalGeneration` |
| Gemma 4 (dense, MoE, unified) | `gemma4_moe`, `gemma4_unified` | `Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration` |
| Nemotron 3 / 3.5 | `nemotron_v3`, `nemotron_omni` | `NemotronHForCausalLM`, `NemotronH_Nano_Omni_Reasoning_V3`, `NemotronH_Omni_Reasoning_V3` |
| Muse Glimmer | `muse_glimmer` | `MuseGlimmerForConditionalGeneration` |
| Inkling | `inkling` | `InklingForConditionalGeneration` |

`llama/` and `gpt_oss/` hold only the RoPE helpers the Qwen and GLM implementations import; there is
no Llama or GPT-OSS model class. Dense HF architectures without a native implementation still load
through the plain transformers path (FSDP2, with TP plans for Qwen2 / Qwen3).

## ✅ Parity with upstream

Two kinds of evidence per family. **Logits**: same checkpoint, same input batch, forward logits compared to the last bit between upstream
`8f73178c` and this tree, both built the way molt builds its actor (FSDP2 + EP mesh, TE attention,
fp32 masters under a bf16 mixed-precision policy, THD packing or the padded forward molt uses for the
GDN families). Families whose checkpoints are hundreds of GB run their real config cut to a few layers
with seeded random weights; run on 8 H100 with the molt `0.1.10` image on 2026-09-29 against tree
`0b35e05`. **molt RL e2e**: the family was trained end to end in molt (vLLM rollout, weight refit,
checkpointing) at the listed scale and parallelism, and the metrics matched upstream's run.

| Family | Build | Parallelism | Tokens compared | max abs diff | Logits | molt RL e2e |
|---|---|---|---|---|---|---|
| Qwen2.5 (R1-Distill-Qwen-1.5B) | real weights | FSDP2 ×8, THD | 924 | 0 | ✅ bit-exact | dense 1.5B |
| Qwen3-MoE (Qwen3-30B-A3B) | real weights | EP8, THD | 924 | 0 | ✅ bit-exact | 30B-A3B, EP8 |
| Qwen3.6 (Qwen3.6-35B-A3B) | real weights | EP8, padded | 2 × 512 | 0 | ✅ bit-exact | 35B-A3B, EP8 / CP8 |
| Qwen3.5 dense (27B config) | seeded, 4 layers | FSDP2 ×8, padded | 2 × 512 | 0 | ✅ bit-exact | — |
| Qwen3.8 Flash Next | seeded, 6 layers | EP8, flex attention, THD | 924 | 0 | ✅ bit-exact | — |
| DeepSeek V4.1 Flash | seeded, 2 layers | EP8, tilelang DSA, padded | 2 × 512 | 0 | ✅ bit-exact | — |
| GLM 5.3 | seeded, 5 layers | EP8, tilelang DSA, THD | 924 | 0 | ✅ bit-exact | GLM-5.2 753B, EP / CP16 |
| GLM 5.3-Flash | seeded, 8 layers | EP8, THD | 924 | 0 | ✅ bit-exact | — |
| Gemma 4 MoE (26B-A4B) | real weights | EP8, THD | 924 | 0 | ✅ bit-exact | 26B-A4B, EP8 |
| Nemotron 3 Nano (30B-A3B) | real weights | EP8, THD | 924 | 0 | ✅ bit-exact, see the note below | Nano 30B-A3B, EP8 |
| Muse Glimmer (30B) | real weights | FSDP2 ×8, THD | 924 | 0 | ✅ bit-exact | 30B, TP2 |
| Inkling Small | seeded, 4 layers | EP8, THD | 924 | 0 | ✅ bit-exact | — |

**Note on Nemotron 3.** On the THD path it is not run-to-run reproducible in upstream itself: seven upstream runs gave
three distinct logit tensors (pairwise max diff 2.4 to 2.7, mean 0.06), and each of the three slim runs
(three commits) is bit-exact with one of them, so the variance is upstream's, not a slim difference. The
Mamba-2 SSD Triton kernels (`_chunk_scan`, `_state_passing`, `_bmm_chunk`, `_chunk_cumsum`) are autotuned
at process start on this path, a known source of run-to-run variance; the padded path is bit-exact every
time. Nemotron 3 Omni shares this model code; no checkpoint was available to run it.

molt RL e2e runs (Qwen2 dense, Qwen3.6-35B, Nemotron 3, Gemma 4, Muse) reproduce the upstream
metrics within run-to-run noise. A 4-hour slim-vs-upstream RL A/B on DeepSeek-R1-Distill-Qwen-1.5B
(same recipe, image, data and seeds; 8 GPUs; ~310 updates each), means over the run:

| | upstream | AutoModel-Slim |
|---|---|---|
| reward | 0.600 | 0.598 |
| `vllm_kl` (train vs rollout) | 5.0e-4 | 5.2e-4 |
| train / rollout log-prob diff | 2.6e-3 | 2.7e-3 |
| gradient norm | 0.121 | 0.116 |
| in-training AIME24 / AIME25 pass@1 at the end | 22.5 / 18.3 | 25.8 / 17.5 |

## 🧱 What is kept

The molt dependency surface, and nothing else:

- **Loading (Hugging Face in, Hugging Face out)** · `_transformers/` (`auto_model`, `model_init`, `infrastructure`, `registry`, `te_attention`, `capabilities`, `mfu`)
  - `NeMoAutoModelForCausalLM` / `NeMoAutoModelForImageTextToText.from_pretrained(..., distributed_setup=, backend=, has_packed_sequence=)`, `AutoMFU`
- **Parallelism** · `components/distributed/`
  - `FSDP2Config`, `MoEParallelizerConfig`, `DistributedSetup`, `MeshContext`, `ParallelismSizes`, `_create_device_meshes`, `get_flat_mesh`, `ContextParallelSharder` (round-robin and block-diagonal CP, model-owned CP hooks), activation checkpointing, THD utilities
- **MoE** · `components/moe/`
  - `Gate` (fp32 router, aux loss), grouped experts (torch / TE / DeepEP), DeepEP + HybridEP dispatch, EP parallelizer with AC recompute replay, `RouterReplay`, `MoEAuxLossAutoScaler`, expert state-dict mixin, expert LoRA
- **Checkpointing** · `components/checkpoint/`
  - `Checkpointer`, `CheckpointingConfig`, `OptimizerState`: DCP save / load (every model, any world size), consolidated HF safetensors export, PEFT adapter export
- **Training utilities** · `components/training/`, `components/_peft/`, `components/quantization/`, `components/optim/`
  - `scale_grads_and_clip_grad_norm` (with the Triton fused grad-norm kernel), `PeftConfig` / `LinearLoRA` (+ MoE expert LoRA), FP8 (incl. MXFP8 experts), Dion / Muon
- **Misc** · `components/utils/`, `components/attention/`, `shared/`
  - `filter_forward_kwargs`, `canonical_parameter_fqn`, FLOPs formulas for the kept architectures, flex / FFPA attention used by Qwen3.8 and Gemma 4 CP

Refit to vLLM uses each model's `state_dict_adapter.convert_single_tensor_to_hf`.

## 🧹 What was removed

- **Training loops and tooling**: recipes and training loops (SFT, KD, VLM), datasets, the CLI and launcher,
  loggers, evaluation, tokenizer wrappers, the capability-based model docs.
- **Model features molt does not train**: speculative decoding (DSpark, EAGLE, drafters), diffusion,
  retrieval encoders and the sequence / token-classification heads, and 48 model families.
- **Parallel strategies and kernels**: pipeline parallelism, DDP and Megatron-FSDP strategies, UCCL-EP and
  Mixture-of-Kittens dispatchers, MagiAttention, Triton LoRA kernels and the fused LoRA MLP, the
  torch.compile wrapper, the parallel strategies of removed families.
- **Quantization and precision paths**: QAT and QLoRA, the bitsandbytes 4-bit loading path.
- **Checkpoint paths**: the full-state-dict and `.bin` load paths (every model, including single-GPU custom
  models, loads through DCP), the diffusers-compatible export flag, the checkpoint retention lifecycle and
  its pointer-file helpers.
- **Leftovers**: transformers-v4 compatibility patches, the MFU formulas of removed families, unreferenced
  training-stack helpers. Tests bound to removed code were dropped.

Kept files were not refactored. Where a kept file imported a removed module, the import and the branch
that used it were deleted; nothing else was rewritten.

## 📦 Installation

AutoModel-Slim needs the same stack as molt: CUDA 13 with `torch==2.13.0+cu130`, TransformerEngine,
flash-linear-attention for the GDN models, DeepEP / HybridEP for MoE, mamba and causal-conv1d for
Nemotron 3, tilelang for the DSA models. The molt container (`hijkzzz/molt`) ships all of it, and every
command below runs inside it unchanged.

**Standalone.** The extras are the upstream ones: `cuda` = TransformerEngine, tilelang, mamba /
causal-conv1d; `fla` = flash-linear-attention; `moe` = DeepEP; `fa`, `ffpa`, `vlm` as upstream.

```bash
pip install "nemo-automodel[cuda,fla,moe] @ git+https://github.com/NVIDIA-NeMo/labs-molt.git@automodel-slim"
```

**From molt.** molt installs upstream AutoModel by default; `setup.py` holds both pins in `AUTOMODEL` and
`MOLT_AUTOMODEL=slim` selects this branch by name, for a local install or an image build:

```bash
MOLT_AUTOMODEL=slim pip install -e ".[vllm]"                                          # local install
docker build --build-arg MOLT_AUTOMODEL=slim -f dockerfile/Dockerfile -t molt:slim .   # image
```

The slim install follows the branch head: a merge here is live on the next molt image build or
reinstall, and `pip` records the exact commit it installed in `direct_url.json`.

**Working on the backend.** Check this branch out next to molt and point the recipes at it; the sibling
checkout wins over the package baked into the image:

```bash
git clone -b automodel-slim https://github.com/NVIDIA-NeMo/labs-molt.git automodel-slim
EXTRA_PYTHONPATH=$PWD/automodel-slim sbatch examples/scripts/slurm/rl_qwen3_6_35b.sh   # slurm recipes
PYTHONPATH=/path/to/automodel-slim python -m molt.cli.train_rl_ray ...                   # in the container
```

The branch is an orphan (one squashed snapshot plus later commits, no upstream history); the same tree
with full upstream history is mirrored at `hijkzzz/Automodel@molt-slim` for cherry-picking.

## 🤝 Contributing

- Open PRs against `automodel-slim`. Keep them to one problem each; deletions are welcome, additions
  need a molt use.
- A maintainer puts the `cicd` label on the PR to run the GPU check on that exact head commit (the
  label is consumed as the run starts; push again, label again). The `Nemo_CICD_Test` check is what
  branch protection looks at.
- Comment `/claude review` for a Claude Code review; it runs at the effort level in `.claude/settings.json`.
- Commits carry `Signed-off-by` (DCO) and follow the upstream Apache-2.0 headers.

## 🔁 CI

`.github/workflows/cicd-automodel-slim.yml` is run the same way as molt's CICD (manual dispatch with an
image and runner input, the `cicd` label on PRs, every push to the branch; failures on push go to the
CI Slack channel) and finishes in about seven minutes on the arm64 GB200 runner:

| job | where | what |
|---|---|---|
| `static` | ubuntu | `compileall`, `ruff --select F821,F401,F822`, `tools/check_closure.py` (every import and every `nemo_automodel.*` string resolves) |
| `test` · unit | molt image, CPU container | the Qwen / MoE / distributed / checkpoint / loading unit tests, ~1,960 cases, alongside the smoke |
| `test` · smoke | molt image, 2 GPUs | `tests/gpu_smoke/smoke.py` on six families, below |

The smoke builds seeded few-layer Qwen2.5, Qwen3, Qwen3-MoE, Qwen3.6-MoE, Inkling and Nemotron 3 (MoE
families on EP2) the way molt builds its actor: FSDP2 + EP mesh, TE attention, THD packing or the padded
forward molt uses for Qwen3.5-MoE. Per family it checks that

- the consolidated HF export (the path molt's vLLM refit uses) loads into `transformers` with no missing or
  unexpected keys;
- that `transformers` model, in bf16, reproduces the forward logits within a relative tolerance sized to bf16
  noise (`rel_fro` 2e-2, `rel_max` 5e-2; measured 3e-3 to 1.6e-2). Qwen3.6 and Inkling get the export check
  only: their tiny random builds disagree with `transformers` far beyond the other families (Qwen3.6's
  real-weight numerics are molt's e2e job; Inkling's disagreement is an open item);
- Qwen3.6 also runs three AdamW steps with molt's mesh-aware clip (the loss must fall) and reloads the DCP
  checkpoint saved beforehand, which must reproduce the original logits.

Each run writes a per-family table (export, logits parity, train loss, reload, seconds) and the unit-test
result to the job summary. There is no Dockerfile here: the image is molt's, and the checkout under test is
mounted over the copy baked into it. JIT-compiled kernels (Triton for GDN, Inductor) persist between runs
through `actions/cache`. Real-weight validation against upstream is the [parity table](#-parity-with-upstream)
above; add an `ARCHS` entry to the smoke when another family needs protecting, with a `configs/*.json`.
Gemma 4 is not in the smoke yet: its per-layer schedule is derived from the layer count by transformers'
config and does not survive the shrink.

## 🔧 Maintenance

- **Delete, do not add.** A change goes in only if molt uses it. Do not refactor kept files; a
  removal cuts the import and the branch that used it, nothing more.
- **Gates before merge.** `python -m compileall -q nemo_automodel`, `ruff check --select F821,F401,F822`,
  and the three closure checks (every import resolves to a kept module, every imported name exists,
  every `nemo_automodel.*` string names a kept module). The branch CI's smoke gates the Qwen families
  against transformers; a change to another family's model code should keep its forward logits
  bit-exact against the previous commit (same checkpoint, same input), and a change to loading, EP or
  CP must run one molt RL e2e (dense Qwen2 and Qwen3.6-35B EP8 / CP8).
- **Picking changes up in molt.** molt's `setup.py` pins this branch by name (`AUTOMODEL["slim"]`), so a
  merge here is live on the next molt image build or reinstall; molt's own e2e CI is the acceptance test.
- **Adding a family.** Copy `components/models/<family>/` from upstream (or the mirror), restore its
  `_transformers/registry.py` entries, run the closure checks to see which removed helpers it needs,
  and compare logits against upstream with real weights or a seeded few-layer build.
- **Known load-bearing pieces that look removable but are not:**
  - `_transformers/capabilities.py`: `model.supports` is read by the EP / CP parallelizers.
  - `components/models/deepseek_v4/kernels/`: loaded by name through `safe_import_from`.
  - `components/distributed/blockdiag_cp/`: imported by the Qwen3.5 / 3.8 model code, also at CP = 1.
  - `checkpoint/addons.py`'s custom-model-code export: Nemotron 3 Nano is a remote-code checkpoint.
  - the `Union` form of `DistributedStrategyConfig`: importers write `... | None`.

## 📄 Provenance and license

Base: NVIDIA-NeMo/Automodel `8f73178ca51d4c1e55ccf05df5da6540a9e24f7e` (the commit molt pinned
before this branch existed). Apache-2.0, unchanged; upstream copyright notices are kept in every file.
