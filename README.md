# AutoModel-Slim

[![CICD AutoModel-Slim](https://github.com/NVIDIA-NeMo/labs-molt/actions/workflows/cicd-automodel-slim.yml/badge.svg?branch=automodel-slim)](https://github.com/NVIDIA-NeMo/labs-molt/actions/workflows/cicd-automodel-slim.yml)

**NeMo AutoModel, trimmed to what [molt](https://github.com/NVIDIA-NeMo/labs-molt) needs. This is where molt's model backend is developed now.**

molt runs RL and SFT with vLLM rollouts and an FSDP2 training actor. The actor delegates model
construction to AutoModel: loading a Hugging Face checkpoint into a native implementation, sharding it
with FSDP2 / tensor / expert / context parallelism, packing sequences for TransformerEngine attention,
and saving DCP checkpoints plus consolidated HF safetensors. This branch carries molt's copy of that
backend: [NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) at commit `8f73178c` with
everything molt does not use removed, the same package name, import path and API.

| | upstream `8f73178c` | AutoModel-Slim |
|---|---|---|
| Python files / lines | 758 / 282k | 249 / ~99k |
| model families | 58 | 10 (below) + shared utilities |
| tests | 1,005 files | 194 files (kept areas only) |
| forward logits vs upstream | | bit-exact on every kept family |

- [Model families](#model-families)
- [What is kept](#what-is-kept-the-molt-dependency-surface) · [What was removed](#what-was-removed)
- [Using it from molt](#using-it-from-molt) · [Contributing](#contributing) · [CI](#ci)
- [Maintenance rules](#maintenance) · [Provenance](#provenance-and-license)

## Model families

| Family | Architectures | Directories | Validation |
|---|---|---|---|
| Qwen2 / Qwen2.5 | `Qwen2ForCausalLM` | `qwen2` | bit-exact logits vs upstream (R1-Distill-Qwen-1.5B); molt RL e2e |
| Qwen3 | `Qwen3ForCausalLM`, `Qwen3MoeForCausalLM`, `Qwen3NextForCausalLM`, `Qwen3VLForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration` | `qwen3`, `qwen3_moe`, `qwen3_next`, `qwen3_vl`, `qwen3_vl_moe` | bit-exact (Qwen3-30B-A3B, EP8); molt RL e2e EP8 |
| Qwen3.5 / 3.6 | `Qwen3_5ForCausalLM`, `Qwen3_5ForConditionalGeneration`, `Qwen3_5MoeForCausalLM`, `Qwen3_5MoeForConditionalGeneration` | `qwen3_5`, `qwen3_5_moe` | bit-exact (Qwen3.6-35B-A3B, EP8); molt RL e2e EP8 / CP8 / routing replay |
| Qwen3.8 Flash Next | `Qwen3_8_FlashNextForConditionalGeneration`, `Qwen4ExpForConditionalGeneration` | `qwen3_8_flash_next` | bit-exact (seeded 6-layer build) |
| DeepSeek V4.1 Flash | `DeepseekV41ForCausalLM` (subclasses the V4 implementation) | `deepseek_v41`, `deepseek_v4`, `deepseek_v3` (shared RoPE / adapter helpers) | bit-exact (seeded 2-layer build, tilelang DSA) |
| GLM 5.x, 5.3-Flash | `GlmMoeDsaForCausalLM`, `Glm5NextForConditionalGeneration` | `glm_moe_dsa`, `glm5_next`, `glm4_moe` (shared adapter helpers) | bit-exact (seeded 5- and 8-layer builds) |
| Gemma 4 (dense, MoE, unified) | `Gemma4ForConditionalGeneration`, `Gemma4UnifiedForConditionalGeneration` | `gemma4_moe`, `gemma4_unified` | bit-exact (Gemma-4-26B-A4B, EP8); molt RL e2e EP8 |
| Nemotron 3 / 3.5 | `NemotronHForCausalLM`, `NemotronH_Nano_Omni_Reasoning_V3`, `NemotronH_Omni_Reasoning_V3` | `nemotron_v3`, `nemotron_omni` | bit-exact (Nemotron-3-Nano-30B-A3B, EP8); molt RL e2e EP8 |
| Muse Glimmer | `MuseGlimmerForConditionalGeneration` | `muse_glimmer` | bit-exact (Muse-Glimmer-30B); molt RL e2e TP2 |
| Inkling | `InklingForConditionalGeneration` | `inkling` | bit-exact (seeded 4-layer build) |

`llama/` and `gpt_oss/` hold only the RoPE helpers the Qwen and GLM implementations import; there is
no Llama or GPT-OSS model class. Dense HF architectures without a native implementation still load
through the plain transformers path (FSDP2, with TP plans for Qwen2 / Qwen3).

"Bit-exact" means: same checkpoint, same input batch, forward logits identical to the last bit
between upstream `8f73178c` and this tree, on the molt `0.1.10` image. For families whose
checkpoints are hundreds of GB, the comparison used the real config cut to a few layers with seeded
random weights.

## What is kept (the molt dependency surface)

| Area | Modules | molt entry points |
|---|---|---|
| Loading | `_transformers/` (`auto_model`, `model_init`, `infrastructure`, `registry`, `te_attention`, `capabilities`, `mfu`) | `NeMoAutoModelForCausalLM` / `NeMoAutoModelForImageTextToText.from_pretrained(..., distributed_setup=, backend=BackendConfig(...), has_packed_sequence=, peft_config=, freeze_config=)`, `get_is_hf_model`, `ModelRegistry`, `AutoMFU` |
| Parallelism | `components/distributed/` | `FSDP2Config`, `MoEParallelizerConfig`, `DistributedSetup`, `MeshContext`, `ParallelismSizes`, `_create_device_meshes`, `get_flat_mesh`, `ContextParallelSharder` (round-robin and THD layouts, model-owned CP hooks, blockdiag CP for packed Qwen3.5 / 3.8) |
| MoE | `components/moe/` | `Gate` (fp32 router, aux loss), grouped experts (torch / TE / DeepEP), DeepEP + HybridEP dispatch, EP parallelizer with AC recompute replay, `RouterReplay`, `MoEAuxLossAutoScaler`, expert state-dict mixin, load-balance metrics, MXFP8 experts |
| Checkpointing | `components/checkpoint/` | `Checkpointer`, `CheckpointingConfig`, `OptimizerState`: DCP save / load (every model, any world size), consolidated HF safetensors export, PEFT adapter export |
| Training utilities | `components/training/`, `components/_peft/`, `components/quantization/`, `components/optim/` | `scale_grads_and_clip_grad_norm` (with the Triton fused grad-norm kernel), `PeftConfig` / `LinearLoRA` (+ MoE expert LoRA), FP8 (`apply_fp8_to_model`), Dion / Muon |
| Misc | `components/utils/`, `components/attention/`, `shared/` | `filter_forward_kwargs`, `canonical_parameter_fqn`, FLOPs formulas for the kept architectures, flex / FFPA attention used by Qwen3.8 and Gemma 4 CP |

Refit to vLLM uses each model's `state_dict_adapter.convert_single_tensor_to_hf`.

## What was removed

Recipes and training loops (SFT, KD, VLM), datasets, the CLI and launcher, loggers, evaluation,
speculative decoding (DSpark, EAGLE, drafters), diffusion, retrieval encoders and the
sequence / token-classification heads, tokenizer wrappers, pipeline parallelism, DDP and
Megatron-FSDP strategies, QAT and QLoRA, UCCL-EP and Mixture-of-Kittens dispatchers,
MagiAttention, Triton LoRA kernels and the fused LoRA MLP, transformers-v4 compatibility patches, the
capability-based model docs, the full-state-dict and `.bin` checkpoint load paths (every model,
including single-GPU custom models, loads through DCP), the diffusers-compatible export flag,
unreferenced training-stack helpers, and 48 model families. Tests bound to removed code were dropped.

Kept files were not refactored. Where a kept file imported a removed module, the import and the branch
that used it were deleted; nothing else was rewritten.

## Using it from molt

molt installs upstream AutoModel by default and switches to this tree with one variable
(`setup.py` swaps the `nemo-automodel` pin for the one in molt's `requirements-automodel-slim.txt`):

```bash
MOLT_AUTOMODEL=slim pip install -e ".[vllm]"                                          # local install
docker build --build-arg MOLT_AUTOMODEL=slim -f dockerfile/Dockerfile -t molt:slim .   # image
```

Working on the backend itself: check this branch out next to molt and point the recipes at it, the
sibling checkout wins over the package baked into the image:

```bash
git clone -b automodel-slim https://github.com/NVIDIA-NeMo/labs-molt.git automodel-slim
EXTRA_PYTHONPATH=$PWD/automodel-slim sbatch examples/scripts/slurm/rl_qwen3_6_35b.sh   # slurm recipes
PYTHONPATH=/path/to/automodel-slim python -m molt.cli.train_rl_ray ...                   # in the container
```

Standalone install (the extras are unchanged from upstream: `cuda` = TransformerEngine, tilelang,
mamba / causal-conv1d; `fla` = flash-linear-attention for the GDN models; `moe` = DeepEP; `fa`,
`ffpa`, `vlm`; the molt image already contains all of them):

```bash
pip install "nemo-automodel[cuda,fla,moe] @ git+https://github.com/NVIDIA-NeMo/labs-molt.git@automodel-slim"
```

The branch is an orphan (one squashed snapshot plus later commits, no upstream history); the same tree
with full upstream history is mirrored at `hijkzzz/Automodel@molt-slim` for cherry-picking. molt pins a
commit sha, not the branch name.

## Contributing

- Open PRs against `automodel-slim`. Keep them to one problem each; deletions are welcome, additions
  need a molt use.
- A maintainer puts the `cicd` label on the PR to run the GPU check on that exact head commit (the
  label is consumed as the run starts; push again, label again). The `Nemo_CICD_Test` check is what
  branch protection looks at.
- Comment `/claude review` for a Claude Code review; it runs at the effort level in `.claude/settings.json`.
- Commits carry `Signed-off-by` (DCO) and follow the upstream Apache-2.0 headers.

## CI

`.github/workflows/cicd-automodel-slim.yml` is run the same way as molt's CICD (manual dispatch with an
image and runner input, the `cicd` label on PRs, every push to the branch; failures on push go to the
CI Slack channel) and finishes in about ten minutes:

| job | where | what |
|---|---|---|
| `static` | ubuntu | `compileall`, `ruff --select F821,F401,F822`, `tools/check_closure.py` (every import and every `nemo_automodel.*` string resolves) |
| `test` · unit | molt image, CPU container alongside the smoke | the Qwen / MoE / distributed / checkpoint / loading unit tests |
| `test` · smoke | molt image, 2 GPUs | `tests/gpu_smoke/qwen_smoke.py`: seeded few-layer Qwen3 dense (FSDP2, THD packing), Qwen3-MoE (EP2, THD) and Qwen3.6-MoE (EP2), built the way molt builds its actor; the base branch is built in the same job and its logits must match the PR bit for bit; then a train step with the mesh-aware grad clip and a DCP save / reload / consolidated-HF-export round trip |

There is no Dockerfile here: the image is molt's, and the checkout under test is mounted over the copy
baked into it. Real-weight validation (bit-exact logits against upstream, molt RL e2e) is done on a
cluster and recorded in the commit messages; add `--arch` cases to the smoke when another family
needs protecting (GLM-5.3 and DeepSeek V4.1 run as seeded 2–5 layer builds in about two minutes).

## Maintenance

- **Delete, do not add.** A change goes in only if molt uses it. Do not refactor kept files; a
  removal cuts the import and the branch that used it, nothing more.
- **Gates before merge.** `python -m compileall -q nemo_automodel`, `ruff check --select F821,F401,F822`,
  and the three closure checks (every import resolves to a kept module, every imported name exists,
  every `nemo_automodel.*` string names a kept module). A GPU change must keep forward logits
  bit-exact against the previous commit for the affected families (same checkpoint, same input), and
  a change to loading, EP or CP must run one molt RL e2e (dense Qwen2 and Qwen3.6-35B EP8 / CP8).
- **Bumping molt.** Update the pin in molt's `requirements.txt` through a molt PR; molt's own e2e CI
  is the acceptance test. Tag this branch at each molt release.
- **Adding a family.** Copy `components/models/<family>/` from upstream (or the mirror), restore its
  `_transformers/registry.py` entries, run the closure checks to see which removed helpers it needs,
  and compare logits against upstream with real weights or a seeded few-layer build.
- **Known load-bearing pieces that look removable but are not:** `_transformers/capabilities.py`
  (`model.supports` is read by the EP / CP parallelizers), `components/models/deepseek_v4/kernels/`
  (loaded by name through `safe_import_from`), `components/distributed/blockdiag_cp/` (imported by
  the Qwen3.5 / 3.8 model code), the `Union` form of `DistributedStrategyConfig` (importers write
  `... | None`), and the `Mistral3ForConditionalGeneration` identifier must stay out of the tree
  because GitHub's push protection reads it as a Mistral API key.

## Provenance and license

Base: NVIDIA-NeMo/Automodel `8f73178ca51d4c1e55ccf05df5da6540a9e24f7e` (the commit molt pinned
before this branch existed). Apache-2.0, unchanged; upstream copyright notices are kept in every file.

