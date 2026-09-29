# AutoModel-Slim

**NeMo AutoModel, trimmed to what [molt](https://github.com/NVIDIA-NeMo/labs-molt) needs.**

molt runs RL (and SFT) with vLLM rollouts and an FSDP2 training actor. The actor delegates model
construction to AutoModel: loading a Hugging Face checkpoint into a native implementation, sharding it
with FSDP2 / tensor / expert / context parallelism, packing sequences for TransformerEngine attention,
and saving DCP checkpoints plus consolidated HF safetensors. Upstream
[NVIDIA-NeMo/Automodel](https://github.com/NVIDIA-NeMo/Automodel) is being discontinued, so this branch
carries that backend for molt: upstream at commit `8f73178c` with everything molt does not use removed.

| | upstream `8f73178c` | AutoModel-Slim |
|---|---|---|
| Python files / lines | 758 / 282k | 249 / ~99k |
| model families | 58 | 10 (below) + shared utilities |
| tests | 1,005 files | 194 files (kept areas only) |

The package name, import path (`nemo_automodel`) and public API are unchanged, so molt needs no code
changes, only a different `requirements.txt` pin.

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

## Install and pinning

This tree is the orphan branch `automodel-slim` of labs-molt (one squashed commit, no upstream
history) so it shares molt's runners and images. molt pins it by commit:

```
# requirements.txt
nemo-automodel @ git+https://github.com/NVIDIA-NeMo/labs-molt.git@<commit-sha>
```

```bash
pip install "nemo-automodel[cuda,fla,moe] @ git+https://github.com/NVIDIA-NeMo/labs-molt.git@automodel-slim"
```

The extras are unchanged from upstream: `cuda` (TransformerEngine, tilelang, mamba / causal-conv1d),
`fla` (flash-linear-attention for the GDN models), `moe` (DeepEP), `fa`, `ffpa`, `vlm`. The molt
image already contains all of them. The same tree with full upstream history is mirrored at
`hijkzzz/Automodel@molt-slim` for cherry-picking.

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

## CI

`.github/workflows/automodel-slim-ci.yml` runs on pushes and PRs to this branch and finishes in about
ten minutes. A CPU job runs `compileall`, `ruff` and `tools/check_closure.py` (every import and every
`nemo_automodel.*` string resolves). A job on molt's two-GPU runner, inside molt's public image,
runs the unit tests of the Qwen / MoE / distributed / checkpoint / loading areas and then
`tests/gpu_smoke/qwen_smoke.py`: seeded few-layer builds of Qwen3 dense (FSDP2, THD packing),
Qwen3-MoE (EP2, THD packing) and Qwen3.6-MoE (EP2), constructed the way molt builds its actor.
The base branch is built in the same job and its forward logits must match the PR bit for bit;
the PR tree then runs a train step with the mesh-aware grad clip and a DCP save / reload /
consolidated-HF-export round trip. Real-weight validation (bit-exact logits against upstream,
molt RL e2e) is done on a cluster and recorded in the commit messages.
