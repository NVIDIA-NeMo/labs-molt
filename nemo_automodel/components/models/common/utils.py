# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import importlib.util
import logging
import os
import warnings
from collections.abc import Collection
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.experimental import register_sharding
from transformers import PretrainedConfig
from transformers.generation import GenerationConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.shared.import_utils import safe_import_from
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from nemo_automodel.shared.utils import dtype_from_str

logger = logging.getLogger(__name__)

HAVE_TE = importlib.util.find_spec("transformer_engine") is not None
HAVE_DEEP_EP = importlib.util.find_spec("deep_ep") is not None
HAVE_UCCL_EP = importlib.util.find_spec("uccl") is not None or importlib.util.find_spec("ep") is not None

# ---------------------------------------------------------------------------
#  Global state flags for training coordination
#  Set by training utility functions, read by TE module patches and MoE modules.
# ---------------------------------------------------------------------------

IS_OPTIM_STEP = False
IS_FIRST_MICROBATCH: bool | None = None


def set_is_optim_step(value: bool) -> None:
    """Set the global IS_OPTIM_STEP flag.

    Args:
        value: Whether we are in an optimization step.
    """
    global IS_OPTIM_STEP
    IS_OPTIM_STEP = value


def get_is_optim_step() -> bool:
    """Get the global IS_OPTIM_STEP flag.

    Returns:
        Whether we are in an optimization step.
    """
    return IS_OPTIM_STEP


def set_is_first_microbatch(value: bool | None) -> None:
    """Set the global IS_FIRST_MICROBATCH flag for FP8 weight caching.

    Args:
        value: True for first microbatch (quantize+cache), False for subsequent
               (use cached), None to disable caching.
    """
    global IS_FIRST_MICROBATCH
    IS_FIRST_MICROBATCH = value


def get_is_first_microbatch() -> bool | None:
    """Get the global IS_FIRST_MICROBATCH flag.

    Returns:
        True/False/None indicating microbatch position for FP8 weight caching.
    """
    return IS_FIRST_MICROBATCH


# Hub loading options ``PreTrainedModel.from_pretrained`` threads into every file it
# reads for a checkpoint, the generation config included (see
# ``GenerativePreTrainedModel.adjust_generation_fn``).
HUB_LOADING_KWARGS = ("cache_dir", "force_download", "proxies", "local_files_only", "token", "revision", "subfolder")


def generation_config_from_model_config(config: PretrainedConfig) -> GenerationConfig:
    """Generation config a custom causal LM starts with, seeded the way transformers does it.

    ``PreTrainedModel.__init__`` derives ``generation_config`` from the model config so
    the bos/eos/pad ids are set from the start. A custom model that builds a bare
    ``GenerationConfig()`` instead has no stop token, and because the consolidated
    export writes ``model.generation_config`` out as ``generation_config.json`` (which
    beats ``config.json`` on reload), the exported model never stops generating either.
    Objects that are not ``PretrainedConfig`` instances (test doubles) get the defaults.
    """
    if not isinstance(config, PretrainedConfig):
        return GenerationConfig()
    try:
        return GenerationConfig.from_model_config(config)
    except NotImplementedError:
        return GenerationConfig()


def load_pretrained_generation_config(
    pretrained_model_name_or_path: str | os.PathLike[str],
    config: PretrainedConfig | None = None,
    config_overrides: Collection[str] | None = None,
    **loading_kwargs: Any,
) -> GenerationConfig | None:
    """The generation config a checkpoint carries, or ``None`` when it carries none.

    Mirrors what ``PreTrainedModel.from_pretrained`` does once the weights are in:
    ``generation_config.json`` wins when present, because that is where hub checkpoints
    keep extra stop tokens (Nemotron-3 lists ``[2, 11]`` there but only ``2`` in
    ``config.json``) and their sampling defaults. Without it, the generation fields of
    the raw ``config.json`` are used, as HF does for legacy checkpoints that still keep
    ``do_sample``/``temperature`` there (the in-memory config has already dropped them).

    Args:
        pretrained_model_name_or_path: Hub id or local directory of the checkpoint.
        config: The model's in-memory config. Its explicit values (an ``eos_token_id``
            passed to ``from_pretrained`` lands here) win over the ``config.json``
            fallback, which cannot know about them.
        config_overrides: Names the caller explicitly set on *config*. Needed because a
            value cannot be recognized as deliberate once it equals a default.
        **loading_kwargs: The ``HUB_LOADING_KWARGS`` the caller was loading with, so the
            generation config is read from the same place as the weights.
    """
    subfolder = loading_kwargs.pop("subfolder", None) or ""
    if subfolder and os.path.isdir(pretrained_model_name_or_path):
        # GenerationConfig.from_pretrained joins ``subfolder`` in front of an absolute
        # local path, which resolves to the parent's file whenever one exists there
        # (transformers 5.12.1), so resolve the directory here and let the child win.
        pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, subfolder)
    elif subfolder:
        loading_kwargs["subfolder"] = subfolder

    try:
        return GenerationConfig.from_pretrained(pretrained_model_name_or_path, **loading_kwargs)
    except OSError:
        logger.info(
            "No generation_config.json in %s, using the generation fields of config.json.",
            pretrained_model_name_or_path,
        )
    try:
        generation_config = GenerationConfig.from_pretrained(
            pretrained_model_name_or_path, config_file_name="config.json", _from_model_config=True, **loading_kwargs
        )
    except (OSError, TypeError):
        # transformers resolves a missing non-default config file to None and then
        # fails to open it (TypeError) instead of raising OSError.
        return None
    if config is not None:
        # The raw file only contributes the legacy sampling fields; the token ids and
        # anything else set on the in-memory config, overrides included, stay on top.
        generation_config.update(**generation_config_from_model_config(config).to_diff_dict())
        # ``to_diff_dict`` drops values that equal a GenerationConfig default, so an
        # explicit ``eos_token_id=None`` would be indistinguishable from "unset" above
        # and the file's stop token would win. Re-apply what the caller actually set.
        if config_overrides:
            generation_fields = set(GenerationConfig().to_dict())
            generation_config.update(
                **{
                    name: getattr(config, name)
                    for name in config_overrides
                    if name in generation_fields and hasattr(config, name)
                }
            )
    return generation_config


def restore_pretrained_generation_config(
    model: nn.Module,
    pretrained_model_name_or_path: str | os.PathLike[str],
    config_overrides: Collection[str] | None = None,
    **loading_kwargs: Any,
) -> None:
    """Replace a model's ``generation_config`` with the one its checkpoint carries, if any.

    Applies only to models that expose a real ``GenerationConfig`` (the ones that can
    generate); a model without one, or a checkpoint without generation settings, is
    left as it is. ``config_overrides`` names what the caller explicitly set on the
    model's config, and ``loading_kwargs`` are the ``HUB_LOADING_KWARGS`` the model was
    loaded with.
    """
    if not isinstance(getattr(model, "generation_config", None), GenerationConfig):
        return
    config = getattr(model, "config", None)
    generation_config = load_pretrained_generation_config(
        pretrained_model_name_or_path,
        config=config if isinstance(config, PretrainedConfig) else None,
        config_overrides=config_overrides,
        **loading_kwargs,
    )
    if generation_config is not None:
        model.generation_config = generation_config


def is_tensor_unallocated(tensor: torch.Tensor) -> bool:
    """Check if tensor is unallocated (meta tensor, fake tensor, etc.).

    TE kernels don't support meta tensors, fake tensors, or unallocated tensors.
    This helper detects such cases for fallback handling.

    Args:
        tensor: Tensor to check

    Returns:
        True if tensor is unallocated or cannot be accessed
    """
    try:
        return tensor.data_ptr() == 0 or tensor.numel() == 0
    except Exception:
        return True


@dataclass(kw_only=True)
class TEFp8Config:
    """Configuration for Transformer Engine FP8 quantization.

    When present (not None) in BackendConfig, FP8 is enabled.
    The ``recipe`` field accepts either a string shorthand (``"current"``, ``"block"``,
    or ``"mxfp8"``) or a pre-built TE recipe object (e.g. ``Float8CurrentScaling(fp8_dpa=True)``).

    ``"mxfp8"`` selects TE's :class:`MXFP8BlockScaling` recipe (e4m3 data + e8m0 block
    scales). Unlike torchao's MXFP8 grouped GEMM, TE's MXFP8 backward is mature (no
    e8m0-overflow NaN), which is why GPT-OSS experts (grouped + bias) use the
    ``experts="te"`` path with this recipe instead of ``experts="torch_mm_mxfp8"``.
    """

    recipe: Literal["current", "block", "mxfp8"] | Any = "current"

    def build_recipe(self):
        """Build and return the TE FP8 recipe object.

        If ``recipe`` is already a TE recipe object (e.g. ``Float8CurrentScaling(...)``),
        it is returned directly.  String values ``"current"``, ``"block"``, and
        ``"mxfp8"`` are mapped to the corresponding TE recipe class.
        """
        if not HAVE_TE:
            return None

        # Pass through pre-built recipe objects directly
        if not isinstance(self.recipe, str):
            return self.recipe

        from transformer_engine.common.recipe import Float8BlockScaling, Float8CurrentScaling

        if self.recipe == "mxfp8":
            try:
                from transformer_engine.common.recipe import MXFP8BlockScaling
            except ImportError as e:  # TE too old for MXFP8 (added ~v2.3)
                raise ImportError(
                    "te_fp8.recipe='mxfp8' requires transformer_engine.common.recipe.MXFP8BlockScaling "
                    "(TE >= ~2.3). The installed TE does not provide it; rebuild on a newer TE image."
                ) from e
            logger.warning("te_fp8.recipe='mxfp8': using TE MXFP8BlockScaling (mature MXFP8 backward).")
            return MXFP8BlockScaling()
        if self.recipe == "block":
            return Float8BlockScaling()
        return Float8CurrentScaling()

    def maybe_te_autocast(self):
        """Return te_autocast context manager for FP8."""
        if not HAVE_TE:
            return nullcontext()
        from transformer_engine.pytorch.quantization import autocast as te_autocast

        return te_autocast(enabled=True, recipe=self.build_recipe())


@dataclass(kw_only=True)
class CudaGraphConfig:
    """Configuration for scoped partial CUDA graphs.

    Attributes:
        modules: Per-layer modules to capture with Transformer Engine, following
            Megatron Core's module names. AutoModel supports whole ``attn``,
            ``moe_router``, and ``moe_preprocess`` scopes, plus the
            AutoModel-specific narrow ``te_dpa`` scope. An empty list disables
            CUDA graphs.
    """

    modules: list[Literal["attn", "te_dpa", "moe_router", "moe_preprocess"]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate the declarative CUDA-graph configuration."""
        if not isinstance(self.modules, list):
            raise TypeError("cuda_graph.modules must be a list")
        if not all(isinstance(module, str) for module in self.modules):
            raise TypeError("cuda_graph.modules entries must be strings")
        supported_modules = {"attn", "te_dpa", "moe_router", "moe_preprocess"}
        unknown_modules = set(self.modules) - supported_modules
        if unknown_modules:
            raise ValueError(
                f"Unsupported cuda_graph.modules: {sorted(unknown_modules)}; "
                f"supported modules are {sorted(supported_modules)}"
            )
        if len(self.modules) != len(set(self.modules)):
            raise ValueError("cuda_graph.modules must not contain duplicates")
        if {"attn", "te_dpa"}.issubset(self.modules):
            raise ValueError("cuda_graph.modules cannot contain both 'attn' and 'te_dpa'")
        if "moe_preprocess" in self.modules and "moe_router" not in self.modules:
            raise ValueError("'moe_preprocess' in cuda_graph.modules requires 'moe_router'")


@dataclass(kw_only=True)
class BackendConfig:
    """Backend configuration for model components.

    Attributes:
        attn: Attention backend ("te", "sdpa", "flex", "eager", "tilelang", or "cudnn").
            For DeepSeek V4, "tilelang" enables the TileLang sparse attention,
            indexer, and Sinkhorn kernels together. For GLM DSA, "tilelang" and
            "cudnn" select their respective packed sparse-attention kernels.
            For Qwen3.8-Flash-Next, "flex" selects FlexAttention sparse GQA on
            CUDA BF16; CPU execution retains the PyTorch numerical oracle.
        sparse_attn: Sparse-attention backend. "generic" preserves each model's
            existing sparse mask plus ``attn`` path; "msa" selects the optional
            SM100 MSA kernels a model provides for its sparse layers only.
        linear: Linear layer backend ("torch", "te", or "quack").
        rms_norm: RMSNorm backend ("torch", "torch_fp32", "te", or "quack").
        rope: Rotary embedding backend ("torch" or "quack"). QuACK is currently
            integrated for Llama-family rotary embeddings.
        rope_fusion: Whether to use fused RoPE (requires TE).
        experts: MoE expert GEMM backend. "torch" uses per-expert loop,
            "te" uses TE GroupedLinear, "torch_mm" uses torch._grouped_mm,
            "gmm" is a deprecated compatibility alias for "torch_mm", and "torch_mm_mxfp8" uses torch._grouped_mm
            dispatch but routes the expert grouped GEMMs through torchao's MXFP8
            scaled grouped GEMM (training-only; GB200/sm_100+ with torchao installed,
            else falls back to torch._grouped_mm at runtime).
        dispatcher: MoE token dispatcher. "torch" uses DTensor all-gather/reduce-scatter,
            "deepep" uses DeepEP, "hybridep" uses HybridEP, "uccl_ep" uses UCCL-EP.
        dispatcher_share_token_dispatcher: Whether flex token dispatchers share a communication
            manager instance across MoE layers.
        dispatcher_async_dispatch: Whether DeepEP/UCCL-EP dispatch and combine should return
            asynchronously and allocate their outputs on the communication stream.
        enable_deepep: Removed and ignored. Logs a warning if set; configure "dispatcher"
            and "experts" explicitly instead.
        fake_balanced_gate: If True, replace the learned Gate with FakeBalancedGate
            that assigns tokens to experts without learned routing weights.
        fake_gate_noise: Noise level [0, 1] for FakeBalancedGate. When > 0, uses
            biased topk selection seeded from the input content so routing varies
            dynamically across training steps (like real Gate) while remaining
            deterministic for activation checkpointing recompute (same input = same
            routing). Only used when fake_balanced_gate=True.
        enable_hf_state_dict_adapter: Whether to enable HuggingFace state dict adapter.
        enable_fsdp_optimizations: Whether to enable FSDP2 optimizations.
        gate_precision: Optional dtype override for the gate computation. Accepts
            torch.dtype or string (e.g., "torch.float32", "float32").
        compile_attn: torch.compile(fullgraph) the attention module's forward — both the
            DeepSeek-V3 MLA and standard GQA attention (e.g. Qwen3-MoE) honor it. Requires
            attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False.
        compile_situ: torch.compile the SiTU expert activation of models that opt in
            (currently Kimi K3): the weighted-SiTU forward and backward run as one fused
            whole-tensor kernel each (low-precision in/out, fp32 math inside, no chunk loop),
            the dense / shared-expert ``SituAndMul`` uses a compiled core, and the attn-res
            mixing chain is compiled as well. Compiled numerics are allclose to eager but
            not bitwise-identical.
        compile_norm: torch.compile the fp32 RMSNorm chain of models that opt in
            (currently Kimi K3), fusing cast/pow/mean/rsqrt/mul into one kernel.
            Same lazy once-per-process pattern as ``compile_situ``; numerics are
            allclose to eager but not bitwise-identical.
        shared_expert_overlap: run the shared experts of opted-in MoE models (currently Kimi K3)
            on a side CUDA stream so their GEMMs overlap the expert-parallel dispatch / combine
            communication of the routed path; numerics unchanged. Default False.
        compile_router_weight: torch.compile the fp32 router-weight multiply applied to
            expert outputs (``GroupedExperts`` / ``GroupedExpertsDeepEP``, all MoE models),
            so each pass is one fused kernel instead of the eager per-chunk cast/multiply/
            cast/slice-assign loop. Same lazy once-per-process pattern as ``compile_situ``;
            numerics are allclose to eager but not bitwise-identical.
        benchmark_static_routing: Benchmark-only. Requires ``fake_balanced_gate=True``
            with ``fake_gate_noise=0.0``, where routing metadata (tokens per expert,
            permuted token counts) is identical for every microbatch. Skips the
            per-microbatch device-to-host reads of that metadata (`.tolist()` /
            `count_nonzero` / dispatcher size checks) by caching the first
            microbatch's values, removing recurring host-sync stalls on the hot
            path. Never enable with a learned gate: cached metadata would go
            stale and silently corrupt expert dispatch.
        cuda_graph: Scoped partial CUDA-graph configuration.
    """

    attn: Literal["te", "sdpa", "flex", "eager", "tilelang", "cudnn"] = (
        "te" if HAVE_TE and torch.cuda.is_available() else "sdpa"
    )
    sparse_attn: Literal["generic", "msa"] = "generic"
    linear: Literal["torch", "te", "quack"] = "te" if HAVE_TE and torch.cuda.is_available() else "torch"
    rms_norm: Literal["torch", "torch_fp32", "te", "quack"] = "torch_fp32"
    rope: Literal["torch", "quack"] = "torch"
    rope_fusion: bool = HAVE_TE and torch.cuda.is_available()
    experts: Literal["torch", "te", "gmm", "torch_mm", "torch_mm_mxfp8"] = (
        "torch_mm" if torch.cuda.is_available() else "torch"
    )
    dispatcher: Literal["torch", "deepep", "hybridep", "uccl_ep"] = (
        "deepep"
        if HAVE_DEEP_EP and torch.cuda.is_available()
        else "uccl_ep"
        if HAVE_UCCL_EP and torch.cuda.is_available()
        else "torch"
    )
    dispatcher_num_sms: int = 20
    dispatcher_share_token_dispatcher: bool = True
    dispatcher_async_dispatch: bool = False
    enable_deepep: bool | None = None  # Removed: ignored with a warning; set dispatcher/experts explicitly
    fake_balanced_gate: bool = False
    # Approximate max/mean load ratios (64 experts, top-8, 4096 tokens):
    # 0.0→1.00x, 0.1→~1.2x, 0.3→~1.6x, 0.5→~2.0x, 1.0→~2.8x.
    fake_gate_noise: float = 0.0
    enable_hf_state_dict_adapter: bool = True
    enable_fsdp_optimizations: bool = False
    te_fp8: TEFp8Config | None = None
    gate_precision: str | torch.dtype | None = None
    # When True, torch.compile(fullgraph=True) the attention module's forward to fuse its many
    # small ops (projections, RoPE, reshapes, SDPA). Applies to both the DeepSeek-V3 MLA and
    # standard GQA attention (e.g. Qwen3-MoE). The compiled region must contain no TE
    # custom-autograd submodules (TE's fused attention/Linear/RMSNorm are black boxes that
    # fullgraph can't trace), so it requires attn="sdpa", linear="torch", rms_norm="torch",
    # rope_fusion=False. Default False.
    compile_attn: bool = False
    # When True, torch.compile the SiTU expert activation of models that opt in (currently
    # Kimi K3): fused whole-tensor weighted-SiTU forward/backward, the dense SituAndMul core
    # and the attn-res chain. Numerics are allclose to eager, not bitwise-identical. Default False.
    compile_situ: bool = False
    # When True, torch.compile the fp32 RMSNorm chain of opted-in models (currently Kimi K3),
    # same lazy once-per-process pattern as compile_situ. Numerics are allclose to eager,
    # not bitwise-identical. Default False.
    compile_norm: bool = False
    # When True, models that opt in (currently Kimi K3) run their shared experts on a side CUDA
    # stream, launched before the routed-expert path and joined after it, so the shared-expert
    # GEMMs overlap the expert-parallel dispatch / combine communication (Megatron-Core's
    # moe_shared_expert_overlap). Same math, only the execution order changes. Default False.
    shared_expert_overlap: bool = False
    # When True, torch.compile the fp32 router-weight multiply on expert outputs (shared by
    # all MoE expert modules), same lazy once-per-process pattern as compile_situ. Numerics
    # are allclose to eager, not bitwise-identical. Default False.
    compile_router_weight: bool = False
    # Benchmark-only: cache per-microbatch routing metadata (tokens per expert, permuted
    # token counts) after the first microbatch to remove recurring device-to-host syncs.
    # Valid ONLY with fake_balanced_gate=True and fake_gate_noise=0.0 (enforced in
    # __post_init__), where that metadata is constant by construction. Default False.
    benchmark_static_routing: bool = False
    cuda_graph: CudaGraphConfig = field(default_factory=CudaGraphConfig)

    def __post_init__(self) -> None:
        # benchmark_static_routing caches routing metadata across microbatches, which is
        # only sound when routing is constant by construction (forced balance, no noise).
        if self.benchmark_static_routing and not (self.fake_balanced_gate and self.fake_gate_noise == 0.0):
            raise ValueError(
                "benchmark_static_routing=True requires fake_balanced_gate=True and "
                "fake_gate_noise=0.0; with a learned or noisy gate the cached routing "
                "metadata would go stale and corrupt expert dispatch."
            )

        if self.sparse_attn not in ("generic", "msa"):
            raise ValueError(f"Unsupported sparse_attn={self.sparse_attn!r}; expected 'generic' or 'msa'.")

        # QuACK consumes position-gathered cosine/sine tables. TE's fused RoPE path
        # instead assumes contiguous [0, seq_len) positions, so combining the two
        # silently produces incorrect phases for packed, offset, or per-example
        # position IDs.
        if self.rope == "quack" and self.rope_fusion:
            logger.warning("rope='quack' is incompatible with rope_fusion=True; disabling rope_fusion.")
            self.rope_fusion = False

        # TEMPORARY: force TE fused RoPE off globally. The fused kernel computes cos/sin
        # in fp32 in-kernel while HF/vLLM rotate with bf16 tables, breaking logprob parity
        # in some models. See #3027. This is the one chokepoint every BackendConfig passes
        # through, so it also overrides an explicit rope_fusion=True from a recipe/config.
        if self.rope_fusion:
            logger.warning("rope_fusion is temporarily force-disabled globally (see #3027).")
        self.rope_fusion = False

        # Normalize te_fp8: dict -> TEFp8Config, None stays None
        if isinstance(self.te_fp8, dict):
            self.te_fp8 = TEFp8Config(**self.te_fp8)

        if isinstance(self.cuda_graph, dict):
            self.cuda_graph = CudaGraphConfig(**self.cuda_graph)
        elif not isinstance(self.cuda_graph, CudaGraphConfig):
            raise TypeError("cuda_graph must be a CudaGraphConfig or mapping")

        if isinstance(self.gate_precision, str):
            self.gate_precision = dtype_from_str(self.gate_precision, default=None)

        if self.experts == "gmm":
            warnings.warn(
                "experts='gmm' is deprecated; use experts='torch_mm' instead.",
                FutureWarning,
                stacklevel=2,
            )

        # enable_deepep was removed. It is no longer honored; warn (once, on rank 0) if a stale
        # config still sets it so the user migrates to explicit dispatcher/experts. The field is
        # retained only so loading an old config does not crash this kw_only dataclass.
        if self.enable_deepep is not None:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.warning(
                    "enable_deepep is no longer supported and is ignored. "
                    "Set 'dispatcher' (deepep/hybridep/torch) and 'experts' explicitly instead. "
                    "Previously enable_deepep=True was equivalent to experts=gmm + dispatcher=deepep, "
                    "and enable_deepep=False to dispatcher=torch."
                )
            self.enable_deepep = None

        # TE grouped experts require a flex dispatcher. Keep the existing standard-
        # dispatcher fallback for the deprecated gmm alias as well.
        if self.experts in ("te", "gmm") and self.dispatcher not in ("deepep", "hybridep", "uccl_ep"):
            if (
                torch.distributed.is_initialized() and torch.distributed.get_rank() == 0
            ) or not torch.distributed.is_initialized():
                logger.info(
                    f"experts='{self.experts}' requires dispatcher='deepep' or 'uccl_ep', "
                    f"but got dispatcher='{self.dispatcher}'. "
                    "Setting dispatcher to torch and experts to torch_mm."
                )
            self.dispatcher = "torch"
            self.experts = "torch_mm"

        graph_modules = self.cuda_graph.modules
        attention_graph_modules = {"attn", "te_dpa"}.intersection(graph_modules)
        if attention_graph_modules and self.attn != "te":
            raise ValueError(f"{sorted(attention_graph_modules)} in cuda_graph.modules requires attn='te'")
        if "attn" in graph_modules and self.linear != "torch":
            raise ValueError("'attn' in cuda_graph.modules currently requires linear='torch'")
        if "attn" in graph_modules and self.rms_norm != "torch":
            raise ValueError("'attn' in cuda_graph.modules currently requires rms_norm='torch'")
        if "attn" in graph_modules and self.rope_fusion:
            raise ValueError("'attn' in cuda_graph.modules currently requires rope_fusion=False")
        if attention_graph_modules and self.te_fp8 is not None:
            recipe_fp8_dpa = getattr(self.te_fp8.recipe, "fp8_dpa", False)
            if recipe_fp8_dpa:
                raise ValueError(
                    f"{sorted(attention_graph_modules)} in cuda_graph.modules requires BF16 dot-product attention "
                    "(fp8_dpa=False)"
                )
        if "moe_router" in graph_modules and self.fake_balanced_gate:
            raise ValueError("'moe_router' in cuda_graph.modules requires the learned Gate (fake_balanced_gate=False)")
        if "moe_preprocess" in graph_modules and self.dispatcher != "hybridep":
            raise ValueError("'moe_preprocess' in cuda_graph.modules requires dispatcher='hybridep'")
        # FP8 requires at least one TE backend (applies to all TE modules: Linear, GroupedLinear, RMSNorm)
        if self.te_fp8 is not None and self.linear != "te" and self.experts != "te":
            raise ValueError(
                "te_fp8 requires at least one TE backend "
                f"(linear='te' or experts='te'), but got linear='{self.linear}', experts='{self.experts}'"
            )


# Keep the forward opaque so grad/no_grad compilation uses the same computation.
@torch.library.custom_op("nemo_automodel::float32_rms_norm", mutates_args=())
def _float32_rms_norm_impl(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Compute RMSNorm in fp32 with an opaque, device-independent forward.

    Args:
        x: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        weight: Tensor of shape [hidden] on the same device as x.
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of shape [..., hidden] in x's dtype, without aliasing either input.
    """
    return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps).to(x.dtype)


@_float32_rms_norm_impl.register_fake
def _float32_rms_norm_meta(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Describe the output metadata for inputs x [..., hidden] and weight [hidden].

    Args:
        x: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        weight: Tensor of shape [hidden] on the same device as x.
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of shape [..., hidden] with x's dtype and device.
    """
    # Native RMSNorm can choose a different output layout on CPU and CUDA.
    return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps).to(x.dtype)


@register_sharding(torch.ops.nemo_automodel.float32_rms_norm.default)
def _float32_rms_norm_sharding(x, weight, eps):
    """Keep the normalized axis complete while allowing leading-axis sharding.

    Args:
        x: Tensor metadata of global shape [..., hidden].
        weight: Tensor metadata of global shape [hidden].
        eps: Epsilon added to the mean square.

    Returns:
        Supported output/input placements on each mesh axis. Leading dimensions
        may be sharded; weight and the hidden dimension must be replicated.
    """
    strategies = [([Replicate()], [Replicate(), Replicate(), None])]
    strategies.extend(([Shard(dim)], [Shard(dim), Replicate(), None]) for dim in range(x.ndim - 1))
    return strategies


def _float32_rms_norm_setup_context(ctx, inputs, output):
    """Save the inputs required for the RMSNorm gradient.

    Args:
        ctx: Autograd context that owns the saved tensors.
        inputs: Tuple containing x [..., hidden], weight [hidden], and eps.
        output: Tensor of shape [..., hidden] in x's dtype.
    """
    x, weight, eps = inputs
    ctx.save_for_backward(x, weight)
    ctx.eps = eps


def _float32_rms_norm_backward(ctx, grad_output):
    """Differentiate the fp32 RMSNorm computation.

    Args:
        ctx: Autograd context containing x [..., hidden] and weight [hidden].
        grad_output: Output gradient of shape [..., hidden].

    Returns:
        Gradients for x [..., hidden] and weight [hidden] in their input dtypes,
        or None for frozen inputs, followed by None for the scalar eps.
    """
    x, weight = ctx.saved_tensors
    x_f32 = x.float()
    w_f32 = weight.float()
    g_f32 = grad_output.float()

    r = torch.rsqrt(x_f32.pow(2).mean(-1, keepdim=True) + ctx.eps)
    xnorm = x_f32 * r

    grad_w = (g_f32 * xnorm).sum_to_size(weight.shape)

    gw = g_f32 * w_f32
    grad_x = r * (gw - xnorm * (gw * xnorm).mean(-1, keepdim=True))

    return (
        grad_x.to(x.dtype) if x.requires_grad else None,
        grad_w.to(weight.dtype) if weight.requires_grad else None,
        None,  # eps is not differentiable
    )


_float32_rms_norm_impl.register_autograd(
    _float32_rms_norm_backward,
    setup_context=_float32_rms_norm_setup_context,
)


@torch.compile(dynamic=True)
def _float32_rms_norm_fwd(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Compiled fp32 RMSNorm forward — standalone function to minimize dynamo guards.

    The opaque forward keeps the same computation in grad and no_grad contexts.

    Args:
        x: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        weight: Tensor of shape [hidden] on the same device as x.
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of shape [..., hidden] in x's dtype.
    """
    return torch.ops.nemo_automodel.float32_rms_norm(x, weight, eps)


class Float32RMSNorm(nn.Module):
    """RMSNorm with explicit fp32 computation for training stability.

    Weights stay in the model's dtype (e.g. bf16) for FSDP2 compatibility.
    Inputs are upcast to fp32, norm is computed in fp32, and the output
    is cast back to the original input dtype.
    """

    def __init__(self, dim, eps=1e-5, device=None, dtype=torch.bfloat16):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=dtype))

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x):
        return _float32_rms_norm_fwd(x, self.weight, self.eps)


def initialize_rms_norm_module(
    rms_norm_impl: str,
    dim: int,
    eps: float = 1e-5,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Initialize RMSNorm module with the specified backend.

    For TE backend, creates TE module directly on specified device.
    Call reset_parameters() to materialize weights if created on meta device.

    Args:
        rms_norm_impl: Backend implementation ("te", "torch", "torch_fp32", or "quack")
            - "te": Transformer Engine fused RMSNorm kernel
            - "torch": PyTorch native nn.RMSNorm (computes in input dtype)
            - "torch_fp32": torch.compiled fp32 RMSNorm for training stability
            - "quack": QuACK CuTe DSL RMSNorm kernel
        dim: Normalized dimension
        eps: Epsilon for numerical stability
        device: Device to create module on (None uses PyTorch default, typically CPU)
        dtype: Parameter dtype

    Returns:
        RMSNorm module
    """
    if rms_norm_impl == "te":
        from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TransformerEngineRMSNorm

        _patch_te_modules()
        return TransformerEngineRMSNorm(normalized_shape=dim, eps=eps, device=device, params_dtype=dtype)
    elif rms_norm_impl == "torch":
        return nn.RMSNorm(dim, eps=eps, device=device, dtype=dtype)
    elif rms_norm_impl == "torch_fp32":
        return Float32RMSNorm(dim, eps=eps, device=device, dtype=dtype)
    elif rms_norm_impl == "quack":
        available, quack_rms_norm = safe_import_from(
            "quack.rmsnorm",
            "QuackRMSNorm",
            msg="rms_norm='quack' requires the 'quack-kernels' package. Install nemo-automodel[cuda].",
        )
        if not available:
            raise ImportError("rms_norm='quack' requires the 'quack-kernels' package. Install nemo-automodel[cuda].")
        return quack_rms_norm(dim, eps=eps, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported RMSNorm implementation: {rms_norm_impl}")


def initialize_linear_module(
    linear_impl: str,
    in_features: int,
    out_features: int,
    bias: bool = False,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Initialize Linear module with the specified backend.

    For TE backend, creates TE module directly on specified device.
    Call reset_parameters() to materialize weights if created on meta device.

    Args:
        linear_impl: Backend implementation ("te", "torch", or "quack")
        in_features: Input features
        out_features: Output features
        bias: Whether to use bias
        device: Device to create module on (None uses PyTorch default, typically CPU)
        dtype: Parameter dtype

    Returns:
        Linear module
    """
    if linear_impl == "torch":
        return nn.Linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    elif linear_impl == "te":
        from transformer_engine.pytorch.module.linear import Linear as TransformerEngineLinear

        _patch_te_modules()
        # Create TE module directly on meta device (same as GroupedExpertsTE)
        return TransformerEngineLinear(
            in_features=in_features, out_features=out_features, bias=bias, device=device, params_dtype=dtype
        )
    elif linear_impl == "quack":
        available, quack_linear = safe_import_from(
            "quack.linear",
            "Linear",
            msg="linear='quack' requires the 'quack-kernels' package. Install nemo-automodel[cuda].",
        )
        if not available:
            raise ImportError("linear='quack' requires the 'quack-kernels' package. Install nemo-automodel[cuda].")
        return quack_linear(in_features, out_features, bias=bias, device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported Linear implementation: {linear_impl}")


def _make_lazy_te_patcher():
    """Return a callable that patches TE modules exactly once.

    Uses a closure instead of module-level global state to track whether the
    patch has already been applied.  The actual ``transformer_engine`` import
    is deferred until the first call so that importing this module never
    triggers heavy native-library loads (flash-attn, CUDA kernels, etc.).

    Two patches are applied:
    1. Unallocated tensor handling: TE kernels don't support meta/fake tensors,
       so we short-circuit with empty tensors for PP shape inference.
    2. is_first_microbatch injection: Reads the global IS_FIRST_MICROBATCH flag and
       passes it to TE Linear/GroupedLinear for FP8 weight caching during
       gradient accumulation (quantize on first microbatch, reuse cached on rest).
    """
    patched = False

    def _patch():
        nonlocal patched
        if patched:
            return
        patched = True

        from transformer_engine.pytorch.module.grouped_linear import GroupedLinear as TEGroupedLinear
        from transformer_engine.pytorch.module.linear import Linear as TELinear
        from transformer_engine.pytorch.module.rmsnorm import RMSNorm as TERMSNorm

        _original_rmsnorm_forward = TERMSNorm.forward
        _original_linear_forward = TELinear.forward
        _original_grouped_linear_forward = TEGroupedLinear.forward

        def _patched_rmsnorm_forward(self, x):
            # Skip the unallocated-tensor short-circuit during torch.compile tracing:
            # fake tensors used by inductor have data_ptr()==0 but are NOT unallocated --
            # returning empty_like here produces a leaf with no grad_fn, breaking AOT autograd.
            if is_tensor_unallocated(x) and not torch.compiler.is_compiling():
                return torch.empty_like(x)
            return _original_rmsnorm_forward(self, x)

        def _patched_linear_forward(self, x, is_first_microbatch=None, **kwargs):
            if is_tensor_unallocated(x):
                out_shape = x.shape[:-1] + (self.weight.shape[0],)
                return torch.empty(out_shape, dtype=x.dtype, device=x.device)
            if is_first_microbatch is None:
                is_first_microbatch = get_is_first_microbatch()
            return _original_linear_forward(self, x, is_first_microbatch=is_first_microbatch, **kwargs)

        def _patched_grouped_linear_forward(self, inp, m_splits, is_first_microbatch=None):
            if is_first_microbatch is None:
                is_first_microbatch = get_is_first_microbatch()
            return _original_grouped_linear_forward(self, inp, m_splits, is_first_microbatch=is_first_microbatch)

        TERMSNorm.forward = _patched_rmsnorm_forward
        TELinear.forward = _patched_linear_forward
        TEGroupedLinear.forward = _patched_grouped_linear_forward

    return _patch


_patch_te_modules = _make_lazy_te_patcher()


def get_rope_config(config) -> tuple[float, dict, float]:
    """Extract rope configuration from ``config.rope_parameters``.

    Args:
        config: A HuggingFace model config object.

    Returns:
        Tuple of (rope_theta, rope_parameters, partial_rotary_factor).
    """
    rope_parameters = config.rope_parameters
    rope_theta = rope_parameters["rope_theta"]
    partial_rotary_factor = rope_parameters.get("partial_rotary_factor", 1.0)
    return rope_theta, rope_parameters, partial_rotary_factor


def cast_model_to_dtype(
    model: nn.Module, dtype: torch.dtype = torch.bfloat16, skip_modules: tuple[str, ...] = ()
) -> None:
    """Cast model parameters to the target dtype, keeping fp32 modules in full precision.

    Respects ``_keep_in_fp32_modules`` / ``_keep_in_fp32_modules_strict`` on
    the model (the same attributes HuggingFace transformers uses).

    Uses ``nn.Module.to()`` which is safe for both plain tensors and DTensors
    (FSDP2 sharded parameters).  When the model is already FSDP2-sharded
    (parameters are DTensors), strict fp32 modules are restored to fp32 because
    they are expected to be isolated as uniform fp32 FSDP units. Non-strict fp32
    hints only restore matching buffers, since their parameters may share an
    FSDP unit with lower-precision parameters.

    Args:
        model: The model whose parameters should be cast.
        dtype: Target dtype (e.g. ``torch.bfloat16``).
        skip_modules: Names of immediate submodules to leave entirely untouched
            (kept at their current dtype). Unlike the ``_keep_in_fp32_modules``
            restore path, these are *detached* during the cast so ``model.to()``
            never visits them — the only reliable way to preserve an fp32
            parameter once it is FSDP2-sharded (post-shard ``.data`` reassignment
            does not stick). The caller must guarantee each skipped submodule is
            its own dtype-uniform FSDP group (e.g. Qwen3.5's ``_fp32_params``
            holder, sharded separately in fp32), so leaving it fp32 cannot break
            FSDP's uniform-dtype rule.
    """
    fp32_keywords = _get_fp32_module_keywords(model)
    strict_fp32_keywords = _get_strict_fp32_module_keywords(model)
    has_dtensor_params = _has_dtensor_params(model)

    if has_dtensor_params:
        fp32_snapshots = _snapshot_fp32_tensors(
            model,
            parameter_keywords=strict_fp32_keywords,
            buffer_keywords=fp32_keywords,
        )
    else:
        fp32_snapshots = _snapshot_fp32_tensors(
            model,
            parameter_keywords=fp32_keywords,
            buffer_keywords=fp32_keywords,
        )

    # Detach skip_modules so ``model.to(dtype)`` does not descend into them. This
    # preserves their exact dtype (e.g. fp32 master weights) through the cast.
    detached: list[tuple[nn.Module, str, nn.Module]] = []
    if skip_modules:
        for _, parent in model.named_modules():
            for child_name, child in list(parent._modules.items()):
                if child is not None and child_name in skip_modules:
                    detached.append((parent, child_name, child))
                    parent._modules[child_name] = None

    try:
        model.to(dtype)
    finally:
        for parent, child_name, child in detached:
            parent._modules[child_name] = child

    if fp32_keywords:
        if has_dtensor_params:
            if strict_fp32_keywords:
                _restore_fp32_tensor_snapshots(
                    model,
                    parameter_snapshots=fp32_snapshots[0],
                    buffer_snapshots={},
                )

            buffer_only_keywords = [kw for kw in fp32_keywords if kw not in strict_fp32_keywords]
            if buffer_only_keywords:
                logger.warning(
                    "Model parameters are DTensors (FSDP2) — skipping fp32 parameter "
                    "restoration for non-strict keywords=%s. Only buffers will be restored to fp32. "
                    "FSDP2 requires uniform dtype within each parameter group.",
                    buffer_only_keywords,
                )
            _restore_fp32_tensor_snapshots(
                model,
                parameter_snapshots={},
                buffer_snapshots=fp32_snapshots[1],
            )
        else:
            _restore_fp32_tensor_snapshots(
                model,
                parameter_snapshots=fp32_snapshots[0],
                buffer_snapshots=fp32_snapshots[1],
            )


@contextmanager
def yield_fp32_model(model: nn.Module, restore_dtype: torch.dtype | None = None):
    """Run a block with the model temporarily in fp32, then cast it to ``restore_dtype``.

    On entry the whole model is cast to fp32; on exit it is cast to ``restore_dtype``
    (which defaults to the model's pre-context floating-point dtype, so by default the
    original dtype is restored). The exit cast is a no-op when the target is already fp32.

    The motivating use is from-scratch weight initialization. Sampling a random init directly
    in a reduced-precision dtype (e.g. bf16) distorts the init's variance/mean schedule: bf16's
    8-bit mantissa quantizes the small init magnitudes and biases the truncation/scaling
    arithmetic used by ``normal_`` / ``trunc_normal_``. In a deep residual stack this compounds
    and produces genuinely huge gradients on the first optimization steps of from-scratch
    pretraining (flat / diverging loss). Sampling in fp32 and then casting back avoids this while
    keeping reduced-precision storage: the round-to-bf16 of a correct fp32 sample is an unbiased
    per-element perturbation that preserves the init statistics. Wrap the body of a model's
    ``initialize_weights`` to keep that round-trip in one place.

    Works whether or not the model is already FSDP2-sharded: both casts are *uniform* whole-model
    casts, so FSDP2's invariant that every parameter in a group shares one dtype is preserved. In
    the AutoModel pipeline ``initialize_weights`` actually runs after sharding (via
    ``checkpointer.initialize_model_weights``), i.e. on DTensor params, which is supported.

    ``_keep_in_fp32_modules`` / ``_keep_in_fp32_modules_strict`` handling is delegated to
    ``cast_model_to_dtype``: on an unsharded model those modules' params and buffers are restored
    to fp32 on exit; on a sharded model, strict fp32 modules are restored while non-strict modules
    only have their buffers restored.

    Args:
        model: The model to run in fp32 within the context.
        restore_dtype: The dtype to cast the model to on exit. Defaults to the model's current
            floating-point dtype (captured before the fp32 cast), i.e. the original dtype.

    Yields:
        The same ``model``, now in fp32.

    Example:
        >>> with yield_fp32_model(self, dtype):
        ...     self.model.init_weights(buffer_device=buffer_device)
    """
    if restore_dtype is None:
        restore_dtype = next((p.dtype for p in model.parameters() if p.is_floating_point()), torch.float32)
    cast_model_to_dtype(model, torch.float32)
    try:
        yield model
    finally:
        cast_model_to_dtype(model, restore_dtype)


def _get_strict_fp32_module_keywords(model: nn.Module) -> list[str]:
    val = getattr(model, "_keep_in_fp32_modules_strict", None)
    if not isinstance(val, (list, set, tuple)):
        return []
    return list(dict.fromkeys(val))


def _get_fp32_module_keywords(model: nn.Module) -> list[str]:
    """Collect module name patterns that must remain in fp32.

    Reads ``_keep_in_fp32_modules`` and ``_keep_in_fp32_modules_strict``
    from the model (the same attributes HuggingFace transformers uses).

    Args:
        model: The model to inspect.

    Returns:
        De-duplicated list of module-name keywords to keep in fp32.
    """
    keywords: list[str] = []
    for attr in ("_keep_in_fp32_modules_strict", "_keep_in_fp32_modules"):
        val = getattr(model, attr, None)
        # HuggingFace's PreTrainedModel.__init__ normalizes a class-level
        # list[str] into an instance-level set[str], so accept both (and tuple).
        if isinstance(val, (list, set, tuple)):
            keywords.extend(val)

    # de-duplicate while preserving order
    return list(dict.fromkeys(keywords))


def _has_dtensor_params(model: nn.Module) -> bool:
    """Check if any model parameter is a DTensor (FSDP2 sharded)."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    return any(isinstance(p, DTensor) for p in model.parameters())


def _snapshot_fp32_tensors(
    model: nn.Module,
    *,
    parameter_keywords: list[str],
    buffer_keywords: list[str],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Clone fp32-preserved tensors before a broad dtype cast.

    Casting ``fp32 -> bf16 -> fp32`` restores the dtype but not the original
    values. Snapshot the matching tensors first so strict fp32 state such as
    router correction bias or recurrent-decay parameters is restored exactly.
    """
    parameter_snapshots = {
        name: param.detach().to(torch.float32).clone()
        for name, param in model.named_parameters()
        if param.is_floating_point() and any(keyword in name for keyword in parameter_keywords)
    }
    buffer_snapshots = {
        name: buf.detach().to(torch.float32).clone()
        for name, buf in model.named_buffers(remove_duplicate=False)
        if buf.is_floating_point() and any(keyword in name for keyword in buffer_keywords)
    }
    return parameter_snapshots, buffer_snapshots


def _restore_fp32_tensor_snapshots(
    model: nn.Module,
    *,
    parameter_snapshots: dict[str, torch.Tensor],
    buffer_snapshots: dict[str, torch.Tensor],
) -> None:
    """Restore fp32-preserved tensors from pre-cast snapshots."""
    named_parameters = dict(model.named_parameters())
    for name, snapshot in parameter_snapshots.items():
        param = named_parameters.get(name)
        if param is None:
            continue
        param.data = snapshot.to(dtype=torch.float32)

    for name, snapshot in buffer_snapshots.items():
        # ActivationWrapper forwards __getattr__ / __setattr__ to the wrapped
        # module. Try the literal FQN first for buffers registered on the wrapped
        # leaf, then strip the wrapper alias as a fallback for forwarded names.
        candidate_names = [name]
        stripped_name = canonical_parameter_fqn(name)
        if stripped_name != name:
            candidate_names.append(stripped_name)

        for candidate_name in candidate_names:
            module_name, _, buffer_name = candidate_name.rpartition(".")
            try:
                module = model.get_submodule(module_name) if module_name else model
            except AttributeError:
                continue
            if buffer_name not in module._buffers:
                continue
            module._buffers[buffer_name] = snapshot.to(dtype=torch.float32)
            break


def _restore_fp32_modules(model: nn.Module, fp32_keywords: list[str]) -> None:
    """Cast modules or individual tensors matching *fp32_keywords* back to float32.

    Only safe for unsharded models (plain tensors). FSDP2 requires uniform
    dtype within each parameter group, so this must not be called on DTensor-sharded
    models. Keywords may name modules (for example ``norm``) or individual
    parameters (for example ``attn_hc.fn``), matching HuggingFace's strict fp32
    module declarations.

    Args:
        model: The model (already cast to the target dtype).
        fp32_keywords: Substrings matched against dot-separated module names.
    """
    for name, module in model.named_modules():
        if any(kw in name for kw in fp32_keywords):
            module.to(torch.float32)
    for name, param in model.named_parameters():
        if any(kw in name for kw in fp32_keywords):
            param.data = param.data.to(torch.float32)
    for name, buf in model.named_buffers():
        if any(kw in name for kw in fp32_keywords):
            module_name, _, buffer_name = name.rpartition(".")
            module = model.get_submodule(module_name) if module_name else model
            module._buffers[buffer_name] = buf.to(torch.float32)


def _restore_fp32_buffers(model: nn.Module, fp32_keywords: list[str]) -> None:
    """Cast only matching buffers (not parameters) back to float32.

    Safe for FSDP2-sharded models because buffers are plain tensors, not
    DTensors managed by FSDP2.

    Args:
        model: The model (already cast to the target dtype).
        fp32_keywords: Substrings matched against dot-separated module names.
    """
    for name, module in model.named_modules():
        if any(kw in name for kw in fp32_keywords):
            for buf_name, buf in module.named_buffers(recurse=False):
                module._buffers[buf_name] = buf.to(torch.float32)
    for name, buf in model.named_buffers():
        if any(kw in name for kw in fp32_keywords):
            module_name, _, buffer_name = name.rpartition(".")
            module = model.get_submodule(module_name) if module_name else model
            module._buffers[buffer_name] = buf.to(torch.float32)


def compute_lm_head_logits(
    lm_head: nn.Module | None,
    hidden_states: torch.Tensor,
    logits_to_keep: int | torch.Tensor = 0,
    is_thd: bool = False,
    fp32_lm_head: bool = False,
    output_hidden_states: bool = False,
) -> CausalLMOutputWithPast:
    """Project hidden states through ``lm_head`` and wrap the result.

    Centralizes the lm_head projection and output packaging shared by every
    custom ``*ForCausalLM`` / ``*ForConditionalGeneration`` ``forward()``. The
    returned ``CausalLMOutputWithPast`` carries the projected ``logits`` and,
    when requested, the final ``hidden_states``; callers that also need ``loss``,
    ``past_key_values``, etc. read ``.logits`` and build their own output.

    - ``lm_head is None`` (e.g. a non-final pipeline-parallel stage that does not
      own the head): ``hidden_states`` is passed through as ``logits`` so the
      next stage receives it.
    - ``logits_to_keep == 0`` (training default): every position is projected.
      The full range is deliberately *not* sliced, because ``slice(0, None)`` on
      a DTensor is unsupported (it raises on the ``aten.alias`` op under tensor
      parallel with sequence parallelism).
    - ``logits_to_keep`` as a positive int or a tensor of indices: only the
      requested positions are projected. Both 2D ``[T, H]`` (THD/packed) and 3D
      ``[B, S, H]`` (BSHD) hidden states are handled.
    - ``is_thd``: THD/packed inputs yield 2D ``[T, V]`` logits; the leading batch
      dim is restored (``unsqueeze(0)`` -> ``[1, T, V]``) so downstream code sees
      a uniform ``[B, S, V]`` layout. The same restoration is applied to the
      ``hidden_states`` field. Only applied while the tensor is still 2D, so an
      ``inputs_embeds`` path that already produced ``[1, T, *]`` is left
      untouched.
    - ``fp32_lm_head``: run the projection in fp32 and cast the logits back to
      the input dtype. Used by models whose ``lm_head.weight`` has been promoted
      to fp32 (e.g. via the MoE ``lm_head_precision`` setting). The matmul goes
      through ``lm_head`` (``nn.Linear``, DTensor-aware under FSDP2) rather than
      ``F.linear`` so DTensor redistribution is preserved.
    - Otherwise, cast the sliced input to the head's effective compute dtype.
      For a standalone FSDP2 head this comes from its mixed-precision policy,
      since its sharded weight may still expose the master-copy dtype before the
      forward pre-hook runs. Other heads use their materialized weight dtype.
    - ``output_hidden_states``: when set, the (full-sequence, THD-restored)
      ``hidden_states`` are attached to the output so the fused cross-entropy
      path can recompute logits over every position; otherwise the field is
      ``None``.

    Args:
        lm_head: The language-model head module, or ``None`` on a pipeline stage
            that does not own it.
        hidden_states: Final hidden states, shaped ``[T, H]`` or ``[B, S, H]``.
        logits_to_keep: ``0`` to project every position; a positive int to keep
            the last ``N`` positions; or a tensor of position indices.
        is_thd: Whether the inputs were THD/packed; if so, a 2D logits (and
            hidden-states) result is unsqueezed back to a leading batch dim of 1.
        fp32_lm_head: Project in fp32 and cast the result back to the input
            dtype. Ignored when ``lm_head`` is ``None``.
        output_hidden_states: Attach the final hidden states to the output.

    Returns:
        A ``CausalLMOutputWithPast`` whose ``logits`` are the projected logits
        (or ``hidden_states`` unchanged when ``lm_head`` is ``None``) and whose
        ``hidden_states`` are the final hidden states when ``output_hidden_states``
        is set, else ``None``.
    """
    if lm_head is None:
        logits = hidden_states
    else:
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            sliced = hidden_states
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            if hidden_states.dim() == 2:
                sliced = hidden_states[slice_indices, :]
            else:
                sliced = hidden_states[:, slice_indices, :]
        if fp32_lm_head:
            logits = lm_head(sliced.float()).to(hidden_states.dtype)
        else:
            compute_dtype = None
            if isinstance(lm_head, FSDPModule):
                compute_dtype = lm_head._get_fsdp_state()._mp_policy.param_dtype
            lm_head_weight = getattr(lm_head, "weight", None)
            if compute_dtype is None and isinstance(lm_head_weight, torch.Tensor):
                compute_dtype = lm_head_weight.dtype
            projection_input = sliced.to(compute_dtype) if compute_dtype is not None else sliced
            logits = lm_head(projection_input)
    if is_thd and logits.dim() == 2:
        logits = logits.unsqueeze(0)

    hidden_out = None
    if output_hidden_states:
        hidden_out = hidden_states.unsqueeze(0) if is_thd and hidden_states.dim() == 2 else hidden_states

    return CausalLMOutputWithPast(logits=logits, hidden_states=hidden_out)


def cast_frozen_modules_to_compute_dtype(model: nn.Module, compute_dtype: torch.dtype | None) -> None:
    """Cast frozen floating-point tensors to ``compute_dtype``.

    When parameters are stored in fp32 (the fp32-master-weights pattern) while compute runs
    in bf16, a fully frozen submodule -- such as a frozen vision tower -- can still produce
    fp32 values that flow into bf16 trainable modules and raise a dtype mismatch in the next
    matmul. This walks each tensor independently so frozen base weights are still cast when
    a module also contains trainable adapter weights. In that mixed-module case, its trainable
    plain-tensor descendants are cast as well so unsharded execution does not introduce a new
    mismatch in the adapter path. Parameters and buffers are handled differently:

    * **Parameters** are cast only when they are plain (unsharded) tensors. Sharded (DTensor)
      params are left as-is: FSDP all-gathers them to the compute dtype during forward, and
      changing a sharded param's dtype in place would desync FSDP's flat-parameter and
      ``orig_dtype`` bookkeeping.
    * **Buffers** are always cast. Buffers are never sharded, so they stay in their stored
      dtype regardless of the wrapper; an fp32 buffer (for example a standardization
      constant) used in a forward op promotes the surrounding bf16 activations to fp32.

    Tensors whose qualified name matches ``_keep_in_fp32_modules`` or
    ``_keep_in_fp32_modules_strict`` are left in fp32. The function is a no-op when
    ``compute_dtype`` is None and for tensors already in ``compute_dtype``. Frozen parameters
    are never updated during training. Trainable plain tensors are cast only in mixed subtrees,
    where unsharded execution must match the requested compute dtype.

    Args:
        model: The model, already materialized, checkpoint-loaded, and sharded.
        compute_dtype: The compute dtype (``mp_policy.param_dtype``); None disables the cast.
    """
    if compute_dtype is None:
        return

    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = ()

    fp32_keywords = _get_fp32_module_keywords(model)

    def _is_fp32_pinned(name: str) -> bool:
        return any(kw in name for kw in fp32_keywords)

    # A LoRA-style module directly owns frozen base params and registers trainable adapter
    # descendants. On a one-rank run FSDP is skipped, so cast the whole mixed subtree to keep
    # both matmul paths uniform. With FSDP, the trainable params are DTensors and remain managed
    # by its mixed-precision policy.
    named_params = list(model.named_parameters())
    frozen_param_owners: set[str] = set()
    parameter_subtrees: set[str] = set()
    trainable_subtrees: set[str] = set()
    for name, param in named_params:
        owner_name, _, _ = name.rpartition(".")
        subtree = owner_name
        while True:
            parameter_subtrees.add(subtree)
            if not subtree:
                break
            subtree, _, _ = subtree.rpartition(".")
        if param.requires_grad:
            subtree = owner_name
            while True:
                trainable_subtrees.add(subtree)
                if not subtree:
                    break
                subtree, _, _ = subtree.rpartition(".")
        elif param.is_floating_point() and not _is_fp32_pinned(name) and not (DTensor and isinstance(param, DTensor)):
            frozen_param_owners.add(owner_name)

    mixed_subtrees = frozen_param_owners & trainable_subtrees

    def _is_in_subtrees(name: str, subtrees: set[str]) -> bool:
        subtree = name
        while True:
            if subtree in subtrees:
                return True
            if not subtree:
                return False
            subtree, _, _ = subtree.rpartition(".")

    # Parameters: cast frozen plain floats and plain floats in mixed adapter subtrees; leave
    # all other trainable params and every sharded (DTensor) param to FSDP mixed precision.
    for name, param in named_params:
        if (param.requires_grad and not _is_in_subtrees(name, mixed_subtrees)) or _is_fp32_pinned(name):
            continue
        if DTensor and isinstance(param, DTensor):
            continue
        if param.is_floating_point() and param.dtype != compute_dtype:
            param.data = param.data.to(compute_dtype)

    # Preserve the existing buffer scope: fully frozen parameter subtrees plus the newly
    # supported mixed adapter subtrees. Buffers never participate in FSDP parameter sharding.
    fully_frozen_subtrees: set[str] = set()
    for module_name, _ in model.named_modules():
        if module_name not in parameter_subtrees or module_name in trainable_subtrees:
            continue
        if _is_in_subtrees(module_name, fully_frozen_subtrees):
            continue
        fully_frozen_subtrees.add(module_name)
    buffer_subtrees = fully_frozen_subtrees | mixed_subtrees

    for name, buf in model.named_buffers():
        if not _is_in_subtrees(name, buffer_subtrees) or _is_fp32_pinned(name):
            continue
        if buf.is_floating_point() and buf.dtype != compute_dtype:
            owner_name, _, leaf = name.rpartition(".")
            owner = model.get_submodule(owner_name) if owner_name else model
            owner._buffers[leaf] = buf.to(compute_dtype)


__all__ = [
    "BackendConfig",
    "Float32RMSNorm",
    "TEFp8Config",
    "cast_frozen_modules_to_compute_dtype",
    "cast_model_to_dtype",
    "compute_lm_head_logits",
    "get_is_first_microbatch",
    "get_is_optim_step",
    "get_rope_config",
    "initialize_linear_module",
    "initialize_rms_norm_module",
    "set_is_first_microbatch",
    "set_is_optim_step",
]
