# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Loading policy for the RL model wrappers.

Everything about turning a checkpoint path into a distributed AutoModel module
lives here: registry probing (native vs Hugging Face fallback), MoE detection,
kernel-backend selection, fail-fast validation, and post-load configuration.
``BaseModel`` in base.py owns only the runtime forward contract.
"""

import os
from importlib.util import find_spec
from typing import Literal, Optional

import torch

from .utils import configure_nemo_moe_aux_loss, is_automodel_custom_model, resolve_ac_mode


def _config_is_moe(config) -> bool:
    """Detect MoE on top-level and nested text configs."""
    if config is None:
        return False
    configs = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        configs.append(text_config)
    for candidate in configs:
        names = [*(getattr(candidate, "architectures", None) or []), getattr(candidate, "model_type", "")]
        if any("moe" in str(name).lower() for name in names):
            return True
        for key in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"):
            count = getattr(candidate, key, None)
            if isinstance(count, int) and count > 1:
                return True
    return False


def _detect_moe_arch(pretrain_or_model) -> bool:
    """Detect MoE without loading weights."""
    if not isinstance(pretrain_or_model, str):
        return _config_is_moe(getattr(pretrain_or_model, "config", None))
    try:
        from transformers import AutoConfig

        return _config_is_moe(AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True))
    except Exception:
        return False


# "tilelang" drives AutoModel's DSA (DeepSeek-style sparse attention) TileLang
# kernels — the indexer + sparse MLA path for glm_moe_dsa / deepseek_v3.2.
_CUSTOM_ATTN_IMPLEMENTATIONS = {"te", "sdpa", "flex", "tilelang"}


def _resolve_custom_backend_attn(attn_implementation: str, packing_samples: bool) -> str:
    if packing_samples:
        if attn_implementation == "te":
            return "te"
        if attn_implementation == "tilelang":
            # DSA (glm_moe_dsa / deepseek_v3.2) is THD-native: its sparse indexer
            # *requires* qkv_format='thd', which is exactly the packed layout.
            return "tilelang"
        raise ValueError(
            "--fsdp.packing_samples requires an AutoModel-native THD backend: "
            "--fsdp.attn_implementation te or tilelang."
        )

    if attn_implementation in _CUSTOM_ATTN_IMPLEMENTATIONS:
        return attn_implementation

    print(f"[Attn] AutoModel custom models do not use {attn_implementation}; using sdpa backend.")
    return "sdpa"


def _will_use_hf_model(pretrain_or_model) -> bool:
    """True if this model would load through the plain HF transformers path.

    The AutoModel (NVIDIA-NeMo/Automodel) backend is the preferred path (native
    CP/EP/TP, custom MoE+EP parallelizer, TE fused attention). HF is a fallback
    only for dense models with no registered native class. MoE always requires
    an AutoModel-native implementation.
    """
    if not isinstance(pretrain_or_model, str):
        return False
    try:
        from nemo_automodel import get_is_hf_model
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
        return get_is_hf_model(cfg, force_hf=False)
    except Exception:
        return True


def _reject_hf_fallback_features(*, is_hf_model: bool, is_moe: bool, moe_aux_loss_coef: float) -> None:
    """Reject features whose old HF-specific implementations were removed."""
    if not is_hf_model:
        return

    unsupported = []
    if is_moe:
        unsupported.append("MoE model training")
    if not is_moe and abs(float(moe_aux_loss_coef or 0.0)) > 1e-8:
        unsupported.append("MoE auxiliary loss")
    if unsupported:
        raise NotImplementedError(
            "Hugging Face fallback models do not support "
            + " or ".join(unsupported)
            + "; use an AutoModel-native implementation or disable the requested feature."
        )


def _automodel_supports_thd_packing(model_or_path) -> bool:
    """Return AutoModel's declared THD capability for a native model."""
    if not isinstance(model_or_path, str) and not is_automodel_custom_model(model_or_path):
        return False
    try:
        from nemo_automodel import query_capabilities

        return query_capabilities(model_or_path, trust_remote_code=True).supports_thd
    except Exception:
        return False


def _mtp_off_kwargs(pretrain_or_model) -> dict:
    """``from_pretrained`` config-override kwargs that disable the MTP head.

    The training actor never uses multi-token prediction (rollout spec-decode
    loads its own copy in vLLM), and some checkpoints declare MTP modules
    without shipping their weights. Returns ``{}`` when MTP is absent."""
    if not isinstance(pretrain_or_model, str):
        return {}
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
    except Exception:
        return {}
    text_config = getattr(cfg, "text_config", None)
    target = text_config if text_config is not None else cfg
    # Model families name the MTP-depth key differently; zero whichever is set.
    disabled = {
        k: 0 for k in ("mtp_num_hidden_layers", "num_nextn_predict_layers", "num_mtp_modules") if getattr(target, k, 0)
    }
    if not disabled:
        return {}
    return {"text_config": disabled} if text_config is not None else disabled


def load_automodel(
    pretrain: str,
    *,
    is_vlm: bool,
    attn_implementation: str,
    param_dtype: str,
    device_mesh,
    moe_mesh,
    cp_size: int,
    distributed_config,
    moe_config,
    activation_checkpointing,
    packing_samples: bool,
    freeze_visual_encoder: bool,
    freeze_moe_router: bool,
    use_fp32_master_weights: bool,
    moe_aux_loss_coef: float,
) -> torch.nn.Module:
    """Load and distribute a checkpoint through NeMo AutoModel.

    Validates the requested feature combination against what the path can
    support BEFORE the (possibly very large) weight load, selects the kernel
    backend, and applies post-load tweaks (router freeze, KV cache off).
    ``configure_loaded_model`` must be called on the result to validate the
    class that actually loaded.
    """
    from molt.utils.utils import convert_to_torch_dtype

    is_moe = _detect_moe_arch(pretrain)
    ep_active = moe_mesh is not None
    use_hf_model = _will_use_hf_model(pretrain)
    _reject_hf_fallback_features(
        is_hf_model=use_hf_model,
        is_moe=is_moe,
        moe_aux_loss_coef=moe_aux_loss_coef,
    )
    if is_moe and not ep_active:
        raise ValueError("MoE models require --fsdp.ep_size > 1 in the AutoModel custom-only branch.")
    # EP dispatch only exists on the AutoModel custom path; an HF-fallback
    # model under active EP would silently mis-shard experts. TP/CP run fine
    # on HF, so only EP is gated here.
    if use_hf_model and ep_active:
        raise RuntimeError(
            f"{pretrain!r}: architecture not in nemo_automodel's ModelRegistry, so molt "
            "would fall back to HF transformers, which has no expert-parallel dispatch. Use a "
            "natively registered checkpoint or run with ep_size=1."
        )
    # Fail fast on packing misconfigurations before the checkpoint load;
    # configure_loaded_model rechecks the loaded model.
    if packing_samples:
        mesh_names = getattr(device_mesh, "mesh_dim_names", ()) or ()
        pp_size = device_mesh["pp"].size() if "pp" in mesh_names else 1
        if use_hf_model:
            if attn_implementation != "flash_attention_2":
                raise ValueError(
                    "Hugging Face fallback packing requires --fsdp.attn_implementation flash_attention_2."
                )
            if cp_size > 1 or pp_size > 1:
                raise NotImplementedError("Hugging Face indexed-mask packing requires cp_size=1 and pp_size=1.")
        elif not _automodel_supports_thd_packing(pretrain):
            raise ValueError(
                "AutoModel custom implementation for this architecture does not declare THD packing support; "
                "use --fsdp.attn_implementation te with a THD-capable custom model or disable packing."
            )

    # The choice list is enforced by argparse; importability is not.
    if attn_implementation == "te" and find_spec("transformer_engine") is None:
        raise ValueError("--fsdp.attn_implementation te requires transformer-engine to be installed.")
    # fp32 master weights, bf16 fwd/bwd via FSDP2 MixedPrecisionPolicy
    # (NVIDIA-NeMo/Automodel PR #2379): a bf16 master rounds away small-LR
    # AdamW updates.
    torch_dtype = torch.float32 if use_fp32_master_weights else convert_to_torch_dtype(param_dtype)

    if is_vlm:
        from nemo_automodel import NeMoAutoModelForImageTextToText as ModelCls
    else:
        from nemo_automodel import NeMoAutoModelForCausalLM as ModelCls

    is_rank_0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if use_hf_model and is_rank_0:
        print(
            f"[AutoModel] WARNING: no native AutoModel implementation matched {pretrain!r} "
            "(architecture not in nemo_automodel ModelRegistry) — falling back to HuggingFace "
            "transformers. Native parallelism, selective activation checkpointing, and TE attention are OFF."
        )
    # AutoModel custom models take their kernels from BackendConfig and get
    # attn_implementation="sdpa" (passing "te" would additionally fire
    # AutoModel's post-init TE injection). HF fallback rejects the `backend`
    # kwarg, so it receives attn_implementation unchanged and no BackendConfig.
    attn_for_from_pretrained = attn_implementation
    backend_kwarg: dict = {}
    if not use_hf_model:
        from nemo_automodel.components.models.common.utils import BackendConfig

        backend_attn = _resolve_custom_backend_attn(attn_implementation, packing_samples)
        using_te = backend_attn == "te"
        backend_cfg = {
            "attn": backend_attn,
            # TE fused RoPE cannot handle multi-axis VLM mRoPE position
            # tensors; keep it off on every path so RoPE stays correct.
            "rope_fusion": False,
            # Pin the MoE dispatcher: BackendConfig otherwise auto-selects
            # on deep_ep importability, silently changing the training path.
            "dispatcher": os.environ.get("MOLT_MOE_DISPATCHER", "hybridep"),
            # bf16 RMSNorm recomputes non-deterministically under activation
            # checkpointing (CheckpointError); default fp32 like AutoModel recipes.
            "rms_norm": os.environ.get("MOLT_RMS_NORM", "torch_fp32"),
            # fp32 router matches vLLM's fp32 routing; a bf16 gate drifts from
            # the rollout engine and inflates vllm_kl. Matches slime/verl.
            "gate_precision": os.environ.get("MOLT_GATE_PRECISION", "float32"),
        }
        # Linear/experts kernels follow the attention choice (TE attn -> TE
        # linear/experts, else torch); env overrides decouple them for models
        # that mix backends (e.g. sdpa attention with TE linear).
        linear_backend = os.environ.get("MOLT_LINEAR_BACKEND")
        experts_backend = os.environ.get("MOLT_MOE_EXPERTS")
        if linear_backend:
            backend_cfg["linear"] = linear_backend
        elif not using_te:
            backend_cfg["linear"] = "torch"
        if experts_backend:
            backend_cfg["experts"] = experts_backend
        elif not using_te:
            backend_cfg["experts"] = "torch_mm"
        attn_for_from_pretrained = "sdpa"
        backend_kwarg = {"backend": BackendConfig(**backend_cfg)}
        if is_rank_0:
            print(f"[Attn] AutoModel custom backend={backend_attn}; config attn_implementation=sdpa.")

    # AutoModel bundles device_mesh/moe_mesh/distributed_config/moe_config/AC into
    # a single DistributedSetup; wrap our pre-built meshes via MeshContext.from_meshes.
    from nemo_automodel.components.distributed.config import DistributedSetup
    from nemo_automodel.components.distributed.mesh import MeshContext

    dist_setup = DistributedSetup(
        mesh_context=MeshContext.from_meshes(device_mesh, moe_mesh),
        strategy_config=distributed_config,
        moe_parallel_config=moe_config,
        # gradient_checkpoint CLI value (str | bool) -> bool | "selective".
        activation_checkpointing=resolve_ac_mode(activation_checkpointing),
    )
    model = ModelCls.from_pretrained(
        pretrain,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation=attn_for_from_pretrained,
        distributed_setup=dist_setup,
        use_liger_kernel=False,
        has_packed_sequence=packing_samples,
        force_hf=False,
        freeze_config={"freeze_vision_tower": True} if freeze_visual_encoder else None,
        # Disable the MTP head via AutoModel's config-override deep-merge (see
        # _mtp_off_kwargs); no-op without MTP.
        **_mtp_off_kwargs(pretrain),
        **backend_kwarg,
    )

    # Optional MoE router freeze (keeps vLLM-vs-actor routing identical).
    # Match by isinstance(Gate), not name: a name match would also catch the
    # gated-MLP `gate_proj`, which is not a router.
    if freeze_moe_router:
        from nemo_automodel.components.moe.layers import Gate

        n_frozen = 0
        for module in model.modules():
            if isinstance(module, Gate):
                for param in module.parameters(recurse=False):
                    param.requires_grad = False
                    n_frozen += 1
        if is_rank_0:
            print(f"[MoE] freeze_moe_router=True: froze {n_frozen} router param tensors")

    # https://github.com/huggingface/transformers/issues/26877
    # Use `model.generate(use_cache=True)` instead.
    model.config.use_cache = False
    return model


def configure_loaded_model(
    model: torch.nn.Module,
    *,
    packing_samples: bool,
    attn_implementation: Optional[str],
    moe_aux_loss_coef: float,
    device_mesh,
    moe_mesh,
    cp_size: int,
) -> Optional[Literal["thd", "indexed_mask"]]:
    """Validate and configure a constructed model; return its packing layout.

    Path/registry inspection during loading is best-effort, so this reruns the
    HF-fallback gate and packing checks against the class that actually loaded.
    ``attn_implementation`` is ``None`` for pre-instantiated models, which
    skips the CLI backend-flag check.
    """
    is_native_model = is_automodel_custom_model(model)
    _reject_hf_fallback_features(
        is_hf_model=not is_native_model,
        is_moe=_detect_moe_arch(model),
        moe_aux_loss_coef=moe_aux_loss_coef,
    )
    if is_native_model:
        configured_aux = configure_nemo_moe_aux_loss(model, moe_aux_loss_coef)
        if abs(float(moe_aux_loss_coef or 0.0)) > 1e-8 and not configured_aux:
            raise ValueError("MoE auxiliary loss was requested, but the AutoModel model has no native MoE gates.")
    if not packing_samples:
        return None

    if is_native_model:
        if attn_implementation is not None and attn_implementation not in {"te", "tilelang"}:
            raise ValueError("AutoModel-native packing requires --fsdp.attn_implementation te or tilelang.")
        if not _automodel_supports_thd_packing(model):
            raise ValueError(
                "This AutoModel custom model does not declare THD packing support; "
                "use a THD-capable custom model or disable --fsdp.packing_samples."
            )
        layout = "thd"
    else:
        mesh_names = getattr(device_mesh, "mesh_dim_names", ()) or ()
        pp_size = device_mesh["pp"].size() if "pp" in mesh_names else 1
        if moe_mesh is not None or cp_size > 1 or pp_size > 1:
            raise NotImplementedError(
                "Hugging Face indexed-mask packing requires cp_size=1, pp_size=1, and ep_size=1."
            )
        from nemo_automodel.components.models.common.packing import configure_packing, get_attn_implementation

        actual_attn = get_attn_implementation(None, model=model)
        if actual_attn != "flash_attention_2":
            raise RuntimeError(
                f"Hugging Face indexed-mask packing requires the model to use flash_attention_2; got {actual_attn!r}."
            )
        configure_packing("flash_attention_2")
        layout = "indexed_mask"

    print(f"[Packing] Using AutoModel {layout} packed path.")
    return layout
