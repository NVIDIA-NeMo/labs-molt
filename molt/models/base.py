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
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

import os
from importlib.util import find_spec
from typing import Union

import torch
import torch.nn as nn

from molt.trainer.fsdp.packing import is_automodel_custom_model

from .utils import (
    configure_nemo_moe_aux_loss,
    resolve_ac_mode,
)


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


_HF_ATTN_IMPLEMENTATIONS = {"eager", "sdpa", "flash_attention_2", "flash_attention_3", "te"}
# "tilelang" drives AutoModel's DSA (DeepSeek-style sparse attention) TileLang
# kernels — the indexer + sparse MLA path for glm_moe_dsa / deepseek_v3.2.
_CUSTOM_ATTN_IMPLEMENTATIONS = {"te", "sdpa", "flex", "tilelang"}
_ALL_ATTN_IMPLEMENTATIONS = _HF_ATTN_IMPLEMENTATIONS | _CUSTOM_ATTN_IMPLEMENTATIONS


def _validate_attn_implementation(attn_implementation: str) -> None:
    if attn_implementation not in _ALL_ATTN_IMPLEMENTATIONS:
        choices = ", ".join(sorted(_ALL_ATTN_IMPLEMENTATIONS))
        raise ValueError(f"Unsupported attention implementation {attn_implementation!r}; choose one of: {choices}")
    if attn_implementation == "te" and find_spec("transformer_engine") is None:
        raise ValueError("--fsdp.attn_implementation te requires transformer-engine to be installed.")


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


def _will_use_hf_model(pretrain_or_model, default: bool = True) -> bool:
    """True if this model would load through the plain HF transformers path.

    The AutoModel (NVIDIA-NeMo/Automodel) backend is the preferred path (native
    CP/EP/TP, custom MoE+EP parallelizer, TE fused attention). HF is a fallback
    only for dense models with no registered native class. MoE always requires
    an AutoModel-native implementation.
    """
    if not isinstance(pretrain_or_model, str):
        return False
    try:
        from nemo_automodel._transformers.model_init import get_is_hf_model
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
        return get_is_hf_model(cfg, force_hf=False)
    except Exception:
        return default


def _reject_hf_fallback_features(
    *, is_hf_model: bool, is_moe: bool, packing_samples: bool, moe_aux_loss_coef: float
) -> None:
    """Reject features whose old HF-specific implementations were removed."""
    if not is_hf_model:
        return

    unsupported = []
    if is_moe:
        unsupported.append("MoE model training")
    if packing_samples:
        unsupported.append("THD sequence packing")
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
        from nemo_automodel._transformers.model_capabilities import query_capabilities

        return query_capabilities(model_or_path, trust_remote_code=True).supports_thd
    except Exception:
        return False


def _first_token_id(config, *attr_names):
    """First integer token id among ``attr_names`` on the VLM config, else None.

    VLM families name the media placeholder id differently (image_token_id /
    image_token_index / img_context_token_id). Uses ``isinstance(int)`` (not
    truthiness) so a valid id of 0 is not skipped.
    """
    for name in attr_names:
        tid = getattr(config, name, None)
        if isinstance(tid, int):
            return tid
    return None


def _mtp_off_kwargs(pretrain_or_model) -> dict:
    """Return the ``from_pretrained`` config-override kwarg that disables the MTP head.

    The training actor never uses the multi-token-prediction head (rollout spec-decode
    loads its own copy in vLLM). AutoModel deep-merges nested config dicts, so
    ``text_config={...}`` patches just that field (same mechanism as the recipe yaml's
    ``text_config.mtp_num_hidden_layers: 0``). Returns ``{}`` when MTP is absent or the
    path can't be introspected."""
    if not isinstance(pretrain_or_model, str):
        return {}
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
    except Exception:
        return {}
    text_config = getattr(cfg, "text_config", None)
    target = text_config if text_config is not None else cfg
    # MoE families name the MTP-depth config key differently (all keys tried below);
    # disable whichever the config enables. Needed even when a checkpoint declares MTP
    # modules but ships no MTP weights (building them would fail the weight load).
    disabled = {
        k: 0 for k in ("mtp_num_hidden_layers", "num_nextn_predict_layers", "num_mtp_modules") if getattr(target, k, 0)
    }
    if not disabled:
        return {}
    return {"text_config": disabled} if text_config is not None else disabled


class BaseModel(nn.Module):
    """Shared base for the RL model wrappers (``Actor`` and ``Critic``).

    Owns model construction through AutoModel, including distributed setup,
    activation checkpointing, optional value-head installation, and R3 binding.
    AutoModel ``Engine`` owns every forward path.
    """

    def __init__(
        self,
        pretrain_or_model,
        attn_implementation: str = "flash_attention_2",
        param_dtype: str = "bf16",
        device_mesh=None,
        moe_mesh=None,
        distributed_config=None,
        moe_config=None,
        activation_checkpointing: Union[bool, str] = False,
        packing_samples: bool = False,
        temperature: float = 1.0,
        freeze_visual_encoder: bool = False,
        freeze_moe_router: bool = False,
        use_fp32_master_weights: bool = True,
        moe_aux_loss_coef: float = 0.0,
        routing_replay: bool = False,
        pre_fsdp_hook=None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected {type(self).__name__} keyword argument(s): {unexpected}")
        self.temperature = temperature
        self.packing_samples = packing_samples
        self._routing_replay_adapter = None

        if not isinstance(pretrain_or_model, str):
            if pre_fsdp_hook is not None:
                raise ValueError("pre_fsdp_hook requires loading the model through NeMoAutoModel")
            self.model = pretrain_or_model
            self.is_vlm = False
            is_native_model = is_automodel_custom_model(self.model)
            is_moe = _detect_moe_arch(self.model)
            _reject_hf_fallback_features(
                is_hf_model=not is_native_model,
                is_moe=is_moe,
                packing_samples=self.packing_samples,
                moe_aux_loss_coef=moe_aux_loss_coef,
            )
            if is_native_model:
                configured_aux = configure_nemo_moe_aux_loss(self.model, moe_aux_loss_coef)
                if abs(float(moe_aux_loss_coef or 0.0)) > 1e-8 and not configured_aux:
                    raise ValueError(
                        "MoE auxiliary loss was requested, but the AutoModel model has no native MoE gates."
                    )
            if self.packing_samples and not _automodel_supports_thd_packing(self.model):
                raise ValueError(
                    "This pre-instantiated AutoModel custom model does not declare THD packing support. "
                    "Use an AutoModel custom TE model or disable --fsdp.packing_samples."
                )
            if routing_replay:
                self._enable_routing_replay()
            return

        from molt.utils.utils import convert_to_torch_dtype, is_vlm_model

        # Trainable actors keep fp32 master weights unless the architecture
        # requires compute-dtype parameters. FSDP2 handles bf16 fwd/bwd via
        # MixedPrecisionPolicy.
        compute_dtype = convert_to_torch_dtype(param_dtype)
        is_moe = _detect_moe_arch(pretrain_or_model)
        ep_active = moe_mesh is not None
        use_hf_model = _will_use_hf_model(pretrain_or_model)
        _reject_hf_fallback_features(
            is_hf_model=use_hf_model,
            is_moe=is_moe,
            packing_samples=packing_samples,
            moe_aux_loss_coef=moe_aux_loss_coef,
        )
        if is_moe and not ep_active:
            raise ValueError("MoE models require --fsdp.ep_size > 1 in the AutoModel custom-only branch.")
        # EP dispatch is a nemo_automodel custom-path feature; HF has no equivalent. An
        # HF-fallback model under active EP would silently mis-shard experts / train on
        # wrong grads, so forbid it loudly. (TP/CP run on HF, so they aren't gated here.)
        if use_hf_model and ep_active:
            raise RuntimeError(
                f"{pretrain_or_model!r}: architecture not in nemo_automodel's ModelRegistry, so molt "
                "would fall back to HF transformers — which has no expert-parallel (EP) dispatch, but "
                "EP is active here (ep_size>1). The HF fallback is forbidden under EP. Use a checkpoint "
                "whose `architectures` is natively registered (e.g. omni3: NemotronH_Nano_Omni_Reasoning_V3, "
                "the official GA model — not a renamed alias), or run with ep_size=1."
            )
        if packing_samples and not _automodel_supports_thd_packing(pretrain_or_model):
            raise ValueError(
                "AutoModel custom implementation for this architecture does not declare THD packing support; "
                "use --fsdp.attn_implementation te with a THD-capable custom model or disable packing."
            )

        _validate_attn_implementation(attn_implementation)
        # fp32 master weights (including MoE): matches AutoModel's master-weight
        # contract (NVIDIA-NeMo/Automodel PR #2379) — load in fp32, let FSDP2's
        # MixedPrecisionPolicy(param_dtype=bf16) do bf16 fwd/bwd. A bf16 master
        # rounds away AdamW updates (~LR < bf16 ULP) at small LR, so the MoE never learns.
        torch_dtype = compute_dtype if not use_fp32_master_weights else torch.float32
        self.is_vlm = is_vlm_model(pretrain_or_model)

        if self.is_vlm:
            from nemo_automodel import NeMoAutoModelForImageTextToText as ModelCls
        else:
            from nemo_automodel import NeMoAutoModelForCausalLM as ModelCls

        # AutoModel owns attention selection (forces sdpa under CP, falls back
        # FA2->sdpa when a model lacks FA2). When no custom path matches a dense
        # model (e.g. Qwen3-8B), it uses HF transformers directly.
        if use_hf_model and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            print(
                f"[AutoModel] WARNING: no native AutoModel implementation matched {pretrain_or_model!r} "
                "(architecture not in nemo_automodel ModelRegistry) — falling back to HuggingFace "
                "transformers. Native parallelism, selective activation checkpointing, and TE attention are OFF."
            )
        # AutoModel custom drives attention/MoE through a BackendConfig and hands
        # from_pretrained "sdpa": passing "te" would also fire AutoModel's own post-init
        # TE injection (auto_model.py) on top of the backend's. HF rejects the `backend`
        # kwarg, so we omit it there and pass attn_implementation through unchanged.
        attn_for_from_pretrained = attn_implementation
        backend_kwarg: dict = {}
        if not use_hf_model:
            from nemo_automodel.components.models.common.utils import BackendConfig

            backend_attn = _resolve_custom_backend_attn(attn_implementation, packing_samples)
            using_te = backend_attn == "te"
            # Disable TE fused RoPE everywhere: VLM mRoPE position tensors don't match
            # the simpler 4D rotary layout the fused kernel expects (first surfaced under
            # Qwen3.5-MoE CP; disabled unconditionally to keep RoPE correct on all paths).
            backend_cfg = {"attn": backend_attn, "rope_fusion": False}
            # Pin the MoE dispatcher (BackendConfig otherwise auto-selects on deep_ep
            # importability, silently changing the training path). Default hybridep to
            # match AutoModel; == deepep on intra-node NVLink. Override MOLT_MOE_DISPATCHER
            # (d580 recipes pin deepep — cross-node hybridep/DOCA-GPUNetIO fails there).
            backend_cfg["dispatcher"] = os.environ.get("MOLT_MOE_DISPATCHER", "hybridep")
            # Linear (GEMM) + experts backend follow the attention choice by default
            # (TE attn -> TE linear/experts, else torch). Some models decouple them
            # (e.g. sparse-attn arch needs sdpa but wants TE linear + gmm experts), so
            # MOLT_LINEAR_BACKEND / MOLT_MOE_EXPERTS override the attn-coupled default.
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
            # RMS-norm precision. bf16 RMSNorm recomputes non-deterministically under
            # activation checkpointing (-> CheckpointError) and destabilizes the MoE grad
            # norm. Default fp32 (matches AutoModel reference recipes). Override: MOLT_RMS_NORM.
            backend_cfg["rms_norm"] = os.environ.get("MOLT_RMS_NORM", "torch_fp32")
            # Force the MoE router to fp32 (BackendConfig defaults the gate linear to
            # the bf16 bulk dtype). A bf16 router drifts from vLLM's fp32 router ->
            # rollout-vs-train logprobs diverge and vllm_kl climbs. Matches slime/verl.
            # Override: MOLT_GATE_PRECISION.
            backend_cfg["gate_precision"] = os.environ.get("MOLT_GATE_PRECISION", "float32")
            attn_for_from_pretrained = "sdpa"
            backend_kwarg = {"backend": BackendConfig(**backend_cfg)}
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"[Attn] AutoModel custom backend={backend_attn}; config attn_implementation=sdpa.")

        # AutoModel bundles device_mesh/moe_mesh/distributed_config/moe_config/AC into
        # a single DistributedSetup; wrap our pre-built meshes via MeshContext.from_meshes.
        from nemo_automodel.components.distributed.config import DistributedSetup
        from nemo_automodel.components.distributed.mesh import MeshContext

        # `activation_checkpointing` is the gradient_checkpoint CLI value (str | bool).
        # Default "full" matches AutoModel's MoE/deepep recipes; pass "selective" for
        # TorchTitan per-op AC.
        ac_setting = resolve_ac_mode(activation_checkpointing)
        dist_setup = DistributedSetup(
            mesh_context=MeshContext.from_meshes(device_mesh, moe_mesh),
            strategy_config=distributed_config,
            moe_parallel_config=moe_config,
            activation_checkpointing=ac_setting,
        )
        self.model = ModelCls.from_pretrained(
            pretrain_or_model,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            attn_implementation=attn_for_from_pretrained,
            distributed_setup=dist_setup,
            use_liger_kernel=False,
            has_packed_sequence=packing_samples,
            force_hf=False,
            pre_fsdp_hook=pre_fsdp_hook,
            freeze_config={"freeze_vision_tower": True} if freeze_visual_encoder else None,
            # Disable the MTP head via AutoModel's config-override deep-merge (see
            # _mtp_off_kwargs); no-op without MTP.
            **_mtp_off_kwargs(pretrain_or_model),
            **backend_kwarg,
        )
        # Registry/config inspection is best-effort. Recheck the loaded class so
        # a late AutoModel -> HF fallback cannot enter either removed feature.
        is_native_model = is_automodel_custom_model(self.model)
        _reject_hf_fallback_features(
            is_hf_model=not is_native_model,
            is_moe=_detect_moe_arch(self.model),
            packing_samples=packing_samples,
            moe_aux_loss_coef=moe_aux_loss_coef,
        )
        if is_native_model:
            configured_aux = configure_nemo_moe_aux_loss(self.model, moe_aux_loss_coef)
            if abs(float(moe_aux_loss_coef or 0.0)) > 1e-8 and not configured_aux:
                raise ValueError("MoE auxiliary loss was requested, but the AutoModel model has no native MoE gates.")
        if routing_replay:
            self._enable_routing_replay()
        if self.packing_samples:
            print("[Packing] Using AutoModel THD/TE packed path.")

        # Optionally freeze the MoE router/gate (keeps vLLM-vs-actor routing identical,
        # stabilizes training). Match by isinstance(Gate), NOT by name: the path varies by
        # arch and a `gate` name match would also catch the gated-MLP `gate_proj.weight`,
        # which is not a router. requires_grad=False drops it from the optimizer and refit.
        if freeze_moe_router:
            try:
                from nemo_automodel.components.moe.layers import Gate
            except ImportError:
                Gate = None
            n_frozen = 0
            if Gate is not None:
                for module in self.model.modules():
                    if isinstance(module, Gate):
                        for param in module.parameters(recurse=False):
                            param.requires_grad = False
                            n_frozen += 1
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f"[MoE] freeze_moe_router=True: froze {n_frozen} router param tensors")

        # https://github.com/huggingface/transformers/issues/26877
        # Use `model.generate(use_cache=True)` instead.
        self.model.config.use_cache = False

        if self.is_vlm:
            vlm_config = self.model.config
            self._image_token_id = _first_token_id(
                vlm_config, "image_token_id", "image_token_index", "img_context_token_id"
            )
            self._video_token_id = _first_token_id(
                vlm_config, "video_token_id", "video_token_index", "video_context_token_id"
            )

    def _enable_routing_replay(self) -> None:
        """Bind AutoModel's model-scoped rollout routing adapter."""
        from nemo_automodel.components.moe.router_replay import RouterReplayAdapter

        self._routing_replay_adapter = RouterReplayAdapter(self.model)
        print(f"[R3] Routing replay enabled at global layer ids {list(self._routing_replay_adapter.layer_ids)}.")
