# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

import inspect
import os
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from molt.trainer.fsdp.packing import (
    cp_dtensor_full_sequence,
    cp_local_seq_index,
    is_automodel_custom_model,
    pack_padded_batch,
    pad_to_cp_multiple,
    unpack_to_padded,
)

from .utils import (
    attach_nemo_moe_aux_loss,
    configure_nemo_moe_aux_loss,
    move_model_to_cpu_for_offload,
    resolve_ac_mode,
)


def _detect_moe_arch(pretrain_or_model) -> bool:
    """Lightweight MoE detection from HF config (no model load)."""
    if not isinstance(pretrain_or_model, str):
        return False
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
        archs = getattr(cfg, "architectures", None) or []
        if any("Moe" in a or "MoE" in a for a in archs):
            return True
        for k in ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts"):
            n = getattr(cfg, k, None)
            if isinstance(n, int) and n > 1:
                return True
    except Exception:
        return False
    return False


def _has_hf_flash_attn_2() -> bool:
    try:
        from transformers.utils import is_flash_attn_2_available

        return bool(is_flash_attn_2_available())
    except Exception:
        return find_spec("flash_attn") is not None


_HF_ATTN_IMPLEMENTATIONS = {"eager", "sdpa", "flash_attention_2", "flash_attention_3", "te"}
_CUSTOM_ATTN_IMPLEMENTATIONS = {"te", "sdpa", "flex"}
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
        if attn_implementation == "flash_attention_2":
            raise ValueError(
                "--fsdp.packing_samples with AutoModel custom models requires --fsdp.attn_implementation te. "
                "HF fallback packing is removed in this branch."
            )
        raise ValueError("--fsdp.packing_samples supports only --fsdp.attn_implementation te or flash_attention_2.")

    if attn_implementation in _CUSTOM_ATTN_IMPLEMENTATIONS:
        return attn_implementation

    print(f"[Attn] AutoModel custom models do not use {attn_implementation}; using sdpa backend.")
    return "sdpa"


def _will_use_hf_model(pretrain_or_model, default: bool = True) -> bool:
    """True if this model would load through the plain HF transformers path.

    The AutoModel (NVIDIA-NeMo/Automodel) backend is the preferred path (native
    CP/EP/TP, custom MoE+EP parallelizer, TE fused attention). HF is a fallback for
    models with no registered native class and supports only text +
    flash_attention_2 + packing (no CP/EP/TP).
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


def _class_source_supports_thd_packing(model_cls) -> bool:
    try:
        source_path = inspect.getsourcefile(model_cls)
    except (TypeError, OSError):
        return False
    if not source_path:
        return False
    try:
        source = Path(source_path).read_text(errors="ignore")
    except OSError:
        return False
    return "qkv_format" in source and "cu_seqlens" in source


def _automodel_arch_supports_thd_packing(pretrain_or_model) -> bool:
    """Return whether AutoModel's custom class consumes THD packing kwargs."""
    # Only reached with a model-path string (the pre-instantiated branch returns earlier).
    try:
        from nemo_automodel._transformers.registry import ModelRegistry
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(pretrain_or_model, trust_remote_code=True)
        archs = getattr(cfg, "architectures", None) or []
        if not archs:
            return False
        # model_arch_name_to_cls is a _LazyArchMapping (no .get()) — use contains/getitem.
        _arch_map = ModelRegistry.model_arch_name_to_cls
        model_cls = _arch_map[archs[0]] if archs[0] in _arch_map else None
        return bool(model_cls) and _class_source_supports_thd_packing(model_cls)
    except Exception:
        return False


def _automodel_custom_supports_thd_packing(model: nn.Module) -> bool:
    if not is_automodel_custom_model(model):
        return False
    return any(_class_source_supports_thd_packing(cls) for cls in type(model).__mro__)


class _AttrDict(dict):
    """Dict output that also supports ``output.foo`` trainer access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def _normalize_output(output):
    if isinstance(output, torch.Tensor):
        return _AttrDict(logits=output)
    if isinstance(output, dict) and not isinstance(output, _AttrDict):
        return _AttrDict(output)
    return output


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

    Owns what they share: building the model via AutoModel's ``from_pretrained``
    (HF weights + per-arch TP plan + FSDP2 wrap + optional CP hooks / activation
    checkpointing), the input prep + model call (``_forward_backbone``), and the
    full-sequence restore. Subclasses add only their head's ``forward``: ``Actor``
    returns log-probs/entropy, ``Critic`` per-token values.
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
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected {type(self).__name__} keyword argument(s): {unexpected}")
        self.temperature = temperature
        self.packing_samples = packing_samples
        self._forward_autocast_dtype = None
        self.device_mesh = device_mesh
        mesh_dims = getattr(device_mesh, "mesh_dim_names", ()) or ()
        self.cp_mesh = device_mesh["cp"] if device_mesh is not None and "cp" in mesh_dims else None
        self.cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1

        if not isinstance(pretrain_or_model, str):
            self.model = pretrain_or_model
            self.is_vlm = False
            self._packing_style = "automodel" if is_automodel_custom_model(self.model) else "hf"
            configure_nemo_moe_aux_loss(self.model, moe_aux_loss_coef)
            if self.packing_samples and self._packing_style == "automodel":
                if not _automodel_custom_supports_thd_packing(self.model):
                    raise ValueError(
                        "This pre-instantiated AutoModel custom model does not consume THD packing kwargs. "
                        "Use an AutoModel custom TE model or disable --fsdp.packing_samples."
                    )
            if self.packing_samples and self._packing_style == "hf":
                cfg = getattr(self.model, "config", None)
                if getattr(cfg, "_attn_implementation", None) != "flash_attention_2" or not _has_hf_flash_attn_2():
                    raise ValueError(
                        "HF packed sequence requires flash_attention_2 and flash-attn. "
                        "Use an AutoModel custom TE model or load the HF model with flash_attention_2."
                    )
            return

        from molt.utils.utils import convert_to_torch_dtype, is_vlm_model

        # Trainable actors keep fp32 master weights unless the architecture
        # requires compute-dtype parameters. FSDP2 handles bf16 fwd/bwd via
        # MixedPrecisionPolicy.
        compute_dtype = convert_to_torch_dtype(param_dtype)
        is_moe = _detect_moe_arch(pretrain_or_model)
        ep_active = moe_mesh is not None
        if is_moe and not ep_active:
            raise ValueError("MoE models require --fsdp.ep_size > 1 in the AutoModel custom-only branch.")
        use_hf_model = _will_use_hf_model(pretrain_or_model)
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
        if packing_samples:
            if not use_hf_model and not _automodel_arch_supports_thd_packing(pretrain_or_model):
                raise ValueError(
                    "AutoModel custom implementation for this architecture does not consume THD packing kwargs; "
                    "use --fsdp.attn_implementation te with a THD-capable custom model or disable packing."
                )
            # HF packing works only with FA2's varlen kernel: sdpa ignores the
            # cu_seq_lens kwargs and would silently fuse packed-row boundaries.
            if use_hf_model and (attn_implementation != "flash_attention_2" or not _has_hf_flash_attn_2()):
                raise ValueError(
                    "HF model packing requires --fsdp.attn_implementation flash_attention_2 with flash-attn installed. "
                    "Disable --fsdp.packing_samples or use FA2 (sdpa would silently fuse packed-batch boundaries)."
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
        # FA2->sdpa when a model lacks FA2). When no custom path matches (e.g. dense
        # Qwen3-8B) it uses HF transformers directly — fine for dense models, which
        # lack the MoE/EP/CP features anyway. Log once so MoE users notice a silent degrade.
        if use_hf_model and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            print(
                f"[AutoModel] WARNING: no native AutoModel implementation matched {pretrain_or_model!r} "
                "(architecture not in nemo_automodel ModelRegistry) — falling back to HuggingFace "
                "transformers. The native path (custom MoE/EP parallelizer, selective activation "
                "checkpointing, TE attention) is OFF. For MoE / hybrid-SSM (Mamba) checkpoints this can "
                "silently degrade throughput AND break activation-checkpoint recompute determinism. "
                "Verify this checkpoint's `architectures` is registered if you expected the native path."
            )
        # With fp32 master weights, autocast the forward to the compute dtype so
        # lm_head/score inputs match the bf16 parameters.
        if compute_dtype != torch.float32:
            self._forward_autocast_dtype = compute_dtype

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
            # Qwen3.5-MoE CP mixes mRoPE position tensors with local sequence
            # shards; TE fused RoPE expects the simpler 4D rotary layout.
            backend_cfg = {"attn": backend_attn, "rope_fusion": using_te and self.cp_size <= 1}
            # Pin the MoE dispatcher (BackendConfig otherwise auto-selects on deep_ep
            # importability, silently changing the training path). Default deepep: makes
            # MoE recompute deterministic under activation checkpointing and gives correct
            # RL grads (with the FsdpStrategy grad-sync fix). Override: MOLT_MOE_DISPATCHER.
            backend_cfg["dispatcher"] = os.environ.get("MOLT_MOE_DISPATCHER", "deepep")
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
            # Disable the MTP head via AutoModel's config-override deep-merge (see
            # _mtp_off_kwargs); no-op without MTP.
            **_mtp_off_kwargs(pretrain_or_model),
            **backend_kwarg,
        )
        self.model = move_model_to_cpu_for_offload(self.model, distributed_config)
        # from_pretrained may downgrade to HF even when custom was requested;
        # re-derive from the loaded class so the forward picks the right pack style.
        self._packing_style = "automodel" if is_automodel_custom_model(self.model) else "hf"
        configure_nemo_moe_aux_loss(self.model, moe_aux_loss_coef)
        if routing_replay:
            # R3: give every MoE gate a RouterReplay handle so the training forward
            # replays the rollout's expert selection. Done post from_pretrained (not via
            # MoEConfig) to stay model-agnostic. vLLM indexes its routing buffer by GLOBAL
            # decoder-layer id, so each gate replays against its own global-id row (read
            # from the module path); uniform-MoE models parse to 0..N-1 (no-op reorder).
            import re

            from nemo_automodel.components.moe.router_replay import RouterReplay

            # _build_routing_targets marks positions with no captured routing with a -1
            # sentinel ("keep the live selection"). Stock RouterReplay returns the target
            # verbatim, so -1 reaches the expert gather (torch.gather forbids negative
            # indices) -> device-side assert. Keep the live selection at the sentinels;
            # outside REPLAY super() returns live indices (>=0), so the where is a no-op.
            # No-ops once AutoModel upstreams the same where-guard.
            class _SentinelRouterReplay(RouterReplay):
                def apply(self, indices: torch.Tensor) -> torch.Tensor:
                    result = super().apply(indices)
                    return torch.where(result >= 0, result, indices)

            RouterReplay.clear_registry()
            gate_layer_ids: list[int] = []
            for name, module in self.model.named_modules():
                if hasattr(module, "router_replay"):  # an AutoModel MoE gate
                    module.router_replay = _SentinelRouterReplay()
                    m = re.search(r"layers\.(\d+)\b", name)
                    gate_layer_ids.append(int(m.group(1)) if m else len(gate_layer_ids))
            if not gate_layer_ids:
                raise RuntimeError("routing_replay is on but the model has no MoE router gates.")
            self._num_routing_gates = len(gate_layer_ids)
            self._moe_layer_global_ids = gate_layer_ids
            print(
                f"[R3] Routing replay enabled on {len(gate_layer_ids)} MoE gates "
                f"at global layer ids {gate_layer_ids}."
            )
        if self.packing_samples:
            print("[Packing] Using AutoModel THD/TE packed path.")

        # VLM: optionally freeze the vision encoder so only the language backbone
        # trains (language params live under "language_model.*" / "lm_head.*").
        #
        # CP>1 forces freezing: PyTorch CP attention runs `resize_` on the sharded
        # inputs_embeds and rejects requires_grad=True. The pre-embed already runs
        # under no_grad, so no gradient reaches the vision tower anyway; freezing
        # just keeps optimizer state from holding never-updated vision params.
        effective_freeze_visual = freeze_visual_encoder
        if self.is_vlm and self.cp_size > 1 and not freeze_visual_encoder:
            effective_freeze_visual = True
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(
                    "[VLM] cp_size>1 forces freeze_visual_encoder=True "
                    "(PyTorch CP requires inputs_embeds.requires_grad=False)."
                )
        if self.is_vlm and effective_freeze_visual:
            for name, param in self.model.named_parameters():
                if "language_model" not in name and "lm_head" not in name:
                    param.requires_grad = False

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
            self._vlm_config = self.model.config
            # Resolve once at construction so forward() doesn't redo the attribute
            # fallback per microbatch (field names vary across VLM families).
            self._image_token_id = _first_token_id(
                self._vlm_config, "image_token_id", "image_token_index", "img_context_token_id"
            )
            self._video_token_id = _first_token_id(
                self._vlm_config, "video_token_id", "video_token_index", "video_context_token_id"
            )

    def _restore_full_sequence(self, t, *, cp_forward, batch, seqlen, indices):
        """Map a per-token tensor back onto the full ``[B, seqlen]`` axis.

        Undoes the two seq-axis transforms the forward applies before the model
        call: CP sharding (gather the CP shards, then trim the CP pad tail back
        to ``seqlen``) and sample packing (scatter the packed THD rows back to
        padded ``[B, seqlen]``). No-op when neither CP nor packing is active.
        """
        if cp_forward:
            t = cp_dtensor_full_sequence(t, self.cp_mesh, seq_dim=1)[:, :seqlen]
        if self.packing_samples:
            t = unpack_to_padded(t, indices, batch, seqlen)
        return t

    def _build_routing_targets(self, routed_experts, indices, cp_forward):
        """Shard the R3 routing ids to this rank's forward token order (RouterReplay).

        ``routed_experts`` is ``(B, vllm_layers, topk, S)`` (rollout top-k expert ids per
        token, seq last). Returns one ``(num_tokens, topk)`` tensor per MoE gate, reordered
        to what the gate sees on this rank (CP-local under cp>1, pad-removed under packing,
        else plain ``B*S``). Positions with no captured routing carry a -1 sentinel that
        ``RouterReplay`` keeps at the live selection.

        vLLM indexes its layer dim by GLOBAL decoder-layer id, so we pick each gate's own
        global-id row (``_moe_layer_global_ids``); uniform-MoE ids are 0..N-1 (plain order).
        """
        n_gates = self._num_routing_gates
        global_ids = self._moe_layer_global_ids
        routing = routed_experts  # (B, vllm_layers, topk, S), seq last
        if cp_forward:
            # Take this rank's CP shard: pad the seq dim as the forward did, with the -1
            # sentinel (so CP pad tokens aren't force-routed to expert 0), then gather this
            # rank's chunks. Local length = padded / cp_size, from `routing` (pre-shard).
            routing = pad_to_cp_multiple(routing, self.cp_size, seq_dim=3, value=-1)
            local_positions = cp_local_seq_index(routing.shape[3] // self.cp_size, self.cp_mesh, routing.device)
            routing = routing.index_select(3, local_positions)
        b, n_layers, topk, s = routing.shape
        if n_layers <= max(global_ids):
            raise ValueError(
                f"rollout routing has {n_layers} layers but a MoE gate maps to global "
                f"layer {max(global_ids)} (have {n_gates} gates at ids {global_ids})."
            )
        # (B, layers, topk, S) seq-last -> (B*S, layers, topk) token-major, one row per token
        per_token = routing.permute(0, 3, 1, 2).reshape(b * s, n_layers, topk).long()
        if self.packing_samples:
            per_token = per_token.index_select(0, indices)  # drop pad tokens to match the packed order
        return [per_token[:, gid, :].contiguous() for gid in global_ids]

    def _forward_backbone(
        self,
        sequences: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        cp_context_stack,
        mm_inputs: dict,
        output_hidden_states: bool = False,
        routed_experts: Optional[torch.Tensor] = None,
    ):
        """Input prep (packing / VLM token-type ids / CP sharding) + model call.

        Returns ``(output, rolled_sequences, cp_forward, indices, batch, seqlen)``:
        ``output`` is the normalized model output (``_AttrDict`` with ``logits`` and,
        for custom MoE, ``aux_loss``); the rest is the state ``_restore_full_sequence``
        needs to map a per-token tensor back onto the dense ``[B, seqlen]`` axis.
        ``Actor`` turns this into log-probs; ``Critic`` into per-token values.

        ``position_ids`` is normally recomputed internally; it stays an input for
        callers that precompute it (packed sequences / VLM mRoPE).
        """
        batch, seqlen = sequences.size()
        attn_kwargs: dict = {}
        indices = None
        cp_forward = False
        cp_ctx_factory = nullcontext
        inputs_embeds = None
        if self.packing_samples:
            sequences, position_ids, rolled_sequences, indices, attn_kwargs = pack_padded_batch(
                sequences, attention_mask, style=self._packing_style
            )
            forward_attention_mask = None
        else:
            # https://github.com/OpenRLHF/OpenRLHF/issues/217
            rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
            forward_attention_mask = attention_mask

            if getattr(self, "is_vlm", False):
                if mm_inputs:
                    image_token_id = self._image_token_id
                    video_token_id = self._video_token_id
                    if image_token_id is None:
                        raise AttributeError(
                            f"VLM config {type(self._vlm_config).__name__} missing image token id "
                            "(expected one of: image_token_id, image_token_index, img_context_token_id)"
                        )
                    token_type_ids = (sequences == image_token_id).to(torch.int32)
                    if video_token_id is not None:
                        token_type_ids[sequences == video_token_id] = 2
                    # Detect silent vision drop: pixel_values present but no image-context
                    # tokens -> text-only logits while the rollout used vision (policy mismatch).
                    if "pixel_values" in mm_inputs and not getattr(self, "_warned_no_image_tokens", False):
                        n_img = int(token_type_ids.eq(1).sum().item())
                        if n_img == 0:
                            import logging as _logging

                            _logging.getLogger(__name__).warning(
                                f"VLM forward: pixel_values present but no image-context tokens "
                                f"(id={image_token_id}) found in sequences; visual features will "
                                f"be dropped. Check that rollout sequences preserve the image "
                                f"placeholder (collapse / dedup may have removed them)."
                            )
                            self._warned_no_image_tokens = True
                    key = "mm_token_type_ids" if "image_grid_thw" in mm_inputs else "token_type_ids"
                    mm_inputs[key] = token_type_ids
            elif position_ids is None:
                if attention_mask is None:
                    position_ids = torch.arange(seqlen, device=sequences.device).unsqueeze(0).expand(batch, -1)
                else:
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)

            if self.cp_size > 1 and attention_mask is not None:
                from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx

                # VLM + CP: the vision tower must run before CP shards the sequence,
                # which AutoModel supports only via its pre-embed hook
                # (`prepare_model_inputs_for_cp` — NVIDIA-NeMo/Automodel#2125). Models
                # without the hook run at cp=1 (use TP/EP for memory). Pre-embed here via
                # `_pre_embed_only` under no_grad (PyTorch CP resize_s the sharded buffer).
                #
                # Pre-embed UNCONDITIONALLY (do not gate on `mm_inputs`): it fires an FSDP
                # all-gather over the dp_cp group, so gating on per-rank images would let
                # image-free ranks skip it -> divergent collective -> NCCL deadlock. A
                # text-only microbatch just embeds its tokens, keeping ranks in lockstep.
                if getattr(self, "is_vlm", False):
                    if not hasattr(self.model, "prepare_model_inputs_for_cp"):
                        raise RuntimeError(
                            "VLM + CP requires the model's AutoModel pre-embed hook "
                            "(prepare_model_inputs_for_cp); this model lacks it — run with "
                            "cp_size=1 (use TP/EP for memory)."
                        )
                    # Fail fast when image-placeholder tokens are present but mm_inputs is empty
                    # (rollout dropped the image): get_rope_index would hit image_grid_thw=None
                    # and crash cryptically. Likely cause: a chat agent attaching images only at
                    # a literal <image> marker that a structured-content VLM already rendered away.
                    if not mm_inputs and self._image_token_id is not None:
                        n_img = int((sequences == self._image_token_id).sum().item())
                        if n_img:
                            raise RuntimeError(
                                f"VLM+CP pre-embed: {n_img} image placeholder token(s) present but no "
                                "multimodal inputs — the rollout dropped the image. Structured-content "
                                "VLMs render <image> to a model placeholder, so agents that interleave "
                                "images at a literal <image> marker attach nothing. Fix the chat agent "
                                "(attach images marker-independently) or use the step runner (geo3k.py)."
                            )
                    with torch.no_grad():
                        inputs_embeds = self.model(input_ids=sequences, **mm_inputs, _pre_embed_only=True)[
                            "inputs_embeds"
                        ]
                    mm_inputs = {}

                # Pad to CP's 2*cp_size divisor before make_cp_batch_and_ctx injects
                # position_ids, so the arange stays dense and shifted-token gather
                # targets stay valid in the pad tail.
                sequences = pad_to_cp_multiple(sequences, self.cp_size, seq_dim=1, value=0)
                attention_mask = pad_to_cp_multiple(attention_mask, self.cp_size, seq_dim=1, value=0)
                rolled_sequences = pad_to_cp_multiple(rolled_sequences, self.cp_size, seq_dim=1, value=0)
                if inputs_embeds is not None:
                    inputs_embeds = pad_to_cp_multiple(inputs_embeds, self.cp_size, seq_dim=1, value=0)

                primary_key, primary_tensor = (
                    ("inputs_embeds", inputs_embeds) if inputs_embeds is not None else ("input_ids", sequences)
                )
                cp_batch = {
                    primary_key: primary_tensor,
                    "attention_mask": attention_mask,
                    "labels": rolled_sequences,
                }
                cp_ctx_factory, cp_batch = make_cp_batch_and_ctx(self.device_mesh, cp_batch)
                position_ids = cp_batch.get("position_ids")
                rolled_sequences = cp_batch["labels"]
                forward_attention_mask = cp_batch.get("attention_mask")
                if inputs_embeds is not None:
                    inputs_embeds = cp_batch["inputs_embeds"]
                else:
                    sequences = cp_batch["input_ids"]
                cp_forward = True

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self._forward_autocast_dtype)
            if self._forward_autocast_dtype is not None and sequences.is_cuda
            else nullcontext()
        )

        forward_ctx = cp_ctx_factory()
        if cp_context_stack is not None and cp_forward:
            # AutoModel CP train context installs backward hooks, so training
            # code keeps it alive until loss.backward() completes.
            cp_context_stack.enter_context(forward_ctx)
            forward_ctx = nullcontext()

        # R3: replay the rollout's per-token expert selection so the training router
        # picks the same experts (kills routing drift). Ids sharded to this rank's order.
        replay_ctx = nullcontext()
        if routed_experts is not None:
            from nemo_automodel.components.moe.router_replay import RouterReplay

            replay_ctx = RouterReplay.replay(self._build_routing_targets(routed_experts, indices, cp_forward))
            # Replay must stay active through the activation-checkpoint recompute in
            # backward, else the recompute reverts to the live router and disagrees with
            # the replayed forward (CheckpointError). Keep it on the caller's stack; the
            # no-grad old-logprob recompute passes no stack -> forward-only.
            if cp_context_stack is not None:
                cp_context_stack.enter_context(replay_ctx)
                replay_ctx = nullcontext()

        with forward_ctx:
            with autocast_ctx:
                # Always pass sequences as keyword `input_ids`: some VLM forwards
                # declare `pixel_values` first positional, so a bare positional would
                # collide with it. In the pre-embed CP path we pass inputs_embeds
                # directly (the model auto-detects it and skips multimodal scatter).
                forward_kwargs = dict(
                    attention_mask=forward_attention_mask,
                    position_ids=position_ids,
                    **attn_kwargs,
                    **mm_inputs,
                )
                if output_hidden_states:
                    forward_kwargs["output_hidden_states"] = True
                if cp_forward and inputs_embeds is not None:
                    forward_kwargs["inputs_embeds"] = inputs_embeds
                else:
                    forward_kwargs["input_ids"] = sequences
                with replay_ctx:
                    output = self.model(**forward_kwargs)
        # AutoModel's custom MoE/LLM models (e.g. Qwen3MoeForCausalLM) return a
        # raw logits Tensor; HF returns a ModelOutput with `.logits`. Normalize.
        output = _normalize_output(output)
        output = attach_nemo_moe_aux_loss(output, self.model)
        return output, rolled_sequences, cp_forward, indices, batch, seqlen
