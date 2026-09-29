# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Model capabilities introspection and input validation.

Provides :class:`ModelSupports` (a read-only descriptor of what a model can
do) and :func:`attach_capabilities_and_validate` which attaches
``model.supports``, ``model.supports_*``, and ``model.validate_for_mesh``
to any ``nn.Module``.

Capabilities are derived from code introspection -- class attributes, mixin
inheritance, forward-signature inspection -- so they stay in sync as models
evolve without manual feature tables.
"""

from __future__ import annotations

import functools
import inspect
import logging
import weakref
from typing import TYPE_CHECKING

from nemo_automodel._transformers.model_capabilities import query_capabilities

if TYPE_CHECKING:
    import torch.nn as nn

    from nemo_automodel.components.distributed.mesh import MeshContext

logger = logging.getLogger(__name__)


def _has_optimized_tp_plan(model_cls: type) -> bool:
    """Check if *model_cls* has an entry in ``PARALLELIZE_FUNCTIONS``."""
    from nemo_automodel.components.distributed.optimized_tp_plans import (
        PARALLELIZE_FUNCTIONS,
        _get_class_qualname,
    )

    return _get_class_qualname(model_cls) in PARALLELIZE_FUNCTIONS


def _is_moe(model_cls: type) -> bool:
    from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin

    return issubclass(model_cls, MoEFSDPSyncMixin)


def _supports_seq_lens(model: "nn.Module") -> bool:
    """True when ``model.forward()`` accepts a ``seq_lens`` kwarg."""
    # @akoumparouli: this is a bit of a hack, but it's the best we can do for now
    # TODO: improve this
    fwd = getattr(model, "forward", None)
    if not callable(fwd):
        return False
    try:
        params = inspect.signature(fwd).parameters
        if "seq_lens" in params:
            return True
        return any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (ValueError, TypeError):
        return False


def _has_backend(model: "nn.Module") -> bool:
    """True for custom models that carry a ``BackendConfig``."""
    backend = getattr(model, "backend", None)
    return backend is not None and hasattr(backend, "attn")


def _uses_te_attention(model: "nn.Module") -> bool:
    """True when the model uses the TE attention backend.

    Covers two cases:
    - Custom models built with ``BackendConfig(attn='te')``.
    - HF models that had TE injected via :func:`inject_te_attention`
      (flagged by ``model._te_attention_injected``).
    """
    backend = getattr(model, "backend", None)
    if getattr(backend, "attn", None) == "te":
        return True
    return getattr(model, "_te_attention_injected", False)


def _uses_magi_attention(model: "nn.Module") -> bool:
    """True when the model uses the MagiAttention (FFA / context-parallel) backend.

    MagiAttention implements context parallelism via its own load-balancing
    dispatch (see ``components/distributed/context_parallel/magi.py``), so it supports CP.
    """
    backend = getattr(model, "backend", None)
    return getattr(backend, "attn", None) == "magi"


def _is_deepseek_v4(model: "nn.Module") -> bool:
    """True when the model is a DeepSeek V4 custom model.

    DSV4 owns its context-parallel attention (Miles-style contiguous query shard
    plus all-gathered K/V), so its CP support is gated on the TileLang attention
    backend rather than the generic TE/SDPA/Magi paths.
    """
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) == "deepseek_v4":
        return True
    # Exact names: sibling architectures such as ``DeepseekV41*`` do not own a CP attention path.
    return type(model).__name__ in ("DeepseekV4ForCausalLM", "DeepseekV4Model")


def _is_glm_moe_dsa(model: "nn.Module") -> bool:
    config = getattr(model, "config", None)
    return getattr(config, "model_type", None) == "glm_moe_dsa" or type(model).__name__.startswith("GlmMoeDsa")


def _is_hybrid(model: "nn.Module") -> bool:
    """True when the model mixes attention with non-attention layers (e.g. Mamba/SSM).

    Detected via config attributes used by NemotronH (``layers_block_type``)
    and HF hybrid models (``hybrid_override_pattern``, ``is_hybrid_model``).
    For VLM wrappers, also inspect the inner ``language_model``'s config.
    """
    candidates = [getattr(model, "config", None)]
    inner = getattr(model, "language_model", None)
    if inner is not None:
        candidates.append(getattr(inner, "config", None))
    # VLM configs nest the decoder config under ``text_config``.
    for c in list(candidates):
        if c is not None:
            candidates.append(getattr(c, "text_config", None))
    for config in candidates:
        if config is None:
            continue
        for attr in ("layers_block_type", "hybrid_override_pattern"):
            pattern = getattr(config, attr, None)
            if pattern and any(str(c).upper() == "M" for c in pattern):
                return True
        if getattr(config, "is_hybrid_model", False) is True:
            return True
        # Qwen3.5 / Qwen3-Next style: per-layer ``layer_types`` mixing
        # ``linear_attention`` (gated-delta / SSM) with ``full_attention``.
        layer_types = getattr(config, "layer_types", None)
        if layer_types and any(str(t) == "linear_attention" for t in layer_types):
            return True
    return False


class ModelSupports:
    """Queryable feature-support descriptor attached to a model instance.

    Every property is derived from introspection of the live model so it
    reflects the actual class hierarchy and forward signature, not a
    hand-maintained table.

    Usage::

        model = NeMoAutoModelForCausalLM.from_pretrained(...)
        model.supports.tp   # True / False
        model.supports.pp   # ...
    """

    __slots__ = ("_model_ref", "_model_cls", "_mesh")

    def __init__(self, model: "nn.Module", mesh: "MeshContext | None" = None) -> None:
        # Hold the model weakly. ``ModelSupports`` is attached back onto the model
        # as ``model._supports``; a strong reference here would form a
        # ``model <-> _supports`` cycle, so the capability descriptor must never be
        # the reason a (multi-GiB) model stays resident after its owner is dropped.
        self._model_ref = weakref.ref(model)
        self._model_cls = type(model)
        self._mesh = mesh

    @property
    def _model(self) -> "nn.Module":
        model = self._model_ref()
        if model is None:
            raise ReferenceError("ModelSupports: underlying model has been garbage-collected")
        return model

    def __repr__(self) -> str:
        names = (
            "tp",
            "pp",
            "cp",
            "ep",
            "sequence_packing",
            "mtp_cp",
            "mtp_cp_pp",
            "cp_vision_frame_sharding",
            "gradient_checkpointing",
            "generate",
        )
        flags = ", ".join("{}={}".format(name, getattr(self, "supports_" + name)) for name in names)
        flags += ", mtp_enabled={}".format(self.mtp_enabled)
        flags += ", is_custom_model={}".format(self.is_custom_model)
        return "ModelSupports({})".format(flags)

    # model kind

    @property
    def is_custom_model(self) -> bool:
        """True when the model class has a custom (non-HF) implementation in the registry."""
        from nemo_automodel._transformers.registry import ModelRegistry

        return ModelRegistry.has_custom_model(self._model_cls.__name__)

    # parallelism

    @property
    def supports_tp(self) -> bool:
        """Model has an optimized or HF-native tensor-parallel plan."""
        return _has_optimized_tp_plan(self._model_cls) or getattr(self._model, "_tp_plan", None) is not None

    @property
    def supports_pp(self) -> bool:
        """Model supports pipeline parallelism.

        True when the model either declares a ``_pp_plan`` or inherits from
        ``MoEFSDPSyncMixin`` (MoE models handle PP via
        ``patched_backward_maybe_with_nosync``).
        """
        return getattr(self._model, "_pp_plan", None) is not None or _is_moe(self._model_cls)

    # alias

    @property
    def supports_tp_plan(self) -> bool:
        return self.supports_tp

    @property
    def supports_pp_plan(self) -> bool:
        return self.supports_pp

    @property
    def supports_cp(self) -> bool:
        """Model supports context parallelism.

        +------------------+----------------+---------+
        | Model kind       | Attention      | CP?     |
        +------------------+----------------+---------+
        | Custom (owns CP) | any            | Yes     |
        | Custom           | TE             | Yes     |
        | Custom           | Magi (FFA)     | Yes     |
        | Custom hybrid    | TE / SDPA      | Yes     |
        | Custom           | FlexAttention  | No      |
        | HF (pure attn)   | SDPA           | Yes     |
        | HF (pure attn)   | no SDPA        | No      |
        | HF hybrid (Mamba)| any            | No      |
        +------------------+----------------+---------+
        """
        if getattr(self._model, "_owns_cp_attention", False):
            # The model ships its own CP batch sharding and per-layer transport for
            # every layer type it has (e.g. Kimi Linear's FLA context for KDA plus
            # gathered-KV FlexAttention for MLA), so no attention backend is required.
            return True
        if _has_backend(self._model):
            backend_attn = getattr(getattr(self._model, "backend", None), "attn", None)
            if _is_deepseek_v4(self._model):
                return backend_attn == "tilelang"
            if _is_glm_moe_dsa(self._model):
                return backend_attn in ("tilelang", "cudnn")
            # Hybrids, and custom models that ship their own CP-aware attention and opt in
            # via ``_supports_cp_sdpa``, may run CP on either TE or SDPA attention.
            if _is_hybrid(self._model) or getattr(self._model, "_supports_cp_sdpa", False):
                backend_attn = getattr(getattr(self._model, "backend", None), "attn", None)
                return backend_attn in ("te", "sdpa")
            return _uses_te_attention(self._model) or _uses_magi_attention(self._model)
        if _is_hybrid(self._model):
            return False
        return getattr(self._model, "_supports_sdpa", False) is True

    @property
    def supports_ep(self) -> bool:
        """Model is a Mixture-of-Experts that supports expert parallelism."""
        return _is_moe(self._model_cls)

    # misc

    @property
    def supports_sequence_packing(self) -> bool:
        """``forward()`` accepts ``seq_lens`` for packed-sequence training."""
        model = self._model
        backend_attn = getattr(getattr(model, "backend", None), "attn", None)
        model_owned_backends = getattr(model, "_packed_cp_attn_backends", ())
        sp_attn_backend = (
            getattr(model, "_supports_sdpa", False) is True
            or _uses_te_attention(model)
            or _uses_magi_attention(model)
            or (self.supports_thd and backend_attn == "tilelang")
            or backend_attn in model_owned_backends
            # Models that build their own per-document masks (Kimi Linear's
            # document-blocked causal mask plus per-document KDA ``cu_seqlens``)
            # need no packing-aware attention backend.
            or getattr(model, "_owns_packed_attention", False)
        )
        return _supports_seq_lens(model) and sp_attn_backend

    @property
    def supports_thd(self) -> bool:
        """Model owns its native THD packed-sequence input path."""
        try:
            capabilities = query_capabilities(self._model)
        except AttributeError:
            return False
        return capabilities.supports_thd

    @property
    def supports_mtp_cp(self) -> bool:
        """Model owns a verified MTP training path under context parallelism."""
        try:
            capabilities = query_capabilities(self._model)
        except AttributeError:
            return False
        return capabilities.supports_mtp_cp

    @property
    def supports_mtp_cp_pp(self) -> bool:
        """Model owns a verified MTP training path with both CP and PP."""
        try:
            capabilities = query_capabilities(self._model)
        except AttributeError:
            return False
        return capabilities.supports_mtp_cp_pp

    @property
    def mtp_enabled(self) -> bool:
        """Whether MTP is active for this configured model instance."""
        mtp_config = getattr(self._model, "mtp_config", None)
        return bool(getattr(mtp_config, "enabled", False))

    @property
    def supports_cp_vision_frame_sharding(self) -> bool:
        """Model owns a verified CP vision frame-sharding integration."""
        try:
            capabilities = query_capabilities(self._model)
        except AttributeError:
            return False
        return capabilities.supports_cp_vision_frame_sharding

    @property
    def supports_generate(self) -> bool:
        """Model has a ``generate()`` method for autoregressive inference."""
        return callable(getattr(self._model, "generate", None))

    @property
    def supports_gradient_checkpointing(self) -> bool:
        """Gradient checkpointing is supported."""
        if self.supports_ep and self.ep_size > 1:
            return False
        for cls in type(self._model).__mro__:
            if "supports_gradient_checkpointing" in cls.__dict__:
                val = cls.__dict__["supports_gradient_checkpointing"]
                if isinstance(val, (property, classmethod, staticmethod)):
                    continue
                return val is True
        return False

    # mesh-aware helpers

    @property
    def cp_size(self) -> int:
        return getattr(self._mesh, "cp_size", 1)

    @property
    def tp_size(self) -> int:
        return getattr(self._mesh, "tp_size", 1)

    @property
    def pp_size(self) -> int:
        return getattr(self._mesh, "pp_size", 1)

    @property
    def ep_size(self) -> int:
        return getattr(self._mesh, "ep_size", 1)

    @property
    def supports_cp_with_sequence_packing(self) -> bool:
        """CP + packed sequences requires a backend with packed CP routing.

        MagiAttention dispatches the packed sequence across the CP group with its
        own load-balancing solver and a per-document varlen mask, so it supports
        CP + packing (see ``context_parallel.magi.magi_prepare_packed_cp``). Models
        with native THD support own their packed CP path in an optimized attention backend.
        Models can restrict a model-owned packed CP path to specific attention
        backends with ``_packed_cp_attn_backends``.
        Models that own their CP end to end (``_owns_cp_attention``) shard the
        packed batch themselves and carry document boundaries into every layer."""
        model = self._model
        if not self.supports_sequence_packing:
            return False
        if self.cp_size <= 1:
            return True
        model_owned_backends = getattr(model, "_packed_cp_attn_backends", None)
        if model_owned_backends is not None:
            backend_attn = getattr(getattr(model, "backend", None), "attn", None)
            return _supports_seq_lens(model) and backend_attn in model_owned_backends
        if getattr(model, "_owns_cp_attention", False):
            return True
        if self.supports_thd:
            backend_attn = getattr(getattr(model, "backend", None), "attn", None)
            return backend_attn == "tilelang"
        return _uses_te_attention(model) or _uses_magi_attention(model)


def validate_for_mesh(model: "nn.Module", mesh: "MeshContext") -> None:
    """Validate *mesh* parallelism sizes against this model's capabilities.

    Works both as a bound method (``model.validate_for_mesh()``) and as a
    standalone call (``validate_for_mesh(model)``).

    Raises :class:`ValueError` with one bullet per violation.
    """
    if mesh is None:
        return

    # If capabilities haven't been attached yet, use a temporary ModelSupports.
    if not hasattr(model, "supports"):
        supports = ModelSupports(model, mesh)
    else:
        supports = model.supports

    tp_size = getattr(mesh, "tp_size", 1)
    pp_size = getattr(mesh, "pp_size", 1)
    ep_size = getattr(mesh, "ep_size", 1)
    cp_size = getattr(mesh, "cp_size", 1)

    arch = type(model).__name__
    errors: list[str] = []

    if tp_size > 1 and not supports.supports_tp:
        errors.append(
            f"Tensor parallelism (tp_size={tp_size}) requested but {arch} "
            f"has no TP plan (not in PARALLELIZE_FUNCTIONS and no `_tp_plan` attribute).\n"
            f"Please re-run with --distributed.tp_size=1 or\n"
            f"modify distributed YAML config section:\n"
            f"distributed:\n"
            f"  tp_size: 1"
        )

    if pp_size > 1 and not supports.supports_pp:
        errors.append(
            f"Pipeline parallelism (pp_size={pp_size}) requires a _pp_plan "
            f"attribute on {arch}, but none was found.\n"
            f"Please re-run with --distributed.pp_size=1 or\n"
            f"modify distributed YAML config section:\n"
            f"distributed:\n"
            f"  pp_size: 1"
        )

    if cp_size > 1 and not supports.supports_cp:
        if _is_hybrid(model) and _has_backend(model):
            errors.append(
                f"Context parallelism (cp_size={cp_size}) for hybrid model {arch} "
                f"requires the TE or SDPA attention backend (backend.attn='te' or 'sdpa').\n"
                f"Please switch attention backend:\n"
                f"model:\n"
                f"  backend:\n"
                f"    attn: te  # or sdpa"
            )
        elif _is_hybrid(model):
            errors.append(
                f"Context parallelism (cp_size={cp_size}) is not supported for "
                f"hybrid model {arch} (contains Mamba/SSM layers).\n"
                f"Please re-run with --distributed.cp_size=1 or\n"
                f"modify distributed YAML config section:\n"
                f"distributed:\n"
                f"  cp_size: 1"
            )
        elif _is_deepseek_v4(model):
            errors.append(
                f"Context parallelism (cp_size={cp_size}) for {arch} requires "
                f"the TileLang attention backend (backend.attn='tilelang').\n"
                f"Please re-run with --distributed.cp_size=1 or switch to TileLang attention:\n"
                f"model:\n"
                f"  backend:\n"
                f"    attn: tilelang"
            )
        elif _is_glm_moe_dsa(model):
            errors.append(
                f"Context parallelism (cp_size={cp_size}) for {arch} requires "
                f"the TileLang or cuDNN DSA attention backend.\n"
                f"Please re-run with --distributed.cp_size=1 or switch attention backend:\n"
                f"model:\n"
                f"  backend:\n"
                f"    attn: tilelang  # or cudnn"
            )
        elif _has_backend(model):
            errors.append(
                f"Context parallelism (cp_size={cp_size}) for {arch} requires "
                f"the TE attention backend (backend.attn='te').\n"
                f"Please re-run with --distributed.cp_size=1 or switch to TE attention:\n"
                f"model:\n"
                f"  backend:\n"
                f"    attn: te"
            )
        else:
            errors.append(
                f"Context parallelism (cp_size={cp_size}) not supported with {arch} "
                f"(model does not declare _supports_sdpa).\n"
                f"Please re-run with --distributed.cp_size=1 or\n"
                f"modify distributed YAML config section:\n"
                f"distributed:\n"
                f"  cp_size: 1"
            )

    if ep_size > 1 and not supports.supports_ep:
        errors.append(
            f"Expert parallelism (ep_size={ep_size}) requires a MoE model, "
            f"but {arch} does not inherit from MoEFSDPSyncMixin.\n"
            f"Please re-run with --distributed.ep_size=1 or\n"
            f"modify distributed YAML config section:\n"
            f"distributed:\n"
            f"  ep_size: 1"
        )

    if errors:
        raise ValueError(f"Unsupported configuration for {arch}:\n" + "\n".join(f"  - {e}" for e in errors))


def _supports_forwarding_property(name: str) -> property:
    """Property that forwards ``model.<name>`` to ``model.supports.<name>``."""

    def fget(self: "nn.Module") -> bool:
        try:
            return getattr(self.supports, name)
        except ReferenceError:
            self._supports = ModelSupports(self, getattr(self, "_mesh", None))  # type: ignore[attr-defined]
            return getattr(self._supports, name)  # type: ignore[attr-defined]

    fget.__name__ = name
    return property(fget)


def _lazy_supports_property(self: "nn.Module") -> ModelSupports:
    supports = getattr(self, "_supports", None)
    if isinstance(supports, ModelSupports) and supports._model_ref() is self:
        return supports
    # Pipeline splitting deep-copies model stages after capabilities were attached.
    # The copied ``_supports`` still points at the source model, so rebuild it for
    # the live stage before any supports_* forwarding property is evaluated.
    self._supports = ModelSupports(self, getattr(self, "_mesh", None))  # type: ignore[attr-defined]
    return self._supports  # type: ignore[attr-defined]


@functools.lru_cache(maxsize=1)
def _build_class_dict() -> dict[str, property | type]:
    cls_dict: dict[str, property | type] = {
        "supports": property(_lazy_supports_property),
    }
    for attr in dir(ModelSupports):
        if attr.startswith("supports_") or attr == "is_custom_model":
            cls_dict[attr] = _supports_forwarding_property(attr)
    return cls_dict


def attach_capabilities_and_validate(model: "nn.Module", mesh: "MeshContext") -> "nn.Module":
    """Attach ``model.supports`` and ``model.supports_*`` and call validate_for_mesh.

    Injects a thin dynamic subclass so that property descriptors (supports_*)
        resolve via ``__getattribute__`` with no ``__getattr__`` overhead,
        which avoids triggering ModelCapabilitiesMixin.__getattr__ for models
        that lack the attribute.
    Safe to call more than once -- subsequent calls are no-ops.
    """
    if "supports" not in type(model).__dict__:
        orig_cls = model.__class__
        new_cls = type(
            orig_cls.__name__,
            (orig_cls,),
            _build_class_dict(),
        )
        new_cls.__module__ = orig_cls.__module__
        new_cls.__qualname__ = orig_cls.__qualname__
        model.__class__ = new_cls
    validate_for_mesh(model, mesh)
    return model
