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

"""State dict adapter for DeepSeek V4.

HF V4 uses different key names compared to V3/V3.2.  This adapter performs
the necessary renaming on top of the standard FP8 dequantization and
per-expert weight aggregation.

Key mapping (HF -> internal):
  embed.weight                          -> model.embed_tokens.weight
  norm.weight                           -> model.norm.weight
  head.weight                           -> lm_head.weight
  layers.{i}.attn_norm.weight           -> model.layers.{i}.input_layernorm.weight
  layers.{i}.ffn_norm.weight            -> model.layers.{i}.post_attention_layernorm.weight
  layers.{i}.attn.*                     -> model.layers.{i}.self_attn.*
  layers.{i}.ffn.gate.weight            -> model.layers.{i}.mlp.gate.weight
  layers.{i}.ffn.gate.bias             -> model.layers.{i}.mlp.gate.e_score_correction_bias
  layers.{i}.ffn.gate.tid2eid          -> model.layers.{i}.mlp.gate.tid2eid  (hash layers only)
  layers.{i}.ffn.shared_experts.w1.*   -> model.layers.{i}.mlp.shared_experts.gate_proj.*
  layers.{i}.ffn.shared_experts.w3.*   -> model.layers.{i}.mlp.shared_experts.up_proj.*
  layers.{i}.ffn.shared_experts.w2.*   -> model.layers.{i}.mlp.shared_experts.down_proj.*
  layers.{i}.ffn.experts.{j}.w1.weight -> aggregated into model.layers.{i}.mlp.experts.gate_and_up_projs
  layers.{i}.ffn.experts.{j}.w3.weight -> aggregated into model.layers.{i}.mlp.experts.gate_and_up_projs
  layers.{i}.ffn.experts.{j}.w2.weight -> aggregated into model.layers.{i}.mlp.experts.down_projs
  layers.{i}.hc_attn_base/fn/scale     -> model.layers.{i}.hc_attn_base/fn/scale
  layers.{i}.hc_ffn_base/fn/scale      -> model.layers.{i}.hc_ffn_base/fn/scale

FP8 note: HF V4 stores scale as `<key>.scale` (not `<key>.weight_scale_inv` like V3).
Both suffixes are handled by the dequantization step.
"""

from __future__ import annotations

import enum
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import (
    BLOCK_SIZE,
    dequantize_from_fp8,
)
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
    get_submesh,
    is_dtensor,
    should_load_expert_for_rank,
    split_experts_weights_dtensor_aware,
)

# V4 Flash routed-expert weights are stored as FP4 (e2m1fn) packed two values per
# int8 byte, with FP8 (e8m0fnu) per-row scales covering 32-column groups:
#   weight: int8 with shape [out, in // 2]         (low nibble + high nibble = 2 fp4 values)
#   scale:  float8_e8m0fnu with shape [out, in // 32]
# Non-expert weights (attention, norms, embed, lm_head, shared experts) use the
# standard FP8 e4m3fn with BLOCK_SIZE×BLOCK_SIZE (128×128) scaling.
FP4_COL_BLOCK = 32

# FP4 e2m1 value table: low 3 bits -> mantissa/exponent, MSB -> sign.
# Layout: [positive values for 0-7, negative values for 8-15].
_FP4_E2M1_TABLE = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


# HF V4 key -> internal key  (simple renames; expert & FP8 handled separately)
_HF_TO_INTERNAL_RENAMES: list[tuple[re.Pattern, str]] = [
    # Optional native vision modules and sentinel embeddings.
    (re.compile(r"^vision\.(.+)$"), r"model.vision.\1"),
    (re.compile(r"^aligner\.(.+)$"), r"model.aligner.\1"),
    (re.compile(r"^(image_start|image_end|image_newline|image_pad)$"), r"model.\1"),
    # Top-level
    (re.compile(r"^embed\.(.+)$"), r"model.embed_tokens.\1"),
    (re.compile(r"^norm\.(.+)$"), r"model.norm.\1"),
    (re.compile(r"^head\.(.+)$"), r"lm_head.\1"),
    # Per-layer norms
    (re.compile(r"^layers\.(\d+)\.attn_norm\.(.+)$"), r"model.layers.\1.input_layernorm.\2"),
    (re.compile(r"^layers\.(\d+)\.ffn_norm\.(.+)$"), r"model.layers.\1.post_attention_layernorm.\2"),
    # Attention sub-keys.  Order matters: specific rules must precede the generic
    # catch-all because regex matching short-circuits on the first match.
    #
    # Two structural divergences between the released DSV4-Flash safetensors
    # (which follow the DeepSeek inference reference's module tree) and HF
    # PR 45616's flattened layout, which we mirror:
    #
    #   1) Compressor's RMSNorm is named ``norm`` on disk but ``kv_norm`` in HF.
    #   2) Indexer has its OWN nested compressor sub-module on disk
    #      (``indexer.compressor.{ape,norm,wgate,wkv}``) but HF flattened those
    #      attributes onto the Indexer itself (``indexer.{ape,kv_norm,wgate,wkv}``).
    #
    # Both renames must run before the generic ``attn.(.+)`` catch-all.
    (re.compile(r"^layers\.(\d+)\.attn\.attn_sink$"), r"model.layers.\1.self_attn.sinks_param.weight"),
    # Indexer in HF is nested under Compressor (Compressor.indexer); on disk
    # Indexer is a sibling of Compressor with its OWN nested compressor:
    #   on-disk  layers.X.attn.indexer.compressor.{ape,norm,wgate,wkv}
    #   on-disk  layers.X.attn.indexer.{wq_b,weights_proj}
    #   our mod  model.layers.X.self_attn.compressor.indexer.{ape,kv_norm,wgate,wkv,wq_b,weights_proj}
    # So all on-disk ``attn.indexer.*`` keys land under ``self_attn.compressor.indexer.*``.
    # Specific ``indexer.compressor.*`` rules first (they collapse the nested compressor),
    # then a catch-all ``indexer.*`` that just adds the ``compressor.`` prefix.
    (
        re.compile(r"^layers\.(\d+)\.attn\.indexer\.compressor\.norm\.(.+)$"),
        r"model.layers.\1.self_attn.compressor.indexer.kv_norm.\2",
    ),
    (
        re.compile(r"^layers\.(\d+)\.attn\.indexer\.compressor\.ape$"),
        r"model.layers.\1.self_attn.compressor.indexer.ape_param.weight",
    ),
    (
        re.compile(r"^layers\.(\d+)\.attn\.indexer\.compressor\.wgate\.(.+)$"),
        r"model.layers.\1.self_attn.compressor.indexer.wgate.\2",
    ),
    (
        re.compile(r"^layers\.(\d+)\.attn\.indexer\.compressor\.wkv\.(.+)$"),
        r"model.layers.\1.self_attn.compressor.indexer.wkv.\2",
    ),
    (
        re.compile(r"^layers\.(\d+)\.attn\.indexer\.(.+)$"),
        r"model.layers.\1.self_attn.compressor.indexer.\2",
    ),
    # Outer compressor's norm rename (after indexer rules so we don't
    # accidentally rewrite ``indexer.compressor.norm`` here).
    (
        re.compile(r"^layers\.(\d+)\.attn\.compressor\.norm\.(.+)$"),
        r"model.layers.\1.self_attn.compressor.kv_norm.\2",
    ),
    (
        re.compile(r"^layers\.(\d+)\.attn\.compressor\.ape$"),
        r"model.layers.\1.self_attn.compressor.ape_param.weight",
    ),
    (re.compile(r"^layers\.(\d+)\.attn\.(.+)$"), r"model.layers.\1.self_attn.\2"),
    # MoE gate (score weight + optional bias correction + hash table)
    (re.compile(r"^layers\.(\d+)\.ffn\.gate\.bias$"), r"model.layers.\1.mlp.gate.e_score_correction_bias"),
    (re.compile(r"^layers\.(\d+)\.ffn\.gate\.(.+)$"), r"model.layers.\1.mlp.gate.\2"),
    # Shared expert (w1=gate, w3=up, w2=down)
    (
        re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w1\.(.+)$"),
        r"model.layers.\1.mlp.shared_experts.gate_proj.\2",
    ),
    (
        re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w3\.(.+)$"),
        r"model.layers.\1.mlp.shared_experts.up_proj.\2",
    ),
    (
        re.compile(r"^layers\.(\d+)\.ffn\.shared_experts\.w2\.(.+)$"),
        r"model.layers.\1.mlp.shared_experts.down_proj.\2",
    ),
    # Latent projections (fc1: hidden→latent, fc2: latent→hidden)
    (
        re.compile(r"^layers\.(\d+)\.ffn\.fc1_latent_proj\.(.+)$"),
        r"model.layers.\1.mlp.fc1_latent_proj.\2",
    ),
    (
        re.compile(r"^layers\.(\d+)\.ffn\.fc2_latent_proj\.(.+)$"),
        r"model.layers.\1.mlp.fc2_latent_proj.\2",
    ),
    # HC (hash-clustering) parameters
    # HC (Hyper-Connection) per-site mixers — HF submodule layout:
    #   layers.{i}.hc_attn_{fn,base,scale}  ->  model.layers.{i}.attn_hc.{fn,base,scale}
    #   layers.{i}.hc_ffn_{fn,base,scale}   ->  model.layers.{i}.ffn_hc.{fn,base,scale}
    (re.compile(r"^layers\.(\d+)\.hc_attn_(base|fn|scale)$"), r"model.layers.\1.attn_hc.\2"),
    (re.compile(r"^layers\.(\d+)\.hc_ffn_(base|fn|scale)$"), r"model.layers.\1.ffn_hc.\2"),
    # MTP-local HC head.  Native MTP keys are normalized to temporary
    # ``layers.{k}.*`` keys before the rename table is applied.
    (re.compile(r"^layers\.(\d+)\.hc_head_(fn|base|scale)$"), r"model.layers.\1.hc_head.hc_\2"),
    # Final HC-head collapse module:
    #   hc_head_{fn,base,scale}  ->  model.hc_head.hc_{fn,base,scale}
    # (HF uses ``hc_fn`` / ``hc_base`` / ``hc_scale`` inside HyperHead, in
    #  contrast to ``fn`` / ``base`` / ``scale`` inside HyperConnection.)
    (re.compile(r"^hc_head_(fn|base|scale)$"), r"model.hc_head.hc_\1"),
]

# Routed-expert pattern in HF V4 format
_EXPERT_PATTERN = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.(w1|w2|w3)\.weight$")


class _HashBiasScope(enum.Enum):
    """Key-format scope for :meth:`DeepSeekV4StateDictAdapter._drop_hash_layer_gate_bias`."""

    INTERNAL = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$")
    HF = re.compile(r"^layers\.(\d+)\.ffn\.gate\.bias$")


class _ExpertQuantLayout(enum.Enum):
    """On-disk routed-expert quantization layout for DeepSeek V4 checkpoints."""

    FP4 = "fp4"
    FP8 = "fp8"


def _rename_hf_key(key: str) -> str:
    """Apply simple rename rules; returns the key unchanged if no rule matches."""
    for pattern, replacement in _HF_TO_INTERNAL_RENAMES:
        new_key, n = pattern.subn(replacement, key)
        if n:
            return new_key
    return key


class DeepSeekV4StateDictAdapter(StateDictAdapter):
    """State dict adapter for DeepSeek V4."""

    def __init__(
        self,
        config: DeepseekV4Config,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._checkpoint_expert_quant_layout_cache: _ExpertQuantLayout | None = None

    # ------------------------------------------------------------------
    # from_hf
    # ------------------------------------------------------------------

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert HF checkpoint to internal format.

        Steps:
          1. Split native ``mtp.{k}.*`` keys (and legacy
             ``layers.{num_hidden_layers+k}.*`` keys) from backbone keys and
             renumber them as temporary ``layers.{k}.*`` keys so the standard
             pipeline (dequantize / aggregate-experts / rename) handles them too.
          2. Dequantize FP8 / FP4 weights for both backbone and MTP.
          3. Aggregate per-expert routed weights into stacked tensors.
          4. Rename keys using the HF -> internal mapping table.
          5. Re-prefix MTP keys: ``model.layers.{k}.*`` -> ``mtp.layers.{k}.*``.
        """
        N = self.config.num_hidden_layers
        num_mtp = int(getattr(self.config, "num_nextn_predict_layers", 0) or 0)
        # HF V4 emits both prefixed (``model.layers.{N+k}.*`` for self_attn /
        # mlp / norms) and unprefixed (``layers.{N+k}.*`` for V4 fusion-only
        # modules eh_proj / enorm / hnorm / final_layernorm) MTP keys, so the
        # split regex must accept either form.
        _layer_re = re.compile(r"^(model\.)?layers\.(\d+)\.")

        # Split MTP keys from backbone keys.  Current DSV4-Flash stores MTP as
        # ``mtp.{k}.*``; HF/intermediate exports can also use
        # ``[model.]layers.{N+k}.*``.  Normalize either format to temporary
        # ``layers.{k}.*`` keys so the standard dequantize / aggregate / rename
        # pipeline can handle FP4 routed experts and FP8 projections uniformly.
        mtp_hf: dict[str, Any] = {}
        backbone_hf: dict[str, Any] = {}
        native_mtp_re = re.compile(r"^mtp\.(\d+)\.")
        for key in list(hf_state_dict.keys()):
            val = hf_state_dict[key]
            native_m = native_mtp_re.match(key)
            if native_m is not None:
                mtp_depth = int(native_m.group(1))
                if mtp_depth < num_mtp:
                    renumbered = f"layers.{mtp_depth}." + key[native_m.end() :]
                    mtp_hf[renumbered] = val
                # Drop checkpoint MTP tensors when the runtime config disables
                # MTP.  Otherwise loading DSV4-Flash with
                # num_nextn_predict_layers=0 produces a large set of dangling
                # ``mtp.0.*`` keys.
                continue

            m = _layer_re.match(key)
            if m and int(m.group(2)) >= N and num_mtp > 0:
                orig_idx = int(m.group(2))
                mtp_depth = orig_idx - N
                if mtp_depth < num_mtp:
                    renumbered = f"layers.{mtp_depth}." + key[m.end() :]
                    mtp_hf[renumbered] = val
                continue

            backbone_hf[key] = val
        hf_state_dict = backbone_hf

        hf_state_dict = self._dequantize(hf_state_dict)
        hf_state_dict = self._aggregate_experts(hf_state_dict, device_mesh)
        state_dict = self._rename_all(hf_state_dict)

        if mtp_hf:
            mtp_hf = self._dequantize(mtp_hf)
            mtp_hf = self._aggregate_experts(mtp_hf, device_mesh)
            mtp_renamed = self._rename_all(mtp_hf)
            for key, val in mtp_renamed.items():
                # After _rename_all, layer-indexed keys are in one of two forms:
                #   - ``model.layers.{k}.*`` if a rename rule matched (norms,
                #     attn, mlp, experts, hc), or
                #   - ``layers.{k}.*`` if no rule matched — V4 MTP-only
                #     modules (``e_proj`` / ``h_proj`` / ``enorm`` / ``hnorm`` /
                #     ``norm``) have no backbone rename rule.
                # Re-prefix both forms into the ``mtp.layers.{k}.*`` namespace.
                if key.startswith("model.layers."):
                    state_dict["mtp" + key[len("model") :]] = val
                elif key.startswith("layers."):
                    state_dict["mtp." + key] = val
                else:
                    state_dict[key] = val

        return state_dict

    def _dequantize(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Dequantize FP8 weights.  Handles both `.scale` and `_scale_inv` suffixes."""
        scale_keys_to_remove: list[str] = []
        for key in list(state_dict.keys()):
            weight = state_dict[key]
            # tid2eid is int32 — skip dequantization entirely
            if key.endswith(".tid2eid"):
                continue
            # HF V4 uses `<base>.scale`; V3 used `<base>.weight_scale_inv`
            scale_key = None
            if key.endswith(".weight"):
                base = key[: -len(".weight")]
                if base + ".scale" in state_dict:
                    scale_key = base + ".scale"
                elif key + "_scale_inv" in state_dict:
                    scale_key = key + "_scale_inv"

            if scale_key is not None:
                scale = state_dict[scale_key]
                if self._is_expert_weight_key(key):
                    state_dict[key] = self._dequantize_expert_weight(key, weight, scale)
                else:
                    state_dict[key] = dequantize_from_fp8(weight, scale, dtype=self.dtype, name=key)
                scale_keys_to_remove.append(scale_key)

        for k in scale_keys_to_remove:
            state_dict.pop(k, None)
        return state_dict

    def _aggregate_experts(
        self,
        state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None,
    ) -> dict[str, Any]:
        """Aggregate per-expert weights (w1/w2/w3) into stacked gate_and_up/down tensors."""
        n_experts = self.moe_config.n_routed_experts

        if device_mesh is not None:
            start_expert, end_expert = get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expected_per_rank = end_expert - start_expert
            rank = (
                get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
        else:
            start_expert, end_expert = 0, n_experts
            expected_per_rank = n_experts
            rank = None

        # layer -> {"gate_and_up": {expert_id: {"w1": ..., "w3": ...}}, "down": {expert_id: tensor}}
        by_layer: dict[str, dict] = {}
        out: dict[str, Any] = {}

        for key in list(state_dict.keys()):
            value = state_dict.pop(key)
            m = _EXPERT_PATTERN.match(key)
            if m is None:
                out[key] = value
                continue

            layer_num, expert_num, which = m.group(1), int(m.group(2)), m.group(3)

            if not should_load_expert_for_rank(expert_num, device_mesh, n_experts):
                continue

            if layer_num not in by_layer:
                by_layer[layer_num] = {"gate_and_up": {}, "down": {}}

            if which in ("w1", "w3"):
                if expert_num not in by_layer[layer_num]["gate_and_up"]:
                    by_layer[layer_num]["gate_and_up"][expert_num] = {}
                by_layer[layer_num]["gate_and_up"][expert_num][which] = value
            else:  # w2 = down_proj
                by_layer[layer_num]["down"][expert_num] = value

            # Once all experts for this layer's gate_and_up are ready, stack them.
            # The sub-dict is popped below, so later iterations that touch the
            # same layer (e.g. the paired w2 key) must tolerate its absence.
            gu_layer = by_layer[layer_num].get("gate_and_up")
            if gu_layer is not None:
                all_ready = len(gu_layer) == expected_per_rank and all(
                    isinstance(d, dict) and "w1" in d and "w3" in d for d in gu_layer.values()
                )
                if all_ready:
                    expert_ids = sorted(gu_layer.keys())
                    tensors = []
                    for eid in expert_ids:
                        gate_w = gu_layer[eid]["w1"]
                        up_w = gu_layer[eid]["w3"]
                        if is_dtensor(gate_w):
                            gate_w = gate_w.to_local()
                        if is_dtensor(up_w):
                            up_w = up_w.to_local()
                        tensors.append(torch.cat([gate_w.T, up_w.T], dim=-1))
                    stacked = torch.stack(tensors, dim=0).to(self.dtype)
                    native_key = f"model.layers.{layer_num}.mlp.experts.gate_and_up_projs"
                    out[native_key] = create_dtensor_from_local(stacked, device_mesh, rank)
                    del by_layer[layer_num]["gate_and_up"]

            # Once all experts for this layer's down are ready, stack them.
            down_layer = by_layer[layer_num].get("down")
            if down_layer is not None and len(down_layer) == expected_per_rank:
                expert_ids = sorted(down_layer.keys())
                tensors = []
                for eid in expert_ids:
                    w = down_layer[eid]
                    if is_dtensor(w):
                        w = w.to_local()
                    tensors.append(w.T)
                stacked = torch.stack(tensors, dim=0).to(self.dtype)
                native_key = f"model.layers.{layer_num}.mlp.experts.down_projs"
                out[native_key] = create_dtensor_from_local(stacked, device_mesh, rank)
                del by_layer[layer_num]["down"]

        return out

    def _rename_all(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Apply the HF->internal rename table to every key."""
        return {_rename_hf_key(k): v for k, v in state_dict.items()}

    # ------------------------------------------------------------------
    # to_hf
    # ------------------------------------------------------------------

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert internal state dict to HF V4 format.

        Splits stacked expert weights back to per-expert w1/w2/w3 tensors,
        applies key renaming in reverse, and optionally quantizes to FP8.
        """
        state_dict = self._drop_hash_layer_gate_bias(state_dict, _HashBiasScope.INTERNAL)

        hf_state_dict: dict[str, Any] = {}

        for fqn, tensor in state_dict.items():
            converted = self.convert_single_tensor_to_hf(
                fqn, tensor, exclude_key_regex=exclude_key_regex, quantization=quantization, **kwargs
            )
            for hf_key, hf_val in converted:
                hf_state_dict[hf_key] = hf_val

        # Belt-and-suspenders: re-run the hash-layer bias filter on the HF-side
        # keys in case any intermediate step emitted them in HF format directly
        # (observed in practice during DCP load even after the internal-side drop).
        hf_state_dict = self._drop_hash_layer_gate_bias(hf_state_dict, _HashBiasScope.HF)
        return hf_state_dict

    def _checkpoint_num_hash_layers(self) -> int:
        """Read ``num_hash_layers`` directly from the checkpoint's config.json.

        We cannot rely on ``self.config.num_hash_layers`` alone: a YAML can
        legitimately override the model's hash-layer count to 0 (e.g. to
        disable hash routing in the forward path), but the on-disk checkpoint
        still has its original value and therefore still omits gate.bias for
        the first ``num_hash_layers`` layers.  To decide what to drop at load
        time we must know the checkpoint's own value.
        """
        import json as _json
        import os as _os

        ckpt_path = getattr(self.config, "_name_or_path", None) or getattr(self.config, "name_or_path", None)
        if not ckpt_path:
            return 0
        cfg_json = _os.path.join(ckpt_path, "config.json")
        if not _os.path.isfile(cfg_json):
            return 0
        try:
            with open(cfg_json) as f:
                data = _json.load(f)
        except Exception:
            return 0
        return int(data.get("num_hash_layers", 0) or 0)

    def _drop_hash_layer_gate_bias(self, state_dict: dict[str, Any], scope: "_HashBiasScope") -> dict[str, Any]:
        """The first ``num_hash_layers`` layers use hash-clustering routing and
        their HF checkpoint has no ``ffn.gate.bias`` / ``e_score_correction_bias``
        tensor.  The model side, however, creates the bias parameter uniformly
        for every layer (Automodel's generic Gate always materializes it when
        ``gate_bias_update_factor > 0``).  Drop those bias keys before load so
        DCP does not raise ``Missing key in checkpoint state_dict`` for them.

        ``scope`` selects which key format to match — the pre-rename internal
        form (``model.layers.{i}.mlp.gate.e_score_correction_bias``) or the
        post-rename HF form (``layers.{i}.ffn.gate.bias``).
        """
        # Vision checkpoints include a normal text correction bias even in
        # hash layers because visual tokens in those layers route by scores.
        if int(getattr(self.config, "vision_n_layers", 0) or 0) > 0:
            return state_dict

        # Prefer the checkpoint's own num_hash_layers over the (possibly YAML
        # overridden) model config — we need to match the on-disk layout.
        num_hash_layers = self._checkpoint_num_hash_layers()
        if num_hash_layers <= 0:
            num_hash_layers = int(getattr(self.config, "num_hash_layers", 0) or 0)
        if num_hash_layers <= 0:
            return state_dict
        hash_layer_ids = {str(i) for i in range(num_hash_layers)}
        pat = scope.value
        filtered: dict[str, Any] = {}
        for key, value in state_dict.items():
            m = pat.match(key)
            if m is not None and m.group(1) in hash_layer_ids:
                continue
            filtered[key] = value
        return filtered

    # Internal -> HF name table (inverse of _HF_TO_INTERNAL_RENAMES)
    _INTERNAL_TO_HF_RENAMES: list[tuple[re.Pattern, str]] = [
        (re.compile(r"^model\.vision\.(.+)$"), r"vision.\1"),
        (re.compile(r"^model\.aligner\.(.+)$"), r"aligner.\1"),
        (re.compile(r"^model\.(image_start|image_end|image_newline|image_pad)$"), r"\1"),
        (re.compile(r"^model\.embed_tokens\.(.+)$"), r"embed.\1"),
        (re.compile(r"^model\.norm\.(.+)$"), r"norm.\1"),
        (re.compile(r"^lm_head\.(.+)$"), r"head.\1"),
        (re.compile(r"^model\.layers\.(\d+)\.input_layernorm\.(.+)$"), r"layers.\1.attn_norm.\2"),
        (re.compile(r"^model\.layers\.(\d+)\.post_attention_layernorm\.(.+)$"), r"layers.\1.ffn_norm.\2"),
        (re.compile(r"^model\.layers\.(\d+)\.self_attn\.sinks_param\.weight$"), r"layers.\1.attn.attn_sink"),
        # Indexer reverse: our ``compressor.indexer.*`` -> on-disk ``indexer.*``
        # with the nested compressor un-flattened for projections.
        (
            re.compile(r"^model\.layers\.(\d+)\.self_attn\.compressor\.indexer\.kv_norm\.(.+)$"),
            r"layers.\1.attn.indexer.compressor.norm.\2",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.self_attn\.compressor\.indexer\.ape_param\.weight$"),
            r"layers.\1.attn.indexer.compressor.ape",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.self_attn\.compressor\.indexer\.wgate\.(.+)$"),
            r"layers.\1.attn.indexer.compressor.wgate.\2",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.self_attn\.compressor\.indexer\.wkv\.(.+)$"),
            r"layers.\1.attn.indexer.compressor.wkv.\2",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.self_attn\.compressor\.indexer\.(.+)$"),
            r"layers.\1.attn.indexer.\2",
        ),
        # Outer compressor: our ``kv_norm`` -> on-disk ``norm``.
        (
            re.compile(r"^model\.layers\.(\d+)\.self_attn\.compressor\.kv_norm\.(.+)$"),
            r"layers.\1.attn.compressor.norm.\2",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.self_attn\.compressor\.ape_param\.weight$"),
            r"layers.\1.attn.compressor.ape",
        ),
        (re.compile(r"^model\.layers\.(\d+)\.self_attn\.(.+)$"), r"layers.\1.attn.\2"),
        # Gate (bias correction key mapped back to `bias`)
        (
            re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.e_score_correction_bias$"),
            r"layers.\1.ffn.gate.bias",
        ),
        (re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.(.+)$"), r"layers.\1.ffn.gate.\2"),
        (
            re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.gate_proj\.(.+)$"),
            r"layers.\1.ffn.shared_experts.w1.\2",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.up_proj\.(.+)$"),
            r"layers.\1.ffn.shared_experts.w3.\2",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.mlp\.shared_experts\.down_proj\.(.+)$"),
            r"layers.\1.ffn.shared_experts.w2.\2",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.mlp\.fc1_latent_proj\.(.+)$"),
            r"layers.\1.ffn.fc1_latent_proj.\2",
        ),
        (
            re.compile(r"^model\.layers\.(\d+)\.mlp\.fc2_latent_proj\.(.+)$"),
            r"layers.\1.ffn.fc2_latent_proj.\2",
        ),
        # Reverse of the HC submodule renames above.
        (re.compile(r"^model\.layers\.(\d+)\.attn_hc\.(fn|base|scale)$"), r"layers.\1.hc_attn_\2"),
        (re.compile(r"^model\.layers\.(\d+)\.ffn_hc\.(fn|base|scale)$"), r"layers.\1.hc_ffn_\2"),
        (re.compile(r"^model\.layers\.(\d+)\.hc_head\.hc_(fn|base|scale)$"), r"layers.\1.hc_head_\2"),
        (re.compile(r"^model\.hc_head\.hc_(fn|base|scale)$"), r"hc_head_\1"),
    ]

    def _internal_key_to_hf(self, key: str) -> str:
        for pattern, replacement in self._INTERNAL_TO_HF_RENAMES:
            new_key, n = pattern.subn(replacement, key)
            if n:
                return new_key
        return key

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        # MTP keys (``mtp.layers.{k}.*``) share the same per-block layout as
        # backbone layers, but current DSV4-Flash stores them under native
        # ``mtp.{k}.*`` keys.  Rewrite to an equivalent temporary
        # ``model.layers.{k}.*`` form for splitting / renaming / quantization,
        # then replace the emitted ``layers.{k}.`` prefix with ``mtp.{k}.``.
        mtp_depth: int | None = None
        if fqn.startswith("mtp."):
            rest = fqn[len("mtp.") :]
            m = re.match(r"^layers\.(\d+)\.", rest)
            if m is None:
                return [(fqn, tensor)]
            mtp_depth = int(m.group(1))
            fqn = f"model.layers.{mtp_depth}." + rest[m.end() :]

        quantization = kwargs.get("quantization", False)
        exclude_key_regex = kwargs.get("exclude_key_regex", None)

        # Split stacked gate_and_up_projs into per-expert w1 + w3
        result = self._split_merged_expert(fqn, tensor)

        if exclude_key_regex:
            result = [(k, v) for k, v in result if not re.match(exclude_key_regex, k)]

        # Rename internal keys to HF keys
        result = [(self._internal_key_to_hf(k), v) for k, v in result]

        if mtp_depth is not None:
            # MTP-only modules (``e_proj`` / ``h_proj`` / ``enorm`` /
            # ``hnorm`` / ``norm``) have no generic backbone rename rule, so
            # they can still carry ``model.layers.{k}.`` here.  Normalize both
            # possible temporary prefixes to the checkpoint's native MTP prefix.
            internal_prefix = f"model.layers.{mtp_depth}."
            layer_prefix = f"layers.{mtp_depth}."
            mtp_prefix = f"mtp.{mtp_depth}."
            result = [
                (
                    mtp_prefix + k[len(internal_prefix) :]
                    if k.startswith(internal_prefix)
                    else mtp_prefix + k[len(layer_prefix) :]
                    if k.startswith(layer_prefix)
                    else k,
                    v,
                )
                for k, v in result
            ]

        if quantization:
            quantized = []
            for key, value in result:
                if key.endswith(".weight") and not self._is_non_quantized(key):
                    base = key[: -len(".weight")]
                    if self._is_expert_weight_key(key):
                        if self._checkpoint_expert_quant_layout() is _ExpertQuantLayout.FP8:
                            fp8_val, scale = self._build_fp8_expert_placeholders(value)
                            quantized.append((key, fp8_val))
                            quantized.append((base + ".scale", scale))
                        else:
                            # V4 Flash routed experts are stored as FP4 e2m1 packed two
                            # values per int8 byte, with per-row / 32-col e8m0 scales.
                            # DCP validates shape + dtype against the checkpoint BEFORE
                            # dequantization happens, so the placeholders must match the
                            # on-disk layout exactly.  We emit empty tensors (content is
                            # overwritten by dcp.load) with the packed shape/dtype.
                            int8_val, e8m0_scale = self._build_fp4_expert_placeholders(value)
                            quantized.append((key, int8_val))
                            quantized.append((base + ".scale", e8m0_scale))
                        continue
                    if is_dtensor(value):
                        # Preserve DTensor structure so DCP knows the global shape
                        # and can shard the checkpoint load correctly.  Converting
                        # only the local shard to a plain tensor strips the mesh /
                        # placement metadata and causes a shape mismatch (e.g.
                        # local [128, 4096] vs checkpoint global [512, 4096]).
                        local = value.to_local()
                        local_fp8 = self._empty_or_cast_fp8(local)
                        fp8_val = DTensor.from_local(local_fp8, value.device_mesh, value.placements)
                    else:
                        fp8_val = self._empty_or_cast_fp8(value)
                    scale = self._build_fp8_global_scale_placeholder(value)
                    quantized.append((key, fp8_val))
                    quantized.append((base + ".scale", scale))
                else:
                    quantized.append((key, value))
            return quantized

        return result

    @staticmethod
    def _build_fp4_expert_placeholders(value: Any) -> tuple[Any, Any]:
        """Return (int8 packed weight, float8_e8m0fnu scale) placeholders whose
        shapes / dtypes match the on-disk V4 Flash routed-expert layout.

        The current `value` is the dequantized bf16 tensor with shape [out, in];
        the checkpoint tensor is int8 [out, in // 2] with an e8m0 scale
        [out, in // 32].  DCP only uses these placeholders for shape/dtype
        validation and as the destination buffer — contents are overwritten on
        load, so we build empty tensors instead of re-packing real data.
        """
        if is_dtensor(value):
            local = value.to_local()
            in_dim = local.shape[-1]
            assert in_dim % FP4_COL_BLOCK == 0, f"V4 expert in-dim {in_dim} must be divisible by {FP4_COL_BLOCK}"
            packed = torch.empty(*local.shape[:-1], in_dim // 2, dtype=torch.int8, device=local.device)
            scale = torch.empty(
                *local.shape[:-1],
                in_dim // FP4_COL_BLOCK,
                dtype=torch.float8_e8m0fnu,
                device=local.device,
            )
            packed_d = DTensor.from_local(packed, value.device_mesh, value.placements)
            scale_d = DTensor.from_local(scale, value.device_mesh, value.placements)
            return packed_d, scale_d

        in_dim = value.shape[-1]
        assert in_dim % FP4_COL_BLOCK == 0, f"V4 expert in-dim {in_dim} must be divisible by {FP4_COL_BLOCK}"
        packed = torch.empty(*value.shape[:-1], in_dim // 2, dtype=torch.int8)
        scale = torch.empty(*value.shape[:-1], in_dim // FP4_COL_BLOCK, dtype=torch.float8_e8m0fnu)
        return packed, scale

    @staticmethod
    def _build_fp8_expert_placeholders(value: Any) -> tuple[Any, Any]:
        """Return placeholders for the DeepSeek V4 Base routed-expert FP8 layout."""
        if is_dtensor(value):
            local_fp8 = DeepSeekV4StateDictAdapter._empty_or_cast_fp8(value.to_local())
            fp8_val = DTensor.from_local(local_fp8, value.device_mesh, value.placements)
        else:
            fp8_val = DeepSeekV4StateDictAdapter._empty_or_cast_fp8(value)

        scale = DeepSeekV4StateDictAdapter._build_fp8_dtensor_scale_placeholder(value)
        return fp8_val, scale

    @staticmethod
    def _build_fp8_global_scale_placeholder(value: Any) -> torch.Tensor:
        if is_dtensor(value):
            local = value.to_local()
            return torch.ones(
                DeepSeekV4StateDictAdapter._scale_shape_from_shape(value.shape),
                dtype=torch.float32,
                device=local.device,
            )

        return torch.ones(DeepSeekV4StateDictAdapter._scale_shape_from_shape(value.shape), dtype=torch.float32)

    @staticmethod
    def _build_fp8_dtensor_scale_placeholder(value: Any) -> Any:
        if is_dtensor(value):
            local = value.to_local()
            scale_local = torch.ones(
                DeepSeekV4StateDictAdapter._scale_shape_from_shape(local.shape),
                dtype=torch.float32,
                device=local.device,
            )
            return DTensor.from_local(scale_local, value.device_mesh, value.placements)

        return DeepSeekV4StateDictAdapter._build_fp8_global_scale_placeholder(value)

    @staticmethod
    def _empty_or_cast_fp8(value: torch.Tensor) -> torch.Tensor:
        if value.is_meta:
            return torch.empty(tuple(value.shape), dtype=torch.float8_e4m3fn, device=value.device)
        return value.to(torch.float8_e4m3fn)

    _NON_QUANTIZED_PATTERNS = [
        # The released vision tower and aligner are plain BF16 and have no
        # companion scale tensors in the native checkpoint.
        "vision.",
        "aligner.",
        "attn_norm.weight",
        "ffn_norm.weight",
        "norm.weight",
        "head.weight",
        "embed.weight",
        "ffn.gate.weight",
        "ffn.gate.bias",
        "ffn.gate.tid2eid",
        "attn.q_norm.weight",
        "attn.kv_norm.weight",
        "attn.attn_sink",
        # Compressor / Indexer projections (compress_ratio>0 layers).  Stored
        # as plain BF16 on disk — the released DSV4-Flash safetensors have NO
        # ``.scale`` companion for any of these, so emitting a fabricated
        # ``.scale`` placeholder makes the DCP planner ask for a key that does
        # not exist on disk.  The Indexer's own ``wq_b`` IS FP8 (has a real
        # ``.scale``) so it stays excluded from this list.
        "attn.compressor.wgate.weight",
        "attn.compressor.wkv.weight",
        "attn.compressor.norm.weight",
        "attn.compressor.ape",
        # Indexer (sibling of compressor on disk).  Its nested compressor's
        # projections are BF16, as are the indexer's ``weights_proj`` and APE.
        "attn.indexer.compressor.wgate.weight",
        "attn.indexer.compressor.wkv.weight",
        "attn.indexer.compressor.norm.weight",
        "attn.indexer.compressor.ape",
        "attn.indexer.weights_proj.weight",
        # Latent projections are stored as BF16 in the V4 checkpoint (not FP8).
        "ffn.fc1_latent_proj.weight",
        "ffn.fc2_latent_proj.weight",
    ]

    def _is_non_quantized(self, hf_key: str) -> bool:
        return any(pat in hf_key for pat in self._NON_QUANTIZED_PATTERNS)

    @staticmethod
    def _is_expert_weight_key(key: str) -> bool:
        return "ffn.experts." in key

    def _scale_shape(self, weight: torch.Tensor) -> tuple[int, int]:
        return self._scale_shape_from_shape(weight.shape)

    @staticmethod
    def _scale_shape_from_shape(shape: torch.Size | tuple[int, ...]) -> tuple[int, int]:
        r, c = shape
        return ((r + BLOCK_SIZE - 1) // BLOCK_SIZE, (c + BLOCK_SIZE - 1) // BLOCK_SIZE)

    def _expert_scale_shape(self, weight: torch.Tensor) -> tuple[int, int]:
        """Scale shape for an FP4 routed-expert weight tensor.

        The weight argument should be the *unpacked* tensor (in the model-side
        state dict, experts are already materialized at full dtype), so its
        last dim is the true `in` dim and the scale has `in // 32` columns.
        """
        r, c = weight.shape
        return (r, (c + FP4_COL_BLOCK - 1) // FP4_COL_BLOCK)

    def _dequantize_expert_weight(self, key: str, weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        layout = self._expert_quant_layout_from_tensors(weight, scale)
        if layout is _ExpertQuantLayout.FP4:
            return self._dequantize_expert_fp4(weight, scale, self.dtype)
        return dequantize_from_fp8(weight, scale, dtype=self.dtype, name=key)

    def _expert_quant_layout_from_tensors(self, weight: torch.Tensor, scale: torch.Tensor) -> _ExpertQuantLayout:
        weight_local = weight.to_local() if is_dtensor(weight) else weight
        scale_local = scale.to_local() if is_dtensor(scale) else scale

        if weight_local.dtype == torch.int8:
            return _ExpertQuantLayout.FP4

        if weight_local.dtype == torch.float8_e4m3fn:
            return _ExpertQuantLayout.FP8

        if tuple(scale_local.shape) == self._scale_shape(weight_local):
            return _ExpertQuantLayout.FP8

        return _ExpertQuantLayout.FP4

    def _checkpoint_expert_quant_layout(self) -> _ExpertQuantLayout:
        override = os.environ.get("NEMO_AUTOMODEL_DSV4_EXPERT_LAYOUT")
        if override:
            normalized = override.lower()
            if normalized in {"fp4", "mxfp4", "flash"}:
                return _ExpertQuantLayout.FP4
            if normalized in {"fp8", "base"}:
                return _ExpertQuantLayout.FP8
            raise ValueError("NEMO_AUTOMODEL_DSV4_EXPERT_LAYOUT must be one of: fp4, mxfp4, flash, fp8, base")

        if self._checkpoint_expert_quant_layout_cache is not None:
            return self._checkpoint_expert_quant_layout_cache

        self._checkpoint_expert_quant_layout_cache = self._detect_checkpoint_expert_quant_layout()
        return self._checkpoint_expert_quant_layout_cache

    def _detect_checkpoint_expert_quant_layout(self) -> _ExpertQuantLayout:
        ckpt_path = getattr(self.config, "_name_or_path", None) or getattr(self.config, "name_or_path", None)
        if not ckpt_path:
            return _ExpertQuantLayout.FP4

        path = Path(ckpt_path)
        if not path.is_dir():
            return _ExpertQuantLayout.FP4

        try:
            from safetensors import safe_open
        except ImportError:
            return _ExpertQuantLayout.FP4

        for sf_path in sorted(path.glob("*.safetensors")):
            with safe_open(sf_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if not _EXPERT_PATTERN.match(key):
                        continue
                    weight = handle.get_tensor(key)
                    return _ExpertQuantLayout.FP4 if weight.dtype == torch.int8 else _ExpertQuantLayout.FP8

        return _ExpertQuantLayout.FP4

    @staticmethod
    def _dequantize_expert_fp4(weight: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Unpack FP4 e2m1 packed-int8 weight and apply the per-row / 32-col e8m0 scale.

        Packed layout: `weight.int8` holds two FP4 values per byte — the low nibble
        at even column index, the high nibble at the following odd column — so the
        logical shape is `[out, weight.size(-1) * 2]`.
        """
        weight_local = weight.to_local() if is_dtensor(weight) else weight
        scale_local = scale.to_local() if is_dtensor(scale) else scale

        # Step 1: unpack two FP4 values from each byte.
        weight_u8 = weight_local.contiguous().view(torch.uint8)
        low = (weight_u8 & 0x0F).long()
        high = ((weight_u8 >> 4) & 0x0F).long()
        # Interleave (low, high) per byte so column indices match the original layout.
        table = _FP4_E2M1_TABLE.to(weight_u8.device)
        fp4_vals = torch.stack([table[low], table[high]], dim=-1).flatten(-2)  # [out, in]

        # Step 2: decode e8m0 scale to fp32. e8m0 stores 2^(e-127), or 0 when e==0.
        # A simple .to(torch.float32) works when PyTorch supports the e8m0 dtype;
        # fall back to the explicit formula otherwise.
        scale_u8 = scale_local.contiguous().view(torch.uint8).int()
        scale_f32 = torch.where(
            scale_u8 == 0,
            torch.zeros_like(scale_u8, dtype=torch.float32),
            torch.pow(2.0, (scale_u8 - 127).float()),
        )

        # Step 3: broadcast scale across the 32 columns it covers.
        scale_expanded = scale_f32.repeat_interleave(FP4_COL_BLOCK, dim=-1)
        scale_expanded = scale_expanded[..., : fp4_vals.shape[-1]]
        return (fp4_vals * scale_expanded).to(dtype)

    def _split_merged_expert(self, fqn: str, tensor: Any) -> list[tuple[str, Any]]:
        """Inverse of expert aggregation: split gate_and_up/down stacks into per-expert keys.

        Handles DTensor inputs (EP-sharded) by working on the local shard only,
        emitting keys only for the experts owned by the current rank.
        """
        gate_up_pat = re.compile(r"^(model\.layers\.(\d+)\.mlp\.experts)\.gate_and_up_projs$")
        down_pat = re.compile(r"^(model\.layers\.(\d+)\.mlp\.experts)\.down_projs$")

        m = gate_up_pat.match(fqn)
        if m:
            layer_num = m.group(2)
            n_total = self.moe_config.n_routed_experts
            expert_tensors, expert_ids = split_experts_weights_dtensor_aware(tensor, n_total)
            result = []
            for t, eid in zip(expert_tensors, expert_ids):
                inter_dim = t.shape[-1] // 2
                # t is [hidden_dim, 2*inter_dim]. If the expert tensor is
                # sharded on hidden dim, keep it as a DTensor so DCP sees the
                # checkpoint's global expert shape.
                gate_t, up_t = t.split(inter_dim, dim=-1)
                result.append((f"layers.{layer_num}.ffn.experts.{eid}.w1.weight", gate_t.T))
                result.append((f"layers.{layer_num}.ffn.experts.{eid}.w3.weight", up_t.T))
            return result

        m = down_pat.match(fqn)
        if m:
            layer_num = m.group(2)
            n_total = self.moe_config.n_routed_experts
            expert_tensors, expert_ids = split_experts_weights_dtensor_aware(tensor, n_total)
            result = []
            for t, eid in zip(expert_tensors, expert_ids):
                result.append((f"layers.{layer_num}.ffn.experts.{eid}.w2.weight", t.T))
            return result

        return [(fqn, tensor)]
