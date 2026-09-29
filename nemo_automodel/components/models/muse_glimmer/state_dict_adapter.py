# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Hugging Face checkpoint adapter for legacy and canonical MuseGlimmer exports."""

from __future__ import annotations

import re
from typing import Any

import torch

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.muse_glimmer.config import MuseGlimmerConfig


class MuseGlimmerStateDictAdapter(StateDictAdapter):
    """Map canonical nested MuseGlimmer checkpoint keys to the native legacy module tree."""

    _supports_low_memory_dcp_load = True

    def __init__(self, config: MuseGlimmerConfig) -> None:
        self.config = config
        self.uses_canonical_layout = "MuseGlimmerForConditionalGeneration" in (config.architectures or [])

    @staticmethod
    def _canonical_to_native_key(key: str) -> str:
        if key.startswith("model.language_model."):
            key = "model." + key[len("model.language_model.") :]
            key = key.replace(".post_attention_layernorm.", ".post_attn_norm.")
            key = key.replace(".pre_feedforward_layernorm.", ".post_attention_layernorm.")
            key = key.replace(".post_feedforward_layernorm.", ".post_ffn_norm.")
            key = key.replace(".self_attn.gate_proj.", ".self_attn.output_gate_proj.")
            return key

        exact = {
            "model.vision_tower.patch_embedder.patch_embedding.weight": "model.vision_encoder.conv1_linear.weight",
            "model.vision_tower.patch_embedder.position_embedding_table.weight": (
                "model.vision_encoder.positional_embedding_vlm"
            ),
            "model.vision_adapter.fc1.weight": "model.vision_adapter.c_fc.weight",
            "model.vision_adapter.fc2.weight": "model.vision_adapter.c_proj.weight",
        }
        if key in exact:
            return exact[key]

        if key.startswith("model.vision_tower.layers."):
            key = "model.vision_encoder.transformer." + key[len("model.vision_tower.layers.") :]
            key = key.replace(".attn.proj.", ".attn.o_proj.")
            key = key.replace(".mlp.fc1.", ".mlp.c_fc.")
            key = key.replace(".mlp.fc2.", ".mlp.c_proj.")
            key = key.replace(".norm1.", ".ln_1.")
            key = key.replace(".norm2.", ".ln_2.")
            return key
        if key.startswith("model.vision_tower."):
            return "model.vision_encoder." + key[len("model.vision_tower.") :]
        return key

    @staticmethod
    def _native_to_canonical_key(key: str) -> str | None:
        if key in {"model.rotary_emb.freqs", "model.vision_encoder.rotary_emb.inv_freq"}:
            return None
        if key.startswith("model.layers.") or key.startswith("model.embed_tokens.") or key.startswith("model.norm."):
            key = "model.language_model." + key[len("model.") :]
            key = key.replace(".post_attention_layernorm.", ".pre_feedforward_layernorm.")
            key = key.replace(".post_attn_norm.", ".post_attention_layernorm.")
            key = key.replace(".post_ffn_norm.", ".post_feedforward_layernorm.")
            key = key.replace(".self_attn.output_gate_proj.", ".self_attn.gate_proj.")
            return key

        exact = {
            "model.vision_encoder.conv1_linear.weight": "model.vision_tower.patch_embedder.patch_embedding.weight",
            "model.vision_encoder.positional_embedding_vlm": (
                "model.vision_tower.patch_embedder.position_embedding_table.weight"
            ),
            "model.vision_adapter.c_fc.weight": "model.vision_adapter.fc1.weight",
            "model.vision_adapter.c_proj.weight": "model.vision_adapter.fc2.weight",
        }
        if key in exact:
            return exact[key]

        if key.startswith("model.vision_encoder.transformer."):
            key = "model.vision_tower.layers." + key[len("model.vision_encoder.transformer.") :]
            key = key.replace(".attn.o_proj.", ".attn.proj.")
            key = key.replace(".mlp.c_fc.", ".mlp.fc1.")
            key = key.replace(".mlp.c_proj.", ".mlp.fc2.")
            key = key.replace(".ln_1.", ".norm1.")
            key = key.replace(".ln_2.", ".norm2.")
            return key
        if key.startswith("model.vision_encoder."):
            return "model.vision_tower." + key[len("model.vision_encoder.") :]
        return key

    def from_hf(self, hf_state_dict: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        del kwargs
        if not any(key.startswith(("model.language_model.", "model.vision_tower.")) for key in hf_state_dict):
            return dict(hf_state_dict)

        state_dict = {self._canonical_to_native_key(key): value for key, value in hf_state_dict.items()}
        if "model.rotary_emb.freqs" not in state_dict:
            positions = torch.arange(0, self.config.head_dim, 2, dtype=torch.float32)
            state_dict["model.rotary_emb.freqs"] = 1.0 / (self.config.rope_theta ** (positions / self.config.head_dim))
        vision_head_dim = self.config.vision_config.hidden_size // self.config.vision_config.num_attention_heads
        vision_spatial_dim = vision_head_dim // 2
        vision_theta = self.config.vision_config.rope_parameters["rope_theta"]
        vision_positions = torch.arange(0, vision_spatial_dim, 2, dtype=torch.float32)
        state_dict["model.vision_encoder.rotary_emb.inv_freq"] = 1.0 / (
            vision_theta ** (vision_positions / vision_spatial_dim)
        )
        return state_dict

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        converted: dict[str, Any] = {}
        for key, value in state_dict.items():
            for mapped_key, mapped_value in self.convert_single_tensor_to_hf(
                key,
                value,
                exclude_key_regex=exclude_key_regex,
                **kwargs,
            ):
                converted[mapped_key] = mapped_value
        return converted

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Convert one native MuseGlimmer tensor to its Hugging Face checkpoint representation."""
        mapped_key = self._native_to_canonical_key(fqn) if self.uses_canonical_layout else fqn
        exclude_key_regex = kwargs.get("exclude_key_regex")
        if mapped_key is None or (exclude_key_regex is not None and re.search(exclude_key_regex, mapped_key)):
            return []
        return [(mapped_key, tensor)]
