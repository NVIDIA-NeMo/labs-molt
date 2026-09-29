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

from typing import Any

import torch


class NemotronHybridCache:
    """Hybrid KV cache for the NemotronH architecture (attention + Mamba2 layers).

    Attention layers accumulate key/value tensors (growing sequence dimension).
    Mamba2 layers maintain fixed-size conv_state and ssm_state tensors.
    MLP/MoE layers have no caching.

    Modeled after ``FalconHybridMambaAttentionDynamicCache`` from transformers.
    """

    is_compileable = False

    def __init__(self, config, batch_size: int, dtype: torch.dtype, device: torch.device):
        self.has_previous_state = False
        self.conv_kernel_size = config.conv_kernel

        intermediate_size = config.mamba_num_heads * config.mamba_head_dim
        conv_dim = intermediate_size + 2 * config.n_groups * config.ssm_state_size

        self.conv_states = {
            i: torch.zeros(
                batch_size,
                conv_dim,
                self.conv_kernel_size,
                device=device,
                dtype=dtype,
            )
            for i in range(config.num_hidden_layers)
        }
        self.ssm_states = {
            i: torch.zeros(
                batch_size,
                config.mamba_num_heads,
                config.mamba_head_dim,
                config.ssm_state_size,
                device=device,
                dtype=dtype,
            )
            for i in range(config.num_hidden_layers)
        }

        # Identify first attention layer for get_seq_length()
        self._attention_layer_idx = None
        for i, block_type in enumerate(config.layers_block_type):
            if block_type in ("attention", "full_attention"):
                self._attention_layer_idx = i
                break

        self.key_cache: list[torch.Tensor] = []
        self.value_cache: list[torch.Tensor] = []

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attention KV cache: append new K/V and return accumulated tensors."""
        if len(self.key_cache) <= layer_idx:
            # Fill skipped layers with empty lists
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append([])
                self.value_cache.append([])
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif len(self.key_cache[layer_idx]) == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def update_conv_state(
        self,
        layer_idx: int,
        new_conv_state: torch.Tensor,
        cache_position: torch.LongTensor,
    ) -> torch.Tensor:
        """Update Mamba conv state: full overwrite (prefill) or roll+update (decode)."""
        conv_state = self.conv_states[layer_idx]
        cache_position = cache_position.clamp(0, self.conv_kernel_size - 1)

        conv_state = conv_state.roll(shifts=-1, dims=-1)
        if len(cache_position) > 1:
            conv_state[:, :, :] = new_conv_state.to(conv_state.device)
        else:
            conv_state[:, :, -1] = new_conv_state[:, :, -1].to(conv_state.device)
        self.conv_states[layer_idx].zero_()
        self.conv_states[layer_idx] += conv_state
        return self.conv_states[layer_idx]

    def get_seq_length(self, layer_idx: int | None = None) -> int:
        """Return attention KV cache sequence length."""
        idx = layer_idx if layer_idx is not None else self._attention_layer_idx
        if idx is None or idx >= len(self.key_cache):
            return 0
        kc = self.key_cache[idx]
        if isinstance(kc, list) and len(kc) == 0:
            return 0
        return kc.shape[-2]

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorder all caches for beam search."""
        for layer_idx in self.conv_states:
            self.conv_states[layer_idx] = self.conv_states[layer_idx][beam_idx]
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx][beam_idx]
        for i in range(len(self.key_cache)):
            kc = self.key_cache[i]
            if not (isinstance(kc, list) and len(kc) == 0):
                self.key_cache[i] = kc[beam_idx]
                self.value_cache[i] = self.value_cache[i][beam_idx]
