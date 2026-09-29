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

"""State-dict conversion for Qwen3.8-Flash-Next and its owner-sharded Engram table.

Dense, grouped-MoE, shared-expert, and GatedDeltaNet parameters reuse the
Qwen3.5-MoE checkpoint layouts.  The PLE table is special: the checkpoint
stores ``split_ngram_parts`` physical shard tensors while the native module
registers one contiguous rank-local row range.  ``to_hf`` exposes the local
range as narrow views of the native parameter, so the checkpoint reader writes
directly into final model storage and the 51.2B-parameter global table is
never materialized.  MTP checkpoint keys are ignored in both directions
because the SFT target does not construct MTP.
"""

from __future__ import annotations

import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.state_dict_adapter import Qwen3_5MoeStateDictAdapter
from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from nemo_automodel.components.models.qwen3_8_flash_next.engram import Qwen3_8_FlashNextOwnerShardedEmbedding
from nemo_automodel.components.moe.layers import MoEConfig


class Qwen3_8_FlashNextStateDictAdapter(Qwen3_5MoeStateDictAdapter):
    """Convert Qwen3.8-Flash-Next checkpoints without gathering the global PLE table."""

    _supports_low_memory_dcp_load = True

    def __init__(
        self,
        config: Qwen3_8_FlashNextTextConfig,
        moe_config: MoEConfig,
        backend: BackendConfig,
        engram_table: Qwen3_8_FlashNextOwnerShardedEmbedding,
        dtype: torch.dtype = torch.bfloat16,
        pretrained_model_name_or_path: str | None = None,
    ) -> None:
        super().__init__(
            config=config,
            moe_config=moe_config,
            backend=backend,
            dtype=dtype,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            mtp_expert_hf_layout="grouped",
            text_only=False,
        )
        self.engram_table = engram_table
        self.split_ngram_parts = int(config.split_ngram_parts)
        ple_layer_idx = int(config.ple_layer_ids[0]) - 1
        self._table_native_key = f"model.language_model.layers.{ple_layer_idx}.ple.ple_embedding.ngram_embedding.weight"
        self._table_hf_prefix = self._table_native_key.removesuffix(".weight")
        self._table_hf_key_pattern = re.compile(rf"{re.escape(self._table_hf_prefix)}\.shard_(\d+)\.weight")

        self._rows_per_checkpoint_shard = engram_table.num_embeddings // self.split_ngram_parts
        start, end = int(engram_table.global_row_start), int(engram_table.global_row_end)
        if start % self._rows_per_checkpoint_shard or end % self._rows_per_checkpoint_shard:
            raise ValueError(
                "The owner-sharded Engram row range must align to complete physical checkpoint shards; "
                f"got [{start}, {end}) with {self._rows_per_checkpoint_shard} rows per shard."
            )
        self._first_local_checkpoint_shard = start // self._rows_per_checkpoint_shard
        self._end_local_checkpoint_shard = end // self._rows_per_checkpoint_shard
        self._view_loaded_native_keys: set[str] = set()

    @property
    def view_loaded_native_keys(self) -> set[str]:
        """Return native parameters already populated through checkpoint views."""
        return set(self._view_loaded_native_keys)

    def get_hf_state_dict_keys(self, state_dict: dict[str, Any]) -> list[str]:
        """Return the rank-independent global HF key set without gathering PLE.

        Consolidated checkpoint planning requires the same global key list on
        every rank, so the one local PLE weight is replaced by all
        ``split_ngram_parts`` physical shard names.  No data is touched.
        """
        keys: list[str] = []
        for fqn, tensor in state_dict.items():
            if fqn.startswith("mtp.") or re.match(r".*_extra_state.*", fqn):
                continue
            if fqn == self._table_native_key:
                keys.extend(
                    f"{self._table_hf_prefix}.shard_{shard_idx}.weight" for shard_idx in range(self.split_ngram_parts)
                )
                continue
            keys.extend(
                key
                for key, _ in self.convert_single_tensor_to_hf(
                    fqn,
                    tensor,
                    exclude_key_regex=r".*_extra_state.*",
                    quantization=False,
                )
            )
        return keys

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Convert one native tensor, specializing the PLE and MTP entries."""
        if fqn.startswith("mtp."):
            return []
        if fqn != self._table_native_key:
            return super().convert_single_tensor_to_hf(fqn, tensor, **kwargs)

        # Split the local table rows into narrow views, one per complete
        # physical checkpoint shard owned by this rank.
        local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
        exclude_key_regex = kwargs.get("exclude_key_regex")
        views: list[tuple[str, torch.Tensor]] = []
        for shard_idx in range(self._first_local_checkpoint_shard, self._end_local_checkpoint_shard):
            key = f"{self._table_hf_prefix}.shard_{shard_idx}.weight"
            if exclude_key_regex and re.match(exclude_key_regex, key):
                continue
            local_row_start = shard_idx * self._rows_per_checkpoint_shard - int(self.engram_table.global_row_start)
            views.append((key, local_tensor.narrow(0, local_row_start, self._rows_per_checkpoint_shard)))
        return views

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Drop PLE and MTP checkpoint keys; the table was written through views.

        The PLE entries returned by :meth:`to_hf` alias the native parameter,
        so by the time DCP hands them back the table is already populated.
        ``view_loaded_native_keys`` records the native key as loaded.
        """
        self._view_loaded_native_keys.clear()
        filtered_state_dict: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            if key.startswith("mtp."):
                continue
            if self._table_hf_key_pattern.fullmatch(key):
                self._view_loaded_native_keys.add(self._table_native_key)
                continue
            filtered_state_dict[key] = value
        return super().from_hf(filtered_state_dict, device_mesh=device_mesh, **kwargs)
