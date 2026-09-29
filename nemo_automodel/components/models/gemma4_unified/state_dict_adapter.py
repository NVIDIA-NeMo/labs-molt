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

"""State-dict key translation for Hugging Face-native Gemma4 Unified models."""

import re
from typing import TYPE_CHECKING, Any

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

# Model FQN prefix -> published Hugging Face checkpoint FQN prefix.
_HF_KEY_RENAMES = {
    "embed_vision.patch_ln1": "vision_embedder.patch_ln1",
    "embed_vision.patch_dense": "vision_embedder.patch_dense",
    "embed_vision.patch_ln2": "vision_embedder.patch_ln2",
    "embed_vision.pos_embedding": "vision_embedder.pos_embedding",
    "embed_vision.pos_norm": "vision_embedder.pos_norm",
    "embed_vision.multimodal_embedder.embedding_projection": "embed_vision.embedding_projection",
}
_MODEL_KEY_RENAMES = {hf_key: model_key for model_key, hf_key in _HF_KEY_RENAMES.items()}


def _rename_key(key: str, renames: dict[str, str]) -> str:
    """Rename one state-dict key while preserving an optional ``model.`` prefix."""
    prefix = "model." if key.startswith("model.") else ""
    unprefixed_key = key[len(prefix) :]
    for source_key, target_key in renames.items():
        if unprefixed_key == source_key or unprefixed_key.startswith(f"{source_key}."):
            return f"{prefix}{target_key}{unprefixed_key[len(source_key) :]}"
    return key


def _rename_state_dict(state_dict: dict[str, Any], renames: dict[str, str]) -> dict[str, Any]:
    """Return a state dict with renamed keys and unchanged tensor values.

    Args:
        state_dict: State mapping whose tensor values retain their original shape,
            dtype, device, and storage.
        renames: Source-prefix to target-prefix mapping.

    Returns:
        A new mapping whose values alias the input values exactly.

    Raises:
        ValueError: If two source keys map to the same output key.
    """
    renamed_state_dict: dict[str, Any] = {}
    for key, value in state_dict.items():
        renamed_key = _rename_key(key, renames)
        if renamed_key in renamed_state_dict:
            raise ValueError(f"Gemma4 Unified key collision for {renamed_key!r}")
        renamed_state_dict[renamed_key] = value
    return renamed_state_dict


class Gemma4UnifiedStateDictAdapter(StateDictAdapter):
    """Translate Gemma4 Unified keys between model and published HF names."""

    _supports_low_memory_dcp_load = True

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Rename model FQNs to published Hugging Face FQNs.

        Args:
            state_dict: Native model state mapping. Tensor values may have arbitrary
                rank and retain their exact shape, dtype, device, and storage.
            exclude_key_regex: Optional regular expression for keys to omit.
            **kwargs: Unused adapter-interface arguments.

        Returns:
            A new Hugging Face-keyed mapping whose tensor values alias the inputs.
        """
        hf_state_dict: dict[str, Any] = {}
        for fqn, value in state_dict.items():
            for hf_fqn, hf_value in self.convert_single_tensor_to_hf(fqn, value, exclude_key_regex=exclude_key_regex):
                if hf_fqn in hf_state_dict:
                    raise ValueError(f"Gemma4 Unified key collision for {hf_fqn!r}")
                hf_state_dict[hf_fqn] = hf_value
        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: "DeviceMesh | None" = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Rename published Hugging Face FQNs back to model FQNs.

        Args:
            hf_state_dict: Hugging Face state mapping. Tensor values may have
                arbitrary rank and retain their exact shape, dtype, device, and storage.
            device_mesh: Unused; Gemma4 Unified renames keys without changing tensor placement.
            **kwargs: Unused adapter-interface arguments.

        Returns:
            A new model-keyed mapping whose tensor values alias the inputs.
        """
        return _rename_state_dict(hf_state_dict, _MODEL_KEY_RENAMES)

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs: Any) -> list[tuple[str, Any]]:
        """Rename one model tensor without changing its layout or storage.

        Args:
            fqn: Fully qualified native model tensor name.
            tensor: Tensor-like value of arbitrary rank. Its shape, dtype, device,
                strides, and storage are preserved exactly.
            **kwargs: Adapter-interface arguments. ``exclude_key_regex`` may contain
                an optional regular expression for keys to omit.

        Returns:
            An empty list when excluded, otherwise one ``(hf_fqn, tensor)`` pair
            whose tensor aliases the input value.
        """
        exclude_key_regex = kwargs.get("exclude_key_regex")
        if exclude_key_regex is not None and re.search(exclude_key_regex, fqn):
            return []
        return [(_rename_key(fqn, _HF_KEY_RENAMES), tensor)]
