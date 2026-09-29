# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Shared structural model traversal utilities."""

from collections.abc import Iterator

from torch import nn

_TEXT_MODULE_ATTRS = ("language_model", "text_model", "text_decoder")


def iter_transformer_and_mtp_blocks(model: nn.Module) -> Iterator[tuple[nn.Module, str, nn.Module]]:
    """Yield transformer and MTP blocks without depending on a recipe or component.

    Args:
        model: Model root containing a transformer layer collection and optional
            multi-token-prediction layers.

    Yields:
        Tuples containing the parent layer collection, child name, and block.
    """
    inner_model = getattr(model, "model", None)
    inner = inner_model if inner_model is not None else model
    text_model = inner
    for attribute in _TEXT_MODULE_ATTRS:
        candidate = getattr(inner, attribute, None)
        if candidate is not None:
            text_model = candidate
            break

    layers = getattr(text_model, "layers", None)
    if layers is not None:
        for layer_id, block in layers.named_children():
            yield layers, layer_id, block

    mtp_layers = getattr(getattr(model, "mtp", None), "layers", None)
    if mtp_layers is not None:
        for layer_id, block in mtp_layers.named_children():
            yield mtp_layers, layer_id, block
