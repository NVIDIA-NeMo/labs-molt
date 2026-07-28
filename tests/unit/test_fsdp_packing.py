# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn

from molt.trainer.fsdp.packing import is_automodel_custom_model


def test_hf_checkpointing_mixin_does_not_mark_hf_model_as_custom():
    hf_mixin = type("HFCheckpointingMixin", (), {"__module__": "nemo_automodel.components.models.common"})
    hf_model = type("Qwen3ForCausalLM", (nn.Module,), {"__module__": "transformers.models.qwen3"})
    wrapped_model = type("Qwen3ForCausalLM", (hf_mixin, hf_model), {"__module__": hf_model.__module__})

    assert not is_automodel_custom_model(wrapped_model())
