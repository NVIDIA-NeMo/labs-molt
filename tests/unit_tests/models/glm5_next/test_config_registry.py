# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from nemo_automodel._transformers.registry import (
    _CUSTOM_CONFIG_REGISTRATIONS,
    MODEL_ARCH_MAPPING,
    resolve_custom_config_cls,
)
from nemo_automodel.components.models.glm5_next.config import Glm5NextConfig, Glm5NextTextConfig
from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration


def test_checkpoint_style_nested_config_resolves_hybrid_patterns():
    config = Glm5NextConfig.from_dict(
        {
            "model_type": "glm5_next",
            "architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": {
                "num_hidden_layers": 4,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "q_lora_rank": 8,
                "qk_rope_head_dim": 0,
                "qk_nope_head_dim": 4,
                "v_head_dim": 4,
                "index_topk": 8,
                "index_kpool": 4,
                "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
                "linear_attn_config": {
                    "head_dim": 4,
                    "num_heads": 2,
                    "short_conv_kernel_size": 2,
                    "gate_lower_bound": -5.0,
                },
            },
            "vision_config": {"depth": 1},
        }
    )

    assert config.text_config.layer_types[-1] == "deepseek_sparse_attention"
    assert config.text_config.linear_lower_bound == -5.0
    assert config.text_config.linear_conv_kernel_dim == 2


def test_config_rejects_rope_and_invalid_kpool_contracts():
    with pytest.raises(ValueError, match="NoPE"):
        Glm5NextTextConfig(qk_rope_head_dim=16)
    with pytest.raises(ValueError, match="divide index_topk"):
        Glm5NextTextConfig(index_topk=7, index_kpool=4)


def test_registry_resolves_native_config_and_model():
    assert MODEL_ARCH_MAPPING["Glm5NextForConditionalGeneration"] == (
        "nemo_automodel.components.models.glm5_next.model",
        "Glm5NextForConditionalGeneration",
    )
    assert _CUSTOM_CONFIG_REGISTRATIONS["glm5_next"] == (
        "nemo_automodel.components.models.glm5_next.config",
        "Glm5NextConfig",
    )
    assert resolve_custom_config_cls("glm5_next") is Glm5NextConfig
    capabilities = Glm5NextForConditionalGeneration.ModelCapabilities()
    assert capabilities.supports_ep and capabilities.supports_cp and capabilities.supports_thd
    assert not capabilities.supports_tp and not capabilities.supports_pp
