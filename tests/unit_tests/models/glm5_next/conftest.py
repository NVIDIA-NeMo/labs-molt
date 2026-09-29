# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.glm5_next.config import (
    Glm5NextConfig,
    Glm5NextTextConfig,
    Glm5NextVisionConfig,
)
from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration


def tiny_glm5_next_config() -> Glm5NextConfig:
    """Build a four-layer hybrid config with one sparse MoE/DSA layer."""
    text = Glm5NextTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_rope_head_dim=0,
        qk_nope_head_dim=4,
        v_head_dim=4,
        index_topk=4,
        index_head_dim=4,
        index_n_heads=2,
        index_kpool=2,
        linear_head_dim=4,
        linear_num_heads=2,
        linear_conv_kernel_dim=2,
        hc_mult=2,
        hc_sinkhorn_iters=3,
        mlp_layer_types=["dense", "dense", "dense", "sparse"],
        layer_types=["linear_attention", "linear_attention", "linear_attention", "deepseek_sparse_attention"],
        pad_token_id=0,
        torch_dtype="float32",
    )
    vision = Glm5NextVisionConfig(
        depth=1,
        hidden_size=8,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=16,
        intermediate_size=16,
        projection_intermediate_size=32,
        torch_dtype="float32",
    )
    return Glm5NextConfig(text_config=text, vision_config=vision, image_token_id=63, pad_token_id=0)


def tiny_backend(*, adapter: bool = True) -> BackendConfig:
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=adapter,
    )


def tiny_glm5_next_model() -> Glm5NextForConditionalGeneration:
    model = Glm5NextForConditionalGeneration(tiny_glm5_next_config(), backend=tiny_backend())
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model
