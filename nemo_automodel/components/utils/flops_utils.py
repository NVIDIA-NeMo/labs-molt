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

import warnings
from typing import Any, Callable


def calculate_mfu(
    tflops: float,
    world_size: int,
    time_seconds: float,
    reference_mfu: float | None = None,
) -> float:
    """Calculate Model FLOPs Utilization (MFU).

    Args:
        tflops: Total TFLOPs across all devices for the measured step.
        world_size: Total number of GPUs.
        time_seconds: Time taken for computation.
        reference_mfu: Peak TFLOPs/s per device for the training precision. The
            legacy default is the H100 dense-FP8 peak.

    Returns:
        MFU as a percentage.
    """
    if reference_mfu is None:
        warnings.warn(
            "Omitting reference_mfu is deprecated; pass the peak TFLOPs/s for the training precision.",
            FutureWarning,
            stacklevel=2,
        )
        reference_mfu = 1979.0
    mfu = tflops / (world_size * time_seconds)
    mfu = mfu / reference_mfu
    return mfu * 100


def gpt3_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for GPT3 family - accepts either AutoConfig or normalized config"""

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    hs = config.hidden_size
    layers = config.num_hidden_layers
    vocab_size = config.vocab_size
    causal_self_attn = True

    return (24 * gbs * seq_len * hs * hs + 4 * gbs * seq_len * seq_len * hs * (0.5 if causal_self_attn else 1)) * (
        3 * layers
    ) + (6 * gbs * seq_len * hs * vocab_size)


def llama2_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for llama2 family - accepts either AutoConfig or normalized config"""

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size
    causal_self_attn = True

    return (
        gbs
        * seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (18 * ffn_hs / hs)
            + (12 * seq_len / hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (layers * hs))
        )
    )


def nemotron_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for nemotron family - accepts either AutoConfig or normalized config"""

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size
    causal_self_attn = True

    return (
        gbs
        * seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (12 * ffn_hs / hs)
            + (12 * seq_len / hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (layers * hs))
        )
    )


def mixtral_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for mixtral family - accepts either AutoConfig or normalized config"""

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 2
    causal_self_attn = True

    return (
        gbs
        * seq_len
        * layers
        * hs
        * hs
        * (
            12
            + (12 * query_groups / attention_heads)
            + (18 * moe_router_topk * ffn_hs / hs)
            + (12 * seq_len / hs) * (0.5 if causal_self_attn else 1)
            + (6 * vocab_size / (layers * hs))
        )
    )


def qwen3_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for Qwen3 family - accepts either AutoConfig or normalized config"""

    # For VL composite configs, use the text_config sub-config
    if hasattr(config, "text_config") and not hasattr(config, "num_hidden_layers"):
        config = config.text_config

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    vocab_size = config.vocab_size
    # Calculate head_dim if not present (for Qwen2) or use directly (for Qwen3)
    head_dim = config.head_dim if hasattr(config, "head_dim") else (hs // attention_heads)
    query_projection_to_hidden_size_ratio = (head_dim * attention_heads) / hs

    # MoE fields - Qwen3 uses "moe_topk" if present, else dense (1)
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 1
    moe_ffn_hidden_size = (
        config.moe_intermediate_size if hasattr(config, "moe_intermediate_size") else config.intermediate_size
    )

    causal_self_attn = True
    hidden_size = hs
    gated_linear_multiplier = 2

    # attention flops for GQA
    attention_flops = (
        3
        * 2
        * gbs
        * layers
        * seq_len
        * hidden_size
        * hidden_size
        * query_projection_to_hidden_size_ratio
        * (
            (query_groups / attention_heads * 2 + 1)  # QKV gemm
            + (seq_len / hidden_size * 2 * (0.5 if causal_self_attn else 1))  # attention
            + 1  # attention proj gemm
        )
    )

    # mlp flops
    mlp_flops = (
        3
        * 2
        * gbs
        * layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * (moe_ffn_hidden_size * moe_router_topk)  # MoE layers
    )

    # vocab flops
    vocab_flops = 3 * 2 * gbs * seq_len * hidden_size * vocab_size

    return attention_flops + mlp_flops + vocab_flops


def bert_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for BERT family - accepts either AutoConfig or normalized config"""

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 512

    layers = config.num_hidden_layers
    hs = config.hidden_size
    vocab_size = config.vocab_size

    return 72 * gbs * layers * seq_len * hs * hs * (1 + (seq_len / (6 * hs)) + (vocab_size / (12 * hs * layers)))


def transformer_flops(config, gbs=1, seq_len=None):
    """Calculate FLOPs for a standard Transformer model - accepts either AutoConfig or normalized config.
    Note: This does not cover encoder-decoder models.
    """
    batch_size = gbs
    if seq_len is None:
        seq_length = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048
    else:
        seq_length = seq_len

    hidden_size = config.hidden_size
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    ffn_hidden_size = config.intermediate_size
    vocab_size = config.vocab_size

    # Handle optional parameters with reasonable defaults
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else num_attention_heads
    causal_self_attn = True  # Default to causal for decoder models
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 0
    kv_channels = hidden_size // num_attention_heads  # Standard dimension per head

    # Calculate query projection size and ratio
    query_projection_size = kv_channels * num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / hidden_size

    # MoE parameters - simplified for NeMo config
    # In this implementation, we assume all layers are dense if num_experts is None
    if moe_router_topk == 0:
        num_dense_layers = num_layers
        num_moe_layers = 0
        num_experts_routed_to = 0
    else:
        # Simplified MoE handling - assuming uniform distribution of MoE layers
        # This can be expanded based on NeMo's actual MoE implementation
        num_moe_layers = num_layers // 2  # Simplified assumption
        num_dense_layers = num_layers - num_moe_layers
        num_experts_routed_to = moe_router_topk

    # Handle SwiGLU vs standard GELU/ReLU
    # Default to standard activation (no SwiGLU)
    gated_linear_multiplier = 1

    # Define the expansion factor as described in the paper
    # 3x: Each GEMM needs forward pass, backward wgrad, and backward dgrad
    # 2x: GEMMs are stacked twice in standard Transformer architectures
    # 2x: A GEMM of m*n with n*k requires 2mnk floating-point operations
    expansion_factor = 3 * 2 * 2
    # Attention
    if not causal_self_attn:
        attention_component = (
            1
            + (query_groups / num_attention_heads)
            # Only half of the attention matrix is non-zero and needs to be multiplied with V
            + (seq_length / hidden_size)  # If causal self attn -> divide by 2.
        ) * query_projection_to_hidden_size_ratio
    else:
        attention_component = (
            1
            + (query_groups / num_attention_heads)
            # Only half of the attention matrix is non-zero and needs to be multiplied with V
            + (seq_length / hidden_size / 2)  # If causal self attn -> divide by 2.
        ) * query_projection_to_hidden_size_ratio

    # Calculate total FLOPs
    total_flops = (
        expansion_factor
        * batch_size
        * seq_length
        * num_layers
        * hidden_size
        * hidden_size
        * (
            attention_component
            # MLP component
            + (
                (
                    # Dense layers
                    (ffn_hidden_size * num_dense_layers)
                    +
                    # MoE layers
                    (
                        (
                            # Routed experts
                            ffn_hidden_size * num_experts_routed_to
                            # Note: Shared experts are not implemented in this version
                        )
                        * num_moe_layers
                    )
                )
                * gated_linear_multiplier
                / (num_layers * hidden_size)
            )
            # Logit component
            + (vocab_size / (2 * num_layers * hidden_size))
        )
    )

    return total_flops


def deepseekv3_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for DeepSeek V3 - accepts either AutoConfig or normalized config"""

    hs = config.hidden_size
    layers = config.num_hidden_layers
    attention_heads = config.num_attention_heads
    ffn_hs = config.intermediate_size
    vocab_size = config.vocab_size

    # DeepSeek V3 specific fields
    q_lora_rank = config.q_lora_rank if hasattr(config, "q_lora_rank") else None
    kv_lora_rank = config.kv_lora_rank
    qk_rope_head_dim = config.qk_rope_head_dim
    qk_nope_head_dim = config.qk_nope_head_dim if hasattr(config, "qk_nope_head_dim") else None

    v_head_dim = config.v_head_dim

    # MoE fields
    moe_intermediate_size = config.moe_intermediate_size
    moe_shared_expert_intermediate_size = moe_intermediate_size
    moe_ffn_hidden_size = moe_intermediate_size
    moe_router_topk = config.num_experts_per_tok

    # MoE layer pattern
    first_k_dense_replace = config.first_k_dense_replace if hasattr(config, "first_k_dense_replace") else 0
    if hasattr(config, "moe_layer_freq"):
        moe_layer_freq = config.moe_layer_freq
    else:
        moe_layer_freq = [0] * first_k_dense_replace + [1] * (layers - first_k_dense_replace)

    # MTP layers (optional)
    mtp_num_layers = config.mtp_num_layers if hasattr(config, "mtp_num_layers") else None

    # DSA / sparse attention (DeepSeek V3.2)
    index_topk = getattr(config, "index_topk", None)
    index_n_heads = getattr(config, "index_n_heads", 0)
    index_head_dim = getattr(config, "index_head_dim", 0)

    # self-attention flops
    if index_topk is not None and index_topk > 0:
        # Sparse: each query attends to index_topk keys
        bmm1_flops = (qk_nope_head_dim + qk_rope_head_dim) * attention_heads * seq_len * index_topk
        bmm2_flops = v_head_dim * attention_heads * seq_len * index_topk
    else:
        # Full causal
        bmm1_flops = 0.5 * (qk_nope_head_dim + qk_rope_head_dim) * attention_heads * (seq_len**2)
        bmm2_flops = 0.5 * v_head_dim * attention_heads * (seq_len**2)
    per_input_attention_flops = 6 * (bmm1_flops + bmm2_flops) * layers
    if mtp_num_layers is not None:
        per_input_attention_flops += 6 * (bmm1_flops + bmm2_flops) * mtp_num_layers

    # DSA indexer overhead (projections + full-sequence BMM per layer)
    if index_topk is not None and index_topk > 0 and index_n_heads > 0:
        idx_proj_params = (q_lora_rank or 0) * index_n_heads * index_head_dim + hs * index_head_dim + hs * index_n_heads
        idx_bmm = index_n_heads * index_head_dim * seq_len * seq_len
        per_layer_indexer = 6 * (idx_proj_params * seq_len + idx_bmm)
        total_indexer_layers = layers + (mtp_num_layers or 0)
        per_input_attention_flops += per_layer_indexer * total_indexer_layers

    # linear layer flops
    if q_lora_rank is not None:
        per_layer_mla_params = hs * q_lora_rank + q_lora_rank * (
            (qk_nope_head_dim + qk_rope_head_dim) * attention_heads
        )  # Q
    else:
        per_layer_mla_params = hs * ((qk_nope_head_dim + qk_rope_head_dim) * attention_heads)  # Q

    per_layer_mla_params += hs * qk_rope_head_dim  # K^R
    per_layer_mla_params += hs * kv_lora_rank + kv_lora_rank * (
        (qk_nope_head_dim + v_head_dim) * attention_heads
    )  # K^C and V^C
    per_layer_mla_params += v_head_dim * attention_heads * hs  # Proj
    mla_params = per_layer_mla_params * layers
    if mtp_num_layers is not None:
        mla_params += per_layer_mla_params * mtp_num_layers

    dense_layer_ffn_params = hs * ffn_hs * 3  # gated linear unit
    per_shared_expert_params = hs * moe_shared_expert_intermediate_size * 3
    per_selected_expert_params = hs * moe_ffn_hidden_size * 3
    ffn_params = 0

    if isinstance(moe_layer_freq, int):
        moe_layer_pattern = [1 if (i % moe_layer_freq == 0) else 0 for i in range(layers)]
    else:
        moe_layer_pattern = moe_layer_freq
    for i in moe_layer_pattern:
        if i == 0:
            ffn_params += dense_layer_ffn_params
        else:
            ffn_params += per_shared_expert_params + (per_selected_expert_params * moe_router_topk)
    if mtp_num_layers is not None:
        for i in range(mtp_num_layers):
            ffn_params += per_shared_expert_params + (per_selected_expert_params * moe_router_topk)
    per_input_params = mla_params + ffn_params
    per_input_linear_flops = 6 * per_input_params * seq_len

    # vocab flops
    per_input_vocab_flops = 6 * vocab_size * hs * seq_len
    if mtp_num_layers is not None:
        for i in range(mtp_num_layers):
            per_input_vocab_flops += 6 * vocab_size * hs * seq_len
            per_input_vocab_flops += 6 * hs * 2 * hs * seq_len

    return (per_input_attention_flops + per_input_linear_flops + per_input_vocab_flops) * gbs


def _nemotronh_mlp_layer_flops(config, gbs, seq_len):
    """Model FLOPs for MLP layer. Assume gated linear unit."""
    return 6 * gbs * seq_len * config.hidden_size * config.intermediate_size * 3


def _nemotronh_moe_layer_flops(config, gbs, seq_len):
    """Model FLOPs for a MoE layer in Nemotron V3/Super V3 (hybrid Mamba/Attention/MoE).

    Nemotron V3 uses relu2 (non-gated) for both routed and shared experts,
    so each expert has 2 linear projections (up_proj + down_proj), not 3.

    When moe_latent_size is set (Super V3), routed experts operate in a reduced
    latent space with additional projection layers (fc1_latent_proj, fc2_latent_proj).
    The shared expert and gate always operate in the full hidden_size dimension.

    Accounts for:
      1. Routed experts: only num_experts_per_tok activated per token.
      2. Shared expert: always active for every token (full hidden_size).
      3. Router/gate: linear projection hidden_size -> n_routed_experts.
      4. Latent projections (if moe_latent_size is set): down and up projections.
    """
    hs = config.hidden_size
    num_tokens = gbs * seq_len

    # Determine if latent MoE is used
    moe_latent_size = getattr(config, "moe_latent_size", None)

    if moe_latent_size is not None:
        # Latent MoE: experts operate in reduced latent space
        expert_dim = moe_latent_size
        # fc1_latent_proj (hs -> latent) + fc2_latent_proj (latent -> hs)
        latent_proj_flops = 6 * num_tokens * hs * moe_latent_size * 2
    else:
        expert_dim = hs
        latent_proj_flops = 0

    # Routed experts: num_experts_per_tok activated, each up_proj + down_proj
    routed_expert_flops = 6 * num_tokens * config.num_experts_per_tok * expert_dim * config.moe_intermediate_size * 2

    # Shared expert: always active on full hidden_size, up_proj + down_proj
    shared_expert_flops = 6 * num_tokens * hs * config.moe_shared_expert_intermediate_size * 2

    # Router/gate: hidden_size -> n_routed_experts (always full dimension)
    gate_flops = 6 * num_tokens * hs * config.n_routed_experts

    return routed_expert_flops + shared_expert_flops + gate_flops + latent_proj_flops


def _non_mla_attn_layer_flops(config, gbs, seq_len):
    """Model FLOPs for attention layer"""
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads

    return (
        6
        * gbs
        * seq_len
        * hs
        * (
            hs  # Q
            + query_groups / attention_heads * hs * 2  # KV
            + seq_len / 2 * 2
            + hs
        )
    )


def _mamba_layer_flops(config, gbs, seq_len):
    """Model FLOPs for Mamba layer.

    Three components:
      - in_proj:  input projections (x_proj, z_proj, dt_proj, B_proj, C_proj)
      - scan:     SSM scan kernel (7x factor accounts for the full SSD scan cost)
      - out_proj: output projection back to hidden_size
    Multiplied by 6 (3x fwd+bwd * 2x FMA) for in_proj/out_proj (standard GEMMs),
    and 7 * 3 = 21 for scan (non-GEMM kernel, higher op count per element).
    """
    hs = config.hidden_size
    if hasattr(config, "mamba_state_dim"):
        mamba_state_dim = config.mamba_state_dim
    elif hasattr(config, "ssm_state_size"):
        mamba_state_dim = config.ssm_state_size
    else:
        raise ValueError("Expected config to have 'mamba_state_dim' or 'ssm_state_size'")
    mamba_head_dim = config.mamba_head_dim
    if hasattr(config, "mamba_num_groups"):
        mamba_num_groups = config.mamba_num_groups
    elif hasattr(config, "n_groups"):
        mamba_num_groups = config.n_groups
    else:
        raise ValueError("Expected config to have 'mamba_num_groups' or 'n_groups'")

    if hasattr(config, "mamba_num_heads") and config.mamba_num_heads:
        nheads = config.mamba_num_heads
    else:
        nheads = 2 * hs // mamba_head_dim  # default expand is 2
    d_in = nheads * mamba_head_dim

    in_proj = 6 * gbs * seq_len * hs * (2 * d_in + 2 * mamba_num_groups * mamba_state_dim + nheads)
    scan = 7 * 3 * gbs * seq_len * d_in * mamba_state_dim
    out_proj = 6 * gbs * seq_len * d_in * hs
    return in_proj + scan + out_proj


def _hybrid_model_flops(config, gbs, seq_len):
    """Model FLOPs for hybrid model"""
    if hasattr(config, "is_hybrid_model"):
        if not config.is_hybrid_model:
            raise ValueError("Config must have is_hybrid_model=True")
    elif not hasattr(config, "hybrid_override_pattern"):
        raise ValueError("Expected config to have `is_hybrid_model` or `hybrid_override_pattern`")

    hybrid_override_pattern = config.hybrid_override_pattern
    hs = config.hidden_size
    vocab_size = config.vocab_size

    num_attn_layers, num_mamba_layers, num_mlp_layers, num_moe_layers = 0, 0, 0, 0
    for c in hybrid_override_pattern:
        if c == "M":
            num_mamba_layers += 1
        elif c == "-":
            num_mlp_layers += 1
        elif c == "*":
            num_attn_layers += 1
        elif c == "E":
            num_moe_layers += 1

    total = 6 * gbs * seq_len * hs * vocab_size
    if num_attn_layers:
        total += num_attn_layers * _non_mla_attn_layer_flops(config, gbs, seq_len)
    if num_mamba_layers:
        total += num_mamba_layers * _mamba_layer_flops(config, gbs, seq_len)
    if num_mlp_layers:
        total += num_mlp_layers * _nemotronh_mlp_layer_flops(config, gbs, seq_len)
    if num_moe_layers:
        total += num_moe_layers * _nemotronh_moe_layer_flops(config, gbs, seq_len)
    return total


def nemotronh_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for NemotronH"""
    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    return _hybrid_model_flops(config, gbs, seq_len)


def attention_flops_calculator(
    seqlen,
    hidden_size,
    num_attention_heads,
    num_query_groups,
    kv_channels: int | None = None,
    is_swa: bool = False,
    swa_window_size: int = 128,
):
    """Calculate the flops for the attention part."""
    kv_channels = kv_channels or (hidden_size // num_attention_heads)

    linear_qkv = seqlen * hidden_size * (kv_channels * (num_attention_heads + num_query_groups * 2))

    linear_proj = seqlen * hidden_size * (kv_channels * num_attention_heads)

    if is_swa:
        attention_mask_nz_elem = (
            swa_window_size * (swa_window_size + 1) / 2 + (seqlen - swa_window_size) * swa_window_size
        )
        attention = num_attention_heads * (attention_mask_nz_elem * kv_channels) * 2
    else:
        bmm_k = kv_channels
        bmm_b = num_attention_heads
        attention_mask_nz_elem = seqlen * (seqlen + 1) / 2
        attention = bmm_b * attention_mask_nz_elem * bmm_k * 2

    return (linear_qkv + linear_proj + attention) * 6


def moe_mlp_flops_calculator(
    seqlen,
    hidden_size,
    moe_ffn_hidden_size,
    moe_router_topk,
    gated_linear_unit: bool = True,
):
    """Calculate the flops for the MLP"""
    total_num_tokens = seqlen * moe_router_topk
    linear_fc1 = total_num_tokens * hidden_size * moe_ffn_hidden_size * (2 if gated_linear_unit else 1)
    linear_fc2 = total_num_tokens * moe_ffn_hidden_size * hidden_size
    return (linear_fc1 + linear_fc2) * 6


def loss_flops_calculator(
    seqlen,
    hidden_size,
    vocab_size,
):
    """Calculate the flops for the loss"""
    return (seqlen * hidden_size * vocab_size) * 6


def gpt_oss_flops_calculator(
    gbs,
    num_layers,
    seqlen,
    hidden_size,
    num_attention_heads,
    num_query_groups,
    moe_ffn_hidden_size,
    moe_router_topk,
    vocab_size,
    kv_channels: int | None = None,
    swa_window_size: int = 128,
    window_attn_skip_freq: int | None = 2,
):
    """Calculate the flops for the GPT-OSS model"""
    flops = 0
    for i in range(num_layers):
        if i % window_attn_skip_freq == 0:
            flops += attention_flops_calculator(
                seqlen,
                hidden_size,
                num_attention_heads,
                num_query_groups,
                kv_channels,
                is_swa=False,
            )
        else:
            flops += attention_flops_calculator(
                seqlen,
                hidden_size,
                num_attention_heads,
                num_query_groups,
                kv_channels,
                is_swa=True,
                swa_window_size=swa_window_size,
            )
        flops += moe_mlp_flops_calculator(
            seqlen,
            hidden_size,
            moe_ffn_hidden_size,
            moe_router_topk,
        )
    flops += loss_flops_calculator(seqlen, hidden_size, vocab_size)
    flops *= gbs
    return flops


def gpt_oss_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for GPT-OSS"""
    # Map config fields
    num_layers = config.num_hidden_layers
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else num_attention_heads
    vocab_size = config.vocab_size

    # GPT-OSS specific fields
    moe_ffn_hidden_size = (
        config.moe_ffn_hidden_size if hasattr(config, "moe_ffn_hidden_size") else config.intermediate_size
    )
    moe_router_topk = config.num_experts_per_tok
    kv_channels = getattr(config, "head_dim", None)
    if kv_channels is None:
        kv_channels = getattr(config, "kv_channels", hidden_size // num_attention_heads)
    swa_window_size = config.window_size[0] if hasattr(config, "window_size") and config.window_size else 128
    window_attn_skip_freq = config.window_attn_skip_freq if hasattr(config, "window_attn_skip_freq") else 2

    return gpt_oss_flops_calculator(
        gbs=gbs,
        num_layers=num_layers,
        seqlen=seq_len,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_query_groups=num_query_groups,
        moe_ffn_hidden_size=moe_ffn_hidden_size,
        moe_router_topk=moe_router_topk,
        vocab_size=vocab_size,
        kv_channels=kv_channels,
        swa_window_size=swa_window_size,
        window_attn_skip_freq=window_attn_skip_freq,
    )


def glm4_moe_flops(config, gbs=1, seq_len=None):
    """Estimate FLOPs for GLM4 MoE model configurations."""
    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    vocab_size = config.vocab_size

    # GLM4 MoE attention config
    head_dim = getattr(config, "head_dim", hs // attention_heads)
    query_projection_to_hidden_size_ratio = (head_dim * attention_heads) / hs

    # MoE config
    ffn_hs = config.intermediate_size  # for dense layers
    moe_intermediate_size = config.moe_intermediate_size if hasattr(config, "moe_intermediate_size") else ffn_hs
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 1
    n_shared_experts = config.n_shared_experts if hasattr(config, "n_shared_experts") else 0
    first_k_dense_replace = config.first_k_dense_replace if hasattr(config, "first_k_dense_replace") else 0

    causal_self_attn = True
    hidden_size = hs
    gated_linear_multiplier = 2  # SwiGLU

    # Attention flops for GQA (Qwen3-style)
    attention_flops = (
        3
        * 2
        * gbs
        * layers
        * seq_len
        * hidden_size
        * hidden_size
        * query_projection_to_hidden_size_ratio
        * (
            (query_groups / attention_heads * 2 + 1)  # QKV gemm
            + (seq_len / hidden_size * 2 * (0.5 if causal_self_attn else 1))  # attention
            + 1  # attention proj gemm
        )
    )

    # MLP flops (DeepSeek V3-style MoE)
    # Dense layers: first_k_dense_replace layers
    dense_mlp_flops = (
        3 * 2 * gbs * first_k_dense_replace * seq_len * hidden_size * (1 + gated_linear_multiplier) * ffn_hs
    )

    # MoE layers: (layers - first_k_dense_replace) layers
    # Each MoE layer has: shared experts + routed experts (topk selected)
    num_moe_layers = layers - first_k_dense_replace

    # Shared expert flops (always computed)
    shared_expert_flops = (
        3
        * 2
        * gbs
        * num_moe_layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * (moe_intermediate_size * n_shared_experts)
    )

    # Routed expert flops (topk selected)
    routed_expert_flops = (
        3
        * 2
        * gbs
        * num_moe_layers
        * seq_len
        * hidden_size
        * (1 + gated_linear_multiplier)
        * (moe_intermediate_size * moe_router_topk)
    )

    mlp_flops = dense_mlp_flops + shared_expert_flops + routed_expert_flops

    # Vocab flops
    vocab_flops = 3 * 2 * gbs * seq_len * hidden_size * vocab_size

    return attention_flops + mlp_flops + vocab_flops


def minimax_m2_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for MiniMax-M2 family - accepts either AutoConfig or normalized config.

    Architecture: GQA attention (Q/K/V/O separate projections, head_dim may differ from
    hidden_size // num_heads) + MoE with SwiGLU (no shared experts by default).
    Optionally includes MTP (Multi-Token Prediction) modules gated by use_mtp.
    """

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    vocab_size = config.vocab_size
    head_dim = getattr(config, "head_dim", hs // attention_heads)
    query_projection_to_hidden_size_ratio = (head_dim * attention_heads) / hs

    # MoE config — all layers are MoE, no shared experts by default
    ffn_hs = config.intermediate_size
    moe_router_topk = config.num_experts_per_tok if hasattr(config, "num_experts_per_tok") else 8
    shared_intermediate_size = getattr(config, "shared_intermediate_size", 0)

    # MTP config (optional, gated by use_mtp)
    use_mtp = getattr(config, "use_mtp", False)
    num_mtp_modules = getattr(config, "num_mtp_modules", 0) if use_mtp else 0
    mtp_transformer_layers = getattr(config, "mtp_transformer_layers", 1)

    causal_self_attn = True
    gated_linear_multiplier = 2  # SwiGLU: gate + up projections

    # --- Attention flops (GQA with separate Q/K/V/O projections) ---
    def _attention_flops_per_layer():
        return (
            6
            * gbs
            * seq_len
            * hs
            * hs
            * query_projection_to_hidden_size_ratio
            * (
                (query_groups / attention_heads * 2 + 1)  # QKV gemm
                + (seq_len / hs * 2 * (0.5 if causal_self_attn else 1))  # BMM (causal)
                + 1  # output proj gemm
            )
        )

    attention_flops = _attention_flops_per_layer() * layers

    # --- MoE MLP flops (SwiGLU, all layers) ---
    def _moe_mlp_flops_per_layer():
        # Routed experts (topk selected)
        routed = 6 * gbs * seq_len * hs * (1 + gated_linear_multiplier) * (ffn_hs * moe_router_topk)
        # Shared experts (if any)
        shared = (
            6 * gbs * seq_len * hs * (1 + gated_linear_multiplier) * shared_intermediate_size
            if shared_intermediate_size > 0
            else 0
        )
        return routed + shared

    mlp_flops = _moe_mlp_flops_per_layer() * layers

    # --- Vocab flops (lm_head) ---
    vocab_flops = 6 * gbs * seq_len * hs * vocab_size

    # --- MTP module flops (optional) ---
    mtp_flops = 0
    if num_mtp_modules > 0:
        total_mtp_layers = num_mtp_modules * mtp_transformer_layers
        # Embedding projection per module: concat(hidden, next_embed) -> hidden  (2*hs -> hs)
        mtp_flops += 6 * gbs * seq_len * hs * 2 * hs * num_mtp_modules
        # Transformer layers (attention + MoE MLP)
        mtp_flops += _attention_flops_per_layer() * total_mtp_layers
        mtp_flops += _moe_mlp_flops_per_layer() * total_mtp_layers
        # Vocab projection per module
        mtp_flops += 6 * gbs * seq_len * hs * vocab_size * num_mtp_modules

    return attention_flops + mlp_flops + vocab_flops + mtp_flops


def _gdn_attention_per_layer_flops(
    gbs,
    seq_len,
    hidden_size,
    linear_key_head_dim,
    linear_value_head_dim,
    linear_num_key_heads,
    linear_num_value_heads,
    linear_conv_kernel_dim,
):
    """FLOPs for a single Gated DeltaNet (GDN / linear attention) layer.

    Based on the GDN FLOPs calculator from Megatron-Bridge PR #2925.
    """
    qk_dim = linear_key_head_dim * linear_num_key_heads
    v_dim = linear_value_head_dim * linear_num_value_heads

    return (
        3
        * 2
        * gbs
        * seq_len
        * (
            hidden_size * (2 * qk_dim + 2 * v_dim + 2 * linear_num_value_heads)
            + linear_conv_kernel_dim * (2 * qk_dim + v_dim)
            + linear_num_value_heads * (linear_value_head_dim**2) * 4
            + hidden_size * v_dim
        )
    )


def qwen3_5_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for Qwen3.5 family (MoE and Dense) with hybrid GDN/full attention.

    Qwen3.5 uses a hybrid attention pattern: 75% GDN (linear attention) layers
    and 25% standard GQA (full attention) layers (full_attention_interval=4).
    Supports both the MoE variant (Qwen3.5-35B-A3B) and Dense variant (Qwen3.5-27B).
    """
    # For VL composite configs, use the text_config sub-config
    if hasattr(config, "text_config") and not hasattr(config, "num_hidden_layers"):
        config = config.text_config

    if seq_len is None:
        seq_len = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 2048

    layers = config.num_hidden_layers
    hs = config.hidden_size
    attention_heads = config.num_attention_heads
    query_groups = config.num_key_value_heads if hasattr(config, "num_key_value_heads") else attention_heads
    vocab_size = config.vocab_size
    head_dim = getattr(config, "head_dim", hs // attention_heads)

    # GDN (linear attention) parameters
    linear_key_head_dim = config.linear_key_head_dim
    linear_value_head_dim = config.linear_value_head_dim
    linear_num_key_heads = config.linear_num_key_heads
    linear_num_value_heads = config.linear_num_value_heads
    linear_conv_kernel_dim = getattr(config, "linear_conv_kernel_dim", 4)

    # Determine layer counts from layer_types or full_attention_interval
    if hasattr(config, "layer_types") and config.layer_types:
        layer_types = config.layer_types
        num_full_attn_layers = sum(1 for lt in layer_types if lt == "full_attention")
        num_gdn_layers = layers - num_full_attn_layers
    else:
        full_attention_interval = getattr(config, "full_attention_interval", 4)
        num_full_attn_layers = layers // full_attention_interval
        num_gdn_layers = layers - num_full_attn_layers

    # MoE fields
    is_moe = hasattr(config, "num_experts") and config.num_experts is not None and config.num_experts > 1
    moe_router_topk = getattr(config, "num_experts_per_tok", 1) if is_moe else 1
    moe_intermediate_size = getattr(config, "moe_intermediate_size", 0) if is_moe else 0
    shared_expert_intermediate_size = getattr(config, "shared_expert_intermediate_size", 0) if is_moe else 0
    ffn_hs = getattr(config, "intermediate_size", 0) if not is_moe else 0

    # MTP layers
    mtp_num_layers = getattr(config, "mtp_num_hidden_layers", 0) or 0

    causal_self_attn = True
    gated_linear_multiplier = 2  # SwiGLU: gate + up projections

    query_projection_to_hidden_size_ratio = (head_dim * attention_heads) / hs

    # Qwen3.5 uses gated attention: Q proj outputs 2x (query + gate), applied as sigmoid(gate)*attn
    attn_output_gate = getattr(config, "attn_output_gate", True)
    q_gate_multiplier = 2 if attn_output_gate else 1

    # --- Standard (full) attention flops per layer ---
    full_attn_per_layer = (
        6
        * gbs
        * seq_len
        * hs
        * hs
        * query_projection_to_hidden_size_ratio
        * (
            (query_groups / attention_heads * 2 + q_gate_multiplier)  # QKV gemm (Q is 2x with gate)
            + (seq_len / hs * 2 * (0.5 if causal_self_attn else 1))  # attention BMM
            + 1  # output proj gemm
        )
    )

    # --- GDN (linear attention) flops per layer ---
    gdn_attn_per_layer = _gdn_attention_per_layer_flops(
        gbs,
        seq_len,
        hs,
        linear_key_head_dim,
        linear_value_head_dim,
        linear_num_key_heads,
        linear_num_value_heads,
        linear_conv_kernel_dim,
    )

    # Total attention flops
    attention_flops = full_attn_per_layer * num_full_attn_layers + gdn_attn_per_layer * num_gdn_layers

    # --- MLP flops ---
    if is_moe:
        # Routed experts (topk selected) + shared experts, all layers are MoE
        routed_expert_flops = (
            6 * gbs * layers * seq_len * hs * (1 + gated_linear_multiplier) * (moe_intermediate_size * moe_router_topk)
        )
        shared_expert_flops = (
            6 * gbs * layers * seq_len * hs * (1 + gated_linear_multiplier) * shared_expert_intermediate_size
        )
        mlp_flops = routed_expert_flops + shared_expert_flops
    else:
        # Dense MLP with SwiGLU
        mlp_flops = 6 * gbs * layers * seq_len * hs * (1 + gated_linear_multiplier) * ffn_hs

    # --- Vocab flops ---
    vocab_flops = 6 * gbs * seq_len * hs * vocab_size

    # --- MTP flops ---
    mtp_flops = 0
    if mtp_num_layers > 0:
        # Embedding projection per MTP layer: 2*hs -> hs
        mtp_flops += 6 * gbs * seq_len * hs * 2 * hs * mtp_num_layers
        # MTP layers reuse the last transformer layer pattern (assumed full attention)
        mtp_flops += full_attn_per_layer * mtp_num_layers
        # MTP MLP (same as main model's last layer)
        if is_moe:
            mtp_mlp_per_layer = (
                6
                * gbs
                * seq_len
                * hs
                * (1 + gated_linear_multiplier)
                * (moe_intermediate_size * moe_router_topk + shared_expert_intermediate_size)
            )
        else:
            mtp_mlp_per_layer = 6 * gbs * seq_len * hs * (1 + gated_linear_multiplier) * ffn_hs
        mtp_flops += mtp_mlp_per_layer * mtp_num_layers
        # Vocab projection per MTP layer
        mtp_flops += 6 * gbs * seq_len * hs * vocab_size * mtp_num_layers

    return attention_flops + mlp_flops + vocab_flops + mtp_flops


# ---------------------------------------------------------------------------
# Shared helpers for MLA (Multi-Latent Attention) + MoE models
# ---------------------------------------------------------------------------


def _mla_attention_per_layer_flops(
    gbs,
    seq_len,
    hs,
    attention_heads,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_head_dim,
    qk_nope_head_dim,
    v_head_dim,
    index_topk=None,
    index_n_heads=0,
    index_head_dim=0,
):
    """Per-layer FLOPs for Multi-Latent Attention (MLA).

    Shared by DeepSeek V3, Kimi K2.5, Mistral Small 4, GLM-5, etc.

    When index_topk is set (DSA / sparse attention), accounts for:
      - Sparse main attention BMM: S * index_topk instead of 0.5 * S^2
      - DSA indexer overhead: Q/K/weights projections + full S^2 indexer BMM
    """
    # --- Main MLA attention BMM ---
    if index_topk is not None and index_topk > 0:
        # Sparse attention: each query attends to index_topk keys (not full causal)
        bmm1 = (qk_nope_head_dim + qk_rope_head_dim) * attention_heads * seq_len * index_topk
        bmm2 = v_head_dim * attention_heads * seq_len * index_topk
    else:
        # Full causal attention
        bmm1 = 0.5 * (qk_nope_head_dim + qk_rope_head_dim) * attention_heads * (seq_len**2)
        bmm2 = 0.5 * v_head_dim * attention_heads * (seq_len**2)
    bmm_flops = 6 * gbs * (bmm1 + bmm2)

    # --- MLA linear projections ---
    if q_lora_rank is not None:
        q_params = hs * q_lora_rank + q_lora_rank * ((qk_nope_head_dim + qk_rope_head_dim) * attention_heads)
    else:
        q_params = hs * ((qk_nope_head_dim + qk_rope_head_dim) * attention_heads)

    kr_params = hs * qk_rope_head_dim
    kv_params = hs * kv_lora_rank + kv_lora_rank * ((qk_nope_head_dim + v_head_dim) * attention_heads)
    o_params = v_head_dim * attention_heads * hs

    linear_flops = 6 * gbs * seq_len * (q_params + kr_params + kv_params + o_params)

    # --- DSA indexer overhead ---
    indexer_flops = 0
    if index_topk is not None and index_topk > 0 and index_n_heads > 0:
        # Indexer projections: wq_b (q_lora -> idx_heads*idx_hd),
        #                      wk (hs -> idx_hd), weights_proj (hs -> idx_heads)
        idx_proj_params = (
            (q_lora_rank or 0) * index_n_heads * index_head_dim  # wq_b
            + hs * index_head_dim  # wk
            + hs * index_n_heads  # weights_proj
        )
        # Indexer full-sequence BMM: Q@K^T over all positions to find top-k
        idx_bmm = index_n_heads * index_head_dim * seq_len * seq_len
        indexer_flops = 6 * gbs * (idx_proj_params * seq_len + idx_bmm)

    return bmm_flops + linear_flops + indexer_flops


def _mla_moe_model_flops(
    gbs,
    seq_len,
    hs,
    layers,
    attention_heads,
    vocab_size,
    q_lora_rank,
    kv_lora_rank,
    qk_rope_head_dim,
    qk_nope_head_dim,
    v_head_dim,
    dense_ffn_hs,
    moe_ffn_hs,
    moe_router_topk,
    moe_shared_expert_hs,
    moe_layer_pattern,
    mtp_num_layers=0,
    index_topk=None,
    index_n_heads=0,
    index_head_dim=0,
):
    """FLOPs for MLA + MoE transformer models (DeepSeek-V3 style).

    Args:
        moe_layer_pattern: List of 0/1 per layer (0=dense, 1=MoE).
        moe_shared_expert_hs: Total intermediate size for all shared experts combined.
        index_topk: If set, use DSA sparse attention with this many selected positions.
        index_n_heads: Number of heads in the DSA indexer.
        index_head_dim: Head dimension of the DSA indexer.
    """
    # --- Attention (MLA on every layer) ---
    mla_per_layer = _mla_attention_per_layer_flops(
        gbs,
        seq_len,
        hs,
        attention_heads,
        q_lora_rank,
        kv_lora_rank,
        qk_rope_head_dim,
        qk_nope_head_dim,
        v_head_dim,
        index_topk=index_topk,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
    )
    attention_flops = mla_per_layer * layers

    # --- FFN (dense or MoE with shared experts, SwiGLU = 3 projections) ---
    dense_layer_ffn_params = hs * dense_ffn_hs * 3
    per_shared_expert_params = hs * moe_shared_expert_hs * 3
    per_selected_expert_params = hs * moe_ffn_hs * 3

    ffn_params = 0
    for is_moe in moe_layer_pattern:
        if is_moe == 0:
            ffn_params += dense_layer_ffn_params
        else:
            ffn_params += per_shared_expert_params + (per_selected_expert_params * moe_router_topk)
    ffn_flops = 6 * gbs * seq_len * ffn_params

    # --- Vocab ---
    vocab_flops = 6 * gbs * seq_len * hs * vocab_size

    # --- MTP ---
    mtp_flops = 0
    if mtp_num_layers > 0:
        mtp_flops += mla_per_layer * mtp_num_layers
        last_is_moe = moe_layer_pattern[-1] if moe_layer_pattern else 0
        if last_is_moe:
            mtp_ffn_params = per_shared_expert_params + (per_selected_expert_params * moe_router_topk)
        else:
            mtp_ffn_params = dense_layer_ffn_params
        mtp_flops += 6 * gbs * seq_len * mtp_ffn_params * mtp_num_layers
        mtp_flops += 6 * gbs * seq_len * hs * vocab_size * mtp_num_layers
        mtp_flops += 6 * gbs * seq_len * hs * 2 * hs * mtp_num_layers  # embedding projection

    return attention_flops + ffn_flops + vocab_flops + mtp_flops


def _build_moe_layer_pattern(config, layers):
    """Build a list of 0/1 indicating dense(0) vs MoE(1) per layer.

    Handles multiple config styles: first_k_dense_replace + moe_layer_freq,
    mlp_layer_types list, etc.
    """
    mlp_layer_types = getattr(config, "mlp_layer_types", None)
    if mlp_layer_types is not None:
        return [0 if lt == "dense" else 1 for lt in mlp_layer_types]

    first_k_dense = getattr(config, "first_k_dense_replace", 0)
    moe_layer_freq = getattr(config, "moe_layer_freq", 1)
    if isinstance(moe_layer_freq, list):
        return moe_layer_freq
    return [0] * first_k_dense + [
        1 if ((i - first_k_dense) % moe_layer_freq == 0) else 0 for i in range(first_k_dense, layers)
    ]


def mla_moe_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for MLA + MoE models (Kimi K2, GLM-5, Mistral Small 4, etc.).

    Handles VL wrappers by extracting text_config if present.
    """
    # Handle VL wrappers with nested text_config
    cfg = config
    if hasattr(config, "text_config") and not hasattr(config, "num_hidden_layers"):
        cfg = config.text_config

    if seq_len is None:
        seq_len = getattr(cfg, "max_position_embeddings", 2048)

    layers = cfg.num_hidden_layers
    hs = cfg.hidden_size
    n_shared = getattr(cfg, "n_shared_experts", 0)

    # MoE intermediate size: try multiple field names
    moe_int_size = getattr(cfg, "moe_intermediate_size", None)
    if moe_int_size is None:
        moe_int_size = getattr(cfg, "expert_ffn_hidden_size", cfg.intermediate_size)

    # Dense FFN intermediate size
    dense_ffn_hs = getattr(cfg, "intermediate_size", None)
    if dense_ffn_hs is None:
        dense_ffn_hs = getattr(cfg, "ffn_hidden_size", moe_int_size)

    # Router top-k: try multiple field names
    moe_topk = getattr(cfg, "num_experts_per_tok", None)
    if moe_topk is None:
        moe_topk = getattr(cfg, "moe_topk", 1)

    moe_layer_pattern = _build_moe_layer_pattern(cfg, layers)

    # MTP: try multiple field names used by different models
    mtp = getattr(cfg, "num_nextn_predict_layers", None)
    if mtp is None:
        mtp = getattr(cfg, "mtp_num_layers", 0)
    mtp = mtp or 0

    # DSA (Dynamic Sparse Attention) indexer fields
    idx_topk = getattr(cfg, "index_topk", None)
    idx_n_heads = getattr(cfg, "index_n_heads", 0)
    idx_head_dim = getattr(cfg, "index_head_dim", 0)

    return _mla_moe_model_flops(
        gbs=gbs,
        seq_len=seq_len,
        hs=hs,
        layers=layers,
        attention_heads=cfg.num_attention_heads,
        vocab_size=cfg.vocab_size,
        q_lora_rank=getattr(cfg, "q_lora_rank", None),
        kv_lora_rank=cfg.kv_lora_rank,
        qk_rope_head_dim=cfg.qk_rope_head_dim,
        qk_nope_head_dim=cfg.qk_nope_head_dim,
        v_head_dim=cfg.v_head_dim,
        dense_ffn_hs=dense_ffn_hs,
        moe_ffn_hs=moe_int_size,
        moe_router_topk=moe_topk,
        moe_shared_expert_hs=moe_int_size * n_shared,
        moe_layer_pattern=moe_layer_pattern,
        mtp_num_layers=mtp,
        index_topk=idx_topk,
        index_n_heads=idx_n_heads,
        index_head_dim=idx_head_dim,
    )


def _kda_attention_per_layer_flops(
    gbs: int,
    seq_len: int,
    hs: int,
    head_dim: int,
    num_heads: int,
    conv_kernel_size: int,
    use_full_rank_gate: bool,
    chunk_size: int = 64,
) -> float:
    """Per-layer FLOPs for Kimi Delta Attention (KDA, gated delta-rule linear attention).

    Projections follow the KDA layer definition (Kimi Linear, arXiv:2510.26692):
    q/k/v with short causal convolutions, the low-rank forget gate
    (``f_a``/``f_b``), the beta projection, the output gate (full-rank ``g_proj``
    or low-rank ``g_a``/``g_b``) and ``o_proj``. The chunkwise kernel is costed
    with the paper's own count, ``FLOPs_KDA(T; C, d_h) = 6 T d_h^2 + 3 T C d_h + T C^2``
    per head (forward, chunk size ``C``), halved to MAC-equivalents so the shared
    ``6 x`` (forward + backward) factor of this module applies.
    """
    proj = num_heads * head_dim
    gate_params = hs * proj if use_full_rank_gate else hs * head_dim + head_dim * proj
    linear_params = (
        3 * hs * proj  # q, k, v projections
        + hs * head_dim
        + head_dim * proj  # forget gate f_a / f_b
        + hs * num_heads  # beta projection
        + gate_params  # output gate
        + proj * hs  # o_proj
    )
    conv_per_token = 3 * proj * conv_kernel_size
    kernel_per_token = num_heads * (6 * head_dim**2 + 3 * chunk_size * head_dim + chunk_size**2) / 2
    return 6 * gbs * seq_len * (linear_params + conv_per_token + kernel_per_token)


def kimi_k3_flops(config: Any, gbs: int = 1, seq_len: int | None = None) -> float:
    """Model FLOPs for Kimi K3 (hybrid KDA / gated-MLA attention + latent MoE with a SiTU shared expert).

    Accepts ``KimiK3TextConfig`` or the multimodal ``KimiK3Config`` wrapper (its
    ``text_config`` is used; the vision tower is not counted, as for other VL entries).

    The layer pattern is read from the config rather than assumed: 1-based
    ``linear_attn_config["kda_layers"]`` marks KDA layers and every other layer
    is MLA (with an extra output-gate projection when ``mla_use_output_gate``);
    ``first_k_dense_replace`` / ``moe_layer_freq`` mark dense vs MoE MLPs, using
    the same rule as ``KimiDecoderLayer``. Routed experts run in a latent space
    of ``routed_expert_hidden_size`` behind shared down/up projections, and the
    shared expert is one SiTU MLP of ``moe_intermediate_size * num_shared_experts``.
    The router GEMM (``hidden_size x num_experts``, about 0.6% of the total), norms
    and the scalar attention-residual projections are omitted, matching the other
    MoE formulas in this module.

    The router top-k field on this config is ``num_experts_per_token`` (not the
    ``num_experts_per_tok`` that the generic ``transformer_flops`` fallback probes),
    which is why K3 was previously costed as a dense 93-layer transformer at well
    under half of its real FLOPs. For the released checkpoint's configuration
    (96 attention heads, dense ``intermediate_size`` 33792) the formula counts
    104.0B active matmul parameters per token and 2.78T total, matching the
    model card ("104B activated / 2.8T") and the tech report's Table 1 (104.2B /
    2.78T) to within 0.2%, which is inside the accounting difference for the
    non-GEMM parameters (router, norms, gate biases). The report's
    single MTP layer is not shipped in the checkpoint (``num_nextn_predict_layers``
    is 0) and is not modelled by Automodel, so it is not counted.
    """
    if hasattr(config, "text_config") and not hasattr(config, "num_hidden_layers"):
        config = config.text_config

    if seq_len is None:
        seq_len = getattr(config, "max_position_embeddings", 2048)

    layers = config.num_hidden_layers
    hs = config.hidden_size
    vocab_size = config.vocab_size

    # --- attention layer pattern (1-based kda_layers list) ---
    linear_attn_config = getattr(config, "linear_attn_config", None) or {}
    kda_layer_ids = set(linear_attn_config.get("kda_layers", []))
    num_kda_layers = sum(1 for layer in range(1, layers + 1) if layer in kda_layer_ids)
    num_mla_layers = layers - num_kda_layers

    attention_heads = config.num_attention_heads
    mla_per_layer = _mla_attention_per_layer_flops(
        gbs,
        seq_len,
        hs,
        attention_heads,
        getattr(config, "q_lora_rank", None),
        config.kv_lora_rank,
        config.qk_rope_head_dim,
        config.qk_nope_head_dim,
        config.v_head_dim,
    )
    if getattr(config, "mla_use_output_gate", False):
        mla_per_layer += 6 * gbs * seq_len * hs * attention_heads * config.v_head_dim

    kda_per_layer = 0
    if num_kda_layers:
        kda_per_layer = _kda_attention_per_layer_flops(
            gbs,
            seq_len,
            hs,
            linear_attn_config["head_dim"],
            linear_attn_config["num_heads"],
            linear_attn_config.get("short_conv_kernel_size", 4),
            linear_attn_config.get("use_full_rank_gate", False),
        )
    attention_flops = num_mla_layers * mla_per_layer + num_kda_layers * kda_per_layer

    # --- MLP layer pattern (same rule as KimiDecoderLayer) ---
    num_experts = getattr(config, "num_experts", None)
    is_moe = num_experts is not None and num_experts > 1
    first_k_dense = getattr(config, "first_k_dense_replace", 0) or 0
    moe_layer_freq = getattr(config, "moe_layer_freq", 1) or 1
    num_moe_layers = sum(1 for i in range(layers) if is_moe and i >= first_k_dense and i % moe_layer_freq == 0)
    num_dense_layers = layers - num_moe_layers

    dense_ffn_params = 3 * hs * config.intermediate_size  # SiTU MLP: gate, up, down
    moe_ffn_params = 0
    if is_moe:
        moe_topk = getattr(config, "num_experts_per_token", None)
        if moe_topk is None:
            moe_topk = getattr(config, "num_experts_per_tok", 1)
        moe_ffn_hs = config.moe_intermediate_size
        latent = getattr(config, "routed_expert_hidden_size", None)
        expert_in = latent or hs
        routed_params = moe_topk * 3 * expert_in * moe_ffn_hs
        latent_proj_params = 2 * hs * latent if latent else 0
        shared_params = 3 * hs * moe_ffn_hs * (getattr(config, "num_shared_experts", 0) or 0)
        moe_ffn_params = routed_params + latent_proj_params + shared_params
    ffn_flops = 6 * gbs * seq_len * (num_dense_layers * dense_ffn_params + num_moe_layers * moe_ffn_params)

    # --- vocab ---
    vocab_flops = 6 * gbs * seq_len * hs * vocab_size

    return attention_flops + ffn_flops + vocab_flops


def _sum_min_floor_div(seq_len: int, ratio: int, cap: int | None) -> int:
    """``sum_{j=1..seq_len} min(cap, floor(j / ratio))`` in closed form (``cap=None`` disables the cap).

    ``floor(j / ratio)`` is the number of complete compressed groups visible to the query at
    zero-based position ``j - 1``; the cap is the sparse top-k. Closed form so million-token
    sequences cost nothing to evaluate.
    """

    def floor_sum(n: int) -> int:
        q, rem = divmod(n, ratio)
        return ratio * q * (q - 1) // 2 + q * (rem + 1)

    if cap is None or seq_len <= cap * ratio:
        return floor_sum(seq_len)
    threshold = cap * ratio  # from here on every query sees at least ``cap`` groups
    return floor_sum(threshold) + (seq_len - threshold) * cap


def _sum_min_window(seq_len: int, window: int) -> int:
    """``sum_{i=0..seq_len-1} min(i + 1, window)``: causal sliding-window keys per query, summed."""
    if seq_len <= window:
        return seq_len * (seq_len + 1) // 2
    return window * (window + 1) // 2 + (seq_len - window) * window


def deepseek_v41_flops(config: Any, gbs: int = 1, seq_len: int | None = None) -> float:
    """Model FLOPs for DeepSeek-V4.1 (CSA2 sparse attention, single-pass mHC, Engram, MoE).

    Accepts ``DeepseekV41TextConfig`` or the multimodal ``DeepseekV41Config`` wrapper (its
    ``text_config`` is used; the vision tower is not counted, as for other VL entries). The
    module shapes follow ``nemo_automodel/components/models/deepseek_v41``:

    * attention linears per layer: ``wq_a`` (hidden x q_lora_rank), ``wq_b`` (q_lora_rank x
      heads*head_dim), ``wkv`` (hidden x head_dim, one shared latent), grouped ``wo_a``
      (heads*head_dim -> o_groups*o_lora_rank) and ``wo_b`` (o_groups*o_lora_rank x hidden);
    * compressors on ``kv_source_layer_ids``: ``wkv`` and, for ratio > 1, ``wgate`` (hidden x
      head_dim each, applied to every token before pooling);
    * sparse attention BMMs: every query attends to ``min(i+1, sliding_window)`` local keys plus
      ``min(index_topk, floor((i+1)/ratio))`` selected compressed keys (SWA-only layers have
      ``compress_ratios[i] == 0``); QK^T and PV each cost ``heads * head_dim`` MACs per key;
    * the indexer is frozen (``_Indexer.requires_grad_(False)``; hard top-k has no gradient), so
      its projections and its causal scoring over the compressed positions are counted forward
      only (2 x MACs) — the implementation scores the full ``[S, S/ratio]`` block before
      masking, which is executed work but not model work;
    * MoE per layer: router GEMM (hidden x n_routed_experts) plus ``num_experts_per_tok +
      n_shared_experts`` experts of ``3 * hidden * moe_intermediate_size``;
    * single-pass mHC: two coefficient projections per layer (``hc_mult*(hc_mult+2)`` x
      ``hc_mult*hidden``); the stream collapse/expand are elementwise and not counted;
    * Engram on ``engram_layer_ids``: the fused key/value projection
      (``(engram_max_ngram_size-1)*engram_n_heads*engram_head_dim`` x ``hidden*(hc_mult+1)``);
      the table gather, hashing, norms, RoPE and quantize/dequantize boundaries are not GEMMs;
    * the FP32 language-model head (hidden x vocab).

    Trainable GEMMs and attention BMMs cost 6 x MACs (forward + backward); activation
    recomputation is excluded (that is HFU). For the released Flash configuration the formula
    counts 16.09B active GEMM parameters per token (model card: 16B activated during decode)
    and 105.6 GFLOPs/token at 4096 tokens, 1.095x the dense-FFN identity ``6 x active``.

    Args:
        config: ``DeepseekV41TextConfig``, or a ``DeepseekV41Config`` wrapper whose ``text_config`` is used.
        gbs: Number of sequences per step.
        seq_len: Tokens per sequence; defaults to ``config.max_position_embeddings``.

    Returns:
        Model FLOPs for one training step of ``gbs`` sequences of ``seq_len`` tokens.
    """
    if hasattr(config, "text_config") and not hasattr(config, "num_hidden_layers"):
        config = config.text_config

    if seq_len is None:
        seq_len = getattr(config, "max_position_embeddings", 4096)
    seq_len = int(seq_len)

    layers = config.num_hidden_layers
    hs = config.hidden_size
    vocab_size = config.vocab_size
    heads = config.num_attention_heads
    head_dim = config.head_dim
    q_lora_rank = config.q_lora_rank
    o_lora_rank = config.o_lora_rank
    o_groups = config.o_groups
    window = config.sliding_window
    compress_ratios = list(config.compress_ratios)
    kv_source_layer_ids = set(config.kv_source_layer_ids)
    index_source_layer_ids = set(config.index_source_layer_ids)
    index_n_heads = config.index_n_heads
    index_head_dim = config.index_head_dim
    index_topk = config.index_topk
    hc_mult = config.hc_mult
    moe_inter = config.moe_intermediate_size
    n_routed = config.n_routed_experts
    topk = config.num_experts_per_tok
    n_shared = config.n_shared_experts
    engram_layers = [lid for lid in config.engram_layer_ids if lid < layers]
    engram_hash_heads = (config.engram_max_ngram_size - 1) * config.engram_n_heads
    engram_head_dim = config.engram_head_dim

    # --- trainable GEMM MACs per token (6x) ---
    attn_linear = (
        hs * q_lora_rank
        + q_lora_rank * heads * head_dim
        + hs * head_dim
        + heads * head_dim * o_lora_rank  # grouped wo_a: (heads*head_dim/o_groups) x o_lora_rank per group
        + o_groups * o_lora_rank * hs
    )
    compressor = sum(
        hs * head_dim * (2 if compress_ratios[lid] > 1 else 1) for lid in kv_source_layer_ids if lid < layers
    )
    experts = (topk + n_shared) * 3 * hs * moe_inter
    router = hs * n_routed
    mhc = 2 * (hc_mult * (hc_mult + 2)) * hc_mult * hs
    engram = len(engram_layers) * engram_hash_heads * engram_head_dim * hs * (hc_mult + 1)
    lm_head = hs * vocab_size
    trainable_macs_per_token = layers * (attn_linear + experts + router + mhc) + compressor + engram + lm_head
    trainable_flops = 6 * gbs * seq_len * trainable_macs_per_token

    # --- sparse attention BMMs (6x), summed over query positions ---
    bmm_macs = 0
    for lid in range(layers):
        ratio = compress_ratios[lid]
        keys = _sum_min_window(seq_len, window)
        if ratio:
            keys += _sum_min_floor_div(seq_len, ratio, index_topk)
        bmm_macs += 2 * heads * head_dim * keys  # QK^T and PV
    attention_flops = 6 * gbs * bmm_macs

    # --- frozen indexer: forward only (2x) ---
    indexer_macs = 0
    for lid in index_source_layer_ids:
        if lid >= layers:
            continue
        ratio = compress_ratios[lid]
        indexer_macs += seq_len * (q_lora_rank * index_n_heads * index_head_dim + hs * index_n_heads)
        if lid in kv_source_layer_ids:
            indexer_macs += (seq_len // ratio) * head_dim * index_head_dim  # wk on compressed tokens
        indexer_macs += index_n_heads * index_head_dim * _sum_min_floor_div(seq_len, ratio, None)
    indexer_flops = 2 * gbs * indexer_macs

    return float(trainable_flops + attention_flops + indexer_flops)


def _sum_mod(seq_len: int, ratio: int) -> int:
    """``sum_{j=1..seq_len} (j mod ratio)`` in closed form.

    ``j mod ratio`` is the length of the incomplete compressed tail visible to the query at
    zero-based position ``j - 1``.
    """
    full_cycles, remainder = divmod(seq_len, ratio)
    return full_cycles * (ratio * (ratio - 1) // 2) + remainder * (remainder + 1) // 2


def qwen3_8_flash_next_flops(config: Any, gbs: int = 1, seq_len: int | None = None) -> float:
    """Model FLOPs for Qwen3.8-Flash-Next (GDN + compressed-block QSA, HyperConnections, Engram PLE, MoE).

    Accepts ``Qwen3_8_FlashNextTextConfig`` or the multimodal ``Qwen3_8_FlashNextConfig`` wrapper
    (its ``text_config`` is used; the vision tower is not loaded by the training path). The module
    shapes follow ``nemo_automodel/components/models/qwen3_8_flash_next``:

    * GatedDeltaNet layers (``layers_block_type == "linear_attention"``): the shared
      ``_gdn_attention_per_layer_flops`` term (QKV/Z/B/A projections, causal conv, chunked
      delta-rule recurrence, output projection);
    * QSA full-attention layers: gated ``q_proj`` (hidden x 2*heads*head_dim), ``k_proj``/``v_proj``
      (hidden x kv_heads*head_dim) and ``o_proj``; the sparse GQA BMMs where the query at zero-based
      position ``t`` attends to ``ratio * min(indexer_budget / ratio, floor((t+1)/ratio))`` routed tokens
      plus the ``(t+1) mod ratio`` tokens of its incomplete causal tail, QK^T and PV each costing
      ``heads * head_dim`` MACs per key;
    * the QSA indexer is frozen (``requires_grad_(False)``; hard top-k has no gradient), so its
      ``index_qk_proj`` and its causal scoring of ``indexer_n_heads`` queries against the
      ``floor((t+1)/ratio)`` compressed keys are counted forward only (2 x MACs);
    * MoE per layer: router GEMM (hidden x num_experts), ``num_experts_per_tok`` routed plus one shared
      expert of ``3 * hidden * moe_intermediate_size`` / ``shared_expert_intermediate_size``, and the
      shared-expert gate (hidden x 1);
    * HyperConnections: two mixers per layer (attention and MoE), each with ``input_mix_weight_down``
      (hc_count*hidden x hc_lowrank), ``input_mix_weight_up`` (hc_lowrank x hc_count*hidden) and
      ``block_inject_weight`` (hc_count*hidden x hc_count); the final read mixer has no inject weight;
    * Engram PLE on ``ple_layer_ids``: ``key_proj`` (ple_embed_dim x hc_count*hidden), ``value_proj``
      (ple_embed_dim x hidden) and the depthwise causal convolution (hc_count*hidden x kernel); the
      table gather, hashing and norms are not GEMMs;
    * the untied language-model head (hidden x vocab). The checkpoint's MTP head is not loaded.

    Trainable GEMMs and attention BMMs cost 6 x MACs (forward + backward); activation recomputation
    is excluded (that is HFU). On the released configuration the formula counts 6.14B active GEMM
    parameters per token excluding the LM head (model card: 6B activated), implies a 125.8B backbone
    excluding the 51.2B Engram table (model card: 125B), and gives 42.0 GFLOPs/token at 4096 tokens,
    1.034x the dense identity ``6 x active``.

    Args:
        config: ``Qwen3_8_FlashNextTextConfig``, or a ``Qwen3_8_FlashNextConfig`` wrapper whose
            ``text_config`` is used.
        gbs: Number of sequences per step.
        seq_len: Tokens per sequence; defaults to ``config.max_position_embeddings``.

    Returns:
        Model FLOPs for one training step of ``gbs`` sequences of ``seq_len`` tokens.
    """
    if hasattr(config, "text_config") and not hasattr(config, "num_hidden_layers"):
        config = config.text_config

    if seq_len is None:
        seq_len = getattr(config, "max_position_embeddings", 4096)
    seq_len = int(seq_len)

    layers = config.num_hidden_layers
    hs = config.hidden_size
    vocab_size = config.vocab_size
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    hc_count = config.hc_count
    hc_lowrank = config.hc_lowrank
    flat_hs = hc_count * hs
    moe_inter = config.moe_intermediate_size
    shared_inter = config.shared_expert_intermediate_size
    n_routed = config.num_experts
    topk = config.num_experts_per_tok
    ratio = config.indexer_compress_ratio
    budget_blocks = config.indexer_budget // ratio
    index_heads = config.indexer_n_heads
    index_kv_heads = config.indexer_kv_heads
    index_head_dim = config.indexer_head_dim
    ple_embed_dim = config.ple_embed_dim
    ple_kernel = config.ple_conv_kernel_size
    num_ple_layers = len(config.ple_layer_ids)

    block_types = config.layers_block_type
    num_full_attn_layers = sum(1 for block_type in block_types if block_type == "attention")
    num_gdn_layers = layers - num_full_attn_layers

    # --- trainable GEMM MACs per token (6x) ---
    attn_linear = hs * heads * head_dim * 2 + 2 * hs * kv_heads * head_dim + heads * head_dim * hs
    experts = topk * 3 * hs * moe_inter + 3 * hs * shared_inter
    router = hs * n_routed + hs  # routed gate + shared-expert gate
    hyper_connection = 2 * (2 * flat_hs * hc_lowrank + flat_hs * hc_count)  # attention + MoE mixers
    final_mixer = 2 * flat_hs * hc_lowrank
    ple = num_ple_layers * (ple_embed_dim * flat_hs + ple_embed_dim * hs + flat_hs * ple_kernel)
    lm_head = hs * vocab_size
    trainable_macs_per_token = (
        num_full_attn_layers * attn_linear
        + layers * (experts + router + hyper_connection)
        + final_mixer
        + ple
        + lm_head
    )
    trainable_flops = 6 * gbs * seq_len * trainable_macs_per_token

    # --- GDN layers (already 6x, includes projections and recurrence) ---
    gdn_flops = num_gdn_layers * _gdn_attention_per_layer_flops(
        gbs,
        seq_len,
        hs,
        config.linear_key_head_dim,
        config.linear_value_head_dim,
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_conv_kernel_dim,
    )

    # --- sparse QSA BMMs (6x), summed over query positions ---
    routed_keys = ratio * _sum_min_floor_div(seq_len, ratio, budget_blocks) + _sum_mod(seq_len, ratio)
    attention_flops = 6 * gbs * num_full_attn_layers * 2 * heads * head_dim * routed_keys  # QK^T and PV

    # --- frozen indexer: forward only (2x) ---
    indexer_macs = seq_len * hs * (index_heads + index_kv_heads) * index_head_dim
    indexer_macs += index_heads * index_head_dim * _sum_min_floor_div(seq_len, ratio, None)
    indexer_flops = 2 * gbs * num_full_attn_layers * indexer_macs

    return float(trainable_flops + gdn_flops + attention_flops + indexer_flops)


def step3_5_flash_flops(config, gbs=1, seq_len=None):
    """Model FLOPs for Step3.5-Flash (GQA + sliding-window / full attention + MoE).

    Architecture: hybrid full/SWA attention with different head counts per type,
    MoE with shared expert on most layers, first few layers dense, SwiGLU.
    """
    if seq_len is None:
        seq_len = getattr(config, "max_position_embeddings", 2048)

    layers = config.num_hidden_layers
    hs = config.hidden_size
    vocab_size = config.vocab_size

    # Attention heads: full vs sliding may differ
    full_attn_heads = config.num_attention_heads
    attn_other = getattr(config, "attention_other_setting", None)
    if attn_other is not None and isinstance(attn_other, dict):
        sliding_attn_heads = attn_other.get("num_attention_heads", full_attn_heads)
    else:
        sliding_attn_heads = full_attn_heads
    num_query_groups = getattr(config, "num_attention_groups", full_attn_heads)
    head_dim = getattr(config, "head_dim", hs // full_attn_heads)
    sliding_window = getattr(config, "sliding_window", 512)

    # MoE config
    moe_top_k = getattr(config, "moe_top_k", 8)
    moe_ffn_hs = getattr(config, "moe_intermediate_size", 1280)
    share_expert_dim = getattr(config, "share_expert_dim", moe_ffn_hs)
    dense_ffn_hs = config.intermediate_size

    # Which layers are MoE? Parse moe_layers_enum (comma-separated string or list)
    moe_layers_raw = getattr(config, "moe_layers_enum", None)
    if moe_layers_raw is not None:
        if isinstance(moe_layers_raw, str):
            moe_layers_set = set(int(x.strip()) for x in moe_layers_raw.split(",") if x.strip())
        else:
            moe_layers_set = set(int(x) for x in moe_layers_raw)
    else:
        # Default: first 3 dense, rest MoE
        moe_layers_set = set(range(3, layers))

    # Layer types (first `layers` entries; remaining are MTP layers)
    layer_types = getattr(config, "layer_types", None)

    # MTP
    mtp_num_layers = getattr(config, "num_nextn_predict_layers", 0) or 0

    # --- Per-layer FLOPs ---
    total_attn = 0
    total_mlp = 0

    for i in range(layers):
        # Determine attention type
        if layer_types and i < len(layer_types):
            is_full = layer_types[i] == "full_attention"
        else:
            is_full = i % 4 == 0  # default: every 4th starting from 0

        if is_full:
            total_attn += attention_flops_calculator(
                seq_len,
                hs,
                full_attn_heads,
                num_query_groups,
                head_dim,
                is_swa=False,
            )
        else:
            total_attn += attention_flops_calculator(
                seq_len,
                hs,
                sliding_attn_heads,
                num_query_groups,
                head_dim,
                is_swa=True,
                swa_window_size=sliding_window,
            )

        # MLP: MoE or dense (SwiGLU = gate + up + down = 3 projections)
        if i in moe_layers_set:
            total_mlp += moe_mlp_flops_calculator(
                seq_len,
                hs,
                moe_ffn_hs,
                moe_top_k,
                gated_linear_unit=True,
            )
            # Shared expert (SwiGLU)
            total_mlp += 6 * seq_len * hs * share_expert_dim * 3
        else:
            total_mlp += 6 * seq_len * hs * dense_ffn_hs * 3

    # Vocab
    total_vocab = loss_flops_calculator(seq_len, hs, vocab_size)

    # MTP
    mtp_total = 0
    if mtp_num_layers > 0:
        # Embedding projection per MTP module (2*hs -> hs)
        mtp_total += 6 * seq_len * hs * 2 * hs * mtp_num_layers
        # Each MTP module has one transformer layer (attention + MoE MLP)
        mtp_total += (
            attention_flops_calculator(
                seq_len,
                hs,
                full_attn_heads,
                num_query_groups,
                head_dim,
                is_swa=False,
            )
            * mtp_num_layers
        )
        mtp_total += (
            moe_mlp_flops_calculator(
                seq_len,
                hs,
                moe_ffn_hs,
                moe_top_k,
                gated_linear_unit=True,
            )
            * mtp_num_layers
        )
        mtp_total += 6 * seq_len * hs * share_expert_dim * 3 * mtp_num_layers
        # Vocab per MTP module
        mtp_total += loss_flops_calculator(seq_len, hs, vocab_size) * mtp_num_layers

    return gbs * (total_attn + total_mlp + total_vocab + mtp_total)


def get_flops_formula_for_hf_config(config: Any) -> Callable | None:
    """
    Get the appropriate FLOPs formula function for a given HuggingFace config.

    Args:
        config: HuggingFace model config object

    Returns:
        The appropriate FLOPs formula function, or None for an unregistered
        composite config. Pass its text config explicitly when only text-backbone
        FLOPs are intended.
    """
    # Get config class name
    config_class_name = config.__class__.__name__

    # Map config class names to FLOPs formulas
    class_name_to_formula = {
        # GPT family
        "GPT2Config": gpt3_flops,
        "GPTNeoConfig": gpt3_flops,
        "GPTNeoXConfig": gpt3_flops,
        "GPTJConfig": gpt3_flops,
        # Llama family
        "LlamaConfig": llama2_flops,  # Llama 1 and 2 use same formula
        # Mixtral (MoE)
        "MixtralConfig": mixtral_flops,
        # Qwen family
        "Qwen2Config": qwen3_flops,
        "Qwen3Config": qwen3_flops,
        "Qwen3MoeConfig": qwen3_flops,
        "Qwen3_5Config": qwen3_5_flops,
        "Qwen3_5MoeConfig": qwen3_5_flops,
        "Qwen3NextConfig": qwen3_5_flops,  # Qwen3.5 Small 4B/9B (GDN + MoE)
        # Qwen3.8-Flash-Next (GDN + compressed-block QSA + HyperConnections + Engram PLE + MoE);
        # the multimodal wrapper and the pre-rename ``qwen4_exp`` aliases use text_config.
        "Qwen3_8_FlashNextConfig": qwen3_8_flash_next_flops,
        "Qwen3_8_FlashNextTextConfig": qwen3_8_flash_next_flops,
        "Qwen3_8_FlashNextLegacyConfig": qwen3_8_flash_next_flops,
        "Qwen3_8_FlashNextLegacyTextConfig": qwen3_8_flash_next_flops,
        "Qwen3VLMoeConfig": qwen3_flops,  # Qwen3 VL 235B text backbone
        "Qwen3VLMoeTextConfig": qwen3_flops,
        "Qwen3VLConfig": qwen3_flops,
        "Qwen3VLTextConfig": qwen3_flops,
        # BERT family
        "BertConfig": bert_flops,
        "RobertaConfig": bert_flops,
        "AlbertConfig": bert_flops,
        "ElectraConfig": bert_flops,
        # DeepSeek V3 / V3.2
        "DeepseekV3Config": deepseekv3_flops,
        # DeepSeek V4.1 (CSA2 sparse attention + single-pass mHC + Engram + MoE; wrapper uses text_config)
        "DeepseekV41Config": deepseek_v41_flops,
        "DeepseekV41TextConfig": deepseek_v41_flops,
        # GPT-OSS
        "GptOssConfig": gpt_oss_flops,
        # GLM family
        "Glm4Config": qwen3_flops,  # Dense GQA + SwiGLU (e.g. GLM-4-9B-0414)
        "Glm4MoeConfig": glm4_moe_flops,  # GLM-4.7 (GQA + MoE)
        "Glm4MoeLiteConfig": mla_moe_flops,  # GLM-4.7-Flash (MLA + MoE)
        "GlmMoeDsaConfig": mla_moe_flops,  # GLM-5 (MLA + MoE)
        # MiniMax-M2 / M2.5
        "MiniMaxM2Config": minimax_m2_flops,
        # MLA + MoE models (Mistral Small 4, Kimi K2.5)
        "Mistral3Config": mla_moe_flops,  # Mistral Small 4 (VL wrapper, extracts text_config)
        "KimiK2Config": mla_moe_flops,  # Kimi K2 / K2.5
        "KimiK25Config": mla_moe_flops,
        "KimiK3TextConfig": kimi_k3_flops,  # Kimi K3 (hybrid KDA/MLA + latent MoE)
        "KimiK3Config": kimi_k3_flops,  # Kimi K3 multimodal wrapper (text_config)
        # Step3.5-Flash
        "Step3p5Config": step3_5_flash_flops,
        "LongcatFlashConfig": mla_moe_flops,  # MLA + MoE
        # T5 family (encoder-decoder)
        "T5Config": transformer_flops,
        "MT5Config": transformer_flops,
        # Nemotron
        "NemotronConfig": nemotron_flops,
        "NemotronHConfig": nemotronh_flops,
        # General transformer fallback
        "OPTConfig": transformer_flops,
        "BloomConfig": transformer_flops,
        "FalconConfig": transformer_flops,
    }

    # Try exact match first
    formula = class_name_to_formula.get(config_class_name)

    # If no exact match, try to match by model_type as fallback
    if formula is None:
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config) and get_text_config() is not config:
            return None
        formula = transformer_flops

    return formula
