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

from __future__ import annotations

from transformers import PretrainedConfig


class DeepseekV4Config(PretrainedConfig):
    """Configuration class for DeepSeek V4.

    DeepSeek V4 differs from V3/V3.2 in several key ways:
    - Attention: GQA (num_key_value_heads=1) with Q-LoRA and grouped O-LoRA instead of MLA.
    - No dense MLP layers: all transformer blocks use MoE FFN.
    - Per-layer sliding/compressed attention via compress_ratios.
    - First num_hash_layers use hash-clustering (HC) attention for dynamic token grouping.
    - Learnable attention sink token for sliding-window layers.
    - New MoE gate scoring: sqrtsoftplus with noaux_tc routing.
    - Next-n prediction (MTP) layers for multi-token prediction.
    """

    model_type = "deepseek_v4"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 129280,
        hidden_size: int = 4096,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 43,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 1,
        # V4 uses a single head_dim for Q, K, V (no separate nope/rope dims at the config level)
        head_dim: int = 512,
        qk_rope_head_dim: int = 64,
        # Q LoRA: hidden -> q_lora_rank -> n_heads * head_dim
        q_lora_rank: int = 1024,
        # O LoRA: n_heads * head_dim -> o_lora_rank -> hidden (with o_groups groups)
        o_lora_rank: int = 1024,
        o_groups: int = 8,
        # MoE routing
        n_routed_experts: int = 256,
        n_shared_experts: int = 1,
        num_experts_per_tok: int = 6,
        routed_scaling_factor: float = 1.5,
        norm_topk_prob: bool = True,
        scoring_func: str = "sqrtsoftplus",
        topk_method: str = "noaux_tc",
        # FFN activation
        hidden_act: str = "silu",
        swiglu_limit: float = 10.0,
        # Position encoding
        max_position_embeddings: int = 1048576,
        rope_theta: float = 10000.0,
        rope_scaling: dict | None = None,
        # Compressed/sliding-window attention (per-layer)
        # compress_ratios[i]: 0 = full attention, >0 = compressed local window
        compress_rope_theta: float = 160000.0,
        compress_ratios: list | None = None,
        sliding_window: int = 128,
        # Hash-clustering attention for the first num_hash_layers layers
        num_hash_layers: int = 3,
        hc_eps: float = 1e-6,
        hc_mult: int = 4,
        hc_sinkhorn_iters: int = 20,
        # Compressor/Indexer for compress-ratio attention layers (not yet implemented)
        index_head_dim: int = 128,
        index_n_heads: int = 64,
        index_topk: int = 512,
        # Optional DeepSeek-V4-Flash-Vision-Exp tower and image-token bridge.
        vision_n_layers: int = 0,
        vision_dim: int = 1024,
        vision_n_heads: int = 16,
        vision_inter_dim: int = 2816,
        vision_patch_size: int = 14,
        vision_rope_theta: float = 10000.0,
        vision_downsample_ratio: int = 3,
        vision_max_n_token: int = 384,
        vision_min_pixels: int = 147456,
        vision_max_wh_ratio: float | None = 8,
        # Multi-token prediction layers appended after the main layers
        num_nextn_predict_layers: int = 1,
        # Standard options
        rms_norm_eps: float = 1e-6,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        pad_token_id: int | None = None,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        pretraining_tp: int = 1,
        tie_word_embeddings: bool = False,
        initializer_range: float = 0.02,
        torch_dtype: str | None = None,
        **kwargs,
    ):
        dtype = kwargs.pop("dtype", None)
        resolved_dtype = dtype if dtype is not None else torch_dtype
        if resolved_dtype is None:
            resolved_dtype = "bfloat16"

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.routed_scaling_factor = routed_scaling_factor
        self.norm_topk_prob = norm_topk_prob
        self.scoring_func = scoring_func
        self.topk_method = topk_method
        self.hidden_act = hidden_act
        self.swiglu_limit = swiglu_limit
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.compress_rope_theta = compress_rope_theta
        self.compress_ratios = compress_ratios or []
        self.sliding_window = sliding_window
        self.num_hash_layers = num_hash_layers
        self.hc_eps = hc_eps
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk = index_topk
        self.vision_n_layers = vision_n_layers
        self.vision_dim = vision_dim
        self.vision_n_heads = vision_n_heads
        self.vision_inter_dim = vision_inter_dim
        self.vision_patch_size = vision_patch_size
        self.vision_rope_theta = vision_rope_theta
        self.vision_downsample_ratio = vision_downsample_ratio
        self.vision_max_n_token = vision_max_n_token
        self.vision_min_pixels = vision_min_pixels
        self.vision_max_wh_ratio = vision_max_wh_ratio
        self.num_nextn_predict_layers = num_nextn_predict_layers
        self.rms_norm_eps = rms_norm_eps
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pretraining_tp = pretraining_tp
        self.initializer_range = initializer_range

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            dtype=resolved_dtype,
            **kwargs,
        )
