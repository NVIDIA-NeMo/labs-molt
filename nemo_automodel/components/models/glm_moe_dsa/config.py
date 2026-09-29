# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Automodel's own GLM MoE DSA config."""

from transformers.configuration_utils import PretrainedConfig

__all__ = ["GlmMoeDsaConfig"]


class GlmMoeDsaConfig(PretrainedConfig):
    """Configuration for GLM MoE DSA (DeepSeek-style sparse attention).

    Standalone rather than a subclass of the transformers config, matching the
    other Automodel-owned configs: Automodel ships its own GLM DSA modeling
    code, so its field protocol must not move when transformers releases do.
    Subclassing also inherited ``attribute_map`` entries, one of which is
    actively harmful -- transformers maps ``head_dim`` onto ``qk_rope_head_dim``
    while the released GLM-5.2 ``config.json`` sets both (``head_dim: 192``, the
    full QK head dim, and ``qk_rope_head_dim: 64``), so the alias overwrote the
    rope dim with 192 and the DSA indexer's nope split went negative
    (``index_head_dim - qk_rope_head_dim = 128 - 192``). Only the expert-count
    alias is kept here.
    """

    model_type = "glm_moe_dsa"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_local_experts": "n_routed_experts"}

    def __init__(
        self,
        vocab_size: int = 154880,
        hidden_size: int = 6144,
        intermediate_size: int = 12288,
        moe_intermediate_size: int = 2048,
        num_hidden_layers: int = 78,
        num_attention_heads: int = 64,
        num_key_value_heads: int = 64,
        # MoE routing
        n_shared_experts: int = 1,
        n_routed_experts: int = 256,
        routed_scaling_factor: float = 2.5,
        n_group: int = 1,
        topk_group: int = 1,
        num_experts_per_tok: int = 8,
        norm_topk_prob: bool = True,
        mlp_layer_types: list[str] | None = None,
        # MLA projections
        kv_lora_rank: int = 512,
        q_lora_rank: int = 2048,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 192,
        v_head_dim: int = 256,
        # DSA lightning indexer
        index_topk: int = 2048,
        index_head_dim: int = 128,
        index_n_heads: int = 32,
        indexer_types: list[str] | None = None,
        index_topk_pattern: str | list[str] | None = None,
        index_topk_freq: int = 1,
        # Misc
        hidden_act: str = "silu",
        max_position_embeddings: int = 202752,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-5,
        use_cache: bool = True,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        rope_parameters: dict | None = None,
        pad_token_id: int | None = None,
        bos_token_id: int | None = 0,
        eos_token_id: int | list[int] | None = 1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        """Initialize the config.

        Args:
            mlp_layer_types: Per-layer ``"dense"``/``"sparse"`` MLP pattern.
                Derived as 3 dense layers then sparse when omitted.
            indexer_types: Per-layer ``"full"``/``"shared"`` indexer pattern.
                Derived from ``index_topk_pattern`` / ``index_topk_freq`` when
                omitted.
            index_topk_pattern: Indexer pattern as an ``F``/``S`` string or an
                explicit list.
            index_topk_freq: Stride used when no pattern is given: layer 0 and
                every ``index_topk_freq``-th layer after it are ``"full"``.
            rope_parameters: RoPE settings (``rope_theta``, ``rope_type``, plus
                the YaRN keys when scaled), carried as a plain dict.
            **kwargs: Forwarded to :class:`~transformers.PretrainedConfig`;
                checkpoint keys Automodel does not read stay available on the
                instance.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        # The attention layers size their Q/K projections from the combined
        # width, so keep it on the config instead of recomputing per layer.
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim

        self.index_topk = index_topk
        self.index_head_dim = index_head_dim
        self.index_n_heads = index_n_heads
        self.index_topk_pattern = index_topk_pattern
        self.index_topk_freq = index_topk_freq

        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.rope_parameters = rope_parameters

        # Released checkpoints spell both lists out; these defaults only cover
        # configs that omit them.
        if mlp_layer_types is None:
            dense_layers = min(3, num_hidden_layers)
            mlp_layer_types = ["dense"] * dense_layers + ["sparse"] * (num_hidden_layers - dense_layers)
        self.mlp_layer_types = mlp_layer_types

        if indexer_types is None:
            if index_topk_pattern is not None:
                indexer_types = (
                    [{"F": "full", "S": "shared"}[char] for char in index_topk_pattern]
                    if isinstance(index_topk_pattern, str)
                    else list(index_topk_pattern)
                )
            else:
                freq = max(int(index_topk_freq), 1)
                indexer_types = ["full" if (max(i - 1, 0) % freq) == 0 else "shared" for i in range(num_hidden_layers)]
        self.indexer_types = indexer_types

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
