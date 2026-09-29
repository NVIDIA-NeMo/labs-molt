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

from dataclasses import dataclass, replace
from typing import Any, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.models.common import (
    BackendConfig,
    compute_lm_head_logits,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import (
    TieSupport,
    reject_unsupported_tie_word_embeddings,
)
from nemo_automodel.components.models.deepseek_v3.rope_utils import (
    freqs_cis_from_position_ids,
    precompute_freqs_cis,
)
from nemo_automodel.components.models.glm_moe_dsa.cp import shard_glm_dsa_packed_cp_batch
from nemo_automodel.components.models.glm_moe_dsa.layers import GlmMoeDsaMLA
from nemo_automodel.components.models.glm_moe_dsa.optimized_kernels import prepare_cudnn_dsa_packed_metadata
from nemo_automodel.components.models.glm_moe_dsa.state_dict_adapter import GlmMoeDsaStateDictAdapter
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MLP, MoE, MoEConfig
from nemo_automodel.components.utils.model_utils import squeeze_input_for_thd
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _uses_indexshare(config: GlmMoeDsaConfig) -> bool:
    """Return whether the model has layers that reuse another layer's DSA indices."""
    indexer_types = getattr(config, "indexer_types", None)
    return indexer_types is not None and "shared" in indexer_types


class Block(nn.Module):
    def __init__(self, layer_idx: int, config: GlmMoeDsaConfig, moe_config: MoEConfig, backend: BackendConfig):
        super().__init__()
        # IndexShare: per-layer indexer mode from `config.indexer_types`. A "shared" layer
        # owns no indexer and reuses the previous "full" layer's top-k selection. Absent the
        # field (e.g. GLM-5.1, which runs a full indexer every layer), every layer is "full".
        indexer_types = getattr(config, "indexer_types", None)
        self.skip_topk = indexer_types is not None and indexer_types[layer_idx] == "shared"
        self.self_attn = GlmMoeDsaMLA(config, backend, skip_topk=self.skip_topk)

        # Thread dtype from config.torch_dtype so the block's own params stay
        # aligned with the rest of the model (fp32 under fp32 master weights).
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        mlp_layer_types = getattr(config, "mlp_layer_types", None)
        if mlp_layer_types is not None:
            is_moe_layer = mlp_layer_types[layer_idx] == "sparse"
        else:
            first_k_dense_replace = getattr(config, "first_k_dense_replace", 0)
            is_moe_layer = layer_idx >= first_k_dense_replace

        if is_moe_layer:
            self.mlp = MoE(moe_config, backend)
        else:
            self.mlp = MLP(config.hidden_size, config.intermediate_size, backend.linear, dtype=dtype)

        self.input_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.post_attention_layernorm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the block and return ``(hidden_states, topk_indices)``.

        ``topk_indices`` is this layer's DSA selection — freshly computed on "full" layers,
        or ``prev_topk_indices`` passed through on "shared" layers — so the caller can thread
        it to subsequent shared layers (GLM IndexShare).
        """
        if attention_mask is not None and padding_mask is None:
            padding_mask = attention_mask.bool().logical_not()

        attn_out, topk_indices = self.self_attn(
            x=self.input_layernorm(x),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            prev_topk_indices=prev_topk_indices,
            return_topk_indices=True,
            **attn_kwargs,
        )
        x = x + attn_out

        mlp_out = self._mlp(x=self.post_attention_layernorm(x), padding_mask=padding_mask)
        x = x + mlp_out
        return x, topk_indices

    def _mlp(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        if isinstance(self.mlp, MLP):
            return self.mlp(x)
        else:
            assert isinstance(self.mlp, MoE)
            return self.mlp(x, padding_mask)

    def init_weights(self, buffer_device: torch.device):
        for norm in (self.input_layernorm, self.post_attention_layernorm):
            norm.reset_parameters()
        self.self_attn.init_weights(buffer_device)
        self.mlp.init_weights(buffer_device)


class GlmMoeDsaModel(nn.Module):
    def __init__(
        self,
        config: GlmMoeDsaConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.config = config
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")

        # Resolve model dtype once; thread it explicitly to every sub-module
        # so fp32 master weights work even when construction is not wrapped in
        # local_torch_dtype().
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=config.moe_intermediate_size,
            n_routed_experts=config.n_routed_experts,
            n_shared_experts=config.n_shared_experts,
            n_activated_experts=config.num_experts_per_tok,
            n_expert_groups=config.n_group,
            n_limited_groups=config.topk_group,
            train_gate=True,
            gate_bias_update_factor=1e-3,
            score_func="sigmoid",
            route_scale=config.routed_scaling_factor,
            aux_loss_coeff=0.0,
            norm_topk_prob=config.norm_topk_prob,
            expert_bias=False,
            router_bias=False,
            expert_activation="swiglu",
            softmax_before_topk=False,
            router_weights_fp32=True,
            dtype=model_dtype,
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=model_dtype)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(config.num_hidden_layers):
            self.layers[str(layer_id)] = Block(layer_id, config, self.moe_config, backend)
        self.norm = initialize_rms_norm_module(
            backend.rms_norm, config.hidden_size, eps=config.rms_norm_eps, dtype=model_dtype
        )

        self.max_seq_len = config.max_position_embeddings
        self.qk_rope_head_dim = config.qk_rope_head_dim

        if hasattr(config, "rope_parameters") and config.rope_parameters is not None:
            rope_theta = config.rope_parameters["rope_theta"]
        else:
            rope_theta = config.rope_theta

        rope_scaling = getattr(config, "rope_scaling", None)

        self.freqs = precompute_freqs_cis(
            qk_rope_head_dim=self.qk_rope_head_dim,
            max_seq_len=self.max_seq_len,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the decoder stack, returning ``(hidden_states, topk_indices)``.

        Args:
            input_ids: Token IDs with shape ``[batch, sequence]`` (or packed
                ``[tokens]``), or previous-stage hidden states with the matching token
                axes plus ``[hidden]`` when this pipeline stage has no embedding.
            position_ids: Optional integer positions with the same token axes as
                ``input_ids``. Packed THD callers supply shape ``[tokens]``.
            attention_mask: Optional token mask with shape ``[batch, sequence]`` or
                additive attention mask broadcastable to ``[batch, heads, query, key]``.
            padding_mask: Optional boolean mask with the input token axes; ``True``
                marks padding. Packed THD callers supply shape ``[tokens]``.
            prev_topk_indices: Optional previous pipeline stage's IndexShare selection,
                shaped ``[batch, sequence, K]`` or packed ``[tokens, 1, K]``. Packed
                values are global THD padded-storage K/V coordinates.
            **attn_kwargs: Attention layout metadata. Packed THD uses ``cu_seqlens``
                and optional ``cu_seqlens_padded`` of shape ``[sequences + 1]``, plus
                optional ``glm_dsa_cp_query_indices`` of shape ``[tokens]`` containing
                global padded-storage query coordinates.

        Returns:
            Hidden states with the input token axes plus ``[hidden]``, and the latest
            IndexShare top-k tensor in the corresponding BSHD or packed THD layout.
        """
        if position_ids is None:
            position_ids = (
                torch.arange(0, input_ids.shape[1], device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
            )

        freqs_cis = freqs_cis_from_position_ids(
            position_ids,
            self.freqs.to(position_ids.device),
            qkv_format=attn_kwargs.get("qkv_format", "bshd"),
            for_fused_rope=self.backend.rope_fusion,
            cp_size=attn_kwargs.get("cp_size", 1),
        )

        h = self.embed_tokens(input_ids) if self.embed_tokens is not None else input_ids

        if self.backend.attn == "cudnn" and attn_kwargs.get("qkv_format") == "thd":
            cu_seqlens = attn_kwargs.get("cu_seqlens")
            if cu_seqlens is None:
                raise ValueError("cuDNN DSA requires 'cu_seqlens' for packed THD input.")
            cu_seqlens = cu_seqlens.flatten().to(device=h.device, dtype=torch.int32).contiguous()
            query_indices = attn_kwargs.get("glm_dsa_cp_query_indices")
            if query_indices is not None:
                query_indices = query_indices.flatten().to(device=h.device, dtype=torch.int32).contiguous()
            cu_seqlens_padded = attn_kwargs.get("cu_seqlens_padded")
            if cu_seqlens_padded is not None:
                cu_seqlens_padded = cu_seqlens_padded.flatten().to(device=h.device, dtype=torch.int32).contiguous()
            cudnn_padding_mask = None
            if padding_mask is not None:
                cudnn_padding_mask = padding_mask.flatten().to(device=h.device, dtype=torch.bool).contiguous()
            cp_size = int(attn_kwargs.get("cp_size", 1))
            packed_metadata = prepare_cudnn_dsa_packed_metadata(
                cu_seqlens,
                h.shape[0] * cp_size,
                query_indices=query_indices,
                cu_seqlens_padded=cu_seqlens_padded,
                padding_mask=cudnn_padding_mask,
            )
            attn_kwargs = dict(attn_kwargs)
            attn_kwargs["cu_seqlens"] = cu_seqlens
            if query_indices is not None:
                attn_kwargs["glm_dsa_cp_query_indices"] = query_indices
            if cu_seqlens_padded is not None:
                attn_kwargs["cu_seqlens_padded"] = cu_seqlens_padded
            attn_kwargs["_cudnn_dsa_packed_metadata"] = packed_metadata
            attn_kwargs["_cudnn_dsa_topk_length"] = packed_metadata.causal_lengths.clamp_max(
                int(self.config.index_topk)
            ).contiguous()
            attn_kwargs["_cudnn_dsa_all_rows_nonempty"] = packed_metadata.all_rows_nonempty
            attn_kwargs["_cudnn_dsa_valid_row_indices"] = packed_metadata.valid_row_indices

        # IndexShare: thread the most recent "full" layer's top-k selection forward so the
        # following "shared" layers can reuse it. Legacy GLM configs have no shared layers, so
        # avoid retaining and propagating their per-layer selections.
        uses_indexshare = _uses_indexshare(self.config)
        topk_indices = prev_topk_indices if uses_indexshare else None
        for layer in self.layers.values():
            h, layer_topk_indices = layer(
                x=h,
                freqs_cis=freqs_cis,
                attention_mask=attention_mask,
                padding_mask=padding_mask,
                prev_topk_indices=topk_indices if uses_indexshare else None,
                **attn_kwargs,
            )
            if uses_indexshare:
                topk_indices = layer_topk_indices

        h = self.norm(h) if self.norm else h
        return h, topk_indices

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")

        with buffer_device:
            if self.embed_tokens is not None:
                nn.init.normal_(self.embed_tokens.weight)
            if self.norm is not None:
                self.norm.reset_parameters()

        for layer in self.layers.values():
            if layer is not None:
                layer.init_weights(buffer_device=buffer_device)

    def update_moe_gate_bias(self) -> None:
        """Update the noaux router correction bias of each local MoE layer; dense layers and disabled gates are skipped."""
        with torch.no_grad():
            for block in self.layers.values():
                if isinstance(block.mlp, MoE) and block.mlp.gate.bias_update_factor > 0:
                    block.mlp.gate.update_bias()


class GlmMoeDsaForCausalLM(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin):
    tie_word_embeddings_support: TieSupport = TieSupport.UNTIED_ONLY
    _packed_cp_attn_backends = ("tilelang", "cudnn")
    _keep_in_fp32_modules_strict = ["e_score_correction_bias"]

    @dataclass(frozen=True)
    class ModelCapabilities:
        """Declared parallelism capabilities for this model class."""

        supports_tp: bool = False
        supports_cp: bool = True
        supports_pp: bool = True
        supports_ep: bool = True
        supports_thd: bool = True

    @classmethod
    def from_config(
        cls,
        config: GlmMoeDsaConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, moe_config, backend, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args,
        **kwargs,
    ):
        config = GlmMoeDsaConfig.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def __init__(
        self,
        config: GlmMoeDsaConfig,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        reject_unsupported_tie_word_embeddings(type(self), config)
        resolved_backend = backend or BackendConfig()
        # HF computes the GLM router projection and selected mixture weights in fp32.
        if resolved_backend.gate_precision is None:
            resolved_backend = replace(resolved_backend, gate_precision=torch.float32)
        self.backend = resolved_backend
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model = GlmMoeDsaModel(
            config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)
        self.lm_head = initialize_linear_module(
            self.backend.linear, config.hidden_size, config.vocab_size, bias=False, dtype=model_dtype
        )
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = GlmMoeDsaStateDictAdapter(
                self.config, self.model.moe_config, self.backend, dtype=model_dtype
            )

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def update_moe_gate_bias(self) -> None:
        """Delegate the noaux router correction-bias update to the inner model."""
        self.model.update_moe_gate_bias()

    def should_pack_validation_with_training(self) -> bool:
        """Return whether validation must use the optimized packed THD layout."""
        return getattr(self.backend, "attn", None) in ("tilelang", "cudnn")

    def prepare_model_inputs_for_cp(
        self,
        batch: dict[str, Any],
        *,
        num_chunks: int = 1,
    ) -> dict[str, Any]:
        """Attach GLM DSA's packed THD context-parallel batch sharder.

        Args:
            batch: The batch dict.
            num_chunks: Number of chunks for load-balanced CP sharding.
        """
        from functools import partial  # noqa: PLC0415

        from nemo_automodel.components.distributed.context_parallel.sharder import (  # noqa: PLC0415
            ContextParallelSharder,
            contiguous_local_indices,
        )

        attn_backend = getattr(self.backend, "attn", None)
        if attn_backend not in self._packed_cp_attn_backends:
            raise NotImplementedError(
                "GLM DSA packed context parallelism requires backend.attn in {'tilelang', 'cudnn'}; "
                f"got backend.attn={attn_backend!r}."
            )

        cp_sharder = ContextParallelSharder(
            shard_batch=partial(
                shard_glm_dsa_packed_cp_batch,
                num_chunks=int(num_chunks),
            ),
            # Contiguous over the packed THD token axis: rank r keeps
            # tokens [r * T/cp, (r + 1) * T/cp).
            local_token_global_indices=contiguous_local_indices,
        )
        return {"cp_sharder": cp_sharder}

    def _is_pipeline_parallel_stage(self) -> bool:
        """True when this module is a trimmed pipeline-parallel stage (not the whole model)."""
        if self.lm_head is None:
            return True
        if self.model.embed_tokens is None:
            return True
        try:
            return len(self.model.layers) != int(self.config.num_hidden_layers)
        except TypeError:
            return False

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Declare PP inter-stage I/O metas, adding a top-k carry only for IndexShare models.

        IndexShare models additionally receive and emit the previous "full" layer's top-k
        selection so a stage that begins with a "shared" layer has the indices it needs. Models
        with all-full indexers need only the hidden-state channel.

        Args:
            is_first: Whether this module is the first pipeline stage.
            microbatch_size: Number of sequences in the pipeline microbatch.
            seq_len: Sequence or packed-token length represented by the metadata.
            dtype: Hidden-state and logits dtype.

        Returns:
            Pair of input and output metadata tuples. Optimized THD stages use hidden-state
            tensors of shape ``[seq_len, hidden]`` and fixed-width top-k tensors of shape
            ``[seq_len, 1, index_topk]``. Dense BSHD stages use ``[microbatch, seq_len, hidden]``
            and ``[microbatch, seq_len, min(index_topk, seq_len)]``. Top-k carry metadata is
            float32 because pipeline receive buffers require gradients.
        """
        hidden_size = self.config.hidden_size
        vocab_size = self.config.vocab_size
        # Optimized indexers always return ``index_topk`` columns. Under TileLang
        # CP the query length is sharded (for example 4096 / cp8 = 512), while
        # K/V are gathered inside the model, so capping by the local query
        # length would under-declare the inter-stage carry shape.
        index_topk = int(self.config.index_topk)
        optimized_thd = self.backend.attn in ("tilelang", "cudnn")
        topk = index_topk if optimized_thd else min(index_topk, seq_len)

        def meta(shape: tuple[int, ...], dt: torch.dtype) -> torch.Tensor:
            return torch.empty(*shape, device="meta", dtype=dt)

        # The inter-stage tensor RANK matches the attention backend's data format, so each stage's
        # forward emits its natural tensors and no per-boundary reshape is needed:
        #   * TileLang/cuDNN DSA run in THD (packed; batch folded into the token axis) -> 2D hidden
        #     ``[T, H]`` and fixed-width top-k ``[T, 1, topk]``. Both backends imply THD.
        #   * sdpa/te/eager run dense bshd -> 3D hidden ``[B, S, H]`` and top-k ``[B, S, topk]``.
        # Top-k indices cross the boundary as float32: torch.distributed.pipelining calls
        # ``requires_grad_(True)`` on recv buffers and int dtypes can't require grad; float32 holds
        # the index values losslessly and ``forward`` casts back (int32 for optimized THD, int64 dense).
        if optimized_thd:
            hidden_meta = meta((seq_len, hidden_size), dtype)
            topk_meta = meta((seq_len, 1, topk), torch.float32)
        else:
            hidden_meta = meta((microbatch_size, seq_len, hidden_size), dtype)
            topk_meta = meta((microbatch_size, seq_len, topk), torch.float32)

        uses_indexshare = _uses_indexshare(self.config)
        if is_first:
            inputs_meta = (meta((microbatch_size, seq_len), torch.long),)
        elif uses_indexshare:
            inputs_meta = (hidden_meta, topk_meta)
        else:
            inputs_meta = (hidden_meta,)

        if self.lm_head is not None:
            # The last stage emits logits; compute_lm_head_logits restores [1, T, V] under THD.
            outputs_meta = (meta((microbatch_size, seq_len, vocab_size), dtype),)
        elif uses_indexshare:
            outputs_meta = (hidden_meta, topk_meta)
        else:
            outputs_meta = (hidden_meta,)

        return inputs_meta, outputs_meta

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *carry: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: bool | None = None,
        **attn_kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...] | torch.Tensor:
        """Forward pass.

        Single process (no pipeline parallelism): returns
        :class:`~transformers.modeling_outputs.CausalLMOutputWithPast`, threading the IndexShare
        top-k internally (seeded ``None``).

        Pipeline parallelism: ``input_ids`` is the upstream hidden state on non-first stages.
        IndexShare models additionally use ``*carry`` for the previous stage's running top-k
        selection. Non-last stages return the declared pipeline outputs and the last stage returns
        the ``logits`` tensor.

        Args:
            input_ids: Token-ID tensor of shape ``[batch, sequence]`` (BSHD) or ``[1, tokens]``
                (packed THD) on the first stage. On later stages, the upstream hidden-state
                tensor has shape ``[batch, sequence, hidden]`` or ``[tokens, hidden]``.
            carry: Optional top-k tensor carried from the previous pipeline stage. Optimized
                THD backends use float32 ``[tokens, 1, index_topk]``; dense BSHD uses float32
                ``[batch, sequence, min(index_topk, sequence)]``.
            position_ids: Optional position-ID tensor of shape ``[batch, sequence]`` or
                ``[1, tokens]`` for packed THD.
            attention_mask: Optional key-mask tensor of shape ``[batch, sequence]`` or additive
                mask of shape ``[batch, 1, sequence, sequence]``.
            padding_mask: Optional padding-mask tensor of shape ``[batch, sequence]`` or
                ``[1, tokens]`` for packed THD.
            logits_to_keep: If ``0``, project all positions; else only the last ``logits_to_keep``.
                A tensor value contains the one-dimensional token indices to project.
            output_hidden_states: When set (single-process), carry final hidden states on the output.
            **attn_kwargs: Additional attention metadata forwarded to the base model. Packed THD
                uses an int32 ``cu_seqlens`` tensor of shape ``[sequences + 1]`` and
                ``qkv_format="thd"``.

        Returns:
            Single-process execution returns a causal-LM output whose logits tensor has shape
            ``[batch, sequence, vocab]``. A non-last pipeline stage returns hidden states and
            float32 top-k carry tensors in the layouts described above; the last stage returns
            a logits tensor of shape ``[batch, sequence, vocab]``.
        """

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        uses_indexshare = _uses_indexshare(self.config)

        # Carry-in arrives as float32 (see get_pipeline_stage_metas, where the pipeline recv
        # buffer must be a grad-capable dtype); restore the int64 index values.
        carry_in = carry[0] if uses_indexshare and carry else None
        is_thd = attn_kwargs.get("qkv_format") == "thd"

        prev_topk_indices = None
        if carry_in is not None:
            # The carry arrives in the backend's natural top-k layout (THD: [T, 1, topk]; bshd:
            # [B, S, topk]) as float32. Optimized THD kernels require int32 indices; the dense
            # path uses int64. Only the dtype differs -- no reshape (see get_pipeline_stage_metas).
            prev_topk_indices = carry_in.to(torch.int32) if is_thd else carry_in.to(torch.int64)

        # THD: squeeze the leading batch dim on EVERY stage. First stage ``input_ids`` is token ids
        # [1, T]; later stages receive the upstream hidden state [1, T, H] (the 3D pipeline meta).
        # squeeze_input_for_thd handles both (plain ``.squeeze(0)``) and also squeezes
        # position_ids / cu_seqlens so the 2D-THD DSA layers get consistent shapes on every stage.
        if is_thd:
            # squeeze_input_for_thd mutates attn_kwargs in place; under activation-checkpointing the
            # stage forward is recomputed for backward, and a mutated (already-squeezed) dict on the
            # second pass yields different input metadata -> CheckpointError. Pass a fresh copy so
            # the forward and its recompute are deterministic.
            attn_kwargs = dict(attn_kwargs)
            input_ids, position_ids, padding_mask, attn_kwargs = squeeze_input_for_thd(
                input_ids, position_ids, padding_mask, attn_kwargs
            )
            attention_mask = None

        hidden, topk_indices = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            padding_mask=padding_mask,
            prev_topk_indices=prev_topk_indices,
            **attn_kwargs,
        )

        if self._is_pipeline_parallel_stage():
            if not uses_indexshare:
                if self.lm_head is not None:
                    return compute_lm_head_logits(self.lm_head, hidden, logits_to_keep, is_thd=is_thd).logits
                return hidden

            # The top-k carry is non-differentiable (integer indices transported as float32), but
            # torch.distributed.pipelining treats every float inter-stage tensor as an activation
            # and demands a gradient for it on backward. We add zero-weight autograd links so the
            # carry it RECEIVES and the carry it SENDS both get a DEFINED (zero) gradient instead of
            # ``None`` — values are unchanged (``+ x.sum() * 0.0``).
            if self.lm_head is not None:
                # Last stage: emit logits for the pipeline loss.
                logits = compute_lm_head_logits(self.lm_head, hidden, logits_to_keep, is_thd=is_thd).logits
                if carry_in is not None:
                    logits = logits + (carry_in.float().sum() * 0.0).to(logits.dtype)
                return logits
            # Non-last stage: emit (hidden, float32 top-k carry) to the next stage. The tensors are
            # already in the backend's natural pipeline shape (THD: [T, H] + [T, 1, topk]; bshd:
            # [B, S, H] + [B, S, topk]) per get_pipeline_stage_metas, so no reshape is needed.
            # Optimized THD indexers always return the fixed ``index_topk`` width declared by
            # the pipeline metadata, including when it exceeds an individual packed sequence.
            zero_from_hidden = hidden.float().sum() * 0.0  # connected to grad-bearing hidden
            carry_out = topk_indices.to(torch.float32) + zero_from_hidden  # requires grad, value unchanged
            if carry_in is not None:
                hidden = hidden + (carry_in.float().sum() * 0.0).to(hidden.dtype)  # defines grad(carry_in)
            return hidden, carry_out

        return compute_lm_head_logits(
            self.lm_head, hidden, logits_to_keep, is_thd=is_thd, output_hidden_states=output_hidden_states
        )

    @torch.no_grad()
    def initialize_weights(
        self, buffer_device: torch.device | None = None, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")
        with buffer_device:
            self.model.init_weights(buffer_device=buffer_device)
            final_out_std = self.config.hidden_size**-0.5
            cutoff_factor = 3
            if self.lm_head is not None:
                nn.init.trunc_normal_(
                    self.lm_head.weight,
                    mean=0.0,
                    std=final_out_std,
                    a=-cutoff_factor * final_out_std,
                    b=cutoff_factor * final_out_std,
                )

        self.to(dtype)
        for layer in self.model.layers.values():
            if isinstance(layer.mlp, MoE):
                layer.mlp.gate.e_score_correction_bias = torch.zeros(
                    (self.config.n_routed_experts), dtype=torch.float32
                ).to(buffer_device)


ModelClass = GlmMoeDsaForCausalLM
