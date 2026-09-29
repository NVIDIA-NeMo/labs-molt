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

"""GLM-5.2 DSA layers.

Contains the GlmMoeDsaIndexer for top-k sparse attention selection
and GlmMoeDsaMLA which integrates the indexer with Multi-head Latent Attention.
"""

from typing import Any

import torch
from torch import nn

# Try to import fast_hadamard_transform, fall back to torch implementation
try:
    from fast_hadamard_transform import hadamard_transform

    _FAST_HADAMARD_AVAILABLE = True
except ImportError:
    _FAST_HADAMARD_AVAILABLE = False

    # Taken from https://github.com/HazyResearch/structured-nets/blob/master/pytorch/structure/hadamard.py#L26
    def hadamard_transform_torch(u, scale: float, normalize=False):
        """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
        n must be a power of 2.
        Parameters:
            u: Tensor of shape (..., n)
            normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
        Returns:
            product: Tensor of shape (..., n)
        """
        import math

        n = u.shape[-1]
        m = int(math.log2(n))
        assert n == 1 << m, "n must be a power of 2"
        x = u.unsqueeze(-1)
        for _ in range(m):
            x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
        x = x.squeeze(-2) / 2 ** (m / 2) if normalize else x.squeeze(-2)
        return x * scale

    def hadamard_transform(x: torch.Tensor, scale: float) -> torch.Tensor:
        """Fallback hadamard_transform when fast_hadamard_transform is not available."""
        return hadamard_transform_torch(x, scale)


from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.deepseek_v3.rope_utils import apply_rotary_emb
from nemo_automodel.components.models.glm_moe_dsa.cp import glm_dsa_cp_all_gather, glm_dsa_cp_enabled
from nemo_automodel.components.models.glm_moe_dsa.optimized_kernels import (
    cudnn_indexer_topk,
    cudnn_sparse_attention,
    is_cudnn_dsa_available,
    is_dsa_kernel_available,
    should_use_tilelang,
    tilelang_indexer_topk,
    tilelang_sparse_attention,
)
from nemo_automodel.components.models.glm_moe_dsa.rope_utils import mla_softmax_scale
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _dsa_kernel_backend(backend: BackendConfig) -> str:
    """Resolve the DSA kernel path from the configured attention backend."""
    return backend.attn if backend.attn in ("tilelang", "cudnn") else "torch"


def _apply_index_rope_half_split(x: torch.Tensor, freqs_cis: torch.Tensor, qkv_format: str) -> torch.Tensor:
    """Apply NON-interleaved (half-split) RoPE to the indexer's rope slice.

    The DSA indexer uses half-split RoPE (``rotate_half``: pair dim ``j`` with ``j + d/2``),
    unlike the main MLA attention which uses interleaved RoPE. ``freqs_cis`` is the same
    complex tensor used by the MLA (``exp(i * theta_j * pos)`` for ``j in [0, d/2)``); we read
    its real/imag parts as cos/sin so the angles match exactly.

    Args:
        x: rope slice, ``[B, S, H, d]`` / ``[B, S, d]`` (bshd) or ``[T, H, d]`` / ``[T, d]`` (thd).
        freqs_cis: complex RoPE table with trailing dim ``d/2``.
        qkv_format: ``"bshd"`` or ``"thd"``.
    """
    d = x.shape[-1]
    half = d // 2
    if qkv_format == "thd":
        fc = freqs_cis.reshape(x.shape[0], *([1] * (x.dim() - 2)), half)
    else:
        fc = freqs_cis.reshape(x.shape[0], x.shape[1], *([1] * (x.dim() - 3)), half)
    cos = fc.real.to(x.dtype)
    sin = fc.imag.to(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _to_additive_key_mask(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a ``{0,1}`` keep-mask (1=attend, 0=mask) to an ADDITIVE key mask (0 / finfo.min).

    Masked positions use ``finfo.min`` rather than ``-inf``: F.scaled_dot_product_attention
    mishandles ``-inf`` float masks (its fused kernels corrupt the softmax). HF builds the
    attention bias with ``create_causal_mask``, which likewise masks padding to ``finfo.min``.
    The recipe, however, hands the model a 2D ``{0,1}`` padding mask; adding it to the scores
    raw (the previous behaviour) both fails to mask padding (0 -> +0 instead of finfo.min) AND adds
    ``+1.0`` to every kept key, which is only softmax-invariant in fp32 — in bf16 the ``+1.0``
    swamps the (scaled) score differences and collapses attention toward uniform. A mask that is
    already additive (values <= 0) is returned unchanged.
    """
    neg = torch.finfo(dtype).min  # SDPA-safe large-negative (NOT -inf, which breaks F.sdpa kernels)
    if mask.dtype == torch.bool:
        return torch.zeros_like(mask, dtype=dtype).masked_fill(~mask, neg)
    if mask.max() > 0:  # {0,1} keep-mask -> additive (kept -> 0, masked -> neg)
        return torch.zeros_like(mask, dtype=dtype).masked_fill(mask <= 0, neg)
    return mask.to(dtype)


def _rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Apply Hadamard rotation activation.

    Reference:
        https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/inference/model.py#L424-L428

    Args:
        x: Input tensor (must be bfloat16).

    Returns:
        Rotated tensor.
    """
    if x.dtype != torch.bfloat16:
        x = x.to(torch.bfloat16)
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


class GlmMoeDsaIndexer(nn.Module):
    """Indexer for top-k sparse attention selection.

    Based on the official GLM-5.2 training implementation. Computes attention
    scores between queries and keys with per-head weights, applies ReLU activation,
    then selects the top-k positions to attend to.

    Key features:
    - Uses LayerNorm (not RMSNorm) for key normalization
    - Has a weights_proj that learns per-head importance weights
    - Optional Hadamard transform (rotate_activation) on Q and K
    - ReLU activation on attention scores before weighting
    """

    def __init__(self, config: GlmMoeDsaConfig, backend: BackendConfig):
        super().__init__()

        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.hidden_size = config.hidden_size
        self.softmax_scale = self.head_dim**-0.5

        self.backend = backend
        linear_impl = backend.linear
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        # Project Q from q_lora residual -> num_heads * head_dim
        self.wq_b = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.q_lora_rank,
            out_features=self.num_heads * self.head_dim,
            bias=False,
            dtype=dtype,
        )

        # Project K from hidden states -> single head_dim (shared across heads)
        self.wk = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.hidden_size,
            out_features=self.head_dim,
            bias=False,
            dtype=dtype,
        )

        # LayerNorm for K (official uses LayerNorm, not RMSNorm). eps=1e-6 matches the reference / HF.
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6, dtype=dtype)

        # Per-head weight projection from hidden states
        self.weights_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.hidden_size,
            out_features=self.num_heads,
            bias=False,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        q_resid: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Compute top-k indices for sparse attention.

        Args:
            x: Hidden-state tensor of shape ``[batch, sequence, hidden]`` for BSHD or
                ``[tokens, hidden]`` for packed THD.
            q_resid: Query-LoRA residual tensor of shape ``[batch, sequence, q_lora_rank]``
                for BSHD or ``[tokens, q_lora_rank]`` for packed THD.
            freqs_cis: Complex RoPE tensor of shape ``[batch, sequence, rope_head_dim / 2]``
                for BSHD or ``[tokens, rope_head_dim / 2]`` for packed THD.
            attention_mask: Optional key mask tensor of shape ``[batch, sequence]`` or
                additive attention tensor of shape ``[batch, 1, sequence, sequence]``.
            **attn_kwargs: Attention metadata. Optimized packed backends require an int32
                ``cu_seqlens`` tensor of shape ``[sequences + 1]``. Packed CP may also
                supply ``glm_dsa_cp_query_indices`` of shape ``[tokens]`` and
                ``cu_seqlens_padded`` of shape ``[sequences + 1]``. The cuDNN backend
                requires CUDA bfloat16 query/key tensors.

        Returns:
            Top-k index tensor. The optimized packed backends return contiguous int32
            ``[tokens, 1, index_topk]`` with a fixed top-k width. The dense path returns
            ``[batch, sequence, min(index_topk, sequence)]`` for BSHD or
            ``[tokens, min(index_topk, tokens)]`` for THD.
        """
        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
            bsz = 1
            seq_len = num_tokens
        else:
            qkv_format = "bshd"
            bsz, seq_len, _ = x.size()

        # Project Q from q_lora residual
        q = self.wq_b(q_resid)
        if qkv_format == "thd":
            q = q.view(num_tokens, self.num_heads, self.head_dim)
        else:
            q = q.view(bsz, seq_len, self.num_heads, self.head_dim)

        # Split Q into pe and nope parts. The indexer lays out the rope slice FIRST
        # ([rope, nope]) — unlike the MLA path ([nope, rope]) — matching the DSA reference / HF.
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.qk_nope_head_dim], dim=-1)

        # Project K from hidden states
        k = self.k_norm(self.wk(x))

        # Split K into pe and nope parts (rope slice first, matching Q)
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.qk_nope_head_dim], dim=-1)

        # Apply NON-interleaved (half-split) RoPE to the pe parts. The indexer uses half-split
        # RoPE (matching the DSA reference / HF), unlike the interleaved RoPE of the MLA path.
        q_pe = _apply_index_rope_half_split(q_pe, freqs_cis, qkv_format)
        k_pe = _apply_index_rope_half_split(k_pe, freqs_cis, qkv_format)

        # Combine pe and nope parts (rope slice first, matching the reference layout)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = torch.cat([k_pe, k_nope], dim=-1)
        cp_group = attn_kwargs.get("_glm_dsa_cp_group")
        cp_enabled = glm_dsa_cp_enabled(cp_group)
        dsa_backend = _dsa_kernel_backend(self.backend)
        if cp_enabled:
            if qkv_format != "thd":
                raise ValueError("GLM DSA context parallelism requires THD/packed sequences.")
            k = glm_dsa_cp_all_gather(k, dim=0, cp_group=cp_group)

        if dsa_backend == "cudnn":
            if qkv_format != "thd":
                raise ValueError(
                    "cuDNN DSA indexer requires THD/packed sequences (qkv_format='thd'); "
                    f"got '{qkv_format}'. Use backend.attn in {{te, sdpa}} for the BSHD dense path."
                )
            cu_seqlens = attn_kwargs.get("cu_seqlens")
            if cu_seqlens is None:
                raise ValueError("cuDNN DSA indexer requires 'cu_seqlens' in attn_kwargs (THD packing metadata).")
            if not is_cudnn_dsa_available():
                raise RuntimeError(
                    "backend.attn='cudnn' requires the optional cuDNN DSA and FlashMLA kernels, "
                    "but they are unavailable in this environment."
                )
            # Preserve the reference scaling in FP32. The cuDNN adapter owns the final bf16 cast and
            # intentionally applies no additional scale to these per-head weights.
            head_weights = self.weights_proj(x).float() * (self.num_heads**-0.5) * self.softmax_scale
            return cudnn_indexer_topk(
                q.contiguous(),
                k.contiguous(),
                head_weights.contiguous(),
                cu_seqlens.flatten().to(torch.int32),
                self.index_topk,
                query_indices=attn_kwargs.get("glm_dsa_cp_query_indices"),
                cu_seqlens_padded=(
                    attn_kwargs["cu_seqlens_padded"].flatten().to(torch.int32)
                    if "cu_seqlens_padded" in attn_kwargs
                    else None
                ),
                packed_metadata=attn_kwargs.get("_cudnn_dsa_packed_metadata"),
            )

        # TileLang sparse path (opt-in via backend.attn == "tilelang"): the fused lighting-indexer
        # kernel computes logits + top-k directly, avoiding the dense [T, T_kv] score tensor. THD only.
        if dsa_backend == "tilelang" and should_use_tilelang(
            dsa_backend,
            available=is_dsa_kernel_available("indexer"),
            kernel_name="indexer",
            tensors=(q, k),
            require_bf16=True,
        ):
            if qkv_format != "thd":
                raise ValueError(
                    "TileLang DSA indexer requires THD/packed sequences (qkv_format='thd'); "
                    f"got '{qkv_format}'. Use backend.attn in {{te, sdpa}} for the bshd dense path."
                )
            cu_seqlens = attn_kwargs.get("cu_seqlens")
            if cu_seqlens is None:
                raise ValueError("TileLang DSA indexer requires 'cu_seqlens' in attn_kwargs (THD packing metadata).")
            # Fold the index softmax_scale (head_dim ** -0.5) into the per-head weight: the kernel
            # computes relu(q·k) * w with no internal scale, and relu(s*x) == s*relu(x) for s > 0.
            head_weights = self.weights_proj(x).float() * (self.num_heads**-0.5) * self.softmax_scale
            return tilelang_indexer_topk(
                q.contiguous(),
                k.contiguous(),
                head_weights.contiguous(),
                cu_seqlens.flatten().to(torch.int32),
                self.index_topk,
                query_indices=attn_kwargs.get("glm_dsa_cp_query_indices"),
                cu_seqlens_padded=(
                    attn_kwargs["cu_seqlens_padded"].flatten().to(torch.int32)
                    if "cu_seqlens_padded" in attn_kwargs
                    else None
                ),
            )

        if cp_enabled:
            raise NotImplementedError(
                "GLM DSA context parallelism is implemented only for backend.attn in {'tilelang', 'cudnn'}."
            )

        # NOTE: the reference Indexer applies a Hadamard rotation (`rotate_activation`) only as part
        # of its FP8 scoring kernel. The Hadamard transform is orthogonal, so it leaves the q·k index
        # scores unchanged; in this bf16/fp32 path we skip it (matching HF) to avoid adding rounding
        # noise that would perturb the top-k selection at near-tie boundaries.

        # Per-head weights from hidden states. Match the reference op order exactly: the
        # (n_heads ** -0.5) factor goes here and softmax_scale is applied to the q·k scores
        # below (relu(scale * x) == scale * relu(x)), then the head reduction is a matmul.
        # weights: [B, S, H] or [T, H]
        weights = self.weights_proj(x).float() * (self.num_heads**-0.5)

        # Per-head q·k scores with K kept single-head (no expand-to-heads), mirroring HF so the
        # index scores — and therefore the top-k selection — track the reference as closely as
        # floating point allows.
        if qkv_format == "thd":
            # q: [T, H, D], k: [T, D]  ->  [T, H, T]
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2))
        else:
            # q: [B, S, H, D], k: [B, S, D]  ->  [B, S, H, T]
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2).unsqueeze(1))
        scores = torch.relu(scores * self.softmax_scale)

        # Head-weighted sum via matmul: weights[..., 1, H] @ scores[..., H, T] -> [..., 1, T].
        scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)  # [T, T] or [B, S, T]

        # Apply attention mask if provided. Convert a {0,1} keep-mask to an additive key mask
        # (kept -> 0, padding -> finfo.min) so padding keys are excluded from the top-k selection,
        # instead of biasing kept keys by +1 (the previous raw-add behaviour).
        if attention_mask is not None:
            am = _to_additive_key_mask(attention_mask, scores.dtype)
            if am.dim() == 4:  # [B, 1, S_q, S_k] additive
                scores = scores + am.squeeze(1) if qkv_format == "bshd" else scores + am.squeeze(0).squeeze(0)
            elif qkv_format == "bshd":  # scores [B, S_q, S_k]; am [B, S_k] -> broadcast over queries
                scores = scores + am.unsqueeze(1)
            else:  # thd: scores [S_q, S_k]; am [S_k]
                scores = scores + am

        # Causal masking: the DSA model is a causal LM, so a query may select only keys at
        # positions <= its own. Without this, when seq_len <= index_topk the top-k picks ALL
        # tokens (including future), which (combined with the is_causal=False sparse-attention
        # path) makes attention bidirectional and leaks the next token. Matches the reference,
        # which causal-masks the index scores. Combines with any additive attention_mask above.
        q_len, k_len = scores.shape[-2], scores.shape[-1]
        causal = torch.ones(q_len, k_len, dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(causal, float("-inf"))

        # Select top-k indices
        actual_topk = min(self.index_topk, seq_len)
        topk_indices = scores.topk(actual_topk, dim=-1).indices

        return topk_indices

    def init_weights(self, init_std: float = 0.02):
        for module in [self.wq_b, self.wk, self.weights_proj]:
            if hasattr(module, "weight"):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=init_std)
        self.k_norm.reset_parameters()


class GlmMoeDsaMLA(nn.Module):
    """Multi-head Latent Attention with Indexer for sparse attention.

    This extends the V3 MLA with an Indexer module that performs
    top-k selection for sparse attention. The indexer uses the
    q_lora residual and hidden states to compute which positions
    to attend to.
    """

    def __init__(self, config: GlmMoeDsaConfig, backend: BackendConfig, skip_topk: bool = False):
        """Initialize MLA with an optional sparse-attention indexer.

        Args:
            config: Model config carrying MLA and indexer dimensions.
            backend: Backend selection for attention/linear/norm kernels.
            skip_topk: When ``True``, this layer owns no indexer and instead reuses the
                top-k selection of the previous "full" indexer layer (GLM IndexShare).
                ``forward`` then requires ``prev_topk_indices`` to be supplied. Defaults
                to ``False`` (the layer runs its own indexer), preserving the full-indexer
                behavior used by GLM configs without IndexShare metadata.
        """
        super().__init__()

        self.skip_topk = skip_topk
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = (
            config.qk_head_dim if hasattr(config, "qk_head_dim") else (self.qk_nope_head_dim + self.qk_rope_head_dim)
        )
        self.v_head_dim = config.v_head_dim
        self.index_topk = config.index_topk

        self.backend = backend
        self.rope_fusion = backend.rope_fusion
        attn_impl = backend.attn
        linear_impl = backend.linear
        rms_norm_impl = backend.rms_norm

        hidden_size = config.hidden_size
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        # GLM-5.2 always uses q_lora (q_lora_rank is not None)
        self.q_a_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=hidden_size,
            out_features=self.q_lora_rank,
            bias=False,
            dtype=dtype,
        )
        # Upstream GLM fixes the MLA latent RMSNorms at 1e-6 independently of
        # the model-wide rms_norm_eps=1e-5 used by the decoder block norms.
        self.q_a_layernorm = initialize_rms_norm_module(
            rms_norm_impl=rms_norm_impl,
            dim=self.q_lora_rank,
            eps=1e-6,
            dtype=dtype,
        )
        self.q_b_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.q_lora_rank,
            out_features=self.n_heads * self.qk_head_dim,
            bias=False,
            dtype=dtype,
        )

        self.kv_a_proj_with_mqa = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=hidden_size,
            out_features=self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.kv_a_layernorm = initialize_rms_norm_module(
            rms_norm_impl=rms_norm_impl,
            dim=self.kv_lora_rank,
            eps=1e-6,
            dtype=dtype,
        )
        self.kv_b_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.kv_lora_rank,
            out_features=self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            dtype=dtype,
        )
        self.o_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.n_heads * self.v_head_dim,
            out_features=hidden_size,
            bias=False,
            dtype=dtype,
        )
        # Shared with the GLM-5.2 DSpark draft, which trains on this model's hidden
        # states and must use the identical (YaRN-corrected) attention temperature.
        self.softmax_scale = mla_softmax_scale(config, self.qk_head_dim)

        if attn_impl in ("tilelang", "cudnn"):
            # Optimized DSA paths run their sparse kernel directly in forward; neither uses the
            # generic dense attention module, which does not implement these backend names.
            self.attn_module, self.attn_func = None, None
        else:
            self.attn_module, self.attn_func = initialize_attn_module_and_func(
                attn_impl=attn_impl,
                num_attention_heads=self.n_heads,
                num_qk_channels=self.qk_head_dim,
                num_v_channels=self.v_head_dim,
                softmax_scale=self.softmax_scale,
            )

        # Initialize the Indexer. "shared" layers (GLM IndexShare) own no indexer and
        # reuse the previous full layer's top-k indices passed in via `prev_topk_indices`.
        self.indexer = None if skip_topk else GlmMoeDsaIndexer(config, backend)

    def _build_sparse_mask(
        self,
        topk_indices: torch.Tensor,
        seq_len: int,
        qkv_format: str,
        bsz: int = 1,
        n_heads: int = 1,
        dtype: torch.dtype = torch.bfloat16,
        attention_mask: torch.Tensor | None = None,
        union_across_batches: bool = False,
    ) -> torch.Tensor:
        """Build a sparse attention mask/bias from top-k indices.

        Creates a mask tensor where non-top-k positions are set to finfo.min.
        Works for both TE (core_attention_bias) and SDPA (attn_mask).

        Uses the same efficient pattern as the official DeepSeek inference code, but with
        finfo.min instead of -inf (F.sdpa mishandles -inf float masks):
        `torch.full(..., finfo.min).scatter_(-1, topk_indices, 0)`

        Args:
            topk_indices: Indices of top-k positions [B, S, topk] or [T, topk]
            seq_len: Sequence length
            qkv_format: 'bshd' or 'thd'
            bsz: Batch size (only used for bshd format)
            n_heads: Number of attention heads to expand to
            dtype: Data type for the output tensor
            attention_mask: Optional attention mask to combine with (for SDPA)
            union_across_batches: If True, union top-k across batches (for TE);
                                  if False, keep per-batch masks (for SDPA)

        Returns:
            sparse_mask: Mask tensor with shape:
                - [1, n_heads, S, S] if union_across_batches=True
                - [B, n_heads, S, S] if union_across_batches=False (bshd)
                - [1, n_heads, T, T] for thd format
        """
        device = topk_indices.device
        # SDPA-safe masked value: F.scaled_dot_product_attention mishandles -inf float masks
        # (its fused kernels corrupt the softmax); transformers/HF use finfo.min for exactly this.
        # TE's core_attention_bias tolerates -inf, which is why only the SDPA path was affected.
        neg = torch.finfo(dtype).min

        if qkv_format == "thd":
            num_tokens = topk_indices.shape[0]
            # Create mask directly in final shape [1, n_heads, T, T]
            # All heads share the same mask, so we create [T, T] and expand
            sparse_mask = torch.full((num_tokens, num_tokens), neg, device=device, dtype=dtype).scatter_(
                -1, topk_indices, 0.0
            )
            # expand creates a view, contiguous makes a copy
            sparse_mask = sparse_mask.view(1, 1, num_tokens, num_tokens).expand(1, n_heads, -1, -1).contiguous()
        else:
            if union_across_batches:
                # For TE: create [B, S, S], scatter, union via max, then expand
                sparse_mask = torch.full((bsz, seq_len, seq_len), neg, device=device, dtype=dtype).scatter_(
                    -1, topk_indices, 0.0
                )
                # Union: max(0, finfo.min) = 0 for any position selected in any batch
                sparse_mask = sparse_mask.max(dim=0, keepdim=True).values
                sparse_mask = sparse_mask.view(1, 1, seq_len, seq_len).expand(1, n_heads, -1, -1).contiguous()
            else:
                # For SDPA: create [B, S, S], scatter, expand (no contiguous needed)
                sparse_mask = torch.full((bsz, seq_len, seq_len), neg, device=device, dtype=dtype).scatter_(
                    -1, topk_indices, 0.0
                )
                sparse_mask = sparse_mask.unsqueeze(1).expand(-1, n_heads, -1, -1)

        # Enforce causality independently of the top-k selection: a query attends only to keys
        # at positions <= its own. Required because at seq_len <= index_topk the top-k selects
        # every token (including future) and the SDPA path consumes this mask with is_causal=False;
        # without it the DSA attention is bidirectional and leaks the next token.
        q_len, k_len = sparse_mask.shape[-2], sparse_mask.shape[-1]
        causal_bool = torch.ones(q_len, k_len, dtype=torch.bool, device=sparse_mask.device).triu(1)
        sparse_mask = sparse_mask.masked_fill(causal_bool, neg)

        # Combine with the attention mask if provided. Convert a {0,1} keep-mask to an additive
        # key mask (kept -> 0, padding -> finfo.min) and broadcast it over the key axis; adding a raw
        # {0,1} mask would bias kept keys by +1 (bf16-lossy) and leave padding unmasked.
        if attention_mask is not None:
            am = _to_additive_key_mask(attention_mask, sparse_mask.dtype)
            while am.dim() < sparse_mask.dim():  # [B, S_key] -> [B, 1, 1, S_key]
                am = am.unsqueeze(1)
            # Clamp so doubly-masked positions (top-k/causal AND padding) don't sum past finfo.min
            # into -inf, which would reintroduce the SDPA -inf problem.
            sparse_mask = (sparse_mask + am).clamp_min(neg)

        return sparse_mask

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        return_topk_indices: bool = False,
        **attn_kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run MLA with (optionally shared) DSA sparse attention.

        Args:
            x: Hidden-state tensor of shape ``[batch, sequence, hidden]`` for BSHD or
                ``[tokens, hidden]`` for packed THD.
            freqs_cis: Complex RoPE tensor of shape ``[batch, sequence, rope_head_dim / 2]``
                for BSHD or ``[tokens, rope_head_dim / 2]`` for packed THD.
            attention_mask: Optional additive attention tensor of shape
                ``[batch, 1, sequence, sequence]`` or key mask of shape ``[batch, sequence]``.
            prev_topk_indices: Top-k index tensor from the most recent full indexer layer.
                Optimized THD backends use int32 ``[tokens, 1, index_topk]``; the dense
                BSHD path uses int64 ``[batch, sequence, min(index_topk, sequence)]``.
                Required only for a shared layer (``skip_topk=True``).
            return_topk_indices: When ``True``, return ``(attn_out, topk_indices)`` so the
                caller can thread the selection to subsequent shared layers (GLM IndexShare).
                When ``False`` (default), return just ``attn_out``.
            **attn_kwargs: Attention metadata. Optimized packed backends require an int32
                ``cu_seqlens`` tensor of shape ``[sequences + 1]``. Packed CP may also
                provide ``_glm_dsa_cp_group`` and packed query-position tensors. The cuDNN
                backend requires CUDA bfloat16 inputs.

        Returns:
            Attention output tensor of shape ``[batch, sequence, hidden]`` for BSHD or
            ``[tokens, hidden]`` for packed THD. When ``return_topk_indices`` is true,
            returns that tensor with the top-k index tensor described above.
        """
        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
            bsz = 1
            seq_len = num_tokens
        else:
            qkv_format = "bshd"
            bsz, seq_len, _ = x.size()

        # Compute q_resid for indexer and main attention path
        q_resid = self.q_a_layernorm(self.q_a_proj(x))

        # Get top-k indices: run our own indexer ("full" layer), or reuse the previous
        # full layer's selection ("shared" layer, GLM IndexShare).
        if self.indexer is not None:
            topk_indices = self.indexer(x, q_resid, freqs_cis, attention_mask, **attn_kwargs)
        else:
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared DSA layers (skip_topk=True) require top-k indices from a previous "
                    "full indexer layer; got prev_topk_indices=None."
                )
            topk_indices = prev_topk_indices

        # Compute Q from q_resid
        q = self.q_b_proj(q_resid)

        if qkv_format == "thd":
            q = q.view(num_tokens, self.n_heads, self.qk_head_dim)
        else:
            q = q.view(bsz, seq_len, self.n_heads, self.qk_head_dim)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj_with_mqa(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_layernorm(kv)

        # For MLA, k_pe needs an extra head dimension for apply_rotary_emb
        head_unsqueeze_dim = 2 if qkv_format == "bshd" else 1
        k_pe = k_pe.unsqueeze(head_unsqueeze_dim)

        # Apply rotary embeddings to q_pe and k_pe
        q_pe = apply_rotary_emb(q_pe, freqs_cis, qkv_format=qkv_format)
        k_pe = apply_rotary_emb(k_pe, freqs_cis, qkv_format=qkv_format)

        # Remove the head dimension we added to k_pe
        k_pe = k_pe.squeeze(head_unsqueeze_dim)
        cp_group = attn_kwargs.get("_glm_dsa_cp_group")
        cp_enabled = glm_dsa_cp_enabled(cp_group)
        dsa_backend = _dsa_kernel_backend(self.backend)
        if cp_enabled:
            if qkv_format != "thd":
                raise ValueError("GLM DSA context parallelism requires THD/packed sequences.")
            kv = glm_dsa_cp_all_gather(kv, dim=0, cp_group=cp_group)
            k_pe = glm_dsa_cp_all_gather(k_pe, dim=0, cp_group=cp_group)

        # Optimized sparse paths run on the absorbed latent representation. The k_nope
        # up-projection is folded into the query, and w_vc maps the latent output back.
        if dsa_backend in ("tilelang", "cudnn"):
            if qkv_format != "thd":
                raise ValueError(
                    f"{dsa_backend} DSA sparse attention requires THD/packed sequences (qkv_format='thd'); "
                    f"got '{qkv_format}'. Use backend.attn in {{te, sdpa}} for the BSHD dense path."
                )
            materialize_effective_weight = getattr(self.kv_b_proj, "materialize_effective_weight", None)
            kv_b_weight = (
                materialize_effective_weight() if materialize_effective_weight is not None else self.kv_b_proj.weight
            )
            w = kv_b_weight.view(self.n_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
            w_kc = w[:, : self.qk_nope_head_dim, :]
            w_vc = w[:, self.qk_nope_head_dim :, :]
            q_absorbed = torch.einsum("thd,hdc->thc", q_nope, w_kc.to(q_nope.dtype))
            q_sparse = torch.cat([q_absorbed, q_pe], dim=-1).to(torch.bfloat16)
            kv_latent = torch.cat([kv, k_pe], dim=-1).unsqueeze(1).to(torch.bfloat16)
            if dsa_backend == "tilelang":
                if not should_use_tilelang(
                    "tilelang",
                    available=is_dsa_kernel_available("sparse_attn"),
                    kernel_name="sparse_attn",
                    tensors=(q_sparse, kv_latent),
                    require_bf16=True,
                ):
                    raise RuntimeError("TileLang sparse attention was selected but did not pass validation.")
                attn_out = tilelang_sparse_attention(
                    q_sparse, kv_latent, topk_indices, w_vc.to(q_sparse.dtype), self.softmax_scale
                )
            else:
                if not is_cudnn_dsa_available():
                    raise RuntimeError(
                        "backend.attn='cudnn' requires the optional cuDNN DSA and FlashMLA kernels, "
                        "but they are unavailable in this environment."
                    )
                expected_topk_shape = (num_tokens, 1, self.index_topk)
                if tuple(topk_indices.shape) != expected_topk_shape:
                    raise ValueError(
                        "cuDNN DSA sparse attention requires fixed-width top-k indices of shape "
                        f"{expected_topk_shape}; got {tuple(topk_indices.shape)}."
                    )
                latent_out = cudnn_sparse_attention(
                    q_sparse,
                    kv_latent,
                    topk_indices,
                    self.softmax_scale,
                    topk_length=attn_kwargs.get("_cudnn_dsa_topk_length"),
                    all_rows_nonempty=bool(attn_kwargs.get("_cudnn_dsa_all_rows_nonempty", False)),
                    valid_row_indices=attn_kwargs.get("_cudnn_dsa_valid_row_indices"),
                )
                attn_out = torch.einsum("thc,hdc->thd", latent_out, w_vc.to(latent_out.dtype))
            x = self.o_proj(attn_out.flatten(1))
            if return_topk_indices:
                return x, topk_indices
            return x

        # Dense reference path (te / sdpa / eager / flex): materialize the sparse mask and run full
        # attention. Build the sparse bias/mask from the top-k indices based on backend.
        if self.backend.attn == "te":
            # For TE: build sparse bias for core_attention_bias (must match Q/K/V dtype)
            # Union across batches since TE expects [1, n_heads, S, S]
            sparse_mask = self._build_sparse_mask(
                topk_indices,
                seq_len,
                qkv_format,
                bsz,
                n_heads=self.n_heads,
                dtype=x.dtype,
                attention_mask=None,
                union_across_batches=True,
            )
        else:
            # For SDPA: build sparse mask, keep per-batch masks
            sparse_mask = self._build_sparse_mask(
                topk_indices,
                seq_len,
                qkv_format,
                bsz,
                n_heads=1,
                dtype=x.dtype,  # model (bf16) dtype, matching HF: an fp32 finfo.min mask downcasts
                # to -inf in bf16 inside F.sdpa -> NaN; a bf16 finfo.min mask is representable.
                attention_mask=attention_mask,
                union_across_batches=False,
            )

        q = torch.cat([q_nope, q_pe], dim=-1)

        kv = self.kv_b_proj(kv)
        if qkv_format == "thd":
            kv = kv.view(num_tokens, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_pe = k_pe.unsqueeze(1).expand([num_tokens, self.n_heads, self.qk_rope_head_dim])
        else:
            kv = kv.view(bsz, seq_len, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_pe = k_pe.unsqueeze(2).expand([bsz, seq_len, self.n_heads, self.qk_rope_head_dim])

        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)

        # Handle attention based on backend
        if self.backend.attn == "te":
            # For TE: use core_attention_bias for sparse attention
            q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
                q, k, v, attention_mask, self.backend.attn, **attn_kwargs
            )
            # Add sparse mask as core_attention_bias
            _attn_kwargs["core_attention_bias_type"] = "post_scale_bias"
            _attn_kwargs["core_attention_bias"] = sparse_mask
        else:
            # For SDPA: use sparse mask (already combined with attention_mask)
            q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
                q, k, v, sparse_mask, self.backend.attn, **attn_kwargs
            )

        x = self.attn_func(q, k, v, **_attn_kwargs)
        x = postprocess_output_for_attn(x, self.backend.attn)

        flatten_dim = 2 if qkv_format == "bshd" else 1
        x = self.o_proj(x.flatten(flatten_dim))
        if return_topk_indices:
            return x, topk_indices
        return x

    def init_weights(self, _buffer_device: torch.device, init_std: float = 0.02):
        linear_list = [
            self.q_a_proj,
            self.q_b_proj,
            self.kv_a_proj_with_mqa,
            self.kv_b_proj,
            self.o_proj,
        ]

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

        norms = [self.kv_a_layernorm, self.q_a_layernorm]
        for norm in norms:
            norm.reset_parameters()

        # Initialize indexer weights ("shared" layers own no indexer).
        if self.indexer is not None:
            self.indexer.init_weights(init_std)
