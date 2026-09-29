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

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import (
    Qwen3NextGatedDeltaNet,
    torch_chunk_gated_delta_rule,
    torch_recurrent_gated_delta_rule,
)

from nemo_automodel.components.attention.utils import (
    initialize_attn_module_and_func,
    postprocess_output_for_attn,
    preprocess_args_and_kwargs_for_attn,
)
from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
)
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk
from nemo_automodel.shared.import_utils import safe_import_from
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


class _SSMGateParam:
    """Get-only descriptor exposing a param from ``_fp32_params`` when present."""

    def __init__(self, name: str):
        self.name = name

    def __get__(self, obj, owner=None):
        if obj is None:
            return self
        holder = obj._modules.get("_fp32_params")
        if holder is not None:
            return getattr(holder, self.name)
        param = obj._parameters.get(self.name)
        if param is not None:
            return param
        raise AttributeError(f"{type(obj).__name__!s} has no parameter {self.name!r}")


class Qwen3NextSSMGate(nn.Module):
    """Owns Qwen3-Next fp32 SSM-gating params and computes the decay gate."""

    def __init__(self, num_v_heads: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.A_log = nn.Parameter(torch.empty(num_v_heads, dtype=dtype))
        self.dt_bias = nn.Parameter(torch.empty(num_v_heads, dtype=dtype))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)


def _install_ssm_gate(mod: nn.Module, fp32_dtype: torch.dtype = torch.float32) -> Qwen3NextSSMGate:
    """Move HF-created bare ``A_log``/``dt_bias`` into a native fp32 holder."""
    num_v_heads = mod._parameters["A_log"].shape[0]
    gate = Qwen3NextSSMGate(num_v_heads, dtype=fp32_dtype)
    for pname in ("A_log", "dt_bias"):
        param = mod._parameters.pop(pname)
        if param.dtype != fp32_dtype:
            param.data = param.data.to(fp32_dtype)
        setattr(gate, pname, param)
    mod.add_module("_fp32_params", gate)
    return gate


class Qwen3NextFp32GatedDeltaNet(Qwen3NextGatedDeltaNet):
    """Qwen3-Next GatedDeltaNet that computes the decay gate via an fp32 holder.

    HF's ``Qwen3NextGatedDeltaNet`` computes the gate inline as
    ``g = -exp(A_log) * softplus(a + dt_bias)`` using the bare ``A_log`` / ``dt_bias``
    parameters. ``A_log`` and ``dt_bias`` are intrinsically fp32 (``A_log`` is
    exponentiated, so bf16 rounding becomes a proportional error on the decay rate that
    the recurrence compounds across the sequence).

    The constructor moves those params into a native ``_fp32_params`` holder so they
    are fp32 resident before any dtype cast or FSDP wrapping. To keep the gate
    computation in fp32 -- and to make FSDP's unshard/reshard + gradient
    reduce-scatter fire for that unit -- the gate is computed inside the holder's
    forward. This subclass overrides ``forward`` to route the gate through
    ``self._compute_gate(a)`` while reproducing the rest of HF's forward verbatim.
    """

    A_log = _SSMGateParam("A_log")
    dt_bias = _SSMGateParam("dt_bias")

    def __init__(self, config: Qwen3NextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        _install_ssm_gate(self)
        self._bind_kernels()

    def _bind_kernels(self) -> None:
        """Restore the kernel callables HF dropped as instance attributes in 5.15.

        transformers <=5.12 set ``causal_conv1d_fn`` / ``causal_conv1d_update`` /
        ``chunk_gated_delta_rule`` / ``recurrent_gated_delta_rule`` on the module and
        left ``causal_conv1d_fn`` as ``None`` when the kernel package was absent.
        5.15 moved them to decorated module-level fallbacks, so ``forward`` below --
        which calls them with the kernel package's own signature (``x=``,
        ``seq_idx=``) -- raised AttributeError. Rebind the installed kernels, and
        keep ``None`` when causal_conv1d is missing so the pure-torch conv branch in
        ``forward`` is taken rather than a signature-incompatible fallback.
        """
        has_conv1d, causal_conv1d_fn = safe_import_from("causal_conv1d", "causal_conv1d_fn")
        _, causal_conv1d_update = safe_import_from("causal_conv1d", "causal_conv1d_update")
        _, chunk_gated_delta_rule = safe_import_from(
            "fla.ops.gated_delta_rule", "chunk_gated_delta_rule", alt=torch_chunk_gated_delta_rule
        )
        _, fused_recurrent_gated_delta_rule = safe_import_from(
            "fla.ops.gated_delta_rule", "fused_recurrent_gated_delta_rule", alt=torch_recurrent_gated_delta_rule
        )

        # safe_import_from returns an UnavailableMeta placeholder rather than None
        # when causal_conv1d is missing, so key the pure-torch branch off the flag.
        self.causal_conv1d_fn = causal_conv1d_fn if has_conv1d else None
        self.causal_conv1d_update = causal_conv1d_update if has_conv1d else None
        self.chunk_gated_delta_rule = chunk_gated_delta_rule
        self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule

    def _compute_gate(self, a: torch.Tensor) -> torch.Tensor:
        """Compute the decay gate ``g`` in fp32, via the holder when it exists."""
        holder = self._modules.get("_fp32_params")
        if holder is not None:
            return holder(a)
        return -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

    def forward(  # pragma: no cover - verbatim HF GDN forward; needs CUDA conv1d/FLA kernels (GPU/functional only)
        self,
        hidden_states: torch.Tensor,
        cache_params: Any | None = None,
        attention_mask: torch.Tensor | None = None,
    ):
        # Mirrors transformers ``Qwen3NextGatedDeltaNet.forward`` with the gate routed
        # through ``self._compute_gate(a)`` so A_log/dt_bias stay fp32 under FSDP.
        from transformers.models.qwen3_next.modeling_qwen3_next import apply_mask_to_padding_states

        # transformers 5.15 dropped this helper's shape guards; it documents a 2D padding
        # mask and a 4D packed causal mask would broadcast to 5D and raise.
        if attention_mask is None or attention_mask.dim() == 2:
            hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

        batch_size, seq_len, _ = hidden_states.shape

        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)

        if use_precomputed_states:
            conv_state = cache_params.layers[self.layer_idx].conv_states
            recurrent_state = cache_params.layers[self.layer_idx].recurrent_states

        projected_states_qkvz = self.in_proj_qkvz(hidden_states)
        projected_states_ba = self.in_proj_ba(hidden_states)
        query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
        query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

        mixed_qkv = torch.cat((query, key, value), dim=-1)
        mixed_qkv = mixed_qkv.transpose(1, 2)

        if use_precomputed_states and seq_len == 1:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            )
        else:
            if use_precomputed_states:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if cache_params is not None:
                new_conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
                cache_params.update_conv_state(new_conv_state, self.layer_idx)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv,
                    weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                    seq_idx=None,
                )
            else:
                mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, : mixed_qkv.shape[-1]])
            if use_precomputed_states:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
        key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
        value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

        beta = b.sigmoid()
        # Gate is computed in fp32 (via the _fp32_params holder when present) so the
        # exponentiated decay rate keeps full precision under bf16 compute.
        g = self._compute_gate(a)
        if self.num_v_heads // self.num_k_heads > 1:
            query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
            key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

        if use_precomputed_states and seq_len == 1:
            core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=recurrent_state if use_precomputed_states else None,
                output_final_state=cache_params is not None,
                use_qk_l2norm_in_kernel=True,
            )

        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)

        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

        output = self.out_proj(core_attn_out)
        return output


class Qwen3NextRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Qwen3Next is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)


class Qwen3NextAttention(nn.Module):
    def __init__(self, config: Qwen3NextConfig, layer_idx: int, backend: BackendConfig):
        super().__init__()
        self.backend = backend
        self.layer_idx = layer_idx

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.num_key_value_groups = self.num_heads // self.num_kv_heads

        # Thread dtype explicitly from config.torch_dtype so callers that do
        # not wrap construction in local_torch_dtype() still get a dtype that
        # matches the model's declared dtype (fp32 under fp32 master weights,
        # bf16 otherwise).
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        # Query projection outputs 2x size for gating
        self.q_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_heads * self.head_dim * 2, False, dtype=dtype
        )
        self.k_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, False, dtype=dtype
        )
        self.v_proj = initialize_linear_module(
            backend.linear, config.hidden_size, self.num_kv_heads * self.head_dim, False, dtype=dtype
        )
        self.o_proj = initialize_linear_module(
            backend.linear, self.num_heads * self.head_dim, config.hidden_size, False, dtype=dtype
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Attention implementation
        self.scaling = self.head_dim**-0.5
        self.attn_module, self.attn_func = initialize_attn_module_and_func(
            attn_impl=backend.attn,
            num_attention_heads=self.num_heads,
            num_qk_channels=self.head_dim,
            num_v_channels=self.head_dim,
            softmax_scale=self.scaling,
            num_gqa_groups=self.num_kv_heads,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        if len(x.shape) == 2:
            qkv_format = "thd"
            num_tokens = x.shape[0]
        else:
            qkv_format = "bshd"
            bsz, seqlen, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Use -1 for the head dim so the (possibly tensor-parallel-sharded) local
        # head count is inferred from the projection output. Without TP this equals
        # the full head count; with TP (colwise q/k/v_proj) it is heads // tp_size.
        if qkv_format == "thd":
            q = q.view(num_tokens, -1, self.head_dim * 2)
            k = k.view(num_tokens, -1, self.head_dim)
            v = v.view(num_tokens, -1, self.head_dim)
        else:
            q = q.view(bsz, seqlen, -1, self.head_dim * 2)
            k = k.view(bsz, seqlen, -1, self.head_dim)
            v = v.view(bsz, seqlen, -1, self.head_dim)

        q, gate = torch.chunk(q, 2, dim=-1)
        gate = gate.reshape(*x.shape[:-1], -1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        q, k = apply_rotary_emb_qk(
            q,
            k,
            freqs_cis,
            format=qkv_format,
            rope_fusion=self.backend.rope_fusion,
            cu_seqlens=attn_kwargs.get("cu_seqlens", None),
            cp_size=attn_kwargs.get("cp_size", 1),
            cp_rank=attn_kwargs.get("cp_rank", 0),
        )

        # Backend-specific attention
        q, k, v, _attn_kwargs = preprocess_args_and_kwargs_for_attn(
            q, k, v, attention_mask, self.backend.attn, **attn_kwargs
        )
        attn_output = self.attn_func(q, k, v, **_attn_kwargs)
        attn_output = postprocess_output_for_attn(attn_output, self.backend.attn)

        # Reshape and apply gating
        attn_output = attn_output.reshape(*x.shape[:-1], -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)

        # Output projection
        attn_output = self.o_proj(attn_output)
        return attn_output

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02):
        linear_list = [self.q_proj, self.k_proj, self.v_proj, self.o_proj]
        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)
        for norm in (self.q_norm, self.k_norm):
            norm.reset_parameters()
