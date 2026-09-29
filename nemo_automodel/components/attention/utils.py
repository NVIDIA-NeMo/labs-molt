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

from typing import Any, Callable, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.attention.flex_attention import FlexAttention


def initialize_attn_module_and_func(
    attn_impl: str,
    num_attention_heads: int,
    num_qk_channels: int,
    num_v_channels: int,
    softmax_scale: float,
    attn_mask_type: str = "causal",
    qkv_format: str = "bshd",
    num_gqa_groups: int | None = None,
    **kwargs: Any,
) -> tuple[nn.Module | None, Callable[..., torch.Tensor]]:
    """Initialize an attention backend module and callable."""
    if attn_impl == "te":
        from transformer_engine.pytorch.attention import DotProductAttention

        attn_module = DotProductAttention(
            num_attention_heads=num_attention_heads,
            kv_channels=(num_qk_channels, num_v_channels),
            attn_mask_type=attn_mask_type,
            qkv_format=qkv_format,
            softmax_scale=softmax_scale,
            num_gqa_groups=num_gqa_groups,
            **kwargs,
        )
        attn_func = attn_module.__call__
        return attn_module, attn_func
    elif attn_impl == "sdpa":
        supported_sdpa_kwargs = {"attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa"}
        unexpected_kwargs = kwargs.keys() - supported_sdpa_kwargs
        if unexpected_kwargs:
            raise TypeError(f"Unsupported SDPA attention kwargs: {sorted(unexpected_kwargs)}")

        default_attn_mask = cast(torch.Tensor | None, kwargs.get("attn_mask", None))
        default_dropout_p = cast(float, kwargs.get("dropout_p", 0.0))
        default_is_causal = cast(bool, kwargs.get("is_causal", attn_mask_type == "causal"))
        default_scale = cast(float | None, kwargs.get("scale", softmax_scale))
        default_enable_gqa = cast(bool, kwargs.get("enable_gqa", num_gqa_groups is not None))

        def attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, **call_kwargs: Any) -> torch.Tensor:
            unexpected_call_kwargs = call_kwargs.keys() - supported_sdpa_kwargs
            if unexpected_call_kwargs:
                raise TypeError(f"Unsupported SDPA attention kwargs: {sorted(unexpected_call_kwargs)}")

            attn_mask = cast(torch.Tensor | None, call_kwargs.get("attn_mask", default_attn_mask))
            dropout_p = cast(float, call_kwargs.get("dropout_p", default_dropout_p))
            is_causal = cast(bool, call_kwargs.get("is_causal", default_is_causal))
            scale = cast(float | None, call_kwargs.get("scale", default_scale))
            enable_gqa = cast(bool, call_kwargs.get("enable_gqa", default_enable_gqa))
            if enable_gqa and attn_mask is not None:
                groups = q.shape[-3] // k.shape[-3]
                k = k.repeat_interleave(groups, dim=-3)
                v = v.repeat_interleave(groups, dim=-3)
                enable_gqa = False
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=scale,
                enable_gqa=enable_gqa,
            )

        return None, attn_func
    elif attn_impl == "flex":
        attn_module = FlexAttention()
        # We still return the module and a reference to its call for parity with other backends
        attn_func = attn_module.__call__
        return attn_module, attn_func
    else:
        raise ValueError(f"Unsupported attention implementation: {attn_impl}")


def preprocess_args_and_kwargs_for_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | None,
    attn_impl: str,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Preprocess attention inputs based on backend requirements."""
    # Create attention kwargs based on backend
    if attn_impl == "te":
        attn_kwargs = {
            "window_size": kwargs.get("window_size", (-1, 0)),
        }
        if attention_mask is not None:
            padding_mask = attention_mask.logical_not()
            attn_kwargs.update(
                {
                    "attn_mask_type": "padding_causal",
                    "attention_mask": padding_mask.unsqueeze(1).unsqueeze(2),
                }
            )
        elif "cu_seqlens" in kwargs:
            attn_kwargs.update(
                {
                    "qkv_format": "thd",
                    "attn_mask_type": "padding_causal",
                    "cu_seqlens_q": kwargs["cu_seqlens"],
                    "cu_seqlens_kv": kwargs["cu_seqlens"],
                }
            )
            if "cu_seqlens_padded" in kwargs:
                attn_kwargs.update(
                    {
                        "cu_seqlens_q_padded": kwargs["cu_seqlens_padded"],
                        "cu_seqlens_kv_padded": kwargs["cu_seqlens_padded"],
                        "pad_between_seqs": True,
                    }
                )
            if "max_seqlen" in kwargs:
                attn_kwargs.update(
                    {
                        "max_seqlen_q": kwargs["max_seqlen"],
                        "max_seqlen_kv": kwargs["max_seqlen"],
                    }
                )
        elif "cu_seqlens_q" in kwargs and "cu_seqlens_kv" in kwargs:
            attn_kwargs.update(
                {
                    "qkv_format": "thd",
                    "attn_mask_type": "padding_causal",
                    "cu_seqlens_q": kwargs["cu_seqlens_q"],
                    "cu_seqlens_kv": kwargs["cu_seqlens_kv"],
                }
            )
            if "cu_seqlens_q_padded" in kwargs:
                attn_kwargs.update(
                    {
                        "cu_seqlens_q_padded": kwargs["cu_seqlens_q_padded"],
                        "pad_between_seqs": True,
                    }
                )
            if "cu_seqlens_kv_padded" in kwargs:
                attn_kwargs["cu_seqlens_kv_padded"] = kwargs["cu_seqlens_kv_padded"]
            if "max_seqlen_q" in kwargs:
                attn_kwargs["max_seqlen_q"] = kwargs["max_seqlen_q"]
            if "max_seqlen_kv" in kwargs:
                attn_kwargs["max_seqlen_kv"] = kwargs["max_seqlen_kv"]

    elif attn_impl == "flex":
        attn_kwargs = kwargs
        # Transpose for SDPA
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
    else:  # sdpa
        attn_kwargs = {}
        # Transpose for SDPA
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()
        v = v.transpose(1, 2).contiguous()
        window_size = kwargs.get("window_size", (-1, 0))
        left_window, right_window = window_size if isinstance(window_size, tuple) else (window_size, 0)
        has_local_window = (left_window is not None and left_window >= 0) or (
            right_window is not None and right_window > 0
        )
        key_mask = None
        explicit_mask = None
        if attention_mask is not None:
            if attention_mask.dim() <= 2:
                key_mask = attention_mask.to(device=q.device, dtype=torch.bool)
                has_padding_mask = not bool(key_mask.all().item())
            else:
                explicit_mask = attention_mask.to(device=q.device)
                has_padding_mask = False
        else:
            has_padding_mask = False

        if has_local_window or has_padding_mask:
            q_len = q.shape[-2]
            kv_len = k.shape[-2]
            kv_offset = max(kv_len - q_len, 0)
            q_pos = torch.arange(q_len, device=q.device) + kv_offset
            kv_pos = torch.arange(kv_len, device=q.device)
            causal_mask = kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1)

            if left_window is not None and left_window >= 0:
                causal_mask = causal_mask & (kv_pos.unsqueeze(0) > q_pos.unsqueeze(1) - left_window)
            if right_window is not None and right_window > 0:
                causal_mask = causal_mask & (kv_pos.unsqueeze(0) <= q_pos.unsqueeze(1) + right_window)

            if has_padding_mask:
                assert key_mask is not None
                if key_mask.shape[-1] != kv_len:
                    key_mask = key_mask[..., -kv_len:]
                causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) & key_mask[:, None, None, :]

            attn_kwargs["attn_mask"] = causal_mask
            attn_kwargs["is_causal"] = False
        elif explicit_mask is not None:
            attn_kwargs["attn_mask"] = explicit_mask
            attn_kwargs["is_causal"] = False
        else:
            attn_kwargs["is_causal"] = True

    return q, k, v, attn_kwargs


def postprocess_output_for_attn(x: torch.Tensor, attn_impl: str) -> torch.Tensor:
    """Postprocess attention output based on attn_impl requirements."""
    if attn_impl in ("sdpa", "flex"):
        x = x.transpose(1, 2).contiguous()
    return x
