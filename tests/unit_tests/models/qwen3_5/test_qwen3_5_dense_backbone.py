# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Unit tests for the native dense Qwen3.5 backbone and fp32 SSMGate machinery.

Covers the custom-model building blocks that replaced the old runtime
``patch_hf_model``: the fp32 ``SSMGate`` holder + ``install_ssm_gate`` +
``_SSMGateParam`` descriptor, the ``Qwen3_5DenseTextBackbone`` forward, and the
fp32-safe rotary embedding. All tests are CPU-only.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

pytest.importorskip("transformers.models.qwen3_5")
pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from nemo_automodel.components.models.common import BackendConfig, packing
from nemo_automodel.components.models.qwen3_5 import packing as qwen3_5_packing
from nemo_automodel.components.models.qwen3_5.model import (
    Fp32SafeQwen3_5TextRotaryEmbedding,
    Qwen3_5DenseTextBackbone,
)
from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import (
    SSMGate,
    _resolve_ssm_dtype,
    _SSMGateParam,
    install_ssm_gate,
)

# Over the default 5s budget on purpose: CUDA gated-delta kernels take about 70s to compile on a cold worker.
# Reduce cold compiler startup before lowering this further.
pytestmark = pytest.mark.timeout(120)


def _backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        # BackendConfig.rope_fusion defaults to (HAVE_TE and cuda), so it is True on
        # the GPU CI runner. Qwen3.5's gated-query RoPE is incompatible with TE's
        # fused rope kernel (it expects a plain 4D layout), which is why every
        # Qwen3.5 recipe sets rope_fusion=false. Pin it here so the test exercises
        # the supported (non-fused) path deterministically on both CPU and GPU.
        rope_fusion=False,
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


def _tiny_config(layer_types=("full_attention",), **kwargs):
    return Qwen3_5TextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=len(layer_types),
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=list(layer_types),
        attn_implementation="eager",
        torch_dtype="float32",
        **kwargs,
    )


# ---------------------------------------------------------------------------
# SSMGate / install_ssm_gate / _SSMGateParam descriptor
# ---------------------------------------------------------------------------


class TestSSMGate:
    def test_params_are_fp32_with_expected_shape(self):
        gate = SSMGate(num_v_heads=4)
        assert gate.A_log.dtype == torch.float32
        assert gate.dt_bias.dtype == torch.float32
        assert tuple(gate.A_log.shape) == (4,)
        assert tuple(gate.dt_bias.shape) == (4,)

    def test_forward_computes_gate_in_fp32(self):
        gate = SSMGate(num_v_heads=4)
        nn.init.zeros_(gate.A_log)  # exp(0) = 1
        nn.init.zeros_(gate.dt_bias)
        a = torch.randn(2, 3, 4, dtype=torch.float32)
        out = gate(a)
        assert out.shape == a.shape
        assert out.dtype == torch.float32
        # g = -exp(A_log) * softplus(a + dt_bias) = -softplus(a) here.
        torch.testing.assert_close(out, -torch.nn.functional.softplus(a))

    def test_forward_upcasts_bf16_input(self):
        gate = SSMGate(num_v_heads=2)
        out = gate(torch.randn(1, 2, 2, dtype=torch.bfloat16))
        # Gate math runs in fp32 regardless of the activation dtype.
        assert out.dtype == torch.float32


class TestInstallSSMGate:
    def _module_with_bare_params(self, dtype=torch.bfloat16, num_v_heads=4):
        mod = nn.Module()
        mod.register_parameter("A_log", nn.Parameter(torch.randn(num_v_heads, dtype=dtype)))
        mod.register_parameter("dt_bias", nn.Parameter(torch.randn(num_v_heads, dtype=dtype)))
        return mod

    def test_moves_params_into_fp32_holder(self):
        mod = self._module_with_bare_params(dtype=torch.bfloat16)
        original_a_log = mod._parameters["A_log"]

        gate = install_ssm_gate(mod, fp32_dtype=torch.float32)

        # Bare params are removed from the parent and live in the holder.
        assert "A_log" not in mod._parameters
        assert "dt_bias" not in mod._parameters
        assert mod._fp32_params is gate
        assert gate.A_log.dtype == torch.float32
        assert gate.dt_bias.dtype == torch.float32
        # The original tensor values are preserved (cast to fp32 in place).
        torch.testing.assert_close(gate.A_log, original_a_log.detach().float())

    def test_already_fp32_params_kept(self):
        mod = self._module_with_bare_params(dtype=torch.float32)
        a_before = mod._parameters["A_log"].detach().clone()
        gate = install_ssm_gate(mod, fp32_dtype=torch.float32)
        torch.testing.assert_close(gate.A_log.detach(), a_before)


class TestSSMGateParamDescriptor:
    def test_descriptor_resolves_to_holder(self):
        class Mod(nn.Module):
            A_log = _SSMGateParam("A_log")
            dt_bias = _SSMGateParam("dt_bias")

            def __init__(self):
                super().__init__()
                self.register_parameter("A_log", nn.Parameter(torch.randn(3)))
                self.register_parameter("dt_bias", nn.Parameter(torch.randn(3)))
                install_ssm_gate(self, fp32_dtype=torch.float32)

        mod = Mod()
        # Attribute reads resolve through the descriptor into the holder.
        assert mod.A_log is mod._fp32_params.A_log
        assert mod.dt_bias is mod._fp32_params.dt_bias

    def test_descriptor_on_class_returns_itself(self):
        class Mod(nn.Module):
            A_log = _SSMGateParam("A_log")

        assert isinstance(Mod.A_log, _SSMGateParam)


class TestResolveSSMDtype:
    def test_default_is_fp32(self):
        class Cfg:
            pass

        assert _resolve_ssm_dtype(Cfg()) == torch.float32

    def test_string_dtype_resolved(self):
        class Cfg:
            mamba_ssm_dtype = "float32"

        assert _resolve_ssm_dtype(Cfg()) == torch.float32

    def test_explicit_dtype_passthrough(self):
        class Cfg:
            mamba_ssm_dtype = torch.float32

        assert _resolve_ssm_dtype(Cfg()) == torch.float32


# ---------------------------------------------------------------------------
# Fp32SafeQwen3_5TextRotaryEmbedding
# ---------------------------------------------------------------------------


class TestFp32SafeRotaryEmbedding:
    def test_inv_freq_stays_fp32_after_dtype_cast(self):
        cfg = _tiny_config()
        rope = Fp32SafeQwen3_5TextRotaryEmbedding(config=cfg)
        assert rope.inv_freq.dtype == torch.float32
        rope.to(torch.bfloat16)
        # The fp32 inv_freq buffer must survive a bulk bf16 cast.
        assert rope.inv_freq.dtype == torch.float32


# ---------------------------------------------------------------------------
# Qwen3_5DenseTextBackbone
# ---------------------------------------------------------------------------


class TestDenseTextBackbone:
    @pytest.mark.parametrize(
        "attention_mask",
        [
            pytest.param(
                torch.tensor(
                    [[1, 1, 1, 2, 2, 0], [1, 1, 2, 2, 3, 3]],
                    dtype=torch.long,
                ),
                id="conflicting-indexed-boundaries",
            ),
            pytest.param(torch.ones((2, 6), dtype=torch.long), id="binary-mask"),
        ],
    )
    def test_packed_metadata_prefers_authoritative_ids_and_matches_reference(
        self, attention_mask: torch.Tensor
    ) -> None:
        """Verify packed IDs override indexed and binary attention masks.

        Args:
            attention_mask: Tensor of shape [batch, sequence] containing either
                indexed document IDs with conflicting boundaries or a binary mask.
        """
        packed_seq_ids = torch.tensor(
            [[1, 1, 2, 2, 2, 0], [1, 2, 2, 3, 3, 3]],
            dtype=torch.long,
        )

        metadata = qwen3_5_packing.prepare_gated_delta_packed_metadata(attention_mask, packed_seq_ids)
        reference_indices, reference_cu_seqlens, _ = packing.get_unpad_data(packed_seq_ids)

        assert metadata is not None
        assert metadata.document_ids is packed_seq_ids
        torch.testing.assert_close(metadata.indices, reference_indices)
        torch.testing.assert_close(metadata.cu_seqlens, reference_cu_seqlens.long())
        torch.testing.assert_close(metadata.cu_seqlens_cpu, reference_cu_seqlens.long())

    def test_packed_metadata_falls_back_from_binary_ids_to_indexed_attention_mask(self):
        attention_mask = torch.tensor([[1, 1, 2, 2]], dtype=torch.long)
        packed_seq_ids = torch.ones_like(attention_mask)

        metadata = qwen3_5_packing.prepare_gated_delta_packed_metadata(attention_mask, packed_seq_ids)

        assert metadata is not None
        assert metadata.document_ids is attention_mask
        assert metadata.cu_seqlens_cpu.tolist() == [0, 2, 4]

    def test_builds_expected_layer_types(self):
        cfg = _tiny_config(layer_types=("full_attention", "linear_attention"))
        backbone = Qwen3_5DenseTextBackbone(cfg, _backend())
        types = [backbone.layers[str(i)].layer_type for i in range(cfg.num_hidden_layers)]
        assert types == ["full_attention", "linear_attention"]
        # The linear-attention block builds the native CP-aware GatedDeltaNet with
        # an fp32 SSMGate holder; the full-attention block does not.
        assert hasattr(backbone.layers["1"], "linear_attn")
        assert hasattr(backbone.layers["1"].linear_attn, "_fp32_params")

    def test_forward_shape_full_attention(self):
        cfg = _tiny_config(layer_types=("full_attention",))
        backbone = Qwen3_5DenseTextBackbone(cfg, _backend())
        out = backbone(input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
        assert out.last_hidden_state.shape == (1, 4, cfg.hidden_size)
        assert out.past_key_values is None

    def test_forward_shape_mixed_layers(self):
        # The linear_attention block runs FLA's gated-delta-rule / causal_conv1d
        # kernels, which are CUDA-only (no CPU fallback when CUDA is present), so
        # build the backbone and inputs on the GPU. Skip when no GPU is available.
        if not torch.cuda.is_available():
            pytest.skip("linear_attention forward requires CUDA (FLA kernels)")
        device = torch.device("cuda")
        cfg = _tiny_config(layer_types=("full_attention", "linear_attention"))
        backbone = Qwen3_5DenseTextBackbone(cfg, _backend()).to(device)
        out = backbone(input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device))
        assert out.last_hidden_state.shape == (1, 4, cfg.hidden_size)

    def test_forward_accepts_inputs_embeds(self):
        cfg = _tiny_config(layer_types=("full_attention",))
        backbone = Qwen3_5DenseTextBackbone(cfg, _backend())
        embeds = torch.randn(1, 4, cfg.hidden_size)
        out = backbone(inputs_embeds=embeds)
        assert out.last_hidden_state.shape == (1, 4, cfg.hidden_size)

    def test_reuses_packed_metadata_across_checkpointed_layers(self):
        torch.manual_seed(123)
        cfg = _tiny_config(layer_types=("linear_attention", "linear_attention"))
        backbone = Qwen3_5DenseTextBackbone(cfg, _backend()).float().train()
        linear_attn_modules = []
        for name in list(backbone.layers):
            linear_attn_modules.append(backbone.layers[name].linear_attn)
            backbone.layers[name] = checkpoint_wrapper(
                backbone.layers[name],
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )

        get_unpad_data = MagicMock(wraps=packing.get_unpad_data)
        chunk_gated_delta_rule = MagicMock(side_effect=lambda *args, **_kwargs: (args[2], None))
        for linear_attn in linear_attn_modules:
            linear_attn.causal_conv1d_fn = MagicMock(side_effect=lambda **kwargs: kwargs["x"])
            linear_attn.chunk_gated_delta_rule = chunk_gated_delta_rule
            linear_attn.norm.forward = MagicMock(side_effect=torch.add)

        with patch.object(qwen3_5_packing, "get_unpad_data", get_unpad_data):
            output = backbone(
                input_ids=torch.tensor([[1, 2, 3, 4]]),
                attention_mask=torch.tensor([[1, 1, 2, 2]]),
            ).last_hidden_state
            output.backward(torch.randn_like(output))

        assert get_unpad_data.call_count == 1
        assert chunk_gated_delta_rule.call_count == 4
        call_kwargs = [call.kwargs for call in chunk_gated_delta_rule.call_args_list]
        device_cu_seqlens = call_kwargs[0]["cu_seqlens"]
        cpu_cu_seqlens = call_kwargs[0]["cu_seqlens_cpu"]
        assert cpu_cu_seqlens.device.type == "cpu"
        assert cpu_cu_seqlens.tolist() == [0, 2, 4]
        assert all(kwargs["cu_seqlens"] is device_cu_seqlens for kwargs in call_kwargs)
        assert all(kwargs["cu_seqlens_cpu"] is cpu_cu_seqlens for kwargs in call_kwargs)

    def test_kv_cache_not_supported(self):
        cfg = _tiny_config(layer_types=("full_attention",))
        backbone = Qwen3_5DenseTextBackbone(cfg, _backend())
        with pytest.raises(NotImplementedError):
            backbone(input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.long), use_cache=True)

    def test_get_set_input_embeddings(self):
        cfg = _tiny_config(layer_types=("full_attention",))
        backbone = Qwen3_5DenseTextBackbone(cfg, _backend())
        new_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        backbone.set_input_embeddings(new_emb)
        assert backbone.get_input_embeddings() is new_emb
