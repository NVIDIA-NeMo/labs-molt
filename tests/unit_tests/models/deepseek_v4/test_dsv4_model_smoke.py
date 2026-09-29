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

"""Smoke tests for DeepSeek V4: forward + backward pass with a tiny config.

Run on the cluster (CPU-only, no checkpoint needed):
    PYTHONPATH=/path/to/Automodel_lao python -m pytest \
        tests/unit_tests/models/deepseek_v4/test_dsv4_model_smoke.py -v -s

Or as a standalone script:
    PYTHONPATH=/path/to/Automodel_lao python \
        tests/unit_tests/models/deepseek_v4/test_dsv4_model_smoke.py
"""

import pytest
import torch
from torch.distributed.fsdp import MixedPrecisionPolicy

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.mtp import MTPConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.deepseek_v4 import fsdp as dsv4_fsdp
from nemo_automodel.components.models.deepseek_v4 import model as dsv4_model_module
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM

# Over the default 5s budget on purpose: CUDA model forward and backward initialize compiled kernels.
# Reduce cold compiler startup before lowering this further.
pytestmark = pytest.mark.timeout(60)

# ``MoE.forward`` (in ``nemo_automodel/components/moe/layers.py``)
# unconditionally creates a ``torch.cuda.Stream()`` when the model has a
# shared expert.  On the CPU-only CI image (``L0_Unit_Tests_CPU``) that
# raises ``AcceleratorError: CUDA driver version is insufficient`` even
# though the model parameters live on the CPU.  Gate the forward-running
# smoke tests on CUDA availability — the rest of the file (HC params,
# weight shapes, attn-sink dtype) is pure metadata and runs everywhere.
_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="MoE.forward unconditionally allocates torch.cuda.Stream() for shared experts",
)


def _tiny_config(**overrides) -> DeepseekV4Config:
    """Tiny V4 config: fits in ~1 GB CPU RAM, exercises all code paths."""
    defaults = dict(
        vocab_size=256,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 64,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        hc_mult=4,
        num_hash_layers=2,
        compress_ratios=[0, 0, 4, 0],  # 4 layers: full, full, csa(fallback), full
        sliding_window=16,
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-6,
        torch_dtype="float32",
    )
    defaults.update(overrides)
    return DeepseekV4Config(**defaults)


def _make_model(config: DeepseekV4Config) -> DeepseekV4ForCausalLM:
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
        dispatcher="torch",
        experts="torch_mm",
    )
    model = DeepseekV4ForCausalLM(config, backend=backend)
    # Zero all float params: Gate/GroupedExperts use torch.empty (uninitialized memory
    # that may contain NaN bit patterns). initialize_weights() is not called in smoke tests.
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.zero_()
    model.eval()
    return model


def test_fp32_config_constructs_all_floating_parameters_in_fp32():
    """An fp32 config must not leave RMSNorm weights in the bf16 helper default."""
    config = _tiny_config(num_hidden_layers=1, num_hash_layers=0, compress_ratios=[4])
    model = _make_model(config)

    mismatches = {
        name: param.dtype
        for name, param in model.named_parameters()
        if param.is_floating_point() and param.dtype != torch.float32
    }

    assert not mismatches


def test_config_honors_canonical_dtype_argument():
    config = DeepseekV4Config(dtype="float32")

    assert config.dtype == torch.float32
    assert config.torch_dtype == torch.float32


def test_config_preserves_fp32_dtype_through_serialization_round_trip():
    config = DeepseekV4Config(dtype="float32")

    restored = DeepseekV4Config.from_dict(config.to_dict())

    assert restored.dtype == torch.float32
    assert restored.torch_dtype == torch.float32


class TestDeepseekV4ModelSmoke:
    def test_dsv4_hca_param_sync_group_uses_only_1d_mesh(self):
        named_hca_group = object()
        unnamed_hca_group = object()
        shape_only_hca_group = object()

        class NamedOneDimMesh:
            mesh_dim_names = ("dp",)

            def size(self):
                return 2

            def get_group(self):
                return named_hca_group

        class UnnamedOneDimMesh:
            mesh_dim_names = None
            ndim = 1

            def size(self):
                return 2

            def get_group(self):
                return unnamed_hca_group

        class UnnamedTwoDimMesh:
            mesh_dim_names = None
            ndim = 2

        class ShapeOnlyOneDimMesh:
            shape = (2,)

            def size(self):
                return 2

            def get_group(self):
                return shape_only_hca_group

        class TwoDimMesh:
            mesh_dim_names = ("dp", "tp")

        assert dsv4_fsdp._hca_param_sync_group_from_1d_mesh(None) is None
        assert dsv4_fsdp._hca_param_sync_group_from_1d_mesh(UnnamedOneDimMesh()) is unnamed_hca_group
        assert dsv4_fsdp._hca_param_sync_group_from_1d_mesh(UnnamedTwoDimMesh()) is None
        assert dsv4_fsdp._hca_param_sync_group_from_1d_mesh(ShapeOnlyOneDimMesh()) is shape_only_hca_group
        assert dsv4_fsdp._hca_param_sync_group_from_1d_mesh(TwoDimMesh()) is None
        assert dsv4_fsdp._hca_param_sync_group_from_1d_mesh(NamedOneDimMesh()) is named_hca_group

    def test_dsv4_fsdp_uses_bf16_compute_for_uniform_fp32_master_weights(self, monkeypatch):
        cfg = _tiny_config(num_hidden_layers=1, num_hash_layers=0, compress_ratios=[128])
        model = _make_model(cfg)
        block = model.model.layers["0"]
        calls = []

        def fake_fully_shard(module, **kwargs):
            calls.append((module, kwargs))
            return module

        monkeypatch.setattr(dsv4_fsdp, "fully_shard", fake_fully_shard)

        hca_group = object()

        class FakeMesh:
            ndim = 1
            mesh_dim_names = None

            def size(self):
                return 2

            def get_group(self):
                return hca_group

        input_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
        )

        dsv4_fsdp.fully_shard_deepseek_v4(
            block,
            mesh=FakeMesh(),
            mp_policy=input_policy,
            offload_policy=object(),
        )

        assert block.self_attn.compressor._hca_param_sync_group is hca_group
        assert calls[-1][0] is block
        assert calls[-1][1]["mp_policy"].param_dtype == torch.bfloat16
        assert calls[-1][1]["mp_policy"].reduce_dtype == torch.float32
        assert any(module is not block and kwargs["mp_policy"].param_dtype == torch.float32 for module, kwargs in calls)

    def test_dsv4_fsdp_preserves_explicit_all_fp32_compute_fast_path(self, monkeypatch):
        cfg = _tiny_config(num_hidden_layers=1, num_hash_layers=0, compress_ratios=[128])
        block = _make_model(cfg).model.layers["0"]
        calls = []

        def fake_fully_shard(module, **kwargs):
            calls.append((module, kwargs))
            return module

        monkeypatch.setattr(dsv4_fsdp, "fully_shard", fake_fully_shard)

        input_policy = MixedPrecisionPolicy(
            param_dtype=torch.float32,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
        )

        dsv4_fsdp.fully_shard_deepseek_v4(
            block,
            mesh=object(),
            mp_policy=input_policy,
            offload_policy=object(),
        )

        assert len(calls) == 1
        assert calls[0][0] is block
        assert calls[0][1]["mp_policy"].param_dtype == torch.float32
        assert calls[0][1]["mp_policy"].reduce_dtype == torch.float32

    def test_dsv4_fsdp_wraps_fp32_islands_without_ignored_trainable_tensors(self, monkeypatch):
        cfg = _tiny_config(
            num_hidden_layers=1,
            num_hash_layers=0,
            compress_ratios=[4],
            torch_dtype="bfloat16",
        )
        backend = BackendConfig(
            attn="sdpa",
            linear="torch",
            rms_norm="torch",
            rope_fusion=False,
            enable_hf_state_dict_adapter=False,
            dispatcher="torch",
            experts="torch_mm",
        )
        old_default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            model = DeepseekV4ForCausalLM(cfg, backend=backend)
        finally:
            torch.set_default_dtype(old_default_dtype)

        block = model.model.layers["0"]
        module_names = {id(module): name for name, module in block.named_modules()}
        module_names[id(block)] = "<block>"
        calls = []

        def fake_fully_shard(module, **kwargs):
            calls.append((module_names.get(id(module), "<unknown>"), kwargs))
            return module

        monkeypatch.setattr(dsv4_fsdp, "fully_shard", fake_fully_shard)

        hca_group = object()

        class FakeMesh:
            mesh_dim_names = ("dp",)

            def size(self):
                return 2

            def get_group(self):
                return hca_group

        input_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
        )
        ignored_expert_params = set(block.mlp.experts.parameters())

        dsv4_fsdp.fully_shard_deepseek_v4(
            block,
            mesh=FakeMesh(),
            mp_policy=input_policy,
            offload_policy=object(),
            ignored_params=ignored_expert_params,
        )

        assert block.self_attn.compressor._hca_param_sync_group is hca_group
        calls_by_name = {name: kwargs for name, kwargs in calls}
        expected_fp32_modules = {
            "attn_hc",
            "ffn_hc",
            "self_attn.sinks_param",
            "self_attn.compressor.wkv",
            "self_attn.compressor.wgate",
            "self_attn.compressor.ape_param",
            "self_attn.compressor.indexer.wkv",
            "self_attn.compressor.indexer.wgate",
            "self_attn.compressor.indexer.ape_param",
        }
        assert expected_fp32_modules.issubset(calls_by_name)
        for name in expected_fp32_modules:
            assert calls_by_name[name]["mp_policy"].param_dtype == torch.float32
            assert calls_by_name[name]["mp_policy"].reduce_dtype == torch.float32

        parent_ignored = calls_by_name["<block>"]["ignored_params"]
        assert ignored_expert_params.issubset(parent_ignored)
        assert len(parent_ignored) == len(ignored_expert_params)

    def test_dsv4_fp32_holder_preserves_explicit_no_reshard(self):
        holder_type = type("DeepseekV4FP32Parameter", (torch.nn.Module,), {})
        kwargs = {"reshard_after_forward": False, "mesh": object()}

        assert dsv4_fsdp._fp32_module_fsdp_kwargs(holder_type(), kwargs) is kwargs

    def test_reference_fp32_parameters_constructed_before_cast(self):
        cfg = _tiny_config(
            num_hidden_layers=2,
            num_hash_layers=0,
            compress_ratios=[4, 0],
            torch_dtype="bfloat16",
        )
        old_default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            model = DeepseekV4ForCausalLM(
                cfg,
                backend=BackendConfig(
                    attn="sdpa",
                    linear="torch",
                    rms_norm="torch",
                    rope_fusion=False,
                    enable_hf_state_dict_adapter=False,
                    dispatcher="torch",
                    experts="torch_mm",
                ),
            )
        finally:
            torch.set_default_dtype(old_default_dtype)

        expected_fp32 = (
            "attn_hc.fn",
            "attn_hc.base",
            "attn_hc.scale",
            "ffn_hc.fn",
            "ffn_hc.base",
            "ffn_hc.scale",
            "hc_head.hc_fn",
            "hc_head.hc_base",
            "hc_head.hc_scale",
            "self_attn.sinks",
            "self_attn.compressor.wkv",
            "self_attn.compressor.wgate",
            "self_attn.compressor.ape",
            "self_attn.compressor.indexer.wkv",
            "self_attn.compressor.indexer.wgate",
            "self_attn.compressor.indexer.ape",
            "lm_head",
        )
        matched = []
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in expected_fp32):
                matched.append(name)
                assert param.dtype == torch.float32, name

        assert matched
        assert model.model.embed_tokens.weight.dtype == torch.bfloat16
        assert model.model.layers["0"].self_attn.wq_a.weight.dtype == torch.bfloat16

    def test_reference_fp32_parameters_survive_bf16_cast(self):
        cfg = _tiny_config(num_hidden_layers=3, num_hash_layers=0, compress_ratios=[0, 4, 0])
        model = DeepseekV4ForCausalLM(
            cfg,
            backend=BackendConfig(
                attn="sdpa",
                linear="torch",
                rms_norm="torch",
                rope_fusion=False,
                enable_hf_state_dict_adapter=False,
                dispatcher="torch",
                experts="torch_mm",
            ),
        )

        cast_model_to_dtype(model, torch.bfloat16)

        expected_fp32 = (
            "attn_hc.fn",
            "attn_hc.base",
            "attn_hc.scale",
            "ffn_hc.fn",
            "ffn_hc.base",
            "ffn_hc.scale",
            "hc_head.hc_fn",
            "hc_head.hc_base",
            "hc_head.hc_scale",
            "self_attn.sinks",
            "self_attn.compressor.wkv",
            "self_attn.compressor.wgate",
            "self_attn.compressor.ape",
            "self_attn.compressor.indexer.wkv",
            "self_attn.compressor.indexer.wgate",
            "self_attn.compressor.indexer.ape",
            "lm_head",
        )
        matched = []
        for name, param in model.named_parameters():
            if any(keyword in name for keyword in expected_fp32):
                matched.append(name)
                assert param.dtype == torch.float32, name

        assert matched
        assert model.model.embed_tokens.weight.dtype == torch.bfloat16

    def test_initialize_weights_does_not_blanket_cast_dtensor_params(self, monkeypatch):
        cfg = _tiny_config(num_hidden_layers=1, num_hash_layers=0, compress_ratios=[0])
        model = DeepseekV4ForCausalLM(
            cfg,
            backend=BackendConfig(
                attn="sdpa",
                linear="torch",
                rms_norm="torch",
                rope_fusion=False,
                enable_hf_state_dict_adapter=False,
                dispatcher="torch",
                experts="torch_mm",
            ),
        )

        monkeypatch.setattr(dsv4_model_module, "_has_dtensor_params", lambda _: True)

        def fail_cast(*args, **kwargs):
            raise AssertionError("cast_model_to_dtype must not run after FSDP2/DTensor wrapping")

        monkeypatch.setattr(dsv4_model_module, "cast_model_to_dtype", fail_cast)

        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)

    def test_initialize_weights_initializes_all_hyper_connection_parameters(self):
        cfg = _tiny_config(
            num_hidden_layers=1,
            num_hash_layers=0,
            compress_ratios=[0],
            num_nextn_predict_layers=1,
        )
        model = DeepseekV4ForCausalLM(
            cfg,
            backend=BackendConfig(
                attn="sdpa",
                linear="torch",
                rms_norm="torch",
                rope_fusion=False,
                enable_hf_state_dict_adapter=False,
                dispatcher="torch",
                experts="torch_mm",
            ),
        )
        block = model.model.layers["0"]
        assert model.mtp is not None
        mtp_block = model.mtp.layers[0]
        hyper_modules = (block.attn_hc, block.ffn_hc, mtp_block.attn_hc, mtp_block.ffn_hc)
        hyper_heads = (model.model.hc_head, mtp_block.hc_head)
        with torch.no_grad():
            for module in hyper_modules:
                module.fn.fill_(float("nan"))
                module.base.fill_(float("nan"))
                module.scale.fill_(float("nan"))
            for head in hyper_heads:
                head.hc_fn.fill_(float("nan"))
                head.hc_base.fill_(float("nan"))
                head.hc_scale.fill_(float("nan"))

        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

        for module in hyper_modules:
            assert torch.isfinite(module.fn).all()
            assert torch.count_nonzero(module.fn) > 0
            torch.testing.assert_close(module.base, torch.zeros_like(module.base))
            torch.testing.assert_close(module.scale, torch.ones_like(module.scale))
        for head in hyper_heads:
            assert torch.isfinite(head.hc_fn).all()
            assert torch.count_nonzero(head.hc_fn) > 0
            torch.testing.assert_close(head.hc_base, torch.zeros_like(head.hc_base))
            torch.testing.assert_close(head.hc_scale, torch.ones_like(head.hc_scale))

    def test_hash_gate_tid2eid_uses_deepep_runtime_int64_dtype(self):
        cfg = _tiny_config(num_hidden_layers=1, num_hash_layers=1, compress_ratios=[0])
        model = DeepseekV4ForCausalLM(
            cfg,
            backend=BackendConfig(
                attn="sdpa",
                linear="torch",
                rms_norm="torch",
                rope_fusion=False,
                enable_hf_state_dict_adapter=False,
                dispatcher="torch",
                experts="torch_mm",
            ),
        )

        gate = model.model.layers["0"].mlp.gate
        assert gate.tid2eid.dtype == torch.int64
        cast_model_to_dtype(model, torch.bfloat16)
        assert gate.tid2eid.dtype == torch.int64

        gate.set_input_ids(torch.tensor([[1, 2, 3]]))
        _, indices, _ = gate(torch.zeros(3, cfg.hidden_size), torch.ones(3, dtype=torch.bool))
        assert indices.dtype == torch.long

    def test_initialize_weights_builds_valid_hash_routes(self):
        cfg = _tiny_config(num_hidden_layers=1, num_hash_layers=1, compress_ratios=[0])
        model = DeepseekV4ForCausalLM(
            cfg,
            backend=BackendConfig(
                attn="sdpa",
                linear="torch",
                rms_norm="torch",
                rope_fusion=False,
                enable_hf_state_dict_adapter=False,
                dispatcher="torch",
                experts="torch_mm",
            ),
        )
        gate = model.model.layers["0"].mlp.gate
        with torch.no_grad():
            gate.weight.fill_(float("nan"))
            gate.tid2eid.zero_()

        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

        assert torch.isfinite(gate.weight).all()
        assert torch.count_nonzero(gate.weight) > 0
        sorted_routes = gate.tid2eid.sort(dim=-1).values
        assert torch.all(sorted_routes[:, 1:] != sorted_routes[:, :-1])
        expert_load = torch.bincount(gate.tid2eid.flatten(), minlength=gate.n_experts)
        assert expert_load.max() - expert_load.min() <= 1

    def test_hc_comb_transpose_used_at_attn_and_mlp_sites(self):
        """Both HC expand sites mix residual streams as ``comb.T @ x``.

        The fake HC modules intentionally expose only ``forward`` so this also
        guards against bypassing module hooks via direct ``compute_weights`` calls.
        """

        class _FixedHC(torch.nn.Module):
            def __init__(self, comb):
                super().__init__()
                self.register_buffer("comb", comb)

            def forward(self, hidden_streams):
                bsz, seq, hc_mult = hidden_streams.shape[:3]
                pre = torch.zeros(bsz, seq, hc_mult, dtype=torch.float32, device=hidden_streams.device)
                post = torch.zeros_like(pre)
                return pre, post, self.comb.expand(bsz, seq, -1, -1)

        class _ZeroAttention(torch.nn.Module):
            def forward(self, hidden_states, **kwargs):
                return torch.zeros_like(hidden_states), None

        class _ZeroMLP(torch.nn.Module):
            def forward(self, hidden_states, padding_mask=None):
                return torch.zeros_like(hidden_states)

        cfg = _tiny_config(num_hidden_layers=1, num_hash_layers=0, compress_ratios=[0])
        model = _make_model(cfg)
        block = model.model.layers["0"]
        block.attn_hc = _FixedHC(
            torch.tensor(
                [
                    [1.0, 2.0, 0.0, 1.0],
                    [0.0, 1.0, 3.0, 0.0],
                    [4.0, 0.0, 1.0, 2.0],
                    [0.0, 5.0, 0.0, 1.0],
                ]
            )
        )
        block.ffn_hc = _FixedHC(
            torch.tensor(
                [
                    [2.0, 0.0, 1.0, 0.0],
                    [1.0, 3.0, 0.0, 2.0],
                    [0.0, 1.0, 4.0, 0.0],
                    [5.0, 0.0, 1.0, 1.0],
                ]
            )
        )
        block.self_attn = _ZeroAttention()
        block.mlp = _ZeroMLP()
        block.input_layernorm = torch.nn.Identity()
        block.post_attention_layernorm = torch.nn.Identity()

        x = torch.arange(cfg.hc_mult * cfg.hidden_size, dtype=torch.float32).view(1, 1, cfg.hc_mult, cfg.hidden_size)
        expected = torch.matmul(block.attn_hc.comb.transpose(-1, -2), x)
        expected = torch.matmul(block.ffn_hc.comb.transpose(-1, -2), expected)
        wrong_orientation = torch.matmul(block.ffn_hc.comb, torch.matmul(block.attn_hc.comb, x))

        actual = block(x, position_embeddings=(torch.empty(0), torch.empty(0)))

        torch.testing.assert_close(actual, expected)
        assert not torch.allclose(expected, wrong_orientation)

    def test_thd_pp_intermediate_hidden_state_keeps_shape(self):
        """Packed-sequence PP stages must not add a fake batch dim to hidden states."""

        class _HiddenStage(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                del input_ids, kwargs
                return torch.zeros(1, 8, 4, 16)

        model = DeepseekV4ForCausalLM.__new__(DeepseekV4ForCausalLM)
        torch.nn.Module.__init__(model)
        model.model = _HiddenStage()
        model.lm_head = None
        model.mtp = None
        model.mtp_config = MTPConfig()

        out = model(torch.ones(1, 8, dtype=torch.long), qkv_format="thd")

        assert out.shape == (1, 8, 4, 16)

    def test_thd_first_stage_keeps_batch_axis(self):
        """DSV4 packed sequence uses seq_lens masks while preserving [1, T] inputs."""
        cfg = _tiny_config(num_hidden_layers=0, compress_ratios=[])
        model = _make_model(cfg)

        input_ids = torch.ones(1, 8, dtype=torch.long)
        position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)
        seq_lens = torch.tensor([[8]], dtype=torch.long)

        with torch.no_grad():
            out = model(
                input_ids,
                position_ids=position_ids,
                qkv_format="thd",
                seq_lens=seq_lens,
                seq_lens_padded=seq_lens,
            )

        assert out.logits.shape == (1, 8, cfg.vocab_size)

    def test_thd_cp1_accepts_standard_cu_seqlens(self, monkeypatch):
        """A non-CP packed caller should not need to translate metadata for DSV4."""
        cfg = _tiny_config(num_hidden_layers=0, compress_ratios=[])
        model = _make_model(cfg)
        captured = {}

        def fake_packed_mask(seq_lens, seq_len, dtype, device, sliding_window=None):
            captured["seq_lens"] = seq_lens
            return torch.zeros(1, 1, seq_len, seq_len, dtype=dtype, device=device)

        monkeypatch.setattr(dsv4_model_module, "build_packed_causal_padding_mask", fake_packed_mask)

        cu_seqlens = torch.tensor([0, 3, 8], dtype=torch.int32)
        with torch.no_grad():
            out = model(
                torch.ones(1, 8, dtype=torch.long),
                position_ids=torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]]),
                qkv_format="thd",
                cu_seqlens=cu_seqlens,
                cu_seqlens_padded=cu_seqlens,
            )

        assert out.logits.shape == (1, 8, cfg.vocab_size)
        torch.testing.assert_close(captured["seq_lens"], torch.tensor([[3, 5]], dtype=torch.int32))

    def test_thd_cp_normalizes_packed_ids_and_derives_padding_mask(self, monkeypatch):
        cfg = _tiny_config(num_hidden_layers=0, compress_ratios=[])
        model = _make_model(cfg)
        cp_group = object()
        captured = {}

        monkeypatch.setattr(dsv4_model_module, "dsv4_cp_enabled", lambda group: group is cp_group)

        def fake_packed_cp_mask(**kwargs):
            captured.update(kwargs)
            seq_len = kwargs["position_ids"].shape[-1]
            return torch.zeros(1, 1, seq_len, seq_len, dtype=kwargs["dtype"], device=kwargs["device"])

        monkeypatch.setattr(dsv4_model_module, "build_dsv4_cp_packed_causal_padding_mask", fake_packed_cp_mask)

        input_ids = torch.ones(1, 4, dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 0]])
        with torch.no_grad():
            out = model(
                input_ids,
                attention_mask=attention_mask,
                qkv_format="thd",
                seq_lens_padded=torch.tensor([[4]]),
                packed_seq_ids=torch.ones(4, dtype=torch.int32),
                _dsv4_cp_group=cp_group,
            )

        assert out.logits.shape == (1, 4, cfg.vocab_size)
        assert captured["packed_seq_ids"].dtype == torch.long
        assert captured["packed_seq_ids"].shape == (1, 4)
        torch.testing.assert_close(captured["padding_mask"], torch.tensor([[False, False, False, True]]))
        assert captured["cp_group"] is cp_group

    @_REQUIRES_CUDA
    def test_forward_shape(self):
        """Forward pass produces logits of the right shape."""
        cfg = _tiny_config()
        model = _make_model(cfg)

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
        with torch.no_grad():
            # forward returns DeepseekV4CausalLMOutput; pull .logits.
            logits = model(input_ids).logits

        assert logits.shape == (bsz, seq, cfg.vocab_size), f"unexpected shape {logits.shape}"
        assert not logits.isnan().any(), "logits contain NaN"
        assert not logits.isinf().any(), "logits contain Inf"

    @_REQUIRES_CUDA
    def test_backward(self):
        """Backward pass produces gradients on all trainable parameters."""
        cfg = _tiny_config()
        model = _make_model(cfg)
        model.train()

        bsz, seq = 2, 8
        input_ids = torch.randint(0, cfg.vocab_size, (bsz, seq))
        labels = torch.randint(0, cfg.vocab_size, (bsz, seq))

        logits = model(input_ids).logits
        loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), labels.view(-1))
        loss.backward()

        # The Indexer's parameters (``compressor.indexer.*``) feed the top-k
        # selection of compressed positions; the indices are integer-valued
        # so they are non-differentiable by design (this matches the official
        # DSV4 inference reference, where those weights are frozen at load
        # time).  Exclude them from the gradient-presence check.
        missing_grad = [
            n
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is None and ".compressor.indexer." not in n
        ]
        assert not missing_grad, f"parameters missing gradients: {missing_grad}"
        assert loss.item() > 0, "loss is zero or negative"

    def test_hc_params_registered(self):
        """HC params are registered on every block via the
        ``DeepseekV4HyperConnection`` submodules ``attn_hc`` / ``ffn_hc``
        (the flat ``hc_attn_*`` / ``hc_ffn_*`` keys exist only in the HF
        checkpoint and are renamed on load).
        """
        cfg = _tiny_config()
        model = _make_model(cfg)

        hc_param_names = {"fn", "base", "scale"}
        for layer_id in range(cfg.num_hidden_layers):
            block = model.model.layers[str(layer_id)]
            for sub_name in ("attn_hc", "ffn_hc"):
                hc = getattr(block, sub_name)
                for pname in hc_param_names:
                    assert hasattr(hc, pname), f"layer {layer_id} missing {sub_name}.{pname}"
                    p = getattr(hc, pname)
                    assert p.dtype == torch.float32, f"{sub_name}.{pname} dtype {p.dtype} != float32"

    def test_hc_attn_fn_shape(self):
        """HC ``fn`` matrix has shape ``(mix_hc, hc_mult * hidden_size)``."""
        cfg = _tiny_config()
        model = _make_model(cfg)

        mix_hc = (2 + cfg.hc_mult) * cfg.hc_mult  # 24
        hc_dim = cfg.hc_mult * cfg.hidden_size  # 256 for tiny

        block = model.model.layers["0"]
        fn = block.attn_hc.fn
        assert fn.shape == (mix_hc, hc_dim), f"attn_hc.fn shape {fn.shape} != ({mix_hc}, {hc_dim})"

    def test_wo_a_weight_shape(self):
        """GroupedOutputProjection has the correct weight shape."""
        cfg = _tiny_config()
        model = _make_model(cfg)

        block = model.model.layers["0"]
        attn = block.self_attn
        n_hpg = cfg.num_attention_heads // cfg.o_groups
        expected = (cfg.o_groups * cfg.o_lora_rank, n_hpg * cfg.head_dim)
        actual = attn.wo_a.weight.shape
        assert actual == expected, f"wo_a weight shape {actual} != {expected}"

    def test_attn_sink_shape_and_dtype(self):
        """``sinks`` is a per-head float32 scalar vector consumed by
        ``eager_attention_with_sink`` (HF PR 45616 named it ``sinks``;
        the HF checkpoint key ``attn_sink`` is renamed to it on load).
        """
        cfg = _tiny_config()
        model = _make_model(cfg)

        block = model.model.layers["0"]
        sink = block.self_attn.sinks
        assert sink.shape == (cfg.num_attention_heads,), f"unexpected shape {sink.shape}"
        assert sink.dtype == torch.float32, f"unexpected dtype {sink.dtype}"

    @_REQUIRES_CUDA
    def test_different_seq_lengths(self):
        """Model runs without error for several sequence lengths."""
        cfg = _tiny_config()
        model = _make_model(cfg)

        # ``Float32RMSNorm.forward`` is ``@torch.compile``'d.  Iterating four
        # different sequence lengths through ``compress_ratio>0`` layers (which
        # build masks of shape ``[..., n_pooled]`` that varies per call)
        # explodes Dynamo's per-shape recompile cache.  Disable the compile
        # for this test — we only care about shape correctness here.
        import torch._dynamo as _dynamo

        with _dynamo.config.patch(recompile_limit=64), _dynamo.config.patch(fail_on_recompile_limit_hit=False):
            for seq in [1, 4, 16, 32]:
                input_ids = torch.randint(0, cfg.vocab_size, (1, seq))
                with torch.no_grad():
                    logits = model(input_ids).logits
                assert logits.shape == (1, seq, cfg.vocab_size)


if __name__ == "__main__":
    import sys

    tests = TestDeepseekV4ModelSmoke()
    cases = [
        ("forward shape", tests.test_forward_shape),
        ("backward + gradients", tests.test_backward),
        ("HC params registered", tests.test_hc_params_registered),
        ("HC attn_fn shape", tests.test_hc_attn_fn_shape),
        ("wo_a weight shape", tests.test_wo_a_weight_shape),
        ("attn_sink shape/dtype", tests.test_attn_sink_shape_and_dtype),
        ("different seq lengths", tests.test_different_seq_lengths),
    ]

    failed = []
    for name, fn in cases:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed.append(name)

    print()
    if failed:
        print(f"FAILED: {len(failed)}/{len(cases)} tests")
        sys.exit(1)
    else:
        print(f"All {len(cases)} tests passed.")
