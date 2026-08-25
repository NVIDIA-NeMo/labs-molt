# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
from torch import nn

import molt.models.loading as loading_module
from molt.models.base import BaseModel
from molt.models.loading import _automodel_supports_thd_packing


class _DeclaredTHDModel(nn.Module):
    _molt_automodel_custom = True

    class ModelCapabilities:
        supports_tp = False
        supports_cp = False
        supports_pp = False
        supports_ep = False
        supports_thd = True


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.aux_loss_coeff = 0.0
        self._track_load_balance = False


Gate.__module__ = "nemo_automodel.components.moe.fake"


def test_thd_support_uses_automodel_capability_declaration():
    assert _automodel_supports_thd_packing(_DeclaredTHDModel())


def test_non_automodel_module_never_claims_thd_support():
    assert not _automodel_supports_thd_packing(nn.Linear(2, 2))


def test_hf_fallback_moe_aux_loss_fails_fast():
    with pytest.raises(NotImplementedError, match="MoE auxiliary loss"):
        BaseModel(nn.Linear(2, 2), moe_aux_loss_coef=0.01)


def test_hf_fallback_moe_model_fails_fast_without_aux_loss_or_ep():
    model = nn.Linear(2, 2)
    model.config = SimpleNamespace(
        architectures=["MixtralForCausalLM"],
        text_config=SimpleNamespace(num_local_experts=8),
    )
    with pytest.raises(NotImplementedError, match="MoE model training"):
        BaseModel(model)


def test_preinstantiated_native_model_records_thd_packing_layout():
    wrapped = BaseModel(_DeclaredTHDModel(), packing_samples=True)

    assert wrapped.packing_layout == "thd"


def test_preinstantiated_hf_model_configures_indexed_mask_packing(monkeypatch):
    import nemo_automodel.components.models.common.packing as packing

    configured = []
    monkeypatch.setattr(packing, "get_attn_implementation", lambda *_args, **_kwargs: "flash_attention_2")
    monkeypatch.setattr(packing, "configure_packing", configured.append)

    wrapped = BaseModel(nn.Linear(2, 2), packing_samples=True)

    assert wrapped.packing_layout == "indexed_mask"
    assert configured == ["flash_attention_2"]


def test_loaded_hf_fallback_configures_indexed_mask_packing(monkeypatch):
    import nemo_automodel
    import nemo_automodel.components.distributed.config as distributed_config
    import nemo_automodel.components.distributed.mesh as distributed_mesh
    import nemo_automodel.components.models.common.packing as packing
    import molt.utils.utils as utils

    model = nn.Linear(2, 2)
    model.config = SimpleNamespace(use_cache=True)
    load_kwargs = {}
    configured = []

    class _FakeAutoModel:
        @classmethod
        def from_pretrained(cls, _path, **kwargs):
            load_kwargs.update(kwargs)
            return model

    monkeypatch.setattr(loading_module, "_detect_moe_arch", lambda _model: False)
    monkeypatch.setattr(loading_module, "_will_use_hf_model", lambda _path: True)
    monkeypatch.setattr(loading_module, "_mtp_off_kwargs", lambda _path: {})
    monkeypatch.setattr(utils, "convert_to_torch_dtype", lambda _dtype: None)
    monkeypatch.setattr(utils, "is_vlm_model", lambda _path: False)
    monkeypatch.setattr(nemo_automodel, "NeMoAutoModelForCausalLM", _FakeAutoModel)
    monkeypatch.setattr(distributed_config, "DistributedSetup", lambda **_kwargs: object())
    monkeypatch.setattr(distributed_mesh.MeshContext, "from_meshes", lambda *_args: object())
    monkeypatch.setattr(packing, "get_attn_implementation", lambda *_args, **_kwargs: "flash_attention_2")
    monkeypatch.setattr(packing, "configure_packing", configured.append)

    wrapped = BaseModel("dense-model", packing_samples=True)

    assert wrapped.packing_layout == "indexed_mask"
    assert load_kwargs["has_packed_sequence"] is True
    assert configured == ["flash_attention_2"]


@pytest.mark.parametrize("mesh_name", ["cp", "pp"])
def test_hf_indexed_mask_packing_rejects_cp_and_pp(monkeypatch, mesh_name):
    class _Mesh:
        mesh_dim_names = (mesh_name,)

        def __getitem__(self, _name):
            return SimpleNamespace(size=lambda: 2)

    monkeypatch.setattr(loading_module, "_detect_moe_arch", lambda _model: False)
    monkeypatch.setattr(loading_module, "_will_use_hf_model", lambda _path: True)

    with pytest.raises(NotImplementedError, match="cp_size=1 and pp_size=1"):
        BaseModel("dense-model", packing_samples=True, device_mesh=_Mesh())


def test_hf_indexed_mask_packing_rejects_non_fa2(monkeypatch):
    monkeypatch.setattr(loading_module, "_detect_moe_arch", lambda _model: False)
    monkeypatch.setattr(loading_module, "_will_use_hf_model", lambda _path: True)

    with pytest.raises(ValueError, match="flash_attention_2"):
        BaseModel("dense-model", packing_samples=True, attn_implementation="flash_attention_3")


def test_automodel_native_moe_uses_aux_loss_autograd_coefficient_without_scalar_tracking():
    model = _DeclaredTHDModel()
    model.gate = Gate()
    model.config = SimpleNamespace(num_local_experts=8, router_aux_loss_coef=0.0)
    model.moe_layer = nn.Module()
    model.moe_layer.moe_config = SimpleNamespace(aux_loss_coeff=0.0)

    wrapped = BaseModel(model, moe_aux_loss_coef=0.25)

    assert wrapped.model is model
    assert model.gate.aux_loss_coeff == pytest.approx(0.25)
    assert model.config.router_aux_loss_coef == pytest.approx(0.25)
    assert model.moe_layer.moe_config.aux_loss_coeff == pytest.approx(0.25)
    assert model.gate._track_load_balance is False
