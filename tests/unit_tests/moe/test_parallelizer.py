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

import sys
import types
from contextlib import contextmanager, nullcontext
from unittest.mock import MagicMock, patch

import pytest

# torch.utils.checkpoint.create_selective_checkpoint_contexts returns
# ``(forward_ctx, recompute_ctx)``, and apply_ac's context wrappers unpack it,
# so the stubs below must return a pair rather than a bare sentinel.
SELECTIVE_CTX = ("CTX_FORWARD", "CTX_RECOMPUTE")


class DummyParam:
    """Mock parameter with requires_grad attribute."""

    def __init__(self, requires_grad=True):
        self.requires_grad = requires_grad


class DummyExperts:
    def __init__(self):
        self._params = {"weight": DummyParam()}

    def named_parameters(self, recurse=False):
        for name, param in self._params.items():
            yield name, param

    def register_parameter(self, name, param):
        self._params[name] = param

    def parameters(self):
        for p in self._params.values():
            yield p


class DummyMoE:
    def __init__(self):
        self.experts = DummyExperts()


class DummyBlock:
    def __init__(self, mlp=None):
        self.mlp = mlp if mlp is not None else DummyMoE()


class LayerContainer:
    def __init__(self, blocks):
        self._blocks = blocks
        self.registered = {}

    def children(self):
        return iter(self._blocks)

    def named_children(self):
        return [(str(i), b) for i, b in enumerate(self._blocks)]

    def register_module(self, name, module):
        self.registered[name] = module


class DummyModel:
    def __init__(self, blocks, embed_tokens=None, embed_norm=None, lm_head=None, audio_tower=None, visual=None):
        self.layers = LayerContainer(blocks)
        self.embed_tokens = embed_tokens
        self.embed_norm = embed_norm
        self.lm_head = lm_head
        self.audio_tower = audio_tower
        self.visual = visual

    def parameters(self):
        """Aggregate child parameters like ``nn.Module.parameters()``."""
        for child in (
            self.layers,
            self.embed_tokens,
            self.embed_norm,
            self.lm_head,
            self.audio_tower,
            self.visual,
        ):
            child_parameters = getattr(child, "parameters", None)
            if callable(child_parameters):
                yield from child_parameters()


def _install_torch_and_layers_stubs(monkeypatch):
    # Build minimal torch stub hierarchy
    torch_stub = types.ModuleType("torch")

    # nn submodule
    nn_stub = types.ModuleType("torch.nn")

    class Parameter:
        def __init__(self, data=None):
            self.data = data

    class Module:
        pass

    # Real containers, so production code can use plain ``isinstance`` checks and
    # runtime ``torch.Tensor`` annotations instead of defensive ``getattr``.
    # Without these the stub silently changes container-detection behavior, and
    # modules annotated with ``torch.Tensor`` (e.g. ``shared/tied_weights.py``)
    # only import when some earlier test happened to cache them under real torch.
    class ModuleList(list):
        def named_children(self):
            return [(str(i), child) for i, child in enumerate(self)]

    class ModuleDict(dict):
        def named_children(self):
            return list(self.items())

    class Tensor:
        pass

    nn_stub.Parameter = Parameter
    nn_stub.Module = Module
    nn_stub.ModuleList = ModuleList
    nn_stub.ModuleDict = ModuleDict
    torch_stub.nn = nn_stub
    torch_stub.Tensor = Tensor

    # cuda submodule
    cuda_stub = types.ModuleType("torch.cuda")

    class Stream:
        def __init__(self):
            pass

    cuda_stub.Stream = Stream
    torch_stub.cuda = cuda_stub

    # distributed submodules and symbols
    dist_stub = types.ModuleType("torch.distributed")

    # device_mesh
    device_mesh_stub = types.ModuleType("torch.distributed.device_mesh")

    class DeviceMesh:
        def __init__(self, *args, **kwargs):
            pass

    device_mesh_stub.DeviceMesh = DeviceMesh

    # fsdp
    fsdp_stub = types.ModuleType("torch.distributed.fsdp")

    def fully_shard(*args, **kwargs):
        return None

    fsdp_stub.fully_shard = fully_shard

    fsdp_fully_stub = types.ModuleType("torch.distributed.fsdp._fully_shard")

    class MixedPrecisionPolicy:
        def __init__(self, *args, **kwargs):
            pass

    class CPUOffloadPolicy:
        def __init__(self, *args, **kwargs):
            pass

    class OffloadPolicy:
        def __init__(self, *args, **kwargs):
            pass

    fsdp_stub.MixedPrecisionPolicy = MixedPrecisionPolicy
    fsdp_stub.CPUOffloadPolicy = CPUOffloadPolicy
    fsdp_fully_stub.MixedPrecisionPolicy = MixedPrecisionPolicy
    fsdp_fully_stub.OffloadPolicy = OffloadPolicy

    # tensor
    tensor_stub = types.ModuleType("torch.distributed.tensor")

    def distribute_module(*args, **kwargs):
        return "DISTRIBUTED"

    def distribute_tensor(*args, **kwargs):
        return object()

    class Shard:
        def __init__(self, *args, **kwargs):
            pass

    class Replicate:
        def __init__(self, *args, **kwargs):
            pass

    tensor_stub.distribute_module = distribute_module
    tensor_stub.distribute_tensor = distribute_tensor
    tensor_stub.Replicate = Replicate
    tensor_stub.Shard = Shard

    # tensor.parallel
    tp_stub = types.ModuleType("torch.distributed.tensor.parallel")

    class ParallelStyle:
        pass

    def parallelize_module(*args, **kwargs):
        return None

    tp_stub.ParallelStyle = ParallelStyle
    tp_stub.parallelize_module = parallelize_module

    # algorithms._checkpoint.checkpoint_wrapper
    alg_stub = types.ModuleType("torch.distributed.algorithms")
    alg_cp_stub = types.ModuleType("torch.distributed.algorithms._checkpoint")
    cpw_stub = types.ModuleType("torch.distributed.algorithms._checkpoint.checkpoint_wrapper")

    def checkpoint_wrapper(*args, **kwargs):
        return args[0]

    class CheckpointImpl:
        NO_REENTRANT = "no_reentrant"
        REENTRANT = "reentrant"

    cpw_stub.checkpoint_wrapper = checkpoint_wrapper
    # components/distributed/activation_checkpointing.py imports this at module
    # scope; without it that module only imports when an earlier test happened to
    # cache it under real torch, making this file order-dependent.
    cpw_stub.CheckpointImpl = CheckpointImpl

    # utils module hierarchy
    utils_stub = types.ModuleType("torch.utils")

    # utils.data
    utils_data_stub = types.ModuleType("torch.utils.data")

    class PinMemory:
        @staticmethod
        def _pin_memory_loop(*args, **kwargs):
            pass

        @staticmethod
        def pin_memory(*args, **kwargs):
            pass

    class DataUtils:
        pin_memory = PinMemory

    utils_data_stub._utils = DataUtils

    # utils.checkpoint
    utils_checkpoint_stub = types.ModuleType("torch.utils.checkpoint")

    class CheckpointPolicy:
        MUST_SAVE = 1
        PREFER_RECOMPUTE = 2

    def create_selective_checkpoint_contexts(policy_factory):
        return SELECTIVE_CTX

    utils_checkpoint_stub.CheckpointPolicy = CheckpointPolicy
    utils_checkpoint_stub._allowed_determinism_checks_to_fns = {"default": object(), "none": object()}
    utils_checkpoint_stub.create_selective_checkpoint_contexts = create_selective_checkpoint_contexts

    # Router ops used by the targeted activation-checkpointing policy.
    aten = types.SimpleNamespace(
        mm=types.SimpleNamespace(default=object()),
        topk=types.SimpleNamespace(default=object()),
    )
    torch_stub.ops = types.SimpleNamespace(aten=aten)

    # dtype and device classes for type annotations
    class dtype:
        pass

    class device:
        pass

    class Tensor:
        pass

    torch_stub.dtype = dtype
    torch_stub.device = device
    torch_stub.Tensor = Tensor

    # common dtypes referenced by code
    torch_stub.bfloat16 = object()
    torch_stub.float32 = object()

    # register into sys.modules via monkeypatch
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_stub)
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda_stub)
    monkeypatch.setitem(sys.modules, "torch.distributed", dist_stub)
    monkeypatch.setitem(sys.modules, "torch.distributed.device_mesh", device_mesh_stub)
    monkeypatch.setitem(sys.modules, "torch.distributed.fsdp", fsdp_stub)
    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.fsdp._fully_shard",
        fsdp_fully_stub,
    )
    monkeypatch.setitem(sys.modules, "torch.distributed.tensor", tensor_stub)
    monkeypatch.setitem(sys.modules, "torch.distributed.tensor.parallel", tp_stub)
    monkeypatch.setitem(sys.modules, "torch.distributed.algorithms", alg_stub)
    monkeypatch.setitem(sys.modules, "torch.distributed.algorithms._checkpoint", alg_cp_stub)
    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
        cpw_stub,
    )
    monkeypatch.setitem(sys.modules, "torch.utils", utils_stub)
    monkeypatch.setitem(sys.modules, "torch.utils.data", utils_data_stub)
    monkeypatch.setitem(sys.modules, "torch.utils.checkpoint", utils_checkpoint_stub)

    # Stub heavy layers import as well to avoid real dependencies
    layers_stub = types.ModuleType("nemo_automodel.components.moe.layers")

    class GroupedExpertsDeepEP:
        pass

    class MoE:
        pass

    class Gate:
        pass

    layers_stub.GroupedExpertsDeepEP = GroupedExpertsDeepEP
    layers_stub.Gate = Gate
    layers_stub.MoE = MoE
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.moe.layers", layers_stub)

    # Stub experts module to avoid importing torch.nn.functional
    experts_stub = types.ModuleType("nemo_automodel.components.moe.experts")

    class GroupedExpertsTE:
        pass

    experts_stub.GroupedExpertsDeepEP = GroupedExpertsDeepEP
    experts_stub.GroupedExpertsTE = GroupedExpertsTE
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.moe.experts", experts_stub)

    # Stub MoK experts to keep this import-isolation test independent of the
    # real DTensor stack, just like the TE and DeepEP expert implementations.
    mok_experts_stub = types.ModuleType("nemo_automodel.components.moe.mok_experts")

    class GroupedExpertsMoK:
        pass

    mok_experts_stub.GroupedExpertsMoK = GroupedExpertsMoK
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.moe.mok_experts", mok_experts_stub)


def _import_parallelizer_with_stubs(monkeypatch):
    import importlib

    # ensure fresh import of parallelizer
    for mod in [
        "nemo_automodel.components.moe.parallelizer",
        "nemo_automodel.components.moe.layers",
        "nemo_automodel.components.moe.experts",
        "nemo_automodel.components.moe.mok_experts",
        "nemo_automodel.components.distributed.pipelining",
        "nemo_automodel.components.distributed.pipelining.config",
        "nemo_automodel.components.distributed.pipelining.hf_utils",
        "nemo_automodel.components.distributed.mesh_utils",
        "nemo_automodel.components.distributed.parallelizer_utils",
    ]:
        if mod in sys.modules:
            sys.modules.pop(mod)

    _install_torch_and_layers_stubs(monkeypatch)

    # Stub the distributed package, config normalization, pipelining module, and hf_utils.
    distributed_stub = types.ModuleType("nemo_automodel.components.distributed")
    distributed_stub.__path__ = []
    config_stub = types.ModuleType("nemo_automodel.components.distributed.config")
    config_stub.normalize_activation_checkpointing_scope = lambda scope: (
        (scope,) if isinstance(scope, str) else tuple(scope)
    )
    pipelining_stub = types.ModuleType("nemo_automodel.components.distributed.pipelining")
    pipelining_stub.__path__ = []
    pipelining_config_stub = types.ModuleType("nemo_automodel.components.distributed.pipelining.config")
    hf_utils_stub = types.ModuleType("nemo_automodel.components.distributed.pipelining.hf_utils")

    class PipelineConfig:
        pass

    def get_text_module(model):
        """Return model.model if exists, otherwise model."""
        if hasattr(model, "model") and model.model is not None:
            return model.model
        return model

    pipelining_config_stub.PipelineConfig = PipelineConfig
    hf_utils_stub.get_text_module = get_text_module
    pipelining_stub.config = pipelining_config_stub
    pipelining_stub.hf_utils = hf_utils_stub

    monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed", distributed_stub)
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed.config", config_stub)
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed.pipelining", pipelining_stub)
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed.pipelining.config", pipelining_config_stub)
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed.pipelining.hf_utils", hf_utils_stub)

    mesh_utils_stub = types.ModuleType("nemo_automodel.components.distributed.mesh_utils")
    mesh_utils_stub.get_submesh = lambda mesh, axis_names: mesh[axis_names]
    mesh_utils_stub.get_fsdp_dp_mesh = lambda mesh, *_axis_names: mesh[("dp_replicate", "dp_shard_cp")]
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed.mesh_utils", mesh_utils_stub)

    parallelizer_utils_stub = types.ModuleType("nemo_automodel.components.distributed.parallelizer_utils")

    def fully_shard_by_dtype(
        module,
        *,
        mesh,
        mp_policy,
        offload_policy,
        fp32_compute_module_names=(),
        reshard_after_forward=None,
        ignored_params=None,
        fully_shard_fn=None,
    ):
        kwargs = {
            "mesh": mesh,
            "mp_policy": mp_policy,
            "offload_policy": offload_policy,
        }
        if reshard_after_forward is not None:
            kwargs["reshard_after_forward"] = reshard_after_forward
        if ignored_params:
            kwargs["ignored_params"] = ignored_params
        fully_shard_fn(module, **kwargs)

    parallelizer_utils_stub.fully_shard_by_dtype = fully_shard_by_dtype
    parallelizer_utils_stub.get_internal_fsdp_mp_policy = lambda mp_policy: ("INTERNAL_MP_POLICY", mp_policy)
    parallelizer_utils_stub.configure_fsdp_unused_param_reduction = lambda module: 0

    def reject_unsupported_mtp_cp(model):
        if model.supports.mtp_enabled and not model.supports.supports_mtp_cp:
            raise RuntimeError("Model does not support MTP with context parallelism")

    parallelizer_utils_stub.reject_unsupported_mtp_cp = reject_unsupported_mtp_cp

    def reject_unsupported_mtp_cp_pp(model):
        is_pp_stage_fn = getattr(model, "_is_pipeline_parallel_stage", None)
        if (
            model.supports.mtp_enabled
            and not model.supports.supports_mtp_cp_pp
            and callable(is_pp_stage_fn)
            and is_pp_stage_fn()
        ):
            raise NotImplementedError("MTP with context and pipeline parallelism is not supported")

    parallelizer_utils_stub.reject_unsupported_mtp_cp_pp = reject_unsupported_mtp_cp_pp

    monkeypatch.setitem(
        sys.modules,
        "nemo_automodel.components.distributed.parallelizer_utils",
        parallelizer_utils_stub,
    )
    distributed_package = importlib.import_module("nemo_automodel.components.distributed")
    monkeypatch.setattr(distributed_package, "parallelizer_utils", parallelizer_utils_stub, raising=False)

    # Stub dtype_from_str utility
    shared_utils_stub = types.ModuleType("nemo_automodel.shared.utils")
    shared_utils_stub.dtype_from_str = lambda val, default=None: default
    monkeypatch.setitem(sys.modules, "nemo_automodel.shared.utils", shared_utils_stub)

    tied_weights_stub = types.ModuleType("nemo_automodel.shared.tied_weights")
    tied_weights_stub.ensure_tied_lm_head = lambda model: None
    monkeypatch.setitem(sys.modules, "nemo_automodel.shared.tied_weights", tied_weights_stub)

    activation_checkpointing_stub = types.ModuleType("nemo_automodel.components.distributed.activation_checkpointing")
    activation_checkpointing_stub.ensure_fsdp_ops_sac_ignored = lambda: None
    activation_checkpointing_stub.ensure_profiler_ops_sac_ignored = lambda: None
    activation_checkpointing_stub.transformer_engine_attention_backend_snapshot_context_fn = lambda context_fn=None: (
        context_fn() if context_fn is not None else (nullcontext(), nullcontext())
    )
    monkeypatch.setitem(
        sys.modules,
        "nemo_automodel.components.distributed.activation_checkpointing",
        activation_checkpointing_stub,
    )

    # apply_ac wraps each block's context in the DeepEP dispatch-replay recorder,
    # which imports fused_a2a lazily. Stub it so this module does not depend on
    # some earlier test having imported it under the real torch.
    fused_a2a_stub = types.ModuleType("nemo_automodel.components.moe.megatron.fused_a2a")

    class _StubDispatchReplayRecorder:
        def __init__(self):
            self.replay_misses = 0
            self.rewind_count = 0

        def rewind(self):
            self.rewind_count += 1

    @contextmanager
    def _stub_dispatch_replay_scope(recorder, mode):
        yield

    fused_a2a_stub.DispatchReplayRecorder = _StubDispatchReplayRecorder
    fused_a2a_stub.dispatch_replay_scope = _stub_dispatch_replay_scope
    monkeypatch.setitem(
        sys.modules,
        "nemo_automodel.components.moe.megatron.fused_a2a",
        fused_a2a_stub,
    )

    distributed_config_stub = types.ModuleType("nemo_automodel.components.distributed.config")
    distributed_config_stub.normalize_activation_checkpointing_scope = lambda value: (
        (value,) if isinstance(value, str) else tuple(value or ("all",))
    )
    monkeypatch.setitem(
        sys.modules,
        "nemo_automodel.components.distributed.config",
        distributed_config_stub,
    )

    parallel_styles_stub = types.ModuleType("nemo_automodel.components.distributed.parallel_styles")
    parallel_styles_stub.translate_to_lora = lambda style: style
    monkeypatch.setitem(
        sys.modules,
        "nemo_automodel.components.distributed.parallel_styles",
        parallel_styles_stub,
    )

    return importlib.import_module("nemo_automodel.components.moe.parallelizer")


def test_expert_parallel_apply_calls_distribute_module(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    ep = P.ExpertParallel()
    module = DummyBlock().mlp.experts
    device_mesh = object()

    distribute_module_mock = MagicMock(return_value="DISTRIBUTED")
    monkeypatch.setattr(P, "distribute_module", distribute_module_mock)

    result = ep._apply(module, device_mesh)

    assert result == "DISTRIBUTED"
    assert distribute_module_mock.call_count == 1
    args, kwargs = distribute_module_mock.call_args
    # (module, device_mesh, partition_fn)
    assert args[0] is module
    assert args[1] is device_mesh
    assert callable(args[2])
    # ensure bound to same instance
    assert isinstance(args[2], types.MethodType) and args[2].__self__ is ep


def test_expert_parallel_partition_fn_shards_and_dispatcher(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)

    # make the target module also look like GroupedExpertsDeepEP
    class DummyGrouped(DummyExperts):
        def __init__(self):
            super().__init__()
            self.dispatch_called_with = None

        def init_token_dispatcher(self, ep_mesh):
            self.dispatch_called_with = ep_mesh

        # override register_parameter to avoid strict type checks
        def register_parameter(self, name, param):
            setattr(self, name, param)

    # patch GroupedExpertsDeepEP symbol used in isinstance checks
    monkeypatch.setattr(P, "GroupedExpertsDeepEP", DummyGrouped)

    # mock distribute_tensor and Shard
    shard_sentinel = object()

    def fake_shard(dim):
        assert dim == 0
        return shard_sentinel

    distributed_obj = object()
    distribute_tensor_mock = MagicMock(return_value=distributed_obj)
    monkeypatch.setattr(P, "Shard", fake_shard)
    monkeypatch.setattr(P, "distribute_tensor", distribute_tensor_mock)

    ep = P.ExpertParallel()
    module = DummyGrouped()
    device_mesh = type("Mesh", (), {"ndim": 1})()

    # original parameter should exist
    assert any(True for _ in module.named_parameters(recurse=False))
    ep._partition_fn("any", module, device_mesh)

    # verify distribute_tensor was called for each top-level parameter with Shard(0)
    for _, param in module.named_parameters(recurse=False):
        pass  # push iterator once for coverage; we validate calls below

    assert distribute_tensor_mock.call_count >= 1
    for args, kwargs in distribute_tensor_mock.call_args_list:
        assert args[1] is device_mesh
        assert isinstance(args[2], list) and args[2][0] is shard_sentinel

    # dispatcher must be initialized
    assert module.dispatch_called_with is device_mesh


def test_expert_parallel_partition_fn_preserves_requires_grad(monkeypatch):
    """Test that _partition_fn preserves the requires_grad attribute of parameters."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    class DummyExpertsWithGrad:
        def __init__(self):
            self._params = {}
            # Create parameters with different requires_grad values
            trainable_param = MagicMock()
            trainable_param.requires_grad = True
            frozen_param = MagicMock()
            frozen_param.requires_grad = False
            self._params["trainable_weight"] = trainable_param
            self._params["frozen_weight"] = frozen_param
            self.registered_params = {}

        def named_parameters(self, recurse=False):
            for name, param in self._params.items():
                yield name, param

        def register_parameter(self, name, param):
            self.registered_params[name] = param

    # Mock distribute_tensor to return a mock Parameter-like object
    def fake_distribute_tensor(param, device_mesh, placements):
        # Return a mock that can be wrapped in nn.Parameter
        mock_tensor = MagicMock()
        return mock_tensor

    monkeypatch.setattr(P, "distribute_tensor", fake_distribute_tensor)
    monkeypatch.setattr(P, "Shard", lambda dim: object())
    # Ensure module doesn't match GroupedExpertsDeepEP to skip dispatcher init
    monkeypatch.setattr(P, "GroupedExpertsDeepEP", type("NotMatching", (), {}))

    ep = P.ExpertParallel()
    module = DummyExpertsWithGrad()
    device_mesh = type("Mesh", (), {"ndim": 1})()

    ep._partition_fn("any", module, device_mesh)

    # Verify requires_grad is preserved for each registered parameter
    assert "trainable_weight" in module.registered_params
    assert "frozen_weight" in module.registered_params
    assert module.registered_params["trainable_weight"].requires_grad is True
    assert module.registered_params["frozen_weight"].requires_grad is False


def test_apply_ep_parallelizes_moe_experts(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    # Patch MoE symbol for isinstance
    monkeypatch.setattr(P, "MoE", DummyMoE)
    parallelize_module_mock = MagicMock()
    monkeypatch.setattr(P, "parallelize_module", parallelize_module_mock)

    block = DummyBlock(mlp=DummyMoE())
    model = DummyModel([block])
    ep_mesh = type("Mesh", (), {"size": lambda self: 2})()

    P.apply_ep(model, ep_mesh)

    assert parallelize_module_mock.call_count == 1
    args, kwargs = parallelize_module_mock.call_args
    assert kwargs["module"] is block.mlp.experts
    assert kwargs["device_mesh"] is ep_mesh
    assert isinstance(kwargs["parallelize_plan"], P.ExpertParallel)


def test_apply_ep_excludes_te_owned_experts_from_tp_replica_sync(monkeypatch):
    """TE's plain local expert tensors remain owned by the folded EP mesh."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    class DummyGroupedExpertsTE:
        def __init__(self):
            self.init_token_dispatcher = MagicMock()

    experts = DummyGroupedExpertsTE()
    moe = DummyMoE()
    moe.experts = experts
    block = DummyBlock(mlp=moe)
    model = DummyModel([block])
    ep_mesh = type("Mesh", (), {"size": lambda self: 2})()
    moe_mesh = object()

    monkeypatch.setattr(P, "GroupedExpertsTE", DummyGroupedExpertsTE)
    exclude_mock = MagicMock()
    tp_replicas_stub = types.ModuleType("nemo_automodel.components.distributed.tp_replicas")
    tp_replicas_stub.exclude_from_tp_replica_sync = exclude_mock
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed.tp_replicas", tp_replicas_stub)

    P.apply_ep(model, ep_mesh, moe_mesh=moe_mesh)

    experts.init_token_dispatcher.assert_called_once_with(ep_mesh=ep_mesh, moe_mesh=moe_mesh)
    exclude_mock.assert_called_once_with(experts)


def test_apply_ep_parallelizes_diffusion_style_block_moe(monkeypatch):
    """Diffusion Gemma exposes the MoE branch as block.moe, not block.mlp."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    parallelize_module_mock = MagicMock()
    monkeypatch.setattr(P, "parallelize_module", parallelize_module_mock)

    class DiffusionBlock:
        def __init__(self):
            self.moe = DummyMoE()

    block = DiffusionBlock()
    model = type("Outer", (), {"model": DummyModel([block])})()
    ep_mesh = type("Mesh", (), {"size": lambda self: 2})()

    P.apply_ep(model, ep_mesh)

    assert parallelize_module_mock.call_count == 1
    _, kwargs = parallelize_module_mock.call_args
    assert kwargs["module"] is block.moe.experts
    assert kwargs["device_mesh"] is ep_mesh
    assert isinstance(kwargs["parallelize_plan"], P.ExpertParallel)


def test_apply_ac_wraps_blocks_with_and_without_context(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    wrapper_returns = [object(), object()]

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        assert preserve_rng_state is True
        # if ignore_router=True, context_fn should be provided
        return wrapper_returns.pop(0)

    wrapper_mock = MagicMock(side_effect=fake_wrapper)
    ctx_mock = MagicMock(return_value=SELECTIVE_CTX)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", wrapper_mock)
    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", ctx_mock)

    blocks = [DummyBlock(), DummyBlock()]
    model = DummyModel(blocks)

    # ignore_router=True path - provide explicit hidden_size and num_experts
    P.apply_ac(model, ignore_router=True, hidden_size=7168, num_experts=256)
    assert wrapper_mock.call_count == 2
    # registration should replace both blocks
    assert len(model.layers.registered) == 2

    # reset for ignore_router=False path
    wrapper_returns.extend([object(), object()])
    model = DummyModel([DummyBlock(), DummyBlock()])
    wrapper_mock.reset_mock()
    model.layers.registered.clear()

    P.apply_ac(model, ignore_router=False, hidden_size=7168, num_experts=256)
    # Router replay does not need a selective policy, but TE attention still
    # receives a context that preserves its forward-time backend cache.
    for _, kwargs in wrapper_mock.call_args_list:
        assert callable(kwargs["context_fn"])
    assert len(model.layers.registered) == 2


@pytest.mark.parametrize("selective", [False, True])
def test_apply_ac_skips_model_owned_eager_block(monkeypatch, selective):
    P = _import_parallelizer_with_stubs(monkeypatch)
    if selective:
        dense_stub = types.ModuleType("nemo_automodel.components.distributed.activation_checkpointing")
        dense_stub.make_selective_checkpoint_context_fn = MagicMock(return_value=object())
        dense_stub.SELECTIVE_AC_WRAPPER_FLAG = "_nemo_selective_ac"
        dense_stub.transformer_engine_attention_backend_snapshot_context_fn = lambda context_fn=None: context_fn
        monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed.activation_checkpointing", dense_stub)

    eager_block = DummyBlock()
    eager_block._nemo_disable_activation_checkpointing = True
    wrapped_block = types.SimpleNamespace()
    wrapper_mock = MagicMock(return_value=wrapped_block)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", wrapper_mock)

    model = DummyModel([DummyBlock(), eager_block])
    P.apply_ac(model, ignore_router=True, hidden_size=7168, num_experts=256, selective=selective)

    wrapper_mock.assert_called_once()
    assert model.layers.registered == {"0": wrapped_block}


def test_apply_ac_warns_when_router_is_recomputed(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=lambda block, **kw: block))
    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", MagicMock(return_value=SELECTIVE_CTX))
    logger_mock = MagicMock()
    monkeypatch.setattr(P, "logger", logger_mock)

    # ignore_router=False under (non-selective) AC recomputes the router -> warn.
    P.apply_ac(DummyModel([DummyBlock()]), ignore_router=False, hidden_size=7168, num_experts=256)
    assert logger_mock.warning.call_count == 1
    assert "ignore_router_for_ac" in logger_mock.warning.call_args[0][0]

    # ignore_router=True (the default) saves the router projection and top-k outputs -> no warning.
    logger_mock.reset_mock()
    P.apply_ac(DummyModel([DummyBlock()]), ignore_router=True, hidden_size=7168, num_experts=256)
    logger_mock.warning.assert_not_called()


@pytest.mark.parametrize("selective", [False, True])
def test_apply_ac_replays_hybridep_layout_for_full_and_selective_ac(monkeypatch, selective):
    P = _import_parallelizer_with_stubs(monkeypatch)
    expert = P.GroupedExpertsDeepEP()
    expert.dispatcher_backend = "hybridep"

    class HybridEPModel(DummyModel):
        def modules(self):
            return iter((self, expert))

    if selective:
        activation_checkpointing_stub = sys.modules["nemo_automodel.components.distributed.activation_checkpointing"]
        activation_checkpointing_stub.make_selective_checkpoint_context_fn = lambda: (
            lambda: (nullcontext(), nullcontext())
        )
        activation_checkpointing_stub.SELECTIVE_AC_WRAPPER_FLAG = "_nemo_selective_ac"

    replay_wrapper = MagicMock(side_effect=lambda context_fn: context_fn)
    monkeypatch.setattr(P, "_replay_hybridep_dispatch_on_recompute", replay_wrapper)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=lambda block, **kwargs: block))

    P.apply_ac(
        HybridEPModel([DummyBlock()]),
        ignore_router=True,
        hidden_size=7168,
        num_experts=384,
        selective=selective,
    )

    replay_wrapper.assert_called_once()


def test_apply_ac_uses_generic_wrapper_even_when_block_local_checkpointing_is_available(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)

    class BlockWithLocalAC(DummyBlock):
        def __init__(self):
            super().__init__()
            self.activation_checkpointing = False

        def set_activation_checkpointing(self, enabled=True):
            self.activation_checkpointing = enabled

    block = BlockWithLocalAC()
    model = DummyModel([block])
    wrapped = object()
    wrapper_mock = MagicMock(return_value=wrapped)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", wrapper_mock)

    P.apply_ac(model, ignore_router=True, hidden_size=7168, num_experts=256)

    wrapper_mock.assert_called_once()
    assert wrapper_mock.call_args.kwargs["preserve_rng_state"] is True
    assert callable(wrapper_mock.call_args.kwargs["context_fn"])
    assert block.activation_checkpointing is False
    assert model.layers.registered["0"] is wrapped


def test_apply_ac_custom_policy_saves_router_projection_and_topk(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_policy = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_policy
        captured_policy = policy_cb
        return SELECTIVE_CTX

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        assert preserve_rng_state is True
        assert callable(context_fn)
        # The selective contexts are wrapped for DeepEP dispatch replay, so what
        # comes back is the wrapper's pair rather than SELECTIVE_CTX itself.
        forward_ctx, recompute_ctx = context_fn()
        assert hasattr(forward_ctx, "__enter__")
        assert hasattr(recompute_ctx, "__enter__")
        return block

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    hidden_size = 17
    num_experts = 31
    model = DummyModel([DummyBlock(), DummyBlock()])

    P.apply_ac(model, ignore_router=True, hidden_size=hidden_size, num_experts=num_experts)

    assert captured_policy is not None

    torch_stub = sys.modules["torch"]
    rhs_match = type("Mat", (), {"shape": (hidden_size, num_experts)})()
    rhs_mismatch = type("Mat", (), {"shape": (hidden_size, num_experts + 1)})()

    policy = captured_policy
    must_save = policy(None, torch_stub.ops.aten.mm.default, object(), rhs_match)
    must_save_topk = policy(None, torch_stub.ops.aten.topk.default, object(), 2)
    prefer_recompute_shape = policy(None, torch_stub.ops.aten.mm.default, object(), rhs_mismatch)
    prefer_recompute_func = policy(None, object(), object(), rhs_match)

    assert must_save == P.CheckpointPolicy.MUST_SAVE
    assert must_save_topk == P.CheckpointPolicy.MUST_SAVE
    assert prefer_recompute_shape == P.CheckpointPolicy.PREFER_RECOMPUTE
    assert prefer_recompute_func == P.CheckpointPolicy.PREFER_RECOMPUTE


def _find_call_by_first_arg(mock_obj, target_first_arg):
    for args, kwargs in mock_obj.call_args_list:
        if args and args[0] is target_first_arg:
            return args, kwargs
    return None


def test_apply_fsdp_calls_with_ignored_params_and_shard_for_experts(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    # Patch MoE symbol for isinstance
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    mp_policy_mock = MagicMock(return_value="MP_POLICY")
    shard_sentinel = object()
    replicate_sentinel = object()
    correction_bias = object()
    distributed_correction_bias = object()

    def fake_shard(dim):
        assert dim == 1
        return shard_sentinel

    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", mp_policy_mock)
    monkeypatch.setattr(P, "Replicate", MagicMock(return_value=replicate_sentinel))
    monkeypatch.setattr(P, "Shard", fake_shard)
    distribute_tensor_mock = MagicMock(return_value=distributed_correction_bias)
    monkeypatch.setattr(P, "distribute_tensor", distribute_tensor_mock)

    block = DummyBlock(mlp=DummyMoE())
    block.mlp.gate = P.Gate()
    block.mlp.gate.e_score_correction_bias = correction_bias
    embed = object()
    embed_norm = object()
    lm = object()
    model = DummyModel([block], embed_tokens=embed, embed_norm=embed_norm, lm_head=lm)

    fsdp_mesh = type("Mesh", (), {"ndim": 1, "size": lambda self: 2})()
    ep_shard_mesh = type("Mesh", (), {"size": lambda self: 2})()
    offload_policy = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=True,
        ep_shard_enabled=True,
        ep_shard_mesh=ep_shard_mesh,
        offload_policy=offload_policy,
    )

    distribute_tensor_mock.assert_called_once_with(
        correction_bias,
        device_mesh=fsdp_mesh,
        placements=[replicate_sentinel],
    )
    assert block.mlp.gate.e_score_correction_bias is distributed_correction_bias

    # Experts should have a dedicated shard call
    experts = block.mlp.experts
    experts_call = _find_call_by_first_arg(fully_shard_mock, experts)
    assert experts_call is not None
    _, experts_kwargs = experts_call
    assert experts_kwargs["mesh"] is ep_shard_mesh
    assert experts_kwargs["reshard_after_forward"] is False
    assert experts_kwargs["offload_policy"] is offload_policy
    assert experts_kwargs["mp_policy"] == ("INTERNAL_MP_POLICY", "MP_POLICY")
    assert callable(experts_kwargs["shard_placement_fn"])  # lambda _: Shard(1)

    # Block should be sharded with ignored_params when ep_enabled
    block_call = _find_call_by_first_arg(fully_shard_mock, block)
    assert block_call is not None
    _, block_kwargs = block_call
    assert block_kwargs["mesh"] is fsdp_mesh
    assert block_kwargs["mp_policy"] == "MP_POLICY"
    ignored = block_kwargs.get("ignored_params")
    assert isinstance(ignored, set) and len(ignored) == len(list(experts.parameters()))

    # embed, post-embedding norm, lm_head and model should also be sharded on fsdp_mesh
    embed_call = _find_call_by_first_arg(fully_shard_mock, embed)
    assert embed_call is not None and embed_call[1]["mesh"] is fsdp_mesh

    embed_norm_call = _find_call_by_first_arg(fully_shard_mock, embed_norm)
    assert embed_norm_call is not None and embed_norm_call[1]["mesh"] is fsdp_mesh

    lm_call = _find_call_by_first_arg(fully_shard_mock, lm)
    assert lm_call is not None and lm_call[1]["mesh"] is fsdp_mesh

    model_call = _find_call_by_first_arg(fully_shard_mock, model)
    assert model_call is not None and model_call[1]["mesh"] is fsdp_mesh


def test_apply_fsdp_installs_accumulated_grad_guard(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    guard_mock = MagicMock()
    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "_patch_fsdp_accumulated_grad_guard", guard_mock)
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    P.apply_fsdp(
        model=DummyModel([DummyBlock(mlp=DummyMoE())]),
        fsdp_mesh=object(),
        ep_enabled=False,
        ep_shard_enabled=False,
    )

    guard_mock.assert_called_once_with()


def test_apply_fsdp_routes_strict_fp32_contract_and_expert_exclusions_to_shared_sharder(monkeypatch):
    """MoE uses the dense dtype-aware sharder with the model and EP contracts."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    shared_sharder_mock = MagicMock()
    monkeypatch.setattr(P.parallelizer_utils, "fully_shard_by_dtype", shared_sharder_mock)

    block = DummyBlock(mlp=DummyMoE())
    model = DummyModel([block])
    model._keep_in_fp32_modules_strict = ["mlp.gate.weight", "mlp.gate.e_score_correction_bias"]
    fsdp_mesh = object()
    mp_policy = MagicMock()
    offload_policy = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=True,
        ep_shard_enabled=False,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        reshard_after_forward=True,
    )

    shared_sharder_mock.assert_called_once_with(
        block,
        mesh=fsdp_mesh,
        mp_policy=mp_policy,
        offload_policy=offload_policy,
        fp32_compute_module_names=(
            "mlp.gate.weight",
            "mlp.gate.e_score_correction_bias",
        ),
        reshard_after_forward=True,
        ignored_params=set(block.mlp.experts.parameters()),
        fully_shard_fn=fully_shard_mock,
    )


def test_apply_fsdp_skips_separate_wrapping_for_tied_embeddings(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    block = DummyBlock(mlp=DummyMoE())
    shared_weight = object()
    embed = types.SimpleNamespace(weight=shared_weight)
    lm_head = types.SimpleNamespace(weight=shared_weight)
    inner_model = DummyModel([block], embed_tokens=embed)
    outer_model = types.SimpleNamespace(model=inner_model, lm_head=lm_head)
    fsdp_mesh = object()

    P.apply_fsdp(
        model=outer_model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=True,
        ep_shard_enabled=False,
        ep_shard_mesh=None,
        wrap_outer_model=True,
    )

    assert _find_call_by_first_arg(fully_shard_mock, embed) is None
    assert _find_call_by_first_arg(fully_shard_mock, lm_head) is None
    assert _find_call_by_first_arg(fully_shard_mock, inner_model) is None

    outer_call = _find_call_by_first_arg(fully_shard_mock, outer_model)
    assert outer_call is not None and outer_call[1]["mesh"] is fsdp_mesh


def test_apply_fsdp_rejects_cross_root_tied_embeddings_without_outer_wrap(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    monkeypatch.setattr(P, "fully_shard", MagicMock())
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    shared_weight = object()
    inner_model = DummyModel(
        [DummyBlock(mlp=DummyMoE())],
        embed_tokens=types.SimpleNamespace(weight=shared_weight),
    )
    outer_model = types.SimpleNamespace(
        model=inner_model,
        lm_head=types.SimpleNamespace(weight=shared_weight),
    )

    with pytest.raises(ValueError, match="wrap_outer_model=False"):
        P.apply_fsdp(
            model=outer_model,
            fsdp_mesh=object(),
            ep_enabled=True,
            ep_shard_enabled=False,
            wrap_outer_model=False,
        )


def test_apply_fsdp_without_ep_enabled_has_no_ignored_params(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    block = DummyBlock(mlp=DummyMoE())
    model = DummyModel([block])
    fsdp_mesh = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=False,
        ep_shard_enabled=False,
        ep_shard_mesh=None,
    )

    block_call = _find_call_by_first_arg(fully_shard_mock, block)
    assert block_call is not None
    _, block_kwargs = block_call
    assert block_kwargs["mesh"] is fsdp_mesh
    assert block_kwargs.get("ignored_params") is None


def test_apply_fsdp_excludes_model_owned_shard_from_block_and_root(monkeypatch):
    """A model-owned parameter shard must not be sharded again by FSDP."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    owner_weight = DummyParam()
    owner_weight._nemo_model_owned_grad_divisor = 1.0

    class OwnerShardedBlock(DummyBlock):
        def parameters(self):
            yield owner_weight

    class OwnerShardedModel(DummyModel):
        def parameters(self):
            yield owner_weight

    block = OwnerShardedBlock(mlp=DummyMoE())
    model = OwnerShardedModel([block])
    fsdp_mesh = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=False,
        ep_shard_enabled=False,
    )

    block_call = _find_call_by_first_arg(fully_shard_mock, block)
    assert block_call is not None
    assert block_call[1]["ignored_params"] == {owner_weight}
    root_call = _find_call_by_first_arg(fully_shard_mock, model)
    assert root_call is not None
    assert root_call[1]["ignored_params"] == {owner_weight}


def test_apply_fsdp_prepares_and_excludes_final_model_owned_dtensor_identity(monkeypatch):
    """The model hook runs before every FSDP unit snapshots ignored params."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    # ``_import_parallelizer_with_stubs`` gives the module its own minimal
    # ``torch.nn.Parameter`` class.  Use that exact runtime class so this test
    # exercises the production type guard instead of mixing the real and
    # stubbed torch modules.
    distributed_weight = P.nn.Parameter()
    distributed_weight._nemo_model_owned_grad_divisor = 2.0

    class ModelOwnedDTensorBlock(DummyBlock):
        def parameters(self):
            yield distributed_weight

    class ModelOwnedDTensorModel(DummyModel):
        def parameters(self):
            yield distributed_weight

        def _nemo_prepare_model_owned_dtensors(self, mesh):
            assert mesh is fsdp_mesh
            return {distributed_weight}

    block = ModelOwnedDTensorBlock(mlp=DummyMoE())
    model = ModelOwnedDTensorModel([block])
    fsdp_mesh = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=False,
        ep_shard_enabled=False,
    )

    block_call = _find_call_by_first_arg(fully_shard_mock, block)
    assert block_call is not None
    assert block_call[1]["ignored_params"] == {distributed_weight}
    root_call = _find_call_by_first_arg(fully_shard_mock, model)
    assert root_call is not None
    assert root_call[1]["ignored_params"] == {distributed_weight}


@pytest.mark.parametrize(
    "audio_trainable, visual_trainable",
    [
        (True, True),
        (False, False),
    ],
)
def test_apply_fsdp_handles_multimodal_components(monkeypatch, audio_trainable, visual_trainable):
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    logging_mock = MagicMock()
    monkeypatch.setattr(P.logger, "info", logging_mock)

    class Tower:
        def __init__(self, requires_grad):
            self._params = [DummyParam(requires_grad=requires_grad)]

        def parameters(self):
            return iter(self._params)

        def named_children(self):
            return []

    audio_tower = Tower(audio_trainable)
    visual_tower = Tower(visual_trainable)

    model = DummyModel([DummyBlock()], audio_tower=audio_tower, visual=visual_tower)

    P.apply_fsdp(
        model=model,
        fsdp_mesh=object(),
        ep_enabled=False,
        ep_shard_enabled=False,
        ep_shard_mesh=None,
    )

    assert (_find_call_by_first_arg(fully_shard_mock, audio_tower) is not None) == audio_trainable
    assert (_find_call_by_first_arg(fully_shard_mock, visual_tower) is not None) == visual_trainable
    if not audio_trainable:
        logging_mock.assert_any_call(
            "Keeping frozen multimodal module %s at FSDP policy %s",
            "audio_tower",
            "root",
        )
    if not visual_trainable:
        logging_mock.assert_any_call(
            "Keeping frozen multimodal module %s at FSDP policy %s",
            "visual",
            "root",
        )


@pytest.mark.parametrize(
    "frozen_multimodal_sharding, expected_tower_sharded, expected_ignored_params",
    [
        ("root", False, None),
        ("per_layer", True, None),
        ("replicate", False, "frozen"),
    ],
)
def test_apply_fsdp_applies_nested_frozen_multimodal_policy(
    monkeypatch, frozen_multimodal_sharding, expected_tower_sharded, expected_ignored_params
):
    """Gemma4-style nested towers honor root, per-layer, and replicate policies."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    def vlm_get_text_module(module):
        return module.language_model if hasattr(module, "language_model") else module

    monkeypatch.setattr(P, "get_text_module", vlm_get_text_module)

    class TreeModule:
        def __init__(self, params=None, **children):
            self._params = params or []
            self._children = children
            for name, child in children.items():
                setattr(self, name, child)

        def parameters(self):
            for param in self._params:
                yield param
            for child in self._children.values():
                parameters = getattr(child, "parameters", None)
                if callable(parameters):
                    yield from parameters()

        def named_children(self):
            return list(self._children.items())

        def named_modules(self):
            yield "", self
            for child_name, child in self._children.items():
                yield child_name, child
                child_named_modules = getattr(child, "named_modules", None)
                if not callable(child_named_modules):
                    continue
                for sub_name, submodule in child_named_modules():
                    if sub_name:
                        yield f"{child_name}.{sub_name}", submodule

    block = DummyBlock(mlp=DummyMoE())
    shared_weight = DummyParam(requires_grad=True)
    vision_param = DummyParam(requires_grad=False)
    embed_vision_param = DummyParam(requires_grad=False)

    embed_tokens = types.SimpleNamespace(weight=shared_weight)
    lm_head = types.SimpleNamespace(weight=shared_weight)
    language_model = DummyModel([block], embed_tokens=embed_tokens)
    language_model.parameters = lambda: iter([shared_weight])
    language_model.named_modules = lambda: iter([("", language_model)])

    vision_tower = TreeModule(params=[vision_param])
    embed_vision = TreeModule(params=[embed_vision_param])
    inner_model = TreeModule(language_model=language_model, vision_tower=vision_tower, embed_vision=embed_vision)
    outer_model = TreeModule(model=inner_model)
    outer_model.lm_head = lm_head

    P.apply_fsdp(
        model=outer_model,
        fsdp_mesh=object(),
        ep_enabled=True,
        ep_shard_enabled=False,
        ep_shard_mesh=None,
        wrap_outer_model=True,
        frozen_multimodal_sharding=frozen_multimodal_sharding,
    )

    assert (_find_call_by_first_arg(fully_shard_mock, vision_tower) is not None) is expected_tower_sharded
    assert (_find_call_by_first_arg(fully_shard_mock, embed_vision) is not None) is expected_tower_sharded
    assert _find_call_by_first_arg(fully_shard_mock, language_model) is None

    outer_call = _find_call_by_first_arg(fully_shard_mock, outer_model)
    assert outer_call is not None
    ignored_params = outer_call[1].get("ignored_params")
    if expected_ignored_params is None:
        assert ignored_params is None
    else:
        assert ignored_params == {vision_param, embed_vision_param}


def test_apply_fsdp_rejects_root_policy_without_an_owner_for_frozen_multimodal_params(monkeypatch):
    """A nested frozen tower cannot use root policy when the outer root is disabled."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    monkeypatch.setattr(P, "fully_shard", MagicMock())
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    class Tower:
        def __init__(self):
            self._params = [DummyParam(requires_grad=False)]

        def parameters(self):
            return iter(self._params)

    class OuterModel:
        def __init__(self):
            self.model = DummyModel([DummyBlock()])
            self.vision_tower = Tower()

        def named_modules(self):
            yield "", self
            yield "vision_tower", self.vision_tower

    with pytest.raises(ValueError, match="requires wrap_outer_model=True.*vision_tower"):
        P.apply_fsdp(
            model=OuterModel(),
            fsdp_mesh=object(),
            ep_enabled=False,
            ep_shard_enabled=False,
            wrap_outer_model=False,
            frozen_multimodal_sharding="root",
        )


@pytest.mark.parametrize(
    "frozen_multimodal_sharding, tower_trainable, expected_tower_sharded",
    [
        ("per_layer", False, True),
        ("replicate", False, False),
        ("root", True, True),
    ],
)
def test_apply_fsdp_without_outer_root_allows_supported_multimodal_policies(
    monkeypatch,
    frozen_multimodal_sharding,
    tower_trainable,
    expected_tower_sharded,
):
    """Per-layer, replicate, and trainable towers do not require an outer root."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    class Tower:
        def __init__(self):
            self._params = [
                DummyParam(requires_grad=False),
                DummyParam(requires_grad=tower_trainable),
            ]

        def parameters(self):
            return iter(self._params)

        def named_children(self):
            return []

    class OuterModel:
        def __init__(self):
            self.model = DummyModel([DummyBlock()])
            self.vision_tower = Tower()

        def named_modules(self):
            yield "", self
            yield "vision_tower", self.vision_tower

    model = OuterModel()
    P.apply_fsdp(
        model=model,
        fsdp_mesh=object(),
        ep_enabled=False,
        ep_shard_enabled=False,
        wrap_outer_model=False,
        frozen_multimodal_sharding=frozen_multimodal_sharding,
    )

    assert (_find_call_by_first_arg(fully_shard_mock, model.vision_tower) is not None) is expected_tower_sharded
    assert _find_call_by_first_arg(fully_shard_mock, model.model) is not None
    assert _find_call_by_first_arg(fully_shard_mock, model) is None


class MeshView:
    def __init__(self, size):
        self._size = size

    def size(self):
        return self._size


class FakeWorldMesh:
    def __init__(self, sizes_by_key, mesh_dim_names):
        self._sizes = sizes_by_key
        self.mesh_dim_names = tuple(mesh_dim_names)
        self._flatten_mapping = {}

    def _get_root_mesh(self):
        return self

    def __getitem__(self, key):
        # Support both string "dp" and tuple ("dp",) lookups
        if key in self._sizes:
            return MeshView(self._sizes[key])
        if isinstance(key, str) and (key,) in self._sizes:
            return MeshView(self._sizes[(key,)])
        raise KeyError(key)


class FakeMoeMesh:
    def __init__(self, sizes_by_key):
        self._sizes = sizes_by_key

    def __getitem__(self, key):
        return MeshView(self._sizes[key])


def test_parallelize_model_uses_root_preserving_hsdp_mesh(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    mesh_utils = sys.modules["nemo_automodel.components.distributed.mesh_utils"]
    hsdp_mesh = MeshView(16)
    get_fsdp_dp_mesh_mock = MagicMock(return_value=hsdp_mesh)
    monkeypatch.setattr(mesh_utils, "get_fsdp_dp_mesh", get_fsdp_dp_mesh_mock)
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)

    world_mesh = FakeWorldMesh(
        {("dp_replicate", "dp_shard_cp"): 16, "tp": 1},
        mesh_dim_names=["dp_replicate", "dp_shard", "cp", "tp"],
    )
    model = type("Outer", (), {"moe_config": type("MoeConfig", (), {"n_routed_experts": 128})()})()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=None,
        dp_axis_names=("dp_replicate", "dp_shard_cp"),
        activation_checkpointing=False,
    )

    get_fsdp_dp_mesh_mock.assert_called_once_with(world_mesh, "dp_replicate", "dp_shard_cp")
    assert apply_fsdp_mock.call_args.args[1] is hsdp_mesh


def test_parallelize_model_calls_subsystems_and_validates(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_ep_mock = MagicMock()
    apply_ac_mock = MagicMock()
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_ep", apply_ep_mock)
    monkeypatch.setattr(P, "apply_ac", apply_ac_mock)
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)

    world_mesh = FakeWorldMesh({"dp": 2, ("dp",): 2, "tp": 1, "cp": 1}, mesh_dim_names=["dp", "tp", "cp"])
    moe_mesh = FakeMoeMesh({"ep": 2, ("es1", "es2"): 2})

    # model.model.moe_config.n_routed_experts must be divisible by ep size (2)
    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 4})()

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=moe_mesh,
        dp_axis_names=("dp",),
        cp_axis_name=None,
        tp_axis_name=None,
        ep_axis_name="ep",
        ep_shard_axis_names=("es1", "es2"),
        activation_checkpointing=True,
    )
    apply_ep_mock.assert_called_once()
    # AC enabled
    apply_ac_mock.assert_called_once_with(
        model, ignore_router=True, selective=False, activation_checkpointing_scope="all"
    )
    # FSDP called with combined flags and derived meshes
    args, kwargs = apply_fsdp_mock.call_args
    # handle positional or keyword invocations
    fsdp_model = kwargs.get("model", args[0] if args else None)
    fsdp_mesh_arg = kwargs.get("fsdp_mesh", args[1] if len(args) > 1 else None)
    ep_enabled = kwargs.get("ep_enabled", args[2] if len(args) > 2 else None)
    ep_shard_enabled = kwargs.get("ep_shard_enabled", args[3] if len(args) > 3 else None)
    ep_shard_mesh_arg = kwargs.get("ep_shard_mesh", args[4] if len(args) > 4 else None)

    assert fsdp_model is model
    assert fsdp_mesh_arg.size() == 2
    assert ep_enabled is True
    assert ep_shard_enabled is True
    assert ep_shard_mesh_arg.size() == 2
    assert kwargs.get("frozen_multimodal_sharding") == "root"


def test_parallelize_model_passes_frozen_multimodal_sharding_to_apply_fsdp(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_ep", MagicMock())
    monkeypatch.setattr(P, "apply_ac", MagicMock())
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)

    world_mesh = FakeWorldMesh({"dp": 2, ("dp",): 2}, mesh_dim_names=["dp"])
    model = type("Outer", (), {"moe_config": type("MC", (), {"n_routed_experts": 4})()})()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=None,
        dp_axis_names=("dp",),
        frozen_multimodal_sharding="replicate",
    )

    apply_fsdp_mock.assert_called_once()
    _, kwargs = apply_fsdp_mock.call_args
    assert kwargs["frozen_multimodal_sharding"] == "replicate"


def test_parallelize_model_accepts_top_level_moe_config(monkeypatch):
    """Custom MoE models may expose moe_config on the outer model itself."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_ep_mock = MagicMock()
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_ep", apply_ep_mock)
    monkeypatch.setattr(P, "apply_ac", MagicMock())
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)

    world_mesh = FakeWorldMesh({"dp": 1, ("dp",): 1}, mesh_dim_names=["dp"])
    moe_mesh = FakeMoeMesh({"ep": 2})
    model = type("Outer", (), {"moe_config": type("MC", (), {"n_routed_experts": 4})()})()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=moe_mesh,
        dp_axis_names=("dp",),
        cp_axis_name=None,
        tp_axis_name=None,
        ep_axis_name="ep",
        ep_shard_axis_names=None,
        activation_checkpointing=False,
    )

    apply_ep_mock.assert_called_once()
    apply_fsdp_mock.assert_not_called()


def test_parallelize_model_rejects_missing_safe_tp_plan_and_invalid_ep_divisibility(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    world_mesh_bad_tp = FakeWorldMesh({"tp": 2, "cp": 1}, mesh_dim_names=["tp", "cp"])
    moe_mesh = FakeMoeMesh({"ep": 2})

    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 3})()  # not divisible by 2

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    # TP requires a registered or explicit plan that passes MoE ownership validation.
    monkeypatch.setattr(P, "_resolve_moe_tp_plan", MagicMock(side_effect=ValueError("No safe TP plan")))
    with pytest.raises(ValueError, match="No safe TP plan"):
        P.parallelize_model(
            model=model,
            world_mesh=world_mesh_bad_tp,
            moe_mesh=moe_mesh,
            dp_axis_names=None,
            cp_axis_name=None,
            tp_axis_name="tp",
            ep_axis_name=None,
            ep_shard_axis_names=None,
            activation_checkpointing=False,
        )

    # EP enabled but divisibility violated -> assertion
    world_mesh_ok = FakeWorldMesh({("dp",): 1, "tp": 1, "cp": 1}, mesh_dim_names=["dp", "tp", "cp"])
    moe_mesh_ep = FakeMoeMesh({"ep": 2})
    with pytest.raises(AssertionError):
        P.parallelize_model(
            model=model,
            world_mesh=world_mesh_ok,
            moe_mesh=moe_mesh_ep,
            dp_axis_names=("dp",),
            cp_axis_name=None,
            tp_axis_name=None,
            ep_axis_name="ep",
            ep_shard_axis_names=None,
            activation_checkpointing=False,
        )


def test_validate_moe_tp_plan_allows_shared_experts_and_rejects_routed_experts(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    safe = {
        "model.language_model.layers.*.mlp.shared_experts.gate_proj": object(),
        "model.language_model.layers.*.mlp.shared_experts.down_proj": object(),
        "lm_head": object(),
    }
    assert P._validate_moe_tp_plan(safe) is safe

    for unsafe_path in (
        "model.layers.*.mlp.experts",
        "model.layers.*.mlp.experts.*.gate_proj",
        "model.layers.*.mlp.gate_and_up_projs",
        "model.layers.*.mlp.down_projs",
        "model.layers.*.mlp.gate",
        "model.layers.*.mlp.router",
        "model.layers.*.mlp.shared_expert_gate",
        "model.layers.*.mlp",
    ):
        with pytest.raises(ValueError, match="EP-owned"):
            P._validate_moe_tp_plan({unsafe_path: object()})


def test_validate_moe_tp_plan_expands_wildcards_and_rejects_zero_matches(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)

    class ConcreteModel:
        def named_modules(self):
            for name in (
                "",
                "model.layers.0.mlp",
                "model.layers.0.mlp.shared_experts",
                "model.layers.0.mlp.shared_experts.gate_proj",
                "model.layers.0.mlp.shared_expert_gate",
                "lm_head",
            ):
                yield name, object()

    model = ConcreteModel()
    safe = {"model.layers.*.mlp.shared_experts.gate_proj": object(), "lm_head": object()}
    assert P._validate_moe_tp_plan(safe, model=model) is safe

    with pytest.raises(ValueError, match="EP-owned"):
        P._validate_moe_tp_plan({"model.layers.*.mlp.shared_expert_*": object()}, model=model)
    with pytest.raises(ValueError, match="must each match"):
        P._validate_moe_tp_plan({"model.layers.*.does_not_exist": object()}, model=model)


def test_resolve_moe_tp_plan_rejects_sequence_parallel_fail_closed(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    with pytest.raises(ValueError, match="sequence_parallel=True"):
        P._resolve_moe_tp_plan(
            object(),
            sequence_parallel=True,
            tp_shard_plan={"lm_head": object()},
            tp_size=2,
        )


def test_resolve_moe_tp_plan_uses_registered_factory_without_dense_fallback(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    optimized_stub = types.ModuleType("nemo_automodel.components.distributed.optimized_tp_plans")
    factory = MagicMock(return_value={"lm_head": object()})
    optimized_stub.PARALLELIZE_FUNCTIONS = {"registered.Model": factory}
    optimized_stub._get_class_qualname = lambda cls: "registered.Model"
    monkeypatch.setitem(
        sys.modules,
        "nemo_automodel.components.distributed.optimized_tp_plans",
        optimized_stub,
    )

    model = type("RegisteredMoe", (), {})()
    plan = P._resolve_moe_tp_plan(
        model,
        sequence_parallel=False,
        tp_shard_plan=None,
        tp_size=2,
    )

    assert set(plan) == {"lm_head"}
    factory.assert_called_once_with(model, False)


def test_resolve_moe_tp_plan_propagates_registered_factory_failure(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    optimized_stub = types.ModuleType("nemo_automodel.components.distributed.optimized_tp_plans")

    def broken_factory(model, sequence_parallel):
        raise RuntimeError("architecture-specific plan failed")

    optimized_stub.PARALLELIZE_FUNCTIONS = {"BrokenMoe": broken_factory}
    optimized_stub._get_class_qualname = lambda cls: "not.registered"
    monkeypatch.setitem(
        sys.modules,
        "nemo_automodel.components.distributed.optimized_tp_plans",
        optimized_stub,
    )

    with pytest.raises(ValueError, match="architecture-specific plan failed"):
        P._resolve_moe_tp_plan(
            type("BrokenMoe", (), {})(),
            sequence_parallel=False,
            tp_shard_plan=None,
            tp_size=2,
        )


def test_parallelize_model_applies_tp_before_cp_ep_ac_and_fsdp(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    calls = []
    safe_plan = {"model.layers.*.mlp.shared_experts.up_proj": object()}
    monkeypatch.setattr(P, "_resolve_moe_tp_plan", MagicMock(return_value=safe_plan))
    monkeypatch.setattr(P, "parallelize_module", MagicMock(side_effect=lambda *args: calls.append("tp")))
    monkeypatch.setattr(P, "apply_cp", MagicMock(side_effect=lambda *args: calls.append("cp")))
    monkeypatch.setattr(P, "apply_ep", MagicMock(side_effect=lambda *args, **kwargs: calls.append("ep")))
    monkeypatch.setattr(P, "apply_ac", MagicMock(side_effect=lambda *args, **kwargs: calls.append("ac")))
    monkeypatch.setattr(P, "apply_fsdp", MagicMock(side_effect=lambda *args, **kwargs: calls.append("fsdp")))
    monkeypatch.setattr(P, "ensure_tied_lm_head", MagicMock(side_effect=lambda model: calls.append("tie")))

    world_mesh = FakeWorldMesh(
        {"tp": 2, "cp": 2, ("dp",): 2},
        mesh_dim_names=["dp", "cp", "tp"],
    )
    moe_mesh = FakeMoeMesh({"ep": 2})
    model = type(
        "Outer",
        (),
        {
            "moe_config": type("MC", (), {"n_routed_experts": 4})(),
            "supports": types.SimpleNamespace(
                mtp_enabled=False,
                supports_mtp_cp=False,
                supports_mtp_cp_pp=False,
            ),
        },
    )()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=moe_mesh,
        dp_axis_names=("dp",),
        cp_axis_name="cp",
        tp_axis_name="tp",
        ep_axis_name="ep",
        activation_checkpointing=True,
        reapply_trainability=lambda _model: calls.append("trainability"),
    )

    assert calls == ["tp", "tie", "cp", "ep", "ac", "trainability", "fsdp"]
    assert model._nemo_moe_tp_requires_replica_sync is True
    assert model._nemo_moe_tp_requires_pretrained_weights is True
    P._resolve_moe_tp_plan.assert_called_once_with(
        model,
        sequence_parallel=False,
        tp_shard_plan=None,
        tp_size=2,
    )


def test_parallelize_model_rejects_cp_mtp_pipeline_stage_before_ep_or_cp(monkeypatch):
    """The MoE/EP path must reject the same unsupported topology on every PP stage."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_cp_mock = MagicMock()
    apply_ep_mock = MagicMock()
    monkeypatch.setattr(P, "apply_cp", apply_cp_mock)
    monkeypatch.setattr(P, "apply_ep", apply_ep_mock)

    world_mesh = FakeWorldMesh(
        {"cp": 2, ("dp",): 2},
        mesh_dim_names=["dp", "cp"],
    )
    moe_mesh = FakeMoeMesh({"ep": 2})
    model = type(
        "PipelineStage",
        (),
        {
            "supports": types.SimpleNamespace(mtp_enabled=True, supports_mtp_cp_pp=False),
            "mtp_config": type("MTP", (), {"enabled": True})(),
            "moe_config": type("MC", (), {"n_routed_experts": 4})(),
            "_is_pipeline_parallel_stage": lambda self: True,
        },
    )()

    with pytest.raises(NotImplementedError, match="MTP with context and pipeline parallelism"):
        P.parallelize_model(
            model=model,
            world_mesh=world_mesh,
            moe_mesh=moe_mesh,
            dp_axis_names=("dp",),
            cp_axis_name="cp",
            ep_axis_name="ep",
        )

    apply_cp_mock.assert_not_called()
    apply_ep_mock.assert_not_called()


def test_parallelize_model_rejects_cp_mtp_without_capability_before_ep_or_cp(monkeypatch):
    """The MoE/EP path must enforce the same MTP+CP capability as the dense path."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_cp_mock = MagicMock()
    apply_ep_mock = MagicMock()
    monkeypatch.setattr(P, "apply_cp", apply_cp_mock)
    monkeypatch.setattr(P, "apply_ep", apply_ep_mock)

    world_mesh = FakeWorldMesh({"cp": 2, ("dp",): 2}, mesh_dim_names=["dp", "cp"])
    moe_mesh = FakeMoeMesh({"ep": 2})
    model = type(
        "UnsupportedMTPModel",
        (),
        {
            "supports": types.SimpleNamespace(
                mtp_enabled=True,
                supports_mtp_cp=False,
                supports_mtp_cp_pp=False,
            ),
            "mtp_config": type("MTP", (), {"enabled": True})(),
            "moe_config": type("MC", (), {"n_routed_experts": 4})(),
        },
    )()

    with pytest.raises(RuntimeError, match="does not support MTP with context parallelism"):
        P.parallelize_model(
            model=model,
            world_mesh=world_mesh,
            moe_mesh=moe_mesh,
            dp_axis_names=("dp",),
            cp_axis_name="cp",
            ep_axis_name="ep",
        )

    apply_cp_mock.assert_not_called()
    apply_ep_mock.assert_not_called()


def test_parallelize_model_forwards_offload_policy_to_fsdp(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)

    world_mesh = FakeWorldMesh({("dp",): 2}, mesh_dim_names=["dp"])
    sentinel = object()
    P.parallelize_model(
        model=object(),
        world_mesh=world_mesh,
        moe_mesh=None,
        dp_axis_names=("dp",),
        offload_policy=sentinel,
    )

    assert apply_fsdp_mock.call_args.kwargs["offload_policy"] is sentinel


def test_parallelize_model_rejects_async_tp_for_custom_moe(monkeypatch):
    P = _import_parallelizer_with_stubs(monkeypatch)
    world_mesh = FakeWorldMesh({"tp": 2}, mesh_dim_names=["tp"])

    with pytest.raises(ValueError, match="enable_async_tensor_parallel=True"):
        P.parallelize_model(
            model=object(),
            world_mesh=world_mesh,
            moe_mesh=None,
            dp_axis_names=(),
            tp_axis_name="tp",
            enable_async_tensor_parallel=True,
        )


def test_apply_fsdp_with_lm_head_precision_fp32(monkeypatch):
    """Test that apply_fsdp applies custom MixedPrecisionPolicy to lm_head when lm_head_precision is fp32."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    mp_policy_mock = MagicMock(return_value="MP_POLICY")
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", mp_policy_mock)

    torch_stub = sys.modules["torch"]
    block = DummyBlock(mlp=DummyMoE())
    lm = object()
    model = DummyModel([block], lm_head=lm)
    fsdp_mesh = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=False,
        ep_shard_enabled=False,
        lm_head_precision=torch_stub.float32,
    )

    # Find the lm_head call
    lm_call = _find_call_by_first_arg(fully_shard_mock, lm)
    assert lm_call is not None
    _, lm_kwargs = lm_call

    # Verify custom MixedPrecisionPolicy was created with fp32 for all dtypes
    assert mp_policy_mock.call_count >= 2  # default + lm_head
    # Find the call for lm_head's custom policy
    fp32_policy_calls = [
        call
        for call in mp_policy_mock.call_args_list
        if call[1].get("param_dtype") == torch_stub.float32
        and call[1].get("reduce_dtype") == torch_stub.float32
        and call[1].get("output_dtype") == torch_stub.float32
    ]
    assert len(fp32_policy_calls) == 1


def test_apply_fsdp_without_lm_head_precision_uses_default_policy(monkeypatch):
    """Test that apply_fsdp uses default MixedPrecisionPolicy for lm_head when lm_head_precision is None."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    mp_policy_mock = MagicMock(return_value="MP_POLICY")
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", mp_policy_mock)

    block = DummyBlock(mlp=DummyMoE())
    lm = object()
    model = DummyModel([block], lm_head=lm)
    fsdp_mesh = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=False,
        ep_shard_enabled=False,
        lm_head_precision=None,
    )

    # Find the lm_head call
    lm_call = _find_call_by_first_arg(fully_shard_mock, lm)
    assert lm_call is not None

    # Should only have one MixedPrecisionPolicy call (the default one)
    assert mp_policy_mock.call_count == 1


@pytest.mark.parametrize("model_owned_sharding", [False, True])
def test_apply_fsdp_uses_model_wrapper_or_default(monkeypatch: pytest.MonkeyPatch, model_owned_sharding: bool) -> None:
    """Language and multimodal units share one model-selected FSDP callback."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    default_shard = MagicMock()
    model_shard = MagicMock()
    monkeypatch.setattr(P, "fully_shard", default_shard)
    block = DummyBlock(mlp=DummyMoE())
    model = DummyModel([block])
    model.visual = DummyExperts()
    if model_owned_sharding:
        model._nemo_fully_shard = model_shard

    P.apply_fsdp(
        model=model,
        fsdp_mesh=object(),
        ep_enabled=False,
        ep_shard_enabled=False,
        lm_head_precision=None,
    )

    selected = model_shard if model_owned_sharding else default_shard
    other = default_shard if model_owned_sharding else model_shard
    for module in (block, model.visual, model):
        assert _find_call_by_first_arg(selected, module) is not None
        assert _find_call_by_first_arg(other, module) is None


def test_parallelize_model_passes_lm_head_precision_to_apply_fsdp(monkeypatch):
    """Test that parallelize_model passes lm_head_precision to apply_fsdp."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)
    monkeypatch.setattr(P, "apply_ep", MagicMock())
    monkeypatch.setattr(P, "apply_ac", MagicMock())

    world_mesh = FakeWorldMesh({("dp",): 2}, mesh_dim_names=["dp"])
    moe_mesh = None

    torch_stub = sys.modules["torch"]

    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 4})()

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=moe_mesh,
        dp_axis_names=("dp",),
        lm_head_precision=torch_stub.float32,
    )

    # Verify apply_fsdp was called with lm_head_precision
    apply_fsdp_mock.assert_called_once()
    _, kwargs = apply_fsdp_mock.call_args
    assert kwargs.get("lm_head_precision") == torch_stub.float32


def test_apply_fsdp_with_lm_head_precision_string_input(monkeypatch):
    """Test that apply_fsdp accepts string input for lm_head_precision and converts to torch.dtype."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    mp_policy_mock = MagicMock(return_value="MP_POLICY")
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", mp_policy_mock)

    torch_stub = sys.modules["torch"]

    # Mock dtype_from_str to convert string to torch.float32
    def mock_dtype_from_str(val, default=None):
        if val == "float32" or val == "torch.float32":
            return torch_stub.float32
        return default

    monkeypatch.setattr(P, "dtype_from_str", mock_dtype_from_str)

    block = DummyBlock(mlp=DummyMoE())
    lm = object()
    model = DummyModel([block], lm_head=lm)
    fsdp_mesh = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=False,
        ep_shard_enabled=False,
        lm_head_precision="float32",
    )

    # Find the lm_head call
    lm_call = _find_call_by_first_arg(fully_shard_mock, lm)
    assert lm_call is not None

    # Verify custom MixedPrecisionPolicy was created with fp32 for all dtypes
    assert mp_policy_mock.call_count >= 2  # default + lm_head
    # Find the call for lm_head's custom policy
    fp32_policy_calls = [
        call
        for call in mp_policy_mock.call_args_list
        if call[1].get("param_dtype") == torch_stub.float32
        and call[1].get("reduce_dtype") == torch_stub.float32
        and call[1].get("output_dtype") == torch_stub.float32
    ]
    assert len(fp32_policy_calls) == 1


def test_parallelize_model_with_lm_head_precision_string_input(monkeypatch):
    """Test that parallelize_model accepts string input for lm_head_precision."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)
    monkeypatch.setattr(P, "apply_ep", MagicMock())
    monkeypatch.setattr(P, "apply_ac", MagicMock())

    world_mesh = FakeWorldMesh({("dp",): 2}, mesh_dim_names=["dp"])
    moe_mesh = None

    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 4})()

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=moe_mesh,
        dp_axis_names=("dp",),
        lm_head_precision="float32",
    )

    # Verify apply_fsdp was called with lm_head_precision as a string
    apply_fsdp_mock.assert_called_once()
    _, kwargs = apply_fsdp_mock.call_args
    assert kwargs.get("lm_head_precision") == "float32"


def test_apply_fsdp_with_wrap_outer_model_true(monkeypatch):
    """Test that apply_fsdp wraps both inner _model and outer model when wrap_outer_model=True."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    block = DummyBlock(mlp=DummyMoE())
    # Create a model with nested structure (model.model exists)
    inner_model = DummyModel([block])

    class OuterModel:
        def __init__(self, inner):
            self.model = inner

    outer_model = OuterModel(inner_model)
    fsdp_mesh = object()

    P.apply_fsdp(
        model=outer_model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=False,
        ep_shard_enabled=False,
        wrap_outer_model=True,
    )

    # Find calls for inner model and outer model
    inner_call = _find_call_by_first_arg(fully_shard_mock, inner_model)
    outer_call = _find_call_by_first_arg(fully_shard_mock, outer_model)

    # Both should be wrapped
    assert inner_call is not None, "Inner model should be wrapped"
    assert outer_call is not None, "Outer model should be wrapped when wrap_outer_model=True"


def test_apply_fsdp_with_wrap_outer_model_false(monkeypatch):
    """Test that apply_fsdp only wraps inner _model when wrap_outer_model=False (default)."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    block = DummyBlock(mlp=DummyMoE())
    # Create a model with nested structure (model.model exists)
    inner_model = DummyModel([block])

    class OuterModel:
        def __init__(self, inner):
            self.model = inner

    outer_model = OuterModel(inner_model)
    fsdp_mesh = object()

    P.apply_fsdp(
        model=outer_model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=False,
        ep_shard_enabled=False,
        wrap_outer_model=False,
    )

    # Find calls for inner model and outer model
    inner_call = _find_call_by_first_arg(fully_shard_mock, inner_model)
    outer_call = _find_call_by_first_arg(fully_shard_mock, outer_model)

    # Only inner should be wrapped
    assert inner_call is not None, "Inner model should be wrapped"
    assert outer_call is None, "Outer model should NOT be wrapped when wrap_outer_model=False"


def test_apply_fsdp_wrap_outer_model_no_nested_structure(monkeypatch):
    """Test that wrap_outer_model has no effect when model has no nested structure."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    block = DummyBlock(mlp=DummyMoE())
    # Create a model without nested structure (no model.model)
    model = DummyModel([block])
    fsdp_mesh = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=False,
        ep_shard_enabled=False,
        wrap_outer_model=True,
    )

    # Find call for model
    model_call = _find_call_by_first_arg(fully_shard_mock, model)

    # Model should be wrapped exactly once (not twice)
    assert model_call is not None, "Model should be wrapped"
    # Count how many times model was passed as first arg
    model_call_count = sum(1 for args, _ in fully_shard_mock.call_args_list if args and args[0] is model)
    assert model_call_count == 1, "Model should only be wrapped once when model == _model"


def test_parallelize_model_passes_wrap_outer_model_to_apply_fsdp(monkeypatch):
    """Test that parallelize_model passes wrap_outer_model to apply_fsdp."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)
    monkeypatch.setattr(P, "apply_ep", MagicMock())
    monkeypatch.setattr(P, "apply_ac", MagicMock())

    world_mesh = FakeWorldMesh({("dp",): 2}, mesh_dim_names=["dp"])
    moe_mesh = None

    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 4})()

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=moe_mesh,
        dp_axis_names=("dp",),
        wrap_outer_model=True,
    )

    # Verify apply_fsdp was called with wrap_outer_model=True
    apply_fsdp_mock.assert_called_once()
    _, kwargs = apply_fsdp_mock.call_args
    assert kwargs.get("wrap_outer_model") is True


def test_parallelize_model_wrap_outer_model_defaults_to_true(monkeypatch):
    """Test that parallelize_model defaults wrap_outer_model to True."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)
    monkeypatch.setattr(P, "apply_ep", MagicMock())
    monkeypatch.setattr(P, "apply_ac", MagicMock())

    world_mesh = FakeWorldMesh({("dp",): 2}, mesh_dim_names=["dp"])
    moe_mesh = None

    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 4})()

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=moe_mesh,
        dp_axis_names=("dp",),
    )

    # Verify apply_fsdp was called with wrap_outer_model=False (default)
    apply_fsdp_mock.assert_called_once()
    _, kwargs = apply_fsdp_mock.call_args
    assert kwargs.get("wrap_outer_model") is True


def test_apply_ac_derives_hidden_size_and_num_experts_from_config(monkeypatch):
    """Test that apply_ac derives hidden_size and num_experts from model.config."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_hidden_size = None
    captured_num_experts = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_hidden_size, captured_num_experts
        # Extract hidden_size and num_experts by testing the policy
        torch_stub = sys.modules["torch"]
        # Test with various shapes to determine what was captured
        for hs in [128, 256, 512, 1024]:
            for ne in [8, 16, 32, 64]:
                rhs = type("Mat", (), {"shape": (hs, ne)})()
                result = policy_cb(None, torch_stub.ops.aten.mm.default, object(), rhs)
                if result == P.CheckpointPolicy.MUST_SAVE:
                    captured_hidden_size = hs
                    captured_num_experts = ne
                    break
            if captured_hidden_size is not None:
                break
        return SELECTIVE_CTX

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        if context_fn is not None:
            context_fn()  # Trigger the context function to capture values
        return block

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    # Create model with config containing hidden_size and num_experts
    class Config:
        hidden_size = 256
        num_experts = 16

    class ModelWithConfig:
        def __init__(self):
            self.config = Config()
            self.layers = LayerContainer([DummyBlock()])

    model = ModelWithConfig()

    P.apply_ac(model, ignore_router=True)

    assert captured_hidden_size == 256
    assert captured_num_experts == 16


def test_apply_ac_raises_when_hidden_size_not_available(monkeypatch):
    """Test that apply_ac raises ValueError when hidden_size is not in config and not provided."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    # Model without config
    class ModelWithoutConfig:
        def __init__(self):
            self.layers = LayerContainer([DummyBlock()])

    model = ModelWithoutConfig()

    with pytest.raises(ValueError, match="hidden_size must be provided"):
        P.apply_ac(model)


def test_apply_ac_raises_when_num_experts_not_available(monkeypatch):
    """Test that apply_ac raises ValueError when num_experts is not in config and not provided."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    # Model with config containing only hidden_size
    class ConfigPartial:
        hidden_size = 256

    class ModelPartialConfig:
        def __init__(self):
            self.config = ConfigPartial()
            self.layers = LayerContainer([DummyBlock()])

    model = ModelPartialConfig()

    with pytest.raises(ValueError, match="num_experts must be provided"):
        P.apply_ac(model)


def test_apply_ac_derives_num_experts_from_num_local_experts(monkeypatch):
    """Test that apply_ac derives num_experts from config.num_local_experts (Mixtral/GPT-OSS style)."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_num_experts = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_num_experts
        torch_stub = sys.modules["torch"]
        for ne in [32, 64, 128]:
            rhs = type("Mat", (), {"shape": (256, ne)})()
            result = policy_cb(None, torch_stub.ops.aten.mm.default, object(), rhs)
            if result == P.CheckpointPolicy.MUST_SAVE:
                captured_num_experts = ne
        return SELECTIVE_CTX

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        if context_fn is not None:
            context_fn()
        return block

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    class ConfigWithNumLocalExperts:
        hidden_size = 256
        num_local_experts = 32

    class ModelWithNumLocalExperts:
        def __init__(self):
            self.config = ConfigWithNumLocalExperts()
            self.layers = LayerContainer([DummyBlock()])

    model = ModelWithNumLocalExperts()

    P.apply_ac(model, ignore_router=True)

    assert captured_num_experts == 32


def test_apply_ac_accepts_explicit_hidden_size_and_num_experts(monkeypatch):
    """Test that apply_ac accepts explicit hidden_size and num_experts parameters."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_hidden_size = None
    captured_num_experts = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_hidden_size, captured_num_experts
        torch_stub = sys.modules["torch"]
        # Test with the expected explicit values
        rhs_match = type("Mat", (), {"shape": (512, 32)})()
        result = policy_cb(None, torch_stub.ops.aten.mm.default, object(), rhs_match)
        if result == P.CheckpointPolicy.MUST_SAVE:
            captured_hidden_size = 512
            captured_num_experts = 32
        return SELECTIVE_CTX

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        if context_fn is not None:
            context_fn()
        return block

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    # Model without config - should work with explicit params
    class ModelWithoutConfig:
        def __init__(self):
            self.layers = LayerContainer([DummyBlock()])

    model = ModelWithoutConfig()

    P.apply_ac(model, ignore_router=True, hidden_size=512, num_experts=32)

    assert captured_hidden_size == 512
    assert captured_num_experts == 32


def test_apply_ac_explicit_params_override_config(monkeypatch):
    """Test that explicit hidden_size and num_experts override model.config values."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_hidden_size = None
    captured_num_experts = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_hidden_size, captured_num_experts
        torch_stub = sys.modules["torch"]
        # Test with explicit override values
        rhs_match = type("Mat", (), {"shape": (1024, 64)})()
        result = policy_cb(None, torch_stub.ops.aten.mm.default, object(), rhs_match)
        if result == P.CheckpointPolicy.MUST_SAVE:
            captured_hidden_size = 1024
            captured_num_experts = 64
        return SELECTIVE_CTX

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        if context_fn is not None:
            context_fn()
        return block

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    # Model with config
    class Config:
        hidden_size = 256
        num_experts = 16

    class ModelWithConfig:
        def __init__(self):
            self.config = Config()
            self.layers = LayerContainer([DummyBlock()])

    model = ModelWithConfig()

    # Explicit params should override config
    P.apply_ac(model, ignore_router=True, hidden_size=1024, num_experts=64)

    assert captured_hidden_size == 1024
    assert captured_num_experts == 64


def test_apply_ac_derives_from_llm_config(monkeypatch):
    """VLM nests LM config under llm_config (not text_config) — apply_ac must fall back to it."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_hidden_size = None
    captured_num_experts = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_hidden_size, captured_num_experts
        torch_stub = sys.modules["torch"]
        for hs in [128, 256, 512, 1024]:
            for ne in [8, 16, 32, 64]:
                rhs = type("Mat", (), {"shape": (hs, ne)})()
                if policy_cb(None, torch_stub.ops.aten.mm.default, object(), rhs) == P.CheckpointPolicy.MUST_SAVE:
                    captured_hidden_size = hs
                    captured_num_experts = ne
                    break
            if captured_hidden_size is not None:
                break
        return SELECTIVE_CTX

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        if context_fn is not None:
            context_fn()
        return block

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    class LLMConfig:
        hidden_size = 512
        num_experts = 32

    # No text_config and no top-level hidden_size/num_experts — must come from llm_config.
    class Config:
        llm_config = LLMConfig()

    class ModelWithLLMConfig:
        def __init__(self):
            self.config = Config()
            self.layers = LayerContainer([DummyBlock()])

    P.apply_ac(ModelWithLLMConfig(), ignore_router=True)

    assert captured_hidden_size == 512
    assert captured_num_experts == 32


def test_apply_ac_text_config_takes_priority_over_llm_config(monkeypatch):
    """When both text_config and llm_config define hidden_size/num_experts, text_config wins."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_hidden_size = None
    captured_num_experts = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_hidden_size, captured_num_experts
        torch_stub = sys.modules["torch"]
        for hs in [128, 256, 512, 1024]:
            for ne in [8, 16, 32, 64]:
                rhs = type("Mat", (), {"shape": (hs, ne)})()
                if policy_cb(None, torch_stub.ops.aten.mm.default, object(), rhs) == P.CheckpointPolicy.MUST_SAVE:
                    captured_hidden_size = hs
                    captured_num_experts = ne
                    break
            if captured_hidden_size is not None:
                break
        return SELECTIVE_CTX

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(
        P,
        "ptd_checkpoint_wrapper",
        MagicMock(side_effect=lambda b, **kw: (kw.get("context_fn") and kw["context_fn"](), b)[1]),
    )

    class TextConfig:
        hidden_size = 256
        num_experts = 16

    class LLMConfig:
        hidden_size = 1024
        num_experts = 64

    class Config:
        text_config = TextConfig()
        llm_config = LLMConfig()

    class ModelBoth:
        def __init__(self):
            self.config = Config()
            self.layers = LayerContainer([DummyBlock()])

    P.apply_ac(ModelBoth(), ignore_router=True)

    assert captured_hidden_size == 256
    assert captured_num_experts == 16


def test_apply_ac_routes_through_get_text_module(monkeypatch):
    """For VLMs, apply_ac must wrap layers under the text sub-module (LM), not the outer wrapper."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    class LM:
        def __init__(self, blocks):
            self.layers = LayerContainer(blocks)

    class VLMInner:
        def __init__(self, lm_blocks, vit_blocks):
            self.language_model = LM(lm_blocks)
            # distractor: vision tower also has .layers — must NOT be wrapped.
            self.vision_tower = type("ViT", (), {"layers": LayerContainer(vit_blocks)})()
            # The inner module also exposes .layers (would be hit pre-fix), but
            # get_text_module redirects past it.
            self.layers = LayerContainer([DummyBlock(), DummyBlock(), DummyBlock()])

    class VLMOuter:
        def __init__(self, lm_blocks, vit_blocks):
            self.config = type("Cfg", (), {"hidden_size": 256, "num_experts": 16})()
            self.model = VLMInner(lm_blocks, vit_blocks)

    # Override get_text_module to drill into language_model (mimics the real helper).
    def vlm_get_text_module(m):
        return m.language_model if hasattr(m, "language_model") else m

    monkeypatch.setattr(P, "get_text_module", vlm_get_text_module)
    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", MagicMock(return_value=SELECTIVE_CTX))
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=lambda b, **kw: b))

    lm_blocks = [DummyBlock(), DummyBlock()]
    vit_blocks = [DummyBlock(), DummyBlock(), DummyBlock(), DummyBlock()]
    model = VLMOuter(lm_blocks, vit_blocks)

    P.apply_ac(model, ignore_router=True)

    # LM layers should be re-registered (wrapped); vision_tower and inner.layers untouched.
    assert set(model.model.language_model.layers.registered.keys()) == {"0", "1"}
    assert model.model.vision_tower.layers.registered == {}
    assert model.model.layers.registered == {}


def test_parallelize_model_passes_ignore_router_for_ac_to_apply_ac(monkeypatch):
    """Test that parallelize_model passes ignore_router_for_ac to apply_ac."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_ac_mock = MagicMock()
    monkeypatch.setattr(P, "apply_ac", apply_ac_mock)
    monkeypatch.setattr(P, "apply_ep", MagicMock())
    monkeypatch.setattr(P, "apply_fsdp", MagicMock())

    world_mesh = FakeWorldMesh({("dp",): 2}, mesh_dim_names=["dp"])
    moe_mesh = None

    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 4})()

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=moe_mesh,
        dp_axis_names=("dp",),
        activation_checkpointing=True,
        ignore_router_for_ac=True,
    )

    # Verify apply_ac was called with ignore_router=True
    apply_ac_mock.assert_called_once()
    args, kwargs = apply_ac_mock.call_args
    assert args[0] is model
    assert kwargs.get("ignore_router") is True


def test_parallelize_model_ignore_router_for_ac_defaults_to_true(monkeypatch):
    """Test that parallelize_model defaults ignore_router_for_ac to True."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_ac_mock = MagicMock()
    monkeypatch.setattr(P, "apply_ac", apply_ac_mock)
    monkeypatch.setattr(P, "apply_ep", MagicMock())
    monkeypatch.setattr(P, "apply_fsdp", MagicMock())

    world_mesh = FakeWorldMesh({("dp",): 2}, mesh_dim_names=["dp"])
    moe_mesh = None

    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 4})()

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=moe_mesh,
        dp_axis_names=("dp",),
        activation_checkpointing=True,
    )

    # Verify apply_ac was called with ignore_router=True (default)
    apply_ac_mock.assert_called_once()
    args, kwargs = apply_ac_mock.call_args
    assert kwargs.get("ignore_router") is True
    # Full (True) AC is not selective.
    assert kwargs.get("selective") is False


def test_parallelize_model_passes_selective_to_apply_ac(monkeypatch):
    """parallelize_model translates activation_checkpointing='selective' into selective=True."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_ac_mock = MagicMock()
    monkeypatch.setattr(P, "apply_ac", apply_ac_mock)
    monkeypatch.setattr(P, "apply_ep", MagicMock())
    monkeypatch.setattr(P, "apply_fsdp", MagicMock())

    world_mesh = FakeWorldMesh({("dp",): 2}, mesh_dim_names=["dp"])

    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 4})()

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=None,
        dp_axis_names=("dp",),
        activation_checkpointing="selective",
    )

    apply_ac_mock.assert_called_once()
    _, kwargs = apply_ac_mock.call_args
    assert kwargs.get("selective") is True
    assert kwargs.get("ignore_router") is True


def test_apply_ac_selective_wraps_blocks_with_shared_policy(monkeypatch):
    """selective=True wraps every block with the shared dense selective policy and
    does not require hidden_size/num_experts (router dims are only needed for the
    router-save policy)."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    sentinel_ctx = object()
    sentinel_flag = "_nemo_selective_ac"
    dense_stub = types.ModuleType("nemo_automodel.components.distributed.activation_checkpointing")
    dense_stub.make_selective_checkpoint_context_fn = MagicMock(return_value=sentinel_ctx)
    dense_stub.SELECTIVE_AC_WRAPPER_FLAG = sentinel_flag
    dense_stub.transformer_engine_attention_backend_snapshot_context_fn = lambda context_fn=None: context_fn
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed.activation_checkpointing", dense_stub)

    wrapped = []

    class _Wrapper:
        def __init__(self, block):
            self.block = block

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        assert preserve_rng_state is True
        assert callable(context_fn)
        assert context_fn() is sentinel_ctx
        w = _Wrapper(block)
        wrapped.append(w)
        return w

    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    blocks = [DummyBlock(), DummyBlock(), DummyBlock()]
    model = DummyModel(blocks)

    # No hidden_size/num_experts provided — selective path must not need them.
    P.apply_ac(model, selective=True)

    assert len(wrapped) == 3
    assert len(model.layers.registered) == 3
    dense_stub.make_selective_checkpoint_context_fn.assert_called_once()
    # Each wrapper is tagged so _apply_per_layer_compile compiles it OUTER
    # (preserving the selective policy) rather than collapsing to inner compile.
    for w in wrapped:
        assert getattr(w, sentinel_flag, False) is True


@pytest.mark.parametrize("selective", [False, True])
def test_apply_ac_repeated_mtp_checkpoints_dense_blocks_and_skips_moe(monkeypatch, selective):
    """Weight-tied MTP AC skips only blocks with the unsafe shared experts group."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    if selective:
        sentinel_flag = "_nemo_selective_ac"
        dense_stub = types.ModuleType("nemo_automodel.components.distributed.activation_checkpointing")
        dense_stub.make_selective_checkpoint_context_fn = MagicMock(return_value=object())
        dense_stub.SELECTIVE_AC_WRAPPER_FLAG = sentinel_flag
        dense_stub.transformer_engine_attention_backend_snapshot_context_fn = lambda context_fn=None: context_fn
        monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed.activation_checkpointing", dense_stub)

    wrapped = []

    class _Wrapper:
        def __init__(self, block):
            self.block = block

    def fake_wrapper(block, **_kwargs):
        wrapper = _Wrapper(block)
        wrapped.append(wrapper)
        return wrapper

    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    backbone = DummyBlock(mlp=object())
    mtp_attention = DummyBlock(mlp=object())
    mtp_moe = DummyBlock(mlp=DummyMoE())
    model = types.SimpleNamespace(
        model=DummyModel([backbone]),
        mtp=types.SimpleNamespace(
            layers=LayerContainer([mtp_attention, mtp_moe]),
            mtp_config=types.SimpleNamespace(use_repeated_layer=True),
        ),
    )

    P.apply_ac(model, selective=selective, hidden_size=8, num_experts=4)

    assert [wrapper.block for wrapper in wrapped] == [backbone, mtp_attention]
    assert set(model.model.layers.registered) == {"0"}
    assert set(model.mtp.layers.registered) == {"0"}
    if selective:
        assert all(getattr(wrapper, sentinel_flag, False) for wrapper in wrapped)


# ============================================================================
# Tests for block.moe attribute handling (Step3p5 style models)
# ============================================================================


class DummyBlockWithMoeAttr:
    """Block with separate moe attribute (Step3p5 style)."""

    def __init__(self, moe=None, mlp=None):
        self.moe = moe
        self.mlp = mlp


def test_apply_ep_handles_block_with_moe_attribute(monkeypatch):
    """Test that apply_ep correctly handles blocks with 'moe' attribute (Step3p5 style)."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    # Patch MoE symbol for isinstance
    monkeypatch.setattr(P, "MoE", DummyMoE)
    parallelize_module_mock = MagicMock()
    monkeypatch.setattr(P, "parallelize_module", parallelize_module_mock)

    moe = DummyMoE()
    # Block has moe attribute instead of mlp
    block = DummyBlockWithMoeAttr(moe=moe, mlp=None)
    model = DummyModel([block])
    ep_mesh = type("Mesh", (), {"size": lambda self: 2})()

    P.apply_ep(model, ep_mesh)

    assert parallelize_module_mock.call_count == 1
    args, kwargs = parallelize_module_mock.call_args
    # Should use block.moe.experts, not block.mlp.experts
    assert kwargs["module"] is moe.experts
    assert kwargs["device_mesh"] is ep_mesh
    assert isinstance(kwargs["parallelize_plan"], P.ExpertParallel)


def test_apply_ep_prefers_moe_over_mlp(monkeypatch):
    """Test that apply_ep prefers block.moe over block.mlp when both exist."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    parallelize_module_mock = MagicMock()
    monkeypatch.setattr(P, "parallelize_module", parallelize_module_mock)

    moe = DummyMoE()
    mlp = DummyMoE()  # A different MoE object
    block = DummyBlockWithMoeAttr(moe=moe, mlp=mlp)
    model = DummyModel([block])
    ep_mesh = type("Mesh", (), {"size": lambda self: 2})()

    P.apply_ep(model, ep_mesh)

    assert parallelize_module_mock.call_count == 1
    args, kwargs = parallelize_module_mock.call_args
    # Should use block.moe.experts (not block.mlp.experts)
    assert kwargs["module"] is moe.experts


def test_apply_ep_falls_back_to_mlp(monkeypatch):
    """Test that apply_ep falls back to block.mlp when block.moe doesn't exist."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)
    parallelize_module_mock = MagicMock()
    monkeypatch.setattr(P, "parallelize_module", parallelize_module_mock)

    mlp = DummyMoE()
    # Block with mlp but no moe attribute
    block = DummyBlock(mlp=mlp)
    model = DummyModel([block])
    ep_mesh = type("Mesh", (), {"size": lambda self: 2})()

    P.apply_ep(model, ep_mesh)

    assert parallelize_module_mock.call_count == 1
    args, kwargs = parallelize_module_mock.call_args
    assert kwargs["module"] is mlp.experts


def test_apply_ac_derives_num_experts_from_moe_num_experts(monkeypatch):
    """Test that apply_ac derives num_experts from config.moe_num_experts when num_experts is absent."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_num_experts = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_num_experts
        torch_stub = sys.modules["torch"]
        # Test with various shapes to determine what was captured
        for ne in [8, 16, 32, 64]:
            rhs = type("Mat", (), {"shape": (256, ne)})()
            result = policy_cb(None, torch_stub.ops.aten.mm.default, object(), rhs)
            if result == P.CheckpointPolicy.MUST_SAVE:
                captured_num_experts = ne
                break
        return SELECTIVE_CTX

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        if context_fn is not None:
            context_fn()
        return block

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    # Create model with config containing only moe_num_experts (not num_experts)
    class Config:
        hidden_size = 256
        moe_num_experts = 32  # Only moe_num_experts, not num_experts

    class ModelWithMoeNumExperts:
        def __init__(self):
            self.config = Config()
            self.layers = LayerContainer([DummyBlock()])

    model = ModelWithMoeNumExperts()

    P.apply_ac(model, ignore_router=True)

    # Should find moe_num_experts
    assert captured_num_experts == 32


def test_apply_ac_prefers_num_experts_over_moe_num_experts(monkeypatch):
    """Test that apply_ac prefers config.num_experts over config.moe_num_experts."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_num_experts = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_num_experts
        torch_stub = sys.modules["torch"]
        for ne in [8, 16, 32, 64]:
            rhs = type("Mat", (), {"shape": (256, ne)})()
            result = policy_cb(None, torch_stub.ops.aten.mm.default, object(), rhs)
            if result == P.CheckpointPolicy.MUST_SAVE:
                captured_num_experts = ne
                break
        return SELECTIVE_CTX

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        if context_fn is not None:
            context_fn()
        return block

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    # Create model with both num_experts and moe_num_experts
    class Config:
        hidden_size = 256
        num_experts = 16  # Should be preferred
        moe_num_experts = 64  # Should be ignored

    class ModelWithBothExperts:
        def __init__(self):
            self.config = Config()
            self.layers = LayerContainer([DummyBlock()])

    model = ModelWithBothExperts()

    P.apply_ac(model, ignore_router=True)

    # Should find num_experts first
    assert captured_num_experts == 16


def test_apply_ac_derives_num_experts_from_moe_config(monkeypatch):
    """Test that apply_ac derives num_experts from model.model.moe_config.n_routed_experts."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_num_experts = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_num_experts
        torch_stub = sys.modules["torch"]
        for ne in [8, 16, 32, 64]:
            rhs = type("Mat", (), {"shape": (256, ne)})()
            result = policy_cb(None, torch_stub.ops.aten.mm.default, object(), rhs)
            if result == P.CheckpointPolicy.MUST_SAVE:
                captured_num_experts = ne
                break
        return SELECTIVE_CTX

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        if context_fn is not None:
            context_fn()
        return block

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    class Config:
        hidden_size = 256

    class MoeConfig:
        n_routed_experts = 32

    class Inner:
        def __init__(self):
            self.moe_config = MoeConfig()
            self.layers = LayerContainer([DummyBlock()])

    class Outer:
        def __init__(self):
            self.config = Config()
            self.model = Inner()

    model = Outer()

    P.apply_ac(model, ignore_router=True)

    assert captured_num_experts == 32


def test_apply_ac_prefers_moe_config_over_config_attrs(monkeypatch):
    """Test that apply_ac prefers moe_config.n_routed_experts over model.config attributes."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_num_experts = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_num_experts
        torch_stub = sys.modules["torch"]
        for ne in [8, 16, 32, 64]:
            rhs = type("Mat", (), {"shape": (256, ne)})()
            result = policy_cb(None, torch_stub.ops.aten.mm.default, object(), rhs)
            if result == P.CheckpointPolicy.MUST_SAVE:
                captured_num_experts = ne
                break
        return SELECTIVE_CTX

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        if context_fn is not None:
            context_fn()
        return block

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    class Config:
        hidden_size = 256
        num_experts = 64  # Should be ignored in favor of moe_config

    class MoeConfig:
        n_routed_experts = 32

    class Inner:
        def __init__(self):
            self.moe_config = MoeConfig()
            self.layers = LayerContainer([DummyBlock()])

    class Outer:
        def __init__(self):
            self.config = Config()
            self.model = Inner()

    model = Outer()

    P.apply_ac(model, ignore_router=True)

    # moe_config should take priority over config.num_experts
    assert captured_num_experts == 32


def test_apply_fsdp_handles_block_with_moe_attribute(monkeypatch):
    """Test that apply_fsdp correctly handles blocks with 'moe' attribute (Step3p5 style)."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    mp_policy_mock = MagicMock(return_value="MP_POLICY")
    shard_sentinel = object()

    def fake_shard(dim):
        assert dim == 1
        return shard_sentinel

    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", mp_policy_mock)
    monkeypatch.setattr(P, "Shard", fake_shard)

    moe = DummyMoE()
    block = DummyBlockWithMoeAttr(moe=moe, mlp=None)
    model = DummyModel([block])

    fsdp_mesh = type("Mesh", (), {"size": lambda self: 2})()
    ep_shard_mesh = type("Mesh", (), {"size": lambda self: 2})()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=True,
        ep_shard_enabled=True,
        ep_shard_mesh=ep_shard_mesh,
    )

    # Experts should have a dedicated shard call using block.moe.experts
    experts = moe.experts
    experts_call = _find_call_by_first_arg(fully_shard_mock, experts)
    assert experts_call is not None
    _, experts_kwargs = experts_call
    assert experts_kwargs["mesh"] is ep_shard_mesh

    # Block should be sharded with ignored_params from moe.experts
    block_call = _find_call_by_first_arg(fully_shard_mock, block)
    assert block_call is not None
    _, block_kwargs = block_call
    ignored = block_kwargs.get("ignored_params")
    assert isinstance(ignored, set) and len(ignored) == len(list(experts.parameters()))


def test_apply_fsdp_uses_moe_for_ignored_params(monkeypatch):
    """Test that apply_fsdp uses block.moe.experts for ignored_params when ep_enabled."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    monkeypatch.setattr(P, "MoE", DummyMoE)

    fully_shard_mock = MagicMock()
    monkeypatch.setattr(P, "fully_shard", fully_shard_mock)
    monkeypatch.setattr(P, "MixedPrecisionPolicy", MagicMock(return_value="MP_POLICY"))

    moe = DummyMoE()
    mlp = DummyMoE()  # Different object
    block = DummyBlockWithMoeAttr(moe=moe, mlp=mlp)
    model = DummyModel([block])
    fsdp_mesh = object()

    P.apply_fsdp(
        model=model,
        fsdp_mesh=fsdp_mesh,
        ep_enabled=True,
        ep_shard_enabled=False,
        ep_shard_mesh=None,
    )

    block_call = _find_call_by_first_arg(fully_shard_mock, block)
    assert block_call is not None
    _, block_kwargs = block_call
    ignored = block_kwargs.get("ignored_params")
    # Should use moe.experts, not mlp.experts
    assert isinstance(ignored, set)
    moe_params = set(moe.experts.parameters())
    assert ignored == moe_params


def test_parallelize_model_passes_mp_policy_to_apply_fsdp(monkeypatch):
    """Test that parallelize_model forwards mp_policy to apply_fsdp."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)
    monkeypatch.setattr(P, "apply_ep", MagicMock())
    monkeypatch.setattr(P, "apply_ac", MagicMock())

    world_mesh = FakeWorldMesh({("dp",): 2}, mesh_dim_names=["dp"])

    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 4})()

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()
    sentinel_policy = object()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=None,
        dp_axis_names=("dp",),
        mp_policy=sentinel_policy,
    )

    apply_fsdp_mock.assert_called_once()
    _, kwargs = apply_fsdp_mock.call_args
    assert kwargs.get("mp_policy") is sentinel_policy


def test_parallelize_model_mp_policy_defaults_to_none(monkeypatch):
    """Test that parallelize_model defaults mp_policy to None when not provided."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    apply_fsdp_mock = MagicMock()
    monkeypatch.setattr(P, "apply_fsdp", apply_fsdp_mock)
    monkeypatch.setattr(P, "apply_ep", MagicMock())
    monkeypatch.setattr(P, "apply_ac", MagicMock())

    world_mesh = FakeWorldMesh({("dp",): 2}, mesh_dim_names=["dp"])

    class Inner:
        def __init__(self):
            self.moe_config = type("MC", (), {"n_routed_experts": 4})()

    class Outer:
        def __init__(self):
            self.model = Inner()

    model = Outer()

    P.parallelize_model(
        model=model,
        world_mesh=world_mesh,
        moe_mesh=None,
        dp_axis_names=("dp",),
    )

    apply_fsdp_mock.assert_called_once()
    _, kwargs = apply_fsdp_mock.call_args
    assert kwargs.get("mp_policy") is None


def test_apply_ac_derives_hidden_size_and_num_experts_from_text_config(monkeypatch):
    """Test that apply_ac resolves hidden_size/num_experts from model.config.text_config (VLM models)."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    captured_hidden_size = None
    captured_num_experts = None

    def fake_create_selective_checkpoint_contexts(policy_cb):
        nonlocal captured_hidden_size, captured_num_experts
        torch_stub = sys.modules["torch"]
        for hs in [256, 512, 2048]:
            for ne in [8, 16, 128]:
                rhs = type("Mat", (), {"shape": (hs, ne)})()
                result = policy_cb(None, torch_stub.ops.aten.mm.default, object(), rhs)
                if result == P.CheckpointPolicy.MUST_SAVE:
                    captured_hidden_size = hs
                    captured_num_experts = ne
                    break
            if captured_hidden_size is not None:
                break
        return SELECTIVE_CTX

    def fake_wrapper(block, preserve_rng_state, determinism_check=None, context_fn=None):
        if context_fn is not None:
            context_fn()
        return block

    monkeypatch.setattr(P, "create_selective_checkpoint_contexts", fake_create_selective_checkpoint_contexts)
    monkeypatch.setattr(P, "ptd_checkpoint_wrapper", MagicMock(side_effect=fake_wrapper))

    # VLM pattern: attrs nested under text_config, NOT at top level
    class TextConfig:
        hidden_size = 2048
        num_experts = 128

    class VLMConfig:
        text_config = TextConfig()

    class VLMModel:
        def __init__(self):
            self.config = VLMConfig()
            self.layers = LayerContainer([DummyBlock()])

    model = VLMModel()
    P.apply_ac(model, ignore_router=True)

    assert captured_hidden_size == 2048
    assert captured_num_experts == 128


# ============================================================================
# Tests for apply_cp – skip non-TE attention modules instead of asserting
# ============================================================================


class _FakeAttnModule:
    """Non-TE attention module (e.g. SDPA)."""

    pass


class _FakeSelfAttn:
    def __init__(self, attn_module):
        self.attn_module = attn_module


class _FakeBlockWithAttn:
    def __init__(self, attn_module, moe=None, layer_type=None, attention_type=None):
        self.self_attn = _FakeSelfAttn(attn_module)
        self.mlp = moe if moe is not None else object()
        if layer_type is not None:
            self.layer_type = layer_type
        if attention_type is not None:
            self.attention_type = attention_type


def _stub_dense_cp_hooks(monkeypatch):
    cp_utils_stub = types.ModuleType("nemo_automodel.components.distributed.context_parallel.utils")
    cp_utils_stub.attach_context_parallel_hooks = MagicMock()
    cp_utils_stub.attach_cp_sdpa_hooks = MagicMock()
    monkeypatch.setitem(sys.modules, "nemo_automodel.components.distributed.context_parallel.utils", cp_utils_stub)
    return cp_utils_stub


def test_apply_cp_warns_on_unsupported_non_te_attention(monkeypatch):
    """Non-TE, non-model-owned attention is unsupported under CP: warn, no hooks."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    cp_utils_stub = _stub_dense_cp_hooks(monkeypatch)

    # Stub DotProductAttention in the TE import inside apply_cp
    te_attn_stub = types.ModuleType("transformer_engine.pytorch.attention")

    class DotProductAttention:
        pass

    te_attn_stub.DotProductAttention = DotProductAttention
    monkeypatch.setitem(sys.modules, "transformer_engine", types.ModuleType("transformer_engine"))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", types.ModuleType("transformer_engine.pytorch"))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.attention", te_attn_stub)

    non_te_attn = _FakeAttnModule()  # not a DotProductAttention, no setup_cp_attention
    block = _FakeBlockWithAttn(non_te_attn)
    model = DummyModel([block])

    cp_mesh = MagicMock()
    cp_mesh.get_group.return_value = MagicMock()

    # Stub get_process_group_ranks to avoid real distributed calls
    dist_stub = sys.modules["torch.distributed"]
    dist_stub.get_process_group_ranks = MagicMock(return_value=[0, 1])

    P.apply_cp(model, cp_mesh)

    assert model._cp_enabled is True
    # No generic CP hooks are attached for unsupported attention.
    cp_utils_stub.attach_context_parallel_hooks.assert_not_called()
    cp_utils_stub.attach_cp_sdpa_hooks.assert_not_called()


def test_apply_cp_model_owned_calls_setup_cp_attention(monkeypatch):
    """A self_attn exposing setup_cp_attention installs its own CP attention; no generic hooks."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    cp_utils_stub = _stub_dense_cp_hooks(monkeypatch)

    te_attn_stub = types.ModuleType("transformer_engine.pytorch.attention")

    class DotProductAttention:
        pass

    te_attn_stub.DotProductAttention = DotProductAttention
    monkeypatch.setitem(sys.modules, "transformer_engine", types.ModuleType("transformer_engine"))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", types.ModuleType("transformer_engine.pytorch"))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.attention", te_attn_stub)

    non_te_attn = _FakeAttnModule()  # not a DotProductAttention
    block = _FakeBlockWithAttn(non_te_attn)
    # The model owns its CP attention (e.g. Gemma4's ring) via setup_cp_attention.
    block.self_attn.setup_cp_attention = MagicMock()
    model = DummyModel([block])

    cp_mesh = MagicMock()
    cp_mesh.get_group.return_value = MagicMock()
    sys.modules["torch.distributed"].get_process_group_ranks = MagicMock(return_value=[0, 1])

    P.apply_cp(model, cp_mesh)

    # Model-owned: setup_cp_attention is called; no generic hooks are attached.
    block.self_attn.setup_cp_attention.assert_called_once_with(cp_mesh)
    cp_utils_stub.attach_context_parallel_hooks.assert_not_called()
    cp_utils_stub.attach_cp_sdpa_hooks.assert_not_called()


def test_apply_cp_skips_attention_without_attn_module(monkeypatch):
    """HF attention without attn_module uses dense CP hooks, not TE CP setup."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    cp_utils_stub = _stub_dense_cp_hooks(monkeypatch)

    te_attn_stub = types.ModuleType("transformer_engine.pytorch.attention")

    class DotProductAttention:
        pass

    te_attn_stub.DotProductAttention = DotProductAttention
    monkeypatch.setitem(sys.modules, "transformer_engine", types.ModuleType("transformer_engine"))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", types.ModuleType("transformer_engine.pytorch"))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.attention", te_attn_stub)

    class AttentionWithoutAttnModuleBlock:
        layer_type = "full_attention"

        def __init__(self):
            self.self_attn = _FakeAttnModule()
            self.mlp = object()

    model = DummyModel([AttentionWithoutAttnModuleBlock()])
    cp_mesh = MagicMock()

    P.apply_cp(model, cp_mesh)
    cp_mesh.get_group.assert_not_called()
    # Unsupported (non-TE, non-model-owned) attention -> warn, no hooks.
    cp_utils_stub.attach_cp_sdpa_hooks.assert_not_called()


def _setup_te_and_dist_stubs(monkeypatch, DotProductAttention):
    """Register TE and torch.distributed stubs needed by apply_cp."""
    te_attn_stub = types.ModuleType("transformer_engine.pytorch.attention")
    te_attn_stub.DotProductAttention = DotProductAttention
    monkeypatch.setitem(sys.modules, "transformer_engine", types.ModuleType("transformer_engine"))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", types.ModuleType("transformer_engine.pytorch"))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.attention", te_attn_stub)

    # apply_cp uses torch.distributed.get_process_group_ranks via attribute access
    torch_mod = sys.modules["torch"]
    dist_stub = sys.modules["torch.distributed"]
    dist_stub.get_process_group_ranks = MagicMock(return_value=[0, 1])
    torch_mod.distributed = dist_stub


def test_apply_cp_configures_te_attention(monkeypatch):
    """apply_cp should call set_context_parallel_group on TE DotProductAttention modules."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    class DotProductAttention:
        def __init__(self):
            self.set_context_parallel_group = MagicMock()

    _setup_te_and_dist_stubs(monkeypatch, DotProductAttention)

    te_attn = DotProductAttention()
    block = _FakeBlockWithAttn(te_attn)
    model = DummyModel([block])

    cp_mesh = MagicMock()
    cp_mesh.get_group.return_value = MagicMock()

    P.apply_cp(model, cp_mesh)

    te_attn.set_context_parallel_group.assert_called_once()


def test_apply_cp_uses_attention_type_and_all_gather_for_sliding_attention(monkeypatch):
    """Blocks that name attention style with ``attention_type`` should still
    receive TE context-parallel setup."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    class DotProductAttention:
        def __init__(self):
            self.set_context_parallel_group = MagicMock()

    _setup_te_and_dist_stubs(monkeypatch, DotProductAttention)

    te_attn = DotProductAttention()
    block = _FakeBlockWithAttn(te_attn, attention_type="sliding_attention")
    model = DummyModel([block])

    cp_mesh = MagicMock()
    cp_mesh.get_group.return_value = MagicMock()

    P.apply_cp(model, cp_mesh, cp_comm_type="p2p")

    te_attn.set_context_parallel_group.assert_called_once()
    _, kwargs = te_attn.set_context_parallel_group.call_args
    assert kwargs["cp_comm_type"] == "all_gather"


def test_apply_cp_mixed_te_and_non_te(monkeypatch):
    """apply_cp configures TE blocks; unsupported non-TE blocks warn (no hooks)."""
    P = _import_parallelizer_with_stubs(monkeypatch)
    cp_utils_stub = _stub_dense_cp_hooks(monkeypatch)

    class DotProductAttention:
        def __init__(self):
            self.set_context_parallel_group = MagicMock()

    _setup_te_and_dist_stubs(monkeypatch, DotProductAttention)

    te_attn = DotProductAttention()
    non_te_attn = _FakeAttnModule()
    block_te = _FakeBlockWithAttn(te_attn)
    block_non_te = _FakeBlockWithAttn(non_te_attn)
    model = DummyModel([block_te, block_non_te])

    cp_mesh = MagicMock()
    cp_mesh.get_group.return_value = MagicMock()

    P.apply_cp(model, cp_mesh)

    te_attn.set_context_parallel_group.assert_called_once()
    # The non-TE block is unsupported under CP -> warn, no generic hooks.
    cp_utils_stub.attach_cp_sdpa_hooks.assert_not_called()


# ============================================================================
# Tests for apply_cp – linear_attention (FLA CP) branches
# ============================================================================


class _FakeLinearAttn:
    """CP-aware linear attention module stub."""

    def __init__(self):
        self._cp_mesh = None


class _FakeLinearAttnNoCPAttr:
    """Linear attention module without _cp_mesh attribute."""

    pass


class _FakeBlockLinearAttn:
    """Block with layer_type='linear_attention'."""

    def __init__(self, linear_attn=None, moe=None):
        self.layer_type = "linear_attention"
        self.layer_idx = 0
        self.linear_attn = linear_attn
        self.mlp = moe if moe is not None else object()


def test_apply_cp_sets_cp_mesh_on_linear_attention(monkeypatch):
    """apply_cp should attach cp_mesh to blocks with layer_type='linear_attention'."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    class DotProductAttention:
        pass

    _setup_te_and_dist_stubs(monkeypatch, DotProductAttention)

    linear_attn = _FakeLinearAttn()
    block = _FakeBlockLinearAttn(linear_attn=linear_attn)
    model = DummyModel([block])

    cp_mesh = MagicMock()

    P.apply_cp(model, cp_mesh)

    assert linear_attn._cp_mesh is cp_mesh


def test_apply_cp_linear_attention_warns_when_no_cp_aware_module(monkeypatch):
    """apply_cp should warn when linear_attention block has no CP-aware linear_attn."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    class DotProductAttention:
        pass

    _setup_te_and_dist_stubs(monkeypatch, DotProductAttention)

    # Block has linear_attn but without _cp_mesh attribute
    block = _FakeBlockLinearAttn(linear_attn=_FakeLinearAttnNoCPAttr())
    block.layer_idx = 3
    model = DummyModel([block])

    cp_mesh = MagicMock()

    with patch.object(P.logger, "warning") as mock_warn:
        P.apply_cp(model, cp_mesh)
    mock_warn.assert_called_once()
    assert "linear_attention" in str(mock_warn.call_args) or "CP-aware" in str(mock_warn.call_args)


def test_apply_cp_linear_attention_warns_when_no_linear_attn_attr(monkeypatch):
    """apply_cp should warn when linear_attention block lacks linear_attn entirely."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    class DotProductAttention:
        pass

    _setup_te_and_dist_stubs(monkeypatch, DotProductAttention)

    # Block without linear_attn attribute at all
    block = _FakeBlockLinearAttn()
    block.linear_attn = None
    delattr(block, "linear_attn")
    model = DummyModel([block])

    cp_mesh = MagicMock()

    with patch.object(P.logger, "warning") as mock_warn:
        P.apply_cp(model, cp_mesh)
    mock_warn.assert_called_once()


def test_apply_cp_mixed_full_and_linear_attention(monkeypatch):
    """apply_cp should handle models with both full_attention and linear_attention blocks."""
    P = _import_parallelizer_with_stubs(monkeypatch)

    class DotProductAttention:
        def __init__(self):
            self.set_context_parallel_group = MagicMock()

    _setup_te_and_dist_stubs(monkeypatch, DotProductAttention)

    # full_attention block
    te_attn = DotProductAttention()
    block_full = _FakeBlockWithAttn(te_attn)

    # linear_attention block
    linear_attn = _FakeLinearAttn()
    block_linear = _FakeBlockLinearAttn(linear_attn=linear_attn)

    model = DummyModel([block_full, block_linear])

    cp_mesh = MagicMock()
    cp_mesh.get_group.return_value = MagicMock()

    P.apply_cp(model, cp_mesh)

    # full_attention block: TE attention configured
    te_attn.set_context_parallel_group.assert_called_once()
    # linear_attention block: cp_mesh attached
    assert linear_attn._cp_mesh is cp_mesh
