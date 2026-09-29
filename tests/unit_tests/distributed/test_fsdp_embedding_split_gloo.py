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

"""Real FSDP numerical coverage for independently sharded embedding tables."""

from __future__ import annotations

import copy
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor

from nemo_automodel.components.distributed.parallelizer import (
    DefaultParallelizationStrategy,
    Qwen3_5ParallelizationStrategy,
)

# Over the default 5s budget on purpose: this module spawns worker processes; every child re-imports torch from scratch.
# Shrink the work or the process count before raising this further.
pytestmark = pytest.mark.timeout(60)


class _ToyLM(nn.Module):
    def __init__(
        self, *, tied: bool, input_in_container: bool, output_in_container: bool, shared_module: bool = False
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=tied)
        self.shared_module = shared_module
        input_embedding = nn.Embedding(32, 8)
        output_embedding = nn.Linear(8, 32, bias=False)
        if tied:
            output_embedding.weight = input_embedding.weight

        # Recursive sharding visits ModuleDict children before the embedding pass.
        # Exercise each table both inside and outside that earlier traversal.
        self.transformer = nn.ModuleDict()
        if input_in_container:
            self.transformer["wte"] = input_embedding
        else:
            self.embed_tokens = input_embedding
        self.transformer["blocks"] = nn.ModuleList([nn.Linear(8, 8)])
        if not shared_module:
            if output_in_container:
                self.transformer["ff_out"] = output_embedding
            else:
                self.lm_head = output_embedding

    def get_input_embeddings(self) -> nn.Module:
        return self.transformer["wte"] if "wte" in self.transformer else self.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        if self.shared_module:
            return self.get_input_embeddings()
        head = self.transformer["ff_out"] if "ff_out" in self.transformer else self.lm_head
        return head[0] if isinstance(head, nn.Sequential) else head

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute token logits.

        Args:
            token_ids: Token IDs of shape [batch, sequence].

        Returns:
            Logits of shape [batch, sequence, vocab].
        """
        hidden = self.get_input_embeddings()(token_ids)
        for block in self.transformer["blocks"]:
            hidden = torch.tanh(block(hidden))
        if self.shared_module:
            return nn.functional.linear(hidden, self.get_output_embeddings().weight)
        head = self.transformer["ff_out"] if "ff_out" in self.transformer else self.lm_head
        return head(hidden)


def _full_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a distributed tensor for numerical comparison.

    Args:
        tensor: Tensor of arbitrary shape, optionally a DTensor sharded on its
            first dimension.

    Returns:
        Replicated tensor with the same global shape and values as ``tensor``.
    """
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _run_case(
    mesh: DeviceMesh,
    *,
    strategy: DefaultParallelizationStrategy,
    tied: bool,
    input_in_container: bool = False,
    output_in_container: bool = False,
    nested_output: bool = False,
    shared_module: bool = False,
) -> None:
    torch.manual_seed(2026)
    model = _ToyLM(
        tied=tied,
        input_in_container=input_in_container,
        output_in_container=output_in_container,
        shared_module=shared_module,
    )
    if nested_output:
        model.transformer["ff_out"] = nn.Sequential(model.transformer["ff_out"])
    reference = copy.deepcopy(model)
    mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)

    if tied and not shared_module and (input_in_container or output_in_container):
        original_parameters = list(model.parameters())
        with pytest.raises(ValueError, match="Distinct tied input/output embedding modules inside a ModuleList or ModuleDict"):
            strategy.parallelize(model, device_mesh=mesh, mp_policy=mp_policy)
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
        assert [id(param) for param in model.parameters()] == [id(param) for param in original_parameters]
        assert not any(isinstance(module, FSDPModule) for module in model.modules())
        return

    strategy.parallelize(model, device_mesh=mesh, mp_policy=mp_policy)

    assert isinstance(model, FSDPModule)
    if tied:
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
        assert len(list(model.parameters())) == len(list(reference.parameters()))
        if shared_module:
            assert model.get_input_embeddings() is model.get_output_embeddings()
            assert isinstance(model.get_input_embeddings(), FSDPModule) == input_in_container
        else:
            assert not isinstance(model.get_input_embeddings(), FSDPModule)
            assert not isinstance(model.get_output_embeddings(), FSDPModule)
    else:
        assert isinstance(model.get_input_embeddings(), FSDPModule)
        assert isinstance(model.get_output_embeddings(), FSDPModule)

    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    model_optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)

    actual = model(token_ids)
    expected = reference(token_ids)
    torch.testing.assert_close(actual, expected)
    actual.square().mean().backward()
    expected.square().mean().backward()

    actual_parameters = dict(model.named_parameters())
    actual_grad_norms = []
    expected_grad_norms = []
    for name, expected_parameter in reference.named_parameters():
        actual_gradient = _full_tensor(actual_parameters[name].grad)
        expected_gradient = expected_parameter.grad
        torch.testing.assert_close(actual_gradient, expected_gradient)
        actual_grad_norms.append(actual_gradient.float().norm())
        expected_grad_norms.append(expected_gradient.float().norm())
    actual_global_norm = torch.stack(actual_grad_norms).norm()
    expected_global_norm = torch.stack(expected_grad_norms).norm()
    torch.testing.assert_close(actual_global_norm, expected_global_norm)

    model_optimizer.step()
    reference_optimizer.step()
    if tied:
        assert model.get_input_embeddings().weight is model.get_output_embeddings().weight
    actual_state = model.state_dict()
    expected_state = reference.state_dict()
    for name, expected_parameter in expected_state.items():
        torch.testing.assert_close(_full_tensor(actual_state[name]), expected_parameter)


def _worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (1, world_size, 1, 1), mesh_dim_names=("dp_replicate", "dp_shard", "cp", "tp"))
        for strategy in (DefaultParallelizationStrategy(), Qwen3_5ParallelizationStrategy()):
            for tied in (False, True):
                for input_in_container in (False, True):
                    for output_in_container in (False, True):
                        _run_case(
                            mesh,
                            strategy=strategy,
                            tied=tied,
                            input_in_container=input_in_container,
                            output_in_container=output_in_container,
                        )
            _run_case(
                mesh,
                strategy=strategy,
                tied=True,
                input_in_container=True,
                output_in_container=True,
                nested_output=True,
            )
            # LLaDA's tied layout exposes the same module from both getters.
            for input_in_container in (False, True):
                _run_case(
                    mesh,
                    strategy=strategy,
                    tied=True,
                    input_in_container=input_in_container,
                    shared_module=True,
                )
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
def test_split_embedding_fsdp_matches_unsharded_reference(world_size: int) -> None:
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)
