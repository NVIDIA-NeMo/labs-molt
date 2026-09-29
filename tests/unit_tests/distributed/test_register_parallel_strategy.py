# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch.nn as nn

from nemo_automodel.components.distributed import parallelizer as p
from nemo_automodel.components.distributed.parallelizer import (
    ParallelizationStrategy,
    get_parallelization_strategy,
    register_parallel_strategy,
)


def test_register_parallel_strategy_decorator_registers_and_resolves_by_model_name():
    original_registry = dict(p.PARALLELIZATION_STRATEGIES)
    try:

        @register_parallel_strategy(name="OutOfTreeModel")
        class OutOfTreeModelStrat(ParallelizationStrategy):
            def parallelize(self, model: nn.Module, *args, **kwargs) -> nn.Module:
                return model

        # Registered under string key
        assert "OutOfTreeModel" in p.PARALLELIZATION_STRATEGIES

        # Shadow with a model class of the same name; lookup should resolve by name
        class OutOfTreeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()

        model = OutOfTreeModel()
        strategy = get_parallelization_strategy(model)

        assert strategy is p.PARALLELIZATION_STRATEGIES["OutOfTreeModel"]
        assert isinstance(strategy, ParallelizationStrategy)
        assert strategy.parallelize(model) is model
    finally:
        # Restore registry to avoid test ordering side effects
        p.PARALLELIZATION_STRATEGIES.clear()
        p.PARALLELIZATION_STRATEGIES.update(original_registry)


def test_register_parallel_strategy_decorator_raises_error_if_not_a_strategy():
    with pytest.raises(ValueError):

        @register_parallel_strategy
        class NotAParallelizationStrategy:
            def parallelize(self, model: nn.Module, *args, **kwargs) -> nn.Module:
                return model


def test_register_parallel_strategy_with_custom_name_registers_under_provided_key():
    original_registry = dict(p.PARALLELIZATION_STRATEGIES)
    try:

        @register_parallel_strategy(name="CustomKeyName")
        class SomeStrategy(ParallelizationStrategy):
            def parallelize(self, model: nn.Module, *args, **kwargs) -> nn.Module:
                return model

        assert "CustomKeyName" in p.PARALLELIZATION_STRATEGIES
        assert isinstance(p.PARALLELIZATION_STRATEGIES["CustomKeyName"], ParallelizationStrategy)

        class CustomKeyName(nn.Module):
            def __init__(self) -> None:
                super().__init__()

        model = CustomKeyName()
        strategy = get_parallelization_strategy(model)
        assert strategy is p.PARALLELIZATION_STRATEGIES["CustomKeyName"]
        assert strategy.parallelize(model) is model
    finally:
        p.PARALLELIZATION_STRATEGIES.clear()
        p.PARALLELIZATION_STRATEGIES.update(original_registry)
