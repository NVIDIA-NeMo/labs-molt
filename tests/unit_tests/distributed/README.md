# Distributed Component Tests

This directory contains unit tests for distributed training components.

## Test Files

### `test_parallelizer.py`

Comprehensive unit tests for the `fsdp2_strategy_parallelize` function in `nemo_automodel.components.distributed.parallelizer`.

#### Test Coverage

The test suite covers the following scenarios:

1. **Basic Functionality**
   - Data parallelism only (no tensor parallelism)
   - Tensor parallelism with custom plans
   - Different model architectures (LLaMA, Gemma3)

2. **Parallel Plan Selection**
   - Custom parallel plans (dictionary format)
   - Custom parallel plans (string import path)
   - Function-based parallel plans
   - Optimized parallel plans based on model type
   - HuggingFace TP plan fallback

3. **Error Handling**
   - Attention heads not divisible by TP size
   - Invalid parallel plan imports
   - Sequence parallel with unsupported plans
   - Hybrid sharding validation

4. **Advanced Features**
   - Activation checkpointing
   - Mixed precision policies
   - CPU offload policies
   - Sequence parallelism
   - Device mesh selection (DP/CP)

5. **Optimization Testing**
   - Reshard optimization for transformer layers
   - Proper FSDP wrapping behavior

#### Test Architecture

The tests use extensive mocking to avoid requiring actual distributed environments:

- **Mock Device Meshes**: Simulate multi-device setups
- **Mock Distributed Environment**: Replace `torch.distributed` calls
- **Mock Model Classes**: Simple test models with required attributes
- **Mock Parallel Plans**: Test different parallelization strategies

#### Running Tests

```bash
# Run from the Automodel root directory
pytest tests/unit_tests/distributed/test_parallelizer.py -v

# Run specific test classes
pytest tests/unit_tests/distributed/test_parallelizer.py::TestFSDP2StrategyParallelize -v

# Run with coverage
pytest tests/unit_tests/distributed/test_parallelizer.py --cov=nemo_automodel.components.distributed.parallelizer
```

#### Dependencies

- `pytest`: Test framework
- `torch`: For neural network components
- `unittest.mock`: For mocking dependencies

The tests are designed to work without requiring:
- Actual GPUs or distributed setup
- Real model weights or training data
- External parallel training libraries

#### Mock Strategy

The tests use a layered mocking approach:

1. **Framework Level**: Mock `torch.distributed`, `device_mesh`, FSDP components
2. **Function Level**: Mock specific functions like `parallelize_module`, `get_hf_tp_shard_plan`
3. **Model Level**: Use lightweight mock models with required interfaces

This allows comprehensive testing of the parallelization logic without the complexity of actual distributed training. 