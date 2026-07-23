# CPU 优先的 RL 学习路径：设计与范围

[English](./README.md) | [详细教程](./WALKTHROUGH_CN.md)

## 摘要

这个目录为 Molt 的核心 RL 逻辑增加了一条单进程、纯 CPU 的执行路径。它面向希望先在个人电脑上理解奖励到参数更新的完整过程，或做轻量修改，再把同一项修改迁移到 GPU 服务器的贡献者。

教程直接复用 Molt 原有的数据集、agent、环境、trajectory、experience、优势估计、replay buffer 和 loss 实现；只替换必须依赖分布式调度或 GPU 模型执行的边界。

它是学习和调试工具，不是第二套生产训练后端。

## 动机

Molt 的目标是让 RL 实验更方便，但生产执行链必然会把学习算法与 Ray、vLLM、AutoModel、FSDP、CUDA 及其通信机制组合在一起。这很适合真实训练，却使贡献者很难在没有 GPU 集群的情况下观察最核心的控制流。

纯 CPU 路径让贡献者可以：

- 从一条数据开始，依次观察 prompt、rollout、reward、experience、loss、梯度和优化器更新；
- 在单进程中使用普通断点逐行调试；
- 先在本地验证轻量的算法或数据流修改；
- 明确迁移到服务器之后，真正还需要验证哪些分布式和硬件行为。

这条显式执行链也方便 coding agent 把一项修改从输入字段或参数一路追踪到实际分支、张量、指标和测试。

## 设计原则

1. **以 Molt 原始实现为准。** 直接导入并执行仓库代码，不把实现复制进教程。
2. **只切断基础设施边界。** 替换 Ray 调度、vLLM 服务、分布式模型执行和权重广播，保留真实的 RL 数据记录与计算。
3. **保持线性执行。** 用一个脚本展示完整流程，不引入并行编排层，便于人和工具直接追踪。
4. **让替换点清晰可见。** 如果教程意外进入 Ray 队列调度或 vLLM 采样配置，本地 shim 会立即报错。
5. **不改生产代码。** 所有 CPU 适配都放在 `examples/tutorial/` 下。

## 本次新增

- `cpu_rl_walkthrough.py`：可直接运行的 CPU 学习路径。
- `requirements-cpu.txt`：不包含 Ray、vLLM 和 GPU runtime 的小型固定依赖环境。
- `WALKTHROUGH.md` 和 `WALKTHROUGH_CN.md`：安装方式、源码对应关系、执行链、小规模设置与建议断点。

教程使用真实的 Qwen3 tokenizer 和 chat template，但不会下载模型权重。本地模型是一个真正运行在 CPU 上的 `Embedding -> Linear` policy；采样 token、行为策略 log probability、反向传播和优化器更新在数学上保持一致。

## 复用范围与替换边界

| 模块 | 本教程中的行为 |
| --- | --- |
| 数据与加载 | 真实的 Hugging Face `Dataset`、`PromptDataset` 和 `StatefulDataLoader` |
| Agent、环境与奖励 | 真实的 math agent、`StepEnvRunner`、环境、grader 和 `Result` |
| RL 数据记录 | 真实的 `Trajectory` 和 `Experience` |
| 采样处理 | 真实的分组完整性检查、动态过滤和 trajectory 转换 |
| Experience 构建 | 真实的 `RemoteExperienceMaker` 逻辑，本地同步解包返回值 |
| 算法 | 真实的 `reinforce_baseline`、replay 拼批、PPO policy loss、重要性修正与 KL-as-loss |
| 参数更新 | 真实的 PyTorch autograd、梯度裁剪、AdamW 和 scheduler |
| Ray | 移除 actor、队列、placement、等待和调度 |
| vLLM | 替换为小型本地生成 transport |
| GPU 模型栈 | 用 rollout/training 共享的 CPU actor 加一份冻结 reference 副本，替换 AutoModel、FSDP、CUDA/NCCL 和权重广播 |

## 非目标

- 提供可用于生产训练的正式 CPU 后端。
- 模拟模型质量、吞吐量、显存压力或分布式时序。
- 验证多 rank、vLLM、FSDP、CUDA 或通信正确性。
- 替代 Molt 完整的开发环境或测试依赖。

这些行为仍应在正常的 GPU/分布式路径中验证。本教程只负责让迁移前的算法逻辑更容易观察，并缩小需要交给服务器验证的范围。

## 验证情况

教程已在 macOS arm64、Python 3.12 的全新环境中完成端到端验证；该环境没有安装 `molt`、Ray 或 vLLM。在线初始化 tokenizer、缓存后的离线运行、`pip check` 以及脚本内部的形状和数据流断言均通过，且没有下载模型权重。

仓库的 `compileall` 和 CPU-safe 单元测试也已通过。收集完整生产测试仍然需要 vLLM；教程环境并不替代这些开发依赖。Linux 依赖解析会选择 PyTorch 官方 CPU wheel，后续适合在 CI 中补充 Linux 运行时 smoke test。

## 后续计划

下一步计划围绕这条 CPU 路径提供一套配套 skill，供用户和 coding agent 使用：

1. 在 CPU 路径上实现并观察轻量修改；
2. 在本地追踪受影响的数据记录、张量、分支和指标；
3. 把修改映射到对应的 GPU/分布式路径；
4. 只把剩余的硬件与分布式验证放到服务器完成。

安装方法、完整执行链、源码映射和建议断点请见[详细教程](./WALKTHROUGH_CN.md)。
