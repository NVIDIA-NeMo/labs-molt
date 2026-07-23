# Molt CPU RL 逐行教程

[说明](./README_CN.md) | [English walkthrough](./WALKTHROUGH.md)

## 动机

Molt 的目标是让 RL 实验更方便，但生产 quick start 会先启动 Ray、vLLM、AutoModel、FSDP 和 GPU 集群，刚接触仓库的贡献者很难直接逐行观察 reward 到参数更新的逻辑。

这个教程为该逻辑提供了一条小型 CPU 学习路径。它复用 Molt 原有的数据集、agent、环境、trajectory、experience、优势估计、replay buffer 和 loss 实现，只替换分布式调度与 GPU 模型执行边界。它并不是第二套训练后端，而是一条适合学习、调试和本地验证轻量修改的可读单进程路径。

同样的结构也让 coding agent 无需模拟集群，就能把一项修改从输入数据一路追踪到 prompt、rollout、reward、experience、loss、梯度和更新后的 policy。

## 哪些部分保持真实

- 通过 Molt `get_tokenizer()` 加载的 Qwen3 tokenizer 与 chat template。
- Hugging Face `Dataset`、`PromptDataset` 和 `StatefulDataLoader`。
- `StepEnvRunner`、math 环境与 grader、`Result` 和 `Trajectory`。
- 分组完整性检查、动态过滤以及 `Trajectory -> Experience` 转换。
- 通过 `RemoteExperienceMaker` 执行的旧策略与冻结参考策略 forward。
- `reinforce_baseline`、replay buffer 拼批、`PolicyLoss`、KL-as-loss、autograd、梯度裁剪、AdamW 与学习率 scheduler。

本地 policy 不是只返回随机形状的 stub，而是一个真正运行在 CPU 上的 `Embedding -> Linear` 模型。本地生成 transport 从它的 softmax 中采样 token，并记录对应的行为策略 log probability。生成与训练共享同一个模型对象，因此优化器更新会直接反映到下一次 forward 中。

## 哪些部分被替换

- Ray actor、队列、placement 以及 `remote/wait` 调度。
- vLLM serving 与 transport。
- AutoModel/FSDP/CUDA/NCCL 模型执行。
- 与上述分布式 runtime 耦合的 `PolicyTrainer`/`FsdpStrategy` 外壳。教程显式写出其对应的 CPU forward、loss、backward、梯度裁剪、optimizer 和 scheduler 操作。
- FSDP 到 vLLM 的权重广播。单进程中 rollout 与训练共享同一个模型对象。

所有替换都集中展示在单个教程文件开头附近。如果代码意外进入 Ray 队列调度或 vLLM `SamplingParams` 等生产路径，它会立即失败，而不是静默跳过。

## 文件与源码对应关系

- `cpu_rl_walkthrough.py` 是唯一可执行入口。它刻意保持线性，便于从第一行开始调试。
- `requirements-cpu.txt` 包含固定版本的 CPU 依赖，不安装 Ray 或 GPU runtime。

教程不复制 Molt 的实现。import 边界对应 `molt/models/__init__.py`、`samples_generator.py` 和 `experience_maker.py` 中的 eager import。`TinyPolicy` 与本地生成 engine 实现 `Actor.forward` 和 `StepEnvRunner` 所消费的接口。显式更新代码块对应 `PolicyTrainer.training_step` 与 `FsdpStrategy.backward/optimizer_step` 中在 CPU 上仍有意义的部分。

现有的 `molt/`、`tests/` 和生产 examples 保持不变；所有适配代码均位于本目录。

## 运行方法

教程依赖与完整仓库依赖刻意分开：

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r examples/tutorial/requirements-cpu.txt
.venv/bin/python examples/tutorial/cpu_rl_walkthrough.py
```

脚本会把仓库根目录加入 `sys.path`，因此不需要运行 `pip install -e . --no-deps`，也不会产生依赖不完整的 `molt` 包元数据。直接依赖均固定版本；Linux 还会查询 PyTorch 官方 CPU wheel index。

tokenizer 固定在 `Qwen/Qwen3-0.6B` 的 revision `c1899de289a04d12100db370d81485cdf75e47ca`。`snapshot_download()` 只允许下载 config、tokenizer、词表和 merge 文件。首次运行下载约 15 MB，不下载模型权重。

缓存准备好以后，可以显式离线运行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv/bin/python examples/tutorial/cpu_rl_walkthrough.py
```

## 实际执行链

```text
Hugging Face Dataset.from_list（本地一条数据）
  -> PromptDataset.__getitem__ / collate_fn
  -> StatefulDataLoader(batch_size=1)
  -> _collect_prompt_batch
  -X Ray dispatch / vLLM transport（直接调用 runner + TinyVllmEngine）
  -> examples/python/agents/math.py
  -> StepEnvRunner.execute -> MathEnv.step -> Result -> Trajectory
  -> SamplesGenerator._filter_group（完整分组检查 + 动态过滤）
  -> SamplesGenerator._process_response_into_experience
  -> RemoteExperienceMaker.build_experiences
  -> TinyPolicy 旧策略 forward + 冻结参考策略 forward
  -> reinforce_baseline advantage
  -> balance_experiences
  -> NaiveReplayBuffer -> torch DataLoader
  -X PolicyTrainer / AutoModel / FSDP Actor（显式 CPU 更新 + TinyPolicy）
  -> log_probs_from_logits / PolicyLoss / compute_approx_kl / agg_loss
  -> loss.backward -> clip_grad_norm_ -> AdamW -> scheduler
  -X FSDP 到 vLLM 的权重广播（共享一个模型对象）
```

`-X` 标记每一个切断的边界。本地 `ray.get` 只负责同步解包本地返回值；真正的 `wait`/队列调度和 vLLM `SamplingParams` 保持禁用。`Result`、`Trajectory` 和 `Experience` 都是 Molt 原有的数据记录，不是教程自定义字典。

## 有意做的小规模调整

- 数据集是 Hugging Face 内存中的一条数据，字段与 Qwen3 math quick start 相同，包含 `prompt` 和 `reward_model`。
- 生产 recipe 每个 prompt 的 8 次采样缩小为 2 次。逐 token 的 inverse-CDF 抽样会确定性地产生一个 `\boxed{4}` 和一个 `\boxed{5}`，同时保留自回归策略与 log probability 的对应关系。
- 两条 rollout 共用一个 group ID，因此真实的 `reinforce_baseline` 会产生正、负 advantage。
- 保留 quick start 中的动态过滤、旧策略/参考策略 forward、KL-as-loss（`k2`，系数 `0.001`）、PPO clipping 和 importance-sampling correction。第一次更新时 actor 与 reference 相同，所以 KL 数值为零，但完整分支确实执行。
- DP、TP、CP 均为 1，关闭 dynamic batching。这些设置去掉吞吐量与显存调度，不改变这里要观察的单 rank 张量语义。
- CPU 上关闭 pinned memory。prompt loader 仍是 `StatefulDataLoader`，训练仍使用 `NaiveReplayBuffer + DataLoader`。
- AdamW 保留 quick start 的 betas、epsilon、零 weight decay、constant scheduler 与 `max_norm=1.0`。只有学习率从 `1e-6` 提高为 `0.01`，使一次教程更新产生可见变化。

## 建议断点顺序

1. `PromptDataset.__getitem__`：tokenization 前的 prompt 渲染。
2. `_collect_prompt_batch`：DataLoader 输出的五个字段。
3. `StepEnvRunner.execute`：tokenizer、本地生成 transport 与真实环境 step。
4. `SamplesGenerator._filter_group`：rollout 完整性，以及均值为 `0.5` 的分组通过动态过滤。
5. `SamplesGenerator._process_response_into_experience`：token 轴 `T` 如何变为 next-token 轴 `T-1`。
6. `RemoteExperienceMaker.make_experience`：本地旧策略/参考策略 forward 与 log probability 字段。
7. `RemoteExperienceMaker.compute_advantages_and_returns`：reward 如何变为正、负 advantage。
8. `NaiveReplayBuffer.append/collate_fn`：rollout batch 如何变为训练 microbatch。
9. `PolicyLoss.forward`，然后是 `compute_approx_kl/agg_loss`：PPO、importance correction 与 KL-as-loss。
10. `total_loss.backward()`：梯度、裁剪、AdamW、scheduler 与 zero-grad。

## 验证情况

教程已在 macOS arm64、Python 3.12 的全新环境中验证，该环境没有安装 `molt`、Ray 或 vLLM。`pip check`、在线 tokenizer 初始化、缓存后的离线执行以及端到端断言均通过。Linux 依赖解析会选择 PyTorch `2.11.0+cpu` wheel，仍建议在 CI 中增加 Linux runtime smoke test。

完整 Molt 测试环境仍需要生产依赖 vLLM。此处的 requirements 只服务于 CPU 教程，不替代整个仓库的开发环境。

## 后续计划

计划中的下一步是围绕这条 CPU 路径提供一套 coding-agent skill，工作流为：

1. 在 CPU 教程中实现并观察轻量修改；
2. 在本地追踪受影响的 Molt 数据记录、张量、分支与指标；
3. 把修改迁移到对应的 GPU/分布式路径；
4. 只在服务器上验证剩余的分布式与硬件行为。

CPU 路径与生产代码保持相近结构，可以让用户和 coding agent 更容易完成这一步迁移。
