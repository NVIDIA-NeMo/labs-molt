# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run Molt's real RL data path with CPU replacements for Ray and the GPU training stack."""

# ruff: noqa: E402

import asyncio
import copy
import sys
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import torch
from datasets import Dataset
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

import molt

# The production package exports GPU model classes eagerly. Keep its package path so
# the tutorial imports the original CPU-safe loss and utility modules directly.
models_package = ModuleType("molt.models")
models_package.__file__ = str(repo_root / "molt/models/__init__.py")
models_package.__package__ = "molt.models"
models_package.__path__ = [str(repo_root / "molt/models")]
sys.modules["molt.models"] = models_package
molt.models = models_package

# Rollout code imports Ray/vLLM at module scope. Local ``get`` unwraps synchronous
# results; queue scheduling and SamplingParams construction still fail immediately.
ray = ModuleType("ray")
ray.get = lambda value: value
ray.wait = Mock(side_effect=RuntimeError("The CPU tutorial does not run Ray scheduling."))
ray.__cpu_tutorial_stub__ = True
sys.modules["ray"] = ray

vllm = ModuleType("vllm")
vllm.SamplingParams = Mock(side_effect=RuntimeError("The CPU tutorial does not run vLLM dispatch."))
vllm.__cpu_tutorial_stub__ = True
sys.modules["vllm"] = vllm

from molt.agents.base import load_agent_runner
from molt.datasets import PromptDataset
from molt.models.loss import PolicyLoss, agg_loss
from molt.models.utils import compute_approx_kl, log_probs_from_logits
from molt.trainer.algorithm import FixedKLController, NaiveReplayBuffer
from molt.trainer.algorithm.experience import balance_experiences
from molt.trainer.rollout.experience_maker import RemoteExperienceMaker
from molt.trainer.rollout.samples_generator import SamplesGenerator, _collect_prompt_batch
from molt.utils import get_tokenizer


class TinyPolicy(torch.nn.Module):
    """CPU replacement for AutoModel/FSDP with Actor-compatible outputs."""

    def __init__(self, vocab_size, transition_sequences, hidden_size=16, target_logit=20.0):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.cls = torch.nn.Linear(hidden_size, vocab_size)

        transitions = {}
        for sequence in transition_sequences:
            for previous, target in zip(sequence, sequence[1:]):
                transitions.setdefault(previous, set()).add(target)
        if len(transitions) > hidden_size:
            raise ValueError("TinyPolicy hidden_size is too small for the requested token transitions")
        with torch.no_grad():
            self.embedding.weight.zero_()
            self.cls.weight.zero_()
            self.cls.bias.zero_()
            for hidden_index, (previous, targets) in enumerate(transitions.items()):
                self.embedding.weight[previous, hidden_index] = 1.0
                for target in targets:
                    self.cls.weight[target, hidden_index] = target_logit

    def forward(self, sequences, action_mask=None, attention_mask=None):
        del attention_mask
        logits = self.cls(self.embedding(sequences))
        next_tokens = torch.roll(sequences, shifts=-1, dims=1)
        log_probs = log_probs_from_logits(logits, next_tokens)[:, :-1]
        output = {"logits": logits, "log_probs": log_probs}
        if action_mask is not None:
            output["action_log_probs"] = log_probs[:, -action_mask.shape[1] :] * action_mask.float()
        return output

    @torch.no_grad()
    def async_run_method_batch(
        self,
        method_name,
        sequences,
        action_mask,
        attention_mask,
        mm_train_inputs_list=None,
        routed_experts=None,
        **kwargs,
    ):
        """Synchronous stand-in for RayActorGroup's batched model forward."""
        del mm_train_inputs_list, routed_experts
        if method_name != "forward" or kwargs:
            raise RuntimeError(f"Unsupported local model call: {method_name}, extras={list(kwargs)}")
        return [
            [
                self(sequence, mask, attention_mask=attention)["action_log_probs"].cpu()
                for sequence, mask, attention in zip(sequences, action_mask, attention_mask)
            ]
        ]


class TinyVllmEngine:
    """Duck-typed vLLM engine that uses the shared TinyPolicy on CPU."""

    def __init__(self, model, tokenizer, uniform_sequences):
        self.model = model
        self.tokenizer = tokenizer
        self.uniform_sequences = iter(uniform_sequences)

    async def generate(self, prompt_token_ids, sampling_params, multi_modal_data=None, session_id=None):
        del multi_modal_data, session_id
        uniforms = iter(next(self.uniform_sequences))
        context = list(prompt_token_ids)
        answer_ids = []
        generation_log_probs = []

        with torch.no_grad():
            for _ in range(sampling_params.max_tokens):
                uniform = next(uniforms)
                sequences = torch.tensor([context], dtype=torch.long)
                next_logits = self.model(sequences)["logits"][0, -1] / sampling_params.temperature
                next_log_probs = next_logits.log_softmax(dim=-1)
                cdf = next_log_probs.exp().cumsum(dim=-1)
                token_id = torch.searchsorted(cdf, torch.tensor(uniform)).clamp_max(cdf.numel() - 1).item()
                answer_ids.append(token_id)
                generation_log_probs.append({token_id: SimpleNamespace(logprob=next_log_probs[token_id].item())})
                context.append(token_id)

        generation = SimpleNamespace(
            token_ids=answer_ids,
            text="",
            finish_reason="length",
            logprobs=generation_log_probs,
            routed_experts=None,
        )
        return SimpleNamespace(outputs=[generation], prompt_routed_experts=None), 0

    @torch.no_grad()
    def mean_answer_log_probability(self, prompt_token_ids, answer):
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        sequences = torch.tensor([list(prompt_token_ids) + answer_ids], dtype=torch.long)
        first_action_step = len(prompt_token_ids) - 1
        action_log_probs = self.model(sequences)["log_probs"][0, first_action_step:]
        return action_log_probs.mean().item()


torch.manual_seed(7)
device = torch.device("cpu")
max_length = 256
correct_answer = r"\boxed{4}"
wrong_answer = r"\boxed{5}"

# These are real Molt config branches, reduced to one CPU rank and two rollouts.
args = SimpleNamespace(
    data=SimpleNamespace(
        input_key="prompt",
        label_key="reward_model",
        tools_key="tools",
        image_key="images",
        apply_chat_template=True,
    ),
    train=SimpleNamespace(
        dynamic_batch_enable=False,
        force_on_policy=False,
        colocate_fsdp_models=False,
    ),
    rollout=SimpleNamespace(batch_size=1, micro_batch_size=1, n_samples_per_prompt=2),
    actor=SimpleNamespace(num_nodes=1, num_gpus_per_node=1),
    fsdp=SimpleNamespace(cp_size=1, tp_size=1),
    algo=SimpleNamespace(
        dynamic_filtering_enable=True,
        dynamic_filtering_range=(0.01, 0.99),
        advantage=SimpleNamespace(
            estimator="reinforce_baseline",
            gamma=1.0,
            lam=1.0,
            no_whiten=False,
        ),
        kl=SimpleNamespace(init_coef=0.001, use_loss=True, estimator="k2"),
    ),
    reward=SimpleNamespace(clip_range=(-10.0, 10.0)),
)
strategy = SimpleNamespace(args=args)
tokenizer_revision = "c1899de289a04d12100db370d81485cdf75e47ca"
tokenizer_path = snapshot_download(
    "Qwen/Qwen3-0.6B",
    revision=tokenizer_revision,
    allow_patterns=["config.json", "tokenizer*", "merges.txt", "vocab.json"],
)
tokenizer = get_tokenizer(tokenizer_path, model=None, padding_side="left")

print("\n1. Real PromptDataset and the same StatefulDataLoader class as rl_trainer.prepare_datasets")
rows = Dataset.from_list(
    [
        {
            "datasource": "local_math",
            "prompt": [{"role": "user", "content": r"What is 2 + 2? Answer with \boxed{}."}],
            "reward_model": {"ground_truth": "4", "style": "rule"},
        }
    ]
)
runner = load_agent_runner(str(repo_root / "examples/python/agents/math.py"))
prompt_dataset = PromptDataset(rows, tokenizer, strategy, prerender=runner.PRERENDER_PROMPT)
prompt_dataloader = StatefulDataLoader(
    prompt_dataset,
    batch_size=1,
    pin_memory=False,
    shuffle=True,
    drop_last=True,
    collate_fn=prompt_dataset.collate_fn,
    num_workers=0,
)
loader_batch = next(iter(prompt_dataloader))
pprint(dict(zip(("datasources", "prompts", "labels", "images", "tools"), loader_batch)))

print("\n2. Real SamplesGenerator prompt-batch collector")
prompts, labels, images, tools, exhausted = _collect_prompt_batch(iter([loader_batch]), num_prompts=1)
pprint({"prompt": prompts[0], "label": labels[0], "exhausted": exhausted})

print("\n3. Tiny CPU model and fake vLLM transport; the model object is shared with training")
prompt_token_ids = tokenizer(prompts[0], add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
correct_answer_ids = tokenizer(correct_answer, add_special_tokens=False)["input_ids"]
wrong_answer_ids = tokenizer(wrong_answer, add_special_tokens=False)["input_ids"]
model = TinyPolicy(
    vocab_size=len(tokenizer),
    transition_sequences=[prompt_token_ids[-1:] + correct_answer_ids, prompt_token_ids[-1:] + wrong_answer_ids],
).to(device)
reference_model = copy.deepcopy(model).eval().requires_grad_(False)
engine = TinyVllmEngine(
    model,
    tokenizer,
    uniform_sequences=[
        [0.10, 0.30, 0.50, 0.25, 0.70],
        [0.20, 0.40, 0.60, 0.75, 0.80],
    ],
)
margin_before = engine.mean_answer_log_probability(
    prompt_token_ids, correct_answer
) - engine.mean_answer_log_probability(prompt_token_ids, wrong_answer)
pprint({name: tuple(parameter.shape) for name, parameter in model.named_parameters()})

print("\n4. Real StepEnvRunner.execute -> real MathEnv.step -> real Result -> real Trajectory")
sampling_params = SimpleNamespace(
    max_tokens=len(correct_answer_ids),
    min_tokens=1,
    logprobs=1,
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
)
trajectories = []
for rollout_index in range(args.rollout.n_samples_per_prompt):
    trajectory = asyncio.run(
        runner.execute(
            prompt=prompts[0],
            label=labels[0],
            sampling_params=sampling_params,
            max_length=max_length,
            hf_tokenizer=tokenizer,
            llm_engine=engine,
            images=images[0],
            tools=tools[0],
        )
    )
    # AgentRunnerActor.run_group normally adds these ids after Ray returns.
    trajectory.group_id = "local-prompt-0"
    trajectory.rollout_id = f"local-rollout-{rollout_index}"
    trajectories.append(trajectory)
    pprint(
        {
            "rollout_id": trajectory.rollout_id,
            "answer": tokenizer.decode(
                trajectory.observation_tokens[trajectory.action_ranges[0][0] :], skip_special_tokens=False
            ),
            "reward": trajectory.reward,
            "truncated": trajectory.truncated,
            "action_ranges": trajectory.action_ranges,
            "num_tokens": len(trajectory.observation_tokens),
        }
    )

print("\n5. Real SamplesGenerator group completeness, dynamic filter, and Trajectory -> Experience conversion")
samples_generator = SamplesGenerator(strategy, prompt_dataloader, None, tokenizer, agent_runners=[])
drop_counts = defaultdict(int)
rollout_samples = samples_generator._filter_group(
    trajectories,
    dynamic_filtering=args.algo.dynamic_filtering_enable,
    drop_counts=drop_counts,
    max_len=max_length,
    n_samples_per_prompt=args.rollout.n_samples_per_prompt,
)
assert not drop_counts
assert len({sample.rollout_ids[0] for sample in rollout_samples}) == args.rollout.n_samples_per_prompt
for sample in rollout_samples:
    pprint(
        {
            "sequences": tuple(sample.sequences.shape),
            "action_mask": tuple(sample.action_mask.shape),
            "rollout_log_probs": tuple(sample.rollout_log_probs.shape),
            "reward": sample.rewards.tolist(),
            "truncated": sample.truncated.tolist(),
        }
    )

print("\n6. Real RemoteExperienceMaker old-policy/reference forwards, KL fields, and grouped advantage")
experience_maker = RemoteExperienceMaker(
    actor_model_group=model,
    initial_model_group=reference_model,
    kl_controller=FixedKLController(args.algo.kl.init_coef),
    strategy=strategy,
    tokenizer=tokenizer,
)
experiences = experience_maker.build_experiences(rollout_samples)
for experience in experiences:
    pprint(
        {
            "index": experience.index,
            "reward": experience.rewards.tolist(),
            "masked_advantage": experience.advantages[experience.action_mask].unique().tolist(),
            "old_action_log_probs": tuple(experience.action_log_probs.shape),
            "base_action_log_probs": tuple(experience.base_action_log_probs.shape),
        }
    )

print("\n7. Real single-rank balancing -> NaiveReplayBuffer -> training DataLoader")
balanced_experiences = balance_experiences(experiences, args)
replay_buffer = NaiveReplayBuffer(sample_batch_size=1, cpu_offload=True, dynamic_batch=False)
for experience in balanced_experiences:
    replay_buffer.append(experience)
train_dataloader = DataLoader(
    replay_buffer,
    batch_size=replay_buffer.sample_batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=False,
    collate_fn=replay_buffer.collate_fn,
)
microbatches = list(train_dataloader)
batch_num_tokens = torch.stack([experience.action_mask.sum() for experience in microbatches]).sum()
pprint(
    {"buffer_items": len(replay_buffer), "microbatches": len(microbatches), "action_tokens": batch_num_tokens.item()}
)

print("\n8. Tiny Actor forward -> real PolicyLoss + KL loss -> autograd -> AdamW/scheduler")
# The recipe uses 1e-6; this one-step tutorial enlarges it so the update is visible.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
loss_fn = PolicyLoss(
    clip_eps_low=0.2,
    clip_eps_high=0.27,
    dual_clip=10.0,
    is_correction_level="geo",
    is_correction_threshold=[0.99, 1.01],
)
optimizer.zero_grad(set_to_none=True)
loss_rows = []
for experience in microbatches:
    experience.to_device(device)
    model_output = model(
        experience.sequences,
        experience.action_mask,
        attention_mask=experience.attention_mask,
    )
    action_log_probs = model_output["action_log_probs"]
    old_action_log_probs = experience.action_log_probs
    if old_action_log_probs is None:
        old_action_log_probs = action_log_probs.detach()
    actor_loss, reported_loss, clip_ratio, policy_kl, vllm_kl, filter_ratio = loss_fn(
        action_log_probs,
        old_action_log_probs,
        experience.advantages,
        action_mask=experience.action_mask,
        rollout_log_probs=experience.rollout_log_probs,
        dp_size=1,
        batch_num_tokens=batch_num_tokens,
    )
    approx_kl = compute_approx_kl(
        action_log_probs,
        experience.base_action_log_probs,
        kl_estimator=args.algo.kl.estimator,
    )
    kl_loss = agg_loss(
        approx_kl,
        experience.action_mask,
        "token-mean",
        dp_size=1,
        batch_num_tokens=batch_num_tokens,
    )
    total_loss = actor_loss + kl_loss * args.algo.kl.init_coef
    total_loss.backward()
    loss_rows.append(
        {
            "policy_loss_contribution": actor_loss.item(),
            "kl_loss_contribution": (kl_loss * args.algo.kl.init_coef).item(),
            "total_loss_contribution": total_loss.item(),
            "reported_loss": reported_loss.item(),
            "clip_ratio": clip_ratio.item(),
            "policy_kl": policy_kl.item(),
            "vllm_kl": None if vllm_kl is None else vllm_kl.item(),
            "filter_ratio": None if filter_ratio is None else filter_ratio.item(),
        }
    )
pprint(loss_rows)
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
pprint(
    {
        "total_grad_norm_before_clip": grad_norm.item(),
        "learning_rate": scheduler.get_last_lr()[0],
        "parameter_gradients": {
            name: {"shape": tuple(parameter.grad.shape), "norm": parameter.grad.norm().item()}
            for name, parameter in model.named_parameters()
        },
    }
)
optimizer.step()
scheduler.step()
optimizer.zero_grad(set_to_none=True)

print("\n9. Shared model replaces FSDP -> vLLM weight broadcast in this one-process tutorial")
margin_after = engine.mean_answer_log_probability(
    prompt_token_ids, correct_answer
) - engine.mean_answer_log_probability(prompt_token_ids, wrong_answer)
pprint({"correct_minus_wrong_logprob_before": margin_before, "correct_minus_wrong_logprob_after": margin_after})

assert [trajectory.reward for trajectory in trajectories] == [1.0, 0.0]
assert all(trajectory.truncated for trajectory in trajectories)
assert all(sample.sequences.shape[1] == sample.action_mask.shape[1] + 1 for sample in rollout_samples)
assert margin_after > margin_before
assert sys.modules["ray"].__cpu_tutorial_stub__
assert sys.modules["vllm"].__cpu_tutorial_stub__
print("\nDONE: Ray scheduling and the GPU model/trainer stack were replaced; the RL records and math stayed real.\n")
