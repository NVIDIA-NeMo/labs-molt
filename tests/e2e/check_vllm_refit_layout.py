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

"""Refit a vLLM engine with its own checkpoint through molt's packed broadcast; teacher-forced log-probs of a
fixed set of sequences must not change beyond the kernel's own run-to-run noise.

Without begin/end_weight_update (--no-bracket) FlashInfer MoE backends keep the checkpoint layout under a kernel
that expects its own and generate garbage, although every weight "loaded". Two GPUs: engine on 0, trainer on 1.

    python3 tests/e2e/check_vllm_refit_layout.py --model <hf dir> --moe-backend flashinfer_trtllm [--no-bracket]
"""

import argparse
import glob
import os
import socket
import sys
import threading

# The packed-broadcast metas carry torch dtypes, which vLLM's default RPC serializer refuses (molt's
# engine actor sets the same).
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

import torch  # noqa: E402
from safetensors import safe_open
from vllm import LLM, SamplingParams

from molt.utils.distributed_util import stateless_init_process_group

PROMPTS = ["The capital of France is", "def fibonacci(n):", "Water boils at", "In 1969, humans first"]


def score(llm, sequences):
    """Teacher-forced log-prob of every response token (greedy re-generation would flip at near-ties on
    non-deterministic kernels and hide the signal behind the cascade)."""
    params = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
    outs = llm.generate([{"prompt_token_ids": ids} for ids, _ in sequences], params)
    return [
        o.prompt_logprobs[i][t].logprob
        for o, (ids, n_prompt) in zip(outs, sequences)
        for i, t in enumerate(ids)
        if i >= n_prompt
    ]


def rpc_during(llm, method, args, trainer_side):
    """The worker's RPC blocks on the collective, so the trainer side runs on a helper thread; the
    engine client's sockets are not thread-safe, so the RPC itself stays on the main thread."""

    def on_trainer_gpu():
        torch.cuda.set_device(1)  # the CUDA device is per thread
        trainer_side()

    thread = threading.Thread(target=on_trainer_gpu)
    thread.start()
    llm.collective_rpc(method, args=args)
    thread.join()


def refit_from_checkpoint(llm, group, model_dir, bracket):
    if bracket:
        llm.collective_rpc("begin_weight_update")
    pending = []

    def flush():
        flat = torch.cat([t.view(torch.uint8).view(-1) for _, t in pending])
        metas = [(name, t.dtype, tuple(t.shape)) for name, t in pending]
        rpc_during(
            llm,
            "update_weights_packed",
            (metas,),
            lambda: group.broadcast(flat, src=0, stream=torch.cuda.current_stream()),
        )
        pending.clear()

    for path in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(path, "pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name).cuda()
                if pending and sum(t.nbytes for _, t in pending) + tensor.nbytes >= 512 << 20:
                    flush()
                pending.append((name, tensor))
    flush()
    if bracket:
        llm.collective_rpc("end_weight_update")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--moe-backend", default=None)
    parser.add_argument("--no-bracket", action="store_true", help="the old refit: plain load_weights flushes")
    args = parser.parse_args()

    torch.cuda.set_device(1)
    engine_kwargs = {"moe_backend": args.moe_backend} if args.moe_backend else {}
    llm = LLM(
        args.model,
        worker_extension_cls="molt.trainer.vllm.vllm_worker_wrap.WorkerWrap",
        gpu_memory_utilization=0.5,
        max_model_len=1024,
        enable_prefix_caching=False,
        seed=0,
        **engine_kwargs,
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    group = []
    rpc_during(
        llm,
        "init_process_group",
        ("127.0.0.1", port, 1, 2, "refit", "nccl"),
        lambda: group.append(stateless_init_process_group("127.0.0.1", port, 0, 2, torch.device("cuda:1"))),
    )

    generated = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=32))
    sequences = [(list(o.prompt_token_ids) + list(o.outputs[0].token_ids), len(o.prompt_token_ids)) for o in generated]
    before, again = score(llm, sequences), score(llm, sequences)
    noise = max(abs(a - b) for a, b in zip(before, again))  # the kernel's run-to-run floor with unchanged weights
    refit_from_checkpoint(llm, group[0], args.model, bracket=not args.no_bracket)
    after = score(llm, sequences)

    delta = max(abs(a - b) for a, b in zip(before, after))
    ok = delta <= max(0.1, 10 * noise)
    print(
        f"{'PASS' if ok else 'FAIL'} moe_backend={args.moe_backend} bracket={not args.no_bracket} "
        f"max|dlogprob|={delta:.3g} noise_floor={noise:.3g}"
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
