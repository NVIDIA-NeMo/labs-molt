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

"""Refit a vLLM engine with its own checkpoint through molt's packed broadcast and require
unchanged greedy logprobs.

A refit that skips vLLM's post-load weight processing leaves MoE weights in checkpoint layout
under a kernel that expects its own (FlashInfer TRT-LLM block layout, CUTLASS swapped gate/up
halves): the engine then generates garbage although every weight "loaded". Two GPUs: the
engine on GPU 0, the broadcasting trainer stand-in on GPU 1.

    python3 tests/e2e/check_vllm_refit_layout.py --model <hf dir> --moe-backend flashinfer_trtllm
    python3 tests/e2e/check_vllm_refit_layout.py --model <hf dir> --moe-backend flashinfer_trtllm --no-bracket  # expected FAIL
"""

import argparse
import glob
import socket
import sys
import threading

import torch
from safetensors import safe_open
from vllm import LLM, SamplingParams

from molt.utils.distributed_util import stateless_init_process_group

PROMPTS = ["The capital of France is", "def fibonacci(n):", "Water boils at", "In 1969, humans first"]


def greedy_logprobs(llm):
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0, max_tokens=32, logprobs=0))
    return [[(t, o.outputs[0].logprobs[i][t].logprob) for i, t in enumerate(o.outputs[0].token_ids)] for o in outs]


def rpc_while_broadcasting(llm, method, args, broadcast):
    """The worker's RPC blocks on the NCCL broadcast, so the trainer side runs it concurrently."""
    thread = threading.Thread(target=llm.collective_rpc, args=(method,), kwargs={"args": args})
    thread.start()
    broadcast()
    thread.join()


def refit_from_checkpoint(llm, group, model_dir, bracket):
    if bracket:
        llm.collective_rpc("begin_weight_update")
    metas, tensors = [], []

    def flush():
        flat = torch.cat([t.view(torch.uint8).view(-1) for t in tensors])
        broadcast = lambda: group.broadcast(flat, src=0, stream=torch.cuda.current_stream())  # noqa: E731
        rpc_while_broadcasting(llm, "update_weights_packed", (list(metas),), broadcast)
        metas.clear()
        tensors.clear()

    for path in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with safe_open(path, "pt", device="cpu") as f:
            for name in f.keys():
                tensor = f.get_tensor(name).cuda()
                if tensors and sum(t.nbytes for t in tensors) + tensor.nbytes >= 512 << 20:
                    flush()
                metas.append((name, tensor.dtype, tuple(tensor.shape)))
                tensors.append(tensor)
    flush()
    if bracket:
        llm.collective_rpc("end_weight_update")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--moe-backend", default=None)
    parser.add_argument("--no-bracket", action="store_true", help="skip begin/end_weight_update (the old refit)")
    args = parser.parse_args()

    torch.cuda.set_device(1)
    llm = LLM(
        args.model,
        worker_extension_cls="molt.trainer.vllm.vllm_worker_wrap.WorkerWrap",
        gpu_memory_utilization=0.5,
        max_model_len=1024,
        enable_prefix_caching=False,
        seed=0,
        **({"moe_backend": args.moe_backend} if args.moe_backend else {}),
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    group = None

    def join_group():
        nonlocal group
        group = stateless_init_process_group("127.0.0.1", port, 0, 2, torch.device("cuda:1"))

    rpc_while_broadcasting(llm, "init_process_group", ("127.0.0.1", port, 1, 2, "refit", "nccl"), join_group)

    before = greedy_logprobs(llm)
    refit_from_checkpoint(llm, group, args.model, bracket=not args.no_bracket)
    after = greedy_logprobs(llm)

    same_tokens = [t for t, _ in sum(before, [])] == [t for t, _ in sum(after, [])]
    max_delta = (
        max(abs(a - b) for (_, a), (_, b) in zip(sum(before, []), sum(after, []))) if same_tokens else float("inf")
    )
    ok = same_tokens and max_delta < 0.05
    print(
        f"{'PASS' if ok else 'FAIL'} moe_backend={args.moe_backend} bracket={not args.no_bracket} "
        f"same_tokens={same_tokens} max|dlogprob|={max_delta:.3g}",
        flush=True,
    )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
