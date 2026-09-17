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

"""Prepare the text-math set the Qwen3-4B SFT/RL recipes read from .tmp/proRL_text_rl.

Train: agentica-org/DeepScaleR-Preview-Dataset (~40K problems — the math pool ProRL
trains on, https://arxiv.org/abs/2505.24864). Eval: HuggingFaceH4/aime_2024 (30
problems). Both ship `problem` / `answer` / `solution`, so one formatter serves both.

Output schema (load_from_disk-compatible):
    datasource: str
    prompt: list[{role: "user", content: str}]
    reward_model: {ground_truth: str, style: "rule"}       # RL label (--data.label_key reward_model)
    response: list[{role: "assistant", content: str}]      # SFT target: solution ending in \\boxed{}
"""

import argparse
from pathlib import Path

from datasets import load_dataset

_INSTRUCTION = "\n\nLet's think step by step and output the final answer within \\boxed{}."


def _format_row(example, datasource):
    answer = str(example["answer"]).strip()
    solution = str(example.get("solution") or "").strip()
    # SFT trains on the reference solution; make sure it ends with the boxed answer
    # the grader looks for (many DeepScaleR solutions stop at the derivation).
    if "\\boxed" not in solution:
        solution = "\n\n".join(s for s in (solution, f"\\boxed{{{answer}}}") if s)
    return {
        "datasource": datasource,
        "prompt": [{"role": "user", "content": str(example["problem"]).strip() + _INSTRUCTION}],
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "response": [{"role": "assistant", "content": solution}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--eval-source", default="HuggingFaceH4/aime_2024")
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap on train rows.")
    parser.add_argument("--max-eval", type=int, default=None, help="Optional cap on eval rows.")
    parser.add_argument("--out-dir", type=Path, default=Path(".tmp/proRL_text_rl"))
    parser.add_argument("--num-proc", type=int, default=8)
    args = parser.parse_args()

    for split, source, cap in (("train", args.train_source, args.max_train), ("eval", args.eval_source, args.max_eval)):
        ds = load_dataset(source, split="train")
        if cap is not None:
            ds = ds.select(range(min(cap, len(ds))))
        out = ds.map(
            _format_row,
            fn_kwargs={"datasource": source.split("/")[-1]},
            num_proc=min(args.num_proc, len(ds)),
            remove_columns=ds.column_names,
        )
        out.save_to_disk(args.out_dir / split)
        print(f"wrote {len(out)} {split} rows to {args.out_dir / split}")


if __name__ == "__main__":
    main()
