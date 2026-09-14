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

"""Convert DAPO-Math-17k to Molt schema.

Source: BytedTsinghua-SIA/DAPO-Math-17k (train; 17K-ish unique problems
across multiple configs, the loaded `default` config materializes ~1.79M
rows). Eval: BytedTsinghua-SIA/AIME-2024 (30 problems).

Both sources already ship `prompt` (chat-style list) and `reward_model`
(`{ground_truth, style}`), so this script only deduplicates the train
split, optionally subsamples, and writes to disk in load_from_disk format.

Any dataset with that schema converts the same way, e.g. the FP16 sanity test
(sail/Sanity-Test-R1D-1.5B: 1,460 MATH problems, AIME 2024 + 2025 as `test`):
  python prepare_dapo.py --train-source sail/Sanity-Test-R1D-1.5B \
    --eval-source sail/Sanity-Test-R1D-1.5B --eval-split test --out-dir .tmp/prep_sanity_r1d
"""

import argparse
from pathlib import Path

from datasets import load_dataset


def _format_row(example):
    return {
        "datasource": example.get("data_source", "dapo_math_17k"),
        "prompt": example["prompt"],
        "reward_model": example["reward_model"],
        # Provide a response field for SFT compatibility — boxed ground-truth.
        "response": [
            {
                "role": "assistant",
                "content": "\\boxed{" + str(example["reward_model"].get("ground_truth", "")) + "}",
            }
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-source",
        default="BytedTsinghua-SIA/DAPO-Math-17k",
        help="HF dataset id for training split.",
    )
    parser.add_argument(
        "--eval-source",
        default="BytedTsinghua-SIA/AIME-2024",
        help="HF dataset id for eval split.",
    )
    parser.add_argument("--config", default="default", help="HF dataset config name (train).")
    parser.add_argument("--eval-split", default="train", help="Split of --eval-source to use.")
    parser.add_argument("--max-train", type=int, default=20000, help="Cap on train rows after dedup.")
    parser.add_argument("--max-eval", type=int, default=None, help="Cap on eval rows.")
    parser.add_argument("--out-dir", type=Path, default=Path(".tmp/dapo_math_17k"))
    parser.add_argument("--num-proc", type=int, default=4)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = load_dataset(args.train_source, args.config, split="train")
    # Dedup on the prompt's user content — many DAPO rows are duplicates across
    # configs/sources. Keep the first occurrence.
    seen: set = set()
    keep_indices = []
    for idx, row in enumerate(train_ds):
        try:
            key = row["prompt"][0]["content"]
        except (KeyError, IndexError, TypeError):
            key = str(row.get("prompt"))
        if key in seen:
            continue
        seen.add(key)
        keep_indices.append(idx)
        if args.max_train is not None and len(keep_indices) >= args.max_train:
            break
    train_ds = train_ds.select(keep_indices)
    print(f"deduped to {len(train_ds)} unique train rows")

    eval_ds = load_dataset(args.eval_source, split=args.eval_split)
    if args.max_eval is not None:
        eval_ds = eval_ds.select(range(min(args.max_eval, len(eval_ds))))

    drop_train = [c for c in train_ds.column_names if c not in ("__index_level_0__",)]
    drop_eval = [c for c in eval_ds.column_names if c not in ("__index_level_0__",)]
    train_out = train_ds.map(_format_row, num_proc=args.num_proc, remove_columns=drop_train)
    eval_out = eval_ds.map(_format_row, num_proc=args.num_proc, remove_columns=drop_eval)

    train_out.save_to_disk(args.out_dir / "train")
    eval_out.save_to_disk(args.out_dir / "eval")
    print(f"wrote {len(train_out)} train + {len(eval_out)} eval rows to {args.out_dir}")


if __name__ == "__main__":
    main()
