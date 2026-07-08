# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert DAPO-Math-17k to Molt schema.

Source: BytedTsinghua-SIA/DAPO-Math-17k (train; 17K-ish unique problems
across multiple configs, the loaded `default` config materializes ~1.79M
rows). Eval: BytedTsinghua-SIA/AIME-2024 (30 problems).

Both sources already ship `prompt` (chat-style list) and `reward_model`
(`{ground_truth, style}`), so this script only deduplicates the train
split, optionally subsamples, and writes to disk in load_from_disk format.
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

    eval_ds = load_dataset(args.eval_source, split="train")
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
