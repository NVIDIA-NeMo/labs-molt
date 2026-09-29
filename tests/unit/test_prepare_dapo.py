# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from pathlib import Path

from datasets import Dataset, load_from_disk


def test_prepare_dapo_deduplicates_eval_prompts(monkeypatch, tmp_path):
    script = Path(__file__).parents[2] / "examples" / "python" / "utils" / "prepare_dapo.py"
    spec = importlib.util.spec_from_file_location("prepare_dapo", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows = [
        {
            "data_source": "math",
            "prompt": [{"role": "user", "content": prompt}],
            "reward_model": {"ground_truth": answer, "style": "rule"},
        }
        for prompt, answer in [("one", "1"), ("two", "2"), ("one", "1")]
    ]
    monkeypatch.setattr(module, "load_dataset", lambda *args, **kwargs: Dataset.from_list(rows))
    monkeypatch.setattr(
        sys,
        "argv",
        [str(script), "--out-dir", str(tmp_path), "--num-proc", "0"],
    )

    module.main()

    eval_ds = load_from_disk(tmp_path / "eval")
    assert len(eval_ds) == 2
    assert [row["prompt"][0]["content"] for row in eval_ds] == ["one", "two"]
