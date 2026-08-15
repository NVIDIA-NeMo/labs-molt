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

"""Prepare AlfWorld data for Molt: RL game files and SFT expert demos.

Both subcommands need ALFWORLD_DATA on the env (set by `alfworld-download`):

  rl   — walk $ALFWORLD_DATA/<glob> (default json_*/train/**/*.tw-pddl), pair
          each game with its sibling traj_data.json task description, and emit
          one jsonl row per game:
              {"prompt": "<task + usage>", "label": "<abs .tw-pddl path>"}
          `prompt` is a bare string (the RL script's --data.apply_chat_template
          wraps it as the user turn); `label` carries the game path the env binds
          (surfaced to the agent via --data.label_key). A fixed seed splits
          train/eval in-distribution.

  sft  — read tw_alfred_seq2seq_{train,eval}_task*_hc.json (from
          `alfworld-download --extra`) and turn each expert episode into a
          multi-turn chat: user(task + obs0) / assistant(action0) / user(obs1) /
          assistant(action1) / ... The demo `obs` field is alfworld's pooled
          seq2seq format ("obs0 [SEP] obs_i [SEP] action_{i-1}"); the current
          turn's obs is the segment just before the trailing embedded action.
          Episodes longer than --max-steps are dropped (won't fit the SFT budget).

Output is jsonl (load_dataset-readable via data_files=).

  python examples/python/utils/prepare_alfworld.py rl   --max-train 50   # self-check
  python examples/python/utils/prepare_alfworld.py sft  --max-train 50
"""

import argparse
import json
import os
import random
from pathlib import Path

# One-line usage hint appended to the task instruction. Surfaces the action
# format and nudges a `look` first — the RL prompt carries no initial room text
# (reset returns the prompt unchanged), so the model must look to see the room.
# SFT prep prepends the same hint to its first user turn so the two stay aligned.
_TASK_SUFFIX = (
    " You are playing a text household game. On each turn reply with exactly ONE "
    "short command (e.g. 'go to drawer 1', 'take cd 1', 'open fridge 1', 'use "
    "lamp 1', 'look'). Start by looking around to see what is in the room."
)

# RL variant: same task framing but a structured <think>/<action> protocol, so the
# command is extracted exactly instead of parsed out of free-form text (the model often
# wraps, prefixes, or chains a bare command). SFT keeps _TASK_SUFFIX so its bare-command
# targets stay consistent with the expert demos.
_RL_TASK_SUFFIX = (
    " You are an expert agent in a text household game. On each turn you are shown the "
    "current observation and a list of admissible actions. First reason step-by-step "
    "inside <think> </think> tags, then choose exactly ONE admissible action and put it "
    "inside <action> </action> tags."
)


def _data_dir() -> Path:
    data_dir = os.environ.get("ALFWORLD_DATA")
    assert data_dir, "Set ALFWORLD_DATA (run alfworld-download) before preparing AlfWorld data."
    return Path(data_dir).expanduser().resolve()


def _task_from_traj(gamefile: Path) -> str:
    traj = json.loads((gamefile.parent / "traj_data.json").read_text(encoding="utf-8"))
    anns = traj.get("turk_annotations", {}).get("anns") or []
    if anns and anns[0].get("task_desc"):
        return anns[0]["task_desc"].strip()
    return ""


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cmd_rl(args):
    data_dir = _data_dir()
    games = sorted(p.resolve() for p in data_dir.glob(args.glob))
    assert games, f"No .tw-pddl under {data_dir}/{args.glob} — run alfworld-download first."

    rows = []
    for gamefile in games:
        task = _task_from_traj(gamefile)
        if task:
            rows.append({"prompt": task + _RL_TASK_SUFFIX, "label": str(gamefile)})

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    n_eval = int(len(rows) * args.eval_frac)
    eval_rows, train_rows = rows[:n_eval], rows[n_eval:]
    if args.max_train:
        train_rows = train_rows[: args.max_train]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "eval.jsonl", eval_rows)
    print(f"alfworld rl: {len(train_rows)} train + {len(eval_rows)} eval games -> {out_dir}")


def _clean_obs(obs: str) -> str:
    # Demo obs are pooled ("obs0 [SEP] obs_i [SEP] action_{i-1}"); the current
    # turn's obs is the segment just before the trailing embedded action. A clean
    # first-turn obs has no [SEP] at all.
    parts = obs.rsplit(" [SEP] ", 2)
    return (parts[-2] if len(parts) >= 2 else parts[0]).strip()


def cmd_sft(args):
    data_dir = _data_dir()
    demos = sorted(data_dir.rglob("tw_alfred_seq2seq_*_hc.json"))
    assert demos, f"No tw_alfred_seq2seq_*_hc.json under {data_dir} — run alfworld-download --extra."

    splits = {"train": [], "eval": []}
    for demo_file in demos:
        split = "train" if "_train_" in demo_file.name else "eval" if "_eval_" in demo_file.name else None
        if split is None:
            continue
        for ep in json.loads(demo_file.read_text(encoding="utf-8")).get("data", []):
            steps = ep.get("steps") or []
            task = (ep.get("task") or "").strip()
            if not task or not (1 <= len(steps) <= args.max_steps):
                continue
            # user=obs[i], assistant=action[i], paired in order — an off-by-one
            # here silently trains on the wrong (obs, action) and zeroes win rate.
            messages = [{"role": "user", "content": f"{task}{_TASK_SUFFIX}\n{_clean_obs(steps[0]['obs'])}"}]
            for i, step in enumerate(steps):
                messages.append({"role": "assistant", "content": str(step["action"]).strip()})
                if i + 1 < len(steps):
                    messages.append({"role": "user", "content": _clean_obs(steps[i + 1]["obs"])})
            splits[split].append({"prompt": messages[:-1], "response": [messages[-1]]})

    if args.max_train:
        splits["train"] = splits["train"][: args.max_train]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", splits["train"])
    _write_jsonl(out_dir / "eval.jsonl", splits["eval"])
    print(f"alfworld sft: {len(splits['train'])} train + {len(splits['eval'])} eval episodes -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare AlfWorld data for Molt (RL game files / SFT expert demos).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    rl = sub.add_parser("rl", help="Emit game-file rows for RL (prompt + .tw-pddl label).")
    rl.add_argument("--out-dir", default=".tmp/alfworld_rl")
    rl.add_argument("--glob", default="json_*/train/**/*.tw-pddl", help="Glob under ALFWORLD_DATA for game files.")
    rl.add_argument("--eval-frac", type=float, default=0.05)
    rl.add_argument("--seed", type=int, default=0)
    rl.add_argument("--max-train", type=int, default=None)
    rl.set_defaults(func=cmd_rl)

    sft = sub.add_parser("sft", help="Emit multi-turn chat episodes from expert demos.")
    sft.add_argument("--out-dir", default=".tmp/alfworld_sft")
    sft.add_argument("--max-steps", type=int, default=40, help="Drop episodes longer than this.")
    sft.add_argument("--max-train", type=int, default=None)
    sft.set_defaults(func=cmd_sft)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
