# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert VeraIsHere/geo3k_imgurl_processed to Molt schema.

Source: https://huggingface.co/datasets/VeraIsHere/geo3k_imgurl_processed
Geometric reasoning with a visible image and a numeric / boxed final answer.
Used for both VLM SFT (single-turn QA) and VLM multi-turn RL (with the math
tool-call env).

Output schema (load_from_disk-compatible):
    datasource: str
    prompt: list[{role: "user", content: str}]   # chat-style, `<image>` literal
    reward_model: {ground_truth: str, style: "rule"}
    response: list[{role: "assistant", content: str}]   # SFT target
    images: list[PIL.Image]
"""

import argparse
import io
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from datasets import load_dataset
from PIL import Image

# OpenAI-style function tool schema for the `python_executor` sandbox.
# The tool runs an arbitrary Python snippet (with math / sympy / numpy
# available) and returns stdout.
# The chat template renders this into a system-side preamble so the model
# emits `<tool_call>{...}</tool_call>` natively (no manual prompt engineering).
GEO3K_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_executor",
        "description": (
            "Run a Python snippet in a sandbox for math/geometry calculations. "
            "Returns captured stdout (capped). math is preloaded; import sympy/numpy "
            "yourself if needed. Use print() to read intermediate values. "
            "Call as many times as needed to verify reasoning steps."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source. Include print() for any value you want to read.",
                },
            },
            "required": ["code"],
        },
    },
}

# Short instruction appended after the bare problem statement — the tool
# schema above is what teaches the model the tool_call format. Pick the answer
# wrapper that matches the model's pretraining distribution:
#   * `boxed`  — `\boxed{...}` (Qwen / DeepSeek-Math convention)
#   * `answer` — `<answer>...</answer>` (Nemotron Omni convention)
# The grader accepts both, so swapping only changes what the model is asked to
# emit; both wrappers grade identically downstream.
_INSTRUCTION_HEAD = (
    " Reason step by step inside <think>...</think>, calling python_executor as "
    "needed to verify intermediate computations. Provide your final answer in "
)
ANSWER_WRAPPERS = {
    "boxed": "\\boxed{}.",
    "answer": "<answer>...</answer>.",
}
RESPONSE_WRAPPERS = {
    "boxed": "\\boxed{{{}}}",
    "answer": "<answer>{}</answer>",
}


def _extract_answer(example: dict[str, Any]) -> str:
    """Best-effort extraction of the numeric ground truth across schema variants."""
    for key in ("answer", "label", "solution", "ground_truth"):
        value = example.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        # Strip a boxed wrapper if present.
        m = re.search(r"\\boxed\{([^{}]*)\}", text)
        if m:
            return m.group(1).strip()
        return text
    return ""


def _load_image(example: dict[str, Any]) -> Image.Image | None:
    """Return a PIL.Image. Source rows ship a PIL object, raw bytes, or a URL string.

    `VeraIsHere/geo3k_imgurl_processed` stores images as remote URLs; we fetch
    each one lazily during `datasets.map`. For local URLs (file://) urlopen
    handles them transparently.
    """
    # Check both `image` and `images` keys; prefer non-None.
    image = example.get("image")
    if image is None:
        image = example.get("images")
    if isinstance(image, list):
        image = image[0] if image else None
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict) and "bytes" in image:
        return Image.open(io.BytesIO(image["bytes"]))
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    if isinstance(image, str):
        # Image URL — fetch and decode. Used by VeraIsHere/geo3k_imgurl_processed.
        with urlopen(image, timeout=30) as response:
            data = response.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image)!r}")


_PROMPT_BOILERPLATE_RE = re.compile(
    r"Solve the following math problem.*?(?=<image>|$)|"
    r"You are a math/geometry expert.*?(?=<image>|$)|"
    r"Follow this protocol:.*?(?=<image>|$)|"
    r"Reason step by step.*?$|"
    r"Answer:.*?\\boxed\{.*?\}",
    re.DOTALL,
)


def _strip_verbose_instructions(text: str) -> str:
    """Drop the source dataset's hand-written tool_call instructions.

    The chat template injects an OpenAI-style tool schema preamble via
    `tools=`, so the verbose hand-rolled protocol in the source `problem`
    field is redundant and bloats the prompt token budget. Keep only the
    bare question + the <image> tag.
    """
    cleaned = _PROMPT_BOILERPLATE_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _format_row(example: dict[str, Any], answer_format: str = "boxed") -> dict[str, Any]:
    problem = str(example.get("problem") or example.get("question") or "").strip()
    answer = _extract_answer(example)
    user_text = _strip_verbose_instructions(problem) + _INSTRUCTION_HEAD + ANSWER_WRAPPERS[answer_format]
    # Defensive: enforce exactly one <image> placeholder so the multimodal
    # processor's image_grid_thw lookup stays aligned with `images`.
    placeholder_count = user_text.count("<image>")
    if placeholder_count == 0:
        user_text = "<image>\n" + user_text
    elif placeholder_count > 1:
        first = user_text.find("<image>")
        head = user_text[: first + len("<image>")]
        tail = user_text[first + len("<image>") :].replace("<image>", "")
        user_text = head + tail
    image = _load_image(example)
    return {
        "datasource": "geo3k_imgurl_processed",
        "prompt": [{"role": "user", "content": user_text}],
        "tools": [GEO3K_TOOL_SCHEMA],
        "reward_model": {"ground_truth": answer, "style": "rule"},
        "response": [{"role": "assistant", "content": RESPONSE_WRAPPERS[answer_format].format(answer)}],
        "images": [image] if image is not None else [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="VeraIsHere/geo3k_imgurl_processed",
        help="HF dataset id.",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Source train split name.",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        help="Source eval split name (fallback to last 5%% of train if missing).",
    )
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap on train rows.")
    parser.add_argument("--max-eval", type=int, default=512, help="Cap on eval rows.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".tmp/geo3k"),
        help="Output dir; writes train/ and eval/ via save_to_disk.",
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--answer-format",
        choices=sorted(ANSWER_WRAPPERS),
        default="boxed",
        help="Wrapper the model is asked to emit: 'boxed' (Qwen / DeepSeek-Math) "
        "or 'answer' (<answer>...</answer>, Nemotron Omni). Grader accepts both.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.source)
    train_ds = ds[args.train_split]
    if args.eval_split in ds:
        eval_ds = ds[args.eval_split]
    else:
        n_eval = max(1, int(0.05 * len(train_ds)))
        eval_ds = train_ds.select(range(len(train_ds) - n_eval, len(train_ds)))
        train_ds = train_ds.select(range(len(train_ds) - n_eval))

    if args.max_train is not None:
        train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))
    eval_ds = eval_ds.select(range(min(args.max_eval, len(eval_ds))))

    columns_to_drop = [c for c in train_ds.column_names if c not in ("__index_level_0__",)]
    fmt_kwargs = {"answer_format": args.answer_format}
    train_out = train_ds.map(_format_row, fn_kwargs=fmt_kwargs, num_proc=args.num_proc, remove_columns=columns_to_drop)
    eval_out = eval_ds.map(_format_row, fn_kwargs=fmt_kwargs, num_proc=args.num_proc, remove_columns=columns_to_drop)

    train_out.save_to_disk(args.out_dir / "train")
    eval_out.save_to_disk(args.out_dir / "eval")
    print(f"wrote {len(train_out)} train + {len(eval_out)} eval rows to {args.out_dir}")


if __name__ == "__main__":
    main()
