# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Write the reference pages of the docs site from the source code.

    python docs/guide/build_reference.py          # rewrite pages/cli-rl.md, cli-sft.md, api-agents.md
    python docs/guide/build_reference.py --check  # fail if they are stale or a page names an unknown flag

Needs only the standard library: the CLI parsers are built without importing the trainers.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAGES = Path(__file__).parent / "pages"
NOTE = "> Generated from `{}` by `docs/guide/build_reference.py`. Edit the source, then rerun the script.\n"
sys.path.insert(0, str(REPO))  # the parser blocks import molt.cli.common_args

pages = {}  # page name -> markdown
flags = set()  # every option string either CLI accepts

# ---- CLI flags: run each entry point's ``__main__`` block up to ``parse_args`` and read the parser.
for page, module in [("cli-rl", "molt.cli.train_rl_ray"), ("cli-sft", "molt.cli.train_sft")]:
    src = module.replace(".", "/") + ".py"
    tree = ast.parse((REPO / src).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.If) and "__main__" in ast.unparse(n))
    setup = main.body[: next(i for i, s in enumerate(main.body) if "parse_args" in ast.unparse(s))]
    scope = {"argparse": argparse}
    exec(compile(ast.Module(setup, []), src, "exec"), scope)

    groups = {}  # section heading -> table rows, in declaration order
    for action in scope["parser"]._actions[1:]:  # [0] is --help
        flags.update(action.option_strings)
        default = "" if action.default is None else f"`{action.default!r}`".replace("'", '"')
        default = re.sub(r"\d{4}T\d\d:\d\d", "<MMDD>T<HH:MM>", default)  # the wandb run name embeds the start time
        text = " ".join((action.help or "").replace("%(default)s", str(action.default)).split())
        # Help text is prose, not Markdown: outside its `code` spans, escape what Markdown would eat.
        text = "`".join(p if i % 2 else re.sub(r"([*~|<>])", r"\\\1", p) for i, p in enumerate(text.split("`")))
        if action.choices and not all(str(c) in text for c in action.choices):
            text += " Choices: " + ", ".join(f"`{c}`" for c in action.choices) + "."
        names = " / ".join(f"`{o}`" for o in action.option_strings)
        name = action.option_strings[0]
        heading = f"`{name.split('.')[0]}.*`" if "." in name else "Other flags"
        groups.setdefault(heading, []).append(f"| {names} | {default} | {text} |")

    lines = [f"# `python -m {module}`", "", NOTE.format(src)]
    for heading, rows in groups.items():
        lines += [f"## {heading}", "", "| Flag | Default | Description |", "|---|---|---|", *rows, ""]
    pages[page] = "\n".join(lines)

# ---- Python API: the classes molt.agents exports, with their fields, methods, and docstrings.
init = ast.parse((REPO / "molt/agents/__init__.py").read_text())
exported = next(
    ast.literal_eval(n.value) for n in init.body if isinstance(n, ast.Assign) and n.targets[0].id == "__all__"
)
lines = ["# `molt.agents`", "", NOTE.format("molt/agents/"), "```text", ast.get_docstring(init), "```", ""]
for src in ["molt/agents/base.py", "molt/agents/chat_agent.py"]:
    source = (REPO / src).read_text().splitlines()
    for cls in ast.parse("\n".join(source)).body:
        if not isinstance(cls, ast.ClassDef) or cls.name not in exported:
            continue
        bases = f"({', '.join(ast.unparse(b) for b in cls.bases)})" if cls.bases else ""
        lines += [f"## `{cls.name}`", "", f"`class {cls.name}{bases}` in `{src}`", ""]
        if ast.get_docstring(cls):
            lines += ["```text", ast.get_docstring(cls), "```", ""]
        fields = [s for s in cls.body if isinstance(s, ast.AnnAssign)]
        if fields:
            lines += ["| Field | Type | Default | Note |", "|---|---|---|---|"]
            for f in fields:
                kind = ast.unparse(f.annotation).replace("|", "\\|")
                default = f"`{ast.unparse(f.value)}`" if f.value else ""
                note = source[f.end_lineno - 1].partition("#")[2].strip()  # the trailing comment, if any
                lines.append(f"| `{f.target.id}` | `{kind}` | {default} | {note} |")
            lines.append("")
        for fn in cls.body:
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name.startswith("_"):
                continue
            prefix = "async def" if isinstance(fn, ast.AsyncFunctionDef) else "def"
            returns = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
            signature = f"{prefix} {fn.name}({ast.unparse(fn.args)}){returns}"
            lines += [f"### `{cls.name}.{fn.name}`", "", "```python", signature, "```", ""]
            if ast.get_docstring(fn):
                lines += ["```text", ast.get_docstring(fn), "```", ""]
pages["api-agents"] = "\n".join(lines)

# ---- Write, or check: generated pages are current and every flag a page names exists.
if "--check" not in sys.argv:
    for name, text in pages.items():
        (PAGES / f"{name}.md").write_text(text)
    sys.exit(0)

# The guide names this flag as not yet on main: its kernels are being upstreamed.
flags.add("--train.batch_invariant")
stale = [name for name, text in pages.items() if (PAGES / f"{name}.md").read_text() != text]
problems = [f"{name}.md is stale: rerun this script" for name in stale]
for path in sorted(PAGES.glob("*.md")):
    # A trailing * is a wildcard such as --actor.lora_*, not a flag.
    named = set(re.findall(r"--[a-z_]+(?:\.[a-z_0-9]+)+(?![\w*])", path.read_text()))
    problems += [f"{path.name} names {flag}, which neither CLI defines" for flag in sorted(named - flags)]
print("\n".join(problems) or "reference pages are current; every flag named in the guide exists")
sys.exit(bool(problems))
