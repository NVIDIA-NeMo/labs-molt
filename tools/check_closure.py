# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static closure checks for the slim tree (no imports executed).

1. every `from nemo_automodel... import` / `import nemo_automodel...` names a module that exists,
2. every imported name is defined in (or a submodule of) that module,
3. every string constant that spells a `nemo_automodel.*` module path points at an existing module
   (kernels and registry entries are loaded by name).

Exit code 1 on any violation. Usage: python tools/check_closure.py [nemo_automodel] [tests]
"""
import ast
import os
import sys

ROOT = "nemo_automodel"


def modules():
    mods = {}
    for dp, _, files in os.walk(ROOT):
        for f in files:
            if f.endswith(".py"):
                p = os.path.join(dp, f)
                m = p[:-3].replace(os.sep, ".")
                mods[m[:-9] if m.endswith(".__init__") else m] = p
    return mods


def defined_names(path):
    tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                names.update(n.id for n in ast.walk(t) if isinstance(n, ast.Name))
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
    if "__getattr__" in names:  # lazy re-export module
        names.add("*")
    return names


def main(roots):
    mods = modules()
    defs = {}
    problems = []
    for root in roots:
        for dp, _, files in os.walk(root):
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(dp, f)
                try:
                    tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
                except SyntaxError as e:
                    problems.append(f"{path}: syntax error {e}")
                    continue
                mod = path[:-3].replace(os.sep, ".")
                pkg = mod[:-9] if mod.endswith(".__init__") else mod.rsplit(".", 1)[0]
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for a in node.names:
                            if a.name.startswith(ROOT) and a.name not in mods:
                                problems.append(f"{path}:{node.lineno}: import {a.name} (missing module)")
                    elif isinstance(node, ast.ImportFrom):
                        if node.level:
                            base = pkg.split(".")
                            base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                            target = ".".join(base + ([node.module] if node.module else []))
                        else:
                            target = node.module or ""
                        if not target.startswith(ROOT):
                            continue
                        if target not in mods:
                            problems.append(f"{path}:{node.lineno}: from {target} import ... (missing module)")
                            continue
                        names = defs.setdefault(target, defined_names(mods[target]))
                        if "*" in names:
                            continue
                        for a in node.names:
                            if a.name != "*" and a.name not in names and f"{target}.{a.name}" not in mods:
                                problems.append(f"{path}:{node.lineno}: from {target} import {a.name} (missing name)")
                    elif path.startswith(ROOT) and isinstance(node, ast.Constant) and isinstance(node.value, str):
                        v = node.value
                        if v.startswith(ROOT + ".") and "." in v[len(ROOT) + 1 :] and " " not in v and v not in mods:
                            parent = v.rsplit(".", 1)[0]
                            # allow "module.attr" strings, flag "package.missing_module"
                            if parent in mods and mods[parent].endswith("__init__.py") and v.split(".")[-1].islower():
                                problems.append(f"{path}:{node.lineno}: string {v!r} names a missing module")
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s) in {len(mods)} modules", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or [ROOT]))
