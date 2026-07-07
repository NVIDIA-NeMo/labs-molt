"""Unit tests for the geo3k math-answer grader.

Covers the two-tier equivalence (string normalization + symbolic) and guards
against over-counting. Run: python3 tests/unit/test_math_grader.py
"""

import importlib.util
from pathlib import Path

_MG = Path(__file__).resolve().parents[2] / "examples" / "python" / "utils" / "math_grader.py"
_spec = importlib.util.spec_from_file_location("math_grader", _MG)
mg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mg)

try:
    import sympy  # noqa: F401

    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False

# Must be judged EQUAL (resolvable by string/normalization, no sympy).
MATCH = [
    ("36", r"36^\circ"),
    (r"36^{\circ}", "36"),
    ("36°", "36"),  # unicode degree
    ("12", "12 cm"),
    ("12cm", "12"),
    ("36 degrees", "36"),
    ("0.5", r"\frac{1}{2}"),
    (r"\frac{1}{2}", "0.5"),
    ("0.5", "1/2"),
    ("30", "x=30"),  # short leading assignment
    ("100", "100"),
    ("3.0", "3"),
]

# Must be judged EQUAL but only via sympy (skipped if sympy absent).
MATCH_SYMPY = [
    (r"\sqrt{12}", r"2\sqrt{3}"),
    (r"2\sqrt{3}", r"\sqrt{12}"),
    ("1/2", "0.50"),
]

# Must be judged DIFFERENT — over-count guard.
NOMATCH = [
    ("36", "37"),
    ("0.5", "0.6"),
    ("12 cm", "13 cm"),
    (r"2\sqrt{3}", r"3\sqrt{2}"),
    ("", "5"),
    ("5", ""),
    ("36", "360"),
    ("1/3", "2/6"),  # unreduced fractions must match exactly
]


def main() -> None:
    fails = []
    for a, b in MATCH:
        if not mg.answers_match(a, b):
            fails.append(f"EXPECTED MATCH but differ: {a!r} vs {b!r}")
    for a, b in NOMATCH:
        if mg.answers_match(a, b):
            fails.append(f"EXPECTED DIFFER but match: {a!r} vs {b!r}")

    # Balanced \boxed extraction through the public scoring entry point.
    r = mg.score_response(r"\boxed{36^\circ}", "", {"ground_truth": "36"})
    if r["reward"] != 1.0:
        fails.append(f"score_response boxed-degree reward={r['reward']} (expected 1.0)")

    if HAVE_SYMPY:
        for a, b in MATCH_SYMPY:
            if not mg.answers_match(a, b):
                fails.append(f"EXPECTED MATCH (sympy) but differ: {a!r} vs {b!r}")
        rs = mg.score_response(r"\boxed{\frac{1}{2}}", "", {"ground_truth": "0.5"})
        if rs["reward"] != 1.0:
            fails.append(f"score_response nested-boxed reward={rs['reward']} (expected 1.0)")
        note = "sympy cases RAN"
    else:
        note = "sympy cases SKIPPED"

    if fails:
        print(f"FAILED ({len(fails)}):")
        for f in fails:
            print("  -", f)
        raise SystemExit(1)
    print(f"ALL GRADER TESTS PASSED. {note}")


if __name__ == "__main__":
    main()
