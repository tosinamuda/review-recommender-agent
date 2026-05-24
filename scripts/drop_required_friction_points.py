"""Remove the legacy `required_friction_points` key from the eval dataset.

Agent-side friction signals were dropped when the runtime collapsed back to the
two-field `user_persona` + `product_details` contract, so `expected.required_friction_points`
no longer participates in scoring. This script rewrites the JSONL file in place
with that key removed, preserving every other field and the line order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PATH = Path("data/eval/review_simulation_eval.jsonl")


def strip_required_friction_points(path: Path) -> tuple[int, int]:
    lines = path.read_text(encoding="utf-8").splitlines()
    rewritten = []
    touched = 0
    for line in lines:
        if not line.strip():
            rewritten.append(line)
            continue
        case = json.loads(line)
        expected = case.get("expected", {})
        if "required_friction_points" in expected:
            del expected["required_friction_points"]
            touched += 1
        rewritten.append(json.dumps(case, ensure_ascii=False))
    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return touched, len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_PATH,
        help="JSONL evaluation dataset path.",
    )
    args = parser.parse_args()
    touched, total = strip_required_friction_points(args.path)
    print(f"Stripped required_friction_points from {touched} / {total} lines in {args.path}.")


if __name__ == "__main__":
    main()
