from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path("data/generated/persona_review_training_cases.jsonl")
DEFAULT_OUTPUT = Path("data/generated/persona_review_training_cases_manual.jsonl")
CANONICAL_INPUT_KEYS = ("user_persona", "product_details", "product_issue")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show one source case or append its manually annotated copy."
    )
    parser.add_argument("case_id")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--user-persona")
    parser.add_argument("--product-issue")
    args = parser.parse_args()

    source_case = find_case(load_jsonl(args.source), args.case_id)
    if args.user_persona is None and args.product_issue is None:
        print(json.dumps(source_case, ensure_ascii=False, indent=2))
        return
    if not args.user_persona or not args.product_issue:
        raise ValueError("Both --user-persona and --product-issue are required to append.")

    existing = load_jsonl(args.output) if args.output.exists() else []
    if any(record["case_id"] == args.case_id for record in existing):
        raise ValueError(f"{args.case_id} already exists in {args.output}.")

    manual_case = annotated_copy(
        source_case,
        user_persona=args.user_persona,
        product_issue=args.product_issue,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(manual_case, ensure_ascii=False) + "\n")

    print(json.dumps(manual_case, ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_case(cases: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    matches = [case for case in cases if case["case_id"] == case_id]
    if not matches:
        raise ValueError(f"Unknown case_id: {case_id}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate case_id in source: {case_id}")
    return matches[0]


def annotated_copy(
    source_case: dict[str, Any],
    *,
    user_persona: str,
    product_issue: str,
) -> dict[str, Any]:
    copied = {
        "case_id": source_case["case_id"],
        "input": {
            key: source_case["input"][key]
            for key in CANONICAL_INPUT_KEYS
        },
        "output": dict(source_case["output"]),
    }
    copied["input"]["user_persona"] = user_persona
    copied["input"]["product_issue"] = product_issue
    return copied


if __name__ == "__main__":
    main()
