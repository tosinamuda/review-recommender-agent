from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path("data/raw/konga_google_play_reviews.jsonl")
DEFAULT_OUTPUT = Path("data/generated/konga_persona_review_training_cases.jsonl")
PRODUCT_DETAILS = (
    "Product: Konga Online Marketplace\n"
    "Category: ecommerce shopping app\n"
    "Description: Online marketplace app for shopping phones, electronics, fashion, "
    "groceries, home items, daily deals, payments, delivery, order tracking, returns, "
    "and customer support.\n"
    "Provider: Konga"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Konga review cases with reviewer names for manual annotation."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = build_cases(load_jsonl(args.source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    print(json.dumps({"written": len(cases), "path": str(args.output)}, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_cases(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        if record.get("app_id") != "com.konga.androida":
            raise ValueError(f"Unexpected Konga app id: {record.get('app_id')}")
        cases.append(
            {
                "case_id": f"konga_{index:03d}",
                "input": {
                    "reviewer_name": str(record.get("author") or ""),
                    "user_persona": "",
                    "product_details": PRODUCT_DETAILS,
                    "product_issue": "",
                },
                "output": {
                    "review": record["review"],
                    "rating": record["rating"],
                },
            }
        )
    return cases


if __name__ == "__main__":
    main()
