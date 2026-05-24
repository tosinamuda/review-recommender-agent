from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NO_MISSING_REVIEW_GOLD_MESSAGE = "No review eval case is missing gold labels."


def display_review_eval_case(
    path: Path,
    *,
    case_id: str | None = None,
    next_missing: bool = False,
) -> dict[str, Any]:
    records = load_jsonl_records(path)
    record = find_case_for_display(records, case_id=case_id, next_missing=next_missing)
    expected = record.get("expected", {})
    return {
        "case_id": record["case_id"],
        "notes": record.get("notes", ""),
        "user_persona": record["user_persona"],
        "product_details": record["product_details"],
        "expected": expected,
        "current_gold": {
            "rating": expected.get("rating"),
            "reference_review": expected.get("reference_review")
            or expected.get("gold_review")
            or expected.get("review"),
        },
    }


def set_review_eval_gold(
    path: Path,
    *,
    case_id: str,
    rating: int,
    reference_review: str,
    force: bool = False,
) -> dict[str, Any]:
    records = load_jsonl_records(path)
    assert_unique_case_ids(records, path)
    target_index = next(
        (index for index, record in enumerate(records) if record["case_id"] == case_id),
        None,
    )
    if target_index is None:
        raise ValueError(f"Unknown case_id: {case_id}")

    target = records[target_index]
    expected = target.setdefault("expected", {})
    if has_gold_labels(expected) and not force:
        raise ValueError(f"{case_id} already has gold labels. Pass --force to overwrite.")
    validate_rating(rating, expected)
    normalized_review = normalize_reference_review(reference_review)

    expected["rating"] = rating
    expected["reference_review"] = normalized_review
    records[target_index] = target
    write_jsonl_records(path, records)
    return target


def find_case_for_display(
    records: list[dict[str, Any]],
    *,
    case_id: str | None,
    next_missing: bool,
) -> dict[str, Any]:
    if bool(case_id) == bool(next_missing):
        raise ValueError("Pass exactly one of case_id or --next-missing.")
    assert_unique_case_ids(records, None)
    if next_missing:
        for record in records:
            if not has_gold_labels(record.get("expected", {})):
                return record
        raise ValueError(NO_MISSING_REVIEW_GOLD_MESSAGE)
    matches = [record for record in records if record["case_id"] == case_id]
    if not matches:
        raise ValueError(f"Unknown case_id: {case_id}")
    return matches[0]


def has_gold_labels(expected: dict[str, Any]) -> bool:
    return expected.get("rating") is not None and bool(
        expected.get("reference_review") or expected.get("gold_review") or expected.get("review")
    )


def validate_rating(rating: int, expected: dict[str, Any]) -> None:
    if rating < 1 or rating > 5:
        raise ValueError("Gold rating must be between 1 and 5.")
    rating_min = expected.get("rating_min")
    rating_max = expected.get("rating_max")
    if rating_min is not None and rating < int(rating_min):
        raise ValueError(f"Gold rating {rating} is below expected rating_min {rating_min}.")
    if rating_max is not None and rating > int(rating_max):
        raise ValueError(f"Gold rating {rating} is above expected rating_max {rating_max}.")


def normalize_reference_review(reference_review: str) -> str:
    normalized = " ".join(reference_review.strip().split())
    if not normalized:
        raise ValueError("Gold reference review must not be empty.")
    return normalized


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Review eval dataset not found: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Review eval dataset is empty: {path}")
    return records


def write_jsonl_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def assert_unique_case_ids(records: list[dict[str, Any]], path: Path | None) -> None:
    seen_ids: set[str] = set()
    for record in records:
        case_id = str(record.get("case_id", ""))
        if not case_id:
            raise ValueError("Review eval record is missing case_id.")
        if case_id in seen_ids:
            location = f" in {path}" if path is not None else ""
            raise ValueError(f"Duplicate case_id{location}: {case_id}")
        seen_ids.add(case_id)


def dumps_pretty(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
