from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.append_sanitized_persona_training_case import (
    append_sanitized_case,
    display_case,
    read_case_context,
    sanitized_copy,
    validate_persona,
)


def test_read_case_context_returns_selected_source_row(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    write_jsonl(source_path, [sample_case()])

    display = read_case_context(source_path, "chowdeck_001")

    assert display == {
        "case_id": "chowdeck_001",
        "current_persona": "Issue-leaky persona.",
        "product_details": "Product: Chowdeck",
        "product_issue": "Vendor substituted an item.",
        "review": "The vendor changed my order.",
        "rating": 2,
    }


def test_display_case_returns_annotation_context() -> None:
    source_case = sample_case()

    display = display_case(source_case)

    assert display["case_id"] == "chowdeck_001"
    assert display["current_persona"] == "Issue-leaky persona."


def test_append_sanitized_case_preserves_everything_except_persona(
    tmp_path: Path,
) -> None:
    source_case = sample_case()
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "sanitized.jsonl"
    write_jsonl(source_path, [source_case])

    copied = append_sanitized_case(
        source_path=source_path,
        output_path=output_path,
        case_id="chowdeck_001",
        user_persona="Regular Nigerian Chowdeck user who orders food and groceries.",
    )

    written_records = load_jsonl(output_path)

    assert copied["case_id"] == source_case["case_id"]
    assert copied["input"]["user_persona"] == (
        "Regular Nigerian Chowdeck user who orders food and groceries."
    )
    assert copied["input"]["product_details"] == source_case["input"]["product_details"]
    assert copied["input"]["product_issue"] == source_case["input"]["product_issue"]
    assert copied["output"] == source_case["output"]
    assert written_records == [copied]


def test_sanitized_copy_preserves_everything_except_persona() -> None:
    source_case = sample_case()

    copied = sanitized_copy(
        source_case,
        "Regular Nigerian Chowdeck user who orders food and groceries.",
    )

    assert copied["input"]["user_persona"] == (
        "Regular Nigerian Chowdeck user who orders food and groceries."
    )
    assert copied["input"]["product_details"] == source_case["input"]["product_details"]
    assert copied["input"]["product_issue"] == source_case["input"]["product_issue"]
    assert copied["output"] == source_case["output"]


def test_append_sanitized_case_rejects_duplicate_case_id(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jsonl"
    output_path = tmp_path / "sanitized.jsonl"
    write_jsonl(source_path, [sample_case()])
    persona = "Regular Nigerian Chowdeck user who orders food and groceries."

    append_sanitized_case(
        source_path=source_path,
        output_path=output_path,
        case_id="chowdeck_001",
        user_persona=persona,
    )

    with pytest.raises(ValueError, match="already exists"):
        append_sanitized_case(
            source_path=source_path,
            output_path=output_path,
            case_id="chowdeck_001",
            user_persona=persona,
        )


def test_validate_persona_rejects_more_than_twenty_two_words() -> None:
    with pytest.raises(ValueError, match="22 words or fewer"):
        validate_persona(
            "One two three four five six seven eight nine ten eleven twelve "
            "thirteen fourteen fifteen sixteen seventeen eighteen nineteen "
            "twenty twentyone twentytwo twentythree."
        )


def test_validate_persona_rejects_multiple_sentences() -> None:
    with pytest.raises(ValueError, match="one sentence"):
        validate_persona("Regular Chowdeck user. Orders groceries often.")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def sample_case() -> dict[str, Any]:
    return {
        "case_id": "chowdeck_001",
        "input": {
            "user_persona": "Issue-leaky persona.",
            "product_details": "Product: Chowdeck",
            "product_issue": "Vendor substituted an item.",
        },
        "output": {
            "review": "The vendor changed my order.",
            "rating": 2,
        },
    }
