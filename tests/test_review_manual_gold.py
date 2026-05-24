from __future__ import annotations

import json
from pathlib import Path

import pytest

from review.manual_gold import display_review_eval_case, set_review_eval_gold


def test_manual_gold_workflow_sets_explicit_rating_and_reference_review(tmp_path: Path) -> None:
    path = write_eval_fixture(tmp_path)

    preview = display_review_eval_case(path, next_missing=True)
    assert preview["case_id"] == "case-1"
    assert preview["current_gold"] == {"rating": None, "reference_review": None}

    updated = set_review_eval_gold(
        path,
        case_id="case-1",
        rating=3,
        reference_review="Pepper was missing, so as a corper I cannot score it high.",
    )

    assert updated["expected"]["rating"] == 3
    assert (
        updated["expected"]["reference_review"]
        == "Pepper was missing, so as a corper I cannot score it high."
    )


def test_manual_gold_workflow_rejects_rating_outside_expected_range(
    tmp_path: Path,
) -> None:
    path = write_eval_fixture(tmp_path)

    with pytest.raises(ValueError, match="above expected rating_max"):
        set_review_eval_gold(
            path,
            case_id="case-1",
            rating=5,
            reference_review="This should not be accepted.",
        )


def write_eval_fixture(tmp_path: Path) -> Path:
    path = tmp_path / "review_eval.jsonl"
    record = {
        "case_id": "case-1",
        "user_persona": "Osun corper who likes peppery food.",
        "product_details": "Rice with no pepper.",
        "expected": {
            "rating_min": 2,
            "rating_max": 3,
            "required_terms": ["pepper"],
            "forbidden_terms": [],
            "minimum_score": 0.7,
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return path
