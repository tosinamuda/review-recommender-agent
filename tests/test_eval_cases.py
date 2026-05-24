from __future__ import annotations

import json
from pathlib import Path

import pytest

from review.eval_cases import (
    DEFAULT_EVAL_DATASET,
    load_review_eval_cases,
    rating_rmse,
    rouge_l_f1,
    score_review_response,
)
from review.retrieval import BehavioralRetriever
from review.review_corpus import load_review_corpus
from review.service import ReviewSimulationService
from tests.fakes import ContractReviewReasoner, FixedReviewRatingCalibrator, HashingEmbeddingModel


def make_service() -> ReviewSimulationService:
    return ReviewSimulationService(
        BehavioralRetriever(
            corpus_path=Path("data/review_exemplars.jsonl"),
            embedding_model=HashingEmbeddingModel(),
            in_memory=True,
        ),
        ContractReviewReasoner(),
        rating_calibrator=FixedReviewRatingCalibrator(),
    )


def test_generated_persona_training_cases_match_prompt_contract() -> None:
    import json
    from pathlib import Path

    path = Path("data/generated/persona_review_training_cases.jsonl")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    assert len(records) == 280
    assert len({record["case_id"] for record in records}) == len(records)
    assert all(set(record) == {"case_id", "input", "output"} for record in records)
    assert all(
        set(record["input"]) == {"user_persona", "product_details", "product_issue"}
        for record in records
    )
    assert all(set(record["output"]) == {"review", "rating"} for record in records)
    assert all(record["input"]["user_persona"] == "" for record in records)
    assert all(record["input"]["product_issue"] == "" for record in records)
    assert all(record["input"]["product_details"].startswith("Product: ") for record in records)
    assert {1, 2, 3, 4, 5}.issubset({record["output"]["rating"] for record in records})
    assert all(record["output"]["review"].strip() for record in records)


def test_manual_persona_training_cases_are_annotated_source_copies() -> None:
    import json
    from pathlib import Path

    path = Path("data/generated/persona_review_training_cases_manual.jsonl")
    if not path.exists():
        return

    source_paths = (
        Path("data/generated/persona_review_training_cases.jsonl"),
        Path("data/generated/konga_persona_review_training_cases.jsonl"),
    )
    source_records = []
    for source_path in source_paths:
        source_records.extend(
            json.loads(line)
            for line in source_path.read_text().splitlines()
            if line.strip()
        )
    source_by_id = {record["case_id"]: canonical_case(record) for record in source_records}
    manual_records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    assert len({record["case_id"] for record in manual_records}) == len(manual_records)
    for manual_record in manual_records:
        source_record = source_by_id[manual_record["case_id"]]
        assert manual_record["output"] == source_record["output"]
        assert (
            manual_record["input"]["product_details"]
            == source_record["input"]["product_details"]
        )
        assert manual_record["input"]["user_persona"].strip()
        assert manual_record["input"]["product_issue"].strip()


def canonical_case(record: dict) -> dict:
    return {
        "case_id": record["case_id"],
        "input": {
            "user_persona": record["input"]["user_persona"],
            "product_details": record["input"]["product_details"],
            "product_issue": record["input"]["product_issue"],
        },
        "output": record["output"],
    }


def test_konga_staging_cases_keep_reviewer_name_outside_canonical_contract() -> None:
    import json
    from pathlib import Path

    raw_path = Path("data/raw/konga_google_play_reviews.jsonl")
    staging_path = Path("data/generated/konga_persona_review_training_cases.jsonl")
    raw_records = [
        json.loads(line)
        for line in raw_path.read_text().splitlines()
        if line.strip()
    ]
    staging_records = [
        json.loads(line)
        for line in staging_path.read_text().splitlines()
        if line.strip()
    ]

    assert len(staging_records) == len(raw_records) == 60
    assert len({record["case_id"] for record in staging_records}) == len(staging_records)
    for raw_record, staging_record in zip(raw_records, staging_records, strict=True):
        assert set(staging_record) == {"case_id", "input", "output"}
        assert set(staging_record["input"]) == {
            "reviewer_name",
            "user_persona",
            "product_details",
            "product_issue",
        }
        assert staging_record["input"]["reviewer_name"] == raw_record["author"]
        assert staging_record["input"]["user_persona"] == ""
        assert staging_record["input"]["product_issue"] == ""
        assert staging_record["input"]["product_details"].startswith(
            "Product: Konga Online Marketplace"
        )
        assert staging_record["output"] == {
            "review": raw_record["review"],
            "rating": raw_record["rating"],
        }


def test_manual_annotation_copy_strips_source_only_reviewer_name() -> None:
    from scripts.append_manual_persona_training_case import annotated_copy

    source_case = {
        "case_id": "konga_001",
        "input": {
            "reviewer_name": "Tochukwu Uzogu",
            "user_persona": "",
            "product_details": "Product: Konga Online Marketplace",
            "product_issue": "",
        },
        "output": {"review": "Late delivery and poor support.", "rating": 1},
    }

    manual_case = annotated_copy(
        source_case,
        user_persona="Konga shopper with an urgent device delivery need.",
        product_issue="KongaNow did not deliver in the promised one to six hours.",
    )

    assert set(manual_case["input"]) == {
        "user_persona",
        "product_details",
        "product_issue",
    }
    assert manual_case["input"]["user_persona"].startswith("Konga shopper")
    assert manual_case["input"]["product_issue"].startswith("KongaNow")
    assert manual_case["output"] == source_case["output"]


def test_sanitized_persona_training_cases_match_manual_source() -> None:
    import json
    import re
    from pathlib import Path

    path = Path("data/generated/persona_review_training_cases_manual_sanitized.jsonl")
    if not path.exists():
        return

    source_path = Path("data/generated/persona_review_training_cases_manual.jsonl")
    source_records = [
        json.loads(line)
        for line in source_path.read_text().splitlines()
        if line.strip()
    ]
    source_by_id = {record["case_id"]: record for record in source_records}
    sanitized_records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]

    assert len(sanitized_records) == 340
    assert len({record["case_id"] for record in sanitized_records}) == 340
    assert [record["case_id"] for record in sanitized_records] == [
        record["case_id"] for record in source_records
    ]
    for sanitized_record in sanitized_records:
        source_record = source_by_id[sanitized_record["case_id"]]
        assert sanitized_record["output"] == source_record["output"]
        assert (
            sanitized_record["input"]["product_details"]
            == source_record["input"]["product_details"]
        )
        assert (
            sanitized_record["input"]["product_issue"]
            == source_record["input"]["product_issue"]
        )
        assert sanitized_record["input"]["user_persona"].strip()
        assert "reviewer_name" not in sanitized_record
        assert "reviewer_name" not in sanitized_record["input"]
        persona_words = re.findall(
            r"\b[\w'-]+\b",
            sanitized_record["input"]["user_persona"],
        )
        assert len(persona_words) <= 22


def test_eval_dataset_loads_with_unique_case_ids() -> None:
    cases = load_review_eval_cases(DEFAULT_EVAL_DATASET)

    assert len(cases) == 28
    assert len({case.case_id for case in cases}) == len(cases)
    assert all(case.product_details.strip() for case in cases)


def test_nigerian_business_holdout_eval_is_separate_from_review_index() -> None:
    holdout_path = Path("data/eval/review_simulation_holdout_nigerian_business_eval.jsonl")
    cases = load_review_eval_cases(holdout_path)
    review_corpus_blob = Path("data/review_exemplars.jsonl").read_text(encoding="utf-8").lower()

    assert len(cases) == 12
    assert len({case.case_id for case in cases}) == len(cases)
    for case in cases:
        product_name = case.product_details.split("Product:\n", 1)[1].split("\n", 1)[0]
        assert case.case_id.startswith("ng_review_holdout_")
        assert product_name.casefold() not in case.user_persona.casefold()
        assert product_name.casefold() not in review_corpus_blob


def test_google_maps_review_holdout_has_source_evidence_without_persona_leakage() -> None:
    holdout_path = Path("data/eval/review_simulation_holdout_google_maps_review_eval.jsonl")
    raw_cases = [
        json.loads(line)
        for line in holdout_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    cases = load_review_eval_cases(holdout_path)
    review_corpus_blob = Path("data/review_exemplars.jsonl").read_text(encoding="utf-8").lower()

    assert len(cases) == 6
    assert len({case.case_id for case in cases}) == len(cases)
    for raw_case, case in zip(raw_cases, cases, strict=True):
        product_name = case.product_details.split("Product:\n", 1)[1].split("\n", 1)[0]
        source_evidence = raw_case["source_evidence"]
        leakage_terms = source_evidence["leakage_terms"]
        persona = case.user_persona.casefold()

        assert case.case_id.startswith("ng_maps_review_")
        assert product_name.casefold() not in persona
        assert product_name.casefold() not in review_corpus_blob
        assert source_evidence["source"] == "google_maps_visible_review_snippets"
        assert source_evidence["source_url"].startswith("https://www.google.com/maps/")
        assert source_evidence["review_signals"]
        assert all(signal.strip() for signal in source_evidence["review_signals"])
        assert leakage_terms
        assert all(term.casefold() not in persona for term in leakage_terms)


def test_review_exemplar_corpus_loads_with_unique_ids() -> None:
    records = load_review_corpus("data/review_exemplars.jsonl")

    assert len(records) == 340
    assert len({record.exemplar_id for record in records}) == len(records)
    assert {1, 2, 3, 4, 5}.issubset({record.rating for record in records})
    assert all(record.product_details for record in records)
    assert all(record.source.startswith("sanitized_training:") for record in records)


def test_retrieval_and_calibration_artifacts_are_checked_in() -> None:
    artifact_paths = [
        Path("data/index/retrieval/persona.faiss"),
        Path("data/index/retrieval/product.faiss"),
        Path("data/index/retrieval/joint.faiss"),
        Path("data/index/retrieval/metadata.json"),
        Path("data/index/rating_calibrator.joblib"),
    ]

    assert all(path.exists() for path in artifact_paths)


def test_eval_scoring_detects_expected_failure() -> None:
    case = next(
        case
        for case in load_review_eval_cases(DEFAULT_EVAL_DATASET)
        if case.case_id == "food_osun_corper_no_pepper"
    )
    service = make_service()
    response = service.run(case.to_request())
    response.rating = 5
    response.review = "The pepper level was solid and perfect for a student."

    result = score_review_response(case, response)

    assert result.passed is False
    assert any("rating 5 outside expected range" in failure for failure in result.failures)
    assert "forbidden term present: pepper level was solid" in result.failures


def test_paper_metric_helpers_compute_rouge_l_and_rmse() -> None:
    assert rouge_l_f1("quick hot food", "quick food") == pytest.approx(0.8)
    assert rating_rmse([1.0, -1.0]) == pytest.approx(1.0)
