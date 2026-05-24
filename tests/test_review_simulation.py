from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.settings import Settings
from review.openrouter_reasoner import (
    GenerateReview,
    SelectBestReview,
    VerifyReview,
    similar_user_review_records,
)
from review.reasoner_contracts import ReviewReasoner
from review.retrieval import BehavioralRetriever
from review.schemas import (
    AggregatedCandidate,
    Candidate,
    Candidates,
    ExemplarSet,
    ReviewRequest,
    SelectedCandidate,
)
from review.service import ReviewSimulationService
from shared import openrouter_lm
from shared.openrouter_lm import configure_dspy_openrouter
from tests.fakes import ContractReviewReasoner, FixedReviewRatingCalibrator, HashingEmbeddingModel

ARCHITECTURE_TRACE_STAGES = [
    "retrieve_similar_user_reviews",
    "generate_review_with_refinement",
    "select_best_review_for_persona",
    "calibrate_rating_from_review",
]

USER_PERSONA = (
    "Ondo-based corper, budget conscious, likes peppery food, quick service, "
    "and generous portions."
)
PRODUCT_DETAILS = (
    "Jollof Bowl with Grilled Chicken. Smoky jollof rice with grilled chicken, "
    "fried plantain, and pepper sauce. NGN 4500. Delivery 35 minutes. Large portion. "
    "Spice level: high."
)


def make_retriever(corpus_path: Path | None = None) -> BehavioralRetriever:
    return BehavioralRetriever(
        corpus_path=corpus_path or Path("data/review_exemplars.jsonl"),
        embedding_model=HashingEmbeddingModel(),
        in_memory=True,
    )


def make_service(reasoner: ReviewReasoner | None = None) -> ReviewSimulationService:
    return ReviewSimulationService(
        make_retriever(),
        reasoner or ContractReviewReasoner(),
        rating_calibrator=FixedReviewRatingCalibrator(),
    )


def sample_request(
    user_persona: str = USER_PERSONA,
    product_details: str = PRODUCT_DETAILS,
) -> ReviewRequest:
    return ReviewRequest(
        user_persona=user_persona,
        product_details=product_details,
    )


def test_retrieval_returns_exemplars_from_seed_corpus() -> None:
    exemplar_set = make_retriever().retrieve(USER_PERSONA, PRODUCT_DETAILS)

    assert exemplar_set.exemplars
    assert exemplar_set.proxy_stats.n_exemplars == 5
    assert exemplar_set.exemplars[0].axis_scores


def test_faiss_retrieval_uses_supplied_jsonl_corpus_and_three_query_axes(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "reviews.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "exemplar_id": "custom_food_001",
                "review_text": "Pepper was strong and the portion was large.",
                "rating": 5,
                "user_profile_summary": "custom Lagos food reviewer",
                "item_category": "food",
                "source": "test_fixture",
                "nigerian": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    embedding_model = HashingEmbeddingModel()

    exemplar_set = BehavioralRetriever(
        corpus_path=corpus_path,
        embedding_model=embedding_model,
        in_memory=True,
    ).retrieve(
        "Lagos buyer who likes spicy food and large portions",
        "Pepper Rice with pepper sauce and a large portion. NGN 3000.",
    )

    assert [exemplar.exemplar_id for exemplar in exemplar_set.exemplars] == ["custom_food_001"]
    assert set(exemplar_set.exemplars[0].axis_scores) == {"persona", "product", "joint"}


def test_service_returns_bounded_rating_and_grounded_review() -> None:
    service = make_service()

    response = service.run(sample_request())

    assert 1 <= response.rating <= 5
    assert response.evidence.rating_distribution
    assert response.evidence.rating_continuous
    assert response.evidence.similar_user_reviews
    assert response.evidence.candidate_reviews
    assert response.evidence.reason
    assert response.evidence.user_persona == USER_PERSONA
    assert response.evidence.product_details == PRODUCT_DETAILS
    assert "Jollof Bowl" in response.review
    assert "corper" in response.review.lower()
    assert "student" not in response.review.lower()
    assert [event.stage for event in response.trace] == ARCHITECTURE_TRACE_STAGES


def test_rating_is_assigned_by_calibration_boundary_after_selection() -> None:
    service = make_service(CandidateRatingMismatchReasoner())

    response = service.run(sample_request())

    assert response.trace[-1].stage == "calibrate_rating_from_review"
    assert "test-only fixed calibrator" in response.trace[-1].message


def test_default_sample_count_is_three() -> None:
    request = ReviewRequest.model_validate(sample_request().model_dump())

    assert request.options.sample_count == 3


def test_no_pepper_product_reflects_persona_friction_in_review() -> None:
    service = make_service()
    request = sample_request(
        user_persona=(
            "Osun-based corper, budget conscious, likes peppery food, quick service, "
            "and generous portions."
        ),
        product_details=(
            "Jollof Bowl with Grilled Chicken. Smoky jollof rice with grilled chicken, "
            "fried plantain, and no pepper. NGN 4500. Delivery 35 minutes. Large portion."
        ),
    )

    response = service.run(request)

    assert response.rating <= 3
    assert "no pepper" in response.review.lower()
    assert "pepper level was solid" not in response.review.lower()
    assert len(response.evidence.similar_user_reviews) == 5


def test_default_retriever_requires_artifacts_when_not_in_memory(tmp_path: Path) -> None:
    retriever = BehavioralRetriever(
        corpus_path=Path("data/review_exemplars.jsonl"),
        index_dir=tmp_path / "missing-index",
        embedding_model=HashingEmbeddingModel(),
    )

    with pytest.raises(FileNotFoundError, match="Review retrieval artifacts are missing"):
        retriever.retrieve(USER_PERSONA, PRODUCT_DETAILS)


def test_dspy_generation_signatures_use_structured_architecture_fields() -> None:
    assert list(GenerateReview.input_fields) == [
        "user_persona",
        "product_details",
        "similar_user_reviews",
    ]
    assert list(GenerateReview.output_fields) == [
        "observed_experience_types",
        "chosen_experience",
        "review",
    ]
    assert "sample_index" not in GenerateReview.input_fields
    assert list(VerifyReview.output_fields) == [
        "persona_voice_preserved",
        "product_details_consistent",
        "experience_grounded",
        "appropriate_length",
        "overall_pass",
        "critique",
    ]
    assert list(SelectBestReview.input_fields) == [
        "user_persona",
        "product_details",
        "similar_user_reviews",
        "candidate_drafts",
    ]
    assert list(SelectBestReview.output_fields) == ["best_draft_index", "reason"]


def test_similar_user_reviews_are_typed_records() -> None:
    exemplar_set = make_retriever().retrieve(USER_PERSONA, PRODUCT_DETAILS)

    records = similar_user_review_records(exemplar_set)

    assert records
    assert records[0].user_persona
    assert records[0].review


def test_openrouter_lm_uses_sampling_without_prompt_cache(monkeypatch) -> None:
    captured = {}

    class FakeLM:
        def __init__(self, model: str, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    def fake_configure(**kwargs) -> None:
        captured["configured_lm"] = kwargs["lm"]

    monkeypatch.setattr(openrouter_lm.dspy, "LM", FakeLM)
    monkeypatch.setattr(openrouter_lm.dspy, "configure", fake_configure)

    configure_dspy_openrouter(Settings())

    assert captured["kwargs"]["temperature"] == pytest.approx(0.7)
    assert captured["kwargs"]["cache"] is False


class CandidateRatingMismatchReasoner(ReviewReasoner):
    provider_name = "candidate-rating-mismatch"

    def generate_candidates(
        self,
        user_persona: str,
        product_details: str,
        exemplar_set: ExemplarSet,
        sample_count: int,
    ) -> Candidates:
        del user_persona, product_details, exemplar_set, sample_count
        return Candidates(
            samples=[
                Candidate(
                    review=(
                        "The Jollof Bowl arrived hot and the large portion helped, "
                        "but there was no pepper at all. The lack of pepper was "
                        "disappointing for someone who loves heat."
                    ),
                )
            ]
        )

    def select_best(
        self,
        user_persona: str,
        product_details: str,
        exemplar_set: ExemplarSet,
        aggregate: AggregatedCandidate,
    ) -> SelectedCandidate:
        del user_persona, product_details, exemplar_set
        return SelectedCandidate(
            review=aggregate.candidates[0].review,
            selection_reason="Selected the only candidate without assigning a rating.",
        )
