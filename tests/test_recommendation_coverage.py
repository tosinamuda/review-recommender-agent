from __future__ import annotations

from pathlib import Path

from recommendation.ranker_contracts import RecommendationRanker
from recommendation.retrieval import ProductRetriever
from recommendation.schemas import (
    CandidateProduct,
    CandidateProductSet,
    CoverageDecision,
    GeneratedRecommendationResult,
    RankingResult,
    RecommendationItem,
    RecommendationRequest,
)
from recommendation.service import RecommendationService
from tests.fakes import (
    ContractGeneratedRecommendationReasoner,
    ContractRecommendationCoverageJudge,
    ContractRecommendationRanker,
    HashingEmbeddingModel,
    RejectingRecommendationCoverageJudge,
)


def test_coverage_limited_response_skips_ranker_and_does_not_leak_candidates(
    tmp_path: Path,
) -> None:
    service = RecommendationService(
        retriever=make_test_retriever(tmp_path),
        coverage_judge=RejectingRecommendationCoverageJudge(),
        generated_reasoner=ContractGeneratedRecommendationReasoner(),
        ranker=ExplodingRecommendationRanker(),
    )

    response = service.run(
        RecommendationRequest(
            user_persona="Lagos student, budget conscious, likes spicy food.",
            context="weekday lunch near Yaba",
            k=5,
        )
    )

    assert response.recommendation_mode == "llm_generated"
    assert response.recommendations == []
    assert len(response.generated_recommendations) == 5
    assert response.coverage.status == "insufficient"
    assert response.coverage.allow_concrete_recommendations is False
    generated_text = " ".join(
        f"{item.product.name} {item.headline} {item.reasoning}"
        for item in response.generated_recommendations
    )
    assert all(
        item.product.metadata["catalogue_grounded"] is False
        for item in response.generated_recommendations
    )
    assert all(
        item.product.product_id.startswith("generated_")
        for item in response.generated_recommendations
    )
    for candidate in response.evidence.candidates:
        assert candidate.product.name not in generated_text
    assert [event.stage for event in response.trace] == [
        "retrieve_candidate_items",
        "judge_candidate_coverage",
        "generate_contextual_recommendations",
        "validate_and_build_response",
    ]


def test_request_supplied_candidates_are_ranked_without_catalogue_retrieval(
    tmp_path: Path,
) -> None:
    service = RecommendationService(
        retriever=make_test_retriever(tmp_path),
        coverage_judge=ContractRecommendationCoverageJudge(),
        generated_reasoner=ContractGeneratedRecommendationReasoner(),
        ranker=ContractRecommendationRanker(),
    )
    supplied = [
        CandidateProduct(
            product_id="gh_food_001",
            name="Osu Pepper Rice Spot",
            category="food",
            description="Affordable spicy rice and grilled chicken near Osu.",
            price=45,
            currency="GHS",
            location="Osu, Accra",
        ),
        CandidateProduct(
            product_id="gh_food_002",
            name="Accra Waakye Bowl",
            category="food",
            description="Large portion waakye with pepper sauce near campus.",
            price=40,
            currency="GHS",
            location="Osu, Accra",
        ),
    ]

    response = service.run(
        RecommendationRequest(
            user_persona="Accra-based student, budget conscious, likes spicy food.",
            context="weekday lunch near Osu",
            k=2,
            candidate_items=supplied,
        )
    )

    assert response.recommendation_mode == "request_supplied_candidates"
    assert response.coverage.candidate_source == "request_supplied"
    assert response.evidence.candidate_count == 2
    assert response.evidence.retrieved_via == ["request_supplied_candidates"]
    assert [item.product.product_id for item in response.recommendations] == [
        "gh_food_001",
        "gh_food_002",
    ]


def test_generated_recommendations_drop_rejected_candidate_name_leaks(
    tmp_path: Path,
) -> None:
    service = RecommendationService(
        retriever=make_test_retriever(tmp_path),
        coverage_judge=RejectingRecommendationCoverageJudge(),
        generated_reasoner=LeakyGeneratedRecommendationReasoner(),
        ranker=ExplodingRecommendationRanker(),
    )

    response = service.run(
        RecommendationRequest(
            user_persona="Lagos student, budget conscious, likes spicy food.",
            context="weekday lunch near Yaba",
            k=2,
        )
    )

    assert response.recommendation_mode == "llm_generated"
    assert len(response.generated_recommendations) == 1
    returned = response.generated_recommendations[0]
    assert returned.product.product_id.startswith("generated_")
    assert returned.product.metadata["catalogue_grounded"] is False
    rejected_names = {candidate.product.name for candidate in response.evidence.candidates}
    returned_text = " ".join(
        [
            returned.product.name,
            returned.product.description,
            returned.headline,
            returned.reasoning,
        ]
    )
    assert all(name not in returned_text for name in rejected_names)


class ExplodingRecommendationRanker(RecommendationRanker):
    provider_name = "test-exploding-recommendation-ranker"

    def rank(
        self,
        user_persona: str,
        context: str,
        candidate_set: CandidateProductSet,
        k: int,
    ) -> RankingResult:
        del user_persona, context, candidate_set, k
        raise AssertionError("Ranker should not run for coverage-limited responses.")


class LeakyGeneratedRecommendationReasoner(ContractGeneratedRecommendationReasoner):
    def generate(
        self,
        *,
        user_persona: str,
        context: str,
        candidate_set: CandidateProductSet,
        coverage_decision: CoverageDecision,
        k: int,
    ) -> GeneratedRecommendationResult:
        del user_persona, context, coverage_decision, k
        leaked_name = candidate_set.candidates[0].product.name
        return GeneratedRecommendationResult(
            candidate_set=candidate_set,
            persona_needs=["budget", "spicy lunch"],
            recommendations=[
                RecommendationItem(
                    rank=1,
                    product=CandidateProduct(
                        product_id="unsafe_id",
                        name=leaked_name,
                        category="food",
                        description="Leaked rejected candidate name.",
                    ),
                    fit_score=0.9,
                    headline=f"Leaked {leaked_name}",
                    reasoning="This should be dropped.",
                ),
                RecommendationItem(
                    rank=2,
                    product=CandidateProduct(
                        product_id="generated_food_02",
                        name="Affordable spicy local eatery near Yaba",
                        category="food",
                        description="Generated recommendation profile, not a catalogue item.",
                    ),
                    fit_score=0.85,
                    headline="Generated fit for lunch near Yaba",
                    reasoning="Matches the persona without naming a rejected candidate.",
                ),
            ],
        )


def make_test_retriever(tmp_path: Path) -> ProductRetriever:
    return ProductRetriever(
        catalogue_path=Path("data/product_catalogue.jsonl"),
        index_dir=tmp_path / "recommendation-index",
        interactions_path=tmp_path / "recommendation-interactions.jsonl",
        eval_cases_path=tmp_path / "recommendation-eval-cases.jsonl",
        embedding_model=HashingEmbeddingModel(),
    )
