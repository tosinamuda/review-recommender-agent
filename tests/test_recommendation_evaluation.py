from __future__ import annotations

from types import SimpleNamespace

import pytest

from recommendation.case_data import RecommendationEvalCase
from recommendation.evaluation import evaluate_recommendation_service


class StubRecommendationService:
    provider_name = "stub-ranker"

    def __init__(self, responses: list[SimpleNamespace]):
        self._responses = responses

    def run(self, request):
        return self._responses.pop(0)


def test_recommendation_evaluator_reports_paper_metrics() -> None:
    service = StubRecommendationService(
        [
            response(
                returned_ids=["relevant-a", "other-b"],
                fit_scores=[0.9, 0.7],
                allow_concrete=True,
                candidate_axis_scores=[{"similar_persona": 0.8}],
            ),
            response(
                returned_ids=["other-c"],
                fit_scores=[0.5],
                allow_concrete=False,
                candidate_axis_scores=[{"product_text": 0.6}],
            ),
        ]
    )
    cases = [
        RecommendationEvalCase(
            case_id="case-1",
            user_persona="Persona",
            context="Context",
            relevant_product_ids=["relevant-a"],
        ),
        RecommendationEvalCase(
            case_id="case-2",
            user_persona="Persona",
            context="Context",
            relevant_product_ids=["relevant-c"],
        ),
    ]

    metrics = evaluate_recommendation_service(service=service, eval_cases=cases, k=10)

    assert metrics["hit_rate_at_10"] == pytest.approx(0.5)
    assert metrics["coverage_accuracy"] == pytest.approx(0.5)
    assert metrics["median_fit_score"] == pytest.approx(0.7)
    assert metrics["cold_start_case_count"] == 1


def response(
    *,
    returned_ids: list[str],
    fit_scores: list[float],
    allow_concrete: bool,
    candidate_axis_scores: list[dict[str, float]],
) -> SimpleNamespace:
    return SimpleNamespace(
        recommendations=[
            SimpleNamespace(
                product=SimpleNamespace(product_id=product_id),
                fit_score=fit_score,
            )
            for product_id, fit_score in zip(returned_ids, fit_scores, strict=True)
        ],
        coverage=SimpleNamespace(allow_concrete_recommendations=allow_concrete),
        evidence=SimpleNamespace(
            returned_product_ids=returned_ids,
            candidate_count=len(candidate_axis_scores),
            candidates=[
                SimpleNamespace(axis_scores=axis_scores)
                for axis_scores in candidate_axis_scores
            ],
        ),
    )
