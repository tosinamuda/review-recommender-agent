from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from statistics import median
from time import sleep
from typing import Any, Protocol

from .case_data import RecommendationEvalCase
from .schemas import RecommendationRequest


class RecommendationEvaluatorService(Protocol):
    provider_name: str

    def run(self, request: RecommendationRequest) -> Any: ...


def load_recommendation_eval_cases(path: Path) -> list[RecommendationEvalCase]:
    if not path.exists():
        raise FileNotFoundError(f"Recommendation eval dataset not found: {path}")
    return [
        RecommendationEvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_recommendation_service(
    *,
    service: RecommendationEvaluatorService,
    eval_cases: list[RecommendationEvalCase],
    k: int = 10,
    delay_seconds: float = 0.0,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict:
    if not eval_cases:
        raise ValueError("Recommendation eval dataset is empty.")
    hit_5 = 0
    hit_10 = 0
    ndcg_10_total = 0.0
    invalid_id_count = 0
    candidate_count_total = 0
    coverage_correct = 0
    cold_start_case_count = 0
    fit_scores: list[float] = []

    for index, case in enumerate(eval_cases):
        response = service.run(
            RecommendationRequest(
                user_persona=case.user_persona,
                context=case.context,
                k=k,
            )
        )
        if (
            response.coverage.allow_concrete_recommendations
            == case.expected_allow_concrete_recommendations
        ):
            coverage_correct += 1
        if is_cold_start_response(response):
            cold_start_case_count += 1
        returned_ids = [item.product.product_id for item in response.recommendations]
        fit_scores.extend(item.fit_score for item in response.recommendations)
        relevant_ids = set(case.relevant_product_ids)
        validated_returned_ids = set(response.evidence.returned_product_ids)
        invalid_id_count += sum(
            1
            for product_id in returned_ids
            if product_id not in validated_returned_ids
        )
        candidate_count_total += response.evidence.candidate_count
        if any(product_id in relevant_ids for product_id in returned_ids[:5]):
            hit_5 += 1
        if any(product_id in relevant_ids for product_id in returned_ids[:10]):
            hit_10 += 1
        ndcg_10_total += ndcg_at_k(returned_ids, relevant_ids, k=10)
        if progress is not None:
            progress(index + 1, len(eval_cases), case.case_id)
        if delay_seconds and index < len(eval_cases) - 1:
            sleep(delay_seconds)

    case_count = len(eval_cases)
    return {
        "case_count": case_count,
        "hit_rate_at_5": round(hit_5 / case_count, 6),
        "hit_rate_at_10": round(hit_10 / case_count, 6),
        "ndcg_at_10": round(ndcg_10_total / case_count, 6),
        "coverage_accuracy": round(coverage_correct / case_count, 6),
        "median_fit_score": round(median(fit_scores), 6) if fit_scores else None,
        "cold_start_case_count": cold_start_case_count,
        "invalid_id_count": invalid_id_count,
        "average_candidate_count": round(candidate_count_total / case_count, 3),
        "model_provider": service.provider_name,
    }


def ndcg_at_k(returned_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for index, product_id in enumerate(returned_ids[:k], 1):
        if product_id in relevant_ids:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant_ids), k)
    if ideal_hits == 0:
        return 0.0
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_hits + 1))
    return dcg / ideal_dcg


def is_cold_start_response(response: Any) -> bool:
    candidates = getattr(getattr(response, "evidence", None), "candidates", [])
    if not candidates:
        return True
    return all(
        float(getattr(candidate, "axis_scores", {}).get("similar_persona", 0.0)) <= 0.0
        for candidate in candidates
    )


def dumps_metrics(metrics: dict) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True)
