from __future__ import annotations

import re
import time
from functools import lru_cache
from typing import Literal

from app.settings import get_settings
from shared.embeddings import SentenceTransformerEmbeddingModel

from .coverage import RecommendationCoverageJudge, build_recommendation_coverage_judge
from .generated import GeneratedRecommendationReasoner, build_generated_recommendation_reasoner
from .reasoning import RecommendationRanker, build_recommendation_ranker
from .retrieval import DEFAULT_CANDIDATE_COUNT, ProductRetriever
from .schemas import (
    CandidateProductSet,
    CoverageDecision,
    GeneratedRecommendationResult,
    RankingResult,
    RecommendationCoverage,
    RecommendationEvidence,
    RecommendationItem,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationTraceEvent,
    RetrievedCandidate,
)


class RecommendationService:
    def __init__(
        self,
        retriever: ProductRetriever,
        coverage_judge: RecommendationCoverageJudge,
        generated_reasoner: GeneratedRecommendationReasoner,
        ranker: RecommendationRanker,
    ):
        self._retriever = retriever
        self._coverage_judge = coverage_judge
        self._generated_reasoner = generated_reasoner
        self._ranker = ranker

    @property
    def provider_name(self) -> str:
        return self._ranker.provider_name

    def run(self, request: RecommendationRequest) -> RecommendationResponse:
        started_at = time.perf_counter()
        trace: list[RecommendationTraceEvent] = []

        stage_started = time.perf_counter()
        candidate_set = self.retrieve(request)
        trace.append(
            RecommendationTraceEvent(
                stage="retrieve_candidate_items",
                message=(
                    f"Pulled {len(candidate_set.candidates)} candidate items from "
                    "the catalogue."
                ),
                duration_ms=elapsed_ms(stage_started),
            )
        )

        stage_started = time.perf_counter()
        coverage_decision = self.judge_coverage(request, candidate_set)
        trace.append(
            RecommendationTraceEvent(
                stage="judge_candidate_coverage",
                message=(
                    f"Coverage judged {coverage_decision.coverage_status}; "
                    f"{len(coverage_decision.viable_product_ids)} candidate ids approved."
                ),
                duration_ms=elapsed_ms(stage_started),
            )
        )

        if coverage_decision.allow_concrete_recommendations:
            stage_started = time.perf_counter()
            ranking_result = self.rank(
                request,
                candidate_set_for_viable_ids(candidate_set, coverage_decision.viable_product_ids),
            )
            trace.append(
                RecommendationTraceEvent(
                    stage="rank_and_reason",
                    message=(
                        "Ranked coverage-approved candidates against the persona and "
                        "generated per-item reasoning."
                    ),
                    duration_ms=elapsed_ms(stage_started),
                )
            )
            generated_result = None
        else:
            stage_started = time.perf_counter()
            generated_result = self.generate_contextual_recommendations(
                request,
                candidate_set,
                coverage_decision,
            )
            ranking_result = RankingResult(
                candidate_set=candidate_set,
                persona_needs=generated_result.persona_needs,
                rankings=[],
            )
            trace.append(
                RecommendationTraceEvent(
                    stage="generate_contextual_recommendations",
                    message=(
                        "Generated non-catalogue recommendations because catalogue "
                        "coverage was insufficient for the request."
                    ),
                    duration_ms=elapsed_ms(stage_started),
                )
            )

        stage_started = time.perf_counter()
        response = self.validate_and_build_response(
            request=request,
            ranking_result=ranking_result,
            coverage_decision=coverage_decision,
            generated_result=generated_result,
            trace=trace,
            elapsed_ms_total=elapsed_ms(started_at),
        )
        response.trace.append(
            RecommendationTraceEvent(
                stage="validate_and_build_response",
                message=(
                    "Validated ranked item ids, attached catalogue metadata, "
                    f"and returned {returned_recommendation_count(response)} recommendations."
                ),
                duration_ms=elapsed_ms(stage_started),
            )
        )
        response.evidence.elapsed_ms = elapsed_ms(started_at)
        return response

    def retrieve(self, request: RecommendationRequest) -> CandidateProductSet:
        if request.candidate_items:
            return CandidateProductSet(
                candidates=[
                    RetrievedCandidate(
                        product=product,
                        score=1.0,
                        axis_scores={"request_supplied": 1.0},
                    )
                    for product in request.candidate_items
                ],
                retrieved_via=["request_supplied_candidates"],
                category=request.category,
                candidate_source="request_supplied",
            )
        return self._retriever.retrieve(
            user_persona=request.user_persona,
            context=request.context,
            category=request.category,
            k=DEFAULT_CANDIDATE_COUNT,
        )

    def judge_coverage(
        self,
        request: RecommendationRequest,
        candidate_set: CandidateProductSet,
    ) -> CoverageDecision:
        if not candidate_set.candidates:
            return CoverageDecision(
                coverage_status="insufficient",
                allow_concrete_recommendations=False,
                unsupported_signals=["no retrieved candidates"],
                fallback_archetypes=[],
                reason="No candidate items were retrieved for the request.",
            )
        return self._coverage_judge.judge(
            user_persona=request.user_persona,
            context=request.context,
            candidate_set=candidate_set,
            coverage_policy=request.coverage_policy,
        )

    def rank(
        self,
        request: RecommendationRequest,
        candidate_set: CandidateProductSet,
    ) -> RankingResult:
        return self._ranker.rank(
            user_persona=request.user_persona,
            context=request.context,
            candidate_set=candidate_set,
            k=request.k,
        )

    def generate_contextual_recommendations(
        self,
        request: RecommendationRequest,
        candidate_set: CandidateProductSet,
        coverage_decision: CoverageDecision,
    ) -> GeneratedRecommendationResult:
        return self._generated_reasoner.generate(
            user_persona=request.user_persona,
            context=request.context,
            candidate_set=candidate_set,
            coverage_decision=coverage_decision,
            k=request.k,
        )

    def validate_and_build_response(
        self,
        request: RecommendationRequest,
        ranking_result: RankingResult,
        coverage_decision: CoverageDecision,
        generated_result: GeneratedRecommendationResult | None,
        trace: list[RecommendationTraceEvent],
        elapsed_ms_total: int,
    ) -> RecommendationResponse:
        candidates_by_id = {
            candidate.product.product_id: candidate
            for candidate in ranking_result.candidate_set.candidates
        }
        recommendations: list[RecommendationItem] = []
        returned_ids: set[str] = set()
        for ranking in sorted(ranking_result.rankings, key=lambda item: item.rank):
            candidate = candidates_by_id.get(ranking.product_id)
            if not candidate or ranking.product_id in returned_ids:
                continue
            returned_ids.add(ranking.product_id)
            recommendations.append(
                RecommendationItem(
                    rank=len(recommendations) + 1,
                    product=candidate.product,
                    fit_score=max(0.0, min(ranking.fit_score, 1.0)),
                    headline=ranking.headline,
                    reasoning=ranking.reasoning,
                )
            )
            candidate.shortlisted = True
            if len(recommendations) >= request.k:
                break

        generated_recommendations = sanitize_generated_recommendations(
            generated_result.recommendations if generated_result is not None else [],
            rejected_candidate_names=[
                candidate.product.name for candidate in ranking_result.candidate_set.candidates
            ],
            k=request.k,
        )
        note = ""
        if len(recommendations) < request.k:
            note = (
                f"Returned {len(recommendations)} of {request.k} requested items after "
                "dropping rankings that did not reference retrieved candidates."
            )
        if not coverage_decision.allow_concrete_recommendations:
            note = coverage_decision.reason or (
                "Catalogue coverage was insufficient for concrete recommendations."
            )

        return RecommendationResponse(
            recommendation_mode=recommendation_mode(
                ranking_result.candidate_set,
                coverage_decision,
            ),
            coverage=RecommendationCoverage(
                status=coverage_decision.coverage_status,
                allow_concrete_recommendations=coverage_decision.allow_concrete_recommendations,
                candidate_source=ranking_result.candidate_set.candidate_source,
                viable_product_ids=coverage_decision.viable_product_ids,
                unsupported_signals=coverage_decision.unsupported_signals,
                reason=coverage_decision.reason,
            ),
            recommendations=recommendations,
            generated_recommendations=generated_recommendations,
            fallback_archetypes=coverage_decision.fallback_archetypes,
            evidence=RecommendationEvidence(
                user_persona=request.user_persona,
                context=request.context,
                persona_needs=(
                    generated_result.persona_needs
                    if generated_result is not None
                    else ranking_result.persona_needs
                ),
                candidate_count=len(ranking_result.candidate_set.candidates),
                retrieved_via=ranking_result.candidate_set.retrieved_via,
                candidates=ordered_candidates(ranking_result.candidate_set.candidates),
                returned_product_ids=[item.product.product_id for item in recommendations],
                note=note,
                elapsed_ms=elapsed_ms_total,
                model_provider=self.provider_name,
            ),
            trace=[*trace],
        )

    def format_response(
        self,
        request: RecommendationRequest,
        ranking_result: RankingResult,
        coverage_decision: CoverageDecision,
        trace: list[RecommendationTraceEvent],
        elapsed_ms_total: int,
        generated_result: GeneratedRecommendationResult | None = None,
    ) -> RecommendationResponse:
        return self.validate_and_build_response(
            request=request,
            ranking_result=ranking_result,
            coverage_decision=coverage_decision,
            generated_result=generated_result,
            trace=trace,
            elapsed_ms_total=elapsed_ms_total,
        )


def ordered_candidates(candidates: list[RetrievedCandidate]) -> list[RetrievedCandidate]:
    return sorted(
        candidates,
        key=lambda candidate: (not candidate.shortlisted, -candidate.score, candidate.product.name),
    )


def candidate_set_for_viable_ids(
    candidate_set: CandidateProductSet,
    viable_product_ids: list[str],
) -> CandidateProductSet:
    viable_ids = set(viable_product_ids)
    return CandidateProductSet(
        candidates=[
            candidate
            for candidate in candidate_set.candidates
            if candidate.product.product_id in viable_ids
        ],
        retrieved_via=candidate_set.retrieved_via,
        category=candidate_set.category,
        candidate_source=candidate_set.candidate_source,
    )


def recommendation_mode(
    candidate_set: CandidateProductSet,
    coverage_decision: CoverageDecision,
) -> Literal[
    "catalogue_grounded",
    "request_supplied_candidates",
    "coverage_limited",
    "llm_generated",
]:
    if not coverage_decision.allow_concrete_recommendations:
        return "llm_generated"
    if candidate_set.candidate_source == "request_supplied":
        return "request_supplied_candidates"
    return "catalogue_grounded"


def returned_recommendation_count(response: RecommendationResponse) -> int:
    return len(response.recommendations) + len(response.generated_recommendations)


def sanitize_generated_recommendations(
    recommendations: list[RecommendationItem],
    *,
    rejected_candidate_names: list[str],
    k: int,
) -> list[RecommendationItem]:
    sanitized = []
    rejected_names = [
        name.strip().lower() for name in rejected_candidate_names if len(name.strip()) >= 4
    ]
    for item in recommendations:
        searchable_text = " ".join(
            [
                item.product.name,
                item.product.description,
                item.headline,
                item.reasoning,
            ]
        ).lower()
        if any(name in searchable_text for name in rejected_names):
            continue
        metadata = {
            **item.product.metadata,
            "catalogue_grounded": False,
            "grounding": "llm_generated",
        }
        product_id = item.product.product_id
        if not product_id.startswith("generated_"):
            product_id = f"generated_{slugify(item.product.category)}_{len(sanitized) + 1:02d}"
        sanitized.append(
            RecommendationItem(
                rank=len(sanitized) + 1,
                product=item.product.model_copy(
                    update={
                        "product_id": product_id,
                        "metadata": metadata,
                    }
                ),
                fit_score=item.fit_score,
                headline=item.headline,
                reasoning=item.reasoning,
            )
        )
        if len(sanitized) >= k:
            break
    return sanitized


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "recommendation"


def elapsed_ms(started_at: float) -> int:
    return max(0, int(round((time.perf_counter() - started_at) * 1000)))


@lru_cache
def get_recommendation_service() -> RecommendationService:
    settings = get_settings()
    embedding_model = SentenceTransformerEmbeddingModel(settings.embedding_model_name)
    return RecommendationService(
        retriever=ProductRetriever(
            catalogue_path=settings.recommendation_catalogue_path,
            embedding_model=embedding_model,
        ),
        coverage_judge=build_recommendation_coverage_judge(settings),
        generated_reasoner=build_generated_recommendation_reasoner(settings),
        ranker=build_recommendation_ranker(settings),
    )
