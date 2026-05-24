from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from recommendation.coverage_contracts import RecommendationCoverageJudge
from recommendation.generated_contracts import GeneratedRecommendationReasoner
from recommendation.ranker_contracts import RecommendationRanker
from recommendation.schemas import (
    CandidateProduct,
    CandidateProductSet,
    CoverageDecision,
    FallbackArchetype,
    GeneratedRecommendationResult,
    RankingResult,
    RecommendationItem,
    RecommendationRanking,
)
from review.rating_calibration import RatingCalibration, RatingCalibrator
from review.reasoner_contracts import ReviewReasoner
from review.schemas import (
    AggregatedCandidate,
    Candidate,
    Candidates,
    ExemplarSet,
    SelectedCandidate,
)
from shared.embeddings import (
    EmbeddingArray,
    EmbeddingModel,
    ensure_normalized_float32,
)


class HashingEmbeddingModel(EmbeddingModel):
    """Deterministic test embedding that avoids loading the production BGE model."""

    def __init__(self, dimension: int = 64, model_name: str = "test-hashing-embedding"):
        self.dimension = dimension
        self.model_name = model_name

    def encode(self, texts: Sequence[str]) -> EmbeddingArray:
        vectors = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row_index, text in enumerate(texts):
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimension
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row_index, bucket] += sign
        return ensure_normalized_float32(vectors)


class ContractReviewReasoner(ReviewReasoner):
    provider_name = "test-contract-reasoner"

    def generate_candidates(
        self,
        user_persona: str,
        product_details: str,
        exemplar_set: ExemplarSet,
        sample_count: int,
    ) -> Candidates:
        samples = []
        for index in range(max(1, min(sample_count, 3))):
            exemplar = exemplar_set.exemplars[index % len(exemplar_set.exemplars)]
            samples.append(
                Candidate(
                    review=contract_review(user_persona, product_details, exemplar),
                    chosen_experience=exemplar.product_issue or exemplar.review_text,
                    verifier_attempts=1,
                )
            )
        return Candidates(samples=samples)

    def select_best(
        self,
        user_persona: str,
        product_details: str,
        exemplar_set: ExemplarSet,
        aggregate: AggregatedCandidate,
    ) -> SelectedCandidate:
        del user_persona, product_details, exemplar_set
        candidate = aggregate.candidates[0]
        return SelectedCandidate(
            review=candidate.review,
            selection_reason="Test-only verbatim selection of first candidate.",
            chosen_experience=candidate.chosen_experience,
            verifier_attempts=candidate.verifier_attempts,
        )


class FixedReviewRatingCalibrator(RatingCalibrator):
    def calibrate(self, review: str) -> RatingCalibration:
        lowered = review.lower()
        if any(term in lowered for term in ("no pepper", "lack of pepper", "slow", "late")):
            distribution = [0.15, 0.35, 0.35, 0.10, 0.05]
        elif any(term in lowered for term in ("arrived hot", "large portion", "proper")):
            distribution = [0.02, 0.06, 0.22, 0.50, 0.20]
        else:
            distribution = [0.05, 0.15, 0.45, 0.25, 0.10]
        continuous = float(np.dot(np.asarray(distribution, dtype=np.float32), [1, 2, 3, 4, 5]))
        return RatingCalibration(
            rating=max(1, min(5, int(round(continuous)))),
            continuous=round(continuous, 4),
            distribution=distribution,
            reason="Rating calibrated by test-only fixed calibrator.",
        )


def contract_review(user_persona: str, product_details: str, exemplar) -> str:
    product_name = first_non_empty_line(product_details)
    persona_summary = " ".join(user_persona.split())
    product_summary = " ".join(product_details.split())
    return (
        f"{product_name} is being reviewed for this persona: {persona_summary}. "
        f"The product details say: {product_summary}. "
        f"A retrieved review anchor is {exemplar.exemplar_id}."
    )


def first_non_empty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            stripped = stripped.split(":", 1)[-1].strip() or stripped
            return stripped
    return text.strip()


class ContractRecommendationRanker(RecommendationRanker):
    provider_name = "test-contract-recommendation-ranker"

    def rank(
        self,
        user_persona: str,
        context: str,
        candidate_set: CandidateProductSet,
        k: int,
    ) -> RankingResult:
        del user_persona
        persona_needs = [
            "budget-friendly Nigerian option",
            "clear fit with the stated context",
            "strong product signal from the catalogue",
        ]
        if context:
            persona_needs.append(f"context: {context}")
        rankings = []
        for index, candidate in enumerate(candidate_set.candidates[:k], 1):
            product = candidate.product
            rankings.append(
                RecommendationRanking(
                    rank=index,
                    product_id=product.product_id,
                    fit_score=max(0.0, 0.95 - ((index - 1) * 0.04)),
                    headline=f"Good match for {product.category} needs",
                    reasoning=(
                        f"{product.name} matches the request through its catalogue "
                        f"description: {product.description}"
                    ),
                )
            )
        return RankingResult(
            candidate_set=candidate_set,
            persona_needs=persona_needs[:4],
            rankings=rankings,
        )


class ContractRecommendationCoverageJudge(RecommendationCoverageJudge):
    provider_name = "test-contract-recommendation-coverage-judge"

    def judge(
        self,
        *,
        user_persona: str,
        context: str,
        candidate_set: CandidateProductSet,
        coverage_policy: str,
    ) -> CoverageDecision:
        del user_persona, context, coverage_policy
        return CoverageDecision(
            coverage_status="sufficient",
            allow_concrete_recommendations=True,
            viable_product_ids=[
                candidate.product.product_id for candidate in candidate_set.candidates
            ],
            reason="Test-only coverage judge approved retrieved candidates.",
        )


class RejectingRecommendationCoverageJudge(RecommendationCoverageJudge):
    provider_name = "test-rejecting-recommendation-coverage-judge"

    def judge(
        self,
        *,
        user_persona: str,
        context: str,
        candidate_set: CandidateProductSet,
        coverage_policy: str,
    ) -> CoverageDecision:
        del user_persona, context, candidate_set, coverage_policy
        return CoverageDecision(
            coverage_status="insufficient",
            allow_concrete_recommendations=False,
            unsupported_signals=[
                "requested locale is not covered by retrieved candidates",
            ],
            fallback_archetypes=[
                FallbackArchetype(
                    rank=1,
                    archetype="budget spicy local lunch",
                    reason=(
                        "Matches the user's budget and spice preference, but no "
                        "concrete local catalogue item is available."
                    ),
                )
            ],
            reason="Retrieved candidates do not fit the requested local context.",
        )


class ContractGeneratedRecommendationReasoner(GeneratedRecommendationReasoner):
    provider_name = "test-contract-generated-recommendation-reasoner"

    def generate(
        self,
        *,
        user_persona: str,
        context: str,
        candidate_set: CandidateProductSet,
        coverage_decision: CoverageDecision,
        k: int,
    ) -> GeneratedRecommendationResult:
        del user_persona, coverage_decision
        names = [
            "Affordable spicy local eatery near Yaba",
            "Student-friendly pepper rice lunch spot",
            "Quick Nigerian fast-food spicy meal",
            "Filling amala and pepper soup option",
            "Budget suya or spicy shawarma stand",
        ]
        generated = []
        for index, name in enumerate(names[:k], 1):
            generated.append(
                RecommendationItem(
                    rank=index,
                    product=CandidateProduct(
                        product_id=f"generated_food_{index:02d}",
                        name=name,
                        category="food",
                        description="Generated recommendation profile, not a catalogue item.",
                        location="Yaba, Lagos" if context else None,
                        metadata={
                            "catalogue_grounded": False,
                            "grounding": "llm_generated",
                        },
                    ),
                    fit_score=max(0.0, 0.88 - ((index - 1) * 0.04)),
                    headline="Generated fit for a budget spicy weekday lunch",
                    reasoning=(
                        "This generated profile fits the budget-conscious student, "
                        "weekday lunch context, and spicy-food preference without "
                        "claiming a concrete catalogue business."
                    ),
                )
            )
        return GeneratedRecommendationResult(
            candidate_set=candidate_set,
            persona_needs=[
                "budget-conscious student",
                "spicy food",
                "weekday lunch near Yaba",
            ],
            recommendations=generated,
        )
