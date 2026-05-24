from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from app.settings import Settings

from .coverage_contracts import RecommendationCoverageJudge
from .schemas import CandidateProductSet, CoverageDecision, FallbackArchetype


class CandidateProductForCoverage(BaseModel):
    product_id: str
    name: str
    category: str
    description: str
    price: float | None = None
    currency: str = "NGN"
    location: str | None = None
    metadata: dict = Field(default_factory=dict)
    retrieval_score: float = 0.0
    retrieval_signals: dict[str, float] = Field(default_factory=dict)


class CoverageArchetype(BaseModel):
    rank: int = Field(ge=1)
    archetype: str
    reason: str


class JudgeCandidateCoverage(dspy.Signature):
    """Judge whether retrieved candidate items can be recommended concretely.

    Use the persona and context as the authority for requested locale, domain,
    budget, and hard constraints. Approve only candidate IDs that are plausible
    concrete recommendations for the request. If retrieved candidates are only
    useful as preference proxies, set allow_concrete_recommendations to false and
    provide generic fallback archetypes without candidate names.
    """

    user_persona: str = dspy.InputField()
    context: str = dspy.InputField()
    candidate_products: list[CandidateProductForCoverage] = dspy.InputField()
    coverage_policy: str = dspy.InputField()
    coverage_status: str = dspy.OutputField(
        desc="One of: sufficient, partial, insufficient."
    )
    allow_concrete_recommendations: bool = dspy.OutputField()
    viable_product_ids: list[str] = dspy.OutputField(
        desc="Candidate product ids that may be passed to the ranker."
    )
    unsupported_signals: list[str] = dspy.OutputField(
        desc="Requested constraints or context not covered by the candidates."
    )
    fallback_archetypes: list[CoverageArchetype] = dspy.OutputField(
        desc="Generic non-concrete recommendation archetypes when coverage is insufficient."
    )
    reason: str = dspy.OutputField(
        desc="Short explanation of the coverage decision."
    )


class DSPyOpenRouterRecommendationCoverageJudge(RecommendationCoverageJudge):
    def __init__(self, settings: Settings):
        self.provider_name = f"dspy-litellm/{settings.lm_model}"
        self._judge = dspy.ChainOfThought(JudgeCandidateCoverage)

    def judge(
        self,
        *,
        user_persona: str,
        context: str,
        candidate_set: CandidateProductSet,
        coverage_policy: str,
    ) -> CoverageDecision:
        prediction = self._judge(
            user_persona=user_persona,
            context=context,
            candidate_products=[
                CandidateProductForCoverage(
                    **candidate.product.model_dump(),
                    retrieval_score=candidate.score,
                    retrieval_signals=candidate.axis_scores,
                )
                for candidate in candidate_set.candidates
            ],
            coverage_policy=coverage_policy,
        )
        return CoverageDecision(
            coverage_status=coerce_coverage_status(
                getattr(prediction, "coverage_status", "insufficient")
            ),
            allow_concrete_recommendations=bool(
                getattr(prediction, "allow_concrete_recommendations", False)
            ),
            viable_product_ids=coerce_str_list(getattr(prediction, "viable_product_ids", [])),
            unsupported_signals=coerce_str_list(
                getattr(prediction, "unsupported_signals", [])
            ),
            fallback_archetypes=[
                coerce_fallback_archetype(item)
                for item in getattr(prediction, "fallback_archetypes", [])
            ],
            reason=str(getattr(prediction, "reason", "")).strip(),
        )


def coerce_coverage_status(
    value: object,
) -> Literal["sufficient", "partial", "insufficient"]:
    normalized = str(value).strip().lower()
    if normalized == "sufficient":
        return "sufficient"
    if normalized == "partial":
        return "partial"
    return "insufficient"


def coerce_fallback_archetype(value: object) -> FallbackArchetype:
    if isinstance(value, FallbackArchetype):
        return value
    if isinstance(value, BaseModel):
        return FallbackArchetype.model_validate(value.model_dump())
    if isinstance(value, Mapping):
        return FallbackArchetype.model_validate(value)
    return FallbackArchetype.model_validate(vars(value))


def coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []
