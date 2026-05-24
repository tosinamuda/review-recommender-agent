from __future__ import annotations

from collections.abc import Mapping

import dspy
from pydantic import BaseModel, Field

from app.settings import Settings

from .ranker_contracts import RecommendationRanker
from .schemas import CandidateProductSet, RankingResult, RecommendationRanking


class CandidateProductForRanking(BaseModel):
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


class RankedProduct(BaseModel):
    rank: int = Field(ge=1)
    product_id: str
    fit_score: float = Field(ge=0.0, le=1.0)
    headline: str
    reasoning: str


class RankAndReason(dspy.Signature):
    """Rank candidate items by fit for this persona and explain every pick.

    Read all candidates before ranking. Use the persona and optional context as
    the authority for user needs. Do not invent product facts that are absent
    from candidate_products.

    Reason in this order:
    1. Identify the persona's strongest needs and constraints.
    2. Group candidates by how they address those needs.
    3. Pick the top k products that best serve the persona and context.
    4. Write one concise fit headline and one explanatory paragraph per pick.
    """

    user_persona: str = dspy.InputField()
    context: str = dspy.InputField()
    candidate_products: list[CandidateProductForRanking] = dspy.InputField()
    k: int = dspy.InputField()
    persona_needs: list[str] = dspy.OutputField(
        desc="Two to four user needs or constraints inferred from the persona."
    )
    rankings: list[RankedProduct] = dspy.OutputField(
        desc="Top k recommended products in ranked order."
    )


class DSPyOpenRouterRecommendationRanker(RecommendationRanker):
    def __init__(self, settings: Settings):
        self.provider_name = f"dspy-litellm/{settings.lm_model}"
        self._rank = dspy.ChainOfThought(RankAndReason)

    def rank(
        self,
        user_persona: str,
        context: str,
        candidate_set: CandidateProductSet,
        k: int,
    ) -> RankingResult:
        prediction = self._rank(
            user_persona=user_persona,
            context=context,
            candidate_products=[
                CandidateProductForRanking(
                    **candidate.product.model_dump(),
                    retrieval_score=candidate.score,
                    retrieval_signals=candidate.axis_scores,
                )
                for candidate in candidate_set.candidates
            ],
            k=k,
        )
        return RankingResult(
            candidate_set=candidate_set,
            persona_needs=coerce_str_list(getattr(prediction, "persona_needs", [])),
            rankings=[
                coerce_recommendation_ranking(ranking)
                for ranking in getattr(prediction, "rankings", [])
            ],
        )


def coerce_recommendation_ranking(value: object) -> RecommendationRanking:
    if isinstance(value, RecommendationRanking):
        return value
    if isinstance(value, BaseModel):
        return RecommendationRanking.model_validate(value.model_dump())
    if isinstance(value, Mapping):
        return RecommendationRanking.model_validate(value)
    return RecommendationRanking.model_validate(vars(value))


def coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []
