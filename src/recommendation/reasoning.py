from __future__ import annotations

from app.settings import Settings

from .openrouter_ranker import DSPyOpenRouterRecommendationRanker
from .ranker_contracts import RecommendationRanker

__all__ = ["RecommendationRanker", "build_recommendation_ranker"]


def build_recommendation_ranker(settings: Settings) -> RecommendationRanker:
    return DSPyOpenRouterRecommendationRanker(settings)

