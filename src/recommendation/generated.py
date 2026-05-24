from __future__ import annotations

from app.settings import Settings

from .generated_contracts import GeneratedRecommendationReasoner
from .openrouter_generated import DSPyOpenRouterGeneratedRecommendationReasoner

__all__ = ["GeneratedRecommendationReasoner", "build_generated_recommendation_reasoner"]


def build_generated_recommendation_reasoner(
    settings: Settings,
) -> GeneratedRecommendationReasoner:
    return DSPyOpenRouterGeneratedRecommendationReasoner(settings)
