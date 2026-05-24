from __future__ import annotations

from app.settings import Settings

from .coverage_contracts import RecommendationCoverageJudge
from .openrouter_coverage_judge import DSPyOpenRouterRecommendationCoverageJudge

__all__ = ["RecommendationCoverageJudge", "build_recommendation_coverage_judge"]


def build_recommendation_coverage_judge(settings: Settings) -> RecommendationCoverageJudge:
    return DSPyOpenRouterRecommendationCoverageJudge(settings)
