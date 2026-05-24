from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import CandidateProductSet, CoverageDecision, GeneratedRecommendationResult


class GeneratedRecommendationReasoner(ABC):
    provider_name: str

    @abstractmethod
    def generate(
        self,
        *,
        user_persona: str,
        context: str,
        candidate_set: CandidateProductSet,
        coverage_decision: CoverageDecision,
        k: int,
    ) -> GeneratedRecommendationResult:
        """Generate non-catalogue recommendations when catalogue coverage is insufficient."""
