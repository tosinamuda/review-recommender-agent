from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import CandidateProductSet, RankingResult


class RecommendationRanker(ABC):
    provider_name: str

    @abstractmethod
    def rank(
        self,
        user_persona: str,
        context: str,
        candidate_set: CandidateProductSet,
        k: int,
    ) -> RankingResult:
        """Rank retrieved product candidates for a persona."""

