from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import CandidateProductSet, CoverageDecision


class RecommendationCoverageJudge(ABC):
    provider_name: str

    @abstractmethod
    def judge(
        self,
        *,
        user_persona: str,
        context: str,
        candidate_set: CandidateProductSet,
        coverage_policy: str,
    ) -> CoverageDecision:
        """Decide whether retrieved candidates are fit enough to recommend."""
