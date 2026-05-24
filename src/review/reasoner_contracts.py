from __future__ import annotations

from abc import ABC, abstractmethod

from .schemas import (
    AggregatedCandidate,
    Candidates,
    ExemplarSet,
    SelectedCandidate,
)


class ReviewReasoner(ABC):
    provider_name: str

    @abstractmethod
    def generate_candidates(
        self,
        user_persona: str,
        product_details: str,
        exemplar_set: ExemplarSet,
        sample_count: int,
    ) -> Candidates:
        """Generate candidate review drafts."""

    @abstractmethod
    def select_best(
        self,
        user_persona: str,
        product_details: str,
        exemplar_set: ExemplarSet,
        aggregate: AggregatedCandidate,
    ) -> SelectedCandidate:
        """Select the final review draft without assigning the final rating."""
