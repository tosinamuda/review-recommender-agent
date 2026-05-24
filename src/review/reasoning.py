from __future__ import annotations

from app.settings import Settings

from .candidate_aggregation import aggregate_candidates
from .openrouter_reasoner import DSPyOpenRouterReasoner
from .reasoner_contracts import ReviewReasoner

__all__ = ["ReviewReasoner", "aggregate_candidates", "build_reasoner"]


def build_reasoner(settings: Settings) -> ReviewReasoner:
    return DSPyOpenRouterReasoner(settings)
