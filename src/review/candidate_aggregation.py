from __future__ import annotations

from .schemas import AggregatedCandidate, Candidates


def aggregate_candidates(candidates: Candidates) -> AggregatedCandidate:
    return AggregatedCandidate(
        candidates=candidates.samples,
        candidate_count=len(candidates.samples),
    )
