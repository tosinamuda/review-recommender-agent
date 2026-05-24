from __future__ import annotations

import statistics

from .schemas import Exemplar, ProxyStats


def proxy_stats(exemplars: list[Exemplar]) -> ProxyStats:
    if not exemplars:
        return ProxyStats(
            n_exemplars=0,
            mean_rating=3.0,
            weighted_mean_rating=3.0,
            std=0.0,
            strictness=0.0,
        )

    ratings = [exemplar.rating for exemplar in exemplars]
    weights = [max(exemplar.score, 0.001) for exemplar in exemplars]
    weighted_mean = sum(
        rating * weight for rating, weight in zip(ratings, weights, strict=True)
    ) / sum(weights)
    return ProxyStats(
        n_exemplars=len(exemplars),
        mean_rating=round(statistics.mean(ratings), 2),
        weighted_mean_rating=round(weighted_mean, 2),
        std=round(statistics.pstdev(ratings), 2) if len(ratings) > 1 else 0.0,
        strictness=round(max(0.0, 4.0 - statistics.mean(ratings)), 2),
    )
