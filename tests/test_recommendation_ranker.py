from __future__ import annotations

from recommendation.openrouter_ranker import (
    RankedProduct,
    coerce_recommendation_ranking,
)


def test_coerce_recommendation_ranking_accepts_dspy_pydantic_output() -> None:
    ranking = coerce_recommendation_ranking(
        RankedProduct(
            rank=1,
            product_id="food_iya_basira_amala",
            fit_score=0.92,
            headline="Strong match on budget and spice profile",
            reasoning="The product aligns with the persona's price and spice needs.",
        )
    )

    assert ranking.rank == 1
    assert ranking.product_id == "food_iya_basira_amala"
    assert ranking.fit_score == 0.92
    assert ranking.headline == "Strong match on budget and spice profile"


def test_coerce_recommendation_ranking_accepts_mapping_output() -> None:
    ranking = coerce_recommendation_ranking(
        {
            "rank": 2,
            "product_id": "food_chicken_republic_refuel",
            "fit_score": 0.86,
            "headline": "Reliable branded spice option",
            "reasoning": "The product is a safe fit for a quick weekday meal.",
        }
    )

    assert ranking.rank == 2
    assert ranking.product_id == "food_chicken_republic_refuel"
    assert ranking.fit_score == 0.86
