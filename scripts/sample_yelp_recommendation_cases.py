from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from recommendation.case_data import (
    StagedRecommendationCase,
    load_manifest,
    load_recommendation_cases,
    normalize_persona,
    write_recommendation_cases,
)
from recommendation.yelp_sampling import sample_yelp_recommendation_cases

DEFAULT_BUSINESS_PATH = Path("data/yelp_dataset/yelp_academic_dataset_business.json")
DEFAULT_REVIEW_PATH = Path("data/yelp_dataset/yelp_academic_dataset_review.json")
DEFAULT_OUTPUT_PATH = Path("data/generated/recommendation_cases_manual.jsonl")
DEFAULT_MANIFEST_PATH = Path("data/generated/recommendation_sample_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Yelp cases for Task 2 recommendations.")
    parser.add_argument("--business", type=Path, default=DEFAULT_BUSINESS_PATH)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEW_PATH)
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--min-reviews", type=int, default=4)
    parser.add_argument("--min-history-reviews", type=int, default=2)
    parser.add_argument("--exclude-source", type=Path, action="append", default=[])
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--case-id-prefix", default="yelp_rec")
    parser.add_argument("--derive-personas-from-history", action="store_true")
    args = parser.parse_args()

    excluded_user_ids, excluded_business_ids, excluded_review_ids = load_exclusions(
        source_paths=args.exclude_source,
        manifest_paths=args.exclude_manifest,
    )
    cases, manifest = sample_yelp_recommendation_cases(
        business_path=args.business,
        review_path=args.reviews,
        output_path=args.output,
        manifest_path=args.manifest,
        sample_size=args.sample_size,
        seed=args.seed,
        min_reviews=args.min_reviews,
        min_history_reviews=args.min_history_reviews,
        excluded_user_ids=excluded_user_ids,
        excluded_business_ids=excluded_business_ids,
        excluded_review_ids=excluded_review_ids,
        case_id_prefix=args.case_id_prefix,
    )
    if args.derive_personas_from_history:
        cases = [case_with_derived_persona(case) for case in cases]
        write_recommendation_cases(args.output, cases)
    print(
        json.dumps(
            {
                "cases": len(cases),
                "products": len(manifest.products),
                "output": str(args.output),
                "manifest": str(args.manifest),
                "seed": args.seed,
            },
            indent=2,
            sort_keys=True,
        )
    )


def load_exclusions(
    *,
    source_paths: list[Path],
    manifest_paths: list[Path],
) -> tuple[set[str], set[str], set[str]]:
    user_ids: set[str] = set()
    business_ids: set[str] = set()
    review_ids: set[str] = set()
    for source_path in source_paths:
        if not source_path.exists():
            continue
        for case in load_recommendation_cases(source_path):
            user_ids.add(case.sampling.source_user_id)
            review_ids.update(case.sampling.history_review_ids)
            review_ids.update(case.sampling.heldout_review_ids)
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        manifest = load_manifest(manifest_path)
        business_ids.update(manifest.product_id_map.values())
        business_ids.update(product.source_business_id for product in manifest.products)
    return user_ids, business_ids, review_ids


def case_with_derived_persona(case: StagedRecommendationCase) -> StagedRecommendationCase:
    return case.model_copy(update={"user_persona": derive_persona_from_history(case)})


def derive_persona_from_history(case: StagedRecommendationCase) -> str:
    place = dominant_place(case)
    categories = category_preferences(case)
    category_phrase = phrase_join(categories[:2]) if categories else "local options"
    persona = (
        f"{place} customer who prefers {category_phrase}, reliable service, "
        "and convenient local choices."
    )
    return normalize_persona(persona)


def dominant_place(case: StagedRecommendationCase) -> str:
    places = [
        " ".join(part for part in [item.city.replace(".", ""), item.state] if part)
        for item in case.persona_context.history
        if item.city or item.state
    ]
    if not places:
        return "Local"
    return Counter(places).most_common(1)[0][0]


def category_preferences(case: StagedRecommendationCase) -> list[str]:
    generic = {"food", "restaurants", "local flavor"}
    context_category = category_from_context(case.context)
    history = case.persona_context.history
    if context_category:
        context_history = [
            item
            for item in history
            if context_category in {category.lower() for category in item.categories}
        ]
        if context_history:
            history = context_history
    categories = [
        category.lower()
        for item in history
        if item.rating >= 4.0
        for category in item.categories
        if category.lower() not in generic
    ]
    if not categories:
        categories = [
            category.lower()
            for item in history
            for category in item.categories
            if category.lower() not in generic
        ]
    return [category for category, _ in Counter(categories).most_common(3)]


def category_from_context(context: str) -> str:
    normalized = " ".join(context.lower().split())
    if " in " not in normalized:
        return normalized
    category, _ = normalized.split(" in ", 1)
    return category.strip()


def phrase_join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return f"{', '.join(items[:-1])}, and {items[-1]}"


if __name__ == "__main__":
    main()
