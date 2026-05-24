from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .case_data import (
    RecommendationHistoryItem,
    RecommendationManifestProduct,
    RecommendationPersonaContext,
    RecommendationSampleManifest,
    RecommendationSamplingInfo,
    StagedRecommendationCase,
    write_manifest,
    write_recommendation_cases,
    yelp_product_id,
)

CONSUMER_CATEGORY_TERMS = {
    "active life",
    "arts",
    "beauty",
    "coffee",
    "event planning",
    "food",
    "health",
    "local flavor",
    "local services",
    "nightlife",
    "restaurants",
    "shopping",
}


@dataclass(frozen=True)
class ContextSignature:
    category: str
    city: str
    state: str

    @property
    def text(self) -> str:
        if self.city or self.state:
            place = ", ".join(part for part in [self.city, self.state] if part)
            return f"{self.category.lower()} in {place}"
        return self.category.lower()


def sample_yelp_recommendation_cases(
    *,
    business_path: Path,
    review_path: Path,
    output_path: Path,
    manifest_path: Path,
    sample_size: int,
    seed: int,
    min_reviews: int = 4,
    min_history_reviews: int = 2,
    min_positive_rating: float = 4.0,
    oversample_factor: int = 12,
    excluded_user_ids: set[str] | None = None,
    excluded_business_ids: set[str] | None = None,
    excluded_review_ids: set[str] | None = None,
    case_id_prefix: str = "yelp_rec",
) -> tuple[list[StagedRecommendationCase], RecommendationSampleManifest]:
    excluded_user_ids = excluded_user_ids or set()
    excluded_business_ids = excluded_business_ids or set()
    excluded_review_ids = excluded_review_ids or set()
    businesses = {
        business_id: business
        for business_id, business in load_consumer_businesses(business_path).items()
        if business_id not in excluded_business_ids
    }
    review_counts = count_reviews_by_user(
        review_path,
        set(businesses),
        min_positive_rating,
        excluded_user_ids=excluded_user_ids,
        excluded_review_ids=excluded_review_ids,
    )
    candidate_user_ids = [
        user_id
        for user_id, counts in review_counts.items()
        if user_id not in excluded_user_ids
        and counts["reviews"] >= min_reviews
        and counts["positive"] >= 1
    ]
    rng = random.Random(seed)
    rng.shuffle(candidate_user_ids)
    selected_user_ids = candidate_user_ids[: max(sample_size * oversample_factor, sample_size)]
    reviews_by_user = collect_reviews_for_users(
        review_path=review_path,
        selected_user_ids=set(selected_user_ids),
        eligible_business_ids=set(businesses),
        excluded_review_ids=excluded_review_ids,
    )

    cases: list[_InternalStagedRecommendationCase] = []
    manifest_products: dict[str, RecommendationManifestProduct] = {}
    product_id_map: dict[str, str] = {}
    for user_id in selected_user_ids:
        case = build_case(
            case_number=len(cases) + 1,
            user_id=user_id,
            reviews=reviews_by_user.get(user_id, []),
            businesses=businesses,
            seed=seed,
            min_history_reviews=min_history_reviews,
            min_positive_rating=min_positive_rating,
            case_id_prefix=case_id_prefix,
        )
        if case is None:
            continue
        cases.append(case)
        for product_id in [*case.history_product_ids, *case.relevant_product_ids]:
            business_id = case.sampling_product_lookup[product_id]
            product_id_map[product_id] = business_id
            manifest_products[product_id] = business_to_manifest_product(
                product_id=product_id,
                business=businesses[business_id],
            )
        if len(cases) == sample_size:
            break

    if len(cases) != sample_size:
        raise ValueError(
            f"Could only build {len(cases)} recommendation cases from {review_path}; "
            f"requested {sample_size}."
        )

    finalized_cases = [strip_internal_case_fields(case) for case in cases]
    manifest = RecommendationSampleManifest(
        seed=seed,
        sample_size=sample_size,
        source_files={
            "business": str(business_path),
            "review": str(review_path),
        },
        selected_case_ids=[case.case_id for case in finalized_cases],
        sampling_rules={
            "min_reviews": min_reviews,
            "min_history_reviews": min_history_reviews,
            "min_positive_rating": min_positive_rating,
            "consumer_category_terms": sorted(CONSUMER_CATEGORY_TERMS),
        },
        product_id_map=product_id_map,
        products=sorted(manifest_products.values(), key=lambda product: product.product_id),
    )
    write_recommendation_cases(output_path, finalized_cases)
    write_manifest(manifest_path, manifest)
    return finalized_cases, manifest


def load_consumer_businesses(path: Path) -> dict[str, dict[str, Any]]:
    businesses: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        categories = parse_categories(record.get("categories"))
        if not categories or not is_consumer_business(categories):
            continue
        business_id = str(record["business_id"])
        businesses[business_id] = {**record, "category_list": categories}
    if not businesses:
        raise ValueError(f"No eligible consumer businesses found in {path}")
    return businesses


def count_reviews_by_user(
    review_path: Path,
    eligible_business_ids: set[str],
    min_positive_rating: float,
    *,
    excluded_user_ids: set[str] | None = None,
    excluded_review_ids: set[str] | None = None,
) -> dict[str, Counter]:
    excluded_user_ids = excluded_user_ids or set()
    excluded_review_ids = excluded_review_ids or set()
    counts: dict[str, Counter] = defaultdict(Counter)
    for review in iter_jsonl(review_path, ignore_invalid_tail=True):
        if review.get("review_id") in excluded_review_ids:
            continue
        user_id = str(review["user_id"])
        if user_id in excluded_user_ids:
            continue
        if review.get("business_id") not in eligible_business_ids:
            continue
        counts[user_id]["reviews"] += 1
        if float(review.get("stars") or 0.0) >= min_positive_rating:
            counts[user_id]["positive"] += 1
    return counts


def collect_reviews_for_users(
    *,
    review_path: Path,
    selected_user_ids: set[str],
    eligible_business_ids: set[str],
    excluded_review_ids: set[str] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    excluded_review_ids = excluded_review_ids or set()
    reviews_by_user: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for review in iter_jsonl(review_path, ignore_invalid_tail=True):
        if review.get("review_id") in excluded_review_ids:
            continue
        user_id = str(review.get("user_id"))
        if user_id not in selected_user_ids:
            continue
        if review.get("business_id") not in eligible_business_ids:
            continue
        reviews_by_user[user_id].append(review)
    return reviews_by_user


def build_case(
    *,
    case_number: int,
    user_id: str,
    reviews: list[dict[str, Any]],
    businesses: dict[str, dict[str, Any]],
    seed: int,
    min_history_reviews: int,
    min_positive_rating: float,
    case_id_prefix: str = "yelp_rec",
) -> _InternalStagedRecommendationCase | None:
    sorted_reviews = sorted(reviews, key=lambda review: str(review.get("date") or ""))
    split_index = preferred_split_index(
        sorted_reviews,
        min_history_reviews=min_history_reviews,
        min_positive_rating=min_positive_rating,
    )
    if split_index is None:
        return None
    history_reviews = sorted_reviews[:split_index]
    heldout_reviews = [
        review
        for review in sorted_reviews[split_index:]
        if float(review.get("stars") or 0.0) >= min_positive_rating
    ]
    if not history_reviews or not heldout_reviews:
        return None

    product_lookup: dict[str, str] = {}
    history_items = []
    history_product_ids = []
    for review in history_reviews:
        business = businesses[str(review["business_id"])]
        product_id = yelp_product_id(str(review["business_id"]))
        product_lookup[product_id] = str(review["business_id"])
        history_product_ids.append(product_id)
        history_items.append(review_to_history_item(review, business, product_id))

    context_signature = context_signature_from_history(history_items)
    matching_heldout_reviews = [
        review
        for review in heldout_reviews
        if business_matches_context(
            business=businesses[str(review["business_id"])],
            context_signature=context_signature,
        )
    ]
    if not matching_heldout_reviews:
        return None

    relevant_product_ids = []
    for review in matching_heldout_reviews:
        product_id = yelp_product_id(str(review["business_id"]))
        product_lookup[product_id] = str(review["business_id"])
        if product_id not in relevant_product_ids:
            relevant_product_ids.append(product_id)

    return _InternalStagedRecommendationCase(
        case_id=f"{case_id_prefix}_{case_number:06d}",
        source="yelp",
        user_persona="",
        context=context_signature.text,
        persona_context=RecommendationPersonaContext(history=history_items),
        history_product_ids=unique_preserve_order(history_product_ids),
        relevant_product_ids=relevant_product_ids,
        sampling=RecommendationSamplingInfo(
            seed=seed,
            source_user_id=user_id,
            history_review_ids=[str(review["review_id"]) for review in history_reviews],
            heldout_review_ids=[str(review["review_id"]) for review in matching_heldout_reviews],
        ),
        sampling_product_lookup=product_lookup,
    )


def preferred_split_index(
    reviews: list[dict[str, Any]],
    *,
    min_history_reviews: int,
    min_positive_rating: float,
) -> int | None:
    if len(reviews) <= min_history_reviews:
        return None
    preferred = max(min_history_reviews, int(round(len(reviews) * 0.6)))
    if any(
        float(review.get("stars") or 0.0) >= min_positive_rating
        for review in reviews[preferred:]
    ):
        return preferred
    positive_indexes = [
        index
        for index, review in enumerate(reviews)
        if index >= min_history_reviews and float(review.get("stars") or 0.0) >= min_positive_rating
    ]
    if not positive_indexes:
        return None
    return positive_indexes[0]


def review_to_history_item(
    review: dict[str, Any],
    business: dict[str, Any],
    product_id: str,
) -> RecommendationHistoryItem:
    return RecommendationHistoryItem(
        review_id=str(review["review_id"]),
        product_id=product_id,
        name=str(business.get("name") or ""),
        categories=list(business.get("category_list") or []),
        city=str(business.get("city") or ""),
        state=str(business.get("state") or ""),
        rating=float(review.get("stars") or 0.0),
        date=str(review.get("date") or ""),
        snippet=snippet(str(review.get("text") or "")),
    )


def business_to_manifest_product(
    *,
    product_id: str,
    business: dict[str, Any],
) -> RecommendationManifestProduct:
    return RecommendationManifestProduct(
        product_id=product_id,
        source_business_id=str(business["business_id"]),
        name=str(business.get("name") or ""),
        categories=list(business.get("category_list") or []),
        city=str(business.get("city") or ""),
        state=str(business.get("state") or ""),
        stars=float(business["stars"]) if business.get("stars") is not None else None,
        review_count=(
            int(business["review_count"])
            if business.get("review_count") is not None
            else None
        ),
        is_open=int(business["is_open"]) if business.get("is_open") is not None else None,
        attributes=business.get("attributes"),
    )


def context_from_history(history: list[RecommendationHistoryItem]) -> str:
    return context_signature_from_history(history).text


def context_signature_from_history(history: list[RecommendationHistoryItem]) -> ContextSignature:
    if not history:
        return ContextSignature(category="local options", city="", state="")
    category_counter = Counter(
        category
        for item in history
        for category in item.categories
        if category.lower() in CONSUMER_CATEGORY_TERMS
    )
    dominant_category = (
        category_counter.most_common(1)[0][0]
        if category_counter
        else "local options"
    )
    place_counter = Counter(
        (item.city, item.state)
        for item in history
        if item.city or item.state
    )
    dominant_city, dominant_state = (
        place_counter.most_common(1)[0][0] if place_counter else ("", "")
    )
    return ContextSignature(
        category=dominant_category,
        city=dominant_city,
        state=dominant_state,
    )


def business_matches_context(
    *,
    business: dict[str, Any],
    context_signature: ContextSignature,
) -> bool:
    categories = {category.lower() for category in business.get("category_list") or []}
    if context_signature.category.lower() not in categories:
        return False
    if context_signature.city and str(business.get("city") or "") != context_signature.city:
        return False
    if context_signature.state and str(business.get("state") or "") != context_signature.state:
        return False
    return True


def parse_categories(value: str | None) -> list[str]:
    if not value:
        return []
    return [category.strip() for category in value.split(",") if category.strip()]


def is_consumer_business(categories: Iterable[str]) -> bool:
    normalized = {category.lower() for category in categories}
    return any(category in normalized for category in CONSUMER_CATEGORY_TERMS)


def snippet(text: str, limit: int = 280) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 1].rstrip() + "..."


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def iter_jsonl(path: Path, *, ignore_invalid_tail: bool = False):
    if not path.exists():
        raise FileNotFoundError(f"Yelp source file not found: {path}")
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if ignore_invalid_tail and not source.read().strip():
                    return
                raise ValueError(f"Invalid JSON in {path} line {line_number}") from exc


class _InternalStagedRecommendationCase(StagedRecommendationCase):
    sampling_product_lookup: dict[str, str]


def strip_internal_case_fields(case: _InternalStagedRecommendationCase) -> StagedRecommendationCase:
    return StagedRecommendationCase.model_validate(
        case.model_dump(exclude={"sampling_product_lookup"})
    )
