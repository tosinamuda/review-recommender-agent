from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .case_data import (
    MAX_PERSONA_WORDS,
    RecommendationManifestProduct,
    StagedRecommendationCase,
    load_manifest,
    load_recommendation_cases,
    normalize_persona,
)


@dataclass(frozen=True)
class RecommendationCaseValidationResult:
    case_count: int
    product_count: int
    failures: list[str]

    @property
    def ok(self) -> bool:
        return not self.failures


def validate_recommendation_cases(
    *,
    source_path: Path,
    manifest_path: Path,
    require_personas: bool = True,
    extra_product_ids: set[str] | None = None,
    extra_product_names: Mapping[str, str] | None = None,
    require_manifest_case_ids: bool = True,
    enforce_context_match: bool = False,
) -> RecommendationCaseValidationResult:
    cases = load_recommendation_cases(source_path)
    manifest = load_manifest(manifest_path)
    product_ids = {product.product_id for product in manifest.products}
    product_names = {product.product_id: product.name for product in manifest.products}
    products_by_id = {product.product_id: product for product in manifest.products}
    if extra_product_ids:
        product_ids.update(extra_product_ids)
    if extra_product_names:
        product_names.update(extra_product_names)
    failures: list[str] = []

    case_ids: set[str] = set()
    for case in cases:
        if case.case_id in case_ids:
            failures.append(f"duplicate case_id: {case.case_id}")
        case_ids.add(case.case_id)
        failures.extend(
            validate_case(
                case,
                product_ids,
                product_names=product_names,
                products_by_id=products_by_id,
                require_personas=require_personas,
                enforce_context_match=enforce_context_match,
            )
        )

    manifest_case_ids = set(manifest.selected_case_ids)
    missing_from_manifest = sorted(case_ids - manifest_case_ids)
    if require_manifest_case_ids and missing_from_manifest:
        failures.append(f"case ids missing from manifest: {', '.join(missing_from_manifest[:5])}")
    duplicate_products = find_duplicates([product.product_id for product in manifest.products])
    for product_id in duplicate_products:
        failures.append(f"duplicate product_id in manifest: {product_id}")

    return RecommendationCaseValidationResult(
        case_count=len(cases),
        product_count=len(product_ids),
        failures=failures,
    )


def validate_case(
    case: StagedRecommendationCase,
    product_ids: set[str],
    *,
    product_names: Mapping[str, str] | None = None,
    products_by_id: Mapping[str, RecommendationManifestProduct] | None = None,
    require_personas: bool,
    enforce_context_match: bool = False,
) -> list[str]:
    failures: list[str] = []
    if require_personas:
        try:
            normalize_persona(case.user_persona)
        except ValueError as exc:
            failures.append(f"{case.case_id}: {exc}")
    elif case.user_persona.strip():
        try:
            normalize_persona(case.user_persona)
        except ValueError as exc:
            failures.append(f"{case.case_id}: {exc}")

    if not case.relevant_product_ids:
        failures.append(f"{case.case_id}: relevant_product_ids must not be empty")
    failures.extend(validate_resolved_product_ids(case, product_ids))
    failures.extend(validate_heldout_context_leakage(case))
    failures.extend(validate_relevant_product_name_leakage(case, product_names or {}))
    if enforce_context_match:
        failures.extend(validate_relevant_products_match_context(case, products_by_id or {}))
    return failures


def validate_resolved_product_ids(
    case: StagedRecommendationCase,
    product_ids: set[str],
) -> list[str]:
    failures: list[str] = []
    for field_name, ids in [
        ("history_product_ids", case.history_product_ids),
        ("relevant_product_ids", case.relevant_product_ids),
    ]:
        unresolved = [product_id for product_id in ids if product_id not in product_ids]
        if unresolved:
            failures.append(
                f"{case.case_id}: unresolved {field_name}: {', '.join(unresolved[:5])}"
            )
    return failures


def validate_heldout_context_leakage(case: StagedRecommendationCase) -> list[str]:
    failures: list[str] = []
    history_review_ids = {item.review_id for item in case.persona_context.history}
    leaked_review_ids = history_review_ids.intersection(case.sampling.heldout_review_ids)
    if leaked_review_ids:
        failures.append(
            f"{case.case_id}: held-out review leaked into persona_context: "
            f"{', '.join(sorted(leaked_review_ids))}"
        )

    context_blob = json.dumps(case.persona_context.model_dump(), sort_keys=True)
    leaked_product_ids = [
        product_id
        for product_id in case.relevant_product_ids
        if product_id in context_blob and product_id not in case.history_product_ids
    ]
    if leaked_product_ids:
        failures.append(
            f"{case.case_id}: held-out product leaked into persona_context: "
            f"{', '.join(leaked_product_ids[:5])}"
        )
    return failures


def validate_relevant_product_name_leakage(
    case: StagedRecommendationCase,
    product_names: Mapping[str, str],
) -> list[str]:
    failures: list[str] = []
    leaked_product_names = find_relevant_product_name_leaks(case=case, product_names=product_names)
    if leaked_product_names:
        failures.append(
            f"{case.case_id}: relevant product name leaked into persona/context: "
            f"{', '.join(leaked_product_names[:5])}"
        )
    return failures


def validate_relevant_products_match_context(
    case: StagedRecommendationCase,
    products_by_id: Mapping[str, RecommendationManifestProduct],
) -> list[str]:
    expected_context = parse_case_context(case.context)
    if expected_context is None:
        return []
    expected_category, expected_city, expected_state = expected_context
    mismatched_products = []
    for product_id in case.relevant_product_ids:
        product = products_by_id.get(product_id)
        if product is None:
            continue
        product_categories = {category.casefold() for category in product.categories}
        category_matches = expected_category in product_categories
        city_matches = not expected_city or product.city.casefold() == expected_city
        state_matches = not expected_state or product.state.casefold() == expected_state
        if not (category_matches and city_matches and state_matches):
            mismatched_products.append(product.name or product_id)
    if not mismatched_products:
        return []
    return [
        f"{case.case_id}: relevant product does not match context {case.context!r}: "
        f"{', '.join(mismatched_products[:5])}"
    ]


def parse_case_context(context: str) -> tuple[str, str, str] | None:
    normalized = " ".join(context.casefold().split())
    if not normalized:
        return None
    if " in " not in normalized:
        return normalized, "", ""
    category, place = normalized.split(" in ", 1)
    if not category.strip():
        return None
    city, separator, state = place.rpartition(", ")
    if not separator:
        return category.strip(), place.strip(), ""
    return category.strip(), city.strip(), state.strip()


def find_relevant_product_name_leaks(
    *,
    case: StagedRecommendationCase,
    product_names: Mapping[str, str],
) -> list[str]:
    visible_text = normalize_leakage_text(f"{case.user_persona} {case.context}")
    leaks: list[str] = []
    for product_id in case.relevant_product_ids:
        product_name = product_names.get(product_id, "")
        normalized_name = normalize_leakage_text(product_name)
        if len(normalized_name) < 4:
            continue
        if re.search(rf"\b{re.escape(normalized_name)}\b", visible_text):
            leaks.append(product_name)
    return leaks


def normalize_leakage_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def find_duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return duplicates


def validation_summary(result: RecommendationCaseValidationResult) -> dict:
    return {
        "ok": result.ok,
        "case_count": result.case_count,
        "product_count": result.product_count,
        "max_persona_words": MAX_PERSONA_WORDS,
        "failures": result.failures,
    }
