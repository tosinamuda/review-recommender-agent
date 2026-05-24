from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .case_data import StagedRecommendationCase, load_recommendation_cases
from .catalogue import load_product_catalogue
from .schemas import CandidateProduct

WORD_RE = re.compile(r"[a-z0-9]+")
GENERIC_NAME_TOKENS = {
    "abuja",
    "africa",
    "african",
    "airtime",
    "amala",
    "app",
    "bank",
    "banking",
    "bistro",
    "bole",
    "bukka",
    "business",
    "cafe",
    "canteen",
    "centre",
    "church",
    "cinema",
    "cinemas",
    "clinic",
    "college",
    "connect",
    "delivery",
    "digital",
    "education",
    "ewedu",
    "family",
    "fitness",
    "food",
    "foods",
    "gari",
    "gbegiri",
    "global",
    "group",
    "harcourt",
    "health",
    "hospital",
    "house",
    "ibadan",
    "ikeja",
    "jabi",
    "kano",
    "lagos",
    "lake",
    "mall",
    "market",
    "mobile",
    "national",
    "network",
    "nigeria",
    "nigerian",
    "online",
    "park",
    "pharmacy",
    "place",
    "plaza",
    "point",
    "port",
    "restaurant",
    "retail",
    "service",
    "services",
    "shopping",
    "spaghetti",
    "state",
    "streaming",
    "technology",
    "theatre",
    "university",
    "wallet",
    "work",
    "yaba",
}


@dataclass(frozen=True)
class PersonaLeakageFinding:
    case_id: str
    product_id: str
    product_name: str
    match_type: Literal["exact_product_name", "distinctive_name_token"]
    matched_text: str
    user_persona: str


def audit_persona_business_leakage(
    *,
    cases_path: Path,
    catalogue_path: Path,
) -> list[PersonaLeakageFinding]:
    products_by_id = {
        product.product_id: product for product in load_product_catalogue(catalogue_path)
    }
    findings: list[PersonaLeakageFinding] = []
    for case in load_recommendation_cases(cases_path):
        persona_tokens = set(normalized_tokens(case.user_persona))
        normalized_persona = normalize_text(case.user_persona)
        for product_id in linked_product_ids(case):
            product = products_by_id.get(product_id)
            if product is None:
                continue
            findings.extend(
                findings_for_product(
                    case=case,
                    product=product,
                    normalized_persona=normalized_persona,
                    persona_tokens=persona_tokens,
                )
            )
    return findings


def findings_for_product(
    *,
    case: StagedRecommendationCase,
    product: CandidateProduct,
    normalized_persona: str,
    persona_tokens: set[str],
) -> list[PersonaLeakageFinding]:
    product_phrase = normalize_text(product.name)
    findings: list[PersonaLeakageFinding] = []
    if product_phrase and product_phrase in normalized_persona:
        findings.append(
            PersonaLeakageFinding(
                case_id=case.case_id,
                product_id=product.product_id,
                product_name=product.name,
                match_type="exact_product_name",
                matched_text=product.name,
                user_persona=case.user_persona,
            )
        )
        return findings

    for token in distinctive_name_tokens(product.name):
        if token in persona_tokens:
            findings.append(
                PersonaLeakageFinding(
                    case_id=case.case_id,
                    product_id=product.product_id,
                    product_name=product.name,
                    match_type="distinctive_name_token",
                    matched_text=token,
                    user_persona=case.user_persona,
                )
            )
    return findings


def linked_product_ids(case: StagedRecommendationCase) -> list[str]:
    seen: set[str] = set()
    product_ids: list[str] = []
    for product_id in [*case.history_product_ids, *case.relevant_product_ids]:
        if product_id not in seen:
            seen.add(product_id)
            product_ids.append(product_id)
    return product_ids


def distinctive_name_tokens(name: str) -> set[str]:
    return {
        token
        for token in normalized_tokens(name)
        if len(token) >= 4 and token not in GENERIC_NAME_TOKENS
    }


def normalize_text(value: str) -> str:
    return " ".join(normalized_tokens(value))


def normalized_tokens(value: str) -> list[str]:
    return [match.group(0) for match in WORD_RE.finditer(value.lower())]
