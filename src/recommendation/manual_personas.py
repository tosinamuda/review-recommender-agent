from __future__ import annotations

import json
from pathlib import Path

from .case_data import (
    StagedRecommendationCase,
    load_recommendation_cases,
    normalize_persona,
    write_recommendation_cases,
)

NO_EMPTY_RECOMMENDATION_PERSONA_MESSAGE = "No empty recommendation persona remains."


def display_case(path: Path, case_id: str | None = None, next_empty: bool = False) -> dict:
    cases = load_recommendation_cases(path)
    case = find_case_for_display(cases, case_id=case_id, next_empty=next_empty)
    return {
        "case_id": case.case_id,
        "current_persona": case.user_persona,
        "context": case.context,
        "history_product_ids": case.history_product_ids,
        "relevant_product_count": len(case.relevant_product_ids),
        "history": [
            {
                "product_id": item.product_id,
                "name": item.name,
                "categories": item.categories,
                "location": ", ".join(part for part in [item.city, item.state] if part),
                "rating": item.rating,
                "date": item.date,
                "snippet": item.snippet,
            }
            for item in case.persona_context.history
        ],
    }


def set_case_persona(
    path: Path,
    *,
    case_id: str,
    persona: str,
    force: bool = False,
) -> StagedRecommendationCase:
    cases = load_recommendation_cases(path)
    seen_ids: set[str] = set()
    target_index: int | None = None
    for index, case in enumerate(cases):
        if case.case_id in seen_ids:
            raise ValueError(f"Duplicate case_id in {path}: {case.case_id}")
        seen_ids.add(case.case_id)
        if case.case_id == case_id:
            target_index = index
    if target_index is None:
        raise ValueError(f"Unknown case_id: {case_id}")

    target = cases[target_index]
    if target.user_persona.strip() and not force:
        raise ValueError(f"{case_id} already has a persona. Pass --force to overwrite.")

    updated = target.model_copy(update={"user_persona": normalize_persona(persona)})
    cases[target_index] = updated
    write_recommendation_cases(path, cases)
    return updated


def find_case_for_display(
    cases: list[StagedRecommendationCase],
    *,
    case_id: str | None,
    next_empty: bool,
) -> StagedRecommendationCase:
    if bool(case_id) == bool(next_empty):
        raise ValueError("Pass exactly one of case_id or --next-empty.")
    seen_ids: set[str] = set()
    for case in cases:
        if case.case_id in seen_ids:
            raise ValueError(f"Duplicate case_id in source: {case.case_id}")
        seen_ids.add(case.case_id)
    if next_empty:
        for case in cases:
            if not case.user_persona.strip():
                return case
        raise ValueError(NO_EMPTY_RECOMMENDATION_PERSONA_MESSAGE)
    matches = [case for case in cases if case.case_id == case_id]
    if not matches:
        raise ValueError(f"Unknown case_id: {case_id}")
    return matches[0]


def dumps_pretty(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)
