from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

MAX_PERSONA_WORDS = 28
WORD_RE = re.compile(r"\b[\w'-]+\b")
SENTENCE_END_RE = re.compile(r"[.!?]")
COMMON_PERSONA_ABBREVIATIONS = (
    "Dr.",
    "Mr.",
    "Mrs.",
    "Ms.",
    "Prof.",
    "Sr.",
    "Jr.",
    "St.",
    "Mt.",
    "Ave.",
    "Blvd.",
    "Rd.",
    "Ln.",
    "Ste.",
    "No.",
    "vs.",
    "e.g.",
    "i.e.",
)


class RecommendationHistoryItem(BaseModel):
    review_id: str
    product_id: str
    name: str
    categories: list[str] = Field(default_factory=list)
    city: str = ""
    state: str = ""
    rating: float
    date: str
    snippet: str = ""


class RecommendationPersonaContext(BaseModel):
    history: list[RecommendationHistoryItem] = Field(default_factory=list)


class RecommendationSamplingInfo(BaseModel):
    seed: int
    source_user_id: str
    history_review_ids: list[str]
    heldout_review_ids: list[str]


class StagedRecommendationCase(BaseModel):
    case_id: str
    source: str = "yelp"
    user_persona: str = ""
    context: str = ""
    persona_context: RecommendationPersonaContext
    history_product_ids: list[str]
    relevant_product_ids: list[str]
    sampling: RecommendationSamplingInfo


class RecommendationManifestProduct(BaseModel):
    product_id: str
    source_business_id: str
    name: str
    categories: list[str] = Field(default_factory=list)
    city: str = ""
    state: str = ""
    stars: float | None = None
    review_count: int | None = None
    is_open: int | None = None
    attributes: dict[str, Any] | None = None


class RecommendationSampleManifest(BaseModel):
    seed: int
    sample_size: int
    source_files: dict[str, str]
    selected_case_ids: list[str]
    sampling_rules: dict[str, Any]
    product_id_map: dict[str, str]
    products: list[RecommendationManifestProduct]


class RecommendationInteraction(BaseModel):
    case_id: str
    product_id: str
    rating: float
    date: str
    liked: bool
    split: str
    source: str = "yelp"


class RecommendationEvalCase(BaseModel):
    case_id: str
    user_persona: str
    context: str = ""
    relevant_product_ids: list[str]
    source: str = "yelp"
    expected_allow_concrete_recommendations: bool = True


def yelp_product_id(business_id: str) -> str:
    digest = hashlib.sha1(business_id.encode("utf-8")).hexdigest()[:16]
    return f"yelp_business_{digest}"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_recommendation_cases(path: Path) -> list[StagedRecommendationCase]:
    return [StagedRecommendationCase.model_validate(record) for record in load_jsonl(path)]


def write_recommendation_cases(
    path: Path,
    cases: list[StagedRecommendationCase],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(case.model_dump_json(exclude_none=True) for case in cases) + "\n",
        encoding="utf-8",
    )


def load_manifest(path: Path) -> RecommendationSampleManifest:
    if not path.exists():
        raise FileNotFoundError(f"Recommendation sample manifest not found: {path}")
    return RecommendationSampleManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: RecommendationSampleManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(exclude_none=True), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def normalize_persona(persona: str) -> str:
    normalized = " ".join(persona.strip().split())
    if not normalized:
        raise ValueError("Persona must not be empty.")
    if "\n" in persona:
        raise ValueError("Persona must be one line.")
    if _word_count(normalized) > MAX_PERSONA_WORDS:
        raise ValueError(
            f"Persona must be {MAX_PERSONA_WORDS} words or fewer: {normalized}"
        )
    if _has_internal_sentence_boundary(normalized):
        raise ValueError("Persona must be one sentence.")
    return normalized


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _has_internal_sentence_boundary(text: str) -> bool:
    scrubbed = text.rstrip(".!?")
    for abbreviation in COMMON_PERSONA_ABBREVIATIONS:
        scrubbed = re.sub(
            re.escape(abbreviation),
            abbreviation.replace(".", ""),
            scrubbed,
            flags=re.IGNORECASE,
        )
    scrubbed = re.sub(
        r"\b(?:[A-Z]\.){2,}",
        lambda match: match.group(0).replace(".", ""),
        scrubbed,
    )
    return bool(SENTENCE_END_RE.search(scrubbed))
