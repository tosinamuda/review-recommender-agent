from __future__ import annotations

import json
import math
import re
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from .schemas import ReviewRequest, ReviewSimulationResponse

DEFAULT_EVAL_DATASET = Path("data/eval/review_simulation_eval.jsonl")


class ExpectedReviewOutcome(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    rating: int | None = Field(default=None, ge=1, le=5)
    rating_min: int = Field(ge=1, le=5)
    rating_max: int = Field(ge=1, le=5)
    reference_review: str | None = Field(
        default=None,
        validation_alias=AliasChoices("reference_review", "gold_review", "review"),
    )
    required_terms: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    minimum_score: float = Field(default=0.7, ge=0.0, le=1.0)


class ReviewEvalCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str = Field(min_length=1)
    notes: str = ""
    user_persona: str = Field(min_length=1)
    product_details: str = Field(min_length=1)
    expected: ExpectedReviewOutcome

    def to_request(self) -> ReviewRequest:
        return ReviewRequest(
            user_persona=self.user_persona,
            product_details=self.product_details,
        )


class ReviewEvalResult(BaseModel):
    case_id: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    failures: list[str]
    rating: int
    review: str
    rating_error: float | None = None
    rouge_l: float | None = None
    bertscore_f1: float | None = None

    @property
    def feedback(self) -> str:
        if not self.failures:
            return "Passes all rubric checks."
        return "Failures: " + "; ".join(self.failures)


def load_review_eval_cases(path: Path = DEFAULT_EVAL_DATASET) -> list[ReviewEvalCase]:
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            cases.append(parse_eval_case(line, path, line_number))
    if not cases:
        raise ValueError(f"Evaluation dataset is empty: {path}")
    return cases


def parse_eval_case(line: str, path: Path, line_number: int) -> ReviewEvalCase:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return ReviewEvalCase.model_validate(payload)


def score_review_response(
    case: ReviewEvalCase,
    response: ReviewSimulationResponse,
) -> ReviewEvalResult:
    failures = []
    expected = case.expected
    review = response.review.lower()

    rating_score = 1.0
    if not expected.rating_min <= response.rating <= expected.rating_max:
        rating_score = 0.0
        failures.append(
            f"rating {response.rating} outside expected range "
            f"{expected.rating_min}-{expected.rating_max}"
        )

    term_score, missing_terms = coverage_score(expected.required_terms, review)
    failures.extend(f"missing required term: {term}" for term in missing_terms)

    forbidden_hits = [term for term in expected.forbidden_terms if term.lower() in review]
    forbidden_score = 0.0 if forbidden_hits else 1.0
    failures.extend(f"forbidden term present: {term}" for term in forbidden_hits)

    score = round(
        0.40 * rating_score + 0.35 * term_score + 0.25 * forbidden_score,
        4,
    )
    return ReviewEvalResult(
        case_id=case.case_id,
        score=score,
        passed=score >= expected.minimum_score,
        failures=failures,
        rating=response.rating,
        review=response.review,
    )


def coverage_score(expected_terms: list[str], text: str) -> tuple[float, list[str]]:
    if not expected_terms:
        return 1.0, []
    lowered = text.lower()
    missing = [term for term in expected_terms if term.lower() not in lowered]
    hits = len(expected_terms) - len(missing)
    return hits / len(expected_terms), missing


def rouge_l_f1(candidate: str, reference: str) -> float:
    candidate_tokens = tokenize_for_text_metric(candidate)
    reference_tokens = tokenize_for_text_metric(reference)
    if not candidate_tokens or not reference_tokens:
        return 0.0
    lcs = longest_common_subsequence_length(candidate_tokens, reference_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(candidate_tokens)
    recall = lcs / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def rating_rmse(errors: list[float]) -> float | None:
    if not errors:
        return None
    return math.sqrt(sum(error**2 for error in errors) / len(errors))


def tokenize_for_text_metric(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def longest_common_subsequence_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, 1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]
