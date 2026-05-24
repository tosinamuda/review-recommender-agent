from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SimulationOptions(BaseModel):
    sample_count: int = Field(default=3, ge=1, le=12)


class ReviewRequest(BaseModel):
    user_persona: str = Field(min_length=1)
    product_details: str = Field(min_length=1)
    options: SimulationOptions = Field(default_factory=SimulationOptions)


class Exemplar(BaseModel):
    exemplar_id: str
    review_text: str
    rating: int = Field(ge=1, le=5)
    user_profile_summary: str
    item_category: str
    score: float = Field(ge=0)
    source: str = "seed_corpus"
    product_details: str | None = None
    product_issue: str | None = None
    axis_scores: dict[str, float] = Field(default_factory=dict)


class ProxyStats(BaseModel):
    n_exemplars: int
    mean_rating: float
    weighted_mean_rating: float
    std: float
    strictness: float
    common_positive_aspects: list[str] = Field(default_factory=list)
    common_negative_aspects: list[str] = Field(default_factory=list)


class ExemplarSet(BaseModel):
    exemplars: list[Exemplar]
    proxy_stats: ProxyStats


class Candidate(BaseModel):
    review: str = Field(min_length=1)
    chosen_experience: str = ""
    observed_experience_types: list[str] = Field(default_factory=list)
    verifier_attempts: int = Field(default=1, ge=1)
    verifier_passed: bool = True
    verifier_critique: str = ""


class Candidates(BaseModel):
    samples: list[Candidate]


class AggregatedCandidate(BaseModel):
    candidates: list[Candidate]
    candidate_count: int = Field(ge=1)


class SelectedCandidate(BaseModel):
    review: str
    selection_reason: str
    chosen_experience: str = ""
    verifier_attempts: int = Field(default=1, ge=1)


class CalibratedCandidate(BaseModel):
    rating: int = Field(ge=1, le=5)
    review: str
    selection_reason: str
    calibration_reason: str
    rating_continuous: float
    rating_distribution: list[float]
    chosen_experience: str = ""
    verifier_attempts: int = Field(default=1, ge=1)


class TraceEvent(BaseModel):
    stage: str
    message: str


class ResponseEvidence(BaseModel):
    user_persona: str
    product_details: str
    similar_user_reviews: list[Exemplar] = Field(default_factory=list)
    candidate_reviews: list[str] = Field(default_factory=list)
    reason: str = ""
    rating_continuous: float | None = None
    rating_distribution: list[float] = Field(default_factory=list)
    verifier_attempts: int | None = None
    model_provider: str


class ReviewSimulationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    rating: int = Field(ge=1, le=5)
    review: str
    evidence: ResponseEvidence
    trace: list[TraceEvent]
