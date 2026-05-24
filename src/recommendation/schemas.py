from __future__ import annotations

from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class CandidateProduct(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    product_id: str = Field(validation_alias=AliasChoices("product_id", "item_id"))
    name: str
    category: str
    description: str
    price: float | None = None
    currency: str = "NGN"
    location: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def normalize(self) -> CandidateProduct:
        self.product_id = self.product_id.strip()
        self.name = self.name.strip()
        self.category = self.category.strip().lower()
        self.description = self.description.strip()
        self.currency = self.currency.strip().upper() if self.currency else ""
        self.location = self.location.strip() if self.location else None
        return self


class RecommendationRequest(BaseModel):
    user_persona: str = Field(min_length=1)
    context: str = ""
    category: str | None = None
    k: int = 5
    candidate_items: list[CandidateProduct] = Field(default_factory=list)
    coverage_policy: Literal["strict", "proxy_allowed"] = "strict"

    @model_validator(mode="after")
    def normalize(self) -> RecommendationRequest:
        self.user_persona = self.user_persona.strip()
        self.context = self.context.strip()
        self.category = self.category.strip().lower() if self.category else None
        self.k = max(1, min(int(self.k), 10))
        return self


class RetrievedCandidate(BaseModel):
    product: CandidateProduct
    score: float = Field(ge=0)
    axis_scores: dict[str, float] = Field(default_factory=dict)
    shortlisted: bool = False


class CandidateProductSet(BaseModel):
    candidates: list[RetrievedCandidate]
    retrieved_via: list[str]
    category: str | None = None
    candidate_source: Literal["built_in_catalogue", "request_supplied"] = "built_in_catalogue"


class RecommendationRanking(BaseModel):
    rank: int = Field(ge=1)
    product_id: str
    fit_score: float = Field(ge=0.0, le=1.0)
    headline: str
    reasoning: str


class RankingResult(BaseModel):
    candidate_set: CandidateProductSet
    persona_needs: list[str] = Field(default_factory=list)
    rankings: list[RecommendationRanking] = Field(default_factory=list)


class FallbackArchetype(BaseModel):
    rank: int = Field(ge=1)
    archetype: str
    reason: str


class CoverageDecision(BaseModel):
    coverage_status: Literal["sufficient", "partial", "insufficient"]
    allow_concrete_recommendations: bool
    viable_product_ids: list[str] = Field(default_factory=list)
    unsupported_signals: list[str] = Field(default_factory=list)
    fallback_archetypes: list[FallbackArchetype] = Field(default_factory=list)
    reason: str = ""

    @model_validator(mode="after")
    def normalize(self) -> CoverageDecision:
        self.viable_product_ids = [
            product_id.strip() for product_id in self.viable_product_ids if product_id.strip()
        ]
        self.unsupported_signals = [
            signal.strip() for signal in self.unsupported_signals if signal.strip()
        ]
        self.reason = self.reason.strip()
        if not self.viable_product_ids:
            self.allow_concrete_recommendations = False
        if self.coverage_status == "insufficient":
            self.allow_concrete_recommendations = False
            self.viable_product_ids = []
        return self


class RecommendationItem(BaseModel):
    rank: int = Field(ge=1)
    product: CandidateProduct
    fit_score: float = Field(ge=0.0, le=1.0)
    headline: str
    reasoning: str


class GeneratedRecommendationResult(BaseModel):
    candidate_set: CandidateProductSet
    persona_needs: list[str] = Field(default_factory=list)
    recommendations: list[RecommendationItem] = Field(default_factory=list)


class RecommendationCoverage(BaseModel):
    status: Literal["sufficient", "partial", "insufficient"]
    allow_concrete_recommendations: bool
    candidate_source: Literal["built_in_catalogue", "request_supplied"]
    viable_product_ids: list[str] = Field(default_factory=list)
    unsupported_signals: list[str] = Field(default_factory=list)
    reason: str = ""


class RecommendationTraceEvent(BaseModel):
    stage: str
    message: str
    duration_ms: int | None = None


class RecommendationEvidence(BaseModel):
    user_persona: str
    context: str = ""
    persona_needs: list[str] = Field(default_factory=list)
    candidate_count: int = 0
    retrieved_via: list[str] = Field(default_factory=list)
    candidates: list[RetrievedCandidate] = Field(default_factory=list)
    returned_product_ids: list[str] = Field(default_factory=list)
    note: str = ""
    elapsed_ms: int = 0
    model_provider: str


class RecommendationResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    recommendation_mode: Literal[
        "catalogue_grounded",
        "request_supplied_candidates",
        "coverage_limited",
        "llm_generated",
    ] = "catalogue_grounded"
    coverage: RecommendationCoverage
    recommendations: list[RecommendationItem]
    generated_recommendations: list[RecommendationItem] = Field(default_factory=list)
    fallback_archetypes: list[FallbackArchetype] = Field(default_factory=list)
    evidence: RecommendationEvidence
    trace: list[RecommendationTraceEvent]
