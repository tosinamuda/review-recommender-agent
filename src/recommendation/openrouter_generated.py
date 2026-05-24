from __future__ import annotations

import re
from collections.abc import Mapping

import dspy
from pydantic import BaseModel, Field

from app.settings import Settings

from .generated_contracts import GeneratedRecommendationReasoner
from .schemas import (
    CandidateProduct,
    CandidateProductSet,
    CoverageDecision,
    FallbackArchetype,
    GeneratedRecommendationResult,
    RecommendationItem,
)


class GeneratedRecommendationDraft(BaseModel):
    rank: int = Field(ge=1)
    name: str
    category: str
    description: str
    location: str | None = None
    fit_score: float = Field(ge=0.0, le=1.0)
    headline: str
    reasoning: str


class GeneratedRecommendationVerification(BaseModel):
    overall_pass: bool
    critique: str


class GenerateContextualRecommendations(dspy.Signature):
    """Generate useful non-catalogue recommendations when catalogue coverage is missing.

    These recommendations are advisory profiles, not concrete catalogue items.
    Do not name retrieved candidate businesses. Do not claim stable item IDs or
    catalogue grounding. Ground every recommendation in the persona and context.
    """

    user_persona: str = dspy.InputField()
    context: str = dspy.InputField()
    coverage_reason: str = dspy.InputField()
    unsupported_signals: list[str] = dspy.InputField()
    fallback_archetypes: list[FallbackArchetype] = dspy.InputField()
    k: int = dspy.InputField(desc="Return exactly this many generated recommendations.")
    persona_needs: list[str] = dspy.OutputField(
        desc="Two to four needs inferred from the persona/context."
    )
    recommendations: list[GeneratedRecommendationDraft] = dspy.OutputField(
        desc="Exactly k top generated recommendation profiles, not catalogue items."
    )


class VerifyGeneratedRecommendations(dspy.Signature):
    """Verify generated recommendations for no-coverage recommendation mode."""

    user_persona: str = dspy.InputField()
    context: str = dspy.InputField()
    k: int = dspy.InputField()
    generated_recommendations: list[GeneratedRecommendationDraft] = dspy.InputField()
    rejected_candidate_names: list[str] = dspy.InputField()
    overall_pass: bool = dspy.OutputField()
    critique: str = dspy.OutputField(
        desc=(
            "Explain any wrong recommendation count, leak of rejected candidate names, "
            "fake catalogue claims, missing persona/context fit, or invalid generated "
            "recommendation."
        )
    )


class GeneratedRecommendationModule(dspy.Module):
    """Generate no-coverage recommendations and refine them through verification."""

    def __init__(self, refine_attempts: int = 3):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateContextualRecommendations)
        self.verify = dspy.ChainOfThought(VerifyGeneratedRecommendations)
        self._refine_attempts = refine_attempts

    def forward(
        self,
        user_persona: str,
        context: str,
        coverage_reason: str,
        unsupported_signals: list[str],
        fallback_archetypes: list[FallbackArchetype],
        rejected_candidate_names: list[str],
        k: int,
    ) -> dspy.Prediction:
        last_verification = GeneratedRecommendationVerification(
            overall_pass=False,
            critique="No verification attempt completed.",
        )

        def reward(_args, prediction) -> float:
            nonlocal last_verification
            recommendations = [
                coerce_generated_draft(item)
                for item in getattr(prediction, "recommendations", [])
            ]
            if not recommendations:
                return 0.0
            if len(recommendations) != k:
                last_verification = GeneratedRecommendationVerification(
                    overall_pass=False,
                    critique=f"Expected exactly {k} recommendations, got {len(recommendations)}.",
                )
                return 0.0
            raw = self.verify(
                user_persona=user_persona,
                context=context,
                k=k,
                generated_recommendations=recommendations,
                rejected_candidate_names=rejected_candidate_names,
            )
            last_verification = GeneratedRecommendationVerification(
                overall_pass=bool(getattr(raw, "overall_pass", False)),
                critique=str(getattr(raw, "critique", "")).strip(),
            )
            return 1.0 if last_verification.overall_pass else 0.0

        refined = dspy.Refine(
            module=self.generate,
            N=self._refine_attempts,
            reward_fn=reward,
            threshold=1.0,
        )
        prediction = refined(
            user_persona=user_persona,
            context=context,
            coverage_reason=coverage_reason,
            unsupported_signals=unsupported_signals,
            fallback_archetypes=fallback_archetypes,
            k=k,
        )
        return dspy.Prediction(
            persona_needs=getattr(prediction, "persona_needs", []),
            recommendations=getattr(prediction, "recommendations", []),
            verifier_passed=last_verification.overall_pass,
            verifier_critique=last_verification.critique,
        )


class DSPyOpenRouterGeneratedRecommendationReasoner(GeneratedRecommendationReasoner):
    def __init__(self, settings: Settings):
        self.provider_name = f"dspy-litellm/{settings.lm_model}"
        self._module = GeneratedRecommendationModule()

    def generate(
        self,
        *,
        user_persona: str,
        context: str,
        candidate_set: CandidateProductSet,
        coverage_decision: CoverageDecision,
        k: int,
    ) -> GeneratedRecommendationResult:
        prediction = self._module(
            user_persona=user_persona,
            context=context,
            coverage_reason=coverage_decision.reason,
            unsupported_signals=coverage_decision.unsupported_signals,
            fallback_archetypes=coverage_decision.fallback_archetypes,
            rejected_candidate_names=[
                candidate.product.name for candidate in candidate_set.candidates
            ],
            k=k,
        )
        drafts = [
            coerce_generated_draft(item)
            for item in getattr(prediction, "recommendations", [])
        ]
        generated_items = [
            generated_item_from_draft(draft, index)
            for index, draft in enumerate(drafts, 1)
        ][:k]
        return GeneratedRecommendationResult(
            candidate_set=candidate_set,
            persona_needs=coerce_str_list(getattr(prediction, "persona_needs", [])),
            recommendations=generated_items,
        )


def generated_item_from_draft(
    draft: GeneratedRecommendationDraft,
    index: int,
) -> RecommendationItem:
    product_id = f"generated_{slugify(draft.category)}_{index:02d}"
    return RecommendationItem(
        rank=index,
        product=CandidateProduct(
            product_id=product_id,
            name=draft.name,
            category=draft.category,
            description=draft.description,
            location=draft.location,
            metadata={
                "catalogue_grounded": False,
                "grounding": "llm_generated",
            },
        ),
        fit_score=max(0.0, min(draft.fit_score, 1.0)),
        headline=draft.headline,
        reasoning=draft.reasoning,
    )


def coerce_generated_draft(value: object) -> GeneratedRecommendationDraft:
    if isinstance(value, GeneratedRecommendationDraft):
        return value
    if isinstance(value, BaseModel):
        return GeneratedRecommendationDraft.model_validate(value.model_dump())
    if isinstance(value, Mapping):
        return GeneratedRecommendationDraft.model_validate(value)
    return GeneratedRecommendationDraft.model_validate(vars(value))


def coerce_str_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if value:
        return [str(value)]
    return []


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "recommendation"
