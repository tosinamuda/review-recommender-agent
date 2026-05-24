from __future__ import annotations

import dspy
from pydantic import BaseModel, Field

from app.settings import Settings

from .json_parsing import parse_int
from .reasoner_contracts import ReviewReasoner
from .schemas import (
    AggregatedCandidate,
    Candidate,
    Candidates,
    ExemplarSet,
    SelectedCandidate,
)


class SimilarUserReview(BaseModel):
    exemplar_id: str
    user_persona: str
    product_details: str
    review: str
    rating: int = Field(ge=1, le=5)


class CandidateDraftForSelection(BaseModel):
    draft_index: int = Field(ge=1)
    review: str
    chosen_experience: str
    verifier_passed: bool
    verifier_critique: str = ""


class ReviewVerification(BaseModel):
    persona_voice_preserved: bool
    product_details_consistent: bool
    experience_grounded: bool
    appropriate_length: bool
    overall_pass: bool
    critique: str = ""


class GenerateReview(dspy.Signature):
    """Write a realistic user review grounded in similar user reviews.

    Reason in this order:
    1. Identify 3-5 distinct experience types present in similar_user_reviews.
    2. Note the rating distribution across the retrieved set.
    3. Pick the experience type most plausible for this user persona.
    4. Write a fresh review in the persona's voice.

    Product details are authoritative. If retrieved reviews conflict with product details,
    follow product details. Do not invent or contradict product features.
    Return no final rating.
    """

    user_persona: str = dspy.InputField(
        desc="The caller-supplied user persona. Preserve its voice and preferences."
    )
    product_details: str = dspy.InputField(
        desc="The caller-supplied product facts. Treat these as authoritative."
    )
    similar_user_reviews: list[SimilarUserReview] = dspy.InputField(
        desc="Five retrieved user reviews used as behavioral analogues, not as product facts."
    )
    observed_experience_types: list[str] = dspy.OutputField(
        desc="Three to five experience patterns observed in the retrieved reviews."
    )
    chosen_experience: str = dspy.OutputField(
        desc="The one experience pattern selected for this persona and product."
    )
    review: str = dspy.OutputField(
        desc="The generated review text only. Do not include a rating."
    )


class VerifyReview(dspy.Signature):
    """Check whether a generated review preserves voice, grounding, and product truth."""

    user_persona: str = dspy.InputField(
        desc="The caller-supplied user persona used to judge voice and preference fit."
    )
    product_details: str = dspy.InputField(
        desc="The authoritative product facts. The review must not contradict these."
    )
    similar_user_reviews: list[SimilarUserReview] = dspy.InputField(
        desc="Retrieved behavioral analogues that ground plausible experiences."
    )
    review: str = dspy.InputField(desc="The candidate review to verify.")
    chosen_experience: str = dspy.InputField(
        desc="The experience pattern the candidate claims to represent."
    )
    persona_voice_preserved: bool = dspy.OutputField(
        desc="True only if the review preserves the persona's vocabulary and register."
    )
    product_details_consistent: bool = dspy.OutputField(
        desc="True only if the review does not invent or contradict product details."
    )
    experience_grounded: bool = dspy.OutputField(
        desc="True only if chosen_experience is supported by retrieved reviews."
    )
    appropriate_length: bool = dspy.OutputField(
        desc="True only if the review length fits the retrieved review style."
    )
    overall_pass: bool = dspy.OutputField(
        desc="True only if all verification checks pass."
    )
    critique: str = dspy.OutputField(desc="Concise explanation for any failed check.")


class SelectBestReview(dspy.Signature):
    """Select one generated draft verbatim; do not synthesize a new review."""

    user_persona: str = dspy.InputField(desc="The caller-supplied user persona.")
    product_details: str = dspy.InputField(desc="The authoritative product facts.")
    similar_user_reviews: list[SimilarUserReview] = dspy.InputField(
        desc="Retrieved behavioral analogues used to judge groundedness."
    )
    candidate_drafts: list[CandidateDraftForSelection] = dspy.InputField(
        desc="Generated drafts with verifier results. Prefer verifier-passed drafts."
    )
    best_draft_index: int = dspy.OutputField(
        desc="The 1-indexed draft_index of the selected candidate."
    )
    reason: str = dspy.OutputField(desc="Why this draft best fits persona and product.")


class ReviewWritingModule(dspy.Module):
    """Generate one verified review: ChainOfThought(generate) refined by ChainOfThought(verify)."""

    def __init__(self, refine_attempts: int = 3):
        super().__init__()
        self.generate = dspy.ChainOfThought(GenerateReview)
        self.verify = dspy.ChainOfThought(VerifyReview)
        self._refine_attempts = refine_attempts

    def forward(
        self,
        user_persona: str,
        product_details: str,
        similar_user_reviews: list[SimilarUserReview],
    ) -> dspy.Prediction:
        attempts = 0
        last_verification: ReviewVerification | None = None

        def reward(_args, prediction) -> float:
            nonlocal attempts, last_verification
            attempts += 1
            review = str(getattr(prediction, "review", "") or "")
            chosen_experience = str(getattr(prediction, "chosen_experience", "") or "")
            if not review or not chosen_experience:
                return 0.0
            raw = self.verify(
                user_persona=user_persona,
                product_details=product_details,
                similar_user_reviews=similar_user_reviews,
                review=review,
                chosen_experience=chosen_experience,
            )
            last_verification = verification_from_prediction(raw)
            return 1.0 if last_verification.overall_pass else 0.0

        refined = dspy.Refine(
            module=self.generate,
            N=self._refine_attempts,
            reward_fn=reward,
            threshold=1.0,
        )
        prediction = refined(
            user_persona=user_persona,
            product_details=product_details,
            similar_user_reviews=similar_user_reviews,
        )
        return dspy.Prediction(
            observed_experience_types=getattr(prediction, "observed_experience_types", []),
            chosen_experience=getattr(prediction, "chosen_experience", ""),
            review=getattr(prediction, "review", ""),
            verifier_attempts=max(1, attempts),
            verifier_passed=last_verification.overall_pass if last_verification else False,
            verifier_critique=last_verification.critique if last_verification else "",
        )


class DSPyOpenRouterReasoner(ReviewReasoner):
    def __init__(self, settings: Settings):
        self.provider_name = f"dspy-litellm/{settings.lm_model}"
        self._writer = ReviewWritingModule()
        self._select = dspy.ChainOfThought(SelectBestReview)
        self._parallel = dspy.Parallel(num_threads=3)

    def generate_candidates(
        self,
        user_persona: str,
        product_details: str,
        exemplar_set: ExemplarSet,
        sample_count: int,
    ) -> Candidates:
        sample_total = min(sample_count, 3)
        similar_user_reviews = similar_user_review_records(exemplar_set)
        exec_pairs = [
            (
                self._writer,
                {
                    "user_persona": user_persona,
                    "product_details": product_details,
                    "similar_user_reviews": similar_user_reviews,
                },
            )
            for _ in range(sample_total)
        ]
        predictions = self._parallel(exec_pairs)
        return Candidates(
            samples=[self._candidate_from_prediction(prediction) for prediction in predictions]
        )

    def select_best(
        self,
        user_persona: str,
        product_details: str,
        exemplar_set: ExemplarSet,
        aggregate: AggregatedCandidate,
    ) -> SelectedCandidate:
        result = self._select(
            user_persona=user_persona,
            product_details=product_details,
            similar_user_reviews=similar_user_review_records(exemplar_set),
            candidate_drafts=candidate_drafts_for_selection(aggregate),
        )
        index = max(
            1,
            min(parse_int(result.best_draft_index), len(aggregate.candidates)),
        )
        candidate = aggregate.candidates[index - 1]
        return SelectedCandidate(
            review=candidate.review,
            selection_reason=str(result.reason or f"Selected draft {index}."),
            chosen_experience=candidate.chosen_experience,
            verifier_attempts=candidate.verifier_attempts,
        )

    def _candidate_from_prediction(self, prediction) -> Candidate:
        return Candidate(
            review=str(prediction.review),
            observed_experience_types=coerce_str_list(prediction.observed_experience_types),
            chosen_experience=str(prediction.chosen_experience or ""),
            verifier_attempts=int(getattr(prediction, "verifier_attempts", 1)),
            verifier_passed=bool(getattr(prediction, "verifier_passed", False)),
            verifier_critique=str(getattr(prediction, "verifier_critique", "")),
        )


def similar_user_review_records(exemplar_set: ExemplarSet) -> list[SimilarUserReview]:
    return [
        SimilarUserReview(
            exemplar_id=exemplar.exemplar_id,
            user_persona=exemplar.user_profile_summary,
            product_details=exemplar.product_details or "",
            review=exemplar.review_text,
            rating=exemplar.rating,
        )
        for exemplar in exemplar_set.exemplars
    ]


def candidate_drafts_for_selection(
    aggregate: AggregatedCandidate,
) -> list[CandidateDraftForSelection]:
    return [
        CandidateDraftForSelection(
            draft_index=index,
            review=candidate.review,
            chosen_experience=candidate.chosen_experience,
            verifier_passed=candidate.verifier_passed,
            verifier_critique=candidate.verifier_critique,
        )
        for index, candidate in enumerate(aggregate.candidates, 1)
    ]


def verification_from_prediction(prediction) -> ReviewVerification:
    return ReviewVerification(
        persona_voice_preserved=coerce_bool(prediction.persona_voice_preserved),
        product_details_consistent=coerce_bool(prediction.product_details_consistent),
        experience_grounded=coerce_bool(prediction.experience_grounded),
        appropriate_length=coerce_bool(prediction.appropriate_length),
        overall_pass=coerce_bool(prediction.overall_pass),
        critique=str(prediction.critique or ""),
    )


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "pass", "passed"}
    return bool(value)


def coerce_str_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value is None:
        return []
    return [str(value)]
