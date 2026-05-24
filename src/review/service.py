from __future__ import annotations

from functools import lru_cache

from app.settings import get_settings
from shared.embeddings import SentenceTransformerEmbeddingModel

from .rating_calibration import (
    RatingCalibrator,
    SklearnOrdinalRatingCalibrator,
    calibrate_rating_from_review,
)
from .reasoning import ReviewReasoner, aggregate_candidates, build_reasoner
from .retrieval import BehavioralRetriever
from .schemas import (
    AggregatedCandidate,
    CalibratedCandidate,
    Candidates,
    ExemplarSet,
    ResponseEvidence,
    ReviewRequest,
    ReviewSimulationResponse,
    SelectedCandidate,
    TraceEvent,
)


class ReviewSimulationService:
    def __init__(
        self,
        retriever: BehavioralRetriever,
        reasoner: ReviewReasoner,
        rating_calibrator: RatingCalibrator | None = None,
    ):
        self._retriever = retriever
        self._reasoner = reasoner
        self._rating_calibrator = rating_calibrator or SklearnOrdinalRatingCalibrator()

    @property
    def provider_name(self) -> str:
        return self._reasoner.provider_name

    def run(self, request: ReviewRequest) -> ReviewSimulationResponse:
        trace: list[TraceEvent] = []

        exemplar_set = self.retrieve(request.user_persona, request.product_details)
        trace.append(
            TraceEvent(
                stage="retrieve_similar_user_reviews",
                message=(
                    f"Retrieved {len(exemplar_set.exemplars)} similar user reviews "
                    "as the user_review_history proxy."
                ),
            )
        )

        candidates = self.generate_candidates(
            request.user_persona,
            request.product_details,
            exemplar_set,
            sample_count=request.options.sample_count,
        )
        trace.append(
            TraceEvent(
                stage="generate_review_with_refinement",
                message=(
                    f"Generated {len(candidates.samples)} candidate drafts through "
                    "the review refinement boundary."
                ),
            )
        )

        aggregate = aggregate_candidates(candidates)
        selected = self.select_best(
            request.user_persona,
            request.product_details,
            exemplar_set,
            aggregate,
        )
        trace.append(
            TraceEvent(
                stage="select_best_review_for_persona",
                message=(
                    f"Selected one review from {aggregate.candidate_count} candidate drafts."
                ),
            )
        )

        calibrated = self.calibrate_rating(selected)
        trace.append(
            TraceEvent(
                stage="calibrate_rating_from_review",
                message=calibrated.calibration_reason,
            )
        )

        return ReviewSimulationResponse(
            rating=calibrated.rating,
            review=calibrated.review,
            evidence=ResponseEvidence(
                user_persona=request.user_persona,
                product_details=request.product_details,
                similar_user_reviews=exemplar_set.exemplars,
                candidate_reviews=[candidate.review for candidate in candidates.samples],
                reason=combined_reason(calibrated),
                rating_continuous=calibrated.rating_continuous,
                rating_distribution=calibrated.rating_distribution,
                verifier_attempts=calibrated.verifier_attempts,
                model_provider=self.provider_name,
            ),
            trace=trace,
        )

    def retrieve(self, user_persona: str, product_details: str) -> ExemplarSet:
        return self._retriever.retrieve(user_persona, product_details)

    def generate_candidates(
        self,
        user_persona: str,
        product_details: str,
        exemplar_set: ExemplarSet,
        sample_count: int,
    ) -> Candidates:
        return self._reasoner.generate_candidates(
            user_persona,
            product_details,
            exemplar_set,
            sample_count=sample_count,
        )

    def select_best(
        self,
        user_persona: str,
        product_details: str,
        exemplar_set: ExemplarSet,
        aggregate: AggregatedCandidate,
    ) -> SelectedCandidate:
        return self._reasoner.select_best(user_persona, product_details, exemplar_set, aggregate)

    def calibrate_rating(self, selected: SelectedCandidate) -> CalibratedCandidate:
        return calibrate_rating_from_review(selected, self._rating_calibrator)


def combined_reason(calibrated: CalibratedCandidate) -> str:
    parts = [calibrated.selection_reason.strip(), calibrated.calibration_reason.strip()]
    if calibrated.chosen_experience.strip():
        parts.insert(0, f"Experience: {calibrated.chosen_experience.strip()}")
    return " ".join(part for part in parts if part)


@lru_cache
def get_review_service() -> ReviewSimulationService:
    settings = get_settings()
    embedding_model = SentenceTransformerEmbeddingModel(settings.embedding_model_name)
    return ReviewSimulationService(
        retriever=BehavioralRetriever(embedding_model=embedding_model),
        reasoner=build_reasoner(settings),
        rating_calibrator=SklearnOrdinalRatingCalibrator(embedding_model=embedding_model),
    )
