from collections.abc import Iterator
from pathlib import Path

import pytest

import app.main as api_main_module
import recommendation.service as recommendation_service_module
import recommendation.workflow as recommendation_workflow_module
import review.service as service_module
import review.workflow as workflow_module
from app.settings import get_settings
from recommendation.retrieval import ProductRetriever
from recommendation.service import (
    RecommendationService,
    get_recommendation_service,
)
from review.retrieval import BehavioralRetriever
from review.service import ReviewSimulationService, get_review_service
from tests.fakes import (
    ContractGeneratedRecommendationReasoner,
    ContractRecommendationCoverageJudge,
    ContractRecommendationRanker,
    ContractReviewReasoner,
    FixedReviewRatingCalibrator,
    HashingEmbeddingModel,
)


@pytest.fixture(autouse=True)
def use_explicit_offline_service(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[None]:
    def service_factory() -> ReviewSimulationService:
        return ReviewSimulationService(
            retriever=BehavioralRetriever(
                corpus_path=Path("data/review_exemplars.jsonl"),
                embedding_model=HashingEmbeddingModel(),
                in_memory=True,
            ),
            reasoner=ContractReviewReasoner(),
            rating_calibrator=FixedReviewRatingCalibrator(),
        )

    def recommendation_service_factory() -> RecommendationService:
        return RecommendationService(
            retriever=ProductRetriever(
                catalogue_path=Path("data/product_catalogue.jsonl"),
                index_dir=tmp_path / "recommendation-index",
                interactions_path=tmp_path / "recommendation-interactions.jsonl",
                eval_cases_path=tmp_path / "recommendation-eval-cases.jsonl",
                embedding_model=HashingEmbeddingModel(),
            ),
            coverage_judge=ContractRecommendationCoverageJudge(),
            generated_reasoner=ContractGeneratedRecommendationReasoner(),
            ranker=ContractRecommendationRanker(),
        )

    monkeypatch.setattr(service_module, "get_review_service", service_factory)
    monkeypatch.setattr(api_main_module, "get_review_service", service_factory)
    monkeypatch.setattr(workflow_module, "get_review_service", service_factory)
    monkeypatch.setattr(
        recommendation_service_module,
        "get_recommendation_service",
        recommendation_service_factory,
    )
    monkeypatch.setattr(
        api_main_module,
        "get_recommendation_service",
        recommendation_service_factory,
    )
    monkeypatch.setattr(
        recommendation_workflow_module,
        "get_recommendation_service",
        recommendation_service_factory,
    )
    get_settings.cache_clear()
    get_review_service.cache_clear()
    get_recommendation_service.cache_clear()
    yield
    get_settings.cache_clear()
    get_review_service.cache_clear()
    get_recommendation_service.cache_clear()
