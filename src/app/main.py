from __future__ import annotations

import logging
import warnings
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from app.settings import get_settings
from recommendation.schemas import RecommendationRequest, RecommendationResponse
from recommendation.service import get_recommendation_service
from review.schemas import ReviewRequest, ReviewSimulationResponse
from review.service import get_review_service
from shared.openrouter_lm import configure_dspy_lm

from . import recommendation_a2a, review_a2a

warnings.filterwarnings(
    "ignore",
    message=r".*\[EXPERIMENTAL\].*",
    category=UserWarning,
)
logging.getLogger("dspy.predict.predict").setLevel(logging.ERROR)

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    settings = get_settings()
    configure_dspy_lm(settings)  # single global configure — must happen before any async tasks

    # Pre-warm both services so embedding models load at startup, not on the first request
    get_review_service()
    get_recommendation_service()

    app = FastAPI(
        title="BCT Hackathon Review Simulation",
        version="0.1.0",
        description="Task 1 review simulation and Task 2 recommendation agents.",
    )

    review_a2a.add_routes(app, settings)
    recommendation_a2a.add_routes(app, settings)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "agent": settings.app_name}

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.post("/api/v1/review-simulation", response_model=ReviewSimulationResponse)
    def simulate_review(request: ReviewRequest) -> ReviewSimulationResponse:
        return get_review_service().run(request)

    @app.post("/api/v1/recommendations", response_model=RecommendationResponse)
    def recommend_products(request: RecommendationRequest) -> RecommendationResponse:
        return get_recommendation_service().run(request)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
