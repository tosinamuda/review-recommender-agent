from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="BCT_APP_",
        extra="ignore",
        populate_by_name=True,
    )

    app_name: str = "bct_challenge_agent"
    public_base_url: str = "http://localhost:8000"
    review_corpus_path: Path = Path("data/review_exemplars.jsonl")
    review_index_dir: Path = Path("data/index/retrieval")
    recommendation_catalogue_path: Path = Field(
        default=Path("data/recommendation/product_catalogue.jsonl")
    )
    recommendation_index_dir: Path = Path("data/index/recommendation")
    recommendation_interactions_path: Path = Field(
        default=Path("data/recommendation/interactions.jsonl")
    )
    recommendation_persona_cases_path: Path = Field(
        default=Path("data/recommendation/persona_cases.jsonl")
    )
    recommendation_eval_cases_path: Path = Field(
        default=Path("data/eval/recommendation_eval_cases.jsonl")
    )
    rating_calibrator_path: Path = Path("data/index/rating_calibrator.joblib")
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    lm_model: str = Field(
        default="openrouter/openai/gpt-oss-120b", validation_alias="LM_MODEL"
    )
    lm_temperature: float = Field(default=0.7, ge=0.0, le=2.0)


@lru_cache
def get_settings() -> Settings:
    return Settings()
