from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from app.settings import get_settings
from shared.embeddings import EmbeddingModel, SentenceTransformerEmbeddingModel

from .review_corpus import ReviewCorpusRecord
from .schemas import CalibratedCandidate, SelectedCandidate

RATING_VALUES = np.asarray([1, 2, 3, 4, 5], dtype=np.float32)


@dataclass(frozen=True)
class RatingCalibration:
    rating: int
    continuous: float
    distribution: list[float]
    reason: str


class RatingCalibrator:
    def calibrate(self, review: str) -> RatingCalibration:
        raise NotImplementedError


class SklearnOrdinalRatingCalibrator(RatingCalibrator):
    def __init__(
        self,
        artifact_path: Path | None = None,
        embedding_model: EmbeddingModel | None = None,
    ):
        settings = get_settings()
        self._artifact_path = (artifact_path or settings.rating_calibrator_path).resolve()
        self._embedding_model = embedding_model or SentenceTransformerEmbeddingModel(
            settings.embedding_model_name
        )
        self._artifact: dict | None = None

    def calibrate(self, review: str) -> RatingCalibration:
        artifact = self._load_artifact()
        vector = self._embedding_model.encode([review])
        gt_probs = [
            positive_class_probability(model, vector)
            for model in artifact["threshold_models"]
        ]
        distribution = threshold_probabilities_to_distribution(gt_probs)
        continuous = float(np.dot(distribution, RATING_VALUES))
        rating = clamp_rating(continuous)
        return RatingCalibration(
            rating=rating,
            continuous=round(continuous, 4),
            distribution=[round(float(value), 6) for value in distribution.tolist()],
            reason=(
                "Rating calibrated from selected-review embedding with "
                "a trained sklearn ordinal threshold model."
            ),
        )

    def _load_artifact(self) -> dict:
        if self._artifact is not None:
            return self._artifact
        if not self._artifact_path.exists():
            raise FileNotFoundError(
                "Rating calibrator artifact is missing. "
                f"Run scripts/build_review_artifacts.py. Missing: {self._artifact_path}"
            )
        artifact = joblib.load(self._artifact_path)
        artifact_model = artifact.get("embedding_model")
        if artifact_model != self._embedding_model.model_name:
            raise ValueError(
                "Rating calibrator embedding model mismatch: "
                f"artifact={artifact_model!r}, runtime={self._embedding_model.model_name!r}"
            )
        self._artifact = artifact
        return artifact


def calibrate_rating_from_review(
    selected: SelectedCandidate,
    calibrator: RatingCalibrator | None = None,
) -> CalibratedCandidate:
    active_calibrator = calibrator or SklearnOrdinalRatingCalibrator()
    calibration = active_calibrator.calibrate(selected.review)
    return CalibratedCandidate(
        rating=calibration.rating,
        review=selected.review,
        selection_reason=selected.selection_reason,
        calibration_reason=calibration.reason,
        rating_continuous=calibration.continuous,
        rating_distribution=calibration.distribution,
        chosen_experience=selected.chosen_experience,
        verifier_attempts=selected.verifier_attempts,
    )


def fit_rating_calibrator(
    records: tuple[ReviewCorpusRecord, ...],
    embedding_model: EmbeddingModel,
    output_path: Path,
) -> None:
    reviews = [record.review_text for record in records]
    ratings = np.asarray([record.rating for record in records], dtype=np.int32)
    vectors = embedding_model.encode(reviews)
    threshold_models = []
    for threshold in range(1, 5):
        target = (ratings > threshold).astype(np.int32)
        model = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=17,
        )
        model.fit(vectors, target)
        threshold_models.append(model)
    artifact = {
        "embedding_model": embedding_model.model_name,
        "record_count": len(records),
        "thresholds": [1, 2, 3, 4],
        "threshold_models": threshold_models,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)


def positive_class_probability(model, vector: np.ndarray) -> float:
    classes = list(model.classes_)
    positive_index = classes.index(1)
    return float(model.predict_proba(vector)[0, positive_index])


def threshold_probabilities_to_distribution(gt_probs: list[float]) -> np.ndarray:
    monotonic = []
    previous = 1.0
    for probability in gt_probs:
        current = min(max(float(probability), 0.0), previous)
        monotonic.append(current)
        previous = current
    p_gt_1, p_gt_2, p_gt_3, p_gt_4 = monotonic
    distribution = np.asarray(
        [
            1.0 - p_gt_1,
            p_gt_1 - p_gt_2,
            p_gt_2 - p_gt_3,
            p_gt_3 - p_gt_4,
            p_gt_4,
        ],
        dtype=np.float32,
    )
    total = float(distribution.sum())
    if total <= 0:
        return np.asarray([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32)
    return distribution / total


def clamp_rating(value: int | float) -> int:
    return max(1, min(5, int(round(float(value)))))
