from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np

from shared.embeddings import EmbeddingModel

from .review_corpus import ReviewCorpusRecord, load_review_corpus
from .schemas import Exemplar

AXES = ("persona", "product", "joint")
AXIS_WEIGHTS = {
    "persona": 0.35,
    "product": 0.30,
    "joint": 0.35,
}


@dataclass(frozen=True)
class ReviewIndexPaths:
    root: Path

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    def axis_index(self, axis: str) -> Path:
        return self.root / f"{axis}.faiss"


class FaissReviewIndex:
    def __init__(
        self,
        records: tuple[ReviewCorpusRecord, ...],
        indexes: dict[str, faiss.Index],
        embedding_model: EmbeddingModel,
    ):
        self.records = records
        self.indexes = indexes
        self.embedding_model = embedding_model

    @classmethod
    def from_artifacts(
        cls,
        corpus_path: Path,
        index_dir: Path,
        embedding_model: EmbeddingModel,
    ) -> FaissReviewIndex:
        paths = ReviewIndexPaths(index_dir)
        missing = [
            path
            for path in [paths.metadata, *(paths.axis_index(axis) for axis in AXES)]
            if not path.exists()
        ]
        if missing:
            missing_list = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(
                "Review retrieval artifacts are missing. "
                f"Run scripts/build_review_artifacts.py. Missing: {missing_list}"
            )
        metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
        if metadata.get("embedding_model") != embedding_model.model_name:
            raise ValueError(
                "Review index embedding model mismatch: "
                f"artifact={metadata.get('embedding_model')!r}, "
                f"runtime={embedding_model.model_name!r}"
            )
        records = load_review_corpus(str(corpus_path.resolve()))
        if len(records) != int(metadata["record_count"]):
            raise ValueError(
                "Review index record count mismatch: "
                f"artifact={metadata['record_count']}, corpus={len(records)}"
            )
        indexes = {axis: faiss.read_index(str(paths.axis_index(axis))) for axis in AXES}
        return cls(records=records, indexes=indexes, embedding_model=embedding_model)

    @classmethod
    def from_records(
        cls,
        records: tuple[ReviewCorpusRecord, ...],
        embedding_model: EmbeddingModel,
    ) -> FaissReviewIndex:
        vectors = build_axis_vectors(records, embedding_model)
        indexes = {axis: build_faiss_index(vectors[axis]) for axis in AXES}
        return cls(records=records, indexes=indexes, embedding_model=embedding_model)

    def search(self, user_persona: str, product_details: str, k: int) -> list[Exemplar]:
        axis_queries = {
            "persona": user_persona,
            "product": product_details,
            "joint": f"{user_persona} {product_details}",
        }
        combined_scores: dict[int, float] = {}
        axis_scores: dict[int, dict[str, float]] = {}
        per_axis_k = min(len(self.records), max(k * 4, k))

        for axis, query in axis_queries.items():
            query_vector = self.embedding_model.encode([query])
            distances, indexes = cast(Any, self.indexes[axis]).search(query_vector, per_axis_k)
            for raw_score, raw_index in zip(distances[0], indexes[0], strict=True):
                if raw_index < 0:
                    continue
                index = int(raw_index)
                score = max(float(raw_score), 0.0)
                axis_scores.setdefault(index, {})[axis] = round(score, 6)
                combined_scores[index] = combined_scores.get(index, 0.0) + (
                    AXIS_WEIGHTS[axis] * score
                )

        ranked = sorted(
            combined_scores,
            key=lambda index: (combined_scores[index], self.records[index].rating),
            reverse=True,
        )
        return [
            record_to_exemplar(
                self.records[index],
                score=combined_scores[index],
                axis_scores=axis_scores.get(index, {}),
            )
            for index in ranked[:k]
        ]


def build_axis_vectors(
    records: tuple[ReviewCorpusRecord, ...],
    embedding_model: EmbeddingModel,
) -> dict[str, np.ndarray]:
    return {
        axis: embedding_model.encode([axis_text(record, axis) for record in records])
        for axis in AXES
    }


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    cast(Any, index).add(vectors)
    return index


def write_review_index(
    records: tuple[ReviewCorpusRecord, ...],
    embedding_model: EmbeddingModel,
    index_dir: Path,
) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    vectors = build_axis_vectors(records, embedding_model)
    for axis in AXES:
        faiss.write_index(build_faiss_index(vectors[axis]), str(index_dir / f"{axis}.faiss"))
    metadata = {
        "embedding_model": embedding_model.model_name,
        "record_count": len(records),
        "axes": list(AXES),
        "axis_weights": AXIS_WEIGHTS,
    }
    (index_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def axis_text(record: ReviewCorpusRecord, axis: str) -> str:
    persona = record.user_profile_summary
    product = record.product_details or record.item_category
    if axis == "persona":
        return persona
    if axis == "product":
        return product
    if axis == "joint":
        return " ".join([persona, product, record.review_text])
    raise ValueError(f"Unknown retrieval axis: {axis}")


def record_to_exemplar(
    record: ReviewCorpusRecord,
    score: float,
    axis_scores: dict[str, float],
) -> Exemplar:
    return Exemplar(
        exemplar_id=record.exemplar_id,
        review_text=record.review_text,
        rating=record.rating,
        user_profile_summary=record.user_profile_summary,
        item_category=record.item_category,
        score=max(round(score, 6), 0.000001),
        source=record.source,
        product_details=record.product_details,
        product_issue=record.product_issue,
        axis_scores=axis_scores,
    )
