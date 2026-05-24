from __future__ import annotations

from pathlib import Path

from app.settings import get_settings
from shared.embeddings import EmbeddingModel, SentenceTransformerEmbeddingModel

from .retrieval_scoring import proxy_stats
from .review_corpus import load_review_corpus
from .review_index import FaissReviewIndex
from .schemas import ExemplarSet


class BehavioralRetriever:
    def __init__(
        self,
        corpus_path: Path | None = None,
        index_dir: Path | None = None,
        embedding_model: EmbeddingModel | None = None,
        in_memory: bool = False,
    ):
        settings = get_settings()
        configured_path = corpus_path or settings.review_corpus_path
        configured_index_dir = index_dir or settings.review_index_dir
        self._corpus_path = configured_path.resolve()
        self._index_dir = configured_index_dir.resolve()
        self._embedding_model = embedding_model or SentenceTransformerEmbeddingModel(
            settings.embedding_model_name
        )
        self._in_memory = in_memory
        self._index: FaissReviewIndex | None = None

    def retrieve(self, user_persona: str, product_details: str, k: int = 5) -> ExemplarSet:
        exemplars = self._load_index().search(user_persona, product_details, k=k)
        return ExemplarSet(exemplars=exemplars, proxy_stats=proxy_stats(exemplars))

    def _load_index(self) -> FaissReviewIndex:
        if self._index is not None:
            return self._index
        records = load_review_corpus(str(self._corpus_path))
        if self._in_memory:
            self._index = FaissReviewIndex.from_records(records, self._embedding_model)
        else:
            self._index = FaissReviewIndex.from_artifacts(
                corpus_path=self._corpus_path,
                index_dir=self._index_dir,
                embedding_model=self._embedding_model,
            )
        return self._index
