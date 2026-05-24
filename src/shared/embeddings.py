from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import numpy as np
import numpy.typing as npt

EmbeddingArray = npt.NDArray[np.float32]


class EmbeddingModel:
    dimension: int
    model_name: str

    def encode(self, texts: Sequence[str]) -> EmbeddingArray:
        raise NotImplementedError


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(self.model_name)

    @cached_property
    def dimension(self) -> int:
        vector = self.encode(["dimension probe"])
        return int(vector.shape[1])

    def encode(self, texts: Sequence[str]) -> EmbeddingArray:
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return ensure_normalized_float32(vectors)


def ensure_normalized_float32(vectors) -> EmbeddingArray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    return (array / safe_norms).astype(np.float32)
