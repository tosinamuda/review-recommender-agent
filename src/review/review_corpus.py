from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class ReviewCorpusRecord(BaseModel):
    exemplar_id: str
    review_text: str = Field(min_length=1)
    rating: int = Field(ge=1, le=5)
    user_profile_summary: str
    item_category: str
    source: str
    nigerian: bool = False
    product_details: str | None = None
    product_issue: str | None = None


class JsonlReviewCorpus:
    def __init__(self, path: Path):
        self._path = path

    def load(self) -> list[ReviewCorpusRecord]:
        if not self._path.exists():
            raise FileNotFoundError(f"Review corpus not found: {self._path}")
        records = []
        for line_number, line in enumerate(self._path.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                records.append(_parse_record(line, line_number, self._path))
        if not records:
            raise ValueError(f"Review corpus is empty: {self._path}")
        return records


@lru_cache
def load_review_corpus(path: str) -> tuple[ReviewCorpusRecord, ...]:
    return tuple(JsonlReviewCorpus(Path(path)).load())


def _parse_record(line: str, line_number: int, path: Path) -> ReviewCorpusRecord:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return ReviewCorpusRecord.model_validate(payload)
