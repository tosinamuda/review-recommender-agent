from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.settings import get_settings
from review.prompt_parsing import infer_category, split_labeled_sections
from review.rating_calibration import fit_rating_calibrator
from review.review_corpus import ReviewCorpusRecord, load_review_corpus
from review.review_index import write_review_index
from shared.embeddings import SentenceTransformerEmbeddingModel

DEFAULT_SOURCE = Path("data/generated/persona_review_training_cases_manual_sanitized.jsonl")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build retrieval and rating artifacts.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--corpus", type=Path, default=settings.review_corpus_path)
    parser.add_argument("--index-dir", type=Path, default=settings.review_index_dir)
    parser.add_argument(
        "--calibrator",
        type=Path,
        default=settings.rating_calibrator_path,
    )
    parser.add_argument(
        "--embedding-model",
        default=settings.embedding_model_name,
    )
    args = parser.parse_args()

    records = normalize_training_cases(args.source)
    write_jsonl_corpus(records, args.corpus)
    embedding_model = SentenceTransformerEmbeddingModel(args.embedding_model)
    write_review_index(
        records=load_review_corpus(str(args.corpus.resolve())),
        embedding_model=embedding_model,
        index_dir=args.index_dir,
    )
    fit_rating_calibrator(
        records=records,
        embedding_model=embedding_model,
        output_path=args.calibrator,
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "corpus": str(args.corpus),
                "index_dir": str(args.index_dir),
                "calibrator": str(args.calibrator),
                "embedding_model": args.embedding_model,
            },
            indent=2,
        )
    )


def normalize_training_cases(source: Path) -> tuple[ReviewCorpusRecord, ...]:
    if not source.exists():
        raise FileNotFoundError(f"Training source not found: {source}")
    records = []
    seen_ids = set()
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        payload = json.loads(line)
        record = training_case_to_record(payload, line_number)
        if record.exemplar_id in seen_ids:
            raise ValueError(f"Duplicate case_id in {source}: {record.exemplar_id}")
        seen_ids.add(record.exemplar_id)
        records.append(record)
    if not records:
        raise ValueError(f"Training source is empty: {source}")
    return tuple(records)


def training_case_to_record(payload: dict[str, Any], line_number: int) -> ReviewCorpusRecord:
    try:
        case_id = str(payload["case_id"])
        input_payload = payload["input"]
        output_payload = payload["output"]
        product_details = str(input_payload["product_details"])
        review = str(output_payload["review"])
        rating = int(output_payload["rating"])
    except KeyError as exc:
        raise ValueError(f"Missing required field in training case line {line_number}") from exc
    sections = split_labeled_sections(product_details)
    category = sections.get("category") or infer_category(product_details)
    return ReviewCorpusRecord(
        exemplar_id=case_id,
        review_text=review,
        rating=rating,
        user_profile_summary=str(input_payload["user_persona"]),
        item_category=category,
        source=f"sanitized_training:{case_id.split('_', 1)[0]}",
        nigerian=True,
        product_details=product_details,
        product_issue=str(input_payload.get("product_issue") or ""),
    )


def write_jsonl_corpus(records: tuple[ReviewCorpusRecord, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        record.model_dump_json(exclude_none=True)
        for record in records
    ]
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
