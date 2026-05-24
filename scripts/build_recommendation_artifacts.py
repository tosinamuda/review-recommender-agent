from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.settings import get_settings
from recommendation.artifact_builder import build_recommendation_artifacts
from shared.embeddings import SentenceTransformerEmbeddingModel

DEFAULT_SOURCE = Path("data/generated/recommendation_cases_manual.jsonl")
DEFAULT_MANIFEST = Path("data/generated/recommendation_sample_manifest.json")
DEFAULT_CATALOGUE = Path("data/recommendation/product_catalogue.jsonl")
DEFAULT_INTERACTIONS = Path("data/recommendation/interactions.jsonl")
DEFAULT_PERSONA_CASES = Path("data/recommendation/persona_cases.jsonl")
DEFAULT_EVAL = Path("data/eval/recommendation_eval_cases.jsonl")
DEFAULT_INDEX_DIR = Path("data/index/recommendation")
DEFAULT_MANUAL_CATALOGUE = Path("data/recommendation/nigerian_catalogue_manual.jsonl")
DEFAULT_MANUAL_CASES = Path("data/generated/recommendation_cases_manual_ng.jsonl")
DEFAULT_HELDOUT_SOURCE = Path("data/generated/recommendation_cases_holdout_yelp.jsonl")
DEFAULT_HELDOUT_MANIFEST = Path("data/generated/recommendation_holdout_sample_manifest.json")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build Task 2 recommendation artifacts.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--interactions", type=Path, default=DEFAULT_INTERACTIONS)
    parser.add_argument("--persona-cases", type=Path, default=DEFAULT_PERSONA_CASES)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--index-dir", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--embedding-model", default=settings.embedding_model_name)
    parser.add_argument("--manual-catalogue", type=Path, default=DEFAULT_MANUAL_CATALOGUE)
    parser.add_argument("--manual-cases", type=Path, default=DEFAULT_MANUAL_CASES)
    parser.add_argument("--heldout-source", type=Path, default=DEFAULT_HELDOUT_SOURCE)
    parser.add_argument("--heldout-manifest", type=Path, default=DEFAULT_HELDOUT_MANIFEST)
    args = parser.parse_args()

    heldout_source = args.heldout_source if args.heldout_source.exists() else None
    heldout_manifest = args.heldout_manifest if args.heldout_manifest.exists() else None
    summary = build_recommendation_artifacts(
        source_path=args.source,
        manifest_path=args.manifest,
        catalogue_path=args.catalogue,
        interactions_path=args.interactions,
        persona_cases_path=args.persona_cases,
        eval_path=args.eval,
        index_dir=args.index_dir,
        embedding_model=SentenceTransformerEmbeddingModel(args.embedding_model),
        manual_catalogue_path=args.manual_catalogue,
        manual_cases_path=args.manual_cases,
        heldout_source_path=heldout_source,
        heldout_manifest_path=heldout_manifest,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
