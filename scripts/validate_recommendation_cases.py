from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommendation.case_validation import (
    validate_recommendation_cases,
    validation_summary,
)
from recommendation.catalogue import load_product_catalogue

DEFAULT_SOURCE = Path("data/generated/recommendation_cases_manual.jsonl")
DEFAULT_MANIFEST = Path("data/generated/recommendation_sample_manifest.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Task 2 recommendation cases.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manual-catalogue", type=Path)
    parser.add_argument("--skip-manifest-case-check", action="store_true")
    parser.add_argument("--allow-empty-personas", action="store_true")
    parser.add_argument("--enforce-context-match", action="store_true")
    args = parser.parse_args()

    extra_product_ids: set[str] = set()
    extra_product_names: dict[str, str] = {}
    if args.manual_catalogue is not None:
        products = load_product_catalogue(args.manual_catalogue)
        extra_product_ids = {product.product_id for product in products}
        extra_product_names = {product.product_id: product.name for product in products}

    result = validate_recommendation_cases(
        source_path=args.source,
        manifest_path=args.manifest,
        require_personas=not args.allow_empty_personas,
        extra_product_ids=extra_product_ids,
        extra_product_names=extra_product_names,
        require_manifest_case_ids=not args.skip_manifest_case_check,
        enforce_context_match=args.enforce_context_match,
    )
    print(json.dumps(validation_summary(result), indent=2, sort_keys=True))
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
