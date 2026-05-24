from __future__ import annotations

import argparse
from pathlib import Path

from review.eval_cases import DEFAULT_EVAL_DATASET
from review.manual_gold import dumps_pretty, set_review_eval_gold


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Replace one Task A eval case's gold rating and reference review from "
            "explicitly supplied arguments."
        )
    )
    parser.add_argument("case_id")
    parser.add_argument("--source", type=Path, default=DEFAULT_EVAL_DATASET)
    parser.add_argument("--rating", type=int, required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    updated = set_review_eval_gold(
        args.source,
        case_id=args.case_id,
        rating=args.rating,
        reference_review=args.review,
        force=args.force,
    )
    print(dumps_pretty(updated))


if __name__ == "__main__":
    main()
