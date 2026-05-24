from __future__ import annotations

import argparse
from pathlib import Path

from review.eval_cases import DEFAULT_EVAL_DATASET
from review.manual_gold import (
    NO_MISSING_REVIEW_GOLD_MESSAGE,
    display_review_eval_case,
    dumps_pretty,
)


def completed_payload() -> dict[str, object]:
    return {
        "complete": True,
        "message": NO_MISSING_REVIEW_GOLD_MESSAGE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show one Task A eval case for manual gold review/rating writing."
    )
    parser.add_argument("case_id", nargs="?")
    parser.add_argument("--next-missing", action="store_true")
    parser.add_argument("--source", type=Path, default=DEFAULT_EVAL_DATASET)
    args = parser.parse_args()

    try:
        payload = display_review_eval_case(
            args.source,
            case_id=args.case_id,
            next_missing=args.next_missing,
        )
    except ValueError as exc:
        if args.next_missing and str(exc) == NO_MISSING_REVIEW_GOLD_MESSAGE:
            print(dumps_pretty(completed_payload()))
            return
        parser.error(str(exc))

    print(dumps_pretty(payload))


if __name__ == "__main__":
    main()
