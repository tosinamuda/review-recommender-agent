from __future__ import annotations

import argparse
from pathlib import Path

from recommendation.manual_personas import (
    NO_EMPTY_RECOMMENDATION_PERSONA_MESSAGE,
    display_case,
    dumps_pretty,
)

DEFAULT_SOURCE = Path("data/generated/recommendation_cases_manual.jsonl")


def completed_payload() -> dict[str, object]:
    return {
        "complete": True,
        "message": NO_EMPTY_RECOMMENDATION_PERSONA_MESSAGE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show one sampled Task 2 recommendation case for manual persona writing."
    )
    parser.add_argument("case_id", nargs="?")
    parser.add_argument("--next-empty", action="store_true")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()

    try:
        payload = display_case(
            args.source,
            case_id=args.case_id,
            next_empty=args.next_empty,
        )
    except ValueError as exc:
        if args.next_empty and str(exc) == NO_EMPTY_RECOMMENDATION_PERSONA_MESSAGE:
            print(dumps_pretty(completed_payload()))
            return
        parser.error(str(exc))

    print(dumps_pretty(payload))


if __name__ == "__main__":
    main()
