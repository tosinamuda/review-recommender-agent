from __future__ import annotations

import argparse
from pathlib import Path

from recommendation.manual_personas import dumps_pretty, set_case_persona

DEFAULT_SOURCE = Path("data/generated/recommendation_cases_manual.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set the manually written persona for one Task 2 recommendation case."
    )
    parser.add_argument("case_id")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    updated = set_case_persona(
        args.source,
        case_id=args.case_id,
        persona=args.persona,
        force=args.force,
    )
    print(dumps_pretty(updated.model_dump(exclude_none=True)))


if __name__ == "__main__":
    main()
