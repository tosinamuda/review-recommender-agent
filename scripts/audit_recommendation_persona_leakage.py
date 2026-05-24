from __future__ import annotations

import argparse
import json
from pathlib import Path

from recommendation.persona_leakage import audit_persona_business_leakage

DEFAULT_SOURCE = Path("data/generated/recommendation_cases_manual_ng.jsonl")
DEFAULT_CATALOGUE = Path("data/recommendation/nigerian_catalogue_manual.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Surface Task 2 personas that may mention linked business names. "
            "This is a manual review queue, not proof that unflagged personas are clean."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    parser.add_argument(
        "--fail-on-findings",
        action="store_true",
        help="Exit non-zero when any possible leakage is found.",
    )
    args = parser.parse_args()

    findings = audit_persona_business_leakage(
        cases_path=args.source,
        catalogue_path=args.catalogue,
    )
    if args.json:
        print(
            json.dumps(
                {
                    "finding_count": len(findings),
                    "findings": [
                        {
                            "case_id": finding.case_id,
                            "product_id": finding.product_id,
                            "product_name": finding.product_name,
                            "match_type": finding.match_type,
                            "matched_text": finding.matched_text,
                            "user_persona": finding.user_persona,
                        }
                        for finding in findings
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"Possible persona leakage findings: {len(findings)}")
        if findings:
            print(
                "Review these manually. Generic food, location, or domain words are "
                "intentionally ignored by this script."
            )
        for finding in findings:
            print()
            print(f"case_id: {finding.case_id}")
            print(f"product_id: {finding.product_id}")
            print(f"product_name: {finding.product_name}")
            print(f"match: {finding.match_type} -> {finding.matched_text}")
            print(f"persona: {finding.user_persona}")

    if findings and args.fail_on_findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
