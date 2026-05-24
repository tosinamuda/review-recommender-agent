from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT = Path("data/generated/persona_review_training_cases.jsonl")


@dataclass(frozen=True)
class ReviewSource:
    path: Path
    case_prefix: str
    app_id: str
    product_details: str


SOURCES = (
    ReviewSource(
        path=Path("data/raw/chowdeck_google_play_reviews.jsonl"),
        case_prefix="chowdeck",
        app_id="com.chowdeck.com",
        product_details=(
            "Product: Chowdeck | Food Delivery\n"
            "Category: food delivery app\n"
            "Description: App for ordering meals from restaurants and getting food delivered.\n"
            "Provider: Chowdeck"
        ),
    ),
    ReviewSource(
        path=Path("data/raw/gtworld_google_play_reviews.jsonl"),
        case_prefix="gtworld",
        app_id="com.gtbank.gtworldv1",
        product_details=(
            "Product: GTWorld\n"
            "Category: banking app\n"
            "Description: Mobile banking app for account access, transfers, bill payments, "
            "and everyday customer banking services.\n"
            "Provider: Guaranty Trust Bank"
        ),
    ),
    ReviewSource(
        path=Path("data/raw/mymtn_ng_google_play_reviews.jsonl"),
        case_prefix="mymtn_ng",
        app_id="ng.mtn.nextgen",
        product_details=(
            "Product: myMTN NG\n"
            "Category: telecom self-service app\n"
            "Description: App for managing MTN Nigeria mobile accounts, airtime, data bundles, "
            "subscriptions, and customer self-service.\n"
            "Provider: MTN Nigeria"
        ),
    ),
    ReviewSource(
        path=Path("data/raw/ninauth_google_play_reviews.jsonl"),
        case_prefix="ninauth",
        app_id="com.ninauth.mobile",
        product_details=(
            "Product: NINAuth\n"
            "Category: government identity app\n"
            "Description: App for managing National Identification Number data sharing, "
            "NIN lock, identity verification, and face verification.\n"
            "Provider: NIMC"
        ),
    ),
    ReviewSource(
        path=Path("data/raw/nismobile_google_play_reviews.jsonl"),
        case_prefix="nismobile",
        app_id="com.irissmart.nismobile",
        product_details=(
            "Product: NIS Mobile\n"
            "Category: government passport app\n"
            "Description: Contactless passport application app for eligibility checks, "
            "biometric capture, payment, and application submission.\n"
            "Provider: Nigeria Immigration Service (NIS)"
        ),
    ),
    ReviewSource(
        path=Path("data/raw/palmpay_google_play_reviews.jsonl"),
        case_prefix="palmpay",
        app_id="com.transsnet.palmpay",
        product_details=(
            "Product: PalmPay - Smarter Way to Bank\n"
            "Category: fintech wallet app\n"
            "Description: Digital finance app for transfers, bill payments, airtime, data, "
            "cashback, savings, and everyday payments.\n"
            "Provider: PalmPay"
        ),
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert raw Google Play reviews into persona/product training cases."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    cases = list(build_training_cases(SOURCES))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    print(json.dumps({"written": len(cases), "path": str(args.output)}, indent=2))


def build_training_cases(sources: tuple[ReviewSource, ...]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for source in sources:
        records = load_source_records(source)
        for index, record in enumerate(records, start=1):
            cases.append(
                {
                    "case_id": f"{source.case_prefix}_{index:03d}",
                    "input": {
                        "user_persona": "",
                        "product_details": source.product_details,
                        "product_issue": "",
                    },
                    "output": {
                        "review": record["review"],
                        "rating": record["rating"],
                    },
                }
            )
    return cases


def load_source_records(source: ReviewSource) -> list[dict[str, Any]]:
    records = [
        json.loads(line)
        for line in source.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    bad_app_ids = {
        record.get("app_id")
        for record in records
        if record.get("app_id") != source.app_id
    }
    if bad_app_ids:
        raise ValueError(f"{source.path} contains unexpected app ids: {sorted(bad_app_ids)}")
    return records


if __name__ == "__main__":
    main()
