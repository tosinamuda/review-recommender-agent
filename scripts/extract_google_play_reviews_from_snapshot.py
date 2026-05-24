from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

TEXT_RE = re.compile(
    r'^(?P<indent>\s*)uid=[^ ]+ (?P<kind>StaticText|image) "(?P<text>.*)"(?: .*)?$'
)
RATING_RE = re.compile(r"Rated (\d+) stars? out of five stars")
DATE_RE = re.compile(r"^[A-Z][a-z]+ \d{1,2}, \d{4}$")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract Google Play review cards from a Chrome accessibility snapshot."
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dialog-title", required=True)
    parser.add_argument("--review-id-prefix", required=True)
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--app-name", required=True)
    parser.add_argument("--source-url", required=True)
    args = parser.parse_args()

    records = extract_records(
        snapshot=args.snapshot,
        dialog_title=args.dialog_title,
        review_id_prefix=args.review_id_prefix,
        app_id=args.app_id,
        app_name=args.app_name,
        source_url=args.source_url,
    )
    args.output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    ratings = {
        str(rating): sum(1 for record in records if record["rating"] == rating)
        for rating in range(1, 6)
    }
    print(
        json.dumps(
            {"written": len(records), "path": str(args.output), "ratings": ratings},
            indent=2,
        )
    )


def extract_records(
    *,
    snapshot: Path,
    dialog_title: str,
    review_id_prefix: str,
    app_id: str,
    app_name: str,
    source_url: str,
) -> list[dict[str, Any]]:
    lines = snapshot.read_text(encoding="utf-8").splitlines()
    lines = lines[_dialog_start(lines, dialog_title) :]
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    expect_review = False
    pending_helpful_number: int | None = None

    for line in lines:
        stripped = line.strip()
        if re.match(r"uid=[^ ]+ banner$", stripped):
            maybe_record = record_from_current(
                current,
                review_id_prefix=review_id_prefix,
                app_id=app_id,
                app_name=app_name,
                source_url=source_url,
            )
            if maybe_record:
                records.append(maybe_record)
            current = {"helpful_count": 0}
            expect_review = False
            pending_helpful_number = None
            continue
        if current is None:
            continue
        match = TEXT_RE.match(line)
        if not match:
            continue
        expect_review, pending_helpful_number = consume_snapshot_value(
            current=current,
            kind=match.group("kind"),
            value=match.group("text"),
            expect_review=expect_review,
            pending_helpful_number=pending_helpful_number,
        )

    maybe_record = record_from_current(
        current,
        review_id_prefix=review_id_prefix,
        app_id=app_id,
        app_name=app_name,
        source_url=source_url,
    )
    if maybe_record:
        records.append(maybe_record)
    return list({record["review_id"]: record for record in records}.values())


def consume_snapshot_value(
    *,
    current: dict[str, Any],
    kind: str,
    value: str,
    expect_review: bool,
    pending_helpful_number: int | None,
) -> tuple[bool, int | None]:
    if kind == "image":
        rating_match = RATING_RE.search(value)
        if rating_match:
            current["rating"] = int(rating_match.group(1))
        return expect_review, pending_helpful_number
    if not current.get("author") and value != "More review actions":
        current["author"] = value
        return expect_review, pending_helpful_number
    if not current.get("date") and DATE_RE.match(value):
        current["date"] = value
        return True, pending_helpful_number
    if expect_review and not current.get("review"):
        current["review"] = value
        return False, pending_helpful_number
    if value.replace(",", "").isdigit():
        return expect_review, int(value.replace(",", ""))
    if "people found this review helpful" in value:
        current["helpful_count"] = pending_helpful_number or 0
        return expect_review, None
    if value == "1 person found this review helpful":
        current["helpful_count"] = 1
    return expect_review, pending_helpful_number


def record_from_current(
    current: dict[str, Any] | None,
    *,
    review_id_prefix: str,
    app_id: str,
    app_name: str,
    source_url: str,
) -> dict[str, Any] | None:
    if (
        not current
        or not current.get("review")
        or not current.get("rating")
        or not current.get("date")
    ):
        return None
    raw_key = "\n".join([current.get("author", ""), current["date"], current["review"]])
    digest = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:12]
    return {
        "review_id": f"{review_id_prefix}_{digest}",
        "app_id": app_id,
        "app_name": app_name,
        "source": "google_play",
        "source_url": source_url,
        "source_view": "most_relevant",
        "author": current.get("author", ""),
        "rating": current["rating"],
        "date": current["date"],
        "review": current["review"],
        "helpful_count": current.get("helpful_count", 0),
    }


def _dialog_start(lines: list[str], dialog_title: str) -> int:
    marker = f' dialog "{dialog_title}"'
    for index, line in enumerate(lines):
        if marker in line:
            return index
    return 0


if __name__ == "__main__":
    main()
