from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

DEFAULT_SOURCE = Path("data/generated/persona_review_training_cases_manual.jsonl")
DEFAULT_OUTPUT = Path("data/generated/persona_review_training_cases_manual_sanitized.jsonl")
MAX_PERSONA_WORDS = 22

WORD_RE = re.compile(r"\b[\w'-]+\b")
SENTENCE_END_RE = re.compile(r"[.!?]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show one manual case or append its persona-sanitized copy."
    )
    parser.add_argument("case_id")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--user-persona")
    args = parser.parse_args()

    if args.user_persona is None:
        print(
            json.dumps(
                read_case_context(args.source, args.case_id),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    sanitized_case = append_sanitized_case(
        source_path=args.source,
        output_path=args.output,
        case_id=args.case_id,
        user_persona=args.user_persona,
    )
    print(json.dumps(sanitized_case, ensure_ascii=False, indent=2))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def find_case(cases: list[dict[str, Any]], case_id: str) -> dict[str, Any]:
    matches = [case for case in cases if case["case_id"] == case_id]
    if not matches:
        raise ValueError(f"Unknown case_id: {case_id}")
    if len(matches) > 1:
        raise ValueError(f"Duplicate case_id in source: {case_id}")
    return matches[0]


def read_case_context(source_path: Path, case_id: str) -> dict[str, Any]:
    source_case = find_case(load_jsonl(source_path), case_id)
    return display_case(source_case)


def append_sanitized_case(
    source_path: Path,
    output_path: Path,
    case_id: str,
    user_persona: str,
) -> dict[str, Any]:
    source_case = find_case(load_jsonl(source_path), case_id)
    sanitized_persona = validate_persona(user_persona)
    existing = load_jsonl(output_path) if output_path.exists() else []
    if any(record["case_id"] == case_id for record in existing):
        raise ValueError(f"{case_id} already exists in {output_path}.")

    sanitized_case = sanitized_copy(source_case, sanitized_persona)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(sanitized_case, ensure_ascii=False) + "\n")
    return sanitized_case


def display_case(source_case: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": source_case["case_id"],
        "current_persona": source_case["input"]["user_persona"],
        "product_details": source_case["input"]["product_details"],
        "product_issue": source_case["input"]["product_issue"],
        "review": source_case["output"]["review"],
        "rating": source_case["output"]["rating"],
    }


def sanitized_copy(source_case: dict[str, Any], user_persona: str) -> dict[str, Any]:
    copied = {
        "case_id": source_case["case_id"],
        "input": dict(source_case["input"]),
        "output": dict(source_case["output"]),
    }
    copied["input"]["user_persona"] = user_persona
    return copied


def validate_persona(user_persona: str) -> str:
    persona = " ".join(user_persona.strip().split())
    if not persona:
        raise ValueError("Persona must not be empty.")
    if "\n" in user_persona:
        raise ValueError("Persona must be one line.")
    if _word_count(persona) > MAX_PERSONA_WORDS:
        raise ValueError(
            f"Persona must be {MAX_PERSONA_WORDS} words or fewer: {persona}"
        )
    if len(SENTENCE_END_RE.findall(persona.rstrip(".!?"))) > 0:
        raise ValueError("Persona must be one sentence.")
    return persona


def _word_count(text: str) -> int:
    return len(WORD_RE.findall(text))


if __name__ == "__main__":
    main()
