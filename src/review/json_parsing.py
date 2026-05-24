from __future__ import annotations

import json
from typing import cast


def parse_json_object(raw: str) -> dict[str, object]:
    text = raw.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        parsed = _parse_embedded_json_object(text, exc)
    if not isinstance(parsed, dict):
        raise ValueError(f"LLM returned non-object JSON: {raw}")
    return cast("dict[str, object]", parsed)


def _parse_embedded_json_object(text: str, original_error: json.JSONDecodeError) -> object:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"LLM returned invalid JSON: {text}") from original_error
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"LLM returned invalid JSON: {text}") from exc


def parse_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"Expected integer-compatible LLM output, got {value!r}")
