from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.settings import get_settings
from recommendation.evaluation import (
    dumps_metrics,
    evaluate_recommendation_service,
    load_recommendation_eval_cases,
)
from recommendation.service import get_recommendation_service
from shared.openrouter_lm import configure_dspy_lm

DEFAULT_EVAL = Path("data/eval/recommendation_eval_cases.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 2 recommendations.")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Sleep this many seconds between eval cases to reduce provider rate-limit pressure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the metrics JSON to this file instead of stdout.",
    )
    args = parser.parse_args()
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be non-negative.")
    configure_dspy_lm(get_settings())

    metrics = evaluate_recommendation_service(
        service=get_recommendation_service(),
        eval_cases=load_recommendation_eval_cases(args.eval),
        k=args.k,
        delay_seconds=args.delay_seconds,
        progress=print_progress,
    )
    output = dumps_metrics(metrics) + "\n"
    if args.output is None:
        print(output, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output, encoding="utf-8")


def print_progress(done: int, total: int, case_id: str) -> None:
    print(f"Evaluated {done}/{total}: {case_id}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
