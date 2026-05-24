from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import sleep

from app.settings import get_settings
from review.eval_cases import (
    DEFAULT_EVAL_DATASET,
    ReviewEvalCase,
    ReviewEvalResult,
    load_review_eval_cases,
    rating_rmse,
    rouge_l_f1,
    score_review_response,
)
from review.service import get_review_service
from shared.openrouter_lm import configure_dspy_lm


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Task 1 review simulation.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_EVAL_DATASET,
        help="JSONL evaluation dataset path.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Print one result JSON object per case instead of a summary.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Sleep this many seconds between eval cases to reduce provider rate-limit pressure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the eval output to this file instead of stdout.",
    )
    parser.add_argument(
        "--bertscore",
        action="store_true",
        help="Compute BERTScore F1. This is opt-in because it loads a large local transformer.",
    )
    args = parser.parse_args()
    if args.delay_seconds < 0:
        parser.error("--delay-seconds must be non-negative.")
    configure_dspy_lm(get_settings())

    cases = load_review_eval_cases(args.dataset)
    service = get_review_service()
    results = []
    for index, case in enumerate(cases):
        results.append(score_review_response(case, service.run(case.to_request())))
        print(
            f"Evaluated {index + 1}/{len(cases)}: {case.case_id}",
            file=sys.stderr,
            flush=True,
        )
        if args.delay_seconds and index < len(cases) - 1:
            sleep(args.delay_seconds)
    attach_paper_metrics(cases, results, compute_bertscore=args.bertscore)

    if args.jsonl:
        output = "\n".join(result.model_dump_json() for result in results) + "\n"
        write_or_print(output, args.output)
        return

    output = json.dumps(
        summarize_results(cases, results, bertscore_enabled=args.bertscore),
        indent=2,
    ) + "\n"
    write_or_print(output, args.output)


def write_or_print(output: str, path: Path | None) -> None:
    if path is None:
        print(output, end="")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(output, encoding="utf-8")


def attach_paper_metrics(
    cases: list[ReviewEvalCase],
    results: list[ReviewEvalResult],
    *,
    compute_bertscore: bool,
) -> None:
    for case, result in zip(cases, results, strict=True):
        if case.expected.rating is not None:
            result.rating_error = float(result.rating - case.expected.rating)
        if case.expected.reference_review:
            result.rouge_l = round(
                rouge_l_f1(result.review, case.expected.reference_review),
                6,
            )

    reference_pairs = [
        (case.expected.reference_review, result.review)
        for case, result in zip(cases, results, strict=True)
        if case.expected.reference_review
    ]
    if compute_bertscore and len(reference_pairs) == len(cases):
        bert_scores = compute_bertscore_f1(
            candidates=[candidate for _reference, candidate in reference_pairs],
            references=[reference for reference, _candidate in reference_pairs],
        )
        if bert_scores is not None:
            for result, bertscore_f1 in zip(results, bert_scores, strict=True):
                result.bertscore_f1 = round(bertscore_f1, 6)


def summarize_results(
    cases: list[ReviewEvalCase],
    results: list[ReviewEvalResult],
    *,
    bertscore_enabled: bool,
) -> dict[str, object]:
    passed = sum(result.passed for result in results)
    avg_score = sum(result.score for result in results) / len(results)
    rating_errors = [
        result.rating_error
        for result in results
        if result.rating_error is not None
    ]
    rouge_l_scores = [
        result.rouge_l for result in results if result.rouge_l is not None
    ]
    bertscore_f1_scores = [
        result.bertscore_f1
        for result in results
        if result.bertscore_f1 is not None
    ]
    missing_gold_fields = missing_paper_metric_inputs(cases)
    return {
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "average_score": round(avg_score, 4),
        "rubric_pass_rate": round(passed / len(results), 6),
        "rating_rmse": rounded_optional(rating_rmse(rating_errors))
        if len(rating_errors) == len(results)
        else None,
        "rouge_l": rounded_optional(average(rouge_l_scores))
        if len(rouge_l_scores) == len(results)
        else None,
        "bertscore_f1": rounded_optional(average(bertscore_f1_scores))
        if len(bertscore_f1_scores) == len(results)
        else None,
        "bertscore_enabled": bertscore_enabled,
        "paper_metric_inputs": {
            "exact_rating_count": len(rating_errors),
            "reference_review_count": len(rouge_l_scores),
            "missing": missing_gold_fields,
        },
        "failures": [
            {
                "case_id": result.case_id,
                "score": result.score,
                "feedback": result.feedback,
            }
            for result in results
            if not result.passed
        ],
    }


def compute_bertscore_f1(
    *,
    candidates: list[str],
    references: list[str],
) -> list[float] | None:
    try:
        from bert_score import score as bert_score
    except ModuleNotFoundError:
        return None

    _precision, _recall, f1_scores = bert_score(
        candidates,
        references,
        lang="en",
        verbose=False,
        rescale_with_baseline=True,
    )
    return [float(score) for score in f1_scores.tolist()]


def missing_paper_metric_inputs(cases: list[ReviewEvalCase]) -> list[str]:
    missing = []
    if any(case.expected.rating is None for case in cases):
        missing.append("expected.rating")
    if any(not case.expected.reference_review for case in cases):
        missing.append("expected.reference_review")
    return missing


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def rounded_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


if __name__ == "__main__":
    main()
