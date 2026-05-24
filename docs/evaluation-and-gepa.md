# Evaluation Dataset And GEPA

## Dataset Roles

`data/review_exemplars.jsonl` is the normalized retrieval/training corpus. It
contains 340 sanitized Nigerian Play Store review cases and preserves the
architecture fields `case_id`, `user_persona`, `product_details`, `review`,
`rating`, and `product_issue`.

`data/eval/review_simulation_eval.jsonl` is the gold evaluation set. It should not
be used for retrieval. It contains 28 cases and mirrors the browser/A2A
contract: each case stores raw `user_persona` text and raw `product_details`
text, plus the expected rating range, required review signals, forbidden review
claims, and expected friction points.

`data/eval/review_simulation_holdout_nigerian_business_eval.jsonl` is an
additional Task 1 holdout over Nigerian businesses from the recommendation
catalogue. These businesses are not in the Task 1 review-exemplar FAISS corpus.
Personas describe stable user context only; business names, locations, menus,
and listing details appear only in `product_details`.

`data/eval/review_simulation_holdout_google_maps_review_eval.jsonl` is a second
Task 1 holdout backed by Google Maps-visible Nigerian business review signals.
The evidence block records the source URL, listing-level rating context, and
paraphrased review signals. Those details are deliberately excluded from the
persona and appear only in `product_details` and `source_evidence`.

This split matters because using the same records for retrieval and evaluation
would reward memorization rather than behavioural fidelity.

## Task 2 Held-Out Evaluation

Task 2 separates the similar-persona retrieval artifact from the reported
ranking evaluation:

- `data/recommendation/persona_cases.jsonl` is embedded into `persona.faiss` and
  can contribute similar-persona liked-product scores at runtime.
- `data/eval/recommendation_eval_cases.jsonl` is the held-out reporting set used
  by `scripts/evaluate_recommendations.py`.

The held-out Yelp cases in `data/eval/recommendation_eval_cases.jsonl` are not
present in `data/recommendation/persona_cases.jsonl`. Their relevant products
are still present in `data/recommendation/product_catalogue.jsonl`; this is
required for NDCG@10 and Hit Rate@10 because ranking evaluation needs a fixed
candidate catalogue. The leakage boundary is the query persona, source user,
review IDs, and relevance labels, not the existence of item metadata in the
catalogue.

## Ablation Framing

Ablation runs should use the full system as the reference condition, then toggle
one component at a time over the same held-out Task 1 or Task 2 cases. That is
closer to feature-flagged offline evaluation than live A/B testing: traffic is
not randomized, users are not split into experimental buckets, and every variant
sees the same fixed cases.

A separate cheap baseline can still be useful for sanity checks, but it is not
the main comparison for an ablation table. The paper should report deltas from
the full system so each row answers what a component contributes or breaks.

## Runtime Workflow

The public live workflow trace follows the four architecture v3.1 boundaries:

```text
raw persona + raw product
-> retrieve_similar_user_reviews
-> generate_review_with_refinement
-> select_best_review_for_persona
-> calibrate_rating_from_review
```

The boundary adapter exists so existing aspect, retrieval, and rendering code can
read fields such as price and delivery time from labels like `Price:` and
`Delivery minutes:`. It runs inside `retrieve_similar_user_reviews`; candidate
aggregation runs inside `select_best_review_for_persona`. Retrieval uses three
persisted FAISS axes (`persona`, `product`, and `joint`) over
`BAAI/bge-small-en-v1.5` embeddings. `calibrate_rating_from_review` loads the
persisted sklearn ordinal-style threshold calibrator and returns both the
rounded rating and the rating probability distribution.

GEPA does not belong in this runtime path. Running GEPA per request would be slow,
expensive, and nondeterministic.

## Where GEPA Belongs

Use GEPA offline, before the demo/runtime:

```text
gold eval dataset
-> run full review simulation program
-> score output with rubric feedback
-> GEPA optimizes DSPy instructions
-> save optimized DSPy program or prompt artifact
-> runtime loads the optimized generator/selector
```

The GEPA target should be the DSPy text components in
`src/review_simulation/domain/openrouter_reasoner.py`:

- `GenerateReview`: improve review generation from user persona/product,
- exemplars, observed experience types, chosen experience, and candidate index.
- `VerifyReview`: improve the verifier used by the internal
  `dspy.Refine(N=3)` loop.
- `SelectBestReview`: improve final candidate selection.

Do not use GEPA to optimize:

- input parsing, because the current contract is raw persona/product details text
- retrieval count, because it is a product decision fixed at 5
- the sklearn calibrator artifact, because rating calibration is trained and
  verified separately from prompt optimization

## Metric Shape

The GEPA metric should wrap the same rubric used by the local evaluator:

- rating is within the expected range
- review contains required persona/product terms
- review avoids forbidden claims
- expected friction points are surfaced
- quality checks have no warnings

GEPA can consume textual feedback such as:

```text
missing required term: corper
missing friction point: spice_below_preferred
forbidden term present: pepper level was solid
rating 4 outside expected range 2-3
```

That feedback is more useful than a scalar score alone because GEPA's strength is
reflective prompt evolution.

## Evaluation Commands

Run the production evaluator:

```bash
uv run --env-file .env python scripts/evaluate_review_simulation.py
```

This uses the same production path as the app: FAISS retrieval, DSPy/OpenRouter
generation and selection, and the persisted sklearn calibrator. It requires
`LM_MODEL` and the matching LiteLLM provider variables in the process
environment.

BERTScore-F1, ROUGE-L, and rating RMSE are populated only for rows with exact
`expected.reference_review` and `expected.rating` labels. Rows that remain
rubric-only still report rubric pass-rate and per-case failure reasons.

For per-case output:

```bash
uv run --env-file .env python scripts/evaluate_review_simulation.py --jsonl
```
