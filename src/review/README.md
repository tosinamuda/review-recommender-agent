# Review Simulation Agent — `src/review`

Task A: given `user_persona` + `product_details`, return one grounded review and a calibrated rating.

## Interfaces

**Web UI** — open `http://127.0.0.1:8000/` after starting the server. Pre-filled Nigerian persona and product examples; shows the 4-node pipeline trace on each run.

![Web interface — Task 1: User Modelling](../../docs/diagrams/ui-screenshot.png)

**REST API** — `POST /api/v1/review-simulation`

```bash
curl -X POST http://localhost:8000/api/v1/review-simulation \
  -H "Content-Type: application/json" \
  -d '{
    "user_persona": "Lagos-based corper, budget conscious, likes peppery food",
    "product_details": "Product: Jollof Bowl\nCategory: food\nDescription: Smoky jollof rice with grilled chicken\nPrice: 4500\nCurrency: NGN"
  }'
```

| | Field | Type | Description |
|---|---|---|---|
| **Request** | `user_persona` | string | User preferences, context, and behaviour |
| | `product_details` | string | Product name, category, price, and metadata |
| | `options.sample_count` | int 1–12 | Parallel candidate drafts (default 3) |
| **Response** | `rating` | int 1–5 | Calibrated star rating from ordinal regression |
| | `review` | string | Selected and verified review text |
| | `evidence` | object | Retrieved exemplars, candidate drafts, rating distribution |
| | `trace` | list | Per-stage execution log across 4 nodes |

## Agent Workflow

![Multi-agent pipeline — Task 1](../../docs/diagrams/task-a-pipeline.png)

```
user_persona + product_details
  │
  ▼
[1] retrieve_similar_user_reviews   ← FAISS 3-axis (persona / product / joint)
  │   returns: 5-8 exemplars {persona, product, review, rating}
  ▼
[2] generate_review_with_refinement ← DSPy ChainOfThought × 3 parallel + Refine loop
  │   GenerateReview → VerifyReview (4 checks) → retry up to 3×
  │   returns: 3 candidate drafts
  ▼
[3] select_best_review_for_persona  ← DSPy ChainOfThought (one scoring call)
  │   returns: 1 draft verbatim (no synthesis)
  ▼
[4] calibrate_rating_from_review    ← sklearn ordinal regression (no LLM)
      embed review → predict P(rating=1..5) → E[rating] → round → clamp [1,5]
```

## Why DSPy

LLM reasoning in nodes 2 and 3 uses DSPy rather than raw prompt strings.

- **Typed signatures** (`dspy.Signature`) enforce input/output contracts at the Python level,
  catching field mismatches before they become silent LLM failures.
- **`ChainOfThought`** adds a hidden reasoning step before each output field, improving
  consistency on structured multi-field outputs (`observed_experience_types`, `chosen_experience`,
  `review`).
- **`dspy.Refine`** composes generation and verification into a single callable with a bounded
  retry loop — the verifier's `critique` is fed back as context on failure, not discarded.
- **GEPA-ready**: because reasoning lives in DSPy modules with named parameters, GEPA can
  evolve the instruction text of `GenerateReview` and `VerifyReview` offline against the
  held-out rubric without touching application code.

## Key Files

| File | Role |
|---|---|
| `workflow.py` | ADK 2.0 Workflow — 4-node graph definition |
| `service.py` | `ReviewSimulationService` — orchestrates retrieve/generate/select/calibrate |
| `openrouter_reasoner.py` | DSPy modules: `GenerateReview`, `VerifyReview`, `SelectBestReview` |
| `retrieval.py` | FAISS 3-axis retrieval over `data/index/retrieval/` |
| `rating_calibration.py` | sklearn ordinal calibrator, loads `data/index/rating_calibrator.joblib` |
| `schemas.py` | Pydantic contracts: `ReviewRequest`, `ReviewSimulationResponse`, etc. |

## Reproduce

```bash
# Start server — web UI at http://127.0.0.1:8000/
uv run app-dev

# Build FAISS indexes and sklearn calibrator from raw corpus
uv run python scripts/build_review_artifacts.py

# Run offline evaluator (requires LM_MODEL + matching provider env)
uv run --env-file .env python scripts/evaluate_review_simulation.py \
  --delay-seconds 5 \
  --output data/eval/results/review_simulation_primary_summary.json

uv run --env-file .env python scripts/evaluate_review_simulation.py \
  --dataset data/eval/review_simulation_holdout_nigerian_business_eval.jsonl \
  --delay-seconds 5 \
  --output data/eval/results/review_simulation_nigerian_business_holdout_summary.json

uv run --env-file .env python scripts/evaluate_review_simulation.py \
  --dataset data/eval/review_simulation_holdout_google_maps_review_eval.jsonl \
  --delay-seconds 5 \
  --output data/eval/results/review_simulation_google_maps_holdout_summary.json

uv run --env-file .env python scripts/evaluate_review_simulation.py --jsonl  # per-case output
```

ROUGE-L and rating RMSE require exact `expected.reference_review` and
`expected.rating` labels in the eval JSONL. BERTScore-F1 is available with
`--bertscore`, but the saved run below leaves it disabled because it loads a
large local transformer.

Artifacts written to `data/index/retrieval/` (3 FAISS axes) and
`data/index/rating_calibrator.joblib`. Corpus source: `data/review_exemplars.jsonl`
(340 indexed retrieval cases).

## Evaluation Data and Results

The retrieval corpus is not used as held-out evaluation data. The saved
evaluation results use gold `expected.rating` and `expected.reference_review`
labels for every row:

| Eval set | Cases | Saved result | Pass-rate | Rating RMSE | ROUGE-L |
|---|---:|---|---:|---:|---:|
| `data/eval/review_simulation_eval.jsonl` | 28 | `data/eval/results/review_simulation_primary_summary.json` | `0.857143` | `0.944911` | `0.14936` |
| `data/eval/review_simulation_holdout_nigerian_business_eval.jsonl` | 12 | `data/eval/results/review_simulation_nigerian_business_holdout_summary.json` | `0.583333` | `1.658312` | `0.125499` |
| `data/eval/review_simulation_holdout_google_maps_review_eval.jsonl` | 6 | `data/eval/results/review_simulation_google_maps_holdout_summary.json` | `0.833333` | `0.707107` | `0.130216` |

The primary failure reasons are preserved in each saved result JSON under
`failures`.

See [../../README.md](../../README.md) for full setup and Docker instructions.
