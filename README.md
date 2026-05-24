# BCT Hackathon — LLM Agent Challenge

DSN × BCT LLM Agent Challenge submission. Two tasks, one application.

- **Task A (User Modelling):** simulate reviews and ratings → [`src/review/`](src/review/README.md)
- **Task B (Recommendation):** coverage-aware personalised recommendations → [`src/recommendation/`](src/recommendation/README.md)

## 1. Setup

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env — set LM_MODEL and the matching provider API key
# This submission was evaluated with: LM_MODEL=openrouter/openai/gpt-oss-120b
# Option A (OpenRouter): get a free key at https://openrouter.ai/keys
# See .env.example for Ollama (local) and other OpenAI-compatible alternatives
```

## 2. Run

Two options — from source or Docker.

**From source:**
```bash
uv run app-dev        # reload mode — watches src/ and loads .env by default
```
Non-reload: `uv run app-start`. Override port: `uv run app-dev --port 8001`.

**Docker:**
```bash
docker build -t bct-agent . && docker run -p 8000:8000 --env-file .env bct-agent
```

Once running, open **http://127.0.0.1:8000/** in your browser to use the web UI, or send requests directly to the REST API (see §4 below).

## 3. Test

**Web UI** — navigate to `http://127.0.0.1:8000/`, select Task 1 or Task 2, fill in the pre-loaded Nigerian persona and product/context fields, and press **Run simulation** / **Get recommendations**. The pipeline trace and result appear in the right panel.

**REST API** — send a `curl` request to the relevant endpoint (see §4 below for the full request/response schema):

## 4. API Reference

The application exposes a web UI (`GET /`) and two REST endpoints.

### Task A — Review Simulation · `POST /api/v1/review-simulation`

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

### Task B — Recommendation · `POST /api/v1/recommendations`

```bash
curl -X POST http://localhost:8000/api/v1/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "user_persona": "Lagos-based student, budget conscious, likes spicy food",
    "context": "weekday lunch, near Yaba",
    "category": "food",
    "k": 5
  }'
```

| | Field | Type | Description |
|---|---|---|---|
| **Request** | `user_persona` | string | User preferences, context, and behaviour |
| | `context` | string | Situational signals: time, location, occasion |
| | `category` | string\|null | Optional category filter |
| | `k` | int 1–10 | Number of recommendations (default 5) |
| | `candidate_items` | list | Optional — bypasses retrieval for explicit re-ranking |
| **Response** | `recommendation_mode` | string | `catalogue_grounded` · `llm_generated` · `coverage_limited` · `request_supplied_candidates` |
| | `coverage` | object | Suitability status, decision boolean, unsupported signals |
| | `recommendations` | list | Ranked catalogue items with fit_score, headline, reasoning |
| | `generated_recommendations` | list | Non-catalogue fallback items (excluded from NDCG/Hit Rate) |
| | `trace` | list | Per-stage execution log with durations |

### Other endpoints

`GET /` web UI · `GET /health` health check · `GET /.well-known/agent-card.json` A2A card · `POST /a2a` A2A streaming

## 5. Rebuild Artifacts

```bash
uv run python scripts/build_review_artifacts.py        # Task A: FAISS + calibrator
uv run python scripts/build_recommendation_artifacts.py # Task B: FAISS + catalogue
```

## 6. Model Configuration

The model used for this submission is **GPT OSS 120b** via OpenRouter:
```
LM_MODEL=openrouter/openai/gpt-oss-120b
```

Change `LM_MODEL` in `.env` to swap models. No code changes needed. Local runs
pass `.env` into the process with `uv run --env-file .env ...`; LiteLLM then
discovers provider variables such as `OPENROUTER_API_KEY` directly from the
process environment.

```bash
# OpenRouter  (prefix: openrouter/<provider>/<model>)
OPENROUTER_API_KEY=sk-or-...
LM_MODEL=openrouter/openai/gpt-oss-120b   # model used for this submission

# Any OpenAI-compatible API  (prefix: openai/<model>)
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=https://api.openai.com/v1
LM_MODEL=openai/gpt-4o

# Ollama local  (no API key needed, prefix: openai/<model>)
OPENAI_API_KEY=anything
OPENAI_API_BASE=http://localhost:11434/v1
LM_MODEL=openai/llama3
```

## 7. Data and Model Disclosure

**Datasets used:**
| Dataset | Source | Use |
|---|---|---|
| Nigerian Play Store reviews | Google Play (Chowdeck, GTWorld, Konga, myMTN, PalmPay, NINAuth, NIS Mobile) | Task A retrieval corpus (340 cases) |
| Nigerian business cases | Manual + Google Maps-visible signals | Task A holdout evaluation (18 cases) |
| Yelp Academic Dataset | [Yelp Open Dataset](https://www.yelp.com/dataset) | Task B catalogue/persona retrieval + 100 held-out eval cases |
| Hand-curated Nigerian catalogue and personas | Manual + Codex-assisted curation | Task B Nigerian-locale retrieval data, not held-out eval |

**Models used:**
| Model | Source | License | Use |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | HuggingFace | MIT | Embeddings (retrieval + calibrator) |
| `openrouter/openai/gpt-oss-120b` (GPT OSS 120b) — used for this submission; configurable via `LM_MODEL` to any OpenRouter or local model | OpenRouter | Varies | LLM generation, ranking, coverage judgment |
| scikit-learn `LogisticRegression` | scikit-learn | BSD | Task A rating calibrator |

**Frameworks:** Google ADK 2.0 (Apache 2.0), DSPy (MIT), LiteLLM (MIT), FAISS (MIT), FastAPI (MIT).

## 8. Solution Papers

- [docs/solution-paper-task-a.md](docs/solution-paper-task-a.md) — Task A (User Modelling)
- [docs/solution-paper-task-b.md](docs/solution-paper-task-b.md) — Task B (Recommendation)

## 9. Evaluate

```bash
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

uv run --env-file .env python scripts/evaluate_recommendations.py \
  --delay-seconds 5 \
  --output data/eval/results/recommendation_eval_summary.json

uv run ruff check . && uv run ty check && uv run pytest  # no API key required
```

Task A reports rubric pass-rate, rating RMSE, and ROUGE-L. BERTScore-F1 is available with `--bertscore` (opt-in — loads a large local transformer).

Latest saved eval run using `LM_MODEL=openrouter/openai/gpt-oss-120b`:

| Task | Cases | Metrics |
|---|---:|---|
| Task A primary | 28 | pass-rate `0.857`; rating RMSE `0.945`; ROUGE-L `0.149` |
| Task A Nigerian business holdout | 12 | pass-rate `0.583`; rating RMSE `1.658`; ROUGE-L `0.125` |
| Task A Google Maps holdout | 6 | pass-rate `0.833`; rating RMSE `0.707`; ROUGE-L `0.130` |
| Task B held-out Yelp | 100 | coverage accuracy `0.86`; hit@5 `0.03`; NDCG@10 `0.028` |
