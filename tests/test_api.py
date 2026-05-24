from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.settings import get_settings
from recommendation.service import get_recommendation_service
from review.service import get_review_service

USER_PERSONA = "Lagos student, budget conscious, likes spicy food and large portions."
PRODUCT_DETAILS = (
    "Jollof Bowl with Grilled Chicken. Smoky jollof rice with grilled chicken and "
    "pepper sauce. NGN 4500. Delivery 35 minutes. Large portion. Spice level: high."
)

ARCHITECTURE_TRACE_STAGES = [
    "retrieve_similar_user_reviews",
    "generate_review_with_refinement",
    "select_best_review_for_persona",
    "calibrate_rating_from_review",
]

RECOMMENDATION_TRACE_STAGES = [
    "retrieve_candidate_items",
    "judge_candidate_coverage",
    "rank_and_reason",
    "validate_and_build_response",
]


def test_agent_card_exposes_a2a_streaming_contract() -> None:
    client = TestClient(create_app())

    response = client.get("/.well-known/agent-card.json")

    assert response.status_code == 200
    body = response.json()
    assert body["capabilities"]["streaming"] is True
    assert body["url"].endswith("/a2a")
    assert body["skills"][0]["id"] == "task_1_review"


def test_review_endpoint_returns_contract_payload() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/review-simulation",
        json={
            "user_persona": USER_PERSONA,
            "product_details": PRODUCT_DETAILS,
            "options": {"sample_count": 4},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert 1 <= body["rating"] <= 5
    assert "Jollof Bowl" in body["review"]
    assert body["evidence"]["model_provider"] == "test-contract-reasoner"
    assert body["evidence"]["user_persona"] == USER_PERSONA
    assert body["evidence"]["product_details"] == PRODUCT_DETAILS
    assert body["evidence"]["similar_user_reviews"]
    assert body["evidence"]["candidate_reviews"]
    assert body["evidence"]["reason"]


def test_recommendation_agent_card_exposes_a2a_streaming_contract() -> None:
    client = TestClient(create_app())

    response = client.get("/recommendation/.well-known/agent-card.json")

    assert response.status_code == 200
    body = response.json()
    assert body["capabilities"]["streaming"] is True
    assert body["url"].endswith("/recommendation/a2a")
    assert body["skills"][0]["id"] == "task_2_recommendation"


def test_recommendations_endpoint_returns_contract_payload() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/api/v1/recommendations",
        json={
            "user_persona": USER_PERSONA,
            "context": "weekday lunch near Yaba",
            "k": 5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 5
    assert [item["rank"] for item in body["recommendations"]] == [1, 2, 3, 4, 5]
    assert all(0 <= item["fit_score"] <= 1 for item in body["recommendations"])
    assert all(item["product"]["product_id"] for item in body["recommendations"])
    assert body["recommendation_mode"] == "catalogue_grounded"
    assert body["coverage"]["status"] == "sufficient"
    assert body["coverage"]["allow_concrete_recommendations"] is True
    assert body["evidence"]["candidate_count"] == 28
    assert body["evidence"]["candidates"]
    assert body["evidence"]["persona_needs"]
    assert [event["stage"] for event in body["trace"]] == RECOMMENDATION_TRACE_STAGES


@pytest.mark.asyncio
async def test_a2a_jsonrpc_stream_returns_trace_and_final_payload(monkeypatch) -> None:
    monkeypatch.setenv("BCT_APP_PUBLIC_BASE_URL", "http://testserver")
    get_settings.cache_clear()
    get_review_service.cache_clear()
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    payload = {
        "user_persona": USER_PERSONA,
        "product_details": PRODUCT_DETAILS,
        "options": {"sample_count": 5},
    }
    rpc_request = {
        "jsonrpc": "2.0",
        "id": "stream-test",
        "method": "message/stream",
        "params": {
            "message": {
                "kind": "message",
                "messageId": "stream-test-message",
                "role": "user",
                "parts": [{"kind": "text", "text": json.dumps(payload)}],
            },
            "configuration": {
                "acceptedOutputModes": ["text/plain", "application/json"],
            },
        },
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with client.stream("POST", "/a2a", json=rpc_request) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = await _read_sse_events(response)

    text_events = [_extract_a2a_text(event) for event in events]
    final_payloads = [
        json.loads(text)
        for text in text_events
        if text.startswith("{") and "rating" in text and "review" in text
    ]

    assert len(events) >= 5
    assert any("similar user reviews" in text for text in text_events)
    assert any("review refinement boundary" in text for text in text_events)
    assert any("Selected one review" in text for text in text_events)
    assert final_payloads
    assert 1 <= final_payloads[-1]["rating"] <= 5
    assert final_payloads[-1]["review"]
    assert [event["stage"] for event in final_payloads[-1]["trace"]] == ARCHITECTURE_TRACE_STAGES


@pytest.mark.asyncio
async def test_recommendation_a2a_jsonrpc_stream_returns_trace_and_final_payload(
    monkeypatch,
) -> None:
    monkeypatch.setenv("BCT_APP_PUBLIC_BASE_URL", "http://testserver")
    get_settings.cache_clear()
    get_review_service.cache_clear()
    get_recommendation_service.cache_clear()
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    payload = {
        "user_persona": USER_PERSONA,
        "context": "weekday lunch near Yaba",
        "k": 5,
    }
    rpc_request = {
        "jsonrpc": "2.0",
        "id": "recommendation-stream-test",
        "method": "message/stream",
        "params": {
            "message": {
                "kind": "message",
                "messageId": "recommendation-stream-test-message",
                "role": "user",
                "parts": [{"kind": "text", "text": json.dumps(payload)}],
            },
            "configuration": {
                "acceptedOutputModes": ["text/plain", "application/json"],
            },
        },
    }

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        async with client.stream("POST", "/recommendation/a2a", json=rpc_request) as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            events = await _read_sse_events(response)

    text_events = [_extract_a2a_text(event) for event in events]
    final_payloads = [
        json.loads(text)
        for text in text_events
        if text.startswith("{") and "recommendations" in text
    ]

    assert len(events) >= 5
    assert any("candidate items" in text for text in text_events)
    assert any("Coverage judged" in text for text in text_events)
    assert final_payloads
    assert len(final_payloads[-1]["recommendations"]) == 5
    assert [event["stage"] for event in final_payloads[-1]["trace"]] == RECOMMENDATION_TRACE_STAGES


async def _read_sse_events(response: httpx.Response) -> list[dict]:
    events = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            events.append(json.loads(line.removeprefix("data:").strip()))
    return events


def _extract_a2a_text(event: dict) -> str:
    result = event.get("result", {})
    update = (
        result.get("statusUpdate")
        or result.get("status_update")
        or result.get("artifactUpdate")
        or result.get("artifact_update")
        or result
    )
    message = (update.get("status") or {}).get("message") or result.get("message") or result.get(
        "msg"
    )
    artifact = update.get("artifact") or result.get("artifact")
    return _extract_parts_text(message) or _extract_parts_text(artifact)


def _extract_parts_text(container: dict | None) -> str:
    if not container:
        return ""
    texts = []
    for part in container.get("parts", []):
        text = part.get("text") or (part.get("root") or {}).get("text")
        if text:
            texts.append(text)
    return "\n".join(texts)
