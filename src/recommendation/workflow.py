from __future__ import annotations

import time

from google.adk import Event, Workflow
from google.adk.workflow import START
from google.genai import types

from .schemas import (
    CandidateProductSet,
    CoverageDecision,
    RankingResult,
    RecommendationRequest,
    RecommendationResponse,
    RecommendationTraceEvent,
)
from .service import (
    RecommendationService,
    candidate_set_for_viable_ids,
    elapsed_ms,
    get_recommendation_service,
)

RANK_ROUTE = "rank_and_reason"
LLM_GENERATED_ROUTE = "llm_generated"


def build_recommendation_workflow() -> Workflow:
    return Workflow(
        name="recommendation_agent",
        description=(
            "Task 2 recommendation workflow: user persona and optional context in; "
            "ranked items with per-item reasoning out."
        ),
        input_schema=RecommendationRequest,
        output_schema=RecommendationResponse,
        edges=[
            (
                START,
                retrieve_candidate_items_node,
                judge_candidate_coverage_node,
                route_by_coverage_node,
            ),
            (
                route_by_coverage_node,
                {
                    RANK_ROUTE: rank_and_reason_node,
                    LLM_GENERATED_ROUTE: generate_contextual_recommendations_node,
                },
            )
        ],
    )


def retrieve_candidate_items_node(node_input: RecommendationRequest, ctx) -> Event:
    service = get_recommendation_service()
    started_at = time.perf_counter()
    candidate_set = service.retrieve(node_input)
    ctx.state["request"] = node_input.model_dump()
    ctx.state["candidate_set"] = candidate_set.model_dump()
    ctx.state["trace"] = []
    message = f"Pulled {len(candidate_set.candidates)} candidate items from the catalogue."
    _append_trace(ctx, "retrieve_candidate_items", message, elapsed_ms(started_at))
    return _event(
        output=candidate_set,
        stage="retrieve_candidate_items",
        message=message,
    )


def judge_candidate_coverage_node(node_input: CandidateProductSet, ctx) -> Event:
    service = get_recommendation_service()
    request = RecommendationRequest.model_validate(ctx.state["request"])
    started_at = time.perf_counter()
    coverage_decision = service.judge_coverage(request, node_input)
    ctx.state["coverage_decision"] = coverage_decision.model_dump()
    message = (
        f"Coverage judged {coverage_decision.coverage_status}; "
        f"{len(coverage_decision.viable_product_ids)} candidate ids approved."
    )
    _append_trace(ctx, "judge_candidate_coverage", message, elapsed_ms(started_at))
    return _event(
        output=coverage_decision,
        stage="judge_candidate_coverage",
        message=message,
    )


def route_by_coverage_node(node_input: CoverageDecision, ctx) -> Event:
    route = RANK_ROUTE if node_input.allow_concrete_recommendations else LLM_GENERATED_ROUTE
    message = f"Selected {route} route."
    return _event(
        output=node_input,
        stage="route_by_coverage",
        message=message,
        route=route,
    )


def rank_and_reason_node(node_input: CoverageDecision, ctx) -> Event:
    service = get_recommendation_service()
    request = RecommendationRequest.model_validate(ctx.state["request"])
    candidate_set = CandidateProductSet.model_validate(ctx.state["candidate_set"])
    started_at = time.perf_counter()
    ranking_result = service.rank(
        request,
        candidate_set_for_viable_ids(candidate_set, node_input.viable_product_ids),
    )
    ctx.state["ranking_result"] = ranking_result.model_dump()
    message = (
        "Ranked coverage-approved candidates against the persona and generated per-item reasoning."
    )
    _append_trace(ctx, "rank_and_reason", message, elapsed_ms(started_at))
    response = validate_and_build_response(
        service,
        request,
        ranking_result,
        node_input,
        ctx,
    )
    return _event(
        output=response,
        stage="rank_and_reason",
        message=response.model_dump_json(),
    )


def generate_contextual_recommendations_node(node_input: CoverageDecision, ctx) -> Event:
    service = get_recommendation_service()
    request = RecommendationRequest.model_validate(ctx.state["request"])
    candidate_set = CandidateProductSet.model_validate(ctx.state["candidate_set"])
    started_at = time.perf_counter()
    generated_result = service.generate_contextual_recommendations(
        request,
        candidate_set,
        node_input,
    )
    ranking_result = RankingResult(
        candidate_set=candidate_set,
        persona_needs=generated_result.persona_needs,
        rankings=[],
    )
    ctx.state["ranking_result"] = ranking_result.model_dump()
    ctx.state["generated_result"] = generated_result.model_dump()
    message = (
        "Generated non-catalogue recommendations because catalogue coverage was insufficient."
    )
    _append_trace(ctx, "generate_contextual_recommendations", message, elapsed_ms(started_at))
    response = validate_and_build_response(
        service,
        request,
        ranking_result,
        node_input,
        ctx,
        generated_result=generated_result,
    )
    return _event(
        output=response,
        stage="generate_contextual_recommendations",
        message=response.model_dump_json(),
    )


def validate_and_build_response(
    service: RecommendationService,
    request: RecommendationRequest,
    ranking_result: RankingResult,
    coverage_decision: CoverageDecision,
    ctx,
    generated_result=None,
) -> RecommendationResponse:
    started_at = time.perf_counter()
    response = service.validate_and_build_response(
        request=request,
        ranking_result=ranking_result,
        coverage_decision=coverage_decision,
        generated_result=generated_result,
        trace=[
            RecommendationTraceEvent.model_validate(item)
            for item in ctx.state.get("trace", [])
        ],
        elapsed_ms_total=0,
    )
    recommendation_count = len(response.recommendations) + len(
        response.generated_recommendations
    )
    response.trace.append(
        RecommendationTraceEvent(
            stage="validate_and_build_response",
            message=(
                "Validated ranked item ids, attached catalogue metadata, "
                f"and returned {recommendation_count} recommendations."
            ),
            duration_ms=elapsed_ms(started_at),
        )
    )
    response.evidence.elapsed_ms = sum(event.duration_ms or 0 for event in response.trace)
    return response


def _append_trace(ctx, stage: str, message: str, duration_ms: int) -> None:
    trace = list(ctx.state.get("trace", []))
    trace.append(
        RecommendationTraceEvent(
            stage=stage,
            message=message,
            duration_ms=duration_ms,
        ).model_dump()
    )
    ctx.state["trace"] = trace


def _event(output, stage: str, message: str, route: str | None = None) -> Event:
    return Event(
        output=output,
        route=route,
        custom_metadata={"stage": stage},
        content=types.Content(
            role="model",
            parts=[types.Part(text=message)],
        ),
    )
