from __future__ import annotations

from google.adk import Event, Workflow
from google.adk.workflow import START
from google.genai import types

from review.reasoning import aggregate_candidates
from review.schemas import (
    Candidates,
    ExemplarSet,
    ResponseEvidence,
    ReviewRequest,
    ReviewSimulationResponse,
    SelectedCandidate,
    TraceEvent,
)
from review.service import (
    combined_reason,
    get_review_service,
)


def build_root_workflow() -> Workflow:
    return Workflow(
        name="review_agent",
        description=(
            "Task 1 review simulation workflow: user persona and product details in; "
            "one grounded review with a calibrated rating out."
        ),
        input_schema=ReviewRequest,
        output_schema=ReviewSimulationResponse,
        edges=[
            (
                START,
                retrieve_similar_user_reviews_node,
                generate_review_with_refinement_node,
                select_best_review_for_persona_node,
                calibrate_rating_from_review_node,
            )
        ],
    )


def retrieve_similar_user_reviews_node(node_input: ReviewRequest, ctx) -> Event:
    service = get_review_service()
    exemplar_set = service.retrieve(node_input.user_persona, node_input.product_details)
    ctx.state["request"] = node_input.model_dump()
    ctx.state["exemplar_set"] = exemplar_set.model_dump()
    ctx.state["trace"] = []
    message = (
        f"Retrieved {len(exemplar_set.exemplars)} similar user reviews "
        "as the user_review_history proxy."
    )
    _append_trace(ctx, "retrieve_similar_user_reviews", message)
    return _event(
        output=exemplar_set,
        stage="retrieve_similar_user_reviews",
        message=message,
    )


def generate_review_with_refinement_node(node_input: ExemplarSet, ctx) -> Event:
    service = get_review_service()
    request = ReviewRequest.model_validate(ctx.state["request"])
    candidates = service.generate_candidates(
        request.user_persona,
        request.product_details,
        node_input,
        sample_count=request.options.sample_count,
    )
    ctx.state["candidates"] = candidates.model_dump()
    message = (
        f"Generated {len(candidates.samples)} candidate drafts through "
        "the review refinement boundary."
    )
    _append_trace(ctx, "generate_review_with_refinement", message)
    return _event(
        output=candidates,
        stage="generate_review_with_refinement",
        message=message,
    )


def select_best_review_for_persona_node(node_input: Candidates, ctx) -> Event:
    service = get_review_service()
    request = ReviewRequest.model_validate(ctx.state["request"])
    exemplar_set = ExemplarSet.model_validate(ctx.state["exemplar_set"])
    aggregate = aggregate_candidates(node_input)
    selected = service.select_best(
        request.user_persona,
        request.product_details,
        exemplar_set,
        aggregate,
    )
    message = f"Selected one review from {aggregate.candidate_count} candidate drafts."
    _append_trace(ctx, "select_best_review_for_persona", message)
    return _event(
        output=selected,
        stage="select_best_review_for_persona",
        message=message,
    )


def calibrate_rating_from_review_node(node_input: SelectedCandidate, ctx) -> Event:
    service = get_review_service()
    request = ReviewRequest.model_validate(ctx.state["request"])
    exemplar_set = ExemplarSet.model_validate(ctx.state["exemplar_set"])
    candidates = Candidates.model_validate(ctx.state["candidates"])
    calibrated = service.calibrate_rating(node_input)
    _append_trace(ctx, "calibrate_rating_from_review", calibrated.calibration_reason)
    response = ReviewSimulationResponse(
        rating=calibrated.rating,
        review=calibrated.review,
        evidence=ResponseEvidence(
            user_persona=request.user_persona,
            product_details=request.product_details,
            similar_user_reviews=exemplar_set.exemplars,
            candidate_reviews=[candidate.review for candidate in candidates.samples],
            reason=combined_reason(calibrated),
            rating_continuous=calibrated.rating_continuous,
            rating_distribution=calibrated.rating_distribution,
            verifier_attempts=calibrated.verifier_attempts,
            model_provider=service.provider_name,
        ),
        trace=[TraceEvent.model_validate(item) for item in ctx.state.get("trace", [])],
    )
    return _event(
        output=response,
        stage="calibrate_rating_from_review",
        message=response.model_dump_json(),
    )


def _append_trace(ctx, stage: str, message: str) -> None:
    trace = list(ctx.state.get("trace", []))
    trace.append(TraceEvent(stage=stage, message=message).model_dump())
    ctx.state["trace"] = trace


def _event(output, stage: str, message: str) -> Event:
    return Event(
        output=output,
        custom_metadata={"stage": stage},
        content=types.Content(
            role="model",
            parts=[types.Part(text=message)],
        ),
    )
