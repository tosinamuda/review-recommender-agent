from __future__ import annotations

from a2a.server.apps import A2AFastAPIApplication, A2ARESTFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotificationConfigStore, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TransportProtocol
from fastapi import FastAPI
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor
from google.adk.a2a.utils.agent_to_a2a import to_a2a

from app.settings import Settings
from review.workflow import build_root_workflow

from .a2a_runtime import build_runner


def add_routes(app: FastAPI, settings: Settings) -> AgentCard:
    workflow = build_root_workflow()
    card = build_review_agent_card(settings)
    runner = build_runner(workflow)
    request_handler = DefaultRequestHandler(
        agent_executor=A2aAgentExecutor(runner=runner),
        task_store=InMemoryTaskStore(),
        push_config_store=InMemoryPushNotificationConfigStore(),
    )

    jsonrpc_app = A2AFastAPIApplication(agent_card=card, http_handler=request_handler)
    jsonrpc_app.add_routes_to_app(app, rpc_url="/a2a")

    rest_app = A2ARESTFastAPIApplication(agent_card=card, http_handler=request_handler)
    rest_routes = rest_app.build().routes
    for route in rest_routes:
        if str(getattr(route, "path", "")).startswith("/v1/"):
            app.router.routes.append(route)

    app.mount(
        "/adk-to-a2a",
        to_a2a(workflow, agent_card=card, runner=runner),
        name="adk_to_a2a_jsonrpc",
    )
    return card


def build_review_agent_card(settings: Settings) -> AgentCard:
    base = settings.public_base_url.rstrip("/")
    return AgentCard(
        name="BCT Review Simulation Agent",
        description=(
            "Simulates a written review from a user persona and product details, "
            "then calibrates the rating from the selected review."
        ),
        url=f"{base}/a2a",
        preferred_transport=TransportProtocol.jsonrpc,
        additional_interfaces=[
            AgentInterface(transport=TransportProtocol.http_json, url=base),
        ],
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
        supports_authenticated_extended_card=False,
        skills=[
            AgentSkill(
                id="task_1_review",
                name="Task 1 Review Simulation",
                description=(
                    "Accepts a user persona and product details, retrieves similar "
                    "user reviews, generates candidate reviews, and returns one final "
                    "review with a calibrated rating."
                ),
                tags=["review-simulation", "adk-workflow", "dspy"],
                input_modes=["text/plain", "application/json"],
                output_modes=["text/plain", "application/json"],
            )
        ],
    )
