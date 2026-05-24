from __future__ import annotations

from a2a.server.apps import A2AFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotificationConfigStore, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, TransportProtocol
from fastapi import FastAPI
from google.adk.a2a.executor.a2a_agent_executor import A2aAgentExecutor

from app.settings import Settings
from recommendation.workflow import build_recommendation_workflow

from .a2a_runtime import build_runner


def add_routes(app: FastAPI, settings: Settings) -> AgentCard:
    workflow = build_recommendation_workflow()
    card = build_recommendation_agent_card(settings)
    runner = build_runner(workflow)
    request_handler = DefaultRequestHandler(
        agent_executor=A2aAgentExecutor(runner=runner),
        task_store=InMemoryTaskStore(),
        push_config_store=InMemoryPushNotificationConfigStore(),
    )
    jsonrpc_app = A2AFastAPIApplication(agent_card=card, http_handler=request_handler)
    jsonrpc_app.add_routes_to_app(
        app,
        agent_card_url="/recommendation/.well-known/agent-card.json",
        rpc_url="/recommendation/a2a",
    )
    return card


def build_recommendation_agent_card(settings: Settings) -> AgentCard:
    base = settings.public_base_url.rstrip("/")
    return AgentCard(
        name="BCT Recommendation Agent",
        description=(
            "Ranks catalogue-grounded item recommendations for a user persona, "
            "judges candidate coverage, and returns per-item reasoning plus evidence."
        ),
        url=f"{base}/recommendation/a2a",
        preferred_transport=TransportProtocol.jsonrpc,
        additional_interfaces=[
            AgentInterface(
                transport=TransportProtocol.http_json,
                url=f"{base}/recommendation",
            ),
        ],
        version="0.1.0",
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text/plain", "application/json"],
        default_output_modes=["text/plain", "application/json"],
        supports_authenticated_extended_card=False,
        skills=[
            AgentSkill(
                id="task_2_recommendation",
                name="Task 2 Recommendation",
                description=(
                    "Accepts a user persona and optional context, retrieves candidate "
                    "items, judges coverage, then ranks concrete matches or returns "
                    "generated non-catalogue recommendations."
                ),
                tags=["recommendation", "adk-workflow", "dspy"],
                input_modes=["text/plain", "application/json"],
                output_modes=["text/plain", "application/json"],
            )
        ],
    )
