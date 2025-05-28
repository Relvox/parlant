# Copyright 2025 Emcie Co Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import httpx
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any, Iterator, Sequence

from lagom import Container
from pytest import fixture

from parlant.core.agents import Agent, AgentId
from parlant.core.common import generate_id
from parlant.core.customers import Customer, CustomerId
from parlant.core.engines.alpha.guideline_matching.guideline_match import (
    GuidelineMatch,
    GuidelineMatchDTO,
    PreviouslyAppliedType,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.sessions import EventKind, EventSource
from parlant.core.tags import Tag, TagId
from parlant.core.tools import ToolContext, ToolId, ToolResult
from parlant.core.services.tools.plugins import tool
from tests.e2e.test_utilities import API, ContextOfTest, run_server
from tests.test_utilities import run_service_server


@fixture
def context() -> Iterator[ContextOfTest]:
    with tempfile.TemporaryDirectory(prefix="parlant-server_cli_test_") as home_dir:
        home_dir_path = Path(home_dir)

        yield ContextOfTest(
            home_dir=home_dir_path,
            api=API(),
        )


async def create_test_agent(
    client: httpx.AsyncClient,
) -> Agent:
    response = await client.post(
        "/agents",
        json={
            "name": "test-agent",
        },
    )
    result_json: dict[str, Any] = response.json()
    result: Agent = Agent(
        creation_utc=datetime.now(timezone.utc),
        **result_json,
    )
    return result


async def create_test_customer(
    client: httpx.AsyncClient,
) -> Customer:
    response = await client.post(
        "/customers",
        json={
            "name": "test-customer",
        },
    )
    result_json: dict[str, Any] = response.json()
    result: Customer = Customer(**result_json)
    return result


async def create_test_session(
    client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_id: CustomerId,
) -> None:
    await client.post(
        "/sessions",
        json={
            "agent_id": agent_id,
            "customer_id": customer_id,
        },
    )


def make_event_params(
    source: EventSource,
    data: dict[str, Any] = {},
    kind: EventKind = EventKind.CUSTOM,
) -> dict[str, Any]:
    return {
        "source": source.value,
        "kind": kind.value,
        "creation_utc": str(datetime.now(timezone.utc)),
        "correlation_id": "dummy_correlation_id",
        "data": data,
        "deleted": False,
    }


def create_guideline_match(
    condition: str,
    action: str,
    score: int,
    rationale: str,
    tags: list[TagId],
    guideline_previously_applied: PreviouslyAppliedType = PreviouslyAppliedType.NO,
    guideline_is_continuous: bool = False,
    should_reapply: bool = False,
) -> GuidelineMatch:
    guideline = Guideline(
        id=GuidelineId(generate_id()),
        creation_utc=datetime.now(timezone.utc),
        content=GuidelineContent(
            condition=condition,
            action=action,
        ),
        enabled=True,
        tags=tags,
        metadata={},
    )

    return GuidelineMatch(
        guideline=guideline,
        score=score,
        rationale=rationale,
        guideline_previously_applied=guideline_previously_applied,
        guideline_is_continuous=guideline_is_continuous,
        should_reapply=should_reapply,
    )


async def get_inferred_tool_calls(
    client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_id: CustomerId,
    tool_enabled_guideline_matches: Sequence[GuidelineMatchDTO],
    conversation_history: list[dict[str, Any]],
    available_tools: list[ToolId],
) -> list[dict[str, Any]]:
    """Get inferred tool calls via the API endpoint."""
    response = await client.post(
        "/test/alpha/tool-call-inference",
        json={
            "agent_id": agent_id,
            "customer_id": customer_id,
            "context_variables": [],
            "interaction_history": [],
            "terms": [],
            "ordinary_guideline_matches": [],
            "tool_enabled_guideline_matches": tool_enabled_guideline_matches,
            "journeys": [],
            "staged_events": conversation_history,
            "available_tools": [
                f"{tool.service_name}:{tool.tool_name}" for tool in available_tools
            ],
        },
    )

    if response.status_code != 202:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    result: dict[str, Any] = response.json()

    return []
    return result.get("tool_calls", [])


@tool
def get_weather(context: ToolContext, location: str, unit: str = "celsius") -> ToolResult:
    """Get the current weather for a location"""
    return ToolResult(f"Weather in {location}: 20°{unit[0].upper()}")


@tool
def calculate(context: ToolContext, expression: str) -> ToolResult:
    """Perform a calculation"""
    return ToolResult(f"Result of {expression}: 42")


@tool
def search(context: ToolContext, query: str) -> ToolResult:
    """Search for information on the web"""
    return ToolResult(f"Search results for: {query}")


async def test_weather_tool_inference(
    context: ContextOfTest,
    container: Container,
) -> None:
    with run_server(context, extra_args=["--test", "--log-level", "debug"]):
        async with context.api.make_client() as client:
            test_agent: Agent = await create_test_agent(client)
            test_customer: Customer = await create_test_customer(client)

            await create_test_session(client, test_agent.id, test_customer.id)

            async with run_service_server([get_weather, calculate, search]) as server:
                await client.put(
                    "/services/my_sdk_service", json={"kind": "sdk", "sdk": {"url": server.url}}
                )

                import asyncio

                await asyncio.sleep(0.5)

                conversation_history: list[dict[str, Any]] = [
                    make_event_params(
                        source=EventSource.CUSTOMER,
                        data={"content": "Hello, I'd like to know the weather."},
                    ),
                    make_event_params(
                        source=EventSource.AI_AGENT,
                        data={
                            "content": "Hi there! I'd be happy to help you with that. What location would you like to know the weather for?"
                        },
                    ),
                    make_event_params(
                        source=EventSource.CUSTOMER,
                        data={"content": "What's the weather like in New York today?"},
                    ),
                ]

                weather_tool = ToolId(service_name="my_sdk_service", tool_name="get_weather")
                calculator_tool = ToolId(service_name="my_sdk_service", tool_name="calculate")
                search_tool = ToolId(service_name="my_sdk_service", tool_name="search")

                tool_enabled_guideline_matches = [
                    (
                        create_guideline_match(
                            condition="Get the current weather for a location",
                            action="a customer asks for the temperature in New York today",
                            score=9,
                            rationale="customer asks for weather today",
                            tags=[Tag.for_agent_id(test_agent.id)],
                        ),
                        [weather_tool],
                    )
                ]

                tool_calls: list[dict[str, Any]] = await get_inferred_tool_calls(
                    client,
                    test_agent.id,
                    test_customer.id,
                    [
                        GuidelineMatchDTO(
                            guideline_id=match.guideline.id,
                            score=match.score,
                            rationale=match.rationale,
                            guideline_previously_applied=match.guideline_previously_applied,
                            guideline_is_continuous=match.guideline_is_continuous,
                            should_reapply=match.should_reapply,
                            associated_tool_ids=[t.to_string() for t in tool_ids],
                        )
                        for (match, tool_ids) in tool_enabled_guideline_matches
                    ],
                    conversation_history,
                    [weather_tool, calculator_tool, search_tool],
                )

                assert len(tool_calls) >= 1

                weather_tool_call = None
                for tool_call in tool_calls:
                    if tool_call["tool_name"] == "get_weather":
                        weather_tool_call = tool_call
                        break

                assert weather_tool_call is not None
                assert "location" in weather_tool_call["parameters"]
                assert weather_tool_call["parameters"]["location"].lower().find("new york") != -1
