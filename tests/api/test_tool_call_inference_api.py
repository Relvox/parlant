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
from typing import Any, Iterator

from lagom import Container
from pytest import fixture

from parlant.api.sessions import EventSourceDTO
from parlant.core.agents import Agent, AgentId
from parlant.core.customers import Customer, CustomerId
from parlant.core.tools import LocalToolService, Tool, ToolId, ToolOverlap
from tests.e2e.test_utilities import API, ContextOfTest, run_server


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


async def create_test_tool(
    container: Container,
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> ToolId:
    service = container[LocalToolService]
    tool: Tool = await service.create_tool(
        name=name,
        module_path=f"test.module.{name}",
        description=description,
        parameters=parameters,
        required=[],
        overlap=ToolOverlap.NONE,
    )

    return ToolId("local", tool.name)


async def get_inferred_tool_calls(
    client: httpx.AsyncClient,
    agent_id: AgentId,
    customer_id: CustomerId,
    conversation_history: list[tuple[EventSourceDTO, str]],
    available_tools: list[ToolId],
) -> list[dict[str, Any]]:
    """TODO: fix this function."""
    response = await client.post(
        "/test/alpha/tool-call-inference",
        json={
            "agent_id": agent_id,
            "customer_id": customer_id,
            "events": [],
            "staged_events": conversation_history,
            "available_tools": available_tools,
        },
    )

    if response.status_code != 202:
        raise Exception(f"API error: {response.status_code} - {response.text}")

    result: dict[str, Any] = response.json()

    return []


async def test_weather_tool_inference(
    context: ContextOfTest,
    container: Container,
) -> None:
    with run_server(context, extra_args=["--test"]):
        async with context.api.make_client() as client:
            test_agent: Agent = await create_test_agent(client)
            test_customer: Customer = await create_test_customer(client)

            await create_test_session(client, test_agent.id, test_customer.id)

            weather_tool: ToolId = await create_test_tool(
                container,
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature",
                    },
                },
            )
            calculator_tool: ToolId = await create_test_tool(
                container,
                name="calculate",
                description="Perform a calculation",
                parameters={
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    },
                },
            )
            search_tool: ToolId = await create_test_tool(
                container,
                name="search",
                description="Search for information on the web",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
            )

            conversation_history: list[tuple[EventSourceDTO, str]] = [
                (EventSourceDTO.CUSTOMER, "Hello, I'd like to know the weather."),
                (
                    EventSourceDTO.AI_AGENT,
                    "Hi there! I'd be happy to help you with that. What location would you like to know the weather for?",
                ),
                (EventSourceDTO.CUSTOMER, "What's the weather like in New York today?"),
            ]

            tool_calls: list[dict[str, Any]] = await get_inferred_tool_calls(
                client,
                test_agent.id,
                test_customer.id,
                conversation_history,
                [weather_tool, calculator_tool, search_tool],
            )

            assert len(tool_calls) >= 1

            weather_tool_call = None
            for tool_call in tool_calls:
                if tool_call["tool_name"] == weather_tool.tool_name:
                    weather_tool_call = tool_call
                    break

            assert weather_tool_call is not None
            assert "location" in weather_tool_call["parameters"]
            assert weather_tool_call["parameters"]["location"].lower().find("new york") != -1


async def test_calculator_tool_inference(
    context: ContextOfTest,
    container: Container,
) -> None:
    with run_server(context, extra_args=["--test"]):
        async with context.api.make_client() as client:
            test_agent: Agent = await create_test_agent(client)
            test_customer: Customer = await create_test_customer(client)

            await create_test_session(client, test_agent.id, test_customer.id)

            weather_tool: ToolId = await create_test_tool(
                container,
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature",
                    },
                },
            )
            calculator_tool: ToolId = await create_test_tool(
                container,
                name="calculate",
                description="Perform a calculation",
                parameters={
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                    },
                },
            )
            search_tool: ToolId = await create_test_tool(
                container,
                name="search",
                description="Search for information on the web",
                parameters={
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                },
            )

            conversation_history: list[tuple[EventSourceDTO, str]] = [
                (EventSourceDTO.CUSTOMER, "Hello, I need help with a calculation."),
                (
                    EventSourceDTO.AI_AGENT,
                    "Hi there! I'd be happy to help you with a calculation. What would you like to calculate?",
                ),
                (EventSourceDTO.CUSTOMER, "What is 145 multiplied by 32?"),
            ]

            tool_calls: list[dict[str, Any]] = await get_inferred_tool_calls(
                client,
                test_agent.id,
                test_customer.id,
                conversation_history,
                [weather_tool, calculator_tool, search_tool],
            )

            assert len(tool_calls) >= 1

            calculator_tool_call = None
            for tool_call in tool_calls:
                if tool_call["tool_name"] == calculator_tool.tool_name:
                    calculator_tool_call = tool_call
                    break

            assert calculator_tool_call is not None
            assert "expression" in calculator_tool_call["parameters"]
            assert "145" in calculator_tool_call["parameters"]["expression"]
            assert "32" in calculator_tool_call["parameters"]["expression"]
