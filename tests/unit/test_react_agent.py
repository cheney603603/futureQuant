"""
Tests for ReActAgent
"""

import pytest

from futureQuant.agent.react_base import ReActAgent
from futureQuant.agent.tools import Tool, ToolResult, tool
from futureQuant.core.llm_client import LLMResponse


class MockLLMClient:
    """Mock LLM client for ReAct testing"""

    def __init__(self, responses):
        """responses: list of LLMResponse to cycle through"""
        self.responses = responses
        self.index = 0
        self.history = []

    def chat(self, messages, tools=None, **kwargs):
        resp = self.responses[self.index % len(self.responses)]
        self.index += 1
        self.history.append({"messages": messages, "tools": tools})
        return resp


class DummyTool(Tool):
    name = "dummy_action"
    description = "A dummy tool that returns what you give it"
    parameters = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }

    def execute(self, value: str) -> ToolResult:
        return ToolResult(success=True, data=f"processed:{value}")


class SimpleReActAgent(ReActAgent):
    def __init__(self, llm_client=None, config=None):
        super().__init__(name="simple_test_agent", config=config, llm_client=llm_client)

    @property
    def system_prompt(self) -> str:
        return "You are a simple test agent. Use tools when needed and finish with FINAL_ANSWER:"


class TestReActAgent:
    def test_thought_action_observation_finish(self):
        """Test basic 3-step ReAct loop"""
        mock_llm = MockLLMClient(
            responses=[
                # Step 1: think + tool call
                LLMResponse(
                    content="I need to process the input first.",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "dummy_action",
                                "arguments": '{"value": "hello"}',
                            },
                        }
                    ],
                    usage=None,
                ),
                # Step 2: final answer
                LLMResponse(
                    content="The result is good. FINAL_ANSWER: Done!",
                    tool_calls=None,
                    usage=None,
                ),
            ]
        )

        agent = SimpleReActAgent(llm_client=mock_llm)
        agent.register_tool(DummyTool())
        result = agent.execute({"task": "Say hello"})

        assert result.is_success
        assert result.data["final_answer"] == "Done!"
        log = agent.get_reasoning_log()
        assert len(log.steps) == 2
        assert log.steps[0].thought == "I need to process the input first."
        assert log.steps[0].observation == "processed:hello"
        assert log.finish_reason == "final_answer"

    def test_max_steps_reached(self):
        """Test that agent stops at max_steps"""
        mock_llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content="Still thinking...",
                    tool_calls=[
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": "dummy_action",
                                "arguments": '{"value": "x"}',
                            },
                        }
                    ],
                    usage=None,
                )
                for i in range(5)
            ]
        )

        agent = SimpleReActAgent(llm_client=mock_llm, config={"max_steps": 3})
        agent.register_tool(DummyTool())
        result = agent.execute({"task": "Never ending task"})

        assert not result.is_success
        assert result.metrics["finish_reason"] == "max_steps_reached"
        assert len(result.metrics["reasoning_log"]["steps"]) == 3

    def test_no_tools_final_answer_directly(self):
        """Agent can finish without any tools"""
        mock_llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content="This is simple. FINAL_ANSWER: 42",
                    tool_calls=None,
                    usage=None,
                ),
            ]
        )

        agent = SimpleReActAgent(llm_client=mock_llm)
        # no tools registered
        result = agent.execute({"task": "What is the answer?"})

        assert result.is_success
        assert result.data["final_answer"] == "42"

    def test_llm_failure(self):
        """Graceful handling when LLM fails"""

        class BadLLM:
            def chat(self, *args, **kwargs):
                raise RuntimeError("Network down")

        agent = SimpleReActAgent(llm_client=BadLLM())
        result = agent.execute({"task": "Do something"})

        assert not result.is_success
        assert result.metrics["finish_reason"] == "llm_error"

    def test_decorator_tool_in_react(self):
        """Use @tool decorated function inside ReAct"""

        @tool(name="multiply", description="Multiply two numbers")
        def multiply(a: int, b: int) -> int:
            return a * b

        mock_llm = MockLLMClient(
            responses=[
                LLMResponse(
                    content="I will multiply 3 and 4.",
                    tool_calls=[
                        {
                            "id": "call_m1",
                            "type": "function",
                            "function": {
                                "name": "multiply",
                                "arguments": '{"a": 3, "b": 4}',
                            },
                        }
                    ],
                    usage=None,
                ),
                LLMResponse(
                    content="FINAL_ANSWER: 12",
                    tool_calls=None,
                    usage=None,
                ),
            ]
        )

        agent = SimpleReActAgent(llm_client=mock_llm)
        agent.register_tool(multiply())
        result = agent.execute({"task": "Calculate 3*4"})

        assert result.is_success
        log = agent.get_reasoning_log()
        assert log.steps[0].observation == "12"
