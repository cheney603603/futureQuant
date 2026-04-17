"""
Tests for BlackboardAgent
"""

import json

import pytest

from futureQuant.agent.blackboard.blackboard_agent import BlackboardAgent, ExecutionPlan
from futureQuant.agent.tools import BlackboardWriteTool
from futureQuant.core.llm_client import LLMResponse


class MockLLMClient:
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.index = 0
        self.history = []

    def chat(self, messages, tools=None, **kwargs):
        resp = self.responses[self.index % len(self.responses)]
        self.index += 1
        self.history.append({"messages": messages, "tools": tools})
        return resp


def build_tool_call_response(tool_calls, content=""):
    return LLMResponse(
        content=content,
        tool_calls=tool_calls,
        usage=None,
    )


class TestExecutionPlan:
    def test_from_dict_and_serialize(self):
        data = {
            "goal": "挖掘螺纹钢因子",
            "steps": [
                {
                    "step_id": 1,
                    "agent": "data_collector",
                    "task": "获取数据",
                    "inputs": {},
                    "outputs": ["RB_daily"],
                    "depends_on": [],
                },
                {
                    "step_id": 2,
                    "agent": "factor_mining",
                    "task": "挖掘因子",
                    "inputs": {"price_data": "RB_daily"},
                    "outputs": ["factors"],
                    "depends_on": [1],
                },
            ],
        }
        plan = ExecutionPlan.from_dict(data)
        assert plan.goal == "挖掘螺纹钢因子"
        assert len(plan.steps) == 2
        assert plan.get_agent_sequence() == ["data_collector", "factor_mining"]
        assert plan.get_step(2)["agent"] == "factor_mining"


class TestBlackboardAgent:
    def test_plan_written_to_blackboard(self):
        plan_json = json.dumps(
            {
                "goal": "Test goal",
                "steps": [
                    {
                        "step_id": 1,
                        "agent": "data_collector",
                        "task": "Get RB data",
                        "inputs": {},
                        "outputs": ["RB_data"],
                        "depends_on": [],
                    }
                ],
            },
            ensure_ascii=False,
        )

        # LLM first outputs the JSON as thought, then calls blackboard_write
        mock_llm = MockLLMClient(
            [
                build_tool_call_response(
                    tool_calls=[
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "blackboard_write",
                                "arguments": json.dumps(
                                    {
                                        "key": "execution_plan",
                                        "value": json.loads(plan_json),
                                        "agent_name": "blackboard_agent",
                                    }
                                ),
                            },
                        }
                    ],
                    content=plan_json,
                ),
                LLMResponse(
                    content="FINAL_ANSWER: 计划已成功生成并写入黑板。",
                    tool_calls=None,
                    usage=None,
                ),
            ]
        )

        agent = BlackboardAgent(llm_client=mock_llm)
        result = agent.execute({"user_query": "帮我获取螺纹钢数据"})

        assert result.is_success
        assert "execution_plan" in result.data
        assert result.data["execution_plan"]["goal"] == "Test goal"
        assert len(result.data["execution_plan"]["steps"]) == 1

        # Verify blackboard content
        bb_entry = agent._bb.read("execution_plan")
        assert bb_entry is not None
        assert bb_entry["goal"] == "Test goal"

    def test_empty_query_fails(self):
        agent = BlackboardAgent()
        result = agent.execute({"user_query": ""})
        assert not result.is_success
        assert "user_query is empty" in result.errors[0]

    def test_extract_plan_from_log(self):
        """If LLM outputs JSON directly without tool call, agent should extract it from log."""
        plan_text = (
            '{"goal": "分析基本面", "steps": [{"step_id": 1, "agent": "fundamental_analysis", '
            '"task": "分析螺纹钢", "inputs": {}, "outputs": ["sentiment"], "depends_on": []}]}'
        )

        mock_llm = MockLLMClient(
            [
                LLMResponse(
                    content=plan_text + " FINAL_ANSWER: 计划已生成",
                    tool_calls=None,
                    usage=None,
                ),
            ]
        )

        agent = BlackboardAgent(llm_client=mock_llm)
        result = agent.execute({"user_query": "分析螺纹钢基本面"})

        assert result.is_success
        plan = result.data.get("execution_plan")
        assert plan is not None
        assert plan["steps"][0]["agent"] == "fundamental_analysis"
