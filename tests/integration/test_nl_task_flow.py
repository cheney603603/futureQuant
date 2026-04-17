"""
End-to-end test for natural language task flow

Uses fully mocked LLM and tools to verify the full pipeline:
BlackboardAgent -> ExecutionPlan -> BlackboardController -> Agent execution
"""

import json

import pytest

from futureQuant.agent.blackboard.blackboard import Blackboard
from futureQuant.agent.blackboard.blackboard_agent import BlackboardAgent
from futureQuant.agent.blackboard.blackboard_controller import BlackboardController
from futureQuant.agent.blackboard.knowledge_source import KnowledgeSourceAdapter
from futureQuant.agent.orchestrator import NaturalLanguageTaskRunner
from futureQuant.core.llm_client import LLMResponse


class MockLLMClient:
    def __init__(self, responses):
        self.responses = responses if isinstance(responses, list) else [responses]
        self.index = 0

    def chat(self, messages, tools=None, **kwargs):
        resp = self.responses[self.index % len(self.responses)]
        self.index += 1
        return resp


class DummyAgent:
    """A simple agent that writes fixed output to blackboard"""
    def __init__(self, name, output_key, output_value):
        self.name = name
        self.output_key = output_key
        self.output_value = output_value

    def run(self, context):
        from futureQuant.agent.base import AgentResult, AgentStatus
        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data=self.output_value,
            metrics={},
        )


def test_blackboard_agent_generates_plan():
    plan_json = json.dumps(
        {
            "goal": "测试任务",
            "steps": [
                {
                    "step_id": 1,
                    "agent": "data_collector",
                    "task": "获取数据",
                    "inputs": {},
                    "outputs": ["price_data"],
                    "depends_on": [],
                }
            ],
        }
    )
    mock_llm = MockLLMClient(
        [
            LLMResponse(
                content=plan_json + " FINAL_ANSWER: 计划已生成",
                tool_calls=None,
                usage=None,
            ),
        ]
    )
    agent = BlackboardAgent(llm_client=mock_llm)
    result = agent.execute({"user_query": "测试任务"})

    assert result.is_success
    assert "execution_plan" in result.data
    assert result.data["execution_plan"]["goal"] == "测试任务"


def test_controller_executes_plan():
    bb = Blackboard()
    bb.write(
        "execution_plan",
        {
            "goal": "测试计划执行",
            "steps": [
                {
                    "step_id": 1,
                    "agent": "data_collector",
                    "task": "获取测试数据",
                    "inputs": {},
                    "outputs": ["price_data"],
                    "depends_on": [],
                },
                {
                    "step_id": 2,
                    "agent": "factor_mining",
                    "task": "挖掘因子",
                    "inputs": {"price_data": "price_data"},
                    "outputs": ["factors"],
                    "depends_on": [1],
                },
            ],
        },
        agent="test",
    )

    controller = BlackboardController(blackboard=bb)
    dummy_data = DummyAgent("data_collector", "price_data", {"close": [1, 2, 3]})
    dummy_miner = DummyAgent("factor_mining", "factors", [{"name": "mom", "ic": 0.05}])

    controller.register_agent(
        dummy_data, name="data_collector", input_keys=[], output_key="price_data"
    )
    controller.register_agent(
        dummy_miner, name="factor_mining", input_keys=["price_data"], output_key="factors"
    )

    result = controller.execute()
    assert result.success
    assert result.n_executed == 2
    assert bb.read("factors") is not None


def test_natural_language_runner_end_to_end():
    plan_json = json.dumps(
        {
            "goal": "获取螺纹钢数据",
            "steps": [
                {
                    "step_id": 1,
                    "agent": "data_collector",
                    "task": "获取数据",
                    "inputs": {},
                    "outputs": ["price_data"],
                    "depends_on": [],
                }
            ],
        }
    )
    mock_llm = MockLLMClient(
        [
            LLMResponse(
                content=plan_json + " FINAL_ANSWER: 计划已生成",
                tool_calls=None,
                usage=None,
            ),
        ]
    )

    bb = Blackboard()
    runner = NaturalLanguageTaskRunner(blackboard=bb, llm_client=mock_llm)
    # 替换已注册的 data_collector 为一个 dummy，避免真实网络调用
    dummy = DummyAgent("data_collector", "price_data", {"symbol": "RB", "close": [3500, 3510]})
    for name in list(runner._controller._sources.keys()):
        runner._controller.unregister(name)
    runner._controller.register_agent(
        dummy, name="data_collector", input_keys=[], output_key="price_data"
    )

    result = runner.run("帮我获取螺纹钢数据")
    assert result["status"] == "success"
    assert result["plan"]["goal"] == "获取螺纹钢数据"
    snapshot = result["blackboard_snapshot"]
    assert "price_data" in snapshot.get("data", {})
