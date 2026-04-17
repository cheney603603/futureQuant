"""
BlackboardAgent - 中央黑板规划 Agent

职责：
1. 接收用户自然语言需求
2. 分析意图并生成结构化 ExecutionPlan
3. 根据各 Agent 的功能描述匹配并分配任务
4. 将 ExecutionPlan 写入 Blackboard

ExecutionPlan Schema:
{
  "goal": "用户原始需求摘要",
  "steps": [
    {
      "step_id": 1,
      "agent": "data_collector",
      "task": "获取螺纹钢最近一周日线数据",
      "inputs": {},
      "outputs": ["RB_daily"],
      "depends_on": []
    }
  ]
}
"""

import json
from typing import Any, Dict, List, Optional

from ..base import AgentStatus
from ..react_base import ReActAgent
from ..tools import BlackboardWriteTool
from .blackboard import Blackboard
from ...core.config import get_config
from ...core.logger import get_logger

logger = get_logger("agent.blackboard_agent")


class ExecutionPlan:
    """执行计划数据类"""

    def __init__(self, goal: str, steps: Optional[List[Dict[str, Any]]] = None):
        self.goal = goal
        self.steps = steps or []

    def to_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal, "steps": self.steps}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionPlan":
        return cls(goal=data.get("goal", ""), steps=data.get("steps", []))

    def get_agent_sequence(self) -> List[str]:
        """获取参与 Agent 列表（按 step_id 顺序）"""
        return [step["agent"] for step in self.steps]

    def get_step(self, step_id: int) -> Optional[Dict[str, Any]]:
        for s in self.steps:
            if s.get("step_id") == step_id:
                return s
        return None


# 预定义的 Agent 能力描述（供 LLM 参考）
AGENT_CATALOG = {
    "data_collector": {
        "name": "data_collector",
        "description": (
            "数据收集 Agent。负责查询本地数据库、可靠链路库，或在无可用链路时"
            "联网搜索并生成数据获取代码。支持期货日线、分钟线、库存、仓单、基差等数据类型。"
        ),
        "inputs": ["品种代码", "数据类型", "时间范围"],
        "outputs": ["price_data", "inventory_data", "basis_data"],
    },
    "factor_mining": {
        "name": "factor_mining",
        "description": (
            "因子挖掘 Agent。基于价格数据和基本面数据，使用 Alpha101、技术指标、"
            "交叉因子等方法挖掘有效因子，计算 IC/ICIR，并将通过筛选的因子存入因子数据库。"
        ),
        "inputs": ["price_data", "basis_data(可选)"],
        "outputs": ["factor_data", "top_factors"],
    },
    "fundamental_analysis": {
        "name": "fundamental_analysis",
        "description": (
            "基本面分析 Agent。基于联网搜索最近一周的新闻、库存、基差、政策等信息，"
            "结合价格数据计算库存周期、支撑压力位，并输出多空情绪评分（-5~+5）。"
        ),
        "inputs": ["target", "price_data(可选)"],
        "outputs": ["sentiment_score", "inventory_cycle", "support_pressure"],
    },
    "backtest": {
        "name": "backtest",
        "description": "回测 Agent。基于因子信号运行向量化的分层回测或组合回测，输出绩效指标。",
        "inputs": ["factor_data", "price_data"],
        "outputs": ["backtest_result"],
    },
}


class BlackboardAgent(ReActAgent):
    """
    中央黑板 Agent

    接收自然语言需求 -> 生成 ExecutionPlan -> 写入 Blackboard
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        blackboard: Optional[Blackboard] = None,
        llm_client=None,
    ):
        super().__init__(name="blackboard_agent", config=config, llm_client=llm_client)
        self._bb = blackboard or Blackboard()
        self.register_tool(BlackboardWriteTool(blackboard=self._bb))

    @property
    def system_prompt(self) -> str:
        varieties = ", ".join(get_config().varieties)
        catalog_json = json.dumps(AGENT_CATALOG, ensure_ascii=False, indent=2)

        return (
            "你是 futureQuant 系统的中央黑板规划 Agent（BlackboardAgent）。\n"
            "你的核心职责是：接收用户的自然语言需求，分析意图，生成结构化的 ExecutionPlan，"
            "并根据系统中各 Agent 的能力描述分配任务。\n\n"
            "系统中可用的 Agent 及其能力如下（JSON 格式）：\n"
            f"{catalog_json}\n\n"
            "规则：\n"
            "1. 只允许使用 blackboard_write 工具将生成的 ExecutionPlan 写入 Blackboard，key 固定为 'execution_plan'。\n"
            "2. ExecutionPlan 必须严格包含 'goal' 和 'steps' 两个字段。\n"
            "3. 每个 step 必须包含：step_id（从1开始）、agent（从上述 catalog 中选取）、"
            "task（该步骤的具体任务描述）、inputs（输入数据键名映射）、outputs（输出数据键名列表）、"
            "depends_on（依赖的 step_id 列表，可为空）。\n"
            "4. 如果用户需求涉及多个品种或多个分析维度，请拆分为多个 steps。\n"
            "5. 如果用户需求无法被现有 Agent 满足，请在 goal 中说明，并返回空的 steps 列表。\n"
            "6. 生成计划后，必须使用 blackboard_write 工具写入。\n"
            "7. 品种代码必须标准化为大写期货代码。系统支持的品种包括：\n"
            f"{varieties}\n\n"
            "示例：用户说'帮我分析螺纹钢最近一周的基本面并挖掘有效因子'\n"
            "你应该生成包含 data_collector（获取数据）、fundamental_analysis（基本面）、"
            "factor_mining（因子挖掘）三个 step 的计划。"
        )

    def execute(self, context: Dict[str, Any]) -> Any:
        """
        执行规划任务

        context 支持 keys:
        - user_query (str): 用户自然语言需求（必须）
        """
        user_query = context.get("user_query", "")
        if not user_query:
            from ..base import AgentResult
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=["user_query is empty"],
            )

        # 将 user_query 包装为 task
        react_context = {"task": user_query}
        result = super().execute(react_context)

        # 尝试从 reasoning_log 中提取 LLM 输出的 execution_plan JSON
        plan = self._extract_plan_from_log()
        if plan is None:
            # 尝试从 blackboard 读取（如果 LLM 成功调用了 blackboard_write）
            plan_entry = self._bb.read("execution_plan", default=None)
            if plan_entry and isinstance(plan_entry, dict):
                plan = ExecutionPlan.from_dict(plan_entry)

        if plan is None:
            # 若仍未找到，说明 LLM 没有正确输出计划，标记失败
            result.status = AgentStatus.FAILED
            if not result.errors:
                result.errors = ["Failed to generate or parse ExecutionPlan"]
            return result

        # 将解析后的计划也写回 blackboard（确保结构正确）
        self._bb.write(
            key="execution_plan",
            value=plan.to_dict(),
            agent=self.name,
            tags={"plan", "latest"},
        )

        # 更新 result data
        result.data = {
            "final_answer": result.data.get("final_answer", ""),
            "execution_plan": plan.to_dict(),
            "task_id": result.data.get("task_id", ""),
        }
        result.metrics["execution_plan"] = plan.to_dict()

        return result

    def _extract_plan_from_log(self) -> Optional[ExecutionPlan]:
        """从 reasoning_log 的 assistant messages 中提取 ExecutionPlan JSON"""
        import re

        log = self.get_reasoning_log()
        if not log:
            return None

        for step in reversed(log.steps):
            thought = step.thought or ""

            # 策略1: 尝试直接解析整个 thought（如果是纯JSON）
            try:
                data = json.loads(thought.strip())
                if "steps" in data:
                    return ExecutionPlan.from_dict(data)
            except json.JSONDecodeError:
                pass

            # 策略2: 提取 ```json ... ``` 代码块
            match = re.search(r"```json\s*(.*?)```", thought, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1).strip())
                    if "steps" in data:
                        return ExecutionPlan.from_dict(data)
                except json.JSONDecodeError:
                    pass

            # 策略3: 从混合文本中提取第一个顶层 JSON 对象 {}
            json_match = re.search(r"(\{.*\})", thought, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1).strip())
                    if "steps" in data:
                        return ExecutionPlan.from_dict(data)
                except json.JSONDecodeError:
                    pass

        return None
