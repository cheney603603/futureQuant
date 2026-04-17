"""
ReAct Agent 基类

实现 Thought → Action → Observation 循环：
1. Thought: LLM 根据当前状态进行推理
2. Action: 调用 ToolRegistry 中的工具
3. Observation: 将工具结果反馈给 LLM
4. 循环直到任务完成或达到最大步数

兼容现有 BaseAgent，可直接注册到 BlackboardController。

使用示例：
    >>> class MyAgent(ReActAgent):
    ...     def __init__(self):
    ...         super().__init__(name="my_agent")
    ...         self.register_tool(WebSearchTool())
    ...
    ...     @property
    ...     def system_prompt(self) -> str:
    ...         return "你是一个测试Agent。"
    ...
    >>> result = MyAgent().run({"task": "搜索今天天气"})
"""

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import AgentResult, AgentStatus, BaseAgent
from .tools import Tool, ToolRegistry
from ..core.llm_client import LLMClient, LLMResponse
from ..core.logger import get_logger

logger = get_logger("agent.react_base")


@dataclass
class ReActStep:
    """ReAct 单步记录"""

    step_num: int
    thought: str = ""
    action: Optional[Dict[str, Any]] = None
    observation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReActLog:
    """完整 ReAct 推理日志"""

    agent_name: str
    task_id: str
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    finish_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "task_id": self.task_id,
            "steps": [
                {
                    "step_num": s.step_num,
                    "thought": s.thought,
                    "action": s.action,
                    "observation": s.observation,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "finish_reason": self.finish_reason,
        }


class ReActAgent(BaseAgent):
    """
    ReAct Agent 抽象基类

    子类需要实现：
    - system_prompt: 返回系统提示词字符串
    - 可选 override `parse_final_answer` 自定义终止逻辑
    """

    DEFAULT_MAX_STEPS = 15
    DEFAULT_FINISH_MARKER = "FINAL_ANSWER:"

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        super().__init__(name=name, config=config)
        self._llm = llm_client or LLMClient()
        self._tools = ToolRegistry()
        self._max_steps = self.config.get("max_steps", self.DEFAULT_MAX_STEPS)
        self._finish_marker = self.config.get(
            "finish_marker", self.DEFAULT_FINISH_MARKER
        )
        self._reasoning_log: Optional[ReActLog] = None

        # 注册默认的 finish 思考辅助
        self._register_default_tools()

    def _register_default_tools(self):
        """注册默认辅助工具（子类可覆盖）"""
        pass

    @property
    def system_prompt(self) -> str:
        """
        系统提示词（子类必须实现）

        提示词中应包含：
        - Agent 身份与任务
        - 可用工具列表（会自动追加）
        - ReAct 格式说明
        - 终止条件说明（使用 FINAL_ANSWER:）
        """
        raise NotImplementedError("Subclasses must implement system_prompt")

    def register_tool(self, tool: Tool) -> "ReActAgent":
        """注册单个工具"""
        self._tools.register(tool)
        return self

    def register_tools(self, *tools: Tool) -> "ReActAgent":
        """批量注册工具"""
        self._tools.register_all(*tools)
        return self

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """获取 OpenAI 格式的工具 schema"""
        return self._tools.to_openai_schema()

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行 ReAct 循环

        Args:
            context: 必须包含 task 或用户通过子类注入的输入数据

        Returns:
            AgentResult: 包含最终答案、reasoning_log、metrics
        """
        task_id = str(uuid.uuid4())[:8]
        self._reasoning_log = ReActLog(agent_name=self.name, task_id=task_id)

        # 构建初始 messages
        messages = self._build_messages(context)

        final_answer = ""
        finish_reason = ""

        for step_num in range(1, self._max_steps + 1):
            logger.debug(f"[{self.name}] ReAct step {step_num}/{self._max_steps}")

            # 1. Thought + Action (LLM Call)
            llm_resp = self._call_llm(messages)
            if llm_resp is None:
                finish_reason = "llm_error"
                final_answer = "LLM 调用失败，无法继续推理。"
                break

            thought = (llm_resp.content or "").strip()

            # 检查是否直接给出最终答案
            parsed_answer = self._parse_final_answer(thought)
            if parsed_answer is not None:
                final_answer = parsed_answer
                finish_reason = "final_answer"
                self._reasoning_log.steps.append(
                    ReActStep(
                        step_num=step_num,
                        thought=thought,
                    )
                )
                break

            # 2. 执行 Action（Tool Calls）
            observations: List[str] = []
            actions: List[Dict[str, Any]] = []

            if llm_resp.tool_calls:
                for tc in llm_resp.tool_calls:
                    actions.append(tc)
                    tool_result = self._tools.execute_tool_call(tc)
                    obs_text = tool_result.to_text()
                    observations.append(obs_text)
                    logger.debug(
                        f"[{self.name}] Tool {tc.get('function', {}).get('name')} -> "
                        f"success={tool_result.success}"
                    )
            else:
                # LLM 没有调用工具，视为单纯思考
                observations.append("[系统提示] 请继续思考并决定下一步行动。如果你已完成任务，请输出 FINAL_ANSWER: <结论>")

            # 记录步骤
            self._reasoning_log.steps.append(
                ReActStep(
                    step_num=step_num,
                    thought=thought,
                    action=actions[0] if actions else None,
                    observation="\n".join(observations),
                    metadata={
                        "usage": {
                            "prompt_tokens": llm_resp.usage.prompt_tokens if llm_resp.usage else 0,
                            "completion_tokens": llm_resp.usage.completion_tokens if llm_resp.usage else 0,
                        }
                    },
                )
            )

            # 3. 将 Observation 加入对话历史
            messages.append(
                {
                    "role": "assistant",
                    "content": thought,
                    "tool_calls": llm_resp.tool_calls,
                }
            )
            if llm_resp.tool_calls:
                for tc, obs in zip(llm_resp.tool_calls, observations):
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.get("id"),
                            "name": tc.get("function", {}).get("name"),
                            "content": obs,
                        }
                    )

            # 4. 检查是否达到最大步数
            if step_num == self._max_steps:
                finish_reason = "max_steps_reached"
                final_answer = thought or "已达到最大推理步数，任务未完成。"

        if not finish_reason:
            finish_reason = "completed"

        self._reasoning_log.final_answer = final_answer
        self._reasoning_log.finish_reason = finish_reason

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS if finish_reason == "final_answer" else AgentStatus.FAILED,
            data={"final_answer": final_answer, "task_id": task_id},
            metrics={
                "reasoning_log": self._reasoning_log.to_dict(),
                "steps": len(self._reasoning_log.steps),
                "finish_reason": finish_reason,
            },
            logs=[f"ReAct finished: {finish_reason} after {len(self._reasoning_log.steps)} steps"],
        )

    def _build_messages(self, context: Dict[str, Any]) -> List[Dict[str, str]]:
        """构建初始 LLM 消息列表"""
        sys_prompt = self.system_prompt

        # 自动追加工具说明
        if self._tools.list_names():
            tool_names = ", ".join(self._tools.list_names())
            sys_prompt += (
                f"\n\n你可以使用以下工具（通过 function calling 调用）：{tool_names}。"
                f"\n当你完成任务时，必须在回复末尾输出 {self._finish_marker} <你的最终结论>"
            )

        task = context.get("task", "")
        user_content = task or json.dumps(context, ensure_ascii=False, default=str)

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

    def _call_llm(self, messages: List[Dict[str, Any]]) -> Optional[LLMResponse]:
        """调用 LLM"""
        try:
            tools_schema = self.get_tools_schema() if self._tools.list_names() else None
            resp = self._llm.chat(messages, tools=tools_schema)
            return resp
        except Exception as exc:
            logger.error(f"[{self.name}] LLM call failed: {exc}")
            return None

    def _parse_final_answer(self, text: str) -> Optional[str]:
        """
        解析最终答案

        如果文本包含 FINISH_MARKER，则提取之后的内容作为最终答案。
        """
        marker = self._finish_marker
        if marker in text:
            idx = text.find(marker)
            return text[idx + len(marker) :].strip()
        return None

    def get_reasoning_log(self) -> Optional[ReActLog]:
        """获取最近一次 ReAct 推理日志"""
        return self._reasoning_log
