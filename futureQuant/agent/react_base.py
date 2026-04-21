"""
增强型 ReAct Agent 基类

在原有 ReActAgent 基础上，引入：
1. 失败检测与恢复机制（_LoopCoordinator 模式）
2. 连续失败计数器（consecutive_empty / consecutive_errors）
3. 恢复提示注入（_inject_recovery）
4. 双终止条件（FINAL_ANSWER + pipeline_ready）
5. 完整执行指标（ReActMetrics）

参考 quant_react_interview-main/agent/react_loop.py 的设计，
但保留 futureQuant 的工具注册机制。

使用示例：
    class MyFactorResearchAgent(EnhancedReActAgent):
        @property
        def system_prompt(self) -> str:
            return "你是因子研究助手..."

        def _register_research_tools(self):
            from .tools.research_tools import get_tool_specs, execute_tool, bind_builder
            from ...engine.nodes import FactorPipelineBuilder
            builder = FactorPipelineBuilder()
            bind_builder(builder)
            # 注册自定义执行器...
            return builder

        def _get_tool_specs(self):
            from .tools.research_tools import get_tool_specs
            return get_tool_specs()
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base import AgentResult, AgentStatus, BaseAgent
from .tools import ToolRegistry
from ..core.llm_client import LLMClient, LLMResponse
from ..core.logger import get_logger

if TYPE_CHECKING:
    from ..engine.nodes.pipeline_builder import FactorPipelineBuilder

logger = get_logger("agent.react_base")


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class ReActStep:
    """单步推理记录。"""
    step_num: int
    thought: str = ""
    action: Optional[Dict[str, Any]] = None
    observation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReActMetrics:
    """ReAct 循环执行指标。"""
    iterations: int = 0
    tool_calls: int = 0
    successful_tool_calls: int = 0
    failed_tool_calls: int = 0
    consecutive_empty: int = 0  # 连续空调用次数
    consecutive_errors: int = 0  # 连续错误次数
    total_errors: int = 0
    pipeline_ready: bool = False

    def reset_empty(self):
        self.consecutive_empty = 0

    def reset_errors(self):
        self.consecutive_errors = 0

    def record_empty(self):
        self.consecutive_empty += 1

    def record_error(self):
        self.consecutive_errors += 1
        self.total_errors += 1

    def record_tool_call(self, success: bool):
        self.tool_calls += 1
        if success:
            self.successful_tool_calls += 1
        else:
            self.failed_tool_calls += 1


@dataclass
class ReActLog:
    """完整 ReAct 推理日志。"""
    agent_name: str
    task_id: str
    steps: List[ReActStep] = field(default_factory=list)
    final_answer: str = ""
    finish_reason: str = ""
    metrics: Optional[ReActMetrics] = None

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
            "metrics": self.metrics.__dict__ if self.metrics else {},
        }


# =============================================================================
# 恢复提示模板
# =============================================================================

RECOVERY_PROMPTS = {
    "empty_response": (
        "Your last response had no tool calls. "
        "You must use tools to build the pipeline. "
        "Start by inspecting the catalog with get_catalog(), then add steps. "
        "If you are unsure where to start, use get_catalog() to see available steps."
    ),
    "tool_error": (
        "A tool call failed. Analyze the error message and repair the affected step. "
        "Use get_details(kind) to understand the correct config format, "
        "then use update_step to fix the config. "
        "Do NOT remove the step - fix it instead."
    ),
    "stuck": (
        "You seem stuck (too many consecutive turns without progress). "
        "Let's reset: call get_catalog() to see available step kinds, "
        "then systematically build the pipeline step by step. "
        "Start with trigger.manual, then data.price_bars, then factor steps."
    ),
    "no_progress": (
        "Multiple turns without progress. Focus on the goal: build a complete pipeline. "
        "If you are unsure about the config format, call get_details(kind) first. "
        "If you have built all steps, call get_pipeline() to export the result."
    ),
    "catalog_first": (
        "Before adding any steps, you should call get_catalog() to understand "
        "what step kinds are available. Call it now."
    ),
}


# =============================================================================
# 增强型 ReAct Agent
# =============================================================================

class EnhancedReActAgent(BaseAgent):
    """
    增强型 ReAct Agent。

    在 ReActAgent 基础上增加了：
    - 失败检测与恢复机制
    - 连续失败计数器
    - 恢复提示注入
    - 流水线就绪标志

    子类需要实现：
    - system_prompt: 系统提示词
    - _register_tools(): 注册工具
    - _get_tool_specs(): 返回工具 schema

    可选覆盖：
    - _get_recovery_prompts(): 自定义恢复提示
    - _is_pipeline_ready(): 判断流水线是否完成
    - _parse_final_answer(): 自定义终止条件解析
    """

    DEFAULT_MAX_ITERS = 15
    DEFAULT_MAX_EMPTY_TURNS = 3
    DEFAULT_MAX_ERROR_TURNS = 3
    DEFAULT_FINISH_MARKER = "FINAL_ANSWER:"

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        super().__init__(name=name, config=config)
        self._llm = llm_client or LLMClient()

        # 循环控制参数
        # 兼容旧配置键 max_steps
        self._max_iters = self.config.get("max_iters", self.config.get("max_steps", self.DEFAULT_MAX_ITERS))
        self._max_empty_turns = self.config.get("max_empty_turns", self.DEFAULT_MAX_EMPTY_TURNS)
        self._max_error_turns = self.config.get("max_error_turns", self.DEFAULT_MAX_ERROR_TURNS)
        self._finish_marker = self.config.get("finish_marker", self.DEFAULT_FINISH_MARKER)

        # 工具注册
        self._tools = ToolRegistry()
        self._builder: Optional[Any] = None  # PipelineBuilder
        self._tool_specs: Optional[List[Dict[str, Any]]] = None

        # 日志
        self._reasoning_log: Optional[ReActLog] = None
        self._metrics: Optional[ReActMetrics] = None

        # 初始化（子类在 execute() 之前可能需要设置 builder）
        self._initialized = False

        # 公开属性（供测试注入 mock）
        self._llm: LLMClient = self._llm  # 已在上面定义，此处仅做类型标注
        self._tool_registry: ToolRegistry = self._tools

    # -------------------------------------------------------------------------
    # 子类必须实现的属性/方法
    # -------------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        """系统提示词（子类必须实现）。"""
        raise NotImplementedError("Subclasses must implement system_prompt")

    def _get_tool_specs(self) -> List[Dict[str, Any]]:
        """
        返回工具 schema（子类可覆盖）。

        默认返回 ToolRegistry 中的注册工具。
        因子研究场景可返回 research_tools.get_tool_specs()。
        """
        if self._tools.list_names():
            return self._tools.to_openai_schema()
        return []

    def _register_tools(self) -> None:
        """
        注册工具（子类可覆盖）。

        在 execute() 首次调用时自动触发。
        默认实现为空，子类可以覆盖或直接使用 register_tool()。
        """
        pass

    # -------------------------------------------------------------------------
    # 公共工具注册方法（保持向后兼容）
    # -------------------------------------------------------------------------

    def register_tool(self, tool: Any) -> "EnhancedReActAgent":
        """注册单个工具（向后兼容别名）。"""
        self._tools.register(tool)
        return self

    def register_tools(self, *tools: Any) -> "EnhancedReActAgent":
        """批量注册工具（向后兼容别名）。"""
        self._tools.register_all(*tools)
        return self

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """获取工具 schema（向后兼容别名）。"""
        return self._tools.to_openai_schema()

    def get_reasoning_log(self) -> Optional[ReActLog]:
        """获取推理日志（公共接口，供测试和调试使用）。"""
        return self._reasoning_log

    # -------------------------------------------------------------------------
    # 子类可覆盖的方法
    # -------------------------------------------------------------------------

    def _get_recovery_prompts(self) -> Dict[str, str]:
        """返回恢复提示字典。"""
        return RECOVERY_PROMPTS

    def _is_pipeline_ready(self) -> bool:
        """
        判断流水线是否已完成。

        默认返回 False。
        子类可以覆盖此方法，在特定条件下（如 get_pipeline 成功）返回 True。
        """
        return False

    def _parse_final_answer(self, text: str) -> Optional[str]:
        """
        解析最终答案。

        默认检查 text 是否包含 FINISH_MARKER。
        子类可以覆盖此方法实现自定义终止条件。
        """
        if self._finish_marker in text:
            idx = text.find(self._finish_marker)
            return text[idx + len(self._finish_marker) :].strip()
        return None

    def _build_messages(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """构建初始 LLM 消息列表。"""
        task = context.get("task", "")
        user_content = task or json.dumps(context, ensure_ascii=False, default=str)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        # 追加工具说明
        tool_specs = self._get_tool_specs()
        if tool_specs:
            tool_names = ", ".join(
                s.get("function", {}).get("name", "unknown") for s in tool_specs
            )
            messages[1]["content"] = (
                f"{user_content}\n\n"
                f"Available tools: {tool_names}. "
                f"When you complete the task, output {self._finish_marker} <your conclusion>"
            )

        return messages

    # -------------------------------------------------------------------------
    # 执行入口
    # -------------------------------------------------------------------------

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行增强型 ReAct 循环。

        Args:
            context: 执行上下文，必须包含 task

        Returns:
            AgentResult
        """
        start_time = time.time()
        task_id = str(uuid.uuid4())[:8]

        # 延迟初始化（允许子类在 execute 之前设置 builder）
        if not self._initialized:
            self._register_tools()
            self._tool_specs = self._get_tool_specs()
            self._initialized = True

        # 初始化日志和指标
        self._reasoning_log = ReActLog(
            agent_name=self.name,
            task_id=task_id,
        )
        self._metrics = ReActMetrics()

        # 构建初始消息
        messages = self._build_messages(context)

        # 执行循环
        coordinator = _LoopCoordinator(
            agent=self,
            messages=messages,
            max_iters=self._max_iters,
            max_empty_turns=self._max_empty_turns,
            max_error_turns=self._max_error_turns,
        )
        loop_result = coordinator.run()

        elapsed = time.time() - start_time
        self._reasoning_log.steps = loop_result["steps"]
        self._reasoning_log.final_answer = loop_result["final_answer"]
        self._reasoning_log.finish_reason = loop_result["finish_reason"]
        self._reasoning_log.metrics = self._metrics

        # 构建结果
        is_success = loop_result["finish_reason"] in ("final_answer", "success", "completed")

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS if is_success else AgentStatus.FAILED,
            data={
                "final_answer": loop_result["final_answer"],
                "pipeline": self._builder.get_pipeline() if self._builder else None,
            },
            metrics={
                "reasoning_log": self._reasoning_log.to_dict(),
                "loop_metrics": self._metrics.__dict__,
                "steps": len(loop_result["steps"]),
                "finish_reason": loop_result["finish_reason"],
            },
            logs=[
                f"ReAct finished: {loop_result['finish_reason']} "
                f"after {len(loop_result['steps'])} steps, "
                f"{self._metrics.tool_calls} tool calls, "
                f"{self._metrics.total_errors} errors"
            ],
            elapsed_seconds=elapsed,
        )


# =============================================================================
# 循环协调器（核心改进）
# =============================================================================

class _LoopCoordinator:
    """
    内部循环协调器。

    负责：
    - 管理循环状态和转换
    - 失败检测（连续空调用、连续错误）
    - 恢复提示注入
    - 终止条件判断

    参考 quant_react_interview 的 _LoopCoordinator，但：
    - 使用 futureQuant 的 ToolRegistry 和 LLMClient
    - 支持 execute_tool 注入（允许子类自定义工具执行）
    - 支持 pipeline_ready 标志
    """

    def __init__(
        self,
        agent: EnhancedReActAgent,
        messages: List[Dict[str, Any]],
        max_iters: int,
        max_empty_turns: int,
        max_error_turns: int,
    ):
        self._agent = agent
        self._messages = messages
        self._max_iters = max_iters
        self._max_empty_turns = max_empty_turns
        self._max_error_turns = max_error_turns

        # 状态
        self._steps: List[ReActStep] = []
        self._metrics = ReActMetrics()
        self._recovery_prompts = agent._get_recovery_prompts()
        self._final_answer = ""
        self._finish_reason = ""
        self._had_llm_error = False  # 追踪是否发生过 LLM 调用失败

    def run(self) -> Dict[str, Any]:
        """执行主循环。"""
        turn_count = 0
        status = "max_iterations"

        while turn_count < self._max_iters:
            turn_count += 1

            # --- 失败检测 ---
            if self._metrics.consecutive_empty >= self._max_empty_turns:
                self._inject_recovery("stuck")
                self._metrics.reset_empty()
                self._metrics.reset_errors()

            if self._metrics.consecutive_errors >= self._max_error_turns:
                self._inject_recovery("no_progress")
                self._metrics.reset_empty()
                self._metrics.reset_errors()

            # --- LLM 调用 ---
            try:
                llm_resp = self._call_llm()
            except Exception as exc:
                self._had_llm_error = True
                self._messages.append({
                    "role": "user",
                    "content": f"API error: {str(exc)}. Please try again.",
                })
                self._metrics.record_error()
                self._metrics.record_empty()
                # 连续 LLM 错误达到上限时直接终止
                if self._metrics.consecutive_errors >= self._max_error_turns:
                    self._finish_reason = "llm_error"
                    self._final_answer = "LLM 调用失败，无法继续推理。"
                    self._steps.append(ReActStep(step_num=turn_count, thought=f"[LLM Error] {exc}"))
                    break
                continue

            thought = (llm_resp.content or "").strip()

            # --- 检查终止条件 ---
            parsed_answer = self._agent._parse_final_answer(thought)
            if parsed_answer is not None:
                self._final_answer = parsed_answer
                self._finish_reason = "final_answer"
                self._steps.append(ReActStep(step_num=turn_count, thought=thought))
                status = "completed"
                break

            # 检查 pipeline_ready
            if self._agent._is_pipeline_ready():
                self._finish_reason = "success"
                self._steps.append(ReActStep(step_num=turn_count, thought=thought))
                status = "success"
                break

            # --- 执行工具调用 ---
            observations: List[str] = []
            actions: List[Dict[str, Any]] = []

            if llm_resp.tool_calls:
                for tc in llm_resp.tool_calls:
                    actions.append(tc)
                    obs_text, had_error = self._execute_tool_call(tc)
                    observations.append(obs_text)
                    self._metrics.record_tool_call(not had_error)
                    if had_error:
                        self._metrics.record_error()
            else:
                observations.append(
                    "[System] No tool calls detected. "
                    "You must use tools to complete the task. "
                    "Call get_catalog() if you are unsure where to start."
                )
                self._metrics.record_empty()

            # --- 记录步骤 ---
            self._steps.append(
                ReActStep(
                    step_num=turn_count,
                    thought=thought,
                    action=actions[0] if actions else None,
                    observation=("\n".join(observations)) if observations else "",
                    metadata={
                        "tool_count": len(actions),
                        "usage": {
                            "prompt_tokens": llm_resp.usage.prompt_tokens if llm_resp.usage else 0,
                            "completion_tokens": llm_resp.usage.completion_tokens if llm_resp.usage else 0,
                        },
                    },
                )
            )

            # --- 更新消息历史 ---
            if llm_resp.tool_calls:
                # 有工具调用：assistant 消息 + tool 响应消息
                self._messages.append({
                    "role": "assistant",
                    "content": thought,
                    "tool_calls": self._encode_tool_calls(llm_resp.tool_calls),
                })
                for tc, obs in zip(llm_resp.tool_calls, observations):
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id"),
                        "name": tc.get("function", {}).get("name"),
                        "content": obs,
                    })
            else:
                # 无工具调用：assistant 纯文本消息
                self._messages.append({
                    "role": "assistant",
                    "content": thought,
                })

            # --- 检查最大步数 ---
            if turn_count == self._max_iters:
                self._finish_reason = "max_steps_reached"
                self._final_answer = (
                    thought or f"Reached max iterations ({self._max_iters}) without completing the task."
                )
                break

        self._agent._reasoning_log = ReActLog(
            agent_name=self._agent.name,
            task_id="unknown",
            steps=self._steps,
            final_answer=self._final_answer,
            finish_reason=self._finish_reason,
            metrics=self._metrics,
        )
        self._agent._metrics = self._metrics

        return {
            "steps": self._steps,
            "final_answer": self._final_answer,
            "finish_reason": self._finish_reason,
            "status": status,
            "iterations": turn_count,
        }

    def _call_llm(self) -> LLMResponse:
        """调用 LLM。"""
        try:
            resp = self._agent._llm.chat(
                self._messages,
                tools=self._agent._tool_specs,
            )
            return resp
        except Exception as exc:
            logger.error(f"[{self._agent.name}] LLM call failed: {exc}")
            raise

    def _execute_tool_call(self, tool_call: Any) -> tuple[str, bool]:
        """执行单个工具调用。"""
        import json as _json

        func = tool_call.get("function", {})
        name = func.get("name", "")
        raw_args = func.get("arguments", "{}")

        try:
            if isinstance(raw_args, str):
                args = _json.loads(raw_args)
            else:
                args = raw_args
        except _json.JSONDecodeError:
            return json.dumps({
                "success": False,
                "error": "Invalid JSON in tool arguments",
            }), True

        # 尝试使用 execute_tool（来自 research_tools）
        try:
            result = self._execute_research_tool(name, args)
            return _json.dumps(result, ensure_ascii=True), not result.get("success", True)
        except (ImportError, AttributeError):
            pass

        # 回退到 ToolRegistry
        try:
            result = self._agent._tools.execute_tool_call(tool_call)
            return result.to_text(), not result.success
        except Exception as exc:
            return json.dumps({"success": False, "error": str(exc)}), True

    def _execute_research_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """执行研究工具（从 pipeline_builder 模块导入）。"""
        from ...engine.nodes.pipeline_builder import execute_tool as research_execute_tool

        import asyncio
        try:
            loop = asyncio.get_running_loop()
            # 在已有事件循环中执行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, research_execute_tool(name, args))
                return future.result(timeout=30)
        except RuntimeError:
            # 没有运行中的事件循环
            return asyncio.run(research_execute_tool(name, args))

    def _inject_recovery(self, kind: str) -> None:
        """注入恢复提示。"""
        prompt = self._recovery_prompts.get(kind, self._recovery_prompts.get("stuck", ""))
        if prompt:
            self._messages.append({
                "role": "user",
                "content": f"[Recovery] {prompt}",
            })
            logger.debug(f"[{self._agent.name}] Injected recovery prompt: {kind}")

    def _encode_tool_calls(self, tool_calls: List[Any]) -> List[Dict[str, Any]]:
        """编码工具调用列表。"""
        result = []
        # 兼容测试 mock 的 dict 格式（直接传入 tool_calls 字典而非列表）
        if tool_calls and isinstance(tool_calls, dict) and "function" in tool_calls:
            tool_calls = [tool_calls]
        for tc in tool_calls:
            func = tc.get("function", {})
            result.append({
                "id": tc.get("id"),
                "type": "function",
                "function": {
                    "name": func.get("name", ""),
                    "arguments": func.get("arguments", "{}"),
                },
            })
        return result


# =============================================================================
# 向后兼容别名
# =============================================================================

# 保留原有的 ReActAgent 作为 EnhancedReActAgent 的别名，
# 确保现有测试和代码继续工作。
ReActAgent = EnhancedReActAgent

