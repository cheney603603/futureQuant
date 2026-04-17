"""
Memory Tool - 记忆存储与管理

封装 AgentLoopController 中的 MemoryBank，供 ReAct Agent 调用。
"""

from typing import Any, Dict, Optional

from .base import Tool, ToolResult
from ..shared.memory_bank import MemoryBank
from ...core.logger import get_logger

logger = get_logger("agent.tools.memory")


class MemoryTool(Tool):
    """
    记忆工具

    支持读取/写入 Agent 执行历史、成功案例和失败教训。
    """

    name = "memory"
    description = (
        "Access persistent agent memory bank. "
        "Can record execution results, retrieve success patterns, or get learned context for a given agent."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["record", "get_history", "get_success_patterns", "get_failure_lessons", "get_learned_context"],
                "description": "Memory action to perform",
            },
            "agent_name": {
                "type": "string",
                "description": "Target agent name",
            },
            "result": {
                "type": "object",
                "description": "Execution result dict (required for 'record' action)",
            },
            "limit": {
                "type": "integer",
                "description": "Max records to retrieve",
                "default": 10,
            },
        },
        "required": ["action", "agent_name"],
    }

    def __init__(self):
        self._bank = MemoryBank()

    def execute(
        self,
        action: str,
        agent_name: str,
        result: Optional[Dict[str, Any]] = None,
        limit: int = 10,
    ) -> ToolResult:
        try:
            if action == "record":
                if result is None:
                    return ToolResult(success=False, error="'result' is required for action='record'")
                self._bank.record_run(agent_name, result)
                return ToolResult(success=True, data={"recorded": True})

            elif action == "get_history":
                history = self._bank.get_history(agent_name, limit=limit)
                return ToolResult(success=True, data=history)

            elif action == "get_success_patterns":
                patterns = self._bank.get_success_patterns(agent_name)
                return ToolResult(success=True, data=patterns)

            elif action == "get_failure_lessons":
                lessons = self._bank.get_failure_lessons(agent_name)
                return ToolResult(success=True, data=lessons)

            elif action == "get_learned_context":
                context = self._bank.get_learned_context(agent_name)
                return ToolResult(success=True, data=context)

            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as exc:
            logger.error(f"MemoryTool error: {exc}")
            return ToolResult(success=False, error=str(exc))
