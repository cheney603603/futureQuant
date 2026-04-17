"""
Blackboard Tool - 中央黑板读写

让 ReAct Agent 能够读取和写入 Blackboard，实现 Agent 间通信。
"""

from typing import Any, Dict, Optional, Set

from .base import Tool, ToolResult
from ..blackboard.blackboard import Blackboard
from ...core.logger import get_logger

logger = get_logger("agent.tools.blackboard")


class BlackboardReadTool(Tool):
    """读取 Blackboard 数据"""

    name = "blackboard_read"
    description = (
        "Read a value from the central blackboard by key. "
        "Returns default if the key does not exist."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Blackboard key to read",
            },
        },
        "required": ["key"],
    }

    def __init__(self, blackboard: Optional[Blackboard] = None):
        self._bb = blackboard

    def execute(self, key: str) -> ToolResult:
        bb = self._bb or Blackboard()
        try:
            value = bb.read(key, default=None)
            entry = bb.read_entry(key)
            meta = {
                "version": entry.version if entry else 0,
                "writer": entry.agent if entry else None,
                "timestamp": entry.timestamp if entry else None,
            } if entry else {}
            return ToolResult(
                success=True,
                data=value if value is not None else "",
                metadata={"exists": entry is not None, **meta},
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))


class BlackboardWriteTool(Tool):
    """写入 Blackboard 数据"""

    name = "blackboard_write"
    description = (
        "Write a value to the central blackboard. "
        "Other agents can read this value later by the specified key."
    )
    parameters = {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Blackboard key to write",
            },
            "value": {
                "description": "Value to write (string, number, dict, or list)",
            },
            "agent_name": {
                "type": "string",
                "description": "Name of the writing agent",
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional tags",
            },
        },
        "required": ["key", "value", "agent_name"],
    }

    def __init__(self, blackboard: Optional[Blackboard] = None):
        self._bb = blackboard

    def execute(
        self,
        key: str,
        value: Any,
        agent_name: str,
        tags: Optional[list] = None,
    ) -> ToolResult:
        bb = self._bb or Blackboard()
        try:
            tag_set = set(tags) if tags else set()
            entry = bb.write(
                key=key,
                value=value,
                agent=agent_name,
                tags=tag_set,
            )
            return ToolResult(
                success=True,
                data={"written": True, "version": entry.version},
                metadata={"key": key, "agent": agent_name},
            )
        except Exception as exc:
            return ToolResult(success=False, error=str(exc))
