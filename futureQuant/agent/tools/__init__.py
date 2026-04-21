"""
Agent 统一工具层

所有 Agent 共享的工具集合，支持自动 schema 生成供 LLM Function Calling 使用。
"""

from .base import Tool, ToolRegistry, ToolResult, tool
from .blackboard_tool import BlackboardReadTool, BlackboardWriteTool
from .code_execution_tool import CodeExecutionTool
from .database_tool import DatabaseTool
from .memory_tool import MemoryTool
from .progress_tool import ProgressTool
from .web_search_tool import WebSearchTool
from .research_tools import get_tool_specs as get_research_tool_specs

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "tool",
    "BlackboardReadTool",
    "BlackboardWriteTool",
    "CodeExecutionTool",
    "DatabaseTool",
    "MemoryTool",
    "ProgressTool",
    "WebSearchTool",
    "get_research_tool_specs",
]
