"""
Tool 基类与注册表

为 Agent 提供统一的工具抽象：
- 每个 Tool 是一个可调用的功能单元
- ToolRegistry 负责收集 Tools 并生成 OpenAI Function Calling Schema
- @tool 装饰器简化工具函数包装

使用示例：
    >>> from futureQuant.agent.tools import Tool, ToolRegistry
    >>> registry = ToolRegistry()
    >>> registry.register(MyTool())
    >>> schema = registry.to_openai_schema()
"""

import inspect
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Type, Union

from ...core.logger import get_logger

logger = get_logger("agent.tools")


@dataclass
class ToolResult:
    """工具执行结果"""

    success: bool = True
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典（供 Observation 使用）"""
        return {
            "success": self.success,
            "data": self.data if self.data is not None else "",
            "error": self.error or "",
            "metadata": self.metadata,
        }

    def to_text(self) -> str:
        """转为文本（简化版 Observation）"""
        if not self.success:
            return f"[Error] {self.error}"
        if isinstance(self.data, str):
            return self.data
        try:
            return json.dumps(self.data, ensure_ascii=False, default=str)
        except Exception:
            return str(self.data)


class Tool(ABC):
    """
    工具抽象基类

    子类需实现：
    - name: 工具名称（LLM 可见）
    - description: 工具功能描述（LLM 可见）
    - execute(**kwargs) -> ToolResult: 执行逻辑
    """

    name: str = ""
    description: str = ""
    parameters: Optional[Dict[str, Any]] = None

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """执行工具逻辑"""
        ...

    def get_schema(self) -> Dict[str, Any]:
        """生成 OpenAI Function Calling Schema"""
        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
            },
        }
        if self.parameters:
            schema["function"]["parameters"] = self.parameters
        else:
            schema["function"]["parameters"] = {
                "type": "object",
                "properties": {},
                "required": [],
            }
        return schema

    def __call__(self, **kwargs) -> ToolResult:
        """直接调用 execute"""
        try:
            return self.execute(**kwargs)
        except Exception as exc:
            logger.error(f"Tool [{self.name}] execution failed: {exc}")
            return ToolResult(success=False, error=f"{type(exc).__name__}: {exc}")


class ToolRegistry:
    """
    工具注册表

    管理多个 Tool，提供 schema 生成和按名称调用。
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool_instance: Tool) -> "ToolRegistry":
        """注册工具"""
        if not tool_instance.name:
            raise ValueError("Tool name cannot be empty")
        self._tools[tool_instance.name] = tool_instance
        logger.debug(f"Tool registered: {tool_instance.name}")
        return self

    def register_all(self, *tools: Tool) -> "ToolRegistry":
        """批量注册"""
        for t in tools:
            self.register(t)
        return self

    def get(self, name: str) -> Optional[Tool]:
        """按名称获取工具"""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """检查是否存在"""
        return name in self._tools

    def list_names(self) -> List[str]:
        """列出所有工具名"""
        return list(self._tools.keys())

    def to_openai_schema(self) -> List[Dict[str, Any]]:
        """导出为 OpenAI tools 格式列表"""
        return [t.get_schema() for t in self._tools.values()]

    def execute(self, name: str, **kwargs) -> ToolResult:
        """按名称执行工具"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(success=False, error=f"Tool '{name}' not found")
        return tool(**kwargs)

    def execute_tool_call(self, tool_call: Dict[str, Any]) -> ToolResult:
        """
        执行 LLM 返回的 tool_call 对象

        tool_call 格式：
            {
                "id": "call_xxx",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": '{"query": "..."}'
                }
            }
        """
        func = tool_call.get("function", {})
        name = func.get("name", "")
        try:
            args = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError as exc:
            return ToolResult(success=False, error=f"Invalid JSON arguments: {exc}")
        return self.execute(name, **args)

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.list_names()})"


def _infer_schema_from_signature(func: Callable) -> Dict[str, Any]:
    """从函数签名推断 JSON Schema（极简版）"""
    sig = inspect.signature(func)
    properties: Dict[str, Any] = {}
    required: List[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        annotation = param.annotation
        prop: Dict[str, Any] = {"description": f"Parameter {param_name}"}

        if annotation is inspect.Parameter.empty:
            prop["type"] = "string"
        elif annotation in (str, "str"):
            prop["type"] = "string"
        elif annotation in (int, "int"):
            prop["type"] = "integer"
        elif annotation in (float, "float"):
            prop["type"] = "number"
        elif annotation in (bool, "bool"):
            prop["type"] = "boolean"
        else:
            prop["type"] = "string"

        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        else:
            prop["default"] = param.default

        properties[param_name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def tool(name: Optional[str] = None, description: Optional[str] = None):
    """
    装饰器：将普通函数包装为 Tool

    使用示例：
        >>> @tool(name="add", description="两数相加")
        ... def add(a: int, b: int) -> int:
        ...     return a + b
    """
    def decorator(func: Callable) -> Type[Tool]:
        tool_name = name or func.__name__
        tool_desc = description or (func.__doc__ or "")
        schema = _infer_schema_from_signature(func)

        class _FunctionTool(Tool):
            name = tool_name
            description = tool_desc
            parameters = schema

            def execute(self, **kwargs) -> ToolResult:
                try:
                    result = func(**kwargs)
                    return ToolResult(success=True, data=result)
                except Exception as exc:
                    return ToolResult(success=False, error=str(exc))

        _FunctionTool.__name__ = f"{func.__name__}Tool"
        return _FunctionTool

    return decorator
