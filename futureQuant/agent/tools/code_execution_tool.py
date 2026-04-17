"""
Code Execution Tool - 受限 Python 代码执行

允许 Agent 生成并执行 Python 代码片段，用于：
- Alpha101 因子验证
- 数据获取链路验证
- 数据清洗与转换

安全策略：
- 仅允许白名单内的模块导入
- 禁止 os/sys/subprocess/eval/compile/exec 等危险操作
- 代码在独立局部环境中运行，无法访问外部变量
- 执行超时控制
"""

import ast
import re
import threading
import traceback
from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult
from ...core.logger import get_logger

logger = get_logger("agent.tools.code_execution")


# 允许导入的模块白名单
ALLOWED_MODULES: set = {
    "pandas",
    "numpy",
    "math",
    "json",
    "datetime",
    "statistics",
    "typing",
    "collections",
    "itertools",
    "fractions",
    "decimal",
    "random",
    "hashlib",
    "re",
    "string",
    "time",
    "warnings",
    # 数据获取（受控）
    "akshare",
    "requests",
    "bs4",
    "lxml",
}

# 禁止的 AST 节点类型
FORBIDDEN_AST_NODES: tuple = (
    ast.ImportFrom,  # 禁止 from x import y，只允许 import x
)

# 禁止的 builtins
FORBIDDEN_BUILTINS: set = {
    "__import__",
    "open",
    "eval",
    "compile",
    "exec",
    "input",
    "exit",
    "quit",
    "help",
}


class CodeExecutionTool(Tool):
    """
    受限代码执行工具

    参数：
    - code: Python 代码字符串
    - timeout: 执行超时秒数（默认 30）
    - context: 传给代码的变量字典（默认空）
    """

    name = "code_execution"
    description = (
        "Execute Python code in a restricted sandbox environment. "
        "Allowed libraries: pandas, numpy, akshare, requests, bs4, math, json, datetime, etc. "
        "Forbidden: os, sys, subprocess, eval, compile, file operations."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute",
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds",
                "default": 30,
            },
            "context": {
                "type": "object",
                "description": "Variables to inject into the execution context",
                "default": {},
            },
        },
        "required": ["code"],
    }

    def execute(
        self,
        code: str,
        timeout: int = 30,
        context: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        code = code.strip()
        if not code:
            return ToolResult(success=False, error="Code is empty")

        # 1. 语法检查
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return ToolResult(success=False, error=f"Syntax error: {exc}")

        # 2. AST 安全检查
        check_result = self._check_ast(tree)
        if not check_result.success:
            return check_result

        # 3. 文本层安全检查（兜底）
        text_check = self._check_text(code)
        if not text_check.success:
            return text_check

        # 4. 构建受限执行环境
        safe_builtins = {
            name: getattr(__builtins__, name)
            for name in dir(__builtins__)
            if name not in FORBIDDEN_BUILTINS and not name.startswith("_")
        }

        # 注入安全的 __import__
        import builtins as _builtins

        _real_import = _builtins.__import__

        def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            top = name.split(".")[0]
            if top not in ALLOWED_MODULES:
                raise ImportError(f"Module '{top}' is not in the whitelist")
            return _real_import(name, globals, locals, fromlist, level)

        safe_builtins["__import__"] = _safe_import

        safe_globals = {"__builtins__": safe_builtins}
        safe_locals: Dict[str, Any] = dict(context or {})

        # 5. 执行（带超时）
        result_container: Dict[str, Any] = {"done": False, "result": None, "error": None}

        def _run():
            try:
                exec(code, safe_globals, safe_locals)
                result_container["result"] = safe_locals.get("result", safe_locals)
                result_container["done"] = True
            except Exception as exc:
                result_container["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                result_container["done"] = True

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return ToolResult(success=False, error=f"Code execution timed out after {timeout}s")

        if result_container["error"]:
            return ToolResult(success=False, error=result_container["error"])

        # 6. 序列化结果（尽量提取 result 变量）
        data = result_container["result"]
        if hasattr(data, "to_dict"):
            try:
                data = data.to_dict()
            except Exception:
                pass
        elif hasattr(data, "tolist"):
            try:
                data = data.tolist()
            except Exception:
                pass

        return ToolResult(
            success=True,
            data=data,
            metadata={"executed_lines": len(code.splitlines()), "timeout": timeout},
        )

    def _check_ast(self, tree: ast.AST) -> ToolResult:
        """AST 层安全检查"""
        for node in ast.walk(tree):
            # 禁止 from ... import ...
            if isinstance(node, ast.ImportFrom):
                return ToolResult(
                    success=False,
                    error=f"Forbidden 'from ... import ...' at line {getattr(node, 'lineno', '?')}",
                )

            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_module = alias.name.split(".")[0]
                    if top_module not in ALLOWED_MODULES:
                        return ToolResult(
                            success=False,
                            error=f"Forbidden module import: '{top_module}'",
                        )

            # 禁止调用特定属性（如 os.system, subprocess.run）
            if isinstance(node, ast.Attribute):
                attr_chain = self._get_attr_chain(node)
                if attr_chain:
                    root = attr_chain[0]
                    if root in ("os", "sys", "subprocess", "pathlib", "shutil", "socket"):
                        return ToolResult(
                            success=False,
                            error=f"Forbidden attribute access: '{'.'.join(attr_chain)}'",
                        )

        return ToolResult(success=True, data="AST check passed")

    @staticmethod
    def _get_attr_chain(node: ast.AST) -> Optional[List[str]]:
        """获取属性调用链，如 os.path.join -> ['os', 'path', 'join']"""
        chain: List[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            chain.insert(0, current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            chain.insert(0, current.id)
            return chain
        return None

    def _check_text(self, code: str) -> ToolResult:
        """文本层安全兜底检查"""
        # 禁止 __import__ 字符串调用
        if "__import__" in code:
            return ToolResult(success=False, error="Forbidden '__import__' detected")

        # 禁止直接 eval/exec/compile 调用（AST 可能漏掉字符串拼接场景）
        forbidden_calls = [r"\beval\s*\(", r"\bexec\s*\(", r"\bcompile\s*\("]
        for pattern in forbidden_calls:
            if re.search(pattern, code):
                return ToolResult(success=False, error=f"Forbidden function call pattern: {pattern}")

        return ToolResult(success=True, data="Text check passed")
