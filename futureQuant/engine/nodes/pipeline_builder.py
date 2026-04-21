"""
因子研究流水线构建器 (Factor Research Pipeline Builder)

参考 quant_react_interview-main/engine/core/builder.py 设计，
为因子研究工作流提供流水线构建和执行能力。

核心功能：
- 使用 add_step() / update_step() / connect_steps() 构建流水线
- 支持 $step_id['field'] 引用语法，在执行时自动解析
- 立即执行验证（add_step 后尝试执行，错误立即返回）
- ExecutionContext 管理步骤间数据传递

使用示例:
    from futureQuant.engine.nodes import FactorPipelineBuilder

    builder = FactorPipelineBuilder()
    builder.add_step(
        kind="trigger.manual",
        config={"target": "RB", "start_date": "2023-01-01"},
        step_id="trigger",
    )
    builder.add_step(
        kind="data.price_bars",
        config={
            "target": "$trigger['target']",
            "start_date": "$trigger['start_date']",
            "end_date": "$trigger['end_date']",
        },
        step_id="price_bars",
    )
    builder.connect_steps("trigger", "price_bars")

    # 导出流水线
    pipeline = builder.get_pipeline()
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .factor_catalog import get_catalog as catalog_list, get_details as catalog_details


# =============================================================================
# 执行上下文
# =============================================================================

@dataclass
class ExecutionContext:
    """
    执行上下文 - 管理流水线中各步骤的数据传递。

    每个步骤执行后，其输出存储在此上下文中，
    供后续步骤通过引用语法（如 $price_bars['data']）访问。

    Attributes:
        pipeline_id: 流水线 ID
        outputs: 步骤 ID -> 步骤输出的映射
        metadata: 流水线元数据（如统计信息）
    """

    pipeline_id: str
    outputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def set_output(self, step_id: str, output: Any) -> None:
        """存储步骤的输出结果。"""
        self.outputs[step_id] = output

    def get_output(self, step_id: str) -> Any:
        """获取指定步骤的输出。"""
        return self.outputs.get(step_id)

    def has_output(self, step_id: str) -> bool:
        """检查指定步骤是否有输出。"""
        return step_id in self.outputs

    def resolve_reference(self, reference: str) -> Any:
        """
        解析 $step_id['field'] 或 $step_id 格式的引用。

        Args:
            reference: 引用字符串，如 "$price_bars['data']"

        Returns:
            解析后的值

        Raises:
            ValueError: 引用格式无效或目标不存在
        """
        # 匹配 $step_id['field'] 格式
        match = re.match(r"^\$([\w]+)\[([\'\"])(.*?)\2\]$", reference)
        if match:
            step_id = match.group(1)
            field_name = match.group(3)
            output = self.get_output(step_id)
            if output is None:
                raise ValueError(f"Step '{step_id}' has no output yet")
            if isinstance(output, dict):
                if field_name not in output:
                    raise ValueError(
                        f"Field '{field_name}' not found in step '{step_id}' output. "
                        f"Available keys: {list(output.keys())}"
                    )
                return output[field_name]
            elif hasattr(output, field_name):
                return getattr(output, field_name)
            else:
                raise ValueError(
                    f"Cannot access '{field_name}' from step '{step_id}' output "
                    f"(type: {type(output).__name__})"
                )

        # 匹配 $step_id 格式（直接引用输出本身）
        if reference.startswith("$"):
            step_id = reference[1:]
            output = self.get_output(step_id)
            if output is None:
                raise ValueError(f"Step '{step_id}' has no output yet")
            return output

        raise ValueError(f"Invalid reference format: {reference}")


# =============================================================================
# 步骤执行器注册表
# =============================================================================

@dataclass
class StepResult:
    """步骤执行结果。"""

    step_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    stage: str = "unknown"  # tooling | validation | execution


class StepExecutorRegistry:
    """
    步骤执行器注册表。

    将步骤类型（kind）映射到具体的执行函数。
    每个执行函数接受 (config, context) 并返回执行结果。

    使用示例:
        registry = StepExecutorRegistry()
        registry.register("data.price_bars", my_data_handler)
    """

    def __init__(self):
        self._handlers: Dict[str, Any] = {}

    def register(self, kind: str, handler: Any) -> None:
        """注册步骤执行器。"""
        self._handlers[kind] = handler

    def get(self, kind: str) -> Optional[Any]:
        """获取步骤执行器。"""
        return self._handlers.get(kind)

    def list_kinds(self) -> List[str]:
        """列出所有已注册的步骤类型。"""
        return sorted(self._handlers.keys())

    def is_registered(self, kind: str) -> bool:
        """检查步骤类型是否已注册。"""
        return kind in self._handlers


# =============================================================================
# 因子研究流水线构建器
# =============================================================================

class FactorPipelineBuilder:
    """
    因子研究流水线构建器。

    提供声明式构建因子研究流水线的能力：
    - add_step: 添加步骤
    - update_step: 更新步骤配置
    - connect_steps: 连接步骤（定义执行顺序）
    - execute_step: 执行单个步骤
    - get_pipeline: 导出完整流水线定义

    支持 $step_id['field'] 引用语法，在执行时自动解析。

    使用示例:
        builder = FactorPipelineBuilder()
        builder.add_step(
            kind="trigger.manual",
            config={"target": "RB", "start_date": "2023-01-01"},
            step_id="trigger",
        )
        # ... 添加更多步骤
        pipeline = builder.get_pipeline()
    """

    def __init__(self, registry: Optional[StepExecutorRegistry] = None):
        """
        初始化流水线构建器。

        Args:
            registry: 步骤执行器注册表，若为 None 则创建新的空注册表。
        """
        self._registry = registry or StepExecutorRegistry()
        self._steps: Dict[str, Dict[str, Any]] = {}
        self._runtime = ExecutionContext(pipeline_id="factor_pipeline")
        self._catalog_kinds = self._load_catalog_kinds()

    def _load_catalog_kinds(self) -> List[str]:
        """从目录加载所有可用的步骤类型。"""
        try:
            return [item["kind"] for item in catalog_list()]
        except Exception:
            return []

    # -------------------------------------------------------------------------
    # 步骤管理
    # -------------------------------------------------------------------------

    def add_step(
        self,
        kind: str,
        config: Dict[str, Any],
        step_id: Optional[str] = None,
    ) -> str:
        """
        添加一个步骤到流水线。

        Args:
            kind: 步骤类型，必须在目录中（如 "data.price_bars"）
            config: 步骤配置字典
            step_id: 步骤 ID（可选，默认从 kind 自动生成）

        Returns:
            分配的步骤 ID

        Raises:
            ValueError: kind 不在目录中
        """
        # 验证 kind 是否有效
        if self._catalog_kinds and kind not in self._catalog_kinds:
            raise ValueError(
                f"Unknown step kind: {kind}. "
                f"Valid kinds: {', '.join(self._catalog_kinds)}"
            )

        # 生成或验证 step_id
        resolved_id = step_id or self._allocate_step_id(kind)
        if resolved_id in self._steps:
            raise ValueError(f"Step ID already exists: {resolved_id}")

        self._steps[resolved_id] = {
            "id": resolved_id,
            "kind": kind,
            "name": kind,
            "config": dict(config),
            "next": [],
        }
        return resolved_id

    def update_step(self, step_id: str, config: Dict[str, Any]) -> None:
        """
        更新现有步骤的配置。

        Args:
            step_id: 要更新的步骤 ID
            config: 新的配置字段（与现有配置合并）

        Raises:
            KeyError: step_id 不存在
        """
        if step_id not in self._steps:
            raise KeyError(f"Step not found: {step_id}. Existing steps: {list(self._steps.keys())}")
        current = self._steps[step_id].setdefault("config", {})
        current.update(config)

    def connect_steps(self, source_id: str, target_id: str) -> None:
        """
        连接两个步骤，定义执行顺序。

        source 步骤先执行，然后执行 target 步骤。
        数据通过 ExecutionContext 传递。

        Args:
            source_id: 上游步骤 ID
            target_id: 下游步骤 ID

        Raises:
            KeyError: 步骤 ID 不存在
        """
        for sid in (source_id, target_id):
            if sid not in self._steps:
                raise KeyError(f"Step not found: {sid}. Existing steps: {list(self._steps.keys())}")

        chain = self._steps[source_id].setdefault("next", [])
        if target_id not in chain:
            chain.append(target_id)

    def remove_step(self, step_id: str) -> bool:
        """移除步骤，返回是否成功。"""
        if step_id not in self._steps:
            return False
        del self._steps[step_id]
        # 从其他步骤的 next 列表中移除
        for step in self._steps.values():
            if "next" in step and step_id in step["next"]:
                step["next"].remove(step_id)
        return True

    # -------------------------------------------------------------------------
    # 步骤执行
    # -------------------------------------------------------------------------

    async def execute_step(self, step_id: str) -> StepResult:
        """
        执行单个步骤。

        步骤的执行流程：
        1. 从 _steps 获取配置
        2. 调用 _materialize_inputs() 解析所有 $reference 引用
        3. 从注册表获取执行器
        4. 调用执行器
        5. 将结果存入 ExecutionContext

        Args:
            step_id: 要执行的步骤 ID

        Returns:
            StepResult

        Raises:
            KeyError: 步骤不存在
        """
        if step_id not in self._steps:
            return StepResult(
                step_id=step_id,
                success=False,
                error=f"Step not found: {step_id}",
                stage="lookup",
            )

        step = self._steps[step_id]
        kind = step["kind"]

        # 检查是否有执行器
        handler = self._registry.get(kind)
        if handler is None:
            return StepResult(
                step_id=step_id,
                success=False,
                error=f"No executor registered for kind: {kind}",
                stage="lookup",
                hint="Register an executor with builder.register_step(kind, handler)",
            )

        # 解析引用
        try:
            materialized_config = self._materialize(step["config"])
        except Exception as exc:
            return StepResult(
                step_id=step_id,
                success=False,
                error=f"Failed to resolve config references: {exc}",
                stage="validation",
            )

        # 执行
        try:
            output = await handler.execute(materialized_config, self._runtime)
            self._runtime.set_output(step_id, output)
            return StepResult(
                step_id=step_id,
                success=True,
                output=output,
                stage="execution",
            )
        except Exception as exc:
            return StepResult(
                step_id=step_id,
                success=False,
                error=str(exc),
                stage="execution",
            )

    def execute_step_sync(self, step_id: str) -> StepResult:
        """同步版本（用于不支持 async 的场景）。"""
        import asyncio
        return asyncio.run(self.execute_step(step_id))

    async def execute_pipeline(self) -> Dict[str, StepResult]:
        """
        执行完整流水线（按连接顺序）。

        Returns:
            dict of step_id -> StepResult
        """
        # 拓扑排序获取执行顺序
        order = self._topological_order()
        results: Dict[str, StepResult] = {}

        for step_id in order:
            result = await self.execute_step(step_id)
            results[step_id] = result
            if not result.success:
                # 关键步骤失败，停止流水线
                break

        return results

    def _topological_order(self) -> List[str]:
        """基于连接关系计算拓扑排序。"""
        visited: set = set()
        order: List[str] = []

        def visit(step_id: str):
            if step_id in visited:
                return
            visited.add(step_id)
            # 先访问依赖
            for next_id in self._steps.get(step_id, {}).get("next", []):
                visit(next_id)
            order.append(step_id)

        # 从 trigger（入口）开始
        trigger_steps = [sid for sid, s in self._steps.items() if s["kind"] == "trigger.manual"]
        if trigger_steps:
            for tid in trigger_steps:
                visit(tid)
        else:
            # 没有 trigger，按注册顺序
            for sid in self._steps:
                visit(sid)

        return order

    # -------------------------------------------------------------------------
    # 引用解析
    # -------------------------------------------------------------------------

    def _materialize(self, value: Any) -> Any:
        """
        递归解析值中的 $reference 引用。

        支持:
        - $step_id['field'] -> 从步骤输出中获取字段
        - $step_id -> 获取步骤输出本身
        - 普通值原样返回
        """
        if isinstance(value, dict):
            return {k: self._materialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._materialize(item) for item in value]
        if isinstance(value, str) and self._is_reference(value):
            return self._resolve_reference(value)
        return value

    def _is_reference(self, value: str) -> bool:
        """判断字符串是否为引用格式。"""
        if not isinstance(value, str):
            return False
        return (
            value.startswith("$") and
            (re.search(r"\$[\w]+\[", value) or re.match(r"^\$[\w]+$", value))
        )

    def _resolve_reference(self, reference: str) -> Any:
        """解析单个引用。"""
        try:
            return self._runtime.resolve_reference(reference)
        except ValueError:
            # 返回原引用（未解析的引用）
            return reference

    # -------------------------------------------------------------------------
    # 注册执行器
    # -------------------------------------------------------------------------

    def register_step(self, kind: str, executor: Any) -> "FactorPipelineBuilder":
        """
        注册步骤执行器。

        Args:
            kind: 步骤类型
            executor: 执行器对象（需要有 execute(config, context) 方法）

        Returns:
            self（支持链式调用）
        """
        self._registry.register(kind, executor)
        return self

    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------

    def _allocate_step_id(self, kind: str) -> str:
        """为步骤类型分配唯一的 ID。"""
        stem = kind.replace(".", "_")
        candidate = stem
        counter = 1
        while candidate in self._steps:
            candidate = f"{stem}_{counter}"
            counter += 1
        return candidate

    def snapshot_step_ids(self) -> List[str]:
        """获取当前所有步骤 ID 的快照。"""
        return list(self._steps.keys())

    def get_pipeline(self) -> Dict[str, Any]:
        """
        导出完整的流水线定义。

        Returns:
            dict，包含:
            - pipeline_id: 流水线 ID
            - steps: 步骤列表（不含运行时输出）
            - metadata: 元数据（统计信息）
        """
        return {
            "pipeline_id": self._runtime.pipeline_id,
            "steps": list(self._steps.values()),
            "metadata": dict(self._runtime.metadata),
        }

    def get_config(self, step_id: str) -> Optional[Dict[str, Any]]:
        """获取步骤的配置（不含解析后的值）。"""
        if step_id not in self._steps:
            return None
        return dict(self._steps[step_id].get("config", {}))

    def __repr__(self) -> str:
        return (
            f"FactorPipelineBuilder("
            f"steps={len(self._steps)}, "
            f"registered_kinds={self._registry.list_kinds()})"
        )


# =============================================================================
# 工具接口（供 LLM 工具调用）
# =============================================================================

_active_builder: Optional[FactorPipelineBuilder] = None


def bind_builder(builder: FactorPipelineBuilder) -> None:
    """将构建器绑定到模块级全局变量（供 execute_tool 使用）。"""
    global _active_builder
    _active_builder = builder


async def execute_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行工具命令（供 LLM 通过工具调用使用）。

    支持的工具:
    - add_step: 添加步骤
    - update_step: 更新步骤配置
    - connect_steps: 连接步骤
    - get_catalog: 获取目录
    - get_details: 获取步骤详情
    - get_pipeline: 导出流水线
    """
    if _active_builder is None:
        return {"success": False, "error": "Builder not bound", "hint": "Call bind_builder() first"}

    handlers = {
        "add_step": _tool_add_step,
        "update_step": _tool_update_step,
        "connect_steps": _tool_connect_steps,
        "get_catalog": _tool_get_catalog,
        "get_details": _tool_get_details,
        "get_pipeline": _tool_get_pipeline,
        "validate_config": _tool_validate_config,
    }

    if name not in handlers:
        return {"success": False, "error": f"Unknown tool: {name}"}

    try:
        return await handlers[name](arguments)
    except Exception as exc:
        return {"success": False, "error": str(exc), "stage": "tooling"}


async def _tool_add_step(payload: Dict[str, Any]) -> Dict[str, Any]:
    """添加工具。"""
    kind = payload.get("kind")
    if not kind:
        return {"success": False, "error": "Missing required field: kind"}

    config = payload.get("config", {})
    step_id = payload.get("step_id")

    # 验证 kind
    valid_kinds = [item["kind"] for item in catalog_list()]
    if kind not in valid_kinds:
        return {
            "success": False,
            "error": f"Invalid kind: {kind}",
            "hint": f"Valid kinds: {', '.join(valid_kinds)}. Call get_catalog() to see all options.",
        }

    try:
        created_id = _active_builder.add_step(kind=kind, config=config, step_id=step_id)
    except ValueError as exc:
        return {"success": False, "error": str(exc)}

    # 尝试执行以验证配置
    result = await _try_execute(_active_builder, created_id)
    result["action"] = "add_step"
    result["step_id"] = created_id
    result["kind"] = kind
    return result


async def _tool_update_step(payload: Dict[str, Any]) -> Dict[str, Any]:
    """更新工具。"""
    step_id = payload.get("step_id")
    if not step_id:
        return {"success": False, "error": "Missing required field: step_id"}

    config = payload.get("config", {})
    if not config:
        return {"success": False, "error": "No config changes provided"}

    existing = _active_builder.snapshot_step_ids()
    if step_id not in existing:
        return {
            "success": False,
            "error": f"Step not found: {step_id}",
            "hint": f"Existing steps: {', '.join(existing)}",
        }

    _active_builder.update_step(step_id, config)
    result = await _try_execute(_active_builder, step_id)
    result["action"] = "update_step"
    return result


async def _tool_connect_steps(payload: Dict[str, Any]) -> Dict[str, Any]:
    """连接工具。"""
    source_id = payload.get("source_id")
    target_id = payload.get("target_id")
    if not source_id or not target_id:
        return {"success": False, "error": "Missing required fields: source_id, target_id"}

    existing = _active_builder.snapshot_step_ids()
    missing = []
    if source_id not in existing:
        missing.append(f"source_id: {source_id}")
    if target_id not in existing:
        missing.append(f"target_id: {target_id}")

    if missing:
        return {
            "success": False,
            "error": f"Steps not found: {', '.join(missing)}",
            "hint": f"Existing steps: {', '.join(existing)}",
        }

    _active_builder.connect_steps(source_id, target_id)
    return {
        "success": True,
        "action": "connect_steps",
        "source_id": source_id,
        "target_id": target_id,
        "message": f"Connected {source_id} -> {target_id}",
    }


async def _tool_get_catalog(_: Dict[str, Any]) -> Dict[str, Any]:
    """目录工具。"""
    return {
        "success": True,
        "action": "get_catalog",
        "catalog": catalog_list(),
        "categories": [
            "trigger - 流水线起点",
            "data - 数据获取",
            "factor - 因子计算",
            "evaluation - 因子评估",
            "fusion - 因子融合",
            "backtest - 回测验证",
            "output - 报告生成",
        ],
        "hint": "Call get_details(kind) for any step to understand its config structure.",
    }


async def _tool_get_details(payload: Dict[str, Any]) -> Dict[str, Any]:
    """详情工具。"""
    kind = payload.get("kind")
    if not kind:
        return {"success": False, "error": "Missing required field: kind"}

    details = catalog_details(kind)
    if "error" in details:
        return {
            "success": False,
            "error": details["error"],
            "hint": details.get("hint", ""),
        }

    return {
        "success": True,
        "action": "get_details",
        "details": details,
    }


async def _tool_get_pipeline(_: Dict[str, Any]) -> Dict[str, Any]:
    """导出流水线工具。"""
    pipeline = _active_builder.get_pipeline()
    step_ids = _active_builder.snapshot_step_ids()

    if not pipeline["steps"]:
        return {
            "success": False,
            "action": "get_pipeline",
            "error": "Pipeline is empty. Add steps with add_step() first.",
            "pipeline": pipeline,
        }

    # 检查连通性
    connected = sum(1 for s in pipeline["steps"] if s.get("next"))
    if connected == 0 and len(step_ids) > 1:
        return {
            "success": False,
            "action": "get_pipeline",
            "error": "Steps are not connected. Use connect_steps() to define execution order.",
            "pipeline": pipeline,
        }

    return {
        "success": True,
        "action": "get_pipeline",
        "pipeline": pipeline,
        "step_count": len(pipeline["steps"]),
        "connected_pairs": [
            {"from": s["id"], "to": n}
            for s in pipeline["steps"]
            for n in s.get("next", [])
        ],
    }


async def _tool_validate_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    """配置验证工具。"""
    kind = payload.get("kind")
    config = payload.get("config", {})
    if not kind:
        return {"success": False, "error": "Missing required field: kind"}

    from .factor_catalog import FactorCatalog

    catalog = FactorCatalog()
    result = catalog.validate_config(kind, config)
    result["action"] = "validate_config"
    result["kind"] = kind
    return result


async def _try_execute(builder: FactorPipelineBuilder, step_id: str) -> Dict[str, Any]:
    """尝试执行步骤，返回结果（用于工具调用反馈）。"""
    result = await builder.execute_step(step_id)
    return {
        "success": result.success,
        "step_id": step_id,
        "error": result.error,
        "stage": result.stage,
        "output": result.output if result.success else None,
        "hint": _get_hint_for_error(result),
    }


def _get_hint_for_error(result: StepResult) -> Optional[str]:
    """根据错误类型返回修复提示。"""
    if not result.error:
        return None

    error_lower = result.error.lower()

    if "no executor" in error_lower or "not found" in error_lower:
        return "This step kind requires an executor to be registered. Skipping execution for now."

    if "not found" in error_lower or "no output" in error_lower:
        return "Check that all referenced steps exist and have been executed."

    if "unknown" in error_lower:
        return "Verify the step kind is correct. Use get_catalog() to see valid kinds."

    return "Use get_details(kind) to check the correct config format for this step."
