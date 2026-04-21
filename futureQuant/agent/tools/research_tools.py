"""
因子研究通用工具表面 (Factor Research Generic Tool Surface)

参考 quant_react_interview-main/agent/tools.py 设计，
为 LLM 提供 6 个通用工具用于构建因子研究流水线。

设计原则：
1. 通用性：工具是通用的操作（add_step, connect_steps 等），而非针对特定步骤
2. 丰富引导：tool description 包含详细的配置格式说明和常见错误
3. 清晰错误：每次返回 success=False 时附带 hint，引导模型修复
4. 即时验证：add_step 后立即尝试执行，错误立即返回

工具列表：
- add_step: 添加步骤到流水线
- update_step: 更新步骤配置
- connect_steps: 连接步骤（定义执行顺序）
- get_catalog: 获取步骤目录
- get_details: 获取步骤详细信息
- get_pipeline: 导出完整流水线
- validate_config: 验证配置格式

使用示例（LLM 调用）：
{
    "name": "add_step",
    "arguments": {
        "kind": "data.price_bars",
        "step_id": "price_bars",
        "config": {
            "target": "$trigger['target']",
            "start_date": "$trigger['start_date']",
            "end_date": "$trigger['end_date']",
            "lookback_days": 60
        }
    }
}
"""

from __future__ import annotations

from typing import Any, Dict, List


# =============================================================================
# 工具规范定义
# =============================================================================

def get_tool_specs() -> List[Dict[str, Any]]:
    """
    获取 OpenAI 格式的工具规范列表。

    Returns:
        List of tool specifications compatible with OpenAI function calling format.
    """
    return [
        _fn(
            name="add_step",
            description=_ADD_STEP_DESC,
            properties={
                "kind": {
                    "type": "string",
                    "description": (
                        "Step kind from the catalog. Must be one of: "
                        "trigger.manual, data.price_bars, data.fundamental, "
                        "factor.technical, factor.fundamental, factor.alpha101, "
                        "evaluation.ic, evaluation.robustness, "
                        "fusion.icir_weight, fusion.multifactor, "
                        "backtest.factor_signal, backtest.walk_forward, "
                        "output.report. "
                        "Call get_catalog first to see all available options."
                    ),
                },
                "step_id": {
                    "type": "string",
                    "description": (
                        "Optional custom ID for the step. "
                        "Use simple names like 'trigger', 'price_bars', 'momentum', 'ic_eval', 'fusion', 'backtest'. "
                        "If omitted, auto-generated from kind."
                    ),
                },
                "config": {
                    "type": "object",
                    "description": (
                        "Step configuration. Structure depends on the step kind. "
                        "ALWAYS call get_details(kind) FIRST to understand required fields and correct format. "
                        "Use $step_id['field'] syntax to reference outputs from other steps."
                    ),
                },
            },
            required=["kind", "config"],
        ),
        _fn(
            name="update_step",
            description=_UPDATE_STEP_DESC,
            properties={
                "step_id": {
                    "type": "string",
                    "description": "ID of the step to update.",
                },
                "config": {
                    "type": "object",
                    "description": (
                        "New or modified config fields. Merges with existing config. "
                        "Call get_details(kind) to understand the correct format."
                    ),
                },
            },
            required=["step_id", "config"],
        ),
        _fn(
            name="connect_steps",
            description=_CONNECT_STEPS_DESC,
            properties={
                "source_id": {
                    "type": "string",
                    "description": "Step that runs first (upstream / producer of data).",
                },
                "target_id": {
                    "type": "string",
                    "description": "Step that runs after (downstream / consumer of data).",
                },
            },
            required=["source_id", "target_id"],
        ),
        _fn(
            name="get_catalog",
            description=_GET_CATALOG_DESC,
            properties={},
            required=[],
        ),
        _fn(
            name="get_details",
            description=_GET_DETAILS_DESC,
            properties={
                "kind": {
                    "type": "string",
                    "description": (
                        "Step kind to inspect. Example: 'data.price_bars' or 'factor.technical'. "
                        "Use the kind exactly as shown in get_catalog()."
                    ),
                },
            },
            required=["kind"],
        ),
        _fn(
            name="get_pipeline",
            description=_GET_PIPELINE_DESC,
            properties={},
            required=[],
        ),
        _fn(
            name="validate_config",
            description=_VALIDATE_CONFIG_DESC,
            properties={
                "kind": {
                    "type": "string",
                    "description": "Step kind to validate.",
                },
                "config": {
                    "type": "object",
                    "description": "Config object to validate.",
                },
            },
            required=["kind", "config"],
        ),
    ]


def _fn(
    name: str,
    description: str,
    properties: Dict[str, Any],
    required: List[str],
) -> Dict[str, Any]:
    """构建 OpenAI 格式的函数规范。"""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


# =============================================================================
# 工具描述（详细版）
# =============================================================================

_ADD_STEP_DESC = """Create a new step in the factor research pipeline.

BEFORE calling this tool:
1. Call get_catalog() to see all available step kinds
2. Call get_details(kind) to understand the config structure for your chosen kind
3. Prepare config with correct field names and $step_id['field'] reference syntax

Reference syntax ($step_id['field']):
- Use $step_id['field'] to access output from another step
- Example: $trigger['target'] to get the target from the trigger step
- Example: $price_bars['data'] to get the data from a data step
- Example: $momentum['factors'] to get the computed factors

Common pipeline order for factor research:
1. trigger.manual (seed parameters: target, dates)
2. data.price_bars (fetch OHLCV data)
3. data.fundamental (optional: fetch basis/inventory data)
4. factor.technical OR factor.alpha101 (compute factors)
5. evaluation.ic (evaluate IC/ICIR)
6. fusion.icir_weight OR fusion.multifactor (combine factors)
7. backtest.factor_signal (backtest the signal)
8. output.report (generate report)

CRITICAL:
- Always call get_details(kind) BEFORE add_step to understand the correct config format
- If config has errors, the tool will return success=False with an error message
- Use update_step to fix errors instead of removing and re-adding steps
"""

_UPDATE_STEP_DESC = """Modify an existing step's configuration.

Use this to:
- Fix a step that had an error
- Add missing required fields
- Adjust parameters (e.g., lookback_days, ic_threshold, windows)

The config MERGES with existing values - only specify fields you want to change.
Step kind cannot be changed after creation.

Common fixes:
- Missing required field -> add the field to config
- Wrong reference syntax -> use $step_id['field'] format
- Wrong data type -> check field_descriptions in get_details output
"""

_CONNECT_STEPS_DESC = """Define execution order between two steps.

The source step runs first, then the target step.
This determines both execution order AND data flow direction.

For a linear pipeline, connect each step to the next:
  connect_steps('trigger', 'price_bars')
  connect_steps('price_bars', 'momentum')
  connect_steps('momentum', 'ic_eval')
  connect_steps('ic_eval', 'fusion')
  ...etc

For a branching pipeline:
  connect_steps('price_bars', 'technical')
  connect_steps('price_bars', 'fundamental')  # Both depend on price_bars
"""

_GET_CATALOG_DESC = """List all available step kinds with brief summaries.

Returns for each kind:
- kind: Unique identifier (e.g., 'data.price_bars')
- category: Grouping (trigger, data, factor, evaluation, fusion, backtest, output)
- purpose: What the step does
- required_fields: Must provide these in config
- optional_fields: Have defaults, can override
- example_config: Shows typical usage

Call this FIRST to understand what steps are available.
Use get_details(kind) to dive deeper into any specific step.
"""

_GET_DETAILS_DESC = """Get detailed information about a specific step kind.

Returns comprehensive metadata including:
- All catalog fields (kind, category, purpose, etc.)
- field_descriptions: Detailed explanation of each config field (semantics, not just types)
- output_shape: What the step outputs and HOW to access it ($step_id['data'], etc.)
- notes: Usage guidance and best practices
- common_mistakes: Things to AVOID - LLM frequently makes these errors

ALWAYS call this BEFORE add_step or update_step to understand:
1. What fields are required vs optional
2. What the correct format for each field is
3. What the output looks like and how to reference it in downstream steps
"""

_GET_PIPELINE_DESC = """Export the current pipeline when complete.

Returns the full pipeline definition with all steps and connections.

ONLY call this when:
- All required steps are added
- Steps are connected in correct order
- No errors in step configs (check all add_step/update_step calls were success=True)

The pipeline is checked for basic validity (non-empty steps, proper connections).
"""

_VALIDATE_CONFIG_DESC = """Validate a config against a step kind's schema without executing.

Use this to:
- Check if a config is valid before calling add_step
- Identify missing required fields
- Catch wrong field names or types early

Returns validation result with errors and warnings.
"""
