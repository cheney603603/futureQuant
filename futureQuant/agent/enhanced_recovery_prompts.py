"""
增强型恢复提示 (Enhanced Recovery Prompts)

针对因子研究场景定制的恢复提示，
在 LLM 遇到失败时提供结构化的修复指导。

核心功能：
1. 错误分类（missing_reference, wrong_format, execution_failed 等）
2. 根据错误类型生成定制化恢复提示
3. 提供具体的修复示例

使用示例：
    from futureQuant.agent.enhanced_recovery_prompts import (
        get_recovery_prompt,
        diagnose_error,
        suggest_fix,
    )
    
    # 诊断错误
    diagnosis = diagnose_error(error_message, step_kind, config)
    
    # 获取恢复提示
    prompt = get_recovery_prompt(diagnosis["error_type"], diagnosis)
    
    # 获取修复建议
    fix = suggest_fix(diagnosis)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ErrorDiagnosis:
    """错误诊断结果。"""
    error_type: str
    error_message: str
    root_cause: str
    affected_field: Optional[str] = None
    affected_step: Optional[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.suggestions is None:
            self.suggestions = []


# =============================================================================
# 错误类型定义
# =============================================================================

ERROR_TYPES = {
    "missing_reference": "引用了不存在的步骤",
    "wrong_reference_format": "引用格式错误",
    "missing_required_field": "缺少必填字段",
    "invalid_field_value": "字段值无效",
    "execution_failed": "步骤执行失败",
    "data_not_found": "数据未找到",
    "connection_failed": "数据库/API 连接失败",
    "invalid_config": "配置格式错误",
    "circular_dependency": "循环依赖",
    "unknown_step_kind": "未知的步骤类型",
}


# =============================================================================
# 恢复提示模板
# =============================================================================

RECOVERY_PROMPTS = {
    # -------------------------------------------------------------------------
    # 引用相关错误
    # -------------------------------------------------------------------------
    "missing_reference": """
配置中引用了不存在的步骤。

**错误详情**:
- 引用的步骤: {referenced_step}
- 已存在的步骤: {existing_steps}

**修复方法**:
1. 检查引用的 step_id 是否正确
2. 确保被引用的步骤已经添加到流水线
3. 确保被引用的步骤已执行（引用只能指向已执行的步骤）

**示例**:
```yaml
# 错误：引用了不存在的步骤
config:
  data: $nonexistent_step['data']

# 正确：引用已存在的步骤
config:
  data: $price_bars['data']  # price_bars 已添加并执行
```

**下一步操作**:
- 使用 `get_pipeline()` 查看当前流水线中的所有步骤
- 使用 `get_details('{step_kind}')` 查看正确的引用格式
""",

    "wrong_reference_format": """
引用格式错误。正确的引用格式为 `$step_id['field']`。

**错误详情**:
- 错误的引用: {wrong_reference}
- 预期格式: $step_id['field']

**常见错误格式**:
1. ❌ 缺少 `$` 前缀: `step_id['field']`
2. ❌ 缺少访问路径: `$step_id`
3. ❌ 使用点号而非括号: `$step_id.field`
4. ❌ 引号类型错误: `$step_id["field"]`（虽然合法，但建议统一使用单引号）

**正确格式示例**:
```yaml
# 引用步骤的输出字段
data: $price_bars['data']

# 引用触发器的参数
target: $trigger['target']
start_date: $trigger['start_date']

# 引用因子的输出
factors: $technical['factors']
```

**修复方法**:
使用 `update_step` 修正配置中的引用格式。

**下一步操作**:
- 使用 `get_details('{step_kind}')` 查看 output_shape 字段，了解正确的访问路径
""",

    # -------------------------------------------------------------------------
    # 配置相关错误
    # -------------------------------------------------------------------------
    "missing_required_field": """
配置缺少必填字段。

**错误详情**:
- 步骤类型: {step_kind}
- 缺少的字段: {missing_fields}
- 已提供的字段: {provided_fields}

**修复方法**:
使用 `update_step` 添加缺少的字段。

**示例**:
```yaml
# 错误：缺少必填字段
config:
  frequency: "daily"

# 正确：补充必填字段
config:
  target: "$trigger['target']"
  start_date: "$trigger['start_date']"
  end_date: "$trigger['end_date']"
  frequency: "daily"
```

**下一步操作**:
- 使用 `get_details('{step_kind}')` 查看 required_fields 字段
- 参考示例配置（example_config 字段）
""",

    "invalid_field_value": """
字段值无效。

**错误详情**:
- 步骤类型: {step_kind}
- 字段名: {field_name}
- 当前值: {current_value}
- 错误原因: {error_reason}

**常见错误**:
1. 数据类型错误（期望 int，实际 str）
2. 值范围错误（如 ic_threshold 应为 0-1）
3. 格式错误（如日期格式）
4. 枚举值错误（如 method 应为 'spearman' 或 'pearson'）

**修复方法**:
使用 `update_step` 修正字段值。

**下一步操作**:
- 使用 `get_details('{step_kind}')` 查看 field_descriptions 字段
""",

    # -------------------------------------------------------------------------
    # 执行相关错误
    # -------------------------------------------------------------------------
    "execution_failed": """
步骤执行失败。

**错误详情**:
- 步骤类型: {step_kind}
- 步骤 ID: {step_id}
- 错误信息: {error_message}

**可能原因**:
1. 数据源不可用（数据库连接失败、API 超时）
2. 参数范围错误（日期范围、窗口大小）
3. 数据质量问题（缺失值、异常值）
4. 依赖步骤未正确执行

**诊断步骤**:
1. 检查数据源连接状态
2. 验证参数范围是否合理
3. 检查依赖步骤的输出

**修复方法**:
根据错误类型调整配置后，使用 `update_step` 更新并重试。

**常见修复示例**:
```yaml
# 数据获取失败 -> 检查日期范围
config:
  start_date: "2023-01-01"  # 确保数据源支持此日期
  end_date: "2024-12-31"

# 指标计算失败 -> 调整窗口大小
config:
  windows:
    momentum: [5, 10, 20]  # 确保有足够的数据
  lookback_days: 60  # 增加回溯天数
```
""",

    "data_not_found": """
数据未找到。

**错误详情**:
- 数据类型: {data_type}
- 查询条件: {query_params}

**可能原因**:
1. 日期范围内没有交易数据（节假日、非交易日）
2. 品种代码错误
3. 数据库中不存在该品种的数据
4. 数据源配置错误

**修复方法**:
1. 验证品种代码是否正确（如 RB、HC、I 等）
2. 检查日期范围是否合理
3. 尝试扩大日期范围
4. 使用 fallback 数据源（如 akshare）

**示例**:
```yaml
# 增加回溯天数
config:
  lookback_days: 120  # 从 60 增加到 120

# 使用 fallback 数据源
config:
  source: "fallback"  # 使用 akshare 作为备用数据源
```
""",

    # -------------------------------------------------------------------------
    # 通用错误
    # -------------------------------------------------------------------------
    "unknown_step_kind": """
未知的步骤类型。

**错误详情**:
- 指定的类型: {unknown_kind}
- 可用的类型: {valid_kinds}

**修复方法**:
使用正确的步骤类型。

**下一步操作**:
- 使用 `get_catalog()` 查看所有可用的步骤类型
""",

    "invalid_config": """
配置格式错误。

**错误详情**:
- 步骤类型: {step_kind}
- 错误信息: {error_message}

**常见错误**:
1. JSON/YAML 格式错误
2. 字段名拼写错误
3. 数据类型不匹配

**修复方法**:
检查配置格式，确保符合步骤的 schema。

**下一步操作**:
- 使用 `get_details('{step_kind}')` 查看正确的配置格式
- 参考示例配置（example_config 字段）
""",

    # -------------------------------------------------------------------------
    # 通用恢复提示
    # -------------------------------------------------------------------------
    "stuck": """
你似乎陷入了循环，多次尝试仍未成功。

**建议重置策略**:
1. 调用 `get_catalog()` 重新了解可用步骤
2. 按标准流程逐步构建流水线：
   - trigger.manual → data.price_bars → factor.technical → evaluation.ic
3. 每添加一个步骤后立即检查结果
4. 遇到错误时，仔细阅读错误信息并修复

**标准流程**:
```
1. trigger.manual    - 定义研究目标（品种、日期）
2. data.price_bars   - 获取价格数据
3. factor.technical  - 计算技术因子
4. evaluation.ic     - 评估因子IC
5. fusion.icir_weight - 合成因子
6. backtest.factor_signal - 回测验证
7. output.report     - 生成报告
```

**下一步操作**:
- 调用 `get_catalog()` 查看所有可用步骤
- 调用 `get_details(kind)` 查看特定步骤的配置格式
""",

    "no_progress": """
多次操作后流水线没有实质性进展。

**当前状态**:
- 已添加步骤: {step_count} 个
- 步骤类型: {step_kinds}

**建议**:
1. 聚焦目标：完成一个完整的因子研究流程
2. 不要纠结于细节，先完成主流程
3. 使用标准流程模板快速构建

**下一步操作**:
- 如果已完成主要步骤，调用 `get_pipeline()` 导出流水线
- 如果缺少关键步骤，按照推荐顺序添加
""",

    "empty_response": """
你的最后一次回复没有包含工具调用。

**重要提示**:
你必须使用工具来构建流水线，而不是直接回复文本。

**可用工具**:
- `add_step`: 添加步骤到流水线
- `update_step`: 更新步骤配置
- `connect_steps`: 连接步骤
- `get_catalog`: 获取步骤目录
- `get_details`: 获取步骤详细信息
- `get_pipeline`: 导出完整流水线

**下一步操作**:
- 调用 `get_catalog()` 了解可用步骤
- 调用 `add_step()` 开始构建流水线
""",
}


# =============================================================================
# 诊断函数
# =============================================================================

def diagnose_error(
    error_message: str,
    step_kind: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> ErrorDiagnosis:
    """
    诊断错误类型和根本原因。
    
    Args:
        error_message: 错误信息
        step_kind: 步骤类型
        config: 步骤配置
        context: 执行上下文
    
    Returns:
        ErrorDiagnosis
    """
    error_lower = error_message.lower()
    config = config or {}
    context = context or {}
    
    # 1. 检查引用错误
    if "not found" in error_lower and "$" in error_message:
        # 提取引用的步骤 ID
        ref_match = re.search(r"\$([\w]+)", error_message)
        referenced_step = ref_match.group(1) if ref_match else "unknown"
        
        return ErrorDiagnosis(
            error_type="missing_reference",
            error_message=error_message,
            root_caure=f"引用了不存在的步骤: {referenced_step}",
            affected_step=referenced_step,
            suggestions=[
                f"检查步骤 '{referenced_step}' 是否已添加到流水线",
                "使用 get_pipeline() 查看当前步骤",
            ],
        )
    
    # 2. 检查引用格式错误
    if "reference" in error_lower or "invalid" in error_lower:
        for field, value in config.items():
            if isinstance(value, str) and value.startswith("$"):
                # 检查是否缺少访问路径
                if not ("[" in value and "]" in value):
                    return ErrorDiagnosis(
                        error_type="wrong_reference_format",
                        error_message=error_message,
                        root_cause=f"引用 '{value}' 缺少访问路径",
                        affected_field=field,
                        suggestions=[
                            f"修正引用格式: {value} -> {value}['field']",
                            "使用 get_details(kind) 查看正确的访问路径",
                        ],
                    )
    
    # 3. 检查缺少必填字段
    if "missing" in error_lower or "required" in error_lower:
        field_match = re.search(r"field[:\s]+['\"]?(\w+)['\"]?", error_message, re.I)
        missing_field = field_match.group(1) if field_match else "unknown"
        
        return ErrorDiagnosis(
            error_type="missing_required_field",
            error_message=error_message,
            root_cause=f"缺少必填字段: {missing_field}",
            affected_field=missing_field,
            suggestions=[
                f"添加字段 '{missing_field}' 到配置",
                f"使用 get_details('{step_kind}') 查看必填字段列表",
            ],
        )
    
    # 4. 检查数据未找到
    if "no data" in error_lower or "data not found" in error_lower:
        return ErrorDiagnosis(
            error_type="data_not_found",
            error_message=error_message,
            root_cause="数据源未返回数据",
            suggestions=[
                "检查日期范围是否合理",
                "验证品种代码是否正确",
                "尝试使用 fallback 数据源",
            ],
        )
    
    # 5. 检查未知步骤类型
    if "unknown" in error_lower and "kind" in error_lower:
        return ErrorDiagnosis(
            error_type="unknown_step_kind",
            error_message=error_message,
            root_cause="指定的步骤类型不存在",
            suggestions=[
                "使用 get_catalog() 查看所有可用的步骤类型",
            ],
        )
    
    # 6. 默认：执行失败
    return ErrorDiagnosis(
        error_type="execution_failed",
        error_message=error_message,
        root_cause="步骤执行过程中发生错误",
        suggestions=[
            "检查错误信息中的具体原因",
            "验证配置参数是否合理",
            "确保依赖步骤已正确执行",
        ],
    )


def get_recovery_prompt(
    error_type: str,
    diagnosis: Optional[ErrorDiagnosis] = None,
) -> str:
    """
    获取恢复提示。
    
    Args:
        error_type: 错误类型
        diagnosis: 错误诊断结果
    
    Returns:
        恢复提示文本
    """
    template = RECOVERY_PROMPTS.get(error_type, RECOVERY_PROMPTS["stuck"])
    
    if diagnosis:
        # 替换模板中的占位符
        try:
            return template.format(
                step_kind=diagnosis.affected_step or "unknown",
                step_id=diagnosis.affected_step or "unknown",
                error_message=diagnosis.error_message,
                referenced_step=diagnosis.affected_step or "unknown",
                missing_fields=diagnosis.affected_field or "unknown",
                wrong_reference=diagnosis.error_message,
                **diagnosis.__dict__,
            )
        except (KeyError, AttributeError):
            return template
    
    return template


def suggest_fix(
    diagnosis: ErrorDiagnosis,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    生成修复建议。
    
    Args:
        diagnosis: 错误诊断结果
        context: 执行上下文
    
    Returns:
        包含修复建议的字典
    """
    fix = {
        "error_type": diagnosis.error_type,
        "root_cause": diagnosis.root_cause,
        "suggestions": diagnosis.suggestions,
        "action": None,
        "updated_config": None,
    }
    
    if diagnosis.error_type == "missing_reference":
        fix["action"] = "update_step"
        fix["suggestions"].append(
            f"修正引用：将 ${diagnosis.affected_step}['field'] 改为正确的步骤 ID"
        )
    
    elif diagnosis.error_type == "wrong_reference_format":
        fix["action"] = "update_step"
        if diagnosis.affected_field:
            fix["suggestions"].append(
                f"更新字段 '{diagnosis.affected_field}' 的引用格式"
            )
    
    elif diagnosis.error_type == "missing_required_field":
        fix["action"] = "update_step"
        if diagnosis.affected_field:
            fix["suggestions"].append(
                f"添加字段: {diagnosis.affected_field}: <value>"
            )
    
    return fix


# =============================================================================
# 向后兼容别名
# =============================================================================

# 保留原有的 RECOVERY_PROMPTS 键名，供 EnhancedReActAgent 使用
RECOVERY_PROMPTS_V1 = {
    "empty_response": RECOVERY_PROMPTS["empty_response"],
    "tool_error": RECOVERY_PROMPTS["execution_failed"],
    "stuck": RECOVERY_PROMPTS["stuck"],
    "no_progress": RECOVERY_PROMPTS["no_progress"],
}
