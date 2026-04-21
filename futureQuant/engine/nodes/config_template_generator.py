"""
配置模板生成器 (Config Template Generator)

根据步骤类型和当前上下文自动生成配置模板，
降低 LLM 手动编写配置的错误率。

核心功能：
1. 自动填充引用字段（如 data: $price_bars['data']）
2. 填充默认值（从 optional_fields）
3. 根据上下文推断推荐值

使用示例：
    from futureQuant.engine.nodes.config_template_generator import ConfigTemplateGenerator
    
    generator = ConfigTemplateGenerator()
    
    # 生成模板
    template = generator.generate_template(
        kind="factor.technical",
        context={"steps": [{"id": "price_bars", "kind": "data.price_bars"}]}
    )
    
    # 验证并修复配置
    fixed_config = generator.validate_and_fix(
        kind="factor.technical",
        config={"data": "$price_bars"},  # 错误：缺少 ['data']
        context=context
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from .factor_catalog import get_details, get_catalog


@dataclass
class TemplateResult:
    """模板生成结果。"""
    config: Dict[str, Any]
    warnings: List[str]
    auto_filled_fields: List[str]


class ConfigTemplateGenerator:
    """
    配置模板生成器。
    
    根据步骤类型和当前上下文自动生成配置模板，
    减少 LLM 手动编写配置时的错误。
    
    功能：
    1. 自动填充引用字段（根据上下文推断）
    2. 填充默认值（从 optional_fields）
    3. 验证配置格式
    4. 修复常见错误
    
    使用示例：
        generator = ConfigTemplateGenerator()
        
        # 方式1：生成空模板
        template = generator.generate_empty_template("data.price_bars")
        
        # 方式2：生成智能模板（根据上下文）
        template = generator.generate_template(
            kind="factor.technical",
            context={"steps": [...], "trigger_id": "trigger"}
        )
        
        # 方式3：验证并修复配置
        fixed = generator.validate_and_fix(
            kind="factor.technical",
            config={"data": "$price_bars"},  # 错误的引用
            context=context
        )
    """
    
    # 引用字段推断规则
    REFERENCE_RULES = {
        # 字段名 -> 依赖的步骤类型
        "target": "trigger.manual",
        "start_date": "trigger.manual",
        "end_date": "trigger.manual",
        "frequency": "trigger.manual",
        "universe": "trigger.manual",
        "data": "data.price_bars",
        "price_data": "data.price_bars",
        "fundamental_data": "data.fundamental",
        "factors": "factor.*",
        "ic_series": "evaluation.ic",
        "passed_factors": "evaluation.ic",
        "composite_factor": "fusion.*",
    }
    
    # 字段访问路径规则
    ACCESSOR_RULES = {
        "data.price_bars": {
            "data": "data",
            "df": "data",
        },
        "data.fundamental": {
            "data": "data",
        },
        "factor.technical": {
            "factors": "factors",
            "data": "factors",
        },
        "factor.alpha101": {
            "data": "data",
            "factors": "data",
        },
        "factor.fundamental": {
            "factors": "factors",
        },
        "evaluation.ic": {
            "ic_series": "ic_series",
            "icir_dict": "icir_dict",
            "passed_factors": "passed_factors",
        },
        "fusion.icir_weight": {
            "composite_factor": "composite_factor",
            "weights": "weights",
        },
    }
    
    def generate_empty_template(
        self,
        kind: str,
    ) -> Dict[str, Any]:
        """
        生成空配置模板（仅包含默认值）。
        
        Args:
            kind: 步骤类型
        
        Returns:
            配置模板字典
        """
        details = get_details(kind)
        if "error" in details:
            return {}
        
        template = {}
        
        # 填充可选字段的默认值
        optional_fields = details.get("optional_fields", {})
        for field, default_value in optional_fields.items():
            template[field] = default_value
        
        return template
    
    def generate_template(
        self,
        kind: str,
        context: Optional[Dict[str, Any]] = None,
        fill_required: bool = True,
        fill_optional: bool = True,
    ) -> TemplateResult:
        """
        生成智能配置模板。
        
        根据上下文自动推断引用字段的值，
        减少手动配置错误。
        
        Args:
            kind: 步骤类型
            context: 当前上下文，包含：
                - steps: 已存在的步骤列表
                - trigger_id: 触发器步骤 ID
                - latest_data_id: 最新数据步骤 ID
                - latest_factor_id: 最新因子步骤 ID
            fill_required: 是否填充必填字段
            fill_optional: 是否填充可选字段
        
        Returns:
            TemplateResult 包含配置、警告、自动填充字段列表
        """
        details = get_details(kind)
        if "error" in details:
            return TemplateResult(
                config={},
                warnings=[f"Unknown step kind: {kind}"],
                auto_filled_fields=[],
            )
        
        context = context or {}
        template = {}
        warnings = []
        auto_filled = []
        
        # 填充必填字段
        if fill_required:
            for field in details.get("required_fields", []):
                value = self._infer_reference(field, context)
                if value is not None:
                    template[field] = value
                    auto_filled.append(field)
                else:
                    warnings.append(
                        f"无法自动推断必填字段 '{field}'，请手动配置。"
                    )
        
        # 填充可选字段
        if fill_optional:
            optional_fields = details.get("optional_fields", {})
            for field, default_value in optional_fields.items():
                if field not in template:
                    template[field] = default_value
                    auto_filled.append(field)
        
        return TemplateResult(
            config=template,
            warnings=warnings,
            auto_filled_fields=auto_filled,
        )
    
    def validate_and_fix(
        self,
        kind: str,
        config: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> TemplateResult:
        """
        验证配置并修复常见错误。
        
        Args:
            kind: 步骤类型
            config: 待验证的配置
            context: 当前上下文
        
        Returns:
            TemplateResult 包含修复后的配置和警告
        """
        details = get_details(kind)
        if "error" in details:
            return TemplateResult(
                config=config,
                warnings=[f"Unknown step kind: {kind}"],
                auto_filled_fields=[],
            )
        
        fixed_config = dict(config)
        warnings = []
        auto_filled = []
        
        # 检查必填字段
        for field in details.get("required_fields", []):
            if field not in fixed_config:
                value = self._infer_reference(field, context)
                if value is not None:
                    fixed_config[field] = value
                    auto_filled.append(field)
                    warnings.append(f"已自动填充缺失的必填字段: {field}")
        
        # 修复引用格式错误
        for field, value in fixed_config.items():
            if isinstance(value, str) and value.startswith("$"):
                fixed_value, fixed = self._fix_reference(value, context)
                if fixed:
                    fixed_config[field] = fixed_value
                    warnings.append(f"已修复字段 '{field}' 的引用格式: {value} -> {fixed_value}")
        
        return TemplateResult(
            config=fixed_config,
            warnings=warnings,
            auto_filled_fields=auto_filled,
        )
    
    def suggest_config_for_step(
        self,
        kind: str,
        existing_steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        根据已存在的步骤推荐配置。
        
        Args:
            kind: 要添加的步骤类型
            existing_steps: 已存在的步骤列表
        
        Returns:
            推荐的配置字典
        """
        context = self._build_context_from_steps(existing_steps)
        result = self.generate_template(kind, context)
        return result.config
    
    # -------------------------------------------------------------------------
    # 私有方法
    # -------------------------------------------------------------------------
    
    def _infer_reference(
        self,
        field: str,
        context: Dict[str, Any],
    ) -> Optional[str]:
        """
        推断字段的引用值。
        
        Args:
            field: 字段名
            context: 上下文
        
        Returns:
            推断的引用字符串（如 "$trigger['target']"）或 None
        """
        # 规则1：触发器字段
        if field in ["target", "start_date", "end_date", "frequency", "universe"]:
            trigger_id = context.get("trigger_id") or self._find_step_by_kind(
                context.get("steps", []), "trigger.manual"
            )
            if trigger_id:
                return f"${trigger_id}['{field}']"
        
        # 规则2：数据字段
        if field in ["data", "price_data"]:
            data_id = context.get("latest_data_id") or self._find_latest_data_step(
                context.get("steps", [])
            )
            if data_id:
                accessor = self._get_accessor(data_id, "data", context)
                return f"${data_id}['{accessor}']"
        
        # 规则3：基本面数据字段
        if field == "fundamental_data":
            fund_id = self._find_step_by_kind(
                context.get("steps", []), "data.fundamental"
            )
            if fund_id:
                return f"${fund_id}['data']"
        
        # 规则4：因子字段
        if field == "factors":
            factor_id = context.get("latest_factor_id") or self._find_latest_factor_step(
                context.get("steps", [])
            )
            if factor_id:
                accessor = self._get_accessor(factor_id, "factors", context)
                return f"${factor_id}['{accessor}']"
        
        # 规则5：IC 评估结果字段
        if field in ["ic_series", "passed_factors"]:
            eval_id = self._find_step_by_kind(
                context.get("steps", []), "evaluation.ic"
            )
            if eval_id:
                return f"${eval_id}['{field}']"
        
        # 规则6：合成因子字段
        if field in ["composite_factor", "factor"]:
            fusion_id = self._find_latest_fusion_step(
                context.get("steps", [])
            )
            if fusion_id:
                return f"${fusion_id}['composite_factor']"
        
        return None
    
    def _fix_reference(
        self,
        reference: str,
        context: Dict[str, Any],
    ) -> tuple[str, bool]:
        """
        修复引用格式错误。
        
        Args:
            reference: 原始引用
            context: 上下文
        
        Returns:
            (修复后的引用, 是否修复)
        """
        # 错误1：缺少访问路径（$step_id 而非 $step_id['field']）
        match = re.match(r"^\$([\w]+)$", reference)
        if match:
            step_id = match.group(1)
            
            # 推断访问路径
            steps = context.get("steps", [])
            step = next((s for s in steps if s.get("id") == step_id), None)
            if step:
                kind = step.get("kind", "")
                accessor = self._get_accessor(step_id, "default", context, kind)
                if accessor:
                    return f"${step_id}['{accessor}']", True
        
        # 错误2：访问路径不存在
        match = re.match(r"^\$([\w]+)\[(['\"])(.*?)\2\]$", reference)
        if match:
            step_id, _, field = match.groups()
            # 验证字段是否存在（需要执行结果，此处简化）
            # 如果上下文中有输出，可以验证
        
        return reference, False
    
    def _get_accessor(
        self,
        step_id: str,
        field_hint: str,
        context: Dict[str, Any],
        kind_hint: Optional[str] = None,
    ) -> str:
        """
        获取步骤的访问路径。
        
        Args:
            step_id: 步骤 ID
            field_hint: 字段提示（如 "data", "factors"）
            context: 上下文
            kind_hint: 步骤类型提示
        
        Returns:
            访问路径字段名
        """
        # 从上下文获取步骤类型
        if kind_hint is None:
            steps = context.get("steps", [])
            step = next((s for s in steps if s.get("id") == step_id), None)
            if step:
                kind_hint = step.get("kind", "")
        
        # 查找访问规则
        if kind_hint:
            # 支持通配符匹配（如 factor.*）
            for pattern, accessors in self.ACCESSOR_RULES.items():
                if pattern.endswith(".*"):
                    prefix = pattern[:-1]  # factor.
                    if kind_hint.startswith(prefix):
                        return accessors.get(field_hint, field_hint)
                elif kind_hint == pattern:
                    return accessors.get(field_hint, field_hint)
        
        # 默认返回字段提示
        return field_hint
    
    def _build_context_from_steps(
        self,
        steps: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """从步骤列表构建上下文。"""
        context = {"steps": steps}
        
        # 查找关键步骤
        context["trigger_id"] = self._find_step_by_kind(steps, "trigger.manual")
        context["latest_data_id"] = self._find_latest_data_step(steps)
        context["latest_factor_id"] = self._find_latest_factor_step(steps)
        
        return context
    
    def _find_step_by_kind(self, steps: List[Dict], kind: str) -> Optional[str]:
        """查找指定类型的步骤 ID。"""
        for step in steps:
            if step.get("kind") == kind:
                return step.get("id")
        return None
    
    def _find_latest_data_step(self, steps: List[Dict]) -> Optional[str]:
        """查找最新的数据步骤 ID。"""
        for step in reversed(steps):
            if step.get("kind", "").startswith("data."):
                return step.get("id")
        return None
    
    def _find_latest_factor_step(self, steps: List[Dict]) -> Optional[str]:
        """查找最新的因子步骤 ID。"""
        for step in reversed(steps):
            if step.get("kind", "").startswith("factor."):
                return step.get("id")
        return None
    
    def _find_latest_fusion_step(self, steps: List[Dict]) -> Optional[str]:
        """查找最新的融合步骤 ID。"""
        for step in reversed(steps):
            if step.get("kind", "").startswith("fusion."):
                return step.get("id")
        return None
