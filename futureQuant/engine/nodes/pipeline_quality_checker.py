"""
流水线质量检查器 (Pipeline Quality Checker)

在 `get_pipeline` 时检查流水线质量，提供评分和改进建议。

核心功能：
1. 完整性检查（是否包含必要步骤）
2. 连通性检查（步骤是否正确连接）
3. 配置正确性检查（引用是否有效）
4. 最佳实践检查（是否符合因子研究最佳实践）

使用示例：
    from futureQuant.engine.nodes.pipeline_quality_checker import PipelineQualityChecker
    
    checker = PipelineQualityChecker()
    pipeline = builder.get_pipeline()
    
    # 检查质量
    result = checker.check_quality(pipeline)
    
    print(f"质量分数: {result.score}/100")
    print(f"等级: {result.grade}")
    print(f"问题: {result.issues}")
    print(f"建议: {result.recommendations}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class QualityCheckResult:
    """质量检查结果。"""
    score: int  # 0-100
    max_score: int = 100
    grade: str = ""
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    breakdown: Dict[str, int] = field(default_factory=dict)
    
    def is_passing(self) -> bool:
        """是否通过检查（分数 >= 60）。"""
        return self.score >= 60
    
    def is_good(self) -> bool:
        """是否良好（分数 >= 80）。"""
        return self.score >= 80
    
    def is_excellent(self) -> bool:
        """是否优秀（分数 >= 90）。"""
        return self.score >= 90


class PipelineQualityChecker:
    """
    流水线质量检查器。
    
    检查维度：
    1. 完整性（40分）：是否包含必要步骤
    2. 连通性（30分）：步骤是否正确连接
    3. 配置正确性（20分）：引用是否有效
    4. 最佳实践（10分）：是否符合因子研究最佳实践
    
    评分标准：
    - 90-100: 优秀 (Excellent)
    - 80-89: 良好 (Good)
    - 70-79: 合格 (Pass)
    - 60-69: 待改进 (Needs Improvement)
    - 0-59: 不完整 (Incomplete)
    
    使用示例：
        checker = PipelineQualityChecker()
        
        # 方式1：完整检查
        result = checker.check_quality(pipeline)
        
        # 方式2：快速检查
        quick_result = checker.quick_check(pipeline)
        
        # 方式3：检查特定维度
        completeness = checker.check_completeness(pipeline)
        connectivity = checker.check_connectivity(pipeline)
    """
    
    # 必需步骤（40分）
    REQUIRED_STEPS = [
        ("trigger.manual", 10, "流水线起点"),
        ("data.price_bars", 10, "价格数据"),
        ("factor.technical", 10, "因子计算"),
        ("evaluation.ic", 10, "因子评估"),
    ]
    
    # 推荐步骤（10分）
    RECOMMENDED_STEPS = [
        ("fusion.icir_weight", 3, "因子合成"),
        ("backtest.factor_signal", 4, "回测验证"),
        ("output.report", 3, "报告生成"),
    ]
    
    def check_quality(
        self,
        pipeline: Dict[str, Any],
    ) -> QualityCheckResult:
        """
        执行完整质量检查。
        
        Args:
            pipeline: 流水线定义
        
        Returns:
            QualityCheckResult
        """
        result = QualityCheckResult(score=0)
        
        # 1. 完整性检查（40分）
        completeness_score, completeness_issues = self._check_completeness(pipeline)
        result.score += completeness_score
        result.breakdown["completeness"] = completeness_score
        result.issues.extend(completeness_issues)
        
        # 2. 连通性检查（30分）
        connectivity_score, connectivity_issues = self._check_connectivity(pipeline)
        result.score += connectivity_score
        result.breakdown["connectivity"] = connectivity_score
        result.issues.extend(connectivity_issues)
        
        # 3. 配置正确性检查（20分）
        config_score, config_issues = self._check_config_validity(pipeline)
        result.score += config_score
        result.breakdown["config_validity"] = config_score
        result.issues.extend(config_issues)
        
        # 4. 最佳实践检查（10分）
        best_practice_score, best_practice_issues = self._check_best_practices(pipeline)
        result.score += best_practice_score
        result.breakdown["best_practices"] = best_practice_score
        result.issues.extend(best_practice_issues)
        
        # 评级
        result.grade = self._grade_score(result.score)
        
        # 生成建议
        result.recommendations = self._generate_recommendations(pipeline, result)
        
        return result
    
    def quick_check(
        self,
        pipeline: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        快速检查（仅检查完整性和连通性）。
        
        Args:
            pipeline: 流水线定义
        
        Returns:
            简化的检查结果字典
        """
        steps = pipeline.get("steps", [])
        step_kinds = {s.get("kind") for s in steps}
        
        # 检查必需步骤
        missing_required = [
            kind for kind, _, _ in self.REQUIRED_STEPS
            if kind not in step_kinds
        ]
        
        # 检查连通性
        connected = sum(1 for s in steps if s.get("next"))
        has_connections = connected > 0
        
        return {
            "is_valid": len(missing_required) == 0 and has_connections,
            "missing_required": missing_required,
            "has_connections": has_connections,
            "step_count": len(steps),
        }
    
    def check_completeness(
        self,
        pipeline: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        检查完整性。
        
        Args:
            pipeline: 流水线定义
        
        Returns:
            完整性检查结果
        """
        score, issues = self._check_completeness(pipeline)
        
        steps = pipeline.get("steps", [])
        step_kinds = {s.get("kind") for s in steps}
        
        missing_required = [
            kind for kind, _, _ in self.REQUIRED_STEPS
            if kind not in step_kinds
        ]
        
        missing_recommended = [
            kind for kind, _, _ in self.RECOMMENDED_STEPS
            if kind not in step_kinds
        ]
        
        return {
            "score": score,
            "max_score": 40,
            "missing_required": missing_required,
            "missing_recommended": missing_recommended,
            "issues": issues,
            "is_complete": len(missing_required) == 0,
        }
    
    def check_connectivity(
        self,
        pipeline: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        检查连通性。
        
        Args:
            pipeline: 流水线定义
        
        Returns:
            连通性检查结果
        """
        score, issues = self._check_connectivity(pipeline)
        
        steps = pipeline.get("steps", [])
        step_ids = {s.get("id") for s in steps}
        
        # 构建图
        edges: Dict[str, Set[str]] = {s.get("id"): set() for s in steps}
        for s in steps:
            for target in s.get("next", []):
                edges[s.get("id")].add(target)
        
        # 检查孤立节点
        isolated = []
        for sid in step_ids:
            has_incoming = any(sid in edges[other] for other in edges if other != sid)
            has_outgoing = len(edges[sid]) > 0
            if not has_incoming and not has_outgoing:
                isolated.append(sid)
        
        # 检查链式连接
        chain_count = sum(1 for s in steps if len(s.get("next", [])) > 0)
        
        return {
            "score": score,
            "max_score": 30,
            "issues": issues,
            "isolated_steps": isolated,
            "chain_count": chain_count,
            "total_steps": len(steps),
        }
    
    # -------------------------------------------------------------------------
    # 私有方法
    # -------------------------------------------------------------------------
    
    def _check_completeness(
        self,
        pipeline: Dict[str, Any],
    ) -> tuple[int, List[str]]:
        """检查完整性。"""
        steps = pipeline.get("steps", [])
        step_kinds = {s.get("kind") for s in steps}
        
        score = 0
        issues = []
        
        for kind, points, description in self.REQUIRED_STEPS:
            if kind in step_kinds:
                score += points
            else:
                issues.append(f"缺少必需步骤: {kind} ({description})")
        
        return score, issues
    
    def _check_connectivity(
        self,
        pipeline: Dict[str, Any],
    ) -> tuple[int, List[str]]:
        """检查连通性。"""
        steps = pipeline.get("steps", [])
        
        if not steps:
            return 0, ["流水线为空"]
        
        score = 0
        issues = []
        
        # 检查1：是否有连接（15分）
        connected = sum(1 for s in steps if s.get("next"))
        if connected == 0 and len(steps) > 1:
            issues.append("步骤之间没有连接，使用 connect_steps 定义执行顺序")
        elif connected > 0:
            score += 15
        
        # 检查2：链式连接（15分）
        if len(steps) > 1:
            if connected >= len(steps) - 1:
                score += 15
            elif connected > 0:
                score += 10
                issues.append("部分步骤未连接到主链")
        
        return score, issues
    
    def _check_config_validity(
        self,
        pipeline: Dict[str, Any],
    ) -> tuple[int, List[str]]:
        """检查配置正确性。"""
        steps = pipeline.get("steps", [])
        
        if not steps:
            return 20, []  # 空流水线视为有效
        
        score = 20
        issues = []
        
        # 检查引用格式
        for step in steps:
            config = step.get("config", {})
            for field, value in config.items():
                if isinstance(value, str) and value.startswith("$"):
                    # 简单检查：是否有访问路径
                    if not ("[" in value and "]" in value):
                        # 可能缺少访问路径
                        issues.append(
                            f"步骤 '{step.get('id')}' 的配置字段 '{field}' "
                            f"引用可能缺少访问路径: {value}。"
                            f"正确格式: $step_id['field']"
                        )
        
        # 有问题则扣分
        if issues:
            score = max(10, score - len(issues) * 2)
        
        return score, issues
    
    def _check_best_practices(
        self,
        pipeline: Dict[str, Any],
    ) -> tuple[int, List[str]]:
        """检查最佳实践。"""
        steps = pipeline.get("steps", [])
        step_kinds = {s.get("kind") for s in steps}
        
        score = 0
        issues = []
        
        # 检查1：是否有回测（4分）
        if any(k.startswith("backtest.") for k in step_kinds):
            score += 4
        else:
            issues.append("建议添加回测步骤验证因子有效性")
        
        # 检查2：是否有报告（3分）
        if "output.report" in step_kinds:
            score += 3
        else:
            issues.append("建议添加报告生成步骤")
        
        # 检查3：是否有因子合成（3分）
        if any(k.startswith("fusion.") for k in step_kinds):
            score += 3
        else:
            issues.append("建议添加因子合成步骤以提升稳健性")
        
        return score, issues
    
    def _generate_recommendations(
        self,
        pipeline: Dict[str, Any],
        result: QualityCheckResult,
    ) -> List[str]:
        """生成改进建议。"""
        recommendations = []
        
        steps = pipeline.get("steps", [])
        step_kinds = {s.get("kind") for s in steps}
        
        # 根据问题生成建议
        if result.breakdown.get("completeness", 0) < 40:
            recommendations.append(
                "流水线不完整。建议添加缺失的必需步骤。"
            )
        
        if result.breakdown.get("connectivity", 0) < 30:
            recommendations.append(
                "步骤之间连接不完整。使用 connect_steps() 定义执行顺序。"
            )
        
        if result.breakdown.get("config_validity", 0) < 20:
            recommendations.append(
                "配置存在潜在问题。检查引用格式是否正确（$step_id['field']）。"
            )
        
        # 根据缺失步骤生成建议
        if "fusion.icir_weight" not in step_kinds:
            recommendations.append(
                "建议在 IC 评估后添加因子合成步骤（fusion.icir_weight），"
                "可以提升因子稳健性。"
            )
        
        if not any(k.startswith("backtest.") for k in step_kinds):
            recommendations.append(
                "建议在因子合成后添加回测步骤（backtest.factor_signal），"
                "验证因子的实际盈利能力。"
            )
        
        if "output.report" not in step_kinds:
            recommendations.append(
                "建议在流程末尾添加报告生成步骤（output.report），"
                "记录研究结果和结论。"
            )
        
        return recommendations
    
    def _grade_score(self, score: int) -> str:
        """根据分数评级。"""
        if score >= 90:
            return "优秀 (Excellent)"
        elif score >= 80:
            return "良好 (Good)"
        elif score >= 70:
            return "合格 (Pass)"
        elif score >= 60:
            return "待改进 (Needs Improvement)"
        else:
            return "不完整 (Incomplete)"


# =============================================================================
# 便捷函数
# =============================================================================

def check_pipeline_quality(pipeline: Dict[str, Any]) -> QualityCheckResult:
    """
    便捷函数：检查流水线质量。
    
    Args:
        pipeline: 流水线定义
    
    Returns:
        QualityCheckResult
    """
    checker = PipelineQualityChecker()
    return checker.check_quality(pipeline)


def quick_validate_pipeline(pipeline: Dict[str, Any]) -> bool:
    """
    便捷函数：快速验证流水线是否有效。
    
    Args:
        pipeline: 流水线定义
    
    Returns:
        是否有效
    """
    checker = PipelineQualityChecker()
    result = checker.quick_check(pipeline)
    return result["is_valid"]
