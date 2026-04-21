"""
智能推荐系统 (Smart Recommender)

根据当前流水线状态，推荐下一步操作，降低 LLM 规划难度。

核心功能：
1. 基于规则推荐下一步（标准因子研究流程）
2. 检查流水线完整性
3. 提供修复建议

使用示例：
    from futureQuant.engine.nodes.smart_recommender import SmartRecommender
    
    recommender = SmartRecommender()
    pipeline = builder.get_pipeline()
    recommendations = recommender.suggest_next_steps(pipeline)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Recommendation:
    """单条推荐。"""
    step_kind: str
    step_name: str
    reason: str
    priority: int  # 1-5, 5最高
    sample_config: Optional[Dict[str, Any]] = None


class SmartRecommender:
    """
    智能推荐系统。
    
    根据因子研究的标准流程和当前流水线状态，
    推荐下一步操作。
    
    标准流程：
    1. trigger.manual -> 定义研究目标
    2. data.price_bars -> 获取价格数据
    3. factor.technical / factor.alpha101 -> 计算因子
    4. evaluation.ic -> 评估因子
    5. fusion.icir_weight -> 合成因子
    6. backtest.factor_signal -> 回测验证
    7. output.report -> 生成报告
    
    使用示例：
        recommender = SmartRecommender()
        
        # 方式1：推荐下一步
        recommendations = recommender.suggest_next_steps(pipeline)
        
        # 方式2：检查完整性
        check_result = recommender.check_completeness(pipeline)
        
        # 方式3：生成完整流程建议
        full_flow = recommender.generate_full_flow(target="RB", start_date="2023-01-01")
    """
    
    # 标准流程定义
    STANDARD_FLOW = [
        ("trigger.manual", "流水线起点"),
        ("data.price_bars", "获取价格数据"),
        ("factor.technical", "计算技术因子"),
        ("evaluation.ic", "评估因子IC"),
        ("fusion.icir_weight", "合成因子"),
        ("backtest.factor_signal", "回测验证"),
        ("output.report", "生成报告"),
    ]
    
    def suggest_next_steps(
        self,
        pipeline: Dict[str, Any],
        max_recommendations: int = 5,
    ) -> List[Recommendation]:
        """
        根据当前流水线状态推荐下一步。
        
        Args:
            pipeline: 当前流水线定义
            max_recommendations: 最多返回多少条推荐
        
        Returns:
            推荐列表，按优先级排序
        """
        steps = pipeline.get("steps", [])
        step_kinds = {s.get("kind") for s in steps}
        step_ids = {s.get("id") for s in steps}
        
        recommendations: List[Recommendation] = []
        
        # 规则1：必须有 trigger
        if "trigger.manual" not in step_kinds:
            recommendations.append(Recommendation(
                step_kind="trigger.manual",
                step_name="流水线起点",
                reason="每个流水线都需要一个触发器来定义研究目标（品种、日期范围）。",
                priority=5,
                sample_config={
                    "target": "RB",
                    "start_date": "2023-01-01",
                    "end_date": "2024-12-31",
                    "frequency": "daily",
                },
            ))
        
        # 规则2：trigger 后需要 data
        if "trigger.manual" in step_kinds and "data.price_bars" not in step_kinds:
            trigger_id = self._find_step_by_kind(steps, "trigger.manual")
            recommendations.append(Recommendation(
                step_kind="data.price_bars",
                step_name="获取价格数据",
                reason="需要价格数据来计算因子。",
                priority=5,
                sample_config={
                    "target": f"${trigger_id}['target']",
                    "start_date": f"${trigger_id}['start_date']",
                    "end_date": f"${trigger_id}['end_date']",
                    "frequency": "daily",
                    "lookback_days": 60,
                },
            ))
        
        # 规则3：data 后需要 factor
        has_data = "data.price_bars" in step_kinds
        has_factors = any(k.startswith("factor.") for k in step_kinds)
        
        if has_data and not has_factors:
            data_id = self._find_step_by_kind(steps, "data.price_bars")
            
            recommendations.append(Recommendation(
                step_kind="factor.technical",
                step_name="计算技术因子",
                reason="技术因子是最基础的因子来源，包含动量、波动率等。",
                priority=5,
                sample_config={
                    "data": f"${data_id}['data']",
                    "indicators": ["momentum", "volatility", "rsi"],
                    "windows": {
                        "momentum": [5, 10, 20, 60],
                        "volatility": [10, 20, 60],
                        "rsi": [6, 14, 21],
                    },
                },
            ))
            
            recommendations.append(Recommendation(
                step_kind="factor.alpha101",
                step_name="计算Alpha101因子",
                reason="Alpha101 是业界经典因子库，可与技术因子互补。",
                priority=4,
                sample_config={
                    "data": f"${data_id}['data']",
                    "top_n": 20,
                },
            ))
        
        # 规则4：factor 后需要 evaluation
        if has_factors and "evaluation.ic" not in step_kinds:
            factor_id = self._find_latest_factor_step(steps)
            
            recommendations.append(Recommendation(
                step_kind="evaluation.ic",
                step_name="评估因子IC",
                reason="IC评估是筛选有效因子的关键步骤。",
                priority=5,
                sample_config={
                    "factors": f"${factor_id}['factors']",
                    "method": "spearman",
                    "ic_threshold": 0.02,
                    "icir_threshold": 0.3,
                },
            ))
        
        # 规则5：evaluation 后需要 fusion
        has_evaluation = "evaluation.ic" in step_kinds
        has_fusion = any(k.startswith("fusion.") for k in step_kinds)
        
        if has_evaluation and not has_fusion:
            eval_id = self._find_step_by_kind(steps, "evaluation.ic")
            factor_id = self._find_latest_factor_step(steps)
            
            recommendations.append(Recommendation(
                step_kind="fusion.icir_weight",
                step_name="ICIR加权合成",
                reason="因子合成可以在保持预测能力的同时降低噪音。",
                priority=5,
                sample_config={
                    "factors": f"${factor_id}['factors']",
                    "ic_series": f"${eval_id}['ic_series']",
                    "corr_threshold": 0.8,
                    "min_icir": 0.3,
                },
            ))
        
        # 规则6：fusion 后需要 backtest
        if has_fusion and not any(k.startswith("backtest.") for k in step_kinds):
            fusion_id = self._find_latest_fusion_step(steps)
            data_id = self._find_step_by_kind(steps, "data.price_bars")
            
            recommendations.append(Recommendation(
                step_kind="backtest.factor_signal",
                step_name="回测验证",
                reason="回测是验证因子实际盈利能力的必要步骤。",
                priority=5,
                sample_config={
                    "factor": f"${fusion_id}['composite_factor']",
                    "price_data": f"${data_id}['data']",
                    "signal_threshold": 1.0,
                    "cost_rate": 0.0003,
                },
            ))
        
        # 规则7：backtest 后需要 report
        has_backtest = any(k.startswith("backtest.") for k in step_kinds)
        if has_backtest and "output.report" not in step_kinds:
            eval_id = self._find_step_by_kind(steps, "evaluation.ic")
            
            recommendations.append(Recommendation(
                step_kind="output.report",
                step_name="生成报告",
                reason="报告是因子研究的重要交付物。",
                priority=4,
                sample_config={
                    "top_factors": f"${eval_id}['passed_factors']",
                    "ic_results": f"${eval_id}['ic_series']",
                    "report_dir": "docs/reports",
                },
            ))
        
        # 按优先级排序
        recommendations.sort(key=lambda r: r.priority, reverse=True)
        
        return recommendations[:max_recommendations]
    
    def check_completeness(
        self,
        pipeline: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        检查流水线完整性。
        
        Args:
            pipeline: 流水线定义
        
        Returns:
            包含 completeness、missing_steps、recommendations 的字典
        """
        steps = pipeline.get("steps", [])
        step_kinds = {s.get("kind") for s in steps}
        
        required_steps = [
            "trigger.manual",
            "data.price_bars",
            "factor.technical",
            "evaluation.ic",
        ]
        
        recommended_steps = [
            "fusion.icir_weight",
            "backtest.factor_signal",
            "output.report",
        ]
        
        missing_required = [s for s in required_steps if s not in step_kinds]
        missing_recommended = [s for s in recommended_steps if s not in step_kinds]
        
        # 计算完整性分数
        total = len(required_steps) + len(recommended_steps)
        present = total - len(missing_required) - len(missing_recommended)
        completeness_score = (present / total) * 100
        
        return {
            "completeness_score": completeness_score,
            "grade": self._grade_score(completeness_score),
            "missing_required": missing_required,
            "missing_recommended": missing_recommended,
            "is_complete": len(missing_required) == 0,
            "recommendations": self.suggest_next_steps(pipeline),
        }
    
    def generate_full_flow(
        self,
        target: str = "RB",
        start_date: str = "2023-01-01",
        end_date: str = "2024-12-31",
        include_alpha101: bool = False,
        include_fundamental: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        生成完整的因子研究流程配置。
        
        Args:
            target: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            include_alpha101: 是否包含 Alpha101 因子
            include_fundamental: 是否包含基本面因子
        
        Returns:
            步骤配置列表，可直接用于构建流水线
        """
        steps = []
        
        # Step 1: Trigger
        steps.append({
            "kind": "trigger.manual",
            "step_id": "trigger",
            "config": {
                "target": target,
                "start_date": start_date,
                "end_date": end_date,
                "frequency": "daily",
            },
        })
        
        # Step 2: Price Data
        steps.append({
            "kind": "data.price_bars",
            "step_id": "price_bars",
            "config": {
                "target": "$trigger['target']",
                "start_date": "$trigger['start_date']",
                "end_date": "$trigger['end_date']",
                "frequency": "daily",
                "lookback_days": 60,
            },
            "depends_on": "trigger",
        })
        
        # Step 3: Technical Factors
        steps.append({
            "kind": "factor.technical",
            "step_id": "technical",
            "config": {
                "data": "$price_bars['data']",
                "indicators": ["momentum", "volatility", "rsi"],
                "windows": {
                    "momentum": [5, 10, 20, 60],
                    "volatility": [10, 20, 60],
                    "rsi": [6, 14, 21],
                },
            },
            "depends_on": "price_bars",
        })
        
        # Optional: Alpha101
        if include_alpha101:
            steps.append({
                "kind": "factor.alpha101",
                "step_id": "alpha101",
                "config": {
                    "data": "$price_bars['data']",
                    "top_n": 20,
                },
                "depends_on": "price_bars",
            })
        
        # Optional: Fundamental
        if include_fundamental:
            steps.append({
                "kind": "data.fundamental",
                "step_id": "fundamental_data",
                "config": {
                    "target": "$trigger['target']",
                    "start_date": "$trigger['start_date']",
                    "end_date": "$trigger['end_date']",
                    "data_type": "basis",
                },
                "depends_on": "trigger",
            })
            
            steps.append({
                "kind": "factor.fundamental",
                "step_id": "fundamental",
                "config": {
                    "fundamental_data": "$fundamental_data['data']",
                    "price_data": "$price_bars['data']",
                    "factor_types": ["basis_ratio", "inventory_change"],
                },
                "depends_on": ["fundamental_data", "price_bars"],
            })
        
        # Step 4: IC Evaluation
        steps.append({
            "kind": "evaluation.ic",
            "step_id": "ic_eval",
            "config": {
                "factors": "$technical['factors']",
                "method": "spearman",
                "ic_threshold": 0.02,
                "icir_threshold": 0.3,
            },
            "depends_on": "technical",
        })
        
        # Step 5: Fusion
        steps.append({
            "kind": "fusion.icir_weight",
            "step_id": "fusion",
            "config": {
                "factors": "$technical['factors']",
                "ic_series": "$ic_eval['ic_series']",
                "corr_threshold": 0.8,
                "min_icir": 0.3,
            },
            "depends_on": ["ic_eval", "technical"],
        })
        
        # Step 6: Backtest
        steps.append({
            "kind": "backtest.factor_signal",
            "step_id": "backtest",
            "config": {
                "factor": "$fusion['composite_factor']",
                "price_data": "$price_bars['data']",
                "signal_threshold": 1.0,
                "cost_rate": 0.0003,
            },
            "depends_on": ["fusion", "price_bars"],
        })
        
        # Step 7: Report
        steps.append({
            "kind": "output.report",
            "step_id": "report",
            "config": {
                "top_factors": "$ic_eval['passed_factors']",
                "ic_results": "$ic_eval['ic_series']",
                "report_dir": "docs/reports",
            },
            "depends_on": "ic_eval",
        })
        
        return steps
    
    # -------------------------------------------------------------------------
    # 辅助方法
    # -------------------------------------------------------------------------
    
    def _find_step_by_kind(self, steps: List[Dict], kind: str) -> Optional[str]:
        """查找指定类型的步骤 ID。"""
        for step in steps:
            if step.get("kind") == kind:
                return step.get("id")
        return None
    
    def _find_latest_factor_step(self, steps: List[Dict]) -> Optional[str]:
        """查找最新的因子计算步骤 ID。"""
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
    
    def _grade_score(self, score: float) -> str:
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
