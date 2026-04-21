"""
因子挖掘自我反思模块 (Factor Mining Self-Reflection)

基于 Agentic AI 理念，实现因子挖掘过程的自我评估与策略调整：

1. 结果评估 - 分析因子挖掘结果的薄弱环节
2. 策略调整 - 根据评估结果调整搜索策略
3. 迭代优化 - 自动触发新一轮搜索
4. 反思报告 - 生成改进建议

设计原则：
- 不依赖外部 LLM，纯规则驱动
- 基于量化指标进行客观评估
- 可配置调整策略
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

from ...core.logger import get_logger

logger = get_logger("agent.factor_mining.self_reflection")


# =============================================================================
# 评估维度与问题识别
# =============================================================================


@dataclass
class WeaknessReport:
    """薄弱环节报告"""
    dimension: str
    severity: str  # 'critical' / 'warning' / 'info'
    description: str
    suggestion: str
    current_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass
class StrategyAdjustment:
    """策略调整建议"""
    original_strategy: Dict[str, Any]
    adjusted_strategy: Dict[str, Any]
    reason: str
    expected_improvement: Optional[str] = None


# =============================================================================
# 评估器
# =============================================================================


class MiningResultEvaluator:
    """
    挖掘结果评估器

    对因子挖掘结果进行多维度评估，识别薄弱环节。
    """

    # 评估阈值配置
    THRESHOLDS = {
        "ic_mean": {"critical": 0.01, "warning": 0.02, "good": 0.03},
        "icir": {"critical": 0.2, "warning": 0.3, "good": 0.5},
        "ic_win_rate": {"critical": 0.45, "warning": 0.50, "good": 0.55},
        "turnover": {"critical": 0.50, "warning": 0.30, "good": 0.15},
        "factor_count": {"critical": 0, "warning": 5, "good": 10},
        "category_diversity": {"critical": 1, "warning": 2, "good": 3},
    }

    def __init__(self) -> None:
        self.logger = logger

    def evaluate(
        self,
        factors: List[Dict[str, Any]],
        price_data: Optional[pd.DataFrame] = None,
        returns: Optional[pd.Series] = None,
    ) -> Tuple[List[WeaknessReport], Dict[str, float]]:
        """
        评估因子挖掘结果

        Args:
            factors: 因子列表（每项含 ic_mean, icir, turnover 等字段）
            price_data: 价格数据（用于时序分析）
            returns: 收益率序列（用于 IC 衰减分析）

        Returns:
            (薄弱环节列表, 摘要统计字典)
        """
        weaknesses: List[WeaknessReport] = []

        if not factors:
            weaknesses.append(WeaknessReport(
                dimension="factor_count",
                severity="critical",
                description="没有找到有效因子",
                suggestion="扩大候选池、增加进化代数、调整变量集",
            ))
            return weaknesses, {}

        # 统计摘要
        summary = {
            "total_factors": len(factors),
            "valid_factors": sum(1 for f in factors if f.get("is_valid", False)),
            "avg_ic": np.mean([f.get("ic_mean", 0) for f in factors]),
            "avg_icir": np.mean([f.get("icir", 0) for f in factors]),
            "avg_turnover": np.mean([f.get("turnover", 0) for f in factors]),
            "avg_win_rate": np.mean([f.get("ic_win_rate", 0) for f in factors]),
            "avg_score": np.mean([f.get("overall_score", 0) for f in factors]),
            "max_ic": max(abs(f.get("ic_mean", 0)) for f in factors),
            "max_icir": max(f.get("icir", 0) for f in factors),
        }

        # 评估每个维度
        self._check_ic_quality(factors, weaknesses)
        self._check_stability(factors, weaknesses)
        self._check_turnover(factors, weaknesses)
        self._check_diversity(factors, weaknesses)
        self._check_ic_decay(factors, price_data, returns, weaknesses)

        return weaknesses, summary

    def _check_ic_quality(
        self,
        factors: List[Dict[str, Any]],
        weaknesses: List[WeaknessReport],
    ) -> None:
        """检查 IC 质量"""
        avg_ic = summary_ic = np.mean([abs(f.get("ic_mean", 0)) for f in factors])
        max_ic = max(abs(f.get("ic_mean", 0)) for f in factors)
        t = self.THRESHOLDS["ic_mean"]

        if max_ic < t["critical"]:
            weaknesses.append(WeaknessReport(
                dimension="ic_quality",
                severity="critical",
                description=f"IC 最高仅 {max_ic:.4f}，低于最低阈值 {t['critical']}",
                suggestion="考虑使用不同时间框架、引入宏观变量、调整因子构造方式",
                current_value=max_ic,
                threshold=t["critical"],
            ))
        elif max_ic < t["warning"]:
            weaknesses.append(WeaknessReport(
                dimension="ic_quality",
                severity="warning",
                description=f"IC 最高 {max_ic:.4f}，低于建议值 {t['good']}",
                suggestion="可尝试增加滚动窗口、引入多周期因子叠加",
                current_value=max_ic,
                threshold=t["warning"],
            ))

    def _check_stability(
        self,
        factors: List[Dict[str, Any]],
        weaknesses: List[WeaknessReport],
    ) -> None:
        """检查稳定性（ICIR）"""
        avg_icir = np.mean([f.get("icir", 0) for f in factors])
        max_icir = max(f.get("icir", 0) for f in factors)
        t = self.THRESHOLDS["icir"]

        if max_icir < t["critical"]:
            weaknesses.append(WeaknessReport(
                dimension="stability",
                severity="critical",
                description=f"ICIR 最高仅 {max_icir:.3f}，因子预测能力不稳定",
                suggestion="建议对因子进行去极值、标准化处理，或使用更长的回测窗口",
                current_value=max_icir,
                threshold=t["critical"],
            ))
        elif max_icir < t["warning"]:
            weaknesses.append(WeaknessReport(
                dimension="stability",
                severity="warning",
                description=f"ICIR 最高 {max_icir:.3f}，稳定性有待提升",
                suggestion="可尝试 Fama-MacBeth 回归平滑因子暴露",
                current_value=max_icir,
                threshold=t["warning"],
            ))

    def _check_turnover(
        self,
        factors: List[Dict[str, Any]],
        weaknesses: List[WeaknessReport],
    ) -> None:
        """检查换手率"""
        avg_turnover = np.mean([f.get("turnover", 0) for f in factors])
        max_turnover = max(f.get("turnover", 0) for f in factors)
        t = self.THRESHOLDS["turnover"]

        if max_turnover > t["warning"]:
            weaknesses.append(WeaknessReport(
                dimension="turnover",
                severity="warning",
                description=f"平均换手率 {avg_turnover:.2%}，可能增加交易成本",
                suggestion="可使用更平滑的加权方式、引入持仓约束",
                current_value=avg_turnover,
                threshold=t["warning"],
            ))

    def _check_diversity(
        self,
        factors: List[Dict[str, Any]],
        weaknesses: List[WeaknessReport],
    ) -> None:
        """检查多样性"""
        categories = set()
        for f in factors:
            cat = f.get("category", "unknown")
            if cat:
                categories.add(cat)

        t = self.THRESHOLDS["category_diversity"]
        if len(categories) < t["warning"]:
            weaknesses.append(WeaknessReport(
                dimension="diversity",
                severity="warning",
                description=f"仅发现 {len(categories)} 个类别的因子，覆盖不足",
                suggestion="建议引入基本面因子、情绪因子或另类数据",
                current_value=float(len(categories)),
                threshold=t["warning"],
            ))

    def _check_ic_decay(
        self,
        factors: List[Dict[str, Any]],
        price_data: Optional[pd.DataFrame],
        returns: Optional[pd.Series],
        weaknesses: List[WeaknessReport],
    ) -> None:
        """检查 IC 随时间的衰减趋势"""
        if price_data is None or returns is None or len(factors) == 0:
            return

        # 取最佳因子检验时序稳定性
        best = max(factors, key=lambda x: x.get("overall_score", 0))
        factor_name = best.get("name", "unknown")

        if factor_name not in price_data.columns:
            return

        factor_values = price_data[factor_name]
        aligned = pd.concat([factor_values, returns], axis=1).dropna()

        if len(aligned) < 60:
            return

        # 计算滚动 IC（最近6个月 vs 更早）
        mid = len(aligned) // 2
        recent = aligned.iloc[mid:]
        older = aligned.iloc[:mid]

        def spearman_ic(x, y):
            r, _ = stats.spearmanr(x, y)
            return r if not np.isnan(r) else 0.0

        ic_recent = spearman_ic(recent.iloc[:, 0], recent.iloc[:, 1])
        ic_older = spearman_ic(older.iloc[:, 0], older.iloc[:, 1])

        # IC 方向反转或大幅下降
        if ic_recent * ic_older < 0:
            weaknesses.append(WeaknessReport(
                dimension="regime_stability",
                severity="critical",
                description=f"因子 '{factor_name}' IC 在近期发生方向反转（前 {ic_older:.4f} → 近 {ic_recent:.4f}）",
                suggestion="因子可能对市场 regime 敏感，建议加入 regime 检测或使用时序自适应权重",
                current_value=ic_recent,
                threshold=0.0,
            ))
        elif abs(ic_recent) < abs(ic_older) * 0.5 and abs(ic_recent) < 0.01:
            weaknesses.append(WeaknessReport(
                dimension="regime_stability",
                severity="warning",
                description=f"因子 '{factor_name}' IC 近半下降超过 50%（前 {ic_older:.4f} → 近 {ic_recent:.4f}）",
                suggestion="因子可能正在衰减，建议增加样本外验证频率",
                current_value=ic_recent,
                threshold=abs(ic_older) * 0.5,
            ))


# =============================================================================
# 策略调整器
# =============================================================================


class StrategyAdjuster:
    """
    因子挖掘策略调整器

    根据评估结果自动调整搜索策略参数。
    """

    DEFAULT_STRATEGY = {
        "candidate_pool_enabled": True,
        "gp_evolution_enabled": False,
        "gp_population_size": 50,
        "gp_generations": 10,
        "gp_max_depth": 4,
        "gp_mutation_prob": 0.25,
        "use_technical": True,
        "use_fundamental": True,
        "use_cross": True,
        "min_ic_threshold": 0.02,
        "min_icir_threshold": 0.3,
        "lookback_periods": [5, 10, 20, 60],
        "macro_enabled": False,
    }

    def __init__(self) -> None:
        self.logger = logger

    def adjust(
        self,
        strategy: Dict[str, Any],
        weaknesses: List[WeaknessReport],
        iteration: int,
    ) -> StrategyAdjustment:
        """
        根据薄弱环节调整策略

        Args:
            strategy: 当前策略配置
            weaknesses: 识别出的薄弱环节
            iteration: 当前迭代轮次

        Returns:
            策略调整建议
        """
        adjusted = copy.deepcopy(strategy)
        reasons: List[str] = []

        for w in weaknesses:
            if w.severity == "critical":
                # 严重问题：强制启用 GP 进化
                if not adjusted.get("gp_evolution_enabled"):
                    adjusted["gp_evolution_enabled"] = True
                    adjusted["gp_generations"] = min(adjusted.get("gp_generations", 10) + 10, 30)
                    adjusted["gp_population_size"] = min(adjusted.get("gp_population_size", 50) + 50, 200)
                    reasons.append(f"[{w.dimension}] 启用 GP 进化，扩大搜索空间")

                # IC 过低：放宽阈值或增加候选池
                if w.dimension == "ic_quality":
                    adjusted["lookback_periods"] = [3, 5, 10, 20, 60, 120]
                    reasons.append(f"[{w.dimension}] 扩展时间周期覆盖")

                # 稳定性问题：延长回测窗口
                if w.dimension == "stability":
                    adjusted["min_ic_threshold"] = max(0.01, adjusted.get("min_ic_threshold", 0.02) * 0.8)
                    reasons.append(f"[{w.dimension}] 适当降低 IC 阈值以保留更多候选")

            elif w.severity == "warning":
                # 警告问题：适度调整
                if w.dimension == "diversity":
                    if not adjusted.get("use_fundamental"):
                        adjusted["use_fundamental"] = True
                        reasons.append("[diversity] 启用基本面因子")
                    if not adjusted.get("macro_enabled"):
                        adjusted["macro_enabled"] = True
                        reasons.append("[diversity] 启用宏观因子")

                if w.dimension == "turnover":
                    adjusted["gp_mutation_prob"] = min(
                        adjusted.get("gp_mutation_prob", 0.25) + 0.1, 0.5
                    )
                    reasons.append("[turnover] 提高变异率以探索更稳定结构")

        # 迭代学习：每轮增加搜索强度
        if iteration > 0:
            adjusted["gp_generations"] = min(
                adjusted.get("gp_generations", 10) + 5, 50
            )

        return StrategyAdjustment(
            original_strategy=strategy,
            adjusted_strategy=adjusted,
            reason="; ".join(reasons) if reasons else "无显著调整",
        )


# =============================================================================
# 自我反思主类
# =============================================================================


class FactorMiningSelfReflection:
    """
    因子挖掘自我反思器

    整合评估与策略调整，实现闭环优化：

    使用示例：
        >>> reflection = FactorMiningSelfReflection()
        >>> reflection.set_max_iterations(3)
        >>>
        >>> # 第一轮
        >>> factors = mining_agent.mine(context)
        >>> should_continue, adjustment = reflection.reflect(
        ...     factors, price_data, returns, current_strategy
        ... )
        >>> if should_continue:
        ...     new_strategy = adjustment.adjusted_strategy
        ...     # 使用新策略重新挖掘
    """

    def __init__(
        self,
        max_iterations: int = 3,
        min_improvement: float = 0.05,
    ) -> None:
        """
        Args:
            max_iterations: 最大反思迭代次数
            min_improvement: 最小改善阈值（分数提升低于此值则停止）
        """
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        self.evaluator = MiningResultEvaluator()
        self.adjuster = StrategyAdjuster()
        self.history: List[Dict[str, Any]] = []
        self.logger = logger

    def reflect(
        self,
        factors: List[Dict[str, Any]],
        price_data: Optional[pd.DataFrame],
        returns: Optional[pd.Series],
        current_strategy: Dict[str, Any],
        iteration: int = 0,
    ) -> Tuple[bool, StrategyAdjustment]:
        """
        执行自我反思

        Args:
            factors: 当前挖掘出的因子列表
            price_data: 价格数据
            returns: 收益率序列
            current_strategy: 当前策略配置
            iteration: 当前迭代序号

        Returns:
            (是否继续迭代, 策略调整建议)
        """
        # 评估
        weaknesses, summary = self.evaluator.evaluate(factors, price_data, returns)

        # 记录历史
        record = {
            "iteration": iteration,
            "weaknesses": [asdict(w) for w in weaknesses],
            "summary": summary,
        }
        self.history.append(record)

        self.logger.info(f"=== Self-Reflection Iteration {iteration} ===")
        self.logger.info(f"  Factors: {summary.get('valid_factors', 0)}/{summary.get('total_factors', 0)} valid")
        self.logger.info(f"  Avg IC: {summary.get('avg_ic', 0):.4f}, Avg ICIR: {summary.get('avg_icir', 0):.3f}")
        self.logger.info(f"  Weaknesses: {len(weaknesses)}")

        for w in weaknesses:
            self.logger.info(f"    [{w.severity.upper()}] {w.dimension}: {w.description}")

        # 决策
        if iteration >= self.max_iterations:
            self.logger.info("Max iterations reached, stopping")
            return False, StrategyAdjustment(
                original_strategy=current_strategy,
                adjusted_strategy=current_strategy,
                reason="达到最大迭代次数",
            )

        # 无严重问题，检查改善空间
        if not any(w.severity == "critical" for w in weaknesses):
            if not factors:
                self.logger.info("No factors found, continuing with adjusted strategy")
            else:
                current_best = max(f.get("overall_score", 0) for f in factors)
                prev_best = self.history[-2]["summary"].get("max_score", 0) if len(self.history) > 1 else 0

                if current_best - prev_best < self.min_improvement and iteration > 0:
                    self.logger.info(f"Improvement {current_best - prev_best:.4f} < {self.min_improvement}, stopping")
                    return False, StrategyAdjustment(
                        original_strategy=current_strategy,
                        adjusted_strategy=current_strategy,
                        reason="改善边际不足",
                    )

        # 调整策略
        adjustment = self.adjuster.adjust(current_strategy, weaknesses, iteration)
        self.logger.info(f"  Strategy adjusted: {adjustment.reason}")

        return True, adjustment

    def generate_report(self) -> str:
        """生成反思报告"""
        lines = ["# 因子挖掘自我反思报告", ""]

        for record in self.history:
            iter_num = record["iteration"]
            lines.append(f"## 迭代 {iter_num}")
            lines.append("")

            summary = record["summary"]
            if summary:
                lines.append(f"- 有效因子: {summary.get('valid_factors', 0)}/{summary.get('total_factors', 0)}")
                lines.append(f"- 平均 IC: {summary.get('avg_ic', 0):.4f}")
                lines.append(f"- 平均 ICIR: {summary.get('avg_icir', 0):.3f}")
                lines.append("")

            weaknesses = record["weaknesses"]
            if weaknesses:
                lines.append(f"**薄弱环节 ({len(weaknesses)} 个):**")
                for w in weaknesses:
                    lines.append(f"- `[{w['severity']}]` {w['dimension']}: {w['description']}")
                    lines.append(f"  - 建议: {w['suggestion']}")
                lines.append("")

        return "\n".join(lines)

    def reset(self) -> None:
        """重置反思历史"""
        self.history = []
