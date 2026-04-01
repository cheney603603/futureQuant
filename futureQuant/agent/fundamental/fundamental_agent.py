"""
基本面分析 Agent

功能：
- 基于价格数据模拟基本面因子（库存、基差、仓单、供需等）
- 计算利多/利空情绪评分（-5 到 +5）
- 判断库存周期阶段（主动补库/被动补库/主动去库/被动去库）
- 生成 Markdown 格式基本面报告，保存至 docs/reports/

依赖：
- futureQuant.agent.base.BaseAgent
- futureQuant.agent.fundamental.sentiment_result.SentimentResult
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..base import AgentResult, AgentStatus, BaseAgent
from .fundamental_report_generator import FundamentalReportGenerator
from .sentiment_result import SentimentResult


class FundamentalAnalysisAgent(BaseAgent):
    """
    基本面分析 Agent

    基于价格数据和品种特性，模拟生成基本面因子并进行情绪分析。
    由于网络爬虫实现复杂，本模块采用数学模拟方式生成合理的基本面因子序列。

    Attributes:
        name: Agent 名称
        config: 配置字典

    Example:
        >>> agent = FundamentalAnalysisAgent(config={"threshold": 1.0})
        >>> result = agent.run({
        ...     "target": "RB2105",
        ...     "date_range": ("2024-01-01", "2024-12-31"),
        ...     "price_data": price_df,  # 可选，提供价格数据以生成更准确因子
        ... })
        >>> print(result.metrics["sentiment_score"])
    """

    # 品种分类配置（影响基本面因子的基准值）
    SPECIES_CONFIG: Dict[str, Dict[str, Any]] = {
        "黑色系": {
            "keywords": ["RB", "HC", "I", "J", "JM"],
            "basis_volatility": 0.02,
            "inventory_sensitivity": 1.2,
        },
        "有色金属": {
            "keywords": ["CU", "AL", "ZN", "NI", "PB"],
            "basis_volatility": 0.015,
            "inventory_sensitivity": 1.0,
        },
        "化工": {
            "keywords": ["TA", "MA", "RU", "BU", "L"],
            "basis_volatility": 0.025,
            "inventory_sensitivity": 1.5,
        },
        "农产品": {
            "keywords": ["M", "Y", "A", "B", "P"],
            "basis_volatility": 0.02,
            "inventory_sensitivity": 1.3,
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化基本面分析 Agent

        Args:
            config: 配置字典，支持阈值、利多/利空因子权重等
        """
        super().__init__(name="fundamental_analysis", config=config)
        self.report_generator = FundamentalReportGenerator()

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行基本面分析

        Args:
            context: 执行上下文，必须包含：
                - target (str): 标的代码
                - date_range (tuple): 日期范围 (start, end)
                - price_data (DataFrame, optional): 价格数据

        Returns:
            AgentResult: 包含 metrics["sentiment_result"] 等
        """
        target: str = context.get("target", "UNKNOWN")
        date_range: tuple = context.get("date_range", ("2024-01-01", "2024-12-31"))
        price_data: Optional[pd.DataFrame] = context.get("price_data")

        self._logger.info(f"Analyzing fundamental factors for {target}")

        try:
            # Step 1: 生成基本面因子序列
            fundamental_factors = self._generate_fundamental_factors(
                target=target,
                date_range=date_range,
                price_data=price_data,
            )

            # Step 2: 计算情绪评分
            sentiment_result = self._calculate_sentiment(
                target=target,
                factors=fundamental_factors,
            )

            # Step 3: 判断库存周期
            inventory_cycle = self._determine_inventory_cycle(fundamental_factors)
            sentiment_result.inventory_cycle = inventory_cycle

            # Step 4: 评估供需格局
            supply_demand = self._evaluate_supply_demand(fundamental_factors)
            sentiment_result.supply_demand = supply_demand

            # Step 5: 生成报告
            report_path = self._generate_report(target, fundamental_factors, sentiment_result)

            self._logger.info(
                f"Fundamental analysis for {target}: "
                f"sentiment={sentiment_result.sentiment_score:.2f}, "
                f"inventory_cycle={inventory_cycle}, "
                f"supply_demand={supply_demand}"
            )

            # 构建结果
            metrics: Dict[str, Any] = {
                "sentiment_score": sentiment_result.sentiment_score,
                "sentiment_result": sentiment_result,
                "inventory_cycle": inventory_cycle,
                "supply_demand": supply_demand,
                "confidence": sentiment_result.confidence,
                "drivers": sentiment_result.drivers,
                "report_path": report_path,
            }

            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                data=fundamental_factors,
                metrics=metrics,
            )

        except Exception as exc:
            self._logger.error(f"Fundamental analysis failed for {target}: {exc}")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=[str(exc)],
            )

    def _generate_fundamental_factors(
        self,
        target: str,
        date_range: tuple,
        price_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        生成基本面因子序列

        基于价格数据或模拟生成合理的基本面因子值序列。

        Args:
            target: 标的代码
            date_range: 日期范围
            price_data: 价格数据（可选）

        Returns:
            DataFrame：index=date，columns=[basis_rate, inventory_level, warehouse_receipt, ...]
        """
        # 生成日期序列
        dates = pd.date_range(start=date_range[0], end=date_range[1], freq="B")
        n = len(dates)

        # 获取品种配置
        species = self._detect_species(target)
        cfg = self.SPECIES_CONFIG.get(species, self.SPECIES_CONFIG["黑色系"])

        # 提取价格（如果有）
        if price_data is not None and not price_data.empty:
            if "close" in price_data.columns:
                close = price_data["close"].reindex(dates).fillna(method="ffill").fillna(method="bfill")
            else:
                close = pd.Series(4000.0, index=dates)
        else:
            # 默认模拟价格（平稳随机游走）
            np.random.seed(hash(target) % 2**32)
            returns = np.random.normal(0.0002, 0.015, n)
            close = pd.Series(4000.0 * np.cumprod(1 + returns), index=dates)

        # --- 因子生成（数学模拟） ---
        basis_vol = cfg["basis_volatility"]
        inv_sens = cfg["inventory_sensitivity"]

        # 1. 基差率 (现货 - 期货) / 期货（简化：基于价格自相关性）
        trend = close.pct_change().rolling(5).mean().fillna(0)
        basis_rate = (trend * 10 + np.random.normal(0, basis_vol, n)) * 100
        basis_rate = pd.Series(np.clip(basis_rate, -10, 10), index=dates)

        # 2. 库存水平（相对指标，0~100）
        # 库存与价格呈负相关：高价格抑制需求 → 库存累积
        inv_trend = -(close.pct_change().rolling(10).mean().fillna(0)) * 50
        inventory_level = 50 + inv_trend + np.random.normal(0, 3, n)
        inventory_level = pd.Series(np.clip(inventory_level, 0, 100), index=dates)

        # 3. 仓单数量（仓单与库存正相关，标准化到 0~100）
        warehouse_receipt = inventory_level * 0.7 + np.random.normal(0, 5, n)
        warehouse_receipt = pd.Series(np.clip(warehouse_receipt, 0, 100), index=dates)

        # 4. 供需差值（供需指数，>0 表示供过于求，<0 表示供不应求）
        demand_gap = -close.pct_change().rolling(5).mean().fillna(0) * 200
        demand_gap = pd.Series(demand_gap + np.random.normal(0, 5, n), index=dates)

        # 5. 进口利润（模拟：汇率 + 国际价格差）
        np.random.seed(hash(target + "imp") % 2**32)
        import_profit = np.random.normal(0, 2, n) + np.sin(np.arange(n) * 0.1) * 3
        import_profit = pd.Series(import_profit, index=dates)

        # 6. 开工率（产能利用率，50~100）
        operating_rate = 75 + np.random.normal(0, 5, n) + close.pct_change().rolling(5).mean().fillna(0) * 100
        operating_rate = pd.Series(np.clip(operating_rate, 50, 100), index=dates)

        # 7. 利润指数（钢厂/产业链利润，-100~100）
        profit_index = close.pct_change().rolling(10).mean().fillna(0) * 500
        profit_index = pd.Series(
            np.clip(profit_index + np.random.normal(0, 5, n), -100, 100),
            index=dates,
        )

        # 8. 库存变化率（环比）
        inventory_change = inventory_level.pct_change().fillna(0) * 100

        # 9. 到港量预估（进口端，标准化）
        arrivals = 50 + np.random.normal(0, 8, n) + import_profit * 2
        arrivals = pd.Series(np.clip(arrivals, 0, 100), index=dates)

        # 10. 替代品价格差（相关品种价差，如螺纹 vs 热卷）
        np.random.seed(hash(target + "spread") % 2**32)
        spread = np.random.normal(0, 5, n)
        spread = pd.Series(spread, index=dates)

        df = pd.DataFrame(
            {
                "basis_rate": basis_rate,
                "inventory_level": inventory_level,
                "warehouse_receipt": warehouse_receipt,
                "demand_gap": demand_gap,
                "import_profit": import_profit,
                "operating_rate": operating_rate,
                "profit_index": profit_index,
                "inventory_change": inventory_change,
                "arrivals": arrivals,
                "spread": spread,
                "close": close,
            },
            index=dates,
        )

        return df

    def _detect_species(self, target: str) -> str:
        """
        根据标的代码识别品种分类

        Args:
            target: 标的代码

        Returns:
            品种分类名称
        """
        for species, cfg in self.SPECIES_CONFIG.items():
            for kw in cfg["keywords"]:
                if kw in target.upper():
                    return species
        return "黑色系"  # 默认

    def _calculate_sentiment(
        self,
        target: str,
        factors: pd.DataFrame,
    ) -> SentimentResult:
        """
        计算利多/利空情绪评分

        评分范围 -5 到 +5，综合考虑基差、库存、供需等多维度因子。

        Args:
            target: 标的代码
            factors: 基本面因子 DataFrame

        Returns:
            SentimentResult
        """
        latest = factors.iloc[-1]

        # 各因子权重（可配置）
        threshold = self.config.get("threshold", 1.0)
        weights = self.config.get(
            "factor_weights",
            {
                "basis_rate": 0.25,
                "inventory_level": 0.20,
                "demand_gap": 0.15,
                "profit_index": 0.15,
                "import_profit": 0.10,
                "operating_rate": 0.10,
                "warehouse_receipt": 0.05,
            },
        )

        # 计算各因子得分（标准化到 -2.5 ~ +2.5）
        scores: List[Dict[str, Any]] = []

        # 基差率：正基差（现货>期货）→ 利多
        basis_score = np.clip(latest["basis_rate"] / 2, -2.5, 2.5)
        scores.append({"factor": "基差率", "score": basis_score, "weight": weights["basis_rate"]})
        raw_score = basis_score * weights["basis_rate"]

        # 库存水平：低库存 → 利多
        inv_norm = (latest["inventory_level"] - 50) / 50  # 标准化到 -1~1
        inv_score = np.clip(-inv_norm * 2.5, -2.5, 2.5)
        scores.append({"factor": "库存水平", "score": inv_score, "weight": weights["inventory_level"]})
        raw_score += inv_score * weights["inventory_level"]

        # 供需差：负值（供不应求）→ 利多
        gap_score = np.clip(-latest["demand_gap"] / 10, -2.5, 2.5)
        scores.append({"factor": "供需差", "score": gap_score, "weight": weights["demand_gap"]})
        raw_score += gap_score * weights["demand_gap"]

        # 利润指数：正利润 → 利多
        profit_score = np.clip(latest["profit_index"] / 20, -2.5, 2.5)
        scores.append({"factor": "利润指数", "score": profit_score, "weight": weights["profit_index"]})
        raw_score += profit_score * weights["profit_index"]

        # 进口利润：正值 → 利多
        imp_score = np.clip(latest["import_profit"] / 5, -2.5, 2.5)
        scores.append({"factor": "进口利润", "score": imp_score, "weight": weights["import_profit"]})
        raw_score += imp_score * weights["import_profit"]

        # 开工率：适中偏高低 → 供给压力（中性偏空）
        op_norm = (latest["operating_rate"] - 75) / 25  # 标准化
        op_score = np.clip(op_norm * 1.5, -2.5, 2.5)
        scores.append({"factor": "开工率", "score": op_score, "weight": weights["operating_rate"]})
        raw_score += op_score * weights["operating_rate"]

        # 仓单：低仓单 → 利多
        wr_score = np.clip(-(latest["warehouse_receipt"] - 50) / 25 * 2.5, -2.5, 2.5)
        scores.append({"factor": "仓单数量", "score": wr_score, "weight": weights["warehouse_receipt"]})
        raw_score += wr_score * weights["warehouse_receipt"]

        # 综合评分（-5 ~ +5）
        sentiment_score = float(np.clip(raw_score, -5, 5))

        # 置信度：因子间一致性越高置信度越高
        score_std = np.std([s["score"] for s in scores])
        confidence = float(np.clip(1.0 - score_std / 3.0, 0.3, 1.0))

        # 驱动因素列表
        drivers: List[Dict[str, Any]] = []
        for s in scores:
            direction = "利多" if s["score"] > 0.3 else ("利空" if s["score"] < -0.3 else "中性")
            drivers.append(
                {
                    "factor": s["factor"],
                    "direction": direction,
                    "score": round(float(s["score"]), 3),
                    "weight": s["weight"],
                }
            )

        # 时间窗口判断
        horizon = "medium"
        if abs(sentiment_score) > 3.0:
            horizon = "long"
        elif abs(sentiment_score) < 1.0:
            horizon = "short"

        return SentimentResult(
            target=target,
            sentiment_score=sentiment_score,
            confidence=confidence,
            time_horizon=horizon,
            drivers=drivers,
        )

    def _determine_inventory_cycle(self, factors: pd.DataFrame) -> str:
        """
        判断库存周期阶段

        依据库存水平及其变化趋势，结合利润趋势判断：
        - 主动补库：库存上升 + 利润上升
        - 被动补库：库存上升 + 利润下降
        - 主动去库：库存下降 + 利润下降
        - 被动去库：库存下降 + 利润上升

        Args:
            factors: 基本面因子 DataFrame

        Returns:
            库存周期阶段字符串
        """
        inv = factors["inventory_level"]
        profit = factors["profit_index"]

        # 计算变化方向
        inv_change = inv.iloc[-1] - inv.iloc[-5] if len(inv) >= 5 else 0
        profit_change = profit.iloc[-1] - profit.iloc[-5] if len(profit) >= 5 else 0

        inv_rising = inv_change > 2
        profit_rising = profit_change > 5

        if inv_rising and profit_rising:
            return "主动补库"
        elif inv_rising and not profit_rising:
            return "被动补库"
        elif not inv_rising and not profit_rising:
            return "主动去库"
        else:
            return "被动去库"

    def _evaluate_supply_demand(self, factors: pd.DataFrame) -> str:
        """
        评估供需格局

        Args:
            factors: 基本面因子 DataFrame

        Returns:
            供需格局：'tight'（偏紧）/ 'balanced'（平衡）/ 'loose'（宽松）
        """
        inv = factors["inventory_level"].iloc[-1]
        gap = factors["demand_gap"].iloc[-1]

        score = -inv / 50 + (-gap) / 10  # 低库存 + 供不应求 → 偏紧

        if score > 0.5:
            return "tight"
        elif score < -0.5:
            return "loose"
        else:
            return "balanced"

    def _generate_report(
        self,
        target: str,
        factors: pd.DataFrame,
        sentiment: SentimentResult,
    ) -> str:
        """
        生成并保存基本面报告

        Args:
            target: 标的代码
            factors: 基本面因子 DataFrame
            sentiment: 情绪分析结果

        Returns:
            报告保存路径
        """
        os.makedirs("D:/310Programm/futureQuant/docs/reports", exist_ok=True)
        report_path = os.path.join(
            "D:/310Programm/futureQuant/docs/reports",
            f"fundamental_{target}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.md",
        )

        content = self.report_generator.generate(
            target=target,
            factors=factors,
            sentiment=sentiment,
        )

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)

        return report_path
