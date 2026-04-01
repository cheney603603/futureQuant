"""
情景分析器

生成 3 种情景（乐观/基准/悲观），每种包含：
- 触发条件
- 价格区间
- 概率
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import pandas as pd


@dataclass
class Scenario:
    """单一情景"""

    name: str  # 'bullish' / 'baseline' / 'bearish'
    probability: float  # 概率（0~1）
    trigger: str  # 触发条件描述
    price_range: tuple[float, float]  # (low, high)
    expected_return: float  # 预期收益率
    holding_period: str  # 持仓周期


class ScenarioAnalyzer:
    """
    情景分析器

    基于当前市场状态和各 Agent 输出，生成三种情景。
    """

    def analyze(
        self,
        current_price: float,
        sentiment_score: float = 0.0,
        atr: float = 0.0,
        market_state: str = "range",
        fundamental_score: float = 0.0,
        quant_signal: int = 0,
        n_days: int = 60,
    ) -> Dict[str, Any]:
        """
        执行情景分析

        Args:
            current_price: 当前价格
            sentiment_score: 基本面情绪评分（-5~5）
            atr: ATR 值
            market_state: 市场状态
            fundamental_score: 基本面评分
            quant_signal: 量化信号方向（1/0/-1）
            n_days: 分析周期天数

        Returns:
            包含三个情景的字典
        """
        # 基准 ATR 移动距离
        unit = atr if atr > 0 else current_price * 0.02

        # === 乐观情景 ===
        bullish_prob = self._compute_bullish_prob(
            sentiment_score, fundamental_score, quant_signal, market_state
        )
        bullish = Scenario(
            name="bullish",
            probability=round(bullish_prob, 3),
            trigger=self._bullish_trigger(sentiment_score, quant_signal, market_state),
            price_range=(
                round(current_price + unit * 1.5, 2),
                round(current_price + unit * 5.0, 2),
            ),
            expected_return=round((unit * 3.0) / current_price, 4),
            holding_period="2-4 周" if n_days < 30 else "1-2 月",
        )

        # === 悲观情景 ===
        bearish_prob = self._compute_bearish_prob(
            sentiment_score, fundamental_score, quant_signal, market_state
        )
        bearish = Scenario(
            name="bearish",
            probability=round(bearish_prob, 3),
            trigger=self._bearish_trigger(sentiment_score, quant_signal, market_state),
            price_range=(
                round(current_price - unit * 5.0, 2),
                round(current_price - unit * 1.5, 2),
            ),
            expected_return=round(-(unit * 3.0) / current_price, 4),
            holding_period="2-4 周" if n_days < 30 else "1-2 月",
        )

        # === 基准情景 ===
        baseline_prob = 1.0 - bullish_prob - bearish_prob
        baseline_prob = max(0.0, min(1.0, baseline_prob))

        # 基准情景价格区间较窄
        if sentiment_score > 0.5:
            baseline_target = current_price + unit * 1.0
        elif sentiment_score < -0.5:
            baseline_target = current_price - unit * 1.0
        else:
            baseline_target = current_price

        baseline = Scenario(
            name="baseline",
            probability=round(baseline_prob, 3),
            trigger="当前趋势延续，震荡整理",
            price_range=(
                round(current_price - unit * 2.0, 2),
                round(current_price + unit * 2.0, 2),
            ),
            expected_return=round((baseline_target - current_price) / current_price, 4),
            holding_period="1-2 周",
        )

        # === 汇总 ===
        scenarios = {
            "bullish": {
                "name": "乐观",
                "probability": bullish.probability,
                "trigger": bullish.trigger,
                "price_range": bullish.price_range,
                "expected_return_pct": round(bullish.expected_return * 100, 2),
                "holding_period": bullish.holding_period,
                "description": f"上涨 {bullish.expected_return*100:.1f}%，"
                f"目标区间 {bullish.price_range[0]:.2f} ~ {bullish.price_range[1]:.2f}",
            },
            "baseline": {
                "name": "基准",
                "probability": baseline.probability,
                "trigger": baseline.trigger,
                "price_range": baseline.price_range,
                "expected_return_pct": round(baseline.expected_return * 100, 2),
                "holding_period": baseline.holding_period,
                "description": f"震荡 {baseline.expected_return*100:+.1f}%，"
                f"区间 {baseline.price_range[0]:.2f} ~ {baseline.price_range[1]:.2f}",
            },
            "bearish": {
                "name": "悲观",
                "probability": bearish.probability,
                "trigger": bearish.trigger,
                "price_range": bearish.price_range,
                "expected_return_pct": round(bearish.expected_return * 100, 2),
                "holding_period": bearish.holding_period,
                "description": f"下跌 {abs(bearish.expected_return)*100:.1f}%，"
                f"目标区间 {bearish.price_range[0]:.2f} ~ {bearish.price_range[1]:.2f}",
            },
        }

        # 期望收益
        expected_total = (
            bullish.probability * bullish.expected_return
            + baseline.probability * baseline.expected_return
            + bearish.probability * bearish.expected_return
        )

        return {
            "scenarios": scenarios,
            "expected_return": round(expected_total, 4),
            "current_price": current_price,
            "unit_move": round(unit, 4),
            "recommendation": self._get_scenario_recommendation(
                bullish.probability, bearish.probability, sentiment_score
            ),
        }

    def _compute_bullish_prob(
        self,
        sentiment: float,
        fundamental: float,
        quant_signal: int,
        market_state: str,
    ) -> float:
        """计算乐观情景概率"""
        prob = 0.20  # 基础概率

        # 情绪评分加成
        if sentiment > 2:
            prob += 0.25
        elif sentiment > 0.5:
            prob += 0.10

        # 基本面加成
        if fundamental > 1:
            prob += 0.10

        # 量化信号加成
        if quant_signal > 0:
            prob += 0.15

        # 趋势加成
        if market_state == "trend_up":
            prob += 0.10

        return min(prob, 0.50)

    def _compute_bearish_prob(
        self,
        sentiment: float,
        fundamental: float,
        quant_signal: int,
        market_state: str,
    ) -> float:
        """计算悲观情景概率"""
        prob = 0.20  # 基础概率

        if sentiment < -2:
            prob += 0.25
        elif sentiment < -0.5:
            prob += 0.10

        if fundamental < -1:
            prob += 0.10

        if quant_signal < 0:
            prob += 0.15

        if market_state == "trend_down":
            prob += 0.10

        return min(prob, 0.50)

    def _bullish_trigger(self, sentiment: float, quant: int, state: str) -> str:
        """乐观情景触发条件"""
        conditions = []
        if sentiment > 1:
            conditions.append("基本面持续改善")
        if quant > 0:
            conditions.append("量化信号做多")
        if state == "trend_up":
            conditions.append("价格突破前高")
        if not conditions:
            conditions.append("无明显利多信号，但市场情绪平稳")
        return " + ".join(conditions)

    def _bearish_trigger(self, sentiment: float, quant: int, state: str) -> str:
        """悲观情景触发条件"""
        conditions = []
        if sentiment < -1:
            conditions.append("基本面恶化")
        if quant < 0:
            conditions.append("量化信号做空")
        if state == "trend_down":
            conditions.append("价格跌破前低")
        if not conditions:
            conditions.append("无明显利空信号，但市场情绪偏弱")
        return " + ".join(conditions)

    def _get_scenario_recommendation(
        self, bull_prob: float, bear_prob: float, sentiment: float
    ) -> str:
        """基于情景概率给出建议"""
        if bull_prob > 0.40 and sentiment > 0:
            return "偏多操作：乐观情景概率最高，建议逢低做多"
        elif bear_prob > 0.40 and sentiment < 0:
            return "偏空操作：悲观情景概率最高，建议逢高做空"
        elif abs(bull_prob - bear_prob) < 0.05:
            return "中性操作：多空情景概率接近，建议观望或对冲"
        else:
            return "谨慎操作：建议等待情景明朗"
