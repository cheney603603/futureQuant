"""
动态权重调整器

根据市场状态（高/低波动、趋势/震荡）动态调整各 Agent 权重。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


@dataclass
class AgentWeights:
    """各 Agent 权重"""

    fundamental: float = 0.30  # 基本面权重
    quant: float = 0.30        # 量化信号权重
    price_behavior: float = 0.25  # 价格行为权重
    backtest: float = 0.15     # 回测验证权重

    def to_dict(self) -> Dict[str, float]:
        return {
            "fundamental": self.fundamental,
            "quant": self.quant,
            "price_behavior": self.price_behavior,
            "backtest": self.backtest,
        }

    def normalize(self) -> "AgentWeights":
        """归一化权重（总和=1）"""
        total = self.fundamental + self.quant + self.price_behavior + self.backtest
        if total < 1e-8:
            return AgentWeights()
        return AgentWeights(
            fundamental=self.fundamental / total,
            quant=self.quant / total,
            price_behavior=self.price_behavior / total,
            backtest=self.backtest / total,
        )


class DynamicWeightEngine:
    """
    动态权重调整引擎

    根据市场状态自动调整各 Agent 的权重占比。

    调整规则：
    - 高波动 → 基本面权重 +20%（宏观驱动增强）
    - 低波动 → 量化权重 +15%（因子效率更高）
    - 趋势明确 → 趋势策略权重 +20%（价格行为增强）
    - 震荡市场 → 基本面权重 +10%（均值回归更有效）
    """

    # 基础权重配置
    BASE_WEIGHTS = AgentWeights(
        fundamental=0.30,
        quant=0.30,
        price_behavior=0.25,
        backtest=0.15,
    )

    def __init__(self) -> None:
        self.base = self.BASE_WEIGHTS

    def compute(
        self,
        volatility_level: Literal["high", "medium", "low"] = "medium",
        trend_strength: float = 0.0,  # -1（下跌）~ 0（震荡）~ +1（上涨）
        market_state: str = "range",   # 'trend_up' / 'trend_down' / 'range' / 'channel'
        confidence_scores: Optional[Dict[str, float]] = None,
    ) -> AgentWeights:
        """
        计算动态权重

        Args:
            volatility_level: 波动率水平
            trend_strength: 趋势强度（-1 到 1）
            market_state: 市场状态
            confidence_scores: 各 Agent 的置信度（可选，用于微调）

        Returns:
            调整后的 AgentWeights
        """
        # 从基础权重开始
        w = AgentWeights(
            fundamental=self.base.fundamental,
            quant=self.base.quant,
            price_behavior=self.base.price_behavior,
            backtest=self.base.backtest,
        )

        # 1. 波动率调整
        if volatility_level == "high":
            # 高波动：基本面权重 +20%，量化权重 -10%，价格行为 -5%
            adj = 0.20
            w.fundamental += adj
            w.quant -= 0.10
            w.price_behavior -= 0.05
        elif volatility_level == "low":
            # 低波动：量化权重 +15%，基本面权重 -5%
            w.quant += 0.15
            w.fundamental -= 0.05
        # medium: 不调整

        # 2. 趋势强度调整
        if abs(trend_strength) > 0.5:
            # 强趋势：价格行为权重 +20%
            w.price_behavior += 0.20
            # 基本面权重略微降低
            w.fundamental -= 0.05
        elif abs(trend_strength) < 0.2:
            # 弱趋势（震荡）：基本面权重 +10%
            w.fundamental += 0.10
            w.price_behavior -= 0.05

        # 3. 市场状态调整
        if market_state in ("trend_up", "trend_down"):
            # 趋势市场：价格行为权重 +10%
            w.price_behavior += 0.10
            w.backtest -= 0.05
        elif market_state == "range":
            # 震荡市场：量化权重 +10%，价格行为权重 -5%
            w.quant += 0.10
            w.price_behavior -= 0.05

        # 4. 置信度微调（可选）
        if confidence_scores:
            # 置信度高的 Agent 权重 +5%，低的 -5%
            avg_conf = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.5
            for key in ["fundamental", "quant", "price_behavior", "backtest"]:
                conf = confidence_scores.get(key, avg_conf)
                if conf > avg_conf + 0.1:
                    adjustment = 0.05
                    if key == "fundamental":
                        w.fundamental += adjustment
                    elif key == "quant":
                        w.quant += adjustment
                    elif key == "price_behavior":
                        w.price_behavior += adjustment
                    elif key == "backtest":
                        w.backtest += adjustment

        # 归一化
        return w.normalize()

    def get_adjusted_direction(
        self,
        direction_scores: Dict[str, float],
        weights: AgentWeights,
    ) -> tuple[str, float]:
        """
        综合各 Agent 方向评分，计算加权方向

        Args:
            direction_scores: 各 Agent 的方向评分（-1 到 1）
                e.g. {"fundamental": 0.8, "quant": -0.3, "price_behavior": 0.5}
            weights: 各 Agent 权重

        Returns:
            (final_direction, weighted_score)
        """
        score = (
            direction_scores.get("fundamental", 0) * weights.fundamental
            + direction_scores.get("quant", 0) * weights.quant
            + direction_scores.get("price_behavior", 0) * weights.price_behavior
            + direction_scores.get("backtest", 0) * weights.backtest
        )

        if score > 0.2:
            direction = "long"
        elif score < -0.2:
            direction = "short"
        else:
            direction = "neutral"

        return direction, float(score)

    def determine_regime(
        self,
        adx: Optional[float] = None,
        volatility: Optional[float] = None,
        trend_direction: int = 0,
    ) -> str:
        """
        Determine market regime based on volatility and trend signals.

        Returns:
            'high_volatility' / 'trending' / 'range'
        """
        # High volatility: if volatility > 3%
        if volatility is not None and volatility > 0.03:
            return 'high_volatility'
        # Trending: non-zero direction and ADX > 25 (or no ADX given)
        if trend_direction != 0 and (adx is None or adx > 25):
            return 'trending'
        # Default: range
        return 'range'

    def get_weights(self, regime: str) -> AgentWeights:
        """
        Get agent weights based on market regime.

        Args:
            regime: 'high_volatility' / 'trending' / 'range'

        Returns:
            AgentWeights
        """
        if regime == 'high_volatility':
            return AgentWeights(
                fundamental=0.45,
                quant=0.20,
                price_behavior=0.20,
                backtest=0.15,
            )
        elif regime == 'trending':
            return AgentWeights(
                fundamental=0.25,
                quant=0.30,
                price_behavior=0.30,
                backtest=0.15,
            )
        else:  # range
            return AgentWeights(
                fundamental=0.25,
                quant=0.35,
                price_behavior=0.25,
                backtest=0.15,
            )
