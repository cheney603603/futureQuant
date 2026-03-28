"""
流动性模型 (Liquidity Model)

流动性约束和成本计算：
- Amihud 非流动性比率
- 成交量占比约束
- 冲击成本估算
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...core.logger import get_logger

logger = get_logger('agent.liquidity_model')


class LiquidityConstraint:
    """流动性约束"""

    def __init__(
        self,
        max_position_pct: float = 0.05,
        max_turnover_pct: float = 0.20,
        min_dollar_volume: float = 1e6,
    ) -> None:
        """
        Args:
            max_position_pct: 单品种最大持仓比例
            max_turnover_pct: 最大换手率
            min_dollar_volume: 最小日成交额
        """
        self.max_position_pct = max_position_pct
        self.max_turnover_pct = max_turnover_pct
        self.min_dollar_volume = min_dollar_volume

    def apply(
        self,
        positions: pd.Series,
        dollar_volume: pd.Series,
    ) -> pd.Series:
        """应用流动性约束"""
        constrained = positions.copy()

        # 成交额约束
        if dollar_volume is not None:
            tradable = (dollar_volume > self.min_dollar_volume).astype(float)
            constrained = constrained * tradable

        # 仓位约束
        constrained = constrained.clip(-self.max_position_pct, self.max_position_pct)

        return constrained


class LiquidityModel:
    """流动性模型"""

    def __init__(
        self,
        amihud_window: int = 20,
        volume_window: int = 20,
    ) -> None:
        """
        Args:
            amihud_window: Amihud 非流动性比率窗口
            volume_window: 成交量窗口
        """
        self.amihud_window = amihud_window
        self.volume_window = volume_window

    def calculate_illiquidity(
        self,
        returns: pd.Series,
        volume: pd.Series,
        price: pd.Series,
    ) -> pd.Series:
        """
        计算 Amihud 非流动性比率

        Args:
            returns: 收益率
            volume: 成交量
            price: 价格

        Returns:
            非流动性比率序列
        """
        abs_ret = returns.abs()
        turnover = volume / volume.rolling(self.volume_window).mean()
        illiq = abs_ret / (turnover + 1e-8)
        return illiq.rolling(self.amihud_window).mean()

    def estimate_market_impact(
        self,
        trade_value: float,
        avg_daily_volume: float,
        volatility: float,
        model: str = 'square_root',
    ) -> float:
        """
        估算市场冲击成本

        Args:
            trade_value: 交易金额
            avg_daily_volume: 日均成交额
            volatility: 波动率
            model: 冲击模型，'linear' / 'square_root'

        Returns:
            冲击成本
        """
        participation = trade_value / (avg_daily_volume + 1e-8)

        if model == 'square_root':
            impact = volatility * np.sqrt(participation)
        else:
            impact = volatility * participation

        return impact * trade_value

    def calculate_liquidity_score(
        self,
        volume: pd.Series,
        price: pd.Series,
        returns: pd.Series,
    ) -> pd.Series:
        """
        计算综合流动性评分

        Args:
            volume: 成交量
            price: 价格
            returns: 收益率

        Returns:
            流动性评分 [0, 1]
        """
        dv = volume * price
        dv_ma = dv.rolling(self.volume_window).mean()
        dv_score = (dv / dv_ma).clip(0, 2)

        spread_proxy = returns.abs().rolling(5).std()
        spread_score = 1.0 / (1.0 + spread_proxy * 100)

        illiq = self.calculate_illiquidity(returns, volume, price)
        illiq_score = 1.0 / (1.0 + illiq * 1e6)

        score = (dv_score * 0.4 + spread_score * 0.3 + illiq_score * 0.3)
        return score.clip(0, 1)
