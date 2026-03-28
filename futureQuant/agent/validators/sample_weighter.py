"""
样本权重计算器 (Sample Weighter)

根据市场状态对样本进行加权：
- 波动率加权：高波动时期降低权重
- 流动性加权：低流动性时期降低权重
- 市场状态加权：根据市场状态调整

权重调整后的 IC 计算能更准确地反映因子在不同市场环境下的表现。
"""

from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from ...core.logger import get_logger
from ...core.base import Factor

logger = get_logger('agent.sample_weighter')


class SampleWeighter:
    """
    样本权重计算器

    根据波动率、流动性、市场状态等因素计算样本权重，
    用于调整因子评估时的权重分布。
    """

    def __init__(
        self,
        volatility_window: int = 20,
        liquidity_window: int = 20,
        volatility_percentile: float = 0.3,
        liquidity_percentile: float = 0.3,
        use_market_state: bool = True,
    ) -> None:
        """
        Args:
            volatility_window: 波动率计算窗口
            liquidity_window: 流动性计算窗口
            volatility_percentile: 波动率高阈值百分位
            liquidity_percentile: 流动性低阈值百分位
            use_market_state: 是否使用市场状态加权
        """
        self.volatility_window = volatility_window
        self.liquidity_window = liquidity_window
        self.volatility_percentile = volatility_percentile
        self.liquidity_percentile = liquidity_percentile
        self.use_market_state = use_market_state

    def calculate_weights(
        self,
        prices: pd.Series,
        returns: pd.Series,
        volume: Optional[pd.Series] = None,
        market_state: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        计算样本权重

        Args:
            prices: 价格序列
            returns: 收益率序列
            volume: 成交量序列（可选）
            market_state: 市场状态序列（可选，1=牛市, 0=震荡, -1=熊市）

        Returns:
            权重序列
        """
        if len(returns) < self.volatility_window * 2:
            return pd.Series(1.0, index=returns.index)

        vol = returns.rolling(self.volatility_window).std()
        vol_rank = vol.rank(pct=True)

        # 波动率加权：高波动期降低权重
        vol_weights = np.where(
            vol_rank > (1 - self.volatility_percentile),
            1 - (vol_rank - (1 - self.volatility_percentile)) / self.volatility_percentile * 0.5,
            1.0,
        )
        vol_weights = np.clip(vol_weights, 0.3, 1.0)

        weights = pd.Series(vol_weights, index=returns.index)

        # 流动性加权
        if volume is not None:
            liq = volume.rolling(self.liquidity_window).mean()
            liq_rank = liq.rank(pct=True)
            liq_weights = np.where(
                liq_rank < self.liquidity_percentile,
                0.5 + liq_rank / self.liquidity_percentile * 0.5,
                1.0,
            )
            weights *= np.clip(liq_weights, 0.5, 1.0)

        # 市场状态加权
        if self.use_market_state and market_state is not None:
            state_weights = np.where(
                market_state == 0, 0.7,
                np.where(market_state == -1, 0.8, 1.0),
            )
            weights *= state_weights

        # 归一化
        weights = weights / weights.mean()
        return weights

    def weighted_ic(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        weights: pd.Series,
    ) -> float:
        """
        计算加权 IC

        Args:
            factor_values: 因子值
            returns: 收益率
            weights: 样本权重

        Returns:
            加权 IC 值
        """
        aligned = pd.concat([factor_values, returns, weights], axis=1).dropna()
        if len(aligned) < 20:
            return 0.0

        fv = aligned.iloc[:, 0]
        rt = aligned.iloc[:, 1]
        wt = aligned.iloc[:, 2]

        corr, _ = stats.spearmanr(fv, rt)
        return corr if not np.isnan(corr) else 0.0

    def get_contribution(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        weights: pd.Series,
        window: int = 60,
    ) -> pd.DataFrame:
        """
        计算各样本对 IC 的贡献

        Args:
            factor_values: 因子值
            returns: 收益率
            weights: 样本权重
            window: 滚动窗口

        Returns:
            IC 贡献度 DataFrame
        """
        aligned = pd.concat([factor_values, returns, weights], axis=1).dropna()
        if len(aligned) < window:
            return pd.DataFrame()

        ic_series = aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1])
        contribution = ic_series * aligned.iloc[:, 2]
        contribution = contribution / contribution.abs().sum()

        return pd.DataFrame({
            'ic': ic_series,
            'weighted_ic': contribution,
        })
