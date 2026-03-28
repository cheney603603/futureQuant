"""
市场状态分析器 (Market State Analyzer)

识别牛市/熊市/震荡市：
- 趋势检测
- 波动率状态
- 市场状态分类
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from ...core.logger import get_logger

logger = get_logger('agent.validators.market_state_analyzer')


class MarketStateAnalyzer:
    """市场状态分析器"""

    STATE_BULL = 1
    STATE_BEAR = -1
    STATE_NEUTRAL = 0

    def __init__(
        self,
        trend_window: int = 60,
        vol_window: int = 20,
        vol_high_percentile: float = 0.8,
        vol_low_percentile: float = 0.2,
    ) -> None:
        """
        Args:
            trend_window: 趋势判断窗口
            vol_window: 波动率窗口
            vol_high_percentile: 高波动率百分位阈值
            vol_low_percentile: 低波动率百分位阈值
        """
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.vol_high = vol_high_percentile
        self.vol_low = vol_low_percentile

    def classify(
        self,
        prices: pd.Series,
    ) -> pd.Series:
        """
        分类市场状态

        Args:
            prices: 价格序列

        Returns:
            市场状态序列 (1=牛市, 0=震荡, -1=熊市)
        """
        trend = self.detect_trend(prices)
        vol_state = self._volatility_state(prices)

        state = pd.Series(0, index=prices.index)

        # 趋势向上 + 低/中波动 = 牛市
        state[(trend > 0) & (vol_state <= 1)] = self.STATE_BULL
        # 趋势向下 + 低/中波动 = 熊市
        state[(trend < 0) & (vol_state <= 1)] = self.STATE_BEAR
        # 高波动 = 震荡
        state[vol_state == 2] = self.STATE_NEUTRAL

        return state

    def detect_trend(self, prices: pd.Series) -> pd.Series:
        """检测价格趋势"""
        ma_short = prices.rolling(20, min_periods=5).mean()
        ma_long = prices.rolling(self.trend_window, min_periods=10).mean()

        trend = pd.Series(0, index=prices.index)
        trend[ma_short > ma_long] = 1
        trend[ma_short < ma_long] = -1

        # 平滑
        return trend.rolling(5, min_periods=1).mean().clip(-1, 1)

    def detect_regime(
        self,
        returns: pd.Series,
    ) -> pd.DataFrame:
        """
        检测市场状态转换

        Args:
            returns: 收益率序列

        Returns:
            状态分析 DataFrame
        """
        vol = returns.rolling(self.vol_window).std() * np.sqrt(252)
        vol_ma = vol.rolling(self.vol_window * 2, min_periods=20).mean()

        regime = pd.DataFrame(index=returns.index)
        regime['volatility'] = vol
        regime['vol_ma'] = vol_ma
        regime['regime'] = 'normal'
        regime.loc[vol > vol_ma * 1.5, 'regime'] = 'high_vol'
        regime.loc[vol < vol_ma * 0.7, 'regime'] = 'low_vol'

        return regime

    def _volatility_state(self, prices: pd.Series) -> pd.Series:
        """计算波动率状态 0=低, 1=中, 2=高"""
        ret = prices.pct_change().dropna()
        vol = ret.rolling(self.vol_window, min_periods=5).std() * np.sqrt(252)
        vol_rank = vol.rolling(self.vol_window * 2, min_periods=20).rank(pct=True)

        state = pd.Series(1, index=prices.index)
        state[vol_rank > self.vol_high] = 2
        state[vol_rank < self.vol_low] = 0

        return state
