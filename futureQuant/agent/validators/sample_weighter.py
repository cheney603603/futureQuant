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

    DEFAULT_CONFIG: Dict[str, Any] = {
        'volatility_window': 20,
        'liquidity_window': 20,
        'volatility_percentile': 0.3,
        'liquidity_percentile': 0.3,
        'use_market_state': True,
        'trend_window': 20,
    }

    def __init__(
        self,
        volatility_window: int = 20,
        liquidity_window: int = 20,
        volatility_percentile: float = 0.3,
        liquidity_percentile: float = 0.3,
        use_market_state: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            volatility_window: 波动率计算窗口
            liquidity_window: 流动性计算窗口
            volatility_percentile: 波动率高阈值百分位
            liquidity_percentile: 流动性低阈值百分位
            use_market_state: 是否使用市场状态加权
            config: 可选配置字典，覆盖具名参数
        """
        # 合并默认配置
        self.config: Dict[str, Any] = dict(self.DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)

        # 从 config 或具名参数初始化属性
        self.volatility_window = int(self.config.get('volatility_window', volatility_window))
        self.liquidity_window = int(self.config.get('liquidity_window', liquidity_window))
        self.volatility_percentile = float(self.config.get('volatility_percentile', volatility_percentile))
        self.liquidity_percentile = float(self.config.get('liquidity_percentile', liquidity_percentile))
        self.use_market_state = bool(self.config.get('use_market_state', use_market_state))
        self.trend_window = int(self.config.get('trend_window', 20))

    # ------------------------------------------------------------------
    # 公共接口方法
    # ------------------------------------------------------------------

    def calculate_volatility_weights(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        基于波动率计算样本权重。

        高波动率时期降低权重（不稳定环境中因子信号噪声更大）。

        Args:
            data: 包含 'close' 列的 DataFrame（也可包含 'volume'）

        Returns:
            归一化权重序列 [0, 1]，与 data 索引对齐
        """
        if data.empty:
            return pd.Series(dtype=float)

        if 'close' not in data.columns:
            return pd.Series(1.0, index=data.index)

        returns = data['close'].pct_change().fillna(0)

        if len(returns) < self.volatility_window * 2:
            weights = pd.Series(1.0, index=data.index)
            return self._normalize_01(weights)

        vol = returns.rolling(self.volatility_window).std().fillna(0)
        vol_rank = vol.rank(pct=True).fillna(0.5)

        # 高波动 → 低权重
        raw_weights = 1.0 - vol_rank * 0.5
        weights = pd.Series(np.clip(raw_weights.values, 0.0, 1.0), index=data.index)
        return self._normalize_01(weights)

    def calculate_liquidity_weights(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        基于流动性（成交量）计算样本权重。

        低流动性时期降低权重（交易成本高，信号难以实现）。

        Args:
            data: 包含 'volume' 列的 DataFrame

        Returns:
            归一化权重序列 [0, 1]，与 data 索引对齐
        """
        if data.empty:
            return pd.Series(dtype=float)

        if 'volume' not in data.columns:
            return pd.Series(1.0, index=data.index)

        volume = data['volume'].fillna(0)
        if len(volume) < self.liquidity_window:
            return pd.Series(1.0, index=data.index)

        liq = volume.rolling(self.liquidity_window).mean().fillna(volume.mean())
        liq_rank = liq.rank(pct=True).fillna(0.5)

        # 高流动性 → 高权重
        raw_weights = 0.5 + liq_rank * 0.5
        weights = pd.Series(np.clip(raw_weights.values, 0.0, 1.0), index=data.index)
        return self._normalize_01(weights)

    def calculate_market_state_weights(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:
        """
        基于市场状态计算样本权重。

        识别趋势/震荡市场，趋势市场中趋势因子权重更高。

        Args:
            data: 包含 'close' 列的 DataFrame

        Returns:
            归一化权重序列 [0, 1]，与 data 索引对齐
        """
        if data.empty:
            return pd.Series(dtype=float)

        if 'close' not in data.columns:
            return pd.Series(1.0, index=data.index)

        close = data['close']
        if len(close) < self.trend_window:
            return pd.Series(1.0, index=data.index)

        # 用移动平均方向判断市场状态
        ma = close.rolling(self.trend_window).mean()
        trend_direction = (close > ma).astype(float)  # 1=趋势向上, 0=趋势向下

        # 趋势市场（连续同向）给予更高权重
        trend_strength = trend_direction.rolling(5).mean().fillna(0.5)
        raw_weights = 0.5 + (trend_strength - 0.5).abs()  # 越偏离0.5越倾向趋势
        weights = pd.Series(np.clip(raw_weights.values, 0.0, 1.0), index=data.index)
        return self._normalize_01(weights)

    def detect_market_state(self, data: pd.DataFrame) -> pd.Series:
        """
        检测市场状态序列。

        Returns:
            市场状态序列：1=上涨趋势, -1=下跌趋势, 0=震荡
        """
        if data.empty or 'close' not in data.columns:
            return pd.Series(dtype=int)

        close = data['close']
        ma = close.rolling(self.trend_window).mean()
        diff = (close - ma) / ma

        state = pd.Series(0, index=data.index)
        state[diff > 0.02] = 1
        state[diff < -0.02] = -1
        return state

    def calculate_combined_weights(
        self,
        data: pd.DataFrame,
        vol_weight: float = 0.4,
        liq_weight: float = 0.3,
        state_weight: float = 0.3,
    ) -> pd.Series:
        """
        综合计算加权权重（波动率 + 流动性 + 市场状态）。

        Args:
            data: 包含 close/volume 列的 DataFrame
            vol_weight: 波动率权重占比
            liq_weight: 流动性权重占比
            state_weight: 市场状态权重占比

        Returns:
            综合权重序列 [0, 1]，与 data 索引对齐
        """
        if data.empty:
            return pd.Series(dtype=float)

        vol_w = self.calculate_volatility_weights(data)
        liq_w = self.calculate_liquidity_weights(data)
        state_w = self.calculate_market_state_weights(data)

        # 对齐并加权求和
        combined = vol_w * vol_weight + liq_w * liq_weight + state_w * state_weight
        combined = combined.fillna(1.0)
        return self._normalize_01(combined)

    def apply_weights(
        self,
        factor_values: pd.Series,
        weights: pd.Series,
    ) -> pd.Series:
        """
        对因子值应用样本权重（返回加权后的因子值）。

        Args:
            factor_values: 因子值序列
            weights: 权重序列

        Returns:
            加权因子值序列
        """
        aligned_w = weights.reindex(factor_values.index).fillna(1.0)
        return factor_values * aligned_w

    # ------------------------------------------------------------------
    # 原有接口（保持向后兼容）
    # ------------------------------------------------------------------

    def calculate_weights(
        self,
        prices: pd.Series,
        returns: pd.Series,
        volume: Optional[pd.Series] = None,
        market_state: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        计算样本权重（原始接口，保持向后兼容）

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

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_01(weights: pd.Series) -> pd.Series:
        """将权重归一化到 [0, 1]"""
        wmin, wmax = weights.min(), weights.max()
        if wmax - wmin < 1e-8:
            return pd.Series(1.0, index=weights.index)
        return (weights - wmin) / (wmax - wmin)

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
