"""
波动率分析器

功能：
- ATR14 计算
- 布林带（20 日，±2 标准差）
- 波动率收缩度：当前 ATR / 20日 ATR 均值
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


class VolatilityAnalyzer:
    """
    波动率分析器

    计算并提供波动率相关指标。
    """

    def __init__(self, atr_period: int = 14, bb_period: int = 20, bb_std: float = 2.0) -> None:
        """
        初始化波动率分析器

        Args:
            atr_period: ATR 周期，默认 14
            bb_period: 布林带周期，默认 20
            bb_std: 布林带标准差倍数，默认 2.0
        """
        self.atr_period = atr_period
        self.bb_period = bb_period
        self.bb_std = bb_std

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        执行波动率分析

        Args:
            df: 价格数据（OHLCV DataFrame）

        Returns:
            包含 atr、atr_ratio、bb_upper、bb_lower 等的字典
        """
        # 提取数据
        if isinstance(df, pd.DataFrame):
            if "close" in df.columns:
                close = df["close"]
            else:
                close = df.iloc[:, 0]

            high = df["high"] if "high" in df.columns else close
            low = df["low"] if "low" in df.columns else close
        else:
            close = df
            high = df
            low = df

        # ATR14
        atr = self._compute_atr(high, low, close)

        # ATR 均值比（当前 / 20日均值）
        atr_ma = atr.rolling(20).mean()
        atr_ratio = atr.iloc[-1] / atr_ma.iloc[-1] if not atr_ma.isna().iloc[-1] else 1.0

        # 布林带
        bb_upper, bb_middle, bb_lower = self._compute_bollinger(close)

        # 波动率等级
        atr_pct = atr.iloc[-1] / close.iloc[-1] if close.iloc[-1] != 0 else 0
        if atr_pct > 0.03:
            vol_level = "high"
        elif atr_pct > 0.015:
            vol_level = "medium"
        else:
            vol_level = "low"

        return {
            "atr": round(float(atr.iloc[-1]), 4) if not pd.isna(atr.iloc[-1]) else 0.0,
            "atr_ma20": round(float(atr_ma.iloc[-1]), 4) if not pd.isna(atr_ma.iloc[-1]) else 0.0,
            "atr_ratio": round(float(atr_ratio), 4) if not pd.isna(atr_ratio) else 1.0,
            "bb_upper": round(float(bb_upper.iloc[-1]), 4) if bb_upper is not None and not pd.isna(bb_upper.iloc[-1]) else None,
            "bb_middle": round(float(bb_middle.iloc[-1]), 4) if bb_middle is not None and not pd.isna(bb_middle.iloc[-1]) else None,
            "bb_lower": round(float(bb_lower.iloc[-1]), 4) if bb_lower is not None and not pd.isna(bb_lower.iloc[-1]) else None,
            "vol_level": vol_level,
            "atr_pct": round(float(atr_pct), 4),
        }

    def _compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        计算 ATR（Average True Range）

        ATR = Wilder 平滑（EMA 变体）的 True Range

        Args:
            high, low, close: 价格序列

        Returns:
            ATR Series
        """
        # True Range = max(H-L, |H-PC|, |L-PC|)
        prev_close = close.shift(1).fillna(close)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder 平滑（使用 EMA 近似，period=atr_period）
        # ATR = (prev_ATR * (period-1) + TR) / period
        atr = tr.ewm(alpha=1.0 / self.atr_period, adjust=False).mean()
        return atr

    def _compute_bollinger(
        self, close: pd.Series
    ) -> tuple:
        """
        计算布林带

        Args:
            close: 收盘价序列

        Returns:
            (bb_upper, bb_middle, bb_lower)
        """
        if len(close) < self.bb_period:
            return None, None, None

        middle = close.rolling(self.bb_period).mean()
        std = close.rolling(self.bb_period).std()
        upper = middle + std * self.bb_std
        lower = middle - std * self.bb_std

        return upper, middle, lower
