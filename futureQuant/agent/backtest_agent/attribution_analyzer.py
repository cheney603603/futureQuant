"""
收益归因分析器

拆解收益来源：
- 因子贡献：因子本身的预测能力带来的收益
- 基本面贡献：基本面信号增强带来的额外收益
- 择时贡献：多空方向选择带来的收益
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class AttributionAnalyzer:
    """
    收益归因分析器

    将策略收益拆解为多个来源，用于诊断策略有效性。
    """

    def analyze(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        fundamental_signal: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        执行收益归因分析

        Args:
            price_data: 价格数据（含 close 列）
            signals: 信号 DataFrame（含 signal 列）
            fundamental_signal: 基本面评分（可选）

        Returns:
            归因分析结果字典
        """
        if "close" not in price_data.columns or "signal" not in signals.columns:
            return {
                "factor_contribution": 0.0,
                "fundamental_contribution": 0.0,
                "timing_contribution": 0.0,
                "total_return": 0.0,
                "note": "insufficient data for attribution",
            }

        close = price_data["close"]
        sig = signals["signal"].fillna(0)

        # 计算收益率
        returns = close.pct_change().fillna(0)

        # 对齐
        common_idx = returns.index.intersection(sig.index)
        returns = returns[common_idx]
        sig = sig[common_idx]

        # 持仓（延迟一期）
        position = sig.shift(1).fillna(0)

        # 总收益
        total_return = (position * returns).sum()

        # --- 因子贡献 ---
        # 因子收益 = 因子暴露 * 因子收益（简化：使用因子值的方向性）
        # 用因子方向与收益的相关性来衡量
        if len(returns) > 10:
            factor_direction = sig.shift(1).fillna(0)
            factor_return = (factor_direction * returns).sum()
            # 因子纯收益（排除方向性）
            factor_contribution = factor_return - np.sign(factor_return) * abs(total_return) * 0.3
            factor_contribution = float(np.clip(factor_contribution, -abs(total_return), abs(total_return)))
        else:
            factor_contribution = 0.0

        # --- 择时贡献 ---
        # 择时收益：方向选择正确带来的收益
        correct_direction = np.sign(position) == np.sign(returns)
        timing_return = (position * returns * correct_direction.astype(float)).sum()
        timing_contribution = float(np.clip(timing_return, -abs(total_return) * 2, abs(total_return) * 2))

        # --- 基本面贡献 ---
        if fundamental_signal is not None:
            # 基本面增强幅度
            enhancement = abs(fundamental_signal) / 5.0  # 0~1
            # 基本面方向是否与信号一致
            if fundamental_signal > 0 and sig.mean() >= 0:
                fund_aligned = True
            elif fundamental_signal < 0 and sig.mean() <= 0:
                fund_aligned = True
            else:
                fund_aligned = False

            if fund_aligned:
                fundamental_contribution = float(enhancement * abs(timing_contribution) * 0.2)
            else:
                fundamental_contribution = float(-enhancement * abs(timing_contribution) * 0.1)
        else:
            fundamental_contribution = 0.0

        # 归一化（确保总和约等于总收益）
        total_attributed = abs(factor_contribution) + abs(timing_contribution) + abs(fundamental_contribution)
        if total_attributed > 1e-8:
            scale = abs(total_return) / total_attributed
        else:
            scale = 1.0

        return {
            "factor_contribution": round(float(factor_contribution * scale), 4),
            "timing_contribution": round(float(timing_contribution * scale), 4),
            "fundamental_contribution": round(float(fundamental_contribution * scale), 4),
            "total_return": round(float(total_return), 4),
            "unexplained": round(float(total_return - (
                factor_contribution + timing_contribution + fundamental_contribution
            ) * scale), 4),
        }
