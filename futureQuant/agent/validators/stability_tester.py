"""
稳定性测试器 (Stability Tester)

因子稳定性检验：
- 滚动 IC 稳定性
- 分组收益单调性
- 时间衰减分析
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ...core.logger import get_logger

logger = get_logger('agent.validators.stability_tester')


class FactorStabilityTester:
    """因子稳定性测试器"""

    def __init__(
        self,
        rolling_window: int = 60,
        n_groups: int = 5,
        min_icir: float = 0.5,
    ) -> None:
        """
        Args:
            rolling_window: 滚动窗口大小
            n_groups: 分组数量
            min_icir: ICIR 最低阈值
        """
        self.rolling_window = rolling_window
        self.n_groups = n_groups
        self.min_icir = min_icir

    def test_stability(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        测试因子稳定性

        Args:
            factor_values: 因子值
            returns: 收益率

        Returns:
            稳定性测试结果
        """
        rolling_ic = self.rolling_ic_test(factor_values, returns)
        monotonic = self._test_monotonicity(factor_values, returns)
        decay = self._test_temporal_decay(factor_values, returns)

        overall_score = (
            rolling_ic['stability_score'] * 0.4 +
            monotonic['monotonicity_score'] * 0.3 +
            decay['decay_score'] * 0.3
        )

        return {
            'rolling_ic': rolling_ic,
            'monotonicity': monotonic,
            'temporal_decay': decay,
            'overall_stability': float(overall_score),
            'is_stable': overall_score >= 0.4,
        }

    def rolling_ic_test(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        滚动 IC 稳定性测试

        Args:
            factor_values: 因子值
            returns: 收益率

        Returns:
            滚动 IC 测试结果
        """
        aligned = pd.concat([factor_values, returns], axis=1).dropna()
        if len(aligned) < self.rolling_window:
            return {'error': 'insufficient data'}

        rolling_ic = aligned.iloc[:, 0].rolling(self.rolling_window).corr(aligned.iloc[:, 1])
        ic_arr = rolling_ic.dropna().values

        if len(ic_arr) < 5:
            return {'error': 'insufficient windows'}

        ic_mean = ic_arr.mean()
        ic_std = ic_arr.std()
        icir = abs(ic_mean) / (ic_std + 1e-8) if ic_std > 1e-8 else 0.0

        win_rate = (ic_arr > 0).mean()
        sign_changes = self._count_sign_changes(ic_arr)

        stability_score = min(icir / 3.0, 1.0) * 0.6 + win_rate * 0.4

        return {
            'ic_series': [float(x) for x in ic_arr],
            'ic_mean': float(ic_mean),
            'ic_std': float(ic_std),
            'icir': float(icir),
            'win_rate': float(win_rate),
            'sign_changes': int(sign_changes),
            'stability_score': float(stability_score),
        }

    def _test_monotonicity(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """测试分组收益单调性"""
        try:
            bins = pd.qcut(factor_values, self.n_groups, duplicates='drop')
            group_returns = returns.groupby(bins).mean()

            if len(group_returns) < 3:
                return {'monotonicity_score': 0.5}

            diffs = np.diff(group_returns.values)
            n_changes = np.sum(np.diff(np.sign(diffs)) != 0)

            if len(diffs) <= 1:
                monotonicity_score = 1.0
            else:
                monotonicity_score = max(1.0 - n_changes / (len(diffs) - 1), 0.0)

            return {
                'group_returns': {str(k): float(v) for k, v in group_returns.items()},
                'monotonicity_score': float(monotonicity_score),
                'n_reversals': int(n_changes),
            }
        except Exception:
            return {'monotonicity_score': 0.5}

    def _test_temporal_decay(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """测试因子时间衰减"""
        aligned = pd.concat([factor_values, returns], axis=1).dropna()
        half_life = self.rolling_window * 2

        recent_ic = self._calculate_ic(
            aligned.iloc[-half_life:, 0],
            aligned.iloc[-half_life:, 1]
        )
        early_ic = self._calculate_ic(
            aligned.iloc[:half_life, 0],
            aligned.iloc[:half_life, 1]
        )

        decay_ratio = recent_ic / (abs(early_ic) + 1e-8) if early_ic != 0 else 0.0

        return {
            'early_ic': float(early_ic),
            'recent_ic': float(recent_ic),
            'decay_ratio': float(decay_ratio),
            'decay_score': float(max(min(decay_ratio, 1.0), 0.0)),
        }

    def _calculate_ic(self, fv: pd.Series, rt: pd.Series) -> float:
        """计算 IC"""
        if len(fv) < 10:
            return 0.0
        corr, _ = stats.spearmanr(fv, rt)
        return corr if not np.isnan(corr) else 0.0

    def _count_sign_changes(self, arr: np.ndarray) -> int:
        """计算符号变化次数"""
        signs = np.sign(arr)
        return int(np.sum(np.diff(signs) != 0))
