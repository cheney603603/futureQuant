"""
压力测试器 (Stress Tester)

极端市场情况测试：
- 历史危机测试
- 情景分析
- 黑天鹅测试
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ...core.logger import get_logger

logger = get_logger('agent.validators.stress_tester')


class StressTester:
    """压力测试器"""

    CRISIS_PERIODS = {
        '2008_financial': ('2008-01-01', '2009-12-31'),
        '2010_flash_crash': ('2010-05-01', '2010-06-30'),
        '2015_china_crash': ('2015-06-01', '2016-02-29'),
        '2020_covid': ('2020-01-01', '2020-12-31'),
        '2022_rate_hike': ('2022-01-01', '2022-12-31'),
    }

    def __init__(
        self,
        drawdown_threshold: float = 0.10,
        vol_multiplier: float = 3.0,
    ) -> None:
        """
        Args:
            drawdown_threshold: 最大回撤阈值
            vol_multiplier: 波动率异常倍数
        """
        self.drawdown_threshold = drawdown_threshold
        self.vol_multiplier = vol_multiplier

    def stress_test(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        执行压力测试

        Args:
            factor_values: 因子值
            returns: 收益率

        Returns:
            压力测试结果
        """
        crisis_results = {}
        for name, (start, end) in self.CRISIS_PERIODS.items():
            mask = (factor_values.index >= start) & (factor_values.index <= end)
            if mask.sum() > 20:
                crisis_results[name] = self._test_period(
                    factor_values[mask],
                    returns[mask]
                )

        black_swan = self._black_swan_test(factor_values, returns)
        scenario = self._scenario_analysis(factor_values, returns)

        overall_score = self._calculate_stress_score(
            crisis_results, black_swan, scenario
        )

        return {
            'crisis_periods': crisis_results,
            'black_swan': black_swan,
            'scenario_analysis': scenario,
            'overall_stress_score': float(overall_score),
            'is_resilient': overall_score >= 0.4,
        }

    def crisis_test(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        period_name: str,
    ) -> Dict[str, Any]:
        """
        测试特定危机时期

        Args:
            factor_values: 因子值
            returns: 收益率
            period_name: 危机时期名称

        Returns:
            危机测试结果
        """
        if period_name not in self.CRISIS_PERIODS:
            return {'error': f'Unknown period: {period_name}'}

        start, end = self.CRISIS_PERIODS[period_name]
        mask = (factor_values.index >= start) & (factor_values.index <= end)

        if mask.sum() < 10:
            return {'error': 'Insufficient data in period'}

        return self._test_period(factor_values[mask], returns[mask])

    def _test_period(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """测试特定时期"""
        aligned = pd.concat([factor_values, returns], axis=1).dropna()
        if len(aligned) < 10:
            return {'error': 'insufficient data'}

        ic, pval = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        ic = ic if not np.isnan(ic) else 0.0

        pct_change = aligned.iloc[:, 0].pct_change().dropna()
        cum = (1 + pct_change).cumprod()
        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        max_dd = abs(drawdown.min())

        return {
            'period_ic': float(ic),
            'max_drawdown': float(max_dd),
            'n_days': int(len(aligned)),
            'survived': max_dd < self.drawdown_threshold,
        }

    def _black_swan_test(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        黑天鹅事件测试

        找出历史上最极端的 5% 收益日，测试因子在这些日子的表现
        """
        pct_ret = returns.pct_change().dropna()
        threshold = pct_ret.quantile(0.95)
        crash_days = pct_ret[pct_ret >= threshold]

        if len(crash_days) < 5:
            return {'error': 'insufficient crash days'}

        aligned = pd.concat([factor_values, returns], axis=1).dropna()
        crash_mask = aligned.iloc[:, 1] >= threshold

        crash_ic, crash_pval = stats.spearmanr(
            aligned[crash_mask].iloc[:, 0],
            aligned[crash_mask].iloc[:, 1]
        )
        normal_ic, _ = stats.spearmanr(
            aligned[~crash_mask].iloc[:, 0],
            aligned[~crash_mask].iloc[:, 1]
        )

        return {
            'crash_days_ic': float(crash_ic) if not np.isnan(crash_ic) else 0.0,
            'normal_days_ic': float(normal_ic) if not np.isnan(normal_ic) else 0.0,
            'n_crash_days': int(crash_mask.sum()),
            'crash_resilient': abs(crash_ic) > abs(normal_ic) * 0.5,
        }

    def _scenario_analysis(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        情景分析

        模拟不同市场情景下的因子表现
        """
        pct_change = factor_values.pct_change().dropna()
        vol = pct_change.std()

        scenarios = {
            'normal': vol,
            'high_vol': vol * 2,
            'extreme': vol * 3,
        }

        results = {}
        for name, vol_level in scenarios.items():
            simulated_impact = vol_level * pct_change.std() * 0.1
            impact_score = 1.0 - min(simulated_impact / vol, 1.0)
            results[name] = {
                'volatility': float(vol_level),
                'impact_score': float(impact_score),
            }

        return results

    def _calculate_stress_score(
        self,
        crisis_results: Dict[str, Any],
        black_swan: Dict[str, Any],
        scenario: Dict[str, Any],
    ) -> float:
        """计算综合压力测试评分"""
        if not crisis_results:
            return 0.5

        crisis_scores = [
            1.0 if r.get('survived', False) else max(0, 1 - r.get('max_drawdown', 1) * 5)
            for r in crisis_results.values()
            if not r.get('error')
        ]

        if not crisis_scores:
            return 0.5

        crisis_avg = np.mean(crisis_scores)

        swan_score = 1.0 if black_swan.get('crash_resilient', False) else 0.3

        scenario_scores = [
            v.get('impact_score', 0.5) for v in scenario.values()
        ]
        scenario_avg = np.mean(scenario_scores) if scenario_scores else 0.5

        return crisis_avg * 0.5 + swan_score * 0.3 + scenario_avg * 0.2
