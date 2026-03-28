"""
多维度评分器 (Multi-Dimensional Scorer)

从五个维度评估因子质量：
- 预测能力：IC 均值与 ICIR
- 稳定性：IC 时间序列的稳定性
- 单调性：因子与收益的单调关系
- 换手率：因子换手导致的交易成本
- 风险调整：因子在极端情况下的表现
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ...core.logger import get_logger
from ...core.base import Factor

logger = get_logger('agent.scorer')


class MultiDimensionalScorer:
    """
    多维度因子评分器

    综合评估因子质量，输出 0-1 评分。
    """

    DEFAULT_WEIGHTS = {
        'predictability': 0.35,
        'stability': 0.25,
        'monotonicity': 0.20,
        'turnover': 0.10,
        'risk': 0.10,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_ic_threshold: float = 0.01,
        min_icir_threshold: float = 0.3,
    ) -> None:
        """
        Args:
            weights: 各维度权重字典
            min_ic_threshold: IC 均值最低阈值
            min_icir_threshold: ICIR 最低阈值
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            for k in self.weights:
                self.weights[k] /= total
        self.min_ic_threshold = min_ic_threshold
        self.min_icir_threshold = min_icir_threshold

    def score(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        计算综合评分

        Args:
            factor_values: 因子值
            returns: 收益率
            volume: 成交量（可选）

        Returns:
            评分字典
        """
        aligned = pd.concat([factor_values, returns], axis=1).dropna()
        if len(aligned) < 30:
            return self._empty_score()

        fv = aligned.iloc[:, 0]
        rt = aligned.iloc[:, 1]

        pred = self._predictability_score(fv, rt)
        stab = self._stability_score(fv, rt)
        mono = self._monotonicity_score(fv, rt)
        turn = self._turnover_score(fv) if volume is not None else 0.5
        risk = self._risk_score(fv, rt)

        scores = {
            'predictability': pred,
            'stability': stab,
            'monotonicity': mono,
            'turnover': turn,
            'risk': risk,
        }

        overall = sum(scores[k] * self.weights[k] for k in scores)

        return {
            **scores,
            'overall': overall,
            'weights': self.weights.copy(),
        }

    def _empty_score(self) -> Dict[str, Any]:
        scores = {k: 0.0 for k in self.DEFAULT_WEIGHTS}
        scores['overall'] = 0.0
        scores['weights'] = self.weights.copy()
        return scores

    def _predictability_score(self, fv: pd.Series, rt: pd.Series) -> float:
        """预测能力评分"""
        corr, pval = stats.spearmanr(fv, rt)
        if np.isnan(corr):
            return 0.0

        ic = abs(corr)
        icir = self._calculate_icir(fv, rt)
        if ic < self.min_ic_threshold or icir < self.min_icir_threshold:
            return 0.0

        ic_score = min(ic / 0.1, 1.0)
        icir_score = min(icir / 2.0, 1.0)
        return (ic_score * 0.5 + icir_score * 0.5)

    def _stability_score(self, fv: pd.Series, rt: pd.Series) -> float:
        """稳定性评分"""
        ic_series = fv.rolling(20).corr(rt)
        ic_arr = ic_series.dropna().values
        if len(ic_arr) < 5:
            return 0.0

        ic_mean = np.abs(ic_arr.mean())
        ic_std = ic_arr.std() if len(ic_arr) > 1 else 1e-8
        if ic_std < 1e-8:
            return 1.0

        icir = ic_mean / ic_std
        return min(icir / 2.0, 1.0)

    def _monotonicity_score(self, fv: pd.Series, rt: pd.Series) -> float:
        """单调性评分：分组收益是否单调"""
        try:
            n_groups = 5
            bins = pd.qcut(fv, n_groups, duplicates='drop')
            group_returns = rt.groupby(bins).mean()

            if len(group_returns) < 3:
                return 0.5

            # 计算单调性：相邻组收益应该递增或递减
            diffs = np.diff(group_returns.values)
            sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)

            if len(diffs) == 0:
                return 0.5

            monotonic_ratio = 1.0 - sign_changes / (len(diffs) - 1)
            return max(monotonic_ratio, 0.0)
        except Exception:
            return 0.5

    def _turnover_score(self, fv: pd.Series) -> float:
        """换手率评分：换手越低越好"""
        turns = fv.pct_change().abs()
        avg_turn = turns.dropna().mean()
        if avg_turn < 1e-8:
            return 1.0
        return max(1.0 - avg_turn * 10, 0.0)

    def _risk_score(self, fv: pd.Series, rt: pd.Series) -> float:
        """风险评分"""
        pct_change = fv.pct_change().dropna()
        if len(pct_change) < 5:
            return 0.5

        # 最大回撤评分
        cum = (1 + pct_change).cumprod()
        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        max_dd = abs(drawdown.min())

        # 波动率评分
        vol = pct_change.std()
        vol_penalty = min(vol * 5, 1.0)

        dd_score = max(1.0 - max_dd * 5, 0.0)
        return (dd_score * 0.6 + (1 - vol_penalty) * 0.4)

    def _calculate_icir(self, fv: pd.Series, rt: pd.Series) -> float:
        """计算 ICIR"""
        ic_series = fv.rolling(20).corr(rt)
        ic_arr = ic_series.dropna().values
        if len(ic_arr) < 5 or ic_arr.std() < 1e-8:
            return 0.0
        return abs(ic_arr.mean()) / ic_arr.std()

    def rank_factors(
        self,
        factors: list[Dict[str, Any]],
    ) -> list[Dict[str, Any]]:
        """
        对多个因子评分并排序

        Args:
            factors: 因子列表，每个元素包含 factor_values 和 returns

        Returns:
            按综合评分排序的因子列表
        """
        scored = []
        for f in factors:
            score = self.score(
                factor_values=f['factor_values'],
                returns=f['returns'],
                volume=f.get('volume'),
            )
            scored.append({**f, **score})

        scored.sort(key=lambda x: x['overall'], reverse=True)
        return scored
