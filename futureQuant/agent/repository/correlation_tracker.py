"""
相关性追踪器 (Correlation Tracker)

追踪因子之间的相关性：
- 因子相关性矩阵
- 相关性变化分析
- 相关性报告
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from ...core.logger import get_logger

logger = get_logger('agent.correlation_tracker')


@dataclass
class CorrelationPair:
    """相关性因子对"""
    factor_a: str
    factor_b: str
    correlation: float
    p_value: float
    is_significant: bool
    window: int = 60


@dataclass
class CorrelationReport:
    """相关性分析报告"""
    matrix: pd.DataFrame
    high_pairs: List[CorrelationPair]
    changing_pairs: List[Dict[str, Any]]
    summary: Dict[str, float]


class CorrelationTracker:
    """
    因子相关性追踪器

    追踪和报告因子间的相关性，帮助避免过度冗余的因子组合。
    """

    def __init__(
        self,
        window: int = 60,
        significance_threshold: float = 0.05,
        correlation_threshold: float = 0.8,
    ) -> None:
        """
        Args:
            window: 滚动窗口大小
            significance_threshold: 显著性阈值
            correlation_threshold: 高相关性阈值
        """
        self.window = window
        self.significance_threshold = significance_threshold
        self.correlation_threshold = correlation_threshold

    def calculate_matrix(
        self,
        factor_dict: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """
        计算因子相关性矩阵

        Args:
            factor_dict: 因子名称到因子值的字典

        Returns:
            相关性矩阵 DataFrame
        """
        df = pd.DataFrame(factor_dict)
        corr = df.rolling(self.window).corr()
        return corr

    def find_high_correlation_pairs(
        self,
        factor_dict: Dict[str, pd.Series],
        threshold: Optional[float] = None,
    ) -> List[CorrelationPair]:
        """
        找出高相关性因子对

        Args:
            factor_dict: 因子字典
            threshold: 相关系数阈值

        Returns:
            高相关性因子对列表
        """
        threshold = threshold or self.correlation_threshold
        df = pd.DataFrame(factor_dict)
        corr = df.corr()

        pairs: List[CorrelationPair] = []
        factors = list(corr.columns)

        for i, f_a in enumerate(factors):
            for j, f_b in enumerate(factors):
                if i >= j:
                    continue
                r = corr.loc[f_a, f_b]
                if abs(r) >= threshold:
                    _, pval = stats.spearmanr(df[f_a].dropna(), df[f_b].dropna())
                    pairs.append(CorrelationPair(
                        factor_a=f_a,
                        factor_b=f_b,
                        correlation=r,
                        p_value=pval,
                        is_significant=pval < self.significance_threshold,
                    ))

        pairs.sort(key=lambda x: abs(x.correlation), reverse=True)
        return pairs

    def track_correlation_change(
        self,
        factor_a: pd.Series,
        factor_b: pd.Series,
        windows: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        追踪两个因子相关性的变化

        Args:
            factor_a: 因子 A
            factor_b: 因子 B
            windows: 不同窗口列表

        Returns:
            相关性变化分析
        """
        windows = windows or [20, 60, 120]
        aligned = pd.concat([factor_a, factor_b], axis=1).dropna()

        if len(aligned) < max(windows):
            return {'error': 'insufficient data'}

        results = {}
        for w in windows:
            if len(aligned) < w:
                continue
            recent = aligned.iloc[-w:]
            r, pval = stats.spearmanr(recent.iloc[:, 0], recent.iloc[:, 1])
            results[f'corr_{w}'] = float(r) if not np.isnan(r) else 0.0
            results[f'pval_{w}'] = float(pval) if not np.isnan(pval) else 1.0

        rolling = aligned.iloc[:, 0].rolling(20).corr(aligned.iloc[:, 1])
        results['change_std'] = float(rolling.dropna().std()) if len(rolling.dropna()) > 1 else 0.0
        results['trend'] = 'increasing' if rolling.diff().mean() > 0 else 'decreasing'

        return results

    def generate_report(
        self,
        factor_dict: Dict[str, pd.Series],
    ) -> CorrelationReport:
        """
        生成完整相关性报告

        Args:
            factor_dict: 因子字典

        Returns:
            CorrelationReport 对象
        """
        df = pd.DataFrame(factor_dict)
        
        # 计算整体相关性矩阵（不是 rolling）
        corr_matrix = df.corr()
        
        high_pairs = self.find_high_correlation_pairs(factor_dict)

        changing_pairs = []
        factors = list(factor_dict.keys())
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                change = self.track_correlation_change(
                    factor_dict[factors[i]],
                    factor_dict[factors[j]],
                )
                if 'change_std' in change and change['change_std'] > 0.1:
                    changing_pairs.append({
                        'factor_a': factors[i],
                        'factor_b': factors[j],
                        **change,
                    })

        # 获取上三角矩阵的索引
        upper_indices = np.triu_indices_from(corr_matrix.values, 1)
        upper_values = corr_matrix.values[upper_indices]
        
        summary = {
            'mean_correlation': float(np.mean(np.abs(upper_values))) if len(upper_values) > 0 else 0.0,
            'max_correlation': float(np.max(np.abs(upper_values))) if len(upper_values) > 0 else 0.0,
            'high_correlation_count': len(high_pairs),
            'changing_pairs_count': len(changing_pairs),
        }

        return CorrelationReport(
            matrix=corr_matrix,
            high_pairs=high_pairs,
            changing_pairs=changing_pairs,
            summary=summary,
        )

    def recommend_factor_removal(
        self,
        factor_dict: Dict[str, pd.Series],
        keep_best: bool = True,
    ) -> List[str]:
        """
        建议需要移除的因子（避免高度冗余）

        Args:
            factor_dict: 因子字典
            keep_best: True 保留 IC 最高的，False 保留相关性最低的

        Returns:
            建议移除的因子名称列表
        """
        pairs = self.find_high_correlation_pairs(factor_dict)
        to_remove: set = set()

        for pair in pairs:
            if keep_best:
                to_remove.add(pair.factor_a)
            else:
                to_remove.add(pair.factor_a if abs(pair.correlation) > abs(pair.correlation) else pair.factor_b)

        return list(to_remove)
