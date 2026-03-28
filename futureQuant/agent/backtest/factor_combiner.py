"""
因子组合器 (Factor Combiner)

多因子权重优化组合：
- 等权组合
- ICIR 加权组合
- 均值方差优化组合
- 去相关正交化组合
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats, optimize

from ...core.logger import get_logger

logger = get_logger('agent.factor_combiner')


class FactorCombiner:
    """多因子组合器"""

    def __init__(
        self,
        method: str = 'icir_weighted',
        correlation_threshold: float = 0.8,
        max_iterations: int = 100,
    ) -> None:
        """
        Args:
            method: 组合方法，'equal' / 'icir_weighted' / 'mean_variance' / 'orthogonal'
            correlation_threshold: 相关性阈值，超过则进行正交化
            max_iterations: 最大迭代次数
        """
        self.method = method
        self.correlation_threshold = correlation_threshold
        self.max_iterations = max_iterations

    def combine(
        self,
        factor_dict: Dict[str, pd.Series],
        returns: pd.Series,
    ) -> pd.Series:
        """
        组合多个因子

        Args:
            factor_dict: 因子名称到因子值的字典
            returns: 收益率序列

        Returns:
            组合因子值
        """
        df = pd.DataFrame(factor_dict)

        if self.method == 'equal':
            weights = self._equal_weights(len(df.columns))
        elif self.method == 'icir_weighted':
            weights = self._icir_weights(df, returns)
        elif self.method == 'mean_variance':
            weights = self._mean_variance_weights(df, returns)
        elif self.method == 'orthogonal':
            df, weights = self._orthogonalize(df, returns)
        else:
            weights = self._equal_weights(len(df.columns))

        combined = (df * weights).sum(axis=1)
        return combined

    def _equal_weights(self, n: int) -> np.ndarray:
        return np.ones(n) / n

    def _icir_weights(
        self,
        df: pd.DataFrame,
        returns: pd.Series,
    ) -> np.ndarray:
        """ICIR 加权"""
        aligned = df.join(returns).dropna()
        if len(aligned) < 30:
            return self._equal_weights(len(df.columns))

        ic_list = []
        for col in df.columns:
            corr, _ = stats.spearmanr(aligned[col], aligned[returns.name or 'return'])
            ic_list.append(abs(corr) if not np.isnan(corr) else 0.0)

        ic_arr = np.array(ic_list)
        rolling_ic = []
        for col in df.columns:
            rolling = df[col].rolling(20).corr(returns).dropna()
            if len(rolling) > 5:
                icir = abs(rolling.mean()) / (rolling.std() + 1e-8)
                rolling_ic.append(icir)
            else:
                rolling_ic.append(0.0)

        icir_arr = np.array(rolling_ic)
        scores = ic_arr * icir_arr
        if scores.sum() < 1e-8:
            return self._equal_weights(len(df.columns))
        return scores / scores.sum()

    def _mean_variance_weights(
        self,
        df: pd.DataFrame,
        returns: pd.Series,
    ) -> np.ndarray:
        """均值方差优化"""
        aligned = df.join(returns).dropna()
        if len(aligned) < 30:
            return self._equal_weights(len(df.columns))

        ic_list = []
        for col in df.columns:
            corr, _ = stats.spearmanr(aligned[col], aligned[returns.name or 'return'])
            ic_list.append(corr if not np.isnan(corr) else 0.0)

        mean_returns = np.array(ic_list)
        cov_matrix = df.corr().values

        n = len(mean_returns)
        cov_flat = cov_matrix.flatten()

        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = np.dot(w, mean_returns)
            port_vol = np.sqrt(np.dot(w, np.dot(cov_matrix, w)) + 1e-8)
            return -port_ret / port_vol

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0.0, 1.0) for _ in range(n)]
        w0 = np.ones(n) / n

        try:
            result = optimize.minimize(neg_sharpe, w0, method='SLSQP',
                                       bounds=bounds, constraints=constraints,
                                       options={'maxiter': self.max_iterations})
            return result.x if result.success else w0
        except Exception:
            return w0

    def _orthogonalize(
        self,
        df: pd.DataFrame,
        returns: pd.Series,
    ) -> tuple:
        """去相关正交化"""
        factors = list(df.columns)
        orthogonal = pd.DataFrame(index=df.index)

        for i, col in enumerate(factors):
            if i == 0:
                orthogonal[col] = df[col]
            else:
                # 对前面已正交化的因子回归
                X = orthogonal.iloc[:, :i].values
                y = df[col].values
                coef = np.linalg.lstsq(X, y, rcond=None)[0]
                orthogonal[col] = df[col] - np.dot(X, coef)

        weights = self._icir_weights(orthogonal, returns)
        return orthogonal, weights

    def calculate_portfolio_ic(
        self,
        combined_factor: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, float]:
        """计算组合因子 IC"""
        aligned = pd.concat([combined_factor, returns], axis=1).dropna()
        if len(aligned) < 20:
            return {'ic': 0.0, 'pval': 1.0}

        corr, pval = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return {
            'ic': float(corr) if not np.isnan(corr) else 0.0,
            'pval': float(pval) if not np.isnan(pval) else 1.0,
        }
