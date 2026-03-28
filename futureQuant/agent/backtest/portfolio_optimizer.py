"""
组合优化器 (Portfolio Optimizer)

均值方差优化、风险预算、最小方差组合：
- Mean-Variance Optimization
- Risk Parity
- Minimum Variance
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize

from ...core.logger import get_logger

logger = get_logger('agent.portfolio_optimizer')


class PortfolioOptimizer:
    """组合优化器"""

    def __init__(
        self,
        method: str = 'mean_variance',
        risk_aversion: float = 1.0,
        max_weight: float = 0.3,
        min_weight: float = 0.0,
    ) -> None:
        """
        Args:
            method: 优化方法，'mean_variance' / 'risk_parity' / 'min_variance' / 'equal'
            risk_aversion: 风险厌恶系数
            max_weight: 最大权重
            min_weight: 最小权重
        """
        self.method = method
        self.risk_aversion = risk_aversion
        self.max_weight = max_weight
        self.min_weight = min_weight

    def optimize(
        self,
        returns: pd.DataFrame,
        factor_exposures: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """
        优化组合权重

        Args:
            returns: 因子收益率 DataFrame
            factor_exposures: 因子暴露度（可选）

        Returns:
            最优权重数组
        """
        n = len(returns.columns)
        if n == 0:
            return np.array([])

        if self.method == 'equal':
            return np.ones(n) / n

        aligned = returns.dropna()
        if len(aligned) < 30:
            return np.ones(n) / n

        mean_ret = aligned.mean().values
        cov = aligned.cov().values

        if self.method == 'mean_variance':
            return self._mean_variance(mean_ret, cov, n)
        elif self.method == 'risk_parity':
            return self._risk_parity(cov, n)
        elif self.method == 'min_variance':
            return self._min_variance(cov, n)
        return np.ones(n) / n

    def _mean_variance(
        self,
        mean_ret: np.ndarray,
        cov: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """均值方差优化"""
        def neg_sharpe(w: np.ndarray) -> float:
            port_ret = np.dot(w, mean_ret)
            port_var = np.dot(w, np.dot(cov, w))
            return -(port_ret - self.risk_aversion * port_var)

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        w0 = np.ones(n) / n

        result = optimize.minimize(neg_sharpe, w0, method='SLSQP',
                                  bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    def _risk_parity(
        self,
        cov: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """风险平价"""
        def risk_contrib(w: np.ndarray) -> np.ndarray:
            port_var = np.dot(w, np.dot(cov, w))
            marginal_var = np.dot(cov, w)
            rc = w * marginal_var / (port_var + 1e-8)
            return rc

        def risk_parity_error(w: np.ndarray) -> float:
            rc = risk_contrib(w)
            target_rc = np.ones(n) * np.sum(rc) / n
            return np.sum((rc - target_rc) ** 2)

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        w0 = np.ones(n) / n

        result = optimize.minimize(risk_parity_error, w0, method='SLSQP',
                                  bounds=bounds, constraints=constraints)
        return result.x if result.success else w0

    def _min_variance(
        self,
        cov: np.ndarray,
        n: int,
    ) -> np.ndarray:
        """最小方差"""
        def portfolio_variance(w: np.ndarray) -> float:
            return np.dot(w, np.dot(cov, w))

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        w0 = np.ones(n) / n

        result = optimize.minimize(portfolio_variance, w0, method='SLSQP',
                                  bounds=bounds, constraints=constraints)
        return result.x if result.success else w0
