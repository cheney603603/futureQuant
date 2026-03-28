"""
鲁棒性测试器 (Robustness Tester)

因子参数敏感性测试：
- 参数扰动测试
- Bootstrap 测试
- 样本外稳定性
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ...core.logger import get_logger

logger = get_logger('agent.validators.robustness_tester')


class FactorRobustnessTester:
    """因子鲁棒性测试器"""

    def __init__(
        self,
        n_bootstrap: int = 100,
        perturbation_scale: float = 0.1,
        significance_level: float = 0.05,
    ) -> None:
        """
        Args:
            n_bootstrap: Bootstrap 迭代次数
            perturbation_scale: 参数扰动幅度
            significance_level: 显著性水平
        """
        self.n_bootstrap = n_bootstrap
        self.perturbation_scale = perturbation_scale
        self.significance_level = significance_level

    def test_sensitivity(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        n_steps: int = 10,
    ) -> Dict[str, Any]:
        """
        测试因子对参数扰动的敏感性

        Args:
            factor_values: 因子值
            returns: 收益率
            n_steps: 扰动步数

        Returns:
            敏感性测试结果
        """
        aligned = pd.concat([factor_values, returns], axis=1).dropna()
        if len(aligned) < 30:
            return {'error': 'insufficient data'}

        base_ic = self._calculate_ic(aligned.iloc[:, 0], aligned.iloc[:, 1])

        perturbations = np.linspace(-self.perturbation_scale,
                                    self.perturbation_scale, n_steps)
        ic_scores = []

        for pert in perturbations:
            perturbed = aligned.iloc[:, 0] * (1 + pert)
            ic = self._calculate_ic(perturbed, aligned.iloc[:, 1])
            ic_scores.append(ic)

        ic_arr = np.array(ic_scores)
        sensitivity = ic_arr.std()

        return {
            'base_ic': float(base_ic),
            'perturbed_ics': [float(x) for x in ic_arr],
            'sensitivity': float(sensitivity),
            'robustness_score': float(max(1.0 - sensitivity / (abs(base_ic) + 1e-8), 0.0)),
            'is_robust': sensitivity < abs(base_ic) * 0.5,
        }

    def bootstrap_test(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        Bootstrap 显著性检验

        Args:
            factor_values: 因子值
            returns: 收益率

        Returns:
            Bootstrap 检验结果
        """
        aligned = pd.concat([factor_values, returns], axis=1).dropna()
        if len(aligned) < 30:
            return {'error': 'insufficient data'}

        base_ic = self._calculate_ic(aligned.iloc[:, 0], aligned.iloc[:, 1])
        n = len(aligned)

        bootstrap_ics = []
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            boot_fv = aligned.iloc[:, 0].iloc[idx]
            boot_ret = aligned.iloc[:, 1].iloc[idx]
            ic = self._calculate_ic(boot_fv, boot_ret)
            bootstrap_ics.append(ic)

        boot_arr = np.array(bootstrap_ics)
        ci_lower = np.percentile(boot_arr, 2.5)
        ci_upper = np.percentile(boot_arr, 97.5)

        return {
            'original_ic': float(base_ic),
            'bootstrap_mean': float(boot_arr.mean()),
            'bootstrap_std': float(boot_arr.std()),
            'ci_95': (float(ci_lower), float(ci_upper)),
            'is_significant': (ci_lower > 0) or (ci_upper < 0),
        }

    def perturbs_test(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
    ) -> Dict[str, Any]:
        """
        参数扰动检验

        Args:
            factor_values: 因子值
            returns: 收益率

        Returns:
            扰动检验结果
        """
        result = self.test_sensitivity(factor_values, returns)
        bootstrap_result = self.bootstrap_test(factor_values, returns)

        combined_score = (
            result.get('robustness_score', 0.0) * 0.5 +
            (1.0 if bootstrap_result.get('is_significant') else 0.0) * 0.5
        )

        return {
            **result,
            **bootstrap_result,
            'combined_robustness_score': float(combined_score),
            'overall_robust': combined_score >= 0.5,
        }

    def _calculate_ic(self, fv: pd.Series, rt: pd.Series) -> float:
        """计算 IC"""
        corr, _ = stats.spearmanr(fv, rt)
        return corr if not np.isnan(corr) else 0.0
