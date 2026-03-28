"""
å®è§å å­ææ Agent

ä½¿ç¨ç¾åææ°ãå©çãååææ°ãéèé¢æç­å®è§å å­ï¼
èèå®è§æ°æ®éå¸¸ä½é¢ï¼æ/å¨ï¼ï¼éè¦ ffill å°æ¥é¢ï¼
éè¿ IC ç­éææå å­ã
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ...core.base import Factor
from ...core.logger import get_logger
from ...factor import (
    DollarIndexFactor,
    InterestRateFactor,
    CommodityIndexFactor,
    InflationExpectationFactor,
)
from ..base import AgentResult, AgentStatus, BaseAgent

logger = get_logger('agent.miners.macro')


class MacroMiningAgent(BaseAgent):
    """
    å®è§å å­ææ Agent

    ä½¿ç¨ç¾åææ°ãå©çãååææ°ãéèé¢æç­å®è§æ°æ®è¿è¡å å­ææã
    å®è§æ°æ®éå¸¸ä½é¢ï¼æ/å¨ï¼ï¼éè¦ ffill å°æ¥é¢ã

    æ¯æçå å­:
    - DollarIndexFactor: ç¾åææ°å½±åå å­
    - InterestRateFactor: å©çå å­
    - CommodityIndexFactor: ååææ°å å­
    - InflationExpectationFactor: éèé¢æå å­

    æ°æ®é¢å¤ç:
    - å®è§æ°æ®éå¸¸ä½é¢ï¼ä½¿ç¨ååå¡«åï¼ffillï¼å°æ¥é¢
    - ç¡®ä¿å®è§å å­ä¸æ¶ççåºåå¯¹é½

    ä½¿ç¨ç¤ºä¾:
        >>> agent = MacroMiningAgent(name='macro_miner')
        >>> result = agent.run({'context': mining_context})
        >>> print(f"Found {result.n_factors} macro factors")
    """

    DEFAULT_CONFIG = {
        'ic_threshold': 0.01,
        'dollar_method': 'change_rate',
        'dollar_window': 20,
        'rate_method': 'change',
        'rate_window': 20,
        'commodity_method': 'beta',
        'commodity_window': 20,
        'inflation_method': 'change',
        'inflation_window': 20,
    }

    # åå å­éè¦çæ°æ®å
    REQUIRED_COLUMNS = {
        'DollarIndexFactor': ['dxy'],
        'InterestRateFactor': ['interest_rate'],
        'CommodityIndexFactor': ['commodity_index'],
        'InflationExpectationFactor': ['inflation_expectation'],
    }

    def __init__(self, name: str = 'macro_miner', config: Optional[Dict[str, Any]] = None):
        """
        åå§åå®è§å å­ææ Agent

        Args:
            name: Agent åç§°
            config: éç½®å­å¸ï¼å¯è¦ç DEFAULT_CONFIG
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        æ§è¡å®è§å å­ææ

        Args:
            context: æ§è¡ä¸ä¸æï¼å¿é¡»åå« 'context' é®ï¼å¼ä¸º MiningContext å¯¹è±¡

        Returns:
            AgentResult: ææç»æï¼åå«éè¿ç­éçå å­åè¡¨åå å­å¼ DataFrame
        """
        mining_context = context.get('context')
        if mining_context is None:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=["Missing 'context' in execution context"],
            )

        data = mining_context.data
        returns = mining_context.returns
        ic_threshold = self.config.get('ic_threshold', 0.01)

        self._logger.info(
            f"Starting macro factor mining: data_shape={data.shape}, "
            f"ic_threshold={ic_threshold}"
        )

        # çæåéå å­
        candidate_factors = self._generate_candidates(data)
        self._logger.info(f"Generated {len(candidate_factors)} candidate factors")

        if not candidate_factors:
            self._logger.warning("No macro factors available (missing required columns)")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                data=pd.DataFrame(index=data.index),
                factors=[],
                metrics={'total_candidates': 0, 'selected_count': 0},
            )

        # è®¡ç®æ¯ä¸ªå å­ç IC å¹¶ç­é
        selected_factors: List[Factor] = []
        factor_data_dict: Dict[str, pd.Series] = {}
        factor_metrics: Dict[str, Dict[str, float]] = {}
        errors: List[str] = []

        for factor in candidate_factors:
            try:
                # è®¡ç®å å­å¼
                factor_values = factor.compute(data)

                if factor_values is None or factor_values.empty:
                    self._logger.warning(f"Factor {factor.name} produced empty values")
                    continue

                # ååå¡«åå°æ¥é¢ï¼å®è§æ°æ®éå¸¸ä½é¢ï¼
                factor_values_ffill = factor_values.ffill()

                # å¯¹é½æ°æ®
                common_idx = factor_values_ffill.index.intersection(returns.index)
                if len(common_idx) < 30:
                    self._logger.warning(
                        f"Factor {factor.name}: insufficient common data points ({len(common_idx)})"
                    )
                    continue

                f_vals = factor_values_ffill.loc[common_idx]
                r_vals = returns.loc[common_idx]

                # å»é¤ç¼ºå¤±å¼
                valid_mask = f_vals.notna() & r_vals.notna()
                if valid_mask.sum() < 30:
                    self._logger.warning(
                        f"Factor {factor.name}: insufficient valid values ({valid_mask.sum()})"
                    )
                    continue

                # è®¡ç® Spearman IC
                ic, p_value = stats.spearmanr(f_vals[valid_mask], r_vals[valid_mask])

                if np.isnan(ic):
                    self._logger.warning(f"Factor {factor.name}: IC is NaN")
                    continue

                # è®°å½ææ 
                factor_metrics[factor.name] = {
                    'ic': float(ic),
                    'ic_abs': abs(float(ic)),
                    'p_value': float(p_value) if not np.isnan(p_value) else 1.0,
                    'n_samples': int(valid_mask.sum()),
                }

                # IC ç­é
                if abs(ic) >= ic_threshold:
                    selected_factors.append(factor)
                    factor_data_dict[factor.name] = factor_values_ffill
                    self._logger.debug(
                        f"Factor {factor.name} selected: IC={ic:.4f}, p={p_value:.4f}"
                    )
                else:
                    self._logger.debug(
                        f"Factor {factor.name} rejected: IC={ic:.4f} (below threshold {ic_threshold})"
                    )

            except Exception as e:
                error_msg = f"Failed to evaluate factor {factor.name}: {e}"
                self._logger.warning(error_msg)
                errors.append(error_msg)
                continue

        self._logger.info(
            f"Macro mining completed: {len(selected_factors)}/{len(candidate_factors)} "
            f"factors selected"
        )

        # æå»ºå å­å¼ DataFrame
        factor_df: Optional[pd.DataFrame] = None
        if factor_data_dict:
            try:
                factor_df = pd.DataFrame(factor_data_dict)
                factor_df.index = data.index
            except Exception as e:
                self._logger.warning(f"Failed to build factor DataFrame: {e}")

        # æ±æ»ææ 
        metrics = {
            'total_candidates': len(candidate_factors),
            'selected_count': len(selected_factors),
            'selection_rate': len(selected_factors) / len(candidate_factors) if candidate_factors else 0,
            'ic_threshold': ic_threshold,
            'factor_metrics': factor_metrics,
        }

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data=factor_df,
            factors=selected_factors,
            metrics=metrics,
            errors=errors if errors else None,
        )

    def _generate_candidates(self, data: pd.DataFrame) -> List[Factor]:
        """
        çæåéå å­å®ä¾ï¼æ ¹æ®æ°æ®å¯ç¨æ§ï¼

        Args:
            data: è¾å¥æ°æ® DataFrame

        Returns:
            åéå å­åè¡¨
        """
        factors: List[Factor] = []

        # ç¾åææ°å å­ï¼éè¦ dxy åï¼
        if 'dxy' in data.columns:
            dollar_method = self.config.get('dollar_method', 'change_rate')
            dollar_window = self.config.get('dollar_window', 20)
            factors.append(
                DollarIndexFactor(method=dollar_method, window=dollar_window)
            )
            self._logger.debug(f"Added DollarIndexFactor (method={dollar_method})")
        else:
            self._logger.debug("Skipping DollarIndexFactor (missing dxy column)")

        # å©çå å­ï¼éè¦ interest_rate åï¼
        if 'interest_rate' in data.columns:
            rate_method = self.config.get('rate_method', 'change')
            rate_window = self.config.get('rate_window', 20)
            factors.append(
                InterestRateFactor(method=rate_method, window=rate_window)
            )
            self._logger.debug(f"Added InterestRateFactor (method={rate_method})")
        else:
            self._logger.debug("Skipping InterestRateFactor (missing interest_rate column)")

        # ååææ°å å­ï¼éè¦ commodity_index å close åï¼
        if 'commodity_index' in data.columns and 'close' in data.columns:
            commodity_method = self.config.get('commodity_method', 'beta')
            commodity_window = self.config.get('commodity_window', 20)
            factors.append(
                CommodityIndexFactor(method=commodity_method, window=commodity_window)
            )
            self._logger.debug(f"Added CommodityIndexFactor (method={commodity_method})")
        else:
            self._logger.debug("Skipping CommodityIndexFactor (missing commodity_index or close)")

        # éèé¢æå å­ï¼éè¦ inflation_expectation åï¼
        if 'inflation_expectation' in data.columns:
            inflation_method = self.config.get('inflation_method', 'change')
            inflation_window = self.config.get('inflation_window', 20)
            factors.append(
                InflationExpectationFactor(method=inflation_method, window=inflation_window)
            )
            self._logger.debug(f"Added InflationExpectationFactor (method={inflation_method})")
        else:
            self._logger.debug("Skipping InflationExpectationFactor (missing inflation_expectation)")

        return factors
