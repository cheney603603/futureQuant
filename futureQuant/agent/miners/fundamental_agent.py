"""
åºæ¬é¢å å­ææ Agent

ä½¿ç¨åºå·®ãåºå­ãä»åç­åºæ¬é¢å å­ï¼
èèæ°æ®åå¸å»¶è¿è¿è¡ lag å¤çï¼
éè¿ IC ç­éææå å­ã
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ...core.base import Factor
from ...core.logger import get_logger
from ...factor import (
    BasisFactor,
    BasisRateFactor,
    TermStructureFactor,
    InventoryChangeFactor,
    InventoryYoYFactor,
    WarehouseReceiptFactor,
    WarehousePressureFactor,
)
from ..base import AgentResult, AgentStatus, BaseAgent

logger = get_logger('agent.miners.fundamental')


class FundamentalMiningAgent(BaseAgent):
    """
    åºæ¬é¢å å­ææ Agent

    ä½¿ç¨åºå·®ãåºå­ãä»åç­åºæ¬é¢æ°æ®è¿è¡å å­ææã
    ç±äºåºæ¬é¢æ°æ®æåå¸å»¶è¿ï¼éè¦å¯¹å å­å¼å lag å¤çã

    æ¯æçå å­:
    - åºå·®å å­: BasisFactor, BasisRateFactor, TermStructureFactor
    - åºå­å å­: InventoryChangeFactor, InventoryYoYFactor
    - ä»åå å­: WarehouseReceiptFactor, WarehousePressureFactor

    æ°æ®å»¶è¿å¤ç:
    - basis: lag=1ï¼æ¥é¢æ°æ®ï¼å»¶è¿1å¤©ï¼
    - inventory: lag=3ï¼å¨åº¦æ°æ®ï¼å»¶è¿3å¤©ï¼
    - warehouse: lag=2ï¼æ¥é¢æ°æ®ï¼å»¶è¿2å¤©ï¼

    ä½¿ç¨ç¤ºä¾:
        >>> agent = FundamentalMiningAgent(name='fundamental_miner')
        >>> result = agent.run({'context': mining_context})
        >>> print(f"Found {result.n_factors} fundamental factors")
    """

    DEFAULT_CONFIG = {
        'ic_threshold': 0.02,
        'basis_lag': 1,
        'inventory_lag': 3,
        'warehouse_lag': 2,
    }

    # åå å­éè¦çæ°æ®å
    REQUIRED_COLUMNS = {
        'BasisFactor': ['spot_price', 'close'],
        'BasisRateFactor': ['spot_price', 'close'],
        'TermStructureFactor': ['near_price', 'far_price'],
        'InventoryChangeFactor': ['inventory'],
        'InventoryYoYFactor': ['inventory'],
        'WarehouseReceiptFactor': ['warehouse_receipt'],
        'WarehousePressureFactor': ['warehouse_receipt', 'volume'],
    }

    # åå å­çå»¶è¿ææ°
    FACTOR_LAGS = {
        'BasisFactor': 1,
        'BasisRateFactor': 1,
        'TermStructureFactor': 1,
        'InventoryChangeFactor': 3,
        'InventoryYoYFactor': 3,
        'WarehouseReceiptFactor': 2,
        'WarehousePressureFactor': 2,
    }

    def __init__(self, name: str = 'fundamental_miner', config: Optional[Dict[str, Any]] = None):
        """
        åå§ååºæ¬é¢å å­ææ Agent

        Args:
            name: Agent åç§°
            config: éç½®å­å¸ï¼å¯è¦ç DEFAULT_CONFIG
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        æ§è¡åºæ¬é¢å å­ææ

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
        ic_threshold = self.config.get('ic_threshold', 0.02)

        self._logger.info(
            f"Starting fundamental factor mining: data_shape={data.shape}, "
            f"ic_threshold={ic_threshold}"
        )

        # çæåéå å­
        candidate_factors = self._generate_candidates(data)
        self._logger.info(f"Generated {len(candidate_factors)} candidate factors")

        if not candidate_factors:
            self._logger.warning("No fundamental factors available (missing required columns)")
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

                # åºç¨å»¶è¿ï¼shiftï¼
                lag_periods = self.FACTOR_LAGS.get(factor.__class__.__name__, 1)
                factor_values_lagged = factor_values.shift(lag_periods)

                # å¯¹é½æ°æ®
                common_idx = factor_values_lagged.index.intersection(returns.index)
                if len(common_idx) < 30:
                    self._logger.warning(
                        f"Factor {factor.name}: insufficient common data points ({len(common_idx)})"
                    )
                    continue

                f_vals = factor_values_lagged.loc[common_idx]
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
                    'lag': lag_periods,
                }

                # IC ç­é
                if abs(ic) >= ic_threshold:
                    selected_factors.append(factor)
                    factor_data_dict[factor.name] = factor_values_lagged
                    self._logger.debug(
                        f"Factor {factor.name} selected: IC={ic:.4f}, p={p_value:.4f}, lag={lag_periods}"
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
            f"Fundamental mining completed: {len(selected_factors)}/{len(candidate_factors)} "
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

        # åºå·®å å­ï¼éè¦ spot_price å closeï¼
        if 'spot_price' in data.columns and 'close' in data.columns:
            factors.append(BasisFactor())
            factors.append(BasisRateFactor())
            self._logger.debug("Added basis factors")
        else:
            self._logger.debug("Skipping basis factors (missing spot_price or close)")

        # æéç»æå å­ï¼éè¦ near_price å far_priceï¼
        if 'near_price' in data.columns and 'far_price' in data.columns:
            factors.append(TermStructureFactor())
            self._logger.debug("Added term structure factor")
        else:
            self._logger.debug("Skipping term structure factor (missing near_price or far_price)")

        # åºå­å å­ï¼éè¦ inventoryï¼
        if 'inventory' in data.columns:
            factors.append(InventoryChangeFactor())
            factors.append(InventoryYoYFactor())
            self._logger.debug("Added inventory factors")
        else:
            self._logger.debug("Skipping inventory factors (missing inventory)")

        # ä»åå å­ï¼éè¦ warehouse_receiptï¼
        if 'warehouse_receipt' in data.columns:
            factors.append(WarehouseReceiptFactor())
            if 'volume' in data.columns:
                factors.append(WarehousePressureFactor())
                self._logger.debug("Added warehouse receipt and pressure factors")
            else:
                self._logger.debug("Added warehouse receipt factor")
        else:
            self._logger.debug("Skipping warehouse factors (missing warehouse_receipt)")

        return factors
