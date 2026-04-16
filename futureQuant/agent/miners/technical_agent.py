"""
ææ¯å å­ææ Agent

éåå¨éãæ³¢å¨çãæäº¤éå å­çåæ°ç©ºé´ï¼
ä½¿ç¨ Spearman IC ç­éææå å­ã
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ...core.base import Factor
from ...core.logger import get_logger
from ...factor import (
    MomentumFactor,
    RSIFactor,
    MACDFactor,
    RateOfChangeFactor,
    ATRFactor,
    VolatilityFactor,
    BollingerBandWidthFactor,
    OBVFactor,
    VolumeRatioFactor,
    VolumeMAFactor,
)
from ..base import AgentResult, AgentStatus, BaseAgent

logger = get_logger('agent.miners.technical')


class TechnicalMiningAgent(BaseAgent):
    """
    ææ¯å å­ææ Agent

    éåææ¯å å­çåæ°ç©ºé´ï¼è®¡ç®æ¯ä¸ªå å­ç Spearman ICï¼
    ä¿ç IC ç»å¯¹å¼è¶è¿éå¼çå å­ã

    æ¯æçå å­ç±»å«:
    - å¨éå å­: MomentumFactor, RSIFactor, MACDFactor, RateOfChangeFactor
    - æ³¢å¨çå å­: ATRFactor, VolatilityFactor, BollingerBandWidthFactor
    - æäº¤éå å­: OBVFactor, VolumeRatioFactor, VolumeMAFactor

    åæ°æç´¢ç©ºé´:
    - momentum_windows: [5, 10, 20, 60, 120]
    - volatility_windows: [10, 20, 60]
    - volume_windows: [5, 10, 20]
    - rsi_windows: [6, 14, 21]

    ä½¿ç¨ç¤ºä¾:
        >>> agent = TechnicalMiningAgent(name='tech_miner')
        >>> result = agent.run({'context': mining_context})
        >>> print(f"Found {result.n_factors} technical factors")
    """

    DEFAULT_CONFIG = {
        'ic_threshold': 0.02,
        'momentum_windows': [5, 10, 20, 60, 120],
        'volatility_windows': [10, 20, 60],
        'volume_windows': [5, 10, 20],
        'rsi_windows': [6, 14, 21],
    }

    def __init__(self, name: str = 'technical_miner', config: Optional[Dict[str, Any]] = None):
        """
        åå§åææ¯å å­ææ Agent

        Args:
            name: Agent åç§°
            config: éç½®å­å¸ï¼å¯è¦ç DEFAULT_CONFIG
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        æ§è¡ææ¯å å­ææ

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

        if isinstance(mining_context, dict):
            data = mining_context.get('data')
            returns = mining_context.get('returns')
        else:
            data = mining_context.data
            returns = mining_context.returns

        if data is None or returns is None:
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=["Missing data or returns in mining context"],
            )
        ic_threshold = self.config.get('ic_threshold', 0.02)

        self._logger.info(
            f"Starting technical factor mining: data_shape={data.shape}, "
            f"ic_threshold={ic_threshold}"
        )

        # çæææåéå å­
        candidate_factors = self._generate_candidates()
        self._logger.info(f"Generated {len(candidate_factors)} candidate factors")

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

                # å¯¹é½æ°æ®
                common_idx = factor_values.index.intersection(returns.index)
                if len(common_idx) < 30:
                    self._logger.warning(
                        f"Factor {factor.name}: insufficient common data points ({len(common_idx)})"
                    )
                    continue

                f_vals = factor_values.loc[common_idx]
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
                    factor_data_dict[factor.name] = factor_values
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
            f"Technical mining completed: {len(selected_factors)}/{len(candidate_factors)} "
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

    def _generate_candidates(self) -> List[Factor]:
        """
        çæææåéå å­å®ä¾

        Returns:
            åéå å­åè¡¨
        """
        factors: List[Factor] = []

        # å¨éå å­åæ°
        momentum_windows = self.config.get('momentum_windows', [5, 10, 20, 60, 120])
        rsi_windows = self.config.get('rsi_windows', [6, 14, 21])

        # æ³¢å¨çå å­åæ°
        volatility_windows = self.config.get('volatility_windows', [10, 20, 60])

        # æäº¤éå å­åæ°
        volume_windows = self.config.get('volume_windows', [5, 10, 20])

        # å¨éå å­
        for period in momentum_windows:
            factors.append(MomentumFactor(name=f"MomentumFactor_{period}", period=period))

        for period in rsi_windows:
            factors.append(RSIFactor(name=f"RSIFactor_{period}", period=period))

        # MACD å å­ï¼æ åæ°ï¼
        factors.append(MACDFactor(name="MACDFactor"))

        for period in momentum_windows:
            factors.append(RateOfChangeFactor(name=f"RateOfChangeFactor_{period}", period=period))

        # æ³¢å¨çå å­
        for period in volatility_windows:
            factors.append(ATRFactor(name=f"ATRFactor_{period}", period=period))

        for period in volatility_windows:
            factors.append(VolatilityFactor(name=f"VolatilityFactor_{period}", period=period))

        for period in volatility_windows:
            factors.append(
                BollingerBandWidthFactor(
                    name=f"BollingerBandWidthFactor_{period}",
                    period=period,
                )
            )

        # æäº¤éå å­
        factors.append(OBVFactor(name="OBVFactor"))

        for period in volume_windows:
            factors.append(VolumeRatioFactor(name=f"VolumeRatioFactor_{period}", period=period))

        for period in volume_windows:
            factors.append(VolumeMAFactor(name=f"VolumeMAFactor_{period}", period=period))

        return factors
