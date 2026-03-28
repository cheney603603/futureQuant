"""
å å­èå Agent

å¯¹æ¥èªåææ Agent çå å­è¿è¡ï¼
1. å»ç¸å³ï¼è®¡ç®å å­é´ Spearman ç¸å³æ§ï¼ç¸å³æ§ > 0.8 çå å­ç»ä¸­ä¿ç IC æé«ç
2. ICIR å æåæï¼è®¡ç®æ¯ä¸ªå å­ç ICIRï¼æ ICIR å½ä¸åæéåæç»¼åå å­
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ...core.base import Factor
from ...core.logger import get_logger
from ..base import AgentResult, AgentStatus, BaseAgent

logger = get_logger('agent.miners.fusion')


class FusionAgent(BaseAgent):
    """
    å å­èå Agent

    å¯¹æ¥èªåææ Agent çå å­è¿è¡å»ç¸å³å ICIR å æåæã

    å¤çæµç¨:
    1. è®¡ç®ææå å­é´ç Spearman ç¸å³æ§ç©éµ
    2. å»ç¸å³ï¼ç¸å³æ§ > threshold çå å­ç»ï¼ä¿ç IC æé«çå å­
    3. è®¡ç®æ¯ä¸ªå å­ç ICIRï¼ä¿¡æ¯æ¯çï¼
    4. æ ICIR å½ä¸åæéåæç»¼åå å­

    éç½®åæ°:
    - corr_threshold: ç¸å³æ§éå¼ï¼é»è®¤ 0.8
    - min_icir: æå° ICIR è¦æ±ï¼é»è®¤ 0.3

    ä½¿ç¨ç¤ºä¾:
        >>> agent = FusionAgent(name='fusion_agent')
        >>> result = agent.run({
        ...     'context': mining_context,
        ...     'factors': factors,
        ...     'factor_data': factor_df,
        ...     'returns': returns,
        ... })
        >>> print(f"Selected {result.n_factors} factors after fusion")
    """

    DEFAULT_CONFIG = {
        'corr_threshold': 0.8,
        'min_icir': 0.3,
    }

    def __init__(self, name: str = 'fusion_agent', config: Optional[Dict[str, Any]] = None):
        """
        åå§åå å­èå Agent

        Args:
            name: Agent åç§°
            config: éç½®å­å¸ï¼å¯è¦ç DEFAULT_CONFIG
        """
        merged_config = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(name, merged_config)

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        æ§è¡å å­èå

        Args:
            context: æ§è¡ä¸ä¸æï¼åå«:
                - 'context': MiningContext å¯¹è±¡
                - 'factors': å å­å®ä¾åè¡¨
                - 'factor_data': å å­å¼ DataFrame
                - 'returns': æ¶çç Series

        Returns:
            AgentResult: èåç»æï¼åå«ç­éåçå å­åè¡¨ååæå å­æ°æ®
        """
        mining_context = context.get('context')
        factors: List[Factor] = context.get('factors', [])
        factor_data: Optional[pd.DataFrame] = context.get('factor_data')
        returns: Optional[pd.Series] = context.get('returns')

        if not factors:
            self._logger.warning("No factors provided for fusion")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.SUCCESS,
                data=pd.DataFrame(),
                factors=[],
                metrics={'message': 'No factors to fuse'},
            )

        self._logger.info(f"Starting factor fusion: {len(factors)} input factors")

        corr_threshold = self.config.get('corr_threshold', 0.8)
        min_icir = self.config.get('min_icir', 0.3)

        # 1. è®¡ç®åå å­ç IC å ICIR
        factor_stats = self._calculate_factor_stats(factors, factor_data, returns)
        self._logger.info(f"Calculated stats for {len(factor_stats)} factors")

        # 2. å»ç¸å³å¤ç
        deduped_factors, deduped_data = self._deduplicate_factors(
            factors, factor_data, factor_stats, corr_threshold
        )
        self._logger.info(
            f"Deduplication: {len(factors)} -> {len(deduped_factors)} factors "
            f"(threshold={corr_threshold})"
        )

        # 3. ICIR ç­é
        selected_factors, selected_data, selected_stats = self._filter_by_icir(
            deduped_factors, deduped_data, factor_stats, min_icir
        )
        self._logger.info(
            f"ICIR filtering: {len(deduped_factors)} -> {len(selected_factors)} factors "
            f"(min_icir={min_icir})"
        )

        # 4. åæç»¼åå å­
        composite_factor = self._build_composite_factor(selected_data, selected_stats)

        # 5. æå»ºæç»ç»æ
        final_data = selected_data.copy() if selected_data is not None else pd.DataFrame()
        if composite_factor is not None and not composite_factor.empty:
            final_data['composite_factor'] = composite_factor

        # æ±æ»ææ 
        metrics = {
            'input_factors': len(factors),
            'after_dedup': len(deduped_factors),
            'after_icir_filter': len(selected_factors),
            'corr_threshold': corr_threshold,
            'min_icir': min_icir,
            'factor_stats': selected_stats,
            'composite_factor_weight': {
                name: stats.get('weight', 0)
                for name, stats in selected_stats.items()
            },
        }

        self._logger.info(
            f"Fusion completed: {len(selected_factors)} factors selected, "
            f"composite factor built"
        )

        return AgentResult(
            agent_name=self.name,
            status=AgentStatus.SUCCESS,
            data=final_data,
            factors=selected_factors,
            metrics=metrics,
        )

    def _calculate_factor_stats(
        self,
        factors: List[Factor],
        factor_data: Optional[pd.DataFrame],
        returns: Optional[pd.Series],
    ) -> Dict[str, Dict[str, float]]:
        """
        è®¡ç®åå å­ç IC å ICIR

        Args:
            factors: å å­å®ä¾åè¡¨
            factor_data: å å­å¼ DataFrame
            returns: æ¶çç Series

        Returns:
            {å å­å: {ic, icir, ic_std, n_samples}} å­å¸
        """
        stats_dict: Dict[str, Dict[str, float]] = {}

        if factor_data is None or returns is None:
            return stats_dict

        for factor in factors:
            factor_name = factor.name

            if factor_name not in factor_data.columns:
                continue

            try:
                f_vals = factor_data[factor_name]

                # å¯¹é½æ°æ®
                common_idx = f_vals.index.intersection(returns.index)
                if len(common_idx) < 30:
                    continue

                f_aligned = f_vals.loc[common_idx]
                r_aligned = returns.loc[common_idx]

                # å»é¤ç¼ºå¤±å¼
                valid_mask = f_aligned.notna() & r_aligned.notna()
                if valid_mask.sum() < 30:
                    continue

                # è®¡ç® IC
                ic, _ = stats.spearmanr(f_aligned[valid_mask], r_aligned[valid_mask])

                if np.isnan(ic):
                    continue

                # è®¡ç®æ»å¨ IC åºåï¼ç¨äº ICIRï¼
                # ä½¿ç¨æ¥åº¦ IC åºåï¼ç®åå¤çï¼åè®¾æ°æ®æ¯æ¥åº¦ï¼
                # å®éåºç¨ä¸­å¯è½éè¦æ´å¤æçåç»è®¡ç®
                rolling_ics = []
                window = 20
                for i in range(window, len(common_idx), window):
                    start_idx = i - window
                    end_idx = i
                    window_f = f_aligned.iloc[start_idx:end_idx]
                    window_r = r_aligned.iloc[start_idx:end_idx]
                    window_mask = window_f.notna() & window_r.notna()
                    if window_mask.sum() >= 10:
                        window_ic, _ = stats.spearmanr(
                            window_f[window_mask], window_r[window_mask]
                        )
                        if not np.isnan(window_ic):
                            rolling_ics.append(window_ic)

                # è®¡ç® ICIR
                if len(rolling_ics) >= 3:
                    ic_mean = np.mean(rolling_ics)
                    ic_std = np.std(rolling_ics)
                    icir = ic_mean / ic_std if ic_std > 0 else 0
                else:
                    ic_mean = ic
                    ic_std = 0
                    icir = 0

                stats_dict[factor_name] = {
                    'ic': float(ic),
                    'ic_mean': float(ic_mean),
                    'ic_std': float(ic_std),
                    'icir': float(icir),
                    'n_samples': int(valid_mask.sum()),
                }

            except Exception as e:
                self._logger.warning(f"Failed to calculate stats for {factor_name}: {e}")
                continue

        return stats_dict

    def _deduplicate_factors(
        self,
        factors: List[Factor],
        factor_data: Optional[pd.DataFrame],
        factor_stats: Dict[str, Dict[str, float]],
        corr_threshold: float,
    ) -> Tuple[List[Factor], Optional[pd.DataFrame]]:
        """
        å»ç¸å³å¤çï¼ç¸å³æ§ > threshold çå å­ç»ï¼ä¿ç IC æé«çå å­

        Args:
            factors: å å­å®ä¾åè¡¨
            factor_data: å å­å¼ DataFrame
            factor_stats: å å­ç»è®¡ä¿¡æ¯
            corr_threshold: ç¸å³æ§éå¼

        Returns:
            (å»éåçå å­åè¡¨, å¯¹åºçå å­å¼ DataFrame)
        """
        if factor_data is None or factor_data.empty:
            return factors, factor_data

        # è®¡ç® Spearman ç¸å³æ§ç©éµ
        try:
            # å»é¤ç¼ºå¤±å¼åè®¡ç®ç¸å³æ§
            clean_data = factor_data.dropna()
            if clean_data.empty or len(clean_data) < 10:
                return factors, factor_data

            corr_matrix = clean_data.corr(method='spearman')
        except Exception as e:
            self._logger.warning(f"Failed to compute correlation matrix: {e}")
            return factors, factor_data

        # æå»ºå å­åå°å å­çæ å°
        factor_map = {f.name: f for f in factors}

        # æ¾åºé«ç¸å³å å­ç»
        n_factors = len(factors)
        factor_names = [f.name for f in factors]
        selected_names: List[str] = []
        excluded_names: set = set()

        for i, name_i in enumerate(factor_names):
            if name_i in excluded_names:
                continue

            # æ¾å°ä¸å½åå å­é«ç¸å³çææå å­
            high_corr_group = [name_i]
            for j, name_j in enumerate(factor_names):
                if i != j and name_j not in excluded_names:
                    if name_i in corr_matrix.index and name_j in corr_matrix.columns:
                        corr_val = abs(corr_matrix.loc[name_i, name_j])
                        if corr_val >= corr_threshold:
                            high_corr_group.append(name_j)

            # å¨é«ç¸å³ç»ä¸­ä¿ç IC æé«çå å­
            if len(high_corr_group) > 1:
                best_factor = max(
                    high_corr_group,
                    key=lambda n: abs(factor_stats.get(n, {}).get('ic', 0))
                )
                selected_names.append(best_factor)
                for name in high_corr_group:
                    if name != best_factor:
                        excluded_names.add(name)
                self._logger.debug(
                    f"Correlation group: {high_corr_group}, selected: {best_factor}"
                )
            else:
                selected_names.append(name_i)

        # æå»ºç»æ
        selected_factors = [factor_map[name] for name in selected_names if name in factor_map]
        selected_data = factor_data[[name for name in selected_names if name in factor_data.columns]]

        return selected_factors, selected_data

    def _filter_by_icir(
        self,
        factors: List[Factor],
        factor_data: Optional[pd.DataFrame],
        factor_stats: Dict[str, Dict[str, float]],
        min_icir: float,
    ) -> Tuple[List[Factor], Optional[pd.DataFrame], Dict[str, Dict[str, float]]]:
        """
        æ ICIR ç­éå å­

        Args:
            factors: å å­å®ä¾åè¡¨
            factor_data: å å­å¼ DataFrame
            factor_stats: å å­ç»è®¡ä¿¡æ¯
            min_icir: æå° ICIR éå¼

        Returns:
            (ç­éåçå å­åè¡¨, å å­å¼ DataFrame, ç­éåçç»è®¡ä¿¡æ¯)
        """
        selected_factors: List[Factor] = []
        selected_names: List[str] = []
        selected_stats: Dict[str, Dict[str, float]] = {}

        for factor in factors:
            factor_name = factor.name
            stats = factor_stats.get(factor_name, {})
            icir = stats.get('icir', 0)

            if icir >= min_icir:
                selected_factors.append(factor)
                selected_names.append(factor_name)
                selected_stats[factor_name] = stats
            else:
                self._logger.debug(
                    f"Factor {factor_name} filtered: ICIR={icir:.4f} < {min_icir}"
                )

        # å¦ææ²¡æå å­éè¿ ICIR ç­éï¼ä¿ç IC æé«çä¸ä¸ª
        if not selected_factors and factors:
            best_factor = max(
                factors,
                key=lambda f: abs(factor_stats.get(f.name, {}).get('ic', 0))
            )
            selected_factors = [best_factor]
            selected_names = [best_factor.name]
            selected_stats[best_factor.name] = factor_stats.get(best_factor.name, {})
            self._logger.warning(
                f"No factors passed ICIR filter, keeping best: {best_factor.name}"
            )

        selected_data = None
        if factor_data is not None and selected_names:
            available_cols = [name for name in selected_names if name in factor_data.columns]
            if available_cols:
                selected_data = factor_data[available_cols]

        return selected_factors, selected_data, selected_stats

    def _build_composite_factor(
        self,
        factor_data: Optional[pd.DataFrame],
        factor_stats: Dict[str, Dict[str, float]],
    ) -> Optional[pd.Series]:
        """
        æ ICIR å½ä¸åæéåæç»¼åå å­

        Args:
            factor_data: å å­å¼ DataFrame
            factor_stats: å å­ç»è®¡ä¿¡æ¯

        Returns:
            ç»¼åå å­ Seriesï¼æ None
        """
        if factor_data is None or factor_data.empty:
            return None

        # è®¡ç® ICIR æé
        weights: Dict[str, float] = {}
        total_icir = 0.0

        for factor_name in factor_data.columns:
            stats = factor_stats.get(factor_name, {})
            icir = stats.get('icir', 0)
            # ä½¿ç¨ ICIR çç»å¯¹å¼ä½ä¸ºæé
            weight = max(0, abs(icir))
            weights[factor_name] = weight
            total_icir += weight

        if total_icir == 0:
            # å¦æææ ICIR é½æ¯ 0ï¼ä½¿ç¨ç­æ
            for factor_name in factor_data.columns:
                weights[factor_name] = 1.0 / len(factor_data.columns)
        else:
            # å½ä¸å
            for factor_name in weights:
                weights[factor_name] /= total_icir

        # æ´æ° factor_stats ä¸­çæé
        for factor_name, weight in weights.items():
            if factor_name in factor_stats:
                factor_stats[factor_name]['weight'] = weight

        # è®¡ç®å æåæå å­
        composite = pd.Series(0.0, index=factor_data.index)
        for factor_name, weight in weights.items():
            if factor_name in factor_data.columns:
                # å¯¹å å­å¼è¿è¡ z-score æ åå
                f_vals = factor_data[factor_name]
                f_mean = f_vals.rolling(60, min_periods=20).mean()
                f_std = f_vals.rolling(60, min_periods=20).std()
                f_zscore = (f_vals - f_mean) / f_std.replace(0, np.nan)
                composite += f_zscore.fillna(0) * weight

        composite.name = 'composite_factor'
        return composite
