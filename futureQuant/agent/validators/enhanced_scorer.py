"""
å¢å¼ºå å­è¯ä¼°æ¨¡å

å¨åæ 5 ç»´è¯ååºç¡ä¸ï¼æ°å¢ 4 ä¸ªç»´åº¦ï¼å½¢æ 9 ç»´è¯åä½ç³»ã
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from datetime import datetime

from ...core.logger import get_logger

logger = get_logger('agent.validators.enhanced_scorer')


class EnhancedMultiDimensionalScorer:
    """
    å¢å¼ºçå¤ç»´åº¦å å­è¯åå¨
    
    å¨åæ 5 ç»´è¯ååºç¡ä¸ï¼æ°å¢ 4 ä¸ªç»´åº¦ï¼
    - å¯äº¤ææ§ (15%): æ¢æçãæµå¨æ§ãæ»ç¹ææ¬
    - é²æ£æ§ (15%): åæ°æææ§ãç¨³å®æ§
    - ç¬ç«æ§ (10%): ä¸å¶ä»å å­çç¸å³æ§
    - åç»æ§ (10%): å¯¹æªæ¥æ¶ççé¢æµè½å
    
    æ°çè¯åæé:
    - é¢æµè½å: 30% (å 35%)
    - ç¨³å®æ§: 20% (å 25%)
    - åè°æ§: 15% (å 20%)
    - å¯äº¤ææ§: 15% (æ°å¢)
    - é²æ£æ§: 15% (æ°å¢)
    - ç¬ç«æ§: 5% (æ°å¢)
    """
    
    def __init__(self):
        """åå§åå¢å¼ºè¯åå¨"""
        self.logger = logger
        
        # æ°çè¯åæé
        self.weights = {
            'predictability': 0.30,      # é¢æµè½å
            'stability': 0.20,            # ç¨³å®æ§
            'monotonicity': 0.15,         # åè°æ§
            'tradability': 0.15,          # å¯äº¤ææ§
            'robustness': 0.15,           # é²æ£æ§
            'independence': 0.05,         # ç¬ç«æ§
        }
    
    def score_tradability(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        volume: pd.Series,
        price: pd.Series,
    ) -> float:
        """
        è®¡ç®å¯äº¤ææ§è¯å
        
        Args:
            factor_values: å å­å¼åºå
            returns: æ¶ççåºå
            volume: æäº¤éåºå
            price: ä»·æ ¼åºå
            
        Returns:
            å¯äº¤ææ§è¯å (0-1)
        """
        try:
            # 1. æ¢æçè¯å
            factor_change = factor_values.diff().abs().mean()
            turnover_score = 1.0 / (1.0 + factor_change)  # æ¢æçè¶ä½è¶å¥½
            
            # 2. æµå¨æ§è¯å
            volume_mean = volume.mean()
            volume_std = volume.std()
            liquidity_score = min(1.0, volume_mean / (volume_std + 1e-8))
            
            # 3. æ»ç¹ææ¬è¯å
            # åºäºä»·æ ¼æ³¢å¨çä¼°è®¡æ»ç¹
            price_volatility = price.pct_change().std()
            slippage_score = 1.0 / (1.0 + price_volatility * 100)
            
            # ç»¼åå¯äº¤ææ§è¯å
            tradability = (turnover_score * 0.4 + 
                          liquidity_score * 0.4 + 
                          slippage_score * 0.2)
            
            return min(1.0, max(0.0, tradability))
            
        except Exception as e:
            self.logger.error(f"Failed to score tradability: {e}")
            return 0.5
    
    def score_robustness(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        param_ranges: Dict[str, Tuple[float, float]],
    ) -> float:
        """
        è®¡ç®é²æ£æ§è¯å
        
        Args:
            factor_values: å å­å¼åºå
            returns: æ¶ççåºå
            param_ranges: åæ°èå´å­å¸
            
        Returns:
            é²æ£æ§è¯å (0-1)
        """
        try:
            # 1. åæ°æææ§åæ
            # è®¡ç®å å­å¼çç¨³å®æ§
            factor_stability = 1.0 - (factor_values.std() / (factor_values.abs().mean() + 1e-8))
            
            # 2. IC ç¨³å®æ§
            ic_values = []
            for i in range(len(factor_values) - 1):
                if len(factor_values[i:i+20]) > 5:
                    ic, _ = spearmanr(factor_values[i:i+20], returns[i:i+20])
                    ic_values.append(ic)
            
            ic_stability = 1.0 - (np.std(ic_values) / (np.abs(np.mean(ic_values)) + 1e-8)) if ic_values else 0.5
            
            # ç»¼åé²æ£æ§è¯å
            robustness = (factor_stability * 0.5 + ic_stability * 0.5)
            
            return min(1.0, max(0.0, robustness))
            
        except Exception as e:
            self.logger.error(f"Failed to score robustness: {e}")
            return 0.5
    
    def score_independence(
        self,
        factor_values: pd.Series,
        other_factors: List[pd.Series],
    ) -> float:
        """
        è®¡ç®ç¬ç«æ§è¯å
        
        Args:
            factor_values: å å­å¼åºå
            other_factors: å¶ä»å å­å¼åºååè¡¨
            
        Returns:
            ç¬ç«æ§è¯å (0-1)
        """
        try:
            if not other_factors:
                return 1.0
            
            # è®¡ç®ä¸å¶ä»å å­çç¸å³æ§
            correlations = []
            for other in other_factors:
                if len(factor_values) == len(other):
                    corr, _ = spearmanr(factor_values, other)
                    correlations.append(abs(corr))
            
            if not correlations:
                return 1.0
            
            # ç¸å³æ§è¶ä½è¶å¥½
            mean_corr = np.mean(correlations)
            independence = 1.0 - mean_corr
            
            return min(1.0, max(0.0, independence))
            
        except Exception as e:
            self.logger.error(f"Failed to score independence: {e}")
            return 0.5
    
    def score_forward_looking(
        self,
        factor_values: pd.Series,
        future_returns: pd.Series,
        lag: int = 1,
    ) -> float:
        """
        è®¡ç®åç»æ§è¯å
        
        Args:
            factor_values: å å­å¼åºå
            future_returns: æªæ¥æ¶ççåºå
            lag: æ»åææ°
            
        Returns:
            åç»æ§è¯å (0-1)
        """
        try:
            # è®¡ç®å å­å¯¹æªæ¥æ¶ççé¢æµè½å
            if len(factor_values) <= lag:
                return 0.5
            
            # è®¡ç® IC
            ic, _ = spearmanr(factor_values[:-lag], future_returns[lag:])
            
            # å° IC è½¬æ¢ä¸ºè¯å (IC èå´ -1 å° 1)
            forward_looking = (ic + 1.0) / 2.0
            
            return min(1.0, max(0.0, forward_looking))
            
        except Exception as e:
            self.logger.error(f"Failed to score forward looking: {e}")
            return 0.5
    
    def calculate_enhanced_score(
        self,
        factor_values: pd.Series,
        returns: pd.Series,
        ic_mean: float,
        icir: float,
        ic_win_rate: float,
        monotonicity: float,
        turnover: float,
        volume: Optional[pd.Series] = None,
        price: Optional[pd.Series] = None,
        other_factors: Optional[List[pd.Series]] = None,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """
        è®¡ç®å¢å¼ºçç»¼åè¯å
        
        Args:
            factor_values: å å­å¼åºå
            returns: æ¶ççåºå
            ic_mean: IC åå¼
            icir: ICIR
            ic_win_rate: IC èç
            monotonicity: åè°æ§
            turnover: æ¢æç
            volume: æäº¤éåºå (å¯é)
            price: ä»·æ ¼åºå (å¯é)
            other_factors: å¶ä»å å­åºå (å¯é)
            param_ranges: åæ°èå´ (å¯é)
            
        Returns:
            åå«åç»´åº¦è¯ååç»¼åè¯åçå­å¸
        """
        try:
            # 1. é¢æµè½åè¯å (30%)
            predictability_score = (ic_mean * 0.4 + 
                                   icir * 0.3 + 
                                   ic_win_rate * 0.3)
            
            # 2. ç¨³å®æ§è¯å (20%)
            stability_score = 1.0 - min(1.0, turnover)
            
            # 3. åè°æ§è¯å (15%)
            monotonicity_score = monotonicity
            
            # 4. å¯äº¤ææ§è¯å (15%)
            if volume is not None and price is not None:
                tradability_score = self.score_tradability(
                    factor_values, returns, volume, price
                )
            else:
                tradability_score = 0.5
            
            # 5. é²æ£æ§è¯å (15%)
            if param_ranges is not None:
                robustness_score = self.score_robustness(
                    factor_values, returns, param_ranges
                )
            else:
                robustness_score = 0.5
            
            # 6. ç¬ç«æ§è¯å (5%)
            if other_factors is not None:
                independence_score = self.score_independence(
                    factor_values, other_factors
                )
            else:
                independence_score = 0.5
            
            # è®¡ç®ç»¼åè¯å
            overall_score = (
                predictability_score * self.weights['predictability'] +
                stability_score * self.weights['stability'] +
                monotonicity_score * self.weights['monotonicity'] +
                tradability_score * self.weights['tradability'] +
                robustness_score * self.weights['robustness'] +
                independence_score * self.weights['independence']
            )
            
            return {
                'overall_score': overall_score,
                'predictability_score': predictability_score,
                'stability_score': stability_score,
                'monotonicity_score': monotonicity_score,
                'tradability_score': tradability_score,
                'robustness_score': robustness_score,
                'independence_score': independence_score,
                'weights': self.weights,
                'timestamp': datetime.now().isoformat(),
            }
            
        except Exception as e:
            self.logger.error(f"Failed to calculate enhanced score: {e}")
            return {
                'overall_score': 0.0,
                'error': str(e),
            }
