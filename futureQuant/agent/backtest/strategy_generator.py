"""
ç­ç¥çæå¨æ¨¡å - StrategyGenerator

åè½ï¼
- å°å å­èªå¨è½¬åä¸ºç­ç¥
- æ¯æåå å­ç­ç¥åå¤å å­ç­ç¥
- èªå¨çæç­ç¥ä»£ç å­ç¬¦ä¸²ï¼å¯æä¹åï¼
- è¿å FactorStrategy å®ä¾
"""

from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
import pandas as pd
import numpy as np

from ...core.base import Factor
from ...core.logger import get_logger
from ...strategy.base import BaseStrategy, SignalType

logger = get_logger('agent.backtest.strategy_generator')


class FactorStrategy(BaseStrategy):
    """
    å å­ç­ç¥ç±»
    
    åºäºå å­å¼çæäº¤æä¿¡å·çç­ç¥ï¼æ¯æåå å­åå¤å å­ç»åã
    
    Attributes:
        factors: å å­åè¡¨
        upper_threshold: ä¸éå¼ï¼å å­å¼å¤§äºæ­¤å¼æ¶åå¤
        lower_threshold: ä¸éå¼ï¼å å­å¼å°äºæ­¤å¼æ¶åç©º
        weighting_method: å¤å å­æéæ¹æ³
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        factors: Optional[List[Factor]] = None,
        upper_threshold: float = 1.0,
        lower_threshold: float = -1.0,
        weighting_method: str = 'equal',
        **params
    ):
        """
        åå§åå å­ç­ç¥
        
        Args:
            name: ç­ç¥åç§°
            factors: å å­åè¡¨
            upper_threshold: ä¸éå¼ï¼å å­å¼ > ä¸éå¼ â åå¤
            lower_threshold: ä¸éå¼ï¼å å­å¼ < ä¸éå¼ â åç©º
            weighting_method: å¤å å­æéæ¹æ³ ('equal', 'ic_weighted', 'custom')
            **params: å¶ä»ç­ç¥åæ°
        """
        super().__init__(name=name, **params)
        
        self.factors = factors or []
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.weighting_method = weighting_method
        
        # å å­æéï¼ç¨äºå¤å å­ç»åï¼
        self.factor_weights: Dict[str, float] = {}
        
        logger.info(f"FactorStrategy initialized: {self.name}, "
                   f"factors={len(self.factors)}, "
                   f"thresholds=({lower_threshold}, {upper_threshold})")
    
    def add_factor(self, factor: Factor, weight: Optional[float] = None):
        """
        æ·»å å å­
        
        Args:
            factor: å å­å®ä¾
            weight: å å­æéï¼å¤å å­æ¶ä½¿ç¨ï¼
        """
        self.factors.append(factor)
        if weight is not None:
            self.factor_weights[factor.name] = weight
        logger.debug(f"Added factor {factor.name} to strategy {self.name}")
    
    def compute_composite_factor(self, data: pd.DataFrame) -> pd.Series:
        """
        è®¡ç®å¤åå å­å¼
        
        æ ¹æ®weighting_methodåå¹¶å¤ä¸ªå å­å¼
        
        Args:
            data: è¾å¥æ°æ®
            
        Returns:
            å¤åå å­å¼åºå
        """
        if not self.factors:
            logger.warning("No factors in strategy")
            return pd.Series(0, index=data.index)
        
        # åå å­æåµ
        if len(self.factors) == 1:
            return self.factors[0].compute(data)
        
        # å¤å å­æåµ
        factor_values = {}
        for factor in self.factors:
            try:
                factor_values[factor.name] = factor.compute(data)
            except Exception as e:
                logger.error(f"Failed to compute factor {factor.name}: {e}")
                continue
        
        if not factor_values:
            return pd.Series(0, index=data.index)
        
        # åå¹¶ä¸ºDataFrame
        factor_df = pd.DataFrame(factor_values)
        
        # æ ¹æ®æéæ¹æ³è®¡ç®å¤åå å­
        if self.weighting_method == 'equal':
            # ç­æé
            composite = factor_df.mean(axis=1)
        elif self.weighting_method == 'ic_weighted':
            # ICå æï¼éè¦é¢åè®¡ç®çICå¼ï¼
            weights = self._get_ic_weights(factor_df.columns)
            composite = (factor_df * weights).sum(axis=1)
        elif self.weighting_method == 'custom':
            # èªå®ä¹æé
            weights = pd.Series(self.factor_weights).reindex(factor_df.columns).fillna(1.0)
            weights = weights / weights.sum()
            composite = (factor_df * weights).sum(axis=1)
        else:
            # é»è®¤ç­æé
            composite = factor_df.mean(axis=1)
        
        return composite
    
    def _get_ic_weights(self, factor_names: List[str]) -> pd.Series:
        """
        è·ååºäºICçæé
        
        Args:
            factor_names: å å­åç§°åè¡¨
            
        Returns:
            æéåºå
        """
        # ç®åçæ¬ï¼ä½¿ç¨ç¸ç­æé
        # å®éå®ç°ä¸­åºè¯¥ä»performance_trackerè·åICåå²
        weights = pd.Series(1.0, index=factor_names)
        return weights / weights.sum()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        çæäº¤æä¿¡å·
        
        ç­ç¥é»è¾ï¼
        - å å­å¼ > ä¸éå¼ â åå¤ (signal=1)
        - å å­å¼ < ä¸éå¼ â åç©º (signal=-1)
        - å¦å â ç©ºä» (signal=0)
        
        Args:
            data: è¾å¥æ°æ®ï¼åå«OHLCVç­
            
        Returns:
            DataFrame with columns: [date, signal, weight, factor_value]
        """
        try:
            # è®¡ç®å¤åå å­å¼
            composite_factor = self.compute_composite_factor(data)
            
            if composite_factor.empty:
                logger.warning("Empty composite factor")
                return pd.DataFrame(columns=['date', 'signal', 'weight', 'factor_value'])
            
            # çæä¿¡å·
            signals = pd.Series(0, index=composite_factor.index, dtype=int)
            
            # å å­å¼ > ä¸éå¼ â åå¤
            signals[composite_factor > self.upper_threshold] = 1
            
            # å å­å¼ < ä¸éå¼ â åç©º
            signals[composite_factor < self.lower_threshold] = -1
            
            # æå»ºç»æDataFrame
            result = pd.DataFrame({
                'signal': signals,
                'weight': 1.0,  # é»è®¤æéä¸º1
                'factor_value': composite_factor,
            })
            
            # æ·»å æ¥æå
            if isinstance(result.index, pd.DatetimeIndex):
                result['date'] = result.index
            else:
                result['date'] = data.index
            
            # åºç¨é£é©ç®¡ç
            result = self.apply_risk_management(result)
            
            self._signals = result
            logger.debug(f"Generated {len(result)} signals for strategy {self.name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate signals: {e}")
            return pd.DataFrame(columns=['date', 'signal', 'weight', 'factor_value'])
    
    def to_code(self) -> str:
        """
        çæç­ç¥ä»£ç å­ç¬¦ä¸²
        
        Returns:
            å¯æä¹åçç­ç¥ä»£ç å­ç¬¦ä¸²
        """
        code_lines = []
        code_lines.append('"""')
        code_lines.append(f'èªå¨çæçå å­ç­ç¥: {self.name}')
        code_lines.append(f'çææ¶é´: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
        code_lines.append('"""')
        code_lines.append('')
        code_lines.append('import pandas as pd')
        code_lines.append('import numpy as np')
        code_lines.append('from futureQuant.strategy.base import BaseStrategy')
        code_lines.append('')
        code_lines.append(f'class {self.name}Strategy(BaseStrategy):')
        code_lines.append('    """')
        code_lines.append(f'    {self.name} ç­ç¥å®ç°')
        code_lines.append('    """')
        code_lines.append('')
        code_lines.append('    def __init__(self, **params):')
        code_lines.append(f"        super().__init__(name='{self.name}', **params)")
        code_lines.append(f"        self.upper_threshold = {self.upper_threshold}")
        code_lines.append(f"        self.lower_threshold = {self.lower_threshold}")
        code_lines.append(f"        self.weighting_method = '{self.weighting_method}'")
        code_lines.append('')
        code_lines.append('    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:')
        code_lines.append('        """çæäº¤æä¿¡å·"""')
        code_lines.append('        # è®¡ç®å å­å¼ï¼è¿ééè¦æ¿æ¢ä¸ºå®éçå å­è®¡ç®é»è¾ï¼')
        code_lines.append('        factor_value = self._compute_factor(data)')
        code_lines.append('')
        code_lines.append('        # çæä¿¡å·')
        code_lines.append('        signals = pd.Series(0, index=data.index)')
        code_lines.append('        signals[factor_value > self.upper_threshold] = 1  # åå¤')
        code_lines.append('        signals[factor_value < self.lower_threshold] = -1  # åç©º')
        code_lines.append('')
        code_lines.append('        return pd.DataFrame({')
        code_lines.append("            'date': data.index,")
        code_lines.append("            'signal': signals,")
        code_lines.append("            'weight': 1.0,")
        code_lines.append("            'factor_value': factor_value,")
        code_lines.append('        })')
        code_lines.append('')
        code_lines.append('    def _compute_factor(self, data: pd.DataFrame) -> pd.Series:')
        code_lines.append('        """è®¡ç®å å­å¼ï¼è¯·å¨æ­¤å¤å®ç°å·ä½çå å­é»è¾ï¼"""')
        code_lines.append('        # TODO: å®ç°å å­è®¡ç®é»è¾')
        code_lines.append('        return pd.Series(0, index=data.index)')
        code_lines.append('')
        
        return '\n'.join(code_lines)
    
    def save_code(self, filepath: str):
        """
        ä¿å­ç­ç¥ä»£ç å°æä»¶
        
        Args:
            filepath: æä»¶è·¯å¾
        """
        try:
            code = self.to_code()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f"Strategy code saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save strategy code: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        å°ç­ç¥è½¬æ¢ä¸ºå­å¸
        
        Returns:
            ç­ç¥éç½®å­å¸
        """
        return {
            'class': 'FactorStrategy',
            'name': self.name,
            'upper_threshold': self.upper_threshold,
            'lower_threshold': self.lower_threshold,
            'weighting_method': self.weighting_method,
            'factors': [
                {'class': f.__class__.__name__, 'name': f.name, 'params': f.params}
                for f in self.factors
            ],
            'params': self.params,
        }


class StrategyGenerator:
    """
    ç­ç¥çæå¨
    
    å°å å­èªå¨è½¬åä¸ºå¯äº¤æçç­ç¥ï¼æ¯æåå å­åå¤å å­ç­ç¥çæã
    
    Attributes:
        default_upper_threshold: é»è®¤ä¸éå¼
        default_lower_threshold: é»è®¤ä¸éå¼
    """
    
    def __init__(
        self,
        default_upper_threshold: float = 1.0,
        default_lower_threshold: float = -1.0,
    ):
        """
        åå§åç­ç¥çæå¨
        
        Args:
            default_upper_threshold: é»è®¤ä¸éå¼
            default_lower_threshold: é»è®¤ä¸éå¼
        """
        self.default_upper_threshold = default_upper_threshold
        self.default_lower_threshold = default_lower_threshold
        
        logger.info("StrategyGenerator initialized")
    
    def generate(
        self,
        factors: Union[Factor, List[Factor]],
        strategy_name: Optional[str] = None,
        upper_threshold: Optional[float] = None,
        lower_threshold: Optional[float] = None,
        weighting_method: str = 'equal',
        **strategy_params
    ) -> FactorStrategy:
        """
        çæå å­ç­ç¥
        
        Args:
            factors: åä¸ªå å­æå å­åè¡¨
            strategy_name: ç­ç¥åç§°ï¼é»è®¤èªå¨çæ
            upper_threshold: ä¸éå¼ï¼é»è®¤ä½¿ç¨çæå¨é»è®¤å¼
            lower_threshold: ä¸éå¼ï¼é»è®¤ä½¿ç¨çæå¨é»è®¤å¼
            weighting_method: å¤å å­æéæ¹æ³
            **strategy_params: å¶ä»ç­ç¥åæ°
            
        Returns:
            FactorStrategyå®ä¾
        """
        # ç»ä¸è½¬æ¢ä¸ºåè¡¨
        if isinstance(factors, Factor):
            factors = [factors]
        
        # çæç­ç¥åç§°
        if strategy_name is None:
            if len(factors) == 1:
                strategy_name = f"{factors[0].name}_Strategy"
            else:
                factor_names = '_'.join([f.name for f in factors[:3]])
                if len(factors) > 3:
                    factor_names += f"_and_{len(factors)-3}_more"
                strategy_name = f"MultiFactor_{factor_names}_Strategy"
        
        # ä½¿ç¨é»è®¤å¼æä¼ å¥å¼
        upper = upper_threshold if upper_threshold is not None else self.default_upper_threshold
        lower = lower_threshold if lower_threshold is not None else self.default_lower_threshold
        
        # åå»ºç­ç¥å®ä¾
        strategy = FactorStrategy(
            name=strategy_name,
            factors=factors,
            upper_threshold=upper,
            lower_threshold=lower,
            weighting_method=weighting_method,
            **strategy_params
        )
        
        logger.info(f"Generated strategy: {strategy_name} with {len(factors)} factors")
        
        return strategy
    
    def generate_from_config(self, config: Dict[str, Any]) -> FactorStrategy:
        """
        ä»éç½®å­å¸çæç­ç¥
        
        Args:
            config: ç­ç¥éç½®å­å¸
            
        Returns:
            FactorStrategyå®ä¾
        """
        # è¿éç®åå¤çï¼å®éå®ç°éè¦å¨æå¯¼å¥å å­ç±»
        factors = config.get('factors', [])
        strategy_name = config.get('name', 'ConfigStrategy')
        upper_threshold = config.get('upper_threshold', self.default_upper_threshold)
        lower_threshold = config.get('lower_threshold', self.default_lower_threshold)
        weighting_method = config.get('weighting_method', 'equal')
        params = config.get('params', {})
        
        # åå»ºç­ç¥ï¼å å­åè¡¨ä¸ºç©ºï¼éè¦åç»­æ·»å ï¼
        strategy = FactorStrategy(
            name=strategy_name,
            factors=[],
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
            weighting_method=weighting_method,
            **params
        )
        
        logger.info(f"Generated strategy from config: {strategy_name}")
        
        return strategy
    
    def batch_generate(
        self,
        factor_groups: List[List[Factor]],
        threshold_grid: Optional[List[tuple]] = None
    ) -> List[FactorStrategy]:
        """
        æ¹éçæç­ç¥
        
        Args:
            factor_groups: å å­ç»ååè¡¨
            threshold_grid: éå¼ç»åç½æ ¼ [(upper1, lower1), (upper2, lower2), ...]
            
        Returns:
            ç­ç¥åè¡¨
        """
        strategies = []
        
        if threshold_grid is None:
            # é»è®¤éå¼ç»å
            threshold_grid = [
                (0.5, -0.5),
                (1.0, -1.0),
                (1.5, -1.5),
                (2.0, -2.0),
            ]
        
        for i, factors in enumerate(factor_groups):
            for j, (upper, lower) in enumerate(threshold_grid):
                strategy_name = f"AutoStrategy_{i}_{j}"
                try:
                    strategy = self.generate(
                        factors=factors,
                        strategy_name=strategy_name,
                        upper_threshold=upper,
                        lower_threshold=lower,
                    )
                    strategies.append(strategy)
                except Exception as e:
                    logger.error(f"Failed to generate strategy {strategy_name}: {e}")
                    continue
        
        logger.info(f"Batch generated {len(strategies)} strategies")
        
        return strategies
