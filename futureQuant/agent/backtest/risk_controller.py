"""
é£é©æ§å¶å¨æ¨¡å - RiskController

åè½ï¼
- é£é©è§åç®¡çï¼æ­¢æãæ­¢çãä»ä½ãåæ¤ï¼
- æ ¹æ®å½åæä»åçäºè°æ´ä»ä½
- æ¯ææ³¢å¨çç®æ ä»ä½è°æ´
- è¿åè°æ´åçä»ä½ä¿¡å·
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np

from ...core.logger import get_logger

logger = get_logger('agent.backtest.risk_controller')


class RiskRuleType(Enum):
    """é£é©è§åç±»å"""
    STOP_LOSS = "stop_loss"           # æ­¢æ
    TAKE_PROFIT = "take_profit"       # æ­¢ç
    MAX_POSITION = "max_position"     # æå¤§ä»ä½
    MAX_DRAWDOWN = "max_drawdown"     # æå¤§åæ¤
    VOLATILITY_TARGET = "volatility_target"  # æ³¢å¨çç®æ 


@dataclass
class RiskRule:
    """
    é£é©è§åæ°æ®ç±»
    
    Attributes:
        rule_type: è§åç±»å
        threshold: éå¼
        enabled: æ¯å¦å¯ç¨
        params: é¢å¤åæ°
    """
    rule_type: RiskRuleType
    threshold: float
    enabled: bool = True
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


class RiskController:
    """
    é£é©æ§å¶å¨
    
    ç®¡çäº¤æé£é©è§åï¼æ ¹æ®å½åæä»åçäºå¨æè°æ´ä»ä½ã
    
    é»è®¤é£é©è§åï¼
    - æ­¢æ: 5%
    - æ­¢ç: 10%
    - æå¤§ä»ä½: 30%
    - æå¤§åæ¤: 15%
    
    Attributes:
        rules: é£é©è§åå­å¸
        volatility_target: æ³¢å¨çç®æ ï¼å¹´åï¼
        current_drawdown: å½ååæ¤
        peak_equity: æçå³°å¼
    """
    
    # é»è®¤é£é©åæ°
    DEFAULT_STOP_LOSS = 0.05       # 5% æ­¢æ
    DEFAULT_TAKE_PROFIT = 0.10     # 10% æ­¢ç
    DEFAULT_MAX_POSITION = 0.30    # 30% æå¤§ä»ä½
    DEFAULT_MAX_DRAWDOWN = 0.15    # 15% æå¤§åæ¤
    DEFAULT_VOL_TARGET = 0.15      # 15% å¹´åæ³¢å¨çç®æ 
    
    def __init__(
        self,
        stop_loss: float = DEFAULT_STOP_LOSS,
        take_profit: float = DEFAULT_TAKE_PROFIT,
        max_position: float = DEFAULT_MAX_POSITION,
        max_drawdown: float = DEFAULT_MAX_DRAWDOWN,
        volatility_target: Optional[float] = DEFAULT_VOL_TARGET,
    ):
        """
        åå§åé£é©æ§å¶å¨
        
        Args:
            stop_loss: æ­¢ææ¯ä¾ï¼é»è®¤5%ï¼
            take_profit: æ­¢çæ¯ä¾ï¼é»è®¤10%ï¼
            max_position: æå¤§ä»ä½æ¯ä¾ï¼é»è®¤30%ï¼
            max_drawdown: æå¤§åæ¤æ¯ä¾ï¼é»è®¤15%ï¼
            volatility_target: æ³¢å¨çç®æ ï¼å¹´åï¼é»è®¤15%ï¼
        """
        self.rules: Dict[RiskRuleType, RiskRule] = {}
        self.volatility_target = volatility_target
        
        # åå§åé»è®¤è§å
        self._init_default_rules(
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_position=max_position,
            max_drawdown=max_drawdown,
        )
        
        # ç¶æè·è¸ª
        self.current_drawdown: float = 0.0
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.position_history: List[Dict] = []
        
        logger.info(f"RiskController initialized: "
                   f"stop_loss={stop_loss:.1%}, "
                   f"take_profit={take_profit:.1%}, "
                   f"max_position={max_position:.1%}, "
                   f"max_drawdown={max_drawdown:.1%}")
    
    def _init_default_rules(
        self,
        stop_loss: float,
        take_profit: float,
        max_position: float,
        max_drawdown: float,
    ):
        """åå§åé»è®¤é£é©è§å"""
        self.rules[RiskRuleType.STOP_LOSS] = RiskRule(
            rule_type=RiskRuleType.STOP_LOSS,
            threshold=stop_loss,
            enabled=True,
        )
        self.rules[RiskRuleType.TAKE_PROFIT] = RiskRule(
            rule_type=RiskRuleType.TAKE_PROFIT,
            threshold=take_profit,
            enabled=True,
        )
        self.rules[RiskRuleType.MAX_POSITION] = RiskRule(
            rule_type=RiskRuleType.MAX_POSITION,
            threshold=max_position,
            enabled=True,
        )
        self.rules[RiskRuleType.MAX_DRAWDOWN] = RiskRule(
            rule_type=RiskRuleType.MAX_DRAWDOWN,
            threshold=max_drawdown,
            enabled=True,
        )
        if self.volatility_target is not None:
            self.rules[RiskRuleType.VOLATILITY_TARGET] = RiskRule(
                rule_type=RiskRuleType.VOLATILITY_TARGET,
                threshold=self.volatility_target,
                enabled=True,
            )
    
    def add_rule(
        self,
        rule_type: RiskRuleType,
        threshold: float,
        enabled: bool = True,
        **params
    ):
        """
        æ·»å é£é©è§å
        
        Args:
            rule_type: è§åç±»å
            threshold: éå¼
            enabled: æ¯å¦å¯ç¨
            **params: é¢å¤åæ°
        """
        self.rules[rule_type] = RiskRule(
            rule_type=rule_type,
            threshold=threshold,
            enabled=enabled,
            params=params,
        )
        logger.debug(f"Added risk rule: {rule_type.value} = {threshold}")
    
    def enable_rule(self, rule_type: RiskRuleType):
        """å¯ç¨é£é©è§å"""
        if rule_type in self.rules:
            self.rules[rule_type].enabled = True
            logger.debug(f"Enabled risk rule: {rule_type.value}")
    
    def disable_rule(self, rule_type: RiskRuleType):
        """ç¦ç¨é£é©è§å"""
        if rule_type in self.rules:
            self.rules[rule_type].enabled = False
            logger.debug(f"Disabled risk rule: {rule_type.value}")
    
    def update_rule_threshold(self, rule_type: RiskRuleType, threshold: float):
        """
        æ´æ°è§åéå¼
        
        Args:
            rule_type: è§åç±»å
            threshold: æ°éå¼
        """
        if rule_type in self.rules:
            self.rules[rule_type].threshold = threshold
            logger.debug(f"Updated {rule_type.value} threshold to {threshold}")
    
    def update_equity(self, equity: float):
        """
        æ´æ°å½åæçå¹¶è®¡ç®åæ¤
        
        Args:
            equity: å½åæç
        """
        self.current_equity = equity
        
        # æ´æ°å³°å¼
        if equity > self.peak_equity:
            self.peak_equity = equity
        
        # è®¡ç®åæ¤
        if self.peak_equity > 0:
            self.current_drawdown = (self.peak_equity - equity) / self.peak_equity
        
        logger.debug(f"Equity updated: {equity:.2f}, "
                    f"peak={self.peak_equity:.2f}, "
                    f"drawdown={self.current_drawdown:.2%}")
    
    def apply_risk_rules(
        self,
        signal: Union[int, float],
        current_position: float,
        entry_price: float,
        current_price: float,
        unrealized_pnl_pct: float = 0.0,
        volatility: Optional[float] = None,
    ) -> Tuple[float, Optional[str]]:
        """
        åºç¨é£é©è§å
        
        æ ¹æ®å½åæä»åçäºè°æ´ä»ä½ä¿¡å·
        
        Args:
            signal: åå§ä¿¡å· (-1, 0, 1)
            current_position: å½åä»ä½æ¯ä¾
            entry_price: å¥åºä»·æ ¼
            current_price: å½åä»·æ ¼
            unrealized_pnl_pct: æªå®ç°çäºæ¯ä¾
            volatility: å½åæ³¢å¨çï¼å¹´åï¼
            
        Returns:
            (è°æ´åä¿¡å·, è§¦åè§åè¯´æ)
            è°æ´åä¿¡å·: -1(åç©º), 0(ç©ºä»), 1(åå¤)
        """
        triggered_rule = None
        adjusted_signal = signal
        
        # æ£æ¥æ­¢æ
        if self.rules[RiskRuleType.STOP_LOSS].enabled:
            stop_threshold = self.rules[RiskRuleType.STOP_LOSS].threshold
            
            # å¤å¤´æ­¢æ
            if current_position > 0 and unrealized_pnl_pct <= -stop_threshold:
                adjusted_signal = 0
                triggered_rule = f"STOP_LOSS (long, pnl={unrealized_pnl_pct:.2%})"
                logger.info(f"Stop loss triggered: {triggered_rule}")
            
            # ç©ºå¤´æ­¢æ
            elif current_position < 0 and unrealized_pnl_pct <= -stop_threshold:
                adjusted_signal = 0
                triggered_rule = f"STOP_LOSS (short, pnl={unrealized_pnl_pct:.2%})"
                logger.info(f"Stop loss triggered: {triggered_rule}")
        
        # æ£æ¥æ­¢ç
        if triggered_rule is None and self.rules[RiskRuleType.TAKE_PROFIT].enabled:
            profit_threshold = self.rules[RiskRuleType.TAKE_PROFIT].threshold
            
            # å¤å¤´æ­¢ç
            if current_position > 0 and unrealized_pnl_pct >= profit_threshold:
                adjusted_signal = 0
                triggered_rule = f"TAKE_PROFIT (long, pnl={unrealized_pnl_pct:.2%})"
                logger.info(f"Take profit triggered: {triggered_rule}")
            
            # ç©ºå¤´æ­¢ç
            elif current_position < 0 and unrealized_pnl_pct >= profit_threshold:
                adjusted_signal = 0
                triggered_rule = f"TAKE_PROFIT (short, pnl={unrealized_pnl_pct:.2%})"
                logger.info(f"Take profit triggered: {triggered_rule}")
        
        # æ£æ¥æå¤§åæ¤
        if triggered_rule is None and self.rules[RiskRuleType.MAX_DRAWDOWN].enabled:
            max_dd_threshold = self.rules[RiskRuleType.MAX_DRAWDOWN].threshold
            
            if self.current_drawdown >= max_dd_threshold:
                # åæ¤è¶éï¼æ¸ä»
                adjusted_signal = 0
                triggered_rule = f"MAX_DRAWDOWN (drawdown={self.current_drawdown:.2%})"
                logger.warning(f"Max drawdown triggered: {triggered_rule}")
        
        # æ£æ¥æå¤§ä»ä½
        if self.rules[RiskRuleType.MAX_POSITION].enabled:
            max_pos_threshold = self.rules[RiskRuleType.MAX_POSITION].threshold
            
            if abs(current_position) >= max_pos_threshold and signal != 0:
                # ä»ä½å·²è¾¾ä¸éï¼ä¸åå ä»
                if (current_position > 0 and signal > 0) or \
                   (current_position < 0 and signal < 0):
                    adjusted_signal = 0
                    triggered_rule = f"MAX_POSITION (pos={current_position:.2%})"
                    logger.debug(f"Max position limit: {triggered_rule}")
        
        return adjusted_signal, triggered_rule
    
    def calculate_volatility_adjusted_position(
        self,
        base_position: float,
        current_volatility: float,
        target_volatility: Optional[float] = None,
    ) -> float:
        """
        æ ¹æ®æ³¢å¨çç®æ è°æ´ä»ä½
        
        å¬å¼: adjusted_position = base_position * (target_vol / current_vol)
        
        Args:
            base_position: åºç¡ä»ä½
            current_volatility: å½åæ³¢å¨çï¼å¹´åï¼
            target_volatility: ç®æ æ³¢å¨çï¼é»è®¤ä½¿ç¨æ§å¶å¨è®¾ç½®
            
        Returns:
            è°æ´åçä»ä½
        """
        if target_volatility is None:
            target_volatility = self.volatility_target
        
        if target_volatility is None or current_volatility <= 0:
            return base_position
        
        # æ³¢å¨çè°æ´ç³»æ°
        vol_ratio = target_volatility / current_volatility
        
        # éå¶è°æ´èå´ (0.25x - 4x)
        vol_ratio = np.clip(vol_ratio, 0.25, 4.0)
        
        adjusted_position = base_position * vol_ratio
        
        # åºç¨æå¤§ä»ä½éå¶
        max_position = self.rules[RiskRuleType.MAX_POSITION].threshold
        adjusted_position = np.clip(adjusted_position, -max_position, max_position)
        
        logger.debug(f"Volatility adjustment: base={base_position:.2%}, "
                    f"current_vol={current_volatility:.2%}, "
                    f"target_vol={target_volatility:.2%}, "
                    f"adjusted={adjusted_position:.2%}")
        
        return adjusted_position
    
    def calculate_position_size_with_risk(
        self,
        capital: float,
        price: float,
        stop_loss_price: float,
        risk_per_trade: float = 0.02,
        max_position_pct: Optional[float] = None,
    ) -> int:
        """
        åºäºé£é©è®¡ç®ä»ä½å¤§å°
        
        å¬å¼: position = capital * risk_per_trade / |entry - stop|
        
        Args:
            capital: æ»èµé
            price: å¥åºä»·æ ¼
            stop_loss_price: æ­¢æä»·æ ¼
            risk_per_trade: åç¬äº¤æé£é©æ¯ä¾ï¼é»è®¤2%ï¼
            max_position_pct: æå¤§ä»ä½æ¯ä¾ï¼é»è®¤ä½¿ç¨è§åè®¾ç½®
            
        Returns:
            å»ºè®®ä»ä½ï¼ææ°ï¼
        """
        if max_position_pct is None:
            max_position_pct = self.rules[RiskRuleType.MAX_POSITION].threshold
        
        # è®¡ç®æ­¢æè·ç¦»
        stop_distance = abs(price - stop_loss_price)
        if stop_distance <= 0:
            logger.warning("Stop loss distance is zero, using max position")
            return int(capital * max_position_pct / price)
        
        # è®¡ç®é£é©éé¢
        risk_amount = capital * risk_per_trade
        
        # è®¡ç®ä»ä½
        position_value = risk_amount / stop_distance * price
        position = int(position_value / price)
        
        # åºç¨æå¤§ä»ä½éå¶
        max_position = int(capital * max_position_pct / price)
        position = min(position, max_position)
        
        logger.debug(f"Position size: capital={capital:.2f}, "
                    f"risk={risk_per_trade:.1%}, "
                    f"position={position}")
        
        return max(0, position)
    
    def check_all_rules(
        self,
        positions: Dict[str, Dict],
        prices: Dict[str, float],
        equity: float,
    ) -> Dict[str, Any]:
        """
        æ£æ¥æææä»çé£é©è§å
        
        Args:
            positions: æä»å­å¸ {symbol: {quantity, entry_price, side}}
            prices: å½åä»·æ ¼å­å¸ {symbol: price}
            equity: å½åæç
            
        Returns:
            é£é©æ£æ¥ç»æ
        """
        self.update_equity(equity)
        
        results = {
            'should_liquidate': [],
            'warnings': [],
            'status': 'ok',
        }
        
        for symbol, pos in positions.items():
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]
            entry_price = pos.get('entry_price', current_price)
            quantity = pos.get('quantity', 0)
            side = pos.get('side', 0)  # 1=long, -1=short
            
            if quantity <= 0:
                continue
            
            # è®¡ç®çäº
            if side > 0:
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            # æ£æ¥æ­¢æ
            if self.rules[RiskRuleType.STOP_LOSS].enabled:
                threshold = self.rules[RiskRuleType.STOP_LOSS].threshold
                if pnl_pct <= -threshold:
                    results['should_liquidate'].append({
                        'symbol': symbol,
                        'reason': 'stop_loss',
                        'pnl_pct': pnl_pct,
                    })
            
            # æ£æ¥æ­¢ç
            if self.rules[RiskRuleType.TAKE_PROFIT].enabled:
                threshold = self.rules[RiskRuleType.TAKE_PROFIT].threshold
                if pnl_pct >= threshold:
                    results['should_liquidate'].append({
                        'symbol': symbol,
                        'reason': 'take_profit',
                        'pnl_pct': pnl_pct,
                    })
        
        # æ£æ¥æå¤§åæ¤
        if self.rules[RiskRuleType.MAX_DRAWDOWN].enabled:
            threshold = self.rules[RiskRuleType.MAX_DRAWDOWN].threshold
            
            if self.current_drawdown >= threshold:
                results['status'] = 'critical'
                results['warnings'].append({
                    'type': 'max_drawdown',
                    'message': f'Max drawdown exceeded: {self.current_drawdown:.2%} >= {threshold:.2%}',
                })
                # æ·»å æææä»å°æ¸ä»åè¡¨
                for symbol in positions.keys():
                    if symbol not in [p['symbol'] for p in results['should_liquidate']]:
                        results['should_liquidate'].append({
                            'symbol': symbol,
                            'reason': 'max_drawdown',
                            'drawdown': self.current_drawdown,
                        })
        
        # æ£æ¥æ»ä»ä½
        if self.rules[RiskRuleType.MAX_POSITION].enabled:
            threshold = self.rules[RiskRuleType.MAX_POSITION].threshold
            total_position = sum(
                pos.get('quantity', 0) * prices.get(symbol, 0)
                for symbol, pos in positions.items()
            ) / equity if equity > 0 else 0
            
            if total_position > threshold:
                results['warnings'].append({
                    'type': 'max_position',
                    'message': f'Total position exceeds limit: {total_position:.2%} > {threshold:.2%}',
                })
        
        if results['should_liquidate']:
            results['status'] = 'liquidate'
        elif results['warnings']:
            results['status'] = 'warning'
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """
        è·åé£é©æ§å¶å¨ç¶æ
        
        Returns:
            ç¶æå­å¸
        """
        return {
            'current_equity': self.current_equity,
            'peak_equity': self.peak_equity,
            'current_drawdown': self.current_drawdown,
            'rules': {
                rule_type.value: {
                    'threshold': rule.threshold,
                    'enabled': rule.enabled,
                }
                for rule_type, rule in self.rules.items()
            },
        }
    
    def reset(self):
        """éç½®é£é©æ§å¶å¨ç¶æ"""
        self.current_drawdown = 0.0
        self.peak_equity = 0.0
        self.current_equity = 0.0
        self.position_history = []
        logger.info("RiskController reset")
