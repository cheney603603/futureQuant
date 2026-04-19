# -*- coding: utf-8 -*-
"""
风控管理器 - 交易风险控制

E 方向实现：
- 仓位限制（单品种/总仓位）
- 止损止盈（固定比例/波动率自适应）
- 回撤控制（最大回撤/连续亏损）
- 风险监控告警

Author: futureQuant Team
Date: 2026-04-19
"""

from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

import pandas as pd
import numpy as np

from futureQuant.core.logger import get_logger
from futureQuant.core.exceptions import RiskError

logger = get_logger('backtest.risk')


class RiskLevel(Enum):
    """风险等级"""
    LOW = "low"           # 低风险
    MEDIUM = "medium"     # 中等风险
    HIGH = "high"         # 高风险
    CRITICAL = "critical" # 严重风险，禁止交易


@dataclass
class RiskCheckResult:
    """风控检查结果"""
    passed: bool                      # 是否通过
    risk_level: RiskLevel            # 风险等级
    reason: str                      # 原因说明
    action: str                      # 建议操作
    metrics: Dict[str, float] = field(default_factory=dict)  # 相关指标


@dataclass
class PositionLimit:
    """仓位限制配置"""
    max_single_position_pct: float = 0.2    # 单品种最大仓位 20%
    max_total_position_pct: float = 0.8     # 总仓位最大 80%
    max_concentration_pct: float = 0.3      # 单一品种集中度 30%
    min_cash_reserve_pct: float = 0.1       # 最小现金储备 10%


@dataclass
class StopLossConfig:
    """止损配置"""
    fixed_pct: float = 0.05                 # 固定止损比例 5%
    trailing_pct: Optional[float] = None    # 追踪止损比例
    volatility_multiplier: float = 2.0      # 波动率倍数（ATR倍数）
    use_volatility: bool = False            # 是否使用波动率自适应


@dataclass
class DrawdownConfig:
    """回撤控制配置"""
    max_daily_drawdown_pct: float = 0.03    # 单日最大回撤 3%
    max_total_drawdown_pct: float = 0.15    # 总回撤限制 15%
    consecutive_loss_days: int = 3          # 连续亏损天数限制


class RiskManager:
    """
    风控管理器
    
    统一管理交易风险，包括：
    - 仓位控制
    - 止损止盈
    - 回撤限制
    - 风险告警
    
    Example:
        >>> risk_mgr = RiskManager(
        ...     max_position_pct=0.5,
        ...     stop_loss_pct=0.05,
        ...     max_drawdown_pct=0.15
        ... )
        >>> 
        >>> # 检查仓位
        >>> can_trade, reason = risk_mgr.check_position_limit(
        ...     current_position=0.3,
        ...     target_position=0.6
        ... )
        >>> 
        >>> # 检查止损
        >>> should_stop, reason = risk_mgr.check_stop_loss(
        ...     entry_price=3000,
        ...     current_price=2800,
        ...     position_side='long'
        ... )
    """
    
    def __init__(
        self,
        position_limits: Optional[PositionLimit] = None,
        stop_loss: Optional[StopLossConfig] = None,
        drawdown: Optional[DrawdownConfig] = None,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        """
        初始化风控管理器
        
        Args:
            position_limits: 仓位限制配置
            stop_loss: 止损配置
            drawdown: 回撤控制配置
            alert_callbacks: 风险告警回调函数列表
        """
        self.position_limits = position_limits or PositionLimit()
        self.stop_loss = stop_loss or StopLossConfig()
        self.drawdown = drawdown or DrawdownConfig()
        self.alert_callbacks = alert_callbacks or []
        
        # 状态追踪
        self.peak_value: float = 0.0
        self.consecutive_losses: int = 0
        self.daily_pnl: List[float] = []
        self.risk_history: List[Dict] = []
        
    def check_position_limit(
        self,
        current_position: float,
        target_position: float,
        portfolio_value: float = 100000,
        variety: Optional[str] = None
    ) -> RiskCheckResult:
        """
        检查仓位限制
        
        Args:
            current_position: 当前仓位（0-1）
            target_position: 目标仓位（0-1）
            portfolio_value: 组合总价值
            variety: 品种代码
            
        Returns:
            RiskCheckResult
        """
        limits = self.position_limits
        
        # 检查总仓位限制
        if target_position > limits.max_total_position_pct:
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.HIGH,
                reason=f"目标仓位 {target_position:.1%} 超过最大限制 {limits.max_total_position_pct:.1%}",
                action="reduce_position",
                metrics={'target_position': target_position, 'max_allowed': limits.max_total_position_pct}
            )
        
        # 检查单品种仓位（如果有品种信息）
        if variety and target_position > limits.max_single_position_pct:
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.MEDIUM,
                reason=f"品种 {variety} 仓位 {target_position:.1%} 超过单品种限制 {limits.max_single_position_pct:.1%}",
                action="reduce_single_position",
                metrics={'variety': variety, 'target_position': target_position}
            )
        
        # 检查现金储备
        cash_reserve = 1 - target_position
        if cash_reserve < limits.min_cash_reserve_pct:
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.MEDIUM,
                reason=f"现金储备 {cash_reserve:.1%} 低于最低要求 {limits.min_cash_reserve_pct:.1%}",
                action="increase_cash_reserve",
                metrics={'cash_reserve': cash_reserve, 'min_required': limits.min_cash_reserve_pct}
            )
        
        return RiskCheckResult(
            passed=True,
            risk_level=RiskLevel.LOW,
            reason="仓位检查通过",
            action="proceed",
            metrics={'target_position': target_position}
        )
    
    def check_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        position_side: str = 'long',
        atr: Optional[float] = None
    ) -> RiskCheckResult:
        """
        检查止损条件
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            position_side: 'long' 或 'short'
            atr: 平均真实波幅（用于波动率自适应止损）
            
        Returns:
            RiskCheckResult
        """
        if position_side == 'long':
            loss_pct = (entry_price - current_price) / entry_price
        else:
            loss_pct = (current_price - entry_price) / entry_price
        
        # 计算止损阈值
        if self.stop_loss.use_volatility and atr is not None:
            stop_threshold = (atr * self.stop_loss.volatility_multiplier) / entry_price
        else:
            stop_threshold = self.stop_loss.fixed_pct
        
        if loss_pct > stop_threshold:
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.HIGH,
                reason=f"触发止损: 亏损 {loss_pct:.2%} 超过阈值 {stop_threshold:.2%}",
                action="close_position",
                metrics={
                    'loss_pct': loss_pct,
                    'stop_threshold': stop_threshold,
                    'entry_price': entry_price,
                    'current_price': current_price
                }
            )
        
        return RiskCheckResult(
            passed=True,
            risk_level=RiskLevel.LOW,
            reason="未触发止损",
            action="hold",
            metrics={'loss_pct': loss_pct, 'stop_threshold': stop_threshold}
        )
    
    def check_take_profit(
        self,
        entry_price: float,
        current_price: float,
        position_side: str = 'long',
        profit_pct: float = 0.1
    ) -> RiskCheckResult:
        """
        检查止盈条件
        
        Args:
            entry_price: 入场价格
            current_price: 当前价格
            position_side: 'long' 或 'short'
            profit_pct: 目标盈利比例
            
        Returns:
            RiskCheckResult
        """
        if position_side == 'long':
            current_profit = (current_price - entry_price) / entry_price
        else:
            current_profit = (entry_price - current_price) / entry_price
        
        if current_profit >= profit_pct:
            return RiskCheckResult(
                passed=False,  # 返回 False 表示触发条件
                risk_level=RiskLevel.LOW,
                reason=f"触发止盈: 盈利 {current_profit:.2%} 达到目标 {profit_pct:.2%}",
                action="take_profit",
                metrics={'profit_pct': current_profit, 'target': profit_pct}
            )
        
        return RiskCheckResult(
            passed=True,
            risk_level=RiskLevel.LOW,
            reason="未触发止盈",
            action="hold",
            metrics={'profit_pct': current_profit}
        )
    
    def check_drawdown(
        self,
        peak_value: float,
        current_value: float,
        daily_pnl: Optional[float] = None
    ) -> RiskCheckResult:
        """
        检查回撤限制
        
        Args:
            peak_value: 历史峰值
            current_value: 当前价值
            daily_pnl: 当日盈亏（用于单日回撤检查）
            
        Returns:
            RiskCheckResult
        """
        # 更新峰值
        if current_value > self.peak_value:
            self.peak_value = current_value
        
        # 计算总回撤
        total_drawdown = (self.peak_value - current_value) / self.peak_value if self.peak_value > 0 else 0
        
        # 检查总回撤
        if total_drawdown > self.drawdown.max_total_drawdown_pct:
            return RiskCheckResult(
                passed=False,
                risk_level=RiskLevel.CRITICAL,
                reason=f"总回撤 {total_drawdown:.2%} 超过限制 {self.drawdown.max_total_drawdown_pct:.2%}",
                action="stop_trading",
                metrics={
                    'total_drawdown': total_drawdown,
                    'peak_value': self.peak_value,
                    'current_value': current_value
                }
            )
        
        # 检查单日回撤
        if daily_pnl is not None:
            daily_return = daily_pnl / current_value if current_value > 0 else 0
            if daily_return < -self.drawdown.max_daily_drawdown_pct:
                self.consecutive_losses += 1
                
                # 检查连续亏损天数
                if self.consecutive_losses >= self.drawdown.consecutive_loss_days:
                    return RiskCheckResult(
                        passed=False,
                        risk_level=RiskLevel.HIGH,
                        reason=f"连续亏损 {self.consecutive_losses} 天",
                        action="pause_trading",
                        metrics={
                            'consecutive_losses': self.consecutive_losses,
                            'daily_return': daily_return
                        }
                    )
            else:
                self.consecutive_losses = 0
        
        return RiskCheckResult(
            passed=True,
            risk_level=RiskLevel.LOW if total_drawdown < 0.05 else RiskLevel.MEDIUM,
            reason=f"回撤检查通过 (当前回撤: {total_drawdown:.2%})",
            action="proceed",
            metrics={'total_drawdown': total_drawdown}
        )
    
    def check_all(
        self,
        position_info: Dict[str, Any],
        price_info: Dict[str, float],
        portfolio_info: Dict[str, float]
    ) -> List[RiskCheckResult]:
        """
        执行所有风控检查
        
        Args:
            position_info: 仓位信息
            price_info: 价格信息
            portfolio_info: 组合信息
            
        Returns:
            List[RiskCheckResult] 所有检查结果
        """
        results = []
        
        # 仓位检查
        if 'current_position' in position_info and 'target_position' in position_info:
            results.append(self.check_position_limit(
                position_info['current_position'],
                position_info['target_position'],
                portfolio_info.get('total_value', 100000),
                position_info.get('variety')
            ))
        
        # 止损检查
        if 'entry_price' in price_info and 'current_price' in price_info:
            results.append(self.check_stop_loss(
                price_info['entry_price'],
                price_info['current_price'],
                position_info.get('side', 'long'),
                price_info.get('atr')
            ))
        
        # 回撤检查
        if 'peak_value' in portfolio_info and 'current_value' in portfolio_info:
            results.append(self.check_drawdown(
                portfolio_info['peak_value'],
                portfolio_info['current_value'],
                portfolio_info.get('daily_pnl')
            ))
        
        # 记录历史
        self.risk_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'passed': r.passed,
                    'risk_level': r.risk_level.value,
                    'reason': r.reason,
                    'action': r.action
                }
                for r in results
            ]
        })
        
        # 触发告警
        for result in results:
            if not result.passed or result.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                self._trigger_alert(result)
        
        return results
    
    def _trigger_alert(self, result: RiskCheckResult):
        """触发风险告警"""
        msg = f"[Risk Alert] {result.risk_level.value.upper()}: {result.reason}"
        logger.warning(msg)
        
        for callback in self.alert_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """获取风险汇总"""
        return {
            'peak_value': self.peak_value,
            'consecutive_losses': self.consecutive_losses,
            'total_checks': len(self.risk_history),
            'recent_alerts': [
                h for h in self.risk_history[-10:]
                if any(not r['passed'] for r in h['results'])
            ]
        }
    
    def reset(self):
        """重置状态"""
        self.peak_value = 0.0
        self.consecutive_losses = 0
        self.daily_pnl = []
        self.risk_history = []
        logger.info("RiskManager reset")


# 便捷函数
def create_default_risk_manager() -> RiskManager:
    """创建默认风控管理器"""
    return RiskManager(
        position_limits=PositionLimit(
            max_single_position_pct=0.2,
            max_total_position_pct=0.8,
            min_cash_reserve_pct=0.1
        ),
        stop_loss=StopLossConfig(
            fixed_pct=0.05,
            use_volatility=False
        ),
        drawdown=DrawdownConfig(
            max_daily_drawdown_pct=0.03,
            max_total_drawdown_pct=0.15,
            consecutive_loss_days=3
        )
    )