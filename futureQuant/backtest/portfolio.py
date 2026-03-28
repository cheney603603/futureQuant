"""
仓位管理模块 - Portfolio

功能：
- 多品种仓位跟踪
- 资金计算与分配
- 风险监控（集中度、敞口）
- 仓位调整与再平衡
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from ..core.logger import get_logger
from ..core.exceptions import BacktestError

logger = get_logger('backtest.portfolio')


class PositionSide(Enum):
    """持仓方向"""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class PositionDetail:
    """持仓详情"""
    symbol: str
    side: PositionSide
    quantity: int = 0
    avg_price: float = 0.0
    current_price: float = 0.0
    margin: float = 0.0
    market_value: float = 0.0
    weight: float = 0.0  # 占组合权重
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def pnl_pct(self) -> float:
        """盈亏百分比"""
        if self.avg_price == 0:
            return 0.0
        return (self.current_price - self.avg_price) / self.avg_price * self.side.value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'current_price': self.current_price,
            'margin': self.margin,
            'market_value': self.market_value,
            'weight': self.weight,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'pnl_pct': self.pnl_pct,
        }


class Portfolio:
    """
    投资组合管理器
    
    功能：
    1. 多品种仓位跟踪
    2. 资金计算与分配
    3. 风险监控（集中度、敞口、保证金使用率）
    4. 仓位调整与再平衡
    
    Attributes:
        initial_capital: 初始资金
        margin_rate: 保证金率
        max_concentration: 最大单品种集中度
        max_leverage: 最大杠杆倍数
        contract_multipliers: 合约乘数字典
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        margin_rate: float = 0.1,
        max_concentration: float = 0.3,
        max_leverage: float = 3.0,
        contract_multipliers: Optional[Dict[str, float]] = None,
    ):
        """
        初始化投资组合
        
        Args:
            initial_capital: 初始资金
            margin_rate: 保证金率
            max_concentration: 最大单品种集中度（默认30%）
            max_leverage: 最大杠杆倍数（默认3倍）
            contract_multipliers: 合约乘数字典
        """
        self.initial_capital = initial_capital
        self.margin_rate = margin_rate
        self.max_concentration = max_concentration
        self.max_leverage = max_leverage
        self.contract_multipliers = contract_multipliers or {}
        
        # 资金状态
        self.cash = initial_capital
        self.available_margin = initial_capital
        self.used_margin = 0.0
        
        # 持仓
        self.positions: Dict[str, PositionDetail] = {}
        
        # 历史记录
        self.position_history: List[Dict] = []
        self.value_history: List[Dict] = []
        
        # 统计
        self.total_trades = 0
        self.total_commission = 0.0
        
        logger.info(f"Portfolio initialized: capital={initial_capital}, "
                   f"margin_rate={margin_rate}, max_concentration={max_concentration}")
    
    def _get_multiplier(self, symbol: str) -> float:
        """获取合约乘数"""
        return self.contract_multipliers.get(symbol, 10.0)
    
    def update_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: int,
        price: float,
        commission: float = 0.0,
    ) -> PositionDetail:
        """
        更新持仓
        
        Args:
            symbol: 合约代码
            side: 持仓方向
            quantity: 数量（绝对值）
            price: 成交价格
            commission: 手续费
            
        Returns:
            更新后的持仓详情
        """
        multiplier = self._get_multiplier(symbol)
        
        if symbol not in self.positions:
            # 新开仓
            market_value = price * quantity * multiplier
            margin = market_value * self.margin_rate
            
            self.positions[symbol] = PositionDetail(
                symbol=symbol,
                side=side,
                quantity=quantity,
                avg_price=price,
                current_price=price,
                margin=margin,
                market_value=market_value,
            )
        else:
            position = self.positions[symbol]
            
            if side == position.side or position.side == PositionSide.FLAT:
                # 同向加仓或新开仓
                if position.side == PositionSide.FLAT:
                    position.side = side
                    position.avg_price = price
                    position.quantity = quantity
                else:
                    # 计算新的均价
                    total_cost = position.avg_price * position.quantity + price * quantity
                    position.quantity += quantity
                    position.avg_price = total_cost / position.quantity
                
                position.market_value = price * position.quantity * multiplier
                position.margin = position.market_value * self.margin_rate
                
            else:
                # 反向操作（平仓或反手）
                if quantity >= position.quantity:
                    # 全部平仓或反手
                    close_qty = position.quantity
                    
                    # 计算实现盈亏
                    price_diff = price - position.avg_price
                    if position.side == PositionSide.SHORT:
                        price_diff = -price_diff
                    realized_pnl = price_diff * close_qty * multiplier
                    position.realized_pnl += realized_pnl
                    
                    remaining = quantity - close_qty
                    if remaining > 0:
                        # 反手开仓
                        position.side = side
                        position.quantity = remaining
                        position.avg_price = price
                        position.market_value = price * remaining * multiplier
                        position.margin = position.market_value * self.margin_rate
                        position.unrealized_pnl = 0.0
                    else:
                        # 全部平仓
                        position.side = PositionSide.FLAT
                        position.quantity = 0
                        position.avg_price = 0.0
                        position.market_value = 0.0
                        position.margin = 0.0
                        position.unrealized_pnl = 0.0
                else:
                    # 部分平仓
                    close_qty = quantity
                    
                    # 计算实现盈亏
                    price_diff = price - position.avg_price
                    if position.side == PositionSide.SHORT:
                        price_diff = -price_diff
                    realized_pnl = price_diff * close_qty * multiplier
                    position.realized_pnl += realized_pnl
                    
                    position.quantity -= close_qty
                    position.market_value = price * position.quantity * multiplier
                    position.margin = position.market_value * self.margin_rate
        
        # 更新权重
        self._update_weights()
        
        # 更新资金
        self._update_margin()
        
        # 更新统计
        self.total_trades += 1
        self.total_commission += commission
        
        logger.debug(f"Position updated: {symbol}, side={side.value}, qty={quantity}, price={price}")
        
        return self.positions[symbol]
    
    def update_prices(self, prices: Dict[str, float]):
        """
        更新持仓价格（用于计算浮动盈亏）
        
        Args:
            prices: {symbol: price} 当前价格字典
        """
        for symbol, price in prices.items():
            if symbol in self.positions:
                position = self.positions[symbol]
                if position.quantity == 0:
                    continue
                
                multiplier = self._get_multiplier(symbol)
                
                # 更新当前价格
                position.current_price = price
                
                # 计算市值
                position.market_value = price * position.quantity * multiplier
                
                # 计算浮动盈亏
                price_diff = price - position.avg_price
                if position.side == PositionSide.SHORT:
                    price_diff = -price_diff
                position.unrealized_pnl = price_diff * position.quantity * multiplier
                
                # 更新保证金（按当前价格）
                position.margin = position.market_value * self.margin_rate
        
        # 更新权重和资金
        self._update_weights()
        self._update_margin()
    
    def _update_weights(self):
        """更新持仓权重"""
        total_value = self.get_total_value()
        if total_value > 0:
            for position in self.positions.values():
                position.weight = abs(position.market_value) / total_value
    
    def _update_margin(self):
        """更新保证金状态"""
        self.used_margin = sum(pos.margin for pos in self.positions.values())
        self.available_margin = self.cash - self.used_margin
    
    def get_position(self, symbol: str) -> Optional[PositionDetail]:
        """
        获取指定品种的持仓
        
        Args:
            symbol: 合约代码
            
        Returns:
            PositionDetail或None
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, PositionDetail]:
        """
        获取所有持仓
        
        Returns:
            {symbol: PositionDetail}字典
        """
        return {k: v for k, v in self.positions.items() if v.quantity > 0}
    
    def get_position_values(self) -> pd.DataFrame:
        """
        获取持仓价值DataFrame
        
        Returns:
            DataFrame with columns: symbol, side, quantity, avg_price, 
                                   current_price, market_value, weight, unrealized_pnl
        """
        positions = self.get_all_positions()
        if not positions:
            return pd.DataFrame()
        
        data = [p.to_dict() for p in positions.values()]
        return pd.DataFrame(data)
    
    def get_total_value(self) -> float:
        """
        获取组合总市值（含浮动盈亏）
        
        Returns:
            总市值
        """
        position_value = sum(abs(pos.market_value) for pos in self.positions.values())
        return self.cash + position_value
    
    def get_net_value(self) -> float:
        """
        获取组合净值（资金 + 保证金 + 浮动盈亏）
        
        Returns:
            净值
        """
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.cash + self.used_margin + unrealized_pnl
    
    def get_available_margin(self) -> float:
        """
        获取可用保证金
        
        Returns:
            可用保证金
        """
        return self.available_margin
    
    def get_margin_info(self) -> Dict[str, float]:
        """
        获取保证金信息
        
        Returns:
            {
                'used': 已用保证金,
                'available': 可用保证金,
                'total': 总资金,
                'ratio': 保证金使用率,
                'leverage': 杠杆倍数
            }
        """
        total = self.get_net_value()
        ratio = self.used_margin / total if total > 0 else 0
        leverage = self.used_margin / total if total > 0 else 0
        
        return {
            'used': self.used_margin,
            'available': self.available_margin,
            'total': total,
            'ratio': ratio,
            'leverage': leverage,
        }
    
    def get_exposure(self) -> Dict[str, float]:
        """
        获取风险敞口
        
        Returns:
            {
                'long_exposure': 多头敞口,
                'short_exposure': 空头敞口,
                'net_exposure': 净敞口,
                'gross_exposure': 总敞口
            }
        """
        long_exposure = sum(
            pos.market_value for pos in self.positions.values() 
            if pos.side == PositionSide.LONG
        )
        short_exposure = sum(
            pos.market_value for pos in self.positions.values() 
            if pos.side == PositionSide.SHORT
        )
        
        return {
            'long_exposure': long_exposure,
            'short_exposure': short_exposure,
            'net_exposure': long_exposure - short_exposure,
            'gross_exposure': long_exposure + short_exposure,
        }
    
    def get_concentration_risk(self) -> Dict[str, Any]:
        """
        获取集中度风险
        
        Returns:
            {
                'max_weight': 最大权重,
                'max_weight_symbol': 最大权重品种,
                'concentration_violation': 是否违反集中度限制,
                'violations': [违规品种列表]
            }
        """
        max_weight = 0.0
        max_symbol = None
        violations = []
        
        for symbol, position in self.positions.items():
            if position.weight > max_weight:
                max_weight = position.weight
                max_symbol = symbol
            
            if position.weight > self.max_concentration:
                violations.append({
                    'symbol': symbol,
                    'weight': position.weight,
                    'limit': self.max_concentration
                })
        
        return {
            'max_weight': max_weight,
            'max_weight_symbol': max_symbol,
            'concentration_violation': len(violations) > 0,
            'violations': violations,
        }
    
    def check_risk_limits(self) -> Dict[str, Any]:
        """
        检查风险限制
        
        Returns:
            {
                'margin_call': 是否触发强平,
                'concentration_violation': 是否违反集中度,
                'leverage_violation': 是否超过杠杆限制,
                'details': 详细信息
            }
        """
        margin_info = self.get_margin_info()
        exposure = self.get_exposure()
        concentration = self.get_concentration_risk()
        
        # 检查保证金
        margin_call = margin_info['ratio'] > 0.9  # 保证金使用率超过90%
        
        # 检查杠杆
        leverage_violation = margin_info['leverage'] > self.max_leverage
        
        return {
            'margin_call': margin_call,
            'concentration_violation': concentration['concentration_violation'],
            'leverage_violation': leverage_violation,
            'details': {
                'margin': margin_info,
                'exposure': exposure,
                'concentration': concentration,
            }
        }
    
    def calculate_position_size(
        self,
        symbol: str,
        target_weight: float,
        price: float,
    ) -> int:
        """
        计算目标仓位大小
        
        Args:
            symbol: 合约代码
            target_weight: 目标权重（-1到1，负数为空头）
            price: 当前价格
            
        Returns:
            目标手数（正数）
        """
        multiplier = self._get_multiplier(symbol)
        total_value = self.get_net_value()
        
        # 目标市值
        target_value = total_value * abs(target_weight)
        
        # 计算手数
        lots = int(target_value / (price * multiplier))
        
        return max(0, lots)
    
    def rebalance(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> List[Dict]:
        """
        组合再平衡
        
        Args:
            target_weights: {symbol: weight} 目标权重字典
            prices: {symbol: price} 当前价格字典
            
        Returns:
            交易指令列表
        """
        orders = []
        
        # 所有涉及的品种
        all_symbols = set(self.positions.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            target_weight = target_weights.get(symbol, 0)
            current_position = self.positions.get(symbol)
            
            multiplier = self._get_multiplier(symbol)
            price = prices.get(symbol)
            
            if price is None:
                continue
            
            # 计算目标手数
            target_lots = self.calculate_position_size(symbol, target_weight, price)
            
            # 确定方向
            if target_weight > 0:
                target_side = PositionSide.LONG
            elif target_weight < 0:
                target_side = PositionSide.SHORT
            else:
                target_side = PositionSide.FLAT
            
            # 计算需要调整的数量
            if current_position is None or current_position.quantity == 0:
                current_lots = 0
                current_side = PositionSide.FLAT
            else:
                current_lots = current_position.quantity
                current_side = current_position.side
            
            # 生成交易指令
            if target_side == PositionSide.FLAT:
                # 全部平仓
                if current_lots > 0:
                    orders.append({
                        'symbol': symbol,
                        'action': 'close',
                        'quantity': current_lots,
                        'side': current_side,
                    })
            elif target_side == current_side:
                # 同向调整
                diff = target_lots - current_lots
                if diff > 0:
                    orders.append({
                        'symbol': symbol,
                        'action': 'open',
                        'quantity': diff,
                        'side': target_side,
                    })
                elif diff < 0:
                    orders.append({
                        'symbol': symbol,
                        'action': 'close',
                        'quantity': abs(diff),
                        'side': current_side,
                    })
            else:
                # 反手：先平后开
                if current_lots > 0:
                    orders.append({
                        'symbol': symbol,
                        'action': 'close',
                        'quantity': current_lots,
                        'side': current_side,
                    })
                if target_lots > 0:
                    orders.append({
                        'symbol': symbol,
                        'action': 'open',
                        'quantity': target_lots,
                        'side': target_side,
                    })
        
        logger.info(f"Rebalance generated {len(orders)} orders")
        return orders
    
    def record_state(self, timestamp: datetime):
        """
        记录当前状态
        
        Args:
            timestamp: 时间戳
        """
        state = {
            'timestamp': timestamp,
            'cash': self.cash,
            'net_value': self.get_net_value(),
            'used_margin': self.used_margin,
            'available_margin': self.available_margin,
            'positions': {s: p.to_dict() for s, p in self.positions.items() if p.quantity > 0},
        }
        self.position_history.append(state)
    
    def get_history_df(self) -> pd.DataFrame:
        """
        获取历史记录DataFrame
        
        Returns:
            历史记录DataFrame
        """
        if not self.position_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.position_history)
    
    def reset(self):
        """重置组合状态"""
        self.cash = self.initial_capital
        self.available_margin = self.initial_capital
        self.used_margin = 0.0
        self.positions = {}
        self.position_history = []
        self.value_history = []
        self.total_trades = 0
        self.total_commission = 0.0
        
        logger.info("Portfolio reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        net_value = self.get_net_value()
        total_return = (net_value - self.initial_capital) / self.initial_capital
        
        exposure = self.get_exposure()
        margin_info = self.get_margin_info()
        
        return {
            'initial_capital': self.initial_capital,
            'final_net_value': net_value,
            'total_return': total_return,
            'total_trades': self.total_trades,
            'total_commission': self.total_commission,
            'current_cash': self.cash,
            'used_margin': self.used_margin,
            'available_margin': self.available_margin,
            'leverage': margin_info['leverage'],
            'position_count': len([p for p in self.positions.values() if p.quantity > 0]),
            'long_exposure': exposure['long_exposure'],
            'short_exposure': exposure['short_exposure'],
            'net_exposure': exposure['net_exposure'],
            'gross_exposure': exposure['gross_exposure'],
        }
