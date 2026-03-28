"""
模拟交易所模块 - Broker

功能：
- 订单管理（提交、撤单、查询）
- 成交撮合（支持滑点、部分成交）
- 手续费计算（开仓/平仓、平今/平昨差异）
- 保证金计算与监控
- 逐日盯市结算（MTM）
- 强平逻辑（保证金不足时）
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from ..core.logger import get_logger
from ..core.exceptions import BrokerError

logger = get_logger('backtest.broker')


class OrderType(Enum):
    """订单类型"""
    MARKET = "market"       # 市价单
    LIMIT = "limit"         # 限价单
    STOP = "stop"           # 止损单


class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"         # 待成交
    PARTIAL = "partial"         # 部分成交
    FILLED = "filled"           # 全部成交
    CANCELLED = "cancelled"     # 已撤销
    REJECTED = "rejected"       # 已拒绝


class TradeSide(Enum):
    """交易方向"""
    BUY = 1         # 买入（开仓/平仓）
    SELL = -1       # 卖出（开仓/平仓）


class PositionSide(Enum):
    """持仓方向"""
    LONG = 1        # 多头
    SHORT = -1      # 空头
    FLAT = 0        # 空仓


@dataclass
class Order:
    """订单数据类"""
    id: str
    symbol: str
    side: TradeSide
    quantity: int
    order_type: OrderType
    price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: float = 0.0
    commission: float = 0.0
    create_time: datetime = field(default_factory=datetime.now)
    update_time: Optional[datetime] = None
    
    @property
    def remaining_quantity(self) -> int:
        """剩余未成交数量"""
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        """是否全部成交"""
        return self.filled_quantity >= self.quantity
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'commission': self.commission,
            'create_time': self.create_time,
            'update_time': self.update_time,
        }


@dataclass
class Trade:
    """成交记录数据类"""
    id: str
    order_id: str
    symbol: str
    side: TradeSide
    quantity: int
    price: float
    commission: float
    trade_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'price': self.price,
            'commission': self.commission,
            'trade_time': self.trade_time,
        }


@dataclass
class Position:
    """持仓数据类"""
    symbol: str
    side: PositionSide
    quantity: int = 0
    avg_price: float = 0.0
    margin: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    def update_unrealized_pnl(self, current_price: float, multiplier: float = 1.0):
        """更新浮动盈亏"""
        if self.quantity == 0:
            self.unrealized_pnl = 0.0
            return
        
        price_diff = current_price - self.avg_price
        self.unrealized_pnl = price_diff * self.quantity * multiplier * self.side.value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'margin': self.margin,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
        }


class Broker:
    """
    模拟交易所
    
    功能：
    1. 订单管理：提交、撤销、查询
    2. 成交撮合：支持市价单、限价单、滑点模拟
    3. 手续费计算：支持不同费率（开仓/平仓、平今/平昨）
    4. 保证金计算：按合约价值比例计算
    5. 逐日盯市：每日结算盈亏
    6. 强平逻辑：保证金不足时强制平仓
    
    Attributes:
        initial_capital: 初始资金
        commission_rate: 手续费率（双边）
        close_today_rate: 平今手续费率（期货特有）
        slippage: 滑点（跳数或比例）
        margin_rate: 保证金率
        maintenance_margin_rate: 维持保证金率（低于此触发强平）
        contract_multipliers: 合约乘数字典 {symbol: multiplier}
        tick_sizes: 最小变动价位字典 {symbol: tick_size}
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission_rate: float = 0.0001,
        close_today_rate: Optional[float] = None,
        slippage: float = 0.0,
        margin_rate: float = 0.1,
        maintenance_margin_rate: float = 0.08,
        contract_multipliers: Optional[Dict[str, float]] = None,
        tick_sizes: Optional[Dict[str, float]] = None,
    ):
        """
        初始化模拟交易所
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率（默认万1）
            close_today_rate: 平今手续费率，None表示与commission_rate相同
            slippage: 滑点（跳数，0表示无滑点）
            margin_rate: 保证金率（默认10%）
            maintenance_margin_rate: 维持保证金率（默认8%）
            contract_multipliers: 合约乘数字典，默认10
            tick_sizes: 最小变动价位字典，默认1
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.close_today_rate = close_today_rate or commission_rate
        self.slippage = slippage
        self.margin_rate = margin_rate
        self.maintenance_margin_rate = maintenance_margin_rate
        self.contract_multipliers = contract_multipliers or {}
        self.tick_sizes = tick_sizes or {}
        
        # 账户状态
        self.cash = initial_capital
        self.available_cash = initial_capital
        self.margin_used = 0.0
        
        # 订单与成交记录
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.daily_positions: Dict[str, Dict[str, int]] = {}  # date -> {symbol: quantity}
        
        # 历史记录
        self.equity_curve: List[Dict] = []
        self.daily_settlements: List[Dict] = []
        
        # 计数器
        self._order_counter = 0
        self._trade_counter = 0
        
        logger.info(f"Broker initialized with capital={initial_capital}, "
                   f"commission={commission_rate}, margin_rate={margin_rate}")
    
    def _get_order_id(self) -> str:
        """生成订单ID"""
        self._order_counter += 1
        return f"ORD{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._order_counter:06d}"
    
    def _get_trade_id(self) -> str:
        """生成成交ID"""
        self._trade_counter += 1
        return f"TRD{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._trade_counter:06d}"
    
    def _get_contract_multiplier(self, symbol: str) -> float:
        """获取合约乘数"""
        return self.contract_multipliers.get(symbol, 10.0)
    
    def _get_tick_size(self, symbol: str) -> float:
        """获取最小变动价位"""
        return self.tick_sizes.get(symbol, 1.0)
    
    def _calculate_slippage_price(self, price: float, side: TradeSide, symbol: str) -> float:
        """
        计算含滑点的成交价格
        
        Args:
            price: 原始价格
            side: 交易方向
            symbol: 合约代码
            
        Returns:
            含滑点的价格
        """
        if self.slippage == 0:
            return price
        
        tick_size = self._get_tick_size(symbol)
        slippage_ticks = int(self.slippage)
        
        # 买入时价格向上滑点，卖出时价格向下滑点
        if side == TradeSide.BUY:
            return price + slippage_ticks * tick_size
        else:
            return price - slippage_ticks * tick_size
    
    def _calculate_commission(
        self, 
        price: float, 
        quantity: int, 
        symbol: str,
        is_close_today: bool = False
    ) -> float:
        """
        计算手续费
        
        Args:
            price: 成交价格
            quantity: 成交数量
            symbol: 合约代码
            is_close_today: 是否平今仓（期货特有）
            
        Returns:
            手续费金额
        """
        multiplier = self._get_contract_multiplier(symbol)
        trade_value = price * quantity * multiplier
        
        # 平今仓使用不同费率
        rate = self.close_today_rate if is_close_today else self.commission_rate
        
        return trade_value * rate
    
    def _calculate_margin(self, price: float, quantity: int, symbol: str) -> float:
        """
        计算保证金
        
        Args:
            price: 价格
            quantity: 数量
            symbol: 合约代码
            
        Returns:
            保证金金额
        """
        multiplier = self._get_contract_multiplier(symbol)
        contract_value = price * quantity * multiplier
        return contract_value * self.margin_rate
    
    def submit_order(
        self,
        symbol: str,
        side: TradeSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        current_price: Optional[float] = None,
    ) -> Optional[Order]:
        """
        提交订单
        
        Args:
            symbol: 合约代码
            side: 交易方向（BUY/SELL）
            quantity: 数量
            order_type: 订单类型
            price: 限价单价格
            current_price: 当前市场价格（用于市价单成交）
            
        Returns:
            Order对象，资金不足时返回None
        """
        if quantity <= 0:
            logger.warning(f"Invalid quantity: {quantity}")
            return None
        
        # 检查资金是否充足（开仓时）
        position = self.positions.get(symbol)
        is_opening = False
        
        if side == TradeSide.BUY:
            # 买入：如果没有空头持仓或买入量大于空头持仓，则为开仓
            if position is None or position.side != PositionSide.SHORT:
                is_opening = True
            elif quantity > position.quantity:
                is_opening = True
        else:  # SELL
            # 卖出：如果没有多头持仓或卖出量大于多头持仓，则为开仓
            if position is None or position.side != PositionSide.LONG:
                is_opening = True
            elif quantity > position.quantity:
                is_opening = True
        
        if is_opening and current_price:
            # 计算所需保证金
            required_margin = self._calculate_margin(current_price, quantity, symbol)
            if required_margin > self.available_cash:
                logger.warning(f"Insufficient margin: required={required_margin:.2f}, "
                              f"available={self.available_cash:.2f}")
                return None
        
        # 创建订单
        order = Order(
            id=self._get_order_id(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=order_type,
            price=price,
        )
        
        self.orders[order.id] = order
        
        # 市价单立即尝试成交
        if order_type == OrderType.MARKET and current_price:
            self._fill_order(order, current_price)
        
        logger.debug(f"Order submitted: {order.id}, {symbol}, {side.value}, {quantity}")
        return order
    
    def _fill_order(self, order: Order, market_price: float, quantity: Optional[int] = None):
        """
        成交订单
        
        Args:
            order: 订单对象
            market_price: 市场价格
            quantity: 成交数量，None表示全部成交
        """
        fill_qty = quantity or order.remaining_quantity
        if fill_qty <= 0:
            return
        
        # 计算成交价格（含滑点）
        fill_price = self._calculate_slippage_price(market_price, order.side, order.symbol)
        
        # 检查是否平今仓
        is_close_today = self._is_close_today(order.symbol, order.side, fill_qty)
        
        # 计算手续费
        commission = self._calculate_commission(
            fill_price, fill_qty, order.symbol, is_close_today
        )
        
        # 更新订单状态
        order.filled_quantity += fill_qty
        order.filled_price = (order.filled_price * (order.filled_quantity - fill_qty) + 
                             fill_price * fill_qty) / order.filled_quantity
        order.commission += commission
        order.update_time = datetime.now()
        
        if order.is_filled:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIAL
        
        # 创建成交记录
        trade = Trade(
            id=self._get_trade_id(),
            order_id=order.id,
            symbol=order.symbol,
            side=order.side,
            quantity=fill_qty,
            price=fill_price,
            commission=commission,
            trade_time=datetime.now(),
        )
        self.trades.append(trade)
        
        # 更新持仓
        self._update_position(order.symbol, order.side, fill_qty, fill_price, commission)
        
        logger.debug(f"Order filled: {order.id}, qty={fill_qty}, price={fill_price:.2f}")
    
    def _is_close_today(self, symbol: str, side: TradeSide, quantity: int) -> bool:
        """
        判断是否平今仓（简化逻辑，实际应按开仓时间判断）
        
        Args:
            symbol: 合约代码
            side: 交易方向
            quantity: 数量
            
        Returns:
            是否平今仓
        """
        position = self.positions.get(symbol)
        if position is None or position.quantity == 0:
            return False
        
        # 如果方向相反且是平仓操作，认为是平今（简化处理）
        if side == TradeSide.SELL and position.side == PositionSide.LONG:
            return True
        if side == TradeSide.BUY and position.side == PositionSide.SHORT:
            return True
        
        return False
    
    def _update_position(
        self, 
        symbol: str, 
        side: TradeSide, 
        quantity: int, 
        price: float,
        commission: float
    ):
        """
        更新持仓
        
        Args:
            symbol: 合约代码
            side: 交易方向
            quantity: 成交数量
            price: 成交价格
            commission: 手续费
        """
        multiplier = self._get_contract_multiplier(symbol)
        
        if symbol not in self.positions:
            # 新开仓
            position_side = PositionSide.LONG if side == TradeSide.BUY else PositionSide.SHORT
            self.positions[symbol] = Position(
                symbol=symbol,
                side=position_side,
                quantity=quantity,
                avg_price=price,
                margin=self._calculate_margin(price, quantity, symbol),
            )
        else:
            position = self.positions[symbol]
            
            # 判断是开仓还是平仓
            is_closing = (side == TradeSide.SELL and position.side == PositionSide.LONG) or \
                        (side == TradeSide.BUY and position.side == PositionSide.SHORT)
            
            if is_closing:
                # 平仓
                close_qty = min(quantity, position.quantity)
                
                # 计算实现盈亏
                price_diff = price - position.avg_price
                if position.side == PositionSide.SHORT:
                    price_diff = -price_diff
                realized_pnl = price_diff * close_qty * multiplier
                position.realized_pnl += realized_pnl
                
                # 更新持仓
                position.quantity -= close_qty
                
                if position.quantity == 0:
                    # 全部平仓
                    position.side = PositionSide.FLAT
                    position.avg_price = 0.0
                    position.margin = 0.0
                else:
                    # 部分平仓，按比例减少保证金
                    position.margin = self._calculate_margin(position.avg_price, position.quantity, symbol)
                
                # 返还保证金和盈亏
                released_margin = self._calculate_margin(price, close_qty, symbol)
                self.cash += released_margin + realized_pnl - commission
                
            else:
                # 加仓
                # 计算新的均价
                total_value = position.avg_price * position.quantity + price * quantity
                position.quantity += quantity
                position.avg_price = total_value / position.quantity
                position.margin = self._calculate_margin(position.avg_price, position.quantity, symbol)
                
                # 扣除保证金和手续费
                added_margin = self._calculate_margin(price, quantity, symbol)
                self.cash -= added_margin + commission
        
        # 更新已用保证金
        self._update_margin_used()
        self._update_available_cash()
    
    def _update_margin_used(self):
        """更新已用保证金"""
        self.margin_used = sum(pos.margin for pos in self.positions.values())
    
    def _update_available_cash(self):
        """更新可用资金"""
        self.available_cash = self.cash - self.margin_used
    
    def cancel_order(self, order_id: str) -> bool:
        """
        撤销订单
        
        Args:
            order_id: 订单ID
            
        Returns:
            是否成功撤销
        """
        order = self.orders.get(order_id)
        if order is None:
            return False
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            return False
        
        order.status = OrderStatus.CANCELLED
        order.update_time = datetime.now()
        
        logger.debug(f"Order cancelled: {order_id}")
        return True
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        获取持仓信息
        
        Args:
            symbol: 合约代码
            
        Returns:
            Position对象或None
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """
        获取所有持仓
        
        Returns:
            {symbol: Position}字典
        """
        return {k: v for k, v in self.positions.items() if v.quantity > 0}
    
    def get_margin(self) -> Dict[str, float]:
        """
        获取保证金信息
        
        Returns:
            {
                'used': 已用保证金,
                'available': 可用保证金,
                'total': 总资金,
                'ratio': 保证金使用率
            }
        """
        total = self.cash + self.margin_used
        ratio = self.margin_used / total if total > 0 else 0
        
        return {
            'used': self.margin_used,
            'available': self.available_cash,
            'total': total,
            'ratio': ratio,
        }
    
    def get_equity(self) -> float:
        """
        获取账户权益（资金 + 保证金 + 浮动盈亏）
        
        Returns:
            账户权益
        """
        unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.cash + self.margin_used + unrealized_pnl
    
    def check_margin_call(self) -> List[str]:
        """
        检查强平条件
        
        Returns:
            需要强平的合约列表
        """
        margin_info = self.get_margin()
        
        # 如果保证金率低于维持保证金率，触发强平
        if margin_info['ratio'] > (1 - self.maintenance_margin_rate / self.margin_rate):
            # 按浮动亏损从大到小排序，优先平亏损最大的
            positions_to_close = sorted(
                self.positions.values(),
                key=lambda p: p.unrealized_pnl
            )
            return [p.symbol for p in positions_to_close if p.quantity > 0]
        
        return []
    
    def liquidate_position(self, symbol: str, current_price: float) -> Optional[Trade]:
        """
        强制平仓
        
        Args:
            symbol: 合约代码
            current_price: 当前价格
            
        Returns:
            成交记录或None
        """
        position = self.positions.get(symbol)
        if position is None or position.quantity == 0:
            return None
        
        # 确定平仓方向
        side = TradeSide.SELL if position.side == PositionSide.LONG else TradeSide.BUY
        
        # 提交市价平仓单
        order = self.submit_order(
            symbol=symbol,
            side=side,
            quantity=position.quantity,
            order_type=OrderType.MARKET,
            current_price=current_price,
        )
        
        if order:
            logger.warning(f"Position liquidated: {symbol}, qty={position.quantity}, "
                          f"side={side.value}")
        
        return order
    
    def settle_daily(self, date: datetime, prices: Dict[str, float]):
        """
        逐日盯市结算
        
        Args:
            date: 结算日期
            prices: {symbol: price} 结算价格
        """
        settlement_pnl = 0.0
        
        for symbol, position in self.positions.items():
            if position.quantity == 0:
                continue
            
            settle_price = prices.get(symbol)
            if settle_price is None:
                continue
            
            multiplier = self._get_contract_multiplier(symbol)
            
            # 计算当日盈亏
            prev_unrealized = position.unrealized_pnl
            position.update_unrealized_pnl(settle_price, multiplier)
            daily_pnl = position.unrealized_pnl - prev_unrealized
            settlement_pnl += daily_pnl
            
            # 更新保证金（按结算价重新计算）
            position.margin = self._calculate_margin(settle_price, position.quantity, symbol)
        
        # 更新资金
        self._update_margin_used()
        self._update_available_cash()
        
        # 记录结算信息
        settlement_record = {
            'date': date,
            'cash': self.cash,
            'margin_used': self.margin_used,
            'equity': self.get_equity(),
            'settlement_pnl': settlement_pnl,
            'positions': {s: p.to_dict() for s, p in self.positions.items() if p.quantity > 0},
        }
        self.daily_settlements.append(settlement_record)
        
        logger.debug(f"Daily settlement: date={date.date()}, equity={settlement_record['equity']:.2f}, "
                    f"pnl={settlement_pnl:.2f}")
    
    def process_pending_orders(self, prices: Dict[str, float]):
        """
        处理未成交订单（限价单等）
        
        Args:
            prices: 当前价格字典 {symbol: price}
        """
        for order in self.orders.values():
            if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIAL]:
                continue
            
            current_price = prices.get(order.symbol)
            if current_price is None:
                continue
            
            if order.order_type == OrderType.LIMIT and order.price:
                # 限价单成交判断
                if order.side == TradeSide.BUY and current_price <= order.price:
                    self._fill_order(order, current_price)
                elif order.side == TradeSide.SELL and current_price >= order.price:
                    self._fill_order(order, current_price)
    
    def reset(self):
        """重置交易所状态"""
        self.cash = self.initial_capital
        self.available_cash = self.initial_capital
        self.margin_used = 0.0
        self.orders = {}
        self.trades = []
        self.positions = {}
        self.daily_positions = {}
        self.equity_curve = []
        self.daily_settlements = []
        self._order_counter = 0
        self._trade_counter = 0
        
        logger.info("Broker reset")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        total_trades = len(self.trades)
        total_commission = sum(t.commission for t in self.trades)
        
        return {
            'initial_capital': self.initial_capital,
            'final_equity': self.get_equity(),
            'total_return': (self.get_equity() - self.initial_capital) / self.initial_capital,
            'total_trades': total_trades,
            'total_commission': total_commission,
            'current_cash': self.cash,
            'margin_used': self.margin_used,
            'position_count': len([p for p in self.positions.values() if p.quantity > 0]),
        }
