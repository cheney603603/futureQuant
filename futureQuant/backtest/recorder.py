"""
交易记录模块 - TradeRecorder

功能：
- 记录每笔交易详情
- 生成交易报告
- 计算绩效指标（收益率、夏普比率、最大回撤等）
- 分析交易统计
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

from ..core.logger import get_logger
from ..core.exceptions import BacktestError

logger = get_logger('backtest.recorder')


class TradeDirection(Enum):
    """交易方向"""
    LONG = 1
    SHORT = -1


class TradeAction(Enum):
    """交易动作"""
    OPEN = "open"
    CLOSE = "close"


@dataclass
class TradeRecord:
    """交易记录数据类"""
    id: str
    symbol: str
    direction: TradeDirection
    action: TradeAction
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    commission: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    
    @property
    def is_closed(self) -> bool:
        """是否已平仓"""
        return self.exit_price is not None
    
    @property
    def holding_period(self) -> Optional[int]:
        """持仓周期（天数）"""
        if self.exit_time is None:
            return None
        return (self.exit_time - self.entry_time).days
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'direction': self.direction.value,
            'action': self.action.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time,
            'exit_time': self.exit_time,
            'commission': self.commission,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'is_closed': self.is_closed,
            'holding_period': self.holding_period,
        }


class TradeRecorder:
    """
    交易记录器
    
    功能：
    1. 记录每笔交易详情
    2. 生成交易报告
    3. 计算绩效指标
    4. 分析交易统计
    
    Attributes:
        initial_capital: 初始资金
        trades: 交易记录列表
        daily_values: 每日净值记录
    """
    
    def __init__(self, initial_capital: float = 1_000_000):
        """
        初始化交易记录器
        
        Args:
            initial_capital: 初始资金
        """
        self.initial_capital = initial_capital
        
        # 交易记录
        self.trades: List[TradeRecord] = []
        self.open_trades: Dict[str, List[TradeRecord]] = {}  # symbol -> [open trades]
        
        # 每日记录
        self.daily_values: List[Dict] = []
        self.daily_returns: List[float] = []
        
        # 统计
        self._trade_counter = 0
        
        logger.info(f"TradeRecorder initialized with capital={initial_capital}")
    
    def _get_trade_id(self) -> str:
        """生成交易ID"""
        self._trade_counter += 1
        return f"TRD{datetime.now().strftime('%Y%m%d%H%M%S')}_{self._trade_counter:06d}"
    
    def record_trade(
        self,
        symbol: str,
        direction: TradeDirection,
        action: TradeAction,
        quantity: int,
        price: float,
        commission: float = 0.0,
        timestamp: Optional[datetime] = None,
    ) -> TradeRecord:
        """
        记录交易
        
        Args:
            symbol: 合约代码
            direction: 交易方向
            action: 交易动作（开仓/平仓）
            quantity: 数量
            price: 价格
            commission: 手续费
            timestamp: 时间戳
            
        Returns:
            TradeRecord对象
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if action == TradeAction.OPEN:
            # 开仓记录
            trade = TradeRecord(
                id=self._get_trade_id(),
                symbol=symbol,
                direction=direction,
                action=action,
                quantity=quantity,
                entry_price=price,
                entry_time=timestamp,
                commission=commission,
            )
            
            if symbol not in self.open_trades:
                self.open_trades[symbol] = []
            self.open_trades[symbol].append(trade)
            self.trades.append(trade)
            
            logger.debug(f"Trade opened: {trade.id}, {symbol}, {direction.value}, qty={quantity}")
            
        else:  # CLOSE
            # 平仓记录 - 找到对应的开仓记录
            trade = self._close_trade(
                symbol=symbol,
                direction=direction,
                quantity=quantity,
                price=price,
                commission=commission,
                timestamp=timestamp,
            )
        
        return trade
    
    def _close_trade(
        self,
        symbol: str,
        direction: TradeDirection,
        quantity: int,
        price: float,
        commission: float,
        timestamp: datetime,
    ) -> Optional[TradeRecord]:
        """
        平仓并更新对应的开仓记录
        
        Args:
            symbol: 合约代码
            direction: 交易方向
            quantity: 平仓数量
            price: 平仓价格
            commission: 手续费
            timestamp: 时间戳
            
        Returns:
            更新的TradeRecord或None
        """
        open_trades = self.open_trades.get(symbol, [])
        if not open_trades:
            logger.warning(f"No open trades found for {symbol}")
            return None
        
        # 找到对应的开仓记录（先进先出）
        remaining = quantity
        total_pnl = 0.0
        
        for trade in open_trades[:]:
            if remaining <= 0:
                break
            
            if trade.is_closed:
                continue
            
            # 计算本次平仓数量
            close_qty = min(remaining, trade.quantity)
            
            # 计算盈亏
            if direction == TradeDirection.LONG:
                # 平空仓
                pnl = (trade.entry_price - price) * close_qty
            else:
                # 平多仓
                pnl = (price - trade.entry_price) * close_qty
            
            total_pnl += pnl
            remaining -= close_qty
            
            # 更新开仓记录
            trade.exit_price = price
            trade.exit_time = timestamp
            trade.pnl = total_pnl
            trade.pnl_pct = total_pnl / (trade.entry_price * quantity) if quantity > 0 else 0
            trade.commission += commission
        
        # 清理已平仓的记录
        self.open_trades[symbol] = [t for t in open_trades if not t.is_closed]
        
        logger.debug(f"Trade closed: {symbol}, qty={quantity}, pnl={total_pnl:.2f}")
        
        return trade
    
    def record_daily_value(
        self,
        date: datetime,
        net_value: float,
        cash: float,
        margin: float,
        positions: Optional[Dict] = None,
    ):
        """
        记录每日净值
        
        Args:
            date: 日期
            net_value: 净值
            cash: 现金
            margin: 保证金
            positions: 持仓信息
        """
        record = {
            'date': date,
            'net_value': net_value,
            'cash': cash,
            'margin': margin,
            'positions': positions or {},
        }
        
        # 计算日收益率
        if self.daily_values:
            prev_value = self.daily_values[-1]['net_value']
            daily_return = (net_value - prev_value) / prev_value if prev_value > 0 else 0
            record['daily_return'] = daily_return
            self.daily_returns.append(daily_return)
        else:
            record['daily_return'] = 0.0
        
        self.daily_values.append(record)
    
    def get_trades(self, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        获取交易记录
        
        Args:
            symbol: 筛选特定品种，None返回全部
            
        Returns:
            交易记录DataFrame
        """
        if not self.trades:
            return pd.DataFrame()
        
        data = [t.to_dict() for t in self.trades]
        df = pd.DataFrame(data)
        
        if symbol is not None and not df.empty:
            df = df[df['symbol'] == symbol]
        
        return df
    
    def get_closed_trades(self) -> pd.DataFrame:
        """
        获取已平仓交易
        
        Returns:
            已平仓交易DataFrame
        """
        closed = [t for t in self.trades if t.is_closed]
        if not closed:
            return pd.DataFrame()
        
        data = [t.to_dict() for t in closed]
        return pd.DataFrame(data)
    
    def get_daily_values(self) -> pd.DataFrame:
        """
        获取每日净值记录
        
        Returns:
            每日净值DataFrame
        """
        if not self.daily_values:
            return pd.DataFrame()
        
        return pd.DataFrame(self.daily_values)
    
    def get_trade_stats(self) -> Dict[str, Any]:
        """
        获取交易统计
        
        Returns:
            交易统计字典
        """
        closed_trades = [t for t in self.trades if t.is_closed]
        
        if not closed_trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
            }
        
        # 盈亏统计
        pnls = [t.pnl for t in closed_trades]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        
        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
            'avg_pnl': np.mean(pnls),
            'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
            'max_win': max(pnls) if pnls else 0,
            'max_loss': min(pnls) if pnls else 0,
            'total_wins': total_wins,
            'total_losses': total_losses,
            'profit_factor': total_wins / total_losses if total_losses > 0 else float('inf'),
            'avg_holding_period': np.mean([t.holding_period for t in closed_trades if t.holding_period is not None]),
        }
    
    def calculate_returns(self) -> pd.Series:
        """
        计算收益率序列
        
        Returns:
            日收益率Series
        """
        if not self.daily_values:
            return pd.Series()
        
        df = pd.DataFrame(self.daily_values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        return df['daily_return']
    
    def calculate_cumulative_returns(self) -> pd.Series:
        """
        计算累计收益率
        
        Returns:
            累计收益率Series
        """
        returns = self.calculate_returns()
        if returns.empty:
            return pd.Series()
        
        return (1 + returns).cumprod() - 1
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.03, periods_per_year: int = 252) -> float:
        """
        计算夏普比率
        
        Args:
            risk_free_rate: 无风险利率（年化）
            periods_per_year: 每年交易天数
            
        Returns:
            夏普比率
        """
        returns = self.calculate_returns()
        if returns.empty or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / periods_per_year
        return excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
    
    def calculate_max_drawdown(self) -> Dict[str, float]:
        """
        计算最大回撤
        
        Returns:
            {
                'max_drawdown': 最大回撤,
                'max_drawdown_pct': 最大回撤百分比,
                'peak_date': 峰值日期,
                'trough_date': 谷值日期,
            }
        """
        if not self.daily_values:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_pct': 0.0,
                'peak_date': None,
                'trough_date': None,
            }
        
        df = pd.DataFrame(self.daily_values)
        df['date'] = pd.to_datetime(df['date'])
        
        # 计算累计净值
        df['cumulative'] = df['net_value'].cummax()
        df['drawdown'] = df['net_value'] - df['cumulative']
        df['drawdown_pct'] = df['drawdown'] / df['cumulative']
        
        # 找到最大回撤
        max_dd_idx = df['drawdown'].idxmin()
        max_dd = df.loc[max_dd_idx, 'drawdown']
        max_dd_pct = df.loc[max_dd_idx, 'drawdown_pct']
        
        # 找到峰值和谷值日期
        peak_date = df.loc[:max_dd_idx, 'net_value'].idxmax()
        trough_date = max_dd_idx
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'peak_date': df.loc[peak_date, 'date'] if peak_date in df.index else None,
            'trough_date': df.loc[trough_date, 'date'] if trough_date in df.index else None,
        }
    
    def calculate_volatility(self, periods_per_year: int = 252) -> float:
        """
        计算波动率
        
        Args:
            periods_per_year: 每年交易天数
            
        Returns:
            年化波动率
        """
        returns = self.calculate_returns()
        if returns.empty:
            return 0.0
        
        return returns.std() * np.sqrt(periods_per_year)
    
    def calculate_calmar_ratio(self, periods_per_year: int = 252) -> float:
        """
        计算卡玛比率
        
        Args:
            periods_per_year: 每年交易天数
            
        Returns:
            卡玛比率
        """
        returns = self.calculate_returns()
        if returns.empty:
            return 0.0
        
        annual_return = returns.mean() * periods_per_year
        max_dd = self.calculate_max_drawdown()
        max_dd_pct = max_dd['max_drawdown_pct']
        
        if max_dd_pct == 0:
            return float('inf') if annual_return > 0 else 0.0
        
        return annual_return / abs(max_dd_pct)
    
    def calculate_sortino_ratio(self, risk_free_rate: float = 0.03, periods_per_year: int = 252) -> float:
        """
        计算索提诺比率
        
        Args:
            risk_free_rate: 无风险利率
            periods_per_year: 每年交易天数
            
        Returns:
            索提诺比率
        """
        returns = self.calculate_returns()
        if returns.empty:
            return 0.0
        
        # 下行标准差（只考虑负收益）
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        
        if downside_std == 0:
            return float('inf') if returns.mean() > 0 else 0.0
        
        excess_return = returns.mean() - risk_free_rate / periods_per_year
        return excess_return / downside_std * np.sqrt(periods_per_year)
    
    def get_performance_metrics(self, risk_free_rate: float = 0.03) -> Dict[str, float]:
        """
        获取完整绩效指标
        
        Args:
            risk_free_rate: 无风险利率
            
        Returns:
            绩效指标字典
        """
        if not self.daily_values:
            return {}
        
        returns = self.calculate_returns()
        
        # 基本指标
        initial_value = self.initial_capital
        final_value = self.daily_values[-1]['net_value']
        total_return = (final_value - initial_value) / initial_value
        
        # 年化收益率
        days = len(self.daily_values)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 风险指标
        volatility = self.calculate_volatility()
        max_dd = self.calculate_max_drawdown()
        
        # 风险调整收益
        sharpe = self.calculate_sharpe_ratio(risk_free_rate)
        sortino = self.calculate_sortino_ratio(risk_free_rate)
        calmar = self.calculate_calmar_ratio()
        
        # 交易统计
        trade_stats = self.get_trade_stats()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_dd['max_drawdown_pct'],
            'win_rate': trade_stats.get('win_rate', 0),
            'profit_factor': trade_stats.get('profit_factor', 0),
            'total_trades': trade_stats.get('total_trades', 0),
            'avg_pnl': trade_stats.get('avg_pnl', 0),
        }
    
    def generate_report(self) -> str:
        """
        生成文字报告
        
        Returns:
            报告字符串
        """
        metrics = self.get_performance_metrics()
        trade_stats = self.get_trade_stats()
        max_dd = self.calculate_max_drawdown()
        
        report = []
        report.append("=" * 60)
        report.append("回测绩效报告")
        report.append("=" * 60)
        report.append("")
        
        # 收益指标
        report.append("【收益指标】")
        report.append(f"初始资金: {self.initial_capital:,.2f}")
        if self.daily_values:
            report.append(f"期末净值: {self.daily_values[-1]['net_value']:,.2f}")
        report.append(f"总收益率: {metrics.get('total_return', 0)*100:.2f}%")
        report.append(f"年化收益率: {metrics.get('annual_return', 0)*100:.2f}%")
        report.append("")
        
        # 风险指标
        report.append("【风险指标】")
        report.append(f"年化波动率: {metrics.get('volatility', 0)*100:.2f}%")
        report.append(f"最大回撤: {max_dd.get('max_drawdown_pct', 0)*100:.2f}%")
        report.append("")
        
        # 风险调整收益
        report.append("【风险调整收益】")
        report.append(f"夏普比率: {metrics.get('sharpe_ratio', 0):.3f}")
        report.append(f"索提诺比率: {metrics.get('sortino_ratio', 0):.3f}")
        report.append(f"卡玛比率: {metrics.get('calmar_ratio', 0):.3f}")
        report.append("")
        
        # 交易统计
        report.append("【交易统计】")
        report.append(f"总交易次数: {trade_stats.get('total_trades', 0)}")
        report.append(f"盈利次数: {trade_stats.get('winning_trades', 0)}")
        report.append(f"亏损次数: {trade_stats.get('losing_trades', 0)}")
        report.append(f"胜率: {trade_stats.get('win_rate', 0)*100:.2f}%")
        report.append(f"盈亏比: {trade_stats.get('profit_factor', 0):.3f}")
        report.append(f"平均盈亏: {trade_stats.get('avg_pnl', 0):.2f}")
        report.append(f"平均盈利: {trade_stats.get('avg_win', 0):.2f}")
        report.append(f"平均亏损: {trade_stats.get('avg_loss', 0):.2f}")
        report.append(f"最大盈利: {trade_stats.get('max_win', 0):.2f}")
        report.append(f"最大亏损: {trade_stats.get('max_loss', 0):.2f}")
        report.append(f"平均持仓周期: {trade_stats.get('avg_holding_period', 0):.1f}天")
        report.append("")
        
        # 月度收益（如果有足够数据）
        if len(self.daily_values) >= 20:
            report.append("【月度收益】")
            monthly_returns = self._calculate_monthly_returns()
            for month, ret in list(monthly_returns.items())[-12:]:  # 最近12个月
                report.append(f"{month}: {ret*100:.2f}%")
        
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def _calculate_monthly_returns(self) -> Dict[str, float]:
        """计算月度收益率"""
        if not self.daily_values:
            return {}
        
        df = pd.DataFrame(self.daily_values)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        # 按月分组计算收益
        monthly = df['net_value'].resample('M').last()
        monthly_returns = monthly.pct_change().dropna()
        
        return {idx.strftime('%Y-%m'): ret for idx, ret in monthly_returns.items()}
    
    def reset(self):
        """重置记录器"""
        self.trades = []
        self.open_trades = {}
        self.daily_values = []
        self.daily_returns = []
        self._trade_counter = 0
        
        logger.info("TradeRecorder reset")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'initial_capital': self.initial_capital,
            'final_value': self.daily_values[-1]['net_value'] if self.daily_values else self.initial_capital,
            'total_trades': len(self.trades),
            'performance': self.get_performance_metrics(),
            'trade_stats': self.get_trade_stats(),
        }
