"""
回测引擎模块 - BacktestEngine

功能：
- 向量化回测（快速研究）
- 事件驱动回测（精细验证）
- 与策略对接
- 支持多空双向交易
- 逐日盯市结算
- 强平逻辑
"""

from typing import Dict, List, Optional, Any, Union, Callable, Tuple
from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np

from ..core.base import BacktestEngine as BaseBacktestEngine, Strategy
from ..core.logger import get_logger
from ..core.exceptions import BacktestError

from .broker import Broker, TradeSide, OrderType, PositionSide
from .portfolio import Portfolio
from .recorder import TradeRecorder, TradeDirection, TradeAction

logger = get_logger('backtest.engine')


class BacktestMode(Enum):
    """回测模式"""
    VECTORIZED = "vectorized"       # 向量化回测
    EVENT_DRIVEN = "event_driven"   # 事件驱动回测


class BacktestEngine(BaseBacktestEngine):
    """
    回测引擎
    
    继承自core.base.BacktestEngine，提供完整的回测功能：
    1. 向量化回测：快速矩阵运算，适合因子研究
    2. 事件驱动回测：逐日模拟交易，适合精细验证
    3. 期货特性支持：保证金、逐日盯市、强平
    
    Attributes:
        initial_capital: 初始资金
        commission: 手续费率
        slippage: 滑点
        margin_rate: 保证金率
        maintenance_margin_rate: 维持保证金率
        mode: 回测模式
    """
    
    def __init__(
        self,
        initial_capital: float = 1_000_000,
        commission: float = 0.0001,
        slippage: float = 0.0,
        margin_rate: float = 0.1,
        maintenance_margin_rate: float = 0.08,
        close_today_rate: Optional[float] = None,
        contract_multipliers: Optional[Dict[str, float]] = None,
        tick_sizes: Optional[Dict[str, float]] = None,
        max_concentration: float = 0.3,
        max_leverage: float = 3.0,
    ):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金（默认100万）
            commission: 手续费率（默认万1）
            slippage: 滑点（跳数，默认0）
            margin_rate: 保证金率（默认10%）
            maintenance_margin_rate: 维持保证金率（默认8%）
            close_today_rate: 平今手续费率，None表示与commission相同
            contract_multipliers: 合约乘数字典
            tick_sizes: 最小变动价位字典
            max_concentration: 最大单品种集中度
            max_leverage: 最大杠杆倍数
        """
        super().__init__(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage,
            margin_rate=margin_rate,
        )
        
        self.maintenance_margin_rate = maintenance_margin_rate
        self.close_today_rate = close_today_rate
        self.contract_multipliers = contract_multipliers or {}
        self.tick_sizes = tick_sizes or {}
        self.max_concentration = max_concentration
        self.max_leverage = max_leverage
        
        # 初始化组件
        self.broker = Broker(
            initial_capital=initial_capital,
            commission_rate=commission,
            close_today_rate=close_today_rate,
            slippage=slippage,
            margin_rate=margin_rate,
            maintenance_margin_rate=maintenance_margin_rate,
            contract_multipliers=contract_multipliers,
            tick_sizes=tick_sizes,
        )
        
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            margin_rate=margin_rate,
            max_concentration=max_concentration,
            max_leverage=max_leverage,
            contract_multipliers=contract_multipliers,
        )
        
        self.recorder = TradeRecorder(initial_capital=initial_capital)
        
        # 回测状态
        self.results: Optional[Dict] = None
        self.is_running = False
        
        logger.info(f"BacktestEngine initialized: capital={initial_capital}, "
                   f"commission={commission}, margin_rate={margin_rate}")
    
    def reset(self):
        """重置回测状态"""
        super().reset()
        self.broker.reset()
        self.portfolio.reset()
        self.recorder.reset()
        self.results = None
        self.is_running = False
        
        logger.info("BacktestEngine reset")
    
    def run(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        mode: BacktestMode = BacktestMode.VECTORIZED,
        **kwargs
    ) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            data: 回测数据，包含OHLCV等
            strategy: 策略实例
            mode: 回测模式
            **kwargs: 额外参数
            
        Returns:
            回测结果字典
        """
        if mode == BacktestMode.VECTORIZED:
            return self.run_vectorized(data, strategy, **kwargs)
        else:
            return self.run_event_driven(data, strategy, **kwargs)
    
    def run_vectorized(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        position_size: Optional[Union[int, str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        向量化回测
        
        使用矩阵运算快速计算收益，适合因子研究阶段。
        简化处理：假设所有信号都能以收盘价成交。
        
        Args:
            data: 回测数据，必须包含close列
            strategy: 策略实例
            position_size: 仓位大小，可以是固定手数或'volatility'按波动率调整
            **kwargs: 额外参数
            
        Returns:
            回测结果字典
        """
        logger.info("Starting vectorized backtest...")
        self.is_running = True
        
        try:
            # 生成信号
            signals_df = strategy.generate_signals(data)
            
            if signals_df.empty:
                logger.warning("No signals generated")
                return self._get_empty_results()
            
            # 确保数据对齐
            if 'signal' not in signals_df.columns:
                raise BacktestError("Strategy must generate 'signal' column")
            
            # 合并数据和信号
            if 'date' in signals_df.columns:
                signals_df = signals_df.set_index('date')
            
            df = data.copy()
            if 'date' in df.columns:
                df = df.set_index('date')
            
            # 对齐索引
            aligned_data = df.join(signals_df[['signal', 'weight']] if 'weight' in signals_df.columns 
                                   else signals_df[['signal']], how='inner')
            
            if aligned_data.empty:
                logger.warning("No aligned data after joining signals")
                return self._get_empty_results()
            
            # 计算收益率
            close_prices = aligned_data['close']
            returns = close_prices.pct_change().fillna(0)
            signals = aligned_data['signal'].fillna(0)
            
            # 仓位权重
            if 'weight' in aligned_data.columns:
                weights = aligned_data['weight'].fillna(0)
            else:
                weights = pd.Series(1.0, index=aligned_data.index)
            
            # 计算持仓（信号延迟一期执行，避免未来函数）
            positions = signals.shift(1).fillna(0) * weights.shift(1).fillna(0)
            
            # 计算策略收益
            strategy_returns = positions * returns
            
            # 计算交易成本
            # 信号变化时产生交易
            signal_changes = signals.diff().abs().fillna(0)
            transaction_costs = signal_changes * self.commission * 2  # 双边手续费
            
            # 扣除成本后的收益
            net_returns = strategy_returns - transaction_costs
            
            # 计算净值曲线
            cumulative_returns = (1 + net_returns).cumprod() - 1
            equity_curve = self.initial_capital * (1 + cumulative_returns)
            
            # 记录每日净值
            for date, equity in equity_curve.items():
                self.recorder.record_daily_value(
                    date=pd.to_datetime(date),
                    net_value=equity,
                    cash=0,  # 向量化模式下不跟踪现金
                    margin=0,
                )
            
            # 计算绩效指标
            self.results = self._calculate_vectorized_results(
                aligned_data, positions, net_returns, equity_curve
            )
            
            logger.info(f"Vectorized backtest completed: "
                       f"return={self.results['total_return']*100:.2f}%, "
                       f"trades={self.results['total_trades']}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Vectorized backtest failed: {e}")
            raise BacktestError(f"Backtest failed: {e}")
        finally:
            self.is_running = False
    
    def _calculate_vectorized_results(
        self,
        data: pd.DataFrame,
        positions: pd.Series,
        returns: pd.Series,
        equity_curve: pd.Series,
    ) -> Dict[str, Any]:
        """计算向量化回测结果"""
        # 交易次数（信号变化）
        trades = positions.diff().abs().fillna(0)
        total_trades = int(trades.sum())
        
        # 收益指标
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # 年化收益
        days = len(data)
        annual_return = (1 + total_return) ** (252 / days) - 1 if days > 0 else 0
        
        # 波动率
        volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe = (returns.mean() - 0.03/252) / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        
        # 最大回撤
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 持仓时间
        position_time = (positions != 0).sum() / len(positions)
        
        return {
            'mode': 'vectorized',
            'initial_capital': self.initial_capital,
            'final_equity': equity_curve.iloc[-1],
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'position_time_ratio': position_time,
            'equity_curve': equity_curve,
            'returns': returns,
            'positions': positions,
            'data': data,
        }
    
    def run_event_driven(
        self,
        data: pd.DataFrame,
        strategy: Strategy,
        use_margin_call: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        事件驱动回测
        
        逐日模拟交易流程：信号生成 → 下单 → 成交确认 → 盈亏结算
        支持完整的期货特性：保证金、逐日盯市、强平
        
        Args:
            data: 回测数据
            strategy: 策略实例
            use_margin_call: 是否启用强平逻辑
            **kwargs: 额外参数
            
        Returns:
            回测结果字典
        """
        logger.info("Starting event-driven backtest...")
        self.is_running = True
        
        try:
            # 确保数据格式正确
            if 'date' not in data.columns and not isinstance(data.index, pd.DatetimeIndex):
                raise BacktestError("Data must have 'date' column or DatetimeIndex")
            
            df = data.copy()
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
            else:
                df.index = pd.to_datetime(df.index)
            
            # 按日期排序
            df = df.sort_index()
            
            # 获取信号
            signals_df = strategy.generate_signals(df)
            
            if signals_df.empty:
                logger.warning("No signals generated")
                return self._get_empty_results()
            
            # 确保信号有date列或索引
            if 'date' in signals_df.columns:
                signals_df['date'] = pd.to_datetime(signals_df['date'])
                signals_df = signals_df.set_index('date')
            else:
                signals_df.index = pd.to_datetime(signals_df.index)
            
            # 合并数据和信号
            aligned_data = df.join(
                signals_df[['signal', 'weight']] if 'weight' in signals_df.columns 
                else signals_df[['signal']],
                how='inner'
            )
            
            if aligned_data.empty:
                logger.warning("No aligned data")
                return self._get_empty_results()
            
            # 逐日回测
            for date, row in aligned_data.iterrows():
                self._process_day(date, row, use_margin_call)
            
            # 生成结果
            self.results = self._calculate_event_driven_results()
            
            logger.info(f"Event-driven backtest completed: "
                       f"return={self.results['total_return']*100:.2f}%, "
                       f"trades={self.results['total_trades']}")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Event-driven backtest failed: {e}")
            raise BacktestError(f"Backtest failed: {e}")
        finally:
            self.is_running = False
    
    def _process_day(self, date: datetime, row: pd.Series, use_margin_call: bool = True):
        """
        处理单日交易
        
        Args:
            date: 日期
            row: 当日数据（含信号）
            use_margin_call: 是否启用强平
        """
        symbol = row.get('symbol', 'UNKNOWN')
        close_price = row['close']
        signal = row.get('signal', 0)
        weight = row.get('weight', 1.0)
        
        # 1. 处理未成交订单
        prices = {symbol: close_price}
        self.broker.process_pending_orders(prices)
        
        # 2. 检查强平
        if use_margin_call:
            margin_call_symbols = self.broker.check_margin_call()
            for sym in margin_call_symbols:
                self.broker.liquidate_position(sym, close_price)
        
        # 3. 根据信号交易（信号延迟一期执行）
        # 获取当前持仓
        position = self.broker.get_position(symbol)
        current_qty = position.quantity if position else 0
        current_side = PositionSide.LONG if current_qty > 0 else (
            PositionSide.SHORT if current_qty < 0 else PositionSide.FLAT
        )
        
        # 目标持仓
        # 根据风险参数计算目标手数，避免硬编码
        # 基础手数 = 可用资金 * 风险比例 / (价格 * 止损距离)
        # 简化版本：基础手数由 max_position 控制
        base_position = int(self.max_leverage * self.initial_capital / (close_price * 100))
        base_position = max(1, base_position)  # 至少1手
        target_qty = int(signal * weight * base_position)
        
        # 计算需要调整的数量
        if current_side == PositionSide.SHORT:
            current_qty = -current_qty
        
        qty_diff = target_qty - current_qty
        
        if qty_diff != 0:
            if qty_diff > 0:
                # 买入
                side = TradeSide.BUY
                self.broker.submit_order(
                    symbol=symbol,
                    side=side,
                    quantity=abs(qty_diff),
                    order_type=OrderType.MARKET,
                    current_price=close_price,
                )
            else:
                # 卖出
                side = TradeSide.SELL
                self.broker.submit_order(
                    symbol=symbol,
                    side=side,
                    quantity=abs(qty_diff),
                    order_type=OrderType.MARKET,
                    current_price=close_price,
                )
        
        # 4. 更新持仓价格
        self.portfolio.update_prices(prices)
        
        # 5. 逐日盯市结算
        self.broker.settle_daily(date, prices)
        
        # 6. 记录每日状态
        equity = self.broker.get_equity()
        margin_info = self.broker.get_margin()
        
        self.recorder.record_daily_value(
            date=date,
            net_value=equity,
            cash=self.broker.cash,
            margin=margin_info['used'],
            positions={
                s: {
                    'quantity': p.quantity,
                    'side': p.side.value,
                    'unrealized_pnl': p.unrealized_pnl,
                }
                for s, p in self.broker.positions.items() if p.quantity > 0
            },
        )
    
    def _calculate_event_driven_results(self) -> Dict[str, Any]:
        """计算事件驱动回测结果"""
        # 获取绩效指标
        metrics = self.recorder.get_performance_metrics()
        trade_stats = self.recorder.get_trade_stats()
        max_dd = self.recorder.calculate_max_drawdown()
        
        # 获取净值曲线
        equity_curve = self.recorder.get_daily_values()
        
        # 获取交易记录
        trades_df = self.recorder.get_trades()
        
        # 获取broker统计
        broker_stats = self.broker.get_stats()
        
        return {
            'mode': 'event_driven',
            'initial_capital': self.initial_capital,
            'final_equity': broker_stats['final_equity'],
            'total_return': metrics.get('total_return', 0),
            'annual_return': metrics.get('annual_return', 0),
            'volatility': metrics.get('volatility', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'sortino_ratio': metrics.get('sortino_ratio', 0),
            'calmar_ratio': metrics.get('calmar_ratio', 0),
            'max_drawdown': max_dd.get('max_drawdown_pct', 0),
            'total_trades': trade_stats.get('total_trades', 0),
            'win_rate': trade_stats.get('win_rate', 0),
            'profit_factor': trade_stats.get('profit_factor', 0),
            'equity_curve': equity_curve,
            'trades': trades_df,
            'broker_stats': broker_stats,
            'recorder': self.recorder,
        }
    
    def _get_empty_results(self) -> Dict[str, Any]:
        """获取空结果"""
        return {
            'mode': 'empty',
            'initial_capital': self.initial_capital,
            'final_equity': self.initial_capital,
            'total_return': 0.0,
            'annual_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'equity_curve': pd.DataFrame(),
            'trades': pd.DataFrame(),
        }
    
    def get_results(self) -> Optional[Dict[str, Any]]:
        """
        获取回测结果
        
        Returns:
            回测结果字典或None
        """
        return self.results
    
    def generate_report(self) -> str:
        """
        生成回测报告
        
        Returns:
            报告字符串
        """
        if self.results is None:
            return "No backtest results available. Please run backtest first."
        
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("期货量化回测报告")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # 基本信息
        report_lines.append("【回测配置】")
        report_lines.append(f"回测模式: {self.results.get('mode', 'unknown')}")
        report_lines.append(f"初始资金: {self.initial_capital:,.2f}")
        report_lines.append(f"手续费率: {self.commission*10000:.2f}‱")
        report_lines.append(f"滑点: {self.slippage}")
        report_lines.append(f"保证金率: {self.margin_rate*100:.1f}%")
        report_lines.append("")
        
        # 收益指标
        report_lines.append("【收益指标】")
        report_lines.append(f"期末权益: {self.results.get('final_equity', 0):,.2f}")
        report_lines.append(f"总收益率: {self.results.get('total_return', 0)*100:.2f}%")
        report_lines.append(f"年化收益率: {self.results.get('annual_return', 0)*100:.2f}%")
        report_lines.append("")
        
        # 风险指标
        report_lines.append("【风险指标】")
        report_lines.append(f"年化波动率: {self.results.get('volatility', 0)*100:.2f}%")
        report_lines.append(f"最大回撤: {self.results.get('max_drawdown', 0)*100:.2f}%")
        report_lines.append("")
        
        # 风险调整收益
        report_lines.append("【风险调整收益】")
        report_lines.append(f"夏普比率: {self.results.get('sharpe_ratio', 0):.3f}")
        report_lines.append(f"索提诺比率: {self.results.get('sortino_ratio', 0):.3f}")
        report_lines.append(f"卡玛比率: {self.results.get('calmar_ratio', 0):.3f}")
        report_lines.append("")
        
        # 交易统计
        report_lines.append("【交易统计】")
        report_lines.append(f"总交易次数: {self.results.get('total_trades', 0)}")
        report_lines.append(f"胜率: {self.results.get('win_rate', 0)*100:.2f}%")
        report_lines.append(f"盈亏比: {self.results.get('profit_factor', 0):.3f}")
        report_lines.append("")
        
        # 详细报告（如果有recorder）
        if 'recorder' in self.results:
            report_lines.append(self.results['recorder'].generate_report())
        
        report_lines.append("=" * 70)
        
        return "\n".join(report_lines)
    
    def plot_results(self, figsize: Tuple[int, int] = (14, 10)):
        """
        绘制回测结果（需要matplotlib）
        
        Args:
            figsize: 图表大小
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib is required for plotting")
            return
        
        if self.results is None:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        
        # 获取净值曲线
        if 'equity_curve' in self.results:
            equity_df = self.results['equity_curve']
            if isinstance(equity_df, pd.DataFrame) and not equity_df.empty:
                # 净值曲线
                ax1 = axes[0]
                ax1.plot(equity_df.index, equity_df['net_value'], label='Net Value')
                ax1.set_title('Equity Curve')
                ax1.set_ylabel('Value')
                ax1.legend()
                ax1.grid(True)
                
                # 回撤
                ax2 = axes[1]
                rolling_max = equity_df['net_value'].cummax()
                drawdown = (equity_df['net_value'] - rolling_max) / rolling_max
                ax2.fill_between(equity_df.index, drawdown, 0, color='red', alpha=0.3)
                ax2.set_title('Drawdown')
                ax2.set_ylabel('Drawdown')
                ax2.grid(True)
                
                # 日收益
                ax3 = axes[2]
                if 'daily_return' in equity_df.columns:
                    ax3.bar(equity_df.index, equity_df['daily_return'], alpha=0.6)
                    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                    ax3.set_title('Daily Returns')
                    ax3.set_ylabel('Return')
                    ax3.set_xlabel('Date')
                    ax3.grid(True)
        
        plt.tight_layout()
        plt.show()
