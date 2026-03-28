"""
回测模块 - 期货量化回测引擎

提供完整的回测功能，包括：
- BacktestEngine: 回测主引擎（向量化 + 事件驱动）
- Broker: 模拟交易所（订单撮合、保证金管理、强平）
- Portfolio: 仓位管理（多品种、风险监控）
- TradeRecorder: 交易记录与绩效分析
"""

from .engine import BacktestEngine
from .broker import Broker
from .portfolio import Portfolio
from .recorder import TradeRecorder

__all__ = [
    'BacktestEngine',
    'Broker',
    'Portfolio',
    'TradeRecorder',
]
