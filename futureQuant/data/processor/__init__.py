"""
data.processor 模块 - 数据预处理

包含：
- cleaner: 数据清洗
- contract_manager: 主力合约切换、连续化
- calendar: 期货交易日历
"""

from .cleaner import DataCleaner
from .contract_manager import ContractManager
from .calendar import FuturesCalendar

__all__ = ['DataCleaner', 'ContractManager', 'FuturesCalendar']
