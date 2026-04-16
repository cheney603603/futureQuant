"""
futureQuant - 期货量化研究框架

功能模块：
- core: 核心基础设施（基类、配置、日志）
- data: 数据管理（获取、存储、预处理）
- factor: 因子库（技术、基本面、宏观因子）
- strategy: 策略框架
- model: 模型训练
- backtest: 回测引擎
- analysis: 绩效分析

使用示例：
    >>> from futureQuant import DataManager
    >>> dm = DataManager()
    >>> df = dm.get_daily_data('RB2501', start_date='2024-01-01')
"""

__version__ = "0.6.0"
__author__ = "futureQuant Team"

# 主要类直接暴露
from .core import (
    Config,
    get_config,
    get_logger,
    setup_logging,
)
from .data import DataManager

__all__ = [
    '__version__',
    'Config',
    'get_config',
    'get_logger',
    'setup_logging',
    'DataManager',
]
