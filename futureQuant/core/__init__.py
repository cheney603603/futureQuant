"""
core 模块 - 期货量化框架核心基础设施

包含：
- base: 抽象基类定义
- config: 全局配置管理
- exceptions: 自定义异常
- logger: 日志系统
"""

from .base import (
    DataFetcher,
    Factor,
    Strategy,
    Model,
    BacktestEngine,
)
from .config import Config, get_config
from .exceptions import (
    FutureQuantError,
    DataError,
    FetchError,
    FactorError,
    StrategyError,
    BacktestError,
)
from .logger import get_logger, setup_logging

__all__ = [
    # Base classes
    'DataFetcher',
    'Factor', 
    'Strategy',
    'Model',
    'BacktestEngine',
    # Config
    'Config',
    'get_config',
    # Exceptions
    'FutureQuantError',
    'DataError',
    'FetchError',
    'FactorError',
    'StrategyError',
    'BacktestError',
    # Logger
    'get_logger',
    'setup_logging',
]
