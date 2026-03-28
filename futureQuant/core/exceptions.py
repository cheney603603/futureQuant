"""
自定义异常类 - 统一错误处理
"""


class FutureQuantError(Exception):
    """框架基础异常"""
    pass


class ConfigError(FutureQuantError):
    """配置错误"""
    pass


class DataError(FutureQuantError):
    """数据相关错误"""
    pass


class FetchError(DataError):
    """数据获取错误"""
    pass


class StorageError(DataError):
    """数据存储错误"""
    pass


class ProcessingError(DataError):
    """数据处理错误"""
    pass


class FactorError(FutureQuantError):
    """因子计算错误"""
    pass


class StrategyError(FutureQuantError):
    """策略错误"""
    pass


class ModelError(FutureQuantError):
    """模型错误"""
    pass


class BacktestError(FutureQuantError):
    """回测错误"""
    pass


class BrokerError(BacktestError):
    """模拟交易所错误"""
    pass


class AnalysisError(FutureQuantError):
    """分析错误"""
    pass
