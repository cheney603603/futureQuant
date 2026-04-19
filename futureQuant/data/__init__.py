"""
data 模块 - 数据管理

包含：
- manager: 数据管理器（统一入口）
- fetcher: 数据获取（akshare + 爬虫）
- storage: 数据存储（SQLite + Parquet）
- processor: 数据预处理（清洗、对齐、主力合约切换）
- validator: 数据验证（时间戳、新鲜度、质量检查）
"""

from .manager import DataManager
from .validator import DataValidator, ValidationError, validate_fetched_data

__all__ = [
    'DataManager',
    'DataValidator',
    'ValidationError',
    'validate_fetched_data',
]
