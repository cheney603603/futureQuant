"""
data 模块 - 数据管理

包含：
- manager: 数据管理器（统一入口）
- fetcher: 数据获取（akshare + 爬虫）
- storage: 数据存储（SQLite + Parquet）
- processor: 数据预处理（清洗、对齐、主力合约切换）
"""

from .manager import DataManager

__all__ = ['DataManager']
