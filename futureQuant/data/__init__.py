"""
data 模块 - 数据管理

包含：
- manager: 数据管理器（统一入口）
- fetcher: 数据获取（akshare + 爬虫 + 降级）
- storage: 数据存储（SQLite + Parquet）
- processor: 数据预处理（清洗、对齐、主力合约切换）
- validator: 数据验证（时间戳、新鲜度、质量检查）P1
- cache_manager: 缓存管理器（TTL过期机制）P2.1
- quality_reporter: 质量报告生成器 P2.3
"""

from .manager import DataManager
from .validator import DataValidator, ValidationError, validate_fetched_data

# P2.1: 缓存管理器
try:
    from .cache_manager import DataCacheManager, get_cache_manager, clear_cache, get_cache_stats
    _has_cache = True
except ImportError:
    _has_cache = False

# P2.3: 质量报告生成器
try:
    from .quality_reporter import DataQualityReporter, check_data_quality, QualityMetrics
    _has_quality = True
except ImportError:
    _has_quality = False

__all__ = [
    'DataManager',
    'DataValidator',
    'ValidationError',
    'validate_fetched_data',
]

if _has_cache:
    __all__.extend(['DataCacheManager', 'get_cache_manager', 'clear_cache', 'get_cache_stats'])

if _has_quality:
    __all__.extend(['DataQualityReporter', 'check_data_quality', 'QualityMetrics'])
