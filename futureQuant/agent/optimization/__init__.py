"""
ä¼åæ¨¡å

æä¾ futureQuant ç³»ç»çæ§è½ä¼åè½åã

åå«ä»¥ä¸å­æ¨¡åï¼
- parallel_calculator: å¹¶è¡è®¡ç®å¼æ
- cache_manager: ç¼å­ç®¡çå¨
- storage_optimizer: å­å¨ä¼åå¨
- query_optimizer: æ¥è¯¢ä¼åå¨
- memory_manager: åå­ç®¡çå¨
- data_preloader: æ°æ®é¢å è½½å¨
- performance_monitor: æ§è½çæ§å¨
"""

from .cache_manager import (
    CacheManager,
    CacheStats,
    CachedFunction,
    DiskCache,
    LRUCache,
    cached,
)
from .data_preloader import (
    BackgroundPreloader,
    DataPreloader,
    PredictivePreloader,
    PreloadStats,
)
from .memory_manager import (
    MemoryManager,
    MemoryMonitor,
    MemoryStats,
)
from .parallel_calculator import (
    BatchCalculator,
    ExecutionMode,
    ParallelCalculator,
    ProgressTracker,
    TaskResult,
    create_calculator,
)
from .performance_monitor import (
    PerformanceAlert,
    PerformanceBenchmark,
    PerformanceMetric,
    PerformanceMonitor,
    PerformanceReporter,
)
from .query_optimizer import (
    BulkQueryExecutor,
    QueryOptimizer,
    QueryStats,
)
from .storage_optimizer import (
    CompressionConfig,
    StorageOptimizer,
)

__all__ = [
    # parallel_calculator
    "ParallelCalculator",
    "BatchCalculator",
    "ExecutionMode",
    "TaskResult",
    "ProgressTracker",
    "create_calculator",
    # cache_manager
    "CacheManager",
    "LRUCache",
    "DiskCache",
    "CacheStats",
    "CachedFunction",
    "cached",
    # storage_optimizer
    "StorageOptimizer",
    "CompressionConfig",
    # query_optimizer
    "QueryOptimizer",
    "BulkQueryExecutor",
    "QueryStats",
    # memory_manager
    "MemoryManager",
    "MemoryMonitor",
    "MemoryStats",
    # data_preloader
    "DataPreloader",
    "BackgroundPreloader",
    "PredictivePreloader",
    "PreloadStats",
    # performance_monitor
    "PerformanceMonitor",
    "PerformanceReporter",
    "PerformanceBenchmark",
    "PerformanceAlert",
    "PerformanceMetric",
]

__version__ = "1.0.0"
__author__ = "futureQuant Team"
__description__ = "Performance optimization module for futureQuant multi-agent factor mining system"
