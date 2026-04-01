"""
data_collector - Agent 1: 数据收集 Agent

功能：
- 多数据源扫描与健康检查（AkShare / TuShare / Baostock / 交易所 API）
- 增量数据拉取与缓存
- 数据验证与清洗
- 自修复引擎（API 变更检测与自动适配）
- 增量更新调度器

使用示例：
    >>> from futureQuant.agent.data_collector import DataCollectorAgent
    >>> agent = DataCollectorAgent()
    >>> result = agent.run({"symbols": ["RB", "HC", "I"], "force_update": False})
    >>> print(result.metrics)
"""

from .loop_controller import DataCollectorAgent
from .data_discovery import (
    DataSourceStatus,
    AkShareSource,
    TuShareSource,
    BaostockSource,
    ExchangeAPISource,
)

__all__ = [
    'DataCollectorAgent',
    'DataSourceStatus',
    'AkShareSource',
    'TuShareSource',
    'BaostockSource',
    'ExchangeAPISource',
]
