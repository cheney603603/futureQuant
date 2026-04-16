"""
data_collector - Agent 1: 数据收集 Agent

三技能架构：
1. DataCollectorAgent     - 数据收集主控制器（数据收集技能）
2. ReliablePathManager    - 可靠链路管理（可靠链路管理技能）
3. PathDiscovery          - 新路径探测（扩展数据收集能力）
4. DataQueryEngine        - 数据查询引擎（数据查询技能）

使用示例：

  # === 数据收集 ===
  >>> from futureQuant.agent.data_collector import DataCollectorAgent
  >>> agent = DataCollectorAgent()
  >>> result = agent.run({"symbols": ["RB", "HC", "I"], "force_update": False})
  >>> print(result.metrics)

  # === 可靠链路管理 ===
  >>> from futureQuant.agent.data_collector import ReliablePathManager
  >>> pm = ReliablePathManager()
  >>> paths = pm.get_reliable_paths(data_type='daily')
  >>> pm.confirm_path('akshare_rb_daily_abc123', success=True, response_ms=850)

  # === 数据查询（自然语言）===
  >>> from futureQuant.agent.data_collector import DataQuerySkill
  >>> skill = DataQuerySkill()
  >>> result = skill.query_nl("螺纹钢最近30天的日线数据")
  >>> print(result.data)
"""

from .loop_controller import DataCollectorAgent
from .reliable_path_manager import ReliablePathManager, ReliablePath, PathStatus
from .path_discovery import PathDiscovery, DiscoveryResult
from .data_query import (
    DataQueryEngine,
    DataQuerySkill,
    NLQueryParser,
    QuerySpec,
    QueryResult,
)
from .data_discovery import (
    DataSourceStatus,
    AkShareSource,
    TuShareSource,
    BaostockSource,
    ExchangeAPISource,
    DataSourceManager,
)

__all__ = [
    # 主控制器
    'DataCollectorAgent',

    # 可靠链路管理
    'ReliablePathManager',
    'ReliablePath',
    'PathStatus',

    # 新路径探测
    'PathDiscovery',
    'DiscoveryResult',

    # 数据查询
    'DataQueryEngine',
    'DataQuerySkill',
    'NLQueryParser',
    'QuerySpec',
    'QueryResult',

    # 数据源
    'DataSourceStatus',
    'AkShareSource',
    'TuShareSource',
    'BaostockSource',
    'ExchangeAPISource',
    'DataSourceManager',
]
