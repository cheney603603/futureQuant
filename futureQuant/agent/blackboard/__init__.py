"""
blackboard - 中央黑板系统

基于黑板模式（Blackboard Pattern）实现多 Agent 协作：
- Blackboard: 中央数据存储，所有 Agent 读写
- KnowledgeSource: 知识源接口，各 Agent 实现
- BlackboardController: 黑板控制器，协调 Agent 执行
- HumanInterventionPoint: 人类介入点

使用示例：
    >>> from futureQuant.agent.blackboard import Blackboard, BlackboardController
    >>> 
    >>> # 创建黑板
    >>> bb = Blackboard()
    >>> 
    >>> # 写入数据
    >>> bb.write("price_data", price_df, agent="data_collector")
    >>> bb.write("factors", factor_list, agent="factor_mining")
    >>> 
    >>> # 读取数据
    >>> price = bb.read("price_data")
    >>> 
    >>> # 创建控制器
    >>> controller = BlackboardController(blackboard=bb)
    >>> controller.register_agent(data_collector_agent)
    >>> controller.register_agent(factor_mining_agent)
    >>> 
    >>> # 执行
    >>> result = await controller.execute()
"""

from .blackboard import Blackboard, BlackboardEntry, BlackboardState
from .knowledge_source import KnowledgeSource, KnowledgeSourceResult
from .blackboard_controller import BlackboardController
from .human_intervention import (
    HumanInterventionPoint,
    InterventionRequest,
    InterventionResponse,
    InterventionType,
    HumanApprovalHandler,
)

__all__ = [
    # 核心黑板
    'Blackboard',
    'BlackboardEntry',
    'BlackboardState',
    
    # 知识源
    'KnowledgeSource',
    'KnowledgeSourceResult',
    
    # 控制器
    'BlackboardController',
    
    # 人类介入
    'HumanInterventionPoint',
    'InterventionRequest',
    'InterventionResponse',
    'InterventionType',
    'HumanApprovalHandler',
]
