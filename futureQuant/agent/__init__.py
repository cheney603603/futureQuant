"""
Agent 模块 - 多 Agent 因子挖掘系统

本模块实现了基于多 Agent 协作的自动化因子挖掘框架：
- BaseAgent: Agent 抽象基类，定义统一接口
- MiningContext: 挖掘上下文，传递数据和中间结果
- MultiAgentFactorMiner: 主入口类，协调各 Agent 工作

挖掘 Agent:
- TechnicalMiningAgent: 技术因子挖掘
- FundamentalMiningAgent: 基本面因子挖掘
- MacroMiningAgent: 宏观因子挖掘
- FusionAgent: 因子融合与去相关

使用示例:
    >>> from futureQuant.agent import MultiAgentFactorMiner
    >>> miner = MultiAgentFactorMiner(
    ...     symbols=['RB', 'HC'],
    ...     start_date='2020-01-01',
    ...     end_date='2023-12-31'
    ... )
    >>> result = miner.run()
    >>> print(result.selected_factors)
"""

from .base import AgentStatus, AgentResult, BaseAgent
from .context import MiningContext
from .orchestrator import MultiAgentFactorMiner, MiningResult

# 挖掘 Agent
from .miners import (
    TechnicalMiningAgent,
    FundamentalMiningAgent,
    MacroMiningAgent,
    FusionAgent,
)

__all__ = [
    # 基础类
    'AgentStatus',
    'AgentResult',
    'BaseAgent',
    'MiningContext',
    # 主入口
    'MultiAgentFactorMiner',
    'MiningResult',
    # 挖掘 Agent
    'TechnicalMiningAgent',
    'FundamentalMiningAgent',
    'MacroMiningAgent',
    'FusionAgent',
]
