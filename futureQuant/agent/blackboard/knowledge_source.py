"""
KnowledgeSource - 知识源接口

知识源（Knowledge Source）是黑板模式中的核心概念：
- 每个知识源可以"观察"黑板，判断自己是否能贡献
- 如果能贡献，则执行并写入黑板
- 知识源之间不直接通信，通过黑板间接协作

在本系统中，每个 Agent 就是一个 KnowledgeSource。
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .blackboard import Blackboard

from ...core.logger import get_logger

logger = get_logger('agent.blackboard.knowledge_source')


class KnowledgeSourceState(Enum):
    """知识源状态"""
    IDLE = "idle"           # 空闲
    CHECKING = "checking"   # 检查是否能贡献
    READY = "ready"         # 准备执行
    EXECUTING = "executing" # 执行中
    SUCCESS = "success"     # 成功
    FAILED = "failed"       # 失败
    SKIPPED = "skipped"     # 跳过（无法贡献）


@dataclass
class KnowledgeSourceResult:
    """
    知识源执行结果
    
    Attributes:
        source_name: 知识源名称
        state: 执行状态
        contributed: 是否贡献了数据到黑板
        data_written: 写入的数据键列表
        metrics: 执行指标
        errors: 错误列表
        elapsed_seconds: 执行耗时
        requires_intervention: 是否需要人类介入
        intervention_request: 介入请求（如有）
    """
    source_name: str
    state: KnowledgeSourceState
    contributed: bool = False
    data_written: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    requires_intervention: bool = False
    intervention_request: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'source_name': self.source_name,
            'state': self.state.value,
            'contributed': self.contributed,
            'data_written': self.data_written,
            'metrics': self.metrics,
            'errors': self.errors,
            'elapsed_seconds': self.elapsed_seconds,
            'requires_intervention': self.requires_intervention,
        }


class KnowledgeSource(ABC):
    """
    知识源抽象基类
    
    知识源是黑板模式中的核心组件：
    1. observe(): 观察黑板，判断是否能贡献
    2. execute(): 执行并写入黑板
    
    子类需要实现：
    - can_contribute(): 判断是否能贡献
    - contribute(): 执行贡献逻辑
    
    使用示例：
        >>> class MyAgent(KnowledgeSource):
        ...     def can_contribute(self, blackboard):
        ...         return blackboard.exists("price_data")
        ...     
        ...     def contribute(self, blackboard):
        ...         price = blackboard.read("price_data")
        ...         factors = self.compute_factors(price)
        ...         blackboard.write("factors", factors, agent=self.name)
        ...         return ["factors"]
    """
    
    def __init__(
        self,
        name: str,
        priority: int = 0,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化知识源
        
        Args:
            name: 知识源名称
            priority: 优先级（数值越大越先执行）
            config: 配置字典
        """
        self.name = name
        self.priority = priority
        self.config = config or {}
        self._state = KnowledgeSourceState.IDLE
        self._logger = get_logger(f'knowledge_source.{name}')
    
    @property
    def state(self) -> KnowledgeSourceState:
        """当前状态"""
        return self._state
    
    def observe(self, blackboard: 'Blackboard') -> bool:
        """
        观察黑板，判断是否能贡献
        
        Args:
            blackboard: 黑板实例
        
        Returns:
            True 如果能贡献，False 否则
        """
        self._state = KnowledgeSourceState.CHECKING
        try:
            can = self.can_contribute(blackboard)
            self._state = KnowledgeSourceState.READY if can else KnowledgeSourceState.SKIPPED
            self._logger.debug(f"[{self.name}] Observe: can_contribute={can}")
            return can
        except Exception as e:
            self._state = KnowledgeSourceState.FAILED
            self._logger.error(f"[{self.name}] Observe failed: {e}")
            return False
    
    def execute(self, blackboard: 'Blackboard') -> KnowledgeSourceResult:
        """
        执行贡献
        
        Args:
            blackboard: 黑板实例
        
        Returns:
            KnowledgeSourceResult
        """
        start_time = time.time()
        self._state = KnowledgeSourceState.EXECUTING
        
        result = KnowledgeSourceResult(
            source_name=self.name,
            state=KnowledgeSourceState.EXECUTING,
        )
        
        try:
            # 设置黑板状态
            blackboard.set_agent_status(self.name, 'running')
            
            # 执行贡献
            data_written = self.contribute(blackboard)
            
            # 成功
            elapsed = time.time() - start_time
            result.state = KnowledgeSourceState.SUCCESS
            result.contributed = True
            result.data_written = data_written or []
            result.elapsed_seconds = elapsed
            
            self._state = KnowledgeSourceState.SUCCESS
            blackboard.set_agent_status(self.name, 'success')
            
            self._logger.info(
                f"[{self.name}] Execute success: "
                f"wrote={result.data_written}, elapsed={elapsed:.2f}s"
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            result.state = KnowledgeSourceState.FAILED
            result.errors.append(str(e))
            result.elapsed_seconds = elapsed
            
            self._state = KnowledgeSourceState.FAILED
            blackboard.set_agent_status(self.name, 'failed')
            
            self._logger.error(f"[{self.name}] Execute failed: {e}")
        
        return result
    
    @abstractmethod
    def can_contribute(self, blackboard: 'Blackboard') -> bool:
        """
        判断是否能贡献
        
        子类必须实现。
        
        Args:
            blackboard: 黑板实例
        
        Returns:
            True 如果能贡献
        """
        ...
    
    @abstractmethod
    def contribute(self, blackboard: 'Blackboard') -> List[str]:
        """
        执行贡献逻辑
        
        子类必须实现。应该将结果写入黑板，并返回写入的数据键列表。
        
        Args:
            blackboard: 黑板实例
        
        Returns:
            写入的数据键列表
        """
        ...
    
    def __repr__(self) -> str:
        return (
            f"KnowledgeSource(name={self.name!r}, "
            f"priority={self.priority}, "
            f"state={self._state.value})"
        )


class KnowledgeSourceAdapter(KnowledgeSource):
    """
    知识源适配器
    
    将现有的 Agent 适配为 KnowledgeSource 接口。
    使得所有现有 Agent 可以无缝接入黑板系统。
    
    使用示例：
        >>> from futureQuant.agent.factor_mining import FactorMiningAgent
        >>> 
        >>> agent = FactorMiningAgent()
        >>> ks = KnowledgeSourceAdapter.wrap(agent)
        >>> 
        >>> # 现在可以注册到黑板控制器
        >>> controller.register(ks)
    """
    
    @classmethod
    def wrap(
        cls,
        agent: Any,
        name: Optional[str] = None,
        priority: int = 0,
        input_keys: Optional[List[str]] = None,
        output_key: Optional[str] = None,
    ) -> 'KnowledgeSourceAdapter':
        """
        包装 Agent 为 KnowledgeSource
        
        Args:
            agent: Agent 实例
            name: 知识源名称（默认使用 agent.name）
            priority: 优先级
            input_keys: 需要的输入数据键（用于判断 can_contribute）
            output_key: 输出数据键（写入黑板时使用）
        
        Returns:
            KnowledgeSourceAdapter 实例
        """
        return cls(
            agent=agent,
            name=name or getattr(agent, 'name', agent.__class__.__name__),
            priority=priority,
            input_keys=input_keys or [],
            output_key=output_key,
        )
    
    def __init__(
        self,
        agent: Any,
        name: str,
        priority: int = 0,
        input_keys: Optional[List[str]] = None,
        output_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化适配器
        
        Args:
            agent: 被包装的 Agent
            name: 知识源名称
            priority: 优先级
            input_keys: 需要的输入数据键
            output_key: 输出数据键
            config: 配置
        """
        super().__init__(name=name, priority=priority, config=config)
        self.agent = agent
        self.input_keys = input_keys or []
        self.output_key = output_key or name
    
    def can_contribute(self, blackboard: 'Blackboard') -> bool:
        """
        判断是否能贡献
        
        默认实现：检查所有 input_keys 是否存在于黑板。
        子类可以覆盖此方法实现更复杂的判断逻辑。
        """
        # 检查输入数据是否就绪
        for key in self.input_keys:
            if not blackboard.exists(key):
                self._logger.debug(f"[{self.name}] Missing input: {key}")
                return False
        
        # 检查是否已经执行过（幂等性）
        if blackboard.exists(self.output_key):
            entry = blackboard.read_entry(self.output_key)
            if entry and entry.agent == self.name:
                self._logger.debug(f"[{self.name}] Already executed")
                return False
        
        return True
    
    def contribute(self, blackboard: 'Blackboard') -> List[str]:
        """
        执行贡献
        
        从黑板读取输入，调用 Agent，将结果写入黑板。
        """
        # 构建上下文
        context = {}
        for key in self.input_keys:
            context[key] = blackboard.read(key)
        
        # 调用 Agent
        if hasattr(self.agent, 'run'):
            result = self.agent.run(context)
        elif hasattr(self.agent, 'execute'):
            result = self.agent.execute(context)
        else:
            raise AttributeError(f"Agent {self.name} has no run() or execute() method")
        
        # 写入黑板
        if hasattr(result, 'data') and result.data is not None:
            blackboard.write(
                key=self.output_key,
                value=result.data,
                agent=self.name,
                metadata={'metrics': getattr(result, 'metrics', {})},
            )
        elif isinstance(result, dict):
            blackboard.write(
                key=self.output_key,
                value=result,
                agent=self.name,
            )
        
        # 保存结果
        blackboard.set_agent_result(self.name, result)
        
        return [self.output_key]
    
    def __repr__(self) -> str:
        return (
            f"KnowledgeSourceAdapter(name={self.name!r}, "
            f"agent={self.agent.__class__.__name__}, "
            f"inputs={self.input_keys}, "
            f"output={self.output_key})"
        )
