"""
Agent 基础设施模块

定义 Agent 系统的核心抽象：
- AgentStatus: Agent 运行状态枚举
- AgentResult: Agent 执行结果数据类
- BaseAgent: 所有 Agent 的抽象基类，提供状态管理、历史记录等通用功能
"""

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.base import Factor
from ..core.logger import get_logger

logger = get_logger('agent.base')


class AgentStatus(Enum):
    """Agent 运行状态枚举"""

    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class AgentResult:
    """
    Agent 执行结果数据类

    Attributes:
        agent_name: Agent 名称
        status: 执行状态
        data: 因子值 DataFrame（行为时间索引，列为因子名）
        factors: 因子实例列表
        metrics: 评估指标字典（如 IC、ICIR 等）
        errors: 错误信息列表
        logs: 日志信息列表
        elapsed_seconds: 执行耗时（秒）
    """

    agent_name: str
    status: AgentStatus
    data: Optional[pd.DataFrame] = None
    factors: Optional[List[Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    errors: Optional[List[str]] = None
    logs: Optional[List[str]] = None
    elapsed_seconds: float = 0.0

    def __post_init__(self):
        """初始化默认值"""
        if self.factors is None:
            self.factors = []
        if self.metrics is None:
            self.metrics = {}
        if self.errors is None:
            self.errors = []
        if self.logs is None:
            self.logs = []

    @property
    def is_success(self) -> bool:
        """是否执行成功"""
        return self.status == AgentStatus.SUCCESS

    @property
    def n_factors(self) -> int:
        """通过筛选的因子数量"""
        return len(self.factors) if self.factors else 0

    def __repr__(self) -> str:
        return (
            f"AgentResult(agent={self.agent_name!r}, "
            f"status={self.status.value!r}, "
            f"n_factors={self.n_factors}, "
            f"elapsed={self.elapsed_seconds:.2f}s)"
        )


class BaseAgent(ABC):
    """
    Agent 抽象基类

    提供统一的 Agent 接口和通用功能：
    - 状态管理（IDLE -> RUNNING -> SUCCESS/FAILED）
    - 执行历史记录
    - 异常捕获与日志记录
    - 配置管理

    子类需要实现 execute() 方法，包含具体的业务逻辑。
    通过 run() 方法调用，run() 负责状态管理和异常处理。

    使用示例:
        class MyAgent(BaseAgent):
            def execute(self, context):
                # 具体逻辑
                return AgentResult(agent_name=self.name, status=AgentStatus.SUCCESS)

        agent = MyAgent(name='my_agent', config={'threshold': 0.05})
        result = agent.run(context)
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        初始化 Agent

        Args:
            name: Agent 名称，用于日志和结果标识
            config: 配置字典，包含 Agent 运行参数
        """
        self.name = name
        self.config = config or {}
        self._status = AgentStatus.IDLE
        self._history: List[AgentResult] = []
        self._logger = get_logger(f'agent.{name}')

    @property
    def status(self) -> AgentStatus:
        """当前 Agent 状态"""
        return self._status

    @abstractmethod
    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行 Agent 核心逻辑（子类必须实现）

        Args:
            context: 执行上下文，包含数据、配置和中间结果

        Returns:
            AgentResult: 执行结果
        """
        ...

    def run(self, context: Dict[str, Any]) -> AgentResult:
        """
        运行 Agent（带状态管理和异常处理）

        执行流程：
        1. 将状态设置为 RUNNING
        2. 调用 execute() 执行核心逻辑
        3. 成功则设置为 SUCCESS，失败则设置为 FAILED
        4. 将结果记录到历史

        Args:
            context: 执行上下文

        Returns:
            AgentResult: 执行结果（即使失败也会返回结果对象）
        """
        self._status = AgentStatus.RUNNING
        self._logger.info(f"Agent [{self.name}] started")
        start_time = time.time()

        try:
            result = self.execute(context)
            elapsed = time.time() - start_time
            result.elapsed_seconds = elapsed

            # 确保状态一致
            if result.status not in (AgentStatus.SUCCESS, AgentStatus.FAILED):
                result.status = AgentStatus.SUCCESS

            self._status = result.status
            self._history.append(result)

            self._logger.info(
                f"Agent [{self.name}] finished: status={result.status.value}, "
                f"n_factors={result.n_factors}, elapsed={elapsed:.2f}s"
            )
            return result

        except Exception as exc:
            elapsed = time.time() - start_time
            error_msg = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()

            self._logger.error(
                f"Agent [{self.name}] failed after {elapsed:.2f}s: {error_msg}\n{tb}"
            )

            self._status = AgentStatus.FAILED
            result = AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=[error_msg],
                logs=[tb],
                elapsed_seconds=elapsed,
            )
            self._history.append(result)
            return result

    def get_history(self) -> List[AgentResult]:
        """
        获取执行历史记录

        Returns:
            历史 AgentResult 列表（按时间顺序）
        """
        return list(self._history)

    def get_last_result(self) -> Optional[AgentResult]:
        """
        获取最近一次执行结果

        Returns:
            最近的 AgentResult，若无历史则返回 None
        """
        return self._history[-1] if self._history else None

    def reset(self):
        """
        重置 Agent 状态

        清除历史记录，将状态恢复为 IDLE。
        """
        self._status = AgentStatus.IDLE
        self._history.clear()
        self._logger.debug(f"Agent [{self.name}] reset to IDLE")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name={self.name!r}, "
            f"status={self._status.value!r}, "
            f"history_len={len(self._history)})"
        )
