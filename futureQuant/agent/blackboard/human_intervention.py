"""
Human Intervention - 人类介入机制

提供 Agent 与人类协作的标准接口：
- InterventionRequest: 介入请求
- InterventionResponse: 介入响应
- HumanInterventionPoint: 介入点基类
- HumanApprovalHandler: 人类审批处理器

设计原则：
1. 非阻塞：请求发出后 Agent 可以继续其他工作
2. 超时机制：避免无限等待
3. 多种介入类型：确认、选择、输入、审批等
4. 可配置：默认启用，可全局禁用

使用示例：
    >>> from futureQuant.agent.blackboard import HumanInterventionPoint, InterventionType
    >>> 
    >>> # 创建介入点
    >>> hip = HumanInterventionPoint()
    >>> 
    >>> # 请求确认
    >>> request = hip.request_confirmation(
    ...     agent_name="decision",
    ...     question="是否执行此交易策略？",
    ...     context={"strategy": "趋势跟踪", "risk": "中等"},
    ... )
    >>> 
    >>> # 等待响应
    >>> response = hip.wait_for_response(request.request_id, timeout=60)
    >>> if response.approved:
    ...     # 执行策略
    ...     pass
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union

from ...core.logger import get_logger

logger = get_logger('agent.blackboard.human_intervention')


class InterventionType(Enum):
    """介入类型枚举"""
    CONFIRMATION = "confirmation"       # 确认（是/否）
    CHOICE = "choice"                   # 选择（从多个选项中选择）
    INPUT = "input"                     # 输入（自由文本输入）
    APPROVAL = "approval"               # 审批（带理由的确认）
    PARAMETER = "parameter"             # 参数调整
    ESCALATION = "escalation"           # 升级（无法处理，需要人类接管）
    REVIEW = "review"                   # 审查（查看并确认结果）


@dataclass
class InterventionRequest:
    """
    介入请求
    
    当 Agent 遇到需要人类决策的情况时，创建介入请求。
    
    Attributes:
        request_id: 唯一请求 ID
        intervention_type: 介入类型
        agent_name: 发起请求的 Agent 名称
        question: 问题/提示文本
        context: 上下文数据（供人类参考）
        options: 选项列表（用于 CHOICE 类型）
        default_response: 默认响应（超时时使用）
        timeout_seconds: 超时时间（秒）
        created_at: 创建时间戳
        priority: 优先级（数值越大越紧急）
        metadata: 元数据
    """
    request_id: str
    intervention_type: InterventionType
    agent_name: str
    question: str
    context: Dict[str, Any] = field(default_factory=dict)
    options: List[str] = field(default_factory=list)
    default_response: Any = None
    timeout_seconds: float = 300.0  # 默认 5 分钟超时
    created_at: float = field(default_factory=time.time)
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """是否已超时"""
        return time.time() > self.created_at + self.timeout_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'intervention_type': self.intervention_type.value,
            'agent_name': self.agent_name,
            'question': self.question,
            'context': self.context,
            'options': self.options,
            'default_response': self.default_response,
            'timeout_seconds': self.timeout_seconds,
            'created_at': self.created_at,
            'priority': self.priority,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InterventionRequest':
        return cls(
            request_id=data['request_id'],
            intervention_type=InterventionType(data['intervention_type']),
            agent_name=data['agent_name'],
            question=data['question'],
            context=data.get('context', {}),
            options=data.get('options', []),
            default_response=data.get('default_response'),
            timeout_seconds=data.get('timeout_seconds', 300.0),
            created_at=data.get('created_at', time.time()),
            priority=data.get('priority', 0),
            metadata=data.get('metadata', {}),
        )


@dataclass
class InterventionResponse:
    """
    介入响应
    
    人类对介入请求的响应。
    
    Attributes:
        request_id: 对应的请求 ID
        response: 响应内容（类型取决于介入类型）
        approved: 是否批准（用于 APPROVAL 类型）
        reason: 理由/说明
        responded_at: 响应时间戳
        responder: 响应者标识（可选）
        metadata: 元数据
    """
    request_id: str
    response: Any
    approved: bool = True
    reason: str = ""
    responded_at: float = field(default_factory=time.time)
    responder: str = "human"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'request_id': self.request_id,
            'response': self.response,
            'approved': self.approved,
            'reason': self.reason,
            'responded_at': self.responded_at,
            'responder': self.responder,
            'metadata': self.metadata,
        }


class HumanInterventionPoint:
    """
    人类介入点
    
    管理 Agent 与人类之间的交互：
    - 创建介入请求
    - 等待人类响应
    - 处理超时
    - 回调通知
    
    使用示例：
        >>> hip = HumanInterventionPoint(enabled=True)
        >>> 
        >>> # 请求确认
        >>> request = hip.request_confirmation(
        ...     agent_name="trader",
        ...     question="是否执行买入操作？",
        ...     context={"symbol": "RB", "price": 3800},
        ... )
        >>> 
        >>> # 非阻塞检查
        >>> if hip.has_response(request.request_id):
        ...     response = hip.get_response(request.request_id)
        ... 
        >>> # 或阻塞等待
        >>> response = hip.wait_for_response(request.request_id, timeout=60)
    """
    
    def __init__(
        self,
        enabled: bool = True,
        default_timeout: float = 300.0,
        on_request: Optional[Callable[[InterventionRequest], None]] = None,
        on_response: Optional[Callable[[InterventionResponse], None]] = None,
    ):
        """
        初始化介入点
        
        Args:
            enabled: 是否启用人类介入（False 则自动批准所有请求）
            default_timeout: 默认超时时间（秒）
            on_request: 请求创建回调
            on_response: 响应接收回调
        """
        self.enabled = enabled
        self.default_timeout = default_timeout
        self.on_request = on_request
        self.on_response = on_response
        
        # 请求和响应存储
        self._requests: Dict[str, InterventionRequest] = {}
        self._responses: Dict[str, InterventionResponse] = {}
        
        logger.info(f"HumanInterventionPoint initialized: enabled={enabled}")
    
    # ---- 创建请求 ----
    
    def request_confirmation(
        self,
        agent_name: str,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        default: bool = False,
        timeout: Optional[float] = None,
    ) -> InterventionRequest:
        """
        请求确认（是/否）
        
        Args:
            agent_name: Agent 名称
            question: 问题
            context: 上下文
            default: 默认值（超时时使用）
            timeout: 超时时间
        
        Returns:
            InterventionRequest
        """
        request = InterventionRequest(
            request_id=self._generate_id(),
            intervention_type=InterventionType.CONFIRMATION,
            agent_name=agent_name,
            question=question,
            context=context or {},
            options=["是", "否"],
            default_response=default,
            timeout_seconds=timeout or self.default_timeout,
        )
        return self._register_request(request)
    
    def request_choice(
        self,
        agent_name: str,
        question: str,
        options: List[str],
        context: Optional[Dict[str, Any]] = None,
        default: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> InterventionRequest:
        """
        请求选择（从多个选项中选择）
        
        Args:
            agent_name: Agent 名称
            question: 问题
            options: 选项列表
            context: 上下文
            default: 默认选项
            timeout: 超时时间
        
        Returns:
            InterventionRequest
        """
        request = InterventionRequest(
            request_id=self._generate_id(),
            intervention_type=InterventionType.CHOICE,
            agent_name=agent_name,
            question=question,
            context=context or {},
            options=options,
            default_response=default or (options[0] if options else None),
            timeout_seconds=timeout or self.default_timeout,
        )
        return self._register_request(request)
    
    def request_input(
        self,
        agent_name: str,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        default: str = "",
        timeout: Optional[float] = None,
    ) -> InterventionRequest:
        """
        请求输入（自由文本）
        
        Args:
            agent_name: Agent 名称
            question: 问题
            context: 上下文
            default: 默认值
            timeout: 超时时间
        
        Returns:
            InterventionRequest
        """
        request = InterventionRequest(
            request_id=self._generate_id(),
            intervention_type=InterventionType.INPUT,
            agent_name=agent_name,
            question=question,
            context=context or {},
            default_response=default,
            timeout_seconds=timeout or self.default_timeout,
        )
        return self._register_request(request)
    
    def request_approval(
        self,
        agent_name: str,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        default: bool = False,
        timeout: Optional[float] = None,
    ) -> InterventionRequest:
        """
        请求审批（带理由的确认）
        
        Args:
            agent_name: Agent 名称
            question: 问题
            context: 上下文
            default: 默认值
            timeout: 超时时间
        
        Returns:
            InterventionRequest
        """
        request = InterventionRequest(
            request_id=self._generate_id(),
            intervention_type=InterventionType.APPROVAL,
            agent_name=agent_name,
            question=question,
            context=context or {},
            options=["批准", "拒绝"],
            default_response=default,
            timeout_seconds=timeout or self.default_timeout,
        )
        return self._register_request(request)
    
    def request_escalation(
        self,
        agent_name: str,
        question: str,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> InterventionRequest:
        """
        请求升级（无法处理，需要人类接管）
        
        Args:
            agent_name: Agent 名称
            question: 问题描述
            context: 上下文
            timeout: 超时时间
        
        Returns:
            InterventionRequest
        """
        request = InterventionRequest(
            request_id=self._generate_id(),
            intervention_type=InterventionType.ESCALATION,
            agent_name=agent_name,
            question=f"⚠️ 升级请求: {question}",
            context=context or {},
            priority=10,  # 最高优先级
            timeout_seconds=timeout or self.default_timeout,
        )
        return self._register_request(request)
    
    # ---- 响应处理 ----
    
    def respond(
        self,
        request_id: str,
        response: Any,
        approved: bool = True,
        reason: str = "",
        responder: str = "human",
    ) -> InterventionResponse:
        """
        提交响应
        
        Args:
            request_id: 请求 ID
            response: 响应内容
            approved: 是否批准
            reason: 理由
            responder: 响应者
        
        Returns:
            InterventionResponse
        """
        resp = InterventionResponse(
            request_id=request_id,
            response=response,
            approved=approved,
            reason=reason,
            responder=responder,
        )
        
        self._responses[request_id] = resp
        
        # 触发回调
        if self.on_response:
            try:
                self.on_response(resp)
            except Exception as e:
                logger.warning(f"on_response callback failed: {e}")
        
        logger.info(
            f"[HumanIntervention] Response: id={request_id}, "
            f"approved={approved}, responder={responder}"
        )
        
        return resp
    
    def has_response(self, request_id: str) -> bool:
        """检查是否有响应"""
        return request_id in self._responses
    
    def get_response(
        self,
        request_id: str,
    ) -> Optional[InterventionResponse]:
        """获取响应（非阻塞）"""
        return self._responses.get(request_id)
    
    def wait_for_response(
        self,
        request_id: str,
        timeout: Optional[float] = None,
        poll_interval: float = 0.5,
    ) -> InterventionResponse:
        """
        等待响应（阻塞）
        
        Args:
            request_id: 请求 ID
            timeout: 超时时间（None 则使用请求的超时时间）
            poll_interval: 轮询间隔（秒）
        
        Returns:
            InterventionResponse（超时则返回默认响应）
        """
        request = self._requests.get(request_id)
        if not request:
            raise ValueError(f"Request {request_id} not found")
        
        # 如果禁用人类介入，自动批准
        if not self.enabled:
            return self._auto_approve(request)
        
        timeout = timeout or request.timeout_seconds
        start = time.time()
        
        while time.time() - start < timeout:
            if request_id in self._responses:
                return self._responses[request_id]
            time.sleep(poll_interval)
        
        # 超时，返回默认响应
        logger.warning(
            f"[HumanIntervention] Timeout: id={request_id}, "
            f"using default response"
        )
        return InterventionResponse(
            request_id=request_id,
            response=request.default_response,
            approved=bool(request.default_response),
            reason="超时，使用默认响应",
            responder="timeout",
        )
    
    # ---- 工具方法 ----
    
    def get_pending_requests(self) -> List[InterventionRequest]:
        """获取所有待处理的请求"""
        pending = []
        for req in self._requests.values():
            if req.request_id not in self._responses and not req.is_expired():
                pending.append(req)
        return sorted(pending, key=lambda r: -r.priority)
    
    def get_all_requests(self) -> List[InterventionRequest]:
        """获取所有请求"""
        return list(self._requests.values())
    
    def clear(self):
        """清空所有请求和响应"""
        self._requests.clear()
        self._responses.clear()
    
    # ---- 内部方法 ----
    
    def _generate_id(self) -> str:
        """生成唯一 ID"""
        return f"int_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    
    def _register_request(self, request: InterventionRequest) -> InterventionRequest:
        """注册请求"""
        self._requests[request.request_id] = request
        
        # 触发回调
        if self.on_request:
            try:
                self.on_request(request)
            except Exception as e:
                logger.warning(f"on_request callback failed: {e}")
        
        logger.info(
            f"[HumanIntervention] Request: id={request.request_id}, "
            f"type={request.intervention_type.value}, agent={request.agent_name}"
        )
        
        return request
    
    def _auto_approve(self, request: InterventionRequest) -> InterventionResponse:
        """自动批准（当人类介入禁用时）"""
        return InterventionResponse(
            request_id=request.request_id,
            response=request.default_response,
            approved=True,
            reason="人类介入已禁用，自动批准",
            responder="auto",
        )


class HumanApprovalHandler:
    """
    人类审批处理器
    
    用于需要人类审批的决策场景：
    - 交易执行前审批
    - 策略变更审批
    - 风险超限审批
    
    使用示例：
        >>> handler = HumanApprovalHandler(
        ...     approval_rules={
        ...         'max_position': 100,  # 超过 100 手需要审批
        ...         'max_loss': 0.05,     # 超过 5% 亏损需要审批
        ...     }
        ... )
        >>> 
        >>> # 检查是否需要审批
        >>> if handler.needs_approval(action):
        ...     request = handler.request_approval(action)
        ...     response = handler.wait(request.request_id)
    """
    
    def __init__(
        self,
        intervention_point: Optional[HumanInterventionPoint] = None,
        approval_rules: Optional[Dict[str, Any]] = None,
        auto_approve_below_threshold: bool = True,
    ):
        """
        初始化审批处理器
        
        Args:
            intervention_point: 介入点实例
            approval_rules: 审批规则
            auto_approve_below_threshold: 低于阈值时是否自动批准
        """
        self.intervention_point = intervention_point or HumanInterventionPoint()
        self.approval_rules = approval_rules or {}
        self.auto_approve_below_threshold = auto_approve_below_threshold
    
    def needs_approval(self, action: Dict[str, Any]) -> bool:
        """
        判断是否需要审批
        
        Args:
            action: 动作字典，包含 position_size, expected_loss 等
        
        Returns:
            True 如果需要审批
        """
        # 检查仓位
        max_position = self.approval_rules.get('max_position', float('inf'))
        if action.get('position_size', 0) > max_position:
            return True
        
        # 检查亏损
        max_loss = self.approval_rules.get('max_loss', float('inf'))
        if abs(action.get('expected_loss', 0)) > max_loss:
            return True
        
        # 检查风险等级
        risk_level = action.get('risk_level', 'low')
        if risk_level in ('high', 'critical'):
            return True
        
        return False
    
    def request_approval(
        self,
        action: Dict[str, Any],
        agent_name: str = "system",
    ) -> InterventionRequest:
        """
        请求审批
        
        Args:
            action: 动作字典
            agent_name: Agent 名称
        
        Returns:
            InterventionRequest
        """
        # 构建问题
        action_type = action.get('type', 'unknown')
        question = f"是否批准执行以下操作？\n\n操作类型: {action_type}"
        
        # 构建上下文
        context = {
            'action': action,
            'rules': self.approval_rules,
            'timestamp': datetime.now().isoformat(),
        }
        
        return self.intervention_point.request_approval(
            agent_name=agent_name,
            question=question,
            context=context,
        )
    
    def wait_for_approval(
        self,
        request_id: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        等待审批结果
        
        Args:
            request_id: 请求 ID
            timeout: 超时时间
        
        Returns:
            True 如果批准
        """
        response = self.intervention_point.wait_for_response(request_id, timeout)
        return response.approved
    
    def process_action(
        self,
        action: Dict[str, Any],
        agent_name: str = "system",
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        处理动作（自动判断是否需要审批）
        
        Args:
            action: 动作字典
            agent_name: Agent 名称
            timeout: 超时时间
        
        Returns:
            处理结果 {'approved': bool, 'reason': str, 'request_id': str}
        """
        result = {
            'approved': False,
            'reason': '',
            'request_id': None,
        }
        
        if not self.needs_approval(action):
            # 不需要审批
            if self.auto_approve_below_threshold:
                result['approved'] = True
                result['reason'] = '低于审批阈值，自动批准'
                return result
        
        # 需要审批
        request = self.request_approval(action, agent_name)
        result['request_id'] = request.request_id
        
        approved = self.wait_for_approval(request.request_id, timeout)
        result['approved'] = approved
        result['reason'] = '人类审批通过' if approved else '人类审批拒绝'
        
        return result
