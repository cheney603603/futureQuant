"""
Blackboard - 中央黑板实现

基于黑板模式（Blackboard Pattern）的中央数据存储：
- 所有 Agent 通过黑板共享数据
- 支持读写锁、版本控制、变更追踪
- 支持条件触发（当某数据写入时触发回调）
- 支持人类介入点

设计原则：
1. 单一数据源：所有 Agent 数据都在黑板中
2. 松耦合：Agent 之间不直接通信，通过黑板间接协作
3. 可观测：所有数据变更都有日志和版本记录
4. 可恢复：支持快照和回滚
"""

from __future__ import annotations

import time
import uuid
import json
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from .human_intervention import InterventionRequest

from ...core.logger import get_logger

logger = get_logger('agent.blackboard')


class BlackboardState(Enum):
    """黑板状态枚举"""
    IDLE = "idle"           # 空闲
    RUNNING = "running"     # 执行中
    PAUSED = "paused"       # 已暂停（等待人类介入）
    SUCCESS = "success"     # 成功完成
    FAILED = "failed"       # 失败
    HUMAN_WAITING = "human_waiting"  # 等待人类响应


@dataclass
class BlackboardEntry:
    """
    黑板条目
    
    每个写入黑板的数据都包装为 BlackboardEntry，包含：
    - 数据本身
    - 写入者（Agent 名称）
    - 时间戳
    - 版本号
    - 元数据
    """
    key: str                           # 数据键名
    value: Any                          # 数据值
    agent: str                          # 写入的 Agent 名称
    timestamp: float                    # 写入时间戳
    version: int = 1                    # 版本号
    metadata: Dict[str, Any] = field(default_factory=dict)  # 元数据
    tags: Set[str] = field(default_factory=set)             # 标签
    
    def to_dict(self) -> Dict[str, Any]:
        """转为字典（用于序列化）"""
        return {
            'key': self.key,
            'value': self.value if not isinstance(self.value, pd.DataFrame) else '<DataFrame>',
            'agent': self.agent,
            'timestamp': self.timestamp,
            'version': self.version,
            'metadata': self.metadata,
            'tags': list(self.tags),
        }
    
    @property
    def datetime_str(self) -> str:
        """人类可读的时间字符串"""
        return datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')


class Blackboard:
    """
    中央黑板
    
    作为多 Agent 系统的中央数据存储，实现：
    - 数据读写（带版本控制）
    - 变更追踪（谁在何时写入了什么）
    - 条件触发（数据变更时触发回调）
    - 快照和恢复
    - 人类介入点集成
    
    线程安全：使用读写锁保护数据访问。
    
    使用示例：
        >>> bb = Blackboard()
        >>> 
        >>> # 写入数据
        >>> bb.write("price_data", price_df, agent="data_collector", tags={"raw", "price"})
        >>> 
        >>> # 读取数据
        >>> price = bb.read("price_data")
        >>> 
        >>> # 订阅变更
        >>> bb.subscribe("factors", my_callback)
        >>> 
        >>> # 创建快照
        >>> snapshot = bb.snapshot()
    """
    
    def __init__(
        self,
        name: str = "main",
        auto_persist: bool = False,
        persist_dir: Optional[Path] = None,
    ):
        """
        初始化黑板
        
        Args:
            name: 黑板名称（用于日志和持久化）
            auto_persist: 是否自动持久化
            persist_dir: 持久化目录
        """
        self.name = name
        self.auto_persist = auto_persist
        self.persist_dir = Path(persist_dir) if persist_dir else None
        
        # 核心数据存储
        self._data: Dict[str, BlackboardEntry] = {}
        self._version: int = 0
        
        # 状态
        self._state: BlackboardState = BlackboardState.IDLE
        self._lock = threading.RLock()  # 可重入锁
        
        # 订阅者
        self._subscribers: Dict[str, List[Callable]] = {}
        
        # 变更历史
        self._history: List[Dict[str, Any]] = []
        
        # 人类介入点
        self._intervention_requests: List[InterventionRequest] = []
        self._intervention_responses: Dict[str, Any] = {}
        
        # Agent 执行状态
        self._agent_status: Dict[str, str] = {}
        self._agent_results: Dict[str, Any] = {}
        
        logger.info(f"Blackboard '{name}' initialized")
    
    # ---- 核心读写 API ----
    
    def write(
        self,
        key: str,
        value: Any,
        agent: str,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        overwrite: bool = True,
    ) -> BlackboardEntry:
        """
        写入数据到黑板
        
        Args:
            key: 数据键名
            value: 数据值
            agent: 写入的 Agent 名称
            metadata: 元数据
            tags: 标签集合
            overwrite: 是否覆盖已有数据（False 则追加）
        
        Returns:
            BlackboardEntry: 写入的条目
        """
        with self._lock:
            timestamp = time.time()
            
            if key in self._data and not overwrite:
                # 追加模式
                existing = self._data[key]
                if isinstance(existing.value, list) and isinstance(value, list):
                    new_value = existing.value + value
                elif isinstance(existing.value, pd.DataFrame) and isinstance(value, pd.DataFrame):
                    new_value = pd.concat([existing.value, value], ignore_index=True)
                else:
                    new_value = value
                version = existing.version + 1
            elif key in self._data:
                # 覆盖模式
                version = self._data[key].version + 1
                new_value = value
            else:
                # 新建
                version = 1
                new_value = value
            
            entry = BlackboardEntry(
                key=key,
                value=new_value,
                agent=agent,
                timestamp=timestamp,
                version=version,
                metadata=metadata or {},
                tags=tags or set(),
            )
            
            self._data[key] = entry
            self._version += 1
            
            # 记录变更历史
            change_record = {
                'action': 'write',
                'key': key,
                'agent': agent,
                'timestamp': timestamp,
                'version': version,
            }
            self._history.append(change_record)
            
            # 触发订阅者
            self._notify_subscribers(key, entry)
            
            logger.debug(
                f"[Blackboard] Write: key='{key}', agent='{agent}', "
                f"version={version}, type={type(value).__name__}"
            )
            
            # 自动持久化
            if self.auto_persist:
                self._persist_entry(entry)
            
            return entry
    
    def read(
        self,
        key: str,
        default: Any = None,
        agent: Optional[str] = None,
    ) -> Any:
        """
        从黑板读取数据
        
        Args:
            key: 数据键名
            default: 默认值（数据不存在时返回）
            agent: 读取的 Agent 名称（用于日志）
        
        Returns:
            数据值，或 default
        """
        with self._lock:
            if key not in self._data:
                logger.debug(f"[Blackboard] Read miss: key='{key}', agent='{agent}'")
                return default
            
            entry = self._data[key]
            logger.debug(
                f"[Blackboard] Read: key='{key}', agent='{agent}', "
                f"version={entry.version}, writer='{entry.agent}'"
            )
            return entry.value
    
    def read_entry(self, key: str) -> Optional[BlackboardEntry]:
        """读取完整条目（包含元数据）"""
        with self._lock:
            return self._data.get(key)
    
    def exists(self, key: str) -> bool:
        """检查数据是否存在"""
        with self._lock:
            return key in self._data
    
    def delete(self, key: str, agent: str) -> bool:
        """删除数据"""
        with self._lock:
            if key not in self._data:
                return False
            
            del self._data[key]
            self._history.append({
                'action': 'delete',
                'key': key,
                'agent': agent,
                'timestamp': time.time(),
            })
            logger.debug(f"[Blackboard] Delete: key='{key}', agent='{agent}'")
            return True
    
    def keys(self) -> List[str]:
        """获取所有数据键"""
        with self._lock:
            return list(self._data.keys())
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有数据（键值对）"""
        with self._lock:
            return {k: v.value for k, v in self._data.items()}
    
    # ---- 订阅机制 ----
    
    def subscribe(
        self,
        key: str,
        callback: Callable[[str, BlackboardEntry], None],
    ):
        """
        订阅数据变更
        
        当 key 对应的数据被写入时，调用 callback。
        
        Args:
            key: 数据键名（支持通配符 '*' 表示所有变更）
            callback: 回调函数 (key, entry) -> None
        """
        with self._lock:
            if key not in self._subscribers:
                self._subscribers[key] = []
            self._subscribers[key].append(callback)
            logger.debug(f"[Blackboard] Subscribe: key='{key}'")
    
    def unsubscribe(self, key: str, callback: Callable):
        """取消订阅"""
        with self._lock:
            if key in self._subscribers:
                try:
                    self._subscribers[key].remove(callback)
                except ValueError:
                    pass
    
    def _notify_subscribers(self, key: str, entry: BlackboardEntry):
        """通知订阅者"""
        # 精确匹配
        if key in self._subscribers:
            for callback in self._subscribers[key]:
                try:
                    callback(key, entry)
                except Exception as e:
                    logger.warning(f"[Blackboard] Subscriber callback failed: {e}")
        
        # 通配符订阅
        if '*' in self._subscribers:
            for callback in self._subscribers['*']:
                try:
                    callback(key, entry)
                except Exception as e:
                    logger.warning(f"[Blackboard] Wildcard subscriber failed: {e}")
    
    # ---- Agent 状态管理 ----
    
    def set_agent_status(self, agent: str, status: str):
        """设置 Agent 状态"""
        with self._lock:
            self._agent_status[agent] = status
            logger.debug(f"[Blackboard] Agent status: {agent} -> {status}")
    
    def get_agent_status(self, agent: str) -> str:
        """获取 Agent 状态"""
        with self._lock:
            return self._agent_status.get(agent, 'unknown')
    
    def set_agent_result(self, agent: str, result: Any):
        """设置 Agent 执行结果"""
        with self._lock:
            self._agent_results[agent] = result
    
    def get_agent_result(self, agent: str) -> Any:
        """获取 Agent 执行结果"""
        with self._lock:
            return self._agent_results.get(agent)
    
    # ---- 人类介入 ----
    
    def request_intervention(
        self,
        request: 'InterventionRequest',
    ) -> str:
        """
        请求人类介入
        
        Args:
            request: 介入请求
        
        Returns:
            请求 ID
        """
        with self._lock:
            self._intervention_requests.append(request)
            self._state = BlackboardState.HUMAN_WAITING
            logger.info(
                f"[Blackboard] Human intervention requested: "
                f"type={request.intervention_type.value}, "
                f"agent={request.agent_name}, "
                f"question={request.question[:50]}..."
            )
            return request.request_id
    
    def respond_intervention(
        self,
        request_id: str,
        response: Any,
    ):
        """
        响应人类介入请求
        
        Args:
            request_id: 请求 ID
            response: 响应内容
        """
        with self._lock:
            self._intervention_responses[request_id] = response
            # 检查是否还有未响应的请求
            pending = [r for r in self._intervention_requests 
                      if r.request_id not in self._intervention_responses]
            if not pending:
                self._state = BlackboardState.RUNNING
            logger.info(f"[Blackboard] Human intervention responded: id={request_id}")
    
    def get_intervention_response(
        self,
        request_id: str,
    ) -> Optional[Any]:
        """获取人类介入响应"""
        with self._lock:
            return self._intervention_responses.get(request_id)
    
    def has_pending_interventions(self) -> bool:
        """是否有待处理的人类介入请求"""
        with self._lock:
            pending = [r for r in self._intervention_requests 
                      if r.request_id not in self._intervention_responses]
            return len(pending) > 0
    
    # ---- 快照与恢复 ----
    
    def snapshot(self) -> Dict[str, Any]:
        """
        创建黑板快照
        
        Returns:
            包含所有数据的快照字典
        """
        with self._lock:
            snapshot_data = {
                'name': self.name,
                'version': self._version,
                'state': self._state.value,
                'timestamp': time.time(),
                'data': {},
                'agent_status': dict(self._agent_status),
                'agent_results': {},
            }
            
            # 序列化数据（DataFrame 转为 JSON）
            for key, entry in self._data.items():
                if isinstance(entry.value, pd.DataFrame):
                    snapshot_data['data'][key] = {
                        'type': 'DataFrame',
                        'value': entry.value.to_json(orient='records', date_format='iso'),
                        'agent': entry.agent,
                        'timestamp': entry.timestamp,
                        'version': entry.version,
                        'metadata': entry.metadata,
                        'tags': list(entry.tags),
                    }
                elif isinstance(entry.value, pd.Series):
                    snapshot_data['data'][key] = {
                        'type': 'Series',
                        'value': entry.value.to_json(),
                        'agent': entry.agent,
                        'timestamp': entry.timestamp,
                        'version': entry.version,
                        'metadata': entry.metadata,
                        'tags': list(entry.tags),
                    }
                else:
                    try:
                        json.dumps(entry.value)  # 测试可序列化
                        snapshot_data['data'][key] = {
                            'type': 'json',
                            'value': entry.value,
                            'agent': entry.agent,
                            'timestamp': entry.timestamp,
                            'version': entry.version,
                            'metadata': entry.metadata,
                            'tags': list(entry.tags),
                        }
                    except (TypeError, ValueError):
                        snapshot_data['data'][key] = {
                            'type': 'unserializable',
                            'value': str(entry.value),
                            'agent': entry.agent,
                            'timestamp': entry.timestamp,
                            'version': entry.version,
                            'metadata': entry.metadata,
                            'tags': list(entry.tags),
                        }
            
            # Agent 结果（简化）
            for agent, result in self._agent_results.items():
                if hasattr(result, 'to_dict'):
                    snapshot_data['agent_results'][agent] = result.to_dict()
                else:
                    snapshot_data['agent_results'][agent] = str(result)[:200]
            
            logger.info(f"[Blackboard] Snapshot created: version={self._version}")
            return snapshot_data
    
    def restore(self, snapshot: Dict[str, Any]):
        """
        从快照恢复黑板
        
        Args:
            snapshot: 快照数据
        """
        with self._lock:
            self._version = snapshot.get('version', 0)
            self._state = BlackboardState(snapshot.get('state', 'idle'))
            self._agent_status = snapshot.get('agent_status', {})
            
            # 恢复数据
            self._data.clear()
            for key, entry_data in snapshot.get('data', {}).items():
                value_type = entry_data.get('type', 'json')
                if value_type == 'DataFrame':
                    value = pd.read_json(entry_data['value'], orient='records')
                elif value_type == 'Series':
                    value = pd.read_json(entry_data['value'], typ='series')
                else:
                    value = entry_data['value']
                
                self._data[key] = BlackboardEntry(
                    key=key,
                    value=value,
                    agent=entry_data['agent'],
                    timestamp=entry_data['timestamp'],
                    version=entry_data['version'],
                    metadata=entry_data.get('metadata', {}),
                    tags=set(entry_data.get('tags', [])),
                )
            
            logger.info(f"[Blackboard] Restored from snapshot: version={self._version}")
    
    # ---- 状态管理 ----
    
    @property
    def state(self) -> BlackboardState:
        """当前状态"""
        with self._lock:
            return self._state
    
    def set_state(self, state: BlackboardState):
        """设置状态"""
        with self._lock:
            old = self._state
            self._state = state
            logger.info(f"[Blackboard] State change: {old.value} -> {state.value}")
    
    # ---- 持久化 ----
    
    def _persist_entry(self, entry: BlackboardEntry):
        """持久化条目到文件"""
        if not self.persist_dir:
            return
        
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        filepath = self.persist_dir / f"{entry.key}_{entry.version}.json"
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"[Blackboard] Persist failed: {e}")
    
    def save(self, path: Optional[Path] = None):
        """保存整个黑板到文件"""
        filepath = path or (self.persist_dir / f"blackboard_{self.name}.json") if self.persist_dir else None
        if not filepath:
            return
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        snapshot = self.snapshot()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[Blackboard] Saved to {filepath}")
    
    def load(self, path: Path):
        """从文件加载黑板"""
        path = Path(path)
        if not path.exists():
            logger.warning(f"[Blackboard] Load file not found: {path}")
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            snapshot = json.load(f)
        
        self.restore(snapshot)
        logger.info(f"[Blackboard] Loaded from {path}")
    
    # ---- 工具方法 ----
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取变更历史"""
        with self._lock:
            return self._history[-limit:]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取黑板摘要"""
        with self._lock:
            return {
                'name': self.name,
                'version': self._version,
                'state': self._state.value,
                'n_entries': len(self._data),
                'keys': list(self._data.keys()),
                'agents': list(self._agent_status.keys()),
                'agent_status': dict(self._agent_status),
                'pending_interventions': self.has_pending_interventions(),
            }
    
    def clear(self):
        """清空黑板"""
        with self._lock:
            self._data.clear()
            self._agent_status.clear()
            self._agent_results.clear()
            self._intervention_requests.clear()
            self._intervention_responses.clear()
            self._history.clear()
            self._version = 0
            self._state = BlackboardState.IDLE
            logger.info(f"[Blackboard] Cleared")
    
    def __repr__(self) -> str:
        return (
            f"Blackboard(name={self.name!r}, "
            f"state={self._state.value}, "
            f"entries={len(self._data)}, "
            f"version={self._version})"
        )
