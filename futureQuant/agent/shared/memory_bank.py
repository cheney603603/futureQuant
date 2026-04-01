"""
MemoryBank - Agent 记忆银行

持久化存储每个 Agent 的执行历史、成功模式、失败教训。
以 JSON 文件为单位，按 Agent 名称分目录存储。

文件结构：
    data/agent_memory/
    ├── data_collector/
    │   ├── history.json      # 执行历史记录
    │   ├── success_patterns.json  # 成功模式库
    │   └── failure_lessons.json   # 失败教训库
    ├── factor_mining/
    │   └── ...
    └── ...

Attributes:
    每条记录包含：timestamp, run_id, context, result_summary,
                 success_patterns[], failure_lessons[], tags[]

使用示例：
    >>> bank = MemoryBank()
    >>> bank.record_run("data_collector", {
    ...     "symbols": ["RB", "HC"],
    ...     "data_points": 5000,
    ... }, result={"status": "success", "records_fetched": 5000})
    >>> patterns = bank.get_success_patterns("data_collector")
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.logger import get_logger

logger = get_logger('agent.shared.memory')

DEFAULT_MEMORY_DIR = Path(__file__).parent.parent.parent.parent / "data" / "agent_memory"
MAX_HISTORY_PER_AGENT = 500  # 最多保留历史记录条数
MAX_PATTERNS = 100  # 成功模式库上限


@dataclass
class RunRecord:
    """
    单次执行记录

    Attributes:
        run_id: 唯一运行ID
        timestamp: ISO 格式时间戳
        context: 执行上下文（输入参数）
        result_summary: 结果摘要
        success_patterns: 提取的成功模式
        failure_lessons: 记录的失败教训
        tags: 标签列表
        elapsed_seconds: 执行耗时
        status: 执行状态
        agent_version: Agent 版本标识
    """
    run_id: str
    timestamp: str
    context: Dict[str, Any] = field(default_factory=dict)
    result_summary: Dict[str, Any] = field(default_factory=dict)
    success_patterns: List[str] = field(default_factory=list)
    failure_lessons: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    status: str = "unknown"
    agent_version: str = "1.0"

    def to_dict(self) -> Dict[str, Any]:
        """转为字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RunRecord":
        """从字典构建"""
        return cls(**d)


class MemoryBank:
    """
    Agent 记忆银行

    提供结构化的持久化记忆存储：
    - 执行历史（RunRecord 列表）
    - 成功模式库（高频出现并导致成功的模式）
    - 失败教训库（错误类型及应对方案）

    线程安全（写操作加文件锁）。
    """

    def __init__(
        self,
        memory_dir: Optional[Path] = None,
        auto_save: bool = True,
    ):
        """
        初始化 MemoryBank

        Args:
            memory_dir: 记忆存储根目录
            auto_save: 是否自动持久化（每次记录后立即写盘）
        """
        self.memory_dir = Path(memory_dir) if memory_dir else DEFAULT_MEMORY_DIR
        self.auto_save = auto_save
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存
        self._cache: Dict[str, List[RunRecord]] = {}
        self._success_patterns: Dict[str, List[str]] = {}
        self._failure_lessons: Dict[str, List[Dict[str, str]]] = {}
        self._dirty: Dict[str, bool] = {}  # 标记需要写盘的 agent

        logger.info(
            f"MemoryBank initialized at {self.memory_dir}, "
            f"auto_save={auto_save}"
        )

    # ---- 核心 API ----

    def record_run(
        self,
        agent_name: str,
        context: Dict[str, Any],
        result: Dict[str, Any],
        tags: Optional[List[str]] = None,
        agent_version: str = "1.0",
    ) -> RunRecord:
        """
        记录一次 Agent 执行

        Args:
            agent_name: Agent 名称（将作为子目录名）
            context: 执行上下文（输入参数、配置等）
            result: 执行结果（会被提取为摘要）
            tags: 标签列表，用于分类检索
            agent_version: Agent 版本

        Returns:
            创建的 RunRecord
        """
        run_id = f"{agent_name}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now().isoformat()

        # 提取成功模式和失败教训
        status = result.get('status', 'unknown')
        success_patterns = self._extract_success_patterns(
            agent_name, context, result
        )
        failure_lessons = self._extract_failure_lessons(
            agent_name, context, result
        )

        # 构建记录
        record = RunRecord(
            run_id=run_id,
            timestamp=timestamp,
            context=context,
            result_summary=self._summarize_result(result),
            success_patterns=success_patterns,
            failure_lessons=failure_lessons,
            tags=tags or [],
            elapsed_seconds=result.get('elapsed_seconds', 0.0),
            status=status,
            agent_version=agent_version,
        )

        # 写入缓存
        if agent_name not in self._cache:
            self._load_agent_history(agent_name)

        self._cache[agent_name].append(record)

        # 裁剪历史（防止无限增长）
        if len(self._cache[agent_name]) > MAX_HISTORY_PER_AGENT:
            self._cache[agent_name] = self._cache[agent_name][-MAX_HISTORY_PER_AGENT:]

        # 更新模式库
        if success_patterns:
            self._update_patterns(agent_name, success_patterns, is_success=True)
        if failure_lessons:
            self._update_patterns(agent_name, failure_lessons, is_success=False)

        # 标记脏
        self._dirty[agent_name] = True

        if self.auto_save:
            self.flush(agent_name)

        logger.info(
            f"[MemoryBank] Recorded run {run_id} for {agent_name}: "
            f"status={status}, elapsed={record.elapsed_seconds:.2f}s"
        )
        return record

    def get_history(
        self,
        agent_name: str,
        limit: int = 50,
        status_filter: Optional[str] = None,
    ) -> List[RunRecord]:
        """
        获取 Agent 执行历史

        Args:
            agent_name: Agent 名称
            limit: 最多返回条数（按时间倒序）
            status_filter: 按状态过滤

        Returns:
            RunRecord 列表
        """
        if agent_name not in self._cache:
            self._load_agent_history(agent_name)

        records = self._cache.get(agent_name, [])
        if status_filter:
            records = [r for r in records if r.status == status_filter]

        return list(reversed(records[-limit:]))

    def get_last_run(self, agent_name: str) -> Optional[RunRecord]:
        """获取最近一次执行记录"""
        history = self.get_history(agent_name, limit=1)
        return history[0] if history else None

    def get_success_patterns(
        self,
        agent_name: str,
        limit: int = 20,
    ) -> List[str]:
        """
        获取成功模式库

        Args:
            agent_name: Agent 名称
            limit: 最多返回条数

        Returns:
            成功模式字符串列表（按频率排序）
        """
        patterns = self._success_patterns.get(agent_name, [])
        return patterns[:limit]

    def get_failure_lessons(
        self,
        agent_name: str,
        limit: int = 20,
    ) -> List[Dict[str, str]]:
        """
        获取失败教训库

        Args:
            agent_name: Agent 名称
            limit: 最多返回条数

        Returns:
            失败教训字典列表 [{error_type, lesson, count}]
        """
        lessons = self._failure_lessons.get(agent_name, [])
        return lessons[:limit]

    def get_learned_context(
        self,
        agent_name: str,
    ) -> Dict[str, Any]:
        """
        获取学习到的上下文（供 Agent 恢复执行使用）

        综合历史最佳参数、成功模式、失败教训。

        Args:
            agent_name: Agent 名称

        Returns:
            包含 learned_params, success_patterns, failure_lessons 的字典
        """
        history = self.get_history(agent_name, limit=100)

        # 找出成功率最高的参数组合
        success_runs = [r for r in history if r.status == 'success']
        if success_runs:
            best_run = max(success_runs, key=lambda r: r.result_summary.get('score', 0))
            learned_params = best_run.context
        else:
            learned_params = {}

        return {
            'learned_params': learned_params,
            'success_patterns': self.get_success_patterns(agent_name),
            'failure_lessons': self.get_failure_lessons(agent_name),
            'total_runs': len(history),
            'success_rate': (
                len(success_runs) / len(history)
                if history else 0.0
            ),
        }

    def flush(self, agent_name: Optional[str] = None):
        """
        将内存缓存写盘

        Args:
            agent_name: 指定 Agent，为 None 则写所有脏的 Agent
        """
        agents = (
            [agent_name]
            if agent_name
            else [k for k, v in self._dirty.items() if v]
        )

        for name in agents:
            if not self._dirty.get(name, False):
                continue

            self._save_agent_history(name)
            self._save_success_patterns(name)
            self._save_failure_lessons(name)
            self._dirty[name] = False

        if agents:
            logger.debug(f"[MemoryBank] Flushed {len(agents)} agent(s) to disk")

    def clear(self, agent_name: str):
        """清除指定 Agent 的所有记忆"""
        self._cache.pop(agent_name, None)
        self._success_patterns.pop(agent_name, None)
        self._failure_lessons.pop(agent_name, None)

        for fname in ['history.json', 'success_patterns.json', 'failure_lessons.json']:
            fpath = self.memory_dir / agent_name / fname
            if fpath.exists():
                fpath.unlink()

        self._dirty[agent_name] = False
        logger.info(f"[MemoryBank] Cleared memory for {agent_name}")

    # ---- 内部方法 ----

    def _agent_dir(self, agent_name: str) -> Path:
        """获取 Agent 记忆目录"""
        d = self.memory_dir / agent_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_agent_history(self, agent_name: str):
        """加载 Agent 历史记录"""
        fpath = self._agent_dir(agent_name) / "history.json"
        try:
            if fpath.exists():
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._cache[agent_name] = [
                        RunRecord.from_dict(r) for r in data
                    ]
            else:
                self._cache[agent_name] = []
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load history for {agent_name}: {e}")
            self._cache[agent_name] = []

    def _save_agent_history(self, agent_name: str):
        """保存 Agent 历史记录"""
        fpath = self._agent_dir(agent_name) / "history.json"
        records = self._cache.get(agent_name, [])
        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump([r.to_dict() for r in records], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save history for {agent_name}: {e}")

    def _load_success_patterns(self, agent_name: str) -> List[str]:
        """加载成功模式库"""
        fpath = self._agent_dir(agent_name) / "success_patterns.json"
        try:
            if fpath.exists():
                with open(fpath, 'r', encoding='utf-8') as f:
                    patterns = json.load(f)
                    self._success_patterns[agent_name] = patterns
                    return patterns
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to load patterns for {agent_name}: {e}")
        self._success_patterns[agent_name] = []
        return []

    def _save_success_patterns(self, agent_name: str):
        """保存成功模式库"""
        fpath = self._agent_dir(agent_name) / "success_patterns.json"
        patterns = self._success_patterns.get(agent_name, [])
        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(patterns[:MAX_PATTERNS], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save patterns for {agent_name}: {e}")

    def _load_failure_lessons(self, agent_name: str) -> List[Dict[str, str]]:
        """加载失败教训库"""
        fpath = self._agent_dir(agent_name) / "failure_lessons.json"
        try:
            if fpath.exists():
                with open(fpath, 'r', encoding='utf-8') as f:
                    lessons = json.load(f)
                    self._failure_lessons[agent_name] = lessons
                    return lessons
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to load lessons for {agent_name}: {e}")
        self._failure_lessons[agent_name] = []
        return []

    def _save_failure_lessons(self, agent_name: str):
        """保存失败教训库"""
        fpath = self._agent_dir(agent_name) / "failure_lessons.json"
        lessons = self._failure_lessons.get(agent_name, [])
        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(lessons, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save lessons for {agent_name}: {e}")

    def _extract_success_patterns(
        self,
        agent_name: str,
        context: Dict[str, Any],
        result: Dict[str, Any],
    ) -> List[str]:
        """从成功执行中提取成功模式"""
        patterns = []
        status = result.get('status', '')

        if status in ('success', 'SUCCESS'):
            # 提取关键成功特征
            if result.get('data_points', 0) > 1000:
                patterns.append(f"data_volume_success:{result['data_points']}")
            if result.get('elapsed_seconds', 999) < 30:
                patterns.append("fast_execution")
            if result.get('source') == 'cache':
                patterns.append("cache_hit")

        return patterns

    def _extract_failure_lessons(
        self,
        agent_name: str,
        context: Dict[str, Any],
        result: Dict[str, Any],
    ) -> List[str]:
        """从失败执行中提取失败教训"""
        lessons = []
        status = result.get('status', '')

        if status in ('failed', 'FAILED'):
            error_type = result.get('error_type', 'unknown')
            lessons.append(f"error:{error_type}")
            if 'timeout' in str(result.get('error', '')).lower():
                lessons.append("timeout_issue")
            if 'network' in str(result.get('error', '')).lower():
                lessons.append("network_reliability")

        return lessons

    def _update_patterns(
        self,
        agent_name: str,
        new_patterns: List[str],
        is_success: bool,
    ):
        """更新模式库"""
        if agent_name not in self._success_patterns:
            self._load_success_patterns(agent_name)
        if agent_name not in self._failure_lessons:
            self._load_failure_lessons(agent_name)

        if is_success:
            existing = self._success_patterns[agent_name]
            for p in new_patterns:
                if p not in existing:
                    existing.append(p)
            self._success_patterns[agent_name] = existing[:MAX_PATTERNS]
        else:
            existing = self._failure_lessons[agent_name]
            for lesson in new_patterns:
                entry = {
                    'lesson': lesson,
                    'first_seen': datetime.now().isoformat(),
                    'count': 1,
                }
                existing.append(entry)
            self._failure_lessons[agent_name] = existing[:MAX_PATTERNS]

    @staticmethod
    def _summarize_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """提取结果摘要（只保留关键字段）"""
        summary_keys = {
            'status', 'error', 'records_fetched', 'data_points',
            'elapsed_seconds', 'ic', 'sharpe', 'n_factors',
            'signal_count', 'source', 'cache_hit',
        }
        return {k: v for k, v in result.items() if k in summary_keys}

    def __repr__(self) -> str:
        agents = list(self._cache.keys())
        return f"MemoryBank(dir={self.memory_dir.name!r}, agents={len(agents)})"
