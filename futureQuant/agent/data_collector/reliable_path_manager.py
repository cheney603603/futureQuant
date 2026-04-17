"""
ReliablePathManager - 可靠链路管理

核心职责：
1. 维护可靠链路库（哪些数据源+参数组合历史上成功过）
2. 记录链路元数据（成功率、响应时间、数据质量评分）
3. 提供链路评分与排序
4. 链路失效时触发降级

数据结构（链路 = Path）：
{
    "path_id": "akshare_rb_daily",
    "source": "akshare",
    "data_type": "daily",
    "symbol_pattern": "RB*",        # 支持通配符
    "params": {"variety": "RB"},
    "status": "active",              # active | degraded | retired
    "success_count": 42,
    "failure_count": 3,
    "success_rate": 0.933,
    "avg_response_ms": 850,
    "last_success": "2026-04-04T01:23:04",
    "last_failure": "2026-04-02T10:15:22",
    "last_try": "2026-04-04T01:23:04",
    "quality_score": 0.95,          # 0-1, 数据质量主观分
    "tags": ["螺纹钢", "日线", "akshare"],
    "created_at": "2026-04-01T00:00:00",
}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...core.logger import get_logger

logger = get_logger('agent.data_collector.path_manager')


class PathStatus(Enum):
    ACTIVE = "active"
    DEGRADED = "degraded"
    RETIRED = "retired"
    UNTESTED = "untested"


@dataclass
class ReliablePath:
    """一条可靠链路"""
    path_id: str
    source: str
    data_type: str
    symbol_pattern: str = "*"
    params: Dict[str, Any] = field(default_factory=dict)
    status: str = "untested"
    success_count: int = 0
    failure_count: int = 0
    avg_response_ms: float = 0.0
    last_success: str = ""
    last_failure: str = ""
    last_try: str = ""
    quality_score: float = 0.5
    tags: List[str] = field(default_factory=list)
    created_at: str = ""
    # 扩展字段：存储最后一次成功的实际数据样本信息
    last_fetch_records: int = 0
    freshness_days: int = 0  # 数据时效（天）
    generated_by_llm: bool = False
    source_code: str = ""

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.0
        return self.success_count / total

    @property
    def is_active(self) -> bool:
        return self.status == PathStatus.ACTIVE.value

    @property
    def is_reliable(self) -> bool:
        """成功率 >= 80% 且最近 7 天内有成功记录"""
        if self.success_count < 3:
            return False
        if self.success_rate < 0.8:
            return False
        if self.last_success:
            last_s = datetime.fromisoformat(self.last_success)
            if (datetime.now() - last_s).days > 7:
                return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['success_rate'] = self.success_rate
        d['is_active'] = self.is_active
        d['is_reliable'] = self.is_reliable
        return d


class ReliablePathManager:
    """
    可靠链路管理器

    职责：
    - 链路库持久化（JSON文件）
    - 新链路注册
    - 链路调用记录（成功/失败）
    - 链路健康度评估
    - 失效链路降级与恢复
    """

    DEFAULT_PATH = "D:/310Programm/futureQuant/data/agent_memory/data_collector/reliable_paths.json"

    # 链路评分权重
    WEIGHTS = {
        'success_rate': 0.40,
        'recency': 0.25,    # 最近成功时间
        'speed': 0.20,     # 响应速度
        'quality': 0.15,   # 数据质量
    }

    def __init__(self, path_file: Optional[str] = None):
        """
        Args:
            path_file: 链路库 JSON 文件路径
        """
        self._path_file = Path(path_file or self.DEFAULT_PATH)
        self._path_file.parent.mkdir(parents=True, exist_ok=True)
        self._paths: Dict[str, ReliablePath] = {}
        self._load()

    # ==================== 持久化 ====================

    def _load(self):
        """从文件加载链路库"""
        if self._path_file.exists():
            try:
                with open(self._path_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                for d in raw if isinstance(raw, list) else raw.get('paths', []):
                    # 过滤掉计算属性（不在 dataclass 字段中）
                    valid_keys = {f.name for f in ReliablePath.__dataclass_fields__.values()}
                    clean = {k: v for k, v in d.items() if k in valid_keys}
                    p = ReliablePath(**clean)
                    self._paths[p.path_id] = p
                logger.info(f"Loaded {len(self._paths)} reliable paths")
            except Exception as exc:
                logger.warning(f"Failed to load path file: {exc}, starting fresh")
                self._paths = {}
        else:
            self._paths = {}

    def _save(self):
        """持久化链路库到文件"""
        try:
            data = {
                'version': '1.0',
                'updated_at': datetime.now().isoformat(),
                'paths': [p.to_dict() for p in self._paths.values()],
            }
            with open(self._path_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"Failed to save path file: {exc}")

    # ==================== 链路注册 ====================

    def register_path(
        self,
        source: str,
        data_type: str,
        symbol_pattern: str = "*",
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        ask_user: bool = True,
    ) -> Tuple[ReliablePath, bool]:
        """
        注册新链路（需要用户确认）

        Args:
            source: 数据源名称
            data_type: 数据类型
            symbol_pattern: 标的匹配模式
            params: 额外参数
            tags: 标签
            ask_user: 是否需要用户授权才真正持久化

        Returns:
            (ReliablePath, 是否已授权)
        """
        path_id = self._make_path_id(source, data_type, symbol_pattern, params)

        if path_id in self._paths:
            logger.info(f"Path {path_id} already exists")
            return self._paths[path_id], True

        now = datetime.now().isoformat()
        path = ReliablePath(
            path_id=path_id,
            source=source,
            data_type=data_type,
            symbol_pattern=symbol_pattern,
            params=params or {},
            status=PathStatus.UNTESTED.value,
            tags=tags or [],
            created_at=now,
            last_try=now,
        )
        self._paths[path_id] = path
        # 暂时不保存，等待第一次成功确认
        logger.info(f"Registered new path: {path_id} (pending confirmation)")
        return path, ask_user

    def confirm_path(self, path_id: str, success: bool, response_ms: float = 0,
                     records: int = 0) -> bool:
        """
        确认链路首次执行结果（成功则永久保存，失败则标记）

        Args:
            path_id: 链路ID
            success: 是否成功
            response_ms: 响应时间
            records: 返回记录数

        Returns:
            是否保存成功
        """
        if path_id not in self._paths:
            logger.warning(f"Path {path_id} not found for confirmation")
            return False

        path = self._paths[path_id]
        now = datetime.now().isoformat()
        path.last_try = now

        if success:
            path.success_count += 1
            path.last_success = now
            path.status = PathStatus.ACTIVE.value
            path.last_fetch_records = records
            if response_ms > 0:
                # 滑动平均更新响应时间
                if path.avg_response_ms == 0:
                    path.avg_response_ms = response_ms
                else:
                    path.avg_response_ms = path.avg_response_ms * 0.7 + response_ms * 0.3
            self._save()
            logger.info(f"Path {path_id} confirmed SUCCESS, rate={path.success_rate:.2%}")
            return True
        else:
            path.failure_count += 1
            path.last_failure = now
            # 连续失败3次以上，降级
            if path.failure_count >= 3:
                path.status = PathStatus.DEGRADED.value
            self._save()
            logger.warning(f"Path {path_id} confirmed FAILURE, failures={path.failure_count}")
            return False

    def record_run(
        self,
        path_id: str,
        success: bool,
        response_ms: float = 0,
        records: int = 0,
        quality_score: float = 0.5,
    ):
        """
        记录链路运行结果（每次调用后调用）

        Args:
            path_id: 链路ID
            success: 是否成功
            response_ms: 响应时间（毫秒）
            records: 返回记录数
            quality_score: 数据质量评分 0-1
        """
        if path_id not in self._paths:
            logger.warning(f"Path {path_id} not registered, cannot record run")
            return

        path = self._paths[path_id]
        now = datetime.now().isoformat()
        path.last_try = now

        if success:
            path.success_count += 1
            path.last_success = now
            path.status = PathStatus.ACTIVE.value
            path.last_fetch_records = records
            path.quality_score = (
                path.quality_score * 0.7 + quality_score * 0.3
            )
            if response_ms > 0:
                if path.avg_response_ms == 0:
                    path.avg_response_ms = response_ms
                else:
                    path.avg_response_ms = path.avg_response_ms * 0.8 + response_ms * 0.2
        else:
            path.failure_count += 1
            path.last_failure = now
            if path.failure_count >= 3:
                path.status = PathStatus.DEGRADED.value

        self._save()

    # ==================== 查询与推荐 ====================

    def get_reliable_paths(
        self,
        data_type: Optional[str] = None,
        symbol: Optional[str] = None,
        min_success_rate: float = 0.7,
        limit: int = 5,
    ) -> List[ReliablePath]:
        """
        获取可靠链路列表（按评分排序）

        Args:
            data_type: 数据类型过滤
            symbol: 标的代码（用于模式匹配）
            min_success_rate: 最低成功率要求
            limit: 返回数量限制

        Returns:
            排序后的链路列表
        """
        candidates = []
        for p in self._paths.values():
            if p.status == PathStatus.RETIRED.value:
                continue
            if data_type and p.data_type != data_type:
                continue
            if p.success_count < 1:
                continue
            if p.success_rate < min_success_rate:
                continue
            candidates.append(p)

        # 评分排序
        scored = [(self._score_path(p), p) for p in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]

    def get_best_path(
        self,
        data_type: str,
        symbol: Optional[str] = None,
    ) -> Optional[ReliablePath]:
        """获取最佳链路"""
        paths = self.get_reliable_paths(data_type=data_type, symbol=symbol, limit=1)
        return paths[0] if paths else None

    def get_all_paths(self) -> List[ReliablePath]:
        """获取所有链路"""
        return list(self._paths.values())

    def get_stats(self) -> Dict[str, Any]:
        """获取链路库统计"""
        total = len(self._paths)
        active = sum(1 for p in self._paths.values() if p.status == PathStatus.ACTIVE.value)
        degraded = sum(1 for p in self._paths.values() if p.status == PathStatus.DEGRADED.value)
        untested = sum(1 for p in self._paths.values() if p.status == PathStatus.UNTESTED.value)
        return {
            'total': total,
            'active': active,
            'degraded': degraded,
            'untested': untested,
            'avg_success_rate': (
                sum(p.success_rate for p in self._paths.values()) / max(total, 1)
            ),
        }

    # ==================== 链路操作 ====================

    def retire_path(self, path_id: str):
        """退役链路"""
        if path_id in self._paths:
            self._paths[path_id].status = PathStatus.RETIRED.value
            self._save()
            logger.info(f"Path {path_id} retired")

    def reactivate_path(self, path_id: str):
        """重新激活链路"""
        if path_id in self._paths:
            self._paths[path_id].status = PathStatus.ACTIVE.value
            self._paths[path_id].failure_count = 0
            self._save()
            logger.info(f"Path {path_id} reactivated")

    def remove_path(self, path_id: str):
        """删除链路"""
        if path_id in self._paths:
            del self._paths[path_id]
            self._save()
            logger.info(f"Path {path_id} removed")

    # ==================== 工具 ====================

    @staticmethod
    def _make_path_id(
        source: str,
        data_type: str,
        symbol_pattern: str,
        params: Optional[Dict[str, Any]],
    ) -> str:
        """生成链路ID"""
        params_key = json.dumps(params or {}, sort_keys=True, ensure_ascii=False)
        # 简短的hash
        import hashlib
        h = hashlib.md5(f"{source}:{data_type}:{symbol_pattern}:{params_key}".encode()).hexdigest()[:8]
        return f"{source}_{data_type}_{symbol_pattern}_{h}"

    def _score_path(self, path: ReliablePath) -> float:
        """计算链路综合评分"""
        # 成功率分数
        rate_score = path.success_rate

        # 新近度分数（最近成功时间）
        recency_score = 0.0
        if path.last_success:
            try:
                last_s = datetime.fromisoformat(path.last_success)
                days_ago = (datetime.now() - last_s).total_seconds() / 86400
                recency_score = max(0, 1 - days_ago / 7)  # 7天内满分，7天外0分
            except Exception:
                recency_score = 0.0

        # 速度分数（响应时间，越快越好）
        speed_score = 0.0
        if path.avg_response_ms > 0:
            # 500ms内满分，5000ms以上0分
            speed_score = max(0, 1 - (path.avg_response_ms - 500) / 4500)

        # 质量分数
        quality_score = path.quality_score

        total = (
            self.WEIGHTS['success_rate'] * rate_score +
            self.WEIGHTS['recency'] * recency_score +
            self.WEIGHTS['speed'] * speed_score +
            self.WEIGHTS['quality'] * quality_score
        )
        return round(total, 4)

    def print_summary(self):
        """打印链路库摘要（调试用）"""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"  可靠链路库摘要")
        print(f"{'='*60}")
        print(f"  总链路数: {stats['total']}")
        print(f"  活跃链路: {stats['active']}")
        print(f"  降级链路: {stats['degraded']}")
        print(f"  未测试:   {stats['untested']}")
        print(f"  平均成功率: {stats['avg_success_rate']:.1%}")
        print(f"{'='*60}")
        for p in sorted(self._paths.values(), key=lambda x: -self._score_path(x)):
            print(f"  [{p.status:10}] {p.path_id}")
            print(f"    成功率: {p.success_rate:.1%} ({p.success_count}/{p.success_count+p.failure_count})")
            print(f"    响应: {p.avg_response_ms:.0f}ms | 质量: {p.quality_score:.2f}")
            print(f"    上次成功: {p.last_success[:19] if p.last_success else 'N/A'}")
        print()
