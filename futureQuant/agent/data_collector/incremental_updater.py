"""
IncrementalUpdater - 增量更新调度器

每日定时任务逻辑：
- 日盘后 16:00 拉取日线数据
- 夜盘后 02:30 补充夜盘数据

维护更新计划表（下次更新时间、各标的优先级）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from ...core.logger import get_logger

logger = get_logger('agent.data_collector.updater')


@dataclass
class UpdateTask:
    """更新任务"""
    symbol: str
    scheduled_time: datetime
    priority: int = 0           # 0=低, 1=中, 2=高
    data_type: str = "daily"     # daily / minute / night
    status: str = "pending"      # pending / running / done / failed
    last_run: Optional[datetime] = None
    last_records: int = 0
    error: str = ""

    @property
    def is_overdue(self) -> bool:
        return datetime.now() > self.scheduled_time and self.status == "pending"


class IncrementalUpdater:
    """
    增量更新调度器

    维护每日定时更新任务，自动计算下次更新时间。
    """

    DEFAULT_SCHEDULE = {
        'day_end': '16:00',       # 日盘结束后
        'night_end': '02:30',     # 夜盘结束后（次日）
    }

    def __init__(
        self,
        schedule: Optional[Dict[str, str]] = None,
    ):
        """
        Args:
            schedule: 自定义时间表 {'day_end': 'HH:MM', 'night_end': 'HH:MM'}
        """
        self.schedule = {**self.DEFAULT_SCHEDULE, **(schedule or {})}

        # 更新计划表
        self._tasks: Dict[str, UpdateTask] = {}

        # 各标的优先级（根据持仓、热点等）
        self._priority_map: Dict[str, int] = {}

    def add_task(
        self,
        symbol: str,
        data_type: str = "daily",
        priority: int = 1,
    ) -> UpdateTask:
        """
        添加更新任务

        Args:
            symbol: 标的代码
            data_type: 数据类型
            priority: 优先级

        Returns:
            创建的 UpdateTask
        """
        scheduled = self._calculate_next_run(data_type)
        task = UpdateTask(
            symbol=symbol,
            scheduled_time=scheduled,
            priority=priority,
            data_type=data_type,
        )
        self._tasks[symbol] = task
        self._priority_map[symbol] = priority
        logger.info(f"Added update task: {symbol} @ {scheduled}")
        return task

    def remove_task(self, symbol: str):
        """移除更新任务"""
        self._tasks.pop(symbol, None)
        self._priority_map.pop(symbol, None)

    def get_pending_tasks(self) -> List[UpdateTask]:
        """获取待执行任务（按优先级排序）"""
        tasks = [t for t in self._tasks.values() if t.status == "pending"]
        return sorted(tasks, key=lambda t: (-t.priority, t.scheduled_time))

    def get_overdue_tasks(self) -> List[UpdateTask]:
        """获取过期任务"""
        return [t for t in self._tasks.values() if t.is_overdue]

    def mark_running(self, symbol: str):
        """标记任务为运行中"""
        if symbol in self._tasks:
            self._tasks[symbol].status = "running"
            self._tasks[symbol].last_run = datetime.now()

    def mark_done(self, symbol: str, records: int):
        """标记任务完成"""
        if symbol in self._tasks:
            t = self._tasks[symbol]
            t.status = "done"
            t.last_records = records
            t.last_run = datetime.now()
            # 计算下次运行时间
            t.scheduled_time = self._calculate_next_run(t.data_type)

    def mark_failed(self, symbol: str, error: str):
        """标记任务失败"""
        if symbol in self._tasks:
            self._tasks[symbol].status = "failed"
            self._tasks[symbol].error = error

    def update_priorities(self, priority_map: Dict[str, int]):
        """批量更新优先级"""
        for symbol, priority in priority_map.items():
            if symbol in self._tasks:
                self._tasks[symbol].priority = priority
        self._priority_map.update(priority_map)

    def _calculate_next_run(self, data_type: str) -> datetime:
        """计算下次运行时间"""
        now = datetime.now()
        time_str = self.schedule.get(
            'night_end' if data_type == 'night' else 'day_end',
            '16:00',
        )
        hour, minute = map(int, time_str.split(':'))

        if data_type == 'night':
            # 夜盘后（次日凌晨）
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)
        else:
            # 日盘后
            next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_run <= now:
                next_run += timedelta(days=1)

        return next_run

    def get_schedule_summary(self) -> Dict[str, Any]:
        """获取调度摘要"""
        pending = self.get_pending_tasks()
        overdue = self.get_overdue_tasks()

        return {
            'total_tasks': len(self._tasks),
            'pending': len(pending),
            'overdue': len(overdue),
            'next_run': min((t.scheduled_time for t in pending), default=None),
            'schedule': self.schedule,
            'priorities': self._priority_map,
        }

    def __repr__(self) -> str:
        pending = len([t for t in self._tasks.values() if t.status == "pending"])
        return f"IncrementalUpdater(pending={pending}, total={len(self._tasks)})"
