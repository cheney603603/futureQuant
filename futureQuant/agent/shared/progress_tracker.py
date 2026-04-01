"""
ProgressTracker - Agent 进度追踪器

记录每个 Agent 的运行状态、耗时、输出摘要。
支持实时进度回调、阶段报告生成。

使用示例：
    >>> tracker = ProgressTracker()
    >>> tracker.start("data_collector", total_steps=7)
    >>> tracker.update("data_collector", step=3, message="Fetching RB data...")
    >>> tracker.complete("data_collector", summary={"records": 5000})
"""
from __future__ import annotations

import time
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.logger import get_logger

logger = get_logger('agent.shared.progress')

DEFAULT_PROGRESS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "agent_progress"


@dataclass
class StepRecord:
    """步骤记录"""
    step: int
    name: str
    message: str = ""
    elapsed: float = 0.0
    status: str = "pending"  # pending / running / done / failed
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AgentProgress:
    """
    Agent 进度快照

    Attributes:
        agent_name: Agent 名称
        status: overall status
        total_steps: 总步骤数
        current_step: 当前步骤
        start_time: 开始时间（epoch 秒）
        end_time: 结束时间（epoch 秒）
        elapsed: 总耗时（秒）
        steps: 各步骤记录
        summary: 执行摘要
        output_summary: 输出摘要
    """
    agent_name: str
    status: str = "idle"
    total_steps: int = 0
    current_step: int = 0
    start_time: float = 0.0
    end_time: float = 0.0
    elapsed: float = 0.0
    steps: List[StepRecord] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    output_summary: Dict[str, Any] = field(default_factory=dict)

    @property
    def progress_pct(self) -> float:
        """完成百分比"""
        if self.total_steps == 0:
            return 0.0
        return (self.current_step / self.total_steps) * 100

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    @property
    def is_done(self) -> bool:
        return self.status in ("success", "failed", "done")

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['progress_pct'] = self.progress_pct
        return d


class ProgressTracker:
    """
    Agent 进度追踪器

    功能：
    - 追踪每个 Agent 的阶段进度
    - 记录每步耗时
    - 生成执行摘要
    - 持久化进度状态（崩溃恢复）
    - 实时进度回调

    使用示例：
        >>> tracker = ProgressTracker(on_progress=my_callback)
        >>> tracker.start("factor_mining", total_steps=5)
        >>> tracker.update("factor_mining", step=2, message="Computing IC...")
        >>> tracker.step_complete("factor_mining", step=2)
        >>> tracker.complete("factor_mining", summary={"n_factors": 50})
    """

    def __init__(
        self,
        progress_dir: Optional[Path] = None,
        auto_save: bool = True,
        on_progress: Optional[callable] = None,
    ):
        """
        初始化 ProgressTracker

        Args:
            progress_dir: 进度文件存储目录
            auto_save: 是否自动持久化
            on_progress: 进度更新回调 (AgentProgress) -> None
        """
        self.progress_dir = Path(progress_dir) if progress_dir else DEFAULT_PROGRESS_DIR
        self.auto_save = auto_save
        self.on_progress = on_progress

        self.progress_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存
        self._progress: Dict[str, AgentProgress] = {}

        # 加载已有进度
        self._load_all()

        logger.info(f"ProgressTracker initialized at {self.progress_dir}")

    # ---- 核心 API ----

    def start(
        self,
        agent_name: str,
        total_steps: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentProgress:
        """
        开始追踪 Agent

        Args:
            agent_name: Agent 名称
            total_steps: 总步骤数
            metadata: 额外元数据

        Returns:
            AgentProgress 对象
        """
        progress = AgentProgress(
            agent_name=agent_name,
            status="running",
            total_steps=total_steps,
            current_step=0,
            start_time=time.time(),
        )
        if metadata:
            progress.summary.update(metadata)

        self._progress[agent_name] = progress
        self._persist(agent_name)
        self._notify(progress)

        logger.info(
            f"[ProgressTracker] Started {agent_name}: "
            f"{total_steps} steps"
        )
        return progress

    def update(
        self,
        agent_name: str,
        step: Optional[int] = None,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentProgress:
        """
        更新进度（设置当前步骤和消息）

        Args:
            agent_name: Agent 名称
            step: 当前步骤（从1开始）
            message: 进度描述
            metadata: 额外元数据

        Returns:
            更新后的 AgentProgress
        """
        progress = self._get_or_create(agent_name)

        if step is not None:
            progress.current_step = step
            # 确保 steps 列表足够长
            while len(progress.steps) < step:
                progress.steps.append(StepRecord(
                    step=len(progress.steps) + 1,
                    name="",
                    status="pending",
                    timestamp=datetime.now().isoformat(),
                ))

        if message:
            if progress.steps and progress.current_step > 0:
                s = progress.steps[progress.current_step - 1]
                s.message = message
            progress.summary['last_message'] = message

        if metadata:
            progress.summary.update(metadata)

        progress.elapsed = time.time() - progress.start_time
        self._persist(agent_name)
        self._notify(progress)

        return progress

    def step_start(
        self,
        agent_name: str,
        step: int,
        name: str = "",
    ) -> AgentProgress:
        """
        标记步骤开始

        Args:
            agent_name: Agent 名称
            step: 步骤编号（从1开始）
            name: 步骤名称

        Returns:
            AgentProgress 对象
        """
        progress = self._get_or_create(agent_name)
        progress.current_step = step

        # 扩展 steps 列表
        while len(progress.steps) < step:
            progress.steps.append(StepRecord(
                step=len(progress.steps) + 1,
                name="",
                status="pending",
                timestamp=datetime.now().isoformat(),
            ))

        s = progress.steps[step - 1]
        s.name = name
        s.status = "running"
        s.timestamp = datetime.now().isoformat()

        self._persist(agent_name)
        self._notify(progress)
        logger.debug(f"[ProgressTracker] {agent_name} step {step} started: {name}")

        return progress

    def step_complete(
        self,
        agent_name: str,
        step: Optional[int] = None,
        message: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentProgress:
        """
        标记步骤完成

        Args:
            agent_name: Agent 名称
            step: 步骤编号（None 表示当前步骤）
            message: 步骤结果描述
            metadata: 步骤元数据

        Returns:
            AgentProgress 对象
        """
        progress = self._get_or_create(agent_name)

        if step is None:
            step = progress.current_step

        if step > 0 and step <= len(progress.steps):
            s = progress.steps[step - 1]
            s.status = "done"
            if message:
                s.message = message
            s.elapsed = time.time() - progress.start_time - sum(
                st.elapsed for st in progress.steps[:step - 1]
            )
            # 简单计算当前步骤耗时
            if step >= 2:
                prev_end = sum(st.elapsed for st in progress.steps[:step - 1])
                curr_start = prev_end
                # 与开始时间差估算
                pass

        if metadata:
            progress.output_summary.update(metadata)

        self._persist(agent_name)
        self._notify(progress)

        pct = progress.progress_pct
        logger.info(
            f"[ProgressTracker] {agent_name} step {step} done "
            f"({pct:.0f}%): {message}"
        )

        return progress

    def step_failed(
        self,
        agent_name: str,
        step: Optional[int] = None,
        error: str = "",
    ) -> AgentProgress:
        """
        标记步骤失败

        Args:
            agent_name: Agent 名称
            step: 步骤编号
            error: 错误信息

        Returns:
            AgentProgress 对象
        """
        progress = self._get_or_create(agent_name)

        if step is None:
            step = progress.current_step

        if step > 0 and step <= len(progress.steps):
            s = progress.steps[step - 1]
            s.status = "failed"
            s.message = error

        progress.status = "failed"
        progress.elapsed = time.time() - progress.start_time

        self._persist(agent_name)
        self._notify(progress)
        logger.error(f"[ProgressTracker] {agent_name} step {step} failed: {error}")

        return progress

    def complete(
        self,
        agent_name: str,
        status: str = "success",
        summary: Optional[Dict[str, Any]] = None,
        output_summary: Optional[Dict[str, Any]] = None,
    ) -> AgentProgress:
        """
        标记 Agent 执行完成

        Args:
            agent_name: Agent 名称
            status: 最终状态
            summary: 执行摘要
            output_summary: 输出摘要

        Returns:
            AgentProgress 对象
        """
        progress = self._get_or_create(agent_name)

        progress.status = status
        progress.end_time = time.time()
        progress.elapsed = progress.end_time - progress.start_time

        if summary:
            progress.summary.update(summary)
        if output_summary:
            progress.output_summary.update(output_summary)

        self._persist(agent_name)
        self._notify(progress)

        logger.info(
            f"[ProgressTracker] {agent_name} completed: "
            f"status={status}, elapsed={progress.elapsed:.2f}s"
        )

        return progress

    def get_progress(self, agent_name: str) -> Optional[AgentProgress]:
        """获取 Agent 进度"""
        return self._progress.get(agent_name)

    def get_all_progress(self) -> Dict[str, AgentProgress]:
        """获取所有 Agent 进度"""
        return dict(self._progress)

    def get_report(self) -> str:
        """
        生成进度报告（Markdown 格式）

        Returns:
            Markdown 格式的进度报告字符串
        """
        lines = [
            "# Agent 执行进度报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "| Agent | 状态 | 进度 | 耗时 |",
            "|-------|------|------|------|",
        ]

        for name, prog in sorted(self._progress.items()):
            status_icon = {
                'success': '✅', 'failed': '❌',
                'running': '🔄', 'idle': '⏸',
            }.get(prog.status, '⬜')

            lines.append(
                f"| {name} | {status_icon} {prog.status} | "
                f"{prog.current_step}/{prog.total_steps} "
                f"({prog.progress_pct:.0f}%) | {prog.elapsed:.1f}s |"
            )

        lines.append("")

        # 详细输出摘要
        for name, prog in sorted(self._progress.items()):
            if prog.output_summary:
                lines.append(f"### {name}")
                for k, v in prog.output_summary.items():
                    lines.append(f"- **{k}**: {v}")
                lines.append("")

        return "\n".join(lines)

    # ---- 内部方法 ----

    def _get_or_create(self, agent_name: str) -> AgentProgress:
        """获取或创建 Progress"""
        if agent_name not in self._progress:
            self._progress[agent_name] = AgentProgress(agent_name=agent_name)
        return self._progress[agent_name]

    def _persist(self, agent_name: str):
        """持久化进度到文件"""
        if not self.auto_save:
            return

        progress = self._progress.get(agent_name)
        if not progress:
            return

        fpath = self.progress_dir / f"{agent_name}.json"
        try:
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(progress.to_dict(), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist progress for {agent_name}: {e}")

    def _load_all(self):
        """加载所有进度文件"""
        if not self.progress_dir.exists():
            return

        for fpath in self.progress_dir.glob("*.json"):
            agent_name = fpath.stem
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._progress[agent_name] = AgentProgress(**{
                        k: v for k, v in data.items()
                        if k in AgentProgress.__dataclass_fields__
                    })
            except Exception as e:
                logger.warning(f"Failed to load progress from {fpath}: {e}")

    def _notify(self, progress: AgentProgress):
        """通知进度更新"""
        if self.on_progress:
            try:
                self.on_progress(progress)
            except Exception as e:
                logger.warning(f"on_progress callback failed: {e}")

    def __repr__(self) -> str:
        running = sum(1 for p in self._progress.values() if p.is_running)
        done = sum(1 for p in self._progress.values() if p.is_done)
        return (
            f"ProgressTracker("
            f"total={len(self._progress)}, "
            f"running={running}, "
            f"done={done})"
        )
