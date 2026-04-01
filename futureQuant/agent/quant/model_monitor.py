"""
模型监控模块

功能：
- 跟踪滚动 IC（信息系数）
- 检测 IC 衰退（短期 IC vs 长期 IC）
- 触发告警：当 IC 下降 > 30% 时

使用方式：
    monitor = ModelMonitor(window_short=20, window_long=60)
    result = monitor.check(current_ic)
    if result['declining']:
        print("⚠️ 模型 IC 衰退告警！")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ICRecord:
    """IC 记录"""

    date: str
    ic: float


@dataclass
class ModelMonitor:
    """
    模型衰退监控器

    Attributes:
        window_short: 短期滚动窗口（默认 20）
        window_long: 长期滚动窗口（默认 60）
        decline_threshold: 衰退阈值（默认 0.30 = 30%）
    """

    window_short: int = 20
    window_long: int = 60
    decline_threshold: float = 0.30

    # IC 历史记录
    _ic_history: List[ICRecord] = field(default_factory=list)

    def record(self, ic: float, date: str = "") -> None:
        """
        记录一条 IC

        Args:
            ic: 当前 IC 值
            date: 日期字符串（可选）
        """
        if date == "":
            from datetime import datetime

            date = datetime.now().strftime("%Y-%m-%d")

        self._ic_history.append(ICRecord(date=date, ic=ic))

        # 保持最近 window_long * 2 条记录
        max_len = self.window_long * 2
        if len(self._ic_history) > max_len:
            self._ic_history = self._ic_history[-max_len:]

    def get_short_ic(self) -> float:
        """
        获取短期滚动 IC 均值

        Returns:
            短期 IC 均值，若不足 window_short 条记录则返回全部均值
        """
        if not self._ic_history:
            return 0.0

        recent = self._ic_history[-self.window_short :]
        if not recent:
            return 0.0
        return sum(r.ic for r in recent) / len(recent)

    def get_long_ic(self) -> float:
        """
        获取长期滚动 IC 均值

        Returns:
            长期 IC 均值
        """
        if not self._ic_history:
            return 0.0

        recent = self._ic_history[-self.window_long :]
        if not recent:
            return 0.0
        return sum(r.ic for r in recent) / len(recent)

    def check(self, current_ic: float, date: str = "") -> Dict[str, Any]:
        """
        检查模型是否衰退

        比较短期 IC（最近 N 天均值）与长期 IC（最近 M 天均值）。
        若短期 IC 较长期 IC 下降超过阈值，则触发告警。

        Args:
            current_ic: 当前 IC 值（会先记录）
            date: 日期

        Returns:
            包含以下键的字典：
                - declining (bool): 是否衰退
                - short_ic (float): 短期 IC 均值
                - long_ic (float): 长期 IC 均值
                - decline_ratio (float): 下降比例
                - signal (str): 告警级别 'ok' / 'warning' / 'critical'
                - n_records (int): 有效 IC 记录数
        """
        # 记录当前 IC
        self.record(current_ic, date)

        short_ic = self.get_short_ic()
        long_ic = self.get_long_ic()

        # 计算下降比例
        if abs(long_ic) < 1e-8:
            decline_ratio = 0.0
        else:
            decline_ratio = (long_ic - short_ic) / abs(long_ic)

        # 判断告警级别
        declining = decline_ratio > self.decline_threshold
        if decline_ratio > 0.5:
            signal = "critical"
        elif decline_ratio > self.decline_threshold:
            signal = "warning"
        else:
            signal = "ok"

        return {
            "declining": declining,
            "short_ic": round(short_ic, 6),
            "long_ic": round(long_ic, 6),
            "decline_ratio": round(decline_ratio, 4),
            "signal": signal,
            "n_records": len(self._ic_history),
        }

    def get_ic_series(self) -> Dict[str, List]:
        """
        获取 IC 历史序列（用于绘图）

        Returns:
            {"dates": [...], "ics": [...]}
        """
        return {
            "dates": [r.date for r in self._ic_history],
            "ics": [r.ic for r in self._ic_history],
        }

    def reset(self) -> None:
        """重置 IC 历史"""
        self._ic_history.clear()
