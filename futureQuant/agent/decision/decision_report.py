from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class DecisionReport:
    """
    决策报告

    Attributes:
        target: 标的代码
        date: 决策日期
        direction: 方向，'long' / 'short' / 'neutral'
        confidence: 置信度（0~1）
        position_size: 建议仓位（0~1）
        price_target: 目标价区间（low, base, high）
        stop_loss: 止损价
        entry_range: 入场区间（min, max）
        risk_points: 风险点列表
        variables_to_watch: 需要监控的变量列表
        strategy_type: 策略类型
        scenario_analysis: 情景分析（乐观/基准/悲观）
    """

    target: str
    date: str
    direction: str = "neutral"
    confidence: float = 0.0
    position_size: float = 0.0
    price_target: tuple = (0.0, 0.0, 0.0)
    stop_loss: float = 0.0
    entry_range: tuple = (0.0, 0.0)
    risk_points: List[Dict[str, Any]] = field(default_factory=list)
    variables_to_watch: List[str] = field(default_factory=list)
    strategy_type: str = "neutral"
    scenario_analysis: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "target": self.target,
            "date": self.date,
            "direction": self.direction,
            "confidence": self.confidence,
            "position_size": self.position_size,
            "price_target": self.price_target,
            "stop_loss": self.stop_loss,
            "entry_range": self.entry_range,
            "risk_points": self.risk_points,
            "variables_to_watch": self.variables_to_watch,
            "strategy_type": self.strategy_type,
            "scenario_analysis": self.scenario_analysis,
        }
