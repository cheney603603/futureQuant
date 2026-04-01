from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class SentimentResult:
    """
    基本面情绪分析结果

    Attributes:
        target: 标的代码（如 RB2105）
        sentiment_score: 情绪评分，范围 -5（极度利空）到 +5（极度利多）
        confidence: 置信度，范围 0~1
        time_horizon: 分析时间窗口，可选 'short' / 'medium' / 'long'
        drivers: 驱动因素列表，每项包含 factor、direction、weight
        inventory_cycle: 库存周期阶段，可选值见 InventoryCycle
        supply_demand: 供需格局，'tight'（偏紧）/ 'balanced'（平衡）/ 'loose'（宽松）
    """

    target: str
    sentiment_score: float = 0.0
    confidence: float = 0.0
    time_horizon: str = "medium"
    drivers: List[Dict[str, Any]] = field(default_factory=list)
    inventory_cycle: str = "unknown"
    supply_demand: str = "balanced"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "target": self.target,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "time_horizon": self.time_horizon,
            "drivers": self.drivers,
            "inventory_cycle": self.inventory_cycle,
            "supply_demand": self.supply_demand,
        }

    def is_bullish(self) -> bool:
        """是否利多（评分 > 0.5）"""
        return self.sentiment_score > 0.5

    def is_bearish(self) -> bool:
        """是否利空（评分 < -0.5）"""
        return self.sentiment_score < -0.5
