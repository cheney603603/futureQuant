from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class MarketState(Enum):
    """
    市场状态枚举

    Attributes:
        TREND_UP: 上升趋势
        TREND_DOWN: 下降趋势
        RANGE: 震荡（价格在均线附近来回穿越）
        CHANNEL: 通道（价格在两条平行线内运行）
    """

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    RANGE = "range"
    CHANNEL = "channel"


@dataclass
class PatternResult:
    """
    形态识别结果

    Attributes:
        pattern_type: 形态类型，可选：
            'triangle'（三角）/ 'rectangle'（矩形）/ 'wedge'（楔形）/
            'flag'（旗形）/ 'double_top'（双顶）/ 'none'（无形态）
        breakout_probability: 突破概率（0~1）
        recommended_direction: 推荐方向：'buy' / 'sell' / 'hold'
        entry_range: 入场区间（min, max）
        stop_loss: 止损价
        target: 目标价
        risk_ratio: 风险收益比
        confidence: 置信度（0~1）
    """

    pattern_type: str = "none"
    breakout_probability: float = 0.5
    recommended_direction: str = "hold"
    entry_range: Tuple[float, float] = (0.0, 0.0)
    stop_loss: float = 0.0
    target: float = 0.0
    risk_ratio: float = 0.0
    confidence: float = 0.5

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "pattern_type": self.pattern_type,
            "breakout_probability": self.breakout_probability,
            "recommended_direction": self.recommended_direction,
            "entry_range": self.entry_range,
            "stop_loss": self.stop_loss,
            "target": self.target,
            "risk_ratio": self.risk_ratio,
            "confidence": self.confidence,
        }
