from dataclasses import dataclass
from typing import List, Optional


@dataclass
class TradingSignal:
    """
    交易信号

    Attributes:
        date: 信号日期
        symbol: 标的代码
        direction: 方向，1=做多, 0=空仓, -1=做空
        confidence: 置信度，0~1
        model_weight: 模型集成权重
        features: 贡献最大的因子列表（可选）
    """

    date: str
    symbol: str
    direction: int
    confidence: float
    model_weight: float
    features: Optional[List[str]] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "date": self.date,
            "symbol": self.symbol,
            "direction": self.direction,
            "confidence": self.confidence,
            "model_weight": self.model_weight,
            "features": self.features,
        }
