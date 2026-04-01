from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class BacktestResult:
    """
    回测结果数据类

    Attributes:
        target: 标的代码
        total_return: 总收益率（小数，如 0.15 表示 15%）
        sharpe_ratio: 夏普比率
        max_drawdown: 最大回撤（小数，如 -0.10 表示 -10%）
        win_rate: 胜率（0~1）
        n_trades: 总交易次数
        equity_curve: 权益曲线（Series，index=date，values=净值）
    """

    target: str
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    n_trades: int = 0
    equity_curve: Optional[pd.Series] = None

    def to_dict(self) -> dict:
        """转换为字典（不含 equity_curve）"""
        return {
            "target": self.target,
            "total_return": self.total_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "n_trades": self.n_trades,
        }
