"""
交易成本模型 (Transaction Cost Model)

计算交易过程中的各种成本：
- 手续费：固定费用 + 比例费用
- 滑点成本：线性、平方根、指数模型
- 市场冲击：临时冲击 + 永久冲击
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from ...core.logger import get_logger

logger = get_logger('agent.cost_model')


class SlippageModel(Enum):
    """滑点模型类型"""
    LINEAR = 'linear'
    SQUARE_ROOT = 'square_root'
    EXPONENTIAL = 'exponential'


@dataclass
class CostResult:
    """成本计算结果"""
    total_cost: float
    commission_cost: float
    slippage_cost: float
    market_impact_cost: float
    cost_ratio: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'total_cost': self.total_cost,
            'commission_cost': self.commission_cost,
            'slippage_cost': self.slippage_cost,
            'market_impact_cost': self.market_impact_cost,
            'cost_ratio': self.cost_ratio,
        }


class CostModel:
    """
    交易成本模型

    综合计算交易成本，包括手续费、滑点和市场冲击。
    """

    def __init__(
        self,
        fixed_cost: float = 5.0,
        commission_rate: float = 0.0001,
        slippage_rate: float = 0.0001,
        slippage_model: str = 'linear',
        impact_model: str = 'sqrt',
        permanent_impact_ratio: float = 0.3,
    ) -> None:
        """
        Args:
            fixed_cost: 固定手续费（每笔）
            commission_rate: 比例手续费率
            slippage_rate: 滑点率
            slippage_model: 滑点模型，'linear' / 'square_root' / 'exponential'
            impact_model: 市场冲击模型，'sqrt' / 'linear'
            permanent_impact_ratio: 永久冲击占总冲击的比例
        """
        self.fixed_cost = fixed_cost
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.slippage_model = SlippageModel(slippage_model)
        self.impact_model = impact_model
        self.permanent_impact_ratio = permanent_impact_ratio

    def calculate_total_cost(
        self,
        trade_value: float,
        volume: float,
        price: float,
        volatility: float = 0.02,
    ) -> CostResult:
        """
        计算总交易成本

        Args:
            trade_value: 交易金额
            volume: 交易量
            price: 当前价格
            volatility: 波动率

        Returns:
            CostResult 对象
        """
        commission = self.calculate_commission(trade_value)
        slippage = self.calculate_slippage(trade_value, volume, price, volatility)
        impact = self.calculate_market_impact(trade_value, volume, price, volatility)

        total = commission + slippage + impact
        ratio = total / trade_value if trade_value > 0 else 0.0

        return CostResult(
            total_cost=total,
            commission_cost=commission,
            slippage_cost=slippage,
            market_impact_cost=impact,
            cost_ratio=ratio,
        )

    def calculate_commission(self, trade_value: float) -> float:
        """计算手续费"""
        return self.fixed_cost + trade_value * self.commission_rate

    def calculate_slippage(
        self,
        trade_value: float,
        volume: float,
        price: float,
        volatility: float,
    ) -> float:
        """计算滑点成本"""
        base_slippage = trade_value * self.slippage_rate

        if self.slippage_model == SlippageModel.LINEAR:
            return base_slippage
        elif self.slippage_model == SlippageModel.SQUARE_ROOT:
            # 滑点与交易量的平方根成正比
            participation = volume / (volume + 1e8)
            return base_slippage * np.sqrt(participation + 1)
        elif self.slippage_model == SlippageModel.EXPONENTIAL:
            return base_slippage * (1 + volatility * 2)
        return base_slippage

    def calculate_market_impact(
        self,
        trade_value: float,
        volume: float,
        price: float,
        volatility: float,
    ) -> float:
        """
        计算市场冲击成本

        Args:
            trade_value: 交易金额
            volume: 成交量
            price: 价格
            volatility: 波动率

        Returns:
            市场冲击成本
        """
        if volume <= 0 or price <= 0:
            return 0.0

        participation = volume / (volume + 1e8)
        if participation < 1e-8:
            return 0.0

        if self.impact_model == 'sqrt':
            # Almgren-Chriss 平方根模型
            base_impact = price * volatility * np.sqrt(participation)
        else:
            # 线性模型
            base_impact = price * volatility * participation

        temporary_impact = base_impact * (1 - self.permanent_impact_ratio)
        permanent_impact = base_impact * self.permanent_impact_ratio

        return trade_value * (temporary_impact / price + permanent_impact / price)

    def estimate_annual_cost(
        self,
        trade_value: float,
        annual_trades: int,
        avg_holding_days: int = 5,
    ) -> Dict[str, float]:
        """
        估算年化交易成本

        Args:
            trade_value: 平均交易金额
            annual_trades: 年交易次数
            avg_holding_days: 平均持仓天数

        Returns:
            年化成本估算
        """
        single_cost = self.calculate_total_cost(trade_value, 0, 0).total_cost
        gross_annual = single_cost * annual_trades

        turnover = 252 / avg_holding_days
        capital = trade_value
        turnover_ratio = 1 / avg_holding_days * 252

        cost_rate = self.commission_rate * 2 + self.slippage_rate * 2
        annual_rate = cost_rate * turnover_ratio

        return {
            'gross_annual_cost': gross_annual,
            'net_annual_cost': capital * annual_rate,
            'annual_cost_ratio': annual_rate,
        }
