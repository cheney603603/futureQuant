"""
宏观因子模块

提供与宏观经济相关的因子计算，包括：
- DollarIndexFactor: 美元指数影响因子
- InterestRateFactor: 利率因子
- CommodityIndexFactor: 商品指数因子
- InflationExpectationFactor: 通胀预期因子
"""

from .macro_factors import (
    DollarIndexFactor,
    InterestRateFactor,
    CommodityIndexFactor,
    InflationExpectationFactor,
)

__all__ = [
    "DollarIndexFactor",
    "InterestRateFactor",
    "CommodityIndexFactor",
    "InflationExpectationFactor",
]
