"""
策略模块

提供策略基类、趋势跟踪、均值回归、套利策略和参数优化功能
"""

from .base import BaseStrategy
from .trend_following import (
    TrendFollowingStrategy,
    DualMAStrategy,
    BreakoutStrategy,
    DonchianChannelStrategy,
)
from .mean_reversion import (
    MeanReversionStrategy,
    RSIStrategy,
    BollingerBandsStrategy,
    CointegrationStrategy,
)
from .arbitrage import (
    ArbitrageStrategy,
    SpreadArbitrageStrategy,
    CrossVarietyArbitrageStrategy,
    FuturesSpotArbitrageStrategy,
)
from .optimizer import StrategyOptimizer, OptimizationResult
from .optuna_optimizer import OptunaOptimizer, quick_optimize

__all__ = [
    # 基类
    'BaseStrategy',
    # 趋势跟踪
    'TrendFollowingStrategy',
    'DualMAStrategy',
    'BreakoutStrategy',
    'DonchianChannelStrategy',
    # 均值回归
    'MeanReversionStrategy',
    'RSIStrategy',
    'BollingerBandsStrategy',
    'CointegrationStrategy',
    # 套利
    'ArbitrageStrategy',
    'SpreadArbitrageStrategy',
    'CrossVarietyArbitrageStrategy',
    'FuturesSpotArbitrageStrategy',
    # 优化
    'StrategyOptimizer',
    'OptimizationResult',
    'OptunaOptimizer',
    'quick_optimize',
]
