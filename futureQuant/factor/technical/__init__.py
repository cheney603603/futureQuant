"""
技术因子模块

提供各种技术分析因子，包括动量、波动率、成交量等类别。
所有因子类均继承自 core.base.Factor 基类。

使用示例：
    >>> from futureQuant.factor.technical import MomentumFactor, RSIFactor, ATRFactor
    >>> 
    >>> # 计算动量因子
    >>> momentum = MomentumFactor(period=20)
    >>> values = momentum.compute(df)
    >>> 
    >>> # 计算RSI因子
    >>> rsi = RSIFactor(period=14)
    >>> rsi_values = rsi.compute(df)
"""

# 动量因子
from .momentum import (
    MomentumFactor,
    RSIFactor,
    MACDFactor,
    MACDDIFFactor,
    RateOfChangeFactor,
)

# 波动率因子
from .volatility import (
    ATRFactor,
    VolatilityFactor,
    BollingerBandWidthFactor,
    TrueRangeFactor,
    ParkisonVolatilityFactor,
)

# 成交量因子
from .volume import (
    OBVFactor,
    VolumeRatioFactor,
    VolumeMAFactor,
    VWAPFactor,
    MFI_Factor,
    VolumePriceTrendFactor,
)

# 导出所有因子类
__all__ = [
    # 动量因子
    'MomentumFactor',
    'RSIFactor',
    'MACDFactor',
    'MACDDIFFactor',
    'RateOfChangeFactor',
    
    # 波动率因子
    'ATRFactor',
    'VolatilityFactor',
    'BollingerBandWidthFactor',
    'TrueRangeFactor',
    'ParkisonVolatilityFactor',
    
    # 成交量因子
    'OBVFactor',
    'VolumeRatioFactor',
    'VolumeMAFactor',
    'VWAPFactor',
    'MFI_Factor',
    'VolumePriceTrendFactor',
]
