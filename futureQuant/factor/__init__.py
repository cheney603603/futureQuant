"""
factor 模块 - 因子库

包含：
- engine: 因子计算引擎
- evaluator: 因子有效性评估（IC/ICIR）
- technical: 技术因子（动量、波动率、成交量）
- fundamental: 基本面因子（基差、库存、仓单）
- macro: 宏观因子

使用示例：
    >>> from futureQuant.factor import FactorEngine, MomentumFactor, RSIFactor
    >>> engine = FactorEngine()
    >>> engine.register(MomentumFactor(window=20))
    >>> engine.register(RSIFactor(window=14))
    >>> factor_df = engine.compute_all(price_data)
"""

from .engine import FactorEngine
from .evaluator import FactorEvaluator
from ..core.base import Factor

# 技术因子
try:
    from .technical import (
        # 动量因子
        MomentumFactor,
        RSIFactor,
        MACDFactor,
        MACDDIFFactor,
        RateOfChangeFactor,
        # 波动率因子
        ATRFactor,
        VolatilityFactor,
        BollingerBandWidthFactor,
        TrueRangeFactor,
        ParkisonVolatilityFactor,
        # 成交量因子
        OBVFactor,
        VolumeRatioFactor,
        VolumeMAFactor,
        VWAPFactor,
        MFI_Factor,
        VolumePriceTrendFactor,
    )
    _technical_imports = [
        'MomentumFactor', 'RSIFactor', 'MACDFactor', 'MACDDIFFactor', 'RateOfChangeFactor',
        'ATRFactor', 'VolatilityFactor', 'BollingerBandWidthFactor', 'TrueRangeFactor', 'ParkisonVolatilityFactor',
        'OBVFactor', 'VolumeRatioFactor', 'VolumeMAFactor', 'VWAPFactor', 'MFI_Factor', 'VolumePriceTrendFactor',
    ]
except ImportError:
    _technical_imports = []

# 基本面因子
try:
    from .fundamental import (
        # 基差因子
        BasisFactor,
        BasisRateFactor,
        TermStructureFactor,
        # 库存因子
        InventoryChangeFactor,
        InventoryYoYFactor,
        # 仓单因子
        WarehouseReceiptFactor,
        WarehousePressureFactor,
    )
    _fundamental_imports = [
        'BasisFactor', 'BasisRateFactor', 'TermStructureFactor',
        'InventoryChangeFactor', 'InventoryYoYFactor',
        'WarehouseReceiptFactor', 'WarehousePressureFactor',
    ]
except ImportError:
    _fundamental_imports = []

# 宏观因子
try:
    from .macro import (
        DollarIndexFactor,
        InterestRateFactor,
        CommodityIndexFactor,
        InflationExpectationFactor,
    )
    _macro_imports = [
        'DollarIndexFactor', 'InterestRateFactor', 
        'CommodityIndexFactor', 'InflationExpectationFactor',
    ]
except ImportError:
    _macro_imports = []

__all__ = [
    # 核心类
    'FactorEngine',
    'FactorEvaluator',
    'Factor',
    # 技术因子
    *_technical_imports,
    # 基本面因子
    *_fundamental_imports,
    # 宏观因子
    *_macro_imports,
]
