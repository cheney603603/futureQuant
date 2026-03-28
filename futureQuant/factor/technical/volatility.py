"""
波动率因子模块

包含ATR、波动率标准差等波动率相关因子
"""

import pandas as pd
import numpy as np
from typing import Optional

from ...core.base import Factor


class ATRFactor(Factor):
    """
    平均真实波幅 (Average True Range) 因子
    
    ATR是衡量市场波动性的技术指标，由J. Welles Wilder开发。
    它考虑了价格跳空的情况，比简单的价格范围更能反映真实波动。
    
    计算公式：
        TR = max(high - low, |high - close_prev|, |low - close_prev|)
        ATR = TR的n周期简单移动平均
        
    参数：
        period: 计算周期，默认为14
        
    应用场景：
        - 设置止损位：止损距离 = n * ATR
        - 仓位管理：波动大时减小仓位
        - 趋势确认：ATR上升表示波动加剧
        
    使用示例：
        >>> factor = ATRFactor(period=14)
        >>> atr = factor.compute(df)
    """
    
    def __init__(self, name: Optional[str] = None, period: int = 14):
        """
        初始化ATR因子
        
        Args:
            name: 因子名称
            period: 计算周期，默认为14
        """
        super().__init__(name=name, period=period)
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算ATR因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，需要'high', 'low', 'close'列
            
        Returns:
            ATR值序列，索引与data对齐
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        # 计算真实波幅 (True Range)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR（使用Wilder的平滑方法，即EMA）
        atr = tr.ewm(alpha=1/self.period, min_periods=self.period).mean()
        atr.name = self.name
        return atr


class VolatilityFactor(Factor):
    """
    波动率因子（收益率标准差）
    
    计算对数收益率的滚动标准差，衡量价格的波动程度。
    是风险管理和量化策略中最常用的波动率度量。
    
    计算公式：
        Returns = ln(Close_t / Close_{t-1})
        Volatility = std(Returns, n) * sqrt(annualization_factor)
        
    参数：
        period: 计算周期，默认为20
        annualize: 是否年化，默认为True
        trading_days: 年交易日数，默认为252
        
    使用示例：
        >>> factor = VolatilityFactor(period=20, annualize=True)
        >>> vol = factor.compute(df)
    """
    
    def __init__(self, 
                 name: Optional[str] = None, 
                 period: int = 20, 
                 annualize: bool = True,
                 trading_days: int = 252):
        """
        初始化波动率因子
        
        Args:
            name: 因子名称
            period: 计算周期，默认为20
            annualize: 是否年化，默认为True
            trading_days: 年交易日数，默认为252
        """
        super().__init__(name=name, period=period, annualize=annualize, 
                        trading_days=trading_days)
        self.period = period
        self.annualize = annualize
        self.trading_days = trading_days
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算波动率因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，至少需要'close'列
            
        Returns:
            波动率值序列，索引与data对齐
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        close = data['close']
        
        # 计算对数收益率
        log_returns = np.log(close / close.shift(1))
        
        # 计算滚动标准差
        volatility = log_returns.rolling(window=self.period).std()
        
        # 年化处理
        if self.annualize:
            volatility = volatility * np.sqrt(self.trading_days)
        
        volatility.name = self.name
        return volatility


class BollingerBandWidthFactor(Factor):
    """
    布林带宽度因子 (Bollinger Band Width)
    
    衡量布林带上下轨之间的距离，反映价格波动性。
    当带宽收窄时，预示着可能的突破；当带宽扩张时，表示波动加剧。
    
    计算公式：
        Middle Band = SMA(Close, n)
        Upper Band = Middle Band + k * std(Close, n)
        Lower Band = Middle Band - k * std(Close, n)
        BB Width = (Upper - Lower) / Middle * 100
        
    参数：
        period: 计算周期，默认为20
        num_std: 标准差倍数，默认为2
        
    使用示例：
        >>> factor = BollingerBandWidthFactor(period=20, num_std=2)
        >>> bb_width = factor.compute(df)
    """
    
    def __init__(self, 
                 name: Optional[str] = None, 
                 period: int = 20, 
                 num_std: float = 2.0):
        """
        初始化布林带宽度因子
        
        Args:
            name: 因子名称
            period: 计算周期，默认为20
            num_std: 标准差倍数，默认为2
        """
        super().__init__(name=name, period=period, num_std=num_std)
        self.period = period
        self.num_std = num_std
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算布林带宽度因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，至少需要'close'列
            
        Returns:
            布林带宽度值序列（百分比），索引与data对齐
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        close = data['close']
        
        # 计算中轨和带宽
        middle = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        
        upper = middle + self.num_std * std
        lower = middle - self.num_std * std
        
        # 计算带宽百分比
        bb_width = (upper - lower) / middle * 100
        bb_width.name = self.name
        return bb_width


class TrueRangeFactor(Factor):
    """
    真实波幅 (True Range) 因子
    
    真实波幅是当日价格波动的真实范围，考虑了跳空情况。
    是ATR的基础组成部分。
    
    计算公式：
        TR = max(high - low, |high - close_prev|, |low - close_prev|)
        
    使用示例：
        >>> factor = TrueRangeFactor()
        >>> tr = factor.compute(df)
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        初始化真实波幅因子
        
        Args:
            name: 因子名称
        """
        super().__init__(name=name)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算真实波幅因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，需要'high', 'low', 'close'列
            
        Returns:
            真实波幅值序列，索引与data对齐
        """
        required_cols = ['high', 'low', 'close']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        high = data['high']
        low = data['low']
        close = data['close']
        
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        tr.name = self.name
        return tr


class ParkisonVolatilityFactor(Factor):
    """
    Parkinson波动率因子
    
    使用日内高低价计算波动率，比收盘价波动率更高效。
    假设价格服从几何布朗运动。
    
    计算公式：
        sigma_p = sqrt(1/(4*n*ln(2)) * sum(ln(high/low)^2))
        
    参数：
        period: 计算周期，默认为20
        annualize: 是否年化，默认为True
        trading_days: 年交易日数，默认为252
        
    使用示例：
        >>> factor = ParkisonVolatilityFactor(period=20)
        >>> park_vol = factor.compute(df)
    """
    
    def __init__(self, 
                 name: Optional[str] = None, 
                 period: int = 20,
                 annualize: bool = True,
                 trading_days: int = 252):
        """
        初始化Parkinson波动率因子
        
        Args:
            name: 因子名称
            period: 计算周期，默认为20
            annualize: 是否年化，默认为True
            trading_days: 年交易日数，默认为252
        """
        super().__init__(name=name, period=period, annualize=annualize,
                        trading_days=trading_days)
        self.period = period
        self.annualize = annualize
        self.trading_days = trading_days
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算Parkinson波动率因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，需要'high', 'low'列
            
        Returns:
            Parkinson波动率值序列，索引与data对齐
        """
        required_cols = ['high', 'low']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        high = data['high']
        low = data['low']
        
        # 计算Parkinson波动率
        log_hl = np.log(high / low)
        parkinson_var = (log_hl ** 2) / (4 * np.log(2))
        parkinson_vol = np.sqrt(parkinson_var.rolling(window=self.period).mean())
        
        # 年化处理
        if self.annualize:
            parkinson_vol = parkinson_vol * np.sqrt(self.trading_days)
        
        parkinson_vol.name = self.name
        return parkinson_vol
