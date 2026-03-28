"""
动量因子模块

包含价格动量、RSI、MACD等技术分析因子
"""

import pandas as pd
import numpy as np
from typing import Optional

from ...core.base import Factor


class MomentumFactor(Factor):
    """
    价格动量因子
    
    计算价格在指定周期内的收益率，反映价格变化的速度和方向。
    动量因子是技术分析中最基础的因子之一，常用于趋势跟踪策略。
    
    计算公式：
        Momentum = (Close_t - Close_{t-n}) / Close_{t-n} * 100
        
    其中：
        - Close_t: 当前收盘价
        - Close_{t-n}: n周期前的收盘价
        
    参数：
        period: 计算周期，默认为20
        
    使用示例：
        >>> factor = MomentumFactor(period=20)
        >>> momentum = factor.compute(df)
    """
    
    def __init__(self, name: Optional[str] = None, period: int = 20):
        """
        初始化动量因子
        
        Args:
            name: 因子名称
            period: 计算周期，默认为20
        """
        super().__init__(name=name, period=period)
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算动量因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，至少需要'close'列
            
        Returns:
            动量值序列，索引与data对齐
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        close = data['close']
        momentum = (close - close.shift(self.period)) / close.shift(self.period) * 100
        momentum.name = self.name
        return momentum


class RSIFactor(Factor):
    """
    相对强弱指数 (Relative Strength Index) 因子
    
    RSI是一种动量震荡指标，用于衡量价格变动的速度和幅度。
    取值范围在0-100之间，通常用于判断超买超卖状态。
    
    计算公式：
        RSI = 100 - (100 / (1 + RS))
        RS = 平均上涨幅度 / 平均下跌幅度
        
    参数：
        period: 计算周期，默认为14
        
    常用阈值：
        - RSI > 70: 超买状态，可能回调
        - RSI < 30: 超卖状态，可能反弹
        
    使用示例：
        >>> factor = RSIFactor(period=14)
        >>> rsi = factor.compute(df)
    """
    
    def __init__(self, name: Optional[str] = None, period: int = 14):
        """
        初始化RSI因子
        
        Args:
            name: 因子名称
            period: 计算周期，默认为14
        """
        super().__init__(name=name, period=period)
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算RSI因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，至少需要'close'列
            
        Returns:
            RSI值序列（0-100），索引与data对齐
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        close = data['close']
        delta = close.diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # 计算平均上涨和平均下跌（使用指数移动平均）
        avg_gain = gain.ewm(alpha=1/self.period, min_periods=self.period).mean()
        avg_loss = loss.ewm(alpha=1/self.period, min_periods=self.period).mean()
        
        # 计算RS和RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi.name = self.name
        return rsi


class MACDFactor(Factor):
    """
    指数平滑异同移动平均线 (MACD) 因子
    
    MACD是一种趋势跟踪动量指标，显示两条移动平均线之间的关系。
    由DIF线、DEA线（信号线）和MACD柱状图（DIF-DEA）组成。
    
    计算公式：
        EMA_fast = 快速周期EMA（默认12）
        EMA_slow = 慢速周期EMA（默认26）
        DIF = EMA_fast - EMA_slow
        DEA = DIF的EMA（默认9）
        MACD = (DIF - DEA) * 2
        
    参数：
        fast_period: 快速EMA周期，默认为12
        slow_period: 慢速EMA周期，默认为26
        signal_period: 信号线EMA周期，默认为9
        
    交易信号：
        - DIF上穿DEA（金叉）：买入信号
        - DIF下穿DEA（死叉）：卖出信号
        
    使用示例：
        >>> factor = MACDFactor(fast_period=12, slow_period=26, signal_period=9)
        >>> macd = factor.compute(df)
    """
    
    def __init__(self, 
                 name: Optional[str] = None, 
                 fast_period: int = 12, 
                 slow_period: int = 26, 
                 signal_period: int = 9):
        """
        初始化MACD因子
        
        Args:
            name: 因子名称
            fast_period: 快速EMA周期，默认为12
            slow_period: 慢速EMA周期，默认为26
            signal_period: 信号线EMA周期，默认为9
        """
        super().__init__(name=name, fast_period=fast_period, 
                        slow_period=slow_period, signal_period=signal_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算MACD因子值（返回MACD柱状图值）
        
        Args:
            data: DataFrame，包含OHLCV数据，至少需要'close'列
            
        Returns:
            MACD柱状图值序列，索引与data对齐
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        close = data['close']
        
        # 计算EMA
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()
        
        # 计算DIF和DEA
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=self.signal_period, adjust=False).mean()
        
        # 计算MACD柱状图
        macd = (dif - dea) * 2
        macd.name = self.name
        return macd


class MACDDIFFactor(Factor):
    """
    MACD DIF线因子
    
    返回MACD的DIF线（快线），用于与其他指标组合分析。
    
    使用示例：
        >>> factor = MACDDIFFactor(fast_period=12, slow_period=26)
        >>> dif = factor.compute(df)
    """
    
    def __init__(self, 
                 name: Optional[str] = None, 
                 fast_period: int = 12, 
                 slow_period: int = 26):
        """
        初始化MACD DIF因子
        
        Args:
            name: 因子名称
            fast_period: 快速EMA周期，默认为12
            slow_period: 慢速EMA周期，默认为26
        """
        super().__init__(name=name, fast_period=fast_period, slow_period=slow_period)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算MACD DIF线
        
        Args:
            data: DataFrame，包含OHLCV数据，至少需要'close'列
            
        Returns:
            DIF值序列，索引与data对齐
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        close = data['close']
        ema_fast = close.ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow_period, adjust=False).mean()
        dif = ema_fast - ema_slow
        dif.name = self.name
        return dif


class RateOfChangeFactor(Factor):
    """
    变动率 (Rate of Change, ROC) 因子
    
    ROC衡量价格在n个周期内的变化率，是一个纯动量指标。
    
    计算公式：
        ROC = ((Close_t - Close_{t-n}) / Close_{t-n}) * 100
        
    参数：
        period: 计算周期，默认为12
        
    使用示例：
        >>> factor = RateOfChangeFactor(period=12)
        >>> roc = factor.compute(df)
    """
    
    def __init__(self, name: Optional[str] = None, period: int = 12):
        """
        初始化ROC因子
        
        Args:
            name: 因子名称
            period: 计算周期，默认为12
        """
        super().__init__(name=name, period=period)
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算ROC因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，至少需要'close'列
            
        Returns:
            ROC值序列，索引与data对齐
        """
        if 'close' not in data.columns:
            raise ValueError("Data must contain 'close' column")
        
        close = data['close']
        roc = ((close - close.shift(self.period)) / close.shift(self.period)) * 100
        roc.name = self.name
        return roc
