"""
成交量因子模块

包含OBV、成交量比率等成交量相关因子
"""

import pandas as pd
import numpy as np
from typing import Optional

from ...core.base import Factor


class OBVFactor(Factor):
    """
    能量潮指标 (On Balance Volume) 因子
    
    OBV由Joseph Granville提出，通过累计成交量来预测价格趋势。
    基本理念是：成交量先于价格变动。
    
    计算公式：
        如果 Close_t > Close_{t-1}: OBV_t = OBV_{t-1} + Volume_t
        如果 Close_t < Close_{t-1}: OBV_t = OBV_{t-1} - Volume_t
        如果 Close_t = Close_{t-1}: OBV_t = OBV_{t-1}
        
    交易信号：
        - OBV上升而价格未涨：可能上涨（量价背离）
        - OBV下降而价格未跌：可能下跌
        - OBV与价格同向：确认趋势
        
    使用示例：
        >>> factor = OBVFactor()
        >>> obv = factor.compute(df)
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        初始化OBV因子
        
        Args:
            name: 因子名称
        """
        super().__init__(name=name)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算OBV因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，需要'close'和'volume'列
            
        Returns:
            OBV值序列，索引与data对齐
        """
        required_cols = ['close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        close = data['close']
        volume = data['volume']
        
        # 计算价格变化方向
        price_change = close.diff()
        
        # 根据价格变化方向确定成交量正负
        signed_volume = volume.copy()
        signed_volume[price_change < 0] = -volume[price_change < 0]
        signed_volume[price_change == 0] = 0
        
        # 累计OBV
        obv = signed_volume.cumsum()
        obv.name = self.name
        return obv


class VolumeRatioFactor(Factor):
    """
    成交量比率 (Volume Ratio) 因子
    
    衡量当前成交量相对于历史平均成交量的比率。
    用于识别异常成交量，可能预示着重要价格变动。
    
    计算公式：
        VR = Volume_t / SMA(Volume, n)
        
    参数：
        period: 计算周期，默认为20
        
    解释：
        - VR > 2: 成交量显著放大
        - VR < 0.5: 成交量显著萎缩
        - VR = 1: 成交量正常
        
    使用示例：
        >>> factor = VolumeRatioFactor(period=20)
        >>> vr = factor.compute(df)
    """
    
    def __init__(self, name: Optional[str] = None, period: int = 20):
        """
        初始化成交量比率因子
        
        Args:
            name: 因子名称
            period: 计算周期，默认为20
        """
        super().__init__(name=name, period=period)
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算成交量比率因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，至少需要'volume'列
            
        Returns:
            成交量比率值序列，索引与data对齐
        """
        if 'volume' not in data.columns:
            raise ValueError("Data must contain 'volume' column")
        
        volume = data['volume']
        volume_ma = volume.rolling(window=self.period).mean()
        
        vr = volume / volume_ma
        vr.name = self.name
        return vr


class VolumeMAFactor(Factor):
    """
    成交量移动平均因子
    
    计算成交量的n周期移动平均，用于平滑成交量数据。
    常与价格指标结合使用，判断量价配合情况。
    
    参数：
        period: 计算周期，默认为20
        
    使用示例：
        >>> factor = VolumeMAFactor(period=20)
        >>> vol_ma = factor.compute(df)
    """
    
    def __init__(self, name: Optional[str] = None, period: int = 20):
        """
        初始化成交量移动平均因子
        
        Args:
            name: 因子名称
            period: 计算周期，默认为20
        """
        super().__init__(name=name, period=period)
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算成交量移动平均因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，至少需要'volume'列
            
        Returns:
            成交量移动平均值序列，索引与data对齐
        """
        if 'volume' not in data.columns:
            raise ValueError("Data must contain 'volume' column")
        
        volume = data['volume']
        vol_ma = volume.rolling(window=self.period).mean()
        vol_ma.name = self.name
        return vol_ma


class VWAPFactor(Factor):
    """
    成交量加权平均价 (Volume Weighted Average Price) 因子
    
    VWAP是衡量日内交易价格的重要指标，常用于算法交易执行。
    价格高于VWAP表示强势，低于VWAP表示弱势。
    
    计算公式：
        VWAP = sum(Typical_Price * Volume) / sum(Volume)
        Typical_Price = (High + Low + Close) / 3
        
    参数：
        period: 计算周期，默认为1（日内VWAP）
        
    应用场景：
        - 算法交易：以VWAP为基准执行大额订单
        - 趋势判断：价格在VWAP上方为多头市场
        - 支撑阻力：VWAP常作为动态支撑/阻力位
        
    使用示例：
        >>> factor = VWAPFactor(period=20)
        >>> vwap = factor.compute(df)
    """
    
    def __init__(self, name: Optional[str] = None, period: int = 20):
        """
        初始化VWAP因子
        
        Args:
            name: 因子名称
            period: 计算周期，默认为20
        """
        super().__init__(name=name, period=period)
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算VWAP因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，需要'high', 'low', 'close', 'volume'列
            
        Returns:
            VWAP值序列，索引与data对齐
        """
        required_cols = ['high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 计算典型价格
        typical_price = (high + low + close) / 3
        
        # 计算VWAP
        vwap = (typical_price * volume).rolling(window=self.period).sum() / \
               volume.rolling(window=self.period).sum()
        vwap.name = self.name
        return vwap


class MFI_Factor(Factor):
    """
    资金流量指标 (Money Flow Index) 因子
    
    MFI是结合了价格和成交量的动量指标，常被称为"成交量RSI"。
    取值范围0-100，用于判断超买超卖状态。
    
    计算公式：
        Typical_Price = (High + Low + Close) / 3
        Raw_Money_Flow = Typical_Price * Volume
        Money_Flow_Ratio = sum(Positive_Money_Flow, n) / sum(Negative_Money_Flow, n)
        MFI = 100 - (100 / (1 + Money_Flow_Ratio))
        
    参数：
        period: 计算周期，默认为14
        
    常用阈值：
        - MFI > 80: 超买状态
        - MFI < 20: 超卖状态
        
    使用示例：
        >>> factor = MFI_Factor(period=14)
        >>> mfi = factor.compute(df)
    """
    
    def __init__(self, name: Optional[str] = None, period: int = 14):
        """
        初始化MFI因子
        
        Args:
            name: 因子名称
            period: 计算周期，默认为14
        """
        super().__init__(name=name, period=period)
        self.period = period
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算MFI因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，需要'high', 'low', 'close', 'volume'列
            
        Returns:
            MFI值序列（0-100），索引与data对齐
        """
        required_cols = ['high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data['volume']
        
        # 计算典型价格
        typical_price = (high + low + close) / 3
        
        # 计算原始资金流量
        raw_money_flow = typical_price * volume
        
        # 判断正负资金流量
        typical_price_diff = typical_price.diff()
        positive_flow = raw_money_flow.where(typical_price_diff > 0, 0)
        negative_flow = raw_money_flow.where(typical_price_diff < 0, 0)
        
        # 计算资金流量比率
        positive_sum = positive_flow.rolling(window=self.period).sum()
        negative_sum = negative_flow.rolling(window=self.period).sum()
        
        money_flow_ratio = positive_sum / negative_sum
        
        # 计算MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))
        mfi.name = self.name
        return mfi


class VolumePriceTrendFactor(Factor):
    """
    量价趋势指标 (Volume Price Trend, VPT) 因子
    
    VPT通过累计成交量和收益率的乘积来衡量资金流向。
    与OBV类似，但考虑了价格变化的幅度。
    
    计算公式：
        VPT_t = VPT_{t-1} + Volume_t * (Close_t - Close_{t-1}) / Close_{t-1}
        
    使用示例：
        >>> factor = VolumePriceTrendFactor()
        >>> vpt = factor.compute(df)
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        初始化VPT因子
        
        Args:
            name: 因子名称
        """
        super().__init__(name=name)
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算VPT因子值
        
        Args:
            data: DataFrame，包含OHLCV数据，需要'close'和'volume'列
            
        Returns:
            VPT值序列，索引与data对齐
        """
        required_cols = ['close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
        
        close = data['close']
        volume = data['volume']
        
        # 计算收益率
        returns = close.pct_change()
        
        # 计算VPT
        vpt_change = volume * returns
        vpt = vpt_change.cumsum()
        vpt.name = self.name
        return vpt
