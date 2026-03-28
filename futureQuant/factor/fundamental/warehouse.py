"""
仓单因子模块

包含仓单变化、仓单压力等基本面因子
"""

import pandas as pd
import numpy as np
from typing import Optional

from ...core.base import Factor


class WarehouseReceiptFactor(Factor):
    """
    仓单因子
    
    计算仓单数量的变化，反映可交割货物数量的变化趋势
    
    仓单变化 = 当期仓单 - 上期仓单
    
    Parameters
    ----------
    receipt_col : str
        仓单数据列名，默认 'receipt'
    period : int
        计算周期，默认 1
    method : str
        计算方法，可选 'change'（变化量）或 'rate'（变化率），默认 'change'
    """
    
    def __init__(self, name: Optional[str] = None,
                 receipt_col: str = 'receipt',
                 period: int = 1,
                 method: str = 'change',
                 **params):
        super().__init__(name, **params)
        self.receipt_col = receipt_col
        self.period = period
        self.method = method
        
        if method not in ['change', 'rate']:
            raise ValueError("method 必须是 'change' 或 'rate'")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算仓单变化因子
        
        Args:
            data: 包含仓单数据的数据框
            
        Returns:
            仓单变化序列
            
        Raises:
            ValueError: 当必需的列不存在时
        """
        # 检查必需的列
        if self.receipt_col not in data.columns:
            raise ValueError(f"数据中缺少必需的列: {self.receipt_col}")
        
        receipt = data[self.receipt_col]
        
        if self.method == 'change':
            # 仓单变化量
            receipt_change = receipt.diff(self.period)
        else:
            # 仓单变化率（百分比）
            receipt_shifted = receipt.shift(self.period)
            receipt_change = (receipt - receipt_shifted) / receipt_shifted * 100
        
        receipt_change.name = self.name
        return receipt_change


class WarehousePressureFactor(Factor):
    """
    仓单压力因子
    
    通过仓单数量与历史水平的比较，衡量交割压力
    
    计算方法：
    1. zscore: 当前仓单在历史分布中的Z分数
    2. percentile: 当前仓单在历史分布中的百分位数
    3. ratio: 当前仓单 / 历史平均仓单
    
    Parameters
    ----------
    receipt_col : str
        仓单数据列名，默认 'receipt'
    window : int
        历史观察窗口，默认 252（一年交易日）
    method : str
        计算方法，可选 'zscore'、'percentile' 或 'ratio'，默认 'zscore'
    """
    
    def __init__(self, name: Optional[str] = None,
                 receipt_col: str = 'receipt',
                 window: int = 252,
                 method: str = 'zscore',
                 **params):
        super().__init__(name, **params)
        self.receipt_col = receipt_col
        self.window = window
        self.method = method
        
        if method not in ['zscore', 'percentile', 'ratio']:
            raise ValueError("method 必须是 'zscore'、'percentile' 或 'ratio'")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算仓单压力因子
        
        Args:
            data: 包含仓单数据的数据框
            
        Returns:
            仓单压力序列
            
        Raises:
            ValueError: 当必需的列不存在时
        """
        # 检查必需的列
        if self.receipt_col not in data.columns:
            raise ValueError(f"数据中缺少必需的列: {self.receipt_col}")
        
        receipt = data[self.receipt_col]
        
        if self.method == 'zscore':
            # Z分数：偏离历史均值的标准差数
            rolling_mean = receipt.rolling(window=self.window).mean()
            rolling_std = receipt.rolling(window=self.window).std()
            pressure = (receipt - rolling_mean) / rolling_std
            
        elif self.method == 'percentile':
            # 百分位数：当前值在历史分布中的位置
            def rolling_percentile(x):
                if len(x) < self.window:
                    return np.nan
                return (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50
            
            pressure = receipt.rolling(window=self.window).apply(
                rolling_percentile, raw=False
            )
            
        else:  # ratio
            # 比率：当前值与历史均值的比值
            rolling_mean = receipt.rolling(window=self.window).mean()
            pressure = receipt / rolling_mean
        
        pressure.name = self.name
        return pressure
