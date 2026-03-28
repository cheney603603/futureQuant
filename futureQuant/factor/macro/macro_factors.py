"""
宏观因子实现

宏观因子通常作为外部数据传入，compute 方法主要进行：
1. 标准化处理（Z-score、Min-Max等）
2. 与期货价格的交互计算（相关性、回归残差等）
3. 动量/变化率计算
"""

from typing import Optional, Literal
import pandas as pd
import numpy as np
from ...core.base import Factor


class DollarIndexFactor(Factor):
    """
    美元指数影响因子
    
    美元指数(DXY)与大宗商品价格通常呈负相关关系。
    该因子计算美元指数的变化率或标准化值，用于衡量美元对商品期货的影响。
    
    Parameters
    ----------
    name : str, optional
        因子名称
    method : str, default 'change_rate'
        计算方法: 'change_rate'(变化率), 'zscore'(Z-score标准化), 
                'momentum'(动量), 'correlation'(与价格相关性)
    window : int, default 20
        计算窗口期
    external_data_col : str, default 'dxy'
        外部数据中美元指数的列名
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        method: Literal['change_rate', 'zscore', 'momentum', 'correlation'] = 'change_rate',
        window: int = 20,
        external_data_col: str = 'dxy',
        **params
    ):
        super().__init__(name, **params)
        self.method = method
        self.window = window
        self.external_data_col = external_data_col
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算美元指数因子
        
        Args:
            data: DataFrame，包含OHLCV数据和外部宏观数据
                  需要包含 external_data_col 指定的列（美元指数数据）
        
        Returns:
            因子值序列，索引与data对齐
        """
        if self.external_data_col not in data.columns:
            # 如果没有外部数据，返回NaN序列
            return pd.Series(np.nan, index=data.index, name=self.name)
        
        dxy = data[self.external_data_col]
        
        if self.method == 'change_rate':
            # 美元指数变化率
            result = dxy.pct_change(self.window)
        elif self.method == 'zscore':
            # Z-score标准化
            result = (dxy - dxy.rolling(self.window).mean()) / dxy.rolling(self.window).std()
        elif self.method == 'momentum':
            # 动量 = 当前值 - N期前的值
            result = dxy - dxy.shift(self.window)
        elif self.method == 'correlation':
            # 美元指数与期货价格的相关性
            if 'close' not in data.columns:
                return pd.Series(np.nan, index=data.index, name=self.name)
            result = dxy.rolling(self.window).corr(data['close'])
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        result.name = self.name
        return result


class InterestRateFactor(Factor):
    """
    利率因子
    
    利率变化影响资金成本和持有成本，对期货价格有重要影响。
    支持多种利率指标：国债收益率、银行间利率等。
    
    Parameters
    ----------
    name : str, optional
        因子名称
    method : str, default 'change'
        计算方法: 'change'(利率变化), 'spread'(期限利差),
                'zscore'(Z-score标准化), 'carry'(持有成本影响)
    window : int, default 20
        计算窗口期
    external_data_col : str, default 'interest_rate'
        外部数据中利率的列名
    tenor_short : str, optional
        短期利率列名（用于计算期限利差）
    tenor_long : str, optional
        长期利率列名（用于计算期限利差）
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        method: Literal['change', 'spread', 'zscore', 'carry'] = 'change',
        window: int = 20,
        external_data_col: str = 'interest_rate',
        tenor_short: Optional[str] = None,
        tenor_long: Optional[str] = None,
        **params
    ):
        super().__init__(name, **params)
        self.method = method
        self.window = window
        self.external_data_col = external_data_col
        self.tenor_short = tenor_short
        self.tenor_long = tenor_long
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算利率因子
        
        Args:
            data: DataFrame，包含OHLCV数据和外部宏观数据
        
        Returns:
            因子值序列，索引与data对齐
        """
        if self.method == 'spread':
            # 计算期限利差
            if self.tenor_short is None or self.tenor_long is None:
                return pd.Series(np.nan, index=data.index, name=self.name)
            if self.tenor_short not in data.columns or self.tenor_long not in data.columns:
                return pd.Series(np.nan, index=data.index, name=self.name)
            result = data[self.tenor_long] - data[self.tenor_short]
        else:
            if self.external_data_col not in data.columns:
                return pd.Series(np.nan, index=data.index, name=self.name)
            
            rate = data[self.external_data_col]
            
            if self.method == 'change':
                # 利率变化（一阶差分）
                result = rate.diff(self.window)
            elif self.method == 'zscore':
                # Z-score标准化
                result = (rate - rate.rolling(self.window).mean()) / rate.rolling(self.window).std()
            elif self.method == 'carry':
                # 持有成本影响 = 利率 * 时间（年化）
                # 假设期货持有期为3个月
                result = rate * 0.25
            else:
                raise ValueError(f"Unknown method: {self.method}")
        
        result.name = self.name
        return result


class CommodityIndexFactor(Factor):
    """
    商品指数因子
    
    使用CRB指数、彭博商品指数等作为商品市场整体走势的代理变量。
    计算商品指数与特定期货品种的相关性或相对强弱。
    
    Parameters
    ----------
    name : str, optional
        因子名称
    method : str, default 'beta'
        计算方法: 'beta'(Beta系数), 'relative_strength'(相对强弱),
                'correlation'(相关性), 'momentum'(动量)
    window : int, default 20
        计算窗口期
    external_data_col : str, default 'commodity_index'
        外部数据中商品指数的列名
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        method: Literal['beta', 'relative_strength', 'correlation', 'momentum'] = 'beta',
        window: int = 20,
        external_data_col: str = 'commodity_index',
        **params
    ):
        super().__init__(name, **params)
        self.method = method
        self.window = window
        self.external_data_col = external_data_col
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算商品指数因子
        
        Args:
            data: DataFrame，包含OHLCV数据和外部宏观数据
        
        Returns:
            因子值序列，索引与data对齐
        """
        if self.external_data_col not in data.columns:
            return pd.Series(np.nan, index=data.index, name=self.name)
        
        if 'close' not in data.columns:
            return pd.Series(np.nan, index=data.index, name=self.name)
        
        commodity_idx = data[self.external_data_col]
        price = data['close']
        
        if self.method == 'beta':
            # Beta系数 = Cov(品种收益, 指数收益) / Var(指数收益)
            commodity_returns = commodity_idx.pct_change()
            price_returns = price.pct_change()
            
            cov = price_returns.rolling(self.window).cov(commodity_returns)
            var = commodity_returns.rolling(self.window).var()
            result = cov / var
        elif self.method == 'relative_strength':
            # 相对强弱 = 品种收益 - 指数收益
            commodity_returns = commodity_idx.pct_change(self.window)
            price_returns = price.pct_change(self.window)
            result = price_returns - commodity_returns
        elif self.method == 'correlation':
            # 滚动相关性
            commodity_returns = commodity_idx.pct_change()
            price_returns = price.pct_change()
            result = price_returns.rolling(self.window).corr(commodity_returns)
        elif self.method == 'momentum':
            # 商品指数动量
            result = commodity_idx.pct_change(self.window)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        result.name = self.name
        return result


class InflationExpectationFactor(Factor):
    """
    通胀预期因子
    
    基于通胀预期数据（如TIPS利差、通胀互换利率等）计算通胀预期变化。
    通胀预期上升通常利好大宗商品。
    
    Parameters
    ----------
    name : str, optional
        因子名称
    method : str, default 'change'
        计算方法: 'change'(通胀预期变化), 'zscore'(Z-score标准化),
                'momentum'(动量), 'regime'(通胀区间)
    window : int, default 20
        计算窗口期
    external_data_col : str, default 'inflation_expectation'
        外部数据中通胀预期的列名
    threshold_high : float, default 0.025
        高通胀阈值（用于regime方法）
    threshold_low : float, default 0.015
        低通胀阈值（用于regime方法）
    """
    
    def __init__(
        self,
        name: Optional[str] = None,
        method: Literal['change', 'zscore', 'momentum', 'regime'] = 'change',
        window: int = 20,
        external_data_col: str = 'inflation_expectation',
        threshold_high: float = 0.025,
        threshold_low: float = 0.015,
        **params
    ):
        super().__init__(name, **params)
        self.method = method
        self.window = window
        self.external_data_col = external_data_col
        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算通胀预期因子
        
        Args:
            data: DataFrame，包含OHLCV数据和外部宏观数据
        
        Returns:
            因子值序列，索引与data对齐
        """
        if self.external_data_col not in data.columns:
            return pd.Series(np.nan, index=data.index, name=self.name)
        
        inflation = data[self.external_data_col]
        
        if self.method == 'change':
            # 通胀预期变化
            result = inflation.diff(self.window)
        elif self.method == 'zscore':
            # Z-score标准化
            result = (inflation - inflation.rolling(self.window).mean()) / inflation.rolling(self.window).std()
        elif self.method == 'momentum':
            # 动量
            result = inflation - inflation.shift(self.window)
        elif self.method == 'regime':
            # 通胀区间分类
            # 1: 高通胀, 0: 温和通胀, -1: 低通胀
            result = pd.Series(0, index=data.index, dtype=float)
            result[inflation > self.threshold_high] = 1
            result[inflation < self.threshold_low] = -1
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        result.name = self.name
        return result
