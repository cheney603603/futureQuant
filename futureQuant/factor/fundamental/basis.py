"""
基差因子模块

包含基差、基差率、期限结构等基本面因子
"""

import pandas as pd
import numpy as np
from typing import Optional

from ...core.base import Factor


class BasisFactor(Factor):
    """
    基差因子
    
    基差 = 现货价格 - 期货价格
    反映现货与期货的价差关系，正值表示现货升水（backwardation），
    负值表示现货贴水（contango）
    
    Parameters
    ----------
    spot_col : str
        现货价格列名，默认 'spot_price'
    close_col : str
        期货收盘价列名，默认 'close'
    """
    
    def __init__(self, name: Optional[str] = None, 
                 spot_col: str = 'spot_price',
                 close_col: str = 'close',
                 **params):
        super().__init__(name, **params)
        self.spot_col = spot_col
        self.close_col = close_col
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算基差因子
        
        Args:
            data: 包含现货价格和期货价格的数据框
            
        Returns:
            基差序列
            
        Raises:
            ValueError: 当必需的列不存在时
        """
        # 检查必需的列
        if self.spot_col not in data.columns:
            raise ValueError(f"数据中缺少必需的列: {self.spot_col}")
        if self.close_col not in data.columns:
            raise ValueError(f"数据中缺少必需的列: {self.close_col}")
        
        # 计算基差
        basis = data[self.spot_col] - data[self.close_col]
        basis.name = self.name
        
        return basis


class BasisRateFactor(Factor):
    """
    基差率因子
    
    基差率 = (现货价格 - 期货价格) / 期货价格 * 100%
    反映基差相对于期货价格的百分比，便于不同品种间的比较
    
    Parameters
    ----------
    spot_col : str
        现货价格列名，默认 'spot_price'
    close_col : str
        期货收盘价列名，默认 'close'
    """
    
    def __init__(self, name: Optional[str] = None,
                 spot_col: str = 'spot_price',
                 close_col: str = 'close',
                 **params):
        super().__init__(name, **params)
        self.spot_col = spot_col
        self.close_col = close_col
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算基差率因子
        
        Args:
            data: 包含现货价格和期货价格的数据框
            
        Returns:
            基差率序列（百分比形式）
            
        Raises:
            ValueError: 当必需的列不存在时
        """
        # 检查必需的列
        if self.spot_col not in data.columns:
            raise ValueError(f"数据中缺少必需的列: {self.spot_col}")
        if self.close_col not in data.columns:
            raise ValueError(f"数据中缺少必需的列: {self.close_col}")
        
        # 计算基差率
        close_price = data[self.close_col]
        basis_rate = (data[self.spot_col] - close_price) / close_price * 100
        basis_rate.name = self.name
        
        return basis_rate


class TermStructureFactor(Factor):
    """
    期限结构因子
    
    通过近月合约和远月合约的价格关系，反映市场供需预期。
    计算方式：(近月价格 - 远月价格) / 远月价格
    
    正值表示backwardation（现货紧张），负值表示contango（供应宽松）
    
    Parameters
    ----------
    near_col : str
        近月合约价格列名，默认 'near_price'
    far_col : str
        远月合约价格列名，默认 'far_price'
    method : str
        计算方法，可选 'spread'（价差）或 'ratio'（比率），默认 'ratio'
    """
    
    def __init__(self, name: Optional[str] = None,
                 near_col: str = 'near_price',
                 far_col: str = 'far_price',
                 method: str = 'ratio',
                 **params):
        super().__init__(name, **params)
        self.near_col = near_col
        self.far_col = far_col
        self.method = method
        
        if method not in ['spread', 'ratio']:
            raise ValueError("method 必须是 'spread' 或 'ratio'")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算期限结构因子
        
        Args:
            data: 包含近月和远月合约价格的数据框
            
        Returns:
            期限结构序列
            
        Raises:
            ValueError: 当必需的列不存在时
        """
        # 检查必需的列
        if self.near_col not in data.columns:
            raise ValueError(f"数据中缺少必需的列: {self.near_col}")
        if self.far_col not in data.columns:
            raise ValueError(f"数据中缺少必需的列: {self.far_col}")
        
        near_price = data[self.near_col]
        far_price = data[self.far_col]
        
        if self.method == 'spread':
            # 价差形式
            term_structure = near_price - far_price
        else:
            # 比率形式（百分比）
            term_structure = (near_price - far_price) / far_price * 100
        
        term_structure.name = self.name
        return term_structure
