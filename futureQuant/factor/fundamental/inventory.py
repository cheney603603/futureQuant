"""
库存因子模块

包含库存变化、库存同比等基本面因子
"""

import pandas as pd
import numpy as np
from typing import Optional

from ...core.base import Factor


class InventoryChangeFactor(Factor):
    """
    库存变化因子
    
    计算库存的环比变化率，反映短期供需变化趋势
    
    库存变化率 = (当期库存 - 上期库存) / 上期库存 * 100%
    
    Parameters
    ----------
    inventory_col : str
        库存数据列名，默认 'inventory'
    period : int
        计算周期，默认 1（周环比）
    method : str
        计算方法，可选 'change'（变化量）或 'rate'（变化率），默认 'rate'
    """
    
    def __init__(self, name: Optional[str] = None,
                 inventory_col: str = 'inventory',
                 period: int = 1,
                 method: str = 'rate',
                 **params):
        super().__init__(name, **params)
        self.inventory_col = inventory_col
        self.period = period
        self.method = method
        
        if method not in ['change', 'rate']:
            raise ValueError("method 必须是 'change' 或 'rate'")
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算库存变化因子
        
        Args:
            data: 包含库存数据的数据框
            
        Returns:
            库存变化序列
            
        Raises:
            ValueError: 当必需的列不存在时
        """
        # 检查必需的列
        if self.inventory_col not in data.columns:
            raise ValueError(f"数据中缺少必需的列: {self.inventory_col}")
        
        inventory = data[self.inventory_col]
        
        if self.method == 'change':
            # 库存变化量
            inventory_change = inventory.diff(self.period)
        else:
            # 库存变化率（百分比）
            inventory_shifted = inventory.shift(self.period)
            inventory_change = (inventory - inventory_shifted) / inventory_shifted * 100
        
        inventory_change.name = self.name
        return inventory_change


class InventoryYoYFactor(Factor):
    """
    库存同比因子
    
    计算库存的同比变化率，反映中长期供需格局变化
    
    库存同比 = (当期库存 - 去年同期库存) / 去年同期库存 * 100%
    
    Parameters
    ----------
    inventory_col : str
        库存数据列名，默认 'inventory'
    periods_per_year : int
        每年的数据周期数，默认 52（周数据）
                         可选 12（月数据）、252（日数据）
    """
    
    def __init__(self, name: Optional[str] = None,
                 inventory_col: str = 'inventory',
                 periods_per_year: int = 52,
                 **params):
        super().__init__(name, **params)
        self.inventory_col = inventory_col
        self.periods_per_year = periods_per_year
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        """
        计算库存同比因子
        
        Args:
            data: 包含库存数据的数据框
            
        Returns:
            库存同比序列（百分比形式）
            
        Raises:
            ValueError: 当必需的列不存在时
        """
        # 检查必需的列
        if self.inventory_col not in data.columns:
            raise ValueError(f"数据中缺少必需的列: {self.inventory_col}")
        
        inventory = data[self.inventory_col]
        
        # 计算同比
        inventory_yoy = (inventory - inventory.shift(self.periods_per_year)) / \
                        inventory.shift(self.periods_per_year) * 100
        
        inventory_yoy.name = self.name
        return inventory_yoy
