"""
数据清洗模块

负责数据的清洗、去极值、缺失值处理等。
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime

from ..core.logger import get_logger

logger = get_logger('data.data_cleaner')


class DataCleaner:
    """
    数据清洗器
    
    负责数据的清洗、去极值、缺失值处理等。
    """
    
    def __init__(self, sigma: float = 3.0):
        """
        初始化数据清洗器
        
        Args:
            sigma: 去极值的标准差倍数 (默认 3-sigma)
        """
        self.sigma = sigma
        self.logger = logger
    
    def remove_outliers(
        self,
        data: pd.Series,
        method: str = 'zscore',
    ) -> pd.Series:
        """
        去除极值
        
        Args:
            data: 数据序列
            method: 去极值方法 ('zscore' 或 'iqr')
            
        Returns:
            清洗后的数据序列
        """
        try:
            if method == 'zscore':
                # Z-score 方法
                mean = data.mean()
                std = data.std()
                z_scores = np.abs((data - mean) / std)
                cleaned = data[z_scores <= self.sigma].copy()
                
                # 用均值填充被去除的极值
                mask = z_scores > self.sigma
                data_cleaned = data.copy()
                data_cleaned[mask] = mean
                
                return data_cleaned
                
            elif method == 'iqr':
                # IQR 方法
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                data_cleaned = data.copy()
                data_cleaned[(data < lower_bound) | (data > upper_bound)] = data.median()
                
                return data_cleaned
            else:
                self.logger.warning(f"Unknown method: {method}, returning original data")
                return data
                
        except Exception as e:
            self.logger.error(f"Failed to remove outliers: {e}")
            return data
    
    def fill_missing_values(
        self,
        data: pd.Series,
        method: str = 'ffill',
        limit: Optional[int] = None,
    ) -> pd.Series:
        """
        填充缺失值
        
        Args:
            data: 数据序列
            method: 填充方法 ('ffill', 'bfill', 'interpolate', 'mean')
            limit: 最大填充数量
            
        Returns:
            填充后的数据序列
        """
        try:
            data_filled = data.copy()
            
            if method == 'ffill':
                # 前向填充
                data_filled = data_filled.ffill(limit=limit)
                # 后向填充剩余的 NaN
                data_filled = data_filled.bfill()
                
            elif method == 'bfill':
                # 后向填充
                data_filled = data_filled.bfill(limit=limit)
                # 前向填充剩余的 NaN
                data_filled = data_filled.ffill()
                
            elif method == 'interpolate':
                # 线性插值
                data_filled = data_filled.interpolate(method='linear', limit=limit)
                # 填充剩余的 NaN
                data_filled = data_filled.ffill().bfill()
                
            elif method == 'mean':
                # 用均值填充
                mean_value = data_filled.mean()
                data_filled = data_filled.fillna(mean_value)
            
            return data_filled
            
        except Exception as e:
            self.logger.error(f"Failed to fill missing values: {e}")
            return data
    
    def smooth_data(
        self,
        data: pd.Series,
        window: int = 5,
        method: str = 'rolling_mean',
    ) -> pd.Series:
        """
        平滑数据
        
        Args:
            data: 数据序列
            window: 窗口大小
            method: 平滑方法 ('rolling_mean', 'ewm', 'savgol')
            
        Returns:
            平滑后的数据序列
        """
        try:
            if method == 'rolling_mean':
                # 滚动平均
                smoothed = data.rolling(window=window, center=True).mean()
                # 填充边界的 NaN
                smoothed = smoothed.bfill().ffill()
                
            elif method == 'ewm':
                # 指数加权平均
                smoothed = data.ewm(span=window).mean()
                
            elif method == 'savgol':
                # Savitzky-Golay 滤波
                from scipy.signal import savgol_filter
                if len(data) >= window:
                    smoothed = pd.Series(
                        savgol_filter(data.fillna(data.mean()), window, 3),
                        index=data.index
                    )
                else:
                    smoothed = data
            else:
                self.logger.warning(f"Unknown method: {method}, returning original data")
                smoothed = data
            
            return smoothed
            
        except Exception as e:
            self.logger.error(f"Failed to smooth data: {e}")
            return data
    
    def clean_data(
        self,
        data: pd.DataFrame,
        outlier_method: str = 'zscore',
        missing_method: str = 'ffill',
        smooth_method: Optional[str] = None,
        smooth_window: int = 5,
    ) -> Tuple[pd.DataFrame, Dict[str, any]]:
        """
        完整的数据清洗流程
        
        Args:
            data: 输入数据框
            outlier_method: 去极值方法
            missing_method: 缺失值填充方法
            smooth_method: 平滑方法 (可选)
            smooth_window: 平滑窗口大小
            
        Returns:
            清洗后的数据框和清洗报告
        """
        try:
            report = {
                'original_shape': data.shape,
                'missing_values_before': data.isnull().sum().to_dict(),
                'outliers_removed': {},
                'timestamp': datetime.now().isoformat(),
            }
            
            data_cleaned = data.copy()
            
            # 1. 去除极值
            for col in data_cleaned.columns:
                if data_cleaned[col].dtype in ['float64', 'int64']:
                    original_count = len(data_cleaned[col])
                    data_cleaned[col] = self.remove_outliers(data_cleaned[col], outlier_method)
                    outliers = original_count - len(data_cleaned[col].dropna())
                    report['outliers_removed'][col] = outliers
            
            # 2. 填充缺失值
            for col in data_cleaned.columns:
                if data_cleaned[col].dtype in ['float64', 'int64']:
                    data_cleaned[col] = self.fill_missing_values(data_cleaned[col], missing_method)
            
            # 3. 平滑数据 (可选)
            if smooth_method:
                for col in data_cleaned.columns:
                    if data_cleaned[col].dtype in ['float64', 'int64']:
                        data_cleaned[col] = self.smooth_data(data_cleaned[col], smooth_window, smooth_method)
            
            report['missing_values_after'] = data_cleaned.isnull().sum().to_dict()
            report['final_shape'] = data_cleaned.shape
            
            self.logger.info(f"Data cleaning completed: {report}")
            
            return data_cleaned, report
            
        except Exception as e:
            self.logger.error(f"Failed to clean data: {e}")
            return data, {'error': str(e)}
