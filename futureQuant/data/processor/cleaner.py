"""
数据清洗模块 - 处理原始数据的异常值、缺失值等
"""

from typing import Optional, List, Dict, Tuple
import pandas as pd
import numpy as np

from ...core.logger import get_logger
from ...core.exceptions import ProcessingError

logger = get_logger('data.processor.cleaner')


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        self.ohlc_cols = ['open', 'high', 'low', 'close']
    
    def clean_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗OHLC数据
        
        检查项：
        1. 价格逻辑：high >= max(open, close, low), low <= min(open, close, high)
        2. 零值检查：价格不能为零或负数
        3. 涨跌停检查：价格变动是否超过限制
        
        Args:
            df: 原始OHLC数据
            
        Returns:
            清洗后的数据
        """
        df = df.copy()
        initial_len = len(df)
        
        # 1. 去除空值
        df = df.dropna(subset=self.ohlc_cols)
        
        # 2. 检查价格逻辑
        invalid_mask = (
            (df['high'] < df[['open', 'close', 'low']].max(axis=1)) |
            (df['low'] > df[['open', 'close', 'high']].min(axis=1)) |
            (df['low'] <= 0) | (df['high'] <= 0)
        )
        
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            logger.warning(f"Found {invalid_count} records with invalid OHLC logic")
            df = df[~invalid_mask]
        
        # 3. 检查成交量
        if 'volume' in df.columns:
            df = df[df['volume'] >= 0]
        
        logger.info(f"Cleaned OHLC data: {initial_len} -> {len(df)} records")
        return df
    
    def handle_missing_values(
        self, 
        df: pd.DataFrame, 
        method: str = 'ffill',
        limit: int = 5
    ) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            df: 输入数据
            method: 填充方法，'ffill'(前向填充), 'bfill'(后向填充), 'interpolate'(插值)
            limit: 最大连续填充天数
            
        Returns:
            填充后的数据
        """
        df = df.copy()
        
        if method == 'ffill':
            df = df.fillna(method='ffill', limit=limit)
        elif method == 'bfill':
            df = df.fillna(method='bfill', limit=limit)
        elif method == 'interpolate':
            df = df.interpolate(method='linear', limit=limit)
        elif method == 'drop':
            df = df.dropna()
        else:
            raise ProcessingError(f"Unknown fill method: {method}")
        
        return df
    
    def remove_outliers(
        self, 
        df: pd.DataFrame, 
        columns: Optional[List[str]] = None,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        去除异常值
        
        Args:
            df: 输入数据
            columns: 需要处理的列，为None时处理所有数值列
            method: 方法，'iqr'(四分位距), 'zscore'(Z分数), 'mad'(中位数绝对偏差)
            threshold: 阈值
            
        Returns:
            清洗后的数据
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
                
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR
                
            elif method == 'zscore':
                mean = df[col].mean()
                std = df[col].std()
                lower = mean - threshold * std
                upper = mean + threshold * std
                
            elif method == 'mad':
                median = df[col].median()
                mad = np.median(np.abs(df[col] - median))
                lower = median - threshold * 1.4826 * mad
                upper = median + threshold * 1.4826 * mad
            else:
                raise ProcessingError(f"Unknown outlier method: {method}")
            
            outlier_mask = (df[col] < lower) | (df[col] > upper)
            outlier_count = outlier_mask.sum()
            
            if outlier_count > 0:
                logger.warning(f"Found {outlier_count} outliers in {col}")
                df.loc[outlier_mask, col] = np.nan
        
        return df
    
    def detect_price_jumps(
        self, 
        df: pd.DataFrame, 
        price_col: str = 'close',
        threshold: float = 0.1
    ) -> pd.DataFrame:
        """
        检测价格跳空（可能的主力合约切换点）
        
        Args:
            df: 价格数据
            price_col: 价格列名
            threshold: 涨跌幅阈值
            
        Returns:
            包含跳空标记的数据
        """
        df = df.copy()
        df['return'] = df[price_col].pct_change()
        df['is_jump'] = np.abs(df['return']) > threshold
        
        jump_count = df['is_jump'].sum()
        if jump_count > 0:
            logger.info(f"Detected {jump_count} price jumps (> {threshold*100:.1f}%)")
        
        return df
    
    def align_to_calendar(
        self, 
        df: pd.DataFrame, 
        calendar_df: pd.DataFrame,
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        对齐到交易日历
        
        Args:
            df: 输入数据
            calendar_df: 交易日历DataFrame
            date_col: 日期列名
            
        Returns:
            对齐后的数据
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # 创建完整的交易日历
        all_dates = calendar_df['date'].values
        
        # 重新索引
        df = df.set_index(date_col).reindex(all_dates).reset_index()
        df = df.rename(columns={'index': date_col})
        
        return df
    
    def resample(
        self, 
        df: pd.DataFrame, 
        rule: str = 'W',
        date_col: str = 'date'
    ) -> pd.DataFrame:
        """
        重采样（日频转周频/月频等）
        
        Args:
            df: 日频数据
            rule: 重采样规则，'W'(周), 'M'(月), 'Q'(季)
            date_col: 日期列名
            
        Returns:
            重采样后的数据
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        
        # OHLCV重采样
        resampled = pd.DataFrame()
        
        if 'open' in df.columns:
            resampled['open'] = df['open'].resample(rule).first()
        if 'high' in df.columns:
            resampled['high'] = df['high'].resample(rule).max()
        if 'low' in df.columns:
            resampled['low'] = df['low'].resample(rule).min()
        if 'close' in df.columns:
            resampled['close'] = df['close'].resample(rule).last()
        if 'volume' in df.columns:
            resampled['volume'] = df['volume'].resample(rule).sum()
        if 'open_interest' in df.columns:
            resampled['open_interest'] = df['open_interest'].resample(rule).last()
        
        resampled = resampled.reset_index()
        resampled = resampled.dropna(subset=['close'])
        
        return resampled
