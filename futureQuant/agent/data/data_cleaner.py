"""
数据清洗器 (Data Cleaner)

数据清洗工具：
- 去极值（Z-Score / IQR / 百分位）
- 缺失值填充（ffill / bfill /  interpolation）
- 平滑处理（移动平均 / 指数移动平均）
- 行业中性化
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

from ...core.logger import get_logger

logger = get_logger('agent.data.data_cleaner')

DEFAULT_FILL_METHOD = 'ffill'


@dataclass
class CleaningReport:
    """数据清洗报告"""
    cleaning_methods: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class DataCleaner:
    """数据清洗器"""

    OUTLIER_METHODS = ('zscore', 'iqr', 'percentile', 'mad')
    FILL_METHODS = ('ffill', 'bfill', 'interpolate', 'mean', 'median', 'zero')
    SMOOTH_METHODS = ('rolling_mean', 'ewm', 'none')

    def __init__(
        self,
        outlier_method: str = 'zscore',
        outlier_threshold: float = 3.0,
        fill_method: str = 'ffill',
        smooth_method: str = 'none',
        smooth_window: int = 5,
    ) -> None:
        """
        Args:
            outlier_method: 去极值方法，'zscore' / 'iqr' / 'percentile' / 'mad'
            outlier_threshold: 去极值阈值
            fill_method: 缺失值填充方法，'ffill' / 'bfill' / 'interpolate' / 'mean' / 'median' / 'zero'
            smooth_method: 平滑方法，'rolling_mean' / 'ewm' / 'none'
            smooth_window: 平滑窗口大小
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.fill_method = fill_method
        self.smooth_method = smooth_method
        self.smooth_window = smooth_window

    def clean_data(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        清洗数据

        Args:
            data: 待清洗的 DataFrame
            columns: 需要清洗的列，None 表示全部

        Returns:
            清洗后的 DataFrame
        """
        result = data.copy()
        cols = columns or result.columns.tolist()

        report = CleaningReport()

        for col in cols:
            if col not in result.columns:
                continue
            if not pd.api.types.is_numeric_dtype(result[col]):
                continue

            # 去极值
            result[col] = self._remove_outliers(result[col])
            if self.outlier_method != 'none':
                report.cleaning_methods.append(f'{col}_outlier_removed')

            # 填充缺失值
            result[col] = self._fill_missing(result[col])
            report.cleaning_methods.append(f'{col}_filled')

            # 平滑
            if self.smooth_method != 'none':
                result[col] = self._smooth(result[col])
                report.cleaning_methods.append(f'{col}_smoothed')

        return result

    def _remove_outliers(self, series: pd.Series) -> pd.Series:
        """去极值"""
        if self.outlier_method == 'none':
            return series

        if self.outlier_method == 'zscore':
            mean = series.mean()
            std = series.std()
            z = ((series - mean) / (std + 1e-8)).abs()
            return series.where(z <= self.outlier_threshold)

        elif self.outlier_method == 'iqr':
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return series.clip(lower, upper)

        elif self.outlier_method == 'percentile':
            lower = series.quantile(self.outlier_threshold / 100)
            upper = series.quantile(1 - self.outlier_threshold / 100)
            return series.clip(lower, upper)

        elif self.outlier_method == 'mad':
            median = series.median()
            mad = (series - median).abs().median()
            modified_z = 0.6745 * (series - median) / (mad + 1e-8)
            return series.where(modified_z.abs() <= self.outlier_threshold)

        return series

    def _fill_missing(self, series: pd.Series) -> pd.Series:
        """填充缺失值"""
        if self.fill_method == 'ffill':
            return series.ffill().bfill()
        elif self.fill_method == 'bfill':
            return series.bfill().ffill()
        elif self.fill_method == 'interpolate':
            return series.interpolate(method='linear').bfill().ffill()
        elif self.fill_method == 'mean':
            return series.fillna(series.mean())
        elif self.fill_method == 'median':
            return series.fillna(series.median())
        elif self.fill_method == 'zero':
            return series.fillna(0.0)
        return series

    def _smooth(self, series: pd.Series) -> pd.Series:
        """平滑处理"""
        if self.smooth_method == 'rolling_mean':
            return series.rolling(self.smooth_window, min_periods=1).mean()
        elif self.smooth_method == 'ewm':
            alpha = 2 / (self.smooth_window + 1)
            return series.ewm(alpha=alpha, adjust=False).mean()
        return series

    def get_report(self) -> CleaningReport:
        """获取清洗报告"""
        return CleaningReport()


def clean_data(
    data: pd.DataFrame,
    outlier_method: str = 'zscore',
    fill_method: str = 'ffill',
    **kwargs,
) -> pd.DataFrame:
    """
    便捷函数：清洗数据

    Args:
        data: 待清洗数据
        outlier_method: 去极值方法
        fill_method: 填充方法
        **kwargs: 其他参数

    Returns:
        清洗后的数据
    """
    cleaner = DataCleaner(
        outlier_method=outlier_method,
        fill_method=fill_method,
        **kwargs,
    )
    return cleaner.clean_data(data)
