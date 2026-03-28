"""
异常检测器 (Anomaly Detector)

检测数据异常模式：
- Z-Score 方法
- IQR 方法
- 移动平均偏离方法
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from ...core.logger import get_logger

logger = get_logger('agent.data.anomaly_detector')


class AnomalyDetector:
    """数据异常检测器"""

    def __init__(
        self,
        method: str = 'zscore',
        threshold: float = 3.0,
        window: int = 20,
    ) -> None:
        """
        Args:
            method: 检测方法，'zscore' / 'iqr' / 'ma_deviation'
            threshold: 异常阈值
            window: 滚动窗口大小
        """
        self.method = method
        self.threshold = threshold
        self.window = window

    def detect(
        self,
        data: pd.Series,
    ) -> pd.Series:
        """
        检测异常值

        Args:
            data: 输入数据

        Returns:
            布尔序列，True 表示异常
        """
        if self.method == 'zscore':
            return self.zscore_method(data)
        elif self.method == 'iqr':
            return self.iqr_method(data)
        elif self.method == 'ma_deviation':
            return self.ma_deviation_method(data)
        return pd.Series(False, index=data.index)

    def zscore_method(self, data: pd.Series) -> pd.Series:
        """Z-Score 方法"""
        rolling_mean = data.rolling(self.window, min_periods=5).mean()
        rolling_std = data.rolling(self.window, min_periods=5).std()
        z_score = ((data - rolling_mean) / (rolling_std + 1e-8)).abs()
        return z_score > self.threshold

    def iqr_method(self, data: pd.Series) -> pd.Series:
        """IQR 方法"""
        rolling_q1 = data.rolling(self.window, min_periods=5).quantile(0.25)
        rolling_q3 = data.rolling(self.window, min_periods=5).quantile(0.75)
        iqr = rolling_q3 - rolling_q1
        lower = rolling_q1 - 1.5 * iqr
        upper = rolling_q3 + 1.5 * iqr
        return (data < lower) | (data > upper)

    def ma_deviation_method(self, data: pd.Series) -> pd.Series:
        """移动平均偏离方法"""
        ma = data.rolling(self.window, min_periods=5).mean()
        mad = (data - ma).abs()
        rolling_mad = mad.rolling(self.window, min_periods=5).median()
        threshold_value = rolling_mad * self.threshold
        return mad > threshold_value

    def get_statistics(self, data: pd.Series) -> Dict[str, float]:
        """获取异常统计"""
        anomalies = self.detect(data)
        n_anomalies = anomalies.sum()
        return {
            'total_points': len(data),
            'n_anomalies': int(n_anomalies),
            'anomaly_ratio': float(n_anomalies / len(data)),
            'mean': float(data.mean()),
            'std': float(data.std()),
            'min': float(data.min()),
            'max': float(data.max()),
        }
