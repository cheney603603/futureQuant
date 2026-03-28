"""
数据验证器 (Data Validator)

数据质量验证：
- 缺失值检查
- 重复值检查
- 数值范围检查
- 时间连续性检查
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ...core.logger import get_logger

logger = get_logger('agent.data.data_validator')


class DataValidator:
    """数据质量验证器"""

    def __init__(
        self,
        check_missing: bool = True,
        check_duplicates: bool = True,
        check_range: bool = True,
        check_continuity: bool = True,
    ) -> None:
        """
        Args:
            check_missing: 检查缺失值
            check_duplicates: 检查重复值
            check_range: 检查数值范围
            check_continuity: 检查时间连续性
        """
        self.check_missing = check_missing
        self.check_duplicates = check_duplicates
        self.check_range = check_range
        self.check_continuity = check_continuity

    def validate(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        验证数据质量

        Args:
            data: 待验证的 DataFrame

        Returns:
            验证报告字典
        """
        report: Dict[str, Any] = {
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {},
        }

        if self.check_missing:
            miss = self.check_missing_values(data)
            if miss['has_issues']:
                report['issues'].append(miss)
                report['is_valid'] = False

        if self.check_duplicates:
            dup = self.check_duplicates(data)
            if dup['has_issues']:
                report['issues'].append(dup)
                report['is_valid'] = False

        if self.check_range:
            rng = self.check_range(data)
            if rng['has_issues']:
                report['warnings'].append(rng)

        if self.check_continuity:
            cont = self.check_continuity(data)
            if cont['has_issues']:
                report['warnings'].append(cont)

        report['statistics'] = self.get_statistics(data)
        return report

    def check_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检查缺失值"""
        miss_count = data.isnull().sum()
        miss_pct = (miss_count / len(data) * 100).round(2)
        total_missing = miss_count.sum()

        result: Dict[str, Any] = {
            'type': 'missing_values',
            'has_issues': total_missing > 0,
            'details': {},
        }

        for col in data.columns:
            if miss_count[col] > 0:
                result['details'][col] = {
                    'count': int(miss_count[col]),
                    'percentage': float(miss_pct[col]),
                }

        return result

    def check_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检查重复值"""
        dup_rows = data.duplicated().sum()
        return {
            'type': 'duplicates',
            'has_issues': dup_rows > 0,
            'count': int(dup_rows),
        }

    def check_range(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检查数值范围异常"""
        issues: Dict[str, Dict[str, float]] = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            val_min, val_max = data[col].min(), data[col].max()
            if val_min < -1e10 or val_max > 1e10:
                issues[col] = {'min': float(val_min), 'max': float(val_max)}

        return {
            'type': 'range_anomaly',
            'has_issues': len(issues) > 0,
            'details': issues,
        }

    def check_continuity(self, data: pd.DataFrame) -> Dict[str, Any]:
        """检查时间连续性"""
        if not isinstance(data.index, pd.DatetimeIndex):
            return {'type': 'continuity', 'has_issues': False}

        diff = data.index.to_series().diff().dropna()
        expected = pd.Timedelta(days=1)
        gaps = diff[diff != expected]

        return {
            'type': 'continuity',
            'has_issues': len(gaps) > 0,
            'n_gaps': int(len(gaps)),
            'largest_gap': float(gaps.max().days) if len(gaps) > 0 else 0,
        }

    def get_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """获取数据统计信息"""
        stats_dict: Dict[str, Dict[str, float]] = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            stats_dict[col] = {
                'mean': float(data[col].mean()),
                'std': float(data[col].std()),
                'min': float(data[col].min()),
                'max': float(data[col].max()),
                'median': float(data[col].median()),
            }

        return stats_dict
