"""
数据验证器

提供数据质量检查功能：
1. 日期范围验证
2. 数据新鲜度检查
3. 价格合理性验证
4. 数据完整性检查

Author: futureQuant Team
Date: 2026-04-19
"""

from datetime import datetime, timedelta
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
import numpy as np

from ..core.logger import get_logger

logger = get_logger('data.validator')


class ValidationError(Exception):
    """数据验证错误"""
    pass


class DataValidator:
    """
    数据验证器
    
    用于验证从各种数据源获取的数据质量。
    """
    
    # 品种的标准价格范围（用于合理性检查）
    PRICE_RANGES: Dict[str, Tuple[float, float]] = {
        # 黑色系
        'RB': (1000, 6000),    # 螺纹钢
        'HC': (1000, 6000),    # 热轧卷板
        'I': (400, 1500),     # 铁矿石
        'J': (800, 3500),     # 焦炭
        'JM': (800, 2500),    # 焦煤
        # 有色金属
        'CU': (40000, 90000),  # 铜
        'AL': (12000, 25000),  # 铝
        'ZN': (15000, 30000),  # 锌
        'NI': (100000, 250000), # 镍
        'AU': (350, 600),     # 黄金
        'AG': (4000, 8000),   # 白银
        # 化工
        'TA': (4000, 8000),   # PTA
        'MA': (1500, 4000),   # 甲醇
        'RU': (10000, 25000),  # 橡胶
        'BU': (2500, 5000),   # 沥青
        # 农产品
        'M': (2500, 5000),     # 豆粕
        'Y': (5500, 10000),    # 豆油
        'C': (2000, 3500),     # 玉米
        'CF': (12000, 20000),  # 棉花
        'SR': (5000, 8000),    # 白糖
        # 玻璃
        'FG': (800, 2500),    # 玻璃
        # 默认范围
        'DEFAULT': (100, 100000),
    }
    
    # 日收益率异常阈值（超过此值视为异常）
    DAILY_RETURN_THRESHOLD = 0.10  # 10%
    
    # 数据最大允许天数
    MAX_DATA_AGE_DAYS = 7  # 7天前的数据视为过时
    
    def __init__(self, variety: Optional[str] = None):
        """
        初始化验证器
        
        Args:
            variety: 品种代码，如 'RB', 'FG' 等
        """
        self.variety = variety
        self.price_range = self.PRICE_RANGES.get(
            variety, 
            self.PRICE_RANGES['DEFAULT']
        )
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
    
    def reset(self):
        """重置验证结果"""
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_price_data(
        self, 
        df: pd.DataFrame,
        strict: bool = False
    ) -> pd.DataFrame:
        """
        验证价格数据
        
        Args:
            df: 价格数据 DataFrame
            strict: 是否严格模式（严格模式下会过滤异常值）
            
        Returns:
            验证后的 DataFrame
            
        Raises:
            ValidationError: 验证失败
        """
        self.reset()
        
        if df is None or df.empty:
            self.validation_errors.append("数据为空")
            raise ValidationError("数据为空")
        
        # 1. 检查必要列
        required_cols = ['date', 'close']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            self.validation_errors.append(f"缺少必要列: {missing_cols}")
            raise ValidationError(f"缺少必要列: {missing_cols}")
        
        # 2. 日期格式检查
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception as e:
            self.validation_errors.append(f"日期格式错误: {e}")
            raise ValidationError(f"日期格式错误: {e}")
        
        # 3. 数据行数检查
        if len(df) < 2:
            self.validation_warnings.append("数据行数过少 (< 2)")
        
        # 4. 价格合理性检查
        close = df['close'].dropna()
        if len(close) == 0:
            self.validation_errors.append("没有有效的收盘价数据")
            raise ValidationError("没有有效的收盘价数据")
        
        min_price, max_price = self.price_range
        out_of_range = ((close < min_price) | (close > max_price)).sum()
        if out_of_range > 0:
            self.validation_warnings.append(
                f"发现 {out_of_range} 个价格在合理范围外 "
                f"({min_price} ~ {max_price})"
            )
            if strict:
                df = df[(df['close'] >= min_price) & (df['close'] <= max_price)]
        
        # 5. 日收益率异常检查
        if 'close' in df.columns and len(df) > 1:
            df_sorted = df.sort_values('date').copy()
            df_sorted['ret'] = df_sorted['close'].pct_change()
            
            # 过滤极端收益率
            extreme_moves = df_sorted['ret'].abs() > self.DAILY_RETURN_THRESHOLD
            n_extreme = extreme_moves.sum()
            
            if n_extreme > 0:
                self.validation_warnings.append(
                    f"发现 {n_extreme} 次超过 "
                    f"{self.DAILY_RETURN_THRESHOLD*100:.0f}% 的价格变动"
                )
                if strict:
                    # 标记异常行
                    df_sorted['is_extreme'] = extreme_moves
                    # 用前后均值替换极端值
                    df_sorted.loc[df_sorted['is_extreme'], 'close'] = np.nan
                    df_sorted['close'] = df_sorted['close'].interpolate()
                    df = df_sorted.drop(columns=['ret', 'is_extreme'])
        
        # 6. 数值列类型转换
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def validate_date_range(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_gap_days: int = 10
    ) -> Dict[str, Any]:
        """
        验证日期范围
        
        Args:
            df: 数据 DataFrame
            start_date: 期望的开始日期
            end_date: 期望的结束日期
            max_gap_days: 允许的最大日期间隔
            
        Returns:
            验证结果字典
        """
        if df is None or df.empty:
            return {'valid': False, 'error': '数据为空'}
        
        result = {
            'valid': True,
            'date_min': None,
            'date_max': None,
            'date_range_days': 0,
            'actual_days': 0,
            'missing_days_pct': 0,
            'issues': [],
            'warnings': [],
        }
        
        # 确保日期列是 datetime 类型
        if df['date'].dtype == 'object':
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
        
        df_sorted = df.sort_values('date')
        date_min = df_sorted['date'].min()
        date_max = df_sorted['date'].max()
        
        result['date_min'] = date_min
        result['date_max'] = date_max
        
        # 计算日期范围
        if start_date:
            start_dt = pd.to_datetime(start_date)
            result['date_range_days'] = (date_max - start_dt).days
        else:
            result['date_range_days'] = (date_max - date_min).days
        
        result['actual_days'] = len(df_sorted)
        
        # 计算缺失天数百分比
        if result['date_range_days'] > 0:
            result['missing_days_pct'] = (
                (result['date_range_days'] - result['actual_days']) / 
                result['date_range_days'] * 100
            )
        
        # 检查1: 开始日期是否匹配
        if start_date:
            if date_min < pd.to_datetime(start_date) - timedelta(days=1):
                result['warnings'].append(
                    f"数据开始日期 ({date_min.date()}) 早于请求 "
                    f"({pd.to_datetime(start_date).date()})"
                )
        
        # 检查2: 结束日期是否匹配
        if end_date:
            if date_max > pd.to_datetime(end_date) + timedelta(days=1):
                result['warnings'].append(
                    f"数据结束日期 ({date_max.date()}) 晚于请求 "
                    f"({pd.to_datetime(end_date).date()})"
                )
        
        # 检查3: 数据完整性
        if result['missing_days_pct'] > 50:
            result['issues'].append(
                f"数据缺失严重: {result['missing_days_pct']:.1f}%"
            )
            result['valid'] = False
        
        # 检查4: 日期间隔异常
        if len(df_sorted) > 1:
            date_diffs = df_sorted['date'].diff().dt.days.dropna()
            large_gaps = date_diffs[date_diffs > max_gap_days]
            if len(large_gaps) > 0:
                result['warnings'].append(
                    f"发现 {len(large_gaps)} 处超过 {max_gap_days} 天的间隔"
                )
        
        return result
    
    def validate_freshness(
        self,
        df: pd.DataFrame,
        max_age_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        验证数据新鲜度
        
        Args:
            df: 数据 DataFrame
            max_age_days: 最大允许天数
            
        Returns:
            验证结果字典
        """
        if df is None or df.empty:
            return {'valid': False, 'error': '数据为空'}
        
        if max_age_days is None:
            max_age_days = self.MAX_DATA_AGE_DAYS
        
        result = {
            'valid': True,
            'latest_date': None,
            'days_old': 0,
            'max_age_days': max_age_days,
            'is_stale': False,
            'issues': [],
        }
        
        # 确保日期列是 datetime 类型
        if df['date'].dtype == 'object':
            df = df.copy()
            df['date'] = pd.to_datetime(df['date'])
        
        latest_date = df['date'].max()
        result['latest_date'] = latest_date
        
        now = datetime.now()
        days_old = (now - latest_date).days
        result['days_old'] = days_old
        
        # 检查1: 不能来自未来
        if latest_date > now:
            result['issues'].append(
                f"数据来自未来: {latest_date.date()} (现在: {now.date()})"
            )
            result['valid'] = False
        
        # 检查2: 不能太旧
        if days_old > max_age_days:
            result['is_stale'] = True
            result['issues'].append(
                f"数据过时: 最新数据是 {days_old} 天前 "
                f"(最大允许: {max_age_days} 天)"
            )
            result['valid'] = False
        
        # 检查3: 年份合理性
        if latest_date.year < now.year - 1:
            result['issues'].append(
                f"数据年份异常: {latest_date.year} (当前: {now.year})"
            )
            result['valid'] = False
        
        return result
    
    def validate_completeness(
        self,
        df: pd.DataFrame,
        required_cols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        验证数据完整性
        
        Args:
            df: 数据 DataFrame
            required_cols: 必需的列名列表
            
        Returns:
            验证结果字典
        """
        if df is None or df.empty:
            return {'valid': False, 'error': '数据为空'}
        
        if required_cols is None:
            required_cols = ['date', 'open', 'high', 'low', 'close']
        
        result = {
            'valid': True,
            'missing_cols': [],
            'null_counts': {},
            'issues': [],
        }
        
        # 检查1: 缺失列
        missing = set(required_cols) - set(df.columns)
        if missing:
            result['missing_cols'] = list(missing)
            result['valid'] = False
            result['issues'].append(f"缺少列: {missing}")
        
        # 检查2: 空值统计
        for col in required_cols:
            if col in df.columns:
                null_count = df[col].isna().sum()
                null_pct = null_count / len(df) * 100
                result['null_counts'][col] = {
                    'count': int(null_count),
                    'percentage': float(null_pct)
                }
                
                if null_pct > 50:
                    result['issues'].append(
                        f"列 '{col}' 空值过多: {null_pct:.1f}%"
                    )
                    result['valid'] = False
        
        return result
    
    def validate_all(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        variety: Optional[str] = None,
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        执行所有验证
        
        Args:
            df: 数据 DataFrame
            start_date: 期望的开始日期
            end_date: 期望的结束日期
            variety: 品种代码
            strict: 是否严格模式
            
        Returns:
            综合验证结果
        """
        if variety:
            self.variety = variety
            self.price_range = self.PRICE_RANGES.get(
                variety,
                self.PRICE_RANGES['DEFAULT']
            )
        
        result = {
            'overall_valid': True,
            'price_valid': True,
            'date_valid': True,
            'freshness_valid': True,
            'completeness_valid': True,
            'all_issues': [],
            'all_warnings': [],
            'cleaned_df': df.copy() if df is not None else None,
        }
        
        try:
            # 1. 价格数据验证
            try:
                cleaned = self.validate_price_data(df, strict=strict)
                result['cleaned_df'] = cleaned
            except ValidationError as e:
                result['price_valid'] = False
                result['overall_valid'] = False
                result['all_issues'].append(f"价格验证失败: {e}")
                return result
            
            # 2. 日期范围验证
            date_result = self.validate_date_range(
                result['cleaned_df'], 
                start_date, 
                end_date
            )
            result['date_valid'] = date_result['valid']
            result['date_range'] = date_result
            if not date_result['valid']:
                result['overall_valid'] = False
            result['all_issues'].extend(date_result.get('issues', []))
            result['all_warnings'].extend(date_result.get('warnings', []))
            
            # 3. 新鲜度验证
            freshness_result = self.validate_freshness(result['cleaned_df'])
            result['freshness_valid'] = freshness_result['valid']
            result['freshness'] = freshness_result
            if not freshness_result['valid']:
                result['overall_valid'] = False
            result['all_issues'].extend(freshness_result.get('issues', []))
            
            # 4. 完整性验证
            completeness_result = self.validate_completeness(result['cleaned_df'])
            result['completeness_valid'] = completeness_result['valid']
            result['completeness'] = completeness_result
            if not completeness_result['valid']:
                result['overall_valid'] = False
            result['all_issues'].extend(completeness_result.get('issues', []))
            
        except Exception as e:
            result['overall_valid'] = False
            result['all_issues'].append(f"验证过程异常: {e}")
        
        # 记录警告日志
        if result['all_warnings']:
            for warning in result['all_warnings']:
                logger.warning(f"[DataValidator] {warning}")
        
        if result['all_issues']:
            for issue in result['all_issues']:
                logger.error(f"[DataValidator] {issue}")
        
        return result
    
    def get_validation_summary(self, result: Dict[str, Any]) -> str:
        """
        获取验证结果摘要
        
        Args:
            result: validate_all 返回的结果
            
        Returns:
            格式化的摘要字符串
        """
        lines = [
            "=" * 50,
            "数据验证结果摘要",
            "=" * 50,
            f"整体验证: {'通过' if result['overall_valid'] else '失败'}",
            f"价格验证: {'通过' if result['price_valid'] else '失败'}",
            f"日期验证: {'通过' if result['date_valid'] else '失败'}",
            f"新鲜度验证: {'通过' if result['freshness_valid'] else '失败'}",
            f"完整性验证: {'通过' if result['completeness_valid'] else '失败'}",
        ]
        
        if result.get('freshness'):
            freshness = result['freshness']
            if freshness.get('latest_date'):
                lines.append(f"最新数据: {freshness['latest_date'].date()}")
                lines.append(f"数据天数: {freshness['days_old']} 天")
        
        if result['all_issues']:
            lines.append("")
            lines.append("问题列表:")
            for issue in result['all_issues']:
                lines.append(f"  - {issue}")
        
        if result['all_warnings']:
            lines.append("")
            lines.append("警告列表:")
            for warning in result['all_warnings']:
                lines.append(f"  - {warning}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


def validate_fetched_data(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    variety: Optional[str] = None,
    strict: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    便捷函数：验证获取的数据
    
    Args:
        df: 获取的原始数据
        start_date: 期望的开始日期
        end_date: 期望的结束日期
        variety: 品种代码
        strict: 是否严格模式
        
    Returns:
        (验证后的数据, 验证结果)
    """
    validator = DataValidator(variety)
    result = validator.validate_all(df, start_date, end_date, variety, strict)
    
    if result['cleaned_df'] is not None:
        return result['cleaned_df'], result
    else:
        return df, result
