"""
因子引擎输入验证 - 数据质量守门员

P2.4 实现：
- 因子计算前自动验证输入数据
- 支持严格/宽松两种模式
- 集成 DataValidator 和 DataQualityReporter
- 防止脏数据流入因子计算链

Author: futureQuant Team
Date: 2026-04-19
"""

from typing import Optional, Dict, List, Any, Callable
import pandas as pd
import numpy as np

from futureQuant.core.logger import get_logger
from futureQuant.core.exceptions import FactorError, DataError
from futureQuant.data.validator import DataValidator
from futureQuant.data.quality_reporter import DataQualityReporter, QualityMetrics

logger = get_logger('factor.validation')


class FactorInputValidator:
    """
    因子输入验证器
    
    在因子计算前对数据进行验证，防止脏数据流入计算链。
    
    使用示例：
        validator = FactorInputValidator(variety='RB', strict=True)
        
        # 验证数据
        is_valid, issues = validator.validate(price_df)
        
        if not is_valid:
            logger.error(f"数据验证失败: {issues}")
            return None
        
        # 继续因子计算
        result = factor.compute(price_df)
    """
    
    # 必需列
    REQUIRED_COLUMNS = ['close']
    
    # 可选列（有则更好）
    RECOMMENDED_COLUMNS = ['open', 'high', 'low', 'volume', 'date']
    
    # 质量阈值
    QUALITY_THRESHOLDS = {
        'strict': {
            'min_rows': 20,
            'min_freshness_score': 0.7,
            'min_completeness_rate': 0.8,
            'max_outlier_rate': 0.05,
        },
        'loose': {
            'min_rows': 5,
            'min_freshness_score': 0.3,
            'min_completeness_rate': 0.5,
            'max_outlier_rate': 0.15,
        }
    }
    
    def __init__(
        self,
        variety: str,
        mode: str = 'strict',
        custom_thresholds: Optional[Dict[str, float]] = None,
        auto_clean: bool = False
    ):
        """
        初始化验证器
        
        Args:
            variety: 品种代码
            mode: 验证模式 ('strict' 或 'loose')
            custom_thresholds: 自定义阈值
            auto_clean: 是否自动清理异常数据
        """
        self.variety = variety
        self.mode = mode
        self.auto_clean = auto_clean
        
        # 合并阈值
        base_thresholds = self.QUALITY_THRESHOLDS.get(mode, self.QUALITY_THRESHOLDS['strict'])
        self.thresholds = {**base_thresholds, **(custom_thresholds or {})}
        
        # 子验证器
        self.data_validator = DataValidator(variety=variety)
        self.quality_reporter = DataQualityReporter()
        
        # 验证结果缓存
        self.last_validation: Optional[Dict] = None
    
    def validate(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        expected_columns: Optional[List[str]] = None
    ) -> tuple[bool, List[str]]:
        """
        验证输入数据
        
        Args:
            data: 输入数据
            start_date: 期望开始日期
            end_date: 期望结束日期
            expected_columns: 期望的列名
            
        Returns:
            (是否通过, 问题列表)
        """
        issues = []
        
        # 1. 基本检查
        if data is None:
            issues.append("输入数据为 None")
            return False, issues
        
        if not isinstance(data, pd.DataFrame):
            issues.append(f"输入类型错误: {type(data)}, 期望 DataFrame")
            return False, issues
        
        if data.empty:
            issues.append("输入数据为空")
            return False, issues
        
        # 2. 列检查
        required = expected_columns or self.REQUIRED_COLUMNS
        missing_cols = [col for col in required if col not in data.columns]
        if missing_cols:
            issues.append(f"缺少必需列: {missing_cols}")
        
        # 3. 行数检查
        if len(data) < self.thresholds['min_rows']:
            issues.append(
                f"数据行数不足: {len(data)} < {self.thresholds['min_rows']}"
            )
        
        # 4. 数据质量检查
        if start_date and end_date:
            quality_report = self.quality_reporter.generate_report(
                data, 'price', self.variety, start_date, end_date,
                expected_columns=expected_columns
            )
            
            # 新鲜度检查
            if quality_report.freshness_score < self.thresholds['min_freshness_score']:
                issues.append(
                    f"数据新鲜度不足: {quality_report.freshness_score:.2f} "
                    f"< {self.thresholds['min_freshness_score']}"
                )
            
            # 完整性检查
            if quality_report.completeness_rate < self.thresholds['min_completeness_rate']:
                issues.append(
                    f"数据完整性不足: {quality_report.completeness_rate:.2f} "
                    f"< {self.thresholds['min_completeness_rate']}"
                )
            
            # 异常值检查
            if quality_report.outlier_rate > self.thresholds['max_outlier_rate']:
                issues.append(
                    f"异常值比例过高: {quality_report.outlier_rate:.2%} "
                    f"> {self.thresholds['max_outlier_rate']}"
                )
            
            # 添加质量报告中的其他问题
            for issue in quality_report.issues:
                if issue not in issues:
                    issues.append(issue)
        
        # 5. 价格合理性检查
        if 'close' in data.columns:
            price_issues = self._validate_prices(data)
            issues.extend(price_issues)
        
        # 记录验证结果
        self.last_validation = {
            'passed': len(issues) == 0,
            'issues': issues,
            'row_count': len(data),
            'column_count': len(data.columns),
            'columns': list(data.columns),
        }
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info(f"数据验证通过: {self.variety}, {len(data)} rows")
        else:
            logger.warning(f"数据验证失败: {self.variety}, 发现 {len(issues)} 个问题")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        return is_valid, issues
    
    def _validate_prices(self, data: pd.DataFrame) -> List[str]:
        """验证价格数据"""
        issues = []
        
        # 检查 NaN
        nan_count = data['close'].isna().sum()
        if nan_count > 0:
            nan_rate = nan_count / len(data)
            if nan_rate > 0.1:  # 超过10% NaN
                issues.append(f"收盘价 NaN 比例过高: {nan_rate:.1%}")
        
        # 检查零值
        zero_count = (data['close'] == 0).sum()
        if zero_count > 0:
            issues.append(f"发现 {zero_count} 个零值收盘价")
        
        # 检查负数
        neg_count = (data['close'] < 0).sum()
        if neg_count > 0:
            issues.append(f"发现 {neg_count} 个负值收盘价")
        
        return issues
    
    def clean_data(
        self,
        data: pd.DataFrame,
        remove_outliers: bool = True,
        fill_missing: bool = False
    ) -> pd.DataFrame:
        """
        清理数据
        
        Args:
            data: 原始数据
            remove_outliers: 是否移除异常值
            fill_missing: 是否填充缺失值
            
        Returns:
            清理后的数据
        """
        df = data.copy()
        
        # 使用 DataValidator 进行清理
        df = self.data_validator.validate_price_data(df, strict=remove_outliers)
        
        # 填充缺失值
        if fill_missing and 'close' in df.columns:
            df['close'] = df['close'].fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"数据清理: {len(data)} -> {len(df)} rows")
        
        return df
    
    def validate_or_raise(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        验证数据，失败则抛出异常
        
        Args:
            data: 输入数据
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            验证通过的数据（或清理后的数据）
            
        Raises:
            DataError: 验证失败
        """
        is_valid, issues = self.validate(data, start_date, end_date, **kwargs)
        
        if not is_valid:
            error_msg = f"数据验证失败 ({self.variety}): " + "; ".join(issues)
            raise DataError(error_msg)
        
        # 如果需要自动清理
        if self.auto_clean:
            return self.clean_data(data)
        
        return data
    
    def get_validation_summary(self) -> Optional[Dict]:
        """获取上次验证摘要"""
        return self.last_validation


# 装饰器模式

def validate_factor_input(
    variety_param: str = 'variety',
    data_param: str = 'data',
    mode: str = 'strict',
    auto_clean: bool = False
):
    """
    因子输入验证装饰器
    
    使用示例：
        @validate_factor_input(variety_param='variety', data_param='price_df')
        def compute_momentum(variety: str, price_df: pd.DataFrame) -> pd.Series:
            # 数据已自动验证
            return price_df['close'].rolling(20).mean()
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # 获取 variety 和 data
            variety = kwargs.get(variety_param)
            data = kwargs.get(data_param)
            
            if variety is None or data is None:
                # 尝试从位置参数获取
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                
                if variety_param in params:
                    idx = params.index(variety_param)
                    if idx < len(args):
                        variety = args[idx]
                
                if data_param in params:
                    idx = params.index(data_param)
                    if idx < len(args):
                        data = args[idx]
            
            if variety and data is not None:
                validator = FactorInputValidator(variety=variety, mode=mode, auto_clean=auto_clean)
                
                is_valid, issues = validator.validate(data)
                
                if not is_valid:
                    raise DataError(
                        f"Factor input validation failed for {func.__name__}: " + 
                        "; ".join(issues)
                    )
                
                # 如果需要清理
                if auto_clean:
                    data = validator.clean_data(data)
                    kwargs[data_param] = data
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# 集成到 FactorEngine 的辅助函数

def create_validated_engine(
    engine_class,
    default_variety: Optional[str] = None,
    validation_mode: str = 'strict'
):
    """
    创建带输入验证的因子引擎
    
    Args:
        engine_class: 因子引擎类
        default_variety: 默认品种
        validation_mode: 验证模式
        
    Returns:
        带验证的引擎类
    """
    class ValidatedEngine(engine_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._validator = None
            self._default_variety = default_variety
            self._validation_mode = validation_mode
        
        def compute(self, data: pd.DataFrame, factor_name: str, use_cache: bool = True, **kwargs):
            """带验证的 compute"""
            # 获取品种
            variety = kwargs.get('variety', self._default_variety)
            
            if variety:
                if self._validator is None or self._validator.variety != variety:
                    self._validator = FactorInputValidator(
                        variety=variety,
                        mode=self._validation_mode
                    )
                
                # 验证数据
                is_valid, issues = self._validator.validate(data)
                
                if not is_valid:
                    logger.error(f"FactorEngine validation failed: {issues}")
                    raise FactorError(f"Input validation failed: {issues}")
            
            return super().compute(data, factor_name, use_cache)
    
    return ValidatedEngine
