"""
数据验证器单元测试

测试内容：
1. 价格数据验证
2. 日期范围验证
3. 数据新鲜度验证
4. 数据完整性验证
5. 端到端验证流程

Author: futureQuant Team
Date: 2026-04-19
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from futureQuant.data.validator import (
    DataValidator,
    ValidationError,
    validate_fetched_data
)


class TestDataValidator:
    """数据验证器测试"""
    
    def setup_method(self):
        """每个测试前设置"""
        self.validator = DataValidator()
    
    def create_sample_price_data(
        self,
        n_days: int = 30,
        start_date: str = None
    ) -> pd.DataFrame:
        """创建样本价格数据"""
        # 默认使用最近7天内的数据（确保新鲜度验证通过）
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        dates = pd.date_range(start=start_date, periods=n_days, freq='D')
        # 模拟价格：100元，每天小幅波动
        np.random.seed(42)
        base_price = 1000
        prices = base_price + np.cumsum(np.random.randn(n_days) * 10)
        
        df = pd.DataFrame({
            'date': dates,
            'open': prices * 0.99,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, n_days),
        })
        
        return df
    
    def test_validate_price_data_valid(self):
        """测试有效价格数据验证"""
        df = self.create_sample_price_data()
        
        result_df = self.validator.validate_price_data(df)
        
        assert len(result_df) == 30
        assert 'date' in result_df.columns
        assert 'close' in result_df.columns
    
    def test_validate_price_data_empty(self):
        """测试空数据验证"""
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_price_data(pd.DataFrame())
        
        assert "数据为空" in str(exc_info.value)
    
    def test_validate_price_data_missing_columns(self):
        """测试缺少必要列的数据"""
        df = pd.DataFrame({'date': ['2026-03-01'], 'volume': [1000]})
        
        with pytest.raises(ValidationError) as exc_info:
            self.validator.validate_price_data(df)
        
        assert "缺少必要列" in str(exc_info.value)
    
    def test_validate_price_data_out_of_range(self):
        """测试价格超出合理范围"""
        df = self.create_sample_price_data()
        # 设置一个超出范围的价格
        df.loc[0, 'close'] = 50  # 太低
        
        result_df = self.validator.validate_price_data(df, strict=False)
        
        # 非严格模式应该产生警告但不抛出异常
        assert len(result_df) == 30  # 数据保留
    
    def test_validate_price_data_extreme_return(self):
        """测试日收益率异常的数据"""
        df = self.create_sample_price_data()
        # 设置一个异常的日收益率（+50%）
        df.loc[1, 'close'] = df.loc[0, 'close'] * 1.5
        
        # 非严格模式
        result_df = self.validator.validate_price_data(df, strict=False)
        assert len(result_df) == 30  # 数据保留
        
        # 严格模式应该过滤
        self.validator.reset()
        result_df_strict = self.validator.validate_price_data(df, strict=True)
        # 严格模式下异常值被处理
    
    def test_validate_date_range_valid(self):
        """测试有效的日期范围"""
        df = self.create_sample_price_data(
            start_date='2026-03-01',
            n_days=30
        )
        
        result = self.validator.validate_date_range(
            df,
            start_date='2026-03-01',
            end_date='2026-03-30'
        )
        
        assert result['valid'] is True
        assert result['date_min'] is not None
        assert result['date_max'] is not None
    
    def test_validate_date_range_partial(self):
        """测试部分在范围内的数据"""
        # 数据开始于请求范围之前
        df = self.create_sample_price_data(
            start_date=(datetime.now() - timedelta(days=45)).strftime('%Y-%m-%d'),
            n_days=60
        )
        
        recent_start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        recent_end = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        result = self.validator.validate_date_range(
            df,
            start_date=recent_start,
            end_date=recent_end
        )
        
        # 应该产生警告但不标记为无效
        assert 'warnings' in result
    
    def test_validate_date_range_future(self):
        """测试包含未来日期的数据"""
        # 数据开始于请求范围之前
        df = self.create_sample_price_data(
            start_date=(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            n_days=10
        )
        
        result = self.validator.validate_date_range(df)
        
        # 应该产生警告
        assert 'issues' in result
    
    def test_validate_freshness_valid(self):
        """测试新鲜数据"""
        # 创建最近3天的数据
        df = self.create_sample_price_data(
            start_date=(datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
            n_days=3
        )
        
        result = self.validator.validate_freshness(df)
        
        assert result['valid'] is True
        assert result['is_stale'] is False
    
    def test_validate_freshness_stale(self):
        """测试过时的数据"""
        df = self.create_sample_price_data(
            start_date=(datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'),
            n_days=30
        )
        
        result = self.validator.validate_freshness(df, max_age_days=7)
        
        assert result['valid'] is False
        assert result['is_stale'] is True
        assert 'issues' in result
    
    def test_validate_freshness_future(self):
        """测试来自未来的数据"""
        df = self.create_sample_price_data(
            start_date=(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d'),
            n_days=10
        )
        
        result = self.validator.validate_freshness(df)
        
        assert result['valid'] is False
        assert any('未来' in issue for issue in result.get('issues', []))
    
    def test_validate_completeness(self):
        """测试数据完整性"""
        df = self.create_sample_price_data()
        
        result = self.validator.validate_completeness(df)
        
        assert result['valid'] is True
        assert len(result['missing_cols']) == 0
    
    def test_validate_completeness_missing_columns(self):
        """测试缺少列的完整性"""
        df = pd.DataFrame({
            'date': ['2026-03-01'],
            'close': [1000],
        })
        
        result = self.validator.validate_completeness(df)
        
        assert result['valid'] is False
        assert len(result['missing_cols']) > 0
    
    def test_validate_all_valid(self):
        """测试完整验证流程 - 有效数据"""
        # 创建最近5天的数据
        df = self.create_sample_price_data(
            start_date=(datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'),
            n_days=5
        )
        
        result = self.validator.validate_all(
            df,
            variety='RB',
            strict=False  # 非严格模式
        )
        
        assert result['price_valid'] is True
        assert result['cleaned_df'] is not None
    
    def test_validate_all_with_errors(self):
        """测试完整验证流程 - 有问题的数据"""
        df = self.create_sample_price_data()
        df.loc[0, 'close'] = 50  # 价格过低
        
        result = self.validator.validate_all(
            df,
            start_date='2026-03-01',
            end_date='2026-03-30',
            variety='RB',
            strict=False
        )
        
        # 应该能处理但产生警告
        assert result['cleaned_df'] is not None
    
    def test_get_validation_summary(self):
        """测试验证摘要生成"""
        df = self.create_sample_price_data()
        result = self.validator.validate_all(df)
        
        summary = self.validator.get_validation_summary(result)
        
        assert "数据验证结果摘要" in summary
        assert "整体验证" in summary


class TestValidateFetchedDataFunction:
    """便捷函数测试"""
    
    def test_validate_fetched_data_valid(self):
        """测试有效数据"""
        start = datetime.now() - timedelta(days=7)
        dates = pd.date_range(start, periods=30, freq='D')
        df = pd.DataFrame({
            'date': dates,
            'open': 1000 + np.cumsum(np.random.randn(30) * 10),
            'high': 1010 + np.cumsum(np.random.randn(30) * 10),
            'low': 990 + np.cumsum(np.random.randn(30) * 10),
            'close': 1000 + np.cumsum(np.random.randn(30) * 10),
        })
        
        cleaned_df, result = validate_fetched_data(
            df,
            start_date=start.strftime('%Y-%m-%d'),
            end_date=datetime.now().strftime('%Y-%m-%d'),
            variety='RB'
        )
        
        assert len(cleaned_df) > 0
        assert result['price_valid'] is True
    
    def test_validate_fetched_data_empty(self):
        """测试空数据"""
        cleaned_df, result = validate_fetched_data(pd.DataFrame())
        
        assert result['overall_valid'] is False


class TestVarietySpecificRanges:
    """品种特定价格范围测试"""
    
    def test_rb_price_range(self):
        """测试螺纹钢价格范围"""
        validator = DataValidator(variety='RB')
        
        assert validator.price_range == (1000, 6000)
    
    def test_fg_price_range(self):
        """测试玻璃价格范围"""
        validator = DataValidator(variety='FG')
        
        assert validator.price_range == (800, 2500)
    
    def test_cu_price_range(self):
        """测试铜价格范围"""
        validator = DataValidator(variety='CU')
        
        assert validator.price_range == (40000, 90000)
    
    def test_unknown_variety_price_range(self):
        """测试未知品种使用默认范围"""
        validator = DataValidator(variety='XX')
        
        assert validator.price_range == (100, 100000)


class TestEdgeCases:
    """边界情况测试"""
    
    def test_single_row_data(self):
        """测试单行数据"""
        df = pd.DataFrame({
            'date': [datetime.now()],
            'close': [1000],
            'open': [990],
            'high': [1010],
            'low': [985],
        })
        
        validator = DataValidator()
        result_df = validator.validate_price_data(df)
        
        assert len(result_df) == 1
    
    def test_all_same_prices(self):
        """测试所有价格相同"""
        df = pd.DataFrame({
            'date': pd.date_range('2026-03-01', periods=10, freq='D'),
            'close': [1000] * 10,
            'open': [1000] * 10,
            'high': [1000] * 10,
            'low': [1000] * 10,
        })
        
        validator = DataValidator()
        result_df = validator.validate_price_data(df)
        
        assert len(result_df) == 10
    
    def test_negative_prices(self):
        """测试负价格"""
        df = pd.DataFrame({
            'date': pd.date_range('2026-03-01', periods=5, freq='D'),
            'close': [1000, 990, -10, 980, 970],  # 有一个负价格
        })
        
        validator = DataValidator()
        result_df = validator.validate_price_data(df, strict=True)
        
        # 负价格应该被处理


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
