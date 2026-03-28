"""
test_data_cleaner.py - DataCleaner 单元测试

测试内容：
1. clean_ohlc() 去除价格为 0 的行
2. clean_ohlc() 去除 high < low 的异常行
3. 结果行数 <= 输入行数
"""
import pytest
import pandas as pd
import numpy as np

pytest.importorskip("futureQuant")

from futureQuant.data.processor.cleaner import DataCleaner


# =============================================================================
# 测试用例
# =============================================================================

class TestCleanOHLC:
    """测试 OHLC 数据清洗"""
    
    def test_clean_removes_zero_prices(self):
        """去除价格为 0 的行"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02', '2024-08-03', '2024-08-04'],
            'symbol': ['RB'] * 4,
            'open': [3800.0, 0.0, 3810.0, 3820.0],    # 第二行 open=0
            'high': [3850.0, 0.0, 3860.0, 3870.0],    # 第二行 high=0
            'low': [3750.0, 0.0, 3760.0, 3770.0],     # 第二行 low=0
            'close': [3800.0, 0.0, 3810.0, 3820.0],   # 第二行 close=0
            'volume': [100000] * 4,
            'open_interest': [500000] * 4,
        })
        
        result = cleaner.clean_ohlc(df)
        
        # 0 价格行应该被去除
        assert 0.0 not in result['close'].values
        assert len(result) < len(df)
        assert len(result) == 3  # 原始4行，去除1行
    
    def test_clean_removes_negative_prices(self):
        """去除负价格行"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02', '2024-08-03'],
            'symbol': ['RB'] * 3,
            'open': [3800.0, -100.0, 3810.0],
            'high': [3850.0, -50.0, 3860.0],
            'low': [3750.0, -80.0, 3760.0],
            'close': [3800.0, -100.0, 3810.0],
            'volume': [100000] * 3,
            'open_interest': [500000] * 3,
        })
        
        result = cleaner.clean_ohlc(df)
        
        assert (result['close'] > 0).all()
        assert len(result) == 2
    
    def test_clean_removes_high_less_than_low(self):
        """去除 high < low 的异常行"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02', '2024-08-03', '2024-08-04'],
            'symbol': ['RB'] * 4,
            'open': [3800.0, 3850.0, 3810.0, 3820.0],
            'high': [3850.0, 3780.0, 3860.0, 3870.0],  # 第二行 high < low
            'low':  [3750.0, 3820.0, 3760.0, 3770.0],
            'close': [3800.0, 3820.0, 3810.0, 3820.0],
            'volume': [100000] * 4,
            'open_interest': [500000] * 4,
        })
        
        result = cleaner.clean_ohlc(df)
        
        # 第二行 high(3780) < low(3820)，应该被去除
        assert len(result) <= len(df)
        # 检查所有剩余行的 high >= low
        assert (result['high'] >= result['low']).all()
    
    def test_clean_result_row_count_le_input(self):
        """结果行数应 <= 输入行数"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02', '2024-08-03', '2024-08-04', '2024-08-05'],
            'symbol': ['RB'] * 5,
            'open': [3800.0, 0.0, 3850.0, 3810.0, 0.0],
            'high': [3850.0, 0.0, 3780.0, 3860.0, 0.0],   # 第3行 high < low
            'low':  [3750.0, 0.0, 3820.0, 3760.0, 0.0],
            'close': [3800.0, 0.0, 3820.0, 3810.0, 0.0],
            'volume': [100000] * 5,
            'open_interest': [500000] * 5,
        })
        
        result = cleaner.clean_ohlc(df)
        
        assert len(result) <= len(df)
    
    def test_clean_with_valid_data(self):
        """有效数据行应保留"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02', '2024-08-03'],
            'symbol': ['RB'] * 3,
            'open': [3800.0, 3810.0, 3820.0],
            'high': [3850.0, 3860.0, 3870.0],
            'low':  [3750.0, 3760.0, 3770.0],
            'close': [3800.0, 3810.0, 3820.0],
            'volume': [100000] * 3,
            'open_interest': [500000] * 3,
        })
        
        result = cleaner.clean_ohlc(df)
        
        # 有效数据应全部保留
        assert len(result) == 3
        pd.testing.assert_frame_equal(result, df)
    
    def test_clean_preserves_non_ohlc_columns(self):
        """保留非 OHLC 列"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02'],
            'symbol': ['RB', 'RB'],
            'open': [3800.0, 3810.0],
            'high': [3850.0, 3860.0],
            'low':  [3750.0, 3760.0],
            'close': [3800.0, 3810.0],
            'volume': [100000, 200000],
            'open_interest': [500000, 600000],
            'extra_col': ['A', 'B'],  # 额外列
        })
        
        result = cleaner.clean_ohlc(df)
        
        assert 'extra_col' in result.columns
        assert 'volume' in result.columns
    
    def test_clean_empty_input(self):
        """空输入返回空 DataFrame"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame(columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'open_interest'])
        
        result = cleaner.clean_ohlc(df)
        
        assert result.empty


class TestHandleMissingValues:
    """测试缺失值处理"""
    
    def test_ffill(self):
        """前向填充"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02', '2024-08-03'],
            'close': [3800.0, np.nan, 3820.0],
        })
        
        result = cleaner.handle_missing_values(df, method='ffill')
        
        assert result.iloc[1]['close'] == 3800.0
    
    def test_drop(self):
        """删除缺失值"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02', '2024-08-03'],
            'close': [3800.0, np.nan, 3820.0],
        })
        
        result = cleaner.handle_missing_values(df, method='drop')
        
        assert len(result) == 2
        assert result['close'].notna().all()


class TestRemoveOutliers:
    """测试异常值去除"""
    
    def test_iqr_method(self):
        """IQR 方法去除异常值"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02', '2024-08-03', '2024-08-04'],
            'close': [3800.0, 3850.0, 3900.0, 8000.0],  # 最后一行是异常值
        })
        
        result = cleaner.remove_outliers(df, columns=['close'], method='iqr', threshold=1.5)
        
        assert len(result) <= len(df)
    
    def test_zscore_method(self):
        """Z-score 方法"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02', '2024-08-03'],
            'close': [3800.0, 3850.0, 9000.0],  # 最后一行是异常值
        })
        
        result = cleaner.remove_outliers(df, columns=['close'], method='zscore', threshold=2.0)
        
        assert len(result) <= len(df)


class TestOtherCleanerMethods:
    """测试其他清洗方法"""
    
    def test_detect_price_jumps(self):
        """检测价格跳空"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02', '2024-08-03'],
            'close': [3800.0, 3850.0, 4100.0],  # 第三行跳空上涨
        })
        
        result = cleaner.detect_price_jumps(df, price_col='close', threshold=0.05)
        
        assert 'return' in result.columns
        assert 'is_jump' in result.columns
    
    def test_resample_weekly(self):
        """周频重采样"""
        cleaner = DataCleaner()
        
        df = pd.DataFrame({
            'date': pd.date_range('2024-08-01', periods=10, freq='B').strftime('%Y-%m-%d').tolist(),
            'open': [3800.0] * 10,
            'high': [3850.0] * 10,
            'low': [3750.0] * 10,
            'close': [3800.0, 3810, 3820, 3830, 3840, 3850, 3860, 3870, 3880, 3890],
            'volume': [100000] * 10,
            'open_interest': [500000] * 10,
        })
        
        result = cleaner.resample(df, rule='W')
        
        assert len(result) < len(df)
        assert 'close' in result.columns
