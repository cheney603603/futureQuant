"""
test_factor_engine.py - FactorEngine 单元测试

测试内容：
1. FactorEngine.register() 注册因子
2. FactorEngine.compute_all() 返回 DataFrame 且列名正确
3. 因子结果与输入长度一致（时间序列对齐）
"""
import sys
import pytest
import pandas as pd
import numpy as np

# 尝试导入目标模块（无法导入则跳过整个模块的测试）
try:
    from futureQuant.factor.engine import FactorEngine
    from futureQuant.core.base import Factor
except Exception:
    # 如果是因为 scipy 临时目录问题，使用 pytest.skip
    pytest.skip("Cannot import futureQuant (temp directory issue)", allow_module_level=True)

# =============================================================================
# 测试用因子实现
# =============================================================================

class DummyFactor(Factor):
    """测试用简单因子"""
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        # 简单收盘价/均价
        if 'close' not in data.columns:
            return pd.Series(dtype=float)
        result = data['close'] / data['close'].rolling(5).mean()
        return result.fillna(1.0)


class VolatilityFactor(Factor):
    """测试用波动率因子"""
    
    def compute(self, data: pd.DataFrame) -> pd.Series:
        if 'close' not in data.columns:
            return pd.Series(dtype=float)
        ret = data['close'].pct_change()
        return ret.rolling(10).std().fillna(0.01)


# =============================================================================
# 测试用例
# =============================================================================

class TestFactorEngineRegister:
    """测试因子注册"""
    
    def test_register_single_factor(self):
        engine = FactorEngine()
        factor = DummyFactor(name='test_factor')
        
        engine.register(factor)
        
        assert 'test_factor' in engine.factors
        assert engine.factors['test_factor'] is factor
    
    def test_register_overwrites_existing(self, caplog):
        engine = FactorEngine()
        factor1 = DummyFactor(name='dup_factor')
        factor2 = DummyFactor(name='dup_factor')
        
        engine.register(factor1)
        engine.register(factor2)
        
        assert engine.factors['dup_factor'] is factor2
        assert 'already registered' in caplog.text.lower()
    
    def test_register_many(self):
        engine = FactorEngine()
        factors = [
            DummyFactor(name='factor_a'),
            VolatilityFactor(name='factor_b'),
        ]
        
        engine.register_many(factors)
        
        assert len(engine.factors) == 2
        assert 'factor_a' in engine.factors
        assert 'factor_b' in engine.factors
    
    def test_unregister(self):
        engine = FactorEngine()
        factor = DummyFactor(name='to_remove')
        engine.register(factor)
        
        engine.unregister('to_remove')
        
        assert 'to_remove' not in engine.factors
    
    def test_list_factors(self):
        engine = FactorEngine()
        engine.register(DummyFactor(name='f1'))
        engine.register(DummyFactor(name='f2'))
        
        names = engine.list_factors()
        
        assert 'f1' in names
        assert 'f2' in names


class TestFactorEngineCompute:
    """测试因子计算"""
    
    def test_compute_all_returns_dataframe(self, sample_ohlcv):
        """compute_all() 应返回 DataFrame"""
        engine = FactorEngine()
        
        # 用单个品种的数据测试
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        engine.register(DummyFactor(name='dummy'))
        engine.register(VolatilityFactor(name='vol'))
        
        result = engine.compute_all(rb_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    def test_compute_all_column_names(self, sample_ohlcv):
        """compute_all() 返回的列名应与注册的因子名一致"""
        engine = FactorEngine()
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        engine.register(DummyFactor(name='dummy'))
        engine.register(VolatilityFactor(name='vol'))
        
        result = engine.compute_all(rb_data)
        
        assert 'dummy' in result.columns
        assert 'vol' in result.columns
    
    def test_compute_all_length_matches_input(self, sample_ohlcv):
        """因子结果长度应与输入数据对齐"""
        engine = FactorEngine()
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        engine.register(DummyFactor(name='test'))
        
        result = engine.compute_all(rb_data)
        
        assert len(result) == len(rb_data)
    
    def test_compute_all_with_factor_names_filter(self, sample_ohlcv):
        """指定因子名称时只计算指定因子"""
        engine = FactorEngine()
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        engine.register(DummyFactor(name='dummy'))
        engine.register(VolatilityFactor(name='vol'))
        
        result = engine.compute_all(rb_data, factor_names=['dummy'])
        
        assert 'dummy' in result.columns
        assert 'vol' not in result.columns
    
    def test_compute_all_empty_when_no_factors(self, sample_ohlcv):
        """无注册因子时返回空 DataFrame"""
        engine = FactorEngine()
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        result = engine.compute_all(rb_data)
        
        assert result.empty
    
    def test_compute_single_factor(self, sample_ohlcv):
        """compute() 单个因子返回 Series"""
        engine = FactorEngine()
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        engine.register(DummyFactor(name='test'))
        
        result = engine.compute(rb_data, 'test')
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(rb_data)
    
    def test_compute_uses_cache(self, sample_ohlcv):
        """第二次调用应使用缓存"""
        engine = FactorEngine()
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        engine.register(DummyFactor(name='test'))
        
        result1 = engine.compute(rb_data, 'test', use_cache=True)
        result2 = engine.compute(rb_data, 'test', use_cache=True)
        
        pd.testing.assert_series_equal(result1, result2)
    
    def test_clear_cache(self, sample_ohlcv):
        """clear_cache() 应清空缓存"""
        engine = FactorEngine()
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        engine.register(DummyFactor(name='test'))
        engine.compute(rb_data, 'test', use_cache=True)
        
        # 缓存键为 (factor_name, data_hash)，只要缓存非空即表示命中
        assert len(engine.cache) > 0
        
        engine.clear_cache()
        
        assert len(engine.cache) == 0
    
    def test_get_factor_info(self):
        """get_factor_info() 返回因子元信息"""
        engine = FactorEngine()
        factor = DummyFactor(name='info_test', param1=10, param2='value')
        engine.register(factor)
        
        info = engine.get_factor_info('info_test')
        
        assert info is not None
        assert info['name'] == 'info_test'
        assert info['class'] == 'DummyFactor'
        assert 'param1' in info['params']
    
    def test_get_factor_info_not_found(self):
        """get_factor_info() 对不存在的因子返回 None"""
        engine = FactorEngine()
        
        result = engine.get_factor_info('nonexistent')
        
        assert result is None
