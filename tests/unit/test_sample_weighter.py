"""
样本权重器测试

测试波动率权重、流动性权重和市场状态权重。
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from futureQuant.agent.validators.sample_weighter import SampleWeighter


# ============================================================================
# SampleWeighter 初始化测试
# ============================================================================

class TestSampleWeighterInit:
    """SampleWeighter 初始化测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        weighter = SampleWeighter()
        
        assert weighter.config is not None
    
    def test_init_custom_config(self):
        """测试自定义配置"""
        config = {
            'volatility_window': 20,
            'liquidity_window': 10,
        }
        weighter = SampleWeighter(config=config)
        
        assert weighter.config['volatility_window'] == 20


# ============================================================================
# 波动率权重测试
# ============================================================================

class TestVolatilityWeighting:
    """波动率权重测试"""
    
    @pytest.fixture
    def sample_price_data(self):
        """生成测试价格数据"""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        # 生成价格序列
        prices = 3500 + np.cumsum(np.random.randn(n) * 10)
        
        data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(100000, 500000, n),
        }, index=dates)
        
        return data
    
    def test_volatility_weight_calculation(self, sample_price_data):
        """测试波动率权重计算"""
        weighter = SampleWeighter(config={'volatility_window': 20})
        
        weights = weighter.calculate_volatility_weights(sample_price_data)
        
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_price_data)
        assert weights.min() >= 0
        assert weights.max() <= 1
    
    def test_volatility_weight_normalization(self, sample_price_data):
        """测试波动率权重归一化"""
        weighter = SampleWeighter()
        
        weights = weighter.calculate_volatility_weights(sample_price_data)
        
        # 权重应该在 0-1 之间
        assert (weights >= 0).all()
        assert (weights <= 1).all()
    
    def test_high_volatility_higher_weight(self):
        """测试高波动率获得更高权重"""
        weighter = SampleWeighter(config={'volatility_window': 10})
        
        # 创建两段数据：一段高波动，一段低波动
        n = 100
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        # 前 50 天：低波动
        low_vol_prices = 3500 + np.random.randn(50) * 1
        # 后 50 天：高波动
        high_vol_prices = 3500 + np.random.randn(50) * 50
        
        prices = np.concatenate([low_vol_prices, high_vol_prices])
        
        data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(100000, 500000, n),
        }, index=dates)
        
        weights = weighter.calculate_volatility_weights(data)
        
        # 后期权重应该更高
        assert weights.iloc[-10:].mean() > weights.iloc[:10].mean()


# ============================================================================
# 流动性权重测试
# ============================================================================

class TestLiquidityWeighting:
    """流动性权重测试"""
    
    @pytest.fixture
    def sample_volume_data(self):
        """生成测试成交量数据"""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        data = pd.DataFrame({
            'volume': np.random.randint(100000, 500000, n),
            'close': 3500 + np.random.randn(n) * 10,
        }, index=dates)
        
        return data
    
    def test_liquidity_weight_calculation(self, sample_volume_data):
        """测试流动性权重计算"""
        weighter = SampleWeighter(config={'liquidity_window': 20})
        
        weights = weighter.calculate_liquidity_weights(sample_volume_data)
        
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_volume_data)
        assert weights.min() >= 0
        assert weights.max() <= 1
    
    def test_high_volume_higher_weight(self):
        """测试高成交量获得更高权重"""
        weighter = SampleWeighter(config={'liquidity_window': 10})
        
        n = 100
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        # 前 50 天：低成交量
        low_volume = np.random.randint(100000, 200000, 50)
        # 后 50 天：高成交量
        high_volume = np.random.randint(400000, 500000, 50)
        
        volumes = np.concatenate([low_volume, high_volume])
        
        data = pd.DataFrame({
            'volume': volumes,
            'close': 3500 + np.random.randn(n) * 10,
        }, index=dates)
        
        weights = weighter.calculate_liquidity_weights(data)
        
        # 后期权重应该更高
        assert weights.iloc[-10:].mean() > weights.iloc[:10].mean()


# ============================================================================
# 市场状态权重测试
# ============================================================================

class TestMarketStateWeighting:
    """市场状态权重测试"""
    
    @pytest.fixture
    def sample_market_data(self):
        """生成测试市场数据"""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        # 生成价格序列
        prices = 3500 + np.cumsum(np.random.randn(n) * 10)
        
        data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(100000, 500000, n),
        }, index=dates)
        
        return data
    
    def test_market_state_weight_calculation(self, sample_market_data):
        """测试市场状态权重计算"""
        weighter = SampleWeighter()
        
        weights = weighter.calculate_market_state_weights(sample_market_data)
        
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_market_data)
        assert weights.min() >= 0
        assert weights.max() <= 1
    
    def test_trending_market_detection(self):
        """测试趋势市场检测"""
        weighter = SampleWeighter()
        
        n = 100
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        # 生成明确的上升趋势
        prices = 3500 + np.arange(n) * 5 + np.random.randn(n) * 1
        
        data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(100000, 500000, n),
        }, index=dates)
        
        weights = weighter.calculate_market_state_weights(data)
        
        # 应该能识别趋势
        assert weights is not None
        assert len(weights) == n


# ============================================================================
# 综合权重测试
# ============================================================================

class TestCombinedWeighting:
    """综合权重测试"""
    
    @pytest.fixture
    def sample_data(self):
        """生成测试数据"""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        prices = 3500 + np.cumsum(np.random.randn(n) * 10)
        
        data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(100000, 500000, n),
        }, index=dates)
        
        return data
    
    def test_combined_weights(self, sample_data):
        """测试综合权重"""
        weighter = SampleWeighter()
        
        weights = weighter.calculate_combined_weights(sample_data)
        
        assert isinstance(weights, pd.Series)
        assert len(weights) == len(sample_data)
        assert weights.min() >= 0
        assert weights.max() <= 1
    
    def test_weight_normalization(self, sample_data):
        """测试权重归一化"""
        weighter = SampleWeighter()
        
        weights = weighter.calculate_combined_weights(sample_data)
        
        # 权重应该在 0-1 之间
        assert (weights >= 0).all()
        assert (weights <= 1).all()
    
    def test_weight_application(self, sample_data):
        """测试权重应用"""
        weighter = SampleWeighter()
        
        # 创建因子值
        factor_values = pd.Series(np.random.randn(len(sample_data)), index=sample_data.index)
        
        # 获取权重
        weights = weighter.calculate_combined_weights(sample_data)
        
        # 应用权重
        weighted_factors = factor_values * weights
        
        assert len(weighted_factors) == len(factor_values)
        assert weighted_factors.notna().sum() > 0


# ============================================================================
# SampleWeighter 边界情况测试
# ============================================================================

class TestSampleWeighterEdgeCases:
    """边界情况测试"""
    
    def test_empty_data(self):
        """测试空数据"""
        weighter = SampleWeighter()
        
        empty_data = pd.DataFrame()
        
        # 应该优雅处理
        try:
            weights = weighter.calculate_volatility_weights(empty_data)
            assert len(weights) == 0
        except (ValueError, KeyError):
            pass
    
    def test_single_row_data(self):
        """测试单行数据"""
        weighter = SampleWeighter()
        
        data = pd.DataFrame({
            'close': [3500],
            'volume': [100000],
        }, index=pd.date_range('2020-01-01', periods=1, freq='B'))
        
        # 应该优雅处理
        try:
            weights = weighter.calculate_volatility_weights(data)
            assert len(weights) == 1
        except (ValueError, IndexError):
            pass
    
    def test_nan_handling(self):
        """测试 NaN 处理"""
        weighter = SampleWeighter()
        
        n = 100
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        data = pd.DataFrame({
            'close': [np.nan] * 50 + list(np.random.randn(50) * 10 + 3500),
            'volume': [np.nan] * 50 + list(np.random.randint(100000, 500000, 50)),
        }, index=dates)
        
        # 应该处理 NaN
        weights = weighter.calculate_volatility_weights(data)
        
        assert len(weights) == n
    
    def test_constant_values(self):
        """测试常数值"""
        weighter = SampleWeighter()
        
        n = 100
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        # 所有价格相同
        data = pd.DataFrame({
            'close': [3500] * n,
            'volume': [100000] * n,
        }, index=dates)
        
        # 应该优雅处理
        weights = weighter.calculate_volatility_weights(data)
        
        assert len(weights) == n
        # 波动率为 0 时权重应该相等或为 0
        assert weights.std() == 0 or weights.sum() == 0
