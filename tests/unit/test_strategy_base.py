"""
test_strategy_base.py - BaseStrategy 单元测试

测试内容：
1. BaseStrategy.validate_params() 对无效参数返回 False
2. calculate_position_size() 返回合理的手数（> 0）
3. generate_signals() 返回 DataFrame 且有 signal 列，值为 -1/0/1
"""
import pytest
import pandas as pd
import numpy as np

pytest.importorskip("futureQuant")

from futureQuant.strategy.base import BaseStrategy, SignalType, PositionSide
from futureQuant.core.exceptions import StrategyError


# =============================================================================
# 测试用策略实现
# =============================================================================

class DummyStrategy(BaseStrategy):
    """测试用简单策略"""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """简单策略：收盘价高于均线做多，低于做空"""
        if 'close' not in data.columns:
            return pd.DataFrame()
        
        ma = data['close'].rolling(5).mean()
        signal = np.where(data['close'] > ma, 1, -1)
        
        result = pd.DataFrame({
            'date': data.index,
            'signal': signal,
            'weight': 1.0,
            'confidence': 0.8,
        })
        
        # 前几行无法计算均线，设为 0
        result.loc[:4, 'signal'] = 0
        result.loc[:4, 'weight'] = 0
        
        return result


class MeanReversionStrategy(BaseStrategy):
    """均值回归策略"""
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        if 'close' not in data.columns:
            return pd.DataFrame()
        
        ma = data['close'].rolling(20).mean()
        std = data['close'].rolling(20).std()
        
        z_score = (data['close'] - ma) / std
        
        # z > 1 做空，z < -1 做多，否则空仓
        signal = np.where(z_score > 1, -1, 0)
        signal = np.where(z_score < -1, 1, signal)
        
        result = pd.DataFrame({
            'date': data.index,
            'signal': signal,
            'weight': 0.5,
        })
        
        # 初始行无法计算
        result.loc[:19, 'signal'] = 0
        result.loc[:19, 'weight'] = 0
        
        return result


# =============================================================================
# 测试用例
# =============================================================================

class TestValidateParams:
    """测试参数验证"""
    
    def test_validate_params_valid(self):
        """有效参数返回 True"""
        strategy = DummyStrategy(
            name='test',
            stop_loss=0.05,
            take_profit=0.10,
            max_position=0.8,
            risk_per_trade=0.02,
        )
        
        assert strategy.validate_params() is True
    
    def test_validate_params_invalid_stop_loss(self):
        """无效止损参数返回 False"""
        strategy = DummyStrategy(
            name='test',
            stop_loss=0.0,  # 必须 > 0
        )
        
        assert strategy.validate_params() is False
    
    def test_validate_params_invalid_take_profit(self):
        """无效止盈参数返回 False"""
        strategy = DummyStrategy(
            name='test',
            take_profit=-0.1,  # 必须 > 0
        )
        
        assert strategy.validate_params() is False
    
    def test_validate_params_invalid_max_position(self):
        """无效最大仓位返回 False"""
        strategy = DummyStrategy(
            name='test',
            max_position=1.5,  # 必须在 0-1 之间
        )
        
        assert strategy.validate_params() is False
    
    def test_validate_params_invalid_risk_per_trade(self):
        """无效单笔风险返回 False"""
        strategy = DummyStrategy(
            name='test',
            risk_per_trade=0.0,  # 必须在 0-1 之间
        )
        
        assert strategy.validate_params() is False
    
    def test_validate_params_none_values(self):
        """None 值参数通过验证（可选参数）"""
        strategy = DummyStrategy(
            name='test',
            stop_loss=None,
            take_profit=None,
        )
        
        # None 表示不使用止损/止盈，应通过验证
        assert strategy.validate_params() is True


class TestCalculatePositionSize:
    """测试仓位计算"""
    
    def test_calculate_position_size_with_stop_loss(self):
        """带止损的仓位计算"""
        strategy = DummyStrategy(
            name='test',
            stop_loss=0.05,
            risk_per_trade=0.02,
        )
        
        position = strategy.calculate_position_size(
            capital=1000000,
            price=3800,
        )
        
        # 手数应 > 0
        assert position > 0
        assert isinstance(position, int)
    
    def test_calculate_position_size_with_atr(self):
        """ATR 法的仓位计算"""
        strategy = DummyStrategy(
            name='test',
            risk_per_trade=0.02,
        )
        
        position = strategy.calculate_position_size(
            capital=1000000,
            price=3800,
            atr=50.0,
        )
        
        assert position > 0
    
    def test_calculate_position_size_with_volatility(self):
        """波动率法的仓位计算"""
        strategy = DummyStrategy(
            name='test',
            risk_per_trade=0.02,
        )
        
        position = strategy.calculate_position_size(
            capital=1000000,
            price=3800,
            volatility=0.02,
        )
        
        assert position > 0
    
    def test_calculate_position_size_default(self):
        """默认仓位计算"""
        strategy = DummyStrategy(name='test')
        
        position = strategy.calculate_position_size(
            capital=1000000,
            price=3800,
        )
        
        assert position >= 0
        assert isinstance(position, int)
    
    def test_calculate_position_size_small_capital(self):
        """资金较少时手数合理"""
        strategy = DummyStrategy(
            name='test',
            stop_loss=0.05,
            risk_per_trade=0.02,
        )
        
        position = strategy.calculate_position_size(
            capital=1000,  # 很少的资金
            price=3800,
        )
        
        # 应该返回 0 而不是负数
        assert position >= 0


class TestGenerateSignals:
    """测试信号生成"""
    
    def test_generate_signals_returns_dataframe(self, sample_ohlcv):
        """generate_signals() 返回 DataFrame"""
        strategy = DummyStrategy(name='test')
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        signals = strategy.generate_signals(rb_data)
        
        assert isinstance(signals, pd.DataFrame)
    
    def test_generate_signals_has_signal_column(self, sample_ohlcv):
        """返回 DataFrame 有 signal 列"""
        strategy = DummyStrategy(name='test')
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        signals = strategy.generate_signals(rb_data)
        
        assert 'signal' in signals.columns
    
    def test_generate_signals_values_in_range(self, sample_ohlcv):
        """signal 值为 -1/0/1"""
        strategy = DummyStrategy(name='test')
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        signals = strategy.generate_signals(rb_data)
        
        valid_signals = signals['signal'].unique()
        assert all(s in [-1, 0, 1] for s in valid_signals)
    
    def test_generate_signals_length_matches_input(self, sample_ohlcv):
        """信号长度与输入数据一致"""
        strategy = DummyStrategy(name='test')
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        signals = strategy.generate_signals(rb_data)
        
        assert len(signals) == len(rb_data)
    
    def test_generate_signals_has_weight_column(self, sample_ohlcv):
        """返回 DataFrame 有 weight 列"""
        strategy = DummyStrategy(name='test')
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        signals = strategy.generate_signals(rb_data)
        
        assert 'weight' in signals.columns
    
    def test_mean_reversion_strategy(self, sample_ohlcv):
        """均值回归策略信号"""
        strategy = MeanReversionStrategy(name='test')
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        signals = strategy.generate_signals(rb_data)
        
        assert 'signal' in signals.columns
        assert all(s in [-1, 0, 1] for s in signals['signal'].unique())


class TestApplyRiskManagement:
    """测试风险管理"""
    
    def test_apply_risk_management_threshold(self, sample_ohlcv):
        """信号阈值过滤"""
        strategy = MeanReversionStrategy(
            name='test',
            signal_threshold=0.5,
        )
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        signals = strategy.generate_signals(rb_data)
        managed = strategy.apply_risk_management(signals)
        
        assert isinstance(managed, pd.DataFrame)
    
    def test_apply_risk_management_empty_input(self):
        """空输入返回空 DataFrame"""
        strategy = DummyStrategy(name='test')
        
        empty_df = pd.DataFrame()
        result = strategy.apply_risk_management(empty_df)
        
        assert result.empty


class TestStrategyStats:
    """测试策略统计"""
    
    def test_get_stats(self):
        """获取策略统计"""
        strategy = DummyStrategy(name='test', stop_loss=0.05)
        
        stats = strategy.get_stats()
        
        assert 'name' in stats
        assert stats['name'] == 'test'
        assert 'params' in stats
        assert 'trade_count' in stats
    
    def test_reset(self, sample_ohlcv):
        """重置策略状态"""
        strategy = DummyStrategy(name='test')
        
        rb_data = sample_ohlcv[sample_ohlcv['symbol'] == 'RB'].copy()
        rb_data['date'] = pd.to_datetime(rb_data['date'])
        rb_data = rb_data.set_index('date').sort_index()
        
        strategy.generate_signals(rb_data)
        
        assert strategy.signals is not None
        
        strategy.reset()
        
        assert strategy.signals is None


class TestSignalType:
    """测试信号类型枚举"""
    
    def test_signal_type_values(self):
        """枚举值正确"""
        assert SignalType.LONG.value == 1
        assert SignalType.SHORT.value == -1
        assert SignalType.FLAT.value == 0
    
    def test_position_side_values(self):
        """持仓方向枚举"""
        assert PositionSide.LONG.value == 1
        assert PositionSide.SHORT.value == -1
        assert PositionSide.FLAT.value == 0


class TestStrategyRepr:
    """测试策略字符串表示"""
    
    def test_repr(self):
        """__repr__ 返回有意义字符串"""
        strategy = DummyStrategy(name='TestStrategy')
        
        repr_str = repr(strategy)
        
        assert 'DummyStrategy' in repr_str
        assert 'TestStrategy' in repr_str
