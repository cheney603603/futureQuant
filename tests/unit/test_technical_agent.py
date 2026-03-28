"""
技术因子挖掘 Agent 测试

测试 TechnicalMiningAgent 的因子生成、IC 计算和筛选逻辑。
"""

from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from futureQuant.agent.base import AgentResult, AgentStatus
from futureQuant.agent.miners.technical_agent import TechnicalMiningAgent


# ============================================================================
# TechnicalMiningAgent 初始化测试
# ============================================================================

class TestTechnicalAgentInit:
    """TechnicalMiningAgent 初始化测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        agent = TechnicalMiningAgent()
        
        assert agent.name == 'technical_miner'
        assert agent.config['ic_threshold'] == 0.02
        assert agent.status == AgentStatus.IDLE
    
    def test_init_custom_name(self):
        """测试自定义名称"""
        agent = TechnicalMiningAgent(name='custom_tech')
        
        assert agent.name == 'custom_tech'
    
    def test_init_custom_config(self):
        """测试自定义配置"""
        custom_config = {
            'ic_threshold': 0.03,
            'momentum_windows': [5, 10],
            'volatility_windows': [10],
            'volume_windows': [5],
            'rsi_windows': [14],
        }
        agent = TechnicalMiningAgent(config=custom_config)
        
        assert agent.config['ic_threshold'] == 0.03
        assert agent.config['momentum_windows'] == [5, 10]
        assert agent.config['volatility_windows'] == [10]
    
    def test_init_partial_config(self):
        """测试部分配置覆盖"""
        partial_config = {
            'ic_threshold': 0.05,
        }
        agent = TechnicalMiningAgent(config=partial_config)
        
        # 覆盖的配置
        assert agent.config['ic_threshold'] == 0.05
        # 保留默认配置
        assert agent.config['momentum_windows'] == [5, 10, 20, 60, 120]


# ============================================================================
# TechnicalMiningAgent 因子生成测试
# ============================================================================

class TestTechnicalAgentFactorGeneration:
    """TechnicalMiningAgent 因子生成测试"""
    
    def test_generate_candidates_count(self):
        """测试候选因子数量"""
        agent = TechnicalMiningAgent()
        candidates = agent._generate_candidates()
        
        # 默认参数空间应该生成足够多的因子
        assert len(candidates) > 0
        # 动量因子: 5 个 momentum + 3 个 RSI + 1 个 MACD + 5 个 ROC = 14
        # 波动率因子: 3 个 ATR + 3 个 Volatility + 3 个 BB Width = 9
        # 成交量因子: 1 个 OBV + 3 个 VolumeRatio + 3 个 VolumeMA = 7
        # 总计: 30 个
        assert len(candidates) == 30
    
    def test_generate_candidates_with_custom_params(self):
        """测试自定义参数生成因子"""
        config = {
            'momentum_windows': [5, 10],
            'rsi_windows': [14],
            'volatility_windows': [10],
            'volume_windows': [5],
        }
        agent = TechnicalMiningAgent(config=config)
        candidates = agent._generate_candidates()
        
        # 更少的参数应该生成更少的因子
        assert len(candidates) > 0
        assert len(candidates) < 30  # 比默认少
    
    def test_generate_candidates_factor_types(self):
        """测试生成因子的类型"""
        agent = TechnicalMiningAgent()
        candidates = agent._generate_candidates()
        
        # 检查因子名称包含预期关键字
        factor_names = [f.name for f in candidates]
        
        # 应该有动量相关因子
        momentum_factors = [n for n in factor_names if 'momentum' in n.lower()]
        assert len(momentum_factors) > 0
        
        # 应该有波动率相关因子
        volatility_factors = [n for n in factor_names if 'volatility' in n.lower() or 'atr' in n.lower()]
        assert len(volatility_factors) > 0
        
        # 应该有成交量相关因子
        volume_factors = [n for n in factor_names if 'volume' in n.lower() or 'obv' in n.lower()]
        assert len(volume_factors) > 0


# ============================================================================
# TechnicalMiningAgent 执行测试
# ============================================================================

class TestTechnicalAgentExecution:
    """TechnicalMiningAgent 执行测试"""
    
    @pytest.fixture
    def mock_mining_context(self, sample_price_data, sample_returns):
        """创建模拟挖掘上下文"""
        context = Mock()
        context.data = sample_price_data
        context.returns = sample_returns
        return context
    
    def test_execute_success(self, mock_mining_context):
        """测试成功执行"""
        agent = TechnicalMiningAgent()
        result = agent.run({'context': mock_mining_context})
        
        assert result.is_success
        assert result.agent_name == 'technical_miner'
        assert 'total_candidates' in result.metrics
        assert 'selected_count' in result.metrics
        assert result.elapsed_seconds >= 0
    
    def test_execute_missing_context(self):
        """测试缺少上下文"""
        agent = TechnicalMiningAgent()
        result = agent.run({})
        
        assert result.status == AgentStatus.FAILED
        assert len(result.errors) > 0
        assert 'Missing' in result.errors[0]
    
    def test_execute_none_context(self):
        """测试 None 上下文"""
        agent = TechnicalMiningAgent()
        result = agent.run({'context': None})
        
        assert result.status == AgentStatus.FAILED
        assert len(result.errors) > 0
    
    def test_execute_ic_threshold_filtering(self, mock_mining_context):
        """测试 IC 阈值筛选"""
        # 使用较高的 IC 阈值
        config = {'ic_threshold': 0.10}
        agent = TechnicalMiningAgent(config=config)
        result = agent.run({'context': mock_mining_context})
        
        assert result.is_success
        # 高阈值应该筛选掉更多因子
        assert result.metrics['selected_count'] <= result.metrics['total_candidates']
    
    def test_execute_factor_metrics(self, mock_mining_context):
        """测试因子指标记录"""
        agent = TechnicalMiningAgent()
        result = agent.run({'context': mock_mining_context})
        
        assert result.is_success
        assert 'factor_metrics' in result.metrics
        assert isinstance(result.metrics['factor_metrics'], dict)
    
    def test_execute_factor_dataframe(self, mock_mining_context):
        """测试因子值 DataFrame"""
        agent = TechnicalMiningAgent()
        result = agent.run({'context': mock_mining_context})
        
        assert result.is_success
        if result.n_factors > 0:
            assert result.data is not None
            assert isinstance(result.data, pd.DataFrame)
            assert result.data.shape[1] == result.n_factors


# ============================================================================
# TechnicalMiningAgent IC 计算测试
# ============================================================================

class TestTechnicalAgentICCalculation:
    """TechnicalMiningAgent IC 计算测试"""
    
    @pytest.fixture
    def mock_mining_context_with_known_pattern(self):
        """创建具有已知模式的上下文"""
        # 创建有明确趋势的数据
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        # 价格有上升趋势
        prices = 3500 + np.cumsum(np.random.randn(n) * 0.5 + 0.1)
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.randn(n) * 0.005),
            'high': prices * (1 + np.abs(np.random.randn(n) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(n) * 0.01)),
            'close': prices,
            'volume': np.random.randint(100000, 500000, n),
        }, index=dates)
        
        # 确保 high >= max(open, close) 和 low <= min(open, close)
        data['high'] = data[['open', 'close', 'high']].max(axis=1)
        data['low'] = data[['open', 'close', 'low']].min(axis=1)
        
        # 未来收益率
        returns = data['close'].pct_change().shift(-1)
        
        context = Mock()
        context.data = data
        context.returns = returns
        
        return context
    
    def test_ic_calculation_with_trending_data(self, mock_mining_context_with_known_pattern):
        """测试趋势数据的 IC 计算"""
        agent = TechnicalMiningAgent(config={'ic_threshold': 0.01})
        result = agent.run({'context': mock_mining_context_with_known_pattern})
        
        assert result.is_success
        # 应该能找到一些有效因子
        assert result.metrics['total_candidates'] > 0
    
    def test_ic_threshold_effect(self, mock_mining_context_with_known_pattern):
        """测试 IC 阈值的影响"""
        # 低阈值
        agent_low = TechnicalMiningAgent(config={'ic_threshold': 0.001})
        result_low = agent_low.run({'context': mock_mining_context_with_known_pattern})
        
        # 高阈值
        agent_high = TechnicalMiningAgent(config={'ic_threshold': 0.10})
        result_high = agent_high.run({'context': mock_mining_context_with_known_pattern})
        
        # 低阈值应该选中更多因子
        assert result_low.metrics['selected_count'] >= result_high.metrics['selected_count']


# ============================================================================
# TechnicalMiningAgent 异常处理测试
# ============================================================================

class TestTechnicalAgentErrorHandling:
    """TechnicalMiningAgent 异常处理测试"""
    
    def test_execute_with_empty_data(self):
        """测试空数据"""
        context = Mock()
        context.data = pd.DataFrame()
        context.returns = pd.Series(dtype=float)
        
        agent = TechnicalMiningAgent()
        result = agent.run({'context': context})
        
        # 应该优雅处理
        assert result.status in [AgentStatus.SUCCESS, AgentStatus.FAILED]
    
    def test_execute_with_nan_data(self):
        """测试包含 NaN 的数据"""
        np.random.seed(42)
        n = 100
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        # 数据包含 NaN
        data = pd.DataFrame({
            'open': np.nan,
            'high': np.nan,
            'low': np.nan,
            'close': np.nan,
            'volume': np.nan,
        }, index=dates)
        
        # 部分非 NaN
        data.iloc[50:80] = np.random.randn(30, 5)
        
        context = Mock()
        context.data = data
        context.returns = pd.Series(np.random.randn(n), index=dates)
        
        agent = TechnicalMiningAgent()
        result = agent.run({'context': context})
        
        # 应该优雅处理 NaN
        assert result.status in [AgentStatus.SUCCESS, AgentStatus.FAILED]
    
    def test_execute_factor_computation_error(self):
        """测试因子计算错误"""
        context = Mock()
        context.data = pd.DataFrame({'close': [1, 2, 3]}, index=pd.date_range('2020-01-01', periods=3, freq='B'))
        context.returns = pd.Series([0.01, 0.02, np.nan], index=context.data.index)
        
        agent = TechnicalMiningAgent()
        result = agent.run({'context': context})
        
        # 应该处理数据不足的情况
        assert result.status in [AgentStatus.SUCCESS, AgentStatus.FAILED]


# ============================================================================
# TechnicalMiningAgent 集成测试
# ============================================================================

class TestTechnicalAgentIntegration:
    """TechnicalMiningAgent 集成测试"""
    
    def test_full_pipeline(self, mining_context):
        """测试完整流程"""
        agent = TechnicalMiningAgent(
            name='test_tech',
            config={
                'ic_threshold': 0.02,
                'momentum_windows': [5, 10, 20],
                'volatility_windows': [10, 20],
                'volume_windows': [5, 10],
                'rsi_windows': [6, 14],
            }
        )
        
        # 运行
        result = agent.run({'context': mining_context})
        
        # 验证结果
        assert result.is_success
        assert result.agent_name == 'test_tech'
        assert result.metrics['total_candidates'] > 0
        
        # 验证历史记录
        history = agent.get_history()
        assert len(history) == 1
        assert history[0] == result
    
    def test_multiple_runs(self, mining_context):
        """测试多次运行"""
        agent = TechnicalMiningAgent()
        
        result1 = agent.run({'context': mining_context})
        result2 = agent.run({'context': mining_context})
        
        assert result1.is_success
        assert result2.is_success
        assert len(agent.get_history()) == 2
    
    def test_reset_and_run(self, mining_context):
        """测试重置后运行"""
        agent = TechnicalMiningAgent()
        
        # 第一次运行
        agent.run({'context': mining_context})
        assert len(agent.get_history()) == 1
        
        # 重置
        agent.reset()
        assert agent.status == AgentStatus.IDLE
        assert len(agent.get_history()) == 0
        
        # 再次运行
        agent.run({'context': mining_context})
        assert len(agent.get_history()) == 1
