"""
未来函数检测器测试

测试 LookAheadDetector 的静态 AST 分析和动态 IC 延迟测试。
"""

import ast
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from futureQuant.agent.validators.lookahead_detector import (
    LookAheadPattern,
    LookAheadDetector,
)


# ============================================================================
# LookAheadPattern AST 分析测试
# ============================================================================

class TestLookAheadPattern:
    """LookAheadPattern AST 分析测试"""
    
    def test_detect_negative_shift(self):
        """检测负向 shift (未来数据)"""
        code = """
def calculate_factor(data):
    return data['close'].shift(-1)
"""
        detector = LookAheadPattern(code)
        tree = ast.parse(code)
        detector.visit(tree)
        
        assert len(detector.dangerous_calls) > 0
        assert any(call['type'] == 'negative_shift' for call in detector.dangerous_calls)
    
    def test_detect_positive_shift_safe(self):
        """检测正向 shift (安全)"""
        code = """
def calculate_factor(data):
    return data['close'].shift(1)
"""
        detector = LookAheadPattern(code)
        tree = ast.parse(code)
        detector.visit(tree)
        
        # 正向 shift 不应该被标记为危险
        negative_shifts = [call for call in detector.dangerous_calls if call['type'] == 'negative_shift']
        assert len(negative_shifts) == 0
    
    def test_detect_suspicious_names(self):
        """检测可疑变量名"""
        code = """
def calculate_factor(data):
    future_price = data['close'].shift(-1)
    return future_price
"""
        detector = LookAheadPattern(code)
        tree = ast.parse(code)
        detector.visit(tree)
        
        assert len(detector.suspicious_names) > 0
        assert 'future_price' in detector.suspicious_names
    
    def test_detect_suspicious_patterns(self):
        """检测可疑模式"""
        code = """
def get_next_close(data):
    return data['close'].shift(-1)
"""
        detector = LookAheadPattern(code)
        tree = ast.parse(code)
        detector.visit(tree)
        
        # 函数名包含 'next' 应该被标记
        suspicious_names = [call for call in detector.dangerous_calls if 'suspicious' in call['type']]
        assert len(suspicious_names) > 0
    
    def test_safe_code_no_warnings(self):
        """安全代码不应产生警告"""
        code = """
def calculate_momentum(data, window=10):
    return data['close'].pct_change(window)
"""
        detector = LookAheadPattern(code)
        tree = ast.parse(code)
        detector.visit(tree)
        
        # pct_change 是安全的，不应产生警告
        negative_shifts = [call for call in detector.dangerous_calls if call['type'] == 'negative_shift']
        assert len(negative_shifts) == 0
    
    def test_rolling_with_negative_shift(self):
        """检测 rolling 后接负向 shift"""
        code = """
def calculate_factor(data):
    return data['close'].rolling(10).mean().shift(-1)
"""
        detector = LookAheadPattern(code)
        tree = ast.parse(code)
        detector.visit(tree)
        
        assert len(detector.dangerous_calls) > 0
    
    def test_multiple_dangerous_patterns(self):
        """检测多个危险模式"""
        code = """
def calculate_factor(data):
    future_close = data['close'].shift(-1)
    next_return = data['close'].pct_change().shift(-2)
    return future_close + next_return
"""
        detector = LookAheadPattern(code)
        tree = ast.parse(code)
        detector.visit(tree)
        
        # 应该检测到多个危险调用
        assert len(detector.dangerous_calls) >= 2
    
    def test_get_dangerous_calls_summary(self):
        """测试获取危险调用摘要"""
        code = """
def calculate_factor(data):
    return data['close'].shift(-1)
"""
        detector = LookAheadPattern(code)
        tree = ast.parse(code)
        detector.visit(tree)
        
        summary = detector.get_dangerous_calls_summary()
        
        assert isinstance(summary, list)
        assert len(summary) > 0


# ============================================================================
# LookAheadDetector 初始化测试
# ============================================================================

class TestLookAheadDetectorInit:
    """LookAheadDetector 初始化测试"""
    
    def test_init_default(self):
        """测试默认初始化"""
        detector = LookAheadDetector()
        
        assert detector.config is not None
        assert 'ic_delay_threshold' in detector.config
    
    def test_init_custom_config(self):
        """测试自定义配置"""
        config = {
            'ic_delay_threshold': 0.05,
            'max_delay_days': 10,
        }
        detector = LookAheadDetector(config=config)
        
        assert detector.config['ic_delay_threshold'] == 0.05


# ============================================================================
# LookAheadDetector 静态检测测试
# ============================================================================

class TestLookAheadDetectorStatic:
    """LookAheadDetector 静态检测测试"""
    
    def test_static_check_safe_factor(self):
        """测试安全因子的静态检查"""
        detector = LookAheadDetector()
        
        # 创建安全因子
        factor = Mock()
        factor.name = 'safe_momentum'
        factor.calculate.__code__ = compile(
            "def calculate(data): return data['close'].pct_change(10)",
            '<string>', 'exec'
        ).co_consts[0]
        
        result = detector.static_check(factor)
        
        assert result['has_lookahead'] is False
        assert result['dangerous_calls'] == []
    
    def test_static_check_dangerous_factor(self):
        """测试危险因子的静态检查"""
        detector = LookAheadDetector()
        
        # 创建危险因子（使用负向 shift）
        def dangerous_calculate(data):
            return data['close'].shift(-1)
        
        factor = Mock()
        factor.name = 'dangerous_factor'
        factor.calculate = dangerous_calculate
        
        result = detector.static_check(factor)
        
        assert result['has_lookahead'] is True
        assert len(result['dangerous_calls']) > 0
    
    def test_static_check_with_source_code(self):
        """测试提供源代码的静态检查"""
        detector = LookAheadDetector()
        
        source = """
def calculate(data):
    return data['close'].shift(-1)
"""
        
        result = detector.static_check_from_source(source)
        
        assert result['has_lookahead'] is True


# ============================================================================
# LookAheadDetector 动态检测测试
# ============================================================================

class TestLookAheadDetectorDynamic:
    """LookAheadDetector 动态检测测试"""
    
    @pytest.fixture
    def sample_data_and_returns(self):
        """生成测试数据和收益率"""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        prices = 3500 + np.cumsum(np.random.randn(n) * 10)
        data = pd.DataFrame({
            'close': prices,
            'open': prices * (1 + np.random.randn(n) * 0.01),
            'high': prices * (1 + np.abs(np.random.randn(n) * 0.01)),
            'low': prices * (1 - np.abs(np.random.randn(n) * 0.01)),
            'volume': np.random.randint(100000, 500000, n),
        }, index=dates)
        
        returns = data['close'].pct_change().shift(-1)
        
        return data, returns
    
    def test_dynamic_check_safe_factor(self, sample_data_and_returns):
        """测试安全因子的动态检查"""
        data, returns = sample_data_and_returns
        detector = LookAheadDetector(config={'ic_delay_threshold': 0.05})
        
        # 创建安全因子
        def safe_calculate(df):
            return df['close'].pct_change(10)
        
        factor = Mock()
        factor.name = 'safe_factor'
        factor.calculate = safe_calculate
        
        result = detector.dynamic_check(factor, data, returns)
        
        assert 'has_lookahead' in result
        assert 'ic_original' in result
        assert 'ic_delayed' in result
    
    def test_dynamic_check_dangerous_factor(self, sample_data_and_returns):
        """测试危险因子的动态检查"""
        data, returns = sample_data_and_returns
        detector = LookAheadDetector(config={'ic_delay_threshold': 0.03})
        
        # 创建危险因子（使用未来数据）
        def dangerous_calculate(df):
            # 使用未来价格，会有高 IC
            return df['close'].shift(-1)
        
        factor = Mock()
        factor.name = 'dangerous_factor'
        factor.calculate = dangerous_calculate
        
        result = detector.dynamic_check(factor, data, returns)
        
        # 应该检测到未来函数
        assert 'has_lookahead' in result
    
    def test_compare_ic_original_vs_delayed(self, sample_data_and_returns):
        """测试原始 IC 与延迟 IC 比较"""
        data, returns = sample_data_and_returns
        detector = LookAheadDetector()
        
        # 创建因子
        def calculate(df):
            return df['close'].pct_change(10)
        
        factor = Mock()
        factor.calculate = calculate
        
        result = detector.dynamic_check(factor, data, returns)
        
        # 延迟 IC 应该记录
        assert 'ic_delayed' in result
        assert 'delay_days' in result


# ============================================================================
# LookAheadDetector 综合检测测试
# ============================================================================

class TestLookAheadDetectorComprehensive:
    """LookAheadDetector 综合检测测试"""
    
    @pytest.fixture
    def sample_data_and_returns(self):
        """生成测试数据"""
        np.random.seed(42)
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='B')
        
        prices = 3500 + np.cumsum(np.random.randn(n) * 10)
        data = pd.DataFrame({
            'close': prices,
            'volume': np.random.randint(100000, 500000, n),
        }, index=dates)
        
        returns = data['close'].pct_change().shift(-1)
        
        return data, returns
    
    def test_comprehensive_check(self, sample_data_and_returns):
        """测试综合检测"""
        data, returns = sample_data_and_returns
        detector = LookAheadDetector()
        
        factor = Mock()
        factor.name = 'test_factor'
        factor.calculate = lambda df: df['close'].pct_change(10)
        
        result = detector.comprehensive_check(factor, data, returns)
        
        assert 'static_check' in result
        assert 'dynamic_check' in result
        assert 'overall_risk' in result
    
    def test_risk_level_assessment(self, sample_data_and_returns):
        """测试风险评估"""
        data, returns = sample_data_and_returns
        detector = LookAheadDetector()
        
        # 安全因子
        safe_factor = Mock()
        safe_factor.name = 'safe'
        safe_factor.calculate = lambda df: df['close'].pct_change(10)
        
        result = detector.comprehensive_check(safe_factor, data, returns)
        assert result['overall_risk'] in ['low', 'medium', 'high']
    
    def test_batch_check(self, sample_data_and_returns):
        """测试批量检测"""
        data, returns = sample_data_and_returns
        detector = LookAheadDetector()
        
        # 创建多个因子
        factors = []
        for i in range(3):
            factor = Mock()
            factor.name = f'factor_{i}'
            factor.calculate = lambda df: df['close'].pct_change(10 * (i + 1))
            factors.append(factor)
        
        results = detector.batch_check(factors, data, returns)
        
        assert len(results) == 3
        assert all('has_lookahead' in r for r in results)


# ============================================================================
# LookAheadDetector 边界情况测试
# ============================================================================

class TestLookAheadDetectorEdgeCases:
    """LookAheadDetector 边界情况测试"""
    
    def test_empty_data(self):
        """测试空数据"""
        detector = LookAheadDetector()
        
        factor = Mock()
        factor.calculate = lambda df: df['close'].pct_change()
        
        empty_data = pd.DataFrame()
        empty_returns = pd.Series(dtype=float)
        
        # 应该优雅处理
        result = detector.dynamic_check(factor, empty_data, empty_returns)
        assert 'error' in result or result.get('has_lookahead') is False
    
    def test_data_with_nans(self):
        """测试包含 NaN 的数据"""
        detector = LookAheadDetector()
        
        np.random.seed(42)
        n = 100
        data = pd.DataFrame({
            'close': [np.nan] * 50 + list(np.random.randn(50) * 10 + 3500),
        }, index=pd.date_range('2020-01-01', periods=n, freq='B'))
        
        returns = data['close'].pct_change().shift(-1)
        
        factor = Mock()
        factor.calculate = lambda df: df['close'].pct_change(10)
        
        result = detector.dynamic_check(factor, data, returns)
        
        # 应该优雅处理 NaN
        assert 'ic_original' in result or 'error' in result
    
    def test_syntax_error_in_source(self):
        """测试源代码语法错误"""
        detector = LookAheadDetector()
        
        invalid_source = """
def calculate(data:
    return data['close']  # 语法错误
"""
        
        result = detector.static_check_from_source(invalid_source)
        
        assert 'error' in result
