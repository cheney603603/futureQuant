"""
pytest 配置文件

定义全局 fixtures、hooks 和配置。
"""

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import pytest

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 设置临时目录
TEMP_DIR = PROJECT_ROOT / 'tests' / 'temp'
TEMP_DIR.mkdir(exist_ok=True, parents=True)
os.environ['TMP'] = str(TEMP_DIR)
os.environ['TEMP'] = str(TEMP_DIR)

# 修补 tempfile
import tempfile as tf
tf.tempdir = str(TEMP_DIR)


# ============================================================================
# 配置选项
# ============================================================================

def pytest_configure(config):
    """pytest 配置钩子"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )


def pytest_collection_modifyitems(config, items):
    """修改测试集合"""
    # 为不需要网络的测试添加标记
    for item in items:
        if "real_data" in item.nodeid or "backtest" in item.nodeid:
            item.add_marker(pytest.mark.slow)


# ============================================================================
# Fixtures: 测试数据
# ============================================================================

@pytest.fixture
def sample_dates() -> pd.DatetimeIndex:
    """
    生成测试日期范围
    
    Returns:
        2020-01-01 到 2024-12-31 的交易日索引（约 1000 天）
    """
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')  # 工作日
    return dates


@pytest.fixture
def sample_price_data(sample_dates) -> pd.DataFrame:
    """
    生成模拟价格数据
    
    Args:
        sample_dates: 日期索引
        
    Returns:
        包含 OHLCV 数据的 DataFrame
    """
    np.random.seed(42)
    n = len(sample_dates)
    
    # 生成价格序列（带趋势和波动）
    returns = np.random.randn(n) * 0.02  # 2% 日波动率
    prices = 3500 * np.cumprod(1 + returns)  # 起始价格 3500
    
    # 生成 OHLCV 数据
    data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(n) * 0.005),
        'high': prices * (1 + np.abs(np.random.randn(n) * 0.01)),
        'low': prices * (1 - np.abs(np.random.randn(n) * 0.01)),
        'close': prices,
        'volume': np.random.randint(100000, 500000, n),
    }, index=sample_dates)
    
    # 确保 high >= max(open, close) 和 low <= min(open, close)
    data['high'] = data[['open', 'close', 'high']].max(axis=1)
    data['low'] = data[['open', 'close', 'low']].min(axis=1)
    
    return data


@pytest.fixture
def sample_returns(sample_price_data) -> pd.Series:
    """
    生成未来收益率数据
    
    Args:
        sample_price_data: 价格数据
        
    Returns:
        未来 1 日收益率序列
    """
    returns = sample_price_data['close'].pct_change().shift(-1)
    return returns


@pytest.fixture
def sample_factor_values(sample_dates) -> pd.DataFrame:
    """
    生成模拟因子值数据
    
    Args:
        sample_dates: 日期索引
        
    Returns:
        包含多个因子值的 DataFrame
    """
    np.random.seed(42)
    n = len(sample_dates)
    
    data = pd.DataFrame({
        'factor_1': np.random.randn(n),
        'factor_2': np.random.randn(n) * 0.5,
        'factor_3': np.random.randn(n) * 2,
    }, index=sample_dates)
    
    return data


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """
    生成多品种 OHLCV 测试数据（含 symbol 列）

    Returns:
        包含 RB、HC 两个品种、共 ~500 行的 OHLCV DataFrame
        columns: [symbol, date, open, high, low, close, volume, open_interest]
    """
    np.random.seed(42)
    records = []
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='B')
    for symbol, base_price in [('RB', 4200.0), ('HC', 3800.0)]:
        prices = base_price * np.cumprod(1 + np.random.randn(len(dates)) * 0.015)
        for i, date in enumerate(dates):
            close = prices[i]
            records.append({
                'symbol': symbol,
                'date': date.strftime('%Y-%m-%d'),
                'open': close * (1 + np.random.randn() * 0.003),
                'high': close * (1 + abs(np.random.randn() * 0.008)),
                'low': close * (1 - abs(np.random.randn() * 0.008)),
                'close': close,
                'volume': int(np.random.randint(80000, 400000)),
                'open_interest': int(np.random.randint(200000, 800000)),
            })
    df = pd.DataFrame(records)
    # 修正 high/low 确保 high >= max(open,close), low <= min(open,close)
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)
    return df


@pytest.fixture
def sample_macro_data(sample_dates) -> pd.DataFrame:
    """
    生成模拟宏观经济数据
    
    Args:
        sample_dates: 日期索引
        
    Returns:
        包含宏观指标的 DataFrame（月频）
    """
    # 生成月频数据
    months = pd.date_range(start='2020-01-01', end='2024-12-31', freq='MS')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'gdp_growth': np.random.uniform(0.01, 0.07, len(months)),
        'cpi_yoy': np.random.uniform(-0.01, 0.05, len(months)),
        'pmi': np.random.uniform(48, 55, len(months)),
        'interest_rate': np.random.uniform(0.02, 0.05, len(months)),
    }, index=months)
    
    return data


@pytest.fixture
def sample_fundamental_data(sample_dates) -> pd.DataFrame:
    """
    生成模拟基本面数据
    
    Args:
        sample_dates: 日期索引
        
    Returns:
        包含基本面指标的 DataFrame（季频）
    """
    # 生成季频数据
    quarters = pd.date_range(start='2020-01-01', end='2024-12-31', freq='QS')
    np.random.seed(42)
    
    data = pd.DataFrame({
        'revenue_growth': np.random.uniform(-0.1, 0.3, len(quarters)),
        'profit_margin': np.random.uniform(0.05, 0.25, len(quarters)),
        'debt_ratio': np.random.uniform(0.3, 0.7, len(quarters)),
    }, index=quarters)
    
    return data


# ============================================================================
# Fixtures: Agent 上下文
# ============================================================================

@pytest.fixture
def mining_context(sample_price_data, sample_returns) -> Dict[str, Any]:
    """
    创建因子挖掘上下文
    
    Args:
        sample_price_data: 价格数据
        sample_returns: 未来收益率
        
    Returns:
        挖掘上下文字典
    """
    return {
        'data': sample_price_data,
        'returns': sample_returns,
        'symbol': 'RB',
        'start_date': '2020-01-01',
        'end_date': '2024-12-31',
        'config': {
            'ic_threshold': 0.02,
            'momentum_windows': [5, 10, 20],
            'volatility_windows': [10, 20],
            'volume_windows': [5, 10],
            'rsi_windows': [6, 14],
        }
    }


@pytest.fixture
def validation_context(sample_factor_values, sample_returns) -> Dict[str, Any]:
    """
    创建验证上下文
    
    Args:
        sample_factor_values: 因子值
        sample_returns: 未来收益率
        
    Returns:
        验证上下文字典
    """
    return {
        'factors': sample_factor_values,
        'returns': sample_returns,
        'config': {
            'n_splits': 5,
            'lookahead_threshold': 0.03,
        }
    }


@pytest.fixture
def backtest_context(sample_price_data, sample_factor_values) -> Dict[str, Any]:
    """
    创建回测上下文
    
    Args:
        sample_price_data: 价格数据
        sample_factor_values: 因子值
        
    Returns:
        回测上下文字典
    """
    return {
        'data': sample_price_data,
        'factors': sample_factor_values,
        'symbol': 'RB',
        'initial_capital': 1000000,
        'config': {
            'stop_loss': 0.05,
            'take_profit': 0.10,
            'max_position': 0.2,
        }
    }


# ============================================================================
# Fixtures: 临时文件和目录
# ============================================================================

@pytest.fixture
def temp_dir():
    """
    创建临时目录（测试结束后自动删除）
    
    Yields:
        临时目录路径
    """
    temp_path = Path(tempfile.mkdtemp(dir=str(TEMP_DIR)))
    yield temp_path
    # 清理
    if temp_path.exists():
        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_db_path(temp_dir) -> Path:
    """
    创建临时数据库路径
    
    Args:
        temp_dir: 临时目录
        
    Returns:
        数据库文件路径
    """
    return temp_dir / 'test_factors.db'


@pytest.fixture
def temp_report_dir(temp_dir) -> Path:
    """
    创建临时报告目录
    
    Args:
        temp_dir: 临时目录
        
    Returns:
        报告目录路径
    """
    report_dir = temp_dir / 'reports'
    report_dir.mkdir(exist_ok=True)
    return report_dir


# ============================================================================
# Fixtures: Mock 对象
# ============================================================================

@pytest.fixture
def mock_factor():
    """
    创建模拟因子对象
    
    Returns:
        模拟因子实例
    """
    from unittest.mock import Mock
    from futureQuant.core.base import Factor
    
    factor = Mock(spec=Factor)
    factor.name = 'test_factor'
    factor.params = {'window': 10}
    factor.category = 'technical'
    factor.calculate.return_value = pd.Series(np.random.randn(100))
    factor.__str__ = lambda: f"Factor(name={factor.name}, params={factor.params})"
    
    return factor


@pytest.fixture
def mock_agent_result():
    """
    创建模拟 Agent 结果
    
    Returns:
        模拟 AgentResult 实例
    """
    from unittest.mock import Mock
    from futureQuant.agent.base import AgentResult, AgentStatus
    
    result = Mock(spec=AgentResult)
    result.agent_name = 'test_agent'
    result.status = AgentStatus.SUCCESS
    result.factors = []
    result.metrics = {'ic': 0.05}
    result.errors = []
    result.logs = []
    result.elapsed_seconds = 1.5
    result.is_success = True
    result.n_factors = 0
    
    return result


# ============================================================================
# Helper Functions
# ============================================================================

def assert_valid_agent_result(result, agent_name: str):
    """
    断言 AgentResult 有效
    
    Args:
        result: AgentResult 实例
        agent_name: 预期的 Agent 名称
    """
    from futureQuant.agent.base import AgentResult, AgentStatus
    
    assert isinstance(result, AgentResult)
    assert result.agent_name == agent_name
    assert isinstance(result.status, AgentStatus)
    assert isinstance(result.elapsed_seconds, float)
    assert result.elapsed_seconds >= 0
    assert isinstance(result.errors, list)
    assert isinstance(result.logs, list)


def create_test_factor(name: str, ic: float = 0.05) -> pd.Series:
    """
    创建测试因子序列
    
    Args:
        name: 因子名称
        ic: 目标 IC 值
        
    Returns:
        因子值序列
    """
    np.random.seed(42)
    n = 100
    factor_values = pd.Series(np.random.randn(n), name=name)
    return factor_values


# 导出 fixtures 供其他 conftest 使用
__all__ = [
    'sample_dates',
    'sample_price_data',
    'sample_returns',
    'sample_factor_values',
    'sample_macro_data',
    'sample_fundamental_data',
    'mining_context',
    'validation_context',
    'backtest_context',
    'temp_dir',
    'temp_db_path',
    'temp_report_dir',
    'mock_factor',
    'mock_agent_result',
    'assert_valid_agent_result',
    'create_test_factor',
]
