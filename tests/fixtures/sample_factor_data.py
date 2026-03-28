"""
样本因子数据生成器

提供生成因子测试数据的工具函数
"""
import numpy as np
import pandas as pd
from typing import Dict


def generate_factor_panel(
    n_days: int = 126,
    n_symbols: int = 5,
    symbols: list = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成横截面因子面板数据，MultiIndex (date, symbol)
    
    Args:
        n_days: 天数
        n_symbols: 品种数
        symbols: 品种代码列表
        seed: 随机种子
        
    Returns:
        DataFrame with MultiIndex (date, symbol), columns: momentum_20d, volatility_20d
    """
    np.random.seed(seed)
    
    if symbols is None:
        symbols = ['RB', 'HC', 'I', 'JM', 'ZC'][:n_symbols]
    
    dates = pd.bdate_range('2024-07-01', periods=n_days).strftime('%Y-%m-%d').tolist()
    n_days = len(dates)
    
    base_prices = {'RB': 3800, 'HC': 3600, 'I': 850, 'JM': 1800, 'ZC': 800}
    index = pd.MultiIndex.from_product([dates, symbols], names=['date', 'symbol'])
    
    records = []
    for date in dates:
        for sym in symbols:
            base = base_prices.get(sym, 3000)
            records.append({
                'date': date,
                'symbol': sym,
                'momentum_20d': float(np.random.randn() * 0.05),
                'volatility_20d': float(max(0.005, np.random.rand() * 0.03)),
                'returns_yf': float(np.random.randn() * 0.01),  # 次日收益率
            })
    
    df = pd.DataFrame(records)
    df = df.set_index(['date', 'symbol'])
    return df


def generate_factor_and_returns(
    n_days: int = 126,
    seed: int = 42
) -> tuple:
    """
    生成因子值和收益率配对数据
    
    Returns:
        (factor_df, returns_series)
    """
    np.random.seed(seed)
    dates = pd.bdate_range('2024-07-01', periods=n_days).strftime('%Y-%m-%d').tolist()
    
    # 因子值：与收益率有轻微正相关
    raw_returns = np.random.randn(n_days) * 0.01
    factor_vals = raw_returns * 0.8 + np.random.randn(n_days) * 0.01
    
    factor_df = pd.DataFrame({
        'momentum': factor_vals,
        'volatility': np.abs(np.random.randn(n_days) * 0.02),
    }, index=dates)
    
    # 收益率：与因子有相关性
    returns = pd.Series(raw_returns, index=dates, name='returns')
    
    return factor_df, returns
