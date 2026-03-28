"""
样本价格数据生成器

提供生成真实感 OHLCV 数据的工具函数，供测试使用
"""
import numpy as np
import pandas as pd
from typing import List, Optional


def generate_ohlcv(
    n_days: int = 180,
    start_date: str = '2024-07-01',
    symbols: Optional[List[str]] = None,
    seed: int = 42
) -> pd.DataFrame:
    """
    生成标准 OHLCV 数据
    
    Args:
        n_days: 天数
        start_date: 开始日期
        symbols: 品种列表
        seed: 随机种子
        
    Returns:
        DataFrame, columns=['date','symbol','open','high','low','close','volume','open_interest']
    """
    np.random.seed(seed)
    
    if symbols is None:
        symbols = ['RB', 'HC', 'I', 'JM', 'ZC']
    
    base_prices = {'RB': 3800, 'HC': 3600, 'I': 850, 'JM': 1800, 'ZC': 800}
    start_ts = pd.Timestamp(start_date)
    
    rows = []
    for symbol in symbols:
        base = base_prices.get(symbol, 3000)
        # 生成带趋势和波动价格序列
        trend = np.linspace(0, 0.03 * (np.random.rand() - 0.3), n_days)
        noise = np.cumsum(np.random.randn(n_days) * base * 0.012)
        price = base * (1 + trend) + noise
        price = np.clip(price, base * 0.6, base * 1.8)
        
        weekday_mask = pd.bdate_range(start=start_date, periods=n_days).weekday < 5
        dates = pd.bdate_range(start=start_date, periods=n_days).strftime('%Y-%m-%d').tolist()
        
        for i, date in enumerate(dates):
            if i >= len(price):
                continue
            close = round(float(price[i]), 2)
            dr = close * (0.005 + np.random.rand() * 0.01)
            open_px = round(close + (np.random.rand() - 0.5) * dr, 2)
            high_px = round(max(open_px, close) + np.random.rand() * dr * 0.4, 2)
            low_px = round(min(open_px, close) - np.random.rand() * dr * 0.4, 2)
            
            rows.append({
                'date': date,
                'symbol': symbol,
                'open': open_px,
                'high': high_px,
                'low': low_px,
                'close': close,
                'volume': int(np.random.randint(50000, 500000)),
                'open_interest': int(np.random.randint(100000, 800000)),
            })
    
    return pd.DataFrame(rows)


def generate_returns(n_days: int = 180, seed: int = 43) -> pd.Series:
    """
    生成模拟收益率序列
    
    Returns:
        Series，index为日期，value为收益率
    """
    np.random.seed(seed)
    dates = pd.bdate_range('2024-07-01', periods=n_days).strftime('%Y-%m-%d').tolist()
    returns = pd.Series(
        np.random.randn(n_days) * 0.01 + 0.0002,
        index=dates,
        name='returns'
    )
    return returns
