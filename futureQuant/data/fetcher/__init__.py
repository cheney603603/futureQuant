"""
data.fetcher 模块 - 数据获取

包含：
- akshare_fetcher: akshare接口封装（期货价格数据）
- fundamental_fetcher: 基本面数据获取（库存、基差、仓单）
- fallback_fetcher: 多数据源降级获取器 P2.2
- crawler: 爬虫模块（交易所、基差、库存）- 框架代码
"""

from .akshare_fetcher import AKShareFetcher

try:
    from .fundamental_fetcher import FundamentalFetcher
    _has_fundamental = True
except ImportError:
    _has_fundamental = False

# P2.2: 多数据源降级
try:
    from .fallback_fetcher import FallbackFetcher, fetch_daily_with_fallback
    _has_fallback = True
except ImportError:
    _has_fallback = False

try:
    from .crawler import ExchangeCrawler, BasisCrawler, InventoryCrawler
    _has_crawler = True
except ImportError:
    _has_crawler = False

__all__ = ['AKShareFetcher']

if _has_fundamental:
    __all__.append('FundamentalFetcher')

if _has_fallback:
    __all__.extend(['FallbackFetcher', 'fetch_daily_with_fallback'])

if _has_crawler:
    __all__.extend(['ExchangeCrawler', 'BasisCrawler', 'InventoryCrawler'])
