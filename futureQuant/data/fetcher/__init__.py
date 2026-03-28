"""
data.fetcher 模块 - 数据获取

包含：
- akshare_fetcher: akshare接口封装
- crawler: 爬虫模块（交易所、基差、库存）
"""

from .akshare_fetcher import AKShareFetcher

try:
    from .crawler import ExchangeCrawler, BasisCrawler, InventoryCrawler
    __all__ = ['AKShareFetcher', 'ExchangeCrawler', 'BasisCrawler', 'InventoryCrawler']
except ImportError:
    __all__ = ['AKShareFetcher']
