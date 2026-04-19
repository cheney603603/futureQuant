"""
多数据源获取器 - 带降级机制的包装器

P2.2 实现：
- 主数据源失败时自动切换备用源
- 支持 akshare、tushare、东方财富等
- 可配置优先级和重试策略
- 统一接口，调用方无感知

Author: futureQuant Team
Date: 2026-04-19
"""

import time
from datetime import datetime
from typing import Optional, Dict, List, Callable, Any
import pandas as pd

from futureQuant.core.base import DataFetcher
from futureQuant.core.logger import get_logger
from futureQuant.core.exceptions import FetchError

logger = get_logger('data.fetcher.multi')


class FallbackFetcher:
    """
    多数据源获取器（带降级机制）
    
    使用示例：
        fetcher = FallbackFetcher()
        df = fetcher.fetch_daily('RB', '2026-01-01', '2026-04-01')
        # 自动尝试多个数据源，直到成功
    """
    
    def __init__(
        self,
        primary: str = 'akshare',
        fallback_order: Optional[List[str]] = None,
        retry_count: int = 2,
        retry_delay: float = 1.0
    ):
        """
        初始化多数据源获取器
        
        Args:
            primary: 主数据源名称
            fallback_order: 备用数据源顺序，None 使用默认
            retry_count: 每个数据源重试次数
            retry_delay: 重试间隔（秒）
        """
        self.primary = primary
        self.fallback_order = fallback_order or ['akshare', 'tushare', 'eastmoney']
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        
        # 数据源实例缓存
        self._fetchers: Dict[str, Optional[DataFetcher]] = {}
        
        # 统计信息
        self._stats = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'fallbacks': 0,
            'by_source': {},
        }
        
        # 初始化主数据源
        self._init_fetcher(primary)
    
    def _init_fetcher(self, source_name: str) -> bool:
        """初始化指定数据源"""
        if source_name in self._fetchers:
            return self._fetchers[source_name] is not None
        
        try:
            if source_name == 'akshare':
                from .akshare_fetcher import AKShareFetcher
                self._fetchers[source_name] = AKShareFetcher()
                logger.info(f"Initialized {source_name} fetcher")
                return True
                
            elif source_name == 'tushare':
                # Tushare 需要 token，暂不实现
                logger.warning(f"{source_name} not implemented yet")
                self._fetchers[source_name] = None
                return False
                
            elif source_name == 'eastmoney':
                # 东方财富接口，暂不实现
                logger.warning(f"{source_name} not implemented yet")
                self._fetchers[source_name] = None
                return False
                
            else:
                logger.error(f"Unknown data source: {source_name}")
                self._fetchers[source_name] = None
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize {source_name}: {e}")
            self._fetchers[source_name] = None
            return False
    
    def _try_fetch(
        self,
        source_name: str,
        fetch_func: Callable,
        *args,
        **kwargs
    ) -> Optional[pd.DataFrame]:
        """尝试从指定数据源获取数据"""
        if not self._init_fetcher(source_name):
            return None
        
        fetcher = self._fetchers[source_name]
        if fetcher is None:
            return None
        
        for attempt in range(self.retry_count):
            try:
                result = fetch_func(fetcher, *args, **kwargs)
                
                if result is not None and not result.empty:
                    # 更新统计
                    self._stats['successes'] += 1
                    if source_name not in self._stats['by_source']:
                        self._stats['by_source'][source_name] = {'success': 0, 'fail': 0}
                    self._stats['by_source'][source_name]['success'] += 1
                    
                    logger.info(f"Successfully fetched from {source_name}")
                    return result
                    
            except Exception as e:
                logger.warning(f"{source_name} attempt {attempt + 1} failed: {e}")
                
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
        
        # 更新失败统计
        if source_name not in self._stats['by_source']:
            self._stats['by_source'][source_name] = {'success': 0, 'fail': 0}
        self._stats['by_source'][source_name]['fail'] += 1
        
        return None
    
    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        获取日频数据（带降级）
        
        Args:
            symbol: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame
            
        Raises:
            FetchError: 所有数据源都失败
        """
        self._stats['requests'] += 1
        
        # 尝试所有数据源
        for source_name in self.fallback_order:
            result = self._try_fetch(
                source_name,
                lambda f, s, sd, ed: f.fetch_daily(s, sd, ed),
                symbol, start_date, end_date
            )
            
            if result is not None:
                if source_name != self.primary:
                    self._stats['fallbacks'] += 1
                    logger.info(f"Used fallback source: {source_name}")
                return result
        
        # 所有数据源都失败
        self._stats['failures'] += 1
        raise FetchError(
            f"Failed to fetch daily data for {symbol} from all sources: "
            f"{self.fallback_order}"
        )
    
    def fetch_inventory(
        self,
        variety: str,
        **kwargs
    ) -> pd.DataFrame:
        """获取库存数据（带降级）"""
        self._stats['requests'] += 1
        
        # 基本面数据目前只有 akshare 支持
        for source_name in ['akshare']:
            result = self._try_fetch(
                source_name,
                lambda f, v: f.fetch_inventory(v) if hasattr(f, 'fetch_inventory') else pd.DataFrame(),
                variety
            )
            
            if result is not None:
                return result
        
        self._stats['failures'] += 1
        logger.warning(f"No fallback available for inventory data")
        return pd.DataFrame()
    
    def fetch_basis(
        self,
        variety: str,
        **kwargs
    ) -> pd.DataFrame:
        """获取基差数据（带降级）"""
        self._stats['requests'] += 1
        
        for source_name in ['akshare']:
            result = self._try_fetch(
                source_name,
                lambda f, v: f.fetch_basis(v) if hasattr(f, 'fetch_basis') else pd.DataFrame(),
                variety
            )
            
            if result is not None:
                return result
        
        self._stats['failures'] += 1
        logger.warning(f"No fallback available for basis data")
        return pd.DataFrame()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = self._stats['requests']
        success_rate = self._stats['successes'] / total if total > 0 else 0
        
        return {
            'total_requests': total,
            'successes': self._stats['successes'],
            'failures': self._stats['failures'],
            'fallbacks': self._stats['fallbacks'],
            'success_rate': round(success_rate, 4),
            'by_source': self._stats['by_source'],
        }
    
    def reset_stats(self):
        """重置统计"""
        self._stats = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'fallbacks': 0,
            'by_source': {},
        }


# 便捷函数

def fetch_daily_with_fallback(
    symbol: str,
    start_date: str,
    end_date: str,
    **kwargs
) -> pd.DataFrame:
    """使用降级机制获取日频数据"""
    fetcher = FallbackFetcher()
    return fetcher.fetch_daily(symbol, start_date, end_date, **kwargs)
