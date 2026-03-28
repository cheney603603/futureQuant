"""
akshare 数据获取器封装

提供期货日线数据获取功能
"""

import time
from datetime import datetime, timedelta
from typing import List, Optional, Dict
import pandas as pd

from ...core.base import DataFetcher
from ...core.exceptions import FetchError
from ...core.logger import get_logger

logger = get_logger('data.fetcher.akshare')


class AKShareFetcher(DataFetcher):
    """akshare数据获取器"""
    
    def __init__(self, timeout: int = 30, retry: int = 3, delay: float = 1.0):
        """
        初始化
        
        Args:
            timeout: 请求超时时间
            retry: 重试次数
            delay: 请求间隔（秒）
        """
        self.timeout = timeout
        self.retry = retry
        self.delay = delay
        self._ak = None
        self._init_akshare()
    
    def _init_akshare(self):
        """初始化akshare"""
        try:
            import akshare as ak
            self._ak = ak
            logger.info("akshare initialized successfully")
        except ImportError:
            logger.error("akshare not installed, please run: pip install akshare")
            raise ImportError("akshare is required for AKShareFetcher")
    
    @property
    def name(self) -> str:
        return "akshare"
    
    def fetch_daily(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        获取期货日线数据
        
        Args:
            symbol: 合约代码，如 'RB2501'
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            DataFrame with columns: [date, open, high, low, close, volume, open_interest, amount]
        """
        for attempt in range(self.retry):
            try:
                # 转换日期格式
                start = pd.to_datetime(start_date).strftime('%Y%m%d')
                end = pd.to_datetime(end_date).strftime('%Y%m%d')
                
                # 获取数据
                df = self._ak.futures_zh_daily_sina(symbol=symbol)
                
                if df is None or df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return pd.DataFrame()
                
                # 标准化列名
                df = self._standardize(df)
                
                # 日期过滤
                df['date'] = pd.to_datetime(df['date'])
                mask = (df['date'] >= start_date) & (df['date'] <= end_date)
                df = df[mask].copy()
                
                logger.info(f"Fetched {len(df)} records for {symbol} from {start_date} to {end_date}")
                
                time.sleep(self.delay)  # 请求间隔
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.retry} failed for {symbol}: {e}")
                if attempt < self.retry - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    raise FetchError(f"Failed to fetch {symbol} after {self.retry} attempts: {e}")
        
        return pd.DataFrame()
    
    def fetch_symbols(self, variety: Optional[str] = None) -> List[str]:
        """
        获取可交易的合约列表
        
        Args:
            variety: 品种代码，如 'RB'，为None时返回所有
            
        Returns:
            合约代码列表
        """
        try:
            # 获取主力合约列表
            df = self._ak.futures_zh_realtime(symbol="主力")
            
            if df is None or df.empty:
                return []
            
            symbols = df['symbol'].tolist()
            
            if variety:
                # 过滤指定品种
                symbols = [s for s in symbols if s.startswith(variety.upper())]
            
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to fetch symbols: {e}")
            return []
    
    def fetch_main_contract(self, variety: str) -> str:
        """
        获取主力合约代码
        
        Args:
            variety: 品种代码，如 'RB'
            
        Returns:
            主力合约代码
        """
        try:
            df = self._ak.futures_zh_realtime(symbol=variety)
            if df is not None and not df.empty:
                return df.iloc[0]['symbol']
            return ""
        except Exception as e:
            logger.error(f"Failed to fetch main contract for {variety}: {e}")
            return ""
    
    def fetch_contract_list(self, variety: str) -> pd.DataFrame:
        """
        获取某品种的所有合约列表
        
        Args:
            variety: 品种代码，如 'RB'
            
        Returns:
            DataFrame with contract info
        """
        try:
            df = self._ak.futures_zh_realtime(symbol=variety)
            return df if df is not None else pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to fetch contract list for {variety}: {e}")
            return pd.DataFrame()
    
    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        标准化数据格式
        
        Args:
            df: 原始数据
            
        Returns:
            标准化后的数据
        """
        # 列名映射
        column_mapping = {
            'date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            '持仓量': 'open_interest',
            'open_interest': 'open_interest',
        }
        
        # 重命名列
        df = df.rename(columns=column_mapping)
        
        # 确保必要列存在
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
        
        # 数值类型转换
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 日期格式
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # 排序
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
