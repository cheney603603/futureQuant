"""
交易所官网爬虫 - 获取仓单、交割等数据

优先从各交易所官网获取数据，稳定性更高
"""

from datetime import datetime
from typing import List, Optional, Dict
import pandas as pd

from .base import BaseCrawler
from ....core.logger import get_logger

logger = get_logger('data.fetcher.crawler.exchange')


class ExchangeCrawler(BaseCrawler):
    """交易所官网数据爬虫"""
    
    # 交易所配置
    EXCHANGES = {
        'SHFE': {
            'name': '上海期货交易所',
            'warehouse_url': 'http://www.shfe.com.cn/data/dailydata/{exchange}weeklystock.dat',
        },
        'DCE': {
            'name': '大连商品交易所',
            'warehouse_url': 'http://www.dce.com.cn/publicweb/quotesdata/wbillWeeklyQuotes.html',
        },
        'CZCE': {
            'name': '郑州商品交易所',
            'warehouse_url': 'http://www.czce.com.cn/cn/DFSStaticFiles/Future/{year}/{date}/FutureDataWhsheet.htm',
        },
        'INE': {
            'name': '上海国际能源交易中心',
            'warehouse_url': 'http://www.ine.cn/data/dailydata/{exchange}weeklystock.dat',
        },
        'GFEX': {
            'name': '广州期货交易所',
            'warehouse_url': '',  # 待补充
        },
        'CFFEX': {
            'name': '中国金融期货交易所',
            'warehouse_url': '',  # 金融期货一般没有仓单
        }
    }
    
    # 品种到交易所的映射
    VARIETY_EXCHANGE = {
        'CU': 'SHFE', 'AL': 'SHFE', 'ZN': 'SHFE', 'PB': 'SHFE', 'NI': 'SHFE', 'SN': 'SHFE',
        'AU': 'SHFE', 'AG': 'SHFE', 'RB': 'SHFE', 'WR': 'SHFE', 'HC': 'SHFE', 'SS': 'SHFE',
        'BU': 'SHFE', 'RU': 'SHFE', 'NR': 'SHFE', 'SP': 'SHFE', 'F': 'SHFE',
        'C': 'DCE', 'CS': 'DCE', 'A': 'DCE', 'B': 'DCE', 'M': 'DCE', 'Y': 'DCE',
        'P': 'DCE', 'FB': 'DCE', 'BB': 'DCE', 'JD': 'DCE', 'L': 'DCE', 'V': 'DCE',
        'PP': 'DCE', 'J': 'DCE', 'JM': 'DCE', 'I': 'DCE', 'EG': 'DCE', 'EB': 'DCE',
        'PG': 'DCE', 'LH': 'DCE', 'RR': 'DCE',
        'WH': 'CZCE', 'PM': 'CZCE', 'CF': 'CZCE', 'SR': 'CZCE', 'TA': 'CZCE', 'OI': 'CZCE',
        'RI': 'CZCE', 'MA': 'CZCE', 'FG': 'CZCE', 'RS': 'CZCE', 'RM': 'CZCE', 'ZC': 'CZCE',
        'JR': 'CZCE', 'LR': 'CZCE', 'SF': 'CZCE', 'SM': 'CZCE', 'CY': 'CZCE', 'AP': 'CZCE',
        'CJ': 'CZCE', 'UR': 'CZCE', 'SA': 'CZCE', 'PF': 'CZCE', 'PK': 'CZCE',
        'SC': 'INE', 'LU': 'INE', 'NR': 'INE', 'BC': 'INE', 'EC': 'INE',
        'SI': 'GFEX', 'LC': 'GFEX',
    }
    
    def __init__(self, delay: float = 3.0, **kwargs):
        super().__init__(delay=delay, **kwargs)
    
    def get_exchange(self, variety: str) -> str:
        """获取品种所属交易所"""
        return self.VARIETY_EXCHANGE.get(variety.upper(), 'UNKNOWN')
    
    def fetch_warehouse_receipts(
        self, 
        variety: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取仓单数据
        
        Args:
            variety: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with columns: [date, variety, warehouse, receipt, change]
        """
        exchange = self.get_exchange(variety)
        
        if exchange == 'SHFE' or exchange == 'INE':
            return self._fetch_shfe_warehouse(variety, start_date, end_date)
        elif exchange == 'DCE':
            return self._fetch_dce_warehouse(variety, start_date, end_date)
        elif exchange == 'CZCE':
            return self._fetch_czce_warehouse(variety, start_date, end_date)
        else:
            logger.warning(f"Unsupported exchange: {exchange} for variety {variety}")
            return pd.DataFrame()
    
    def _fetch_shfe_warehouse(
        self, 
        variety: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取上期所仓单数据"""
        try:
            # 上期所提供JSON接口
            url = f"http://www.shfe.com.cn/data/dailydata/{variety.lower()}weeklystock.dat"
            response = self.get(url)
            data = response.json()
            
            if 'o_cursor' not in data:
                return pd.DataFrame()
            
            records = []
            for item in data['o_cursor']:
                record = {
                    'date': item.get('DATE'),
                    'variety': variety.upper(),
                    'warehouse': item.get('WHABBRNAME', ''),
                    'receipt': int(item.get('WRTWGHTS', 0)),
                    'change': int(item.get('WRTCHANGE', 0)),
                }
                records.append(record)
            
            df = pd.DataFrame(records)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                
                # 日期过滤
                if start_date:
                    df = df[df['date'] >= start_date]
                if end_date:
                    df = df[df['date'] <= end_date]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch SHFE warehouse receipts for {variety}: {e}")
            return pd.DataFrame()
    
    def _fetch_dce_warehouse(
        self, 
        variety: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取大商所仓单数据"""
        try:
            # 大商所需要POST请求
            url = "http://www.dce.com.cn/publicweb/quotesdata/wbillWeeklyQuotes.html"
            
            # 获取最新数据
            payload = {
                'wbillWeeklyQuotes.variety': variety.upper(),
                'year': datetime.now().year,
                'month': datetime.now().month - 1,
                'day': datetime.now().day,
            }
            
            response = self.post(url, data=payload)
            # 解析HTML（简化版，实际需要根据页面结构调整）
            # 这里使用pandas读取HTML表格
            dfs = pd.read_html(response.text)
            
            if not dfs:
                return pd.DataFrame()
            
            df = dfs[0]
            # 处理列名和数据格式...
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch DCE warehouse receipts for {variety}: {e}")
            return pd.DataFrame()
    
    def _fetch_czce_warehouse(
        self, 
        variety: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取郑商所仓单数据"""
        try:
            # 郑商所仓单数据页面
            today = datetime.now()
            url = f"http://www.czce.com.cn/cn/DFSStaticFiles/Future/{today.year}/{today.strftime('%Y%m%d')}/FutureDataWhsheet.htm"
            
            response = self.get(url)
            dfs = pd.read_html(response.text)
            
            if not dfs:
                return pd.DataFrame()
            
            # 找到对应品种的数据表
            for df in dfs:
                if variety.upper() in str(df.values):
                    # 处理数据...
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Failed to fetch CZCE warehouse receipts for {variety}: {e}")
            return pd.DataFrame()
    
    def fetch_delivery_data(self, variety: str, year: Optional[int] = None) -> pd.DataFrame:
        """
        获取交割数据
        
        Args:
            variety: 品种代码
            year: 年份，默认当前年
            
        Returns:
            DataFrame with delivery data
        """
        # 各交易所交割数据获取逻辑
        # 这是一个占位实现
        logger.info(f"Delivery data fetch not yet implemented for {variety}")
        return pd.DataFrame()
