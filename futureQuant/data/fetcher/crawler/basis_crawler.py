"""
基差数据爬虫 - 从多个来源获取基差数据

基差 = 现货价格 - 期货价格
"""

from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import re

from .base import BaseCrawler
from ....core.logger import get_logger

logger = get_logger('data.fetcher.crawler.basis')


class BasisCrawler(BaseCrawler):
    """基差数据爬虫"""
    
    # 基差数据源
    SOURCES = {
        '100ppi': {
            'name': '生意社',
            'url': 'https://www.100ppi.com/sf/day-{}-{}.html',
        },
        'sina': {
            'name': '新浪财经',
            'url': 'https://finance.sina.com.cn/futures/quotes/{variety}.shtml',
        }
    }
    
    # 品种代码映射（从期货代码到现货网站代码）
    VARIETY_MAPPING = {
        'RB': {'100ppi': 'xianhuo', 'sina': '螺纹钢'},
        'HC': {'100ppi': 'rezha', 'sina': '热轧板卷'},
        'I': {'100ppi': 'tiekuang', 'sina': '铁矿石'},
        'J': {'100ppi': 'jiaotan', 'sina': '焦炭'},
        'JM': {'100ppi': 'jiaomei', 'sina': '焦煤'},
        'CU': {'100ppi': 'tong', 'sina': '铜'},
        'AL': {'100ppi': 'lv', 'sina': '铝'},
        'ZN': {'100ppi': 'xin', 'sina': '锌'},
        'NI': {'100ppi': 'nie', 'sina': '镍'},
        'SN': {'100ppi': 'xi', 'sina': '锡'},
        'AU': {'100ppi': 'huangjin', 'sina': '黄金'},
        'AG': {'100ppi': 'baiyin', 'sina': '白银'},
        'TA': {'100ppi': 'PTA', 'sina': 'PTA'},
        'MA': {'100ppi': 'jiaochun', 'sina': '甲醇'},
        'PP': {'100ppi': 'jubingxi', 'sina': '聚丙烯'},
        'L': {'100ppi': 'pe', 'sina': '聚乙烯'},
        'PVC': {'100ppi': 'pvc', 'sina': 'PVC'},
        'EG': {'100ppi': 'yierchun', 'sina': '乙二醇'},
        'M': {'100ppi': 'doupo', 'sina': '豆粕'},
        'Y': {'100ppi': 'douyou', 'sina': '豆油'},
        'P': {'100ppi': 'zonglvyou', 'sina': '棕榈油'},
        'OI': {'100ppi': 'caiyou', 'sina': '菜籽油'},
        'RM': {'100ppi': 'caipo', 'sina': '菜粕'},
        'CF': {'100ppi': 'mianhua', 'sina': '棉花'},
        'SR': {'100ppi': 'tang', 'sina': '白糖'},
        'RU': {'100ppi': 'xiangjiao', 'sina': '天然橡胶'},
        'SC': {'100ppi': 'yuanyou', 'sina': '原油'},
    }
    
    def __init__(self, delay: float = 3.0, **kwargs):
        super().__init__(delay=delay, **kwargs)
    
    def fetch_basis(
        self, 
        variety: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: str = '100ppi'
    ) -> pd.DataFrame:
        """
        获取基差数据
        
        Args:
            variety: 品种代码，如 'RB'
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            source: 数据源，'100ppi' 或 'sina'
            
        Returns:
            DataFrame with columns: [date, variety, spot_price, futures_price, basis, basis_rate]
        """
        variety = variety.upper()
        
        if source == '100ppi':
            return self._fetch_from_100ppi(variety, start_date, end_date)
        elif source == 'sina':
            return self._fetch_from_sina(variety, start_date, end_date)
        else:
            logger.error(f"Unknown source: {source}")
            return pd.DataFrame()
    
    def _fetch_from_100ppi(
        self, 
        variety: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """从生意社获取基差数据"""
        try:
            mapping = self.VARIETY_MAPPING.get(variety, {})
            ppi_code = mapping.get('100ppi')
            
            if not ppi_code:
                logger.warning(f"No 100ppi mapping for variety: {variety}")
                return pd.DataFrame()
            
            # 获取现货价格数据
            url = f"https://www.100ppi.com/sf/day-{ppi_code}.html"
            response = self.get(url)
            
            # 解析HTML表格
            dfs = pd.read_html(response.text)
            
            if not dfs:
                return pd.DataFrame()
            
            # 找到价格数据表（通常是第一个表格）
            df = dfs[0]
            
            # 处理列名
            if len(df.columns) >= 3:
                df.columns = ['date', 'spot_price', 'change'][:len(df.columns)]
            
            # 清理数据
            df = df.dropna(subset=['date', 'spot_price'])
            df['date'] = pd.to_datetime(df['date'])
            df['spot_price'] = pd.to_numeric(df['spot_price'], errors='coerce')
            df['variety'] = variety
            
            # 日期过滤
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]
            
            # 获取对应期货价格并计算基差
            # 这里简化处理，实际需要获取对应日期的期货价格
            df['futures_price'] = None  # 需要另外获取
            df['basis'] = None
            df['basis_rate'] = None
            
            logger.info(f"Fetched {len(df)} basis records for {variety} from 100ppi")
            return df[['date', 'variety', 'spot_price', 'futures_price', 'basis', 'basis_rate']]
            
        except Exception as e:
            logger.error(f"Failed to fetch basis from 100ppi for {variety}: {e}")
            return pd.DataFrame()
    
    def _fetch_from_sina(
        self, 
        variety: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """从新浪财经获取基差数据"""
        # 新浪财经基差数据获取逻辑较复杂，需要解析JS接口
        # 这里作为占位实现
        logger.info(f"Sina basis data fetch not yet fully implemented for {variety}")
        return pd.DataFrame()
    
    def fetch_basis_history(
        self, 
        variety: str,
        days: int = 252
    ) -> pd.DataFrame:
        """
        获取历史基差数据
        
        Args:
            variety: 品种代码
            days: 历史天数
            
        Returns:
            DataFrame with historical basis data
        """
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        return self.fetch_basis(variety, start_date, end_date)
    
    def calculate_basis(
        self, 
        spot_df: pd.DataFrame, 
        futures_df: pd.DataFrame,
        variety: str
    ) -> pd.DataFrame:
        """
        计算基差
        
        Args:
            spot_df: 现货价格数据
            futures_df: 期货价格数据
            variety: 品种代码
            
        Returns:
            DataFrame with basis calculation
        """
        # 合并数据
        df = pd.merge(
            spot_df[['date', 'close']].rename(columns={'close': 'spot_price'}),
            futures_df[['date', 'close']].rename(columns={'close': 'futures_price'}),
            on='date',
            how='inner'
        )
        
        # 计算基差
        df['basis'] = df['spot_price'] - df['futures_price']
        df['basis_rate'] = df['basis'] / df['futures_price'] * 100
        df['variety'] = variety
        
        return df[['date', 'variety', 'spot_price', 'futures_price', 'basis', 'basis_rate']]
