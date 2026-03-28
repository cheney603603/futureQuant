"""
数据管理器 - 统一数据入口

整合数据获取、存储、预处理功能，提供简洁的API
"""

from typing import List, Optional, Dict, Union, Literal
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from ..core.base import DataFetcher
from ..core.config import get_config
from ..core.logger import get_logger
from ..core.exceptions import DataError

from .fetcher.akshare_fetcher import AKShareFetcher
from .fetcher.crawler import ExchangeCrawler, BasisCrawler, InventoryCrawler
from .storage.db_manager import DBManager
from .processor.cleaner import DataCleaner
from .processor.contract_manager import ContractManager
from .processor.calendar import FuturesCalendar

logger = get_logger('data.manager')


class DataManager:
    """数据管理器 - 统一数据入口"""
    
    def __init__(
        self, 
        cache_dir: Optional[str] = None,
        db_path: Optional[str] = None,
        auto_update: bool = True
    ):
        """
        初始化数据管理器
        
        Args:
            cache_dir: 缓存目录
            db_path: 数据库路径
            auto_update: 是否自动更新数据
        """
        config = get_config()
        
        self.cache_dir = Path(cache_dir or config.data.cache_dir)
        self.db_path = db_path or config.data.db_path
        self.auto_update = auto_update
        
        # 初始化组件
        self.db = DBManager(self.db_path, self.cache_dir)
        self.cleaner = DataCleaner()
        self.contract_manager = ContractManager()
        self.calendar = FuturesCalendar()
        
        # 初始化数据获取器
        self.fetchers: Dict[str, DataFetcher] = {}
        self._init_fetchers()
    
    def _init_fetchers(self):
        """初始化数据获取器"""
        # akshare获取器
        try:
            self.fetchers['akshare'] = AKShareFetcher()
            logger.info("AKShare fetcher initialized")
        except ImportError:
            logger.warning("AKShare not available")
        
        # 爬虫获取器
        try:
            self.crawlers = {
                'exchange': ExchangeCrawler(),
                'basis': BasisCrawler(),
                'inventory': InventoryCrawler(),
            }
            logger.info("Crawlers initialized")
        except Exception as e:
            logger.warning(f"Some crawlers not available: {e}")
            self.crawlers = {}
    
    # ==================== 基础数据获取 ====================
    
    def get_daily_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: str = 'akshare',
        use_cache: bool = True,
        auto_clean: bool = True
    ) -> pd.DataFrame:
        """
        获取日线数据
        
        Args:
            symbol: 合约代码，如 'RB2501'
            start_date: 开始日期 (YYYY-MM-DD)，默认一年前
            end_date: 结束日期 (YYYY-MM-DD)，默认今天
            source: 数据源
            use_cache: 是否使用缓存
            auto_clean: 是否自动清洗数据
            
        Returns:
            DataFrame with columns: [date, open, high, low, close, volume, open_interest]
        """
        # 默认日期范围
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        # 尝试从缓存加载
        if use_cache:
            df = self.db.load_price_data(symbol, start_date, end_date)
            if not df.empty:
                logger.info(f"Loaded {len(df)} records from cache for {symbol}")
                return df
        
        # 从数据源获取
        fetcher = self.fetchers.get(source)
        if not fetcher:
            raise DataError(f"Unknown data source: {source}")
        
        df = fetcher.fetch_daily(symbol, start_date, end_date)
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}")
            return df
        
        # 数据清洗
        if auto_clean:
            df = self.cleaner.clean_ohlc(df)
        
        # 保存到缓存
        if use_cache:
            self.db.save_price_data(df, symbol)
            self.db.log_update('daily', symbol, start_date, end_date, len(df))
        
        return df
    
    def get_continuous_contract(
        self,
        variety: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        adjust_method: Literal['backward', 'forward', 'none'] = 'backward',
        rollover_method: Literal['volume', 'open_interest'] = 'open_interest',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取连续合约数据
        
        Args:
            variety: 品种代码，如 'RB'
            start_date: 开始日期
            end_date: 结束日期
            adjust_method: 复权方式
            rollover_method: 主力合约识别方法
            use_cache: 是否使用缓存
            
        Returns:
            连续合约DataFrame
        """
        cache_key = f"{variety}_CONTINUOUS_{adjust_method}"
        
        # 尝试从缓存加载
        if use_cache:
            df = self.db.load_price_data(cache_key, start_date, end_date)
            if not df.empty:
                logger.info(f"Loaded continuous contract from cache for {variety}")
                return df
        
        # 获取所有合约数据
        # 这里简化处理，实际应该获取该品种的所有合约
        symbols = self._get_variety_contracts(variety, start_date, end_date)
        
        if not symbols:
            logger.warning(f"No contracts found for variety {variety}")
            return pd.DataFrame()
        
        # 添加合约数据到管理器
        for symbol in symbols:
            df = self.get_daily_data(symbol, start_date, end_date, use_cache=True)
            if not df.empty:
                self.contract_manager.add_contract_data(symbol, df)
        
        # 创建连续合约
        continuous_df = self.contract_manager.create_continuous_contract(
            variety=variety,
            adjust_method=adjust_method,
            rollover_method=rollover_method
        )
        
        # 保存到缓存
        if use_cache and not continuous_df.empty:
            self.db.save_price_data(continuous_df, cache_key)
        
        return continuous_df
    
    def _get_variety_contracts(
        self, 
        variety: str, 
        start_date: str, 
        end_date: str
    ) -> List[str]:
        """
        获取某品种在日期范围内的所有历史合约代码。
        
        优先使用 akshare 交易所接口获取真实合约列表，
        如果失败则基于品种的活跃月份规律生成候选合约。

        Args:
            variety: 品种代码，如 'RB', 'I', 'JM'
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            合约代码列表
        """
        contracts = []
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # 方案一：从 akshare 获取主力合约历史列表
        try:
            akshare_fetcher = self.fetchers.get('akshare')
            if akshare_fetcher:
                # akshare 的 get_futures_major_symbol() 返回当前主力合约
                # 配合品种代码可推断历史合约，但无法直接获取历史主力序列
                # 这里退而求其次：尝试获取近期主力，再往前推
                logger.info(f"Trying to fetch contracts for {variety} from akshare")
                # 如果 akshare 有对应接口则调用，无则跳过
                major_symbols = getattr(akshare_fetcher, 'get_major_symbols', None)
                if major_symbols and callable(major_symbols):
                    major = major_symbols(variety)
                    if major:
                        contracts.append(major)
                        logger.info(f"Got major contract {major} for {variety}")
        except Exception as e:
            logger.warning(f"akshare fetch failed for {variety}: {e}")

        # 方案二：根据品种活跃月份规律生成候选合约
        # 不同期货品种的活跃月份不同，这里按大品种规律分组
        ACTIVE_MONTHS_BY_VARIETY = {
            # 黑色系（活跃月份多）
            'RB': [1, 5, 10], 'HC': [1, 5, 10], 'I': [1, 5, 9],
            'J': [1, 5, 9], 'JM': [1, 5, 9], '焦煤': [1, 5, 9],
            '螺纹': [1, 5, 10], '热卷': [1, 5, 10], '铁矿': [1, 5, 9],
            # 有色金属
            'CU': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'AL': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'ZN': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'NI': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            # 化工
            'TA': [1, 5, 9], 'MA': [1, 5, 9], 'RU': [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            'L': [1, 5, 9], 'PP': [1, 5, 9], 'PVC': [1, 5, 9],
            # 农产品
            'M': [1, 3, 5, 7, 8, 9, 11],
            'Y': [1, 3, 5, 7, 8, 9, 12],
            'A': [1, 3, 5, 7, 9, 11],
            'CS': [1, 3, 5, 7, 9, 11],
            # 油脂
            'P': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            # 贵金属
            'AU': [1, 2, 4, 6, 8, 10, 12], 'AG': [1, 2, 4, 6, 8, 10, 12],
            # 金融
            'IC': [1, 3, 5, 7, 9, 12], 'IF': [1, 3, 5, 7, 9, 12],
            'IH': [1, 3, 5, 7, 9, 12], 'IM': [1, 3, 5, 7, 9, 12],
            'T': [3, 6, 9, 12], 'TF': [3, 6, 9, 12], 'TS': [3, 6, 9, 12],
        }

        # 大小写兼容
        variety_upper = variety.upper()
        active_months = ACTIVE_MONTHS_BY_VARIETY.get(variety_upper, [1, 5, 9])

        # 生成年份范围内所有可能的合约
        for year in range(start_dt.year, end_dt.year + 1):
            for month in active_months:
                contract_code = f"{variety_upper}{str(year)[2:]}{month:02d}"
                if contract_code not in contracts:
                    contracts.append(contract_code)

        # 去重并排序（按年份+月份）
        contracts = sorted(set(contracts), key=lambda x: (x[-4:-2], x[-2:]))

        logger.info(
            f"Generated {len(contracts)} contract candidates for {variety} "
            f"({start_date} ~ {end_date}): {contracts[:5]}{'...' if len(contracts)>5 else ''}"
        )
        return contracts
    
    # ==================== 基本面数据获取 ====================
    
    def get_basis_data(
        self,
        variety: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        source: str = '100ppi'
    ) -> pd.DataFrame:
        """
        获取基差数据
        
        Args:
            variety: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            source: 数据源
            
        Returns:
            DataFrame with basis data
        """
        if 'basis' not in self.crawlers:
            logger.warning("Basis crawler not available")
            return pd.DataFrame()
        
        crawler = self.crawlers['basis']
        return crawler.fetch_basis(variety, start_date, end_date, source)
    
    def get_warehouse_receipts(
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
            DataFrame with warehouse receipt data
        """
        if 'exchange' not in self.crawlers:
            logger.warning("Exchange crawler not available")
            return pd.DataFrame()
        
        crawler = self.crawlers['exchange']
        return crawler.fetch_warehouse_receipts(variety, start_date, end_date)
    
    def get_inventory_data(
        self,
        variety: str,
        inventory_type: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        获取库存数据
        
        Args:
            variety: 品种代码
            inventory_type: 库存类型
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with inventory data
        """
        if 'inventory' not in self.crawlers:
            logger.warning("Inventory crawler not available")
            return pd.DataFrame()
        
        crawler = self.crawlers['inventory']
        return crawler.fetch_inventory(variety, inventory_type, start_date, end_date)
    
    # ==================== 数据管理 ====================
    
    def update_all_data(self, varieties: Optional[List[str]] = None):
        """
        更新所有数据
        
        Args:
            varieties: 品种列表，为None时更新所有配置的品种
        """
        config = get_config()
        varieties = varieties or config.varieties
        
        logger.info(f"Starting data update for {len(varieties)} varieties")
        
        for variety in varieties:
            try:
                # 更新主力合约数据
                main_contract = self._get_main_contract(variety)
                if main_contract:
                    self.get_daily_data(main_contract, use_cache=True)
                
                # 更新基本面数据
                self.get_basis_data(variety)
                self.get_warehouse_receipts(variety)
                
                logger.info(f"Updated data for {variety}")
                
            except Exception as e:
                logger.error(f"Failed to update {variety}: {e}")
    
    def _get_main_contract(self, variety: str) -> Optional[str]:
        """获取主力合约代码"""
        if 'akshare' in self.fetchers:
            return self.fetchers['akshare'].fetch_main_contract(variety)
        return None
    
    def get_data_summary(self) -> Dict:
        """获取数据存储摘要"""
        return self.db.get_data_summary()
    
    def clear_cache(self, symbol: Optional[str] = None):
        """
        清除缓存
        
        Args:
            symbol: 合约代码，为None时清除所有缓存
        """
        if symbol:
            # 清除特定合约缓存
            for f in self.cache_dir.glob(f"{symbol}*.parquet"):
                f.unlink()
                logger.info(f"Removed cache: {f}")
        else:
            # 清除所有缓存
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
            logger.info("Cleared all cache")
