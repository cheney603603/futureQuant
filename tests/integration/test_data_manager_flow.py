"""
test_data_manager_flow.py - DataManager 集成测试

测试内容：
1. DataManager 初始化成功
2. get_daily_data() 在有缓存时从缓存读取（mock DBManager）
3. get_basis_data() 在爬虫不可用时返回空 DataFrame（而非报错）
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

pytest.importorskip("futureQuant")

from futureQuant.data.manager import DataManager


# =============================================================================
# 测试用例
# =============================================================================

class TestDataManagerInit:
    """测试 DataManager 初始化"""
    
    def test_init_default(self):
        """默认初始化成功"""
        # 需要 mock 掉依赖的组件
        with patch('futureQuant.data.manager.DBManager'), \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner'), \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher'):
            
            dm = DataManager()
            
            assert dm is not None
            assert hasattr(dm, 'db')
            assert hasattr(dm, 'cleaner')
            assert hasattr(dm, 'contract_manager')
            assert hasattr(dm, 'calendar')
    
    def test_init_with_custom_paths(self):
        """自定义路径初始化"""
        with patch('futureQuant.data.manager.DBManager') as mock_db, \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner'), \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher'):
            
            dm = DataManager(
                cache_dir='./test_cache',
                db_path='./test.db',
                auto_update=False
            )
            
            assert dm.auto_update is False
    
    def test_init_fetcher_akshare(self):
        """AKShare fetcher 初始化"""
        with patch('futureQuant.data.manager.DBManager'), \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner'), \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher') as mock_ak:
            
            dm = DataManager()
            
            # 如果 akshare 可用，应该有 fetcher
            # 注意：可能因为导入失败而不存在
            if hasattr(dm, 'fetchers'):
                assert dm.fetchers is not None


class TestGetDailyData:
    """测试获取日线数据"""
    
    def test_get_daily_data_from_cache(self):
        """有缓存时从缓存读取"""
        cached_data = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02'],
            'symbol': ['RB2501', 'RB2501'],
            'open': [3800.0, 3810.0],
            'high': [3850.0, 3860.0],
            'low': [3750.0, 3760.0],
            'close': [3800.0, 3810.0],
            'volume': [100000, 110000],
            'open_interest': [500000, 510000],
        })
        
        with patch('futureQuant.data.manager.DBManager') as mock_db_class, \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner'), \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher'):
            
            mock_db = MagicMock()
            mock_db.load_price_data.return_value = cached_data
            mock_db_class.return_value = mock_db
            
            dm = DataManager()
            
            result = dm.get_daily_data(
                symbol='RB2501',
                start_date='2024-08-01',
                end_date='2024-08-02',
                use_cache=True
            )
            
            # 应该从缓存读取，不调用 fetcher
            mock_db.load_price_data.assert_called()
            assert not result.empty
            assert len(result) == 2
    
    @pytest.mark.skip(reason="需要真实 akshare 环境或完整 mock，依赖网络可用性")
    def test_get_daily_data_cache_miss_fetches(self):
        """缓存未命中时从数据源获取"""
        empty_cache = pd.DataFrame()
        
        fetched_data = pd.DataFrame({
            'date': ['2024-08-01'],
            'symbol': ['RB2501'],
            'open': [3800.0],
            'high': [3850.0],
            'low': [3750.0],
            'close': [3800.0],
            'volume': [100000],
            'open_interest': [500000],
        })
        
        with patch('futureQuant.data.manager.DBManager') as mock_db_class, \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner') as mock_cleaner_class, \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher') as mock_ak_class:
            
            mock_db = MagicMock()
            mock_db.load_price_data.return_value = empty_cache
            mock_db_class.return_value = mock_db
            
            mock_cleaner = MagicMock()
            mock_cleaner.clean_ohlc.return_value = fetched_data
            mock_cleaner_class.return_value = mock_cleaner
            
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_daily.return_value = fetched_data
            mock_ak_class.return_value = mock_fetcher
            
            dm = DataManager()
            
            result = dm.get_daily_data(
                symbol='RB2501',
                start_date='2024-08-01',
                end_date='2024-08-01',
                use_cache=True
            )
            
            # 应该调用 fetcher
            mock_fetcher.fetch_daily.assert_called()
            # 应该保存到缓存
            mock_db.save_price_data.assert_called()
    
    def test_get_daily_data_no_cache(self):
        """不使用缓存时直接从数据源获取"""
        fetched_data = pd.DataFrame({
            'date': ['2024-08-01'],
            'symbol': ['RB2501'],
            'open': [3800.0],
            'high': [3850.0],
            'low': [3750.0],
            'close': [3800.0],
            'volume': [100000],
            'open_interest': [500000],
        })
        
        with patch('futureQuant.data.manager.DBManager') as mock_db_class, \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner') as mock_cleaner_class, \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher') as mock_ak_class:
            
            mock_db = MagicMock()
            mock_db_class.return_value = mock_db
            
            mock_cleaner = MagicMock()
            mock_cleaner.clean_ohlc.return_value = fetched_data
            mock_cleaner_class.return_value = mock_cleaner
            
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_daily.return_value = fetched_data
            mock_ak_class.return_value = mock_fetcher
            
            dm = DataManager()
            
            result = dm.get_daily_data(
                symbol='RB2501',
                start_date='2024-08-01',
                end_date='2024-08-01',
                use_cache=False
            )
            
            # 不应该从缓存读取
            mock_db.load_price_data.assert_not_called()


class TestGetBasisData:
    """测试基差数据获取"""
    
    def test_get_basis_data_crawler_unavailable_returns_empty(self):
        """爬虫不可用时返回空 DataFrame 而非报错"""
        with patch('futureQuant.data.manager.DBManager'), \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner'), \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher'):
            
            dm = DataManager()
            
            # 模拟爬虫不可用的情况
            dm.crawlers = {}  # 空字典
            
            result = dm.get_basis_data(variety='RB')
            
            # 应该返回空 DataFrame，而不是抛出异常
            assert isinstance(result, pd.DataFrame)
            assert result.empty
    
    def test_get_basis_data_with_crawler(self):
        """爬虫可用时正常返回"""
        expected_data = pd.DataFrame({
            'date': ['2024-08-01', '2024-08-02'],
            'variety': ['RB', 'RB'],
            'basis': [50.0, 60.0],
        })
        
        with patch('futureQuant.data.manager.DBManager'), \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner'), \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher'), \
             patch('futureQuant.data.manager.BasisCrawler') as mock_crawler_class:
            
            mock_crawler = MagicMock()
            mock_crawler.fetch_basis.return_value = expected_data
            mock_crawler_class.return_value = mock_crawler
            
            dm = DataManager()
            
            result = dm.get_basis_data(variety='RB')
            
            # 应该调用爬虫
            mock_crawler.fetch_basis.assert_called()


class TestOtherDataManagerMethods:
    """测试其他 DataManager 方法"""
    
    def test_get_continuous_contract_basic(self):
        """获取连续合约数据"""
        # _get_variety_contracts 是 DataManager 实例方法，需要 patch.object 绑定到类
        with patch('futureQuant.data.manager.DBManager') as mock_db_class, \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner'), \
             patch('futureQuant.data.manager.ContractManager') as mock_cm_class, \
             patch('futureQuant.data.manager.AKShareFetcher'), \
             patch.object(DataManager, '_get_variety_contracts', return_value=['RB2501']):
            
            mock_db = MagicMock()
            mock_db.load_price_data.return_value = pd.DataFrame()  # 缓存未命中
            mock_db_class.return_value = mock_db
            
            mock_cm = MagicMock()
            mock_cm.create_continuous_contract.return_value = pd.DataFrame({
                'date': ['2024-08-01'],
                'close': [3800.0],
                'symbol': ['RB2501'],
            })
            mock_cm_class.return_value = mock_cm
            
            dm = DataManager()
            
            result = dm.get_continuous_contract(
                variety='RB',
                start_date='2024-08-01',
                end_date='2024-08-01'
            )
            
            assert isinstance(result, pd.DataFrame)
    
    def test_get_inventory_data_crawler_unavailable(self):
        """库存爬虫不可用时返回空 DataFrame"""
        with patch('futureQuant.data.manager.DBManager'), \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner'), \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher'):
            
            dm = DataManager()
            dm.crawlers = {}
            
            result = dm.get_inventory_data(variety='RB')
            
            assert isinstance(result, pd.DataFrame)
            assert result.empty
    
    def test_clear_cache(self):
        """清除缓存"""
        with patch('futureQuant.data.manager.DBManager'), \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner'), \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher'):
            
            dm = DataManager()
            dm.clear_cache()
            
            # 不应抛出异常
    
    def test_get_data_summary(self):
        """获取数据摘要"""
        with patch('futureQuant.data.manager.DBManager') as mock_db_class, \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner'), \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher'):
            
            mock_db = MagicMock()
            mock_db.get_data_summary.return_value = {'total_records': 1000}
            mock_db_class.return_value = mock_db
            
            dm = DataManager()
            summary = dm.get_data_summary()
            
            assert isinstance(summary, dict)


class TestDataManagerEdgeCases:
    """测试边界情况"""
    
    def test_get_daily_data_empty_result(self):
        """数据源返回空结果"""
        with patch('futureQuant.data.manager.DBManager') as mock_db_class, \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner'), \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher') as mock_ak_class:
            
            mock_db = MagicMock()
            mock_db.load_price_data.return_value = pd.DataFrame()
            mock_db_class.return_value = mock_db
            
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_daily.return_value = pd.DataFrame()  # 空结果
            mock_ak_class.return_value = mock_fetcher
            
            dm = DataManager()
            
            result = dm.get_daily_data(
                symbol='INVALID',
                start_date='2024-08-01',
                end_date='2024-08-01'
            )
            
            # 返回空 DataFrame
            assert result.empty
    
    def test_update_all_data(self):
        """批量更新数据"""
        with patch('futureQuant.data.manager.DBManager'), \
             patch('futureQuant.data.manager.FuturesCalendar'), \
             patch('futureQuant.data.manager.DataCleaner'), \
             patch('futureQuant.data.manager.ContractManager'), \
             patch('futureQuant.data.manager.AKShareFetcher') as mock_ak_class:
            
            mock_fetcher = MagicMock()
            mock_fetcher.fetch_main_contract.return_value = 'RB2501'
            mock_ak_class.return_value = mock_fetcher
            
            dm = DataManager()
            dm.update_all_data(varieties=['RB'])
            
            # 不应抛出异常
