"""
FundamentalFetcher 单元测试

测试基本面数据获取器的核心功能。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from futureQuant.data.fetcher.fundamental_fetcher import (
    FundamentalFetcher,
    INVENTORY_SYMBOLS,
    VARIETY_CN,
    get_inventory,
    get_basis,
    get_fundamental_summary,
)


class TestFundamentalFetcherInit:
    """测试初始化"""
    
    def test_init_success(self):
        """测试正常初始化"""
        fetcher = FundamentalFetcher()
        assert fetcher.name == "fundamental"
        assert fetcher._ak is not None
        assert fetcher.timeout == 30
        assert fetcher.retry == 3
        assert fetcher.delay == 1.0
    
    def test_init_custom_params(self):
        """测试自定义参数"""
        fetcher = FundamentalFetcher(timeout=60, retry=5, delay=2.0)
        assert fetcher.timeout == 60
        assert fetcher.retry == 5
        assert fetcher.delay == 2.0


class TestFetchInventory:
    """测试库存数据获取"""
    
    def test_fetch_inventory_success(self):
        """测试成功获取库存数据"""
        with patch('akshare.futures_inventory_em') as mock_func:
            mock_df = pd.DataFrame({
                '日期': ['2026-01-07', '2026-01-08', '2026-01-09'],
                '库存': [56844, 55633, 55633],
                '增减': [None, -1211, 0],
            })
            mock_func.return_value = mock_df
            
            fetcher = FundamentalFetcher()
            result = fetcher.fetch_inventory('RB')
            
            assert not result.empty
            assert 'date' in result.columns
            assert 'inventory' in result.columns
            assert 'variety' in result.columns
            assert result['variety'].iloc[0] == 'RB'
    
    def test_fetch_inventory_empty_response(self):
        """测试空响应"""
        with patch('akshare.futures_inventory_em') as mock_func:
            mock_func.return_value = pd.DataFrame()
            
            fetcher = FundamentalFetcher()
            result = fetcher.fetch_inventory('RB')
            
            assert result.empty
    
    def test_fetch_inventory_with_date_filter(self):
        """测试日期过滤"""
        with patch('akshare.futures_inventory_em') as mock_func:
            mock_df = pd.DataFrame({
                '日期': ['2026-01-01', '2026-01-15', '2026-02-01'],
                '库存': [50000, 55000, 60000],
                '增减': [0, 5000, 5000],
            })
            mock_func.return_value = mock_df
            
            fetcher = FundamentalFetcher()
            result = fetcher.fetch_inventory(
                'RB',
                start_date='2026-01-10',
                end_date='2026-01-20',
            )
            
            mock_func.assert_called_once()
    
    def test_fetch_inventory_retry_on_error(self):
        """测试重试机制"""
        with patch('akshare.futures_inventory_em') as mock_func:
            # 第一次失败，第二次成功
            mock_func.side_effect = [
                Exception("Network error"),
                pd.DataFrame({'日期': ['2026-01-01'], '库存': [100], '增减': [0]}),
            ]
            
            fetcher = FundamentalFetcher()
            result = fetcher.fetch_inventory('RB')
            
            assert not result.empty
            assert mock_func.call_count == 2


class TestFetchBasis:
    """测试基差数据获取"""
    
    def test_fetch_basis_success(self):
        """测试成功获取基差数据"""
        with patch('akshare.futures_spot_price') as mock_func:
            mock_df = pd.DataFrame({
                'date': ['2024-04-30'],
                'symbol': ['RB'],
                'spot_price': [3586.22],
                'near_basis': [-93.22],
                'dom_basis': [69.78],
                'near_basis_rate': [-0.026],
                'dom_basis_rate': [0.019],
            })
            mock_func.return_value = mock_df
            
            fetcher = FundamentalFetcher()
            result = fetcher.fetch_basis('RB')
            
            assert not result.empty
            mock_func.assert_called_once()
    
    def test_fetch_basis_all_varieties(self):
        """测试获取所有品种基差"""
        with patch('akshare.futures_spot_price') as mock_func:
            mock_df = pd.DataFrame({
                'date': ['2024-04-30'] * 3,
                'symbol': ['RB', 'HC', 'I'],
                'spot_price': [3586, 3700, 800],
            })
            mock_func.return_value = mock_df
            
            fetcher = FundamentalFetcher()
            result = fetcher.fetch_basis()
            
            assert not result.empty
            assert len(result) == 3


class TestFetchWarehouseReceipt:
    """测试仓单数据获取"""
    
    def test_fetch_warehouse_receipt_shfe(self):
        """测试上期所仓单"""
        with patch('akshare.futures_shfe_warehouse_receipt') as mock_func:
            mock_df = pd.DataFrame({
                '品种': ['CU', 'AL'],
                '仓单数量': [10000, 8000],
            })
            mock_func.return_value = mock_df
            
            fetcher = FundamentalFetcher()
            result = fetcher.fetch_warehouse_receipt('SHFE')
            
            mock_func.assert_called_once()
    
    def test_fetch_warehouse_receipt_dce(self):
        """测试大商所仓单"""
        with patch('akshare.futures_warehouse_receipt_dce') as mock_func:
            mock_df = pd.DataFrame({
                '品种': ['I', 'J'],
                '仓单数量': [5000, 3000],
            })
            mock_func.return_value = mock_df
            
            fetcher = FundamentalFetcher()
            result = fetcher.fetch_warehouse_receipt('DCE')
            
            mock_func.assert_called_once()
    
    def test_fetch_warehouse_receipt_unknown_exchange(self):
        """测试未知交易所"""
        fetcher = FundamentalFetcher()
        result = fetcher.fetch_warehouse_receipt('UNKNOWN')
        
        assert result.empty


class TestFetchFundamentalSummary:
    """测试基本面汇总"""
    
    def test_fetch_summary_success(self):
        """测试成功获取汇总"""
        with patch('akshare.futures_inventory_em') as mock_inv, \
             patch('akshare.futures_spot_price') as mock_basis:
            # 模拟库存数据
            mock_inv_df = pd.DataFrame({
                '日期': ['2026-04-17'],
                '库存': [83390],
                '增减': [0],
            })
            mock_inv.return_value = mock_inv_df
            
            # 模拟基差数据
            mock_basis_df = pd.DataFrame({
                'date': ['2024-04-30'],
                'symbol': ['RB'],
                'spot_price': [3586.22],
                'near_basis': [-93.22],
                'dom_basis': [69.78],
                'near_basis_rate': [-0.026],
            })
            mock_basis.return_value = mock_basis_df
            
            fetcher = FundamentalFetcher()
            result = fetcher.fetch_fundamental_summary('RB')
            
            assert result['variety'] == 'RB'
            assert result['variety_cn'] == '螺纹钢'
            assert 'inventory' in result
            assert 'basis' in result
            assert 'update_time' in result


class TestFetchSymbols:
    """测试获取支持品种"""
    
    def test_fetch_symbols(self):
        """测试获取品种列表"""
        fetcher = FundamentalFetcher()
        symbols = fetcher.fetch_symbols()
        
        assert len(symbols) > 0
        assert 'RB' in symbols
        assert 'I' in symbols
        assert 'CU' in symbols


class TestBatchFetch:
    """测试批量获取"""
    
    def test_fetch_all_inventory(self):
        """测试批量获取库存"""
        with patch('akshare.futures_inventory_em') as mock_func:
            mock_func.return_value = pd.DataFrame({
                '日期': ['2026-01-01'],
                '库存': [100],
                '增减': [0],
            })
            
            fetcher = FundamentalFetcher()
            result = fetcher.fetch_all_inventory(['RB', 'HC'])
            
            assert not result.empty
            assert mock_func.call_count == 2


class TestConvenienceFunctions:
    """测试便捷函数"""
    
    @patch('futureQuant.data.fetcher.fundamental_fetcher.FundamentalFetcher')
    def test_get_inventory(self, mock_fetcher_class):
        """测试 get_inventory 便捷函数"""
        mock_fetcher = MagicMock()
        mock_fetcher_class.return_value = mock_fetcher
        mock_fetcher.fetch_inventory.return_value = pd.DataFrame()
        
        result = get_inventory('RB')
        
        mock_fetcher.fetch_inventory.assert_called_once_with('RB')
    
    @patch('futureQuant.data.fetcher.fundamental_fetcher.FundamentalFetcher')
    def test_get_basis(self, mock_fetcher_class):
        """测试 get_basis 便捷函数"""
        mock_fetcher = MagicMock()
        mock_fetcher_class.return_value = mock_fetcher
        mock_fetcher.fetch_basis.return_value = pd.DataFrame()
        
        result = get_basis('RB')
        
        mock_fetcher.fetch_basis.assert_called_once_with('RB')
    
    @patch('futureQuant.data.fetcher.fundamental_fetcher.FundamentalFetcher')
    def test_get_fundamental_summary(self, mock_fetcher_class):
        """测试 get_fundamental_summary 便捷函数"""
        mock_fetcher = MagicMock()
        mock_fetcher_class.return_value = mock_fetcher
        mock_fetcher.fetch_fundamental_summary.return_value = {}
        
        result = get_fundamental_summary('RB')
        
        mock_fetcher.fetch_fundamental_summary.assert_called_once_with('RB')


class TestStandardization:
    """测试数据标准化"""
    
    def test_standardize_inventory(self):
        """测试库存数据标准化"""
        fetcher = FundamentalFetcher()
        
        df = pd.DataFrame({
            '日期': ['2026-01-07', '2026-01-08'],
            '库存': [56844, 55633],
            '增减': [None, -1211],
        })
        
        result = fetcher._standardize_inventory(df, 'RB')
        
        assert 'date' in result.columns
        assert 'inventory' in result.columns
        assert 'variety' in result.columns
        assert result['variety'].iloc[0] == 'RB'
    
    def test_standardize_basis(self):
        """测试基差数据标准化"""
        fetcher = FundamentalFetcher()
        
        df = pd.DataFrame({
            'date': ['2024-04-30'],
            'symbol': ['RB'],
            'spot_price': ['3586.22'],  # 字符串格式
            'near_basis': ['-93.22'],
        })
        
        result = fetcher._standardize_basis(df)
        
        assert 'date' in result.columns
        # 数值列应已转换
        assert result['spot_price'].dtype in [np.float64, np.int64]


class TestConstants:
    """测试常量配置"""
    
    def test_variety_cn_mapping(self):
        """测试品种中文名映射"""
        assert VARIETY_CN['RB'] == '螺纹钢'
        assert VARIETY_CN['I'] == '铁矿石'
        assert VARIETY_CN['CU'] == '铜'
    
    def test_inventory_symbols_mapping(self):
        """测试库存品种代码映射"""
        assert INVENTORY_SYMBOLS['RB'] == 'rb'
        assert INVENTORY_SYMBOLS['I'] == 'i'
        assert INVENTORY_SYMBOLS['CU'] == 'cu'


class TestErrorHandling:
    """测试错误处理"""
    
    def test_network_error_returns_empty(self):
        """测试网络错误返回空 DataFrame"""
        with patch('akshare.futures_inventory_em') as mock_func:
            mock_func.side_effect = Exception("Network error")
            
            fetcher = FundamentalFetcher(retry=1)
            result = fetcher.fetch_inventory('RB')
            
            assert result.empty
    
    def test_invalid_data_handling(self):
        """测试无效数据处理"""
        with patch('akshare.futures_inventory_em') as mock_func:
            # 返回 None
            mock_func.return_value = None
            
            fetcher = FundamentalFetcher()
            result = fetcher.fetch_inventory('RB')
            
            assert result.empty


class TestIntegration:
    """集成测试（使用真实数据）"""
    
    @pytest.mark.integration
    def test_real_inventory_fetch(self):
        """测试真实库存数据获取"""
        fetcher = FundamentalFetcher()
        result = fetcher.fetch_inventory('RB')
        
        # 如果网络正常，应该获取到数据
        if not result.empty:
            assert 'date' in result.columns
            assert 'inventory' in result.columns
            assert len(result) > 0
    
    @pytest.mark.integration
    def test_real_basis_fetch(self):
        """测试真实基差数据获取"""
        fetcher = FundamentalFetcher()
        result = fetcher.fetch_basis('RB')
        
        # 如果网络正常，应该获取到数据
        if not result.empty:
            assert 'spot_price' in result.columns or len(result.columns) > 0
