"""
基本面数据获取器 - 基于 akshare 实现

提供：
- 库存数据（东方财富）
- 仓单数据（交易所）
- 基差数据（现货-期货价差）

Author: futureQuant Team
Date: 2026-04-18
"""

import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd

from ...core.base import DataFetcher
from ...core.exceptions import FetchError
from ...core.logger import get_logger

logger = get_logger('data.fetcher.fundamental')


# 品种中文名映射
VARIETY_CN = {
    'RB': '螺纹钢', 'HC': '热轧板卷', 'I': '铁矿石', 'J': '焦炭', 'JM': '焦煤',
    'CU': '铜', 'AL': '铝', 'ZN': '锌', 'NI': '镍', 'SN': '锡', 'PB': '铅',
    'AU': '黄金', 'AG': '白银',
    'TA': 'PTA', 'MA': '甲醇', 'PP': '聚丙烯', 'L': '聚乙烯', 'V': 'PVC',
    'EG': '乙二醇', 'EB': '苯乙烯', 'PG': '液化石油气',
    'RU': '天然橡胶', 'NR': '20号胶', 'BU': '石油沥青', 'FU': '燃料油', 'SC': '原油',
    'M': '豆粕', 'Y': '豆油', 'A': '豆一', 'B': '豆二', 'P': '棕榈油',
    'C': '玉米', 'CS': '玉米淀粉', 'JD': '鸡蛋', 'LH': '生猪',
    'CF': '棉花', 'SR': '白糖', 'OI': '菜籽油', 'RM': '菜粕',
    'AP': '苹果', 'CJ': '红枣', 'PK': '花生',
    'FG': '玻璃',  # 玻璃期货
}

# akshare 库存品种代码映射（英文代码或中文名）
INVENTORY_SYMBOLS = {
    'RB': '螺纹钢', 'HC': '热轧卷板', 'I': '铁矿石', 'J': '焦炭', 'JM': '焦煤',
    'CU': '铜', 'AL': '铝', 'ZN': '锌', 'NI': '镍', 'SN': '锡', 'PB': '铅',
    'AU': '黄金', 'AG': '白银',
    'TA': 'PTA', 'MA': '甲醇', 'PP': '聚丙烯', 'L': '聚乙烯', 'V': 'PVC',
    'EG': '乙二醇', 'EB': '苯乙烯', 'PG': '液化石油气',
    'RU': '天然橡胶', 'NR': '20号胶', 'BU': '石油沥青', 'FU': '燃料油',
    'M': '豆粕', 'Y': '豆油', 'A': '豆一', 'B': '豆二', 'P': '棕榈油',
    'C': '玉米', 'CS': '玉米淀粉', 'JD': '鸡蛋', 'LH': '生猪',
    'CF': '棉花', 'SR': '白糖', 'OI': '菜籽油', 'RM': '菜粕',
    'FG': '玻璃',  # 玻璃期货
}


class FundamentalFetcher(DataFetcher):
    """
    基本面数据获取器

    基于 akshare 获取：
    1. 库存数据 - 东方财富数据源
    2. 基差数据 - 现货价格与期货价格对比
    3. 仓单数据 - 交易所官网（部分接口可能不稳定）
    """

    def __init__(self, timeout: int = 30, retry: int = 3, delay: float = 1.0):
        self.timeout = timeout
        self.retry = retry
        self.delay = delay
        self._ak = None
        self._init_akshare()

    def _init_akshare(self):
        try:
            import akshare as ak
            self._ak = ak
            logger.info("FundamentalFetcher: akshare initialized")
        except ImportError:
            raise ImportError("akshare not installed. Run: pip install akshare")

    @property
    def name(self) -> str:
        return "fundamental"

    # ============================================================
    # 抽象方法实现（DataFetcher 接口）
    # ============================================================

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        获取日线数据（基本面数据暂不支持此接口）

        基本面数据请使用：
        - fetch_inventory() 获取库存
        - fetch_basis() 获取基差
        """
        logger.warning("FundamentalFetcher does not support fetch_daily. Use fetch_inventory or fetch_basis.")
        return pd.DataFrame()

    def fetch_symbols(self) -> List[str]:
        """获取支持的品种列表"""
        return list(INVENTORY_SYMBOLS.keys())

    # ============================================================
    # 库存数据
    # ============================================================

    def fetch_inventory(
        self,
        variety: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取库存数据（来自东方财富）

        Args:
            variety: 品种代码，如 'RB', 'I'
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame with columns: [date, variety, inventory, change]
        """
        variety = variety.upper()
        symbol = INVENTORY_SYMBOLS.get(variety, variety.lower())

        for attempt in range(self.retry):
            try:
                df = self._ak.futures_inventory_em(symbol=symbol)

                if df is None or df.empty:
                    logger.warning(f"No inventory data for {variety}")
                    return pd.DataFrame()

                # 标准化列名（中文 → 英文）
                df = self._standardize_inventory(df, variety)

                # 日期过滤
                if start_date and 'date' in df.columns:
                    df = df[df['date'] >= start_date]
                if end_date and 'date' in df.columns:
                    df = df[df['date'] <= end_date]

                logger.info(
                    f"[FundamentalFetcher] Inventory {variety}: "
                    f"{len(df)} records"
                )
                time.sleep(self.delay)
                return df

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for inventory {variety}: {e}")
                if attempt < self.retry - 1:
                    time.sleep(2 ** attempt)

        return pd.DataFrame()

    def _standardize_inventory(self, df: pd.DataFrame, variety: str) -> pd.DataFrame:
        """标准化库存数据列名"""
        # akshare 返回中文列名：日期、库存、增减
        rename_map = {}
        cols_lower = [str(c).lower() for c in df.columns]

        for i, col in enumerate(df.columns):
            col_s = str(col).strip()
            if '日期' in col_s or col_s == 'date':
                rename_map[col] = 'date'
            elif '库存' in col_s or col_s == 'inventory':
                rename_map[col] = 'inventory'
            elif '增减' in col_s or 'change' in col_s.lower():
                rename_map[col] = 'change'

        df = df.rename(columns=rename_map)

        # 日期转换
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # 数值转换
        for col in ['inventory', 'change']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['variety'] = variety

        # 保留标准列
        std_cols = ['date', 'variety', 'inventory', 'change']
        keep_cols = [c for c in std_cols if c in df.columns]
        df = df[keep_cols].dropna(subset=['date'])

        return df.sort_values('date').reset_index(drop=True)

    # ============================================================
    # 基差数据
    # ============================================================

    def fetch_basis(
        self,
        variety: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取基差数据（现货价格与期货价格对比）

        Args:
            variety: 品种代码，如 'RB'。为 None 时返回所有品种
            start_date: 开始日期（暂不支持，仅返回最新数据）
            end_date: 结束日期

        Returns:
            DataFrame with columns:
            [date, symbol, spot_price, near_contract, near_contract_price,
             dominant_contract, dominant_contract_price, near_basis, dom_basis,
             near_basis_rate, dom_basis_rate]
        """
        for attempt in range(self.retry):
            try:
                df = self._ak.futures_spot_price()

                if df is None or df.empty:
                    logger.warning("No basis data available")
                    return pd.DataFrame()

                # 品种过滤
                if variety:
                    variety = variety.upper()
                    # 尝试多种匹配方式
                    df = df[
                        (df['symbol'] == variety) |
                        (df['symbol'].str.upper() == variety) |
                        (df['symbol'].str.startswith(variety))
                    ]

                # 标准化
                df = self._standardize_basis(df)

                logger.info(
                    f"[FundamentalFetcher] Basis data: "
                    f"{len(df)} records"
                )
                time.sleep(self.delay)
                return df

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for basis: {e}")
                if attempt < self.retry - 1:
                    time.sleep(2 ** attempt)

        return pd.DataFrame()

    def _standardize_basis(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化基差数据"""
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # 数值列
        numeric_cols = [
            'spot_price', 'near_contract_price', 'dominant_contract_price',
            'near_basis', 'dom_basis', 'near_basis_rate', 'dom_basis_rate'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.reset_index(drop=True)

    # ============================================================
    # 仓单数据
    # ============================================================

    def fetch_warehouse_receipt(
        self,
        exchange: str,
        date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        获取仓单数据

        Args:
            exchange: 交易所代码 ('SHFE', 'DCE', 'CZCE', 'GFEX')
            date: 日期 (YYYYMMDD)，默认最新

        Returns:
            DataFrame with warehouse receipt data

        Note:
            仓单接口可能不稳定，交易所网站经常变更
        """
        if date is None:
            date = datetime.now().strftime('%Y%m%d')

        exchange = exchange.upper()

        try:
            if exchange == 'SHFE':
                df = self._ak.futures_shfe_warehouse_receipt(date=date)
            elif exchange == 'DCE':
                df = self._ak.futures_warehouse_receipt_dce(date=date)
            elif exchange == 'CZCE':
                df = self._ak.futures_warehouse_receipt_czce(date=date)
            elif exchange == 'GFEX':
                df = self._ak.futures_gfex_warehouse_receipt(date=date)
            else:
                logger.warning(f"Unknown exchange: {exchange}")
                return pd.DataFrame()

            if df is not None and not df.empty:
                df = self._standardize_warehouse(df, exchange)
                logger.info(f"[FundamentalFetcher] Warehouse receipts {exchange}: {len(df)} records")
                return df

        except Exception as e:
            logger.warning(f"Warehouse receipt fetch failed for {exchange}: {e}")
            # 接口可能不稳定，返回空 DataFrame 而非抛出异常

        return pd.DataFrame()

    def _standardize_warehouse(self, df: pd.DataFrame, exchange: str) -> pd.DataFrame:
        """标准化仓单数据"""
        df['exchange'] = exchange
        df['fetch_date'] = datetime.now()
        return df

    # ============================================================
    # 综合基本面数据
    # ============================================================

    def fetch_fundamental_summary(
        self,
        variety: str,
    ) -> Dict[str, Any]:
        """
        获取品种基本面数据汇总

        Args:
            variety: 品种代码

        Returns:
            包含库存、基差等信息的汇总字典
        """
        variety = variety.upper()

        summary = {
            'variety': variety,
            'variety_cn': VARIETY_CN.get(variety, '未知'),
            'inventory': None,
            'basis': None,
            'update_time': datetime.now().isoformat(),
        }

        # 库存数据
        try:
            inv_df = self.fetch_inventory(variety)
            if not inv_df.empty:
                latest = inv_df.iloc[-1]
                summary['inventory'] = {
                    'date': latest.get('date'),
                    'value': latest.get('inventory'),
                    'change': latest.get('change'),
                }
        except Exception as e:
            logger.warning(f"Failed to fetch inventory for {variety}: {e}")

        # 基差数据
        try:
            basis_df = self.fetch_basis(variety)
            if not basis_df.empty:
                latest = basis_df.iloc[0]
                summary['basis'] = {
                    'spot_price': latest.get('spot_price'),
                    'near_basis': latest.get('near_basis'),
                    'dom_basis': latest.get('dom_basis'),
                    'near_basis_rate': latest.get('near_basis_rate'),
                }
        except Exception as e:
            logger.warning(f"Failed to fetch basis for {variety}: {e}")

        return summary

    # ============================================================
    # 批量获取
    # ============================================================

    def fetch_all_inventory(
        self,
        varieties: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        批量获取多个品种的库存数据

        Args:
            varieties: 品种列表，默认获取所有支持品种

        Returns:
            合并的库存数据 DataFrame
        """
        if varieties is None:
            varieties = list(INVENTORY_SYMBOLS.keys())

        all_data = []
        for variety in varieties:
            try:
                df = self.fetch_inventory(variety)
                if not df.empty:
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to fetch inventory for {variety}: {e}")

        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()

    def fetch_all_basis(self) -> pd.DataFrame:
        """
        获取所有品种的基差数据

        Returns:
            所有品种基差数据 DataFrame
        """
        return self.fetch_basis()


# ============================================================
# 便捷函数
# ============================================================

def get_inventory(variety: str) -> pd.DataFrame:
    """获取单个品种库存数据"""
    fetcher = FundamentalFetcher()
    return fetcher.fetch_inventory(variety)


def get_basis(variety: Optional[str] = None) -> pd.DataFrame:
    """获取基差数据"""
    fetcher = FundamentalFetcher()
    return fetcher.fetch_basis(variety)


def get_fundamental_summary(variety: str) -> Dict[str, Any]:
    """获取品种基本面汇总"""
    fetcher = FundamentalFetcher()
    return fetcher.fetch_fundamental_summary(variety)
