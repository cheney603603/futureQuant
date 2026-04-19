"""
akshare 数据获取器

修复说明 (2026-04-11):
  - 新浪接口 futures_zh_daily_sina(symbol=variety) 已失效（返回空数据）
  - 改用 akshare.get_futures_daily(market=exchange) 获取全量数据再按品种过滤
  - 支持: SHFE, CZCE, CFFEX, INE
  - DCE 交易所接口已失效
"""

import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ...core.base import DataFetcher
from ...core.exceptions import FetchError
from ...core.logger import get_logger

logger = get_logger('data.fetcher.akshare')

# 品种代码 → 交易所 映射
VARIETY_EXCHANGE: Dict[str, str] = {
    # SHFE
    'AG': 'SHFE', 'AL': 'SHFE', 'AO': 'SHFE', 'AU': 'SHFE',
    'BC': 'SHFE', 'BR': 'SHFE', 'BU': 'SHFE', 'CU': 'SHFE',
    'EC': 'SHFE', 'FU': 'SHFE', 'HC': 'SHFE', 'LU': 'SHFE',
    'NI': 'SHFE', 'NR': 'SHFE', 'PB': 'SHFE', 'RB': 'SHFE',
    'RU': 'SHFE', 'SC': 'SHFE', 'SN': 'SHFE', 'SP': 'SHFE',
    'SS': 'SHFE', 'WR': 'SHFE', 'ZN': 'SHFE',
    # CZCE
    'AP': 'CZCE', 'CF': 'CZCE', 'CJ': 'CZCE', 'CY': 'CZCE',
    'FG': 'CZCE', 'JR': 'CZCE', 'LR': 'CZCE', 'MA': 'CZCE',
    'OI': 'CZCE', 'PF': 'CZCE', 'PK': 'CZCE', 'PM': 'CZCE',
    'PX': 'CZCE', 'RI': 'CZCE', 'RM': 'CZCE', 'RS': 'CZCE',
    'SA': 'CZCE', 'SF': 'CZCE', 'SH': 'CZCE', 'SM': 'CZCE',
    'SR': 'CZCE', 'TA': 'CZCE', 'UR': 'CZCE', 'WH': 'CZCE',
    'ZC': 'CZCE',
    # CFFEX
    'IC': 'CFFEX', 'IF': 'CFFEX', 'IH': 'CFFEX', 'IM': 'CFFEX',
    'T': 'CFFEX', 'TF': 'CFFEX', 'TL': 'CFFEX', 'TS': 'CFFEX',
    # INE
    'NR': 'INE',  # NR 同时在 SHFE 和 INE，取 INE
    'SC_TAS': 'INE',
}

# 交易所 → akshare market 参数
EXCHANGE_AK: Dict[str, str] = {
    'SHFE': 'SHFE', 'CZCE': 'CZCE',
    'CFFEX': 'CFFEX', 'INE': 'INE',
}

# CZCE 列名特殊映射（无 open_interest/turnover 列）
_CZCE_COLS_STD: Dict[str, str] = {
    'date': 'date', 'open': 'open', 'high': 'high',
    'low': 'low', 'close': 'close', 'volume': 'volume',
}
# SHFE/CFFEX/INE 标准列
_STD_COLS: Dict[str, str] = {
    'date': 'date', 'open': 'open', 'high': 'high',
    'low': 'low', 'close': 'close', 'volume': 'volume',
    'open_interest': 'open_interest',
}


class AKShareFetcher(DataFetcher):
    """
    akshare 数据获取器（修复版）

    使用 akshare.get_futures_daily(market=exchange) 按交易所拉取，
    再按品种过滤，替代已失效的新浪接口。
    """

    def __init__(self, timeout: int = 30, retry: int = 3, delay: float = 1.0):
        self.timeout = timeout
        self.retry = retry
        self.delay = delay
        self._ak = None
        self._cache: Dict[str, pd.DataFrame] = {}  # market+date → DataFrame
        self._init_akshare()

    def _init_akshare(self):
        try:
            import akshare as ak
            self._ak = ak
            logger.info("akshare initialized successfully")
        except ImportError:
            raise ImportError("akshare not installed. Run: pip install akshare")

    @property
    def name(self) -> str:
        return "akshare"

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs,
    ) -> pd.DataFrame:
        """
        获取期货日线数据

        Args:
            symbol: 合约代码，如 'RB'（品种前缀）/'RB2501'（具体合约）
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
        """
        for attempt in range(self.retry):
            try:
                return self._fetch_impl(symbol, start_date, end_date)
            except FetchError:
                if attempt < self.retry - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        return pd.DataFrame()

    def _fetch_impl(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        内部获取实现：
        1. 品种前缀 → 交易所
        2. 按交易所拉取全量数据（带缓存）
        3. 过滤指定品种/合约
        4. 严格日期范围验证
        """
        import pandas as pd
        from datetime import datetime, timedelta
        
        # 标准化 symbol: 取前2字符作为品种前缀
        variety = symbol[:2].upper()
        contract_code = symbol.upper()

        # 确认交易所
        exchange = VARIETY_EXCHANGE.get(variety)
        if not exchange:
            raise FetchError(
                f"Unknown variety '{variety}' for symbol '{symbol}'. "
                f"Supported: {sorted(VARIETY_EXCHANGE.keys())}"
            )

        # 按交易日期拉取（缓存一天内多次调用）
        ak_market = EXCHANGE_AK.get(exchange, exchange)
        all_data = self._fetch_exchange(ak_market, start_date, end_date)

        if all_data.empty:
            raise FetchError(
                f"No data returned from {exchange} for {variety} "
                f"({start_date} ~ {end_date})"
            )

        # 标准化列（先做，用于日期列转换）
        df = self._standardize(all_data.copy(), exchange)

        # 严格日期过滤：确保数据在请求范围内
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # 移除无效日期
            df = df[df['date'].notna()]
            
            # 严格按日期范围过滤
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            date_mask = (df['date'] >= start_dt) & (df['date'] <= end_dt)
            
            before_count = len(df)
            df = df[date_mask].copy()
            
            # 如果过滤后数据过少，发出警告
            if len(df) < before_count * 0.5:
                logger.warning(
                    f"[AKShareFetcher] {symbol}: 日期过滤后数据减少 "
                    f"({before_count} → {len(df)})，可能存在数据问题"
                )

        # 过滤：品种列匹配 或 合约代码前缀匹配
        if 'variety' in df.columns:
            mask = df['variety'] == variety
        else:
            mask = df['symbol'].str.upper().str.startswith(variety)

        # 如果输入是具体合约（如 RB2501），再过滤合约代码
        if len(contract_code) >= 4 and contract_code not in VARIETY_EXCHANGE:
            mask_contract = (
                df['symbol'].str.upper().str.startswith(contract_code)
            )
            mask = mask & mask_contract

        df = df[mask].copy()

        # 日期排序
        df = df.sort_values('date').reset_index(drop=True)
        
        # 数据新鲜度检查
        if len(df) > 0 and 'date' in df.columns:
            latest_date = df['date'].max()
            days_old = (datetime.now() - latest_date).days
            if days_old > 7:
                logger.warning(
                    f"[AKShareFetcher] {symbol}: 数据可能过时，"
                    f"最新数据 {days_old} 天前 ({latest_date.date()})"
                )

        logger.info(
            f"[AKShareFetcher] {symbol} @ {exchange}: "
            f"{len(df)} records ({start_date} ~ {end_date})"
        )
        time.sleep(self.delay)
        return df

    def _fetch_exchange(
        self,
        market: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """按交易所拉取全量数据（带日期缓存）"""
        # 转换日期格式
        sd = pd.to_datetime(start_date).strftime('%Y%m%d')
        ed = pd.to_datetime(end_date).strftime('%Y%m%d')

        # 缓存键
        cache_key = f"{market}:{sd}:{ed}"

        if cache_key in self._cache:
            return self._cache[cache_key]

        # 调用 akshare.get_futures_daily
        df = self._ak.get_futures_daily(
            start_date=sd,
            end_date=ed,
            market=market,
        )

        if df is None or df.empty:
            return pd.DataFrame()

        # 清理返回数据
        if 'index' in df.columns:
            df = df.drop(columns=['index'])

        self._cache[cache_key] = df
        return df

    def fetch_symbols(self, variety: Optional[str] = None) -> List[str]:
        """
        获取可交易合约列表
        """
        try:
            # 获取近期交易日
            end = datetime.now().strftime('%Y%m%d')
            start = (datetime.now().replace(day=1)).strftime('%Y%m%d')

            if variety:
                exchange = VARIETY_EXCHANGE.get(variety.upper())
                if exchange:
                    df = self._fetch_exchange(EXCHANGE_AK[exchange], start, end)
                    if not df.empty and 'symbol' in df.columns:
                        return sorted(df['symbol'].str.upper().unique().tolist())
            else:
                # 返回所有支持品种的合约
                symbols = []
                for ex in ['SHFE', 'CZCE', 'CFFEX']:
                    df = self._fetch_exchange(EXCHANGE_AK[ex], start, end)
                    if not df.empty and 'symbol' in df.columns:
                        symbols.extend(df['symbol'].str.upper().unique().tolist())
                return sorted(set(symbols))
        except Exception as e:
            logger.warning(f"fetch_symbols failed: {e}")
        return []

    def fetch_main_contract(self, variety: str) -> str:
        """获取主力合约代码"""
        symbols = self.fetch_symbols(variety=variety)
        if not symbols:
            return ""
        # 取最近到期的合约
        return sorted(symbols)[0]

    def _standardize(self, df: pd.DataFrame, exchange: str) -> pd.DataFrame:
        """标准化列名"""
        if df.empty:
            return df

        # 列名清理（akshare 返回中文列名）
        rename_map: Dict[str, str] = {}
        for col in df.columns:
            col_s = str(col).strip()
            if col_s in ('\u65e5\u671f', 'date'):  # 日期
                rename_map[col] = 'date'
            elif col_s in ('\u5f00\u76d8', 'open'):  # 开盘
                rename_map[col] = 'open'
            elif col_s in ('\u6700\u9ad8', 'high'):  # 最高
                rename_map[col] = 'high'
            elif col_s in ('\u6700\u4f4e', 'low'):  # 最低
                rename_map[col] = 'low'
            elif col_s in ('\u6536\u76d8', 'close'):  # 收盘
                rename_map[col] = 'close'
            elif col_s in ('\u6210\u4ea4\u91cf', 'volume'):  # 成交量
                rename_map[col] = 'volume'
            elif col_s in ('\u6301\u4ed3\u91cf', 'open_interest', 'oi'):  # 持仓量
                rename_map[col] = 'open_interest'
            elif col_s in ('symbol', 'symbol'):
                rename_map[col] = 'symbol'
            elif col_s in ('variety', '\u54c1\u79cd'):
                rename_map[col] = 'variety'

        df = df.rename(columns=rename_map)

        # 数值列
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'open_interest']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 日期列统一转换：int(20240102) 或 str('20240102') → datetime
        if 'date' in df.columns:
            if df['date'].dtype == 'int64' or df['date'].dtype.name == 'int64':
                # CZCE: 整数格式 20240102
                df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y%m%d')
            else:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # 品种列
        if 'variety' not in df.columns and 'symbol' in df.columns:
            df['variety'] = df['symbol'].str[:2]

        # 保留标准列
        std_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'open_interest', 'symbol', 'variety']
        keep_cols = [c for c in std_cols if c in df.columns]
        df = df[keep_cols]

        return df
