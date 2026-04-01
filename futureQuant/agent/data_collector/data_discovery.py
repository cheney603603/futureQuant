"""
Data Discovery - 数据源扫描与评估

扫描类:
- AkShareSource: AkShare 数据源
- TuShareSource: TuShare 数据源
- BaostockSource: Baostock 数据源
- ExchangeAPISource: 交易所 API 数据源

每个 Source 实现统一接口:
- health_check() -> bool
- get_instruments() -> list
- fetch(symbol, start_date, end_date) -> pd.DataFrame
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import pandas as pd

from ...core.logger import get_logger

logger = get_logger('agent.data_collector.discovery')


class DataSourceStatus(Enum):
    """数据源状态枚举"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    UNKNOWN = "unknown"


@dataclass
class DataSourceInfo:
    """数据源信息"""

    name: str
    status: DataSourceStatus
    response_time_ms: float = 0.0
    error_message: str = ""
    last_check: str = ""
    instruments_count: int = 0
    supported_types: List[str] = None

    def __post_init__(self):
        if self.supported_types is None:
            self.supported_types = []


class BaseDataSource(ABC):
    """数据源抽象基类"""

    def __init__(self, name: str):
        self.name = name
        self._last_health: Optional[bool] = None
        self._response_times: List[float] = []

    @property
    @abstractmethod
    def supported_types(self) -> List[str]:
        """支持的数据类型"""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """
        检查数据源健康状态

        Returns:
            True = 健康, False = 不健康
        """
        pass

    @abstractmethod
    def get_instruments(self, variety: Optional[str] = None) -> List[str]:
        """
        获取可交易的标的列表

        Args:
            variety: 品种代码, 为 None 时返回所有

        Returns:
            标的代码列表
        """
        pass

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        获取标的日线数据

        Args:
            symbol: 合约代码
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame, 列: [date, open, high, low, close, volume, open_interest]
        """
        pass

    def get_info(self) -> DataSourceInfo:
        """获取数据源信息"""
        health = self.health_check()
        avg_response = (
            sum(self._response_times[-10:]) / len(self._response_times[-10:])
            if self._response_times else 0.0
        )
        return DataSourceInfo(
            name=self.name,
            status=(
                DataSourceStatus.HEALTHY if health
                else DataSourceStatus.UNAVAILABLE
            ),
            response_time_ms=avg_response,
            last_check=datetime.now().isoformat(),
            supported_types=self.supported_types,
        )


class AkShareSource(BaseDataSource):
    """AkShare 数据源 - 基于 akshare 库获取期货数据"""

    def __init__(self):
        super().__init__("akshare")
        self._client = None

    @property
    def supported_types(self) -> List[str]:
        return ["daily", "minute", "basis", "inventory", "warehouse_receipt"]

    def _get_client(self):
        """延迟初始化 akshare 客户端"""
        if self._client is None:
            try:
                import akshare as ak
                self._client = ak
                logger.debug("AkShare client initialized")
            except ImportError:
                raise RuntimeError("akshare not installed. Run: pip install akshare")
        return self._client

    def health_check(self) -> bool:
        """检查 AkShare 接口健康"""
        start = time.time()
        try:
            ak = self._get_client()
            ak.futures_zh_daily_sina(symbol="RB2501")
            elapsed = (time.time() - start) * 1000
            self._response_times.append(elapsed)
            self._last_health = True
            logger.debug(f"AkShare health OK: {elapsed:.0f}ms")
            return True
        except Exception as exc:
            self._last_health = False
            logger.warning(f"AkShare health check failed: {exc}")
            return False

    def get_instruments(self, variety: Optional[str] = None) -> List[str]:
        """获取期货合约列表"""
        try:
            ak = self._get_client()
            df = ak.futures_zh_daily_sina(symbol=variety or "RB")
            if df is not None and 'symbol' in df.columns:
                return df['symbol'].unique().tolist()
        except Exception as exc:
            logger.warning(f"Failed to get instruments from AkShare: {exc}")
        return []

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """获取日线数据"""
        try:
            ak = self._get_client()
            df = ak.futures_zh_daily_sina(symbol=symbol)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.rename(columns={
                '\u65e5\u671f': 'date',
                '\u5f00\u76d8': 'open',
                '\u6700\u9ad8': 'high',
                '\u6700\u4f4e': 'low',
                '\u6536\u76d8': 'close',
                '\u6210\u4ea4\u91cf': 'volume',
                '\u6301\u4ed3\u91cf': 'open_interest',
            })
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df[
                    (df['date'] >= start_date) & (df['date'] <= end_date)
                ].reset_index(drop=True)
            return df
        except Exception as exc:
            logger.error(f"AkShare fetch failed for {symbol}: {exc}")
            raise


class TuShareSource(BaseDataSource):
    """TuShare 数据源 - 基于 tushare 获取期货数据"""

    def __init__(self):
        super().__init__("tushare")
        self._client = None

    @property
    def supported_types(self) -> List[str]:
        return ["daily", "minute"]

    def _get_client(self):
        if self._client is None:
            try:
                import tushare as ts
                self._client = ts
            except ImportError:
                raise RuntimeError("tushare not installed. Run: pip install tushare")
        return self._client

    def health_check(self) -> bool:
        start = time.time()
        try:
            ts = self._get_client()
            elapsed = (time.time() - start) * 1000
            self._response_times.append(elapsed)
            self._last_health = True
            return True
        except Exception as exc:
            self._last_health = False
            logger.warning(f"TuShare health check failed: {exc}")
            return False

    def get_instruments(self, variety: Optional[str] = None) -> List[str]:
        return []

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        raise NotImplementedError("TuShare requires token configuration. "
                                  "Please set your tushare token in config.")


class BaostockSource(BaseDataSource):
    """Baostock 数据源 - 基于 baostock 获取期货数据"""

    def __init__(self):
        super().__init__("baostock")
        self._bs = None

    @property
    def supported_types(self) -> List[str]:
        return ["daily"]

    def _get_client(self):
        if self._bs is None:
            try:
                import baostock as bs
                bs.login()
                self._bs = bs
            except ImportError:
                raise RuntimeError("baostock not installed. Run: pip install baostock")
            except Exception as exc:
                logger.warning(f"Baostock login failed: {exc}")
                raise
        return self._bs

    def health_check(self) -> bool:
        start = time.time()
        try:
            self._get_client()
            elapsed = (time.time() - start) * 1000
            self._response_times.append(elapsed)
            self._last_health = True
            return True
        except Exception as exc:
            self._last_health = False
            logger.warning(f"Baostock health check failed: {exc}")
            return False

    def get_instruments(self, variety: Optional[str] = None) -> List[str]:
        return []

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        try:
            bs = self._get_client()
            code = symbol[2:] + "." + symbol[:2].upper()
            rs = bs.query_history_k_data_plus(
                code,
                "date,open,high,low,close,volume,turn",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
            )
            data = bs.result2data(rs)
            if data is None:
                return pd.DataFrame()
            df = pd.DataFrame(data, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'turn'])
            df = df.rename(columns={'turn': 'open_interest'})
            return df
        except Exception as exc:
            logger.error(f"Baostock fetch failed for {symbol}: {exc}")
            raise


class ExchangeAPISource(BaseDataSource):
    """交易所 API 数据源 - 通过期货交易所官方 API 获取数据"""

    def __init__(self):
        super().__init__("exchange_api")

    @property
    def supported_types(self) -> List[str]:
        return ["daily", "minute", "tick"]

    def health_check(self) -> bool:
        start = time.time()
        try:
            elapsed = (time.time() - start) * 1000
            self._response_times.append(elapsed)
            self._last_health = False
            logger.warning("Exchange API not configured (simulated)")
            return False
        except Exception as exc:
            self._last_health = False
            logger.warning(f"Exchange API health check failed: {exc}")
            return False

    def get_instruments(self, variety: Optional[str] = None) -> List[str]:
        return []

    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        raise NotImplementedError("Exchange API requires CTP configuration")


class DataSourceManager:
    """
    多数据源统一管理

    提供:
    - 按健康度排序数据源
    - 批量健康检查
    - 数据源切换策略
    """

    def __init__(self):
        self._sources: Dict[str, BaseDataSource] = {
            'akshare': AkShareSource(),
            'tushare': TuShareSource(),
            'baostock': BaostockSource(),
            'exchange_api': ExchangeAPISource(),
        }

    def health_check_all(self) -> Dict[str, DataSourceInfo]:
        """批量健康检查"""
        results: Dict[str, DataSourceInfo] = {}
        for name, source in self._sources.items():
            try:
                info = source.get_info()
                results[name] = info
            except Exception as exc:
                results[name] = DataSourceInfo(
                    name=name,
                    status=DataSourceStatus.UNAVAILABLE,
                    error_message=str(exc),
                    last_check=datetime.now().isoformat(),
                )
        return results

    def get_healthy_sources(self) -> List[str]:
        """获取健康数据源列表(按响应时间排序)"""
        healthy: List[str] = []
        for name, source in self._sources.items():
            if source.health_check():
                healthy.append(name)
        healthy.sort(key=lambda n: (
            sum(self._sources[n]._response_times[-10:]) /
            max(len(self._sources[n]._response_times[-10:]), 1)
        ))
        return healthy

    def get_source(self, name: str) -> Optional[BaseDataSource]:
        return self._sources.get(name)

    def register_source(self, name: str, source: BaseDataSource) -> None:
        """注册自定义数据源"""
        self._sources[name] = source
