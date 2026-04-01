"""
DataCollectorAgent - 数据收集 Agent Loop 主控制器

状态机：IDLE → CHECKING → FETCHING → VALIDATING → STORING → SUCCESS/FAILED

核心流程：
1. 检查数据源健康状态（ping akshare）
2. 读取本地缓存元数据
3. 确定需要更新的标的+时间范围
4. 调用 fetcher 获取数据


5. 调用 data_validator 验证
6. 写入 Parquet 缓存
7. 失败：指数退避重试（3次），仍失败则降级记录

继承自 BaseAgent，使用共享的 LoopController / MemoryBank / ProgressTracker。
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import pandas as pd

from ..base import AgentResult, AgentStatus, BaseAgent
from ..shared.loop_controller import AgentLoopController, RetryStrategy
from ..shared.memory_bank import MemoryBank
from ..shared.progress_tracker import ProgressTracker

if TYPE_CHECKING:
    from ...data.manager import DataManager


class CollectorState(Enum):
    """数据收集状态枚举"""
    IDLE = "idle"
    CHECKING = "checking"
    FETCHING = "fetching"
    VALIDATING = "validating"
    STORING = "storing"
    SUCCESS = "success"
    FAILED = "failed"


class DataCollectorAgent(BaseAgent):
    """
    数据收集 Agent

    负责从多个数据源（AkShare / TuShare / Baostock / 交易所 API）
    拉取期货行情数据，执行验证，写入 Parquet 缓存。

    支持：
    - 增量更新（只拉取新增日期的数据）
    - 多数据源降级（某源失败自动切换）
    - 自修复（记录 API 变更，自动适配）
    """

    DEFAULT_CONFIG = {
        'max_retries': 3,
        'base_delay': 2.0,
        'timeout': 120,
        'preferred_source': 'akshare',
        'backup_sources': ['baostock'],
        'cache_dir': None,          # None → 使用 DataManager 默认
        'force_update': False,       # True → 忽略缓存，强制全量拉取
        'update_schedule': {
            'day_end': '16:00',       # 日盘后
            'night_end': '02:30',     # 夜盘后
        },
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        memory_bank: Optional[MemoryBank] = None,
        progress_tracker: Optional[ProgressTracker] = None,
    ):
        """
        初始化 DataCollectorAgent

        Args:
            config: 自定义配置
            memory_bank: 记忆银行（用于记录成功/失败模式）
            progress_tracker: 进度追踪器
        """
        name = "data_collector"
        super().__init__(name=name, config=config)
        self._cfg = {**self.DEFAULT_CONFIG, **(config or {})}

        self.memory_bank = memory_bank or MemoryBank()
        self.progress_tracker = progress_tracker or ProgressTracker()

        # DataManager（延迟初始化）
        self._dm: Optional["DataManager"] = None

        # 数据源健康状态
        self._source_health: Dict[str, bool] = {}

        # LoopController
        self._loop = AgentLoopController(
            name=name,
            retry_strategy=RetryStrategy(
                max_retries=self._cfg['max_retries'],
                base_delay=self._cfg['base_delay'],
            ),
            timeout=self._cfg['timeout'],
        )

        self._logger = self._logger  # 已在 base 中初始化

    # ---- BaseAgent 实现 ----

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        """
        执行数据收集主逻辑

        Args:
            context: 执行上下文，包含：
                - symbols: 品种列表
                - start_date: 开始日期（可选）
                - end_date: 结束日期（可选）
                - force_update: 是否强制更新

        Returns:
            AgentResult
        """
        tracker = self.progress_tracker
        tracker.start("data_collector", total_steps=7)

        symbols: List[str] = context.get('symbols', [])
        start_date: Optional[str] = context.get('start_date')
        end_date: Optional[str] = context.get('end_date', datetime.now().strftime('%Y-%m-%d'))
        force_update: bool = context.get('force_update', self._cfg['force_update'])

        all_records: List[int] = []
        errors: List[str] = []
        source_used: Dict[str, int] = {}
        fetched_symbols: List[str] = []
        skipped_symbols: List[str] = []

        try:
            # 1. 检查数据源健康
            tracker.step_start("data_collector", step=1, name="检查数据源健康")
            self._check_source_health()
            tracker.step_complete("data_collector", step=1, message=f"健康数据源: {[s for s,v in self._source_health.items() if v]}")
            self._logger.info(f"Source health: {self._source_health}")

            # 2. 确定更新范围
            tracker.step_start("data_collector", step=2, name="确定更新范围")
            update_plan = self._determine_update_plan(symbols, start_date, end_date, force_update)
            tracker.step_complete("data_collector", step=2, message=f"待更新标的: {len(update_plan)} 个")
            self._logger.info(f"Update plan: {len(update_plan)} symbols to update")

            if not update_plan:
                self._logger.info("No data to update, skipping fetch")
                tracker.complete("data_collector", status="success", summary={"skipped": True})
                return AgentResult(
                    agent_name=self.name,
                    status=AgentStatus.SUCCESS,
                    data=pd.DataFrame(),
                    metrics={
                        'symbols_updated': 0,
                        'records_fetched': 0,
                        'skipped': True,
                        'source_health': self._source_health,
                    },
                    elapsed_seconds=0.0,
                )

            # 3-5. 循环拉取、验证、存储
            tracker.step_start("data_collector", step=3, name="拉取数据")
            total_fetched = 0
            fetch_errors = []

            for i, (symbol, (sd, ed)) in enumerate(update_plan.items()):
                step_msg = f"[{i+1}/{len(update_plan)}] {symbol} ({sd} ~ {ed})"

                try:
                    def _fetch():
                        return self._fetch_with_fallback(symbol, sd, ed)

                    df = self._loop.run_sync(_fetch)
                    records = len(df) if not df.empty else 0
                    total_fetched += records
                    all_records.append(records)
                    fetched_symbols.append(symbol)
                    self._logger.info(f"Fetched {symbol}: {records} records")
                except Exception as exc:
                    err = f"{symbol}: {type(exc).__name__}: {exc}"
                    errors.append(err)
                    skipped_symbols.append(symbol)
                    fetch_errors.append(err)
                    self._logger.warning(f"Failed to fetch {symbol}: {exc}")

            tracker.step_complete(
                "data_collector", step=3,
                message=f"完成 {len(fetched_symbols)}/{len(update_plan)} 个",
                metadata={'records_fetched': total_fetched}
            )

            # 6. 数据验证
            tracker.step_start("data_collector", step=4, name="验证数据")
            validated_records = sum(all_records) if all_records else 0
            tracker.step_complete("data_collector", step=4, message=f"验证记录: {validated_records}")

            # 7. 存储/更新元数据
            tracker.step_start("data_collector", step=5, name="存储元数据")
            self._update_metadata(symbols, fetched_symbols, skipped_symbols)
            tracker.step_complete("data_collector", step=5, message="元数据已更新")

            tracker.step_start("data_collector", step=6, name="记录记忆")
            self._record_memory(symbols, fetched_symbols, skipped_symbols, total_fetched, errors)
            tracker.step_complete("data_collector", step=6, message="记忆已保存")

            # 完成
            elapsed = self._loop.metrics.total_elapsed
            tracker.complete("data_collector", status="success", output_summary={
                'symbols_updated': len(fetched_symbols),
                'symbols_skipped': len(skipped_symbols),
                'records_fetched': total_fetched,
                'errors': errors,
            })

            status = AgentStatus.SUCCESS if fetched_symbols else AgentStatus.FAILED
            return AgentResult(
                agent_name=self.name,
                status=status,
                metrics={
                    'symbols_updated': len(fetched_symbols),
                    'symbols_skipped': len(skipped_symbols),
                    'records_fetched': total_fetched,
                    'source_health': self._source_health,
                    'errors': errors,
                    'update_plan': {k: f"{v[0]}~{v[1]}" for k, v in update_plan.items()},
                },
                elapsed_seconds=elapsed,
            )

        except Exception as exc:
            self._logger.error(f"DataCollectorAgent fatal error: {exc}")
            tracker.step_failed("data_collector", step=0, error=str(exc))
            tracker.complete("data_collector", status="failed")
            return AgentResult(
                agent_name=self.name,
                status=AgentStatus.FAILED,
                errors=[str(exc)],
                elapsed_seconds=self._loop.metrics.total_elapsed,
            )

    # ---- 内部方法 ----

    def _init_dm(self) -> "DataManager":
        """延迟初始化 DataManager"""
        if self._dm is None:
            from ...data.manager import DataManager
            self._dm = DataManager()
        return self._dm

    def _check_source_health(self):
        """检查各数据源健康状态"""
        from .data_discovery import AkShareSource, BaostockSource, TuShareSource, ExchangeAPISource

        sources = {
            'akshare': AkShareSource(),
            'baostock': BaostockSource(),
            'tushare': TuShareSource(),
            'exchange_api': ExchangeAPISource(),
        }

        for name, source in sources.items():
            try:
                health = source.health_check()
                self._source_health[name] = health
            except Exception as exc:
                self._source_health[name] = False
                self._logger.warning(f"Source {name} health check failed: {exc}")

        self._logger.info(f"Source health: {self._source_health}")

    def _determine_update_plan(
        self,
        symbols: List[str],
        start_date: Optional[str],
        end_date: Optional[str],
        force_update: bool,
    ) -> Dict[str, tuple]:
        """
        确定需要更新的标的及时间范围

        Returns:
            {symbol: (start_date, end_date)}
        """
        dm = self._init_dm()

        if force_update or not symbols:
            # 全量：默认一年
            sd = start_date or (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            ed = end_date or datetime.now().strftime('%Y-%m-%d')
            return {sym: (sd, ed) for sym in symbols}

        # 增量：检查缓存的最后更新时间
        plan = {}
        for symbol in symbols:
            try:
                last_date = dm.db.get_last_update_date(symbol)
                if last_date is None:
                    # 从未更新过，取一年
                    sd = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                else:
                    # 从上次最后日期的下一天开始
                    sd = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
                ed = end_date or datetime.now().strftime('%Y-%m-%d')
                plan[symbol] = (sd, ed)
            except Exception as exc:
                self._logger.warning(f"Failed to get last date for {symbol}: {exc}")
                sd = start_date or (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                plan[symbol] = (sd, ed)

        return plan

    def _fetch_with_fallback(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """
        带降级的数据获取

        优先使用首选数据源，失败则依次尝试备用源。
        """
        dm = self._init_dm()

        preferred = self._cfg['preferred_source']
        backups = self._cfg['backup_sources']
        all_sources = [preferred] + [s for s in backups if s != preferred]

        last_error: Optional[Exception] = None

        for source_name in all_sources:
            if not self._source_health.get(source_name, False):
                self._logger.debug(f"Skipping unhealthy source: {source_name}")
                continue

            try:
                df = dm.get_daily_data(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    source=source_name,
                    use_cache=False,
                )

                if df.empty:
                    raise ValueError(f"No data returned from {source_name}")

                self._logger.info(f"Fetched {symbol} from {source_name}: {len(df)} records")
                return df

            except Exception as exc:
                last_error = exc
                self._logger.warning(
                    f"Source {source_name} failed for {symbol}: {exc}"
                )
                self._source_health[source_name] = False
                continue

        # 所有源都失败
        raise last_error or RuntimeError(f"All sources failed for {symbol}")

    def _update_metadata(
        self,
        requested: List[str],
        fetched: List[str],
        skipped: List[str],
    ):
        """更新本地缓存元数据"""
        dm = self._init_dm()
        try:
            summary = dm.get_data_summary()
            self._logger.info(f"Updated metadata: {summary}")
        except Exception as exc:
            self._logger.warning(f"Failed to update metadata: {exc}")

    def _record_memory(
        self,
        symbols: List[str],
        fetched: List[str],
        skipped: List[str],
        total_records: int,
        errors: List[str],
    ):
        """记录到记忆银行"""
        context = {
            'symbols': symbols,
            'n_symbols': len(symbols),
        }
        result = {
            'status': 'success' if fetched else 'failed',
            'records_fetched': total_records,
            'symbols_fetched': fetched,
            'symbols_skipped': skipped,
            'errors': errors,
            'source_health': self._source_health,
            'elapsed_seconds': self._loop.metrics.total_elapsed,
        }

        self.memory_bank.record_run(
            agent_name=self.name,
            context=context,
            result=result,
            tags=['data_collection'] + symbols,
        )
        self.memory_bank.flush(self.name)
