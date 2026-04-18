"""
SelfHealer - 自修复引擎

检测数据获取失败的根因：
- API 变更（字段名变化/接口下线）
- 网络问题（超时/连接失败）
- 格式异常（列名变化/数据类型错误）

自动修复：
- 更新参数/列名映射
- 调整时间格式
- 记录修复历史到 memory_bank
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ...core.logger import get_logger

logger = get_logger('agent.data_collector.healer')


@dataclass
class HealAction:
    """修复动作记录"""
    action_type: str          # "column_rename", "date_format", "param_update", "retry"
    description: str
    before: Any
    after: Any
    timestamp: str = ""
    success: bool = False
    error: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class HealResult:
    """修复结果"""
    healed: bool = False
    data: Optional[pd.DataFrame] = None
    actions: List[HealAction] = field(default_factory=list)
    message: str = ""
    degraded_level: int = 0  # 0=未降级, 1=轻度, 2=中度, 3=重度

    @property
    def n_actions(self) -> int:
        return len(self.actions)


class SelfHealer:
    """
    自修复引擎

    工作流程：
    1. 检测失败类型
    2. 根据错误模式选择修复策略
    3. 尝试自动修复
    4. 记录修复历史到 memory_bank
    """

    # 标准列名映射（用于修复列名变更）
    COLUMN_ALIASES: Dict[str, List[str]] = {
        'date': ['日期', 'date', 'datetime', '时间', 'trade_date', '交易日期'],
        'open': ['开盘', 'open', '开盘价', '开盘价(元/吨)'],
        'high': ['最高', 'high', '最高价', '最高价(元/吨)'],
        'low': ['最低', 'low', '最低价', '最低价(元/吨)'],
        'close': ['收盘', 'close', '收盘价', '收盘价(元/吨)'],
        'volume': ['成交量', 'volume', 'vol', '成交'],
        'open_interest': ['持仓量', 'open_interest', 'oi', 'position', '持仓'],
    }

    # 已知日期格式
    DATE_FORMATS = [
        '%Y-%m-%d',
        '%Y/%m/%d',
        '%Y%m%d',
        '%d/%m/%Y',
        '%m/%d/%Y',
        '%Y-%m-%d %H:%M:%S',
        '%Y%m%d %H:%M:%S',
    ]

    def __init__(
        self,
        memory_bank: Optional[object] = None,
        max_heal_attempts: int = 3,
    ):
        """
        Args:
            memory_bank: 记忆银行实例（用于记录修复历史）
            max_heal_attempts: 最大修复尝试次数
        """
        self.memory_bank = memory_bank
        self.max_heal_attempts = max_heal_attempts

        # 列名修复映射缓存
        self._column_mappings: Dict[str, Dict[str, str]] = {}

        # 历史修复记录
        self._heal_history: List[HealResult] = []

    def diagnose_and_heal(
        self,
        symbol: str,
        error: Exception,
        raw_data: Any,
        source: str = "unknown",
    ) -> HealResult:
        """
        诊断并自动修复

        Args:
            symbol: 标的代码
            error: 捕获的异常
            raw_data: 原始数据（可能是 None 或部分数据）
            source: 数据源名称

        Returns:
            HealResult
        """
        error_msg = str(error)
        error_type = type(error).__name__
        logger.info(f"[Healer] Diagnosing {symbol} error: {error_type}: {error_msg}")

        result = HealResult()

        # 1. 检测错误类型
        error_category = self._classify_error(error_msg)

        # 2. 根据类型执行修复
        if error_category == 'api_change':
            result = self._heal_api_change(symbol, raw_data)
        elif error_category == 'column_rename':
            result = self._heal_column_rename(raw_data)
        elif error_category == 'date_format':
            result = self._heal_date_format(raw_data)
        elif error_category == 'network':
            result = self._heal_network(raw_data)
        elif error_category == 'data_quality':
            result = self._heal_data_quality(raw_data)
        elif error_category == 'unknown':
            result = self._heal_unknown(symbol, raw_data, error)

        # 3. 记录修复历史
        self._record_heal(symbol, error_type, error_msg, result, source)

        return result

    def _classify_error(self, error_msg: str) -> str:
        """分类错误类型"""
        msg_lower = error_msg.lower()

        # API 变更
        if any(kw in msg_lower for kw in ['404', 'not found', 'no data returned', 'api']):
            if any(kw in msg_lower for kw in ['column', 'field', 'key']):
                return 'column_rename'
            return 'api_change'

        # 网络问题
        if any(kw in msg_lower for kw in [
            'timeout', 'connection', 'network', 'refused',
            'proxy', 'dns', 'ssl', 'temporarily unavailable',
        ]):
            return 'network'

        # 格式问题
        if any(kw in msg_lower for kw in [
            'date', 'datetime', 'parse', 'format',
            'expected', 'got', 'type',
        ]):
            if 'column' in msg_lower or 'field' in msg_lower:
                return 'column_rename'
            return 'date_format'

        # 数据质量问题
        if any(kw in msg_lower for kw in [
            'empty', 'null', 'none', 'missing',
            'duplicate', 'inconsistent',
        ]):
            return 'data_quality'

        return 'unknown'

    def _heal_column_rename(self, raw_data: Any) -> HealResult:
        """修复列名变更"""
        result = HealResult()
        if raw_data is None:
            return result

        if not isinstance(raw_data, pd.DataFrame):
            return result

        df = raw_data.copy()
        renamed_cols: Dict[str, str] = {}

        # 尝试匹配标准列名
        for std_name, aliases in self.COLUMN_ALIASES.items():
            for col in df.columns:
                if col == std_name:
                    continue
                if any(re.match(rf'^{re.escape(alias)}$', str(col), re.IGNORECASE) for alias in aliases):
                    renamed_cols[col] = std_name

        if renamed_cols:
            action = HealAction(
                action_type='column_rename',
                description=f"Renamed {len(renamed_cols)} columns to standard names",
                before=list(renamed_cols.keys()),
                after=list(renamed_cols.values()),
            )
            try:
                df = df.rename(columns=renamed_cols)
                action.success = True
                result.healed = True
                result.data = df
                result.message = f"Renamed columns: {renamed_cols}"
            except Exception as exc:
                action.success = False
                action.error = str(exc)
                result.message = f"Column rename failed: {exc}"
            result.actions.append(action)
            logger.info(f"[Healer] Column rename: {renamed_cols}")
        else:
            result.message = "No columns to rename"

        return result

    def _heal_date_format(self, raw_data: Any) -> HealResult:
        """修复日期格式"""
        result = HealResult()
        if raw_data is None or not isinstance(raw_data, pd.DataFrame):
            return result

        df = raw_data.copy()

        # 查找日期列
        date_col = None
        for col in df.columns:
            if any(kw in col.lower() for kw in ['date', '时间', 'trade']):
                date_col = col
                break

        if date_col is None:
            result.message = "No date column found"
            return result

        # 尝试解析日期
        parsed = False
        for fmt in self.DATE_FORMATS:
            try:
                pd.to_datetime(df[date_col], format=fmt)
                df[date_col] = pd.to_datetime(df[date_col], format=fmt)
                parsed = True
                action = HealAction(
                    action_type='date_format',
                    description=f"Parsed dates with format: {fmt}",
                    before=df[date_col].dtype,
                    after='datetime64[ns]',
                )
                action.success = True
                result.actions.append(action)
                result.healed = True
                result.data = df
                result.message = f"Date format fixed: {fmt}"
                logger.info(f"[Healer] Date format fixed: {fmt}")
                break
            except Exception:
                continue

        if not parsed:
            # 尝试自动推断格式
            try:
                df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
                action = HealAction(
                    action_type='date_format',
                    description="Parsed dates with auto-inferred format",
                    before=df[date_col].dtype,
                    after='datetime64[ns]',
                )
                action.success = True
                result.actions.append(action)
                result.healed = True
                result.data = df
                result.message = "Date format fixed: auto-inferred"
            except Exception as exc:
                result.message = f"Date format fix failed: {exc}"

        return result

    def _heal_api_change(self, symbol: str, raw_data: Any) -> HealResult:
        """修复 API 变更"""
        result = HealResult()
        result.message = "API change detected - using fallback"
        result.degraded_level = 2

        # 记录 API 变更事件
        action = HealAction(
            action_type='api_change',
            description=f"API change detected for symbol {symbol}",
            before="original API",
            after="fallback strategy",
        )
        result.actions.append(action)

        # 如果有原始数据，尝试修复后返回
        if isinstance(raw_data, pd.DataFrame) and not raw_data.empty:
            healed = self._heal_column_rename(raw_data)
            if healed.healed:
                result.healed = True
                result.data = healed.data
                result.message = "API change recovered with column mapping"

        return result

    def _heal_network(self, raw_data: Any) -> HealResult:
        """修复网络问题（添加重试延迟）"""
        result = HealResult()
        result.message = "Network issue - suggest retry with delay"
        result.degraded_level = 1

        action = HealAction(
            action_type='retry',
            description="Network error - suggest exponential backoff retry",
            before=None,
            after="retry_with_delay",
        )
        result.actions.append(action)

        return result

    def _heal_data_quality(self, raw_data: Any) -> HealResult:
        """修复数据质量问题"""
        result = HealResult()
        if raw_data is None or not isinstance(raw_data, pd.DataFrame):
            return result

        df = raw_data.copy()
        issues = []

        # 移除全空行
        before_len = len(df)
        df = df.dropna(how='all')
        if len(df) < before_len:
            issues.append(f"dropped {before_len - len(df)} empty rows")

        # 移除重复行
        before_len = len(df)
        df = df.drop_duplicates()
        if len(df) < before_len:
            issues.append(f"dropped {before_len - len(df)} duplicate rows")

        # 数值列填充
        for col in df.select_dtypes(include=['number']).columns:
            if df[col].isna().any():
                df[col] = df[col].ffill().bfill()
                issues.append(f"filled NaN in {col}")

        if issues:
            result.healed = True
            result.data = df
            result.message = "; ".join(issues)
            result.degraded_level = 1

            action = HealAction(
                action_type='data_quality',
                description="Fixed data quality issues",
                before=before_len,
                after=len(df),
                success=True,
            )
            result.actions.append(action)
            logger.info(f"[Healer] Data quality fixed: {issues}")
        else:
            result.message = "No quality issues found"

        return result

    def _heal_unknown(self, symbol: str, raw_data: Any, error: Exception) -> HealResult:
        """处理未知错误"""
        result = HealResult()
        result.message = f"Unknown error type: {type(error).__name__}: {error}"
        result.degraded_level = 3

        action = HealAction(
            action_type='unknown',
            description=f"Unknown error for {symbol}",
            before=type(error).__name__,
            after="no_recovery",
            success=False,
            error=str(error),
        )
        result.actions.append(action)

        return result

    def _record_heal(
        self,
        symbol: str,
        error_type: str,
        error_msg: str,
        result: HealResult,
        source: str,
    ):
        """记录修复历史到 memory_bank"""
        self._heal_history.append(result)

        if self.memory_bank:
            try:
                self.memory_bank.record_run(
                    agent_name='self_healer',
                    context={
                        'symbol': symbol,
                        'source': source,
                        'error_type': error_type,
                        'error_msg': error_msg,
                    },
                    result={
                        'status': 'healed' if result.healed else 'failed',
                        'healed': result.healed,
                        'n_actions': result.n_actions,
                        'message': result.message,
                        'degraded_level': result.degraded_level,
                    },
                    tags=['healing', f'level_{result.degraded_level}', symbol],
                )
                self.memory_bank.flush('self_healer')
            except Exception as exc:
                logger.warning(f"Failed to record heal history: {exc}")

    def get_heal_stats(self) -> Dict[str, Any]:
        """获取修复统计"""
        if not self._heal_history:
            return {'total': 0, 'healed': 0, 'failed': 0, 'rate': 0.0}

        total = len(self._heal_history)
        healed = sum(1 for r in self._heal_history if r.healed)
        failed = total - healed

        return {
            'total': total,
            'healed': healed,
            'failed': failed,
            'heal_rate': healed / total if total else 0.0,
            'avg_degraded_level': (
                sum(r.degraded_level for r in self._heal_history) / total
            ),
        }
