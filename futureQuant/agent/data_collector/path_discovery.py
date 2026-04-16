"""
PathDiscovery - 新路径探测引擎

核心职责：
1. 联网搜索获取数据获取方案（使用 Web 搜索 API）
2. 尝试探测新数据源/接口
3. 验证探测结果
4. 将成功的新链路注册到可靠链路库

探测策略：
- 按数据类型分别探测：
  - daily（日线）
  - minute（分钟线）
  - tick（Tick数据）
  - inventory（库存）
  - warehouse_receipt（仓单）
  - basis（基差）
  - macroeconomic（宏观数据）
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from ...core.logger import get_logger
from .reliable_path_manager import ReliablePath, ReliablePathManager

logger = get_logger('agent.data_collector.discovery')


@dataclass
class DiscoveryResult:
    """探测结果"""
    success: bool = False
    path: Optional[ReliablePath] = None
    data: Optional[pd.DataFrame] = None
    message: str = ""
    source: str = ""
    data_type: str = ""
    records: int = 0
    response_ms: float = 0.0
    quality_score: float = 0.5
    attempts: int = 0
    error: str = ""
    suggestions: List[str] = field(default_factory=list)  # 下次探测建议


class PathDiscovery:
    """
    新路径探测引擎

    工作流程：
    1. 分析需求（数据类型、标的、时间范围）
    2. 联网搜索可行方案
    3. 逐个尝试方案
    4. 验证结果
    5. 成功则注册到链路库
    """

    def __init__(
        self,
        path_manager: Optional[ReliablePathManager] = None,
    ):
        """
        Args:
            path_manager: 可靠链路管理器
        """
        self._pm = path_manager or ReliablePathManager()
        self._discovery_history: List[DiscoveryResult] = []

    # ==================== 主动探测 ====================

    def discover(
        self,
        data_type: str,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_attempts: int = 3,
        user_permission_callback: Optional[Callable[[str, Dict], bool]] = None,
    ) -> DiscoveryResult:
        """
        执行新路径探测

        Args:
            data_type: 数据类型 (daily/minute/tick/inventory/warehouse_receipt/basis)
            symbol: 标的代码（可选，用于精确探测）
            start_date: 开始日期
            end_date: 结束日期
            max_attempts: 最大尝试次数
            user_permission_callback: 用户授权回调，签名 f(title, details) -> bool

        Returns:
            DiscoveryResult
        """
        logger.info(f"[Discovery] Starting discovery for {data_type}, symbol={symbol}")
        result = DiscoveryResult(
            source="discovery",
            data_type=data_type,
        )

        # 1. 联网搜索可行方案
        plans = self._search_plans(data_type, symbol)
        if not plans:
            result.message = "未找到可行的数据获取方案"
            result.error = "search_no_results"
            logger.warning("[Discovery] No plans found from web search")
            return result

        result.suggestions = plans
        result.attempts = len(plans)

        # 2. 逐个尝试
        for i, plan in enumerate(plans[:max_attempts]):
            plan_name = plan.get('name', f"plan_{i}")
            plan_source = plan.get('source', 'unknown')
            plan_params = plan.get('params', {})
            logger.info(f"[Discovery] Trying plan {i+1}/{len(plans)}: {plan_name} (source={plan_source})")

            try:
                start_ms = time.time() * 1000
                df = self._execute_plan(
                    plan_source=plan_source,
                    plan_params=plan_params,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    data_type=data_type,
                )
                elapsed_ms = time.time() * 1000 - start_ms
                result.response_ms = elapsed_ms

                if df is None or df.empty:
                    logger.warning(f"[Discovery] Plan {plan_name} returned empty data")
                    continue

                # 3. 验证数据质量
                quality = self._assess_quality(df, data_type)
                if quality < 0.3:
                    logger.warning(f"[Discovery] Plan {plan_name} data quality too low: {quality:.2f}")
                    continue

                # 4. 数据有效，记录
                result.success = True
                result.data = df
                result.records = len(df)
                result.quality_score = quality
                result.message = f"通过 {plan_source} 获取 {len(df)} 条记录"

                # 5. 注册新链路
                path, need_confirm = self._pm.register_path(
                    source=plan_source,
                    data_type=data_type,
                    symbol_pattern=symbol or "*",
                    params=plan_params,
                    tags=[symbol or "", data_type],
                    ask_user=True,
                )

                if user_permission_callback and need_confirm:
                    ok = user_permission_callback(
                        title=f"发现新数据链路",
                        details={
                            "source": plan_source,
                            "data_type": data_type,
                            "records": len(df),
                            "quality": quality,
                            "response_ms": elapsed_ms,
                        }
                    )
                    if ok:
                        self._pm.confirm_path(path.path_id, success=True,
                                            response_ms=elapsed_ms, records=len(df))
                        result.path = path
                        result.message += " (已注册到可靠链路库)"
                    else:
                        result.message += " (用户取消注册)"
                else:
                    self._pm.confirm_path(path.path_id, success=True,
                                        response_ms=elapsed_ms, records=len(df))
                    result.path = path
                    result.message += " (已自动注册)"

                logger.info(f"[Discovery] SUCCESS: {result.message}")
                return result

            except Exception as exc:
                logger.warning(f"[Discovery] Plan {plan_name} failed: {exc}")
                result.error = str(exc)
                continue

        result.message = f"所有方案尝试完毕均失败，最后错误: {result.error}"
        logger.error(f"[Discovery] FAILED: {result.message}")
        return result

    def discover_for_symbol(
        self,
        symbol: str,
        data_types: Optional[List[str]] = None,
    ) -> Dict[str, DiscoveryResult]:
        """
        为单个标的探测所有数据类型

        Args:
            symbol: 标的代码
            data_types: 数据类型列表，默认全部

        Returns:
            {data_type: DiscoveryResult}
        """
        if data_types is None:
            data_types = ['daily', 'minute', 'basis', 'inventory', 'warehouse_receipt']

        results = {}
        for dtype in data_types:
            results[dtype] = self.discover(
                data_type=dtype,
                symbol=symbol,
            )
        return results

    # ==================== 方案搜索（联网） ====================

    def _search_plans(
        self,
        data_type: str,
        symbol: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        联网搜索可行方案

        返回格式：
        [{
            "name": "akshare_螺纹钢日线",
            "source": "akshare",
            "params": {"symbol": "RB", "market": "shfe"},
            "confidence": 0.95,
        }, ...]
        """
        plans: List[Dict[str, Any]] = []

        # 构建搜索查询
        query = self._build_search_query(data_type, symbol)
        logger.info(f"[Discovery] Searching for: {query}")

        try:
            # 使用 web_search 工具
            search_results = self._web_search(query)
            plans = self._parse_search_results(search_results, data_type, symbol)
        except Exception as exc:
            logger.warning(f"[Discovery] Web search failed: {exc}, using fallback plans")
            plans = self._fallback_plans(data_type, symbol)

        # 无论如何，至少返回 fallback plans
        if not plans:
            plans = self._fallback_plans(data_type, symbol)

        # 按置信度排序
        plans.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return plans

    def _build_search_query(self, data_type: str, symbol: Optional[str]) -> str:
        """构建搜索查询"""
        symbol_name = self._get_symbol_name(symbol) if symbol else ""

        queries = {
            'daily': f"{symbol_name}期货 日线数据 akshare tushare python 获取",
            'minute': f"{symbol_name}期货 分钟数据 API python 实时",
            'tick': f"{symbol_name}期货 tick数据 获取方式 python",
            'inventory': f"{symbol_name}期货 库存数据 获取 2024",
            'warehouse_receipt': f"{symbol_name}期货 仓单数据 获取",
            'basis': f"{symbol_name}期货 基差数据 获取",
            'macroeconomic': "中国 宏观经济数据 API python GDP CPI",
        }
        return queries.get(data_type, f"{symbol_name}期货 数据 获取")

    def _get_symbol_name(self, symbol: str) -> str:
        """标的代码映射到中文名"""
        mapping = {
            'RB': '螺纹钢', 'HC': '热卷', 'I': '铁矿石',
            'J': '焦炭', 'JM': '焦煤', 'ZC': '动力煤',
            'AL': '铝', 'CU': '铜', 'ZN': '锌',
            'NI': '镍', 'AU': '黄金', 'AG': '白银',
            'RU': '橡胶', 'FU': '燃油', 'BU': '沥青',
            'PP': '聚丙烯', 'PE': '聚乙烯', 'PVC': 'PVC',
            'MA': '甲醇', 'TA': 'PTA', 'EG': '乙二醇',
            'L': '塑料', 'V': 'PVC',
        }
        prefix = symbol[:2] if len(symbol) >= 2 else symbol[:1]
        return mapping.get(prefix.upper(), symbol)

    def _web_search(self, query: str) -> List[Dict[str, str]]:
        """执行网络搜索（调用 web_search 工具）"""
        try:
            from ...core.web_search import web_search
            results = web_search(query, count=8)
            return results
        except ImportError:
            logger.warning("web_search not available, using fallback")
            return []
        except Exception as exc:
            logger.warning(f"web_search error: {exc}")
            return []

    def _parse_search_results(
        self,
        results: List[Dict[str, str]],
        data_type: str,
        symbol: Optional[str],
    ) -> List[Dict[str, Any]]:
        """解析搜索结果为可行方案"""
        plans = []
        for r in results:
            title = r.get('title', '').lower()
            snippet = r.get('snippet', '').lower()

            # 识别数据源
            source = None
            confidence = 0.5
            if 'akshare' in title or 'akshare' in snippet:
                source = 'akshare'
                confidence = 0.9
            elif 'tushare' in title or 'tushare' in snippet:
                source = 'tushare'
                confidence = 0.85
            elif 'baostock' in title or 'baostock' in snippet:
                source = 'baostock'
                confidence = 0.7
            elif '东方财富' in title or 'eastmoney' in title:
                source = 'eastmoney'
                confidence = 0.75
            elif '新浪' in title or 'sina' in title:
                source = 'sina'
                confidence = 0.6

            if source:
                prefix = symbol[:2] if symbol else 'RB'
                params = {'symbol': prefix, 'variety': prefix}
                if data_type == 'daily':
                    params = {'symbol': prefix}
                elif data_type == 'minute':
                    params = {'symbol': prefix, 'period': '15min'}

                plans.append({
                    'name': f"{source}_{data_type}_{prefix}",
                    'source': source,
                    'params': params,
                    'confidence': confidence,
                    'url': r.get('url', ''),
                })

        return plans

    def _fallback_plans(
        self,
        data_type: str,
        symbol: Optional[str],
    ) -> List[Dict[str, Any]]:
        """后备方案（当搜索不可用时）"""
        prefix = symbol[:2] if symbol else 'RB'
        plans = []

        if data_type == 'daily':
            plans = [
                {'name': f'akshare_{prefix}_daily', 'source': 'akshare',
                 'params': {'symbol': prefix}, 'confidence': 0.95},
                {'name': f'baostock_{prefix}_daily', 'source': 'baostock',
                 'params': {'code': prefix}, 'confidence': 0.7},
            ]
        elif data_type == 'minute':
            plans = [
                {'name': f'akshare_{prefix}_minute', 'source': 'akshare',
                 'params': {'symbol': prefix, 'period': '15min'}, 'confidence': 0.8},
            ]
        elif data_type in ('inventory', 'warehouse_receipt', 'basis'):
            plans = [
                {'name': f'akshare_{data_type}_{prefix}', 'source': 'akshare',
                 'params': {'symbol': prefix, 'varity': prefix}, 'confidence': 0.75},
            ]

        return plans

    # ==================== 方案执行 ====================

    def _execute_plan(
        self,
        plan_source: str,
        plan_params: Dict[str, Any],
        symbol: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        data_type: str,
    ) -> pd.DataFrame:
        """执行一个具体方案"""
        from ...data.fetcher.akshare_fetcher import AKShareFetcher
        from ...data.manager import DataManager

        dm = DataManager()

        # 构造标的
        sym = symbol or plan_params.get('symbol', 'RB')
        sd = start_date or plan_params.get('start_date', '')
        ed = end_date or plan_params.get('end_date', '')

        if plan_source == 'akshare':
            fetcher = AKShareFetcher()
            df = fetcher.fetch_daily(symbol=sym, start_date=sd, end_date=ed)
            return df

        elif plan_source == 'baostock':
            # Baostock 通过 DataManager 间接调用
            df = dm.get_daily_data(symbol=sym, start_date=sd, end_date=ed, source='baostock')
            return df

        elif plan_source == 'tushare':
            raise NotImplementedError("TuShare requires token configuration")

        # 默认：走 DataManager
        df = dm.get_daily_data(symbol=sym, start_date=sd, end_date=ed, source=plan_source)
        return df

    # ==================== 质量评估 ====================

    def _assess_quality(self, df: pd.DataFrame, data_type: str) -> float:
        """评估数据质量，返回 0-1 分数"""
        if df is None or df.empty:
            return 0.0

        score = 0.0
        checks = 0

        # 1. 必填列检查
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        has_required = all(c in df.columns for c in required_cols)
        if has_required:
            score += 0.3
        checks += 1

        # 2. 行数检查
        if len(df) >= 10:
            score += 0.2
        elif len(df) >= 1:
            score += 0.1
        checks += 1

        # 3. 日期列有效性
        if 'date' in df.columns:
            try:
                dates = pd.to_datetime(df['date'])
                if dates.notna().all():
                    score += 0.15
                # 检查是否有未来日期（异常）
                if (dates > pd.Timestamp.now()).any():
                    score -= 0.1
                checks += 1
            except Exception:
                pass

        # 4. OHLC 逻辑一致性
        if all(c in df.columns for c in ['open', 'high', 'low', 'close']):
            valid_ohlc = (
                (df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            ).sum()
            ratio = valid_ohlc / len(df) if len(df) > 0 else 0
            score += 0.25 * ratio
            checks += 1

        # 5. 数值合理性（价格不能为0或负）
        if 'close' in df.columns:
            reasonable = (df['close'] > 0).sum() / max(len(df), 1)
            score += 0.1 * reasonable
            checks += 1

        # 归一化
        if checks > 0:
            score = score / max(checks, 1) * (5 / 3)  # 放大到接近1
            score = min(1.0, max(0.0, score))

        return round(score, 3)

    # ==================== 历史与统计 ====================

    def get_discovery_history(self) -> List[DiscoveryResult]:
        """获取探测历史"""
        return self._discovery_history

    def get_discovery_stats(self) -> Dict[str, Any]:
        """获取探测统计"""
        if not self._discovery_history:
            return {'total': 0, 'success': 0, 'success_rate': 0.0}

        total = len(self._discovery_history)
        success = sum(1 for r in self._discovery_history if r.success)
        avg_quality = (
            sum(r.quality_score for r in self._discovery_history) / total
        )
        return {
            'total': total,
            'success': success,
            'success_rate': success / total if total else 0.0,
            'avg_quality': avg_quality,
            'avg_response_ms': (
                sum(r.response_ms for r in self._discovery_history) / total
            ),
        }
