"""
DataQuery - 数据查询 Skill

支持两种查询方式：
1. 自然语言查询：用户用日常语言描述需求
2. 结构化 JSON 查询：标准化查询参数

核心职责：
1. 解析用户查询意图
2. 优先从本地数据库/缓存查询
3. 数据库无结果 → 通过可靠链路查询
4. 链路也无结果 → 触发新路径探测
5. 将查询结果存入数据库（用户授权）
6. 返回结构化结果

查询语言支持（自然语言）：
- "螺纹钢最近30天的日线数据"
- "RB2405 2024年1月到3月的收盘价"
- "AL所有合约的库存数据"
- "获取黄金的分钟数据"
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from ...core.logger import get_logger
from .reliable_path_manager import ReliablePath, ReliablePathManager
from .path_discovery import PathDiscovery, DiscoveryResult

logger = get_logger('agent.data_collector.query')


# ==================== 查询请求/响应数据结构 ====================

@dataclass
class QuerySpec:
    """标准化查询规格"""
    symbols: List[str]  # 标的列表
    data_type: str  # daily | minute | tick | inventory | warehouse_receipt | basis | macroeconomic
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    fields: Optional[List[str]] = None  # 返回字段筛选
    limit: int = 0  # 最多返回条数（0=不限）
    require_user_permission: bool = True  # 查询结果是否需要用户授权才存储


@dataclass
class QueryResult:
    """查询结果"""
    success: bool = False
    data: Optional[pd.DataFrame] = None
    records: int = 0
    source: str = ""  # "db" | "reliable_path" | "discovery" | "none"
    path_id: str = ""
    message: str = ""
    from_cache: bool = False
    cached_records: int = 0
    fresh_records: int = 0
    query_time_ms: float = 0.0
    # 查询详情
    query_spec: Optional[QuerySpec] = None
    # 存储授权状态
    store_authorized: bool = False


# ==================== 自然语言解析器 ====================

class NLQueryParser:
    """
    自然语言查询解析器

    将中文自然语言转换为 QuerySpec

    支持模式：
    - "{品种} 最近 N 天/个月 的 {数据类型}"
    - "{品种} {合约} {时间段} 的 {字段}"
    - "获取 {品种} 的 {数据类型}"
    - "{时间} 的 {品种} {数据类型}"
    """

    # 品种名 → 标准代码映射
    VARIETY_MAP = {
        '螺纹钢': 'RB', '热卷': 'HC', '铁矿石': 'I', '矿石': 'I',
        '焦炭': 'J', '焦煤': 'JM', '动力煤': 'ZC',
        '铝': 'AL', '铜': 'CU', '锌': 'ZN', '镍': 'NI',
        '黄金': 'AU', '白银': 'AG', '螺纹': 'RB',
        '橡胶': 'RU', '燃油': 'FU', '沥青': 'BU',
        '塑料': 'L', '聚乙烯': 'PE', '聚丙烯': 'PP',
        '甲醇': 'MA', 'PTA': 'TA', '乙二醇': 'EG',
        'PVC': 'PVC',
    }

    # 数据类型关键词
    DATA_TYPE_MAP = {
        '日线': 'daily', '日k': 'daily', '日K': 'daily', 'daily': 'daily',
        '分钟': 'minute', '分钟线': 'minute', 'min': 'minute', 'min': 'minute',
        'tick': 'tick', '逐笔': 'tick',
        '库存': 'inventory',
        '仓单': 'warehouse_receipt',
        '基差': 'basis',
        '宏观': 'macroeconomic',
        '行情': 'daily',
        '收盘价': 'daily', '收盘': 'daily',
    }

    # 时间关键词
    TIME_MAP = {
        '今天': 0, '今日': 0,
        '昨天': 1, '昨日': 1,
        '最近': None,  # 需要配合具体天数
        '上周': 7, '本周': 7,
        '上月': 30, '本月': 30,
        '上个月': 30, '这个月': 30,
    }

    CONTRACT_RE = re.compile(r'([A-Z]{2,4})\d{3,4}[CYP]?', re.IGNORECASE)
    DATE_RANGE_RE = re.compile(
        r'(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})?\s*[到~\-至\s]+\s*(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})?'
    )
    RECENT_DAYS_RE = re.compile(r'最近(\d+)[天日月年]')
    YEAR_RE = re.compile(r'(\d{4})年')

    def parse(self, text: str) -> QuerySpec:
        """
        解析自然语言查询

        Args:
            text: 自然语言查询文本

        Returns:
            QuerySpec

        Raises:
            ValueError: 无法解析
        """
        text = text.strip()
        if not text:
            raise ValueError("查询文本不能为空")

        # 1. 提取标的
        symbols = self._extract_symbols(text)
        if not symbols:
            # 默认螺纹钢
            symbols = ['RB']

        # 2. 提取数据类型
        data_type = self._extract_data_type(text)
        if not data_type:
            data_type = 'daily'

        # 3. 提取时间范围
        start_date, end_date = self._extract_date_range(text)
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            # 默认最近30天
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        # 4. 提取字段（如果有）
        fields = self._extract_fields(text)

        return QuerySpec(
            symbols=symbols,
            data_type=data_type,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
            limit=0,
        )

    def _extract_symbols(self, text: str) -> List[str]:
        """提取标的代码"""
        symbols = []

        # 1. 中文品种名
        for name, code in self.VARIETY_MAP.items():
            if name in text:
                if code not in symbols:
                    symbols.append(code)

        # 2. 合约代码（如 RB2405）
        for m in self.CONTRACT_RE.finditer(text.upper()):
            sym = m.group(0).upper()
            if sym not in symbols:
                symbols.append(sym)

        # 3. 纯大写字母（仅限2-4字母的期货品种代码，不接受单字母）
        # 排除常见英文缩写词
        exclude = {'API', 'CSV', 'PDF', 'SQL', 'ETF', 'GDP', 'CPI', 'USA', 'USD', 'EUR', 'GBP', 'HK', 'SSE', 'SZSE'}
        for m in re.finditer(r'\b([A-Z]{2,4})\b', text.upper()):
            sym = m.group(1)
            if sym not in exclude and sym not in symbols:
                symbols.append(sym)

        return symbols

    def _extract_data_type(self, text: str) -> Optional[str]:
        """提取数据类型"""
        for keyword, dtype in self.DATA_TYPE_MAP.items():
            if keyword in text:
                return dtype
        return None

    def _extract_date_range(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        """提取时间范围"""
        # 显式日期范围
        m = self.DATE_RANGE_RE.search(text)
        if m:
            y1, m1 = int(m.group(1)), int(m.group(2))
            d1 = int(m.group(3) or 1)
            y2, m2 = int(m.group(4)), int(m.group(5))
            d2 = int(m.group(6) or 28)
            sd = f"{y1}-{m1:02d}-{d1:02d}"
            ed = f"{y2}-{m2:02d}-{d2:02d}"
            return sd, ed

        # 最近N天
        m = self.RECENT_DAYS_RE.search(text)
        if m:
            n = int(m.group(1))
            ed = datetime.now().strftime('%Y-%m-%d')
            sd = (datetime.now() - timedelta(days=n)).strftime('%Y-%m-%d')
            return sd, ed

        # 年份
        years = self.YEAR_RE.findall(text)
        if years:
            year = int(years[0])
            if '1月' in text or '到' in text and '3月' in text:
                return f"{year}-01-01", f"{year}-03-31"
            return f"{year}-01-01", f"{year}-12-31"

        # 今天/昨天/最近
        for kw, days in self.TIME_MAP.items():
            if kw in text:
                if days is not None:
                    ed = datetime.now().strftime('%Y-%m-%d')
                    sd = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                    return sd, ed
                else:
                    # "最近" 需要配合具体数字（已在 RECENT_DAYS_RE 处理）
                    pass

        return None, None

    def _extract_fields(self, text: str) -> Optional[List[str]]:
        """提取返回字段"""
        field_map = {
            '收盘': ['close'], '开盘': ['open'],
            '最高': ['high'], '最低': ['low'],
            '成交量': ['volume'], '持仓': ['open_interest'],
        }
        fields = []
        for kw, cols in field_map.items():
            if kw in text:
                fields.extend(cols)
        return fields if fields else None


# ==================== 主查询引擎 ====================

class DataQueryEngine:
    """
    数据查询引擎

    三层查询策略：
    1. 本地数据库/缓存查询（最快）
    2. 可靠链路查询（利用历史成功路径）
    3. 新路径探测（最后手段）

    查询 → 结果 → 用户授权存储
    """

    def __init__(
        self,
        path_manager: Optional[ReliablePathManager] = None,
        path_discovery: Optional[PathDiscovery] = None,
    ):
        self._pm = path_manager or ReliablePathManager()
        self._pd = path_discovery or PathDiscovery(path_manager=self._pm)
        self._parser = NLQueryParser()

    # ==================== 对外 API ====================

    def query(
        self,
        query_input: Union[str, Dict[str, Any], QuerySpec],
        store_on_success: bool = False,
        user_permission_callback: Optional[callable] = None,
    ) -> QueryResult:
        """
        执行数据查询

        Args:
            query_input: 查询输入，可以是：
                - str: 自然语言查询
                - dict: JSON 格式查询
                - QuerySpec: 标准化查询对象
            store_on_success: 成功后是否自动存储（需用户授权）
            user_permission_callback: 用户授权回调 f(title, details) -> bool

        Returns:
            QueryResult
        """
        start_ms = time.time() * 1000

        # 1. 解析查询
        try:
            spec = self._parse_query(query_input)
        except ValueError as exc:
            return QueryResult(
                success=False,
                message=f"查询解析失败: {exc}",
            )

        logger.info(f"[Query] {spec.symbols} {spec.data_type} {spec.start_date}~{spec.end_date}")

        # 2. 优先查本地数据库
        result = self._query_local(spec)
        if result.success and result.records > 0:
            result.query_time_ms = time.time() * 1000 - start_ms
            result.source = "db"
            result.from_cache = True
            logger.info(f"[Query] Found {result.records} records in local DB")
            return result

        # 3. 通过可靠链路查询
        result = self._query_via_reliable_paths(spec)
        if result.success and result.records > 0:
            result.query_time_ms = time.time() * 1000 - start_ms
            result.source = "reliable_path"
            logger.info(f"[Query] Found {result.records} via reliable path: {result.path_id}")
            return result

        # 4. 触发新路径探测
        result = self._query_via_discovery(spec, user_permission_callback)
        result.query_time_ms = time.time() * 1000 - start_ms
        if result.success:
            result.source = "discovery"
            logger.info(f"[Query] Discovered {result.records} records")
        else:
            result.source = "none"
            result.message = f"所有途径均未找到数据: {'; '.join([r.message for r in [result]])}"
            logger.warning(f"[Query] No data found for {spec.symbols}")

        return result

    def query_json(self, json_str: str, **kwargs) -> QueryResult:
        """快捷方法：JSON 字符串查询"""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            return QueryResult(success=False, message=f"JSON 解析失败: {exc}")
        return self.query(data, **kwargs)

    def query_nl(self, nl_text: str, **kwargs) -> QueryResult:
        """快捷方法：自然语言查询"""
        return self.query(nl_text, **kwargs)

    # ==================== 内部查询 ====================

    def _query_local(self, spec: QuerySpec) -> QueryResult:
        """查询本地数据库/缓存"""
        from ...data.manager import DataManager
        from ...data.storage.db_manager import DBManager

        result = QueryResult()
        result.query_spec = spec
        all_dfs = []

        try:
            dm = DataManager()
            db = DBManager()

            for symbol in spec.symbols:
                # 1. 查 parquet 缓存
                df_cache = db.load_price_data(
                    symbol=symbol,
                    start_date=spec.start_date,
                    end_date=spec.end_date,
                    data_type=spec.data_type,
                )
                if not df_cache.empty:
                    all_dfs.append(df_cache)
                    continue

                # 2. 查 DataManager
                df = dm.get_daily_data(
                    symbol=symbol,
                    start_date=spec.start_date,
                    end_date=spec.end_date,
                    source='auto',
                    use_cache=True,
                )
                if not df.empty:
                    all_dfs.append(df)

            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                combined = combined.drop_duplicates(subset=['date', 'symbol'], keep='last')
                combined = combined.sort_values('date')
                if spec.fields:
                    available = [c for c in spec.fields if c in combined.columns]
                    if available:
                        combined = combined[available + ['date']]
                if spec.limit > 0:
                    combined = combined.tail(spec.limit)

                result.success = True
                result.data = combined
                result.records = len(combined)
                result.cached_records = len(combined)
                result.message = f"本地缓存找到 {len(combined)} 条记录"
                return result

            result.message = "本地数据库无记录"
            return result

        except Exception as exc:
            logger.warning(f"[Query] Local DB query failed: {exc}")
            result.message = f"本地查询异常: {exc}"
            return result

    def _query_via_reliable_paths(self, spec: QuerySpec) -> QueryResult:
        """通过可靠链路查询"""
        result = QueryResult()
        result.query_spec = spec

        # 获取该数据类型的可靠链路
        paths = self._pm.get_reliable_paths(
            data_type=spec.data_type,
            min_success_rate=0.6,
            limit=3,
        )

        if not paths:
            result.message = "无可用可靠链路"
            return result

        logger.info(f"[Query] Trying {len(paths)} reliable paths")

        for path in paths:
            try:
                start_ms = time.time() * 1000
                df = self._execute_path(path, spec)
                elapsed_ms = time.time() * 1000 - start_ms

                if df is None or df.empty:
                    self._pm.record_run(path.path_id, success=False)
                    continue

                # 质量检查
                quality = self._pd._assess_quality(df, spec.data_type)
                if quality < 0.3:
                    self._pm.record_run(path.path_id, success=False, quality_score=quality)
                    continue

                result.success = True
                result.data = df
                result.records = len(df)
                result.path_id = path.path_id
                result.message = f"通过链路 [{path.path_id}] 获取 {len(df)} 条记录"
                self._pm.record_run(
                    path.path_id, success=True,
                    response_ms=elapsed_ms,
                    records=len(df),
                    quality_score=quality,
                )
                return result

            except Exception as exc:
                logger.warning(f"[Query] Reliable path {path.path_id} failed: {exc}")
                self._pm.record_run(path.path_id, success=False)
                continue

        result.message = f"所有可靠链路均失败"
        return result

    def _query_via_discovery(
        self,
        spec: QuerySpec,
        user_callback: Optional[callable],
    ) -> QueryResult:
        """触发新路径探测"""
        result = QueryResult()
        result.query_spec = spec

        for symbol in spec.symbols:
            logger.info(f"[Query] Starting discovery for {symbol} ({spec.data_type})")
            disc_result = self._pd.discover(
                data_type=spec.data_type,
                symbol=symbol,
                start_date=spec.start_date,
                end_date=spec.end_date,
                user_permission_callback=user_callback,
            )

            if disc_result.success and disc_result.data is not None:
                result.success = True
                result.data = disc_result.data
                result.records = len(disc_result.data)
                result.path_id = disc_result.path.path_id if disc_result.path else ""
                result.message = disc_result.message
                result.quality_score = disc_result.quality_score
                return result
            else:
                result.message = f"{symbol} 探测失败: {disc_result.message}"

        return result

    def _execute_path(self, path: ReliablePath, spec: QuerySpec) -> pd.DataFrame:
        """执行单个可靠链路"""
        from ...data.fetcher.akshare_fetcher import AKShareFetcher

        if path.source == 'akshare':
            fetcher = AKShareFetcher()
            for symbol in spec.symbols:
                df = fetcher.fetch_daily(
                    symbol=symbol,
                    start_date=spec.start_date or '',
                    end_date=spec.end_date or '',
                )
                if df is not None and not df.empty:
                    return df

        elif path.source == 'baostock':
            # Baostock 通过 DataManager 间接调用
            from ...data.manager import DataManager
            dm = DataManager()
            for symbol in spec.symbols:
                df = dm.get_daily_data(
                    symbol=symbol,
                    start_date=spec.start_date or '',
                    end_date=spec.end_date or '',
                    source='baostock',
                    use_cache=False,
                )
                if df is not None and not df.empty:
                    return df
        else:
            raise RuntimeError(f"No executor for source: {path.source}")

    # ==================== 解析 ====================

    def _parse_query(self, query_input: Union[str, Dict[str, Any], QuerySpec]) -> QuerySpec:
        """统一解析入口"""
        if isinstance(query_input, QuerySpec):
            return query_input
        elif isinstance(query_input, dict):
            return self._parse_dict(query_input)
        elif isinstance(query_input, str):
            # 尝试先当 JSON 解析
            try:
                data = json.loads(query_input)
                return self._parse_dict(data)
            except json.JSONDecodeError:
                return self._parser.parse(query_input)
        else:
            raise ValueError(f"Unsupported query type: {type(query_input)}")

    def _parse_dict(self, data: Dict[str, Any]) -> QuerySpec:
        """解析字典格式查询"""
        # 标准化字段名
        symbols = data.get('symbols') or data.get('symbol') or data.get('品种') or []
        if isinstance(symbols, str):
            symbols = [symbols]

        data_type = (
            data.get('data_type') or
            data.get('datatype') or
            data.get('数据类型') or
            'daily'
        )

        return QuerySpec(
            symbols=symbols,
            data_type=data_type,
            start_date=data.get('start_date') or data.get('start') or data.get('开始日期'),
            end_date=data.get('end_date') or data.get('end') or data.get('结束日期'),
            fields=data.get('fields') or data.get('columns') or data.get('字段'),
            limit=int(data.get('limit') or 0),
            require_user_permission=bool(data.get('require_permission', True)),
        )

    # ==================== 结果存储 ====================

    def store_result(
        self,
        result: QueryResult,
        ask_user: bool = True,
    ) -> bool:
        """
        将查询结果存入数据库

        Args:
            result: 查询结果
            ask_user: 是否需要用户授权

        Returns:
            是否存储成功
        """
        if not result.success or result.data is None:
            logger.info("[Query] No data to store")
            return False

        try:
            from ...data.storage.db_manager import DBManager
            db = DBManager()

            for symbol in (result.query_spec.symbols if result.query_spec else []):
                sym_data = result.data
                if 'symbol' in result.data.columns:
                    sym_data = result.data[result.data['symbol'] == symbol]
                if sym_data.empty:
                    continue

                db.save_price_data(
                    df=sym_data,
                    symbol=symbol,
                    data_type=result.query_spec.data_type if result.query_spec else 'daily',
                )
                logger.info(f"[Query] Stored {len(sym_data)} records for {symbol}")

            return True
        except Exception as exc:
            logger.error(f"[Query] Failed to store result: {exc}")
            return False


# ==================== Skill 入口 ====================

class DataQuerySkill:
    """
    数据查询 Skill（面向 Agent 的包装器）

    暴露给 Agent 调用的统一入口：
    - skill.query_nl(nl_text)  # 自然语言
    - skill.query_json(json_text)  # JSON 查询
    - skill.get_stats()  # 查看链路库状态
    """

    def __init__(self):
        self._engine = DataQueryEngine()

    def query(self, query_input: Union[str, Dict], **kwargs) -> QueryResult:
        """
        统一查询入口

        用法：
            >>> skill = DataQuerySkill()
            >>> result = skill.query("螺纹钢最近30天的日线数据")
            >>> result = skill.query({"symbols": ["RB"], "data_type": "daily", "start_date": "2024-01-01", "end_date": "2024-03-01"})
        """
        return self._engine.query(query_input, **kwargs)

    def query_nl(self, text: str, **kwargs) -> QueryResult:
        """自然语言查询"""
        return self._engine.query_nl(text, **kwargs)

    def query_json(self, json_str: str, **kwargs) -> QueryResult:
        """JSON 查询"""
        return self._engine.query_json(json_str, **kwargs)

    def get_path_stats(self) -> Dict[str, Any]:
        """获取可靠链路库统计"""
        return self._engine._pm.get_stats()

    def list_paths(self) -> List[Dict[str, Any]]:
        """列出所有链路"""
        return [p.to_dict() for p in self._engine._pm.get_all_paths()]

    def run_discovery(
        self,
        data_type: str,
        symbol: Optional[str] = None,
        **kwargs
    ) -> DiscoveryResult:
        """手动触发新路径探测"""
        return self._engine._pd.discover(data_type=data_type, symbol=symbol, **kwargs)

    def get_result_summary(self, result: QueryResult) -> Dict[str, Any]:
        """将 QueryResult 转为可读摘要"""
        return {
            'success': result.success,
            'records': result.records,
            'source': result.source,
            'path_id': result.path_id,
            'message': result.message,
            'query_time_ms': round(result.query_time_ms, 1),
            'data_preview': (
                result.data.head(5).to_dict('records') if result.data is not None else []
            ),
        }
