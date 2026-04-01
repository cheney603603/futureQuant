"""
AdapterGenerator - 适配器代码生成器

给定品种名（如 "RB" 螺纹钢），自动生成对应的 fetcher 适配器。
使用 LLM 理解数据源 API 文档，生成适配器代码并写入 futureQuant/data/fetcher/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.logger import get_logger

logger = get_logger('agent.data_collector.adapter')

TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "data" / "fetcher"


class AdapterGenerator:
    """
    适配器代码生成器

    功能：
    - 根据品种元数据生成 fetcher 适配器代码
    - 自动推断交易所、合约规则
    - 生成标准接口实现
    """

    # 品种元数据（用于代码生成）
    VARIETY_METADATA: Dict[str, Dict[str, Any]] = {
        'RB': {
            'name': '螺纹钢',
            'exchange': 'SHFE',
            'variety_code': 'RB',
            'unit': '吨',
            'tick': 1,
            'margin_rate': 0.1,
            'active_months': [1, 5, 10],
            'description': '上海期货交易所螺纹钢期货',
        },
        'HC': {
            'name': '热轧卷板',
            'exchange': 'SHFE',
            'variety_code': 'HC',
            'unit': '吨',
            'tick': 1,
            'margin_rate': 0.1,
            'active_months': [1, 5, 10],
            'description': '上海期货交易所热轧卷板期货',
        },
        'I': {
            'name': '铁矿石',
            'exchange': 'DCE',
            'variety_code': 'I',
            'unit': '吨',
            'tick': 0.5,
            'margin_rate': 0.1,
            'active_months': [1, 5, 9],
            'description': '大连商品交易所铁矿石期货',
        },
        'J': {
            'name': '焦炭',
            'exchange': 'DCE',
            'variety_code': 'J',
            'unit': '吨',
            'tick': 0.5,
            'margin_rate': 0.1,
            'active_months': [1, 5, 9],
            'description': '大连商品交易所焦炭期货',
        },
        'JM': {
            'name': '焦煤',
            'exchange': 'DCE',
            'variety_code': 'JM',
            'unit': '吨',
            'tick': 0.5,
            'margin_rate': 0.1,
            'active_months': [1, 5, 9],
            'description': '大连商品交易所焦煤期货',
        },
        'CU': {
            'name': '铜',
            'exchange': 'SHFE',
            'variety_code': 'CU',
            'unit': '吨',
            'tick': 10,
            'margin_rate': 0.1,
            'active_months': list(range(1, 13)),
            'description': '上海期货交易所铜期货',
        },
        'AL': {
            'name': '铝',
            'exchange': 'SHFE',
            'variety_code': 'AL',
            'unit': '吨',
            'tick': 5,
            'margin_rate': 0.1,
            'active_months': list(range(1, 13)),
            'description': '上海期货交易所铝期货',
        },
        'ZN': {
            'name': '锌',
            'exchange': 'SHFE',
            'variety_code': 'ZN',
            'unit': '吨',
            'tick': 5,
            'margin_rate': 0.1,
            'active_months': list(range(1, 13)),
            'description': '上海期货交易所锌期货',
        },
        'AU': {
            'name': '黄金',
            'exchange': 'SHFE',
            'variety_code': 'AU',
            'unit': '克',
            'tick': 0.05,
            'margin_rate': 0.08,
            'active_months': [2, 4, 6, 8, 10, 12],
            'description': '上海期货交易所黄金期货',
        },
        'AG': {
            'name': '白银',
            'exchange': 'SHFE',
            'variety_code': 'AG',
            'unit': '千克',
            'tick': 1,
            'margin_rate': 0.1,
            'active_months': [2, 4, 6, 8, 10, 12],
            'description': '上海期货交易所白银期货',
        },
        'RU': {
            'name': '天然橡胶',
            'exchange': 'SHFE',
            'variety_code': 'RU',
            'unit': '吨',
            'tick': 5,
            'margin_rate': 0.1,
            'active_months': [1, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            'description': '上海期货交易所天然橡胶期货',
        },
        'M': {
            'name': '豆粕',
            'exchange': 'DCE',
            'variety_code': 'M',
            'unit': '吨',
            'tick': 1,
            'margin_rate': 0.1,
            'active_months': [1, 3, 5, 7, 8, 9, 11],
            'description': '大连商品交易所豆粕期货',
        },
        'Y': {
            'name': '豆油',
            'exchange': 'DCE',
            'variety_code': 'Y',
            'unit': '吨',
            'tick': 2,
            'margin_rate': 0.1,
            'active_months': [1, 3, 5, 7, 8, 9, 12],
            'description': '大连商品交易所豆油期货',
        },
        'P': {
            'name': '棕榈油',
            'exchange': 'DCE',
            'variety_code': 'P',
            'unit': '吨',
            'tick': 2,
            'margin_rate': 0.1,
            'active_months': [2, 3, 4, 5, 6, 7, 8, 9, 10, 12],
            'description': '大连商品交易所棕榈油期货',
        },
    }

    def __init__(self, output_dir: Optional[Path] = None):
        """
        初始化生成器

        Args:
            output_dir: 输出目录，默认 futureQuant/data/fetcher/
        """
        self.output_dir = output_dir or TEMPLATE_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        variety: str,
        source: str = "akshare",
    ) -> str:
        """
        为指定品种生成适配器代码

        Args:
            variety: 品种代码，如 "RB"
            source: 数据源

        Returns:
            生成的代码字符串
        """
        meta = self.VARIETY_METADATA.get(variety.upper(), {
            'name': variety,
            'exchange': 'UNKNOWN',
            'variety_code': variety.upper(),
            'unit': '吨',
            'tick': 1,
            'margin_rate': 0.1,
            'active_months': [1, 5, 9],
            'description': f'{variety} 期货',
        })

        code = self._render_template(variety, meta, source)
        return code

    def generate_and_save(
        self,
        variety: str,
        source: str = "akshare",
    ) -> Path:
        """
        生成代码并写入文件

        Args:
            variety: 品种代码
            source: 数据源

        Returns:
            生成的文件路径
        """
        code = self.generate(variety, source)
        filename = f"{variety.lower()}_fetcher_{source}.py"
        filepath = self.output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)

        logger.info(f"Generated adapter: {filepath}")
        return filepath

    def _render_template(
        self,
        variety: str,
        meta: Dict[str, Any],
        source: str,
    ) -> str:
        """渲染适配器代码模板"""
        variety_upper = variety.upper()
        variety_lower = variety.lower()

        code = f'''"""
{meta["name"]} 期货数据适配器

自动生成的数据适配器
数据源: {source}
交易所: {meta["exchange"]}
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd

from ....core.logger import get_logger
from ....core.base import DataFetcher

logger = get_logger('data.fetcher.{variety_lower}')


class {variety_upper}Fetcher(DataFetcher):
    """
    {meta["name"]} ({variety_upper}) 数据适配器

    数据源: {source}
    交易所: {meta["exchange"]}
    交易单位: {meta["unit"]}
    最小变动价位: {meta["tick"]}
    保证金率: {meta["margin_rate"]:.0%}

    活跃月份: {meta["active_months"]}
    """

    def __init__(self, token: Optional[str] = None):
        """
        初始化 {meta["name"]} 适配器

        Args:
            token: API Token（{source} 需要时传入）
        """
        self.source = "{source}"
        self.variety = "{variety_upper}"
        self.exchange = "{meta["exchange"]}"
        self._client = None

    @property
    def name(self) -> str:
        return "{variety_lower}_fetcher"

    def _get_client(self):
        """获取数据源客户端"""
        if self._client is None:
            if self.source == "akshare":
                import akshare as ak
                self._client = ak
            elif self.source == "tushare":
                import tushare as ts
                self._client = ts
            else:
                raise ValueError(f"Unsupported source: {{self.source}}")
        return self._client

    def fetch_daily(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        获取 {meta["name"]} 日线数据

        Args:
            symbol: 合约代码，如 "RB2501"
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)

        Returns:
            DataFrame，列：[date, open, high, low, close, volume, open_interest]
        """
        try:
            client = self._get_client()
            df = client.futures_zh_daily_sina(symbol=symbol)

            if df is None or df.empty:
                return pd.DataFrame()

            # 列名标准化
            column_map = {{
                "日期": "date",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume",
                "持仓量": "open_interest",
            }}
            df = df.rename(columns=column_map)

            # 日期过滤
            df["date"] = pd.to_datetime(df["date"])
            df = df[
                (df["date"] >= start_date) & (df["date"] <= end_date)
            ].copy()
            df = df.sort_values("date").reset_index(drop=True)

            logger.info(
                f"Fetched {{symbol}}: {{len(df)}} records "
                f"[{{start_date}}, {{end_date}}]"
            )
            return df

        except Exception as exc:
            logger.error(f"Failed to fetch {{symbol}}: {{exc}}")
            raise

    def fetch_symbols(self, variety: Optional[str] = None) -> List[str]:
        """
        获取 {meta["name"]} 合约列表

        Args:
            variety: 品种代码

        Returns:
            合约代码列表
        """
        try:
            client = self._get_client()
            df = client.futures_zh_daily_sina(symbol="{variety}" or "{variety_upper}")
            if df is not None and "symbol" in df.columns:
                return df["symbol"].unique().tolist()
        except Exception as exc:
            logger.warning(f"Failed to fetch symbols: {{exc}}")
        return []

    def get_active_contracts(self) -> List[str]:
        """获取当前活跃合约列表"""
        # {meta["name"]} 活跃月份: {meta["active_months"]}
        return self.fetch_symbols()

    def get_variety_metadata(self) -> Dict:
        """获取品种元数据"""
        return {{
            "variety": "{variety_upper}",
            "name": "{meta["name"]}",
            "exchange": "{meta["exchange"]}",
            "unit": "{meta["unit"]}",
            "tick": {meta["tick"]},
            "margin_rate": {meta["margin_rate"]},
            "active_months": {meta["active_months"]},
        }}
'''
        return code

    def generate_batch(self, varieties: List[str]) -> List[Path]:
        """
        批量生成适配器

        Args:
            varieties: 品种代码列表

        Returns:
            生成的文件路径列表
        """
        paths = []
        for v in varieties:
            try:
                path = self.generate_and_save(v)
                paths.append(path)
            except Exception as exc:
                logger.error(f"Failed to generate adapter for {v}: {exc}")
        return paths


def generate_adapter(variety: str, source: str = "akshare") -> str:
    """
    快捷函数：为品种生成适配器代码并返回

    Args:
        variety: 品种代码
        source: 数据源

    Returns:
        代码字符串
    """
    gen = AdapterGenerator()
    return gen.generate(variety, source)
