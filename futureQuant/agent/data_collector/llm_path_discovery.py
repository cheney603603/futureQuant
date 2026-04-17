"""
LLM Path Discovery - 基于 LLM 的新路径探测引擎

增强现有 PathDiscovery：
1. 联网搜索可行方案
2. 将搜索结果传给 LLM 生成 Python 数据获取代码
3. 使用 CodeExecutionTool 验证代码
4. 成功则注册到 ReliablePathManager（标记为 llm_generated）
5. 失败 3 次则返回失败，建议人工介入
"""

from typing import Any, Dict, List, Optional

from .path_discovery import DiscoveryResult
from .reliable_path_manager import ReliablePathManager
from ..tools import CodeExecutionTool, WebSearchTool
from ...core.llm_client import LLMClient
from ...core.logger import get_logger

logger = get_logger("agent.data_collector.llm_discovery")


class LLMPathDiscovery:
    """
    LLM 驱动的新路径探测引擎
    """

    MAX_ATTEMPTS = 3

    def __init__(
        self,
        path_manager: Optional[ReliablePathManager] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        self._pm = path_manager or ReliablePathManager()
        self._llm = llm_client or LLMClient()
        self._web_search = WebSearchTool()
        self._code_exec = CodeExecutionTool()

    def discover(
        self,
        data_type: str,
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> DiscoveryResult:
        """
        执行 LLM 驱动的新路径探测
        """
        result = DiscoveryResult(
            source="llm_discovery",
            data_type=data_type,
        )

        query = self._build_search_query(data_type, symbol)
        logger.info(f"[LLMDiscovery] Starting: {query}")

        # 1. 联网搜索
        search_res = self._web_search.execute(query=query, max_results=8)
        if not search_res.success or not search_res.data:
            result.message = "联网搜索未返回结果"
            result.error = "web_search_empty"
            return result

        search_context = self._format_search_results(search_res.data)

        # 2. 尝试生成并验证代码（最多3次）
        for attempt in range(1, self.MAX_ATTEMPTS + 1):
            logger.info(f"[LLMDiscovery] Attempt {attempt}/{self.MAX_ATTEMPTS}")

            code = self._generate_fetcher_code(
                data_type=data_type,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                search_context=search_context,
                previous_errors=result.error if result.error else None,
            )

            if not code:
                result.message = f"第 {attempt} 次代码生成失败"
                continue

            # 3. 验证代码
            exec_res = self._code_exec.execute(code=code, timeout=30)
            result.attempts = attempt

            if not exec_res.success:
                logger.warning(f"[LLMDiscovery] Code execution failed: {exec_res.error}")
                result.error = exec_res.error
                continue

            # 4. 检查返回值是否像 DataFrame（至少应该是 dict/list）
            data = exec_res.data
            if data is None or (isinstance(data, dict) and not data) or (isinstance(data, list) and not data):
                result.error = "代码执行成功但返回空数据"
                continue

            # 5. 注册新链路
            path, _ = self._pm.register_path(
                source="llm_generated",
                data_type=data_type,
                symbol_pattern=symbol or "*",
                params={"code": code, "symbol": symbol},
                tags=[symbol or "", data_type, "llm_generated"],
                ask_user=False,
            )
            self._pm.confirm_path(path.path_id, success=True, records=1)

            result.success = True
            result.data = data
            result.records = len(data) if isinstance(data, list) else 1
            result.message = f"通过 LLM 生成代码成功获取数据（尝试 {attempt} 次）"
            result.path = path
            return result

        result.message = f"所有 {self.MAX_ATTEMPTS} 次尝试均失败，最后错误: {result.error}"
        logger.error(f"[LLMDiscovery] {result.message}")
        return result

    def _build_search_query(self, data_type: str, symbol: Optional[str]) -> str:
        mapping = {
            "RB": "螺纹钢", "HC": "热卷", "I": "铁矿石",
            "J": "焦炭", "JM": "焦煤", "AL": "铝",
            "CU": "铜", "ZN": "锌", "AU": "黄金",
            "TA": "PTA", "MA": "甲醇", "RU": "橡胶",
        }
        name = mapping.get(symbol, symbol) if symbol else ""
        type_names = {
            "daily": "日线数据",
            "minute": "分钟数据",
            "inventory": "库存数据",
            "warehouse_receipt": "仓单数据",
            "basis": "基差数据",
        }
        type_name = type_names.get(data_type, data_type)
        return f"{name}期货 {type_name} python akshare API 获取"

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        lines = []
        for i, r in enumerate(results[:5], 1):
            lines.append(f"{i}. {r.get('title', '')}\n{r.get('body', '')}\n{r.get('href', '')}")
        return "\n".join(lines)

    def _generate_fetcher_code(
        self,
        data_type: str,
        symbol: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        search_context: str,
        previous_errors: Optional[str],
    ) -> Optional[str]:
        """调用 LLM 生成数据获取代码"""
        prompt = (
            "你是一个 Python 数据工程师。请根据以下搜索结果，"
            "编写一段仅使用 pandas / numpy / akshare / requests / bs4 的 Python 代码，"
            "用于获取指定期货数据。代码必须:\n"
            "1. 以 result 变量保存最终数据（DataFrame 或 dict list）\n"
            "2. 不使用 os / sys / subprocess / eval / file write\n"
            "3. 如果 akshare 有相关接口，优先使用 akshare\n"
            "4. 代码应能直接 exec 执行\n\n"
            f"数据类型: {data_type}\n"
            f"品种代码: {symbol or '任意'}\n"
            f"时间范围: {start_date or '不限'} ~ {end_date or '不限'}\n\n"
            f"搜索结果摘要:\n{search_context}\n"
        )
        if previous_errors:
            prompt += f"\n上次尝试失败原因: {previous_errors}\n请修正后重新生成。\n"
        prompt += "\n请只输出 Python 代码，不要输出任何解释。"

        try:
            resp = self._llm.chat([{"role": "user", "content": prompt}], temperature=0.2)
            code = resp.content or ""
            # 清洗 markdown 代码块
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0]
            elif "```" in code:
                code = code.split("```")[1].split("```")[0]
            return code.strip()
        except Exception as exc:
            logger.error(f"[LLMDiscovery] LLM code generation failed: {exc}")
            return None
