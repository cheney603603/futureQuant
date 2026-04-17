"""
Web Fundamental Agent - 基于真实网络数据的基本面分析 Agent

功能：
1. 联网搜索最近一周的 {品种} 新闻、库存、基差、政策
2. 结合价格数据计算支撑压力位（近期高低点/枢轴点）
3. 输出库存周期、多空情绪评分（-5~+5）、支撑压力位
4. 将结构化结果写入 Blackboard
"""

from typing import Any, Dict, Optional

import pandas as pd

from ..base import AgentResult, AgentStatus
from ..react_base import ReActAgent
from ..tools import (
    BlackboardReadTool,
    BlackboardWriteTool,
    CodeExecutionTool,
    WebSearchTool,
)
from ...core.config import get_config
from ...core.logger import get_logger

logger = get_logger("agent.fundamental.web")


class WebFundamentalAgent(ReActAgent):
    """
    基本面分析 Agent（ReAct 版）

    context 支持:
    - target (str): 品种代码，如 "RB"
    - price_data_key (str): blackboard 上价格数据 key，默认 "price_data"
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        blackboard=None,
        llm_client=None,
    ):
        super().__init__(
            name="fundamental_analysis",
            config=config,
            llm_client=llm_client,
        )
        self.register_tools(
            WebSearchTool(),
            CodeExecutionTool(),
            BlackboardReadTool(blackboard),
            BlackboardWriteTool(blackboard),
        )

    @property
    def system_prompt(self) -> str:
        varieties = ", ".join(get_config().varieties)
        return (
            "你是 futureQuant 的基本面分析 Agent。你的任务是基于最新的网络信息，"
            "对指定期货品种进行基本面分析并给出多空评分。\n\n"
            "执行规则（严格按顺序）：\n"
            "1. 使用 web_search 搜索最近一周（time_range='week'）的 {品种} 相关新闻、库存、基差、政策。"
            "至少进行 2-3 轮不同关键词的搜索（如 '{品种} 库存 最近一周'、'{品种} 基差'、'{品种} 新闻'）。\n"
            "2. 如果 blackboard 上有价格数据，读取最近 20-60 日的 K 线，"
            "使用 code_execution 计算近期高点、近期低点、枢轴点（Pivot Points）作为支撑压力位。\n"
            "3. 综合搜索结果和价格数据，使用 code_execution 进行数据清洗和评分计算。\n"
            "4. 输出一个结构化 JSON 结论，并使用 blackboard_write 写入黑板，key 为 'fundamental_analysis'。"
            "JSON 必须包含以下字段：\n"
            "   - sentiment_score (float, -5~+5): 多空情绪评分\n"
            "   - inventory_cycle (str): 主动补库 / 被动补库 / 主动去库 / 被动去库\n"
            "   - supply_demand (str): tight / balanced / loose\n"
            "   - support_level (float): 支撑位\n"
            "   - resistance_level (float): 压力位\n"
            "   - pivot_level (float): 枢轴点\n"
            "   - drivers (list): 驱动因素列表，每项包含 factor/direction/score\n"
            "5. 最后输出 FINAL_ANSWER: <多空评分和关键结论的简要文字总结>\n\n"
            "支持的期货品种：" + varieties
        )

    def execute(self, context: Dict[str, Any]) -> AgentResult:
        target = context.get("target", "UNKNOWN")
        price_data_key = context.get("price_data_key", "price_data")

        # 先读取价格数据（如果有），注入到 context 中供 ReAct 使用
        price_data = self._get_price_data(price_data_key)
        enriched_context = {
            "task": (
                f"分析 {target} 的基本面。"
                f"价格数据: {'可用' if price_data is not None else '不可用'}"
            ),
            "target": target,
            "price_data": price_data.to_dict("list") if price_data is not None else None,
        }

        # 运行 ReAct 循环（LLM 会自主搜索、计算、写黑板）
        result = super().execute(enriched_context)

        # 从黑板读取最终结构化结果（如果 LLM 正确写入）
        fa_entry = self._tools.execute("blackboard_read", key="fundamental_analysis")
        if fa_entry.success and fa_entry.data:
            fa_data = fa_entry.data
            if isinstance(fa_data, dict):
                result.data = fa_data
                result.metrics["fundamental_analysis"] = fa_data
                result.status = AgentStatus.SUCCESS
                result.logs.append(f"基本面分析完成：sentiment={fa_data.get('sentiment_score')}")
            else:
                result.status = AgentStatus.FAILED
                result.errors.append("Blackboard 'fundamental_analysis' is not a dict")
        else:
            # LLM 可能没有正确写入，检查 reasoning_log 是否包含足够信息
            log = self.get_reasoning_log()
            if log and log.final_answer:
                result.data = {"summary": log.final_answer}
            else:
                result.status = AgentStatus.FAILED
                result.errors.append("Failed to retrieve fundamental_analysis from blackboard")

        return result

    def _get_price_data(self, key: str) -> Optional[pd.DataFrame]:
        bb_res = self._tools.execute("blackboard_read", key=key)
        if bb_res.success and bb_res.data:
            try:
                return pd.DataFrame(bb_res.data)
            except Exception:
                pass
        return None
