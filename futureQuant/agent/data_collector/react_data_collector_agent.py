"""
React DataCollector Agent - 基于 ReAct 的数据收集 Agent

职责：
1. 优先查询 ReliablePathManager 中的已有链路
2. 若无可用链路，联网搜索并生成数据获取代码
3. 验证代码并执行
4. 将成功结果存入数据库并注册新链路
5. 连续失败后登记 intervention_request 到黑板
"""

from typing import Any, Dict, Optional

from ..react_base import ReActAgent
from ..tools import (
    BlackboardReadTool,
    BlackboardWriteTool,
    CodeExecutionTool,
    DatabaseTool,
    MemoryTool,
    ProgressTool,
    WebSearchTool,
)
from .reliable_path_manager import ReliablePathManager
from ...core.config import get_config
from ...core.logger import get_logger

logger = get_logger("agent.data_collector.react")


class ReactDataCollectorAgent(ReActAgent):
    """
    数据收集 Agent（ReAct 版）

    context 支持 keys:
    - data_type (str): daily | minute | tick | inventory | warehouse_receipt | basis
    - symbol (str): 品种代码，如 "RB"
    - start_date (str): 开始日期
    - end_date (str): 结束日期
    - store_on_success (bool): 成功后是否存储到数据库
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        blackboard=None,
        llm_client=None,
    ):
        super().__init__(
            name="data_collector",
            config=config,
            llm_client=llm_client,
        )
        self._pm = ReliablePathManager()
        self.register_tools(
            WebSearchTool(),
            CodeExecutionTool(),
            DatabaseTool(),
            MemoryTool(),
            ProgressTool(),
            BlackboardReadTool(blackboard),
            BlackboardWriteTool(blackboard),
        )

    @property
    def system_prompt(self) -> str:
        varieties = ", ".join(get_config().varieties)
        return (
            "你是 futureQuant 的数据收集 Agent。你的任务是根据用户需求获取期货相关数据。\n"
            "支持的品种包括：\n" + varieties + "\n\n"
            "执行规则（严格按顺序）：\n"
            "1. 首先使用 web_search 或直接推理检查 blackboard 上是否已有相关数据。"
            "如果 blackboard_read 能读到目标数据键，直接确认完成任务。\n"
            "2. 如果 blackboard 上没有，查询 memory 工具查看历史成功链路。\n"
            "3. 如果记忆中也没有可用方案，使用 web_search 搜索 '{品种} 期货 {数据类型} python API 获取'。"
            "根据搜索结果，使用 code_execution 生成并验证一段获取数据的 Python 代码。"
            "代码只允许使用 pandas, numpy, akshare, requests, bs4 等安全库。\n"
            "4. 若代码验证成功并返回有效 DataFrame，使用 database 工具将数据保存到数据库，"
            "并使用 blackboard_write 将结果键写回黑板。\n"
            "5. 若连续尝试 3 次均失败，使用 blackboard_write 写入一个 intervention_request，"
            "说明失败原因，并输出 FINAL_ANSWER: 失败。\n"
            "6. 一切顺利时，输出 FINAL_ANSWER: 成功，并简要说明数据来源和记录数。"
        )

    def execute(self, context: Dict[str, Any]) -> Any:
        """包装 ReAct 执行，注入任务描述"""
        symbol = context.get("symbol", "RB")
        data_type = context.get("data_type", "daily")
        start_date = context.get("start_date", "")
        end_date = context.get("end_date", "")

        task = (
            f"获取 {symbol} 的 {data_type} 数据"
            f"（时间范围：{start_date} ~ {end_date}）"
        )
        react_context = {"task": task, **context}
        return super().execute(react_context)
