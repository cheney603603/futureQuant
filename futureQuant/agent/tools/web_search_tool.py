"""
Web Search Tool - 联网搜索

使用 duckduckgo-search 进行匿名搜索，支持时间范围过滤。
"""

from typing import Any, Dict, List, Optional

from .base import Tool, ToolResult
from ...core.logger import get_logger

logger = get_logger("agent.tools.web_search")


class WebSearchTool(Tool):
    """
    联网搜索工具

    支持参数：
    - query: 搜索关键词
    - time_range: 时间范围过滤 (day, week, month, year)
    - max_results: 最大返回结果数
    """

    name = "web_search"
    description = (
        "Perform a web search using DuckDuckGo. "
        "Useful for finding latest news, API documentation, research reports, and data sources."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query string",
            },
            "time_range": {
                "type": "string",
                "enum": ["day", "week", "month", "year"],
                "description": "Filter results by time range",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 8,
            },
        },
        "required": ["query"],
    }

    def execute(
        self,
        query: str,
        time_range: Optional[str] = None,
        max_results: int = 8,
    ) -> ToolResult:
        try:
            from duckduckgo_search import DDGS
        except ImportError as exc:
            return ToolResult(
                success=False,
                error=f"duckduckgo-search not installed: {exc}",
            )

        logger.info(f"[WebSearch] query='{query}', time_range={time_range}, max_results={max_results}")

        try:
            with DDGS() as ddgs:
                results = ddgs.text(
                    keywords=query,
                    region="cn-zh",
                    safesearch="off",
                    timelimit=time_range,
                    max_results=max_results,
                )
        except Exception as exc:
            logger.warning(f"[WebSearch] failed: {exc}")
            return ToolResult(success=False, error=str(exc))

        parsed: List[Dict[str, Any]] = []
        for r in results:
            parsed.append(
                {
                    "title": r.get("title", ""),
                    "href": r.get("href", ""),
                    "body": r.get("body", ""),
                }
            )

        if not parsed:
            return ToolResult(
                success=True,
                data=[],
                metadata={"query": query, "time_range": time_range},
            )

        return ToolResult(
            success=True,
            data=parsed,
            metadata={
                "query": query,
                "time_range": time_range,
                "count": len(parsed),
            },
        )
