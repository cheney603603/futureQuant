"""
Progress Tool - 进度追踪

封装 ProgressTracker，供 Agent 在 ReAct 循环中报告进度。
"""

from typing import Any, Dict, Optional

from .base import Tool, ToolResult
from ..shared.progress_tracker import ProgressTracker
from ...core.logger import get_logger

logger = get_logger("agent.tools.progress")


class ProgressTool(Tool):
    """
    进度追踪工具

    Agent 可以在执行过程中更新进度，生成可读的进度报告。
    """

    name = "progress_tracker"
    description = (
        "Track and report task progress. "
        "Useful for long-running tasks to record step start/completion and generate markdown reports."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["start", "step_start", "step_complete", "step_failed", "complete", "generate_report"],
                "description": "Progress action",
            },
            "task_name": {
                "type": "string",
                "description": "Unique task identifier",
            },
            "step_name": {
                "type": "string",
                "description": "Step name (for step_* actions)",
            },
            "message": {
                "type": "string",
                "description": "Optional message",
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata dict",
            },
        },
        "required": ["action", "task_name"],
    }

    def __init__(self):
        self._trackers: Dict[str, ProgressTracker] = {}

    def _get_tracker(self, task_name: str) -> ProgressTracker:
        if task_name not in self._trackers:
            self._trackers[task_name] = ProgressTracker(task_name=task_name)
        return self._trackers[task_name]

    def execute(
        self,
        action: str,
        task_name: str,
        step_name: Optional[str] = None,
        message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolResult:
        try:
            tracker = self._get_tracker(task_name)
            meta = metadata or {}

            if action == "start":
                tracker.start(message=message, metadata=meta)
                return ToolResult(success=True, data={"status": "started"})

            elif action == "step_start":
                if step_name is None:
                    return ToolResult(success=False, error="step_name required")
                tracker.step_start(step_name, message=message)
                return ToolResult(success=True, data={"step": step_name, "status": "started"})

            elif action == "step_complete":
                if step_name is None:
                    return ToolResult(success=False, error="step_name required")
                tracker.step_complete(step_name, message=message, metadata=meta)
                return ToolResult(success=True, data={"step": step_name, "status": "completed"})

            elif action == "step_failed":
                if step_name is None:
                    return ToolResult(success=False, error="step_name required")
                tracker.step_failed(step_name, error=message or "Unknown error")
                return ToolResult(success=True, data={"step": step_name, "status": "failed"})

            elif action == "complete":
                tracker.complete(message=message, metadata=meta)
                return ToolResult(success=True, data={"status": "completed"})

            elif action == "generate_report":
                path = tracker.generate_report()
                return ToolResult(success=True, data={"report_path": path})

            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except Exception as exc:
            logger.error(f"ProgressTool error: {exc}")
            return ToolResult(success=False, error=str(exc))
