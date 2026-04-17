"""
Database Tool - 统一数据库操作

自动适配 MySQL（优先）和 SQLite（兜底），支持：
- SQL 查询
- DataFrame 写入
- 表存在性检查
"""

from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from .base import Tool, ToolResult
from ...core.config import get_config
from ...core.logger import get_logger

logger = get_logger("agent.tools.database")


class DatabaseTool(Tool):
    """
    数据库工具

    参数：
    - action: "query" | "execute" | "save_dataframe" | "list_tables"
    - sql: SQL 语句（query/execute 时使用）
    - table: 表名（save_dataframe 时使用）
    - data: DataFrame 字典表示（save_dataframe 时使用）
    - if_exists: "append" | "replace" | "fail"
    """

    name = "database"
    description = (
        "Execute SQL queries or save DataFrames to the database. "
        "Auto-switches between MySQL (primary) and SQLite (fallback)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["query", "execute", "save_dataframe", "list_tables"],
                "description": "Database action",
            },
            "sql": {
                "type": "string",
                "description": "SQL statement for query or execute",
            },
            "table": {
                "type": "string",
                "description": "Target table name for save_dataframe",
            },
            "data": {
                "type": "object",
                "description": "DataFrame dict records for save_dataframe",
            },
            "if_exists": {
                "type": "string",
                "enum": ["append", "replace", "fail"],
                "description": "Behavior when table exists",
                "default": "append",
            },
        },
        "required": ["action"],
    }

    def __init__(self):
        self._engine = None
        self._dialect = "sqlite"
        self._init_engine()

    def _init_engine(self):
        """初始化数据库连接引擎"""
        cfg = get_config()
        mysql_cfg = cfg.data.mysql

        # 尝试 MySQL
        if mysql_cfg and mysql_cfg.get("enabled"):
            try:
                user = mysql_cfg.get("user", "")
                password = mysql_cfg.get("password", "")
                host = mysql_cfg.get("host", "127.0.0.1")
                port = mysql_cfg.get("port", 3306)
                database = mysql_cfg.get("database", "futurequant")

                if user and password:
                    engine_url = (
                        f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
                        "?charset=utf8mb4"
                    )
                    engine = create_engine(engine_url, pool_pre_ping=True)
                    # 测试连接
                    with engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    self._engine = engine
                    self._dialect = "mysql"
                    logger.info(f"DatabaseTool using MySQL: {host}:{port}/{database}")
                    return
            except Exception as exc:
                logger.warning(f"MySQL connection failed, falling back to SQLite: {exc}")

        # 回退 SQLite
        db_path = cfg.data.db_path
        self._engine = create_engine(f"sqlite:///{db_path}")
        self._dialect = "sqlite"
        logger.info(f"DatabaseTool using SQLite: {db_path}")

    def execute(
        self,
        action: str,
        sql: Optional[str] = None,
        table: Optional[str] = None,
        data: Optional[List[Dict[str, Any]]] = None,
        if_exists: str = "append",
    ) -> ToolResult:
        if self._engine is None:
            return ToolResult(success=False, error="Database engine not initialized")

        try:
            if action == "query":
                if not sql:
                    return ToolResult(success=False, error="'sql' is required for query")
                with self._engine.connect() as conn:
                    df = pd.read_sql(text(sql), conn)
                return ToolResult(
                    success=True,
                    data=df.to_dict("records") if not df.empty else [],
                    metadata={"rows": len(df), "columns": list(df.columns)},
                )

            elif action == "execute":
                if not sql:
                    return ToolResult(success=False, error="'sql' is required for execute")
                with self._engine.begin() as conn:
                    result = conn.execute(text(sql))
                    rowcount = result.rowcount if result else 0
                return ToolResult(success=True, data={"rowcount": rowcount})

            elif action == "save_dataframe":
                if not table:
                    return ToolResult(success=False, error="'table' is required for save_dataframe")
                if data is None:
                    return ToolResult(success=False, error="'data' is required for save_dataframe")
                df = pd.DataFrame(data)
                df.to_sql(table, self._engine, if_exists=if_exists, index=False)
                return ToolResult(success=True, data={"saved_rows": len(df), "table": table})

            elif action == "list_tables":
                if self._dialect == "sqlite":
                    query = "SELECT name FROM sqlite_master WHERE type='table'"
                else:
                    query = "SHOW TABLES"
                with self._engine.connect() as conn:
                    df = pd.read_sql(text(query), conn)
                return ToolResult(
                    success=True,
                    data=df.iloc[:, 0].tolist() if not df.empty else [],
                )

            else:
                return ToolResult(success=False, error=f"Unknown action: {action}")
        except SQLAlchemyError as exc:
            logger.error(f"DatabaseTool SQL error: {exc}")
            return ToolResult(success=False, error=f"SQL error: {exc}")
        except Exception as exc:
            logger.error(f"DatabaseTool error: {exc}")
            return ToolResult(success=False, error=str(exc))
