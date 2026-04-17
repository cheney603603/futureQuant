"""
Factor MySQL Store - 因子统一存储（MySQL 优先 + SQLite 兜底）

表结构 factor_library：
- factor_id (PK)
- name
- category
- variety
- frequency
- source (alpha101/technical/fundamental/llm)
- logic_description
- ic
- icir
- code_text
- created_at
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from ...core.config import get_config
from ...core.logger import get_logger

logger = get_logger("agent.factor_mining.factor_store")


class FactorMySQLStore:
    """因子存储器（MySQL/SQLite 自适应）"""

    def __init__(self):
        self._engine = None
        self._dialect = "sqlite"
        self._init_engine()
        self._ensure_table()

    def _init_engine(self):
        cfg = get_config()
        mysql_cfg = cfg.data.mysql

        if mysql_cfg and mysql_cfg.get("enabled"):
            try:
                user = mysql_cfg.get("user", "")
                password = mysql_cfg.get("password", "")
                host = mysql_cfg.get("host", "127.0.0.1")
                port = mysql_cfg.get("port", 3306)
                database = mysql_cfg.get("database", "futurequant")

                if user and password:
                    url = (
                        f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}"
                        "?charset=utf8mb4"
                    )
                    engine = create_engine(url, pool_pre_ping=True)
                    with engine.connect() as conn:
                        conn.execute(text("SELECT 1"))
                    self._engine = engine
                    self._dialect = "mysql"
                    logger.info(f"FactorStore using MySQL: {host}:{port}/{database}")
                    return
            except Exception as exc:
                logger.warning(f"MySQL failed, fallback to SQLite: {exc}")

        db_path = cfg.data.db_path
        self._engine = create_engine(f"sqlite:///{db_path}")
        self._dialect = "sqlite"
        logger.info(f"FactorStore using SQLite: {db_path}")

    def _ensure_table(self):
        """确保 factor_library 表存在"""
        # 使用通用 SQL，兼容 MySQL 和 SQLite
        stmt = """
        CREATE TABLE IF NOT EXISTS factor_library (
            factor_id VARCHAR(64) PRIMARY KEY,
            name VARCHAR(128) NOT NULL,
            category VARCHAR(64),
            variety VARCHAR(32),
            frequency VARCHAR(32),
            source VARCHAR(64),
            logic_description TEXT,
            ic REAL,
            icir REAL,
            code_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        try:
            with self._engine.begin() as conn:
                conn.execute(text(stmt))
        except SQLAlchemyError as exc:
            logger.error(f"Failed to ensure table: {exc}")

    def save_factor(self, metadata: Dict[str, Any]) -> bool:
        """保存因子元数据"""
        required = ["factor_id", "name"]
        for r in required:
            if r not in metadata:
                logger.error(f"Missing required field: {r}")
                return False

        sql = text(
            """
            INSERT INTO factor_library
            (factor_id, name, category, variety, frequency, source,
             logic_description, ic, icir, code_text, created_at)
            VALUES
            (:factor_id, :name, :category, :variety, :frequency, :source,
             :logic_description, :ic, :icir, :code_text, :created_at)
            ON DUPLICATE KEY UPDATE
                category=VALUES(category),
                variety=VALUES(variety),
                frequency=VALUES(frequency),
                source=VALUES(source),
                logic_description=VALUES(logic_description),
                ic=VALUES(ic),
                icir=VALUES(icir),
                code_text=VALUES(code_text)
            """
            if self._dialect == "mysql"
            else """
            INSERT INTO factor_library
            (factor_id, name, category, variety, frequency, source,
             logic_description, ic, icir, code_text, created_at)
            VALUES
            (:factor_id, :name, :category, :variety, :frequency, :source,
             :logic_description, :ic, :icir, :code_text, :created_at)
            ON CONFLICT(factor_id) DO UPDATE SET
                category=excluded.category,
                variety=excluded.variety,
                frequency=excluded.frequency,
                source=excluded.source,
                logic_description=excluded.logic_description,
                ic=excluded.ic,
                icir=excluded.icir,
                code_text=excluded.code_text
            """
        )

        params = {
            "factor_id": metadata["factor_id"],
            "name": metadata["name"],
            "category": metadata.get("category", ""),
            "variety": metadata.get("variety", ""),
            "frequency": metadata.get("frequency", "daily"),
            "source": metadata.get("source", "unknown"),
            "logic_description": metadata.get("logic_description", ""),
            "ic": metadata.get("ic"),
            "icir": metadata.get("icir"),
            "code_text": metadata.get("code_text", ""),
            "created_at": datetime.now(),
        }

        try:
            with self._engine.begin() as conn:
                conn.execute(sql, params)
            logger.info(f"Factor saved: {metadata['factor_id']}")
            return True
        except SQLAlchemyError as exc:
            logger.error(f"Failed to save factor: {exc}")
            return False

    def list_factors(self, variety: Optional[str] = None, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出因子"""
        query = "SELECT * FROM factor_library WHERE 1=1"
        params: Dict[str, Any] = {}
        if variety:
            query += " AND variety = :variety"
            params["variety"] = variety
        if source:
            query += " AND source = :source"
            params["source"] = source
        query += " ORDER BY created_at DESC"

        try:
            with self._engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
            return df.to_dict("records")
        except SQLAlchemyError as exc:
            logger.error(f"Failed to list factors: {exc}")
            return []

    def get_factor(self, factor_id: str) -> Optional[Dict[str, Any]]:
        """获取单个因子"""
        try:
            with self._engine.connect() as conn:
                df = pd.read_sql(
                    text("SELECT * FROM factor_library WHERE factor_id = :factor_id"),
                    conn,
                    params={"factor_id": factor_id},
                )
            if df.empty:
                return None
            return df.iloc[0].to_dict()
        except SQLAlchemyError as exc:
            logger.error(f"Failed to get factor: {exc}")
            return None
