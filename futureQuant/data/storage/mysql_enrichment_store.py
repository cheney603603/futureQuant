"""
MySQL store for fundamental and macro enrichment data.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from ...core.logger import get_logger
from .mysql_price_loader import MySQLPriceLoaderConfig

logger = get_logger("data.mysql_enrichment")


class MySQLEnrichmentStore:
    """Read/write external fundamental and macro features in MySQL."""

    def __init__(self, config: MySQLPriceLoaderConfig):
        self.config = config
        self._engine = create_engine(
            (
                "mysql+pymysql://"
                f"{config.user}:{config.password}"
                f"@{config.host}:{config.port}/{config.database}"
                "?charset=utf8mb4"
            ),
            pool_pre_ping=True,
        )
        self.ensure_tables()

    def ensure_tables(self) -> None:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS fundamental_daily (
              variety VARCHAR(32) NOT NULL,
              `date` DATE NOT NULL,
              spot_price DOUBLE NULL,
              futures_price DOUBLE NULL,
              basis DOUBLE NULL,
              basis_rate DOUBLE NULL,
              inventory DOUBLE NULL,
              inventory_type VARCHAR(64) NULL,
              warehouse_receipt DOUBLE NULL,
              warehouse_change DOUBLE NULL,
              near_price DOUBLE NULL,
              far_price DOUBLE NULL,
              source VARCHAR(64) NULL,
              llm_provider VARCHAR(64) NULL,
              llm_model VARCHAR(128) NULL,
              updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (variety, `date`)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS macro_daily (
              `date` DATE NOT NULL,
              dxy DOUBLE NULL,
              interest_rate DOUBLE NULL,
              commodity_index DOUBLE NULL,
              inflation_expectation DOUBLE NULL,
              source VARCHAR(64) NULL,
              llm_provider VARCHAR(64) NULL,
              llm_model VARCHAR(128) NULL,
              updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
              PRIMARY KEY (`date`)
            )
            """,
        ]

        with self._engine.begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))

    def save_fundamental_data(
        self,
        df: pd.DataFrame,
        variety: str,
        source: str = "crawler",
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> int:
        if df is None or df.empty:
            return 0

        required_cols = ["date"]
        if not all(col in df.columns for col in required_cols):
            return 0

        working = df.copy()
        working["date"] = pd.to_datetime(working["date"]).dt.date
        working["variety"] = variety.upper()
        working["source"] = source
        working["llm_provider"] = llm_provider
        working["llm_model"] = llm_model

        cols = [
            "variety",
            "date",
            "spot_price",
            "futures_price",
            "basis",
            "basis_rate",
            "inventory",
            "inventory_type",
            "warehouse_receipt",
            "warehouse_change",
            "near_price",
            "far_price",
            "source",
            "llm_provider",
            "llm_model",
        ]
        for col in cols:
            if col not in working.columns:
                working[col] = None

        records = working[cols].drop_duplicates(subset=["variety", "date"]).to_dict("records")
        if not records:
            return 0

        sql = text(
            """
            INSERT INTO fundamental_daily
            (variety, `date`, spot_price, futures_price, basis, basis_rate, inventory,
             inventory_type, warehouse_receipt, warehouse_change, near_price, far_price,
             source, llm_provider, llm_model)
            VALUES
            (:variety, :date, :spot_price, :futures_price, :basis, :basis_rate, :inventory,
             :inventory_type, :warehouse_receipt, :warehouse_change, :near_price, :far_price,
             :source, :llm_provider, :llm_model)
            ON DUPLICATE KEY UPDATE
              spot_price = VALUES(spot_price),
              futures_price = VALUES(futures_price),
              basis = VALUES(basis),
              basis_rate = VALUES(basis_rate),
              inventory = VALUES(inventory),
              inventory_type = VALUES(inventory_type),
              warehouse_receipt = VALUES(warehouse_receipt),
              warehouse_change = VALUES(warehouse_change),
              near_price = VALUES(near_price),
              far_price = VALUES(far_price),
              source = VALUES(source),
              llm_provider = VALUES(llm_provider),
              llm_model = VALUES(llm_model)
            """
        )
        try:
            with self._engine.begin() as conn:
                conn.execute(sql, records)
            return len(records)
        except SQLAlchemyError as e:
            logger.warning(f"Failed to save fundamental data for {variety}: {e}")
            return 0

    def save_macro_data(
        self,
        df: pd.DataFrame,
        source: str = "crawler",
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
    ) -> int:
        if df is None or df.empty or "date" not in df.columns:
            return 0

        working = df.copy()
        working["date"] = pd.to_datetime(working["date"]).dt.date
        working["source"] = source
        working["llm_provider"] = llm_provider
        working["llm_model"] = llm_model

        cols = [
            "date",
            "dxy",
            "interest_rate",
            "commodity_index",
            "inflation_expectation",
            "source",
            "llm_provider",
            "llm_model",
        ]
        for col in cols:
            if col not in working.columns:
                working[col] = None

        records = working[cols].drop_duplicates(subset=["date"]).to_dict("records")
        if not records:
            return 0

        sql = text(
            """
            INSERT INTO macro_daily
            (`date`, dxy, interest_rate, commodity_index, inflation_expectation,
             source, llm_provider, llm_model)
            VALUES
            (:date, :dxy, :interest_rate, :commodity_index, :inflation_expectation,
             :source, :llm_provider, :llm_model)
            ON DUPLICATE KEY UPDATE
              dxy = VALUES(dxy),
              interest_rate = VALUES(interest_rate),
              commodity_index = VALUES(commodity_index),
              inflation_expectation = VALUES(inflation_expectation),
              source = VALUES(source),
              llm_provider = VALUES(llm_provider),
              llm_model = VALUES(llm_model)
            """
        )
        try:
            with self._engine.begin() as conn:
                conn.execute(sql, records)
            return len(records)
        except SQLAlchemyError as e:
            logger.warning(f"Failed to save macro data: {e}")
            return 0

    def load_fundamental_data(
        self,
        variety: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        query = """
        SELECT `date`, variety, spot_price, futures_price, basis, basis_rate,
               inventory, inventory_type, warehouse_receipt, warehouse_change,
               near_price, far_price
        FROM fundamental_daily
        WHERE variety = :variety
        """
        params = {"variety": variety.upper()}
        if start_date:
            query += " AND `date` >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND `date` <= :end_date"
            params["end_date"] = end_date
        query += " ORDER BY `date`"

        try:
            with self._engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
        except SQLAlchemyError as e:
            logger.warning(f"Failed to load fundamental data for {variety}: {e}")
            return pd.DataFrame()

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df

    def load_macro_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        query = """
        SELECT `date`, dxy, interest_rate, commodity_index, inflation_expectation
        FROM macro_daily
        WHERE 1=1
        """
        params = {}
        if start_date:
            query += " AND `date` >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += " AND `date` <= :end_date"
            params["end_date"] = end_date
        query += " ORDER BY `date`"

        try:
            with self._engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
        except SQLAlchemyError as e:
            logger.warning(f"Failed to load macro data: {e}")
            return pd.DataFrame()

        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
        return df
