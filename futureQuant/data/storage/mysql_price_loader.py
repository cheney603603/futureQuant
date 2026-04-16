"""
MySQL price loader for DataManager.

This is an optional read path used when local MySQL credentials are provided
via environment variables. It allows the project to prefer a user-owned MySQL
price table before falling back to network fetchers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from ...core.logger import get_logger

logger = get_logger("data.mysql_loader")


@dataclass
class MySQLPriceLoaderConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    table: str = "daily_price"
    date_column: str = "date"
    symbol_column: str = "symbol"
    open_column: str = "open"
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"
    volume_column: str = "volume"
    open_interest_column: str = "open_interest"

    @classmethod
    def from_mapping(cls, data: Optional[Dict[str, Any]]) -> Optional["MySQLPriceLoaderConfig"]:
        if not data or not data.get("enabled"):
            return None

        required_keys = ["host", "user", "password", "database"]
        if not all(data.get(key) for key in required_keys):
            return None

        return cls(
            host=str(data["host"]),
            port=int(data.get("port", 3306)),
            user=str(data["user"]),
            password=str(data["password"]),
            database=str(data["database"]),
            table=str(data.get("table", "daily_price")),
            date_column=str(data.get("date_column", "date")),
            symbol_column=str(data.get("symbol_column", "symbol")),
            open_column=str(data.get("open_column", "open")),
            high_column=str(data.get("high_column", "high")),
            low_column=str(data.get("low_column", "low")),
            close_column=str(data.get("close_column", "close")),
            volume_column=str(data.get("volume_column", "volume")),
            open_interest_column=str(
                data.get("open_interest_column", "open_interest")
            ),
        )

    @classmethod
    def from_env(cls) -> Optional["MySQLPriceLoaderConfig"]:
        host = os.getenv("FUTUREQUANT_MYSQL_HOST")
        user = os.getenv("FUTUREQUANT_MYSQL_USER")
        password = os.getenv("FUTUREQUANT_MYSQL_PASSWORD")
        database = os.getenv("FUTUREQUANT_MYSQL_DATABASE")

        if not all([host, user, password, database]):
            return None

        return cls(
            host=host,
            port=int(os.getenv("FUTUREQUANT_MYSQL_PORT", "3306")),
            user=user,
            password=password,
            database=database,
            table=os.getenv("FUTUREQUANT_MYSQL_TABLE", "daily_price"),
            date_column=os.getenv("FUTUREQUANT_MYSQL_DATE_COLUMN", "date"),
            symbol_column=os.getenv("FUTUREQUANT_MYSQL_SYMBOL_COLUMN", "symbol"),
            open_column=os.getenv("FUTUREQUANT_MYSQL_OPEN_COLUMN", "open"),
            high_column=os.getenv("FUTUREQUANT_MYSQL_HIGH_COLUMN", "high"),
            low_column=os.getenv("FUTUREQUANT_MYSQL_LOW_COLUMN", "low"),
            close_column=os.getenv("FUTUREQUANT_MYSQL_CLOSE_COLUMN", "close"),
            volume_column=os.getenv("FUTUREQUANT_MYSQL_VOLUME_COLUMN", "volume"),
            open_interest_column=os.getenv(
                "FUTUREQUANT_MYSQL_OPEN_INTEREST_COLUMN",
                "open_interest",
            ),
        )


class MySQLPriceLoader:
    """Read OHLCV-style daily futures data from MySQL."""

    def __init__(self, config: Optional[MySQLPriceLoaderConfig] = None):
        self.config = config or MySQLPriceLoaderConfig.from_env()
        if self.config is None:
            raise ValueError("MySQL loader config is not available")

        self._engine = create_engine(
            (
                "mysql+pymysql://"
                f"{self.config.user}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}"
                "?charset=utf8mb4"
            ),
            pool_pre_ping=True,
        )

    @property
    def is_enabled(self) -> bool:
        return self.config is not None

    def load_daily_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        cfg = self.config
        assert cfg is not None

        query = f"""
        SELECT
            {cfg.date_column} AS date,
            {cfg.symbol_column} AS symbol,
            {cfg.open_column} AS open,
            {cfg.high_column} AS high,
            {cfg.low_column} AS low,
            {cfg.close_column} AS close,
            {cfg.volume_column} AS volume,
            {cfg.open_interest_column} AS open_interest
        FROM {cfg.table}
        WHERE {cfg.symbol_column} = :symbol
        """
        params = {"symbol": symbol}

        if start_date:
            query += f" AND {cfg.date_column} >= :start_date"
            params["start_date"] = start_date
        if end_date:
            query += f" AND {cfg.date_column} <= :end_date"
            params["end_date"] = end_date

        query += f" ORDER BY {cfg.date_column}"

        try:
            with self._engine.connect() as conn:
                df = pd.read_sql(text(query), conn, params=params)
        except SQLAlchemyError as e:
            logger.warning(f"MySQL daily data load failed for {symbol}: {e}")
            return pd.DataFrame()

        if df.empty:
            return df

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        return df
