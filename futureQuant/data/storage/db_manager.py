"""
数据库管理器 - SQLite + Parquet 双存储方案

- SQLite: 存储元数据、配置、小表
- Parquet: 存储大表（价格数据、因子数据），列式存储高效
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import numpy as np

from ...core.config import get_config
from ...core.logger import get_logger
from ...core.exceptions import StorageError

logger = get_logger('data.storage')


class DBManager:
    """数据库管理器"""
    
    def __init__(self, db_path: Optional[str] = None, cache_dir: Optional[str] = None):
        """
        初始化数据库管理器
        
        Args:
            db_path: SQLite数据库路径
            cache_dir: 缓存目录（存放Parquet文件）
        """
        config = get_config()
        
        self.db_path = db_path or config.data.db_path
        self.cache_dir = Path(cache_dir or config.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self._init_db()
    
    def _init_db(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 品种信息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS varieties (
                    code TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    category TEXT,
                    contract_multiplier INTEGER,
                    price_tick REAL,
                    margin_rate REAL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 合约信息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS contracts (
                    symbol TEXT PRIMARY KEY,
                    variety TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    list_date TEXT,
                    expire_date TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (variety) REFERENCES varieties(code)
                )
            ''')
            
            # 数据更新记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS update_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_type TEXT NOT NULL,
                    symbol TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    records_count INTEGER,
                    status TEXT,
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 因子元数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS factor_meta (
                    name TEXT PRIMARY KEY,
                    category TEXT,
                    description TEXT,
                    params TEXT,
                    dependencies TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    # ==================== 价格数据操作 ====================
    
    def save_price_data(self, df: pd.DataFrame, symbol: str, data_type: str = 'daily'):
        """
        保存价格数据到Parquet
        
        Args:
            df: 价格数据
            symbol: 合约代码
            data_type: 数据类型，如 'daily', 'minute'
        """
        try:
            file_path = self.cache_dir / f"{symbol}_{data_type}.parquet"
            
            # 如果文件已存在，合并数据
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=['date'], keep='last')
                df = df.sort_values('date')
            
            df.to_parquet(file_path, index=False, compression='zstd')
            logger.info(f"Saved {len(df)} records to {file_path}")
            
        except Exception as e:
            raise StorageError(f"Failed to save price data: {e}")
    
    def load_price_data(
        self, 
        symbol: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        data_type: str = 'daily'
    ) -> pd.DataFrame:
        """
        加载价格数据
        
        Args:
            symbol: 合约代码
            start_date: 开始日期
            end_date: 结束日期
            data_type: 数据类型
            
        Returns:
            DataFrame with price data
        """
        try:
            file_path = self.cache_dir / f"{symbol}_{data_type}.parquet"
            
            if not file_path.exists():
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path)
            
            # 日期过滤
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load price data: {e}")
            return pd.DataFrame()
    
    # ==================== 因子数据操作 ====================
    
    def save_factor_data(self, df: pd.DataFrame, factor_name: str):
        """
        保存因子数据
        
        Args:
            df: 因子数据，格式为 [date, symbol, factor_value]
            factor_name: 因子名称
        """
        try:
            file_path = self.cache_dir / f"factor_{factor_name}.parquet"
            
            # 合并已有数据
            if file_path.exists():
                existing_df = pd.read_parquet(file_path)
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.drop_duplicates(subset=['date', 'symbol'], keep='last')
                df = df.sort_values(['date', 'symbol'])
            
            df.to_parquet(file_path, index=False, compression='zstd')
            logger.info(f"Saved factor {factor_name} with {len(df)} records")
            
        except Exception as e:
            raise StorageError(f"Failed to save factor data: {e}")
    
    def load_factor_data(
        self, 
        factor_name: str,
        symbols: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        加载因子数据
        
        Args:
            factor_name: 因子名称
            symbols: 品种列表，为None时返回所有
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            DataFrame with factor data
        """
        try:
            file_path = self.cache_dir / f"factor_{factor_name}.parquet"
            
            if not file_path.exists():
                return pd.DataFrame()
            
            df = pd.read_parquet(file_path)
            
            # 过滤
            if symbols:
                df = df[df['symbol'].isin(symbols)]
            if start_date:
                df = df[df['date'] >= start_date]
            if end_date:
                df = df[df['date'] <= end_date]
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load factor data: {e}")
            return pd.DataFrame()
    
    # ==================== SQLite 元数据操作 ====================
    
    def save_variety_info(self, info: Dict[str, Any]):
        """保存品种信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO varieties 
                (code, name, exchange, category, contract_multiplier, price_tick, margin_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                info['code'],
                info['name'],
                info['exchange'],
                info.get('category'),
                info.get('contract_multiplier'),
                info.get('price_tick'),
                info.get('margin_rate')
            ))
            conn.commit()
    
    def get_variety_info(self, code: str) -> Optional[Dict]:
        """获取品种信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM varieties WHERE code = ?', (code,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def save_contract_info(self, info: Dict[str, Any]):
        """保存合约信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO contracts 
                (symbol, variety, exchange, list_date, expire_date)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                info['symbol'],
                info['variety'],
                info['exchange'],
                info.get('list_date'),
                info.get('expire_date')
            ))
            conn.commit()
    
    def get_contract_info(self, symbol: str) -> Optional[Dict]:
        """获取合约信息"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM contracts WHERE symbol = ?', (symbol,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def log_update(
        self, 
        data_type: str, 
        symbol: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        records_count: int = 0,
        status: str = 'success',
        message: Optional[str] = None
    ):
        """记录数据更新日志"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO update_log 
                (data_type, symbol, start_date, end_date, records_count, status, message)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (data_type, symbol, start_date, end_date, records_count, status, message))
            conn.commit()
    
    def get_update_history(
        self, 
        data_type: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """获取更新历史"""
        query = 'SELECT * FROM update_log WHERE 1=1'
        params = []
        
        if data_type:
            query += ' AND data_type = ?'
            params.append(data_type)
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    def save_factor_meta(self, meta: Dict[str, Any]):
        """保存因子元数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO factor_meta 
                (name, category, description, params, dependencies)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                meta['name'],
                meta.get('category'),
                meta.get('description'),
                json.dumps(meta.get('params', {})),
                json.dumps(meta.get('dependencies', []))
            ))
            conn.commit()
    
    def get_factor_meta(self, name: str) -> Optional[Dict]:
        """获取因子元数据"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM factor_meta WHERE name = ?', (name,))
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['params'] = json.loads(result['params'])
                result['dependencies'] = json.loads(result['dependencies'])
                return result
            return None
    
    def list_factors(self, category: Optional[str] = None) -> pd.DataFrame:
        """列出所有因子"""
        query = 'SELECT * FROM factor_meta'
        params = []
        
        if category:
            query += ' WHERE category = ?'
            params.append(category)
        
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(query, conn, params=params)
    
    # ==================== 工具方法 ====================
    
    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据存储摘要"""
        summary = {
            'db_path': self.db_path,
            'cache_dir': str(self.cache_dir),
            'parquet_files': [],
            'db_tables': []
        }
        
        # Parquet文件
        for f in self.cache_dir.glob('*.parquet'):
            summary['parquet_files'].append({
                'name': f.name,
                'size_mb': round(f.stat().st_size / (1024 * 1024), 2)
            })
        
        # SQLite表
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            
            for (table_name,) in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                count = cursor.fetchone()[0]
                summary['db_tables'].append({
                    'name': table_name,
                    'records': count
                })
        
        return summary
