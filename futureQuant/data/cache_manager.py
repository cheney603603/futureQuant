"""
数据缓存管理器 - 带过期机制的持久化缓存

P2.1 实现：
- 支持 TTL (Time To Live) 过期机制
- SQLite 元数据 + Parquet 数据存储
- 自动清理过期缓存
- 缓存统计和监控

Author: futureQuant Team
Date: 2026-04-19
"""

import json
import hashlib
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from futureQuant.core.logger import get_logger
from futureQuant.core.exceptions import StorageError

logger = get_logger('data.cache')


class DataCacheManager:
    """
    数据缓存管理器
    
    特性：
    1. TTL 过期机制 - 可配置缓存有效期
    2. 持久化存储 - SQLite 元数据 + Parquet 数据
    3. 自动清理 - 定期清理过期缓存
    4. 统计监控 - 缓存命中率、大小等
    """
    
    # 默认 TTL 配置（小时）
    DEFAULT_TTL = {
        'price': 24,          # 价格数据：1天
        'fundamental': 168,   # 基本面数据：7天
        'inventory': 72,      # 库存数据：3天
        'basis': 24,          # 基差数据：1天
        'warehouse': 168,     # 仓单数据：7天
        'factor': 24,         # 因子数据：1天
    }
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        ttl_config: Optional[Dict[str, int]] = None,
        auto_cleanup: bool = True
    ):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            ttl_config: TTL 配置 {数据类型: 小时数}
            auto_cleanup: 是否自动清理过期缓存
        """
        from futureQuant.core.config import get_config
        
        config = get_config()
        self.cache_dir = Path(cache_dir or config.data.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.ttl_config = {**self.DEFAULT_TTL, **(ttl_config or {})}
        self.auto_cleanup = auto_cleanup
        
        # 数据库路径
        self.db_path = self.cache_dir / 'cache_metadata.db'
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'writes': 0,
            'deletes': 0,
        }
        
        # 初始化数据库
        self._init_db()
        
        # 启动时清理过期缓存
        if auto_cleanup:
            self.cleanup_expired()
    
    def _init_db(self):
        """初始化缓存元数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 缓存元数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_metadata (
                    cache_key TEXT PRIMARY KEY,
                    data_type TEXT NOT NULL,
                    variety TEXT,
                    start_date TEXT,
                    end_date TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    row_count INTEGER,
                    hit_count INTEGER DEFAULT 0,
                    last_hit_at TIMESTAMP
                )
            ''')
            
            # 缓存统计表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS cache_stats (
                    date TEXT PRIMARY KEY,
                    hits INTEGER DEFAULT 0,
                    misses INTEGER DEFAULT 0,
                    writes INTEGER DEFAULT 0,
                    deletes INTEGER DEFAULT 0
                )
            ''')
            
            conn.commit()
    
    def _generate_key(
        self,
        data_type: str,
        variety: str,
        start_date: str,
        end_date: str,
        extra_params: Optional[Dict] = None
    ) -> str:
        """生成缓存键"""
        key_str = f"{data_type}:{variety}:{start_date}:{end_date}"
        if extra_params:
            key_str += ":" + json.dumps(extra_params, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def get(
        self,
        data_type: str,
        variety: str,
        start_date: str,
        end_date: str,
        extra_params: Optional[Dict] = None,
        check_freshness: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        获取缓存数据
        
        Args:
            data_type: 数据类型 (price, inventory, basis, etc.)
            variety: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            extra_params: 额外参数
            check_freshness: 是否检查数据新鲜度
            
        Returns:
            DataFrame 或 None
        """
        cache_key = self._generate_key(data_type, variety, start_date, end_date, extra_params)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 查询缓存
            cursor.execute('''
                SELECT file_path, expires_at, row_count
                FROM cache_metadata
                WHERE cache_key = ?
            ''', (cache_key,))
            
            row = cursor.fetchone()
            
            if not row:
                self.stats['misses'] += 1
                logger.debug(f"Cache miss: {cache_key}")
                return None
            
            file_path, expires_at, row_count = row
            
            # 检查是否过期
            if datetime.now() > datetime.fromisoformat(expires_at):
                self.stats['misses'] += 1
                logger.debug(f"Cache expired: {cache_key}")
                self._delete_cache_entry(cache_key, file_path)
                return None
            
            # 检查文件是否存在
            parquet_path = Path(file_path)
            if not parquet_path.exists():
                self.stats['misses'] += 1
                logger.warning(f"Cache file missing: {file_path}")
                self._delete_cache_entry(cache_key, file_path)
                return None
            
            try:
                # 读取数据
                df = pd.read_parquet(parquet_path)
                
                # 更新命中统计
                cursor.execute('''
                    UPDATE cache_metadata
                    SET hit_count = hit_count + 1,
                        last_hit_at = CURRENT_TIMESTAMP
                    WHERE cache_key = ?
                ''', (cache_key,))
                conn.commit()
                
                self.stats['hits'] += 1
                logger.debug(f"Cache hit: {cache_key} ({len(df)} rows)")
                
                return df
                
            except Exception as e:
                logger.error(f"Failed to read cache file: {e}")
                self._delete_cache_entry(cache_key, file_path)
                return None
    
    def put(
        self,
        data: pd.DataFrame,
        data_type: str,
        variety: str,
        start_date: str,
        end_date: str,
        extra_params: Optional[Dict] = None,
        ttl_hours: Optional[int] = None
    ) -> bool:
        """
        存储缓存数据
        
        Args:
            data: 要缓存的数据
            data_type: 数据类型
            variety: 品种代码
            start_date: 开始日期
            end_date: 结束日期
            extra_params: 额外参数
            ttl_hours: 自定义 TTL（小时），None 使用默认值
            
        Returns:
            是否成功
        """
        if data.empty:
            logger.warning("Cannot cache empty DataFrame")
            return False
        
        cache_key = self._generate_key(data_type, variety, start_date, end_date, extra_params)
        
        # 确定 TTL
        if ttl_hours is None:
            ttl_hours = self.ttl_config.get(data_type, 24)
        
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        # 文件路径
        file_name = f"{cache_key}.parquet"
        file_path = self.cache_dir / file_name
        
        try:
            # 保存数据
            data.to_parquet(file_path, index=False, compression='snappy')
            
            # 记录元数据
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO cache_metadata
                    (cache_key, data_type, variety, start_date, end_date,
                     expires_at, file_path, file_size, row_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    cache_key,
                    data_type,
                    variety,
                    start_date,
                    end_date,
                    expires_at.isoformat(),
                    str(file_path),
                    file_path.stat().st_size,
                    len(data)
                ))
                
                conn.commit()
            
            self.stats['writes'] += 1
            logger.debug(f"Cache written: {cache_key} ({len(data)} rows, TTL={ttl_hours}h)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to write cache: {e}")
            return False
    
    def _delete_cache_entry(self, cache_key: str, file_path: str):
        """删除缓存条目"""
        try:
            # 删除文件
            Path(file_path).unlink(missing_ok=True)
            
            # 删除元数据
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM cache_metadata WHERE cache_key = ?', (cache_key,))
                conn.commit()
            
            self.stats['deletes'] += 1
            logger.debug(f"Cache deleted: {cache_key}")
            
        except Exception as e:
            logger.error(f"Failed to delete cache entry: {e}")
    
    def cleanup_expired(self) -> int:
        """
        清理过期缓存
        
        Returns:
            清理的条目数
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 查询过期条目
            cursor.execute('''
                SELECT cache_key, file_path
                FROM cache_metadata
                WHERE expires_at < ?
            ''', (datetime.now().isoformat(),))
            
            expired = cursor.fetchall()
            
            count = 0
            for cache_key, file_path in expired:
                self._delete_cache_entry(cache_key, file_path)
                count += 1
            
            if count > 0:
                logger.info(f"Cleaned up {count} expired cache entries")
            
            return count
    
    def clear_all(self) -> int:
        """
        清空所有缓存
        
        Returns:
            删除的条目数
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT cache_key, file_path FROM cache_metadata')
            all_entries = cursor.fetchall()
            
            for cache_key, file_path in all_entries:
                self._delete_cache_entry(cache_key, file_path)
            
            logger.info(f"Cleared all {len(all_entries)} cache entries")
            return len(all_entries)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 总条目数
            cursor.execute('SELECT COUNT(*) FROM cache_metadata')
            total_entries = cursor.fetchone()[0]
            
            # 总大小
            cursor.execute('SELECT SUM(file_size) FROM cache_metadata')
            total_size = cursor.fetchone()[0] or 0
            
            # 各类型统计
            cursor.execute('''
                SELECT data_type, COUNT(*), SUM(file_size), SUM(hit_count)
                FROM cache_metadata
                GROUP BY data_type
            ''')
            type_stats = {
                row[0]: {
                    'count': row[1],
                    'size_mb': round(row[2] / (1024 * 1024), 2) if row[2] else 0,
                    'hits': row[3],
                }
                for row in cursor.fetchall()
            }
            
            # 计算命中率
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                'session_stats': self.stats,
                'hit_rate': round(hit_rate, 4),
                'total_entries': total_entries,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'by_type': type_stats,
            }
    
    def get_cache_info(
        self,
        data_type: Optional[str] = None,
        variety: Optional[str] = None
    ) -> List[Dict]:
        """获取缓存详细信息"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            query = '''
                SELECT cache_key, data_type, variety, start_date, end_date,
                       created_at, expires_at, file_size, row_count, hit_count
                FROM cache_metadata
                WHERE 1=1
            '''
            params = []
            
            if data_type:
                query += ' AND data_type = ?'
                params.append(data_type)
            
            if variety:
                query += ' AND variety = ?'
                params.append(variety)
            
            query += ' ORDER BY created_at DESC'
            
            cursor.execute(query, params)
            
            columns = ['cache_key', 'data_type', 'variety', 'start_date', 'end_date',
                      'created_at', 'expires_at', 'file_size', 'row_count', 'hit_count']
            
            return [dict(zip(columns, row)) for row in cursor.fetchall()]


# 全局缓存管理器实例
_cache_manager: Optional[DataCacheManager] = None


def get_cache_manager() -> DataCacheManager:
    """获取全局缓存管理器实例"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = DataCacheManager()
    return _cache_manager


def clear_cache():
    """清空所有缓存"""
    manager = get_cache_manager()
    manager.clear_all()


def get_cache_stats() -> Dict[str, Any]:
    """获取缓存统计"""
    manager = get_cache_manager()
    return manager.get_stats()
