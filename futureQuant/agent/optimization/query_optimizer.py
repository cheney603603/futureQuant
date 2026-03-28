"""
æ¥è¯¢ä¼åå¨æ¨¡å

æä¾æ°æ®åºæ¥è¯¢ä¼åè½åï¼
- æ°æ®åºç´¢å¼ä¼å
- æ¥è¯¢è®¡ååæ
- æ¥è¯¢ç»æç¼å­
- æ¹éæ¥è¯¢ä¼å
"""

import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QueryStats:
    """æ¥è¯¢ç»è®¡ä¿¡æ¯"""
    query_count: int = 0
    total_time_ms: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def avg_time_ms(self) -> float:
        """å¹³åæ¥è¯¢æ¶é´"""
        return self.total_time_ms / self.query_count if self.query_count > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """ç¼å­å½ä¸­ç"""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def __repr__(self) -> str:
        return (
            f"QueryStats(queries={self.query_count}, "
            f"avg_time={self.avg_time_ms:.2f}ms, "
            f"cache_hit_rate={self.cache_hit_rate * 100:.1f}%)"
        )


class QueryOptimizer:
    """
    æ¥è¯¢ä¼åå¨
    
    æä¾æ°æ®åºæ¥è¯¢ä¼åãç¼å­åæ¹éæ¥è¯¢åè½ã
    """
    
    def __init__(
        self,
        db_path: str = ":memory:",
        enable_cache: bool = True,
        cache_size: int = 1000,
    ):
        """
        åå§åæ¥è¯¢ä¼åå¨
        
        Args:
            db_path: æ°æ®åºè·¯å¾ï¼":memory:" è¡¨ç¤ºåå­æ°æ®åºï¼
            enable_cache: æ¯å¦å¯ç¨æ¥è¯¢ç¼å­
            cache_size: ç¼å­å¤§å°
        """
        self.db_path = db_path
        self.enable_cache = enable_cache
        self.cache_size = cache_size
        self.query_cache: Dict[str, pd.DataFrame] = {}
        self.stats = QueryStats()
        self.connection: Optional[sqlite3.Connection] = None
        
        self._init_connection()
        logger.info(
            f"QueryOptimizer initialized: db={db_path}, "
            f"cache={enable_cache}, cache_size={cache_size}"
        )
    
    def _init_connection(self) -> None:
        """åå§åæ°æ®åºè¿æ¥"""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.row_factory = sqlite3.Row
            logger.debug(f"Database connection established: {self.db_path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")
            raise
    
    def create_index(
        self,
        table_name: str,
        column_names: List[str],
        index_name: Optional[str] = None,
    ) -> None:
        """
        åå»ºæ°æ®åºç´¢å¼
        
        Args:
            table_name: è¡¨å
            column_names: åååè¡¨
            index_name: ç´¢å¼åï¼å¦æä¸º None åèªå¨çæï¼
        """
        if index_name is None:
            index_name = f"idx_{table_name}_{'_'.join(column_names)}"
        
        columns_str = ", ".join(column_names)
        sql = f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({columns_str})"
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            self.connection.commit()
            logger.info(f"Created index: {index_name} on {table_name}({columns_str})")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
    
    def analyze_query_plan(self, query: str) -> List[str]:
        """
        åææ¥è¯¢è®¡å
        
        Args:
            query: SQL æ¥è¯¢è¯­å¥
        
        Returns:
            æ¥è¯¢è®¡åä¿¡æ¯åè¡¨
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            plan = cursor.fetchall()
            return [str(row) for row in plan]
        except Exception as e:
            logger.error(f"Error analyzing query plan: {e}")
            return []
    
    def execute_query(
        self,
        query: str,
        params: Optional[Tuple] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        æ§è¡æ¥è¯¢
        
        Args:
            query: SQL æ¥è¯¢è¯­å¥
            params: æ¥è¯¢åæ°
            use_cache: æ¯å¦ä½¿ç¨ç¼å­
        
        Returns:
            æ¥è¯¢ç»æ DataFrame
        """
        # çæç¼å­é®
        cache_key = f"{query}|{params}" if params else query
        
        # æ£æ¥ç¼å­
        if use_cache and self.enable_cache and cache_key in self.query_cache:
            self.stats.cache_hits += 1
            logger.debug(f"Cache hit: {cache_key[:50]}...")
            return self.query_cache[cache_key].copy()
        
        self.stats.cache_misses += 1
        
        # æ§è¡æ¥è¯¢
        start_time = time.time()
        try:
            if params:
                df = pd.read_sql_query(query, self.connection, params=params)
            else:
                df = pd.read_sql_query(query, self.connection)
            
            elapsed_ms = (time.time() - start_time) * 1000
            self.stats.query_count += 1
            self.stats.total_time_ms += elapsed_ms
            
            logger.debug(f"Query executed in {elapsed_ms:.2f}ms: {query[:50]}...")
            
            # å­å¥ç¼å­
            if use_cache and self.enable_cache:
                if len(self.query_cache) >= self.cache_size:
                    # å é¤ææ§çç¼å­é¡¹
                    oldest_key = next(iter(self.query_cache))
                    del self.query_cache[oldest_key]
                
                self.query_cache[cache_key] = df.copy()
            
            return df
            
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def batch_query(
        self,
        queries: List[str],
        use_cache: bool = True,
    ) -> List[pd.DataFrame]:
        """
        æ¹éæ§è¡æ¥è¯¢
        
        Args:
            queries: SQL æ¥è¯¢è¯­å¥åè¡¨
            use_cache: æ¯å¦ä½¿ç¨ç¼å­
        
        Returns:
            æ¥è¯¢ç»æ DataFrame åè¡¨
        """
        results = []
        
        for query in queries:
            try:
                df = self.execute_query(query, use_cache=use_cache)
                results.append(df)
            except Exception as e:
                logger.error(f"Error in batch query: {e}")
                results.append(pd.DataFrame())
        
        return results
    
    def load_dataframe(
        self,
        df: pd.DataFrame,
        table_name: str,
        if_exists: str = "replace",
    ) -> None:
        """
        å° DataFrame å è½½å°æ°æ®åº
        
        Args:
            df: è¦å è½½ç DataFrame
            table_name: è¡¨å
            if_exists: è¡¨å­å¨æ¶çå¤çæ¹å¼ ("replace", "append", "fail")
        """
        try:
            df.to_sql(table_name, self.connection, if_exists=if_exists, index=False)
            logger.info(f"Loaded DataFrame to table: {table_name} ({df.shape[0]} rows)")
        except Exception as e:
            logger.error(f"Error loading DataFrame: {e}")
            raise
    
    def clear_cache(self) -> None:
        """æ¸ç©ºæ¥è¯¢ç¼å­"""
        self.query_cache.clear()
        logger.info("Query cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """è·åæ¥è¯¢ç»è®¡ä¿¡æ¯"""
        return {
            "query_count": self.stats.query_count,
            "total_time_ms": self.stats.total_time_ms,
            "avg_time_ms": self.stats.avg_time_ms,
            "cache_hits": self.stats.cache_hits,
            "cache_misses": self.stats.cache_misses,
            "cache_hit_rate": self.stats.cache_hit_rate * 100,
            "cache_size": len(self.query_cache),
        }
    
    def close(self) -> None:
        """å³é­æ°æ®åºè¿æ¥"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
    
    def __repr__(self) -> str:
        return f"QueryOptimizer(db={self.db_path}, {self.stats})"
    
    def __enter__(self):
        """ä¸ä¸æç®¡çå¨å¥å£"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """ä¸ä¸æç®¡çå¨åºå£"""
        self.close()


class BulkQueryExecutor:
    """
    æ¹éæ¥è¯¢æ§è¡å¨
    
    ä¼åå¤§éæ¥è¯¢çæ§è¡æçã
    """
    
    def __init__(self, optimizer: QueryOptimizer, batch_size: int = 100):
        """
        åå§åæ¹éæ¥è¯¢æ§è¡å¨
        
        Args:
            optimizer: æ¥è¯¢ä¼åå¨å®ä¾
            batch_size: æ¹éå¤§å°
        """
        self.optimizer = optimizer
        self.batch_size = batch_size
    
    def execute_bulk_queries(
        self,
        queries: List[str],
        use_cache: bool = True,
    ) -> List[pd.DataFrame]:
        """
        æ§è¡å¤§éæ¥è¯¢
        
        Args:
            queries: æ¥è¯¢è¯­å¥åè¡¨
            use_cache: æ¯å¦ä½¿ç¨ç¼å­
        
        Returns:
            æ¥è¯¢ç»æåè¡¨
        """
        all_results = []
        
        for i in range(0, len(queries), self.batch_size):
            batch = queries[i:i + self.batch_size]
            logger.info(f"Processing batch {i // self.batch_size + 1}: {len(batch)} queries")
            
            batch_results = self.optimizer.batch_query(batch, use_cache=use_cache)
            all_results.extend(batch_results)
        
        return all_results
    
    def execute_parameterized_queries(
        self,
        query_template: str,
        params_list: List[Tuple],
        use_cache: bool = True,
    ) -> List[pd.DataFrame]:
        """
        æ§è¡åæ°åæ¥è¯¢
        
        Args:
            query_template: æ¥è¯¢æ¨¡æ¿
            params_list: åæ°åè¡¨
            use_cache: æ¯å¦ä½¿ç¨ç¼å­
        
        Returns:
            æ¥è¯¢ç»æåè¡¨
        """
        results = []
        
        for params in params_list:
            try:
                df = self.optimizer.execute_query(
                    query_template,
                    params=params,
                    use_cache=use_cache
                )
                results.append(df)
            except Exception as e:
                logger.error(f"Error executing parameterized query: {e}")
                results.append(pd.DataFrame())
        
        return results
