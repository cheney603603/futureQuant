"""
åå­ç®¡çå¨æ¨¡å

æä¾åå­ä¼åè½åï¼
- æ°æ®ååå è½½
- åå­ä½¿ç¨çæ§
- èªå¨åå¾åæ¶
- åå­æ³æ¼æ£æµ
"""

import gc
import logging
import psutil
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """åå­ç»è®¡ä¿¡æ¯"""
    process_memory_mb: float = 0.0
    system_memory_mb: float = 0.0
    memory_percent: float = 0.0
    peak_memory_mb: float = 0.0
    
    def __repr__(self) -> str:
        return (
            f"MemoryStats(process={self.process_memory_mb:.1f}MB, "
            f"system={self.system_memory_mb:.1f}MB, "
            f"percent={self.memory_percent:.1f}%)"
        )


class MemoryMonitor:
    """
    åå­çæ§å¨
    
    çæ§è¿ç¨åç³»ç»åå­ä½¿ç¨æåµã
    """
    
    def __init__(self):
        """åå§ååå­çæ§å¨"""
        self.process = psutil.Process()
        self.peak_memory = 0.0  # 先声明，避免 _get_memory_stats() 内访问旧值
        self.initial_memory = self._get_memory_stats()
        self.peak_memory = self.initial_memory.process_memory_mb
        logger.info(f"MemoryMonitor initialized: {self.initial_memory}")
    
    def _get_memory_stats(self) -> MemoryStats:
        """è·åå½ååå­ç»è®¡"""
        process_info = self.process.memory_info()
        process_memory_mb = process_info.rss / 1024 / 1024
        
        virtual_memory = psutil.virtual_memory()
        system_memory_mb = virtual_memory.available / 1024 / 1024
        memory_percent = virtual_memory.percent
        
        return MemoryStats(
            process_memory_mb=process_memory_mb,
            system_memory_mb=system_memory_mb,
            memory_percent=memory_percent,
            peak_memory_mb=max(self.peak_memory, process_memory_mb),
        )
    
    def get_current_memory(self) -> MemoryStats:
        """è·åå½ååå­ä½¿ç¨æåµ"""
        stats = self._get_memory_stats()
        self.peak_memory = max(self.peak_memory, stats.process_memory_mb)
        return stats
    
    def get_memory_delta(self) -> float:
        """è·ååå­å¢é¿éï¼MBï¼"""
        current = self.get_current_memory()
        return current.process_memory_mb - self.initial_memory.process_memory_mb
    
    def __repr__(self) -> str:
        current = self.get_current_memory()
        delta = self.get_memory_delta()
        return f"MemoryMonitor({current}, delta={delta:.1f}MB)"


class MemoryManager:
    """
    åå­ç®¡çå¨
    
    æä¾æ°æ®ååå è½½ãåå­çæ§ååå¾åæ¶åè½ã
    """
    
    def __init__(
        self,
        chunk_size: int = 10000,
        memory_threshold_mb: float = 1000.0,
        enable_gc: bool = True,
    ):
        """
        åå§ååå­ç®¡çå¨
        
        Args:
            chunk_size: ååå¤§å°ï¼è¡æ°ï¼
            memory_threshold_mb: åå­éå¼ï¼MBï¼ï¼è¶è¿æ¶è§¦ååå¾åæ¶
            enable_gc: æ¯å¦å¯ç¨èªå¨åå¾åæ¶
        """
        self.chunk_size = chunk_size
        self.memory_threshold_mb = memory_threshold_mb
        self.enable_gc = enable_gc
        self.monitor = MemoryMonitor()
        logger.info(
            f"MemoryManager initialized: chunk_size={chunk_size}, "
            f"threshold={memory_threshold_mb}MB, gc={enable_gc}"
        )
    
    def load_dataframe_chunked(
        self,
        file_path: str,
        chunk_size: Optional[int] = None,
    ) -> Generator[pd.DataFrame, None, None]:
        """
        ååå è½½ CSV æä»¶
        
        Args:
            file_path: æä»¶è·¯å¾
            chunk_size: ååå¤§å°ï¼å¦æä¸º None åä½¿ç¨é»è®¤å¼ï¼
        
        Yields:
            æ¯ä¸ªååç DataFrame
        """
        chunk_size = chunk_size or self.chunk_size
        
        try:
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                yield chunk
                
                # æ£æ¥åå­ä½¿ç¨
                if self.enable_gc:
                    self._check_and_cleanup()
                    
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            raise
    
    def process_dataframe_chunked(
        self,
        df: pd.DataFrame,
        process_func: Callable,
        chunk_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        ååå¤ç DataFrame
        
        Args:
            df: è¾å¥ DataFrame
            process_func: å¤çå½æ°
            chunk_size: ååå¤§å°
        
        Returns:
            å¤çåç DataFrame
        """
        chunk_size = chunk_size or self.chunk_size
        results = []
        
        for i in range(0, len(df), chunk_size):
            chunk = df.iloc[i:i + chunk_size]
            processed_chunk = process_func(chunk)
            results.append(processed_chunk)
            
            # æ£æ¥åå­ä½¿ç¨
            if self.enable_gc:
                self._check_and_cleanup()
        
        return pd.concat(results, ignore_index=True)
    
    def _check_and_cleanup(self) -> None:
        """æ£æ¥åå­ä½¿ç¨å¹¶è¿è¡æ¸ç"""
        current_memory = self.monitor.get_current_memory()
        
        if current_memory.process_memory_mb > self.memory_threshold_mb:
            logger.warning(
                f"Memory usage high: {current_memory.process_memory_mb:.1f}MB > "
                f"{self.memory_threshold_mb}MB, triggering garbage collection"
            )
            gc.collect()
            
            # åæ¬¡æ£æ¥
            new_memory = self.monitor.get_current_memory()
            freed = current_memory.process_memory_mb - new_memory.process_memory_mb
            logger.info(f"Garbage collection freed {freed:.1f}MB")
    
    def detect_memory_leak(
        self,
        func: Callable,
        iterations: int = 10,
        threshold_mb: float = 50.0,
    ) -> Dict[str, Any]:
        """
        æ£æµåå­æ³æ¼
        
        Args:
            func: è¦æµè¯çå½æ°
            iterations: è¿­ä»£æ¬¡æ°
            threshold_mb: åå­å¢é¿éå¼ï¼MBï¼
        
        Returns:
            æ£æµç»æå­å¸
        """
        gc.collect()
        initial_memory = self.monitor.get_current_memory().process_memory_mb
        memory_samples = [initial_memory]
        
        for i in range(iterations):
            try:
                func()
                gc.collect()
                current_memory = self.monitor.get_current_memory().process_memory_mb
                memory_samples.append(current_memory)
                logger.debug(f"Iteration {i + 1}: {current_memory:.1f}MB")
            except Exception as e:
                logger.error(f"Error in iteration {i + 1}: {e}")
        
        # åæåå­å¢é¿è¶å¿
        memory_growth = memory_samples[-1] - memory_samples[0]
        avg_growth_per_iteration = memory_growth / iterations if iterations > 0 else 0
        
        has_leak = memory_growth > threshold_mb
        
        result = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": memory_samples[-1],
            "total_growth_mb": memory_growth,
            "avg_growth_per_iteration_mb": avg_growth_per_iteration,
            "has_leak": has_leak,
            "memory_samples": memory_samples,
        }
        
        if has_leak:
            logger.warning(
                f"Potential memory leak detected: "
                f"{memory_growth:.1f}MB growth over {iterations} iterations"
            )
        else:
            logger.info(f"No memory leak detected: {memory_growth:.1f}MB growth")
        
        return result
    
    def get_object_size(self, obj: Any) -> int:
        """
        è·åå¯¹è±¡å¤§å°ï¼å­èï¼
        
        Args:
            obj: å¯¹è±¡
        
        Returns:
            å¯¹è±¡å¤§å°ï¼å­èï¼
        """
        return sys.getsizeof(obj)
    
    def get_dataframe_size(self, df: pd.DataFrame) -> int:
        """
        è·å DataFrame å¤§å°ï¼å­èï¼
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame å¤§å°ï¼å­èï¼
        """
        return df.memory_usage(deep=True).sum()
    
    def optimize_dataframe_memory(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """
        ä¼å DataFrame åå­ä½¿ç¨
        
        Args:
            df: è¾å¥ DataFrame
        
        Returns:
            ä¼ååç DataFrame åèççåå­ï¼å­èï¼
        """
        original_size = self.get_dataframe_size(df)
        df_optimized = df.copy()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            # ä¼åæ´æ°ç±»å
            if col_type == 'int64':
                c_min = df_optimized[col].min()
                c_max = df_optimized[col].max()
                
                if c_min > -128 and c_max < 127:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif c_min > -32768 and c_max < 32767:
                    df_optimized[col] = df_optimized[col].astype('int16')
                elif c_min > -2147483648 and c_max < 2147483647:
                    df_optimized[col] = df_optimized[col].astype('int32')
            
            # ä¼åæµ®ç¹ç±»å
            elif col_type == 'float64':
                df_optimized[col] = df_optimized[col].astype('float32')
            
            # ä¼åå¯¹è±¡ç±»å
            elif col_type == 'object':
                num_unique = len(df_optimized[col].unique())
                num_total = len(df_optimized[col])
                
                if num_unique / num_total < 0.5:
                    df_optimized[col] = df_optimized[col].astype('category')
        
        optimized_size = self.get_dataframe_size(df_optimized)
        saved_memory = original_size - optimized_size
        
        logger.info(
            f"Optimized DataFrame memory: {original_size / 1024 / 1024:.1f}MB -> "
            f"{optimized_size / 1024 / 1024:.1f}MB (saved {saved_memory / 1024 / 1024:.1f}MB)"
        )
        
        return df_optimized, saved_memory
    
    def get_stats(self) -> Dict[str, Any]:
        """è·ååå­ç»è®¡ä¿¡æ¯"""
        current = self.monitor.get_current_memory()
        return {
            "process_memory_mb": current.process_memory_mb,
            "system_memory_mb": current.system_memory_mb,
            "memory_percent": current.memory_percent,
            "peak_memory_mb": current.peak_memory_mb,
            "memory_delta_mb": self.monitor.get_memory_delta(),
        }
    
    def __repr__(self) -> str:
        return f"MemoryManager({self.monitor})"
