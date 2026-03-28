"""
æ°æ®é¢å è½½å¨æ¨¡å

æä¾æºè½æ°æ®é¢å è½½è½åï¼
- ç­æ°æ®é¢å è½½
- é¢æµæ§å è½½
- åå°å è½½çº¿ç¨
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PreloadStats:
    """é¢å è½½ç»è®¡ä¿¡æ¯"""
    total_preloaded: int = 0
    successful_preloads: int = 0
    failed_preloads: int = 0
    total_preload_time_ms: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """æåç"""
        total = self.successful_preloads + self.failed_preloads
        return self.successful_preloads / total if total > 0 else 0.0
    
    def __repr__(self) -> str:
        return (
            f"PreloadStats(total={self.total_preloaded}, "
            f"success={self.successful_preloads}, "
            f"failed={self.failed_preloads}, "
            f"success_rate={self.success_rate * 100:.1f}%)"
        )


class DataPreloader:
    """
    æ°æ®é¢å è½½å¨
    
    æ¯æç­æ°æ®é¢å è½½åé¢æµæ§å è½½ã
    """
    
    def __init__(
        self,
        max_preload_size: int = 100,
        preload_timeout_seconds: float = 30.0,
    ):
        """
        åå§åæ°æ®é¢å è½½å¨
        
        Args:
            max_preload_size: æå¤§é¢å è½½æ°æ®é
            preload_timeout_seconds: é¢å è½½è¶æ¶æ¶é´
        """
        self.max_preload_size = max_preload_size
        self.preload_timeout_seconds = preload_timeout_seconds
        self.preloaded_data: Dict[str, Any] = {}
        self.preload_queue: List[str] = []
        self.stats = PreloadStats()
        self.lock = threading.Lock()
        logger.info(
            f"DataPreloader initialized: max_size={max_preload_size}, "
            f"timeout={preload_timeout_seconds}s"
        )
    
    def preload_data(
        self,
        data_key: str,
        load_func: Callable,
        *args,
        **kwargs
    ) -> bool:
        """
        é¢å è½½æ°æ®
        
        Args:
            data_key: æ°æ®é®
            load_func: å è½½å½æ°
            *args: å è½½å½æ°çä½ç½®åæ°
            **kwargs: å è½½å½æ°çå³é®å­åæ°
        
        Returns:
            æ¯å¦é¢å è½½æå
        """
        try:
            start_time = time.time()
            
            # æ§è¡å è½½å½æ°
            data = load_func(*args, **kwargs)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            with self.lock:
                # æ£æ¥ç¼å­å¤§å°
                if len(self.preloaded_data) >= self.max_preload_size:
                    # å é¤ææ§çæ°æ®
                    oldest_key = self.preload_queue.pop(0)
                    del self.preloaded_data[oldest_key]
                    logger.debug(f"Evicted preloaded data: {oldest_key}")
                
                # å­å¨é¢å è½½çæ°æ®
                self.preloaded_data[data_key] = data
                self.preload_queue.append(data_key)
            
            self.stats.total_preloaded += 1
            self.stats.successful_preloads += 1
            self.stats.total_preload_time_ms += elapsed_ms
            
            logger.info(f"Preloaded data: {data_key} ({elapsed_ms:.2f}ms)")
            return True
            
        except Exception as e:
            self.stats.total_preloaded += 1
            self.stats.failed_preloads += 1
            logger.error(f"Error preloading data {data_key}: {e}")
            return False
    
    def get_preloaded_data(self, data_key: str) -> Optional[Any]:
        """
        è·åé¢å è½½çæ°æ®
        
        Args:
            data_key: æ°æ®é®
        
        Returns:
            é¢å è½½çæ°æ®ï¼å¦æä¸å­å¨åè¿å None
        """
        with self.lock:
            return self.preloaded_data.get(data_key)
    
    def is_preloaded(self, data_key: str) -> bool:
        """
        æ£æ¥æ°æ®æ¯å¦å·²é¢å è½½
        
        Args:
            data_key: æ°æ®é®
        
        Returns:
            æ¯å¦å·²é¢å è½½
        """
        with self.lock:
            return data_key in self.preloaded_data
    
    def clear_preloaded_data(self) -> None:
        """æ¸ç©ºææé¢å è½½çæ°æ®"""
        with self.lock:
            self.preloaded_data.clear()
            self.preload_queue.clear()
        logger.info("Cleared all preloaded data")
    
    def get_stats(self) -> Dict[str, Any]:
        """è·åé¢å è½½ç»è®¡ä¿¡æ¯"""
        with self.lock:
            return {
                "total_preloaded": self.stats.total_preloaded,
                "successful_preloads": self.stats.successful_preloads,
                "failed_preloads": self.stats.failed_preloads,
                "success_rate": self.stats.success_rate * 100,
                "avg_preload_time_ms": (
                    self.stats.total_preload_time_ms / self.stats.successful_preloads
                    if self.stats.successful_preloads > 0 else 0
                ),
                "current_preloaded_count": len(self.preloaded_data),
            }
    
    def __repr__(self) -> str:
        return f"DataPreloader({self.stats})"


class BackgroundPreloader:
    """
    åå°é¢å è½½å¨
    
    å¨åå°çº¿ç¨ä¸­æ§è¡æ°æ®é¢å è½½ã
    """
    
    def __init__(
        self,
        preloader: DataPreloader,
        max_workers: int = 2,
    ):
        """
        åå§ååå°é¢å è½½å¨
        
        Args:
            preloader: æ°æ®é¢å è½½å¨å®ä¾
            max_workers: æå¤§å·¥ä½çº¿ç¨æ°
        """
        self.preloader = preloader
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_tasks: Dict[str, threading.Future] = {}
        self.lock = threading.Lock()
        logger.info(f"BackgroundPreloader initialized: max_workers={max_workers}")
    
    def preload_async(
        self,
        data_key: str,
        load_func: Callable,
        *args,
        **kwargs
    ) -> None:
        """
        å¼æ­¥é¢å è½½æ°æ®
        
        Args:
            data_key: æ°æ®é®
            load_func: å è½½å½æ°
            *args: å è½½å½æ°çä½ç½®åæ°
            **kwargs: å è½½å½æ°çå³é®å­åæ°
        """
        with self.lock:
            # æ£æ¥æ¯å¦å·²æå¾å¤çä»»å¡
            if data_key in self.pending_tasks:
                logger.debug(f"Preload task already pending: {data_key}")
                return
        
        # æäº¤åå°ä»»å¡
        future = self.executor.submit(
            self.preloader.preload_data,
            data_key,
            load_func,
            *args,
            **kwargs
        )
        
        with self.lock:
            self.pending_tasks[data_key] = future
        
        logger.debug(f"Submitted async preload task: {data_key}")
    
    def wait_for_preload(
        self,
        data_key: str,
        timeout: Optional[float] = None,
    ) -> bool:
        """
        ç­å¾é¢å è½½å®æ
        
        Args:
            data_key: æ°æ®é®
            timeout: è¶æ¶æ¶é´ï¼ç§ï¼
        
        Returns:
            æ¯å¦é¢å è½½æå
        """
        with self.lock:
            future = self.pending_tasks.get(data_key)
        
        if future is None:
            return self.preloader.is_preloaded(data_key)
        
        try:
            result = future.result(timeout=timeout)
            with self.lock:
                del self.pending_tasks[data_key]
            return result
        except Exception as e:
            logger.error(f"Error waiting for preload: {e}")
            return False
    
    def shutdown(self, wait: bool = True) -> None:
        """
        å³é­åå°é¢å è½½å¨
        
        Args:
            wait: æ¯å¦ç­å¾ææä»»å¡å®æ
        """
        self.executor.shutdown(wait=wait)
        logger.info("BackgroundPreloader shutdown")
    
    def __repr__(self) -> str:
        return f"BackgroundPreloader(pending={len(self.pending_tasks)})"


class PredictivePreloader:
    """
    é¢æµæ§é¢å è½½å¨
    
    åºäºè®¿é®æ¨¡å¼é¢æµå¹¶é¢å è½½æ°æ®ã
    """
    
    def __init__(
        self,
        preloader: DataPreloader,
        history_size: int = 100,
    ):
        """
        åå§åé¢æµæ§é¢å è½½å¨
        
        Args:
            preloader: æ°æ®é¢å è½½å¨å®ä¾
            history_size: è®¿é®åå²å¤§å°
        """
        self.preloader = preloader
        self.history_size = history_size
        self.access_history: List[str] = []
        self.access_patterns: Dict[str, Set[str]] = {}
        self.lock = threading.Lock()
        logger.info(f"PredictivePreloader initialized: history_size={history_size}")
    
    def record_access(self, data_key: str) -> None:
        """
        è®°å½æ°æ®è®¿é®
        
        Args:
            data_key: æ°æ®é®
        """
        with self.lock:
            self.access_history.append(data_key)
            
            # ä¿æåå²å¤§å°
            if len(self.access_history) > self.history_size:
                self.access_history.pop(0)
    
    def predict_next_access(self) -> Optional[str]:
        """
        é¢æµä¸ä¸ä¸ªè®¿é®çæ°æ®
        
        Returns:
            é¢æµçæ°æ®é®ï¼å¦ææ æ³é¢æµåè¿å None
        """
        with self.lock:
            if len(self.access_history) < 2:
                return None
            
            # ç®åçé¢æµï¼æ¥æ¾æåä¸ä¸ªè®¿é®åæå¸¸è·éçæ°æ®
            last_access = self.access_history[-1]
            
            # æå»ºè®¿é®æ¨¡å¼
            for i in range(len(self.access_history) - 1):
                current = self.access_history[i]
                next_access = self.access_history[i + 1]
                
                if current not in self.access_patterns:
                    self.access_patterns[current] = set()
                
                self.access_patterns[current].add(next_access)
            
            # è¿åæå¯è½çä¸ä¸ä¸ªè®¿é®
            if last_access in self.access_patterns:
                candidates = self.access_patterns[last_access]
                if candidates:
                    return max(candidates, key=lambda x: self.access_history.count(x))
        
        return None
    
    def preload_predicted(
        self,
        load_func: Callable,
        *args,
        **kwargs
    ) -> bool:
        """
        é¢å è½½é¢æµçæ°æ®
        
        Args:
            load_func: å è½½å½æ°
            *args: å è½½å½æ°çä½ç½®åæ°
            **kwargs: å è½½å½æ°çå³é®å­åæ°
        
        Returns:
            æ¯å¦é¢å è½½æå
        """
        predicted_key = self.predict_next_access()
        
        if predicted_key is None:
            return False
        
        if self.preloader.is_preloaded(predicted_key):
            return True
        
        logger.info(f"Preloading predicted data: {predicted_key}")
        return self.preloader.preload_data(predicted_key, load_func, *args, **kwargs)
    
    def __repr__(self) -> str:
        return f"PredictivePreloader(history={len(self.access_history)})"
