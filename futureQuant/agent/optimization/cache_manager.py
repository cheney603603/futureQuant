"""
ç¼å­ç®¡çå¨æ¨¡å

æä¾é«æçç¼å­æºå¶ï¼
- LRU ç¼å­ï¼åå­ï¼
- ç£çç¼å­ï¼æä¹åï¼
- ç¼å­å½ä¸­çç»è®¡
- ç¼å­å¤±æç­ç¥
- ç¼å­é¢ç­
"""

import hashlib
import json
import logging
import os
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """ç¼å­ç»è®¡ä¿¡æ¯"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        """ç¼å­å½ä¸­ç"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    @property
    def hit_rate_percent(self) -> float:
        """ç¼å­å½ä¸­çç¾åæ¯"""
        return self.hit_rate * 100
    
    def __repr__(self) -> str:
        return (
            f"CacheStats(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate_percent:.1f}%, "
            f"size={self.total_size_bytes / 1024 / 1024:.1f}MB)"
        )


class LRUCache:
    """
    LRUï¼æè¿æå°ä½¿ç¨ï¼ç¼å­å®ç°
    
    ç¹ç¹ï¼
    - åºå®å¤§å°ï¼è¶è¿å®¹éæ¶å é¤æå°ä½¿ç¨çé¡¹
    - O(1) æ¶é´å¤æåº¦çæ¥è¯¢åæå¥
    - çº¿ç¨ä¸å®å¨ï¼éè¦å¤é¨åæ­¥ï¼
    """
    
    def __init__(self, max_size: int = 1000):
        """
        åå§å LRU ç¼å­
        
        Args:
            max_size: æå¤§ç¼å­é¡¹æ°
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, Any] = OrderedDict()
        self.stats = CacheStats()
    
    def get(self, key: str) -> Optional[Any]:
        """
        è·åç¼å­å¼
        
        Args:
            key: ç¼å­é®
        
        Returns:
            ç¼å­å¼ï¼å¦æä¸å­å¨åè¿å None
        """
        if key not in self.cache:
            self.stats.misses += 1
            return None
        
        # ç§»å°æ«å°¾ï¼æè¿ä½¿ç¨ï¼
        self.cache.move_to_end(key)
        self.stats.hits += 1
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """
        è®¾ç½®ç¼å­å¼
        
        Args:
            key: ç¼å­é®
            value: ç¼å­å¼
        """
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # å é¤æå°ä½¿ç¨çé¡¹ï¼ç¬¬ä¸ä¸ªï¼
                removed_key, removed_value = self.cache.popitem(last=False)
                self.stats.evictions += 1
                logger.debug(f"Evicted cache entry: {removed_key}")
        
        self.cache[key] = value
    
    def clear(self) -> None:
        """æ¸ç©ºç¼å­"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def size(self) -> int:
        """è·åç¼å­é¡¹æ°"""
        return len(self.cache)
    
    def __repr__(self) -> str:
        return f"LRUCache(size={self.size()}/{self.max_size}, {self.stats})"


class DiskCache:
    """
    ç£çç¼å­å®ç°
    
    ç¹ç¹ï¼
    - æä¹åå­å¨
    - æ¯æè¿ææ¶é´
    - èªå¨æ¸çè¿ææ°æ®
    """
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        ttl_seconds: Optional[int] = None,
    ):
        """
        åå§åç£çç¼å­
        
        Args:
            cache_dir: ç¼å­ç®å½
            ttl_seconds: ç¼å­è¿ææ¶é´ï¼ç§ï¼ï¼None è¡¨ç¤ºæ°¸ä¸è¿æ
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = ttl_seconds
        self.stats = CacheStats()
        logger.info(f"DiskCache initialized: dir={cache_dir}, ttl={ttl_seconds}s")
    
    def _get_cache_path(self, key: str) -> Path:
        """è·åç¼å­æä»¶è·¯å¾"""
        # ä½¿ç¨ MD5 åå¸ä½ä¸ºæä»¶åï¼é¿åè·¯å¾é®é¢
        hash_key = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{hash_key}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """
        è·åç¼å­å¼
        
        Args:
            key: ç¼å­é®
        
        Returns:
            ç¼å­å¼ï¼å¦æä¸å­å¨æå·²è¿æåè¿å None
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            self.stats.misses += 1
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # æ£æ¥è¿ææ¶é´
            if self.ttl_seconds is not None:
                if time.time() - data['timestamp'] > self.ttl_seconds:
                    cache_path.unlink()
                    self.stats.misses += 1
                    logger.debug(f"Cache expired: {key}")
                    return None
            
            self.stats.hits += 1
            return data['value']
            
        except Exception as e:
            logger.error(f"Error reading cache: {e}")
            self.stats.misses += 1
            return None
    
    def put(self, key: str, value: Any) -> None:
        """
        è®¾ç½®ç¼å­å¼
        
        Args:
            key: ç¼å­é®
            value: ç¼å­å¼
        """
        cache_path = self._get_cache_path(key)
        
        try:
            data = {
                'value': value,
                'timestamp': time.time(),
                'key': key,
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            
            # æ´æ°ç»è®¡ä¿¡æ¯
            self.stats.total_size_bytes += cache_path.stat().st_size
            
        except Exception as e:
            logger.error(f"Error writing cache: {e}")
    
    def clear(self) -> None:
        """æ¸ç©ºææç¼å­"""
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                cache_file.unlink()
            except Exception as e:
                logger.error(f"Error deleting cache file: {e}")
        
        self.stats = CacheStats()
        logger.info("Disk cache cleared")
    
    def cleanup_expired(self) -> int:
        """
        æ¸çè¿æç¼å­
        
        Returns:
            å é¤çç¼å­æä»¶æ°
        """
        if self.ttl_seconds is None:
            return 0
        
        deleted_count = 0
        current_time = time.time()
        
        for cache_file in self.cache_dir.glob("*.cache"):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                if current_time - data['timestamp'] > self.ttl_seconds:
                    cache_file.unlink()
                    deleted_count += 1
                    
            except Exception as e:
                logger.error(f"Error checking cache file: {e}")
        
        logger.info(f"Cleaned up {deleted_count} expired cache entries")
        return deleted_count
    
    def __repr__(self) -> str:
        return f"DiskCache(dir={self.cache_dir}, ttl={self.ttl_seconds}s, {self.stats})"


class CacheManager:
    """
    ç¼å­ç®¡çå¨
    
    æ´ååå­ç¼å­åç£çç¼å­ï¼æä¾ç»ä¸çç¼å­æ¥å£ã
    """
    
    def __init__(
        self,
        memory_cache_size: int = 1000,
        disk_cache_dir: Optional[str] = None,
        disk_cache_ttl: Optional[int] = None,
        use_disk_cache: bool = True,
    ):
        """
        åå§åç¼å­ç®¡çå¨
        
        Args:
            memory_cache_size: åå­ç¼å­å¤§å°
            disk_cache_dir: ç£çç¼å­ç®å½
            disk_cache_ttl: ç£çç¼å­è¿ææ¶é´ï¼ç§ï¼
            use_disk_cache: æ¯å¦å¯ç¨ç£çç¼å­
        """
        self.memory_cache = LRUCache(max_size=memory_cache_size)
        self.use_disk_cache = use_disk_cache
        
        if use_disk_cache:
            cache_dir = disk_cache_dir or "./cache"
            self.disk_cache = DiskCache(cache_dir=cache_dir, ttl_seconds=disk_cache_ttl)
        else:
            self.disk_cache = None
        
        logger.info(
            f"CacheManager initialized: memory_size={memory_cache_size}, "
            f"disk_cache={use_disk_cache}"
        )
    
    def get(self, key: str) -> Optional[Any]:
        """
        è·åç¼å­å¼ï¼åæ¥åå­ï¼åæ¥ç£çï¼
        
        Args:
            key: ç¼å­é®
        
        Returns:
            ç¼å­å¼
        """
        # åæ¥åå­ç¼å­
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # åæ¥ç£çç¼å­
        if self.use_disk_cache and self.disk_cache:
            value = self.disk_cache.get(key)
            if value is not None:
                # åååå­ç¼å­
                self.memory_cache.put(key, value)
                return value
        
        return None
    
    def put(self, key: str, value: Any) -> None:
        """
        è®¾ç½®ç¼å­å¼ï¼åæ¶åå¥åå­åç£çï¼
        
        Args:
            key: ç¼å­é®
            value: ç¼å­å¼
        """
        self.memory_cache.put(key, value)
        
        if self.use_disk_cache and self.disk_cache:
            self.disk_cache.put(key, value)
    
    def clear(self) -> None:
        """æ¸ç©ºææç¼å­"""
        self.memory_cache.clear()
        if self.use_disk_cache and self.disk_cache:
            self.disk_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """è·åç¼å­ç»è®¡ä¿¡æ¯"""
        stats = {
            "memory_cache": {
                "hits": self.memory_cache.stats.hits,
                "misses": self.memory_cache.stats.misses,
                "hit_rate": self.memory_cache.stats.hit_rate_percent,
                "size": self.memory_cache.size(),
            }
        }
        
        if self.use_disk_cache and self.disk_cache:
            stats["disk_cache"] = {
                "hits": self.disk_cache.stats.hits,
                "misses": self.disk_cache.stats.misses,
                "hit_rate": self.disk_cache.stats.hit_rate_percent,
                "size_bytes": self.disk_cache.stats.total_size_bytes,
            }
        
        return stats
    
    def __repr__(self) -> str:
        return (
            f"CacheManager(memory={self.memory_cache}, "
            f"disk={self.disk_cache if self.use_disk_cache else 'disabled'})"
        )


class CachedFunction:
    """
    ç¼å­è£é¥°å¨
    
    ç¨äºç¼å­å½æ°çæ§è¡ç»æã
    """
    
    def __init__(
        self,
        func: Callable,
        cache_manager: CacheManager,
        key_prefix: str = "",
    ):
        """
        åå§åç¼å­å½æ°
        
        Args:
            func: è¦ç¼å­çå½æ°
            cache_manager: ç¼å­ç®¡çå¨
            key_prefix: ç¼å­é®åç¼
        """
        self.func = func
        self.cache_manager = cache_manager
        self.key_prefix = key_prefix
    
    def _make_key(self, *args, **kwargs) -> str:
        """çæç¼å­é®"""
        key_parts = [self.key_prefix, self.func.__name__]
        
        # æ·»å åæ°å°é®
        for arg in args:
            if isinstance(arg, (str, int, float)):
                key_parts.append(str(arg))
            elif isinstance(arg, pd.DataFrame):
                # ä½¿ç¨ DataFrame çå½¢ç¶ååå
                key_parts.append(f"df_{arg.shape}_{hash(tuple(arg.columns))}")
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float)):
                key_parts.append(f"{k}={v}")
        
        return "|".join(key_parts)
    
    def __call__(self, *args, **kwargs) -> Any:
        """æ§è¡ç¼å­å½æ°"""
        cache_key = self._make_key(*args, **kwargs)
        
        # å°è¯ä»ç¼å­è·å
        cached_value = self.cache_manager.get(cache_key)
        if cached_value is not None:
            logger.debug(f"Cache hit: {cache_key}")
            return cached_value
        
        # æ§è¡å½æ°
        logger.debug(f"Cache miss: {cache_key}")
        result = self.func(*args, **kwargs)
        
        # å­å¥ç¼å­
        self.cache_manager.put(cache_key, result)
        
        return result


def cached(cache_manager: CacheManager, key_prefix: str = ""):
    """
    ç¼å­è£é¥°å¨å·¥åå½æ°
    
    ä½¿ç¨æ¹å¼ï¼
        @cached(cache_manager, key_prefix="factor")
        def calculate_factor(data):
            ...
    """
    def decorator(func: Callable) -> Callable:
        return CachedFunction(func, cache_manager, key_prefix)
    
    return decorator
