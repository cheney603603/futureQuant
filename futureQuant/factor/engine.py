"""
因子计算引擎 - 因子注册、批量计算、缓存管理
"""

from typing import List, Dict, Optional, Type, Union, Tuple
from datetime import datetime
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

from ..core.base import Factor
from ..core.logger import get_logger
from ..core.exceptions import FactorError

logger = get_logger('factor.engine')


def _data_hash(data: pd.DataFrame) -> str:
    """
    计算 DataFrame 的轻量哈希值，用于缓存 key 区分不同输入数据。

    使用 index 范围 + shape + 首尾行数据摘要，兼顾速度与准确性。

    Args:
        data: 输入数据

    Returns:
        16 位十六进制哈希字符串
    """
    try:
        index_repr = f"{data.index[0]}_{data.index[-1]}_{len(data)}"
        shape_repr = f"{data.shape}"
        # 取首尾各 2 行数值做摘要，避免对大数据集全量哈希
        sample = pd.concat([data.iloc[:2], data.iloc[-2:]]).values.tobytes()
        raw = f"{index_repr}_{shape_repr}_{hash(sample)}".encode()
        return hashlib.md5(raw).hexdigest()[:16]
    except Exception:
        # 兜底：返回固定值，缓存失效但不崩溃
        return "fallback"




class FactorEngine:
    """因子计算引擎"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        初始化因子引擎
        
        Args:
            cache_dir: 因子缓存目录
        """
        self.factors: Dict[str, Factor] = {}
        # 缓存键为 (factor_name, data_hash)，避免换数据后返回旧结果
        self.cache: Dict[Tuple[str, str], pd.Series] = {}
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def register(self, factor: Factor):
        """
        注册因子
        
        Args:
            factor: 因子实例
        """
        if factor.name in self.factors:
            logger.warning(f"Factor {factor.name} already registered, overwriting")
        
        self.factors[factor.name] = factor
        logger.info(f"Registered factor: {factor.name}")
    
    def register_many(self, factors: List[Factor]):
        """批量注册因子"""
        for factor in factors:
            self.register(factor)
    
    def unregister(self, name: str):
        """注销因子"""
        if name in self.factors:
            del self.factors[name]
            logger.info(f"Unregistered factor: {name}")
    
    def compute(
        self, 
        data: pd.DataFrame, 
        factor_name: str,
        use_cache: bool = True
    ) -> pd.Series:
        """
        计算单个因子
        
        Args:
            data: 输入数据
            factor_name: 因子名称
            use_cache: 是否使用缓存
            
        Returns:
            因子值序列
        """
        if factor_name not in self.factors:
            raise FactorError(f"Factor not found: {factor_name}")
        
        factor = self.factors[factor_name]
        
        # 检查缓存 —— 以 (factor_name, data_hash) 为键，防止数据变化后命中旧缓存
        cache_key = (factor_name, _data_hash(data)) if use_cache else None
        if use_cache and cache_key in self.cache:
            logger.debug(f"Using cached factor: {factor_name}")
            return self.cache[cache_key]
        
        # 计算因子
        logger.info(f"Computing factor: {factor_name}")
        result = factor.compute(data)
        
        # 保存缓存
        if use_cache and cache_key is not None:
            self.cache[cache_key] = result
        
        return result
    
    def compute_all(
        self, 
        data: pd.DataFrame,
        factor_names: Optional[List[str]] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        批量计算因子
        
        Args:
            data: 输入数据
            factor_names: 因子名称列表，为None时计算所有
            use_cache: 是否使用缓存
            
        Returns:
            DataFrame，每列是一个因子
        """
        names = factor_names or list(self.factors.keys())
        results = {}
        
        for name in names:
            try:
                result = self.compute(data, name, use_cache)
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to compute factor {name}: {e}")
        
        if not results:
            return pd.DataFrame()
        
        # 合并结果
        df = pd.DataFrame(results)
        df.index = data.index
        
        return df
    
    def compute_panel(
        self, 
        data_dict: Dict[str, pd.DataFrame],
        factor_name: str
    ) -> pd.DataFrame:
        """
        横截面计算因子
        
        Args:
            data_dict: {symbol: DataFrame}
            factor_name: 因子名称
            
        Returns:
            DataFrame，列为品种，行为日期
        """
        if factor_name not in self.factors:
            raise FactorError(f"Factor not found: {factor_name}")
        
        factor = self.factors[factor_name]
        return factor.compute_panel(data_dict)
    
    def get_factor_info(self, name: str) -> Optional[Dict]:
        """获取因子信息"""
        if name not in self.factors:
            return None
        
        factor = self.factors[name]
        return {
            'name': factor.name,
            'class': factor.__class__.__name__,
            'params': factor.params,
        }
    
    def list_factors(self) -> List[str]:
        """列出所有已注册因子"""
        return list(self.factors.keys())
    
    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
        logger.info("Factor cache cleared")
    
    def save_factors(self, path: str):
        """保存因子配置"""
        import json
        
        config = {
            name: {
                'class': f.__class__.__name__,
                'params': f.params,
            }
            for name, f in self.factors.items()
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(config)} factors to {path}")
