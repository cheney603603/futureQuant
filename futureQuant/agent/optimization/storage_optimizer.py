"""
å­å¨ä¼åå¨æ¨¡å

æä¾æ°æ®å­å¨ä¼åè½åï¼
- Parquet æä»¶åç¼©
- æ°æ®ååºå­å¨
- åå¼å­å¨ä¼å
- æä»¶åå¹¶
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class CompressionConfig:
    """åç¼©éç½®"""
    
    COMPRESSION_TYPES = {
        'snappy': 'snappy',
        'gzip': 'gzip',
        'brotli': 'brotli',
        'lz4': 'lz4',
        'zstd': 'zstd',
        'none': None,
    }
    
    def __init__(
        self,
        compression: str = 'snappy',
        compression_level: Optional[int] = None,
    ):
        """
        åå§ååç¼©éç½®
        
        Args:
            compression: åç¼©ç®æ³ ('snappy', 'gzip', 'brotli', 'lz4', 'zstd', 'none')
            compression_level: åç¼©çº§å«ï¼åå³äºç®æ³ï¼
        """
        if compression not in self.COMPRESSION_TYPES:
            raise ValueError(f"Unsupported compression: {compression}")
        
        self.compression = self.COMPRESSION_TYPES[compression]
        self.compression_level = compression_level
    
    def __repr__(self) -> str:
        return f"CompressionConfig(compression={self.compression}, level={self.compression_level})"


class StorageOptimizer:
    """
    å­å¨ä¼åå¨
    
    æä¾ Parquet æä»¶ä¼åãæ°æ®ååºãåå¼å­å¨ç­åè½ã
    """
    
    def __init__(
        self,
        storage_dir: str = "./data",
        compression: str = 'snappy',
    ):
        """
        åå§åå­å¨ä¼åå¨
        
        Args:
            storage_dir: å­å¨ç®å½
            compression: åç¼©ç®æ³
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.compression_config = CompressionConfig(compression=compression)
        logger.info(f"StorageOptimizer initialized: dir={storage_dir}, compression={compression}")
    
    def save_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        partition_cols: Optional[List[str]] = None,
    ) -> Path:
        """
        ä¿å­ DataFrame ä¸º Parquet æä»¶
        
        Args:
            df: è¦ä¿å­ç DataFrame
            name: æä»¶åï¼ä¸å«æ©å±åï¼
            partition_cols: ååºå
        
        Returns:
            ä¿å­çæä»¶è·¯å¾
        """
        output_path = self.storage_dir / f"{name}.parquet"
        
        try:
            # è½¬æ¢ä¸º PyArrow Table
            table = pa.Table.from_pandas(df)
            
            # åå¥ Parquet æä»¶
            pq.write_table(
                table,
                output_path,
                compression=self.compression_config.compression,
                compression_level=self.compression_config.compression_level,
            )
            
            original_size = df.memory_usage(deep=True).sum()
            compressed_size = output_path.stat().st_size
            compression_ratio = (1 - compressed_size / original_size) * 100 if original_size > 0 else 0
            
            logger.info(
                f"Saved {name}: {original_size / 1024 / 1024:.1f}MB -> "
                f"{compressed_size / 1024 / 1024:.1f}MB ({compression_ratio:.1f}% compression)"
            )
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving DataFrame: {e}")
            raise
    
    def save_partitioned(
        self,
        df: pd.DataFrame,
        name: str,
        partition_cols: List[str],
    ) -> Path:
        """
        ä¿å­ååº Parquet æä»¶
        
        Args:
            df: è¦ä¿å­ç DataFrame
            name: ç®å½å
            partition_cols: ååºå
        
        Returns:
            ä¿å­çç®å½è·¯å¾
        """
        output_dir = self.storage_dir / name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # è½¬æ¢ä¸º PyArrow Table
            table = pa.Table.from_pandas(df)
            
            # åå¥ååº Parquet æä»¶
            pq.write_to_dataset(
                table,
                root_path=output_dir,
                partition_cols=partition_cols,
                compression=self.compression_config.compression,
            )
            
            logger.info(f"Saved partitioned dataset: {name} with partitions {partition_cols}")
            return output_dir
            
        except Exception as e:
            logger.error(f"Error saving partitioned dataset: {e}")
            raise
    
    def load_dataframe(self, name: str) -> pd.DataFrame:
        """
        å è½½ Parquet æä»¶ä¸º DataFrame
        
        Args:
            name: æä»¶åï¼ä¸å«æ©å±åï¼
        
        Returns:
            å è½½ç DataFrame
        """
        file_path = self.storage_dir / f"{name}.parquet"
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            table = pq.read_table(file_path)
            df = table.to_pandas()
            logger.info(f"Loaded {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading DataFrame: {e}")
            raise
    
    def load_partitioned(self, name: str) -> pd.DataFrame:
        """
        å è½½ååº Parquet æ°æ®é
        
        Args:
            name: ç®å½å
        
        Returns:
            å è½½ç DataFrame
        """
        dir_path = self.storage_dir / name
        
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        try:
            table = pq.read_table(dir_path)
            df = table.to_pandas()
            logger.info(f"Loaded partitioned dataset {name}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading partitioned dataset: {e}")
            raise
    
    def get_file_stats(self, name: str) -> Dict[str, Any]:
        """
        è·åæä»¶ç»è®¡ä¿¡æ¯
        
        Args:
            name: æä»¶åï¼ä¸å«æ©å±åï¼
        
        Returns:
            ç»è®¡ä¿¡æ¯å­å¸
        """
        file_path = self.storage_dir / f"{name}.parquet"
        
        if not file_path.exists():
            return {}
        
        try:
            file_size = file_path.stat().st_size
            table = pq.read_table(file_path)
            df = table.to_pandas()
            
            return {
                "file_size_bytes": file_size,
                "file_size_mb": file_size / 1024 / 1024,
                "rows": df.shape[0],
                "columns": df.shape[1],
                "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
                "compression_ratio": (1 - file_size / df.memory_usage(deep=True).sum()) * 100,
            }
            
        except Exception as e:
            logger.error(f"Error getting file stats: {e}")
            return {}
    
    def merge_files(
        self,
        file_names: List[str],
        output_name: str,
    ) -> Path:
        """
        åå¹¶å¤ä¸ª Parquet æä»¶
        
        Args:
            file_names: è¦åå¹¶çæä»¶ååè¡¨
            output_name: è¾åºæä»¶å
        
        Returns:
            åå¹¶åçæä»¶è·¯å¾
        """
        dfs = []
        
        for file_name in file_names:
            try:
                df = self.load_dataframe(file_name)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_name}: {e}")
        
        if not dfs:
            raise ValueError("No files loaded successfully")
        
        # åå¹¶ DataFrame
        merged_df = pd.concat(dfs, ignore_index=True)
        
        # ä¿å­åå¹¶åçæä»¶
        output_path = self.save_dataframe(merged_df, output_name)
        logger.info(f"Merged {len(file_names)} files into {output_name}")
        
        return output_path
    
    def optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ä¼å DataFrame çæ°æ®ç±»åä»¥åå°åå­å ç¨
        
        Args:
            df: è¾å¥ DataFrame
        
        Returns:
            ä¼ååç DataFrame
        """
        original_memory = df.memory_usage(deep=True).sum()
        
        for col in df.columns:
            col_type = df[col].dtype
            
            # ä¼åæ´æ°ç±»å
            if col_type == 'int64':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            
            # ä¼åæµ®ç¹ç±»å
            elif col_type == 'float64':
                df[col] = df[col].astype(np.float32)
            
            # ä¼åå¯¹è±¡ç±»åï¼å­ç¬¦ä¸²ï¼
            elif col_type == 'object':
                if df[col].dtype == 'object':
                    num_unique = len(df[col].unique())
                    num_total = len(df[col])
                    
                    if num_unique / num_total < 0.5:
                        df[col] = df[col].astype('category')
        
        optimized_memory = df.memory_usage(deep=True).sum()
        reduction = (1 - optimized_memory / original_memory) * 100
        
        logger.info(
            f"Optimized dtypes: {original_memory / 1024 / 1024:.1f}MB -> "
            f"{optimized_memory / 1024 / 1024:.1f}MB ({reduction:.1f}% reduction)"
        )
        
        return df
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """è·åå­å¨ç»è®¡ä¿¡æ¯"""
        total_size = 0
        file_count = 0
        
        for file_path in self.storage_dir.glob("**/*.parquet"):
            total_size += file_path.stat().st_size
            file_count += 1
        
        return {
            "total_files": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / 1024 / 1024,
            "storage_dir": str(self.storage_dir),
        }
    
    def __repr__(self) -> str:
        stats = self.get_storage_stats()
        return (
            f"StorageOptimizer(dir={self.storage_dir}, "
            f"files={stats['total_files']}, "
            f"size={stats['total_size_mb']:.1f}MB)"
        )


# å¯¼å¥ numpy ç¨äºæ°æ®ç±»åä¼å
import numpy as np
