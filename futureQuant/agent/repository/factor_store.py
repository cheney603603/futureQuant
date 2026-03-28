"""
å å­å­å¨æ¨¡å

è´è´£å å­åæ°æ®åå å­å¼çæä¹åå­å¨ã
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import sqlite3
import json
import pandas as pd
import numpy as np

from ...core.logger import get_logger
from ...core.base import Factor

logger = get_logger('agent.repository.factor_store')


class FactorRepository:
    """
    å å­åºç®¡çå¨
    
    è´è´£å å­çæä¹åå­å¨ãæ¥è¯¢åç®¡çã
    """
    
    def __init__(self, storage_dir: str = './factor_repo'):
        """
        åå§åå å­åº
        
        Args:
            storage_dir: å­å¨ç®å½
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # åæ°æ®æ°æ®åº
        self.db_path = self.storage_dir / 'metadata.db'
        self.values_dir = self.storage_dir / 'values'
        self.values_dir.mkdir(parents=True, exist_ok=True)
        
        # åå§åæ°æ®åº
        self._init_db()
        
        self.logger = logger
    
    def _init_db(self):
        """åå§åæ°æ®åºè¡¨"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # å å­åæ°æ®è¡¨
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factor_metadata (
                factor_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT,
                sub_category TEXT,
                description TEXT,
                formula TEXT,
                parameters TEXT,
                data_dependencies TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                status TEXT DEFAULT 'active'
            )
        ''')
        
        # å å­æ§è½è¡¨
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factor_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                factor_id TEXT NOT NULL,
                version_id TEXT,
                period TEXT,
                start_date DATE,
                end_date DATE,
                ic_mean REAL,
                icir REAL,
                ic_win_rate REAL,
                monotonicity REAL,
                turnover REAL,
                max_drawdown REAL,
                overall_score REAL,
                created_at TIMESTAMP,
                FOREIGN KEY (factor_id) REFERENCES factor_metadata(factor_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_factor(
        self,
        factor: Factor,
        values: pd.DataFrame,
        performance: Optional[Dict[str, float]] = None,
        version_id: str = 'v1.0'
    ) -> str:
        """
        ä¿å­å å­
        
        Args:
            factor: å å­å®ä¾
            values: å å­å¼ DataFrame
            performance: æ§è½ææ å­å¸
            version_id: çæ¬ ID
            
        Returns:
            å å­ ID
        """
        factor_id = f"{factor.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # 1. ä¿å­åæ°æ®
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO factor_metadata
                (factor_id, name, category, parameters, created_at, updated_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                factor_id,
                factor.name,
                getattr(factor, 'category', 'unknown'),
                json.dumps(factor.params),
                datetime.now(),
                datetime.now(),
                'active'
            ))
            
            conn.commit()
            conn.close()
            
            # 2. ä¿å­å å­å¼ï¼Parquet æ ¼å¼ï¼
            values_path = self.values_dir / f"{factor_id}.parquet"
            values.to_parquet(str(values_path))
            
            # 3. ä¿å­æ§è½ææ 
            if performance:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO factor_performance
                    (factor_id, version_id, ic_mean, icir, ic_win_rate, overall_score, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    factor_id,
                    version_id,
                    performance.get('ic_mean', 0),
                    performance.get('icir', 0),
                    performance.get('ic_win_rate', 0),
                    performance.get('overall_score', 0),
                    datetime.now()
                ))
                
                conn.commit()
                conn.close()
            
            self.logger.info(f"Factor {factor_id} saved successfully")
            return factor_id
            
        except Exception as e:
            self.logger.error(f"Failed to save factor: {e}")
            raise
    
    def get_factor(
        self,
        factor_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        è·åå å­
        
        Args:
            factor_id: å å­ ID
            start_date: å¼å§æ¥æ
            end_date: ç»ææ¥æ
            
        Returns:
            å å­æ°æ®å­å¸
        """
        try:
            # è·ååæ°æ®
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT * FROM factor_metadata WHERE factor_id = ?',
                (factor_id,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if not row:
                return None
            
            # è·åå å­å¼
            values_path = self.values_dir / f"{factor_id}.parquet"
            if values_path.exists():
                values = pd.read_parquet(str(values_path))
                
                # ææ¥æç­é
                if start_date and end_date:
                    values = values.loc[start_date:end_date]
            else:
                values = None
            
            return {
                'factor_id': factor_id,
                'name': row[1],
                'category': row[2],
                'values': values,
                'created_at': row[8],
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get factor: {e}")
            return None
    
    def list_factors(
        self,
        category: Optional[str] = None,
        status: str = 'active'
    ) -> List[str]:
        """
        ååºå å­
        
        Args:
            category: å å­ç±»å«
            status: å å­ç¶æ
            
        Returns:
            å å­ ID åè¡¨
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            if category:
                cursor.execute(
                    'SELECT factor_id FROM factor_metadata WHERE category = ? AND status = ?',
                    (category, status)
                )
            else:
                cursor.execute(
                    'SELECT factor_id FROM factor_metadata WHERE status = ?',
                    (status,)
                )
            
            factors = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            return factors
            
        except Exception as e:
            self.logger.error(f"Failed to list factors: {e}")
            return []
    
    def update_factor_status(self, factor_id: str, status: str):
        """
        æ´æ°å å­ç¶æ
        
        Args:
            factor_id: å å­ ID
            status: æ°ç¶æ (active/inactive/observed)
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute(
                'UPDATE factor_metadata SET status = ?, updated_at = ? WHERE factor_id = ?',
                (status, datetime.now(), factor_id)
            )
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Factor {factor_id} status updated to {status}")
            
        except Exception as e:
            self.logger.error(f"Failed to update factor status: {e}")
    
    def delete_factor(self, factor_id: str):
        """
        å é¤å å­
        
        Args:
            factor_id: å å­ ID
        """
        try:
            # å é¤åæ°æ®
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM factor_metadata WHERE factor_id = ?', (factor_id,))
            cursor.execute('DELETE FROM factor_performance WHERE factor_id = ?', (factor_id,))
            
            conn.commit()
            conn.close()
            
            # å é¤å å­å¼æä»¶
            values_path = self.values_dir / f"{factor_id}.parquet"
            if values_path.exists():
                values_path.unlink()
            
            self.logger.info(f"Factor {factor_id} deleted")
            
        except Exception as e:
            self.logger.error(f"Failed to delete factor: {e}")
