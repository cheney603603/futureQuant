"""
å å­çæ¬ç®¡çæ¨¡å

è´è´£å å­çæ¬çåå»ºãæ¥è¯¢åå¯¹æ¯ã
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import sqlite3
import json
from pathlib import Path

from ...core.logger import get_logger

logger = get_logger('agent.repository.version_control')


class FactorVersionControl:
    """
    å å­çæ¬ç®¡çå¨
    
    è´è´£å å­çæ¬çåå»ºãæ¥è¯¢ãå¯¹æ¯ååæ»ã
    """
    
    def __init__(self, db_path: str):
        """
        åå§åçæ¬ç®¡çå¨
        
        Args:
            db_path: æ°æ®åºè·¯å¾
        """
        self.db_path = db_path
        self._init_version_table()
        self.logger = logger
    
    def _init_version_table(self):
        """åå§åçæ¬è¡¨"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS factor_version (
                version_id TEXT PRIMARY KEY,
                factor_id TEXT NOT NULL,
                version_number TEXT,
                parameters TEXT,
                code TEXT,
                change_reason TEXT,
                created_at TIMESTAMP,
                FOREIGN KEY (factor_id) REFERENCES factor_metadata(factor_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_version(
        self,
        factor_id: str,
        version_number: str,
        parameters: Dict[str, Any],
        code: str,
        change_reason: str = ''
    ) -> str:
        """
        åå»ºæ°çæ¬
        
        Args:
            factor_id: å å­ ID
            version_number: çæ¬å· (å¦ v1.0, v1.1)
            parameters: åæ°éç½®
            code: è®¡ç®ä»£ç 
            change_reason: åæ´åå 
            
        Returns:
            çæ¬ ID
        """
        version_id = f"{factor_id}_{version_number}"
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO factor_version
                (version_id, factor_id, version_number, parameters, code, change_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                version_id,
                factor_id,
                version_number,
                json.dumps(parameters),
                code,
                change_reason,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Version {version_id} created")
            return version_id
            
        except Exception as e:
            self.logger.error(f"Failed to create version: {e}")
            raise
    
    def get_version_history(self, factor_id: str) -> List[Dict[str, Any]]:
        """
        è·åå å­çæ¬åå²
        
        Args:
            factor_id: å å­ ID
            
        Returns:
            çæ¬åå²åè¡¨
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT * FROM factor_version WHERE factor_id = ? ORDER BY created_at DESC',
                (factor_id,)
            )
            
            rows = cursor.fetchall()
            conn.close()
            
            history = []
            for row in rows:
                history.append({
                    'version_id': row[0],
                    'version_number': row[2],
                    'parameters': json.loads(row[3]),
                    'change_reason': row[5],
                    'created_at': row[6],
                })
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get version history: {e}")
            return []
    
    def compare_versions(
        self,
        version_id_1: str,
        version_id_2: str
    ) -> Dict[str, Any]:
        """
        å¯¹æ¯ä¸¤ä¸ªçæ¬
        
        Args:
            version_id_1: çæ¬ ID 1
            version_id_2: çæ¬ ID 2
            
        Returns:
            å¯¹æ¯ç»æ
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # è·åä¸¤ä¸ªçæ¬çä¿¡æ¯
            cursor.execute('SELECT * FROM factor_version WHERE version_id = ?', (version_id_1,))
            row1 = cursor.fetchone()
            
            cursor.execute('SELECT * FROM factor_version WHERE version_id = ?', (version_id_2,))
            row2 = cursor.fetchone()
            
            conn.close()
            
            if not row1 or not row2:
                return {}
            
            params1 = json.loads(row1[3])
            params2 = json.loads(row2[3])
            
            # å¯¹æ¯åæ°
            diff = {
                'version_1': row1[2],
                'version_2': row2[2],
                'parameter_changes': {},
                'code_changed': row1[4] != row2[4],
            }
            
            # æ¾åºåæ°å·®å¼
            all_keys = set(params1.keys()) | set(params2.keys())
            for key in all_keys:
                if params1.get(key) != params2.get(key):
                    diff['parameter_changes'][key] = {
                        'old': params1.get(key),
                        'new': params2.get(key),
                    }
            
            return diff
            
        except Exception as e:
            self.logger.error(f"Failed to compare versions: {e}")
            return {}
    
    def rollback(self, factor_id: str, version_number: str) -> bool:
        """
        åæ»å°æå®çæ¬
        
        Args:
            factor_id: å å­ ID
            version_number: çæ¬å·
            
        Returns:
            æ¯å¦æå
        """
        try:
            version_id = f"{factor_id}_{version_number}"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # è·åæå®çæ¬çä¿¡æ¯
            cursor.execute('SELECT * FROM factor_version WHERE version_id = ?', (version_id,))
            row = cursor.fetchone()
            
            if not row:
                self.logger.warning(f"Version {version_id} not found")
                return False
            
            # åå»ºæ°çæ¬ä½ä¸ºåæ»çæ¬
            new_version = f"{version_number}_rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            cursor.execute('''
                INSERT INTO factor_version
                (version_id, factor_id, version_number, parameters, code, change_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"{factor_id}_{new_version}",
                factor_id,
                new_version,
                row[3],
                row[4],
                f"Rollback from {version_number}",
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Rolled back to version {version_number}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback: {e}")
            return False
